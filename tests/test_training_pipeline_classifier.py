"""The over/under classifier strategy, and proof it left the regressors alone.

Half of this file is about the classifier working; the other half is
characterization -- pinning that adding a third strategy changed nothing about
how the two regressors behave. That split is deliberate: the risky part of this
change was never the new code, it was the shared machinery it had to be
threaded through.
"""

import numpy as np
import optuna
import pandas as pd
import pytest
from xgboost import XGBClassifier, XGBRegressor

from training_pipeline import pipeline as pipeline_module
from training_pipeline.betting import evaluate_betting, expected_value
from training_pipeline.calibration import (
    brier_score,
    calibration_summary,
    calibration_table,
    log_loss,
)
from training_pipeline.config import (
    OVER_LABEL_COL,
    BettingConfig,
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    HoldoutConfig,
    OptunaConfig,
    PredictionStrategy,
    SearchSpaceConfig,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.data import PreparedDataset, add_over_under_label
from training_pipeline.decisions import predict_decisions, primary_threshold
from training_pipeline.pipeline import run_experiment
from training_pipeline.tuning import (
    get_strategy,
    select_best_classifier_trial_lexicographic,
    select_best_classifier_trial_lexicographic_pooled,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _frame(n_games: int = 260, games_per_day: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_days = n_games // games_per_day
    dates = np.repeat(
        pd.date_range("2025-11-01", periods=n_days, freq="D").to_numpy(), games_per_day
    )
    line = rng.uniform(200, 240, size=n_games).round(1)
    df = pd.DataFrame(
        {
            "GAME_DATE": dates,
            "SEASON_YEAR": 2025,
            "TOTAL_POINTS": (line + rng.normal(0, 12, n_games)).round(1),
            "ODDS_TOTAL_LINE_bet365": line,
            "FEATURE_A": rng.normal(size=n_games),
            "FEATURE_B": rng.normal(size=n_games),
        }
    )
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    df[OVER_LABEL_COL] = (df["LINE_ERROR"] > 0).astype(int)
    return df


def _prepared(df: pd.DataFrame, config: ExperimentConfig) -> PreparedDataset:
    features = ["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]
    return PreparedDataset(
        df_full=df,
        X=df[features],
        y=df[config.target_col],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=features,
        dataset_checksum="sha256:test",
    )


def _config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "clf",
        "prediction_strategy": PredictionStrategy.OVER_UNDER_CLASSIFIER,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="x.csv"),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_games=20),
        "walk_forward": WalkForwardConfig(
            test_games=20, step_games_between_tests=40, train_games=120,
            min_train_games=40, max_folds=2,
        ),
        "optuna": OptunaConfig(
            n_trials=2,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# --- configuration ----------------------------------------------------------


def test_prediction_strategy_implies_the_target_family(tmp_path):
    config = _config(tmp_path)
    assert config.target_family == TargetFamily.OVER_UNDER
    assert config.is_classifier
    assert config.target_col == OVER_LABEL_COL


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("total_points", PredictionStrategy.TOTAL_POINTS_REGRESSOR),
        ("line_error", PredictionStrategy.LINE_ERROR_REGRESSOR),
    ],
)
def test_a_legacy_config_setting_only_target_family_still_loads(
    tmp_path, family, expected
):
    """Every config and saved run written before prediction_strategy existed
    must keep working untouched.
    """
    config = ExperimentConfig(
        experiment_name="legacy",
        target_family=family,
        line_col="ODDS_TOTAL_LINE_bet365" if family == "total_points" else None,
        data=DataConfig(csv_path="x.csv"),
    )
    assert config.strategy == expected


def test_contradictory_strategy_and_family_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="implies target_family"):
        ExperimentConfig(
            experiment_name="bad",
            prediction_strategy=PredictionStrategy.OVER_UNDER_CLASSIFIER,
            target_family="total_points",
            line_col="ODDS_TOTAL_LINE_bet365",
            data=DataConfig(csv_path="x.csv"),
        )


def test_classifier_requires_a_line_because_the_label_depends_on_it(tmp_path):
    with pytest.raises(ValueError, match="line_col is required"):
        ExperimentConfig(
            experiment_name="bad",
            prediction_strategy=PredictionStrategy.OVER_UNDER_CLASSIFIER,
            data=DataConfig(csv_path="x.csv"),
        )


def test_classifier_switches_the_xgboost_objective(tmp_path):
    """Training a classifier under a regression loss would fit and produce
    numbers, just meaningless ones.
    """
    assert _config(tmp_path).optuna.objective_name == "binary:logistic"


def test_an_explicit_regression_objective_on_a_classifier_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="is a regression objective"):
        _config(
            tmp_path, optuna=OptunaConfig(objective_name="reg:absoluteerror")
        )


def test_the_label_is_excluded_from_the_features(tmp_path):
    """It is a deterministic function of TOTAL_POINTS and the line, so leaving
    it in X would hand the model the answer.
    """
    assert OVER_LABEL_COL in _config(tmp_path).exclude_cols


# --- label construction -----------------------------------------------------


def test_label_is_one_when_the_total_beats_the_line():
    df = pd.DataFrame(
        {"TOTAL_POINTS": [210.0, 190.0, 200.0], "LINE": [200.0, 200.0, 200.0]}
    )
    labelled, n_pushes = add_over_under_label(df, line_col="LINE")

    assert n_pushes == 1  # the 200 == 200 game
    assert list(labelled[OVER_LABEL_COL]) == [1, 0]


def test_pushes_are_dropped_from_training_but_counted():
    """A push has no OVER/UNDER answer, so a label would have to be invented --
    on exactly the games where the market was most precisely right.
    """
    df = pd.DataFrame(
        {"TOTAL_POINTS": [210.0, 200.0, 200.0, 180.0], "LINE": [200.0] * 4}
    )
    labelled, n_pushes = add_over_under_label(df, line_col="LINE")

    assert n_pushes == 2
    assert len(labelled) == 2


def test_all_pushes_raises_rather_than_returning_an_empty_frame():
    df = pd.DataFrame({"TOTAL_POINTS": [200.0, 200.0], "LINE": [200.0, 200.0]})
    with pytest.raises(ValueError, match="No rows left after dropping pushes"):
        add_over_under_label(df, line_col="LINE")


# --- expected value ---------------------------------------------------------


def test_expected_value_is_zero_exactly_at_break_even():
    """At -110 the break-even probability is 1/1.909091 = 52.38%."""
    odds = np.array([1.0 + 100.0 / 110.0])
    np.testing.assert_allclose(
        expected_value(np.array([1.0 / odds[0]]), odds), [0.0], atol=1e-12
    )
    assert expected_value(np.array([0.60]), odds)[0] > 0
    assert expected_value(np.array([0.50]), odds)[0] < 0


def test_classifier_bets_the_side_with_the_better_price(tmp_path):
    """With asymmetric prices the higher-EV side can be the LESS likely one --
    which is why the side is chosen on EV, not on probability.
    """
    config = _config(tmp_path)

    class _Stub:
        def predict_proba(self, X):
            # 45% OVER: less likely, but priced at 3.0 it is still the better bet.
            return np.column_stack([np.full(len(X), 0.55), np.full(len(X), 0.45)])

    stub = _Stub()
    stub.__class__ = type("S", (XGBClassifier,), {"predict_proba": _Stub.predict_proba})

    decisions = predict_decisions(
        stub,
        pd.DataFrame({"a": [1.0]}),
        config=config,
        target_line=np.array([220.0]),
        decimal_odds_over=np.array([3.0]),
        decimal_odds_under=np.array([1.2]),
    )
    # EV over = .45*3 - 1 = +0.35; EV under = .55*1.2 - 1 = -0.34.
    assert decisions.bets_over[0]
    assert decisions.selection_score[0] == pytest.approx(0.35)


def test_selection_score_defaults_to_the_edge_magnitude():
    """The regressors' behaviour must be byte-identical to before the
    selection_score parameter existed.
    """
    kwargs = {
        "predicted_edge": np.array([3.0, -1.0, 5.0]),
        "actual_total": np.array([210.0, 200.0, 190.0]),
        "line": np.array([205.0, 205.0, 205.0]),
        "min_edge": 2.0,
    }
    implicit = evaluate_betting(**kwargs)
    explicit = evaluate_betting(**kwargs, selection_score=np.abs(kwargs["predicted_edge"]))
    assert implicit.model_dump() == explicit.model_dump()
    assert implicit.n_bets == 2  # the |-1.0| bet is filtered out


# --- calibration metrics ----------------------------------------------------


def test_log_loss_and_brier_match_hand_computed_values():
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert log_loss(y, p) == pytest.approx(expected)
    assert brier_score(y, p) == pytest.approx(((0.8 - 1) ** 2 + 0.3**2) / 2)


def test_a_coin_flip_scores_the_textbook_value():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(0.693147, abs=1e-5)
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)


def test_calibration_summary_flags_overconfidence():
    """Predicting 90% on games that go over only half the time is the failure
    mode that matters: the bet rule compares to an absolute threshold.
    """
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(float)
    summary = calibration_summary(y, np.full(400, 0.9))

    assert summary.mean_bias > 0.3
    assert summary.expected_calibration_error > 0.3
    # Worse than simply predicting the base rate.
    assert summary.log_loss_improvement < 0


def test_a_perfectly_calibrated_model_has_near_zero_bias():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 0.9, size=4000)
    y = (rng.random(4000) < p).astype(float)
    summary = calibration_summary(y, p)

    assert abs(summary.mean_bias) < 0.02
    assert summary.expected_calibration_error < 0.03
    assert summary.log_loss_improvement > 0


def test_calibration_table_drops_empty_buckets_rather_than_reporting_zeros():
    y = np.array([1.0, 1.0, 0.0])
    table = calibration_table(y, np.array([0.85, 0.85, 0.85]), n_buckets=10)
    assert len(table) == 1
    assert table.iloc[0]["n"] == 3


# --- end to end -------------------------------------------------------------


@pytest.fixture
def patched_classifier(monkeypatch, tmp_path):
    config = _config(tmp_path)
    prepared = _prepared(_frame(), config)
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    return config


def test_classifier_runs_end_to_end_and_produces_probabilities(patched_classifier):
    result = run_experiment(patched_classifier)
    predictions = result.walk_forward_result.predictions

    assert "p_over" in predictions.columns
    assert predictions["p_over"].between(0.0, 1.0).all()
    assert "expected_value" in predictions.columns
    assert result.walk_forward_result.calibration is not None


def test_classifier_reports_no_point_error_metrics(patched_classifier):
    """MAE against a 0/1 label is not a points error and must never be ranked
    beside a regressor's.
    """
    result = run_experiment(patched_classifier)
    assert np.isnan(result.walk_forward_result.mae)
    assert np.isnan(result.walk_forward_result.rmse)
    assert np.isnan(result.walk_forward_result.r2)


def test_classifier_has_no_alternative_line_comparison(patched_classifier):
    """Its label was defined relative to one specific line, so the same
    prediction cannot be re-scored against a different one.
    """
    config = patched_classifier
    config.betting.comparison_line_cols = ("ODDS_TOTAL_LINE_consensus_opener",)
    assert run_experiment(config).walk_forward_result.line_comparison is None


def test_classifier_cv_betting_uses_expected_value_thresholds(patched_classifier):
    result = run_experiment(patched_classifier)
    cv = result.cv_betting

    assert cv is not None
    assert cv.calibration is not None
    assert "p_over" in cv.predictions.columns
    # Thresholds are EV, not points.
    assert list(cv.betting_sweep["min_edge"]) == list(
        patched_classifier.betting.ev_thresholds
    )


def test_classifier_fits_a_classifier_not_a_regressor(patched_classifier):
    config = patched_classifier
    config.refit.train_production_model = True
    result = run_experiment(config, save_model=True)
    assert isinstance(result.model, XGBClassifier)


def test_primary_threshold_switches_units_with_the_strategy(tmp_path):
    classifier = _config(tmp_path)
    assert primary_threshold(classifier) == classifier.betting.primary_ev_threshold

    regressor = _config(
        tmp_path,
        prediction_strategy=PredictionStrategy.TOTAL_POINTS_REGRESSOR,
        betting=BettingConfig(primary_edge_threshold=2.0),
    )
    assert primary_threshold(regressor) == 2.0


# --- characterization: the regressors must be untouched ---------------------


@pytest.mark.parametrize(
    ("strategy", "line_col", "expected_target"),
    [
        (PredictionStrategy.TOTAL_POINTS_REGRESSOR, "ODDS_TOTAL_LINE_bet365", "TOTAL_POINTS"),
        (PredictionStrategy.LINE_ERROR_REGRESSOR, None, "LINE_ERROR"),
    ],
)
def test_regressors_still_train_on_their_original_target(
    monkeypatch, tmp_path, strategy, line_col, expected_target
):
    config = _config(tmp_path, prediction_strategy=strategy, line_col=line_col)
    assert config.target_col == expected_target
    assert not config.is_classifier
    assert config.optuna.objective_name == "reg:squarederror"

    prepared = _prepared(_frame(), config)
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    result = run_experiment(config)

    # Real point-error metrics, not NaN.
    assert np.isfinite(result.walk_forward_result.mae)
    assert isinstance(result.model, XGBRegressor) or result.model is None


def test_regressor_strategies_are_unchanged_objects(tmp_path):
    """get_strategy must still return the original regressor strategies."""
    from training_pipeline.tuning import LineErrorStrategy, TotalPointsStrategy

    total = _config(tmp_path, prediction_strategy=PredictionStrategy.TOTAL_POINTS_REGRESSOR)
    assert isinstance(get_strategy(total), TotalPointsStrategy)

    line = _config(
        tmp_path,
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        line_col=None,
    )
    assert isinstance(get_strategy(line), LineErrorStrategy)


def test_regressor_edge_thresholds_are_still_in_points(tmp_path, monkeypatch):
    config = _config(
        tmp_path, prediction_strategy=PredictionStrategy.TOTAL_POINTS_REGRESSOR
    )
    prepared = _prepared(_frame(), config)
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    sweep = run_experiment(config).walk_forward_result.betting_sweep
    assert list(sweep["min_edge"]) == list(config.betting.edge_thresholds)


# --- review findings: five defects, each pinned -----------------------------


def test_outcome_derived_columns_are_excluded_for_every_strategy(tmp_path):
    """LINE_ERROR = TOTAL_POINTS - line, and the line IS a feature, so it
    reconstructs the total for a regressor and its sign alone is the
    classifier's label. Three CSVs under data/train_data ship this column, so
    this is a live hazard, not a hypothetical one.
    """
    for strategy, line_col in (
        (PredictionStrategy.TOTAL_POINTS_REGRESSOR, "ODDS_TOTAL_LINE_bet365"),
        (PredictionStrategy.LINE_ERROR_REGRESSOR, None),
        (PredictionStrategy.OVER_UNDER_CLASSIFIER, "ODDS_TOTAL_LINE_bet365"),
    ):
        config = _config(tmp_path, prediction_strategy=strategy, line_col=line_col)
        for column in ("TOTAL_POINTS", "LINE_ERROR", OVER_LABEL_COL):
            assert column in config.exclude_cols, (strategy.value, column)


def test_leak_guard_rejects_an_outcome_column_in_the_feature_matrix():
    from training_pipeline.data import assert_no_leaking_features

    with pytest.raises(ValueError, match="reached the feature matrix"):
        assert_no_leaking_features(pd.DataFrame({"FEATURE_A": [1.0], "LINE_ERROR": [2.0]}))


def test_leak_guard_keeps_the_engineered_before_rollups():
    """DIFF_FROM_LINE_*_BEFORE_* are legitimate pre-game features; a substring
    match here would silently delete hundreds of real columns.
    """
    from training_pipeline.data import assert_no_leaking_features

    assert_no_leaking_features(
        pd.DataFrame(
            {
                "DIFF_FROM_LINE_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME": [1.0],
                "ODDS_TOTAL_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME": [2.0],
            }
        )
    )


def test_classifier_betting_rules_are_post_hoc_to_the_study(tmp_path):
    """Prices and thresholds no longer select a classifier trial."""
    def _mk(odds, ev):
        return _config(
            tmp_path,
            betting=BettingConfig(
                flat_decimal_odds=odds, primary_ev_threshold=ev,
                ev_thresholds=(0.0, 0.02, 0.05),
            ),
        )

    assert _mk(1.9090909090909092, 0.0).fingerprint() == _mk(2.5, 0.05).fingerprint()
    assert _mk(2.5, 0.05).fingerprint() == _mk(2.5, 0.05).fingerprint()


def test_classifier_selection_ignores_roi_when_accuracy_ties():
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(
            value=0.690,
            user_attrs={"mean_logloss": 0.690, "mean_ou_acc": 0.54, "mean_roi": -0.9},
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            value=0.691,
            user_attrs={"mean_logloss": 0.691, "mean_ou_acc": 0.54, "mean_roi": 9.0},
        )
    )

    selected = select_best_classifier_trial_lexicographic(
        study, logloss_tolerance_abs=0.002
    )

    assert selected.number == 0


def test_pooled_classifier_selection_ignores_roi_when_accuracy_ties():
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(
            value=0.690,
            user_attrs={
                "pooled_log_loss": 0.690,
                "pooled_ou_acc": 0.54,
                "pooled_roi": -0.9,
            },
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            value=0.691,
            user_attrs={
                "pooled_log_loss": 0.691,
                "pooled_ou_acc": 0.54,
                "pooled_roi": 9.0,
            },
        )
    )

    selected = select_best_classifier_trial_lexicographic_pooled(
        study, logloss_tolerance_abs=0.002
    )

    assert selected.number == 0


def test_a_post_hoc_sweep_list_still_does_not_fork_the_study(tmp_path):
    """Changing a reporting sweep must not throw away a tuned study."""
    base = _config(
        tmp_path,
        betting=BettingConfig(ev_thresholds=(0.0, 0.02), primary_ev_threshold=0.0),
    )
    wider = _config(
        tmp_path,
        betting=BettingConfig(ev_thresholds=(0.0, 0.02, 0.09), primary_ev_threshold=0.0),
    )
    assert base.fingerprint() == wider.fingerprint()


def test_regressor_betting_settings_remain_post_hoc(tmp_path):
    """Nothing about the fix may make a regressor's study fork on a threshold."""
    def _mk(edge):
        return _config(
            tmp_path,
            prediction_strategy=PredictionStrategy.TOTAL_POINTS_REGRESSOR,
            betting=BettingConfig(
                edge_thresholds=(0.0, edge), primary_edge_threshold=edge
            ),
        )

    assert _mk(1.0).fingerprint() == _mk(2.0).fingerprint()


def test_real_prices_reach_the_side_selection_not_just_settlement(tmp_path):
    """Choosing the side on flat odds and settling on real ones would score a
    bet the model never would have placed.
    """
    from training_pipeline.decisions import collect_prices

    config = _config(
        tmp_path,
        betting=BettingConfig(over_price_col="ODDS_OVER", under_price_col="ODDS_UNDER"),
    )
    df = pd.DataFrame({"ODDS_OVER": [3.0, 2.0], "ODDS_UNDER": [1.2, 1.9]})

    over, under = collect_prices(df, config)
    np.testing.assert_allclose(over, [3.0, 2.0])
    np.testing.assert_allclose(under, [1.2, 1.9])

    # And positional selection, for callers scoring a subset.
    over_subset, _ = collect_prices(df, config, positions=np.array([1]))
    np.testing.assert_allclose(over_subset, [2.0])


def test_collect_prices_returns_none_when_unconfigured(tmp_path):
    from training_pipeline.decisions import collect_prices

    assert collect_prices(pd.DataFrame({"a": [1]}), _config(tmp_path)) == (None, None)


def test_classifier_cv_metrics_are_not_labelled_mae(patched_classifier, tmp_path):
    """The classifier's Optuna value is log loss. Filing it under cv_mae would
    report 0.69 as though it were a points error.
    """
    import json

    config = patched_classifier
    config.save_experiment_artifacts = True
    config.experiment_root_dir = tmp_path / "artifacts"
    result = run_experiment(config)

    metrics = json.loads((result.run_dir / "final_test_metrics.json").read_text())
    assert metrics["cv"]["mae"] is None
    assert metrics["cv"]["log_loss"] is not None
    # A coin flip scores 0.693; anything near it must never appear as an MAE.
    assert 0.5 < metrics["cv"]["log_loss"] < 1.0


def test_recovered_classifier_hyperparameters_report_log_loss(patched_classifier, tmp_path):
    from training_pipeline.reuse import load_run_hyperparameters

    config = patched_classifier
    config.save_experiment_artifacts = True
    config.experiment_root_dir = tmp_path / "artifacts"
    run_dir = run_experiment(config).run_dir

    recovered = load_run_hyperparameters(run_dir)
    assert recovered.cv_mae is None
    assert recovered.cv_logloss is not None
    assert "CV log loss" in recovered.to_yaml_block()
    assert "CV MAE" not in recovered.to_yaml_block()


def test_promoting_a_classifier_is_refused_with_an_explanation(patched_classifier, tmp_path):
    """The serving path reads model.predict() as a points total, so a promoted
    classifier would not crash -- it would quietly treat class "1" as a
    one-point game.
    """
    from training_pipeline.promote import train_production_model_from_run

    config = patched_classifier
    config.save_experiment_artifacts = True
    config.experiment_root_dir = tmp_path / "artifacts"
    run_dir = run_experiment(config).run_dir

    with pytest.raises(ValueError, match="cannot be promoted yet"):
        train_production_model_from_run(run_dir)


# --- the search space must be measured in the objective's own units ---------


def _space_config(tmp_path, **overrides) -> ExperimentConfig:
    """A config that does NOT override optuna.search_space.

    The shared _config helper shrinks the space (n_estimators=8) so the suite
    runs fast -- which makes it useless here: a shrunken space is not the
    default, so the classifier swap never fires and the regressor's space is not
    the default either. Testing the search space needs a config that leaves it
    alone.
    """
    kwargs = {
        "experiment_name": "space",
        "prediction_strategy": PredictionStrategy.OVER_UNDER_CLASSIFIER,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="x.csv"),
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_classifier_gets_a_hessian_scaled_search_space(tmp_path):
    """The defaults are reasoned in squared-error terms, where the hessian is 1
    per sample. Logistic loss has hessian ~0.25 at a 50% base rate, so the same
    numbers mean something else: min_child_weight 20-250 becomes 80-1000
    SAMPLES per leaf, and a gamma floor of 1.0 exceeds the ~0.49 chance-level
    split gain, pruning every split.
    """
    from training_pipeline.config import CLASSIFIER_SEARCH_SPACE, SearchSpaceConfig

    classifier = _space_config(tmp_path)
    regressor = _space_config(
        tmp_path, prediction_strategy=PredictionStrategy.TOTAL_POINTS_REGRESSOR
    )

    assert classifier.optuna.search_space == CLASSIFIER_SEARCH_SPACE
    assert regressor.optuna.search_space == SearchSpaceConfig()

    # The two ratios the scaling is derived from, rounded to round numbers:
    # hessian 0.25 puts min_child_weight at ~1/4 (62.5 -> 60), and the measured
    # 387x gain-scale ratio puts gamma at ~1/387 (1.29 -> 2.0 / 0.0026 -> 0.002).
    space = classifier.optuna.search_space
    assert space.min_child_weight.high == 60.0
    assert space.gamma.high == 2.0
    assert space.gamma.low == 0.002


def test_only_the_scale_dependent_parameters_move(tmp_path):
    """max_depth, subsample, colsample_bytree and learning_rate are
    dimensionless here; changing them would be unjustified churn.
    """
    from training_pipeline.config import SearchSpaceConfig

    classifier = _space_config(tmp_path).optuna.search_space
    regression = SearchSpaceConfig()

    for unchanged in ("max_depth", "subsample", "colsample_bytree", "learning_rate"):
        assert getattr(classifier, unchanged) == getattr(regression, unchanged), unchanged
    for rescaled in ("min_child_weight", "gamma", "reg_alpha", "reg_lambda"):
        assert getattr(classifier, rescaled) != getattr(regression, rescaled), rescaled


def test_a_deliberate_search_space_is_never_overridden(tmp_path):
    """The swap means "this space was inherited, not chosen". Anyone who states
    one -- including deliberately reusing the regression ranges -- keeps it.
    """
    from training_pipeline.config import FloatRange, SearchSpaceConfig

    chosen = SearchSpaceConfig(gamma=FloatRange(low=7.0, high=9.0, log=True))
    config = _space_config(tmp_path, optuna=OptunaConfig(search_space=chosen))
    assert config.optuna.search_space.gamma.low == 7.0


def test_regressors_are_completely_unaffected(tmp_path):
    from training_pipeline.config import SearchSpaceConfig

    for strategy, line_col in (
        (PredictionStrategy.TOTAL_POINTS_REGRESSOR, "ODDS_TOTAL_LINE_bet365"),
        (PredictionStrategy.LINE_ERROR_REGRESSOR, None),
    ):
        config = _space_config(
            tmp_path, prediction_strategy=strategy, line_col=line_col
        )
        assert config.optuna.search_space == SearchSpaceConfig(), strategy.value


def test_the_two_spaces_do_not_share_a_study(tmp_path):
    """The search space is part of what a trial MEANS, so a classifier run must
    not resume a study tuned under regression-scaled ranges.
    """
    classifier = _space_config(tmp_path)
    regressor = _space_config(
        tmp_path, prediction_strategy=PredictionStrategy.TOTAL_POINTS_REGRESSOR
    )
    assert classifier.fingerprint() != regressor.fingerprint()


def test_the_scaled_space_can_actually_fit_a_signal(tmp_path):
    """The regression space's upper half drives a classifier to a constant
    prediction. This is the regression test for that: given a planted signal,
    a model built at the CEILING of the classifier space must still separate
    games, where the regression ceiling produces a flat 0.504.
    """
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    n = 2500  # the campaign's reference window
    X = pd.DataFrame(rng.normal(size=(n, 60)).astype(np.float32))
    logit = 0.30 * X[0] + 0.25 * X[1] + 0.20 * X[2]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))

    space = _space_config(tmp_path).optuna.search_space
    shared = dict(
        objective="binary:logistic", tree_method="hist", n_estimators=200,
        learning_rate=0.05, random_state=16, n_jobs=2, verbosity=0,
    )
    ceiling = XGBClassifier(
        min_child_weight=space.min_child_weight.high,
        gamma=space.gamma.high,
        reg_lambda=space.reg_lambda.high,
        **shared,
    ).fit(X, y)
    spread = np.ptp(ceiling.predict_proba(X)[:, 1])
    assert spread > 0.10, (
        f"even at its most conservative the classifier space must separate "
        f"games; got a probability spread of {spread:.3f}"
    )
