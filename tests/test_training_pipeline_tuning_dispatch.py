import numpy as np
import optuna
import pandas as pd
import pytest

from training_pipeline.config import (
    DataConfig,
    ExperimentConfig,
    OptunaConfig,
    SampleWeightConfig,
    SearchSpaceConfig,
    TargetFamily,
)
from training_pipeline.tuning import (
    LineErrorStrategy,
    TotalPointsStrategy,
    get_strategy,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _total_points_config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "dispatch_test",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="data/train_data/example.csv"),
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _line_error_config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "dispatch_test_le",
        "target_family": TargetFamily.LINE_ERROR,
        "data": DataConfig(csv_path="data/train_data/example.csv"),
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_get_strategy_returns_total_points_strategy_with_configured_line_col():
    config = _total_points_config()
    strategy = get_strategy(config)
    assert isinstance(strategy, TotalPointsStrategy)
    assert strategy.line_col == "ODDS_TOTAL_LINE_bet365"


def test_get_strategy_returns_line_error_strategy_with_configured_sample_weight():
    sw = SampleWeightConfig(enabled=True, lambda_=0.01)
    config = _line_error_config(sample_weight=sw)
    strategy = get_strategy(config)
    assert isinstance(strategy, LineErrorStrategy)
    assert strategy.sample_weight == sw


def _tiny_problem(n: int = 40):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": rng.uniform(200, 240, n),
            "FEATURE_A": rng.normal(size=n),
        }
    )
    y = pd.Series(X["ODDS_TOTAL_LINE_bet365"].to_numpy() + rng.normal(0, 10, n))
    splits = [(np.arange(0, 30), np.arange(30, n))]
    dates = pd.Series(pd.date_range("2026-01-01", periods=n, freq="D"))
    return X, y, splits, dates


def _fast(config_factory, **overrides):
    """One trial, tiny forest -- we are testing wiring, not model quality."""
    return config_factory(
        optuna=OptunaConfig(
            n_trials=1,
            search_space=SearchSpaceConfig(n_estimators=10, early_stopping_rounds=5),
        ),
        **overrides,
    )


def test_total_points_tuning_scores_folds_against_the_configured_line(monkeypatch):
    """The total_points family must score folds with the betting line, so the
    configured line_col has to reach evaluate_fold_total_points.
    """
    import training_pipeline.tuning as tuning_module

    seen: dict = {}
    real = tuning_module._total_points.evaluate_fold_total_points

    def spy(model, X_valid, y_valid, line_col, *, fold, n_train):
        seen["line_col"] = line_col
        return real(model, X_valid, y_valid, line_col, fold=fold, n_train=n_train)

    monkeypatch.setattr(
        tuning_module._total_points, "evaluate_fold_total_points", spy
    )

    X, y, splits, _ = _tiny_problem()
    config = _fast(_total_points_config)
    study = get_strategy(config).tune(X=X, y=y, splits=splits, config=config)

    assert seen["line_col"] == "ODDS_TOTAL_LINE_bet365"
    assert len(study.trials) == 1


def test_line_error_tuning_uses_the_error_fold_evaluator(monkeypatch):
    """The line_error family scores off the sign of the predicted error and
    must never be routed through the line-based evaluator.
    """
    import training_pipeline.tuning as tuning_module

    called = {"error_line": 0, "total_points": 0}
    real = tuning_module._error_line.evaluate_fold_error_line

    def spy(model, X_valid, y_valid, *, fold, n_train):
        called["error_line"] += 1
        return real(model, X_valid, y_valid, fold=fold, n_train=n_train)

    def forbidden(*args, **kwargs):
        called["total_points"] += 1
        raise AssertionError("line_error must not use the total_points evaluator")

    monkeypatch.setattr(tuning_module._error_line, "evaluate_fold_error_line", spy)
    monkeypatch.setattr(
        tuning_module._total_points, "evaluate_fold_total_points", forbidden
    )

    X, y, splits, _ = _tiny_problem()
    config = _fast(_line_error_config)
    get_strategy(config).tune(X=X, y=y, splits=splits, config=config)

    assert called["error_line"] == 1
    assert called["total_points"] == 0


def test_recency_weighting_is_applied_only_when_enabled(monkeypatch):
    import training_pipeline.tuning as tuning_module

    calls = {"n": 0}
    real = tuning_module.build_recency_sample_weights

    def spy(dates, **kwargs):
        calls["n"] += 1
        return real(dates, **kwargs)

    monkeypatch.setattr(tuning_module, "build_recency_sample_weights", spy)
    X, y, splits, dates = _tiny_problem()

    disabled = _fast(_line_error_config)
    get_strategy(disabled).tune(X=X, y=y, splits=splits, config=disabled, dates=dates)
    assert calls["n"] == 0

    enabled = _fast(
        _line_error_config, sample_weight=SampleWeightConfig(enabled=True, lambda_=0.01)
    )
    get_strategy(enabled).tune(X=X, y=y, splits=splits, config=enabled, dates=dates)
    assert calls["n"] == 1


def test_tuned_lambda_is_recorded_on_the_trial():
    """When the decay rate is tuned it becomes part of the trial, so downstream
    refitting can recover it.
    """
    X, y, splits, dates = _tiny_problem()
    config = _fast(
        _line_error_config,
        sample_weight=SampleWeightConfig(
            enabled=True,
            tune_lambda=True,
            lambda_bounds=(1e-4, 0.02),
            # Isolate the always-weighted path; the opt-out is covered in
            # test_training_pipeline_search_space.py.
            allow_unweighted=False,
        ),
    )
    study = get_strategy(config).tune(
        X=X, y=y, splits=splits, config=config, dates=dates
    )

    trial = study.trials[0]
    assert "sample_weight_lambda" in trial.params
    assert 1e-4 <= trial.params["sample_weight_lambda"] <= 0.02
    assert trial.user_attrs["sample_weight_lambda"] == trial.params["sample_weight_lambda"]


def test_objective_records_the_user_attrs_downstream_helpers_read():
    """select_best_trial_lexicographic and summarize_optuna_trials depend on
    these keys; losing one silently breaks trial selection.
    """
    X, y, splits, _ = _tiny_problem()
    config = _fast(_total_points_config)
    study = get_strategy(config).tune(X=X, y=y, splits=splits, config=config)

    attrs = study.trials[0].user_attrs
    for key in (
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "mean_ou_acc",
        "mean_ou_acc_edge_2",
        "mean_best_iteration",
        "median_best_iteration",
        "fold_metrics",
    ):
        assert key in attrs, f"missing user_attr: {key}"


# ---------------------------------------------------------------------------
# the tie band: the rule that decides whether MAE or OU accuracy picks the model
# ---------------------------------------------------------------------------


def _tie(values, **overrides):
    from training_pipeline.config import TieTolerancePolicy
    from training_pipeline.tuning import resolve_tie_tolerance

    kwargs = {
        "policy": TieTolerancePolicy.QUANTILE,
        "fixed_abs": 0.10,
        "fixed_pct": None,
        "max_fraction": 0.10,
        "floor": 0.001,
        "cap": 0.10,
        "warn_fraction": 0.25,
    }
    kwargs.update(overrides)
    return resolve_tie_tolerance(values, **kwargs)


#: Cell A of public_betting_tradeoff_2026_08, to scale: 60 completed trials
#: whose entire pooled-MAE spread was 0.0987 on a base of 14.36. Under the old
#: fixed 0.10 tolerance, 58 of the 60 were candidates -- the primary metric
#: ranked nothing and pooled OU accuracy chose the model.
_PACKED_FRONTIER = [14.3591 + 0.0987 * i / 59 for i in range(60)]


def test_the_old_fixed_tolerance_swallowed_almost_the_whole_study():
    """Not a hypothetical: this is the measured failure the redesign is for."""
    from training_pipeline.config import TieTolerancePolicy

    result = _tie(_PACKED_FRONTIER, policy=TieTolerancePolicy.FIXED)

    assert result.tolerance == pytest.approx(0.10)
    assert result.n_candidates == 60
    assert result.fraction == pytest.approx(1.0)
    assert result.warning is not None


def test_the_quantile_rule_keeps_the_tie_break_to_a_genuine_tie():
    """Same trials, same data: the band collapses to the gap spanning the best
    tenth, so OU accuracy breaks a tie instead of making the decision."""
    result = _tie(_PACKED_FRONTIER)

    assert result.n_candidates == 7
    assert result.fraction <= 0.15
    # An order of magnitude tighter than the 0.10 it replaces. (On the real
    # cell A distribution, which is not uniformly spaced, it resolved to
    # 0.0055 and admitted the same 7 trials.)
    assert result.tolerance < 0.02
    assert result.warning is None


def test_a_discriminating_study_does_not_widen_the_band():
    """The inversion a dispersion rule would get wrong. When MAE separates the
    trials well, the band must NOT expand to match the spread -- a rank rule
    admits the same small share either way."""
    spread_out = [14.0 + i * 0.5 for i in range(60)]

    packed = _tie(_PACKED_FRONTIER)
    spread = _tie(spread_out)

    assert spread.n_candidates <= packed.n_candidates
    assert spread.fraction <= 0.15


def test_the_cap_is_a_hard_maximum():
    """However the data falls, the band can never be more permissive than the
    constant it replaced."""
    result = _tie([14.0, 99.0, 99.0, 99.0], max_fraction=1.0, cap=0.05)

    assert result.tolerance == pytest.approx(0.05)
    assert result.n_candidates == 1


def test_the_floor_admits_trials_separated_by_numerical_dust():
    """Ranking on a 1e-9 MAE difference is ranking on noise. The floor exists
    to stop that, and it is allowed to admit more than max_fraction."""
    values = [14.0 + i * 1e-9 for i in range(60)]

    result = _tie(values, floor=0.01)

    assert result.tolerance == pytest.approx(0.01)
    assert result.n_candidates == 60


def test_exact_ties_are_never_split_by_arrival_order():
    """Trials genuinely equal on the primary metric all reach the tie-break;
    cutting the set at max_fraction would break them by trial number, which is
    not a criterion."""
    values = [14.0] * 8 + [15.0] * 2

    result = _tie(values, max_fraction=0.10)

    assert result.n_candidates == 8


def test_a_smoke_sized_study_does_not_raise_the_diagnostic():
    """'2 of 2 trials tied' is arithmetic. Warning there would teach readers to
    ignore the warning that matters."""
    assert _tie([14.0, 14.0]).warning is None
    assert _tie([14.0] * 40).warning is not None


def test_the_band_and_its_diagnostics_are_recorded():
    """A band that swallowed the study is a fact about the run: it must be in
    the artifacts, not only in a log line."""
    summary = _tie(_PACKED_FRONTIER).summary()

    assert summary["tie_policy"] == "quantile"
    assert summary["tie_n_completed"] == 60
    assert summary["tie_n_candidates"] == 7
    assert summary["tie_candidate_fraction"] == pytest.approx(7 / 60)
    assert summary["tie_tolerance"] > 0


def test_non_finite_trial_values_are_ignored_not_ranked():
    result = _tie([float("nan"), 14.0, 14.05, float("inf")])

    assert result.best == pytest.approx(14.0)
    assert result.n_completed == 2
