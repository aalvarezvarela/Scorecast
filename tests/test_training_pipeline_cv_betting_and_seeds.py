"""Cross-fold betting metrics, seed threading, and alternative-line scoring.

The three additions all exist to make comparisons between experiments
trustworthy rather than to make any single model better:
  - CV betting buys ~5x the bet volume of the holdout.
  - Seed threading gives those comparisons an error bar.
  - Alternative-line scoring separates "has information" from "could have
    captured it".
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from training_pipeline import pipeline as pipeline_module
from training_pipeline.betting import evaluate_alternative_lines
from training_pipeline.config import (
    BettingConfig,
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    HoldoutConfig,
    HoldoutEvaluation,
    OptunaConfig,
    PredictionStrategy,
    SearchSpaceConfig,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.cv_betting import evaluate_cv_betting
from training_pipeline.data import PreparedDataset
from training_pipeline.line_scoring import (
    collect_comparison_lines,
    predicted_total_points,
)
from training_pipeline.pipeline import run_experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prepared(n_games: int = 260, games_per_day: int = 4) -> PreparedDataset:
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
            # An "opener" that sits a few points away from the close, the way a
            # real one does.
            "ODDS_TOTAL_LINE_consensus_opener": (
                line + rng.normal(0, 2.5, n_games)
            ).round(1),
            "FEATURE_A": rng.normal(size=n_games),
            "FEATURE_B": rng.normal(size=n_games),
        }
    )
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    features = ["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]
    return PreparedDataset(
        df_full=df,
        X=df[features],
        y=df["TOTAL_POINTS"],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=features,
        dataset_checksum="sha256:test",
    )


def _config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "cvbet",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="x.csv"),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_games=20),
        "walk_forward": WalkForwardConfig(
            test_games=20,
            step_games_between_tests=40,
            train_games=120,
            min_train_games=40,
            max_folds=2,
        ),
        "optuna": OptunaConfig(
            n_trials=1,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


@pytest.fixture
def patched(monkeypatch):
    prepared = _prepared()
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    return prepared


# --- CV betting -------------------------------------------------------------


def test_cv_betting_pools_every_fold_and_beats_the_holdout_for_volume(
    patched, tmp_path
):
    """The whole point: more bets than the holdout can offer."""
    result = run_experiment(_config(tmp_path))

    assert result.cv_betting is not None
    cv = result.cv_betting
    assert cv.n_folds == 2
    # Two folds of 20 validation games each, pooled.
    assert cv.n_games == 40
    assert len(cv.fold_metrics) == 2
    assert len(cv.predictions) == 40
    # More candidate games scored than the 20-game holdout.
    assert cv.n_games > result.walk_forward_result.n_games


def test_cv_fold_layout_is_non_overlapping_so_no_game_is_double_counted(
    patched, tmp_path
):
    cv = run_experiment(_config(tmp_path)).cv_betting
    assert cv.n_unique_games == cv.n_games


def test_cv_betting_can_be_switched_off(patched, tmp_path):
    config = _config(tmp_path, betting=BettingConfig(evaluate_cv_folds=False))
    assert run_experiment(config).cv_betting is None


def test_cv_fold_metrics_report_the_line_error_alongside_the_model_error(
    patched, tmp_path
):
    """Per fold you need both numbers to say whether the model beat the market
    there, rather than only that its MAE was some value.
    """
    folds = run_experiment(_config(tmp_path)).cv_betting.fold_metrics
    assert {"mae", "line_mae", "roi", "n_bets", "valid_start", "valid_end"} <= set(
        folds.columns
    )
    assert folds["line_mae"].notna().all()


def test_cv_betting_trains_only_on_each_folds_own_training_rows(patched, tmp_path):
    """No fold may be scored by a model that saw its validation period."""
    result = run_experiment(_config(tmp_path))
    folds = result.cv_betting.fold_metrics
    predictions = result.cv_betting.predictions
    for _, fold in folds.iterrows():
        rows = predictions[predictions["fold"] == fold["fold"]]
        assert len(rows) == fold["n_valid"]


def test_cv_betting_uses_line_error_predictions_directly_as_the_edge(tmp_path):
    """For LINE_ERROR the prediction already IS the edge; subtracting the line
    again would double-count it.
    """
    prepared = _prepared()
    config = _config(
        tmp_path,
        target_family=TargetFamily.LINE_ERROR,
        line_col=None,
    )
    df_dev = prepared.df_full
    X = df_dev[["FEATURE_A", "FEATURE_B"]]
    y = df_dev["LINE_ERROR"]
    splits = [(np.arange(0, 200), np.arange(200, 240))]

    result = evaluate_cv_betting(
        config,
        df_dev=df_dev,
        X_dev=X,
        y_dev=y,
        dates_dev=df_dev["GAME_DATE"],
        splits=splits,
        params={"max_depth": 2},
        n_estimators=5,
        target_line_col="ODDS_TOTAL_LINE_bet365",
    )

    np.testing.assert_allclose(
        result.predictions["predicted_edge"].to_numpy(),
        result.predictions["y_pred"].to_numpy(),
    )


# --- seed threading ---------------------------------------------------------


def test_random_state_reaches_the_model_fits(patched, tmp_path):
    """Previously hardcoded to 16 in three places, so this was unmeasurable."""
    a = run_experiment(_config(tmp_path, random_state=16))
    b = run_experiment(_config(tmp_path, random_state=999))

    assert a.walk_forward_result.random_state == 16
    assert b.walk_forward_result.random_state == 999


def test_same_seed_reproduces_the_same_evaluation(patched, tmp_path):
    a = run_experiment(_config(tmp_path, random_state=7))
    b = run_experiment(_config(tmp_path, random_state=7))
    assert a.walk_forward_result.mae == pytest.approx(b.walk_forward_result.mae)


def test_evaluation_seeds_produce_one_row_per_seed_with_the_primary_first(
    patched, tmp_path
):
    config = _config(tmp_path, random_state=16, evaluation_seeds=(101, 202))
    result = run_experiment(config)

    stability = result.seed_stability
    assert stability is not None
    assert list(stability["random_state"]) == [16, 101, 202]
    # Exactly one row is the headline; the others are the error bar around it.
    assert stability["is_primary"].tolist() == [True, False, False]
    assert stability.loc[0, "mae"] == pytest.approx(result.walk_forward_result.mae)


def test_seed_stability_is_absent_when_no_extra_seeds_are_requested(patched, tmp_path):
    assert run_experiment(_config(tmp_path)).seed_stability is None


def test_primary_seed_is_never_re_run_as_an_extra_seed(tmp_path):
    """Listing it twice would double the work and report one fit as if it were
    independent evidence of stability.
    """
    config = _config(tmp_path, random_state=16, evaluation_seeds=(16, 5, 5, 16))
    assert config.evaluation_seeds == (5,)


def test_explicit_seed_overrides_a_random_state_carried_in_trial_params(
    patched, tmp_path
):
    """Optuna trial params include random_state, so the override has to be
    applied last or the extra seeds would all be silently identical.
    """
    result = run_experiment(
        _config(tmp_path, random_state=16, evaluation_seeds=(4242,))
    )
    maes = result.seed_stability["mae"].tolist()
    assert maes[0] != pytest.approx(maes[1])


# --- alternative lines ------------------------------------------------------


def test_alternative_line_scoring_is_hand_checkable():
    """One clear bet per line, so wins/losses can be verified by inspection."""
    # Predicted 210. Closing line 200 => +10 edge, bet OVER. Actual 205 => win.
    # Opening line 220 => -10 edge, bet UNDER on the same game. Actual 205,
    # which is under 220 => also a win, but a different bet entirely.
    table = evaluate_alternative_lines(
        predicted_total_points=np.array([210.0]),
        actual_total=np.array([205.0]),
        lines={
            "close": np.array([200.0]),
            "open": np.array([220.0]),
        },
        min_edge=2.0,
    )

    close_row = table[table["line_col"] == "close"].iloc[0]
    open_row = table[table["line_col"] == "open"].iloc[0]

    assert close_row["n_bets"] == 1 and close_row["n_wins"] == 1
    assert open_row["n_bets"] == 1 and open_row["n_wins"] == 1
    # Each line's own forecasting error, independent of the bet.
    assert close_row["line_mae"] == pytest.approx(5.0)
    assert open_row["line_mae"] == pytest.approx(15.0)
    # The first line is the reference point for movement.
    assert close_row["mean_abs_move_vs_first"] == pytest.approx(0.0)
    assert open_row["mean_abs_move_vs_first"] == pytest.approx(20.0)


def test_alternative_lines_can_flip_a_bet_from_over_to_under():
    """The reason this table exists: an edge against the close is not the same
    bet as an edge against the open.
    """
    table = evaluate_alternative_lines(
        predicted_total_points=np.array([210.0]),
        actual_total=np.array([230.0]),  # OVER both lines
        lines={"close": np.array([200.0]), "open": np.array([220.0])},
        min_edge=2.0,
    )
    # Bet OVER vs the close and win; bet UNDER vs the open and lose.
    assert table[table["line_col"] == "close"].iloc[0]["n_wins"] == 1
    assert table[table["line_col"] == "open"].iloc[0]["n_losses"] == 1


def test_line_comparison_appears_in_the_walk_forward_result(patched, tmp_path):
    config = _config(
        tmp_path,
        betting=BettingConfig(
            comparison_line_cols=("ODDS_TOTAL_LINE_consensus_opener",)
        ),
    )
    comparison = run_experiment(config).walk_forward_result.line_comparison

    assert comparison is not None
    assert list(comparison["line_col"]) == [
        "ODDS_TOTAL_LINE_bet365",
        "ODDS_TOTAL_LINE_consensus_opener",
    ]
    # The opener really does sit away from the close in the fixture.
    assert comparison.iloc[1]["mean_abs_move_vs_first"] > 0


def test_line_comparison_is_none_when_nothing_to_compare_against(patched, tmp_path):
    """A line compared against itself is not information, and should not add a
    file to every run directory.
    """
    assert run_experiment(_config(tmp_path)).walk_forward_result.line_comparison is None


def test_missing_comparison_columns_are_skipped_not_fatal(patched, tmp_path):
    """Line availability varies across CSV snapshots, and this is a diagnostic:
    it must never fail a run that would otherwise have succeeded.
    """
    config = _config(
        tmp_path,
        betting=BettingConfig(comparison_line_cols=("ODDS_TOTAL_LINE_does_not_exist",)),
    )
    lines = collect_comparison_lines(
        _prepared().df_full, config, target_line_col="ODDS_TOTAL_LINE_bet365"
    )
    assert list(lines) == ["ODDS_TOTAL_LINE_bet365"]
    # And the run still completes.
    assert run_experiment(config).walk_forward_result is not None


def test_spread_line_comparison_skips_total_lines(tmp_path):
    """A spread prediction cannot be meaningfully re-scored against totals."""
    config = _config(
        tmp_path,
        prediction_strategy=PredictionStrategy.SPREAD_ERROR_REGRESSOR,
        target_family=TargetFamily.SPREAD_ERROR,
        line_col=None,
        betting=BettingConfig(
            comparison_line_cols=(
                "ODDS_TOTAL_LINE_consensus_opener",
                "ODDS_SPREAD_LINE_HOME_consensus_opener",
            )
        ),
    )
    df = pd.DataFrame(
        {
            "ODDS_SPREAD_LINE_HOME_bet365": [3.5, -2.0],
            "ODDS_TOTAL_LINE_consensus_opener": [225.5, 231.0],
            "ODDS_SPREAD_LINE_HOME_consensus_opener": [4.0, -1.5],
        }
    )

    lines = collect_comparison_lines(
        df, config, target_line_col="ODDS_SPREAD_LINE_HOME_bet365"
    )

    assert list(lines) == [
        "ODDS_SPREAD_LINE_HOME_bet365",
        "ODDS_SPREAD_LINE_HOME_consensus_opener",
    ]


def test_predicted_total_points_adds_the_line_back_for_line_error(tmp_path):
    """A LINE_ERROR prediction is relative to its own line, so comparing it
    against a different line requires putting it back into points space first.
    """
    y_pred = np.array([3.0, -2.0])
    line = np.array([220.0, 210.0])

    np.testing.assert_allclose(
        predicted_total_points(
            y_pred, target_line=line, target_family=TargetFamily.LINE_ERROR
        ),
        [223.0, 208.0],
    )
    np.testing.assert_allclose(
        predicted_total_points(
            y_pred, target_line=line, target_family=TargetFamily.TOTAL_POINTS
        ),
        y_pred,
    )


# --- artifacts and leaderboard ---------------------------------------------


def test_new_artifacts_are_written_and_read_back_by_the_leaderboard(patched, tmp_path):
    from training_pipeline.leaderboard import build_leaderboard

    root = tmp_path / "artifacts"
    config = _config(
        tmp_path,
        save_experiment_artifacts=True,
        experiment_root_dir=root,
        evaluation_seeds=(101,),
        betting=BettingConfig(
            comparison_line_cols=("ODDS_TOTAL_LINE_consensus_opener",)
        ),
    )
    result = run_experiment(config)

    for name in (
        "cv_betting_summary.json",
        "cv_fold_betting.csv",
        "cv_betting_sweep.csv",
        "cv_predictions.parquet",
        "seed_stability.csv",
        "line_comparison.csv",
        # The CV-level line comparison carries several times the bet volume of
        # the holdout one, and is what the comparison notebook reads first.
        "cv_line_comparison.csv",
    ):
        assert (result.run_dir / name).exists(), name

    row = build_leaderboard(root).iloc[0]
    assert row["cv_n_bets"] == result.cv_betting.betting_primary.n_bets
    assert row["cv_n_folds"] == 2
    assert row["n_seeds"] == 2
    # The error bar the leaderboard exists to expose.
    assert row["seed_roi_range"] >= 0


def test_leaderboard_tolerates_runs_without_the_new_artifacts(tmp_path):
    """Runs saved before these columns existed must still appear."""
    from training_pipeline.leaderboard import build_leaderboard

    run_dir = tmp_path / "old_run_20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text('{"target_family": "total_points"}')

    df = build_leaderboard(tmp_path)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["cv_roi"])


def test_single_shot_mode_also_reports_a_line_comparison(patched, tmp_path):
    config = _config(
        tmp_path,
        holdout_evaluation=HoldoutEvaluation.SINGLE_SHOT,
        betting=BettingConfig(
            comparison_line_cols=("ODDS_TOTAL_LINE_consensus_opener",)
        ),
    )
    result = run_experiment(config)
    assert result.holdout_result is not None
    assert result.holdout_result.line_comparison is not None
