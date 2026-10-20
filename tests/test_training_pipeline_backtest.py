import numpy as np
import pandas as pd
import pytest

from training_pipeline.backtest import run_daily_backtest, xgb_params_from_trial
from training_pipeline.config import (
    BacktestConfig,
    BettingConfig,
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    SampleWeightConfig,
    TargetFamily,
)
from training_pipeline.data import PreparedDataset


class _FrozenTrial:
    """Minimal stand-in for optuna.trial.FrozenTrial."""

    def __init__(self, params, user_attrs=None):
        self.params = params
        self.user_attrs = user_attrs or {}


def _synthetic_prepared(n_games: int = 60, games_per_day: int = 3) -> PreparedDataset:
    """A small, fully deterministic dataset spanning several game-days."""
    rng = np.random.default_rng(0)
    n_days = n_games // games_per_day
    dates = np.repeat(
        pd.date_range("2026-01-01", periods=n_days, freq="D").to_numpy(), games_per_day
    )
    line = rng.uniform(200, 240, size=n_games).round(1)
    total_points = line + rng.normal(0, 12, size=n_games).round(1)

    df = pd.DataFrame(
        {
            "GAME_DATE": dates,
            "SEASON_YEAR": 2025,
            "TOTAL_POINTS": total_points,
            "ODDS_TOTAL_LINE_bet365": line,
            "FEATURE_A": rng.normal(size=n_games),
            "FEATURE_B": rng.normal(size=n_games),
        }
    )
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]

    feature_names = ["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]
    return PreparedDataset(
        df_full=df,
        X=df[feature_names],
        y=df["TOTAL_POINTS"],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=feature_names,
    )


def _config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "backtest_test",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="x.csv"),
        "cleaning": CleaningConfig(verbose=0),
        "backtest": BacktestConfig(test_games=15, n_estimators=5, show_progress=False),
        "betting": BettingConfig(primary_edge_threshold=0.0),
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_backtest_predicts_every_game_in_the_window():
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    assert result.n_games == 15
    assert result.n_days == 5  # 15 games / 3 per day
    assert len(result.predictions) == 15
    assert len(result.daily_results) == 5


def test_backtest_never_trains_on_a_future_game():
    """The core guarantee: for each prediction day, every training row must
    come from a strictly earlier date.
    """
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    for _, row in result.daily_results.iterrows():
        assert row["train_end_date"] < row["date"], (
            f"day {row['date']} trained on data up to {row['train_end_date']}"
        )


def test_training_set_grows_as_backtest_days_are_played():
    """Completed backtest days must become training data for later days --
    that is what makes this a simulation of daily retraining.
    """
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    train_sizes = result.daily_results["train_n_games"].tolist()
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[-1] > train_sizes[0]
    # Each additional day should add that day's games to history.
    assert train_sizes[1] - train_sizes[0] == 3


def test_rolling_train_window_caps_the_training_set():
    prepared = _synthetic_prepared()
    config = _config(
        backtest=BacktestConfig(
            test_games=15, n_estimators=5, train_games=20, show_progress=False
        )
    )
    result = run_daily_backtest(config, prepared=prepared)

    assert result.daily_results["train_n_games"].max() <= 20


def test_backtest_reports_baseline_and_betting_metrics():
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    assert result.baseline.n_games == 15
    assert result.betting_primary.n_candidates == 15
    assert not result.betting_sweep.empty
    summary = result.summary()
    assert summary["n_games"] == 15
    assert "roi" in summary and "baseline_mae" in summary


def test_total_points_edge_is_prediction_minus_line():
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    expected = result.predictions["y_pred"] - result.predictions["target_line"]
    assert result.predictions["predicted_edge"].to_numpy() == pytest.approx(
        expected.to_numpy()
    )


def test_line_error_edge_is_the_prediction_itself():
    prepared = _synthetic_prepared()
    config = _config(target_family=TargetFamily.LINE_ERROR, line_col=None)
    result = run_daily_backtest(config, prepared=prepared)

    assert result.predictions["predicted_edge"].to_numpy() == pytest.approx(
        result.predictions["y_pred"].to_numpy()
    )
    # y_true must be the line error, not the raw total.
    assert result.predictions["y_true"].abs().max() < 100


def test_predictions_carry_the_actual_total_for_settlement():
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    merged = result.predictions
    assert (merged["TOTAL_POINTS"] > 100).all()
    # For a TOTAL_POINTS run, y_true is the actual total.
    assert merged["y_true"].to_numpy() == pytest.approx(merged["TOTAL_POINTS"].to_numpy())


def test_xgb_params_from_trial_separates_sample_weight_lambda():
    """sample_weight_lambda is a training-protocol parameter, not an XGBoost
    one -- feeding it to XGBRegressor would be silently ignored.
    """
    trial = _FrozenTrial(
        params={"max_depth": 3, "learning_rate": 0.05, "sample_weight_lambda": 0.004},
        user_attrs={"median_best_iteration": 120},
    )
    params, n_estimators, lambda_ = xgb_params_from_trial(trial)

    assert "sample_weight_lambda" not in params
    assert params == {"max_depth": 3, "learning_rate": 0.05}
    assert n_estimators == 120
    assert lambda_ == 0.004


def test_backtest_applies_recency_weighting_when_configured():
    """A weighted run must complete and produce the same shape as unweighted."""
    prepared = _synthetic_prepared()
    config = _config(
        target_family=TargetFamily.LINE_ERROR,
        line_col=None,
        sample_weight=SampleWeightConfig(enabled=True, lambda_=0.01),
    )
    result = run_daily_backtest(config, prepared=prepared)

    assert result.sample_weight_lambda == 0.01
    assert result.n_games == 15


# ---------------------------------------------------------------------------
# the "line + its historical drift" null
# ---------------------------------------------------------------------------


def test_backtest_reports_a_drift_null_distinct_from_the_model():
    """The walk-forward path used to have no bias-corrected null of its own.
    run_experiment filled the gap by falling back to the MODEL's own metrics,
    so roi_vs_bias_baseline came out exactly 0.00 on every daily_walk_forward
    run -- a model always ties itself. Nothing errored; the number was simply
    meaningless."""
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    null = result.baseline_bias_corrected_betting
    assert null.n_candidates == 15
    # A constant edge is not selective: it stands on every candidate game.
    assert null.n_bets == 15
    # And it is a genuinely separate measurement, not a copy of the model's.
    assert result.baseline_bias_corrected.mae != result.mae


def test_the_drift_null_is_fitted_on_history_not_on_the_scored_games():
    """A null fitted on the rows it is scored on is not a null. The default
    must come from df_history, which is by construction everything before the
    evaluation window."""
    prepared = _synthetic_prepared()
    result = run_daily_backtest(_config(), prepared=prepared)

    history = prepared.df_full.iloc[:-15]
    expected = float(
        (history["TOTAL_POINTS"] - history["ODDS_TOTAL_LINE_bet365"]).mean()
    )
    assert result.dev_line_error_bias == pytest.approx(expected)

    scored = prepared.df_full.iloc[-15:]
    on_scored_rows = float(
        (scored["TOTAL_POINTS"] - scored["ODDS_TOTAL_LINE_bet365"]).mean()
    )
    assert result.dev_line_error_bias != pytest.approx(on_scored_rows)


def test_an_explicit_bias_overrides_the_history_default():
    """run_experiment computes the bias once on dev and passes it in, so the
    holdout and walk-forward paths are scored against the SAME null."""
    from training_pipeline.backtest import run_walk_forward_evaluation

    prepared = _synthetic_prepared()
    config = _config()
    df_history = prepared.df_full.iloc[:-15]
    df_evaluation = prepared.df_full.iloc[-15:].reset_index(drop=True)

    result = run_walk_forward_evaluation(
        config,
        prepared=prepared,
        df_history=df_history,
        df_evaluation=df_evaluation,
        train_games=None,
        xgb_params={},
        n_estimators=5,
        show_progress=False,
        dev_line_error_bias=7.5,
    )

    assert result.dev_line_error_bias == pytest.approx(7.5)
    # +7.5 on every line means betting OVER on all of them, at that edge.
    assert result.baseline_bias_corrected_betting.n_bets == 15
    assert "+7.500" in result.baseline_bias_corrected.line_col
