"""Regressions for three defects found in review of the spread implementation.

Each one produced no exception at the point of the mistake, which is why each
gets a test rather than a fix alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_pipeline.betting import OUTCOME_COLUMN
from training_pipeline.config import (
    RESIDUAL_TARGET_COLUMNS,
    CleaningConfig,
    DataConfig,
    DatasetType,
    ExperimentConfig,
    HoldoutConfig,
    PredictionStrategy,
)
from training_pipeline.data import ensure_spread_error_column, prepare_dataset
from training_pipeline.decisions import _RESIDUAL_REGRESSORS
from training_pipeline.reporting import loaders

# --- 1. the closing dataset never derived SPREAD_ERROR -----------------------


def _closing_frame(n_days=120, games_per_day=4, seed=3):
    rng = np.random.default_rng(seed)
    rows = [
        {"GAME_ID": f"00221{i * games_per_day + g:05d}", "GAME_DATE": day,
         "SEASON_YEAR": 2024}
        for i, day in enumerate(pd.date_range("2024-10-20", periods=n_days, freq="D"))
        for g in range(games_per_day)
    ]
    df = pd.DataFrame(rows)
    line = rng.uniform(205, 240, len(df)).round(1)
    df["ODDS_TOTAL_LINE_bet365"] = line
    df["TOTAL_POINTS"] = (line + rng.normal(0, 12, len(df))).round(1)
    df["PTS_TEAM_AWAY"] = 110
    df["PTS_TEAM_HOME"] = 110 + rng.integers(-25, 26, len(df))
    df["HOME_MARGIN"] = df["PTS_TEAM_HOME"] - df["PTS_TEAM_AWAY"]
    df["ODDS_SPREAD_LINE_HOME_bet365"] = rng.normal(0, 5, len(df)).round(1)
    df["FEATURE_A"] = rng.normal(size=len(df))
    return df


def _closing_config(tmp_path, csv_path):
    return ExperimentConfig(
        experiment_name="closing_spread",
        prediction_strategy=PredictionStrategy.SPREAD_ERROR_REGRESSOR,
        data=DataConfig(csv_path=csv_path, dataset_type=DatasetType.CLOSING_LINE),
        cleaning=CleaningConfig(verbose=0),
        holdout=HoldoutConfig(test_size=None, test_days=30),
        save_experiment_artifacts=False,
        experiment_root_dir=tmp_path / "a",
        model_output_root=tmp_path / "m",
    )


def test_closing_spread_run_trains_without_a_precomputed_target(tmp_path):
    """The closing CSV carries HOME_MARGIN and the spread line but NOT
    SPREAD_ERROR -- only the intermediate builder derives that upstream. Before
    the fix this raised KeyError and no closing spread run could start."""
    path = tmp_path / "closing.csv"
    _closing_frame().to_csv(path, index=False)

    prepared = prepare_dataset(_closing_config(tmp_path, path))

    assert prepared.y.name == "SPREAD_ERROR"
    expected = (
        prepared.df_full["HOME_MARGIN"]
        - prepared.df_full["ODDS_SPREAD_LINE_HOME_bet365"]
    )
    pd.testing.assert_series_equal(
        prepared.y.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
    assert "SPREAD_ERROR" not in prepared.X.columns


def test_ensure_never_overwrites_an_existing_per_snapshot_target():
    """The safety property that makes deriving-at-training-time safe.

    An intermediate frame arrives with one target per snapshot, each built
    against that snapshot's line. Recomputing here against whatever single
    anchor survived would replace them all -- the exact per-game-join failure the
    intermediate target exists to avoid.
    """
    frame = pd.DataFrame(
        {
            "HOME_MARGIN": [8.0, 8.0, 8.0],
            "ODDS_SPREAD_LINE_HOME_bet365": [4.0, 5.0, 6.0],
            "SPREAD_ERROR": [4.0, 3.0, 2.0],
        }
    )
    out = ensure_spread_error_column(
        frame, spread_line_col="ODDS_SPREAD_LINE_HOME_bet365", outcome_col="HOME_MARGIN"
    )
    assert out["SPREAD_ERROR"].tolist() == [4.0, 3.0, 2.0]
    assert out is frame  # untouched, not even copied


def test_ensure_derives_when_absent():
    frame = pd.DataFrame(
        {"HOME_MARGIN": [8.0], "ODDS_SPREAD_LINE_HOME_bet365": [4.0]}
    )
    out = ensure_spread_error_column(
        frame, spread_line_col="ODDS_SPREAD_LINE_HOME_bet365", outcome_col="HOME_MARGIN"
    )
    assert out["SPREAD_ERROR"].tolist() == [4.0]


def test_ensure_reports_the_missing_column_by_name():
    with pytest.raises(KeyError, match="schema 2_1"):
        ensure_spread_error_column(
            pd.DataFrame({"HOME_MARGIN": [8.0]}),
            spread_line_col="ODDS_SPREAD_LINE_HOME_bet365",
            outcome_col="HOME_MARGIN",
        )


# --- 2. reporting silently skipped spread runs -------------------------------


def _write_predictions(path, *, outcome_column):
    pd.DataFrame(
        {
            "predicted_edge": [3.0, -3.0, 2.0],
            "selection_score": [3.0, 3.0, 2.0],
            "target_line": [4.0, 4.0, 4.0],
            outcome_column: [9.0, 1.0, 4.0],  # home covers, away covers, push
        }
    ).to_parquet(path)


def _Row(run_dir):
    """A leaderboard row as the loaders actually receive it: a pandas Series."""
    return pd.Series({"run_dir": str(run_dir), "prediction_strategy": "spread_error_regressor"})


def test_spread_predictions_are_not_silently_skipped(tmp_path):
    """Before the fix the loader required TOTAL_POINTS and `continue`d past a
    spread run -- no error, no count, no named skip. The run simply vanished."""
    _write_predictions(tmp_path / "cv_predictions.parquet", outcome_column=OUTCOME_COLUMN)

    found = loaders.load_all_predictions(_Row(tmp_path), drop_pushes=False)

    assert len(found) == 1, "the spread run was dropped from the comparison"
    source, frame = found[0]
    assert source == "cross-validation"
    assert OUTCOME_COLUMN in frame.columns
    # home covers -> win; away covers with an away bet -> win; landing on -> push
    assert frame["won"].tolist() == [1, 1, 0]
    assert frame["push"].tolist() == [False, False, True]


def test_archived_totals_predictions_still_load(tmp_path):
    """Backward compatibility: runs that wrote only TOTAL_POINTS."""
    _write_predictions(tmp_path / "cv_predictions.parquet", outcome_column="TOTAL_POINTS")

    found = loaders.load_all_predictions(_Row(tmp_path), drop_pushes=False)

    assert len(found) == 1
    _, frame = found[0]
    assert frame[OUTCOME_COLUMN].tolist() == [9.0, 1.0, 4.0]
    assert frame["won"].tolist() == [1, 1, 0]


def test_a_frame_with_no_outcome_at_all_is_still_skipped(tmp_path):
    """The normalisation must not turn a genuinely unusable run into a usable one."""
    pd.DataFrame(
        {"predicted_edge": [1.0], "selection_score": [1.0], "target_line": [4.0]}
    ).to_parquet(tmp_path / "cv_predictions.parquet")

    assert loaders.load_all_predictions(_Row(tmp_path), drop_pushes=False) == []


# --- 3. baseline_pred sat in the wrong space for spread ----------------------


def test_spread_error_is_registered_as_a_residual_target():
    """`baseline_pred` is zeroed for residual targets. SPREAD_ERROR is one; when
    it was missing from this set the saved parquet put a spread LINE beside a
    target that is already a residual -- two spaces, adjacent columns, no error."""
    assert "SPREAD_ERROR" in RESIDUAL_TARGET_COLUMNS
    assert "LINE_ERROR" in RESIDUAL_TARGET_COLUMNS
    assert "TOTAL_POINTS" not in RESIDUAL_TARGET_COLUMNS


def test_the_two_residual_definitions_agree():
    """config names them by target column, decisions by strategy. They describe
    one fact, so they must not drift."""
    from_strategies = {s.target_family.value.upper() for s in _RESIDUAL_REGRESSORS}
    assert from_strategies == {c.upper() for c in RESIDUAL_TARGET_COLUMNS}
