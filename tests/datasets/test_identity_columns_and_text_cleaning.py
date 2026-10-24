"""Identifier columns are declared by exact name; everything else is a number.

Pins the bug this replaced: readers treated every column whose name merely
contained "ID" as an identifier and read it as text, and cleaning then dropped
it as a text column. On the 2_5 intermediate dataset that silently removed seven
unique market-dynamics features (``..._REACTION_RESIDUAL``, ``..._RIDGE_...``,
``..._WITHOUT_SIDE_MOVE_60``) from every model, and on the closing dataset 212
``..._line_mid_...`` columns. Nothing errored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.config.identity_columns import (
    IDENTIFIER_COLUMNS,
    NON_FEATURE_COLUMNS,
    apply_identifier_dtypes,
    identifier_dtypes,
)
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)

from training_pipeline.config import CleaningConfig
from training_pipeline.data import (
    load_raw_training_csv,
)
from training_pipeline.parquet_dataset import convert_csv_to_parquet

#: Real 2_5 feature names that contain "ID" without being identifiers.
FEATURES_CONTAINING_ID = [
    "ODDS_book_total_line_mid_bet365_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME",
    "ODDS_SNAP_NEWS_TOT_BET365_REACTION_RESIDUAL",
    "ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE",
    "ODDS_SNAP_XMKT_TOTAL_MOVE_WITHOUT_SIDE_MOVE_60",
    "IS_US_HOLIDAY_BEFORE",
]


def _write_csv(path, n=12):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "GAME_ID": [f"00224{i:05d}" for i in range(n)],
            "SEASON_ID": ["22024"] * n,
            "TEAM_ID_TEAM_HOME": [1610612737 + i % 3 for i in range(n)],
            "TEAM_ID_TEAM_AWAY": [1610612740 + i % 3 for i in range(n)],
            "TEAM_NAME_TEAM_HOME": [f"Team {i % 3}" for i in range(n)],
            "SEASON_TYPE": ["Regular Season"] * n,
            "GAME_DATE": [f"2025-01-{i + 1:02d}" for i in range(n)],
            "SEASON_YEAR": [2024] * n,
            "TOTAL_POINTS": rng.normal(225, 15, n).round(),
            "ODDS_TOTAL_LINE_bet365": rng.normal(224, 10, n).round() + 0.5,
            "IS_US_HOLIDAY_BEFORE": [int(i == 4) for i in range(n)],
        }
    )
    for column in FEATURES_CONTAINING_ID[:-1]:
        frame[column] = rng.normal(0, 1, n)
    frame.loc[2, FEATURES_CONTAINING_ID[1]] = np.nan
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------


def test_identifier_dtypes_names_only_declared_identifiers():
    columns = ["GAME_ID", "TEAM_ID_TEAM_HOME", *FEATURES_CONTAINING_ID]
    assert identifier_dtypes(columns) == {"GAME_ID": str, "TEAM_ID_TEAM_HOME": str}


def test_every_identifier_is_a_non_feature_column():
    assert set(IDENTIFIER_COLUMNS) <= set(NON_FEATURE_COLUMNS)


def test_apply_identifier_dtypes_reads_numbers_as_text_like_read_csv():
    """A Parquet file written from a builder's frame stores TEAM_ID as a number."""
    df = pd.DataFrame(
        {
            "TEAM_ID_TEAM_HOME": [1610612737, 1610612738],
            "TEAM_ID_TEAM_AWAY": [1610612739.0, np.nan],
            "GAME_ID": ["0022400001", None],
            "PACE": [1.0, 2.0],
        }
    )

    out = apply_identifier_dtypes(df)

    assert out["TEAM_ID_TEAM_HOME"].tolist() == ["1610612737", "1610612738"]
    assert out["TEAM_ID_TEAM_AWAY"].iloc[0] == "1610612739"
    missing = out["TEAM_ID_TEAM_AWAY"].iloc[1]
    assert isinstance(missing, float) and np.isnan(missing)  # not "<NA>" or "nan"
    assert out["GAME_ID"].iloc[1] is None  # already text: left alone
    assert out["PACE"].dtype == np.float64
    assert df["TEAM_ID_TEAM_HOME"].dtype == np.int64  # input not mutated


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def test_csv_reads_features_containing_id_as_numbers(tmp_path):
    df = load_raw_training_csv(_write_csv(tmp_path / "train.csv"))

    for column in FEATURES_CONTAINING_ID:
        assert pd.api.types.is_numeric_dtype(df[column]), column
    assert df["GAME_ID"].iloc[0] == "0022400000"
    assert df["TEAM_ID_TEAM_HOME"].iloc[0] == "1610612737"


def test_parquet_copy_reads_identically(tmp_path):
    csv = _write_csv(tmp_path / "train.csv")
    parquet = convert_csv_to_parquet(csv)

    pd.testing.assert_frame_equal(
        load_raw_training_csv(parquet), load_raw_training_csv(csv), check_exact=True
    )


def test_a_copy_typed_by_the_old_loader_fails_cleaning_loudly(tmp_path):
    """Copies converted before the fix stored "ID"-substring features as text.
    Nothing marks them as stale; the text guard in cleaning is what catches
    them, so a stale copy cannot quietly train on fewer features."""
    csv = _write_csv(tmp_path / "train.csv")
    header = pd.read_csv(csv, nrows=0).columns
    stale = csv.with_suffix(".parquet")
    pd.read_csv(csv, dtype={c: str for c in header if "ID" in c.upper()}).to_parquet(
        stale, index=False
    )

    with pytest.raises(ValueError, match="RESIDUAL.*regenerate it"):
        clean_dataframe_for_training(
            load_raw_training_csv(stale), verbose=0, keep_columns=["GAME_DATE"]
        )


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------


def test_features_containing_id_survive_load_and_cleaning(tmp_path):
    df = load_raw_training_csv(_write_csv(tmp_path / "train.csv"))

    cleaned, report = clean_dataframe_for_training(
        df,
        verbose=0,
        keep_all_cols=True,
        keep_columns=["GAME_DATE"],
        return_report=True,
    )

    for column in FEATURES_CONTAINING_ID:
        assert column in cleaned.columns, column
    for column in ("GAME_ID", "SEASON_ID", "TEAM_ID_TEAM_HOME", "SEASON_TYPE"):
        assert report.why_dropped(column)["step"] == "non_feature_columns"


def test_numeric_columns_named_like_ids_are_kept():
    """The old _ID and _NAME rules matched substrings of any name."""
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [220.0, 221.0, 222.0],
            "ODDS_TOTAL_LINE_bet365": [219.5, 220.5, 221.5],
            "N_IDLE_DAYS_BEFORE_TEAM_HOME": [1.0, 2.0, 3.0],
            "PLAYER_NAME_LENGTH_BEFORE": [4.0, 5.0, 6.0],
        }
    )

    cleaned = clean_dataframe_for_training(df, verbose=0, keep_all_cols=True)

    assert "N_IDLE_DAYS_BEFORE_TEAM_HOME" in cleaned.columns
    assert "PLAYER_NAME_LENGTH_BEFORE" in cleaned.columns


def _frame_with_text(value):
    return pd.DataFrame(
        {
            "TOTAL_POINTS": [220.0, 221.0, 222.0],
            "ODDS_TOTAL_LINE_bet365": [219.5, 220.5, 221.5],
            "A_FEATURE_READ_AS_TEXT": [value, value, None],
        }
    )


def test_an_undeclared_text_column_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="A_FEATURE_READ_AS_TEXT"):
        clean_dataframe_for_training(_frame_with_text("246.5"), verbose=0)


def test_the_prediction_path_can_drop_undeclared_text_and_records_it():
    cleaned, report = clean_dataframe_for_training(
        _frame_with_text("19:30"),
        verbose=0,
        keep_all_cols=True,
        unexpected_text="drop",
        return_report=True,
    )

    assert "A_FEATURE_READ_AS_TEXT" not in cleaned.columns
    assert (
        report.why_dropped("A_FEATURE_READ_AS_TEXT")["step"]
        == "unexpected_text_columns"
    )


def test_a_protected_text_column_is_not_an_error():
    cleaned = clean_dataframe_for_training(
        _frame_with_text("19:30"),
        verbose=0,
        keep_columns=["A_FEATURE_READ_AS_TEXT"],
    )

    assert "A_FEATURE_READ_AS_TEXT" in cleaned.columns


def test_an_empty_object_column_is_judged_as_an_empty_column():
    df = _frame_with_text(None)

    _, report = clean_dataframe_for_training(
        df, verbose=0, nan_threshold=50.0, return_report=True
    )

    assert report.why_dropped("A_FEATURE_READ_AS_TEXT")["step"] == "high_nan_columns"


def test_missing_flags_are_retired():
    with pytest.raises(ValueError, match="create_missing_flags"):
        CleaningConfig(create_missing_flags=True)
