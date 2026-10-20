"""Tests for the public-betting / extra-seasons trade switch.

The switch exists because the two halves are one decision. Each test below
pins one half, plus the reason they must not come apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    advanced_column_cleaning,
    clean_dataframe_for_training,
    find_season_gated_columns,
)

from training_pipeline.config import (
    DEFAULT_SEASON_YEAR_FLOOR,
    EXTENDED_SEASON_NAN_SPREAD,
    EXTENDED_SEASON_YEAR_FLOOR,
    PUBLIC_BETTING_SUBSTRINGS,
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    PredictionStrategy,
)


def build_config(**data_kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(csv_path="x.csv", **data_kwargs),
        cleaning=CleaningConfig(),
    )


# ---------------------------------------------------------------------------
# off (the default)
# ---------------------------------------------------------------------------


def test_disabled_by_default_and_changes_nothing():
    config = build_config(season_year_floor=DEFAULT_SEASON_YEAR_FLOOR)
    assert config.data.extend_history_dropping_season_gated_columns is False
    assert config.data.season_year_floor == DEFAULT_SEASON_YEAR_FLOOR
    assert config.cleaning.exclude_cols_containing is None


def test_disabled_leaves_existing_exclusions_alone():
    config = ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(csv_path="x.csv"),
        cleaning=CleaningConfig(exclude_cols_containing=["fanatics_sportsbook"]),
    )
    assert config.cleaning.exclude_cols_containing == ["fanatics_sportsbook"]


# ---------------------------------------------------------------------------
# on: both halves must fire
# ---------------------------------------------------------------------------


def test_enabling_lowers_the_floor_and_drops_the_columns():
    config = build_config(
        season_year_floor=DEFAULT_SEASON_YEAR_FLOOR,
        extend_history_dropping_season_gated_columns=True,
    )
    assert config.data.season_year_floor == EXTENDED_SEASON_YEAR_FLOOR
    assert config.cleaning.max_seasonal_nan_spread == EXTENDED_SEASON_NAN_SPREAD


def test_enabling_with_no_floor_set_still_lowers_it():
    config = build_config(extend_history_dropping_season_gated_columns=True)
    assert config.data.season_year_floor == EXTENDED_SEASON_YEAR_FLOOR


def test_unrelated_exclusions_are_left_alone():
    """fanatics_sportsbook is excluded for an unrelated reason and must survive.
    The flag no longer touches exclude_cols_containing at all -- it computes the
    offenders instead of naming them."""
    config = ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(
            csv_path="x.csv", extend_history_dropping_season_gated_columns=True
        ),
        cleaning=CleaningConfig(exclude_cols_containing=["fanatics_sportsbook"]),
    )
    assert config.cleaning.exclude_cols_containing == ["fanatics_sportsbook"]
    assert config.cleaning.max_seasonal_nan_spread == EXTENDED_SEASON_NAN_SPREAD


def test_an_explicit_spread_is_not_overridden():
    """A caller who set one has answered this question more precisely than the
    flag can, e.g. 50 to also remove partially-gated columns."""
    config = ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(
            csv_path="x.csv", extend_history_dropping_season_gated_columns=True
        ),
        cleaning=CleaningConfig(max_seasonal_nan_spread=50.0),
    )
    assert config.cleaning.max_seasonal_nan_spread == 50.0
    assert config.data.season_year_floor == EXTENDED_SEASON_YEAR_FLOOR


def test_a_floor_below_the_standard_one_is_left_alone():
    """An explicit 2020 means "the COVID season but not the bubble". The flag
    only ever relaxes the standard floor; it must not override a deliberate
    narrowing."""
    config = build_config(
        season_year_floor=2020, extend_history_dropping_season_gated_columns=True
    )
    assert config.data.season_year_floor == 2020
    assert config.cleaning.max_seasonal_nan_spread == EXTENDED_SEASON_NAN_SPREAD


# ---------------------------------------------------------------------------
# the resolution must be visible to everything downstream
# ---------------------------------------------------------------------------


def test_resolution_is_recorded_in_the_dumped_config():
    """--dry-run, config.json and the run record must show what actually ran,
    not the flag that implied it."""
    config = build_config(
        season_year_floor=DEFAULT_SEASON_YEAR_FLOOR,
        extend_history_dropping_season_gated_columns=True,
    )
    dumped = config.model_dump(mode="json")
    assert dumped["data"]["season_year_floor"] == EXTENDED_SEASON_YEAR_FLOOR
    assert dumped["cleaning"]["max_seasonal_nan_spread"] == EXTENDED_SEASON_NAN_SPREAD


def test_flag_and_spelled_out_equivalent_share_a_fingerprint():
    """The flag is shorthand, so setting it and writing out what it resolves to
    describe the same trials and must resume the same Optuna study."""
    via_flag = build_config(
        season_year_floor=DEFAULT_SEASON_YEAR_FLOOR,
        extend_history_dropping_season_gated_columns=True,
    )
    spelled_out = ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(
            csv_path="x.csv",
            season_year_floor=EXTENDED_SEASON_YEAR_FLOOR,
        ),
        cleaning=CleaningConfig(max_seasonal_nan_spread=EXTENDED_SEASON_NAN_SPREAD),
    )
    assert via_flag.data.extend_history_dropping_season_gated_columns is True
    assert spelled_out.data.extend_history_dropping_season_gated_columns is False
    assert via_flag.fingerprint() == spelled_out.fingerprint()


def test_enabling_changes_the_fingerprint():
    """It resolves into hashed fields, so the two arms of the A/B cannot share
    a study even though the flag itself is excluded."""
    off = build_config(season_year_floor=DEFAULT_SEASON_YEAR_FLOOR)
    on = build_config(
        season_year_floor=DEFAULT_SEASON_YEAR_FLOOR,
        extend_history_dropping_season_gated_columns=True,
    )
    assert off.fingerprint() != on.fingerprint()


def test_dropping_columns_alone_differs_from_dropping_and_extending():
    """The campaign's middle cell. Same exclusions, different floor -- these
    must not collide, or the attribution the campaign exists for is lost."""
    columns_only = ExperimentConfig(
        experiment_name="t",
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        data=DataConfig(csv_path="x.csv", season_year_floor=DEFAULT_SEASON_YEAR_FLOOR),
        cleaning=CleaningConfig(max_seasonal_nan_spread=EXTENDED_SEASON_NAN_SPREAD),
    )
    both = build_config(
        season_year_floor=DEFAULT_SEASON_YEAR_FLOOR,
        extend_history_dropping_season_gated_columns=True,
    )
    assert columns_only.fingerprint() != both.fingerprint()


# ---------------------------------------------------------------------------
# what the patterns actually match
# ---------------------------------------------------------------------------


def test_patterns_match_the_public_betting_family():
    from nba_ou.data_processing.missing_data.clean_df_for_training import (
        _get_cols_matching_patterns,
    )

    df = pd.DataFrame(
        columns=[
            "ODDS_total_consensus_pct_over",
            "ODDS_total_consensus_pct_under",
            "ODDS_spread_consensus_pct_away",
            "ODDS_total_pct_bets_over",
            "ODDS_total_pct_money_under",
            "ODDS_moneyline_pct_bets_home",
            "ODDS_total_consensus_pct_over_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME",
        ]
    )
    matched = _get_cols_matching_patterns(df, list(PUBLIC_BETTING_SUBSTRINGS))
    assert matched == set(df.columns)


def test_patterns_do_not_touch_the_consensus_opening_line():
    """ODDS_TOTAL_LINE_consensus_opener is the betting.comparison_line_cols
    baseline. Matching on "consensus" rather than "consensus_pct" would remove
    it and silently kill the closing-vs-opening comparison."""
    from nba_ou.data_processing.missing_data.clean_df_for_training import (
        _get_cols_matching_patterns,
    )

    df = pd.DataFrame(
        columns=[
            "ODDS_TOTAL_LINE_consensus_opener",
            "ODDS_TOTAL_LINE_bet365",
            "ODDS_SPREAD_consensus_opener",
        ]
    )
    assert _get_cols_matching_patterns(df, list(PUBLIC_BETTING_SUBSTRINGS)) == set()


def test_public_betting_columns_are_dropped_for_every_season():
    """The whole point: dropping them only from the old seasons would leave the
    column set itself telling the model which season a row came from."""
    from nba_ou.data_processing.missing_data.clean_df_for_training import (
        advanced_column_cleaning,
    )

    rng = np.random.default_rng(16)
    df = pd.DataFrame(
        {
            "SEASON_YEAR": [2019] * 50 + [2024] * 50,
            "ODDS_total_pct_bets_over": [np.nan] * 50 + list(rng.normal(size=50)),
            "ODDS_TOTAL_LINE_bet365": rng.normal(size=100),
        }
    )
    cleaned = advanced_column_cleaning(
        df,
        exclude_cols_containing=list(PUBLIC_BETTING_SUBSTRINGS),
        verbose=0,
    )
    assert "ODDS_total_pct_bets_over" not in cleaned.columns
    assert not any(
        cleaned[col].isna().groupby(df["SEASON_YEAR"]).mean().nunique() > 1
        for col in cleaned.columns
        if col != "SEASON_YEAR"
    ), "no surviving column may be NaN in one season and present in another"


@pytest.mark.parametrize("pattern", PUBLIC_BETTING_SUBSTRINGS)
def test_each_pattern_is_non_empty(pattern):
    assert pattern and pattern == pattern.strip()


# ---------------------------------------------------------------------------
# the detection rule itself
# ---------------------------------------------------------------------------


def _seasonal_frame() -> pd.DataFrame:
    """Three seasons. One column absent in the first, one uniformly gappy."""
    n = 30
    return pd.DataFrame(
        {
            "SEASON_YEAR": [2019] * n + [2020] * n + [2021] * n,
            "gated": [np.nan] * n + list(range(n)) + list(range(n)),
            "uniformly_gappy": ([1.0, np.nan] * (n // 2)) * 3,
            "always_present": list(range(3 * n)),
        }
    )


def test_a_column_absent_for_a_whole_season_is_gated():
    gated = find_season_gated_columns(
        _seasonal_frame(), season_col="SEASON_YEAR", max_spread=90.0
    )
    assert set(gated) == {"gated"}
    assert "100.0% NaN in 2019" in gated["gated"]


def test_a_uniformly_missing_column_is_not_gated():
    """50% NaN in every season identifies nothing. A plain nan_threshold cannot
    tell this apart from the gated column; spread can."""
    gated = find_season_gated_columns(
        _seasonal_frame(), season_col="SEASON_YEAR", max_spread=90.0
    )
    assert "uniformly_gappy" not in gated
    assert "always_present" not in gated


def test_a_single_season_frame_gates_nothing():
    """The same-day prediction path sees one season; nothing there can identify
    a season, and dropping columns would break the served feature schema."""
    frame = _seasonal_frame()
    one = frame[frame["SEASON_YEAR"] == 2021]
    assert (
        find_season_gated_columns(one, season_col="SEASON_YEAR", max_spread=90.0) == {}
    )


def test_missing_season_column_raises_rather_than_skipping():
    with pytest.raises(KeyError, match="SEASON_YEAR"):
        find_season_gated_columns(
            pd.DataFrame({"x": [1.0, 2.0]}), season_col="SEASON_YEAR", max_spread=90.0
        )


def test_cleaning_drops_gated_columns_from_every_season():
    """The point of the whole mechanism: dropping them only from the old seasons
    would leave the column set itself announcing which season a row came from."""
    cleaned = advanced_column_cleaning(
        _seasonal_frame(),
        max_seasonal_nan_spread=90.0,
        nan_threshold=100.0,
        verbose=0,
    )
    assert "gated" not in cleaned.columns
    assert "uniformly_gappy" in cleaned.columns


def test_the_step_is_off_by_default():
    cleaned = advanced_column_cleaning(
        _seasonal_frame(), nan_threshold=100.0, verbose=0
    )
    assert "gated" in cleaned.columns


def test_protected_columns_are_never_gated_away():
    cleaned = advanced_column_cleaning(
        _seasonal_frame(),
        max_seasonal_nan_spread=90.0,
        nan_threshold=100.0,
        keep_columns=["gated"],
        verbose=0,
    )
    assert "gated" in cleaned.columns


def test_the_report_explains_a_gated_drop():
    _, report = clean_dataframe_for_training(
        _seasonal_frame().assign(TOTAL_POINTS=220.0, ODDS_TOTAL_LINE_bet365=219.0),
        max_seasonal_nan_spread=90.0,
        nan_threshold=100.0,
        verbose=0,
        return_report=True,
    )
    entry = report.why_dropped("gated")
    assert entry["step"] == "season_gated_columns"
    assert "varies" in entry["reason"]
