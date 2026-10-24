"""Trend slopes must not be NaN at the start of a season.

``compute_trend_slope`` groups by ``(TEAM_ID, SEASON_YEAR)`` with
``min_periods=2`` after a ``shift(1)``, so the first two games of every season
had no value -- across ~1,100 slope columns in the closing-line dataset and ~160
in the intermediate one. That is enough missingness per row for the downstream
``max_na_per_row`` cleaning to discard the row entirely, which is why the models
had never been trained on the opening games of a season.

The fix mirrors ``_SEASON_BEFORE_AVG`` in ``statistics.compute_rolling_stats``:
current season (preferred) -> previous season -> a defined "no history" value.
The per-season grouping is kept deliberately -- a slope spanning the offseason is
not a trend -- so the gap is closed by the fallback rather than by widening the
window across the boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.team.rolling import (
    TREND_NO_HISTORY_VALUE,
    compute_trend_slope,
)

TREND_COL = "PTS_TREND_SLOPE_LAST_10_GAMES_BEFORE"
HOME_AWAY_COL = "PTS_TREND_SLOPE_LAST_10_HOME_AWAY_GAMES_BEFORE"


def _season(team, year, n, start, base, step, season_type="Regular Season"):
    """A team-season with a perfectly linear scoring trend of slope ``step``."""
    return pd.DataFrame(
        {
            "TEAM_ID": team,
            "SEASON_YEAR": year,
            "GAME_DATE": pd.date_range(start, periods=n, freq="3D"),
            "HOME": [i % 2 for i in range(n)],
            "SEASON_TYPE": season_type,
            "PTS": [base + step * i for i in range(n)],
        }
    )


def _ordered(result):
    return result.sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"])


def test_opening_games_inherit_the_previous_seasons_closing_trend():
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    )

    out = _ordered(compute_trend_slope(df.copy(), parameter="PTS", window=10))
    team = out[out.TEAM_ID == 1]
    closing_2023 = team[team.SEASON_YEAR == 2023][TREND_COL].iloc[-1]
    opening_2024 = team[team.SEASON_YEAR == 2024][TREND_COL]

    assert opening_2024.iloc[0] == pytest.approx(closing_2023)
    assert opening_2024.iloc[1] == pytest.approx(closing_2023)


def test_the_fallback_reads_the_regular_season_not_the_playoffs():
    """A playoff run is a different regime, and only 16 teams have one. Sourcing
    the carried-over trend from it would make the feature mean something
    different depending on how far a team went."""
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            # Deep collapse in the playoffs, chronologically the last thing the
            # team played before the new season.
            _season(1, 2023, 6, "2024-04-20", 130, -8.0, season_type="Playoffs"),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    )

    out = _ordered(compute_trend_slope(df.copy(), parameter="PTS", window=10))
    team = out[out.TEAM_ID == 1]
    regular = team[(team.SEASON_YEAR == 2023) & (team.SEASON_TYPE == "Regular Season")]
    playoffs = team[(team.SEASON_YEAR == 2023) & (team.SEASON_TYPE == "Playoffs")]
    opening_2024 = team[team.SEASON_YEAR == 2024][TREND_COL].iloc[0]

    assert opening_2024 == pytest.approx(regular[TREND_COL].iloc[-1])
    assert opening_2024 != pytest.approx(playoffs[TREND_COL].iloc[-1])


def test_a_team_with_no_prior_season_gets_the_no_history_value():
    df = _season(2, 2024, 6, "2024-11-01", 100, 1.0)

    out = _ordered(compute_trend_slope(df, parameter="PTS", window=10))

    assert out[TREND_COL].iloc[0] == TREND_NO_HISTORY_VALUE
    assert out[TREND_COL].iloc[1] == TREND_NO_HISTORY_VALUE


def test_no_trend_column_is_left_with_nan():
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
            _season(2, 2024, 6, "2024-11-01", 100, 1.0),
        ],
        ignore_index=True,
    )

    out = compute_trend_slope(df, parameter="PTS", window=10)

    trend_columns = [c for c in out.columns if "_TREND_SLOPE_" in c]
    assert trend_columns
    for column in trend_columns:
        assert not out[column].isna().any(), column


def test_the_home_away_column_is_filled_before_the_subtraction():
    """It is derived as (home/away slope - overall slope). Filling only the
    overall side would leave NaN - value = NaN, so the relative column would stay
    missing exactly where it was before."""
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    )

    out = _ordered(compute_trend_slope(df, parameter="PTS", window=10))
    opening = out[out.SEASON_YEAR == 2024][HOME_AWAY_COL]

    assert not opening.isna().any()


def test_within_season_values_are_untouched():
    """The fallback must only fill gaps. Once a season has enough games the value
    has to be exactly the within-season slope it always was."""
    df = _season(1, 2024, 12, "2024-11-01", 100, 2.0)

    out = _ordered(compute_trend_slope(df.copy(), parameter="PTS", window=10))
    settled = out[TREND_COL].iloc[3:]

    # A perfectly linear +2.0 scoring progression has a +2.0 slope.
    assert settled.to_numpy() == pytest.approx(2.0)


def test_a_frame_without_season_type_falls_back_to_no_history():
    """Minimal frames have no season type to filter on. Rather than quietly
    sourcing the value from whatever games are present -- playoffs included --
    the chain drops through to the defined no-history value."""
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    ).drop(columns=["SEASON_TYPE"])

    out = _ordered(compute_trend_slope(df, parameter="PTS", window=10))
    opening_2024 = out[out.SEASON_YEAR == 2024][TREND_COL]

    assert opening_2024.iloc[0] == TREND_NO_HISTORY_VALUE
    assert not out[TREND_COL].isna().any()


def test_short_minus_long_variant_is_also_filled():
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    )

    out = compute_trend_slope(
        df,
        parameter="PTS",
        window=10,
        include_home_away_relative=False,
        relative_to_window=5,
    )

    relative_col = "PTS_TREND_SLOPE_LAST_5_MINUS_LAST_10_GAMES_BEFORE"
    assert relative_col in out.columns
    assert not out[relative_col].isna().any()


def test_the_fallback_is_keyed_per_team():
    """Team 2's opening games must not inherit team 1's trend."""
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 5.0),
            _season(2, 2023, 12, "2023-11-01", 100, -5.0),
            _season(1, 2024, 4, "2024-11-01", 110, 0.5),
            _season(2, 2024, 4, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    )

    out = _ordered(compute_trend_slope(df, parameter="PTS", window=10))
    opening = out[out.SEASON_YEAR == 2024].groupby("TEAM_ID")[TREND_COL].first()

    assert opening.loc[1] == pytest.approx(5.0)
    assert opening.loc[2] == pytest.approx(-5.0)


def test_a_duplicated_index_does_not_scramble_the_fill():
    """The pipeline hands this function frames whose index is not unique. The
    fallback is computed positionally for that reason; an index-aligned fillna
    would raise or misalign here."""
    df = pd.concat(
        [
            _season(1, 2023, 12, "2023-11-01", 100, 2.0),
            _season(1, 2024, 6, "2024-11-01", 110, 0.5),
        ],
        ignore_index=True,
    )
    df.index = np.zeros(len(df), dtype=int)

    out = _ordered(compute_trend_slope(df, parameter="PTS", window=10))
    opening_2024 = out[out.SEASON_YEAR == 2024][TREND_COL]

    assert not opening_2024.isna().any()
    assert opening_2024.iloc[0] == pytest.approx(2.0)
