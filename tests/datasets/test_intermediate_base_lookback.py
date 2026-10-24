"""Rolling features need history from before the first line-history season.

The line-history store starts at 2021-22; the NBA database goes back further.
``season_start_date`` used to be derived from ``min(season_years)``, which pinned
the two together: the earliest line-history season began with every rolling
window empty, every ``_SEASON_BEFORE_AVG`` unfilled and every trend slope on its
no-history value.

The obvious workaround does not work. ``season_years`` is intersected against
``available_seasons()``, so older years passed there are silently dropped -- they
cannot reach ``create_base_game_features``. Hence a separate knob.
"""

from __future__ import annotations

import inspect

import pytest
from nba_ou.create_training_data.create_intermediate_line_df import (
    DEFAULT_BASE_LOOKBACK_SEASONS,
    create_intermediate_line_df,
)


def _signature():
    return inspect.signature(create_intermediate_line_df).parameters


def test_the_knob_exists_and_is_keyword_only():
    parameter = _signature()["base_lookback_seasons"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_it_defaults_to_covering_the_previous_season_fallbacks():
    """One season satisfies ``_SEASON_BEFORE_AVG``, the trend-slope chain and
    roster continuity; two leaves margin for the 20-game rollups."""
    assert DEFAULT_BASE_LOOKBACK_SEASONS >= 1
    assert _signature()["base_lookback_seasons"].default == (
        DEFAULT_BASE_LOOKBACK_SEASONS
    )


def test_the_start_year_is_the_first_store_season_minus_the_lookback():
    """The arithmetic the parameter exists to perform, stated once so a change
    to it has to be deliberate."""
    first_line_history_season = 2021

    assert first_line_history_season - DEFAULT_BASE_LOOKBACK_SEASONS == 2019


@pytest.mark.parametrize("value", [-1, -2])
def test_a_negative_lookback_is_refused(value):
    """It would move the start *later* than the first line-history season,
    silently discarding games the store has ticks for."""
    with pytest.raises(ValueError, match="base_lookback_seasons must be >= 0"):
        create_intermediate_line_df(base_lookback_seasons=value, season_years=[2021])


# --- the closing-odds floor -------------------------------------------------


def test_the_lookback_lands_exactly_on_the_first_season_with_odds():
    """Measured, not assumed: ``odds_sportsbook`` holds 32 rows for season 2018
    (all from one 2019 postseason) and nothing earlier, so 2019-20 is the first
    season that carries closing odds at all. The line-history store starts at
    2021-22, which makes a 2-season lookback the largest one that buys anything.
    """
    from nba_ou.create_training_data.create_intermediate_line_df import (
        FIRST_SEASON_WITH_CLOSING_ODDS,
    )

    first_line_history_season = 2021

    assert (
        first_line_history_season - DEFAULT_BASE_LOOKBACK_SEASONS
        == FIRST_SEASON_WITH_CLOSING_ODDS
    )


def test_a_deeper_lookback_would_reach_seasons_with_no_odds():
    """Not an error -- team and player history is still real down there -- but it
    lengthens the build without warming a single odds rollup, so it warns."""
    from nba_ou.create_training_data.create_intermediate_line_df import (
        FIRST_SEASON_WITH_CLOSING_ODDS,
    )

    first_line_history_season = 2021

    assert first_line_history_season - 3 < FIRST_SEASON_WITH_CLOSING_ODDS
