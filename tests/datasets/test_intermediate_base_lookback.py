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
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.create_training_data import create_intermediate_line_df as builder
from nba_ou.create_training_data.create_intermediate_line_df import (
    DEFAULT_BASE_LOOKBACK_SEASONS,
    create_intermediate_line_df,
)


class _BaseReached(Exception):
    """Stop after observing the actual call into the base feature builder."""


@pytest.fixture
def base_start(monkeypatch):
    observed = {}
    monkeypatch.setattr(builder, "available_seasons", lambda **kw: [2021, 2022])
    monkeypatch.setattr(
        builder,
        "fetch_pregame_ticks",
        lambda *a, **kw: pd.DataFrame({"game_id": ["g"]}),
    )
    monkeypatch.setattr(
        builder, "fetch_games", lambda *a, **kw: pd.DataFrame({"game_id": ["g"]})
    )
    monkeypatch.setattr(
        builder,
        "_build_snapshot_side",
        lambda *a, **kw: SimpleNamespace(wide=pd.DataFrame(), ridge_closing_lines={}),
    )
    monkeypatch.setattr(
        builder,
        "add_prior_game_line_dynamics",
        lambda *a, **kw: pd.DataFrame({"game_id": ["g"]}),
    )

    def capture(**kwargs):
        observed.update(kwargs)
        raise _BaseReached

    monkeypatch.setattr(builder, "create_base_game_features", capture)

    def run(**kwargs):
        with pytest.raises(_BaseReached):
            create_intermediate_line_df(
                season_years=[2022, 2021],
                combine_fanatics_and_caesars=False,
                verbose=False,
                **kwargs,
            )
        return observed["season_start_date"]

    return run


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


def test_the_start_year_is_the_first_store_season_minus_the_lookback(base_start):
    """The real builder forwards the requested lookback to base feature loading."""
    assert base_start() == pd.Timestamp("2019-10-01")
    for lookback, year in [(0, 2021), (1, 2020), (3, 2018)]:
        assert base_start(base_lookback_seasons=lookback) == pd.Timestamp(
            year=year, month=10, day=1
        )


@pytest.mark.parametrize("value", [-1, -2])
def test_a_negative_lookback_is_refused(value):
    """It would move the start *later* than the first line-history season,
    silently discarding games the store has ticks for."""
    with pytest.raises(ValueError, match="base_lookback_seasons must be >= 0"):
        create_intermediate_line_df(base_lookback_seasons=value, season_years=[2021])


# --- the closing-odds floor -------------------------------------------------


def test_the_lookback_lands_exactly_on_the_first_season_with_odds(base_start, capsys):
    from nba_ou.create_training_data.create_intermediate_line_df import (
        FIRST_SEASON_WITH_CLOSING_ODDS,
    )

    start = base_start()
    assert start == pd.Timestamp(year=FIRST_SEASON_WITH_CLOSING_ODDS, month=10, day=1)
    assert "WARNING:" not in capsys.readouterr().out


def test_a_deeper_lookback_would_reach_seasons_with_no_odds(base_start, capsys):
    assert base_start(base_lookback_seasons=3) == pd.Timestamp("2018-10-01")
    output = capsys.readouterr().out
    assert "WARNING: base features start at 2018" in output
    assert "closing-odds tables begin at 2019" in output
    assert "without warming any odds rollup" in output


# --- seasons only the Covers fallback holds ---------------------------------


def _covers_seasons(**kw):
    """SBR from 2021; the Covers fallback adds 2019 (its backfilled games)."""
    return [2021, 2022] if kw.get("covers_fallback") is False else [2019, 2021, 2022]


def _scope(monkeypatch, base_start, **kwargs):
    monkeypatch.setattr(builder, "available_seasons", _covers_seasons)
    observed = {}

    def capture(**kw):
        observed.update(kw)
        raise _BaseReached

    monkeypatch.setattr(builder, "create_base_game_features", capture)
    with pytest.raises(_BaseReached):
        create_intermediate_line_df(
            combine_fanatics_and_caesars=False, verbose=False, **kwargs
        )
    return observed["season_start_date"]


def test_a_covers_only_season_does_not_widen_the_default_scope(monkeypatch, base_start):
    # The default scope (and so the base lookback) is SBR's: 2021 - 2 seasons.
    assert _scope(monkeypatch, base_start) == pd.Timestamp("2019-10-01")


def test_a_covers_only_season_asked_for_is_kept(monkeypatch, base_start):
    start = _scope(monkeypatch, base_start, season_years=[2019, 2021])
    assert start == pd.Timestamp("2017-10-01")
