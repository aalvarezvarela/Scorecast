"""The run's line source: SBR, or Covers when SBR is down, frozen or empty.

The decision (``resolve_line_source`` / ``check_line_feed``) and the two
readers that act on it: the closing frame's day-page odds and the
intermediate frame's live line history, each filling from Covers the games
SBR does not serve and recording every game's source.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.create_training_data import get_all_info_for_scheduled_games as info
from nba_ou.create_training_data import live_game_day
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven.feed_freshness import assess_feed_freshness
from nba_ou.postgre_db.line_history_aiven.live import LiveLineHistory
from nba_ou.postgre_db.line_history_covers.live import CoversLiveDay
from nba_ou.prediction import line_source as ls

GAME_DATE = date(2026, 10, 24)
AS_OF = pd.Timestamp("2026-10-24 21:00", tz="UTC")
TIPOFF = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)
GAME_A, GAME_B = "0022600104", "0022600105"
TEAMS = {
    GAME_A: ("Miami Heat", "Charlotte Hornets"),
    GAME_B: ("Brooklyn Nets", "Cleveland Cavaliers"),
}


def _freshness(minutes_since_tick: float | None, source="SBR"):
    ticks = (
        []
        if minutes_since_tick is None
        else [AS_OF - pd.Timedelta(minutes=minutes_since_tick)]
    )
    return assess_feed_freshness(
        ticks, [TIPOFF], game_date=GAME_DATE, as_of=AS_OF, source=source
    )


def _scheduled() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [GAME_A, GAME_B],
            "GAME_DATE_EST": ["2026-10-24T00:00:00"] * 2,
            "HOME_TEAM_ID": [TEAM_ID_MAP[TEAMS[g][1]] for g in (GAME_A, GAME_B)],
            "VISITOR_TEAM_ID": [TEAM_ID_MAP[TEAMS[g][0]] for g in (GAME_A, GAME_B)],
        }
    )


def _covers_game(game_id: str, minutes: int = -150) -> ScrapedGame:
    away, home = TEAMS[game_id]
    tick = LineTick(
        market="totals",
        book_slug="bet365",
        book_name="bet365",
        line_ts=TIPOFF + timedelta(minutes=minutes),
        minutes_to_tip=minutes,
        is_opener=False,
        left_line=228.5,
        left_price=-110,
        right_line=228.5,
        right_price=-110,
    )
    return ScrapedGame(
        event_id=int(game_id[-3:]),
        game_date=GAME_DATE,
        season_year=2026,
        tipoff_utc=TIPOFF,
        team_away=away,
        team_home=home,
        status_text="Scheduled",
        away_score=None,
        home_score=None,
        ticks=(tick,),
    )


class FakeCovers:
    """Stands in for ``fetch_covers_live_day``; records each request."""

    def __init__(self, game_ids=(GAME_A, GAME_B), *, minutes: int = -150):
        self.available = set(game_ids)
        self.minutes = minutes
        self.calls: list[list[str]] = []

    def __call__(
        self, game_date, *, scheduled_games, schedule=None, game_ids=None, **_
    ):
        wanted = sorted(
            scheduled_games["GAME_ID"].astype(str) if game_ids is None else game_ids
        )
        self.calls.append(wanted)
        # A Covers tick 150 min before tip is 30 min before AS_OF: fresh.
        return CoversLiveDay(
            game_date=game_date,
            as_of=AS_OF,
            games={
                g: _covers_game(g, self.minutes) for g in wanted if g in self.available
            },
            missing={g: "not listed" for g in wanted if g not in self.available},
        )


# --- the decision ---------------------------------------------------------------


def test_fresh_sbr_never_asks_covers():
    covers = FakeCovers()
    decision = ls.resolve_line_source(
        GAME_DATE,
        load_scheduled_games=_scheduled,
        check_sbr=lambda day: _freshness(10),
        fetch_covers=covers,
    )
    assert decision.source == ls.SBR and not decision.stale
    assert covers.calls == []


def test_frozen_sbr_reads_a_fresh_covers():
    decision = ls.resolve_line_source(
        GAME_DATE,
        load_scheduled_games=_scheduled,
        check_sbr=lambda day: _freshness(None),
        fetch_covers=FakeCovers(),
    )
    assert decision.source == ls.COVERS and not decision.stale
    assert "frozen" in decision.reason
    assert not decision.covers_freshness.stale


def test_both_frozen_is_stale_on_sbr():
    decision = ls.resolve_line_source(
        GAME_DATE,
        load_scheduled_games=_scheduled,
        check_sbr=lambda day: _freshness(None),
        fetch_covers=FakeCovers(minutes=-600),
    )
    assert decision.source == ls.SBR and decision.stale
    assert "Covers feed" in decision.reason


def test_a_failed_check_without_covers_carries_on_unchecked():
    def broken(day):
        raise RuntimeError("SBR down")

    decision = ls.resolve_line_source(
        GAME_DATE,
        load_scheduled_games=_scheduled,
        check_sbr=broken,
        fetch_covers=FakeCovers(game_ids=()),
    )
    assert decision.source == ls.SBR and not decision.stale
    assert "SBR down" in decision.reason


def test_a_failed_check_with_covers_reads_covers():
    def broken(day):
        raise RuntimeError("SBR down")

    decision = ls.resolve_line_source(
        GAME_DATE,
        load_scheduled_games=_scheduled,
        check_sbr=broken,
        fetch_covers=FakeCovers(),
    )
    assert decision.source == ls.COVERS


def test_check_line_feed_stops_on_stale_lines_unless_allowed():
    kwargs = dict(
        load_scheduled_games=_scheduled,
        check_sbr=lambda day: _freshness(None),
        fetch_covers=FakeCovers(minutes=-600),
        report=lambda *a, **k: None,
    )
    with pytest.raises(SystemExit):
        ls.check_line_feed(GAME_DATE, allow_stale_lines=False, **kwargs)
    assert ls.check_line_feed(GAME_DATE, allow_stale_lines=True, **kwargs).stale


def test_forced_covers_checks_nothing():
    def never(day):
        raise AssertionError("checked SBR")

    decision = ls.check_line_feed(
        GAME_DATE,
        allow_stale_lines=False,
        load_scheduled_games=_scheduled,
        report=lambda *a, **k: None,
        force="covers",
        fetch_covers=FakeCovers(),
    )
    assert decision.uses_covers


def test_covers_is_fetched_once_per_game_unless_refreshed():
    covers = FakeCovers()
    source = ls.LineSource(GAME_DATE, fetch_covers=covers)
    source.covers_day(_scheduled(), game_ids=[GAME_A])
    day = source.covers_day(_scheduled())
    assert covers.calls == [[GAME_A], [GAME_B]]
    assert set(day.games) == {GAME_A, GAME_B}
    refreshed = source.covers_day(_scheduled(), game_ids=[GAME_B], refresh=True)
    assert covers.calls[-1] == [GAME_B] and set(refreshed.games) == {GAME_B}


def test_a_listed_game_without_lines_is_not_priced():
    frame = pd.DataFrame(
        {
            "game_id": [GAME_A, GAME_B],
            "total_bet365_line_over": [228.5, None],
            "total_consensus_opener_line_over": [231.5, 231.5],
        }
    )
    assert ls.priced_game_ids(frame) == {GAME_A}
    assert ls.priced_game_ids(pd.DataFrame()) == set()


# --- the closing reader -----------------------------------------------------------


def _scheduled_for_merge() -> pd.DataFrame:
    frame = _scheduled()
    frame["team_away"] = [TEAMS[g][0] for g in frame["GAME_ID"]]
    frame["team_home"] = [TEAMS[g][1] for g in frame["GAME_ID"]]
    frame["game_date"] = GAME_DATE
    frame["game_id"] = frame["GAME_ID"]
    return frame


def _sbr_day(*, priced: bool) -> pd.DataFrame:
    """SBR's day frame: both games listed, only the first priced (or none)."""
    from nba_ou.fetch_data.odds_covers.day_odds import covers_day_odds

    frame = covers_day_odds(
        [_covers_game(GAME_A), _covers_game(GAME_B)],
        game_date=GAME_DATE,
        as_of=AS_OF,
    )
    lines = [c for c in frame.columns if c.startswith(("total_", "spread_", "ml_"))]
    frame.loc[frame.index[1:] if priced else frame.index, lines] = None
    return frame


def _patch_sbr(monkeypatch, frame):
    async def scrape(days, **_):
        return frame

    monkeypatch.setattr(info, "scrape_sportsbook_days", scrape)


def test_closing_odds_fill_the_games_sbr_does_not_price(monkeypatch):
    _patch_sbr(monkeypatch, _sbr_day(priced=True))
    covers = FakeCovers()
    source = ls.LineSource(GAME_DATE, fetch_covers=covers)
    odds = info.get_sportsbook_prediction_data(
        "2026-10-24", _scheduled_for_merge(), headless=True, line_source=source
    )
    assert sorted(odds["game_id"]) == [GAME_A, GAME_B]
    assert source.closing_sources == {GAME_A: ls.SBR, GAME_B: ls.COVERS}
    assert covers.calls == [[GAME_B]]
    assert source.source == ls.SBR


def test_closing_odds_without_any_sbr_line_switch_the_run(monkeypatch):
    _patch_sbr(monkeypatch, _sbr_day(priced=False))
    source = ls.LineSource(GAME_DATE, fetch_covers=FakeCovers())
    odds = info.get_sportsbook_prediction_data(
        "2026-10-24", _scheduled_for_merge(), headless=True, line_source=source
    )
    assert source.uses_covers
    assert set(source.closing_sources.values()) == {ls.COVERS}
    assert odds["total_bet365_line_over"].notna().all()


def test_closing_odds_on_covers_never_ask_sbr(monkeypatch):
    async def scrape(days, **_):
        raise AssertionError("SBR read")

    monkeypatch.setattr(info, "scrape_sportsbook_days", scrape)
    source = ls.LineSource(GAME_DATE, source=ls.COVERS, fetch_covers=FakeCovers())
    odds = info.get_sportsbook_prediction_data(
        "2026-10-24", _scheduled_for_merge(), headless=True, line_source=source
    )
    assert len(odds) == 2


def test_without_a_line_source_sbr_is_read_as_before(monkeypatch):
    frame = _sbr_day(priced=True)
    _patch_sbr(monkeypatch, frame)
    odds = info.get_sportsbook_prediction_data(
        "2026-10-24", _scheduled_for_merge(), headless=True
    )
    assert len(odds) == 2  # the unpriced row is kept, as it always was
    assert asyncio.iscoroutinefunction(info.scrape_sportsbook_days)


# --- the intermediate reader ------------------------------------------------------


def _history(game_ids, minutes_before_tip) -> LiveLineHistory:
    ticks = pd.DataFrame(
        {
            "game_id": game_ids,
            "season_year": 2026,
            "market": "totals",
            "book": "bet365",
            "line_ts": [TIPOFF - timedelta(minutes=m) for m in minutes_before_tip],
            "minutes_before_tip": [float(m) for m in minutes_before_tip],
            "is_opener": False,
            "left_line": 228.5,
            "left_price": -110,
            "right_line": 228.5,
            "right_price": -110,
        }
    )
    games = pd.DataFrame(
        {
            "game_id": sorted(set(game_ids)),
            "game_date": pd.Timestamp(GAME_DATE),
            "season_year": 2026,
            "tipoff_utc": TIPOFF,
            "team_home": "CHA",
            "team_away": "MIA",
        }
    )
    return LiveLineHistory(ticks, games, AS_OF)


def _patch_history(monkeypatch, sbr_history, covers_history):
    monkeypatch.setattr(
        live_game_day, "fetch_live_line_history", lambda *a, **k: sbr_history
    )
    monkeypatch.setattr(
        live_game_day,
        "covers_live_line_history",
        lambda day, *, game_ids, **_: LiveLineHistory(
            covers_history.ticks[covers_history.ticks["game_id"].isin(game_ids)],
            covers_history.games[covers_history.games["game_id"].isin(game_ids)],
            AS_OF,
        ),
    )


def _read(source, *, as_of=AS_OF) -> LiveLineHistory:
    return live_game_day._line_history_with_fallback(
        pd.Timestamp(GAME_DATE),
        scheduled_games=_scheduled(),
        team_games=pd.DataFrame(),
        schedule=pd.DataFrame(),
        as_of=as_of,
        line_source=source,
        refresh=True,
    )


def test_history_fills_the_games_sbr_has_no_tick_for(monkeypatch):
    _patch_history(
        monkeypatch, _history([GAME_A], [30]), _history([GAME_A, GAME_B], [20, 25])
    )
    source = ls.LineSource(GAME_DATE, fetch_covers=FakeCovers())
    history = _read(source)
    assert source.history_sources == {GAME_A: ls.SBR, GAME_B: ls.COVERS}
    by_game = history.ticks.groupby("game_id")["minutes_before_tip"].min().to_dict()
    assert by_game == {GAME_A: 30.0, GAME_B: 25.0}


def test_history_without_any_sbr_tick_switches_the_run(monkeypatch):
    empty = _history([GAME_A], [30])
    empty = LiveLineHistory(empty.ticks.iloc[0:0], empty.games.iloc[0:0], AS_OF)
    _patch_history(monkeypatch, empty, _history([GAME_A, GAME_B], [20, 25]))
    source = ls.LineSource(GAME_DATE, fetch_covers=FakeCovers())
    _read(source)
    assert source.uses_covers
    assert source.history_sources == {GAME_A: ls.COVERS, GAME_B: ls.COVERS}


def test_a_truncated_tipped_game_moves_only_to_a_fresher_covers(monkeypatch):
    after_tip = pd.Timestamp(TIPOFF) + pd.Timedelta(hours=3)
    # SBR stops 240 min before tip for both; Covers goes on for A only.
    _patch_history(
        monkeypatch,
        _history([GAME_A, GAME_B], [240, 240]),
        _history([GAME_A, GAME_B], [5, 300]),
    )
    source = ls.LineSource(GAME_DATE, fetch_covers=FakeCovers())
    history = _read(source, as_of=after_tip)
    assert source.history_sources == {GAME_A: ls.COVERS, GAME_B: ls.SBR}
    by_game = history.ticks.groupby("game_id")["minutes_before_tip"].min().to_dict()
    assert by_game == {GAME_A: 5.0, GAME_B: 240.0}
