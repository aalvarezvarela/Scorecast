"""Covers as the prediction's SBR fallback: the day-page odds and live history.

The closing frame must come out of Covers with SBR's columns exactly, so
nothing downstream can tell the source apart but the values: the test builds
the payload SBR would serve for the same quotes and compares the two frames
cell for cell. The live line history must hold SBR's books only.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from nba_ou.fetch_data.odds_covers.day_odds import (
    SBR_DAY_PAGE_BOOKS,
    covers_day_odds,
    latest_tick,
)
from nba_ou.fetch_data.odds_covers.parse import PickConsensus, parse_consensus
from nba_ou.fetch_data.odds_sportsbook import scrape_sportsbook as sbr
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven.live import LiveLineHistory
from nba_ou.postgre_db.line_history_covers.live import (
    CoversLiveDay,
    covers_freshness,
    covers_live_line_history,
    fill_live_line_history,
    truncated_game_ids,
)

GAME_DATE = date(2026, 10, 24)
TIPOFF = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)
AS_OF = pd.Timestamp("2026-10-24 22:00", tz="UTC")
GAME_ID = "0022600104"


def _tick(market, book, minutes, left, right, *, opener=False) -> LineTick:
    return LineTick(
        market=market,
        book_slug=book,
        book_name=book,
        line_ts=TIPOFF + timedelta(minutes=minutes),
        minutes_to_tip=minutes,
        is_opener=opener,
        left_line=left[0],
        left_price=left[1],
        right_line=right[0],
        right_price=right[1],
    )


def _ticks() -> list[LineTick]:
    return [
        # bet365: an older quote, the current one, and an in-play one.
        _tick("totals", "bet365", -600, (226.5, -110), (226.5, -110), opener=True),
        _tick("totals", "bet365", -75, (228.5, -106), (228.5, -114)),
        _tick("totals", "bet365", 15, (240.5, -110), (240.5, -110)),
        _tick("point_spread", "bet365", -75, (-3.5, -113), (3.5, -107)),
        _tick("money_line", "bet365", -75, (None, -155), (None, 130)),
        # BetMGM's opener is the consensus opener (Covers rebuilds it).
        _tick("totals", "betmgm", -900, (231.5, -275), (231.5, 200), opener=True),
        _tick(
            "totals", "consensus_opener", -900, (231.5, -275), (231.5, 200), opener=True
        ),
        # After as_of: not known yet.
        _tick("totals", "fanduel", -30, (229.5, -110), (229.5, -110)),
        # Books SBR's day page does not draw.
        _tick("totals", "thescore", -70, (228.5, -115), (228.5, -105)),
        _tick("totals", "hard_rock_bet", -70, (228.5, -110), (228.5, -110)),
    ]


def _game(ticks=None, *, tipoff=TIPOFF) -> ScrapedGame:
    return ScrapedGame(
        event_id=388590,
        game_date=GAME_DATE,
        season_year=2026,
        tipoff_utc=tipoff,
        team_away="Miami Heat",
        team_home="Charlotte Hornets",
        status_text="Scheduled",
        away_score=None,
        home_score=None,
        ticks=tuple(_ticks() if ticks is None else ticks),
    )


def _sbr_payload(consensus: dict | None = None) -> dict[str, dict]:
    """What SBR's day pages serve for the same quotes, as of ``AS_OF``."""
    books = [{"name": name} for name, _ in SBR_DAY_PAGE_BOOKS]
    view = {
        "gameId": 388590,
        "startDate": TIPOFF.isoformat(),
        "gameStatusText": "Scheduled",
        "awayTeamScore": 0,
        "homeTeamScore": 0,
        "awayTeam": {"displayName": "Miami Heat"},
        "homeTeam": {"displayName": "Charlotte Hornets"},
        "consensus": consensus or {},
    }

    def row(current: dict, opening: dict | None, betmgm: dict | None = None) -> dict:
        return {
            "gameView": view,
            # BetMGM and bet365 are the first two books; the others have no quote.
            "oddsViews": [
                None if betmgm is None else {"currentLine": betmgm},
                {"currentLine": current},
                *[None] * 6,
            ],
            "openingLineViews": [None if opening is None else {"openingLine": opening}],
        }

    return {
        "totals": {
            "sportsbooks": books,
            "gameRows": [
                row(
                    {"total": 228.5, "overOdds": -106, "underOdds": -114},
                    {"total": 231.5, "overOdds": -275, "underOdds": 200},
                    # BetMGM's only quote is its opener, so also its current line.
                    betmgm={"total": 231.5, "overOdds": -275, "underOdds": 200},
                )
            ],
        },
        "spread": {
            "sportsbooks": books,
            "gameRows": [
                row(
                    {
                        "awaySpread": -3.5,
                        "homeSpread": 3.5,
                        "awayOdds": -113,
                        "homeOdds": -107,
                    },
                    None,
                )
            ],
        },
        "ml": {
            "sportsbooks": books,
            "gameRows": [row({"awayOdds": -155, "homeOdds": 130}, None)],
        },
    }


def _sbr_frame(consensus: dict | None = None) -> pd.DataFrame:
    models = _sbr_payload(consensus)
    raw = {
        name: sbr._with_season(parse(models[name], GAME_DATE), GAME_DATE)
        for name, parse in (
            ("totals", sbr.parse_sbr_totals_rows),
            ("spread", sbr.parse_sbr_spread_rows),
            ("ml", sbr.parse_sbr_moneyline_rows),
        )
    }
    return sbr.attach_scrape_metadata(
        sbr._merge_day(raw["totals"], raw["spread"], raw["ml"]), models["totals"], AS_OF
    )


# --- the closing frame's day-page odds -----------------------------------------


def test_covers_day_odds_is_the_frame_sbr_serves_for_the_same_quotes():
    covers = covers_day_odds([_game()], game_date=GAME_DATE, as_of=AS_OF)
    pd.testing.assert_frame_equal(covers, _sbr_frame())


def test_pick_counts_become_sbr_consensus_percentages():
    counts = PickConsensus(
        spread_away=312, spread_home=410, total_over=213, total_under=275
    )
    covers = covers_day_odds(
        [_game()], game_date=GAME_DATE, as_of=AS_OF, consensus={388590: counts}
    )
    pd.testing.assert_frame_equal(covers, _sbr_frame(counts.sbr_consensus()))
    # SBR's day page rounds the shares: 410/722 -> 57, 213/488 -> 44.
    row = covers.iloc[0]
    assert row["spread_consensus_pct_home"] == 57
    assert row["total_consensus_pct_over"] == 44


CONSENSUS_PAGE = """
<div id="consensus_analysis_content">
  <div class="covers-CoversConsensusDetailsTable-finalWagersleft">368</div>
  <div class="covers-CoversConsensusDetailsTable-picksFinal--center">Total Picks</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersRight">198</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersleft"> 167 </div>
  <div class="covers-CoversConsensusDetailsTable-picksFinal--center">Total Picks</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersRight"> 197 </div>
</div>
<div id="experts_analysis_content" style="display: none">
  <div class="covers-CoversConsensusDetailsTable-finalWagersleft">21</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersRight">7</div>
</div>
"""


def test_consensus_page_reads_the_public_ats_and_ou_totals_only():
    assert parse_consensus(CONSENSUS_PAGE) == PickConsensus(368, 198, 167, 197)
    shares = parse_consensus(CONSENSUS_PAGE).sbr_consensus()
    assert shares["awaySpreadPickPercent"] == pytest.approx(65.018, abs=1e-3)
    assert shares["overPickPercent"] == pytest.approx(45.879, abs=1e-3)
    assert parse_consensus("<html></html>") is None
    assert PickConsensus(0, 0, 0, 0).sbr_consensus()["overPickPercent"] == 0.0


def test_covers_day_odds_has_no_column_sbr_lacks():
    columns = covers_day_odds([_game()], game_date=GAME_DATE, as_of=AS_OF).columns
    assert not [c for c in columns if "thescore" in c or "hard_rock" in c]
    assert "total_bet365_line_over" in columns
    assert "total_consensus_opener_line_over" in columns


def test_current_line_is_the_latest_pregame_tick_known_at_as_of():
    ticks = _ticks()
    tick = latest_tick(ticks, market="totals", book="bet365", as_of=AS_OF)
    assert (tick.left_line, tick.minutes_to_tip) == (228.5, -75)
    earlier = latest_tick(
        ticks, market="totals", book="bet365", as_of=TIPOFF - timedelta(minutes=300)
    )
    assert earlier.left_line == 226.5
    # In play, never a current line: the day page before tip shows the close.
    after_tip = latest_tick(
        ticks, market="totals", book="bet365", as_of=TIPOFF + timedelta(hours=1)
    )
    assert after_tip.left_line == 228.5
    assert latest_tick(ticks, market="totals", book="fanduel", as_of=AS_OF) is None


def test_no_game_is_an_empty_frame():
    assert covers_day_odds([], game_date=GAME_DATE, as_of=AS_OF).empty


# --- the intermediate frame's live line history -------------------------------


def _team_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [GAME_ID, GAME_ID],
            "game_date": [GAME_DATE] * 2,
            "team_name": ["Charlotte Hornets", "Miami Heat"],
            "home": [True, False],
            "season_year": [2026, 2026],
        }
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "team_home": ["CHA"],
            "team_away": ["MIA"],
            "tipoff_utc": [TIPOFF],
        }
    )


def _day(game: ScrapedGame | None = None, *, as_of=AS_OF) -> CoversLiveDay:
    return CoversLiveDay(
        game_date=GAME_DATE, as_of=as_of, games={GAME_ID: game or _game()}
    )


def test_live_history_keeps_only_the_books_sbr_stores():
    history = covers_live_line_history(
        _day(), team_games=_team_games(), schedule=_schedule()
    )
    assert set(history.ticks["book"]) == {"bet365", "betmgm", "hard_rock_bet"}
    assert set(history.ticks["game_id"]) == {GAME_ID}
    # Cut at the fetch's moment, pre-game only.
    assert history.ticks["line_ts"].max() <= AS_OF
    assert history.games["team_home"].tolist() == ["CHA"]


def test_live_history_is_never_later_than_the_fetch():
    later = AS_OF + pd.Timedelta(hours=1)
    history = covers_live_line_history(
        _day(), team_games=_team_games(), schedule=_schedule(), as_of=later
    )
    assert history.as_of == AS_OF


def test_covers_freshness_reads_the_fetch_moment():
    freshness = covers_freshness(_day())
    assert freshness.source == "Covers"
    assert freshness.describe().startswith("Covers feed")
    # The latest tick known at the fetch: T-70 (theScore, Hard Rock).
    assert freshness.quiet_minutes == pytest.approx(70 - 60)
    assert not freshness.stale


def _history(game_ids, minutes_before_tip, tipoff=TIPOFF) -> LiveLineHistory:
    ticks = pd.DataFrame(
        {
            "game_id": game_ids,
            "season_year": 2026,
            "market": "totals",
            "book": "bet365",
            "line_ts": [tipoff - timedelta(minutes=m) for m in minutes_before_tip],
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
            "tipoff_utc": tipoff,
            "team_home": "CHA",
            "team_away": "MIA",
        }
    )
    return LiveLineHistory(ticks, games, AS_OF)


def test_fill_takes_the_fallback_games_and_keeps_the_others():
    sbr_history = _history(["a", "b"], [30, 40])
    covers = _history(["b", "c"], [10, 20])
    filled = fill_live_line_history(sbr_history, covers, ["b", "c"])
    by_game = filled.ticks.set_index("game_id")["minutes_before_tip"].to_dict()
    assert by_game == {"a": 30.0, "b": 10.0, "c": 20.0}
    assert filled.games["game_id"].tolist() == ["a", "b", "c"]
    assert filled.as_of == sbr_history.as_of


def test_only_a_tipped_game_can_be_truncated():
    history = _history(["frozen", "fine"], [240, 5])
    before_tip = TIPOFF - timedelta(minutes=1)
    assert truncated_game_ids(history, before_tip) == set()
    assert truncated_game_ids(history, TIPOFF + timedelta(hours=3)) == {"frozen"}
