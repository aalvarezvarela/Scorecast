"""Tonight's SBR day-page odds, rebuilt from Covers line history.

The closing prediction frame reads SBR's day pages (``scrape_sportsbook_days``):
per book, the current line of every market, plus the consensus opener and
pick percentages. When SBR is down or frozen, the prediction falls back to
Covers, whose line history is the same feed (``docs/odds/covers_line_history.md``).

Rather than map Covers ticks onto the final columns, this builds the payload
SBR's JSON engine parses -- one ``oddsTableModel`` per market -- and runs it
through the engine's own parsers and day merge. Every column the SBR path
yields therefore comes out with the same name, order and rounding, and no
other: the book list is SBR's (``SBR_DAY_PAGE_BOOKS``), so a book only Covers
carries (theScore) never gets a column, and the eighth book (Hard Rock Bet) is
dropped by the parsers exactly as on SBR.

Per book and market the current line is the latest pre-tip tick at or before
``as_of``: what SBR's day page shows before tip. The opener is Covers'
``consensus_opener`` (BetMGM's first quote, as on SBR). The consensus pick
percentages are the shares of the Covers Consensus pick counts
(``parse.PickConsensus``), which SBR's are; a game without them gets SBR's
empty consensus.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import (
    _merge_day,
    _with_season,
    attach_scrape_metadata,
    parse_sbr_moneyline_rows,
    parse_sbr_spread_rows,
    parse_sbr_totals_rows,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)

from .books import CONSENSUS_OPENER_ID, book_slug
from .parse import PickConsensus

#: SBR's day-page sportsbooks, as its payload names and orders them (read
#: 2026-10-24), with the line-history slug of each. The page draws the first
#: ``scrape_sportsbook.RENDERED_SPORTSBOOKS``; the parsers drop the rest.
SBR_DAY_PAGE_BOOKS: tuple[tuple[str, str], ...] = (
    ("BetMGM", "betmgm"),
    ("bet365", "bet365"),
    ("DraftKings", "draftkings"),
    ("FanDuel", "fanduel"),
    ("Fanatics Sportsbook", "fanatics_sportsbook"),
    ("BetRivers", "betrivers"),
    ("Caesars Odds", "caesars"),
    ("Hard Rock Bet", "hard_rock_bet"),
)

#: Day-page market -> (line-history market, parser).
_MARKETS = {
    "totals": ("totals", parse_sbr_totals_rows),
    "spread": ("point_spread", parse_sbr_spread_rows),
    "ml": ("money_line", parse_sbr_moneyline_rows),
}

CONSENSUS_OPENER = book_slug(CONSENSUS_OPENER_ID)


def _as_utc(moment) -> pd.Timestamp:
    moment = pd.Timestamp(moment)
    return (
        moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    )


def latest_tick(
    ticks: Iterable[LineTick], *, market: str, book: str, as_of: pd.Timestamp
) -> LineTick | None:
    """The book's latest pre-tip quote of ``market`` known at ``as_of``."""
    known = [
        tick
        for tick in ticks
        if tick.market == market
        and tick.book_slug == book
        and tick.is_pregame
        and _as_utc(tick.line_ts) <= as_of
    ]
    return max(known, key=lambda tick: tick.line_ts) if known else None


def _line_view(market: str, tick: LineTick | None) -> dict[str, Any]:
    """A tick as SBR's ``currentLine``/``openingLine``: left is away (OVER)."""
    if tick is None:
        return {}
    if market == "totals":
        total = tick.left_line if tick.left_line is not None else tick.right_line
        return {
            "total": total,
            "overOdds": tick.left_price,
            "underOdds": tick.right_price,
        }
    if market == "point_spread":
        away, home = tick.left_line, tick.right_line
        if away is None and home is not None:
            away = -home
        if home is None and away is not None:
            home = -away
        return {
            "awaySpread": away,
            "homeSpread": home,
            "awayOdds": tick.left_price,
            "homeOdds": tick.right_price,
        }
    return {"awayOdds": tick.left_price, "homeOdds": tick.right_price}


def _game_row(
    game: ScrapedGame,
    market: str,
    as_of: pd.Timestamp,
    consensus: PickConsensus | None,
) -> dict[str, Any]:
    current = [
        latest_tick(game.ticks, market=market, book=slug, as_of=as_of)
        for _, slug in SBR_DAY_PAGE_BOOKS
    ]
    opener = latest_tick(game.ticks, market=market, book=CONSENSUS_OPENER, as_of=as_of)
    return {
        "gameView": {
            "gameId": game.event_id,
            "startDate": _as_utc(game.tipoff_utc).isoformat(),
            "gameStatusText": game.status_text,
            # Before tip SBR serves zero scores, not empty ones.
            "awayTeamScore": game.away_score or 0,
            "homeTeamScore": game.home_score or 0,
            "awayTeam": {"displayName": game.team_away},
            "homeTeam": {"displayName": game.team_home},
            "consensus": {} if consensus is None else consensus.sbr_consensus(),
        },
        "oddsViews": [
            None if tick is None else {"currentLine": _line_view(market, tick)}
            for tick in current
        ],
        "openingLineViews": [
            None if opener is None else {"openingLine": _line_view(market, opener)}
        ],
    }


def day_page_models(
    games: Iterable[ScrapedGame],
    *,
    as_of: pd.Timestamp,
    consensus: Mapping[int, PickConsensus] | None = None,
) -> dict[str, dict[str, Any]]:
    """Day-page market -> the ``oddsTableModel`` SBR's JSON engine would parse.

    ``consensus`` maps a game's ``event_id`` (its Covers id) to its pick counts.
    """
    as_of = _as_utc(as_of)
    consensus = consensus or {}
    games = sorted(games, key=lambda game: (game.tipoff_utc, game.event_id))
    sportsbooks = [{"name": name} for name, _ in SBR_DAY_PAGE_BOOKS]
    return {
        name: {
            "sportsbooks": sportsbooks,
            "gameRows": [
                _game_row(game, market, as_of, consensus.get(int(game.event_id)))
                for game in games
            ],
        }
        for name, (market, _) in _MARKETS.items()
    }


def covers_day_odds(
    games: Iterable[ScrapedGame],
    *,
    game_date: date,
    as_of: datetime | pd.Timestamp | None = None,
    consensus: Mapping[int, PickConsensus] | None = None,
) -> pd.DataFrame:
    """``scrape_sportsbook_days([game_date])``'s frame, from Covers ticks and,
    when given, pick counts (by Covers id).

    Empty when no game is given, as SBR's is for a date without odds.
    """
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else _as_utc(as_of)
    models = day_page_models(games, as_of=as_of, consensus=consensus)
    if not models["totals"]["gameRows"]:
        return pd.DataFrame()
    raw = {
        name: _with_season(parse(models[name], game_date), game_date)
        for name, (_, parse) in _MARKETS.items()
    }
    if raw["totals"] is None:
        return pd.DataFrame()
    return attach_scrape_metadata(
        _merge_day(raw["totals"], raw["spread"], raw["ml"]), models["totals"], as_of
    )
