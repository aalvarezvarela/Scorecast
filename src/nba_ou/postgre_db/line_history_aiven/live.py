"""Tonight's line history, read live for a prediction and never stored.

The store only ever holds finished games (``ingest.is_finished``): a game is
written the day after it is played, with its whole history, the same rule the
closing-odds tables follow. A prediction made before tip-off still needs the
history *so far*, so this module scrapes it into memory and hands back frames
shaped exactly like the store's reads -- ``fetch_pregame_ticks`` and
``fetch_games`` -- for ``create_intermediate_line_df`` to append to them.

The scraped games go through the ingest's own ``build_frames`` (game-id
resolution, the spread-price-bleed and implausible-line repairs, the storage
encodings) and the fetch's own ``_decode_ticks``: the only step skipped is the
insert. A tick read live is therefore the value the store would have returned
for it the next day.

``as_of`` truncates the history to what was known at that moment. Live, it is
simply "now"; replaying a past date with an earlier ``as_of`` is how the
train/serve parity of the whole path is tested.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    ScrapedGame,
    discover_games_for_date,
    new_session,
    scrape_events,
)
from nba_ou.postgre_db.config.db_config import connect_line_history_db

from . import ingest as ingest_mod
from . import schema as schema_mod
from .fetch import (
    ALL_MARKETS,
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    GAME_COLUMNS,
    TICK_COLUMNS,
    _decode_ticks,
)
from .schema import SCHEMA


@dataclass(frozen=True)
class LiveLineHistory:
    """One date's line history as of ``as_of``, in the store's read shapes."""

    ticks: pd.DataFrame
    games: pd.DataFrame
    as_of: pd.Timestamp
    unmatched_games: list[str] = field(default_factory=list)

    @property
    def game_ids(self) -> list[str]:
        return sorted(self.games["game_id"].astype(str).unique())


def scheduled_team_games(scheduled_games: pd.DataFrame, *, season_year: int) -> pd.DataFrame:
    """Today's NBA schedule as the team-game rows the ingest resolves against.

    ``ingest.build_game_index`` expects ``nba_games``-shaped rows (one per team,
    with ``home``), which tonight's games do not have yet. The schedule feed
    carries the same GAME_ID and team ids, so this builds those rows from it.
    """
    id_to_name = {str(team_id): name for name, team_id in TEAM_ID_MAP.items()}
    rows = []
    for game in scheduled_games.itertuples(index=False):
        game_date = pd.to_datetime(game.GAME_DATE_EST).date()
        for team_id, home in ((game.HOME_TEAM_ID, True), (game.VISITOR_TEAM_ID, False)):
            rows.append(
                {
                    "game_id": str(game.GAME_ID),
                    "game_date": game_date,
                    "team_name": id_to_name.get(str(team_id)),
                    "home": home,
                    "season_year": season_year,
                }
            )
    return pd.DataFrame(rows, columns=["game_id", "game_date", "team_name", "home", "season_year"])


def live_frames_from_scraped(
    games: Sequence[ScrapedGame],
    *,
    team_games: pd.DataFrame,
    book_ids: dict[str, int],
    as_of: pd.Timestamp,
    schedule: pd.DataFrame | None = None,
    markets: tuple[str, ...] = ALL_MARKETS,
    exclude_books: tuple[str, ...] = (),
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
) -> LiveLineHistory:
    """Scraped games -> the frames the store would return for them. Pure.

    Same filters as ``fetch_pregame_ticks``: pre-game only, at least
    ``min_minutes_before_tip`` before tip, the requested markets, the excluded
    books dropped. On top of that, nothing after ``as_of``.
    """
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    market_ids = schema_mod.market_ids()
    rows, game_dim, stats = ingest_mod.build_frames(
        games,
        game_index=ingest_mod.build_game_index(team_games),
        book_ids=book_ids,
        market_ids=market_ids,
        # The season schedule feed: it supplies the team codes the store's game
        # rows carry (CHA, not "Charlotte Hornets"), which the prior-game line
        # dynamics join on. Without it the live rows would name teams
        # differently from every stored game.
        schedule=schedule,
    )

    games_frame = game_dim.reindex(columns=GAME_COLUMNS).copy()
    if not games_frame.empty:
        games_frame["game_id"] = games_frame["game_id"].astype(str)
        games_frame["game_date"] = pd.to_datetime(games_frame["game_date"])
        games_frame["tipoff_utc"] = pd.to_datetime(games_frame["tipoff_utc"], utc=True)
        games_frame["season_year"] = pd.to_numeric(games_frame["season_year"]).astype("int64")
    games_frame = games_frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    if rows.empty:
        return LiveLineHistory(
            pd.DataFrame(columns=TICK_COLUMNS), games_frame, as_of, stats.unmatched_games
        )

    code_by_market = {market_id: code for code, market_id in market_ids.items()}
    slug_by_book = {book_id: slug for slug, book_id in book_ids.items()}
    line_ts = pd.to_datetime(rows["line_ts"], utc=True)
    keep = (
        rows["is_pregame"].astype(bool)
        & (pd.to_numeric(rows["mins_to_tip"]) <= -int(min_minutes_before_tip))
        & (line_ts <= as_of)
    )
    rows = rows[keep]
    # The store hands psycopg ints and NULLs back as Python objects, which
    # _decode_ticks then infers into int64/float64. The encoded columns here are
    # pandas nullable Int64; route them through the same objects so the decoded
    # dtypes match a stored read rather than coming out as Int64/Float64.
    for column in ["left_line", "left_price", "right_line", "right_price", "mins_to_tip"]:
        values = rows[column].astype(object)
        rows = rows.assign(**{column: values.where(rows[column].notna(), None)})
    ticks = pd.DataFrame(
        {
            "game_id": rows["game_id"].astype(str),
            "season_year": rows["season_year"],
            "market": rows["market_id"].map(code_by_market),
            "book": rows["book_id"].map(slug_by_book),
            "line_ts": rows["line_ts"],
            # Named as the store's query aliases it; _decode_ticks flips the sign.
            "minutes_before_tip": rows["mins_to_tip"],
            "is_opener": rows["is_opener"],
            "left_line": rows["left_line"],
            "left_price": rows["left_price"],
            "right_line": rows["right_line"],
            "right_price": rows["right_price"],
        },
        columns=TICK_COLUMNS,
    )
    ticks = ticks[ticks["market"].isin(markets) & ~ticks["book"].isin(exclude_books)]
    # The store's read order, so every downstream groupby sees the same sequence.
    ticks = ticks.sort_values(["game_id", "market", "book", "line_ts"])
    return LiveLineHistory(
        _decode_ticks(ticks.reset_index(drop=True)),
        games_frame,
        as_of,
        stats.unmatched_games,
    )


def read_book_ids(conn: psycopg.Connection | None = None) -> dict[str, int]:
    """The store's slug -> book_id map. Read-only: a book the store has never
    held is left out, exactly as it would be absent from a training read."""
    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT slug, book_id FROM {}.lh_book").format(sql.Identifier(SCHEMA))
            )
            return {slug: book_id for slug, book_id in cur.fetchall()}
    finally:
        if owned:
            conn.close()


def fetch_live_line_history(
    game_date: date,
    *,
    team_games: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    markets: tuple[str, ...] = ALL_MARKETS,
    exclude_books: tuple[str, ...] = (),
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
    conn: psycopg.Connection | None = None,
) -> LiveLineHistory:
    """Scrape ``game_date``'s line history into memory. Writes nothing.

    ``team_games`` maps SBR's games onto NBA GAME_IDs: tonight's comes from
    ``scheduled_team_games``; a replayed past date can pass ``nba_games`` rows.
    ``schedule`` is the season schedule feed (``fetch_schedules``), the source
    of the team codes the store writes -- pass it, or team names will not match
    stored games.
    """
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    session = new_session()
    try:
        summaries = discover_games_for_date(session, game_date)
        scraped = list(
            scrape_events([s.event_id for s in summaries], session=session, markets=markets)
        )
    finally:
        session.close()
    return live_frames_from_scraped(
        scraped,
        team_games=team_games,
        book_ids=read_book_ids(conn),
        as_of=as_of,
        schedule=schedule,
        markets=markets,
        exclude_books=exclude_books,
        min_minutes_before_tip=min_minutes_before_tip,
    )
