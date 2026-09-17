"""Bring one season of line history up to date.

Shaped like the other per-source updaters (``update_odds_sportsbook_database``
and friends): one call, one season, safe to run daily.

A run does three things, in order of how often they matter:

1. **Refresh a rolling window of recent dates.** A game's lines keep moving
   right up to tipoff, so a game fetched on the morning it is played is
   *present* but not *final*. Presence alone would never bring it back, which
   is why the recent window is re-fetched unconditionally rather than diffed.
2. **Fill games the store has never seen**, using ``nba_games`` as the
   reference for what should exist.
3. **Top up games missing a book that most games on their own date carry** --
   the signature of a partial scrape, as opposed to a book that had simply not
   launched yet or skipped that one game.
4. **Optionally backfill named books** (``backfill_books``): re-fetch every
   stored game with no tick at all for them. Step 3 cannot do this for a book
   the store has never held, because no game on any date "expects" it.

All writes are insert-only, so re-fetching costs a request and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    ALL_MARKETS,
    ScrapedGame,
    discover_games_for_date,
    new_session,
    scrape_events,
)

from . import ingest as ingest_mod
from .schema import SCHEMA

#: SBR stopped serving Caesars, so its absence is not a gap and a game must
#: never be re-fetched on account of it -- that would mean re-fetching those
#: games forever, waiting for data the source no longer has. Historical Caesars
#: rows stay exactly where they are; they are simply not *expected* any more.
DISCONTINUED_BOOKS: frozenset[str] = frozenset({"caesars"})

#: A book counts as *expected* on a date only once it priced at least this
#: share of that date's games. Guards against a book's launch day (one game out
#: of the slate) making every other game on that date look partial.
DEFAULT_EXPECTED_BOOK_SHARE = 0.5

#: Scraped games are written every this many dates rather than once at the end,
#: so a multi-season backfill that fails late keeps what it already fetched.
#: Inserts are idempotent, so re-running after a failure only re-requests.
DEFAULT_FLUSH_EVERY_DATES = 7

#: How far back the unconditional refresh reaches. Three days covers a run that
#: was skipped, plus the gap between a morning fetch and the closing line.
DEFAULT_REFRESH_DAYS = 3


@dataclass
class UpdateResult:
    season_year: int
    refreshed_dates: list[date] = field(default_factory=list)
    gap_dates: list[date] = field(default_factory=list)
    incomplete_dates: list[date] = field(default_factory=list)
    backfill_dates: list[date] = field(default_factory=list)
    scraped_games: int = 0
    inserted_ticks: int = 0
    inserted_games: int = 0
    retimed_games: list[str] = field(default_factory=list)
    failed_dates: list[str] = field(default_factory=list)

    @property
    def target_dates(self) -> list[date]:
        return sorted(
            set(self.refreshed_dates)
            | set(self.gap_dates)
            | set(self.incomplete_dates)
            | set(self.backfill_dates)
        )


def current_season_year(today: date | None = None) -> int:
    """NBA season start year for ``today`` (Jan 2026 belongs to 2025-26)."""
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def resolve_season_years(
    start: int | None = None,
    end: int | None = None,
    *,
    today: date | None = None,
) -> list[int]:
    """Season years to update, defaulting to the current one.

    Defaulting rather than hardcoding means the daily job needs no yearly edit:
    in-season it resolves to the season in progress, out of season to the one
    just finished.
    """
    if start is None and end is None:
        return [current_season_year(today)]

    first = int(start if start is not None else end)  # type: ignore[arg-type]
    last = int(end if end is not None else start)  # type: ignore[arg-type]
    if first > last:
        raise ValueError(f"start season {first} is after end season {last}")
    return list(range(first, last + 1))


def _discontinued_book_ids(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT book_id, slug FROM {}.lh_book").format(
                sql.Identifier(SCHEMA)
            )
        )
        return [
            book_id for book_id, slug in cur.fetchall() if slug in DISCONTINUED_BOOKS
        ]


def find_incomplete_games(
    conn: psycopg.Connection,
    season_year: int,
    *,
    min_share: float = DEFAULT_EXPECTED_BOOK_SHARE,
) -> pd.DataFrame:
    """Stored games missing a book that most games on their own date carry.

    A book is *expected* on a date only if it priced at least ``min_share`` of
    that date's games. Two things make that threshold necessary rather than
    just comparing against the best-covered game:

    * A book's launch day has it on one game out of the slate (Fanatics, first
      seen 2025-11-05, covered 1 of 11). Treating the best-covered game as the
      bar would mark the other ten partial forever, since re-fetching can never
      produce data the source never had.
    * Books occasionally skip an individual game, which is a real absence
      rather than a scrape failure.

    Discontinued books are excluded outright, so a missing Caesars never makes
    a game look partial.
    """
    excluded = _discontinued_book_ids(conn) or [-1]
    query = sql.SQL(
        """
        WITH game_books AS (
            SELECT DISTINCT g.game_id, g.game_date, l.book_id
            FROM {schema}.lh_game g
            JOIN {schema}.lh_line l USING (game_id)
            WHERE g.season_year = %s
              AND l.book_id <> ALL(%s)
        ),
        games_per_date AS (
            SELECT game_date, COUNT(DISTINCT game_id) AS n_games
            FROM game_books GROUP BY game_date
        ),
        expected AS (
            SELECT gb.game_date, gb.book_id
            FROM game_books gb
            JOIN games_per_date d USING (game_date)
            GROUP BY gb.game_date, gb.book_id, d.n_games
            HAVING COUNT(DISTINCT gb.game_id)::numeric / d.n_games >= %s
        ),
        games AS (SELECT DISTINCT game_id, game_date FROM game_books)
        SELECT g.game_id, g.game_date, COUNT(*) AS missing_books
        FROM games g
        JOIN expected e USING (game_date)
        LEFT JOIN game_books gb
               ON gb.game_id = g.game_id AND gb.book_id = e.book_id
        WHERE gb.book_id IS NULL
        GROUP BY g.game_id, g.game_date
        ORDER BY g.game_date, g.game_id
        """
    ).format(schema=sql.Identifier(SCHEMA))

    with conn.cursor() as cur:
        cur.execute(query, (int(season_year), excluded, float(min_share)))
        rows = cur.fetchall()

    return pd.DataFrame(rows, columns=["game_id", "game_date", "missing_books"])


def find_games_missing_books(
    conn: psycopg.Connection,
    season_year: int,
    books: tuple[str, ...],
) -> pd.DataFrame:
    """Stored games of ``season_year`` holding no tick for at least one of ``books``.

    Unlike ``find_incomplete_games`` this needs no evidence that the book is
    expected on the date, which is exactly what lets it backfill a book the
    store has never held. The price is that a game the book genuinely never
    priced (a season before it existed on SBR) is re-fetched on every run that
    asks for it -- so this only runs when a backfill is requested explicitly.
    """
    slugs = sorted({str(b).strip().lower() for b in books if str(b).strip()})
    if not slugs:
        return pd.DataFrame(columns=["game_id", "game_date"])

    query = sql.SQL(
        """
        SELECT g.game_id, g.game_date
        FROM {schema}.lh_game g
        WHERE g.season_year = %s
          AND (
            SELECT COUNT(DISTINCT b.slug)
            FROM {schema}.lh_line l
            JOIN {schema}.lh_book b USING (book_id)
            WHERE l.game_id = g.game_id
              AND l.season_year = g.season_year
              AND b.slug = ANY(%s)
          ) < %s
        ORDER BY g.game_date, g.game_id
        """
    ).format(schema=sql.Identifier(SCHEMA))

    with conn.cursor() as cur:
        cur.execute(query, (int(season_year), slugs, len(slugs)))
        rows = cur.fetchall()

    return pd.DataFrame(rows, columns=["game_id", "game_date"])


def season_game_dates(games_df: pd.DataFrame, season_year: int) -> set[date]:
    if games_df.empty or "season_year" not in games_df.columns:
        return set()
    season = games_df[
        pd.to_numeric(games_df["season_year"], errors="coerce") == int(season_year)
    ]
    dates = pd.to_datetime(season["game_date"], errors="coerce").dt.date
    return {d for d in dates if d is not None and not pd.isna(d)}


def recent_dates(
    games_df: pd.DataFrame,
    season_year: int,
    *,
    refresh_days: int,
    today: date | None = None,
) -> list[date]:
    """Dates in the last ``refresh_days`` that actually had games.

    Out of season this is empty, so an offseason run does gap-filling only.
    """
    if refresh_days <= 0:
        return []
    today = today or date.today()
    window_start = today - timedelta(days=refresh_days)
    return sorted(
        d
        for d in season_game_dates(games_df, season_year)
        if window_start <= d <= today
    )


def plan_update(
    conn: psycopg.Connection,
    games_df: pd.DataFrame,
    season_year: int,
    *,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    include_incomplete: bool = True,
    min_book_share: float = DEFAULT_EXPECTED_BOOK_SHARE,
    backfill_books: tuple[str, ...] = (),
    today: date | None = None,
) -> UpdateResult:
    """Work out which dates need fetching, without fetching anything."""
    result = UpdateResult(season_year=season_year)
    result.refreshed_dates = recent_dates(
        games_df, season_year, refresh_days=refresh_days, today=today
    )

    season_dates = season_game_dates(games_df, season_year)
    missing = ingest_mod.find_missing_games(conn, games_df)
    if not missing.empty:
        result.gap_dates = sorted(
            d for d in ingest_mod.missing_dates(missing) if d in season_dates
        )

    if include_incomplete:
        incomplete = find_incomplete_games(conn, season_year, min_share=min_book_share)
        if not incomplete.empty:
            result.incomplete_dates = sorted(set(incomplete["game_date"]))

    if backfill_books:
        lacking = find_games_missing_books(conn, season_year, tuple(backfill_books))
        if not lacking.empty:
            result.backfill_dates = sorted(set(lacking["game_date"]))

    return result


def update_line_history_database(
    season_year: int,
    *,
    games_df: pd.DataFrame,
    conn: psycopg.Connection,
    schedule: pd.DataFrame | None = None,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    include_incomplete: bool = True,
    min_book_share: float = DEFAULT_EXPECTED_BOOK_SHARE,
    markets: tuple[str, ...] = ALL_MARKETS,
    backfill_books: tuple[str, ...] = (),
    flush_every_dates: int = DEFAULT_FLUSH_EVERY_DATES,
    dry_run: bool = False,
    today: date | None = None,
    progress: bool = True,
) -> UpdateResult:
    """Bring ``season_year`` up to date. Safe to run repeatedly.

    ``dry_run`` reports the plan and stops before the first request, so it can
    be used to see what a run would touch without loading SBR at all.
    """
    result = plan_update(
        conn,
        games_df,
        season_year,
        refresh_days=refresh_days,
        include_incomplete=include_incomplete,
        min_book_share=min_book_share,
        backfill_books=backfill_books,
        today=today,
    )
    dates = result.target_dates
    if progress:
        print(
            f"Season {season_year}: {len(dates)} date(s) to fetch "
            f"({len(result.refreshed_dates)} recent, {len(result.gap_dates)} gaps, "
            f"{len(result.incomplete_dates)} partial, "
            f"{len(result.backfill_dates)} backfill)"
        )
    if not dates or dry_run:
        return result

    def flush(batch: list[ScrapedGame]) -> None:
        if not batch:
            return
        stats = ingest_mod.ingest_scraped_games(
            conn, batch, games_df=games_df, schedule=schedule
        )
        result.inserted_ticks += stats.inserted_ticks
        result.inserted_games += stats.inserted_games
        result.retimed_games.extend(stats.retimed_games)
        batch.clear()

    session = new_session()
    batch: list[ScrapedGame] = []
    try:
        for index, day in enumerate(dates):
            # Checked before any `continue`, so a failed date never delays a write.
            if index and flush_every_dates > 0 and index % flush_every_dates == 0:
                flush(batch)
            try:
                summaries = discover_games_for_date(session, day)
                games = list(
                    scrape_events(
                        [s.event_id for s in summaries],
                        session=session,
                        markets=markets,
                    )
                )
            except Exception as exc:  # one bad date must not end the run
                result.failed_dates.append(f"{day}: {exc}")
                if progress:
                    print(f"  ! {day}: {exc}")
                continue

            if not games:
                continue
            batch.extend(games)
            result.scraped_games += len(games)
            if progress:
                ticks = sum(len(g.ticks) for g in games)
                print(f"  {day}: {len(games)} game(s), {ticks} ticks")
    finally:
        session.close()

    flush(batch)
    return result
