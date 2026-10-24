"""Daily update of the Covers line-history store.

The plan is per game: one request returns every book and market of a game
(``fetch_data.odds_covers.client``).

* a game ``nba_games`` has and the store does not -> fetched;
* a stored game of the last ``retry_days`` days missing one of
  ``books.ACTIVE_BOOKS`` -> fetched again (the insert keeps what is stored), so
  a book that came late is picked up for a few days and then left alone (some
  books never price some games);
* anything older -> nothing. Covers' history is final once a game is over.

A game enters ``lh_game`` only with its ticks, in one transaction, so a game
whose answer had no line history (or whose write failed) is never mistaken for
a stored one: it stays missing and is fetched again on every run. A truncated
history -- the freshest pre-tip tick more than ``TRUNCATED_HISTORY_MINUTES``
before tip, the signature of a feed freeze -- is held back as the SBR ingest
does (``ingest.is_held_back``): missing, and so fetched again, for
``TRUNCATED_HOLD_DAYS`` after tip-off, then stored as it is.

Only games ``SETTLED_AFTER`` past tip-off are fetched, and the season starts at
``FIRST_DATE``: the store holds the 2026-27 season onward. The one exception is
the backfill (``plan_backfill``): past games whose SBR history is missing or
incomplete, listed by ``scripts/line_history/covers/audit_sbr_gaps.py``. Those
seasons hold only the backfilled games.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import psycopg

from nba_ou.fetch_data.odds_covers import archive as archive_mod
from nba_ou.fetch_data.odds_covers.books import ACTIVE_BOOKS
from nba_ou.fetch_data.odds_covers.client import CoversClient, CoversError
from nba_ou.fetch_data.odds_covers.parse import (
    CoversGame,
    is_settled,
    parse_matchups,
    scraped_game,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    EASTERN_TZ,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven import ingest as sbr_ingest
from nba_ou.postgre_db.line_history_aiven.schema import market_ids
from nba_ou.utils.ci import log_group

from . import schema as schema_mod
from . import store

#: First date of the store: opening night of 2026-27.
FIRST_DATE = date(2026, 10, 20)
FIRST_SEASON = 2026

DEFAULT_RETRY_DAYS = 3


@dataclass
class UpdateResult:
    planned_games: int = 0
    fetched_games: int = 0
    unsettled_games: list[str] = field(default_factory=list)
    unmatched_games: list[str] = field(default_factory=list)
    unlisted_games: list[str] = field(default_factory=list)
    feed_tipoff_missing: list[str] = field(default_factory=list)
    empty_games: list[str] = field(default_factory=list)
    held_back: list[str] = field(default_factory=list)
    unknown_books: set[int] = field(default_factory=set)
    failed: list[str] = field(default_factory=list)
    archive_failures: int = 0
    inserted_games: int = 0
    inserted_ticks: int = 0
    prepared_ticks: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    repaired: dict[str, int] = field(default_factory=dict)
    frames: list[tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=list)

    def absorb(self, stats: sbr_ingest.IngestStats) -> None:
        self.prepared_ticks += stats.prepared_ticks
        self.unmatched_games.extend(stats.unmatched_games)
        for reason, count in stats.dropped.items():
            self.dropped[reason] = self.dropped.get(reason, 0) + count
        for reason, count in stats.repaired.items():
            self.repaired[reason] = self.repaired.get(reason, 0) + count


def season_games(games_df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """One row per game of ``season_year`` from the team-game ``nba_games`` frame."""
    games = games_df[
        pd.to_numeric(games_df["season_year"], errors="coerce") == int(season_year)
    ][["game_id", "game_date"]].copy()
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    return games.dropna().drop_duplicates("game_id").reset_index(drop=True)


def plan_requests(
    games: pd.DataFrame,
    stored: dict[str, set[int]],
    *,
    today: date,
    retry_days: int = DEFAULT_RETRY_DAYS,
    first_date: date = FIRST_DATE,
    books: Iterable[int] = ACTIVE_BOOKS,
) -> dict[date, dict[str, set[int]]]:
    """date -> game_id -> the books it lacks. Each planned game is fetched whole;
    the books only say why. Games of ``today`` are not ready."""
    books = set(int(b) for b in books)
    retry_from = today - timedelta(days=retry_days)
    plan: dict[date, dict[str, set[int]]] = {}
    for row in games.itertuples(index=False):
        if not first_date <= row.game_date < today:
            continue
        if row.game_id not in stored:
            wanted = set(books)
        elif row.game_date >= retry_from:
            wanted = books - stored[row.game_id]
        else:
            continue
        if wanted:
            plan.setdefault(row.game_date, {})[row.game_id] = wanted
    return dict(sorted(plan.items()))


def backfill_game_ids(
    flagged: pd.DataFrame, reasons: Iterable[str] | None = None
) -> set[str]:
    """The games of an ``audit_sbr_gaps.py --out`` file flagged with one of
    ``reasons`` (every flagged game when ``reasons`` is None)."""
    wanted = None if reasons is None else set(reasons)
    flags = flagged["reasons"].fillna("").astype(str).str.split(";")
    keep = flags.apply(
        lambda names: any(n and (wanted is None or n in wanted) for n in names)
    )
    return set(flagged.loc[keep, "game_id"].astype(str).str.zfill(10))


def plan_backfill(
    games: pd.DataFrame,
    stored: dict[str, set[int]],
    *,
    today: date,
    books: Iterable[int] = ACTIVE_BOOKS,
) -> dict[date, dict[str, set[int]]]:
    """date -> game_id -> books for the past ``games`` the store does not hold.

    ``FIRST_DATE`` does not apply. A stored game is never fetched again, so a
    backfill run again picks up only the games still missing (an empty answer,
    a failed request).
    """
    books = set(int(b) for b in books)
    plan: dict[date, dict[str, set[int]]] = {}
    for row in games.itertuples(index=False):
        if row.game_date < today and row.game_id not in stored:
            plan.setdefault(row.game_date, {})[row.game_id] = set(books)
    return dict(sorted(plan.items()))


def feed_tipoffs(schedule: pd.DataFrame | None) -> dict[str, datetime]:
    if schedule is None or schedule.empty:
        return {}
    feed = schedule.dropna(subset=["tipoff_utc"]).drop_duplicates("game_id")
    return {
        str(row.game_id): pd.Timestamp(row.tipoff_utc).tz_convert("UTC").to_pydatetime()
        for row in feed.itertuples(index=False)
    }


def _resolve(listed: list[CoversGame], lookup: dict) -> dict[str, CoversGame]:
    """game_id -> Covers game, for the listed games ``nba_games`` knows."""
    resolved: dict[str, CoversGame] = {}
    for game in listed:
        hit = sbr_ingest.resolve_game_id(
            lookup,
            game_date=game.game_date,
            team_away=game.team_away,
            team_home=game.team_home,
        )
        if hit is not None:
            resolved[hit[0]] = game
    return resolved


def ingest_games(
    conn: psycopg.Connection | None,
    games: list[ScrapedGame],
    *,
    games_df: pd.DataFrame,
    schedule: pd.DataFrame | None,
    result: UpdateResult,
    dry_run: bool,
) -> None:
    """Encode ``games`` with the SBR ingest's ``build_frames`` and insert them:
    the games with ticks and their ticks, in one transaction."""
    rows, game_dim, stats = sbr_ingest.build_frames(
        games,
        game_index=sbr_ingest.build_game_index(games_df),
        book_ids=schema_mod.book_ids(),
        market_ids=market_ids(),
        schedule=schedule,
    )
    result.absorb(stats)
    # A game without ticks stays out of lh_game, so the plan keeps asking for it.
    game_dim = game_dim[game_dim["game_id"].isin(rows["game_id"])]
    result.frames.append((rows, game_dim))
    if dry_run or conn is None or game_dim.empty:
        return
    seasons = sorted(int(s) for s in rows["season_year"].dropna().unique())
    for season_year in seasons:  # DDL, committed on its own
        schema_mod.create_season_partition(conn, season_year)
    try:
        games = store.insert_games(conn, game_dim, commit=False)
        ticks = 0
        for season_year in seasons:
            season_rows = rows[rows["season_year"] == season_year]
            ticks += store.insert_lines(conn, season_rows, season_year, commit=False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    result.inserted_games += games
    result.inserted_ticks += ticks


def _archive_put(archive, key: str, text: str, result: UpdateResult, progress) -> None:
    try:
        archive.put(key, text)
    except Exception as exc:  # the archive must not stop the store
        result.archive_failures += 1
        progress(f"  ! archive write failed ({type(exc).__name__}: {exc})")


def update_covers_line_history(
    plan: dict[date, dict[str, set[int]]],
    *,
    games_df: pd.DataFrame,
    schedule: pd.DataFrame | None,
    conn: psycopg.Connection | None,
    client: CoversClient,
    archive: archive_mod.CoversArchive | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    progress: Callable[[str], None] = print,
) -> UpdateResult:
    """Fetch, archive and store every planned game. One date at a time, each
    written before the next is fetched, so a failure keeps what is done."""
    now = now or datetime.now(UTC)
    result = UpdateResult(planned_games=sum(len(day) for day in plan.values()))
    lookup = sbr_ingest.build_game_lookup(sbr_ingest.build_game_index(games_df))
    tipoffs = feed_tipoffs(schedule)
    archiving = archive is not None and not dry_run

    for day, wanted in plan.items():
        with log_group(f"{day}: {len(wanted)} game(s) planned", progress):
            _update_day(
                day,
                wanted,
                lookup=lookup,
                tipoffs=tipoffs,
                games_df=games_df,
                schedule=schedule,
                conn=conn,
                client=client,
                archive=archive if archiving else None,
                dry_run=dry_run,
                now=now,
                result=result,
                progress=progress,
            )
    return result


def _update_day(
    day: date,
    wanted: dict[str, set[int]],
    *,
    lookup: dict,
    tipoffs: dict[str, datetime],
    games_df: pd.DataFrame,
    schedule: pd.DataFrame | None,
    conn: psycopg.Connection | None,
    client: CoversClient,
    archive: archive_mod.CoversArchive | None,
    dry_run: bool,
    now: datetime,
    result: UpdateResult,
    progress: Callable[[str], None],
) -> None:
    """One date of the plan: its matchups, then each wanted game, then one write.

    Every game gets one log line saying what happened to it, so a run can be
    followed (and a skipped game explained) from the workflow log alone.
    """
    try:
        matchups_text = client.matchups(day)
    except CoversError as exc:
        result.failed.append(f"{day} matchups: {exc}")
        progress(f"  ! matchups page failed: {exc}")
        return
    if archive is not None:
        key = archive_mod.matchups_key(day, datetime.now(UTC))
        _archive_put(archive, key, matchups_text, result, progress)

    games = parse_matchups(matchups_text, day)
    listed = _resolve(games, lookup)
    progress(f"  matchups: {len(games)} listed, {len(listed)} matched to nba_games")
    scraped: list[ScrapedGame] = []
    for game_id in sorted(wanted):
        game = listed.get(game_id)
        if game is None:
            result.unlisted_games.append(f"{day} {game_id}")
            progress(f"  ? {game_id}: not on the matchups page")
            continue
        label = f"{day} {game.team_away} @ {game.team_home} ({game_id})"
        short = f"{game.team_away} @ {game.team_home} ({game_id})"
        tipoff = tipoffs.get(game_id)
        if tipoff is None:
            # Covers has no tip time for finished games, and without one the
            # in-play ticks cannot be told apart. Left for the next run.
            result.feed_tipoff_missing.append(label)
            progress(f"  - {short}: no schedule-feed tipoff, next run")
            continue
        if not is_settled(tipoff, now):
            result.unsettled_games.append(label)
            hours = (now - tipoff).total_seconds() / 3600
            progress(f"  - {short}: not finished (tip {hours:+.1f} h ago), next run")
            continue
        started = time.monotonic()
        try:
            text = client.line_history(game.covers_id)
        except CoversError as exc:
            result.failed.append(f"{label}: {exc}")
            progress(f"  ! {short}: request failed: {exc}")
            continue
        elapsed = time.monotonic() - started
        fetched_at = datetime.now(UTC)
        result.fetched_games += 1
        if archive is not None:
            key = archive_mod.game_key(day, game.covers_id, fetched_at)
            _archive_put(archive, key, text, result, progress)
        try:
            parsed = scraped_game(
                game, text=text, tipoff_utc=tipoff, fetched_at=fetched_at
            )
        except Exception as exc:  # one odd page must not stop the run
            result.failed.append(f"{label}: parse error ({type(exc).__name__}: {exc})")
            progress(f"  ! {short}: parse error: {type(exc).__name__}: {exc}")
            continue
        result.unknown_books |= parsed.unknown_books
        books = parsed.game.books
        gap = sbr_ingest.pregame_gap_minutes(parsed.game)
        detail = (
            f"{len(books)} book(s), {len(parsed.game.ticks)} tick(s), last pre-tip "
            f"tick {gap if gap is not None else '-'} min before tip "
            f"[{len(text) / 1e6:.1f} MB, {elapsed:.1f} s]"
        )
        if not books:
            result.empty_games.append(label)
            progress(f"  - {short}: no line history, next run")
            continue
        if sbr_ingest.is_held_back(parsed.game, now, sbr_ingest.TRUNCATED_HOLD_DAYS):
            result.held_back.append(f"{label}: last pre-tip tick {gap} min before tip")
            progress(f"  - {short}: truncated, held back: {detail}")
            continue
        progress(f"  + {short}: {detail}")
        scraped.append(parsed.game)

    if not scraped:
        progress("  nothing to store for this date")
        return
    games_before, ticks_before = result.inserted_games, result.inserted_ticks
    prepared_before = result.prepared_ticks
    ingest_games(
        conn,
        scraped,
        games_df=games_df,
        schedule=schedule,
        result=result,
        dry_run=dry_run,
    )
    prepared = result.prepared_ticks - prepared_before
    if dry_run or conn is None:
        progress(
            f"  prepared {prepared} tick(s) of {len(scraped)} game(s), not written"
        )
    else:
        progress(
            f"  stored {result.inserted_games - games_before} new game(s) and "
            f"{result.inserted_ticks - ticks_before} new tick(s) "
            f"(of {prepared} prepared)"
        )


def today_eastern(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(EASTERN_TZ).date()
