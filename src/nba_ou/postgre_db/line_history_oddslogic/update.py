"""Daily update of the OddsLogic line-history store.

The plan is per game and book rather than per date, because a request is one
book of one game (``fetch_data.odds_oddslogic.client``):

* a game ``nba_games`` has and the store does not -> every book;
* a stored game of the last ``retry_days`` days -> only its missing books, so
  an answer that came back empty is asked again for a few days and then left
  alone (some books never price some games);
* anything older -> nothing. The widget's archive is final once a game is
  over, so there is no movement to refresh.

Only games ``SETTLED_AFTER`` past tip-off are fetched, and the season starts at
``FIRST_DATE``: this store holds the 2026-27 season onward.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import psycopg

from nba_ou.fetch_data.odds_oddslogic import archive as archive_mod
from nba_ou.fetch_data.odds_oddslogic.books import ACTIVE_BOOKS, OL_BOOKS
from nba_ou.fetch_data.odds_oddslogic.client import (
    REQUEST_TYPES,
    OddsLogicClient,
    OddsLogicError,
    history_url,
)
from nba_ou.fetch_data.odds_oddslogic.parse import (
    OddsLogicGame,
    is_settled,
    parse_schedule,
    scraped_game,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    EASTERN_TZ,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven import ingest as sbr_ingest
from nba_ou.postgre_db.line_history_aiven.schema import market_ids

from . import schema as schema_mod
from . import store

#: First date of the store: opening night of 2026-27.
FIRST_DATE = date(2026, 10, 20)
FIRST_SEASON = 2026

DEFAULT_RETRY_DAYS = 3


@dataclass
class UpdateResult:
    planned_games: int = 0
    planned_requests: int = 0
    fetched_games: int = 0
    unsettled_games: list[str] = field(default_factory=list)
    unmatched_games: list[str] = field(default_factory=list)
    unlisted_games: list[str] = field(default_factory=list)
    feed_tipoff_missing: list[str] = field(default_factory=list)
    empty_answers: list[str] = field(default_factory=list)
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
    """date -> game_id -> book ids to fetch. Games of ``today`` are not ready."""
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


def feed_tipoffs(schedule: pd.DataFrame | None) -> dict[str, datetime]:
    if schedule is None or schedule.empty:
        return {}
    feed = schedule.dropna(subset=["tipoff_utc"]).drop_duplicates("game_id")
    return {
        str(row.game_id): pd.Timestamp(row.tipoff_utc).tz_convert("UTC").to_pydatetime()
        for row in feed.itertuples(index=False)
    }


def _resolve(listed: list[OddsLogicGame], lookup: dict) -> dict[str, OddsLogicGame]:
    """game_id -> widget game, for the listed games ``nba_games`` knows."""
    resolved: dict[str, OddsLogicGame] = {}
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


def _url(game: OddsLogicGame, book_id: int, request_type: str) -> str:
    return history_url(
        game.game_date,
        rotation=game.rotation,
        book_id=book_id,
        request_type=request_type,
        team_away=game.team_away,
        team_home=game.team_home,
        away_abbr=game.away_abbr,
        home_abbr=game.home_abbr,
    )


def _request(
    client: OddsLogicClient,
    game: OddsLogicGame,
    book_id: int,
    request_type: str,
    failures: list[str],
) -> str | None:
    """One history answer; a request that fails after its retries is recorded
    in ``failures`` and given as ``None``, like an empty answer, so one book
    never costs the game's other books."""
    try:
        return client.history(_url(game, book_id, request_type))
    except OddsLogicError as exc:
        failures.append(
            f"{game.game_date} {game.away_abbr}@{game.home_abbr} "
            f"{OL_BOOKS[book_id][0]} t={request_type}: {exc}"
        )
        return None


def fetch_game(
    client: OddsLogicClient,
    game: OddsLogicGame,
    book_ids: Iterable[int],
    failures: list[str],
) -> dict[int, dict[str, str | None]]:
    """Every wanted book's answers for one game: book id -> request type -> text.

    An answer that stayed empty or failed is ``None``; ``refetch_empty`` asks
    again. The public-betting series (``books.PUBLIC_FEEDS``) are not fetched.
    """
    return {
        book_id: {
            request_type: _request(client, game, book_id, request_type, failures)
            for request_type in REQUEST_TYPES
        }
        for book_id in sorted(book_ids)
    }


def refetch_empty(
    client: OddsLogicClient,
    game: OddsLogicGame,
    history: dict[int, dict[str, str | None]],
    failures: list[str],
) -> None:
    """Ask once more for the answers of ``history`` that stayed empty.

    Run after the rest of the date: the empty answers come in bursts, and the
    same request usually succeeds a few minutes later.
    """
    for book_id, answers in history.items():
        for request_type, text in answers.items():
            if text is None:
                answers[request_type] = _request(
                    client, game, book_id, request_type, failures
                )


def ingest_games(
    conn: psycopg.Connection | None,
    games: list[ScrapedGame],
    *,
    games_df: pd.DataFrame,
    schedule: pd.DataFrame | None,
    result: UpdateResult,
    dry_run: bool,
) -> None:
    """Encode ``games`` with the SBR ingest's ``build_frames`` and insert them."""
    rows, game_dim, stats = sbr_ingest.build_frames(
        games,
        game_index=sbr_ingest.build_game_index(games_df),
        book_ids=schema_mod.book_ids(),
        market_ids=market_ids(),
        schedule=schedule,
    )
    result.absorb(stats)
    result.frames.append((rows, game_dim))
    if dry_run or conn is None or game_dim.empty:
        return
    for season_year in sorted(rows["season_year"].dropna().unique()):
        schema_mod.create_season_partition(conn, int(season_year))
    result.inserted_games += store.insert_games(conn, game_dim)
    for season_year in sorted(rows["season_year"].dropna().unique()):
        season_rows = rows[rows["season_year"] == season_year]
        result.inserted_ticks += store.insert_lines(conn, season_rows, int(season_year))


def update_oddslogic_line_history(
    plan: dict[date, dict[str, set[int]]],
    *,
    games_df: pd.DataFrame,
    schedule: pd.DataFrame | None,
    conn: psycopg.Connection | None,
    client: OddsLogicClient,
    archive: archive_mod.OddsLogicArchive | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    progress: Callable[[str], None] = print,
) -> UpdateResult:
    """Fetch, archive and store every planned (game, book). One date at a time,
    each written before the next is fetched, so a failure keeps what is done."""
    now = now or datetime.now(UTC)
    result = UpdateResult(
        planned_games=sum(len(day) for day in plan.values()),
        planned_requests=sum(
            len(books) * len(REQUEST_TYPES)
            for day in plan.values()
            for books in day.values()
        ),
    )
    lookup = sbr_ingest.build_game_lookup(sbr_ingest.build_game_index(games_df))
    tipoffs = feed_tipoffs(schedule)

    for day, wanted in plan.items():
        try:
            schedule_text = client.schedule(day)
        except OddsLogicError as exc:
            result.failed.append(f"{day} schedule: {exc}")
            continue
        fetched_at = datetime.now(UTC)
        if archive is not None and not dry_run:
            try:
                archive.put(
                    archive_mod.schedule_key(day, fetched_at), {"text": schedule_text}
                )
            except Exception as exc:  # the archive must not stop the store
                result.archive_failures += 1
                progress(f"  ! archive write failed ({type(exc).__name__}: {exc})")

        listed = _resolve(parse_schedule(schedule_text, day), lookup)
        fetched: list[tuple] = []
        for game_id, books in sorted(wanted.items()):
            game = listed.get(game_id)
            if game is None:
                result.unlisted_games.append(f"{day} {game_id}")
                continue
            tipoff = tipoffs.get(game_id)
            if tipoff is None:
                result.feed_tipoff_missing.append(game_id)
                tipoff = game.tipoff_utc
            label = f"{day} {game.away_abbr}@{game.home_abbr} ({game_id})"
            if not is_settled(tipoff, now):
                result.unsettled_games.append(label)
                continue
            history = fetch_game(client, game, books, result.failed)
            result.fetched_games += 1
            progress(f"  {label}: {len(books)} book(s)")
            fetched.append((game, tipoff, history))

        scraped: list[ScrapedGame] = []
        for game, tipoff, history in fetched:
            refetch_empty(client, game, history, result.failed)
            for book_id, by_type in history.items():
                for request_type, text in by_type.items():
                    if text is None:
                        result.empty_answers.append(
                            f"{day} {game.away_abbr}@{game.home_abbr} "
                            f"{OL_BOOKS[book_id][0]} t={request_type}"
                        )
            if archive is not None and not dry_run:
                stamp = datetime.now(UTC)
                try:
                    archive.put(
                        archive_mod.game_key(game, stamp),
                        archive_mod.game_bundle(
                            game, fetched_at=stamp, history=history
                        ),
                    )
                except Exception as exc:  # the archive must not stop the store
                    result.archive_failures += 1
                    progress(f"  ! archive write failed ({type(exc).__name__}: {exc})")
            answers = {
                book_id: [text for text in by_type.values() if text]
                for book_id, by_type in history.items()
            }
            scraped.append(scraped_game(game, answers=answers, tipoff_utc=tipoff))

        if scraped:
            ingest_games(
                conn,
                scraped,
                games_df=games_df,
                schedule=schedule,
                result=result,
                dry_run=dry_run,
            )
    return result


def today_eastern(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(EASTERN_TZ).date()
