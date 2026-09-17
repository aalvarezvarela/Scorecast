"""Load freshly scraped SBR line history into the Aiven store.

The companion to ``transform.py``, which handles the CSV-era scrape. Those rows
carried naive local timestamps that had to be localized per season; the rows
produced by ``fetch_data.odds_sportsbook.scrape_sportsbook_line_history`` are
already tz-aware UTC and carry the game's own tipoff, so the whole
localize/DST-ambiguity stage disappears. Everything downstream of the timestamp
is unchanged -- game_id resolution, the storage encodings, and the two
data-quality repairs -- so those are imported from ``transform`` rather than
restated here.

Writes are insert-only. Both the fact table and the game dimension go in with
``ON CONFLICT DO NOTHING``, so re-running a date range adds whatever is new and
never rewrites what is already stored. That matters concretely: SBR has since
dropped Caesars, and the 270k historical Caesars rows cannot be refetched, so a
refresh must never be allowed to replace a game's rows wholesale.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import ScrapedGame
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    _normalize_sbr_team_name,
    build_games_home_away_for_line_history,
)

from . import load as loader
from . import schema as schema_mod
from .schema import SCHEMA
from .tipoff_corrections import retime_games
from .transform import (
    GAME_DIM_COLUMNS,
    OUTPUT_COLUMNS,
    _encode_line,
    _encode_price,
    null_implausible_pregame_lines,
    repair_spread_price_bleed,
)

#: A page tipoff further than this from the schedule feed's is reported rather
#: than trusted silently. Real disagreements are schedule changes; small ones
#: are rounding between the two sources.
TIPOFF_TOLERANCE_MINUTES = 15


@dataclass
class IngestStats:
    scraped_games: int = 0
    matched_games: int = 0
    source_ticks: int = 0
    prepared_ticks: int = 0
    inserted_ticks: int = 0
    inserted_games: int = 0
    unmatched_games: list[str] = field(default_factory=list)
    tipoff_disagreements: list[str] = field(default_factory=list)
    retimed_games: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    repaired: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    def repair(self, reason: str, count: int) -> None:
        if count:
            self.repaired[reason] = self.repaired.get(reason, 0) + int(count)


def load_stored_game_ids(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT game_id FROM {}.lh_game").format(sql.Identifier(SCHEMA))
        )
        return {str(row[0]) for row in cur.fetchall()}


def stored_date_range(conn: psycopg.Connection) -> tuple[date | None, date | None]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT MIN(game_date), MAX(game_date) FROM {}.lh_game").format(
                sql.Identifier(SCHEMA)
            )
        )
        row = cur.fetchone()
    # An empty table still returns a row of NULLs; None only happens if the
    # query returned nothing at all.
    return (row[0], row[1]) if row else (None, None)


def find_missing_games(
    conn: psycopg.Connection,
    games_df: pd.DataFrame,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Games that ``nba_games`` knows about but the line-history store does not.

    This is the staleness signal: ``nba_games`` is updated daily, so any game
    present there and absent here is either a gap in an old scrape or a date
    that has not been fetched yet.

    ``start`` defaults to the store's own earliest game -- before that there is
    no line history to be missing, only history that was never collected.
    """
    if games_df.empty:
        return pd.DataFrame(columns=["game_id", "game_date", "season_year"])

    stored = load_stored_game_ids(conn)
    first_stored, _ = stored_date_range(conn)
    if start is None:
        start = first_stored

    games = games_df.copy()
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    games = games.dropna(subset=["game_date"]).drop_duplicates("game_id")

    missing = games[~games["game_id"].isin(stored)]
    if start is not None:
        missing = missing[missing["game_date"] >= start]
    if end is not None:
        missing = missing[missing["game_date"] <= end]

    columns = [
        c for c in ["game_id", "game_date", "season_year"] if c in missing.columns
    ]
    return missing[columns].sort_values("game_date").reset_index(drop=True)


def missing_dates(missing: pd.DataFrame) -> list[date]:
    """The dates to hand the scraper -- one page fetch covers a whole slate."""
    if missing.empty:
        return []
    return sorted({d for d in missing["game_date"] if d is not None})


def build_game_index(games_df: pd.DataFrame) -> pd.DataFrame:
    """(game_date, team_home, team_away) -> game_id, in standardized team names.

    Both sides of the join are pushed through the same standardization the
    CSV-era loader used, so SBR's "Charlotte Hornets" and ``nba_games``'
    "Charlotte" meet in the middle.
    """
    return build_games_home_away_for_line_history(games_df)


def build_game_lookup(
    game_index: pd.DataFrame,
) -> dict[tuple, tuple[str, int | None]]:
    """(game_date, team_home, team_away) -> (game_id, season_year)."""
    if game_index.empty:
        return {}
    return {
        (row.game_date, row.team_home, row.team_away): (
            str(row.game_id),
            (
                int(row.game_season_year)
                if pd.notna(getattr(row, "game_season_year", None))
                else None
            ),
        )
        for row in game_index.itertuples(index=False)
    }


def resolve_game_id(
    lookup: dict[tuple, tuple[str, int | None]],
    *,
    game_date: date,
    team_away: str,
    team_home: str,
) -> tuple[str, int | None] | None:
    """Look one SBR game up in ``nba_games``, or ``None`` if it is not there.

    A miss is almost always preseason, which ``nba_games`` does not carry.
    """
    try:
        home = _normalize_sbr_team_name(team_home)
        away = _normalize_sbr_team_name(team_away)
    except RuntimeError:
        return None
    return lookup.get((game_date, home, away))


def _resolve_game_ids(
    games: Sequence[ScrapedGame],
    game_index: pd.DataFrame,
    stats: IngestStats,
) -> dict[int, tuple[str, int]]:
    """event_id -> (nba game_id, season_year) for the games that match.

    Unmatched games are reported and skipped rather than loaded without a key.
    """
    lookup = build_game_lookup(game_index)
    if not lookup:
        return {}

    resolved: dict[int, tuple[str, int]] = {}
    for game in games:
        hit = resolve_game_id(
            lookup,
            game_date=game.game_date,
            team_away=game.team_away,
            team_home=game.team_home,
        )
        if hit is None:
            stats.unmatched_games.append(
                f"{game.event_id} {game.game_date} "
                f"{game.team_away} @ {game.team_home}"
            )
            continue

        game_id, season_year = hit
        resolved[game.event_id] = (game_id, season_year or game.season_year)

    return resolved


def _check_tipoffs(
    games: Sequence[ScrapedGame],
    resolved: dict[int, tuple[str, int]],
    schedule: pd.DataFrame | None,
    stats: IngestStats,
) -> None:
    """Compare each page tipoff with the schedule feed's, and report drift.

    The page value is the one used -- it is the same payload the ticks came
    from, so ``mins_to_tip`` stays internally consistent even if the feed
    disagrees. The feed is only here to make a disagreement visible.
    """
    if schedule is None or schedule.empty:
        return

    feed = (
        schedule.dropna(subset=["tipoff_utc"])
        .drop_duplicates("game_id")
        .set_index(
            schedule.dropna(subset=["tipoff_utc"])
            .drop_duplicates("game_id")["game_id"]
            .astype(str)
        )
    )
    for game in games:
        hit = resolved.get(game.event_id)
        if hit is None or hit[0] not in feed.index:
            continue
        feed_tipoff = pd.to_datetime(feed.loc[hit[0], "tipoff_utc"], utc=True)
        delta = (
            abs((pd.Timestamp(game.tipoff_utc) - feed_tipoff).total_seconds()) / 60.0
        )
        if delta > TIPOFF_TOLERANCE_MINUTES:
            stats.tipoff_disagreements.append(
                f"{hit[0]} (event {game.event_id}): page {game.tipoff_utc} "
                f"vs feed {feed_tipoff} -- {delta:.0f} min apart"
            )


def _team_codes(schedule: pd.DataFrame | None) -> dict[str, tuple[str, str]]:
    if schedule is None or schedule.empty:
        return {}
    return {
        str(row.game_id): (row.team_home, row.team_away)
        for row in schedule.dropna(subset=["team_home", "team_away"]).itertuples(
            index=False
        )
    }


def build_frames(
    games: Sequence[ScrapedGame],
    *,
    game_index: pd.DataFrame,
    book_ids: dict[str, int],
    market_ids: dict[str, int],
    schedule: pd.DataFrame | None = None,
    stats: IngestStats | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, IngestStats]:
    """Scraped games -> (fact rows, game dimension) ready for ``lh_line``/``lh_game``."""
    stats = stats or IngestStats()
    stats.scraped_games += len(games)
    empty = (
        pd.DataFrame(columns=OUTPUT_COLUMNS),
        pd.DataFrame(columns=GAME_DIM_COLUMNS),
        stats,
    )
    if not games:
        return empty

    resolved = _resolve_game_ids(games, game_index, stats)
    stats.drop("preseason_or_unmatched_game", len(games) - len(resolved))
    if not resolved:
        return empty

    _check_tipoffs(games, resolved, schedule, stats)
    codes = _team_codes(schedule)

    records: list[dict] = []
    dim_records: list[dict] = []
    unknown_ticks = 0
    for game in games:
        hit = resolved.get(game.event_id)
        if hit is None:
            continue
        game_id, season_year = hit
        stats.matched_games += 1

        team_home, team_away = codes.get(game_id, (game.team_home, game.team_away))
        dim_records.append(
            {
                "game_id": game_id,
                "game_date": game.game_date,
                "season_year": season_year,
                "tipoff_utc": game.tipoff_utc,
                "event_id": game.event_id,
                "team_home": team_home,
                "team_away": team_away,
            }
        )

        for tick in game.ticks:
            stats.source_ticks += 1
            market_id = market_ids.get(tick.market)
            book_id = book_ids.get(tick.book_slug)
            if market_id is None or book_id is None:
                unknown_ticks += 1
                continue
            records.append(
                {
                    "game_id": game_id,
                    "season_year": season_year,
                    "market_id": market_id,
                    "book_id": book_id,
                    "line_ts": tick.line_ts,
                    "mins_to_tip": tick.minutes_to_tip,
                    "is_pregame": tick.is_pregame,
                    "is_opener": tick.is_opener,
                    "left_line": tick.left_line,
                    "left_price": tick.left_price,
                    "right_line": tick.right_line,
                    "right_price": tick.right_price,
                }
            )

    stats.drop("unknown_market_or_book", unknown_ticks)

    if not records:
        return (
            pd.DataFrame(columns=OUTPUT_COLUMNS),
            pd.DataFrame(dim_records, columns=GAME_DIM_COLUMNS),
            stats,
        )

    rows = pd.DataFrame(records)

    rows, bled = repair_spread_price_bleed(rows, market_ids.get("point_spread"))
    stats.repair("spread_price_bleed", bled)
    rows, implausible = null_implausible_pregame_lines(rows, market_ids)
    stats.repair("implausible_pregame_line", implausible)

    for column in ["left_line", "right_line"]:
        rows[column] = _encode_line(rows[column])
    for column in ["left_price", "right_price"]:
        rows[column] = _encode_price(rows[column])

    rows["mins_to_tip"] = rows["mins_to_tip"].astype("Int64")
    rows["market_id"] = rows["market_id"].astype("Int64")
    rows["book_id"] = rows["book_id"].astype("Int64")
    rows["game_id"] = rows["game_id"].astype(str)

    before = len(rows)
    rows = rows.drop_duplicates(
        subset=["game_id", "market_id", "book_id", "line_ts"], keep="last"
    )
    stats.drop("duplicate_timepoint", before - len(rows))
    stats.prepared_ticks += len(rows)

    game_dim = pd.DataFrame(dim_records, columns=GAME_DIM_COLUMNS).drop_duplicates(
        "game_id"
    )
    return rows[OUTPUT_COLUMNS].reset_index(drop=True), game_dim, stats


def insert_games(conn: psycopg.Connection, game_dim: pd.DataFrame) -> int:
    """Add unseen games. Existing rows are left exactly as they are.

    Deliberately not an upsert: ``mins_to_tip`` on already-stored ticks was
    computed against the stored ``tipoff_utc``, so silently moving it would
    desynchronise rows this run is not touching.
    """
    if game_dim.empty:
        return 0

    columns = loader.GAME_DIM_INSERT_COLUMNS
    rows = [
        tuple(loader._to_native(value) for value in row)
        for row in game_dim[columns].itertuples(index=False, name=None)
    ]
    query = sql.SQL(
        "INSERT INTO {}.lh_game ({cols}) VALUES ({vals}) "
        "ON CONFLICT (game_id) DO NOTHING"
    ).format(
        sql.Identifier(SCHEMA),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
        vals=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
    )
    with conn.cursor() as cur:
        inserted = 0
        for row in rows:
            cur.execute(query, row)
            inserted += cur.rowcount
    conn.commit()
    return inserted


def retime_to_page_tipoffs(conn: psycopg.Connection, game_dim: pd.DataFrame) -> list[str]:
    """Move stored games onto the page tipoff where the two disagree.

    New ticks are timed against the page's tipoff. A game stored earlier may
    carry another one (the bulk load used the schedule feed, which keeps the
    original slot of a moved game), and inserting the new ticks as they are
    would leave two books of one game on two clocks. The page tipoff matched
    the NBA scoreboard on all 66 disagreements checked on 2026-09-17, so the
    stored game is retimed -- tipoff and every existing tick -- before the new
    ticks go in.
    """
    if game_dim.empty:
        return []
    page = game_dim.drop_duplicates("game_id").set_index(
        game_dim.drop_duplicates("game_id")["game_id"].astype(str)
    )["tipoff_utc"]
    page = pd.to_datetime(page, utc=True)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, tipoff_utc FROM {}.lh_game WHERE game_id = ANY(%s)"
            ).format(sql.Identifier(SCHEMA)),
            (page.index.tolist(),),
        )
        stored = {
            str(game_id): pd.Timestamp(tipoff).tz_convert("UTC")
            for game_id, tipoff in cur.fetchall()
        }
    moved = {
        game_id: tipoff
        for game_id, tipoff in page.items()
        if game_id in stored and stored[game_id] != tipoff
    }
    if moved:
        retime_games(conn, moved)
    return sorted(moved)


def ingest_scraped_games(
    conn: psycopg.Connection,
    games: Sequence[ScrapedGame],
    *,
    games_df: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    dry_run: bool = False,
) -> IngestStats:
    """Resolve, encode and insert a batch of scraped games. Safe to re-run."""
    stats = IngestStats()
    if not games:
        return stats

    book_slugs = sorted({tick.book_slug for game in games for tick in game.ticks})
    book_ids = (
        schema_mod.ensure_books(conn, book_slugs)
        if not dry_run
        else _existing_books(conn, book_slugs)
    )

    rows, game_dim, stats = build_frames(
        games,
        game_index=build_game_index(games_df),
        book_ids=book_ids,
        market_ids=schema_mod.market_ids(),
        schedule=schedule,
        stats=stats,
    )
    if dry_run or rows.empty:
        return stats

    for season_year in sorted(rows["season_year"].dropna().unique()):
        schema_mod.create_season_partition(conn, int(season_year))

    stats.inserted_games = insert_games(conn, game_dim)
    stats.retimed_games = retime_to_page_tipoffs(conn, game_dim)

    # Staged per season: the merge targets one partition at a time, which keeps
    # the write off the 1 GB instance's WAL in a single spike.
    for season_year in sorted(rows["season_year"].dropna().unique()):
        season_rows = rows[rows["season_year"] == season_year]
        loader.copy_rows(conn, season_rows)
        stats.inserted_ticks += loader.merge_staging(conn, int(season_year))

    return stats


def _existing_books(conn: psycopg.Connection, slugs: Iterable[str]) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT slug, book_id FROM {}.lh_book").format(
                sql.Identifier(SCHEMA)
            )
        )
        mapping = {slug: book_id for slug, book_id in cur.fetchall()}
    missing = sorted(set(slugs) - set(mapping))
    if missing:
        print(f"  (dry run) unregistered books would be added: {missing}")
    return mapping
