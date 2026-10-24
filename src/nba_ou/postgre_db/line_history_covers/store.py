"""Reads and writes of the Covers line-history store.

Writes are insert-only, as in the SBR store: ``lh_game`` and ``lh_line`` both
go in with ``ON CONFLICT DO NOTHING``, ticks through ``COPY`` into an UNLOGGED
staging table and one ``INSERT ... SELECT`` per season. With ``commit=False``
the writes join the caller's transaction, so a game and its lines land
together or not at all (``update.ingest_games``). Reads return the frames
``line_history_aiven.fetch`` returns, decoded the same way.

``lh_consensus`` (the Covers Consensus pick counts, one row per stored game)
is insert-only too: a game's counts are read once, after it is over.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.postgre_db.line_history_aiven.fetch import (
    ALL_MARKETS,
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    GAME_COLUMNS,
    TICK_COLUMNS,
    _decode_ticks,
    _rows_to_frame,
)
from nba_ou.postgre_db.line_history_aiven.load import (
    GAME_DIM_INSERT_COLUMNS,
    STAGING_TABLE,
    _to_native,
)
from nba_ou.postgre_db.line_history_aiven.transform import OUTPUT_COLUMNS

from .schema import SCHEMA

#: ``lh_consensus`` columns, in insert order.
CONSENSUS_COLUMNS = [
    "game_id",
    "season_year",
    "spread_away_picks",
    "spread_home_picks",
    "total_over_picks",
    "total_under_picks",
    "fetched_at",
]
_PICK_COLUMNS = CONSENSUS_COLUMNS[2:6]

#: ``lh_game`` with its Covers id: what a consensus request needs.
GAME_EVENT_COLUMNS = [
    "game_id",
    "game_date",
    "season_year",
    "tipoff_utc",
    "event_id",
    "team_home",
    "team_away",
]


def insert_games(
    conn: psycopg.Connection, game_dim: pd.DataFrame, *, commit: bool = True
) -> int:
    """Add unseen games; a stored game is never changed."""
    if game_dim.empty:
        return 0
    query = sql.SQL(
        "INSERT INTO {}.lh_game ({cols}) VALUES ({vals}) "
        "ON CONFLICT (game_id) DO NOTHING"
    ).format(
        sql.Identifier(SCHEMA),
        cols=sql.SQL(", ").join(map(sql.Identifier, GAME_DIM_INSERT_COLUMNS)),
        vals=sql.SQL(", ").join(sql.Placeholder() * len(GAME_DIM_INSERT_COLUMNS)),
    )
    inserted = 0
    with conn.cursor() as cur:
        for row in game_dim[GAME_DIM_INSERT_COLUMNS].itertuples(index=False, name=None):
            cur.execute(query, tuple(_to_native(value) for value in row))
            inserted += cur.rowcount
    if commit:
        conn.commit()
    return inserted


def insert_lines(
    conn: psycopg.Connection,
    rows: pd.DataFrame,
    season_year: int,
    *,
    commit: bool = True,
) -> int:
    """COPY one season's rows to staging and merge them; returns rows inserted."""
    staging = sql.Identifier(STAGING_TABLE)
    schema = sql.Identifier(SCHEMA)
    cols = sql.SQL(", ").join(map(sql.Identifier, OUTPUT_COLUMNS))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{}").format(schema, staging))
        cur.execute(
            sql.SQL(
                "CREATE UNLOGGED TABLE {}.{} (LIKE {}.lh_line INCLUDING DEFAULTS)"
            ).format(schema, staging, schema)
        )
        copy_sql = sql.SQL("COPY {}.{} ({cols}) FROM STDIN").format(
            schema, staging, cols=cols
        )
        with cur.copy(copy_sql) as copy:
            for row in rows[OUTPUT_COLUMNS].itertuples(index=False, name=None):
                copy.write_row(tuple(_to_native(value) for value in row))
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.lh_line ({cols})
                SELECT {cols} FROM {schema}.{staging}
                WHERE season_year = %s
                ON CONFLICT (game_id, market_id, book_id, line_ts, season_year)
                DO NOTHING
                """
            ).format(schema=schema, staging=staging, cols=cols),
            (int(season_year),),
        )
        inserted = cur.rowcount
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{}").format(schema, staging))
    if commit:
        conn.commit()
    return inserted


def record_load(
    conn: psycopg.Connection,
    *,
    season_year: int,
    source_rows: int,
    loaded_rows: int,
    dropped: dict[str, int],
) -> None:
    """``lh_load_meta`` for the season: the last run's counts, UTC, high confidence."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {}.lh_load_meta
                    (season_year, timezone, confidence, source_rows,
                     loaded_rows, dropped_rows, loaded_at)
                VALUES (%s, 'UTC', 'high', %s, %s, %s, now())
                ON CONFLICT (season_year) DO UPDATE SET
                    source_rows = EXCLUDED.source_rows,
                    loaded_rows = EXCLUDED.loaded_rows,
                    dropped_rows = EXCLUDED.dropped_rows,
                    loaded_at = EXCLUDED.loaded_at
                """
            ).format(sql.Identifier(SCHEMA)),
            (int(season_year), int(source_rows), int(loaded_rows), json.dumps(dropped)),
        )
    conn.commit()


def stored_game_books(
    conn: psycopg.Connection, season_year: int
) -> dict[str, set[int]]:
    """game_id -> book ids holding ticks in all three markets, for every stored
    game of the season. A book missing a market counts as missing, so an answer
    that came back empty for one request type is asked for again.

    A stored game with no complete book maps to an empty set.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT g.game_id, l.book_id
                FROM {schema}.lh_game g
                LEFT JOIN (
                    SELECT game_id, book_id FROM {schema}.lh_line
                    WHERE season_year = %s
                    GROUP BY game_id, book_id
                    HAVING COUNT(DISTINCT market_id) = %s
                ) l USING (game_id)
                WHERE g.season_year = %s
                """
            ).format(schema=sql.Identifier(SCHEMA)),
            (int(season_year), len(ALL_MARKETS), int(season_year)),
        )
        stored: dict[str, set[int]] = {}
        for game_id, book_id in cur.fetchall():
            books = stored.setdefault(str(game_id), set())
            if book_id is not None:
                books.add(int(book_id))
    return stored


def fetch_games(season_years: Iterable[int], conn: psycopg.Connection) -> pd.DataFrame:
    """The ``lh_game`` rows of ``season_years``, typed like the SBR reader's."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT game_id, game_date, season_year, tipoff_utc, team_home, team_away
                FROM {}.lh_game WHERE season_year = ANY(%s)
                ORDER BY game_date, game_id
                """
            ).format(sql.Identifier(SCHEMA)),
            ([int(s) for s in season_years],),
        )
        games = _rows_to_frame(cur, GAME_COLUMNS)
    if games.empty:
        return games
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["tipoff_utc"] = pd.to_datetime(games["tipoff_utc"], utc=True)
    games["season_year"] = pd.to_numeric(games["season_year"]).astype("int64")
    return games.reset_index(drop=True)


def fetch_pregame_ticks(
    season_years: Iterable[int],
    conn: psycopg.Connection,
    *,
    markets: tuple[str, ...] = ALL_MARKETS,
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
) -> pd.DataFrame:
    """Every pre-game tick of ``season_years``, in ``TICK_COLUMNS`` and decoded
    like ``line_history_aiven.fetch.fetch_pregame_ticks``."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT l.game_id, l.season_year, m.code, b.slug,
                       l.line_ts, l.mins_to_tip, l.is_opener,
                       l.left_line, l.left_price, l.right_line, l.right_price
                FROM {schema}.lh_line l
                JOIN {schema}.lh_market m USING (market_id)
                JOIN {schema}.lh_book   b USING (book_id)
                WHERE l.season_year = ANY(%s)
                  AND l.is_pregame
                  AND l.mins_to_tip <= %s
                  AND m.code = ANY(%s)
                ORDER BY l.game_id, m.code, b.slug, l.line_ts
                """
            ).format(schema=sql.Identifier(SCHEMA)),
            (
                [int(s) for s in season_years],
                -int(min_minutes_before_tip),
                list(markets),
            ),
        )
        ticks = _rows_to_frame(cur, TICK_COLUMNS)
    return _decode_ticks(ticks)


def fetch_closing_ticks(
    conn: psycopg.Connection,
    *,
    season_years: Iterable[int] | None = None,
    game_ids: Iterable[str] | None = None,
    last_n: int | None = 5,
    markets: tuple[str, ...] = ALL_MARKETS,
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
) -> pd.DataFrame:
    """``line_history_aiven.fetch._fetch_closing_ticks_once`` on this schema:
    the last ``last_n`` priced pre-game ticks per (game, market, book) of
    finished games. The clock check is kept, though Covers' ``mins_to_tip`` is
    derived from the tip-off and always agrees."""
    filters, params = [], []
    if season_years is not None:
        filters.append(sql.SQL("l.season_year = ANY(%s)"))
        params.append([int(s) for s in season_years])
    if game_ids is not None:
        filters.append(sql.SQL("l.game_id = ANY(%s)"))
        params.append([str(g) for g in game_ids])
    if not filters:
        raise ValueError("Pass season_years or game_ids.")
    query = sql.SQL(
        """
        WITH eligible AS (
            SELECT l.game_id, l.season_year, m.code AS market,
                   b.slug AS book, l.line_ts,
                   l.mins_to_tip AS minutes_before_tip, l.is_opener,
                   l.left_line, l.left_price, l.right_line, l.right_price,
                   ROW_NUMBER() OVER (
                       PARTITION BY l.game_id, m.code, b.slug
                       ORDER BY l.line_ts DESC
                   ) AS recency
            FROM {schema}.lh_line l
            JOIN {schema}.lh_game   g USING (game_id)
            JOIN {schema}.lh_market m USING (market_id)
            JOIN {schema}.lh_book   b USING (book_id)
            WHERE ({filters})
              AND l.is_pregame
              AND l.mins_to_tip <= %s
              AND m.code = ANY(%s)
              AND l.left_price IS NOT NULL
              AND l.right_price IS NOT NULL
              AND g.tipoff_utc < now()
              AND g.tipoff_utc - l.line_ts >= make_interval(mins => %s)
              AND ABS(
                  EXTRACT(EPOCH FROM (g.tipoff_utc - l.line_ts)) / 60.0
                  + l.mins_to_tip
              ) <= 1
        )
        SELECT game_id, season_year, market, book, line_ts,
               minutes_before_tip, is_opener,
               left_line, left_price, right_line, right_price
        FROM eligible
        WHERE %s::int IS NULL OR recency <= %s::int
        ORDER BY game_id, market, book, line_ts
        """
    ).format(schema=sql.Identifier(SCHEMA), filters=sql.SQL(" OR ").join(filters))
    params += [
        -int(min_minutes_before_tip),
        list(markets),
        int(min_minutes_before_tip),
        last_n,
        last_n,
    ]
    with conn.cursor() as cur:
        cur.execute(query, params)
        ticks = _rows_to_frame(cur, TICK_COLUMNS)
    return _decode_ticks(ticks)


def season_fingerprint(conn: psycopg.Connection, season: int) -> list | None:
    """The SBR store's season fingerprint (``line_history_season_fingerprint``)
    on this schema, or None when the store holds no game of the season."""
    from nba_ou.postgre_db.line_history_aiven.fetch import _CLOSING_TICKS_FINGERPRINT

    query = sql.SQL(_CLOSING_TICKS_FINGERPRINT).format(schema=sql.Identifier(SCHEMA))
    with conn.cursor() as cur:
        cur.execute(query, {"season": int(season)})
        values = [None if v is None else str(v) for v in cur.fetchone()]
    return None if values[2] in (None, "0") else values


def insert_consensus(
    conn: psycopg.Connection, rows: pd.DataFrame, *, commit: bool = True
) -> int:
    """Add the pick counts of games that have none; stored counts are kept."""
    if rows.empty:
        return 0
    query = sql.SQL(
        "INSERT INTO {}.lh_consensus ({cols}) VALUES ({vals}) "
        "ON CONFLICT (game_id) DO NOTHING"
    ).format(
        sql.Identifier(SCHEMA),
        cols=sql.SQL(", ").join(map(sql.Identifier, CONSENSUS_COLUMNS)),
        vals=sql.SQL(", ").join(sql.Placeholder() * len(CONSENSUS_COLUMNS)),
    )
    inserted = 0
    with conn.cursor() as cur:
        for row in rows[CONSENSUS_COLUMNS].itertuples(index=False, name=None):
            cur.execute(query, tuple(_to_native(value) for value in row))
            inserted += cur.rowcount
    if commit:
        conn.commit()
    return inserted


def stored_consensus_game_ids(conn: psycopg.Connection, season_year: int) -> set[str]:
    """The games of the season whose pick counts are stored."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id FROM {}.lh_consensus WHERE season_year = %s"
            ).format(sql.Identifier(SCHEMA)),
            (int(season_year),),
        )
        return {str(game_id) for (game_id,) in cur.fetchall()}


def fetch_game_events(conn: psycopg.Connection, season_year: int) -> pd.DataFrame:
    """The season's ``lh_game`` rows with their Covers id (``event_id``)."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT {cols} FROM {schema}.lh_game WHERE season_year = %s "
                "ORDER BY game_date, game_id"
            ).format(
                cols=sql.SQL(", ").join(map(sql.Identifier, GAME_EVENT_COLUMNS)),
                schema=sql.Identifier(SCHEMA),
            ),
            (int(season_year),),
        )
        games = _rows_to_frame(cur, GAME_EVENT_COLUMNS)
    if games.empty:
        return games
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games["tipoff_utc"] = pd.to_datetime(games["tipoff_utc"], utc=True)
    return games.reset_index(drop=True)


def fetch_consensus(
    conn: psycopg.Connection,
    season_years: Iterable[int] | None = None,
    game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Stored pick counts in ``CONSENSUS_COLUMNS``; every game when both filters
    are None. ``parse.PickConsensus.sbr_consensus`` turns a row into SBR's
    shares."""
    filters, params = [], []
    if season_years is not None:
        filters.append(sql.SQL("season_year = ANY(%s)"))
        params.append([int(s) for s in season_years])
    if game_ids is not None:
        filters.append(sql.SQL("game_id = ANY(%s)"))
        params.append([str(g) for g in game_ids])
    where = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(filters) if filters else sql.SQL("")
    )
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT {cols} FROM {schema}.lh_consensus{where} ORDER BY game_id"
            ).format(
                cols=sql.SQL(", ").join(map(sql.Identifier, CONSENSUS_COLUMNS)),
                schema=sql.Identifier(SCHEMA),
                where=where,
            ),
            params,
        )
        consensus = _rows_to_frame(cur, CONSENSUS_COLUMNS)
    consensus["game_id"] = consensus["game_id"].astype(str)
    for column in ("season_year", *_PICK_COLUMNS):
        consensus[column] = pd.to_numeric(consensus[column]).astype("int64")
    consensus["fetched_at"] = pd.to_datetime(consensus["fetched_at"], utc=True)
    return consensus.reset_index(drop=True)
