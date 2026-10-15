"""Read ``SELECT * FROM <table>`` loaders from S3 season pieces plus the database.

The plain table loaders (players, games, odds, injuries, referees, All-Star)
all run ``SELECT * ... WHERE season_year = ANY(seasons) OR game_id = ANY(extra
ids) [ORDER BY ...]``. Past seasons rarely change, so a refresher stores each
past season in S3 (``season_store``), validated against a server fingerprint.
A loader then reads those seasons from S3 and queries the database only for
the rest: the current season, any season S3 cannot vouch for, and the extra
game ids outside the requested seasons.

The result must be the frame the single query gives through
``read_sql_canonical``, and is built to be:

* rows are rebuilt as plain Python values (``None`` where missing, as the
  driver returns them) and converted by pandas' own ``read_sql`` path, so each
  column's dtype is inferred over all rows together, exactly as for one query,
  even where a column is entirely NULL in one season;
* the ``ORDER BY`` runs are put back in query order (by sorting on keys that
  sort the same everywhere, or by concatenating seasons in query order), and
  ``canonical_row_order`` fixes the order inside each run.

A season is read from S3 only if its manifest was validated within
``MAX_VALIDATION_AGE``; otherwise it is queried, so a refresher that stops
running makes loads slower, never stale.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import pandas.io.sql as pandas_sql
import pyarrow
from psycopg import sql

from nba_ou.postgre_db.input_cache import code_key
from nba_ou.postgre_db.row_order import (
    canonical_row_order,
    final_order_by,
    read_sql_canonical,
)
from nba_ou.postgre_db.season_store import SeasonStore
from nba_ou.utils.general_utils import get_season_year_from_date

#: A season validated longer ago than this is read from the database.
MAX_VALIDATION_AGE = pd.Timedelta(hours=36)


def s3_seasons_enabled() -> bool:
    """``NBA_OU_S3_SEASONS=0`` makes every loader query the database only."""
    return os.environ.get("NBA_OU_S3_SEASONS", "1") != "0"


@dataclass(frozen=True)
class SeasonTable:
    """A table read with ``SELECT *`` by season, and how its rows are ordered.

    ``order_by`` is the loader's SQL ``ORDER BY`` text. ``sort_keys`` lists its
    keys as ``(column, ascending)`` when they sort identically in pandas and in
    Postgres (dates, numbers, digit-only ids), so assembled rows can be put back
    in query order by sorting. Otherwise the query must start with
    ``season_year`` and pieces are concatenated in season order; such a table
    cannot take extra game ids.
    """

    schema: str
    table: str
    order_by: str = ""
    sort_keys: tuple[tuple[str, bool], ...] = field(default=())

    @property
    def dataset(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def order_columns(self) -> tuple[str, ...]:
        return (
            final_order_by(f"SELECT 1 ORDER BY {self.order_by}")
            if self.order_by
            else ()
        )

    def code_key(self) -> str:
        """What the stored bytes depend on: the season query, its row order,
        pandas' conversion and the library versions. Not the reading side, so
        changing how pieces are assembled keeps the store valid."""
        return code_key(
            season_piece,
            season_fingerprint,
            read_sql_canonical,
            canonical_row_order,
            final_order_by,
            pandas_sql._wrap_result,
            pandas_sql._convert_arrays_to_dataframe,
            dataset=self.dataset,
            order_by=self.order_by,
            pyarrow=pyarrow.__version__,
        )

    def _from(self) -> sql.Composable:
        return sql.SQL("{}.{}").format(
            sql.Identifier(self.schema), sql.Identifier(self.table)
        )

    def _order(self) -> sql.Composable:
        return sql.SQL(" ORDER BY " + self.order_by) if self.order_by else sql.SQL("")


def season_piece(spec: SeasonTable, conn, season: int) -> pd.DataFrame:
    """One season of the table, in ``read_sql_canonical`` order."""
    query = sql.SQL("SELECT * FROM {} WHERE season_year = %s{}").format(
        spec._from(), spec._order()
    )
    return read_sql_canonical(query.as_string(conn), conn, params=(int(season),))


def season_fingerprint(spec: SeasonTable, conn, season: int) -> list:
    """Row count and a hash over every column of the season's rows."""
    query = sql.SQL(
        "SELECT count(*), sum(hashtextextended(t::text, 0)) FROM {} t "
        "WHERE season_year = %s"
    ).format(spec._from())
    with conn.cursor() as cur:
        cur.execute(query, (int(season),))
        return [None if v is None else str(v) for v in cur.fetchone()]


def rows_as_records(frame: pd.DataFrame) -> list[tuple]:
    """``frame``'s rows as values pandas' ``read_sql`` conversion infers the
    same dtypes from as from the driver's rows.

    Missing values go back as ``None``, as the driver returns NULL, except in
    float columns: there they stay NaN. A float column means its piece held a
    non-NULL value (a number, or a ``NUMERIC`` NaN, which pandas also turns
    into NaN), and a NaN marks that, where ``None`` would let an otherwise
    empty column fall back to ``object``.
    """
    columns = []
    for name in frame.columns:
        values = frame[name].tolist()
        if frame[name].dtype.kind != "f":
            values = [None if _missing(v) else v for v in values]
        columns.append(values)
    return list(zip(*columns)) if columns else []


def _missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # list-like values are never "missing"
        return False


def assemble(spec: SeasonTable, pieces: list[pd.DataFrame]) -> pd.DataFrame:
    """The frame one query over all ``pieces`` gives through ``read_sql_canonical``.

    pandas converts each column on its own, so columns are assembled one by
    one. Where every piece has the same dtype, concatenating gives what the
    conversion would: it inferred that dtype from each piece's values, and
    over their union infers it again (an ``object`` piece held values that
    block conversion, which still block it). Only a column whose dtype differs
    between pieces -- typically NULL for whole seasons -- is rebuilt from
    driver-style values and converted again.

    ``pieces`` must be in season order matching the query's first ``ORDER BY``
    key when the table has no ``sort_keys``.
    """
    pieces = [p.reset_index(drop=True) for p in pieces if p is not None]
    columns = list(pieces[0].columns)
    for piece in pieces[1:]:
        if list(piece.columns) != columns:
            raise ValueError(f"{spec.dataset}: season pieces have different columns")
    data = {}
    for name in columns:
        parts = [piece[name] for piece in pieces]
        if len({str(part.dtype) for part in parts}) == 1:
            data[name] = pd.concat(parts, ignore_index=True)
        else:
            data[name] = _reconverted(name, parts)
    frame = pd.DataFrame(data, columns=columns)
    if spec.sort_keys:
        keys = [k for k, _ in spec.sort_keys]
        frame = frame.sort_values(
            keys,
            ascending=[a for _, a in spec.sort_keys],
            kind="mergesort",
            # Postgres: ASC puts NULLs last, DESC first.
            na_position="last" if spec.sort_keys[0][1] else "first",
        ).reset_index(drop=True)
    return canonical_row_order(frame, spec.order_columns)


def _reconverted(name: str, parts: list[pd.Series]) -> pd.Series:
    """One column converted from driver-style values, as ``read_sql`` would."""
    values = [row[0] for part in parts for row in rows_as_records(part.to_frame(name))]
    converted = pandas_sql._wrap_result(
        [(v,) for v in values], [name], coerce_float=True
    )
    return converted[name]


#: The loaders this module serves, by dataset.
SEASON_TABLES: dict[str, SeasonTable] = {
    spec.dataset: spec
    for spec in (
        SeasonTable("nba_players", "nba_players"),
        SeasonTable("nba_injuries", "nba_injuries"),
        SeasonTable("nba_refs", "nba_refs"),
        SeasonTable(
            "nba_games",
            "nba_games",
            "game_date DESC, game_id DESC",
            (("game_date", False), ("game_id", False)),
        ),
        SeasonTable(
            "odds_yahoo", "odds_yahoo", "game_date DESC", (("game_date", False),)
        ),
        SeasonTable(
            "odds_sportsbook",
            "odds_sportsbook",
            "game_date DESC",
            (("game_date", False),),
        ),
        SeasonTable(
            "nba_all_star_voting",
            "nba_all_star_voting",
            "season_year, conference, position, fan_rank NULLS LAST",
        ),
    )
}


def current_season_year(now: pd.Timestamp | None = None) -> int:
    now = pd.Timestamp.now(tz="US/Eastern") if now is None else pd.Timestamp(now)
    return int(get_season_year_from_date(now.tz_localize(None) if now.tz else now))


def read_season_table(
    spec: SeasonTable,
    season_years: list[int],
    extra_game_ids: list[str] | None,
    *,
    store: SeasonStore,
    conn,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The loader's frame from S3 past seasons plus one database query.

    Returns the frame and a summary (which seasons came from where).
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current_season_year(now)
    key = spec.code_key()
    requested = sorted({int(s) for s in season_years})
    extras = sorted({str(g) for g in (extra_game_ids or [])})
    if extras and spec.order_by and not spec.sort_keys:
        raise ValueError(f"{spec.dataset} cannot assemble extra game ids")

    def _stored(season: int) -> pd.DataFrame | None:
        entry = store.entry(spec.dataset, key, season)
        if entry is None or now - entry.validated_at > MAX_VALIDATION_AGE:
            return None
        return store.read(spec.dataset, key, season, entry)

    past = [s for s in requested if s < current]
    # Seasons download in parallel: each is two small GETs, mostly latency.
    with ThreadPoolExecutor(max_workers=8) as pool:
        pieces_by_season = dict(zip(past, pool.map(_stored, past)))
    from_s3 = {s: p for s, p in pieces_by_season.items() if p is not None}
    from_db = [s for s in requested if s not in from_s3]

    db_frame = None
    if from_db or extras:
        conditions, params = [], []
        if from_db:
            conditions.append(sql.SQL("season_year = ANY(%s)"))
            params.append(from_db)
        if extras:
            # Rows of the requested seasons are already in; these are the
            # extra games' rows from any other season.
            conditions.append(
                sql.SQL("(game_id = ANY(%s) AND NOT season_year = ANY(%s))")
            )
            params += [extras, requested]
        # With the table's ORDER BY, so a season queried here keeps the runs
        # the server orders, as its stored twin does.
        query = sql.SQL("SELECT * FROM {} WHERE {}{}").format(
            spec._from(), sql.SQL(" OR ").join(conditions), spec._order()
        )
        db_frame = read_sql_canonical(query.as_string(conn), conn, params=tuple(params))

    summary = {"s3": sorted(from_s3), "db": from_db, "extra_ids": len(extras)}
    if not from_s3:
        return db_frame, summary
    pieces = list(from_s3.values())
    if db_frame is not None and len(db_frame):
        pieces.append(db_frame)
    if spec.order_by and not spec.sort_keys:
        # Keys that only Postgres can order (text by collation), led by
        # season_year: put whole seasons in query order instead of sorting.
        pieces = _by_season_in_query_order(spec, pieces)
    return assemble(spec, pieces), summary


def _by_season_in_query_order(spec: SeasonTable, pieces) -> list[pd.DataFrame]:
    if spec.order_columns[0] != "season_year":
        raise ValueError(f"{spec.dataset}: ORDER BY must start with season_year")
    seasons: dict[int, pd.DataFrame] = {}
    for piece in pieces:
        for season, part in piece.groupby("season_year", sort=False):
            if int(season) in seasons:
                raise ValueError(f"{spec.dataset}: season {season} in two pieces")
            seasons[int(season)] = part.reset_index(drop=True)
    descending = spec.order_by.split(",")[0].strip().upper().endswith("DESC")
    return [seasons[s] for s in sorted(seasons, reverse=descending)]


def refresh_season_table(
    spec: SeasonTable,
    seasons: list[int],
    *,
    store: SeasonStore,
    conn,
    now: pd.Timestamp | None = None,
    force: bool = False,
) -> dict[int, str]:
    """Validate (and if changed, refetch and upload) each past season.

    The fingerprint is taken before the fetch, so data changing in between
    leaves an entry the next refresh replaces.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current_season_year(now)
    key = spec.code_key()
    outcome = {}
    for season in sorted({int(s) for s in seasons}):
        if season >= current:
            outcome[season] = "current season, not stored"
            continue
        fingerprint = season_fingerprint(spec, conn, season)
        entry = store.entry(spec.dataset, key, season)
        if entry is not None and entry.fingerprint == fingerprint and not force:
            store.touch(spec.dataset, key, season, entry, now)
            outcome[season] = "unchanged"
            continue
        piece = season_piece(spec, conn, season)
        store.write(spec.dataset, key, season, piece, fingerprint, now)
        outcome[season] = f"uploaded {len(piece):,} rows"
    return outcome


_shared: dict[str, SeasonStore | None] = {}


def _shared_store() -> SeasonStore | None:
    """One store (one S3 client) per process; None when S3 is not configured."""
    if "store" not in _shared:
        from nba_ou.postgre_db.season_store import default_store

        _shared["store"] = default_store()
    return _shared["store"]


def read_through(
    dataset: str, season_years: list[int] | None, extra_game_ids, conn
) -> pd.DataFrame | None:
    """For a table loader: its frame from S3 past seasons plus the database.

    Returns None whenever the loader should run its own query instead: S3
    seasons switched off (``NBA_OU_S3_SEASONS=0``) or not configured, no
    season filter (an all-rows or extra-ids-only query), or any failure on the
    S3 side. Whatever happens here, the loader's result is the same frame.
    """
    if not s3_seasons_enabled() or not season_years:
        return None
    try:
        store = _shared_store()
        if store is None:
            return None
        frame, summary = read_season_table(
            SEASON_TABLES[dataset],
            season_years,
            list(extra_game_ids or []),
            store=store,
            conn=conn,
        )
    except Exception as exc:  # S3 down, credentials, a damaged piece ...
        print(
            f"! {dataset}: S3 seasons unavailable ({type(exc).__name__}: {exc}); "
            "querying the database"
        )
        try:
            conn.rollback()  # leave the connection usable for the fallback
        except Exception:
            pass
        return None
    print(
        f"  {dataset}: seasons {summary['s3'] or 'none'} from S3, "
        f"{summary['db'] or 'none'} from the database"
    )
    return frame
