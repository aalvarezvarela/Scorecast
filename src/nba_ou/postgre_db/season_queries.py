"""Season-filtered queries served from the S3 input cache: past seasons stored,
the rest from one database query.

``season_tables`` does this for ``SELECT *`` loaders. Some loaders run their own
query instead -- a join, an aggregate, a ``DISTINCT`` -- over rows that belong
to one season each (every row comes from one game, and a game belongs to one
season). Such a query's result over all seasons is the union of its results
per season, so:

* each finished season is stored as the query's own result for that season,
  fingerprinted on the server by a hash over that very result (so any change
  to any table it reads, in a way that changes its rows, refuses the entry);
* a build reads the stored seasons and runs the query once for every other
  row -- ``season IS NULL OR NOT season = ANY(stored)`` -- so nothing a single
  query would return is missed;
* the pieces are assembled as ``season_tables.assemble`` does and put in
  ``canonical_row_order``, which for a query without ``ORDER BY`` is a total
  order on the rows: the union's order is the single query's order.

A query is only eligible if every row's season is the season the filter
tests, its result has no ``ORDER BY``, and it reads no clock: those are the
conditions under which a finished season's result is fixed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import pandas.io.sql as pandas_sql
import pyarrow

from nba_ou.postgre_db.input_cache import code_key
from nba_ou.postgre_db.row_order import (
    canonical_row_order,
    final_order_by,
    read_sql_canonical,
)
from nba_ou.postgre_db.season_store import SeasonStore
from nba_ou.postgre_db.season_tables import (
    SeasonTable,
    assemble,
    check_outcome,
    current_season_year,
    servable,
)

#: Seasons a "every season" read probes in the store.
FIRST_STORED_SEASON = 2010


@dataclass(frozen=True)
class SeasonQuery:
    """A loader query with a ``{season_filter}`` placeholder in its WHERE clause."""

    dataset: str
    template: str
    season_column: str
    database: str
    params: tuple[tuple[str, object], ...] = field(default=())

    def __post_init__(self) -> None:
        if final_order_by(self.query("TRUE")):
            raise ValueError(f"{self.dataset}: a season query cannot have ORDER BY")

    def query(self, season_filter: str) -> str:
        return self.template.format(season_filter=season_filter)

    def code_key(self) -> str:
        return code_key(
            season_query_piece,
            season_query_fingerprint,
            read_sql_canonical,
            canonical_row_order,
            pandas_sql._wrap_result,
            pandas_sql._convert_arrays_to_dataframe,
            dataset=self.dataset,
            template=self.template,
            season_column=self.season_column,
            params=self.params,
            pyarrow=pyarrow.__version__,
        )

    def _params(self, **extra) -> dict:
        return {**dict(self.params), **extra}


#: ``assemble`` needs only the ordering rules: none here.
_UNORDERED = SeasonTable("query", "query")


def season_query_piece(spec: SeasonQuery, conn, season: int) -> pd.DataFrame:
    query = spec.query(f"{spec.season_column} = %(store_season)s")
    return read_sql_canonical(
        query, conn, params=spec._params(store_season=int(season))
    )


def season_query_fingerprint(spec: SeasonQuery, conn, season: int) -> list:
    piece = spec.query(f"{spec.season_column} = %(store_season)s")
    query = f"SELECT count(*), sum(hashtextextended(q::text, 0)) FROM ({piece}) q"
    with conn.cursor() as cur:
        cur.execute(query, spec._params(store_season=int(season)))
        return [None if v is None else str(v) for v in cur.fetchone()]


def read_season_query(
    spec: SeasonQuery,
    season_years: list[int] | None,
    *,
    store: SeasonStore,
    conn,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The query's result for ``season_years`` (None: every season)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current_season_year(now)
    key = spec.code_key()
    candidates = (
        list(range(FIRST_STORED_SEASON, current))
        if season_years is None
        else sorted({int(s) for s in season_years if int(s) < current})
    )

    def stored(season):
        entry = store.entry(spec.dataset, key, season)
        if not servable(entry, now):
            return None
        return store.read(spec.dataset, key, season, entry)

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = dict(zip(candidates, pool.map(stored, candidates), strict=True))
    from_s3 = {s: piece for s, piece in found.items() if piece is not None}

    column = spec.season_column
    conditions = [f"({column} IS NULL OR NOT {column} = ANY(%(store_stored)s))"]
    params = spec._params(store_stored=sorted(from_s3))
    if season_years is not None:
        conditions.append(f"{column} = ANY(%(store_requested)s)")
        params["store_requested"] = sorted({int(s) for s in season_years})
    db_frame = read_sql_canonical(
        spec.query(" AND ".join(conditions)), conn, params=params
    )
    summary = {"s3": sorted(from_s3)}
    if not from_s3:
        return db_frame, summary
    pieces = [from_s3[s] for s in sorted(from_s3)]
    if len(db_frame):
        pieces.append(db_frame)
    return assemble(_UNORDERED, pieces), summary


def refresh_season_query(
    spec: SeasonQuery,
    seasons: list[int],
    *,
    store: SeasonStore,
    conn,
    now: pd.Timestamp | None = None,
    force: bool = False,
    check_only: bool = False,
) -> dict[int, str]:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current_season_year(now)
    key = spec.code_key()
    outcome = {}
    for season in sorted({int(s) for s in seasons}):
        if season >= current:
            outcome[season] = "current season, not stored"
            continue
        fingerprint = season_query_fingerprint(spec, conn, season)
        entry = store.entry(spec.dataset, key, season)
        if check_only:
            outcome[season] = check_outcome(entry, fingerprint)
            continue
        if entry is not None and entry.fingerprint == fingerprint and not force:
            store.touch(spec.dataset, key, season, entry, now)
            outcome[season] = "unchanged"
            continue
        piece = season_query_piece(spec, conn, season)
        store.write(spec.dataset, key, season, piece, fingerprint, now)
        outcome[season] = f"uploaded {len(piece):,} rows"
    return outcome


def read_through_query(spec: SeasonQuery, season_years, conn) -> pd.DataFrame | None:
    """``read_season_query`` from the shared store, or None to run the loader's
    own query: S3 off or not configured, or any failure (the connection is
    rolled back so the loader can still use it)."""
    from nba_ou.postgre_db import season_tables

    if not season_tables.s3_seasons_enabled():
        return None
    try:
        store = season_tables._shared_store()
        if store is None:
            return None
        frame, summary = read_season_query(spec, season_years, store=store, conn=conn)
    except Exception as exc:  # noqa: BLE001 -- S3 is only a shortcut
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        print(
            f"! {spec.dataset}: S3 input cache unavailable ({type(exc).__name__}: {exc})"
        )
        return None
    print(f"  {spec.dataset}: seasons {summary['s3'] or 'none'} from S3")
    return frame


def registered_season_queries() -> dict[str, tuple[SeasonQuery, str]]:
    """Every loader query served this way, with the SQL listing its seasons."""
    from nba_ou.data_processing.injury_status.status_history import (
        _box_history_query,
    )
    from nba_ou.postgre_db.injury_report_aiven import fetch as report_fetch
    from nba_ou.postgre_db.injury_report_aiven.schema import SCHEMA

    report_seasons = f"SELECT DISTINCT season_year FROM {SCHEMA}.ir_game"
    specs = {
        "nba_players.box_history": (
            _box_history_query(),
            "SELECT DISTINCT season_year FROM nba_players.nba_players "
            "WHERE season_year >= 2017",
        )
    }
    for name, query, params in (
        ("last_status_before_tip", report_fetch._LAST_STATUS_BEFORE_TIP, ()),
        ("filing_at_tip", report_fetch._FILING_AT_TIP, ()),
        ("report_age_at_tip", report_fetch._REPORT_AGE_AT_TIP, ()),
        (
            "listed_status_events",
            report_fetch._LISTED_STATUS_EVENTS,
            (("statuses", list(report_fetch.LISTED_STATUS_CODES)),),
        ),
        ("listed_pairs", report_fetch._LISTED_PAIRS, ()),
    ):
        spec = report_fetch._season_query(name, query, params)
        specs[spec.dataset] = (spec, report_seasons)
    return specs
