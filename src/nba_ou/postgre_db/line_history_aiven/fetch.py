"""Read pre-game ticks out of the Aiven line-history store.

Read-only. The store's storage encodings are undone exactly once -- here -- so
that no feature module downstream has to remember them:

* ``left_line``/``right_line`` are half-points doubled into ``SMALLINT``
  (``449 -> 224.5``). Divided by 2 on the way out.
* ``left_price``/``right_price`` are **American** odds, already null where the
  book was off the board.
* ``mins_to_tip`` is negative pre-game. Callers think in *positive* minutes
  before tip, so this module emits ``minutes_before_tip = -mins_to_tip`` and the
  sign convention never leaks into feature code.

Per-market column semantics (see ``transform.py`` and the Phase 0 findings):

* ``totals`` -- ``left`` is OVER, ``right`` is UNDER, and a valid quote has
  ``left_line == right_line``.
* ``point_spread`` -- mirrored, ``left_line == -right_line``. The price-bleed
  repair leaves some rows with a valid price and a NULL line, so a present price
  does not imply a present line.
* ``money_line`` -- prices only; both line columns are NULL by nature.
"""

from __future__ import annotations

import time

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.config.odds_columns import HISTORY_ONLY_BOOKS
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.input_cache import SeasonCache, code_key

from .schema import SCHEMA

#: Market codes as stored in ``lh_market``.
MARKET_TOTALS = "totals"
MARKET_SPREAD = "point_spread"
MARKET_MONEYLINE = "money_line"
ALL_MARKETS: tuple[str, ...] = (MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE)

#: Books that only ever appear in part of the history. Carrying one of these
#: unflagged lets a model recover the season from mere column availability.
PARTIAL_COVERAGE_BOOKS: tuple[str, ...] = ("fanatics_sportsbook",)

#: Safety margin on the leakage filter. Phase 0 recommends staying clear of the
#: tipoff boundary rather than trusting it to the minute, because a per-game
#: tipoff that is a few minutes stale would otherwise admit an in-play tick.
DEFAULT_MIN_MINUTES_BEFORE_TIP = 5

#: A game whose freshest pre-tip tick (any book, any market) is further than
#: this from tip-off has a truncated history: SBR's feed froze before the close
#: and nothing after it can be read. Measured 2026-10-21: per-season median gap
#: 2-8 minutes, under 1% of games past 60 outside 2019-20, and every feed freeze
#: (2024-03-28, 2026-03-23, 2026-04-20, 2026-05-01, 2026-10-20) 3+ hours.
TRUNCATED_HISTORY_MINUTES = 60

TICK_COLUMNS = [
    "game_id",
    "season_year",
    "market",
    "book",
    "line_ts",
    "minutes_before_tip",
    "is_opener",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]

GAME_COLUMNS = [
    "game_id",
    "game_date",
    "season_year",
    "tipoff_utc",
    "team_home",
    "team_away",
]


def _rows_to_frame(cursor: psycopg.Cursor, columns: list[str]) -> pd.DataFrame:
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _decode_line(values: pd.Series) -> pd.Series:
    """Undo the doubled-half-point SMALLINT encoding."""
    return pd.to_numeric(values, errors="coerce") / 2.0


def fetch_games(
    season_years: list[int],
    *,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """Return the ``lh_game`` dimension for ``season_years``."""
    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        query = sql.SQL(
            """
            SELECT game_id, game_date, season_year, tipoff_utc,
                   team_home, team_away
            FROM {}.lh_game
            WHERE season_year = ANY(%s)
            ORDER BY game_date, game_id
            """
        ).format(sql.Identifier(SCHEMA))
        with conn.cursor() as cur:
            cur.execute(query, ([int(s) for s in season_years],))
            games = _rows_to_frame(cur, GAME_COLUMNS)
    finally:
        if owned:
            conn.close()

    if games.empty:
        return games

    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["tipoff_utc"] = pd.to_datetime(games["tipoff_utc"], utc=True)
    games["season_year"] = pd.to_numeric(games["season_year"]).astype("int64")
    return games.reset_index(drop=True)


#: Attempts per chunk. Aiven cuts long result transfers now and then -- seen as
#: "SSL error: unexpected eof" and "AdminShutdown: terminating connection due to
#: administrator command" -- without restarting; a fresh connection succeeds.
FETCH_ATTEMPTS = 3
CLOSING_FETCH_GAME_CHUNK = 500


def _fetch_one_with_retries(chunk: dict, fetch_one, label) -> pd.DataFrame:
    """``fetch_one(**chunk)`` on its own connection, retrying a dropped one."""
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            return fetch_one(**chunk)
        except psycopg.OperationalError as exc:
            if attempt == FETCH_ATTEMPTS:
                raise
            print(
                f"! line-history fetch for {label(chunk)} dropped "
                f"({type(exc).__name__}); retrying ({attempt}/{FETCH_ATTEMPTS - 1})"
            )
            time.sleep(5 * attempt)


def _concat_ticks(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=TICK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _fetch_with_retries(chunks: list[dict], fetch_one, label) -> pd.DataFrame:
    """Run ``fetch_one(**chunk)`` per chunk on its own connection, retrying drops."""
    return _concat_ticks(
        [_fetch_one_with_retries(chunk, fetch_one, label) for chunk in chunks]
    )


def fetch_pregame_ticks(
    season_years: list[int],
    *,
    markets: tuple[str, ...] = ALL_MARKETS,
    exclude_books: tuple[str, ...] = (),
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
    conn: psycopg.Connection | None = None,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Return every pre-game tick for ``season_years``, decoded.

    One query per season, each on a fresh connection and retried if Aiven drops
    it: all seasons at once is ~2.5M rows, and on 2026-09-17 that single
    transfer was cut ("AdminShutdown"). With an explicit ``conn`` the query runs
    once, as given.

    ``use_cache`` reuses a season from the local input cache while its
    server-side fingerprint is unchanged, as ``fetch_closing_ticks`` does.
    """
    kwargs = {
        "markets": markets,
        "exclude_books": exclude_books,
        "min_minutes_before_tip": min_minutes_before_tip,
    }
    if conn is not None:
        return _fetch_pregame_ticks_once(season_years, conn=conn, **kwargs)
    if use_cache:
        return _cached_season_ticks(
            "pregame_ticks",
            [int(s) for s in season_years],
            lambda **chunk: _fetch_pregame_ticks_once(chunk["season_years"], **kwargs),
            (_fetch_pregame_ticks_once,),
            kwargs,
        )
    return _fetch_with_retries(
        [{"season_years": [int(s)]} for s in season_years],
        lambda season_years: _fetch_pregame_ticks_once(season_years, **kwargs),
        _chunk_label,
    )


def _fetch_pregame_ticks_once(
    season_years: list[int],
    *,
    markets: tuple[str, ...] = ALL_MARKETS,
    exclude_books: tuple[str, ...] = (),
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """Every pre-game tick for ``season_years`` in a single query, decoded.

    The ``is_pregame`` flag alone is not the leakage filter -- it only separates
    pre-game from in-play. Snapshot horizons are applied later, per snapshot, in
    ``snapshots.py``; this function's ``min_minutes_before_tip`` is only the
    blanket safety margin around the tipoff boundary itself.
    """
    if not markets:
        raise ValueError("markets must not be empty.")
    unknown = set(markets) - set(ALL_MARKETS)
    if unknown:
        raise ValueError(f"Unknown market(s): {sorted(unknown)}")
    if min_minutes_before_tip < 0:
        raise ValueError("min_minutes_before_tip must be >= 0.")

    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        query = sql.SQL(
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
              AND NOT (b.slug = ANY(%s))
            ORDER BY l.game_id, m.code, b.slug, l.line_ts
            """
        ).format(schema=sql.Identifier(SCHEMA))
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    [int(s) for s in season_years],
                    -int(min_minutes_before_tip),
                    list(markets),
                    list(exclude_books),
                ),
            )
            ticks = _rows_to_frame(cur, TICK_COLUMNS)
    finally:
        if owned:
            conn.close()

    return _decode_ticks(ticks)


def _decode_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    """Undo the storage encodings on a frame shaped like ``TICK_COLUMNS``."""
    if ticks.empty:
        return ticks

    ticks["game_id"] = ticks["game_id"].astype(str)
    ticks["line_ts"] = pd.to_datetime(ticks["line_ts"], utc=True)
    ticks["season_year"] = pd.to_numeric(ticks["season_year"]).astype("int64")

    # Stored as negative minutes relative to tip; downstream thinks in positive
    # "minutes before tip", which is also how the snapshot grid is expressed.
    ticks["minutes_before_tip"] = -pd.to_numeric(
        ticks["minutes_before_tip"], errors="coerce"
    ).astype("float64")

    ticks["is_opener"] = ticks["is_opener"].astype(bool)

    for column in ["left_line", "right_line"]:
        ticks[column] = _decode_line(ticks[column])
    for column in ["left_price", "right_price"]:
        ticks[column] = pd.to_numeric(ticks[column], errors="coerce")

    return ticks.reset_index(drop=True)


def fetch_closing_ticks(
    *,
    season_years: list[int] | None = None,
    game_ids: list[str] | None = None,
    last_n: int | None = 5,
    markets: tuple[str, ...] = ALL_MARKETS,
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
    conn: psycopg.Connection | None = None,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Priced pre-game ticks of finished games, for reconstructing the close.

    ``use_cache`` reuses a season's ticks from the local input cache while its
    server-side fingerprint (``line_history_season_fingerprint``) is unchanged; see
    ``nba_ou.postgre_db.input_cache``. It applies to season fetches without an
    explicit ``conn``, and returns exactly what fetching would.

    Fetched one season (or 500 game ids) per query, each on a fresh connection
    and retried on a dropped one: a single all-seasons query moves ~725k rows
    for 90+ seconds, long enough for Aiven to cut it. With an explicit ``conn``
    the query runs once, as given. See ``_fetch_closing_ticks_once`` for what is
    returned.
    """
    kwargs = {
        "last_n": last_n,
        "markets": markets,
        "min_minutes_before_tip": min_minutes_before_tip,
    }
    if conn is not None or (season_years is not None and game_ids is not None):
        return _fetch_closing_ticks_once(
            season_years=season_years, game_ids=game_ids, conn=conn, **kwargs
        )
    if season_years is None and game_ids is None:
        raise ValueError("Pass season_years or game_ids.")

    if season_years is not None and use_cache:
        return _cached_season_ticks(
            "closing_ticks",
            [int(s) for s in season_years],
            lambda **chunk: _fetch_closing_ticks_once(**chunk, **kwargs),
            (_fetch_closing_ticks_once,),
            kwargs,
        )
    if season_years is not None:
        chunks = [{"season_years": [int(s)]} for s in season_years]
    else:
        ids = [str(g) for g in game_ids]
        chunks = [
            {"game_ids": ids[i : i + CLOSING_FETCH_GAME_CHUNK]}
            for i in range(0, len(ids), CLOSING_FETCH_GAME_CHUNK)
        ]

    return _fetch_with_retries(
        chunks,
        lambda **chunk: _fetch_closing_ticks_once(**chunk, **kwargs),
        _chunk_label,
    )


#: Every input of a season's tick query, as one query. Written for
#: ``_fetch_closing_ticks_once``, the widest reader; ``_fetch_pregame_ticks_once``
#: reads a subset (no games, no clock), so it is covered too, if conservatively:
#: the season's lines, the games they join (matched by season and by id, so a
#: game filed under another season is still covered), both dimensions, and how
#: many of those games have tipped off -- the query keeps only those, so the
#: clock is an input too. Row text renders NULLs by position, so moving a value
#: between columns changes the hash.
_CLOSING_TICKS_FINGERPRINT = """
    WITH season_games AS (
        -- A UNION, not "season_year = s OR game_id IN (...)": Postgres runs
        -- that OR as a rescan of the lines per game (10+ minutes a season).
        SELECT g.* FROM {schema}.lh_game g WHERE g.season_year = %(season)s
        UNION
        SELECT g.* FROM {schema}.lh_game g
        JOIN (
            SELECT DISTINCT l.game_id FROM {schema}.lh_line l
            WHERE l.season_year = %(season)s
        ) season_lines USING (game_id)
    )
    SELECT
        (SELECT count(*) FROM {schema}.lh_line l WHERE l.season_year = %(season)s),
        (SELECT sum(hashtextextended(l::text, 0)) FROM {schema}.lh_line l
         WHERE l.season_year = %(season)s),
        (SELECT count(*) FROM season_games),
        (SELECT sum(hashtextextended(g::text, 0)) FROM season_games g),
        (SELECT count(*) FROM season_games WHERE tipoff_utc < now()),
        (SELECT sum(hashtextextended(m::text, 0)) FROM {schema}.lh_market m),
        (SELECT sum(hashtextextended(b::text, 0)) FROM {schema}.lh_book b)
"""


#: A fingerprint is a few seconds a season; one this slow means something is
#: wrong server-side, and the season is fetched rather than waited on.
FINGERPRINT_TIMEOUT = "120s"


def line_history_season_fingerprint(conn: psycopg.Connection, season: int) -> list:
    """Server-side fingerprint of everything one season's tick queries read."""
    query = sql.SQL(_CLOSING_TICKS_FINGERPRINT).format(schema=sql.Identifier(SCHEMA))
    try:
        with conn.cursor() as cur:
            cur.execute(query, {"season": int(season)})
            return [None if value is None else str(value) for value in cur.fetchone()]
    except psycopg.Error:
        conn.rollback()  # keep the connection usable for the next season
        raise


def _season_ticks_key(loaders: tuple, kwargs: dict) -> str:
    return code_key(
        *loaders,
        _decode_ticks,
        _rows_to_frame,
        _decode_line,
        line_history_season_fingerprint,
        fingerprint_sql=_CLOSING_TICKS_FINGERPRINT,
        **kwargs,
    )


def _ticks_dataset(name: str) -> str:
    return f"line_history_aiven.{name}"


def _s3_season_ticks(name: str, key: str, seasons: list[int]) -> dict:
    """Finished seasons the S3 input cache holds, validated recently enough.

    The refresher (``refresh_season_ticks``) re-checks each finished season's
    server fingerprint daily and uploads it when it changed; whether a stored
    season is served is ``season_tables.servable``. Any failure means "not in
    S3": the caller fetches it.
    """
    from concurrent.futures import ThreadPoolExecutor

    from nba_ou.postgre_db import season_tables

    if not season_tables.s3_seasons_enabled():
        return {}
    try:
        store = season_tables._shared_store()
        if store is None:
            return {}
        current = season_tables.current_season_year()
        now = pd.Timestamp.now(tz="UTC")
        dataset = _ticks_dataset(name)

        def one(season):
            entry = store.entry(dataset, key, season)
            if not season_tables.servable(entry, now):
                return None
            frame = store.read(dataset, key, season, entry)
            return None if frame is None else (frame, entry.fingerprint)

        past = [s for s in seasons if s < current]
        with ThreadPoolExecutor(max_workers=8) as pool:
            found = dict(zip(past, pool.map(one, past), strict=True))
        return {s: found_s for s, found_s in found.items() if found_s is not None}
    except Exception as exc:  # noqa: BLE001 -- S3 is only a shortcut
        print(f"! {name}: S3 input cache unavailable ({type(exc).__name__}: {exc})")
        return {}


def _cached_season_ticks(
    name: str, seasons: list[int], fetch_one, loaders: tuple, kwargs: dict
) -> pd.DataFrame:
    """One season per query, as ``_fetch_with_retries`` runs it, through the caches.

    Finished seasons come from the S3 input cache when it holds them (see
    ``_s3_season_ticks``); the rest from the local input cache while their
    server fingerprint is unchanged, or from the server.
    """
    key = _season_ticks_key(loaders, kwargs)
    from_s3 = _s3_season_ticks(name, key, seasons)
    cache = SeasonCache(name, key)
    frames = []
    missing = [s for s in seasons if s not in from_s3]
    if missing:
        with connect_line_history_db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{FINGERPRINT_TIMEOUT}'")
            fetched = {
                season: cache.get(
                    season,
                    lambda s: line_history_season_fingerprint(conn, s),
                    lambda s: _fetch_one_with_retries(
                        {"season_years": [s]}, fetch_one, _chunk_label
                    ),
                )
                for season in missing
            }
    else:
        fetched = {}
    frames = [from_s3[s][0] if s in from_s3 else fetched[s] for s in seasons]
    label = name.replace("_", " ")
    print(
        f"  {label}: {len(from_s3)} season(s) from S3, {cache.hits} from the "
        f"local input cache, {cache.misses} fetched"
    )
    ticks = _concat_ticks(frames)
    # The server fingerprint each season's ticks were validated under, for
    # callers that key derived results on the line history (the odds repair).
    ticks.attrs["season_fingerprints"] = {
        **cache.fingerprints,
        **{s: fingerprint for s, (_, fingerprint) in from_s3.items()},
    }
    return ticks


def pregame_tick_fingerprints(
    season_years: list[int],
    *,
    exclude_books: tuple[str, ...] = (),
    markets: tuple[str, ...] = ALL_MARKETS,
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
) -> dict[int, list]:
    """Each season's server fingerprint, as ``fetch_pregame_ticks`` would
    validate it, without reading the ticks.

    A finished season the S3 input cache validated recently gives its stored
    fingerprint -- the refresher re-checks it against the server daily --
    and any other the server's (``line_history_season_fingerprint``).
    """
    kwargs = {
        "markets": markets,
        "exclude_books": tuple(exclude_books),
        "min_minutes_before_tip": min_minutes_before_tip,
    }
    key = _season_ticks_key((_fetch_pregame_ticks_once,), kwargs)
    out = {}
    try:
        from nba_ou.postgre_db import season_tables

        store = (
            season_tables._shared_store()
            if season_tables.s3_seasons_enabled()
            else None
        )
        current = season_tables.current_season_year()
        now = pd.Timestamp.now(tz="UTC")
        for season in season_years:
            if store is None or season >= current:
                continue
            entry = store.entry(_ticks_dataset("pregame_ticks"), key, season)
            if season_tables.servable(entry, now):
                out[int(season)] = entry.fingerprint
    except Exception as exc:  # noqa: BLE001 -- S3 is only a shortcut
        print(
            f"! pregame ticks: S3 fingerprints unavailable ({type(exc).__name__}: {exc})"
        )
    missing = [int(s) for s in season_years if int(s) not in out]
    if missing:
        with connect_line_history_db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{FINGERPRINT_TIMEOUT}'")
            for season in missing:
                out[season] = line_history_season_fingerprint(conn, season)
    return {season: out[int(season)] for season in season_years}


def _closing_ticks_spec() -> tuple:
    """Name, loaders, arguments and fetcher of ``fetch_closing_ticks``' season
    fetches with its default arguments, as the odds repair calls it."""
    import inspect

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(fetch_closing_ticks).parameters.items()
    }
    kwargs = {
        "last_n": defaults["last_n"],
        "markets": defaults["markets"],
        "min_minutes_before_tip": defaults["min_minutes_before_tip"],
    }
    return (
        "closing_ticks",
        (_fetch_closing_ticks_once,),
        kwargs,
        lambda **chunk: _fetch_closing_ticks_once(**chunk, **kwargs),
    )


def _pregame_ticks_spec(exclude_books: tuple[str, ...] = HISTORY_ONLY_BOOKS) -> tuple:
    """``_closing_ticks_spec`` for ``fetch_pregame_ticks``' season fetches, with
    the books the intermediate builder excludes by default."""
    import inspect

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(fetch_pregame_ticks).parameters.items()
    }
    kwargs = {
        "markets": defaults["markets"],
        "exclude_books": tuple(exclude_books),
        "min_minutes_before_tip": defaults["min_minutes_before_tip"],
    }
    return (
        "pregame_ticks",
        (_fetch_pregame_ticks_once,),
        kwargs,
        lambda **chunk: _fetch_pregame_ticks_once(chunk["season_years"], **kwargs),
    )


#: What ``refresh_season_ticks`` can store: the odds repair's closing ticks and
#: the intermediate builder's pre-game ticks.
TICK_SPECS = {
    "closing_ticks": _closing_ticks_spec,
    "pregame_ticks": _pregame_ticks_spec,
}


def refresh_season_ticks(
    store,
    seasons: list[int],
    *,
    force: bool = False,
    now=None,
    name: str = "closing_ticks",
    check_only: bool = False,
) -> dict[int, str]:
    """Validate finished seasons of ticks in S3, uploading changed ones.

    ``name`` is a ``TICK_SPECS`` entry. The fingerprint is taken before the
    fetch, so ticks changing mid-fetch leave an entry the next refresh
    replaces. The current season is skipped. ``check_only`` compares the
    fingerprints and writes nothing.
    """
    from nba_ou.postgre_db import season_tables

    name, loaders, kwargs, fetch_one = TICK_SPECS[name]()
    key = _season_ticks_key(loaders, kwargs)
    dataset = _ticks_dataset(name)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = season_tables.current_season_year()
    outcome = {}
    with connect_line_history_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{FINGERPRINT_TIMEOUT}'")
        for season in seasons:
            if season >= current:
                outcome[season] = "current season, not stored"
                continue
            fingerprint = line_history_season_fingerprint(conn, season)
            entry = store.entry(dataset, key, season)
            if check_only:
                outcome[season] = season_tables.check_outcome(entry, fingerprint)
                continue
            if entry is not None and entry.fingerprint == fingerprint and not force:
                store.touch(dataset, key, season, entry, now)
                outcome[season] = "unchanged"
                continue
            frame = _fetch_one_with_retries(
                {"season_years": [season]}, fetch_one, _chunk_label
            )
            written = store.write(dataset, key, season, frame, fingerprint, now)
            outcome[season] = f"uploaded {written.rows:,} rows"
    return outcome


def _chunk_label(chunk: dict) -> str:
    if "season_years" in chunk:
        return f"season {chunk['season_years'][0]}"
    return f"{len(chunk['game_ids'])} game(s)"


def _fetch_closing_ticks_once(
    *,
    season_years: list[int] | None = None,
    game_ids: list[str] | None = None,
    last_n: int | None = 5,
    markets: tuple[str, ...] = ALL_MARKETS,
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """Priced pre-game ticks of finished games, for reconstructing the close.

    Differs from ``fetch_pregame_ticks`` in three ways, all aimed at the closing
    line rather than the snapshot grid:

    * Only the ``last_n`` most recent ticks per (game, market, book) are
      returned; the close needs the tail, not the whole path. ``None`` returns
      every tick. Ticks without both prices are skipped in SQL, so the tail is
      not wasted on off-board rows.
    * ``mins_to_tip`` must agree with ``tipoff_utc - line_ts`` to the minute,
      and both clocks must clear the safety margin. Measured 2026-09-17, the
      two agree exactly for every book except BetRivers (8,262 ticks, off by up
      to 65 days) and Hard Rock Bet; a tick whose clocks disagree cannot be
      placed before tip, so it is not trusted.
    * Games whose tip is still in the future are excluded: their last tick is
      a current line, not a close.
    """
    if season_years is None and game_ids is None:
        raise ValueError("Pass season_years or game_ids.")
    if last_n is not None and last_n < 1:
        raise ValueError("last_n must be >= 1 or None.")
    if min_minutes_before_tip < 0:
        raise ValueError("min_minutes_before_tip must be >= 0.")

    filters = []
    params: list = []
    if season_years is not None:
        filters.append(sql.SQL("l.season_year = ANY(%s)"))
        params.append([int(s) for s in season_years])
    if game_ids is not None:
        filters.append(sql.SQL("l.game_id = ANY(%s)"))
        params.append([str(g) for g in game_ids])

    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
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
        ).format(
            schema=sql.Identifier(SCHEMA),
            filters=sql.SQL(" OR ").join(filters),
        )
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
    finally:
        if owned:
            conn.close()

    return _decode_ticks(ticks)


def available_seasons(*, conn: psycopg.Connection | None = None) -> list[int]:
    """Seasons actually present in the store.

    2019-20 and 2020-21 are deliberately absent (Phase 0 could not pin their
    timezone), so this is the honest source for "what can we train on" rather
    than any hardcoded range.
    """
    owned = conn is None
    conn = conn or connect_line_history_db()
    try:
        query = sql.SQL("SELECT DISTINCT season_year FROM {}.lh_game").format(
            sql.Identifier(SCHEMA)
        )
        with conn.cursor() as cur:
            cur.execute(query)
            return sorted(int(row[0]) for row in cur.fetchall())
    finally:
        if owned:
            conn.close()
