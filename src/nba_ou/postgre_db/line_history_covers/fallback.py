"""Covers as SBR's fallback in the stored line-history reads.

``line_history_aiven.fetch``'s readers (pre-game ticks, closing ticks, games)
add the Covers store's games where SBR's history is missing or cut short. The
choice is per game, and a game is read whole from one source:

* SBR has no tick of the game (the 2024-03-29 slate, 2019-04..06, the
  2026-10-23 freeze) -> Covers;
* Covers' latest tick is more than ``MARGIN_MINUTES`` later than SBR's (SBR
  froze before the close: 2024-03-28) -> Covers;
* otherwise SBR. Both serve the same feed, so on a sound game they agree.

Books only Covers carries (``COVERS_ONLY_BOOKS``) are dropped, so a game reads
the same books whichever source it comes from.

The season fingerprints that key the caches built on these reads get the
Covers season's fingerprint appended -- only when Covers holds a game of that
season, so a season it does not touch keeps its keys. A failed Covers read
leaves SBR's frame as it is and says so.
"""

from __future__ import annotations

import pandas as pd
import psycopg

from . import schema as schema_mod
from . import store

#: How much later Covers' last tick must be for a game SBR has to come from it.
MARGIN_MINUTES = 30

#: Covers books the SBR store does not carry (``books.COVERS_BOOKS``).
COVERS_ONLY_BOOKS: tuple[str, ...] = ("thescore", "consensus_opener")

TICK_ORDER = ["game_id", "market", "book", "line_ts"]


def fallback_game_ids(
    primary: pd.DataFrame, fallback: pd.DataFrame, *, margin_minutes=MARGIN_MINUTES
) -> list[str]:
    """Games to read from ``fallback``: absent from ``primary``, or whose last
    tick there is more than ``margin_minutes`` later. Pure."""
    if fallback.empty:
        return []
    latest_fallback = fallback.groupby(fallback["game_id"].astype(str))["line_ts"].max()
    if primary.empty:
        return sorted(latest_fallback.index)
    latest_primary = primary.groupby(primary["game_id"].astype(str))["line_ts"].max()
    joined = latest_primary.reindex(latest_fallback.index)
    margin = pd.Timedelta(minutes=margin_minutes)
    take = joined.isna() | (latest_fallback > joined + margin)
    return sorted(latest_fallback.index[take])


def merge_by_game(
    primary: pd.DataFrame, fallback: pd.DataFrame, game_ids: list[str], order
) -> pd.DataFrame:
    """``primary`` with ``game_ids`` replaced by ``fallback``'s rows. Pure."""
    if not game_ids:
        return primary
    ids = set(game_ids)
    parts = [
        frame
        for frame in (
            primary[~primary["game_id"].astype(str).isin(ids)],
            fallback[fallback["game_id"].astype(str).isin(ids)],
        )
        if not frame.empty
    ]
    merged = pd.concat(parts, ignore_index=True) if parts else primary
    merged = merged.sort_values(order, kind="stable").reset_index(drop=True)
    merged.attrs = dict(primary.attrs)
    return merged


def apply(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """SBR's ticks with the games Covers should supply. ``attrs`` keeps SBR's
    and gains ``covers_game_ids``."""
    fallback = fallback[~fallback["book"].isin(COVERS_ONLY_BOOKS)]
    game_ids = fallback_game_ids(primary, fallback)
    out = merge_by_game(primary, fallback, game_ids, TICK_ORDER)
    out.attrs["covers_game_ids"] = game_ids
    return out


def _with_conn(conn: psycopg.Connection | None, read):
    if conn is not None:
        try:
            return read(conn)
        except Exception:
            conn.rollback()  # the caller's connection stays usable
            raise
    # The SBR readers' connection (same database): one patched in tests, or
    # swapped in an environment, covers both stores.
    from nba_ou.postgre_db.line_history_aiven import fetch as sbr_fetch

    with sbr_fetch.connect_line_history_db() as owned:
        return read(owned)


def _guarded(label: str, primary: pd.DataFrame, read) -> pd.DataFrame:
    """``read()`` or, when Covers cannot be read, ``primary`` with a warning."""
    try:
        return read()
    except Exception as exc:  # noqa: BLE001 -- SBR alone is still a valid read
        print(f"! {label}: Covers fallback unavailable ({type(exc).__name__}: {exc})")
        return primary


def _report(label: str, ticks: pd.DataFrame) -> pd.DataFrame:
    ids = ticks.attrs.get("covers_game_ids") or []
    if ids:
        print(f"  {label}: {len(ids)} game(s) from Covers")
    return ticks


def _candidate_game_ids(
    conn: psycopg.Connection,
    primary: pd.DataFrame,
    *,
    season_years=None,
    game_ids=None,
    markets: tuple[str, ...],
    min_minutes_before_tip: int,
    exclude_books: tuple[str, ...] = (),
) -> list[str]:
    """The games Covers should supply, decided on each game's latest Covers
    tick (one aggregate row a game), so only those games' ticks are moved."""
    latest = store.latest_pregame_ticks(
        conn,
        season_years=season_years,
        game_ids=game_ids,
        markets=markets,
        min_minutes_before_tip=min_minutes_before_tip,
        exclude_books=(*exclude_books, *COVERS_ONLY_BOOKS),
    )
    return fallback_game_ids(primary, latest)


def pregame_ticks(
    primary: pd.DataFrame,
    season_years: list[int],
    *,
    markets: tuple[str, ...],
    exclude_books: tuple[str, ...],
    min_minutes_before_tip: int,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """``fetch_pregame_ticks``' frame with Covers' games where SBR falls short."""

    def read() -> pd.DataFrame:
        def query(c):
            if not schema_mod.schema_exists(c):
                return primary
            ids = _candidate_game_ids(
                c,
                primary,
                season_years=season_years,
                markets=markets,
                min_minutes_before_tip=min_minutes_before_tip,
                exclude_books=exclude_books,
            )
            if not ids:
                return primary
            covers = store.fetch_pregame_ticks(
                season_years,
                c,
                markets=markets,
                min_minutes_before_tip=min_minutes_before_tip,
                game_ids=ids,
            )
            covers = covers[~covers["book"].isin(exclude_books)]
            return apply(primary, covers)

        return _report("pregame ticks", _with_conn(conn, query))

    return _guarded("pregame ticks", primary, read)


def closing_ticks(
    primary: pd.DataFrame,
    *,
    season_years: list[int] | None,
    game_ids: list[str] | None,
    last_n: int | None,
    markets: tuple[str, ...],
    min_minutes_before_tip: int,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """``fetch_closing_ticks``' frame with Covers' games where SBR falls short.

    A season's fingerprint in ``attrs["season_fingerprints"]`` gets the Covers
    season's appended when Covers holds games of it, so the odds repair's memo
    (keyed on those) sees a Covers change.
    """

    def read() -> pd.DataFrame:
        def query(c):
            if not schema_mod.schema_exists(c):
                return primary
            ids = _candidate_game_ids(
                c,
                primary,
                season_years=season_years,
                game_ids=game_ids,
                markets=markets,
                min_minutes_before_tip=min_minutes_before_tip,
            )
            covers = (
                store.fetch_closing_ticks(
                    c,
                    game_ids=ids,
                    last_n=last_n,
                    markets=markets,
                    min_minutes_before_tip=min_minutes_before_tip,
                )
                if ids
                else pd.DataFrame()
            )
            out = apply(primary, covers) if not covers.empty else primary
            fingerprints = primary.attrs.get("season_fingerprints")
            if fingerprints:
                out.attrs["season_fingerprints"] = extend_fingerprints(fingerprints, c)
            return out

        return _report("closing ticks", _with_conn(conn, query))

    return _guarded("closing ticks", primary, read)


def games(
    primary: pd.DataFrame,
    season_years: list[int],
    *,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """``fetch_games``' frame plus the Covers games SBR does not hold."""

    def read() -> pd.DataFrame:
        def query(c):
            if not schema_mod.schema_exists(c):
                return primary
            covers = store.fetch_games(season_years, c)
            if covers.empty:
                return primary
            known = set(primary["game_id"].astype(str)) if not primary.empty else set()
            extra = covers[~covers["game_id"].astype(str).isin(known)]
            return merge_by_game(
                primary,
                extra,
                sorted(extra["game_id"].astype(str)),
                ["game_date", "game_id"],
            )

        return _with_conn(conn, query)

    return _guarded("games", primary, read)


def extend_fingerprints(fingerprints: dict, conn: psycopg.Connection) -> dict:
    """SBR season fingerprints with the Covers season's appended where it holds
    games. A season Covers does not hold keeps its fingerprint unchanged."""
    if not schema_mod.schema_exists(conn):
        return dict(fingerprints)
    out = {}
    for season, fingerprint in fingerprints.items():
        covers = store.season_fingerprint(conn, int(season))
        out[season] = (
            fingerprint if covers is None else [*fingerprint, "covers", *covers]
        )
    return out
