"""Keep stored line-history clocks on the real tipoff.

``mins_to_tip`` and ``is_pregame`` are only as good as ``lh_game.tipoff_utc``.
Two things broke that, both found 2026-09-17:

* The bulk load took tipoffs from the NBA season-schedule feed, which keeps the
  *originally scheduled* time. Games the league moved -- flexed TV slots, the
  January 2026 storm, postponements weeks later -- kept a wrong tipoff, and in
  the worst case (``0022501111``, stored three weeks after it was played) the
  live ticks of the real game were labelled pre-game.
* Incremental ingest timed each tick against the SBR page's tipoff but never
  moved a tipoff already stored, so backfilled books (BetRivers, Hard Rock)
  disagreed with every other book of the same game.

The NBA daily scoreboard (``ScoreboardV3.gameTimeUTC``) carries the tipoff as
played, and on every one of the 66 disagreements checked it matched the SBR
page to the minute. It is the reference here.

A retime rewrites ``lh_game.tipoff_utc`` and recomputes ``mins_to_tip`` /
``is_pregame`` from ``line_ts`` for *every* book of the game, so no two books of
one game can disagree about when it started.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import (
    fetch_tipoffs_for_dates,
    tipoffs_from_cache,
)
from nba_ou.fetch_data.nba_schedule.tipoff_corrections import TIPOFF_CORRECTIONS

from .schema import SCHEMA

REPORT_COLUMNS = [
    "game_id",
    "status",
    "stored_tipoff_utc",
    "tipoff_utc",
    "shift_minutes",
    "pregame_ticks_before",
    "pregame_ticks_after",
    "ticks_with_new_clock",
]

_MINUTES_FROM_TIP = "ROUND(EXTRACT(EPOCH FROM (line_ts - %s)) / 60.0)"


def stored_tipoffs(
    conn: psycopg.Connection, season_years: list[int] | None = None
) -> pd.DataFrame:
    """``lh_game`` rows: game_id, season_year, game_date, tipoff_utc."""
    where = sql.SQL("")
    params: tuple = ()
    if season_years is not None:
        where = sql.SQL("WHERE season_year = ANY(%s)")
        params = ([int(s) for s in season_years],)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, season_year, game_date, tipoff_utc FROM {}.lh_game {}"
            ).format(sql.Identifier(SCHEMA), where),
            params,
        )
        frame = pd.DataFrame(
            cur.fetchall(), columns=["game_id", "season_year", "game_date", "tipoff_utc"]
        )
    frame["game_id"] = frame["game_id"].astype(str)
    frame["tipoff_utc"] = pd.to_datetime(frame["tipoff_utc"], utc=True)
    return frame


def scoreboard_tipoffs(
    game_dates: pd.DataFrame, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """Tipoff as played, per game, from the daily scoreboard.

    ``game_dates`` needs ``game_id`` and ``game_date`` -- the date the game was
    *played* (``nba_games``), since a postponed game is only on the scoreboard
    of its new date. With ``cache_dir`` the scoreboards are read from the cache
    filled by ``cache_scoreboards`` instead of being requested.
    """
    dates = sorted({str(pd.Timestamp(d).date()) for d in game_dates["game_date"]})
    board = (
        tipoffs_from_cache(dates, cache_dir)
        if cache_dir is not None
        else fetch_tipoffs_for_dates(dates)
    )
    board["game_id"] = board["game_id"].astype(str)
    board["tipoff_utc"] = pd.to_datetime(board["tipoff_utc"], utc=True)
    wanted = set(game_dates["game_id"].astype(str))
    return (
        board[board["game_id"].isin(wanted)]
        .drop_duplicates("game_id")[["game_id", "tipoff_utc"]]
        .reset_index(drop=True)
    )


SBR_MAX_WORKERS = 4


def _sbr_cache_path(cache_dir: Path, day: str) -> Path:
    return cache_dir / f"sbr_daily_{day}.json"


def sbr_tipoffs(
    game_dates: pd.DataFrame,
    games_df: pd.DataFrame,
    cache_dir: str | Path,
    *,
    max_workers: int = SBR_MAX_WORKERS,
) -> tuple[pd.DataFrame, list[str]]:
    """Tipoff per game from SBR's daily odds pages, plus the dates that failed.

    The fallback when stats.nba.com is unreachable. The page's ``startDate`` is
    the same clock the line-history ticks were scraped against, and it matched
    the NBA scoreboard on all 66 moved games checked on 2026-09-17. A postponed
    game is listed on the date it was played, which is why ``game_date`` must be
    the played date from ``nba_games``.

    One page per date, a few in flight at once; each date is cached as soon as
    it arrives, so a rerun only fetches what is still missing.
    """
    from concurrent.futures import ThreadPoolExecutor
    import json
    import threading

    from tqdm import tqdm

    from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
        discover_games_for_date,
        new_session,
    )

    from .ingest import build_game_index, build_game_lookup, resolve_game_id

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = sorted({str(pd.Timestamp(d).date()) for d in game_dates["game_date"]})
    pending = [d for d in days if not _sbr_cache_path(cache_dir, d).exists()]
    if len(pending) < len(days):
        print(f"SBR cache: {len(days) - len(pending)}/{len(days)} date(s) already cached")

    local = threading.local()

    def fetch(day: str) -> str | None:
        if not hasattr(local, "session"):
            local.session = new_session()
        try:
            summaries = discover_games_for_date(
                local.session, pd.Timestamp(day).date(), timeout=20, retries=2
            )
        except Exception as exc:  # noqa: BLE001 - reported, retried next run
            return f"{day}: {type(exc).__name__}"
        records = [
            {
                "event_id": g.event_id,
                "game_date": g.game_date.isoformat(),
                "tipoff_utc": g.tipoff_utc.isoformat(),
                "team_away": g.team_away,
                "team_home": g.team_home,
            }
            for g in summaries
        ]
        path = _sbr_cache_path(cache_dir, day)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records))
        tmp.replace(path)
        return None

    failed: list[str] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            for error in tqdm(
                pool.map(fetch, pending), total=len(pending), desc="SBR pages", unit="date"
            ):
                if error:
                    failed.append(error)

    lookup = build_game_lookup(build_game_index(games_df))
    rows = []
    for day in days:
        path = _sbr_cache_path(cache_dir, day)
        if not path.exists():
            continue
        for record in json.loads(path.read_text()):
            hit = resolve_game_id(
                lookup,
                game_date=pd.Timestamp(record["game_date"]).date(),
                team_away=record["team_away"],
                team_home=record["team_home"],
            )
            if hit is not None:
                rows.append({"game_id": str(hit[0]), "tipoff_utc": record["tipoff_utc"]})

    reference = pd.DataFrame(rows, columns=["game_id", "tipoff_utc"])
    reference["tipoff_utc"] = pd.to_datetime(reference["tipoff_utc"], utc=True)
    wanted = set(game_dates["game_id"].astype(str))
    reference = reference[reference["game_id"].isin(wanted)].drop_duplicates("game_id")
    return reference.reset_index(drop=True), failed


#: Shifts this small are scoreboard noise, not moved games: the only ones seen
#: in 2019-20..2024-25 are +1 minute on season opening nights.
TIPOFF_SHIFT_TOLERANCE_MINUTES = 2.0


def find_tipoff_mismatches(stored: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Games whose stored tipoff differs from ``reference``: game_id, both tips."""
    merged = stored.merge(
        reference.rename(columns={"tipoff_utc": "reference_tipoff_utc"}),
        on="game_id",
        how="inner",
    )
    shift = (
        merged["reference_tipoff_utc"] - merged["tipoff_utc"]
    ).dt.total_seconds() / 60.0
    out = merged.assign(shift_minutes=shift)[
        shift.abs().gt(TIPOFF_SHIFT_TOLERANCE_MINUTES)
    ]
    return out[
        ["game_id", "season_year", "tipoff_utc", "reference_tipoff_utc", "shift_minutes"]
    ].reset_index(drop=True)


def games_with_inconsistent_clocks(
    conn: psycopg.Connection, season_years: list[int] | None = None
) -> list[str]:
    """Games holding a tick whose ``mins_to_tip`` disagrees with its own timestamp."""
    where = sql.SQL("")
    params: tuple = ()
    if season_years is not None:
        where = sql.SQL("AND l.season_year = ANY(%s)")
        params = ([int(s) for s in season_years],)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT l.game_id
                FROM {schema}.lh_line l
                JOIN {schema}.lh_game g USING (game_id)
                WHERE ABS(
                    EXTRACT(EPOCH FROM (l.line_ts - g.tipoff_utc)) / 60.0
                    - l.mins_to_tip
                ) > 1 {where}
                """
            ).format(schema=sql.Identifier(SCHEMA), where=where),
            params,
        )
        return sorted(str(row[0]) for row in cur.fetchall())


def retime_games(
    conn: psycopg.Connection,
    tipoffs: Mapping[str, pd.Timestamp],
    *,
    dry_run: bool = False,
    commit: bool = True,
) -> pd.DataFrame:
    """Set each game's tipoff and recompute every tick's clock against it.

    Passing a game's current tipoff is valid: it realigns ticks that were timed
    against a different one without moving the game.
    """
    report = []
    schema = sql.Identifier(SCHEMA)
    with conn.cursor() as cur:
        for game_id, tipoff in tipoffs.items():
            tipoff = pd.Timestamp(tipoff).tz_convert("UTC")
            cur.execute(
                sql.SQL(
                    "SELECT tipoff_utc, season_year FROM {}.lh_game WHERE game_id = %s"
                ).format(schema),
                (str(game_id),),
            )
            row = cur.fetchone()
            if row is None:
                report.append({"game_id": str(game_id), "status": "not in lh_game"})
                continue
            stored_tipoff, season_year = row
            stored_tipoff = pd.Timestamp(stored_tipoff).tz_convert("UTC")
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FILTER (WHERE is_pregame), "
                    f"COUNT(*) FILTER (WHERE {_MINUTES_FROM_TIP} < 0), "
                    f"COUNT(*) FILTER (WHERE mins_to_tip <> {_MINUTES_FROM_TIP}) "
                    "FROM {}.lh_line WHERE game_id = %s AND season_year = %s"
                ).format(schema),
                (tipoff, tipoff, str(game_id), season_year),
            )
            pregame_before, pregame_after, clock_changes = cur.fetchone()
            report.append(
                {
                    "game_id": str(game_id),
                    "status": "dry run" if dry_run else "retimed",
                    "stored_tipoff_utc": stored_tipoff,
                    "tipoff_utc": tipoff,
                    "shift_minutes": (tipoff - stored_tipoff).total_seconds() / 60.0,
                    "pregame_ticks_before": pregame_before,
                    "pregame_ticks_after": pregame_after,
                    "ticks_with_new_clock": clock_changes,
                }
            )
            if dry_run:
                continue
            if tipoff != stored_tipoff:
                cur.execute(
                    sql.SQL(
                        "UPDATE {}.lh_game SET tipoff_utc = %s WHERE game_id = %s"
                    ).format(schema),
                    (tipoff, str(game_id)),
                )
            if clock_changes or pregame_before != pregame_after:
                cur.execute(
                    sql.SQL(
                        f"UPDATE {{}}.lh_line SET mins_to_tip = {_MINUTES_FROM_TIP}::integer, "
                        f"is_pregame = {_MINUTES_FROM_TIP} < 0 "
                        "WHERE game_id = %s AND season_year = %s"
                    ).format(schema),
                    (tipoff, tipoff, str(game_id), season_year),
                )
    if dry_run:
        conn.rollback()
    elif commit:
        conn.commit()
    return pd.DataFrame(report, columns=REPORT_COLUMNS)


def correct_stored_tipoffs(
    conn: psycopg.Connection, *, dry_run: bool = False
) -> pd.DataFrame:
    """Apply ``TIPOFF_CORRECTIONS`` to the stored games."""
    return retime_games(
        conn,
        {game_id: c.tipoff_utc for game_id, c in TIPOFF_CORRECTIONS.items()},
        dry_run=dry_run,
    )
