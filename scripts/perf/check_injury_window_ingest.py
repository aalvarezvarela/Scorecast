#!/usr/bin/env python3
"""A window ingest builds the stored spans of its games, exactly.

    python scripts/perf/check_injury_window_ingest.py --first 2026-04-09 --last 2026-04-10

Dry-runs ``ingest_season(write_game_dates=(first, last))`` on the reports from
the day before ``first`` to ``last`` -- what ``daily_injury_reports.py`` writes
-- and compares its spans and filing spans with the store's for those games,
which a whole-season load wrote. Read-only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]

import pandas as pd  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import manifest as mf  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import urls as U  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive.manifest import (  # noqa: E402
    ManifestStore,
)
from nba_ou.fetch_data.injury_reports.archive.storage import S3Storage  # noqa: E402
from nba_ou.postgre_db.config.db_config import connect_nba_db  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven.ingest import ingest_season  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven.schema import SCHEMA  # noqa: E402

ET = ZoneInfo("America/New_York")


def _stored(conn, game_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spans = pd.read_sql_query(
        f"""
        SELECT s.game_id, s.player_id::text AS player_id,
               t.nba_team_id::text AS team_id, s.valid_from, s.valid_to,
               s.status_id, rc.code || '|' || r.detail AS reason,
               s.mins_to_tip, s.is_pregame
        FROM {SCHEMA}.ir_status_span s
        JOIN {SCHEMA}.ir_team t ON t.team_id = s.team_id
        JOIN {SCHEMA}.ir_reason r ON r.reason_id = s.reason_id
        JOIN {SCHEMA}.ir_reason_category rc ON rc.category_id = r.category_id
        WHERE s.game_id = ANY(%(ids)s)
        """,
        conn,
        params={"ids": game_ids},
    )
    filings = pd.read_sql_query(
        f"""
        SELECT f.game_id, t.nba_team_id::text AS team_id, f.valid_from,
               f.valid_to, f.submitted
        FROM {SCHEMA}.ir_filing_span f
        JOIN {SCHEMA}.ir_team t ON t.team_id = f.team_id
        WHERE f.game_id = ANY(%(ids)s)
        """,
        conn,
        params={"ids": game_ids},
    )
    return spans, filings


def _normalise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("valid_from", "valid_to"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("game_id", "player_id", "team_id"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    if "status_id" in frame:
        frame["status_id"] = pd.to_numeric(frame["status_id"]).astype("Float64")
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first", required=True)
    parser.add_argument("--last", required=True)
    parser.add_argument("--manifest-root", default="data/injury_reports/manifest")
    args = parser.parse_args()
    first, last = date.fromisoformat(args.first), date.fromisoformat(args.last)
    report_start = first - timedelta(days=1)

    store = ManifestStore(Path(args.manifest_root))
    storage = S3Storage(
        bucket="adrian-nba-model-registry-eu-west-1",
        profile="adrian-personal",
        region="eu-west-1",
    )
    season = U.season_label(first)
    manifest = store.load(season)
    published = mf.published_utc(manifest).dt.tz_convert(ET).dt.date
    window = manifest[(published >= report_start) & (published <= last)]
    collected: dict = {}
    with connect_nba_db(env="supabase") as source:
        summary = ingest_season(
            season,
            manifest=window,
            storage=storage,
            aiven=None,
            source=source,
            dry_run=True,
            quiet=True,
            write_game_dates=(first, last),
            collect=collected,
        )
    print(summary.describe())
    spans = collected["spans"].assign(
        player_id=lambda f: f["player_id"].astype("Int64").astype(str),
        team_id=lambda f: f["team_id"].astype("Int64").astype(str),
        reason=lambda f: f["reason_id"].astype(str),
    )
    filings = collected["filing_spans"].assign(
        team_id=lambda f: f["team_id"].astype("Int64").astype(str)
    )
    game_ids = sorted(set(spans["game_id"].astype(str)) | set(filings["game_id"]))
    with connect_nba_db("aiven") as conn:
        stored_spans, stored_filings = _stored(conn, game_ids)
    span_keys = ["game_id", "player_id", "valid_from"]
    filing_keys = ["game_id", "team_id", "valid_from"]
    span_columns = list(stored_spans.columns)
    ok = True
    for name, built, stored, keys, columns in (
        ("spans", spans, stored_spans, span_keys, span_columns),
        ("filing spans", filings, stored_filings, filing_keys, list(stored_filings)),
    ):
        built = _normalise(built[columns], keys)
        stored = _normalise(stored[columns], keys)
        joined = built.merge(stored, on=columns, how="outer", indicator=True)
        only_built = joined[joined["_merge"].eq("left_only")]
        only_stored = joined[joined["_merge"].eq("right_only")]
        same = only_built.empty and only_stored.empty and len(built) == len(stored)
        ok &= same
        print(
            f"--- {name}: {len(built)} built, {len(stored)} stored, "
            f"{len(only_built)} only built, {len(only_stored)} only stored"
        )
        if not same:
            print(only_built.head(10).to_string(index=False))
            print(only_stored.head(10).to_string(index=False))
    print("RESULT:", "EQUAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
