#!/usr/bin/env python3
"""Tonight's report states, parsed live, equal the stored ones for a past date.

    python scripts/perf/check_live_injury_reports.py --date 2026-04-10 \
        --as-of 2026-04-10T22:00:00Z

Replays ``date`` as of ``as_of``: the reports published before it (from the
S3 archive) go through ``live.build_live_injury_reports``, and the status,
filing and report-age reads at every snapshot of that date's games already
passed are compared with the same reads over the store, which a whole-season
load wrote. Read-only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

import pandas as pd  # noqa: E402
from nba_ou.data_processing.line_history.snapshots import (  # noqa: E402
    DEFAULT_SNAPSHOT_GRID,
)
from nba_ou.fetch_data.injury_reports.archive import urls as U  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive.manifest import (  # noqa: E402
    ManifestStore,
)
from nba_ou.fetch_data.injury_reports.archive.storage import S3Storage  # noqa: E402
from nba_ou.postgre_db.config.db_config import connect_nba_db  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven import fetch, live  # noqa: E402
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_games  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--manifest-root", default="data/injury_reports/manifest")
    args = parser.parse_args()
    game_date = date.fromisoformat(args.date)
    as_of = pd.Timestamp(args.as_of)
    season = U.season_label(game_date)

    games = fetch_games([int(season[:4])])
    games = games[pd.to_datetime(games["game_date"]).dt.date.eq(game_date)]
    game_ids = sorted(games["game_id"].astype(str))
    cutoffs = pd.DataFrame(
        [
            (g, m, pd.Timestamp(t) - pd.Timedelta(minutes=m))
            for g, t in zip(games["game_id"].astype(str), games["tipoff_utc"])
            for m in DEFAULT_SNAPSHOT_GRID
        ],
        columns=["game_id", "snapshot_minutes", "as_of"],
    )
    cutoffs = cutoffs[cutoffs["as_of"] < as_of].reset_index(drop=True)
    print(f"{len(game_ids)} games, {len(cutoffs)} passed cutoffs")

    storage = S3Storage(
        bucket="adrian-nba-model-registry-eu-west-1",
        profile="adrian-personal",
        region="eu-west-1",
    )
    manifest = ManifestStore(Path(args.manifest_root)).load(season)
    with connect_nba_db(env="supabase") as source:
        tonight = live.build_live_injury_reports(
            manifest,
            storage,
            source,
            game_date=game_date,
            game_ids=game_ids,
            as_of=as_of,
            season=season,
        )
    print(
        f"live: {len(tonight.reports)} reports, {len(tonight.spans)} spans, "
        f"{len(tonight.filing_spans)} filing spans"
    )
    ok = True
    with connect_nba_db("aiven") as conn:
        stored = fetch.report_state_at_snapshots(conn, cutoffs)
        current = live.state_at_snapshots(conn, cutoffs, tonight)
    for name, a, b in zip(("statuses", "filings", "report age"), stored, current):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            print(f"{name}: EQUAL ({len(a)} rows)")
        except AssertionError as exc:
            ok = False
            print(f"{name}: DIFFERENT ({len(a)} stored, {len(b)} live)\n{exc}")
    print("RESULT:", "EQUAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
