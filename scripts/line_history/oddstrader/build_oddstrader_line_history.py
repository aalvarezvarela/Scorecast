"""Rebuild the OddsTrader ``lh_*`` Parquet tables from the raw S3 archive.

Reads ``train_data/odds_sources/oddstrader/raw/v1/`` and writes
``train_data/odds_sources/oddstrader/line_history/`` with the tables, columns and
encodings of the Aiven ``line_history`` schema, plus ``manifest.json``. Game
ids are resolved against ``nba_games`` (read only). Deterministic: rerunning
reproduces the tables.

Usage:
    python scripts/line_history/oddstrader/build_oddstrader_line_history.py
    python scripts/line_history/oddstrader/build_oddstrader_line_history.py --seasons 2023
    python scripts/line_history/oddstrader/build_oddstrader_line_history.py --dry-run

``--seasons`` rewrites only those seasons' ``lh_line_<season>`` files, but
``lh_game``, ``lh_book``, ``lh_load_meta`` and the manifest then describe only
those seasons too; run without it for the full archive.
"""

from __future__ import annotations

import argparse
import json
import sys

from nba_ou.fetch_data.odds_oddstrader.build import build_tables
from nba_ou.fetch_data.odds_oddstrader.store import make_store
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", help="season start years")
    parser.add_argument(
        "--dry-run", action="store_true", help="build and report, write nothing"
    )
    args = parser.parse_args()

    games_df = load_games_for_line_history_creation()
    if games_df.empty:
        print("nba_games returned no games.")
        return 1
    manifest = build_tables(
        make_store(), games_df=games_df, season_years=args.seasons, dry_run=args.dry_run
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("seasons", "raw_games", "tables")}, indent=2
        )
    )
    for season, stats in manifest["stats_by_date_season"].items():
        print(
            f"  {season}: {stats['scraped_games']} games, {stats['matched_games']} "
            f"matched, {stats['unmatched_games']} unmatched, "
            f"dropped {stats['dropped']}, repaired {stats['repaired']}"
        )
    if manifest["unreadable_bundles"]:
        print(f"{len(manifest['unreadable_bundles'])} unreadable bundles:")
        for line in manifest["unreadable_bundles"][:20]:
            print(f"  ! {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
