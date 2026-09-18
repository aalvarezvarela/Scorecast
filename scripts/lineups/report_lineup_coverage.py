"""Report archived-response and validated-stint coverage by season."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def coverage(root: Path, *, expected: dict[int, int] | None = None) -> pd.DataFrame:
    rows = []
    manifest_root = root / "nba_api_raw" / "manifest"
    for path in sorted(manifest_root.glob("season=*/manifest.parquet")):
        year = int(path.parent.name.split("=", 1)[1])
        manifest = pd.read_parquet(path)
        total = manifest.game_id.nunique()
        good = manifest.loc[manifest.status.eq("ok")]
        paired = len(
            set(good.loc[good.endpoint.eq("gamerotation"), "game_id"])
            & set(good.loc[good.endpoint.eq("playbyplayv3"), "game_id"])
        )
        status_path = root / "lineup_stints" / f"season={year}" / "game_status.parquet"
        statuses = pd.read_parquet(status_path) if status_path.exists() else pd.DataFrame()
        built = int(statuses.status.eq("ok").sum()) if not statuses.empty else 0
        failed = int(statuses.status.eq("failed").sum()) if not statuses.empty else 0
        reasons = (
            statuses.loc[statuses.status.eq("failed"), "reason"].value_counts().to_dict()
            if not statuses.empty else {}
        )
        expected_games = expected.get(year) if expected is not None else None
        rows.append(dict(
            season_year=year, expected_games=expected_games, games_seen=total,
            raw_complete=paired,
            raw_coverage=(paired / expected_games if expected_games else None),
            stint_ok=built, stint_failed=failed,
            stint_coverage=(built / paired if paired else None),
            stint_pass_rate=(built / (built + failed) if built + failed else None),
            failure_reasons=reasons,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--with-db", action="store_true", help="Include expected finished-game counts")
    args = parser.parse_args()
    expected = None
    if args.with_db:
        from scripts.lineups.backfill_lineup_raw import finished_games

        expected = {}
        for season, _ in finished_games():
            expected[season] = expected.get(season, 0) + 1
    report = coverage(args.local_root, expected=expected)
    print(report.to_string(index=False) if not report.empty else "No lineup archive yet")


if __name__ == "__main__":
    main()
