"""Read-only comparison of bet365 coverage, closes (T-5) and snapshots.

--from-dir consumes the MoviOdds CLI's Parquet output. --sbr-ticks optionally
consumes a local decoded TICK_COLUMNS Parquet instead of reading SBR in Aiven.
Differences are reported, never treated as an error or adjusted away.
"""

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_pregame_ticks
from nba_ou.postgre_db.line_history_moviodds import compare, store


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    parser.add_argument("--from-dir", type=Path)
    parser.add_argument("--sbr-ticks", type=Path)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args(argv)
    if args.from_dir:
        movi = compare.ticks_from_dir(args.from_dir)
    else:
        with connect_line_history_db() as conn:
            conn.read_only = True
            movi = store.fetch_pregame_ticks(args.seasons, conn)
    if movi.empty:
        print("No MoviOdds pregame ticks.")
        return 1
    if args.sbr_ticks:
        sbr = pd.read_parquet(args.sbr_ticks)
    else:
        with connect_line_history_db() as conn:
            conn.read_only = True
            sbr = fetch_pregame_ticks(args.seasons, conn=conn)
    sbr = sbr[(sbr["book"] == "bet365") & sbr["game_id"].isin(movi["game_id"])]
    comparison = compare.snapshot_comparison(movi, sbr)
    summary = compare.summarize(comparison)
    print(
        f"{movi['game_id'].nunique()} MoviOdds games; {sbr['game_id'].nunique()} in SBR"
    )
    print(
        "snapshot_minutes=0: closing with the shared five-minute margin. Ages measure last change, not polling freshness."
    )
    print(summary.to_string(index=False))
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.save_dir / "snapshots.csv", index=False)
        summary.to_csv(args.save_dir / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
