"""Rewrite stored SBR closing lines from the line history.

SBR day pages show each book's last number, which for many games is a live
line. This replaces every stored per-book close with the last valid pre-tip
quote from the Aiven line history, clears clear cross-book mistakes, keeps the
untouched SBR row in ``odds_sportsbook_sbr_raw`` and an audit row per cell in
``odds_sportsbook_close_repair``. See
``nba_ou.postgre_db.odds_sportsbook.repair_closes``.

Only games not yet repaired are processed, so it is cheap to run daily -- after
the line-history update, which is where last night's ticks come from.

Examples::

    # current season, what would change
    python scripts/update_databases/repair_sportsbook_closes.py --dry-run

    # one-off correction of everything stored
    python scripts/update_databases/repair_sportsbook_closes.py --all-seasons

    # recompute already-repaired games from the raw SBR backup (e.g. after
    # changing thresholds or correcting tipoffs)
    python scripts/update_databases/repair_sportsbook_closes.py --all-seasons --redo
"""

from __future__ import annotations

import argparse

from nba_ou.postgre_db.odds_sportsbook.repair_closes import (
    current_season_year,
    repair_stored_closes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    seasons = parser.add_mutually_exclusive_group()
    seasons.add_argument(
        "--season-year",
        type=int,
        action="append",
        help="season start year; repeatable (default: the current season)",
    )
    seasons.add_argument(
        "--all-seasons", action="store_true", help="every stored season"
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="also recompute games already repaired, starting from the raw SBR rows",
    )
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()

    season_years = (
        None if args.all_seasons else (args.season_year or [current_season_year()])
    )
    print(f"Seasons: {'all' if season_years is None else season_years}")
    summary = repair_stored_closes(
        season_years=season_years, redo=args.redo, dry_run=args.dry_run
    )
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
