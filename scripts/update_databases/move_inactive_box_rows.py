"""Move the inactive-player rows stored in nba_players to nba_injuries.

From 2026-27 the NBA's box score lists inactive players as ``Inactive - ...``
rows and the summary's inactive list is empty. Ingestion now splits them out
(``nba_ou.fetch_data.fetch_nba_data.inactive_rows``); this moves the ones
stored before that. See ``nba_ou.postgre_db.injuries_refs.inactive_backfill``.

Examples::

    # current season, what would move (writes nothing)
    python scripts/update_databases/move_inactive_box_rows.py

    # move them
    python scripts/update_databases/move_inactive_box_rows.py --execute
"""

from __future__ import annotations

import argparse

from nba_ou.postgre_db.injuries_refs.inactive_backfill import (
    already_in_injuries,
    execute_backfill,
    load_backfill,
)
from nba_ou.postgre_db.season_tables import current_season_year


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="season start year (default: the current season)",
    )
    parser.add_argument(
        "--execute", action="store_true", help="write; without it, report only"
    )
    args = parser.parse_args()

    season_year = args.season_year or current_season_year()
    backfill = load_backfill(season_year)
    print(backfill.describe())
    if backfill.injuries.empty:
        return 0
    print(
        f"{len(backfill.injuries)} rows to nba_injuries; "
        f"{already_in_injuries(backfill)} of them are there already."
    )

    if not args.execute:
        print("Dry run: nothing written. Re-run with --execute.")
        return 0

    deleted = execute_backfill(backfill)
    print(f"Moved {deleted} rows from nba_players to nba_injuries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
