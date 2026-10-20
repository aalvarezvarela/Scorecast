"""
NBA Over/Under Predictor - Database Update Script

Updates NBA games, player game logs, odds and refs/injuries in PostgreSQL by
iterating season-by-season. Uses the existing `update_database` function (which
fetches only missing game_ids).

    python scripts/update_databases/update_all_databases.py                  # current season
    python scripts/update_databases/update_all_databases.py --start 2023 --end 2024

Seasons default to the current one (the season in progress, or out of season the
one just finished), so the daily job needs no yearly edit.
"""

from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update NBA games, player logs, odds and refs/injuries by season."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First season start year (e.g. 2023 for 2023-24). "
        "Defaults to the current season.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last season start year. Defaults to --start.",
    )
    parser.add_argument(
        "--only-new-games",
        action="store_true",
        default=False,
        help="Only update odds and refs/injuries if new games are found",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=False,
        help="Run Playwright in headed mode (default uses config.ini)",
    )

    args = parser.parse_args()
    headless = False if args.headed else True

    update_all_databases(
        start_season_year=args.start,
        end_season_year=args.end,
        only_new_games=args.only_new_games,
        headless=headless,
    )
