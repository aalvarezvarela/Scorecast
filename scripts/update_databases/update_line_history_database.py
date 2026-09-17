"""Daily line-history update, one season at a time.

The counterpart to ``update_all_databases.py`` for the SBR line-history store.
Run it *after* the games update, so ``nba_games`` already knows about last
night's games.

Each run refreshes a rolling window of recent dates (lines move right up to
tipoff, so a game fetched in the morning is present but not final), fills any
game ``nba_games`` has that the store does not, and tops up games holding fewer
books than their own date's best-covered game. Writes are insert-only, so
running it repeatedly is safe and cheap.

Seasons default to the current one, so the scheduled job needs no yearly edit.

Examples::

    # what the daily job does: current season only
    python scripts/update_databases/update_line_history_database.py

    # plan the work without fetching or writing
    python scripts/update_databases/update_line_history_database.py --dry-run

    # past seasons
    python scripts/update_databases/update_line_history_database.py --start 2021 --end 2024

    # add a book the store never held to every stored game (BetRivers exists on
    # SBR from 2021-22, so earlier seasons would only be re-requested for nothing)
    python scripts/update_databases/update_line_history_database.py \
        --start 2021 --end 2025 --backfill-books betrivers --refresh-days 0
"""

from __future__ import annotations

import argparse
import sys

from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import ALL_MARKETS
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven import update as update_mod
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--refresh-days",
        type=int,
        default=update_mod.DEFAULT_REFRESH_DAYS,
        help=(
            "re-fetch dates in the last N days even when their games are "
            "already stored, to pick up line movement toward tipoff "
            f"(default: {update_mod.DEFAULT_REFRESH_DAYS}; 0 disables)"
        ),
    )
    parser.add_argument(
        "--skip-incomplete-check",
        action="store_true",
        help="do not look for games missing a book their date's other games have",
    )
    parser.add_argument(
        "--min-book-share",
        type=float,
        default=update_mod.DEFAULT_EXPECTED_BOOK_SHARE,
        help=(
            "a book counts as expected on a date only once it priced this "
            "share of that date's games. Raise it if a game with a genuine "
            "book absence keeps being re-fetched "
            f"(default: {update_mod.DEFAULT_EXPECTED_BOOK_SHARE})"
        ),
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=list(ALL_MARKETS),
        choices=list(ALL_MARKETS),
        help="markets to store (default: all three)",
    )
    parser.add_argument(
        "--backfill-books",
        nargs="*",
        default=[],
        metavar="SLUG",
        help=(
            "re-fetch every stored game with no tick for these books (e.g. "
            "betrivers). Needed once for a book the store has never held, "
            "since the partial-game check only expects books already stored"
        ),
    )
    parser.add_argument(
        "--flush-every-dates",
        type=int,
        default=update_mod.DEFAULT_FLUSH_EVERY_DATES,
        help=(
            "write scraped games every N dates, so a long backfill keeps its "
            f"progress if it fails (default: {update_mod.DEFAULT_FLUSH_EVERY_DATES}; "
            "0 writes once at the end)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the plan and scrape nothing",
    )
    args = parser.parse_args()

    try:
        seasons = update_mod.resolve_season_years(args.start, args.end)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    print(f"Updating line history for season(s): {seasons}")
    print("Loading games from nba_games ...")
    games_df = load_games_for_line_history_creation()
    if games_df.empty:
        print("No games returned from nba_games; nothing to do.")
        return 1

    try:
        schedule = fetch_schedules(seasons)
    except Exception as exc:  # the feed is only a tipoff cross-check
        print(f"! schedule feed unavailable ({exc}); continuing without it")
        schedule = None

    failures: list[str] = []
    with connect_line_history_db() as conn:
        for season_year in seasons:
            print(f"\n=== Season {season_year}-{str(season_year + 1)[-2:]} ===")
            result = update_mod.update_line_history_database(
                season_year,
                games_df=games_df,
                conn=conn,
                schedule=schedule,
                refresh_days=args.refresh_days,
                include_incomplete=not args.skip_incomplete_check,
                min_book_share=args.min_book_share,
                markets=tuple(args.markets),
                backfill_books=tuple(args.backfill_books),
                flush_every_dates=args.flush_every_dates,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print(
                    f"  dry run: would fetch {len(result.target_dates)} date(s), "
                    "nothing written"
                )
            else:
                print(
                    f"  scraped {result.scraped_games} game(s); inserted "
                    f"{result.inserted_ticks} tick(s) and "
                    f"{result.inserted_games} new game(s)"
                )
                if result.retimed_games:
                    print(
                        f"  moved {len(result.retimed_games)} stored game(s) to "
                        f"the page tipoff: {', '.join(result.retimed_games[:10])}"
                    )
            failures.extend(result.failed_dates)

    if failures:
        # Transient 5xx on one date leaves a gap the next run will re-detect,
        # so this is a warning rather than a failure.
        print(f"\n{len(failures)} date(s) could not be fetched:")
        for item in failures[:10]:
            print(f"  - {item}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
