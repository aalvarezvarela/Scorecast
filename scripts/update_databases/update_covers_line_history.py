"""Daily update of the Covers line-history store in Aiven.

A line-history source beside SBR (``update_line_history_database.py``): SBR's
feed for the nine US books, in the SBR store's tables in the schema
``line_history_covers``, plus SBR's ``consensus_opener`` rebuilt from BetMGM's
opener. From the 2026-27 season on. Run it after the games update, so
``nba_games`` knows last night's games.

Each run fetches every finished game the store does not hold yet, and again
the games of the last ``--retry-days`` days missing a book. Raw answers are
archived to S3 (insert-only) before they are parsed. Writes are insert-only.

Exits 1 when a request or a parse failed, so a source outage shows on the
workflow step (``continue-on-error`` keeps the job going); 0 otherwise, even
when games were left for the next run.

Examples::

    # first run: create the schema, then what the daily job does
    python scripts/update_databases/update_covers_line_history.py --create-schema

    # plan and parse without writing anything (no Aiven or S3 writes)
    python scripts/update_databases/update_covers_line_history.py --dry-run

    # specific dates; with --dry-run any season, written to Parquet for checks
    python scripts/update_databases/update_covers_line_history.py \
        --dry-run --dates 2026-01-15 --save-dir /tmp/covers
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.fetch_data.odds_covers.archive import make_archive
from nba_ou.fetch_data.odds_covers.client import CoversClient
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_covers import schema as schema_mod
from nba_ou.postgre_db.line_history_covers import store
from nba_ou.postgre_db.line_history_covers import update as update_mod
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)
from nba_ou.utils.ci import annotate


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _requested_dates(args) -> list[date] | None:
    if args.dates:
        return sorted(set(args.dates))
    if args.start_date or args.end_date:
        start = args.start_date or update_mod.FIRST_DATE
        end = args.end_date or update_mod.today_eastern()
        return [d.date() for d in pd.date_range(start, end, freq="D")]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create-schema",
        action="store_true",
        help="create the schema and its tables first (safe if they exist)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and parse, but write nothing to Aiven or S3",
    )
    parser.add_argument("--dates", nargs="*", type=_parse_date, metavar="YYYY-MM-DD")
    parser.add_argument("--start-date", type=_parse_date)
    parser.add_argument("--end-date", type=_parse_date)
    parser.add_argument(
        "--retry-days",
        type=int,
        default=update_mod.DEFAULT_RETRY_DAYS,
        help="fetch again the games this recent that miss a book "
        f"(default: {update_mod.DEFAULT_RETRY_DAYS})",
    )
    parser.add_argument(
        "--no-archive", action="store_true", help="do not archive raw answers to S3"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="also write the prepared lh_line/lh_game rows here as Parquet",
    )
    args = parser.parse_args()

    dates = _requested_dates(args)
    if dates and not args.dry_run and dates[0] < update_mod.FIRST_DATE:
        print(
            f"error: the store starts on {update_mod.FIRST_DATE}; earlier dates "
            "are only allowed with --dry-run"
        )
        return 2

    today = update_mod.today_eastern()
    seasons = (
        sorted({season_year_for_date(d) for d in dates})
        if dates
        else [season_year_for_date(today)]
    )
    if not dates and seasons[0] < update_mod.FIRST_SEASON:
        print(f"Season {seasons[0]} is before the store's first season; nothing to do.")
        return 0

    print(f"Covers line history, season(s) {seasons}")
    games_df = load_games_for_line_history_creation()
    if games_df.empty:
        print("No games returned from nba_games; nothing to do.")
        return 1
    try:
        schedule = fetch_schedules(seasons)
    except Exception as exc:  # without tipoffs nothing can be stored this run
        annotate("warning", f"Covers: schedule feed unavailable ({exc})")
        schedule = None

    with connect_line_history_db() as conn:
        if args.create_schema and not args.dry_run:
            schema_mod.create_schema(conn)
            print(f"Schema {schema_mod.SCHEMA} ready.")
        exists = schema_mod.schema_exists(conn)
        if not exists and not args.dry_run:
            print(
                f"error: schema {schema_mod.SCHEMA} does not exist; "
                "run once with --create-schema"
            )
            return 2

        plan: dict = {}
        for season_year in seasons:
            stored = store.stored_game_books(conn, season_year) if exists else {}
            games = update_mod.season_games(games_df, season_year)
            if dates:
                season_plan = update_mod.plan_requests(
                    games[games["game_date"].isin(dates)],
                    stored,
                    today=max(today, dates[-1]),
                    retry_days=10**5,
                    first_date=dates[0],
                )
            else:
                season_plan = update_mod.plan_requests(
                    games, stored, today=today, retry_days=args.retry_days
                )
            plan.update(season_plan)

        result = update_mod.UpdateResult()
        if plan:
            print(
                f"Plan: {sum(len(g) for g in plan.values())} game(s) on "
                f"{len(plan)} date(s)"
            )
            archive = None
            if not args.dry_run and not args.no_archive:
                try:
                    archive = make_archive()
                except Exception as exc:
                    annotate("warning", f"Covers: no S3 archive ({exc})")
            with CoversClient() as client:
                result = update_mod.update_covers_line_history(
                    plan,
                    games_df=games_df,
                    schedule=schedule,
                    conn=None if args.dry_run else conn,
                    client=client,
                    archive=archive,
                    dry_run=args.dry_run,
                    # Flushed per line, so the workflow log follows the run live.
                    progress=lambda line: print(line, flush=True),
                )
                requests_made = client.requests_made
            if result.inserted_ticks or result.inserted_games:
                for season_year in seasons:
                    store.record_load(
                        conn,
                        season_year=season_year,
                        source_rows=result.prepared_ticks,
                        loaded_rows=result.inserted_ticks,
                        dropped=result.dropped,
                    )
        else:
            requests_made = 0
            print("Nothing to fetch.")

    _report(result, requests_made, args)
    return 1 if result.failed else 0


def _report(result: update_mod.UpdateResult, requests_made: int, args) -> None:
    print(
        f"\nFetched {result.fetched_games} game(s) in {requests_made} request(s); "
        f"prepared {result.prepared_ticks} tick(s)"
    )
    if args.dry_run:
        print("Dry run: nothing written.")
    else:
        print(
            f"Inserted {result.inserted_games} game(s) and "
            f"{result.inserted_ticks} tick(s)"
        )
    if result.dropped:
        print(f"Dropped: {result.dropped}")
    if result.repaired:
        print(f"Repaired: {result.repaired}")
    if result.unsettled_games:
        print(f"Not finished yet, left for the next run: {len(result.unsettled_games)}")
    if result.held_back:
        # Normal for a day or two after a feed freeze; past the hold it is stored.
        print(
            f"Truncated history held back, left for the next run: "
            f"{len(result.held_back)}: " + "; ".join(result.held_back[:10])
        )
    for label, items in (
        ("game(s) not on the matchups page", result.unlisted_games),
        ("game(s) not matched to nba_games", result.unmatched_games),
        (
            "game(s) without a schedule-feed tipoff, left for the next run",
            result.feed_tipoff_missing,
        ),
        ("game(s) without any line history, left for the next run", result.empty_games),
        ("request(s) or parse(s) failed", result.failed),
    ):
        if items:
            annotate(
                "warning", f"Covers: {len(items)} {label}: " + "; ".join(items[:10])
            )
    if result.unknown_books:
        # A book Covers added: worth mapping in books.COVERS_BOOKS.
        annotate(
            "warning",
            "Covers: rows of unmapped book id(s) skipped: "
            + ", ".join(str(b) for b in sorted(result.unknown_books)),
        )
    if result.archive_failures:
        annotate(
            "warning",
            f"Covers: {result.archive_failures} S3 archive write(s) failed",
        )

    if args.save_dir and result.frames:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        rows = pd.concat([f[0] for f in result.frames], ignore_index=True)
        games = pd.concat([f[1] for f in result.frames], ignore_index=True)
        rows.to_parquet(args.save_dir / "lh_line.parquet", index=False)
        games.to_parquet(args.save_dir / "lh_game.parquet", index=False)
        print(f"Saved {len(rows)} row(s) and {len(games)} game(s) to {args.save_dir}")


if __name__ == "__main__":
    sys.exit(main())
