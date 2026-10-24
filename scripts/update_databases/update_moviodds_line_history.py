"""Store finished NBA games' bet365 histories in the separate MoviOdds schema.

Unused and unmaintained since 2026-10-24: too slow and unstable for the value
it adds. Automatic updates are disabled; kept for possible future reuse.
See docs/odds/moviodds_line_history.md.

First deployment: --create-schema. Inspect without remote writes: --dry-run
--start-date 2026-10-20 --end-date 2026-10-22 --save-dir /tmp/moviodds.
Explicit dates re-fetch even complete games; existing rows are never replaced.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.fetch_data.odds_moviodds.archive import make_archive
from nba_ou.fetch_data.odds_moviodds.client import MoviOddsClient
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_moviodds import schema, store, update
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (
    load_games_for_line_history_creation,
)
from nba_ou.utils.ci import annotate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-schema", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="no PostgreSQL or S3 writes"
    )
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--retry-days", type=int, default=update.DEFAULT_RETRY_DAYS)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args(argv)
    if args.retry_days < 0:
        parser.error("--retry-days must be non-negative")
    if args.dates and (args.start_date or args.end_date):
        parser.error("use --dates or a date range")
    today = update.today_eastern()
    dates = sorted(set(args.dates)) if args.dates else None
    if args.start_date or args.end_date:
        start, end = args.start_date or update.FIRST_DATE, args.end_date or today
        if start > end:
            parser.error("--start-date must not exceed --end-date")
        dates = [d.date() for d in pd.date_range(start, end)]
    if dates and dates[0] < update.FIRST_DATE:
        parser.error(f"the store starts on {update.FIRST_DATE}")
    # Include every season since launch: absent older games must not disappear
    # from automatic discovery when a new season starts.
    seasons = (
        sorted({season_year_for_date(d) for d in dates})
        if dates
        else list(
            range(
                season_year_for_date(update.FIRST_DATE), season_year_for_date(today) + 1
            )
        )
    )
    games_df = load_games_for_line_history_creation()
    if games_df.empty:
        print("No played games returned from nba_games.")
        return 1
    try:
        schedule = fetch_schedules(seasons)
    except Exception as exc:
        annotate(
            "warning",
            f"MoviOdds: NBA schedule unavailable ({exc}); games without clocks postponed",
        )
        schedule = None
    targets = update.game_targets(games_df, schedule)
    targets = targets[targets["season_year"].isin(seasons)]
    with connect_line_history_db() as conn:
        if args.dry_run:
            conn.read_only = True
        if args.create_schema and not args.dry_run:
            schema.create_schema(conn)
        exists = schema.schema_exists(conn)
        if not exists and not args.dry_run:
            print(f"Missing {schema.SCHEMA}; run once with --create-schema.")
            return 2
        stored = store.stored_game_state(conn, seasons) if exists else {}
        plan = update.plan_requests(
            targets, stored, today=today, retry_days=args.retry_days, dates=dates
        )
        print(f"MoviOdds: {len(plan)} game(s) planned, seasons {seasons}")
        archive = None
        if not args.dry_run and not args.no_archive and not plan.empty:
            try:
                archive = make_archive()
            except Exception as exc:
                annotate("warning", f"MoviOdds: no S3 archive ({exc})")
        with MoviOddsClient() as client:
            result = update.update_moviodds_line_history(
                plan,
                stored=stored,
                client=client,
                conn=None if args.dry_run else conn,
                archive=archive,
                dry_run=args.dry_run,
            )
            requests_made = client.requests_made
        if not args.dry_run:
            for season, counts in result.seasons.items():
                store.record_load(conn, season_year=season, **counts)
    prepared = sum(len(rows) for rows, _ in result.frames)
    print(
        f"Fetched {result.fetched_games} games in {requests_made} requests; prepared {prepared} ticks"
    )
    if args.dry_run:
        print("Dry run: no PostgreSQL or S3 writes.")
    else:
        print(
            f"Inserted {result.inserted_games} games and {result.inserted_ticks} ticks"
        )
    for warning in result.warnings:
        annotate("warning", f"MoviOdds: {warning}")
    for failure in result.failures:
        annotate("warning", f"MoviOdds: {failure}")
    print(f"Discard reasons: {dict(result.dropped)}")
    print(f"Repairs: {dict(result.repaired)}")
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        if result.frames:
            pd.concat([r for r, _ in result.frames], ignore_index=True).to_parquet(
                args.save_dir / "lh_line.parquet", index=False
            )
            pd.concat([g for _, g in result.frames], ignore_index=True).to_parquet(
                args.save_dir / "lh_game.parquet", index=False
            )
        # Keep failures visible alongside the inspectable local frames.
        import json

        (args.save_dir / "run.json").write_text(
            json.dumps(
                {
                    "planned": len(plan),
                    "fetched": result.fetched_games,
                    "ticks": prepared,
                    "requests": requests_made,
                    "failures": result.failures,
                    "warnings": result.warnings,
                    "dropped": dict(result.dropped),
                    "repaired": dict(result.repaired),
                },
                indent=2,
            )
            + "\n"
        )
    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
