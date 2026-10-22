"""Load scraped SBR line history from CSV into the Aiven Postgres store.

Reads games from the default DB (Supabase) and writes to Aiven, one season at a
time so disk and WAL never spike on the 1 GB instance. Safe to re-run: the merge
step ignores rows that are already present.

Examples::

    # what would happen, no writes
    python scripts/line_history/load_line_history_to_aiven.py --dry-run

    # create schema and load the high-confidence seasons
    python scripts/line_history/load_line_history_to_aiven.py

    # one season
    python scripts/line_history/load_line_history_to_aiven.py --seasons 2024-25

    # include the two COVID seasons whose timezone is unresolved (see
    # docs/odds/line_history_phase0_findings.md)
    python scripts/line_history/load_line_history_to_aiven.py --include-low-confidence

    # start over
    python scripts/line_history/load_line_history_to_aiven.py --reset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import (
    fetch_schedules,
    fetch_tipoffs_for_dates,
)
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven import load as loader
from nba_ou.postgre_db.line_history_aiven import schema as schema_mod
from nba_ou.postgre_db.line_history_aiven import transform as tf
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    _normalize_sbr_team_name,
    build_games_home_away_for_line_history,
    load_games_for_line_history_creation,
)

DEFAULT_ROOT = Path("data/sbr_line_history")
# Aiven free tier: 1 GB covering heap, indexes and WAL.
SIZE_BUDGET_BYTES = 1_000_000_000
SIZE_WARN_FRACTION = 0.80


def season_label_to_year(label: str) -> int:
    return int(str(label).split("-")[0])


def discover_seasons(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and "-" in p.name)


def resolve_schedule(
    season_years: list[int],
    needed: pd.DataFrame,
) -> pd.DataFrame:
    """Season schedules, topped up from the daily scoreboard where they fall short.

    ``needed`` is the (game_id, game_date) set the line history actually refers
    to; the dates of any missing ids drive the fallback lookup.
    """
    schedule = fetch_schedules(season_years)

    missing = needed[~needed["game_id"].isin(set(schedule["game_id"]))]
    if not missing.empty:
        dates = sorted(missing["game_date"].dropna().unique())
        print(
            f"  {len(missing)} game(s) absent from the season feed "
            f"across {len(dates)} date(s); querying the daily scoreboard"
        )
        extra = fetch_tipoffs_for_dates(dates)
        recovered = extra[extra["game_id"].isin(set(missing["game_id"]))]
        if not recovered.empty:
            schedule = pd.concat([schedule, recovered], ignore_index=True)
        print(f"  recovered {recovered['game_id'].nunique()} of {len(missing)}")

    return schedule.dropna(subset=["tipoff_utc"]).drop_duplicates("game_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument(
        "--include-low-confidence",
        action="store_true",
        help="also load seasons whose timezone calibration was inconclusive",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="transform and report, but write nothing",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DROP the line_history schema before loading",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"Line-history root not found: {root}", file=sys.stderr)
        return 1

    seasons = args.seasons or discover_seasons(root)

    # Hold back seasons Phase 0 could not pin a timezone for.
    skipped = []
    selected = []
    for label in seasons:
        _tz, confidence = tf.season_timezone(season_label_to_year(label))
        if confidence == "low" and not args.include_low_confidence:
            skipped.append(label)
        else:
            selected.append(label)

    if skipped:
        print(
            f"Holding back {', '.join(skipped)} (timezone unresolved). "
            "Pass --include-low-confidence to load anyway.\n"
        )
    if not selected:
        print("Nothing to load.")
        return 0

    print(f"Seasons to load: {', '.join(selected)}\n")

    print("Reading games from the default DB ...")
    games = build_games_home_away_for_line_history(
        load_games_for_line_history_creation()
    )
    print(f"  {len(games):,} home/away pairs\n")

    # Read every season's CSVs first so the schedule can be resolved in one pass.
    print("Reading CSVs ...")
    raw_by_season: dict[str, pd.DataFrame] = {}
    for label in selected:
        raw = tf.read_season_csvs(root, label)
        print(
            f"  {label}: {len(raw):>9,} rows"
            + ("  (empty, skipping)" if raw.empty else "")
        )
        if not raw.empty:
            raw_by_season[label] = raw

    selected = [label for label in selected if label in raw_by_season]
    if not selected:
        print("\nNo rows to load.")
        return 0

    print("\nResolving tipoffs ...")
    needed_frames = []
    for raw in raw_by_season.values():
        if raw.empty:
            continue
        probe = raw[["game_date", "team_home", "team_away"]].drop_duplicates()
        probe["team_home"] = probe["team_home"].map(_normalize_sbr_team_name)
        probe["team_away"] = probe["team_away"].map(_normalize_sbr_team_name)
        merged = probe.merge(
            games[["game_id", "game_date", "team_home", "team_away"]],
            on=["game_date", "team_home", "team_away"],
            how="inner",
        )
        needed_frames.append(merged[["game_id", "game_date"]])

    needed = (
        pd.concat(needed_frames, ignore_index=True).drop_duplicates("game_id")
        if needed_frames
        else pd.DataFrame(columns=["game_id", "game_date"])
    )
    schedule = resolve_schedule([season_label_to_year(s) for s in selected], needed)
    print(f"  {len(schedule):,} games with tipoff\n")

    if args.dry_run:
        conn = None
        book_ids: dict[str, int] = {}
        for raw in raw_by_season.values():
            if raw.empty:
                continue
            for slug in raw["bookmaker_slug"].dropna().astype(str).str.lower().unique():
                book_ids.setdefault(slug, len(book_ids) + 1)
    else:
        conn = connect_line_history_db()
        if args.reset:
            print("Dropping existing line_history schema ...")
            schema_mod.drop_schema(conn)
        schema_mod.create_schema(conn)
        all_slugs = [
            slug
            for raw in raw_by_season.values()
            if not raw.empty
            for slug in raw["bookmaker_slug"].dropna().astype(str).str.lower().unique()
        ]
        book_ids = schema_mod.ensure_books(conn, all_slugs)
        print(f"Schema ready. Books: {', '.join(sorted(book_ids))}\n")

    market_ids = schema_mod.market_ids()
    grand_total = 0

    try:
        for label in selected:
            season_year = season_label_to_year(label)
            raw = raw_by_season[label]
            tz, confidence = tf.season_timezone(season_year)
            print(f"--- {label}  (tz={tz}, confidence={confidence}) ---")

            rows, game_dim, stats = tf.transform_season(
                raw,
                season_year=season_year,
                games=games,
                schedule=schedule,
                book_ids=book_ids,
                market_ids=market_ids,
                normalize_team=_normalize_sbr_team_name,
            )

            print(f"  source {stats.source_rows:>9,} -> output {stats.output_rows:>9,}")
            for reason, count in sorted(stats.dropped.items(), key=lambda kv: -kv[1]):
                print(f"    dropped {count:>8,}  {reason}")
            for reason, count in sorted(stats.repaired.items(), key=lambda kv: -kv[1]):
                print(f"    repaired {count:>7,}  {reason}")
            if stats.output_rows:
                pregame = int(rows["is_pregame"].sum())
                print(
                    f"    pregame {pregame:>8,}  in-play {stats.output_rows - pregame:,}"
                    f"  games {rows['game_id'].nunique():,}"
                )

            if args.dry_run or rows.empty:
                grand_total += stats.output_rows
                print()
                continue

            schema_mod.create_season_partition(conn, season_year)
            games_written = loader.upsert_games(conn, game_dim)
            staged = loader.copy_rows(conn, rows)
            inserted = loader.merge_staging(conn, season_year)
            loader.record_load(
                conn,
                season_year=season_year,
                timezone=tz,
                confidence=confidence,
                source_rows=stats.source_rows,
                loaded_rows=inserted,
                dropped=stats.dropped,
            )
            loader.vacuum_analyze(conn, season_year)

            size_bytes, size_pretty = loader.database_size(conn)
            grand_total += inserted
            print(
                f"    games {games_written:,} | staged {staged:,} | "
                f"inserted {inserted:,} | db {size_pretty}"
            )
            if size_bytes > SIZE_BUDGET_BYTES * SIZE_WARN_FRACTION:
                print(
                    f"    WARNING: {size_bytes / SIZE_BUDGET_BYTES:.0%} of the "
                    "1 GB budget used"
                )
            print()

        if conn is not None:
            size_bytes, size_pretty = loader.database_size(conn)
            print("=" * 60)
            print(f"Loaded {grand_total:,} rows. Database size: {size_pretty}")
            print(f"Budget used: {size_bytes / SIZE_BUDGET_BYTES:.1%} of 1 GB")
            print("=" * 60)
        else:
            print("=" * 60)
            print(f"DRY RUN: {grand_total:,} rows would be loaded. Nothing written.")
            print("=" * 60)
    finally:
        if conn is not None:
            conn.close()

    return 0


if __name__ == "__main__":
    os.environ.setdefault("DB_ENV", "supabase")
    raise SystemExit(main())
