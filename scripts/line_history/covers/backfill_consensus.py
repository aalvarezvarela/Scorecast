"""Backfill a past season's Covers Consensus pick counts and percentages.

SBR's consensus pick percentages (``*_consensus_pct_*``) are the Covers
Consensus shares. For 2023-24 the stored ones are missing for about 60% of the
games and unreliable where present (``docs/pipeline/improvements_2_5.md``).
Covers still serves that season's pick counts, and SBR's event ids are Covers'
game ids, so each game is one request.

Two steps, each a dry run unless ``--execute`` is given:

``fetch``
    Asks Covers for the pick counts of every game of the season in the SBR
    line-history store that has none in ``line_history_covers.lh_consensus``,
    and stores them there. A game whose page has no counts is left for a later
    run. ``--save-dir`` also writes the rows to ``lh_consensus.parquet``;
    ``--from-parquet`` stores such a file without asking Covers again. The
    first ``--execute`` drops ``lh_consensus``'s reference to ``lh_game``, so
    the counts of games without stored Covers lines fit.

``apply``
    Overwrites the four percentage columns of ``odds_sportsbook`` for the
    season's games with stored counts (SBR's rendering: rounded, empty for 0).
    The untouched SBR rows stay in ``odds_sportsbook_sbr_raw``.

The training data changes only after the season is rebuilt.

Examples::

    # what would be fetched; rows kept for the real run
    python scripts/line_history/covers/backfill_consensus.py fetch \\
        --season-year 2023 --save-dir /tmp/covers_consensus_2023

    # store the rows of that dry run
    python scripts/line_history/covers/backfill_consensus.py fetch \\
        --season-year 2023 --execute \\
        --from-parquet /tmp/covers_consensus_2023/lh_consensus.parquet

    # what would change in odds_sportsbook, then change it
    python scripts/line_history/covers/backfill_consensus.py apply --season-year 2023
    python scripts/line_history/covers/backfill_consensus.py apply \\
        --season-year 2023 --execute
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.odds_covers.archive import make_archive
from nba_ou.fetch_data.odds_covers.client import CoversClient
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_covers import schema as schema_mod
from nba_ou.postgre_db.line_history_covers import store
from nba_ou.postgre_db.line_history_covers import update as update_mod
from nba_ou.postgre_db.odds_sportsbook.covers_pct_overwrite import (
    PCT_COLUMN_NAMES,
    overwrite_season_percentages,
)
from nba_ou.utils.ci import annotate


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _fetch(args) -> int:
    season = args.season_year
    with connect_line_history_db() as conn:
        if not schema_mod.consensus_table_exists(conn):
            print(f"error: {schema_mod.SCHEMA}.lh_consensus does not exist")
            return 2
        stored = store.stored_consensus_game_ids(conn, season)

        if args.from_parquet:
            frame = pd.read_parquet(args.from_parquet)
            frame = frame[frame["season_year"].astype(int) == season]
            frame = frame[~frame["game_id"].astype(str).isin(stored)]
            print(
                f"{args.from_parquet}: {len(frame)} game(s) of {season} without "
                "stored counts"
            )
            if not args.execute or frame.empty:
                print("Dry run: nothing written." if not frame.empty else "")
                return 0
            schema_mod.drop_consensus_game_reference(conn)
            inserted = store.insert_consensus(conn, frame[store.CONSENSUS_COLUMNS])
            print(f"Stored the consensus of {inserted} game(s).")
            return 0

        games = store.fetch_sbr_game_events(conn, season)
        if args.dates and not games.empty:
            games = games[games["game_date"].isin(args.dates)]
        targets = update_mod.plan_consensus(games, stored, now=datetime.now(UTC))
        planned = sum(len(day) for day in targets.values())
        print(
            f"Season {season}: {len(games)} SBR game(s), {len(stored)} with stored "
            f"counts; {planned} to fetch on {len(targets)} date(s)"
        )
        if not planned:
            return 0

        archive = None
        if args.execute:
            schema_mod.drop_consensus_game_reference(conn)
            if not args.no_archive:
                try:
                    archive = make_archive()
                except Exception as exc:  # the counts are still stored
                    annotate("warning", f"Covers: no S3 archive ({exc})")
        with CoversClient() as client:
            result = update_mod.update_consensus(
                targets,
                conn=conn if args.execute else None,
                client=client,
                archive=archive,
                dry_run=not args.execute,
                progress=lambda line: print(line, flush=True),
            )

    prepared = sum(len(f) for f in result.consensus_frames)
    print(
        f"Fetched {result.consensus_fetched} page(s): {prepared} game(s) with "
        f"counts, {len(result.consensus_empty)} without, "
        f"{len(result.failed)} failed; stored {result.consensus_inserted}."
    )
    for label in result.consensus_empty:
        print(f"  no counts: {label}")
    for label in result.failed:
        print(f"  failed: {label}")
    if args.save_dir and result.consensus_frames:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        path = args.save_dir / "lh_consensus.parquet"
        pd.concat(result.consensus_frames, ignore_index=True).to_parquet(
            path, index=False
        )
        print(f"Saved {prepared} row(s) to {path}")
    if not args.execute:
        print("Dry run: nothing written to Aiven or S3.")
    return 1 if result.failed else 0


def _apply(args) -> int:
    plan = overwrite_season_percentages(args.season_year, dry_run=not args.execute)
    replaced = int(plan["had_sbr_value"].sum()) if not plan.empty else 0
    verb = "Rewrote" if args.execute else "Would rewrite"
    print(
        f"{verb} the pick percentages of {len(plan)} game(s) of "
        f"{args.season_year}: {replaced} replace SBR's values, "
        f"{len(plan) - replaced} fill empty ones."
    )
    if not plan.empty:
        print(plan[PCT_COLUMN_NAMES].describe().loc[["mean", "std", "min", "max"]])
    if not args.execute:
        print("Dry run: nothing written.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    steps = parser.add_subparsers(dest="step", required=True)
    for name in ("fetch", "apply"):
        step = steps.add_parser(name)
        step.add_argument("--season-year", type=int, required=True)
        step.add_argument(
            "--execute", action="store_true", help="write (default: dry run)"
        )
    fetch = steps.choices["fetch"]
    fetch.add_argument("--dates", nargs="*", type=_parse_date, metavar="YYYY-MM-DD")
    fetch.add_argument("--save-dir", type=Path, help="also write the rows as Parquet")
    fetch.add_argument(
        "--from-parquet", type=Path, help="store these rows; no request is made"
    )
    fetch.add_argument(
        "--no-archive", action="store_true", help="do not archive raw answers to S3"
    )
    args = parser.parse_args()
    return _fetch(args) if args.step == "fetch" else _apply(args)


if __name__ == "__main__":
    sys.exit(main())
