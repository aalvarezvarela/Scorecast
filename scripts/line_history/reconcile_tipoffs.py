"""Put stored line-history games on the tipoff they were actually played at.

For each stored game, the tipoff is compared with the tipoff published for the
date ``nba_games`` says it was played -- by default on SBR's daily odds page
(``--source sbr``), optionally on the NBA daily scoreboard (``--source nba``).
Every game that differs is retimed: ``lh_game.tipoff_utc`` is replaced and
``mins_to_tip`` / ``is_pregame`` are recomputed from ``line_ts`` for all its
ticks. Games whose tipoff is right but hold ticks timed against another clock
(backfilled books) are realigned too. Idempotent.

``--write-corrections`` also regenerates
``nba_ou/fetch_data/nba_schedule/tipoff_corrections_data.py``, the override the
season-schedule feed goes through, so a bulk reload or the injury-report
resolver cannot bring the stale times back.

Why SBR by default: stats.nba.com (``nba_api`` ``ScoreboardV3``) stops answering
after a few hundred requests and stayed blocked for hours on 2026-09-17, while
SBR served a whole season (214 dates) in about a minute. SBR's tipoff agreed with
the NBA scoreboard on all 1,321 games of 2024-25 and on all 66 moved games of
2025-26, and it is the clock the ticks themselves were scraped against.

Either way one page is fetched per date (~200 a season) and cached under
``--cache-dir`` as soon as it arrives. A run that ends with dates missing raises
*before touching the database*; rerun and it continues from the cache --
``scripts/line_history/reconcile_tipoffs.sh`` does that. With ``--source nba``
a run also stops after ``--max-requests``.

Examples::

    scripts/line_history/reconcile_tipoffs.sh --season-year 2025 --dry-run
    scripts/line_history/reconcile_tipoffs.sh --all-seasons --write-corrections
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import cache_scoreboards
from nba_ou.fetch_data.nba_schedule.tipoff_corrections import (
    CORRECTIONS_DATA_PATH,
    TIPOFF_CORRECTIONS,
    render_corrections_module,
)
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven.tipoff_corrections import (
    find_tipoff_mismatches,
    games_with_inconsistent_clocks,
    retime_games,
    sbr_tipoffs,
    scoreboard_tipoffs,
    stored_tipoffs,
)
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)


DEFAULT_CACHE_DIRS = {"sbr": "data/cache/sbr_daily", "nba": "data/cache/nba_scoreboard"}
DEFAULT_MAX_REQUESTS = 290


class ScoreboardsIncomplete(RuntimeError):
    """Raised when a run ends with reference dates still uncached."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    seasons = parser.add_mutually_exclusive_group(required=True)
    seasons.add_argument("--season-year", type=int, action="append")
    seasons.add_argument("--all-seasons", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="report only")
    parser.add_argument(
        "--write-corrections",
        action="store_true",
        help="regenerate the schedule-feed override from the scoreboard result",
    )
    parser.add_argument(
        "--source",
        choices=("sbr", "nba"),
        default="sbr",
        help="tipoff reference: SBR daily odds pages (default) or the NBA scoreboard",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"where pages are cached (default: {DEFAULT_CACHE_DIRS})",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=(
            "--source nba only: requests per run before raising, kept under the "
            f"~300 at which the API stops answering (default: {DEFAULT_MAX_REQUESTS})"
        ),
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIRS[args.source]

    conn = connect_line_history_db()
    try:
        stored = stored_tipoffs(conn, None if args.all_seasons else args.season_year)
        if stored.empty:
            print("No stored games for those seasons.")
            return 0
        print(f"{len(stored)} stored game(s); reading nba_games ...")
        games = load_games_for_line_history_creation()
        games["game_id"] = games["game_id"].astype(str)
        played = games.drop_duplicates("game_id")[["game_id", "game_date"]]
        played = played[played["game_id"].isin(set(stored["game_id"]))]

        if args.source == "sbr":
            reference, failed = sbr_tipoffs(played, games, cache_dir)
            remaining = failed
        else:
            remaining = cache_scoreboards(
                played["game_date"].tolist(), cache_dir, max_requests=args.max_requests
            )
        if remaining:
            raise ScoreboardsIncomplete(
                f"{len(remaining)} date(s) still to fetch ({', '.join(remaining[:5])}); "
                "nothing written. Run again to continue from the cache."
            )
        if args.source == "nba":
            reference = scoreboard_tipoffs(played, cache_dir=cache_dir)
        missing = sorted(set(stored["game_id"]) - set(reference["game_id"]))
        if missing:
            print(
                f"! {len(missing)} game(s) have no {args.source} tipoff, left as "
                f"stored: {', '.join(missing[:10])}"
            )

        mismatches = find_tipoff_mismatches(stored, reference)
        tipoffs = dict(
            zip(mismatches["game_id"], mismatches["reference_tipoff_utc"], strict=True)
        )
        stored_by_id = stored.set_index("game_id")["tipoff_utc"]
        for game_id in games_with_inconsistent_clocks(
            conn, sorted(stored["season_year"].unique().tolist())
        ):
            tipoffs.setdefault(game_id, stored_by_id[game_id])

        print(
            f"{len(mismatches)} game(s) off the {args.source} tipoff; "
            f"{len(tipoffs) - len(mismatches)} more with inconsistent tick clocks"
        )
        report = retime_games(conn, tipoffs, dry_run=args.dry_run)
    finally:
        conn.close()

    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(report.sort_values("shift_minutes").to_string(index=False))

    if args.write_corrections:
        corrections = {
            game_id: (c.feed_tipoff_utc, c.tipoff_utc)
            for game_id, c in TIPOFF_CORRECTIONS.items()
        }
        for row in mismatches.itertuples(index=False):
            feed = corrections.get(row.game_id, (row.tipoff_utc, None))[0]
            corrections[row.game_id] = (feed, row.reference_tipoff_utc)
        CORRECTIONS_DATA_PATH.write_text(render_corrections_module(corrections))
        print(f"Wrote {len(corrections)} correction(s) to {CORRECTIONS_DATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
