"""Bring the Aiven line-history store up to date with ``nba_games``.

``nba_games`` (Supabase) is refreshed daily, so it is the reference for what
*should* exist. Any game present there and absent from ``line_history.lh_game``
is a gap, and its date is a page worth fetching -- one daily-odds request covers
a whole slate, then one request per game returns every book across totals,
spread and moneyline.

Everything is insert-only, so running this repeatedly is safe: it adds what is
missing and never rewrites what is stored.

Examples::

    # what is missing, no fetching and no writes
    python scripts/line_history/update_line_history.py --report-only

    # fill every gap nba_games knows about
    python scripts/line_history/update_line_history.py

    # a specific window, e.g. the 2026 playoffs
    python scripts/line_history/update_line_history.py \
        --start-date 2026-04-13 --end-date 2026-06-13

    # scrape but do not write, to inspect what would land
    python scripts/line_history/update_line_history.py --limit-dates 3 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    ALL_MARKETS,
    ScrapedGame,
    discover_games_for_date,
    new_session,
    scrape_events,
)
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven import ingest as ingest_mod
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _report(missing: pd.DataFrame) -> None:
    if missing.empty:
        print("Line-history store is up to date with nba_games.")
        return

    dates = ingest_mod.missing_dates(missing)
    print(f"Missing {len(missing)} game(s) across {len(dates)} date(s).")
    if "season_year" in missing.columns:
        by_season = missing.groupby("season_year").agg(
            games=("game_id", "size"),
            first=("game_date", "min"),
            last=("game_date", "max"),
        )
        print(by_season.to_string())
    print(f"\nDate range: {dates[0]} -> {dates[-1]}")


def _print_stats(stats: ingest_mod.IngestStats, *, dry_run: bool) -> None:
    print(
        f"\nGames scraped: {stats.scraped_games} | matched to nba_games: "
        f"{stats.matched_games}"
    )
    print(
        f"Ticks scraped: {stats.source_ticks} | ready to store: {stats.prepared_ticks}"
    )
    if dry_run:
        print("Dry run: nothing written.")
    else:
        already = stats.prepared_ticks - stats.inserted_ticks
        print(
            f"Inserted {stats.inserted_ticks} new tick(s); "
            f"{already} were already stored."
        )
        print(f"New games added to lh_game: {stats.inserted_games}")
    if stats.repaired:
        print(f"Repaired: {stats.repaired}")
    if stats.dropped:
        print(f"Dropped: {stats.dropped}")
    if stats.unmatched_games:
        shown = stats.unmatched_games[:10]
        print(f"\nUnmatched games ({len(stats.unmatched_games)}, usually preseason):")
        for item in shown:
            print(f"  - {item}")
        if len(stats.unmatched_games) > len(shown):
            print(f"  ... and {len(stats.unmatched_games) - len(shown)} more")
    if stats.tipoff_disagreements:
        print(
            f"\nTipoff disagreements vs schedule feed ({len(stats.tipoff_disagreements)}):"
        )
        for item in stats.tipoff_disagreements[:10]:
            print(f"  - {item}")
    if stats.retimed_games:
        print(
            f"\nStored games moved to the page tipoff ({len(stats.retimed_games)}): "
            + ", ".join(stats.retimed_games[:10])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Named --start-date/--end-date, not --start/--end, because the daily
    # updater's --start/--end are season years.
    parser.add_argument("--start-date", type=_parse_date, default=None)
    parser.add_argument("--end-date", type=_parse_date, default=None)
    parser.add_argument(
        "--dates",
        nargs="+",
        type=_parse_date,
        default=None,
        help=(
            "scrape these dates regardless of the nba_games diff. Use to top up "
            "games that are already stored but missing a book or market; writes "
            "stay insert-only either way."
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
        "--report-only",
        action="store_true",
        help="show the gap against nba_games and stop",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scrape and report, but write nothing",
    )
    parser.add_argument(
        "--limit-dates",
        type=int,
        default=None,
        help="fetch at most this many dates, oldest first",
    )
    parser.add_argument(
        "--batch-dates",
        type=int,
        default=10,
        help="write to Postgres every N dates rather than at the end",
    )
    parser.add_argument(
        "--full-slate",
        action="store_true",
        help=(
            "fetch every game listed on each date, not just the ones missing "
            "from the store. Slower, but tops up games that are already stored "
            "with an incomplete set of books or markets."
        ),
    )
    args = parser.parse_args()

    print("Loading games from nba_games ...")
    games_df = load_games_for_line_history_creation()
    if games_df.empty:
        print("No games returned from nba_games; nothing to do.")
        return 1

    with connect_line_history_db() as conn:
        missing = ingest_mod.find_missing_games(
            conn, games_df, start=args.start_date, end=args.end_date
        )
        _report(missing)
        if args.report_only:
            return 0

        if args.dates:
            dates = sorted(set(args.dates))
            print(f"\nScraping {len(dates)} explicitly requested date(s).")
        elif missing.empty:
            return 0
        else:
            dates = ingest_mod.missing_dates(missing)

        if args.limit_dates is not None:
            dates = dates[: args.limit_dates]
            print(f"\nLimiting to the first {len(dates)} date(s).")

        seasons = sorted({d.year if d.month >= 10 else d.year - 1 for d in dates})
        print(f"\nFetching tipoff schedule for cross-checks: seasons {seasons}")
        try:
            schedule = fetch_schedules(seasons)
        except Exception as exc:  # the feed is only a cross-check; never fatal
            print(f"  ! schedule feed unavailable ({exc}); continuing without it")
            schedule = None

        print(f"\nScraping {len(dates)} date(s) from SportsbookReview ...")
        totals = ingest_mod.IngestStats()
        batch: list[ScrapedGame] = []
        seen_dates: set[date] = set()
        session = new_session()

        def flush() -> None:
            if not batch:
                return
            stats = ingest_mod.ingest_scraped_games(
                conn,
                batch,
                games_df=games_df,
                schedule=schedule,
                dry_run=args.dry_run,
            )
            for name in (
                "scraped_games",
                "matched_games",
                "source_ticks",
                "prepared_ticks",
                "inserted_ticks",
                "inserted_games",
            ):
                setattr(totals, name, getattr(totals, name) + getattr(stats, name))
            totals.unmatched_games.extend(stats.unmatched_games)
            totals.tipoff_disagreements.extend(stats.tipoff_disagreements)
            totals.retimed_games.extend(stats.retimed_games)
            for reason, count in stats.dropped.items():
                totals.drop(reason, count)
            for reason, count in stats.repaired.items():
                totals.repair(reason, count)
            batch.clear()

        lookup = ingest_mod.build_game_lookup(ingest_mod.build_game_index(games_df))
        wanted = set(missing["game_id"]) if not args.dates else None

        def event_ids_for(day: date) -> list[int]:
            """The events on ``day`` worth fetching.

            Fetching a whole slate to reach one missing game means ~7x the
            requests for no gain, since the rest collide with rows already
            stored. Resolving each listed game first keeps the backfill to
            what is actually absent.
            """
            summaries = discover_games_for_date(session, day)
            if args.full_slate or wanted is None:
                return [s.event_id for s in summaries]

            keep: list[int] = []
            for summary in summaries:
                hit = ingest_mod.resolve_game_id(
                    lookup,
                    game_date=summary.game_date,
                    team_away=summary.team_away,
                    team_home=summary.team_home,
                )
                # An unresolved game is preseason; it would be dropped on
                # ingest anyway, so there is nothing to gain by fetching it.
                if hit is not None and hit[0] in wanted:
                    keep.append(summary.event_id)
            return keep

        try:
            for day in dates:
                try:
                    event_ids = event_ids_for(day)
                except Exception as exc:  # one bad date must not end a backfill
                    print(f"  ! {day}: {exc}")
                    continue

                if not event_ids:
                    continue

                games = list(
                    scrape_events(event_ids, session=session, markets=args.markets)
                )
                if not games:
                    continue

                books = sorted({tick.book_slug for g in games for tick in g.ticks})
                ticks = sum(len(g.ticks) for g in games)
                print(
                    f"  {day}: {len(games)} game(s), {ticks} ticks, "
                    f"{len(books)} books ({', '.join(books)})"
                )

                batch.extend(games)
                seen_dates.add(day)
                if len(seen_dates) % max(args.batch_dates, 1) == 0:
                    flush()
            flush()
        except KeyboardInterrupt:
            print("\nInterrupted; flushing what was scraped so far ...")
            flush()
        finally:
            session.close()

        _print_stats(totals, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
