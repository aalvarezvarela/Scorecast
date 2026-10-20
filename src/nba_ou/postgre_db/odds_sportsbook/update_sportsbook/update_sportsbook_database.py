import argparse
import asyncio
from collections.abc import Sequence
from datetime import date

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    merge_sportsbook_with_games,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import (
    scrape_sportsbook_days,
)
from nba_ou.postgre_db.odds_sportsbook.create_db.create_odds_sportsbook_db import (
    book_columns,
    create_odds_sportsbook_table,
    upsert_odds_sportsbook_df,
)
from nba_ou.postgre_db.odds_sportsbook.update_sportsbook.update_database_utils import (
    game_ids_on_or_after,
    get_dates_for_game_ids,
    get_first_stored_game_date,
    get_game_ids_missing_columns,
    get_missing_game_ids_to_scrape,
    load_games_for_sportsbook_update,
    select_target_game_ids,
)

DEFAULT_WRITE_EVERY_DATES = 14


def _normalize_game_id(df: pd.DataFrame, col: str = "game_id") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str)
    return out


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize team_home and team_away columns using TEAM_NAME_STANDARDIZATION."""
    out = df.copy()
    if "team_home" in out.columns:
        out["team_home"] = out["team_home"].map(
            lambda x: TEAM_NAME_STANDARDIZATION.get(x, x) if pd.notna(x) else x
        )
    if "team_away" in out.columns:
        out["team_away"] = out["team_away"].map(
            lambda x: TEAM_NAME_STANDARDIZATION.get(x, x) if pd.notna(x) else x
        )
    return out


def _empty_summary(**overrides: int) -> dict[str, int]:
    summary = {
        "target_game_ids": 0,
        "missing_game_ids": 0,
        "backfill_game_ids": 0,
        "scraped_rows": 0,
        "merged_rows": 0,
        "inserted_rows": 0,
        "backfilled_rows": 0,
    }
    summary.update(overrides)
    return summary


def update_odds_sportsbook_database(
    *,
    last_n_games: int | None = None,
    season_year: str | int | None = None,
    headless: bool = True,
    backfill_books: Sequence[str] = (),
    engine: str = "json",
    write_every_dates: int = DEFAULT_WRITE_EVERY_DATES,
    skip_before_coverage: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Incrementally update odds_sportsbook DB by scraping only missing GAME_IDs.

    Flow:
    1) Load all games from DB (optionally filter by season)
    2) Select target GAME_IDs (optionally latest N)
    3) Find missing GAME_IDs in odds_sportsbook table
    3b) With ``backfill_books``, also find stored GAME_IDs whose columns for
        those books are all NULL (a book added after the game was scraped)
    4) Scrape only corresponding dates from sportsbookreview
    5) Merge scraped rows with games DB to resolve canonical GAME_ID
    6) Insert missing games; fill only the backfilled books' NULL columns on
       stored games, so no existing value is ever overwritten

    Steps 4-6 run per ``write_every_dates`` dates, so an interrupted run keeps
    what it already wrote and a rerun resumes from there.

    ``skip_before_coverage`` (the all-seasons mode) ignores missing games dated
    before the first odds ever stored, and backfill games dated before the
    first stored value of the backfilled books. SBR has nothing earlier, so
    without it every run would re-request those dates forever. A book never
    stored at all has no such date yet, so its first run searches everything.

    ``dry_run`` reports what would be scraped and stops before any request.
    """
    if not create_odds_sportsbook_table(drop_existing=False):
        raise RuntimeError("Failed to create/validate odds_sportsbook table.")

    fill_columns = book_columns(tuple(backfill_books)) if backfill_books else []
    if backfill_books and not fill_columns:
        raise ValueError(
            f"No odds_sportsbook columns for {list(backfill_books)}; add the book "
            "to TOTAL_BOOKS / SPREAD_BOOKS / ML_BOOKS first."
        )

    games_df = load_games_for_sportsbook_update(season_year=season_year)
    if games_df.empty:
        print("No games found in games database for sportsbook update.")
        return _empty_summary()

    target_game_ids = select_target_game_ids(games_df, last_n_games=last_n_games)
    missing_game_ids = get_missing_game_ids_to_scrape(target_game_ids)
    backfill_game_ids = (
        get_game_ids_missing_columns(target_game_ids, fill_columns)
        if fill_columns
        else []
    )

    print(f"Target games: {len(target_game_ids)}")
    if skip_before_coverage:
        odds_start = get_first_stored_game_date()
        missing_game_ids = game_ids_on_or_after(games_df, missing_game_ids, odds_start)
        print(f"Odds coverage starts: {odds_start or 'nothing stored yet'}")
    print(f"Missing sportsbook games: {len(missing_game_ids)}")
    if fill_columns:
        if skip_before_coverage:
            book_start = get_first_stored_game_date(fill_columns)
            backfill_game_ids = game_ids_on_or_after(
                games_df, backfill_game_ids, book_start
            )
            print(
                f"{list(backfill_books)} coverage starts: "
                f"{book_start or 'never stored, searching every season'}"
            )
        print(f"Stored games without {list(backfill_books)}: {len(backfill_game_ids)}")

    counts = {
        "target_game_ids": len(target_game_ids),
        "missing_game_ids": len(missing_game_ids),
        "backfill_game_ids": len(backfill_game_ids),
    }
    if not missing_game_ids and not backfill_game_ids:
        print("No missing game IDs to scrape. Database is up to date for target games.")
        return _empty_summary(**counts)

    scrape_dates_ts = get_dates_for_game_ids(
        games_df, set(missing_game_ids) | set(backfill_game_ids)
    )
    scrape_dates: list[date] = [d.date() for d in scrape_dates_ts]
    games_df = _normalize_game_id(games_df)
    missing_set = set(missing_game_ids)
    backfill_set = set(backfill_game_ids)

    if dry_run:
        by_season = pd.Series(
            [d.year if d.month >= 8 else d.year - 1 for d in scrape_dates]
        ).value_counts()
        print(f"Dry run: would scrape {len(scrape_dates)} date(s)")
        for season, n in sorted(by_season.items()):
            print(f"  {season}-{str(season + 1)[-2:]}: {n} date(s)")
        return _empty_summary(**counts)

    print(f"Scraping sportsbook for {len(scrape_dates)} date(s)...")
    totals = {
        "scraped_rows": 0,
        "merged_rows": 0,
        "inserted_rows": 0,
        "backfilled_rows": 0,
    }
    chunk_size = write_every_dates if write_every_dates > 0 else len(scrape_dates)
    for start in range(0, len(scrape_dates), chunk_size):
        chunk = scrape_dates[start : start + chunk_size]
        scraped_df = asyncio.run(
            scrape_sportsbook_days(chunk, headless=headless, engine=engine)
        )
        if scraped_df.empty:
            continue
        totals["scraped_rows"] += len(scraped_df)

        scraped_df = _normalize_game_id(scraped_df)
        scraped_df = _normalize_team_names(scraped_df)
        merged_df = merge_sportsbook_with_games(scraped_df, games_df)
        if "game_id" not in merged_df:
            continue
        merged_df = _normalize_game_id(merged_df.dropna(subset=["game_id"]))
        if merged_df.empty:
            continue

        new_rows = merged_df[merged_df["game_id"].isin(missing_set)]
        backfill_rows = merged_df[merged_df["game_id"].isin(backfill_set)]
        totals["merged_rows"] += len(new_rows) + len(backfill_rows)
        totals["inserted_rows"] += upsert_odds_sportsbook_df(new_rows)
        if fill_columns:
            totals["backfilled_rows"] += upsert_odds_sportsbook_df(
                backfill_rows, fill_columns=fill_columns
            )
        print(
            f"Written through {chunk[-1].isoformat()}: "
            f"inserted {totals['inserted_rows']}, "
            f"backfilled {totals['backfilled_rows']}",
            flush=True,
        )

    if not totals["scraped_rows"]:
        print("No sportsbook rows scraped for missing games.")
    elif not totals["merged_rows"]:
        print("Scraped rows could not be mapped to game_id.")

    print(f"Scraped rows: {totals['scraped_rows']}")
    print(f"Mapped rows: {totals['merged_rows']}")
    print(f"Inserted rows: {totals['inserted_rows']}")
    if fill_columns:
        print(f"Backfilled rows ({list(backfill_books)}): {totals['backfilled_rows']}")

    return _empty_summary(**counts, **totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Update odds_sportsbook DB by scraping missing game IDs from sportsbookreview"
        )
    )
    parser.add_argument(
        "--last-n-games",
        type=int,
        default=None,
        help="If provided, only check and update the latest N games from the games DB",
    )
    seasons = parser.add_mutually_exclusive_group()
    seasons.add_argument(
        "--season-year",
        type=int,
        default=2018,
        help="Season start year to check (e.g., 2025 for 2025-26 season)",
    )
    seasons.add_argument(
        "--all-seasons",
        action="store_true",
        help=(
            "Check every season, skipping dates before the stored odds (and "
            "backfilled books) begin"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be scraped per season, then stop",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=False,
        help=(
            "Browser engine only: show the window (debugging; needs a display). "
            "Default is headless, which works over SSH/screen."
        ),
    )

    parser.add_argument(
        "--backfill-books",
        nargs="*",
        default=[],
        metavar="SLUG",
        help=(
            "Also re-scrape stored games whose columns for these books are all "
            "NULL, and fill only those columns (e.g. --backfill-books betrivers)"
        ),
    )

    parser.add_argument(
        "--engine",
        choices=("json", "browser"),
        default="json",
        help=(
            "json (default): read the odds embedded in the page over plain HTTP. "
            "browser: drive Playwright through the rendered table (slow fallback)."
        ),
    )
    parser.add_argument(
        "--write-every-dates",
        type=int,
        default=DEFAULT_WRITE_EVERY_DATES,
        help="Write to the database after this many scraped dates (0 = at the end)",
    )

    args = parser.parse_args()

    results = update_odds_sportsbook_database(
        last_n_games=args.last_n_games,
        season_year=None if args.all_seasons else args.season_year,
        headless=not args.headed,
        backfill_books=args.backfill_books,
        engine=args.engine,
        write_every_dates=args.write_every_dates,
        skip_before_coverage=args.all_seasons,
        dry_run=args.dry_run,
    )
    print("Update summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
