import argparse
import asyncio

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    merge_sportsbook_with_games,
)
from nba_ou.fetch_data.odds_yahoo.scrape_yahoo import scrape_yahoo_days
from nba_ou.postgre_db.odds_yahoo.create_db.create_odds_yahoo_db import (
    create_odds_yahoo_table,
    upsert_odds_yahoo_df,
)
from nba_ou.postgre_db.odds_yahoo.update_yahoo.update_database_utils import (
    get_dates_for_game_ids,
    get_missing_game_ids_to_scrape,
    load_games_for_yahoo_update,
    select_target_game_ids,
)


def _normalize_game_id(df: pd.DataFrame, col: str = "game_id") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str)
    return out


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _map(name: object) -> object:
        if pd.isna(name):
            return name
        n = str(name).strip()
        if "all-star" in n.lower():
            return None
        return TEAM_NAME_STANDARDIZATION.get(n, n)

    if "team_home" in out.columns:
        out["team_home"] = out["team_home"].map(_map)
    if "team_away" in out.columns:
        out["team_away"] = out["team_away"].map(_map)

    return out


def update_odds_yahoo_database(
    *,
    last_n_games: int | None = None,
    season_year: str | int | None = None,
) -> dict[str, int]:
    """
    Incrementally update odds_yahoo DB by scraping only missing GAME_IDs.

    Flow:
    1) Load games from DB and optionally filter to latest N games
    2) Find target game IDs missing in odds_yahoo
    3) Scrape Yahoo for only the corresponding (US) game dates
    4) Merge with games DB to recover canonical game_id
    5) Upsert missing rows into odds_yahoo
    """
    if not create_odds_yahoo_table(drop_existing=False):
        raise RuntimeError("Failed to create/validate odds_yahoo table.")

    results = {
        "target_game_ids": 0,
        "missing_game_ids": 0,
        "scraped_games": 0,
        "merged_rows": 0,
        "inserted_rows": 0,
    }

    games_df = load_games_for_yahoo_update(season_year=season_year)
    if games_df.empty:
        print("No games found in games database for Yahoo update.")
        return results

    target_game_ids = select_target_game_ids(games_df, last_n_games=last_n_games)
    missing_game_ids = get_missing_game_ids_to_scrape(target_game_ids)
    results["target_game_ids"] = len(target_game_ids)
    results["missing_game_ids"] = len(missing_game_ids)

    print(f"Target games: {len(target_game_ids)}")
    print(f"Missing Yahoo games: {len(missing_game_ids)}")

    if not missing_game_ids:
        print(
            "No missing game IDs to scrape. Yahoo database is up to date for target games."
        )
        return results

    scrape_dates = [
        d.date() for d in get_dates_for_game_ids(games_df, missing_game_ids)
    ]
    print(f"Scraping Yahoo for {len(scrape_dates)} date(s)...")
    print(f"Scrape dates: {[d.isoformat() for d in scrape_dates]}")
    game_level_df = asyncio.run(scrape_yahoo_days(scrape_dates))
    results["scraped_games"] = len(game_level_df)

    if game_level_df.empty:
        print("No Yahoo rows scraped for missing games.")
        return results

    game_level_df = _normalize_team_names(game_level_df)
    games_df = _normalize_game_id(games_df)

    merged_df = merge_sportsbook_with_games(game_level_df, games_df)
    merged_df = (
        merged_df.dropna(subset=["game_id"])
        if "game_id" in merged_df.columns
        else pd.DataFrame()
    )

    if merged_df.empty:
        print("Yahoo game rows could not be mapped to game_id.")
        return results

    merged_df = _normalize_game_id(merged_df)
    missing_set = set(missing_game_ids)
    merged_df = merged_df[merged_df["game_id"].isin(missing_set)]
    results["merged_rows"] = len(merged_df)

    results["inserted_rows"] = upsert_odds_yahoo_df(merged_df)

    print(f"Yahoo game rows: {len(game_level_df)}")
    print(f"Mapped rows: {len(merged_df)}")
    print(f"Inserted rows: {results['inserted_rows']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update odds_yahoo DB by scraping missing game IDs from Yahoo odds"
    )
    parser.add_argument(
        "--last-n-games",
        type=int,
        default=None,
        help="If provided, only check/update the latest N games from the games DB",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=2025,
        help="Optional season start year filter (e.g., 2025 for 2025-26 season)",
    )

    args = parser.parse_args()

    results = update_odds_yahoo_database(
        last_n_games=args.last_n_games,
        season_year=args.season_year,
    )
    print("Update summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
