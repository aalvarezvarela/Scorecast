"""
Module for merging odds data with games data based on game_id.

This provides an alternative to the date-based merge, using game_id for exact matching.
Only merges total line, spread, and moneyline - no additional columns.
"""

from typing import Literal

import numpy as np
import pandas as pd

from nba_ou.config.odds_columns import (
    get_main_book,
    moneyline_col,
    spread_col,
    total_line_col,
)

DEFAULT_PRIMARY_BOOK = get_main_book()


def _extract_available_books(
    df_odds: pd.DataFrame,
    *,
    market_prefix: str,
    required_suffixes: tuple[str, ...],
) -> list[str]:
    """
    Extract books that have all required columns for a market.
    """
    books = set()
    for col in df_odds.columns:
        if not col.startswith(f"{market_prefix}_"):
            continue
        for suffix in required_suffixes:
            token = f"_{suffix}"
            if col.endswith(token):
                book = col[len(market_prefix) + 1 : -len(token)]
                if book:
                    books.add(book)
                break

    available = []
    for book in sorted(books):
        has_all = all(
            f"{market_prefix}_{book}_{suffix}" in df_odds.columns
            for suffix in required_suffixes
        )
        if has_all:
            available.append(book)
    return available


def _assert_book_available(
    *,
    selected_book: str,
    available_books: list[str],
    market_name: str,
) -> None:
    if selected_book in available_books:
        return
    raise ValueError(
        f"Configured {market_name} book '{selected_book}' is not available in odds data. "
        f"Available {market_name} books: {available_books}"
    )


def merge_odds_percentages_and_prices_by_game_id(
    df_odds: pd.DataFrame,
    df_team: pd.DataFrame,
    exclude_yahoo: bool = False,
) -> pd.DataFrame:
    """
    Merge odds percentages and prices with team dataframe BEFORE computing rolling stats.

    This function merges:
    - Consensus percentages (game-level, same for both teams)
    - Yahoo percentages (team-specific based on home/away)
    - Price columns (total=game-level, spread/ml=team-specific)

    Team-specific columns are assigned based on whether the team is home or away.

    Args:
        df_odds: Odds dataframe with game_id
        df_team: Team dataframe (one row per team per game) with GAME_ID and HOME columns
        exclude_yahoo: If True, exclude Yahoo-specific betting columns

    Returns:
        DataFrame with odds percentages and prices merged per team
    """
    if df_odds.empty or "game_id" not in df_odds.columns:
        print("Warning: Odds dataframe empty or missing game_id")
        return df_team

    if "GAME_ID" not in df_team.columns or "HOME" not in df_team.columns:
        print("Warning: Team dataframe missing GAME_ID or HOME column")
        return df_team

    # Identify available columns
    consensus_pct_cols = [
        c
        for c in df_odds.columns
        if c.startswith("total_consensus_pct_")
        or c.startswith("spread_consensus_pct_")
        or c.startswith("moneyline_consensus_pct_")
    ]

    yahoo_cols = []
    if not exclude_yahoo:
        yahoo_patterns = ["_pct_bets_", "_pct_money_"]
        yahoo_cols = [c for c in df_odds.columns if any(p in c for p in yahoo_patterns)]

    # Identify price columns
    total_price_cols = [
        c for c in df_odds.columns if c.startswith("total_") and "_price_" in c
    ]
    spread_price_cols = [
        c for c in df_odds.columns if c.startswith("spread_") and "_price_" in c
    ]
    ml_price_cols = [
        c for c in df_odds.columns if c.startswith("ml_") and "_price_" in c
    ]

    # Build subset of columns to merge
    cols_to_merge = (
        ["game_id"]
        + consensus_pct_cols
        + yahoo_cols
        + total_price_cols
        + spread_price_cols
        + ml_price_cols
    )
    cols_to_merge = [c for c in cols_to_merge if c in df_odds.columns]

    if len(cols_to_merge) <= 1:
        print("Warning: No percentages or prices found in odds data")
        return df_team

    df_odds_subset = df_odds[cols_to_merge].copy()
    df_odds_subset = df_odds_subset.rename(columns={"game_id": "GAME_ID"})

    # Merge with team data
    df_merged = df_team.merge(
        df_odds_subset, on="GAME_ID", how="left", suffixes=("", "_odds")
    )

    # Process team-specific columns (Yahoo percentages, spread prices, ML prices)
    home_mask = df_merged["HOME"].astype(bool)

    # Yahoo percentages: assign based on home/away
    if not exclude_yahoo:
        yahoo_home_away_pairs = [
            ("total_pct_bets_over", "total_pct_bets_over"),  # Same for both
            ("total_pct_bets_under", "total_pct_bets_under"),  # Same for both
            ("total_pct_money_over", "total_pct_money_over"),  # Same for both
            ("total_pct_money_under", "total_pct_money_under"),  # Same for both
            ("spread_pct_bets", "spread_pct_bets_home", "spread_pct_bets_away"),
            ("spread_pct_money", "spread_pct_money_home", "spread_pct_money_away"),
            (
                "moneyline_pct_bets",
                "moneyline_pct_bets_home",
                "moneyline_pct_bets_away",
            ),
            (
                "moneyline_pct_money",
                "moneyline_pct_money_home",
                "moneyline_pct_money_away",
            ),
        ]

        for pair in yahoo_home_away_pairs:
            if len(pair) == 2:  # Game-level (totals)
                continue  # Already correct
            elif len(pair) == 3:  # Team-specific
                new_col, home_col, away_col = pair
                if home_col in df_merged.columns and away_col in df_merged.columns:
                    df_merged[new_col] = np.where(
                        home_mask, df_merged[home_col], df_merged[away_col]
                    )
                    df_merged = df_merged.drop(
                        columns=[home_col, away_col], errors="ignore"
                    )

    # Spread prices: assign based on home/away
    # Sorted: set order depends on the per-process hash seed, which made the
    # output column order differ between otherwise identical builds.
    spread_books = sorted(
        set(
            [
                c.replace("spread_", "")
                .replace("_price_home", "")
                .replace("_price_away", "")
                for c in spread_price_cols
            ]
        )
    )
    for book in spread_books:
        home_col = f"spread_{book}_price_home"
        away_col = f"spread_{book}_price_away"
        if home_col in df_merged.columns and away_col in df_merged.columns:
            new_col = f"spread_{book}_price"
            df_merged[new_col] = np.where(
                home_mask, df_merged[home_col], df_merged[away_col]
            )
            df_merged = df_merged.drop(columns=[home_col, away_col], errors="ignore")

    # ML prices: assign based on home/away
    # Sorted: set order depends on the per-process hash seed, which made the
    # output column order differ between otherwise identical builds.
    ml_books = sorted(
        set(
            [
                c.replace("ml_", "")
                .replace("_price_home", "")
                .replace("_price_away", "")
                for c in ml_price_cols
            ]
        )
    )
    for book in ml_books:
        home_col = f"ml_{book}_price_home"
        away_col = f"ml_{book}_price_away"
        if home_col in df_merged.columns and away_col in df_merged.columns:
            new_col = f"ml_{book}_price"
            df_merged[new_col] = np.where(
                home_mask, df_merged[home_col], df_merged[away_col]
            )
            df_merged = df_merged.drop(columns=[home_col, away_col], errors="ignore")

    # Total prices are already game-level (same for both teams), no changes needed

    print(
        f"Merged {len(cols_to_merge) - 1} odds percentage/price columns (exclude_yahoo={exclude_yahoo})"
    )

    return df_merged


def merge_total_spread_moneyline_by_game_id(
    df_odds: pd.DataFrame,
    df_team: pd.DataFrame,
    book: str = DEFAULT_PRIMARY_BOOK,
    total_line_book: str | None = None,
    total_lines_mode: Literal["selected", "all", "none"] = "all",
    debug: bool = False,
    exclude_yahoo: bool = False,
) -> pd.DataFrame:
    """
    Merge team dataframe with odds data using game_id.

    total_lines_mode:
      - "none": do not merge any total line
      - "selected": merge TOTAL_LINE_<total_line_book> only
      - "all": merge total lines for all known books into TOTAL_LINE_<book> columns

    exclude_yahoo: If True, exclude Yahoo-specific betting columns (pct_bets, pct_money)
                   but keep metadata columns (game_date, teams, etc.)
    """
    if total_line_book is None:
        total_line_book = book

    if df_odds.empty:
        print("Warning: Odds dataframe is empty")
        return df_team

    if "game_id" not in df_odds.columns:
        print("Warning: 'game_id' column not found in odds dataframe")
        return df_team

    if "GAME_ID" not in df_team.columns:
        print("Warning: 'GAME_ID' column not found in team dataframe")
        return df_team

    # Security check: validate configured books exist in incoming odds data.
    available_spread_books = _extract_available_books(
        df_odds,
        market_prefix="spread",
        required_suffixes=("line_home", "line_away"),
    )
    available_ml_books = _extract_available_books(
        df_odds,
        market_prefix="ml",
        required_suffixes=("price_home", "price_away"),
    )
    available_total_books = _extract_available_books(
        df_odds,
        market_prefix="total",
        required_suffixes=("line_over",),
    )

    _assert_book_available(
        selected_book=book,
        available_books=available_spread_books,
        market_name="spread",
    )
    _assert_book_available(
        selected_book=book,
        available_books=available_ml_books,
        market_name="moneyline",
    )
    _assert_book_available(
        selected_book=total_line_book,
        available_books=available_total_books,
        market_name="total-line",
    )

    # Debug: print df_team rows whose GAME_ID is not in df_odds
    if debug:
        required_team_cols = {"GAME_DATE", "TEAM_NAME"}
        missing_team_cols = [c for c in required_team_cols if c not in df_team.columns]
        if missing_team_cols:
            print(
                f"Debug warning: df_team missing columns needed for debug print: {missing_team_cols}"
            )
        else:
            games_df = df_team[
                ~df_team["SEASON_TYPE"].isin(["Preseason", "All Star"])
            ].copy()
            games_df = games_df[games_df["SEASON_YEAR"] >= 2019]
            team_game_ids = set(games_df["GAME_ID"].dropna().unique())
            odds_game_ids = set(df_odds["game_id"].dropna().unique())

            missing_game_ids = team_game_ids - odds_game_ids
            if missing_game_ids:
                df_missing = (
                    games_df.loc[
                        games_df["GAME_ID"].isin(missing_game_ids),
                        ["GAME_ID", "TEAM_NAME", "GAME_DATE"],
                    ]
                    .drop_duplicates()
                    .sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"])
                )
                print(
                    f"[DEBUG] Games in df_team but not in df_odds: "
                    f"{len(missing_game_ids)} GAME_IDs, {len(df_missing)} team-rows"
                )
                print(df_missing.to_string(index=False))
            else:
                print("[DEBUG] All df_team GAME_IDs are present in df_odds.")

    # Base columns for the selected book (spread + moneyline)
    spread_home_col = f"spread_{book}_line_home"
    spread_away_col = f"spread_{book}_line_away"
    ml_home_col = f"ml_{book}_price_home"
    ml_away_col = f"ml_{book}_price_away"

    required_cols = [
        "game_id",
        spread_home_col,
        spread_away_col,
        ml_home_col,
        ml_away_col,
    ]

    # Totals handling
    known_total_sources = [
        ("consensus_opener", "total_consensus_opener_line_over"),
        ("betmgm", "total_betmgm_line_over"),
        ("fanduel", "total_fanduel_line_over"),
        ("caesars", "total_caesars_line_over"),
        ("bet365", "total_bet365_line_over"),
        ("draftkings", "total_draftkings_line_over"),
        ("fanatics_sportsbook", "total_fanatics_sportsbook_line_over"),
    ]

    total_selected_col = f"total_{total_line_book}_line_over"

    if total_lines_mode == "selected":
        required_cols.append(total_selected_col)

    elif total_lines_mode == "all":
        required_cols.append(total_selected_col)
        # Only require sources actually present in df_odds -- a book can be
        # legitimately absent (e.g. Caesars, once excluded/merged upstream via
        # combine_caesars_and_fanatics) without that being an error.
        required_cols.extend(
            [
                col
                for _, col in known_total_sources
                if col != total_selected_col and col in df_odds.columns
            ]
        )

    # Check required columns exist
    missing_cols = [c for c in required_cols if c not in df_odds.columns]
    if missing_cols:
        print(f"Warning: Missing columns in odds dataframe: {missing_cols}")
        raise ValueError(f"Missing columns in odds dataframe: {missing_cols}")
        # return df_team

    # Subset odds
    df_odds_subset = df_odds[required_cols].copy()

    # Filter out Yahoo-specific columns if requested
    if exclude_yahoo:
        yahoo_cols = [
            "total_pct_bets_over",
            "total_pct_bets_under",
            "total_pct_money_over",
            "total_pct_money_under",
            "spread_pct_bets_away",
            "spread_pct_bets_home",
            "spread_pct_money_away",
            "spread_pct_money_home",
            "moneyline_pct_bets_away",
            "moneyline_pct_bets_home",
            "moneyline_pct_money_away",
            "moneyline_pct_money_home",
        ]
        yahoo_cols_to_drop = [
            col for col in yahoo_cols if col in df_odds_subset.columns
        ]
        if yahoo_cols_to_drop:
            df_odds_subset = df_odds_subset.drop(columns=yahoo_cols_to_drop)

    # Rename
    selected_spread_col = spread_col(book)
    selected_moneyline_col = moneyline_col(book)
    rename_map = {
        "game_id": "GAME_ID",
        spread_home_col: f"{selected_spread_col}_HOME",
        spread_away_col: f"{selected_spread_col}_AWAY",
        ml_home_col: f"{selected_moneyline_col}_HOME",
        ml_away_col: f"{selected_moneyline_col}_AWAY",
    }

    if total_lines_mode in {"selected", "all"}:
        rename_map[total_selected_col] = total_line_col(total_line_book)

    if total_lines_mode == "all":
        for src, col in known_total_sources:
            if col == total_selected_col:
                continue  # already mapped
            rename_map[col] = total_line_col(src)

    df_odds_subset = df_odds_subset.rename(columns=rename_map)

    # Merge
    df_merged = df_team.merge(
        df_odds_subset,
        on="GAME_ID",
        how="left",
        suffixes=("", "_odds"),
    )

    # Vectorized assignment for selected spread and moneyline columns
    if "HOME" in df_merged.columns:
        home_mask = df_merged["HOME"].astype(bool)
        df_merged[selected_spread_col] = np.where(
            home_mask,
            df_merged[f"{selected_spread_col}_HOME"],
            df_merged[f"{selected_spread_col}_AWAY"],
        )
        df_merged[selected_moneyline_col] = np.where(
            home_mask,
            df_merged[f"{selected_moneyline_col}_HOME"],
            df_merged[f"{selected_moneyline_col}_AWAY"],
        )
    else:
        print("Warning: 'HOME' column not found in team dataframe")
        df_merged[selected_spread_col] = df_merged[f"{selected_spread_col}_HOME"]
        df_merged[selected_moneyline_col] = df_merged[f"{selected_moneyline_col}_HOME"]

    # Drop temp columns
    df_merged = df_merged.drop(
        columns=[
            f"{selected_spread_col}_HOME",
            f"{selected_spread_col}_AWAY",
            f"{selected_moneyline_col}_HOME",
            f"{selected_moneyline_col}_AWAY",
        ],
        errors="ignore",
    )

    print(
        f"Merged {len(df_merged)} rows with odds data from {book} "
        f"(total_line_book={total_line_book}) "
        f"(total_lines_mode={total_lines_mode}, exclude_yahoo={exclude_yahoo})"
    )
    return df_merged


def merge_remaining_odds_by_game_id(
    df_odds: pd.DataFrame,
    df_merged: pd.DataFrame,
    exclude_books: list[str] | None = None,
    exclude_yahoo: bool = False,
    exclude_total_lines: bool = True,
) -> pd.DataFrame:
    """
    Merge all remaining odds data with merged home/away dataframe using game_id.

    This function should be called AFTER merge_home_away_data(), which creates one row
    per game. It merges all odds columns except those already merged by
    merge_total_spread_moneyline_by_game_id().

    Args:
        df_odds: Merged odds dataframe (from Yahoo + Sportsbook merge)
        df_merged: Merged home/away dataframe (one row per game) with GAME_ID column
        exclude_books: List of books to exclude specific columns for (default: none)
            This will exclude total_X_line_over/under, spread_X_line_home/away,
            ml_X_price_home/away for each book in the list.
        exclude_yahoo: If True, exclude Yahoo-specific betting columns (pct_bets, pct_money)
                       but keep metadata columns (game_date, teams, etc.)
        exclude_total_lines: If True (default), exclude total line columns that were already merged
                            by merge_total_spread_moneyline_by_game_id (only if they exist in df_merged).
                            This prevents duplicates while allowing flexibility for different merge modes.

    Returns:
        pd.DataFrame: Merged dataframe with all remaining odds columns added
    """
    if exclude_books is None:
        exclude_books = []
    elif isinstance(exclude_books, str):
        exclude_books = [exclude_books]

    if df_odds.empty:
        print("Warning: Odds dataframe is empty")
        return df_merged

    # Ensure game_id column exists in both dataframes
    if "game_id" not in df_odds.columns:
        print("Warning: 'game_id' column not found in odds dataframe")
        return df_merged

    if "GAME_ID" not in df_merged.columns:
        print("Warning: 'GAME_ID' column not found in merged dataframe")
        return df_merged

    # Build list of columns to exclude
    # For the primary book (exclude_books), only exclude columns that were RENAMED
    # in merge_total_spread_moneyline_by_game_id:
    #   - spread_{book}_line_home/away → renamed to SPREAD_<book>
    #   - ml_{book}_price_home/away → renamed to MONEYLINE_<book>
    # Total lines are handled separately by exclude_total_lines.
    # Other columns (prices, opener prices, etc.) are allowed through because:
    #   1. They have different names from the _TEAM_HOME/_TEAM_AWAY versions
    #   2. They're needed by engineer_odds_features at game level
    columns_to_exclude = set()
    for book in exclude_books:
        columns_to_exclude.update(
            [
                f"spread_{book}_line_home",
                f"spread_{book}_line_away",
                f"ml_{book}_price_home",
                f"ml_{book}_price_away",
            ]
        )

    # Exclude total line columns that were already merged (to prevent duplicates)
    # Only exclude if the corresponding uppercase column already exists in df_merged
    if exclude_total_lines:
        known_total_sources = [
            "consensus_opener",
            "betmgm",
            "fanduel",
            "caesars",
            "bet365",
            "draftkings",
            "fanatics_sportsbook",
        ]
        for book in known_total_sources:
            # Check if the uppercase version already exists in df_merged
            uppercase_col = total_line_col(book)
            if uppercase_col in df_merged.columns:
                # Only exclude if it was already merged
                columns_to_exclude.update(
                    [
                        f"total_{book}_line_over",
                        f"total_{book}_line_under",
                    ]
                )

    # Also exclude metadata columns that are already in df_merged
    columns_to_exclude.update(
        [
            "game_date",
            "season_year",
            "team_home",
            "team_away",
        ]
    )

    # Note: We do NOT exclude prices, consensus %, or Yahoo % here.
    # These columns were already merged at team-level by merge_odds_percentages_and_prices_by_game_id
    # (for rolling stats), but those team-level versions have _TEAM_HOME/_TEAM_AWAY suffixes
    # after merge_home_away_data — different names from the raw game-level columns here.
    # The raw game-level columns are needed by engineer_odds_features downstream.
    # select_training_columns handles which versions to keep.

    # Exclude Yahoo-specific betting columns if requested
    if exclude_yahoo:
        yahoo_cols = [
            "total_pct_bets_over",
            "total_pct_bets_under",
            "total_pct_money_over",
            "total_pct_money_under",
            "spread_pct_bets_away",
            "spread_pct_bets_home",
            "spread_pct_money_away",
            "spread_pct_money_home",
            "moneyline_pct_bets_away",
            "moneyline_pct_bets_home",
            "moneyline_pct_money_away",
            "moneyline_pct_money_home",
        ]
        columns_to_exclude.update(yahoo_cols)

    # Select columns from odds to merge (all except excluded and already-present columns).
    # Skipping pre-existing columns prevents pandas from creating duplicated *_odds columns
    # when this function is called on a dataframe that already contains some game-level odds.
    existing_cols = set(df_merged.columns)
    skipped_existing_cols = []
    cols_to_merge = ["game_id"]  # Always include game_id
    for col in df_odds.columns:
        if col == "game_id" or col in columns_to_exclude:
            continue
        if col in existing_cols:
            skipped_existing_cols.append(col)
            continue
        cols_to_merge.append(col)

    # Check if there are any columns to merge besides game_id
    if len(cols_to_merge) <= 1:
        print("Warning: No remaining odds columns to merge after exclusions")
        return df_merged

    # Log what types of columns are being merged (for verification)
    spread_lines = [
        c for c in cols_to_merge if c.startswith("spread_") and "_line_" in c
    ]
    total_lines = [c for c in cols_to_merge if c.startswith("total_") and "_line_" in c]
    other_cols = [
        c for c in cols_to_merge if c not in spread_lines + total_lines + ["game_id"]
    ]

    if len(cols_to_merge) > 1:
        print(
            f"Merging remaining odds: {len(spread_lines)} spread lines, "
            f"{len(total_lines)} total lines, {len(other_cols)} other columns"
        )
    if skipped_existing_cols:
        print(
            f"Skipped {len(skipped_existing_cols)} already-existing odds column(s) "
            f"to avoid duplicate merge columns."
        )

    df_odds_subset = df_odds[cols_to_merge].copy()

    # Rename game_id to GAME_ID for merging
    df_odds_subset = df_odds_subset.rename(columns={"game_id": "GAME_ID"})

    # Merge with merged dataframe
    df_result = df_merged.merge(
        df_odds_subset,
        on="GAME_ID",
        how="left",
        suffixes=("", "_odds"),
    )

    return df_result
