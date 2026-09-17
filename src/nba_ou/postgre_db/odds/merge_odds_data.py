"""
Module for merging Yahoo and Sportsbook odds data.

This module provides functions to:
- Load odds data from both Yahoo and Sportsbook databases
- Merge them based on game_id
- Fill missing BetMGM values with Yahoo data
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from nba_ou.data_processing.odds.book_combination import (
    combine_caesars_and_fanatics,
    resolve_combine_books,
)
from nba_ou.data_processing.odds.closing_line_repair import (
    audit_repaired_closes,
    print_repair_summary,
    repair_closing_lines,
)
from nba_ou.data_processing.odds.normalize_total_lines import (
    normalize_total_lines_inplace,
)
from nba_ou.data_processing.odds.normalize_spread_lines import (
    normalize_spread_lines_inplace,
)
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db.fetch_data_from_odds_sportsbook_db import (
    load_odds_sportsbook_from_db,
)
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_closing_ticks
from nba_ou.postgre_db.odds_yahoo.fetch_data_from_db.fetch_data_from_odds_yahoo_db import (
    load_odds_yahoo_from_db,
)


def merge_yahoo_sportsbook_odds(
    df_yahoo: pd.DataFrame,
    df_sportsbook: pd.DataFrame,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    closing_line_repair: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Merge Yahoo and Sportsbook odds data.

    This function:
    - Merges both dataframes on game_id
    - Removes duplicate columns (team names, dates) keeping sportsbook versions
    - Fills missing BetMGM values with Yahoo data:
      - total_betmgm_line_over <- total_line
      - spread_betmgm_line_home <- spread_home
      - spread_betmgm_line_away <- spread_away
      - ml_betmgm_price_home <- moneyline_home
      - ml_betmgm_price_away <- moneyline_away

    Args:
        df_yahoo (pd.DataFrame): Yahoo odds data
        df_sportsbook (pd.DataFrame): Sportsbook odds data
        normalize_total_lines (bool): If True, replace asymmetrically priced
            total markets with their estimated 50/50 line and -110/-110 prices.
        normalize_spread_lines (bool): If True, replace asymmetrically priced
            spread markets with their estimated 50/50 line and -110/-110 prices.
        null_extreme_spread_prices (bool): If True, set implausibly extreme
            spread price cells to NaN before spread centering.
        closing_line_repair: Applied to the merged frame while prices are
            still American, after the Yahoo fill and before centering, so a
            Yahoo-filled value is verified like any other.

    Returns:
        pd.DataFrame: Merged odds dataframe
    """
    if df_yahoo.empty and df_sportsbook.empty:
        return pd.DataFrame()

    if df_yahoo.empty:
        print("Yahoo odds data is empty, returning sportsbook data only")
        return _convert_prices_and_normalize_markets(
            df_sportsbook.copy(),
            normalize_total_lines=normalize_total_lines,
            normalize_spread_lines=normalize_spread_lines,
            null_extreme_spread_prices=null_extreme_spread_prices,
            closing_line_repair=closing_line_repair,
        )

    if df_sportsbook.empty:
        print("Sportsbook odds data is empty, returning yahoo data only")
        return _convert_prices_and_normalize_markets(
            df_yahoo.copy(),
            normalize_total_lines=normalize_total_lines,
            normalize_spread_lines=normalize_spread_lines,
            null_extreme_spread_prices=null_extreme_spread_prices,
            closing_line_repair=closing_line_repair,
        )

    # Select all relevant columns from yahoo for merging
    # Include all Yahoo columns except duplicate team name columns
    yahoo_cols_to_exclude = [
        "team_home",
        "team_away",
        "team_home_abbr",
        "team_away_abbr",
    ]

    yahoo_cols_to_merge = [
        col
        for col in df_yahoo.columns
        if col not in yahoo_cols_to_exclude or col == "game_id"
    ]

    # Filter yahoo to only include columns that exist
    yahoo_cols_available = [
        col for col in yahoo_cols_to_merge if col in df_yahoo.columns
    ]
    df_yahoo_subset = df_yahoo[yahoo_cols_available].copy()

    # Merge on game_id
    df_merged = df_sportsbook.merge(
        df_yahoo_subset,
        on="game_id",
        how="outer",
        suffixes=("", "_yahoo"),
    )

    # Coalesce game_date and season_year - fill NaN from one source with the other
    if "game_date_yahoo" in df_merged.columns:
        df_merged["game_date"] = df_merged["game_date"].fillna(
            df_merged["game_date_yahoo"]
        )
        df_merged = df_merged.drop(columns=["game_date_yahoo"])

    if "season_year_yahoo" in df_merged.columns:
        df_merged["season_year"] = df_merged["season_year"].fillna(
            df_merged["season_year_yahoo"]
        )
        df_merged = df_merged.drop(columns=["season_year_yahoo"])

    # Fill missing BetMGM values with Yahoo data
    # Total line - fill BetMGM with Yahoo total_line
    if "total_line" in df_merged.columns:
        if "total_betmgm_line_over" in df_merged.columns:
            df_merged["total_betmgm_line_over"] = df_merged[
                "total_betmgm_line_over"
            ].fillna(df_merged["total_line"])
        if "total_betmgm_line_under" in df_merged.columns:
            df_merged["total_betmgm_line_under"] = df_merged[
                "total_betmgm_line_under"
            ].fillna(df_merged["total_line"])
        # Drop yahoo column after filling since it's redundant with BetMGM
        df_merged = df_merged.drop(columns=["total_line"])

    # Spread home/away - fill BetMGM with Yahoo spread values
    if (
        "spread_home" in df_merged.columns
        and "spread_betmgm_line_home" in df_merged.columns
    ):
        df_merged["spread_betmgm_line_home"] = df_merged[
            "spread_betmgm_line_home"
        ].fillna(df_merged["spread_home"])
        # Drop yahoo column after filling since it's redundant with BetMGM
        df_merged = df_merged.drop(columns=["spread_home"])

    if (
        "spread_away" in df_merged.columns
        and "spread_betmgm_line_away" in df_merged.columns
    ):
        df_merged["spread_betmgm_line_away"] = df_merged[
            "spread_betmgm_line_away"
        ].fillna(df_merged["spread_away"])
        # Drop yahoo column after filling since it's redundant with BetMGM
        df_merged = df_merged.drop(columns=["spread_away"])

    # Moneyline home/away - fill BetMGM with Yahoo moneyline values
    if (
        "moneyline_home" in df_merged.columns
        and "ml_betmgm_price_home" in df_merged.columns
    ):
        df_merged["ml_betmgm_price_home"] = df_merged["ml_betmgm_price_home"].fillna(
            df_merged["moneyline_home"]
        )
        # Drop yahoo column after filling since it's redundant with BetMGM
        df_merged = df_merged.drop(columns=["moneyline_home"])

    if (
        "moneyline_away" in df_merged.columns
        and "ml_betmgm_price_away" in df_merged.columns
    ):
        df_merged["ml_betmgm_price_away"] = df_merged["ml_betmgm_price_away"].fillna(
            df_merged["moneyline_away"]
        )
        # Drop yahoo column after filling since it's redundant with BetMGM
        df_merged = df_merged.drop(columns=["moneyline_away"])

    # Keep all other Yahoo columns as they provide unique information
    # These include:
    # - total_pct_bets_over, total_pct_bets_under
    # - total_pct_money_over, total_pct_money_under
    # - spread_pct_bets_away, spread_pct_bets_home
    # - spread_pct_money_away, spread_pct_money_home
    # - moneyline_pct_bets_away, moneyline_pct_bets_home
    # - moneyline_pct_money_away, moneyline_pct_money_home

    print(f"Merged odds data: {len(df_merged)} rows")
    # sort by date, newest first
    if "game_date" in df_merged.columns:
        df_merged = df_merged.sort_values(by="game_date", ascending=False).reset_index(
            drop=True
        )

    # Drop redundant columns
    cols_to_drop = [
        "team_home",
        "team_away",
        "home_points",
        "away_points",
        "total_points",
    ]
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_merged.columns]
    if cols_to_drop_existing:
        df_merged = df_merged.drop(columns=cols_to_drop_existing)

    return _convert_prices_and_normalize_markets(
        df_merged,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        closing_line_repair=closing_line_repair,
    )


def american_to_decimal_series(odds: pd.Series) -> pd.Series:
    """
    Convert American odds to decimal odds (payout multiplier, includes stake).
    Example: -110 -> 1.9091 ; +150 -> 2.5
    """
    o = pd.to_numeric(odds, errors="coerce").astype(float)

    dec = np.where(
        o > 0,
        1.0 + (o / 100.0),
        np.where(
            o < 0,
            1.0 + (100.0 / np.abs(o)),
            np.nan,  # o == 0 invalid
        ),
    )
    return pd.Series(dec, index=odds.index)


def _convert_prices_and_normalize_markets(
    df_odds: pd.DataFrame,
    *,
    normalize_total_lines: bool,
    normalize_spread_lines: bool,
    null_extreme_spread_prices: bool,
    closing_line_repair: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Convert raw American prices, then optionally center total/spread markets."""
    if closing_line_repair is not None:
        df_odds = closing_line_repair(df_odds)
    price_cols = [column for column in df_odds.columns if "_price" in column]
    for column in price_cols:
        df_odds[column] = american_to_decimal_series(df_odds[column])

    normalize_total_lines_inplace(
        df_odds,
        enabled=normalize_total_lines,
        odds_format="decimal",
    )
    normalize_spread_lines_inplace(
        df_odds,
        enabled=normalize_spread_lines,
        null_extreme_prices=null_extreme_spread_prices,
        odds_format="decimal",
    )
    return df_odds


def load_and_merge_odds_yahoo_sportsbookreview(
    season_years: list[str] | None = None,
    extra_game_ids=None,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    exclude_caesars: bool = False,
    combine_fanatics_and_caesars: bool | None = None,
    repair_closing_lines_from_history: bool = True,
) -> pd.DataFrame:
    """
    Load odds data from Yahoo and Sportsbook databases, merge them, and merge with games.

    This function:
    1. Loads Yahoo odds data
    2. Loads Sportsbook odds data
    3. Loads games data
    4. Merges Yahoo and Sportsbook odds
    5. Merges the combined odds with games to get game_id

    Args:
        season_years (list[str], optional): List of seasons to load (e.g., ["2023-24", "2024-25"])
        normalize_total_lines (bool): Whether to center asymmetrically priced
            total markets. Defaults to True.
        normalize_spread_lines (bool): Whether to center asymmetrically priced
            spread markets. Defaults to True.
        null_extreme_spread_prices (bool): Whether to set implausibly extreme
            spread prices to NaN before centering. Defaults to True.
        exclude_caesars (bool): If True, drop all Caesars-named odds columns.
            Default False.
        combine_fanatics_and_caesars (bool | None): Coalesce fanatics_sportsbook
            columns with Caesars (fanatics kept where present, Caesars fills
            NaNs) and drop the standalone Caesars columns. Defaults to None,
            which means "combine unless an exclusion was explicitly asked for"
            -- see resolve_combine_books in
            nba_ou.data_processing.odds.book_combination.
        repair_closing_lines_from_history (bool): Replace every per-book close
            with the last valid pre-tip quote in the Aiven line history, then
            clear cross-book outliers. SBR stores in-play values as closes for
            27% of games, so turning this off reintroduces that leak; it needs
            a line-history connection. See
            nba_ou.data_processing.odds.closing_line_repair. Default True.

    Returns:
        pd.DataFrame: Complete merged odds dataframe with game_id, all odds columns
    """
    print(f"Loading odds data for seasons: {season_years}")

    # Load data from databases
    df_yahoo = load_odds_yahoo_from_db(
        seasons=season_years, extra_game_ids=extra_game_ids
    )
    df_sportsbook = load_odds_sportsbook_from_db(
        seasons=season_years, extra_game_ids=extra_game_ids
    )

    # Handle None returns
    if df_yahoo is None:
        df_yahoo = pd.DataFrame()
    if df_sportsbook is None:
        df_sportsbook = pd.DataFrame()

    # If both odds sources are empty, try to return empty with games structure
    if df_yahoo.empty and df_sportsbook.empty:
        print("Warning: Both Yahoo and Sportsbook odds data are empty")
        return pd.DataFrame()

    # Merge Yahoo and Sportsbook odds
    df_odds_merged = merge_yahoo_sportsbook_odds(
        df_yahoo,
        df_sportsbook,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        closing_line_repair=(
            _repair_from_line_history if repair_closing_lines_from_history else None
        ),
    )

    df_odds_merged = combine_caesars_and_fanatics(
        df_odds_merged,
        exclude_caesars=exclude_caesars,
        combine_with_fanatics=resolve_combine_books(
            combine=combine_fanatics_and_caesars, exclude_caesars=exclude_caesars
        ),
    )

    print(f"Final merged odds: {len(df_odds_merged)} rows")

    return df_odds_merged


def _repair_from_line_history(df_odds: pd.DataFrame) -> pd.DataFrame:
    """Fetch closing ticks for the frame's seasons and apply the repair."""
    if df_odds.empty or "game_id" not in df_odds.columns:
        return df_odds
    frame = df_odds.reset_index(drop=True)
    seasons = sorted(
        int(season)
        for season in pd.to_numeric(frame["season_year"], errors="coerce")
        .dropna()
        .unique()
    )
    ticks = fetch_closing_ticks(season_years=seasons)
    repaired, audit = repair_closing_lines(
        frame,
        ticks,
        game_tick_loader=lambda game_ids: fetch_closing_ticks(
            game_ids=game_ids, last_n=None
        ),
    )
    audit_repaired_closes(repaired, audit)
    print_repair_summary(audit)
    return repaired


if __name__ == "__main__":
    # Test the merging function
    seasons = ["2023-24", "2024-25"]
    df_merged = load_and_merge_odds_yahoo_sportsbookreview(season_years=seasons)

    if not df_merged.empty:
        print(f"\nMerged odds shape: {df_merged.shape}")
        print(f"Columns: {df_merged.columns.tolist()}")
        print("\nFirst few rows:")
        print(df_merged.head())
    else:
        print("\nNo merged odds data available")
