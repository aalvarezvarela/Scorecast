import pandas as pd
from nba_ou.config.yahoo_features import (
    YAHOO_RAW_SOURCES,
    is_yahoo_percentage_column,
)
from nba_ou.postgre_db.odds.merge_odds_data import (
    merge_yahoo_sportsbook_odds,
)


def _log_nan_columns(df: pd.DataFrame) -> None:
    """Log which columns have NaN values and how many rows are affected.

    Args:
        df (pd.DataFrame): DataFrame to inspect for NaN values.
    """
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if nan_cols.empty:
        return
    print("  Columns with NaN values (count of affected rows):")
    for col, count in nan_cols.items():
        print(f"    - {col}: {count}/{len(df)} rows")


def merge_and_validate_scheduled_odds(
    df_odds: pd.DataFrame,
    df_odds_yahoo: pd.DataFrame,
    df_odds_sportsbook: pd.DataFrame,
    strict_mode: int = 0,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
) -> pd.DataFrame:
    """Merge and validate scheduled odds with historical odds data.

    This function merges Yahoo and Sportsbook odds for scheduled games, validates
    column consistency, and optionally checks for null values in strict mode.

    When strict mode would remove ALL prediction rows (e.g. because optional columns
    such as public-betting percentages are unavailable for play-in / playoff games),
    a warning is logged and the rows are kept so the pipeline can still run.

    Args:
        df_odds (pd.DataFrame): Historical odds data from database
        df_odds_yahoo (pd.DataFrame): Yahoo odds for scheduled games
        df_odds_sportsbook (pd.DataFrame): Sportsbook odds for scheduled games
        strict_mode (int, optional): Maximum number of columns allowed to have NaN/None values
            per row.  Rows exceeding this threshold are dropped.
            Use 0 for no columns with nulls allowed, -1 or any negative value to disable the
            check entirely.  When ALL prediction rows would be removed the strict-mode filter is
            skipped with a warning so the pipeline can still produce predictions.  Default is 0.
        normalize_total_lines (bool): Whether to center asymmetrically priced
            scheduled total markets. Defaults to True.
        normalize_spread_lines (bool): Whether to center asymmetrically priced
            scheduled spread markets. Defaults to True.
        null_extreme_spread_prices (bool): Whether to set implausibly extreme
            scheduled spread prices to NaN before centering. Defaults to True.

    Returns:
        pd.DataFrame: Combined odds dataframe with historical and scheduled games

    Raises:
        ValueError: If column validation fails (prediction data is missing columns that exist
            in the historical odds).
    """
    # Merge Yahoo and Sportsbook scheduled odds
    df_odds_predict = merge_yahoo_sportsbook_odds(
        df_odds_yahoo,
        df_odds_sportsbook,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
    )

    # An unavailable Yahoo feed is optional. Keep historical schema columns
    # present with NaN instead of failing column validation for today's games.
    missing_yahoo = {
        c: float("nan")
        for c in YAHOO_RAW_SOURCES
        if c in df_odds.columns and c not in df_odds_predict.columns
    }
    if missing_yahoo:
        df_odds_predict = df_odds_predict.assign(**missing_yahoo)

    # Validate columns
    df_odds_cols = set(df_odds.columns)
    df_odds_predict_cols = set(df_odds_predict.columns)

    # Check if df_odds has columns that df_odds_predict doesn't have
    missing_in_predict = df_odds_cols - df_odds_predict_cols
    if missing_in_predict:
        raise ValueError(
            f"df_odds_predict is missing columns that exist in df_odds: {missing_in_predict}"
        )

    # Drop extra columns from df_odds_predict that are not in df_odds
    extra_in_predict = df_odds_predict_cols - df_odds_cols
    if extra_in_predict:
        print(f"Dropping extra columns from df_odds_predict: {extra_in_predict}")
        df_odds_predict = df_odds_predict.drop(columns=list(extra_in_predict))

    # Strict mode: check for NaN or None values
    if strict_mode >= 0:
        # Count NaNs per row
        required_columns = [
            c for c in df_odds_predict.columns if not is_yahoo_percentage_column(c)
        ]
        nan_counts_per_row = df_odds_predict[required_columns].isnull().sum(axis=1)

        # Rows exceeding strict mode threshold
        rows_exceeding = nan_counts_per_row > strict_mode
        num_rows_exceeding = rows_exceeding.sum()

        if num_rows_exceeding > 0:
            print(
                f"\nStrict mode: Found {num_rows_exceeding} rows with NaNs in more than {strict_mode} columns"
            )
            _log_nan_columns(df_odds_predict)

            if num_rows_exceeding == len(df_odds_predict):
                print(
                    "WARNING: Strict mode would remove every scheduled odds "
                    "row; retaining them so predictions can still be generated."
                )
            else:
                print(
                    f"Dropping {num_rows_exceeding} scheduled odds row(s) that "
                    "exceed the strict-mode threshold."
                )
                df_odds_predict = df_odds_predict.loc[~rows_exceeding].copy()

    # Concatenate and sort
    df_odds_combined = pd.concat([df_odds, df_odds_predict], ignore_index=True)
    df_odds_combined.sort_values(by="game_date", inplace=True, ascending=False)
    df_odds_combined.reset_index(drop=True, inplace=True)

    return df_odds_combined
