import pandas as pd
from nba_ou.config.yahoo_features import YAHOO_RAW_SOURCES
from nba_ou.postgre_db.odds.merge_odds_data import (
    merge_yahoo_sportsbook_odds,
)


def _as_history_floats(predict: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """``predict`` with each column that is ``float64`` in ``history`` as ``float64``.

    Tonight's scraped odds mark a missing line as ``pd.NA``, which makes the
    column ``object`` -- and, once concatenated, the stored seasons' column
    too. The column's dtype then followed tonight's rows, and with it the
    history memo's digest of every season. Stored odds hold NaN, so tonight's
    rows take it as well. A column whose values are not numbers is left as it
    was.
    """
    out = predict.copy()
    for column in predict.columns:
        if column not in history.columns or history[column].dtype != "float64":
            continue
        if out[column].dtype == "float64":
            continue
        values = out[column].astype("object").where(out[column].notna(), float("nan"))
        try:
            out[column] = values.astype("float64")
        except (TypeError, ValueError):
            pass
    return out


def merge_and_validate_scheduled_odds(
    df_odds: pd.DataFrame,
    df_odds_yahoo: pd.DataFrame,
    df_odds_sportsbook: pd.DataFrame,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
) -> pd.DataFrame:
    """Merge and validate scheduled odds with historical odds data.

    This function merges Yahoo and Sportsbook odds for scheduled games, validates
    column consistency, and preserves missing quotes. Each model applies its
    own missing-feature budget when selecting prediction rows.

    Args:
        df_odds (pd.DataFrame): Historical odds data from database
        df_odds_yahoo (pd.DataFrame): Yahoo odds for scheduled games
        df_odds_sportsbook (pd.DataFrame): Sportsbook odds for scheduled games
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

    # A scheduled game has one odds row: tonight's. The stored odds never hold
    # it before tip-off, but a replay of a played date reads them after.
    if "game_id" in df_odds.columns and "game_id" in df_odds_predict.columns:
        scheduled_ids = set(df_odds_predict["game_id"].astype(str))
        df_odds = df_odds[~df_odds["game_id"].astype(str).isin(scheduled_ids)]

    df_odds_predict = _as_history_floats(df_odds_predict, df_odds)

    # Concatenate and sort. Stable, so a date's games keep the loader's order:
    # an unstable sort put them in an order that depended on every other date,
    # tonight's included, and so changed from one day to the next.
    df_odds_combined = pd.concat([df_odds, df_odds_predict], ignore_index=True)
    df_odds_combined.sort_values(
        by="game_date", inplace=True, ascending=False, kind="stable"
    )
    df_odds_combined.reset_index(drop=True, inplace=True)

    return df_odds_combined
