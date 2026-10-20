"""Normalize asymmetrically priced NBA spreads to an estimated 50/50 line."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nba_ou.data_processing.line_history.normalization import (
    MARGIN_SIGMA,
    center_two_way_line,
)
from nba_ou.data_processing.odds.normalize_total_lines import (
    DEFAULT_CENTERED_AMERICAN_ODDS,
    DEFAULT_PRICE_DECIMAL_PLACES,
    DEFAULT_QUOTE_INCREMENT,
    odds_to_decimal,
)

DEFAULT_MIN_SPREAD_PRICE_DECIMAL = 1.50
DEFAULT_MAX_SPREAD_PRICE_DECIMAL = 2.50
DEFAULT_MAX_REASONABLE_SPREAD_ABS = 60.0


def _spread_market_prefixes(column_names: pd.Index | list[str]) -> list[str]:
    prefixes: list[str] = []
    for column in column_names:
        if not column.endswith("_line_home"):
            continue
        prefix = column.removesuffix("_line_home")
        required_columns = {
            f"{prefix}_line_home",
            f"{prefix}_line_away",
            f"{prefix}_price_home",
            f"{prefix}_price_away",
        }
        if required_columns.issubset(column_names):
            prefixes.append(prefix)
    return prefixes


def _decimal_prices(prices: pd.Series, *, odds_format: str) -> pd.Series:
    values = pd.to_numeric(prices, errors="coerce").astype("float64")
    if odds_format in {"decimal", "european", "eu"}:
        return values.where(values > 1.0)
    if odds_format in {"american", "us"}:
        decimal = pd.Series(np.nan, index=values.index, dtype="float64")
        positive = values > 0.0
        negative = values < 0.0
        decimal[positive] = 1.0 + values[positive] / 100.0
        decimal[negative] = 1.0 + 100.0 / values[negative].abs()
        return decimal
    raise ValueError("odds_format must be 'decimal' or 'american'.")


def spread_price_extreme_mask(
    prices: pd.Series,
    *,
    odds_format: str = "decimal",
    min_decimal: float = DEFAULT_MIN_SPREAD_PRICE_DECIMAL,
    max_decimal: float = DEFAULT_MAX_SPREAD_PRICE_DECIMAL,
) -> pd.Series:
    """Rows where a spread price is too extreme to trust as ordinary juice.

    The default range is intentionally broad: decimal 1.50 to 2.50, roughly
    American -200 to +150. Prices outside it are often alternate-line or
    wrong-market bleed rather than the two-way price for the displayed spread.
    """
    decimal = _decimal_prices(prices, odds_format=odds_format)
    present = pd.to_numeric(prices, errors="coerce").notna()
    return present & (decimal.lt(min_decimal) | decimal.gt(max_decimal))


def _centered_price(*, odds_format: str, centered_american_odds: float) -> float:
    if odds_format in {"decimal", "european", "eu"}:
        return odds_to_decimal(centered_american_odds, odds_format="american")
    if odds_format in {"american", "us"}:
        return centered_american_odds
    raise ValueError("odds_format must be 'decimal' or 'american'.")


def normalize_spread_lines_inplace(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    null_extreme_prices: bool = True,
    null_invalid_line_pairs: bool = True,
    odds_format: str = "decimal",
    sigma: float = MARGIN_SIGMA,
    quote_increment: float = DEFAULT_QUOTE_INCREMENT,
    centered_american_odds: float = DEFAULT_CENTERED_AMERICAN_ODDS,
    price_decimal_places: int = DEFAULT_PRICE_DECIMAL_PLACES,
    min_price_decimal: float = DEFAULT_MIN_SPREAD_PRICE_DECIMAL,
    max_price_decimal: float = DEFAULT_MAX_SPREAD_PRICE_DECIMAL,
    max_spread_abs: float = DEFAULT_MAX_REASONABLE_SPREAD_ABS,
) -> pd.DataFrame:
    """Center every complete, asymmetrically priced bookmaker spread in ``df``.

    Raw closing spread columns are home/away handicaps. The centering helper
    works in canonical home-margin space, so each home handicap is negated
    before centering and negated back when written into the raw columns.
    """
    if df.empty:
        return df
    if min_price_decimal <= 1.0 or max_price_decimal <= min_price_decimal:
        raise ValueError("price decimal bounds must satisfy 1 < min < max.")

    total_comparable_markets = 0
    total_changed_markets = 0
    total_extreme_prices = 0
    total_invalid_line_pairs = 0
    centered_price = round(
        _centered_price(
            odds_format=odds_format,
            centered_american_odds=centered_american_odds,
        ),
        price_decimal_places,
    )

    for prefix in _spread_market_prefixes(df.columns):
        line_home_col = f"{prefix}_line_home"
        line_away_col = f"{prefix}_line_away"
        price_home_col = f"{prefix}_price_home"
        price_away_col = f"{prefix}_price_away"

        line_home = pd.to_numeric(df[line_home_col], errors="coerce")
        line_away = pd.to_numeric(df[line_away_col], errors="coerce")
        price_home = pd.to_numeric(df[price_home_col], errors="coerce").round(
            price_decimal_places
        )
        price_away = pd.to_numeric(df[price_away_col], errors="coerce").round(
            price_decimal_places
        )

        df.loc[price_home.notna(), price_home_col] = price_home.loc[
            price_home.notna()
        ]
        df.loc[price_away.notna(), price_away_col] = price_away.loc[
            price_away.notna()
        ]

        if null_invalid_line_pairs:
            both_lines = line_home.notna() & line_away.notna()
            invalid_line_pair = both_lines & (
                line_home.abs().gt(max_spread_abs)
                | line_away.abs().gt(max_spread_abs)
                | ~np.isclose(line_home + line_away, 0.0, rtol=0.0, atol=1e-10)
            )
            if invalid_line_pair.any():
                df.loc[
                    invalid_line_pair,
                    [line_home_col, line_away_col, price_home_col, price_away_col],
                ] = np.nan
                total_invalid_line_pairs += int(invalid_line_pair.sum())
                line_home = line_home.mask(invalid_line_pair)
                line_away = line_away.mask(invalid_line_pair)
                price_home = price_home.mask(invalid_line_pair)
                price_away = price_away.mask(invalid_line_pair)

        if null_extreme_prices:
            extreme_home = spread_price_extreme_mask(
                price_home,
                odds_format=odds_format,
                min_decimal=min_price_decimal,
                max_decimal=max_price_decimal,
            )
            extreme_away = spread_price_extreme_mask(
                price_away,
                odds_format=odds_format,
                min_decimal=min_price_decimal,
                max_decimal=max_price_decimal,
            )
            extreme_price = extreme_home | extreme_away
            if extreme_price.any():
                df.loc[extreme_home, price_home_col] = np.nan
                df.loc[extreme_away, price_away_col] = np.nan
                total_extreme_prices += int(extreme_price.sum())
                price_home = price_home.mask(extreme_home)
                price_away = price_away.mask(extreme_away)

        comparable_mask = (
            line_home.notna()
            & line_away.notna()
            & price_home.notna()
            & price_away.notna()
            & line_home.abs().le(max_spread_abs)
            & line_away.abs().le(max_spread_abs)
            & np.isclose(line_home + line_away, 0.0, rtol=0.0, atol=1e-10)
        )
        total_comparable_markets += int(np.count_nonzero(comparable_mask))

        if not enabled:
            continue

        candidate_mask = comparable_mask & ~np.isclose(
            price_home, price_away, rtol=0.0, atol=1e-10
        )
        if not candidate_mask.any():
            continue

        canonical_line = -line_home
        centered_line = center_two_way_line(
            canonical_line,
            price_away,
            price_home,
            sigma=sigma,
            odds_format=odds_format,
            quote_increment=quote_increment,
            left_wins_above=False,
        )
        update_mask = candidate_mask & centered_line.notna()
        if not update_mask.any():
            continue

        df.loc[update_mask, line_home_col] = -centered_line.loc[update_mask]
        df.loc[update_mask, line_away_col] = centered_line.loc[update_mask]
        df.loc[update_mask, price_home_col] = centered_price
        df.loc[update_mask, price_away_col] = centered_price
        total_changed_markets += int(update_mask.sum())

    changed_percentage = (
        100.0 * total_changed_markets / total_comparable_markets
        if total_comparable_markets
        else 0.0
    )
    print(
        "Spread-line normalization: "
        f"{total_changed_markets}/{total_comparable_markets} bookmaker-game "
        f"markets changed ({changed_percentage:.2f}%). "
        f"Extreme-price rows nulled: {total_extreme_prices}. "
        f"Invalid line-pair rows nulled: {total_invalid_line_pairs}."
    )

    return df
