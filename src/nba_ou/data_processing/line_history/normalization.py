"""Center asymmetrically priced snapshot quotes onto a -110/-110 equivalent.

Why this exists at all: a book can move its *price* without moving its *line*,
so two quotes of "224.5" priced -105/-115 and -120/+100 are not the same market
view. Comparing them across books, or across snapshots of one book, is only
meaningful once each quote is restated as the line it would carry at a
symmetric -110. This mirrors what ``create_df_to_predict(normalize_total_lines=True)``
already does for the closing dataset, so a line at T means the same thing as a
line in the existing training data.

The estimator is the one already in
``nba_ou.data_processing.odds.normalize_total_lines``; only ``sigma`` and the
column plumbing are market-specific. This module adds two things that module
does not provide:

1. **Vectorisation.** The existing entry points are scalar and run a
   100-iteration bisection per integer-line quote. The snapshot dataset has
   hundreds of thousands of quotes, so both branches are done over numpy arrays
   here. ``tests/test_line_history_normalization.py`` asserts this path agrees
   with the scalar original.
2. **Signed lines.** ``round_to_increment`` rejects negatives and spreads are
   signed, so rounding is done on the magnitude and the sign reapplied -- i.e.
   half-away-from-zero, which is the symmetric choice for a mirrored market.

Moneyline is deliberately *not* given a centered line: there is no line to
shift, so normalisation there means devigging only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_ou.data_processing.odds.normalize_total_lines import (
    DEFAULT_CENTERED_AMERICAN_ODDS,
    DEFAULT_QUOTE_INCREMENT,
    DEFAULT_TOTAL_POINTS_SIGMA,
)

#: Std. dev. of the total around its fair line. Reused from the existing module
#: so both datasets center on identical assumptions.
TOTAL_SIGMA = DEFAULT_TOTAL_POINTS_SIGMA

#: Std. dev. of the game *margin* around its fair spread. Calibrated on this
#: repo's data rather than assumed: over 6,112 games from 2021-22 onward the
#: residual of (home margin - closing spread) has std 13.46, against a raw
#: margin std of 15.57. Recalibrate with
#: ``scripts/create_train_data/create_intermediate_line_train_data.py --calibrate-sigma``
#: if the seasons in scope change materially.
MARGIN_SIGMA = 13.46

CENTERED_AMERICAN_ODDS = DEFAULT_CENTERED_AMERICAN_ODDS
QUOTE_INCREMENT = DEFAULT_QUOTE_INCREMENT

_BISECTION_ITERATIONS = 100
_BISECTION_HALF_WIDTH_SIGMAS = 10.0


def american_to_decimal(prices: pd.Series) -> pd.Series:
    """Vectorised American -> decimal odds.

    The store holds American odds. The scalar helpers in the existing module
    default to ``odds_format="decimal"``, and a positive American price like
    ``+105`` would pass their ``> 1.0`` validity check and convert silently
    wrong -- so conversion is done explicitly here rather than relying on a
    format argument being remembered at every call site.
    """
    values = pd.to_numeric(prices, errors="coerce").astype("float64")
    # Zero is not a valid American price, and it is the one value where both
    # branches below would produce nonsense rather than NaN.
    values = values.mask(values == 0.0)

    decimal = pd.Series(np.nan, index=values.index, dtype="float64")
    positive = values > 0
    negative = values < 0
    decimal[positive] = 1.0 + values[positive] / 100.0
    decimal[negative] = 1.0 + 100.0 / values[negative].abs()
    return decimal


def devig_two_way(
    left_price: pd.Series,
    right_price: pd.Series,
    *,
    odds_format: str = "american",
) -> pd.DataFrame:
    """Fair two-way probabilities and the raw overround.

    Returns columns ``fair_left``, ``fair_right``, ``overround``. Rows missing
    either price yield NaN throughout rather than a one-sided guess.
    """
    if odds_format == "american":
        left_decimal = american_to_decimal(left_price)
        right_decimal = american_to_decimal(right_price)
    elif odds_format in {"decimal", "european", "eu"}:
        left_decimal = pd.to_numeric(left_price, errors="coerce").astype("float64")
        right_decimal = pd.to_numeric(right_price, errors="coerce").astype("float64")
        left_decimal = left_decimal.mask(left_decimal <= 1.0)
        right_decimal = right_decimal.mask(right_decimal <= 1.0)
    else:
        raise ValueError("odds_format must be 'american' or 'decimal'.")

    raw_left = 1.0 / left_decimal
    raw_right = 1.0 / right_decimal
    total = raw_left + raw_right

    return pd.DataFrame(
        {
            "fair_left": raw_left / total,
            "fair_right": raw_right / total,
            "overround": total - 1.0,
        },
        index=left_price.index,
    )


def round_to_increment_signed(
    values: pd.Series, increment: float = QUOTE_INCREMENT
) -> pd.Series:
    """Round to a quote increment, half away from zero.

    The existing ``round_to_increment`` raises on negatives, which spreads are.
    Rounding the magnitude and reapplying the sign keeps a mirrored market
    symmetric: a fair -3.25 and a fair +3.25 must land on -3.5 and +3.5, not on
    -3.0 and +3.5 as half-up would give.
    """
    if increment <= 0.0:
        raise ValueError("increment must be positive.")
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    magnitude = np.floor(numeric.abs() / increment + 0.5) * increment
    return pd.Series(
        np.sign(numeric) * magnitude, index=numeric.index, dtype="float64"
    )


def _center_half_point_lines(
    line: np.ndarray, fair_left: np.ndarray, sigma: float
) -> np.ndarray:
    """Closed form: no push is possible, so the quantile maps straight across."""
    return line + sigma * norm.ppf(fair_left)


def _center_integer_lines(
    line: np.ndarray, fair_left: np.ndarray, sigma: float
) -> np.ndarray:
    """Vectorised bisection for lines where an exact push can occur.

    Mirrors ``_estimate_integer_line_center``: the quoted probability is
    conditional on *not* pushing, so the mapping is solved numerically rather
    than inverted. The conditional is monotonically increasing in the center,
    which is what makes plain bisection valid.
    """
    low = line - _BISECTION_HALF_WIDTH_SIGMAS * sigma
    high = line + _BISECTION_HALF_WIDTH_SIGMAS * sigma

    for _ in range(_BISECTION_ITERATIONS):
        midpoint = (low + high) / 2.0
        left_win = 1.0 - norm.cdf((line + 0.5 - midpoint) / sigma)
        right_win = norm.cdf((line - 0.5 - midpoint) / sigma)
        conditional = left_win / (left_win + right_win)

        below = conditional < fair_left
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)

    return (low + high) / 2.0


def center_two_way_line(
    line: pd.Series,
    left_price: pd.Series,
    right_price: pd.Series,
    *,
    sigma: float,
    odds_format: str = "american",
    quote_increment: float = QUOTE_INCREMENT,
    left_wins_above: bool = True,
) -> pd.Series:
    """Restate a two-way quote as the line it would carry at -110/-110.

    ``left_wins_above`` says which side of the quote wins when the outcome comes
    in *above* the line, and it differs by market:

    * **Totals** (``True``) -- OVER is the left side and wins when the total
      exceeds the line.
    * **Spread** (``False``) -- the left side is the AWAY team, and the resolved
      line is the expected HOME margin. The away side covers when the margin
      comes in *below* the line, so the probability adjustment must run the
      other way.

    Getting this wrong is not a rounding difference, it inverts the correction:
    a quote of away +4.5 at -130 / home -4.5 at +110 means the market thinks the
    away side covers 54% of the time, so the fair margin is *below* 4.5 (~3.0).
    Treating it like a total pushes it to 6.0 -- wrong by 3 points and in the
    wrong direction. Symmetric -110/-110 quotes hide the bug entirely, which is
    why the tests use asymmetric prices.

    Rows with a missing line or either price return NaN: a one-sided quote
    carries no information about where the fair line sits.
    """
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive.")

    numeric_line = pd.to_numeric(line, errors="coerce").astype("float64")
    fair = devig_two_way(left_price, right_price, odds_format=odds_format)
    # The machinery below is written for "the side that wins above". When the
    # left side wins below, the right side is that side -- and because
    # Phi^-1(1-p) == -Phi^-1(p), substituting it flips the correction exactly,
    # in the push-aware integer branch as well as the closed-form one.
    fair_left = fair["fair_left"] if left_wins_above else fair["fair_right"]

    usable = numeric_line.notna() & fair_left.notna()
    # A fair probability of exactly 0 or 1 sends the quantile to infinity; such
    # a quote is degenerate rather than informative.
    usable &= fair_left.gt(0.0) & fair_left.lt(1.0)

    centered = pd.Series(np.nan, index=numeric_line.index, dtype="float64")
    if not usable.any():
        return centered

    line_values = numeric_line[usable].to_numpy()
    fair_values = fair_left[usable].to_numpy()

    is_integer = np.isclose(line_values, np.round(line_values), atol=1e-10)
    result = np.empty_like(line_values)
    if (~is_integer).any():
        result[~is_integer] = _center_half_point_lines(
            line_values[~is_integer], fair_values[~is_integer], sigma
        )
    if is_integer.any():
        result[is_integer] = _center_integer_lines(
            line_values[is_integer], fair_values[is_integer], sigma
        )

    centered[usable] = result
    return round_to_increment_signed(centered, quote_increment)


def center_totals(
    line: pd.Series,
    over_price: pd.Series,
    under_price: pd.Series,
    *,
    sigma: float = TOTAL_SIGMA,
    odds_format: str = "american",
) -> pd.Series:
    """Centered total. OVER is the "left" side."""
    return center_two_way_line(
        line, over_price, under_price, sigma=sigma, odds_format=odds_format
    )


def center_spread(
    line: pd.Series,
    left_price: pd.Series,
    right_price: pd.Series,
    *,
    sigma: float = MARGIN_SIGMA,
    odds_format: str = "american",
) -> pd.Series:
    """Centered spread, sign preserved.

    Two market-specific corrections, both of which a totals-shaped call would
    get wrong: the margin distribution's sigma (not the total's -- they differ
    by over two points), and ``left_wins_above=False``, because the away side
    covers when the home margin lands *below* the line.
    """
    return center_two_way_line(
        line,
        left_price,
        right_price,
        sigma=sigma,
        odds_format=odds_format,
        left_wins_above=False,
    )
