"""Tests for -110 centering of snapshot quotes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.line_history.normalization import (
    MARGIN_SIGMA,
    TOTAL_SIGMA,
    american_to_decimal,
    center_spread,
    center_totals,
    devig_two_way,
    round_to_increment_signed,
)
from nba_ou.data_processing.odds.normalize_total_lines import (
    estimate_centered_total_line,
)


def test_american_to_decimal_matches_definition():
    got = american_to_decimal(pd.Series([105.0, -125.0, -110.0, 200.0]))
    assert got.tolist() == pytest.approx([2.05, 1.8, 1 + 100 / 110, 3.0])


def test_zero_price_is_not_a_valid_american_quote():
    assert american_to_decimal(pd.Series([0.0])).isna().all()


def test_positive_american_price_would_be_silently_wrong_as_decimal():
    """The trap this module exists to close.

    A price of +105 is 2.05 in decimal. Read as if it were already decimal it
    passes the "> 1.0" validity check and sails through as 105.0 -- an implied
    probability of 0.95 instead of 0.49. Negative prices would at least raise;
    positive ones fail silently, which is why the American conversion is
    explicit rather than a default argument.
    """
    as_american = american_to_decimal(pd.Series([105.0])).iloc[0]
    read_as_decimal = 105.0
    assert as_american == pytest.approx(2.05)
    assert 1.0 / as_american == pytest.approx(0.4878, abs=1e-3)
    assert 1.0 / read_as_decimal == pytest.approx(0.0095, abs=1e-3)


def test_vectorised_centering_matches_scalar_original():
    """Half-point and integer lines both, since they take different branches."""
    rng = np.random.default_rng(7)
    lines = rng.choice([220.0, 224.5, 228.0, 231.5], 250)
    over = rng.choice([-130, -120, -115, -110, -105, 100, 110], 250)
    under = rng.choice([-130, -120, -115, -110, -105, 100, 110], 250)

    vectorised = center_totals(pd.Series(lines), pd.Series(over), pd.Series(under))
    scalar = [
        estimate_centered_total_line(line, o, u, odds_format="american")
        for line, o, u in zip(lines, over, under, strict=True)
    ]
    assert vectorised.to_numpy() == pytest.approx(np.array(scalar))


def test_symmetric_price_leaves_line_untouched():
    centered = center_totals(
        pd.Series([224.5]), pd.Series([-110.0]), pd.Series([-110.0])
    )
    assert centered.iloc[0] == pytest.approx(224.5)


def test_cheaper_over_pulls_the_total_up():
    """A shorter price on the over means the market thinks over is likelier."""
    centered = center_totals(
        pd.Series([224.5]), pd.Series([-130.0]), pd.Series([110.0])
    )
    assert centered.iloc[0] > 224.5


def test_signed_rounding_is_symmetric_about_zero():
    """Spreads are signed; half-up rounding would break the mirror."""
    got = round_to_increment_signed(pd.Series([3.25, -3.25, 0.2, -0.2]))
    assert got.tolist() == pytest.approx([3.5, -3.5, 0.0, -0.0])


def test_spread_centering_handles_negative_lines():
    """``round_to_increment`` rejects negatives; the signed path must not."""
    centered = center_spread(
        pd.Series([-5.5, 5.5]), pd.Series([-110.0, -110.0]), pd.Series([-110.0, -110.0])
    )
    assert centered.tolist() == pytest.approx([-5.5, 5.5])


def test_spread_uses_margin_sigma_not_total_sigma():
    """Using the total's sigma on a spread biases every centered spread."""
    assert MARGIN_SIGMA != TOTAL_SIGMA
    # A pronounced price asymmetry, so the two sigmas disagree by more than the
    # 0.5 quote increment they are both rounded to.
    line, left, right = pd.Series([5.5]), pd.Series([-200.0]), pd.Series([165.0])
    with_margin = center_spread(line, left, right).iloc[0]
    with_total = center_spread(line, left, right, sigma=TOTAL_SIGMA).iloc[0]
    assert with_margin != with_total


def test_devig_probabilities_sum_to_one_and_expose_overround():
    fair = devig_two_way(pd.Series([-110.0]), pd.Series([-110.0]))
    assert fair["fair_left"].iloc[0] + fair["fair_right"].iloc[0] == pytest.approx(1.0)
    assert fair["overround"].iloc[0] == pytest.approx(0.0476, abs=1e-3)


def test_one_sided_quote_yields_no_centered_line():
    centered = center_totals(pd.Series([224.5]), pd.Series([-110.0]), pd.Series([None]))
    assert centered.isna().all()


def test_spread_centering_direction_asymmetric_prices():
    """Regression: the away side covers when the margin lands BELOW the line.

    The stored spread is the expected HOME margin and ``left`` is the AWAY side,
    so a short away price must pull the fair margin DOWN. Treating a spread like
    a total inverts this. Symmetric -110/-110 quotes hide the bug completely,
    which is why every case here is asymmetric.
    """
    line = pd.Series([4.5, 4.5])
    # away short (likely to cover) -> margin below 4.5; away long -> above.
    away = pd.Series([-130.0, 110.0])
    home = pd.Series([110.0, -130.0])
    centered = center_spread(line, away, home)
    assert centered.iloc[0] < 4.5
    assert centered.iloc[1] > 4.5
    assert centered.tolist() == pytest.approx([3.0, 6.0])


def test_spread_centering_is_symmetric_under_mirroring():
    """The two sides of one spread must center to exact negatives."""
    away = pd.Series([-130.0])
    home = pd.Series([110.0])
    from_away = center_spread(pd.Series([4.5]), away, home).iloc[0]
    from_home = center_spread(pd.Series([-4.5]), home, away).iloc[0]
    assert from_away == pytest.approx(-from_home)


def test_totals_direction_is_unchanged_by_the_spread_fix():
    """A short OVER price still pushes the total UP."""
    centered = center_totals(
        pd.Series([224.5]), pd.Series([-130.0]), pd.Series([110.0])
    )
    assert centered.iloc[0] > 224.5
