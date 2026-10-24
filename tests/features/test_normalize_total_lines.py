import pandas as pd
import pytest
from nba_ou.data_processing.odds.normalize_total_lines import (
    estimate_centered_total_line,
    normalize_total_lines_inplace,
    odds_to_decimal,
)
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    load_one_day_totals_csv,
)
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds


def _total_quote(
    line: float = 209.5,
    over_price: float = 1.62,
    under_price: float = 2.20,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_bet365_line_over": [line],
            "total_bet365_line_under": [line],
            "total_bet365_price_over": [over_price],
            "total_bet365_price_under": [under_price],
        }
    )


def test_estimate_centered_total_line_from_decimal_and_american_odds():
    assert estimate_centered_total_line(209.5, 1.62, 2.20) == 212.5
    assert (
        estimate_centered_total_line(
            209.5,
            -161.3,
            120.0,
            odds_format="american",
        )
        == 212.5
    )


def test_normalize_total_lines_modifies_existing_columns_only():
    df = _total_quote()
    original_columns = df.columns.tolist()

    returned = normalize_total_lines_inplace(df)

    assert returned is df
    assert df.columns.tolist() == original_columns
    assert df.at[0, "total_bet365_line_over"] == 212.5
    assert df.at[0, "total_bet365_line_under"] == 212.5
    expected_centered_price = 1.91
    assert df.at[0, "total_bet365_price_over"] == pytest.approx(
        expected_centered_price
    )
    assert df.at[0, "total_bet365_price_under"] == pytest.approx(
        expected_centered_price
    )


def test_normalizer_prints_changed_market_count_and_percentage(capsys):
    df = pd.concat(
        [
            _total_quote(),
            _total_quote(line=220.5, over_price=1.91, under_price=1.91),
        ],
        ignore_index=True,
    )

    normalize_total_lines_inplace(df)

    assert (
        "Total-line normalization: 1/2 bookmaker-game markets changed (50.00%)."
        in capsys.readouterr().out
    )


def test_normalizer_rounds_prices_before_comparing_small_differences():
    df = _total_quote(over_price=1.903, under_price=1.904)

    normalize_total_lines_inplace(df)

    assert df.at[0, "total_bet365_line_over"] == 209.5
    assert df.at[0, "total_bet365_line_under"] == 209.5
    assert df.at[0, "total_bet365_price_over"] == 1.90
    assert df.at[0, "total_bet365_price_under"] == 1.90


@pytest.mark.parametrize(
    ("over_price", "under_price"),
    [(1.91, 2.05), (1.75, 1.91)],
)
def test_normalizer_skips_market_when_either_side_is_minus_110(
    over_price,
    under_price,
):
    df = _total_quote(over_price=over_price, under_price=under_price)

    normalize_total_lines_inplace(df)

    assert df.at[0, "total_bet365_line_over"] == 209.5
    assert df.at[0, "total_bet365_line_under"] == 209.5
    assert df.at[0, "total_bet365_price_over"] == over_price
    assert df.at[0, "total_bet365_price_under"] == under_price


def test_normalizer_leaves_centered_disabled_and_incomplete_quotes_unchanged():
    centered = _total_quote(over_price=1.91, under_price=1.91)
    centered_before = centered.copy(deep=True)
    normalize_total_lines_inplace(centered)
    pd.testing.assert_frame_equal(centered, centered_before)

    disabled = _total_quote()
    disabled_before = disabled.copy(deep=True)
    normalize_total_lines_inplace(disabled, enabled=False)
    pd.testing.assert_frame_equal(disabled, disabled_before)

    mismatched_lines = _total_quote()
    mismatched_lines.at[0, "total_bet365_line_under"] = 210.5
    mismatched_before = mismatched_lines.copy(deep=True)
    normalize_total_lines_inplace(mismatched_lines)
    pd.testing.assert_frame_equal(mismatched_lines, mismatched_before)


def test_merge_normalizes_sportsbook_only_quotes_after_decimal_conversion():
    sportsbook = _total_quote(
        over_price=-161.3,
        under_price=120.0,
    )

    normalized = merge_yahoo_sportsbook_odds(
        pd.DataFrame(),
        sportsbook,
    )
    original = merge_yahoo_sportsbook_odds(
        pd.DataFrame(),
        sportsbook,
        normalize_total_lines=False,
    )

    assert normalized.at[0, "total_bet365_line_over"] == 212.5
    assert normalized.at[0, "total_bet365_line_under"] == 212.5
    assert normalized.at[0, "total_bet365_price_over"] == pytest.approx(
        1.91
    )
    assert original.at[0, "total_bet365_line_over"] == 209.5
    assert original.at[0, "total_bet365_price_over"] == pytest.approx(
        odds_to_decimal(-161.3, "american")
    )


def test_integer_line_uses_push_adjustment_and_stays_on_half_point_grid():
    centered_line = estimate_centered_total_line(210.0, 1.62, 2.20)
    assert centered_line * 2 == round(centered_line * 2)


def test_live_scraper_rows_are_normalized_by_the_scheduled_merge_path():
    scraper_rows = pd.DataFrame(
        {
            "date": ["2026-02-04", "2026-02-04"],
            "season": [2025, 2025],
            "event_id": [362774, 362774],
            "row_index": [0, 1],
            "team_name": ["Denver", "New York"],
            "score": [None, None],
            "consensus_pct": [60, 40],
            "consensus_opener_side": ["O", "U"],
            "bet365_line": [209.5, 209.5],
            "bet365_price": [-161.3, 120.0],
        }
    )
    scheduled_sportsbook = load_one_day_totals_csv(scraper_rows)

    normalized = merge_yahoo_sportsbook_odds(
        pd.DataFrame(),
        scheduled_sportsbook,
    )

    assert normalized.at[0, "total_bet365_line_over"] == 212.5
    assert normalized.at[0, "total_bet365_line_under"] == 212.5
    assert normalized.at[0, "total_bet365_price_over"] == pytest.approx(
        1.91
    )
    assert normalized.at[0, "total_bet365_price_under"] == pytest.approx(
        1.91
    )
