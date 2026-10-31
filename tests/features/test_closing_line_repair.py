"""Tests for repairing SBR closing lines from the line-history store.

The leak being closed: SBR stores a book's *last* number, which is often an
in-play line. The repair must replace it with the last valid pre-tip quote --
the whole quote, in the right orientation -- and must only clear cross-book
outliers that are very likely data mistakes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.line_history.normalization import devig_two_way
from nba_ou.data_processing.odds.closing_line_repair import (
    ACTION_HISTORY_FILLED,
    ACTION_HISTORY_REPLACED,
    ACTION_HISTORY_STALE,
    ACTION_HISTORY_SAME,
    ACTION_OUTLIER_MEDIAN,
    ACTION_OUTLIER_NULLED,
    ACTION_STALE_BOOK_MEDIAN,
    ACTION_STALE_BOOK_NULLED,
    ACTION_UNVERIFIED,
    audit_repaired_closes,
    repair_closing_lines,
    select_history_closes,
    stale_history_games,
)
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds

GAME = "0022300001"
TIP = pd.Timestamp("2024-01-10 00:00:00", tz="UTC")
BOOKS = ("bet365", "betmgm", "draftkings", "fanduel", "caesars")


def wide_row(**values) -> pd.DataFrame:
    """One SBR row with every book's totals/spread/moneyline columns present."""
    row: dict = {"game_id": GAME, "season_year": 2023}
    for book in BOOKS:
        for column in (
            f"total_{book}_line_over",
            f"total_{book}_line_under",
            f"total_{book}_price_over",
            f"total_{book}_price_under",
            f"spread_{book}_line_away",
            f"spread_{book}_line_home",
            f"spread_{book}_price_away",
            f"spread_{book}_price_home",
            f"ml_{book}_price_away",
            f"ml_{book}_price_home",
        ):
            row[column] = np.nan
    row.update(values)
    return pd.DataFrame([row])


def tick(
    book: str,
    minutes_before_tip: float,
    *,
    market: str = "totals",
    line: float | None = 220.5,
    left_price: float = -110.0,
    right_price: float = -110.0,
    game_id: str = GAME,
) -> dict:
    """A tick shaped like ``fetch_closing_ticks`` output (left = OVER / AWAY)."""
    return {
        "game_id": game_id,
        "season_year": 2023,
        "market": market,
        "book": book,
        "line_ts": TIP - pd.Timedelta(minutes=minutes_before_tip),
        "minutes_before_tip": float(minutes_before_tip),
        "is_opener": False,
        "left_line": line,
        "left_price": left_price,
        "right_line": (
            None if line is None else (-line if market == "point_spread" else line)
        ),
        "right_price": right_price,
    }


def ticks(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def cell(audit: pd.DataFrame, book: str, market: str = "totals") -> pd.Series:
    rows = audit[audit["book"].eq(book) & audit["market"].eq(market)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_in_play_sbr_total_is_replaced_by_whole_pre_tip_quote():
    df = wide_row(
        total_fanduel_line_over=231.5,
        total_fanduel_line_under=231.5,
        total_fanduel_price_over=-150.0,
        total_fanduel_price_under=120.0,
    )
    history = ticks(
        [
            tick("fanduel", 300, line=219.5),
            tick("fanduel", 20, line=220.5, left_price=-105, right_price=-115),
        ]
    )

    repaired, audit = repair_closing_lines(df, history)

    row = repaired.iloc[0]
    assert row["total_fanduel_line_over"] == 220.5
    assert row["total_fanduel_line_under"] == 220.5
    assert row["total_fanduel_price_over"] == -105.0
    assert row["total_fanduel_price_under"] == -115.0
    assert cell(audit, "fanduel")["action"] == ACTION_HISTORY_REPLACED
    audit_repaired_closes(repaired, audit)


def test_spread_replacement_keeps_home_away_orientation():
    # Home favoured by 6: away handicap +6, home handicap -6.
    df = wide_row(
        spread_draftkings_line_away=1.5,
        spread_draftkings_line_home=-1.5,
        spread_draftkings_price_away=-110.0,
        spread_draftkings_price_home=-110.0,
    )
    history = ticks(
        [
            tick(
                "draftkings",
                15,
                market="point_spread",
                line=6.0,
                left_price=-108,
                right_price=-112,
            )
        ]
    )

    repaired, audit = repair_closing_lines(df, history)

    row = repaired.iloc[0]
    assert row["spread_draftkings_line_away"] == 6.0
    assert row["spread_draftkings_line_home"] == -6.0
    assert row["spread_draftkings_price_away"] == -108.0
    assert row["spread_draftkings_price_home"] == -112.0
    audit_repaired_closes(repaired, audit)


def test_identical_quote_is_reported_as_same():
    df = wide_row(
        ml_bet365_price_away=150.0,
        ml_bet365_price_home=-170.0,
    )
    history = ticks(
        [tick("bet365", 10, market="money_line", line=None, left_price=150, right_price=-170)]
    )

    _, audit = repair_closing_lines(df, history)

    assert cell(audit, "bet365", "money_line")["action"] == ACTION_HISTORY_SAME


def test_glitch_tick_with_extreme_price_is_skipped_for_the_close():
    history = ticks(
        [
            tick("bet365", 60, line=220.5),
            tick("bet365", 10, line=191.5, left_price=-740, right_price=450),
        ]
    )

    closes = select_history_closes(history)

    assert closes["line"].tolist() == [220.5]


def test_book_missing_from_history_keeps_sbr_as_unverified():
    df = wide_row(
        total_caesars_line_over=221.0,
        total_caesars_line_under=221.0,
        total_caesars_price_over=-110.0,
        total_caesars_price_under=-110.0,
    )

    repaired, audit = repair_closing_lines(df, ticks([tick("bet365", 10)]))

    assert repaired.iloc[0]["total_caesars_line_over"] == 221.0
    assert cell(audit, "caesars")["action"] == ACTION_UNVERIFIED


def test_no_line_history_keeps_sbr_and_passes_the_audit():
    # Games stored before any tick is fetched (e.g. a season's first nights).
    df = wide_row(
        total_caesars_line_over=221.0,
        total_caesars_line_under=221.0,
        total_caesars_price_over=-110.0,
        total_caesars_price_under=-110.0,
    )

    repaired, audit = repair_closing_lines(df, pd.DataFrame())
    audit_repaired_closes(repaired, audit)

    assert audit["final_level"].dtype == "float64"
    assert repaired.iloc[0]["total_caesars_line_over"] == 221.0
    assert cell(audit, "caesars")["action"] == ACTION_UNVERIFIED


def test_history_that_froze_hours_before_tip_is_flagged_stale_not_dropped():
    # Every book's last quote ~5h before tip: an SBR feed freeze. The values
    # are still the last pre-tip quotes; the cells are flagged, not verified.
    df = wide_row(
        total_bet365_line_over=222.0,
        total_bet365_line_under=222.0,
        total_bet365_price_over=-110.0,
        total_bet365_price_under=-110.0,
    )
    history = ticks(
        [
            tick("bet365", 300, line=222.0),
            tick("betmgm", 299, line=223.5),
            tick("draftkings", 310, line=222.5),
        ]
    )

    repaired, audit = repair_closing_lines(df, history)
    audit_repaired_closes(repaired, audit)

    assert set(audit["action"]) == {ACTION_HISTORY_STALE}
    assert stale_history_games(audit) == [GAME]
    assert repaired.iloc[0]["total_bet365_line_over"] == 222.0
    assert repaired.iloc[0]["total_betmgm_line_over"] == 223.5


def test_one_fresh_book_keeps_the_game_verified():
    # Books that stop quoting early are normal; the game is stale only when
    # even its freshest book closed long before tip.
    history = ticks([tick("bet365", 300, line=222.0), tick("betmgm", 4, line=223.5)])

    _, audit = repair_closing_lines(wide_row(), history)

    assert ACTION_HISTORY_STALE not in set(audit["action"])
    assert stale_history_games(audit) == []


def test_history_fills_a_cell_sbr_left_empty():
    repaired, audit = repair_closing_lines(wide_row(), ticks([tick("betmgm", 30, line=218.0)]))

    assert repaired.iloc[0]["total_betmgm_line_over"] == 218.0
    assert cell(audit, "betmgm")["action"] == ACTION_HISTORY_FILLED


def _consensus_ticks(line: float, minutes: float = 10) -> list[dict]:
    return [
        tick("draftkings", minutes, line=line),
        tick("fanduel", minutes, line=line + 0.5),
        tick("caesars", minutes, line=line),
    ]


def test_stale_outlier_is_nulled_for_a_non_anchor_book():
    history = ticks(
        [
            *_consensus_ticks(227.0),
            *_consensus_ticks(226.0, minutes=300),
            # Quoted hours early (within the stale-book lag, so step B's case)
            # and already far from every other book then.
            tick("draftkings", 700, line=226.0),
            tick("fanduel", 700, line=226.0),
            tick("betmgm", 640, line=214.5),
        ]
    )

    repaired, audit = repair_closing_lines(wide_row(), history)

    assert np.isnan(repaired.iloc[0]["total_betmgm_line_over"])
    assert np.isnan(repaired.iloc[0]["total_betmgm_price_over"])
    assert cell(audit, "betmgm")["action"] == ACTION_OUTLIER_NULLED
    audit_repaired_closes(repaired, audit)


def _unpriced_total(book: str, line: float) -> dict:
    """An SBR total with a line but no prices, so it has no normalized level."""
    return {f"total_{book}_line_over": line, f"total_{book}_line_under": line}


@pytest.mark.parametrize("line", [114.5, 190.5, 216.5])
def test_unpriced_total_far_from_the_market_is_nulled(line):
    # The early-2021 BetMGM shape: other books verified around 228, BetMGM's
    # SBR cell carries a line and no prices.
    repaired, audit = repair_closing_lines(
        wide_row(**_unpriced_total("betmgm", line)), ticks(_consensus_ticks(228.0))
    )

    assert np.isnan(repaired.iloc[0]["total_betmgm_line_over"])
    assert np.isnan(repaired.iloc[0]["total_betmgm_line_under"])
    assert cell(audit, "betmgm")["action"] == ACTION_OUTLIER_NULLED
    audit_repaired_closes(repaired, audit)


def test_unpriced_total_within_the_wider_threshold_is_left_alone():
    # 9 points off: an outlier for a priced cell (6), not for an unpriced one (10).
    repaired, audit = repair_closing_lines(
        wide_row(**_unpriced_total("betmgm", 219.0)), ticks(_consensus_ticks(228.0))
    )

    assert repaired.iloc[0]["total_betmgm_line_over"] == 219.0
    assert cell(audit, "betmgm")["action"] == ACTION_UNVERIFIED


def test_implausible_unverified_total_is_nulled_without_other_books():
    repaired, audit = repair_closing_lines(
        wide_row(**_unpriced_total("betmgm", 114.5)), ticks([tick("bet365", 10)])
    )

    assert np.isnan(repaired.iloc[0]["total_betmgm_line_over"])
    assert cell(audit, "betmgm")["action"] == ACTION_OUTLIER_NULLED


def test_unpriced_anchor_total_is_filled_with_verified_median():
    repaired, audit = repair_closing_lines(
        wide_row(**_unpriced_total("bet365", 114.5)), ticks(_consensus_ticks(228.0))
    )

    row = repaired.iloc[0]
    assert row["total_bet365_line_over"] == 228.0
    assert row["total_bet365_price_over"] == -110.0
    assert cell(audit, "bet365")["action"] == ACTION_OUTLIER_MEDIAN
    audit_repaired_closes(repaired, audit)


def test_anchor_outlier_is_filled_with_verified_median_at_minus_110():
    df = wide_row(
        spread_bet365_line_away=-10.0,
        spread_bet365_line_home=10.0,
        spread_bet365_price_away=-110.0,
        spread_bet365_price_home=-110.0,
    )
    history = ticks(
        [
            tick("draftkings", 10, market="point_spread", line=1.5),
            tick("fanduel", 10, market="point_spread", line=1.5),
            tick("caesars", 10, market="point_spread", line=2.0),
        ]
    )

    repaired, audit = repair_closing_lines(df, history)

    row = repaired.iloc[0]
    assert row["spread_bet365_line_away"] == 1.5
    assert row["spread_bet365_line_home"] == -1.5
    assert row["spread_bet365_price_away"] == -110.0
    assert cell(audit, "bet365", "point_spread")["action"] == ACTION_OUTLIER_MEDIAN
    audit_repaired_closes(repaired, audit)


def test_moneyline_anchor_median_fill_devigs_back_to_the_median():
    history = ticks(
        [
            tick("draftkings", 10, market="money_line", line=None, left_price=300, right_price=-380),
            tick("fanduel", 10, market="money_line", line=None, left_price=290, right_price=-370),
            tick("caesars", 10, market="money_line", line=None, left_price=310, right_price=-400),
            tick("bet365", 600, market="money_line", line=None, left_price=-105, right_price=-115),
        ]
    )

    repaired, audit = repair_closing_lines(wide_row(), history)

    target = cell(audit, "bet365", "money_line")
    assert target["action"] == ACTION_OUTLIER_MEDIAN
    row = repaired.iloc[0]
    fair = devig_two_way(
        pd.Series([row["ml_bet365_price_away"]]), pd.Series([row["ml_bet365_price_home"]])
    )
    assert fair["fair_right"].iloc[0] == pytest.approx(target["others_median"])
    audit_repaired_closes(repaired, audit)


def test_book_that_closed_before_a_market_wide_jump_is_not_corrected():
    # bet365 closed at 220.5 when everyone agreed; the others then jumped 9
    # points together in the last minutes (a likely early tip). Replacing bet365
    # with that jump would import the very leak this repair removes.
    history = ticks(
        [
            tick("bet365", 17, line=220.5),
            tick("draftkings", 20, line=221.0),
            tick("fanduel", 20, line=220.5),
            tick("caesars", 20, line=221.0),
            *_consensus_ticks(229.5, minutes=8),
        ]
    )

    repaired, audit = repair_closing_lines(wide_row(), history)

    assert repaired.iloc[0]["total_bet365_line_over"] == 220.5
    assert cell(audit, "bet365")["action"] == ACTION_HISTORY_FILLED


def test_no_outlier_when_other_books_disagree_among_themselves():
    history = ticks(
        [
            tick("draftkings", 10, line=220.0),
            tick("fanduel", 10, line=224.0),
            tick("caesars", 10, line=228.0),
            tick("betmgm", 2000, line=214.0),
        ]
    )

    _, audit = repair_closing_lines(wide_row(), history)

    assert cell(audit, "betmgm")["action"] == ACTION_HISTORY_FILLED


def test_deviation_below_threshold_is_left_alone():
    history = ticks([*_consensus_ticks(227.0), tick("betmgm", 600, line=222.0)])

    _, audit = repair_closing_lines(wide_row(), history)

    assert cell(audit, "betmgm")["action"] == ACTION_HISTORY_FILLED


def test_book_recorded_only_at_its_opener_is_nulled_not_kept():
    # The 2026-27 Fanatics shape (0022600020): the feed kept only the opener,
    # ten days before tip, while every other book went on to close ~237.
    # Neither that opener nor SBR's own value is a verified close.
    df = wide_row(
        total_betmgm_line_over=237.0,
        total_betmgm_line_under=237.0,
        total_betmgm_price_over=-110.0,
        total_betmgm_price_under=-110.0,
    )
    history = ticks([*_consensus_ticks(237.0, minutes=60), tick("betmgm", 14186, line=242.0)])

    repaired, audit = repair_closing_lines(df, history)

    assert np.isnan(repaired.iloc[0]["total_betmgm_line_over"])
    assert np.isnan(repaired.iloc[0]["total_betmgm_price_under"])
    target = cell(audit, "betmgm")
    assert target["action"] == ACTION_STALE_BOOK_NULLED
    assert target["book_lag_minutes"] == pytest.approx(14126.0)
    audit_repaired_closes(repaired, audit)


def test_stale_anchor_book_is_filled_with_the_fresh_median():
    history = ticks([*_consensus_ticks(237.0, minutes=60), tick("bet365", 3000, line=242.0)])

    repaired, audit = repair_closing_lines(wide_row(), history)

    row = repaired.iloc[0]
    assert row["total_bet365_line_over"] == 237.0
    assert row["total_bet365_price_over"] == -110.0
    assert cell(audit, "bet365")["action"] == ACTION_STALE_BOOK_MEDIAN
    audit_repaired_closes(repaired, audit)


def test_stale_book_still_in_line_with_the_market_is_kept():
    # A book that did not move all day: its old quote is still the market's.
    history = ticks([*_consensus_ticks(227.0), tick("betmgm", 1500, line=228.5)])

    repaired, audit = repair_closing_lines(wide_row(), history)

    assert repaired.iloc[0]["total_betmgm_line_over"] == 228.5
    assert cell(audit, "betmgm")["action"] == ACTION_HISTORY_FILLED


def test_lag_is_measured_against_the_freshest_book_of_the_game():
    # A whole-game feed freeze leaves every book equally old: none is stale
    # against the others, so none is emptied.
    history = ticks(
        [
            tick("draftkings", 3000, line=227.0),
            tick("fanduel", 3000, line=227.0),
            tick("caesars", 3000, line=227.0),
            tick("betmgm", 3100, line=231.0),
        ]
    )

    repaired, audit = repair_closing_lines(wide_row(), history)

    assert repaired.iloc[0]["total_betmgm_line_over"] == 231.0
    assert ACTION_STALE_BOOK_NULLED not in set(audit["action"])


def test_write_back_audit_catches_a_tampered_frame():
    repaired, audit = repair_closing_lines(wide_row(), ticks([tick("betmgm", 30, line=218.0)]))
    repaired.loc[0, "total_betmgm_line_over"] = 230.0

    with pytest.raises(AssertionError, match="write-back mismatch"):
        audit_repaired_closes(repaired, audit)


def test_merge_applies_repair_on_american_prices_before_centering():
    seen: dict = {}

    def repair(frame: pd.DataFrame) -> pd.DataFrame:
        seen["price"] = frame["total_bet365_price_over"].iloc[0]
        repaired, _ = repair_closing_lines(
            frame, ticks([tick("bet365", 10, line=221.5, left_price=-110, right_price=-110)])
        )
        return repaired

    sportsbook = wide_row(
        total_bet365_line_over=235.5,
        total_bet365_line_under=235.5,
        total_bet365_price_over=-200.0,
        total_bet365_price_under=160.0,
    )

    merged = merge_yahoo_sportsbook_odds(
        pd.DataFrame(), sportsbook, closing_line_repair=repair
    )

    assert seen["price"] == -200.0
    assert merged["total_bet365_line_over"].iloc[0] == 221.5
    assert merged["total_bet365_price_over"].iloc[0] == pytest.approx(1.91, abs=0.01)
