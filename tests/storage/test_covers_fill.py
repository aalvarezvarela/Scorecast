"""odds_sportsbook's consensus percentages and opener from the Covers store
(``odds_sportsbook.covers_fill``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nba_ou.postgre_db.line_history_aiven.fetch import TICK_COLUMNS
from nba_ou.postgre_db.odds_sportsbook import covers_fill


def test_pick_percentages_are_sbrs_rendered_shares():
    consensus = pd.DataFrame(
        {
            "game_id": ["0022600094"],
            "spread_away_picks": [296],
            "spread_home_picks": [517],
            "total_over_picks": [396],
            "total_under_picks": [299],
        }
    )
    row = covers_fill.pick_percentages(consensus).iloc[0]
    # 517 / 813 = 63.59 -> 64; 396 / 695 = 56.98 -> 57 (SBR's own example).
    assert row["spread_consensus_pct_home"] == 64.0
    assert row["spread_consensus_pct_away"] == 36.0
    assert row["total_consensus_pct_over"] == 57.0
    assert row["total_consensus_pct_under"] == 43.0


def _opener_ticks() -> pd.DataFrame:
    ts = pd.Timestamp("2026-10-23T03:00Z")
    rows = [
        ("totals", ts, 230.5, -110, 230.5, -110),
        # A later consensus_opener tick is not the opener.
        ("totals", ts + pd.Timedelta(hours=2), 228.5, -110, 228.5, -110),
        ("point_spread", ts, 4.5, -110, -4.5, -110),
        ("money_line", ts, np.nan, 160, np.nan, -190),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": "0022600094",
                "season_year": 2026,
                "market": market,
                "book": "consensus_opener",
                "line_ts": line_ts,
                "minutes_before_tip": 900.0,
                "is_opener": True,
                "left_line": ll,
                "left_price": lp,
                "right_line": rl,
                "right_price": rp,
            }
            for market, line_ts, ll, lp, rl, rp in rows
        ],
        columns=TICK_COLUMNS,
    )


def test_opener_quotes_take_the_first_tick_per_market():
    row = covers_fill.opener_quotes(_opener_ticks()).iloc[0]
    assert row["total_consensus_opener_line_over"] == 230.5
    assert row["total_consensus_opener_price_under"] == -110
    assert row["spread_consensus_opener_line_away"] == 4.5
    assert row["spread_consensus_opener_line_home"] == -4.5
    assert row["ml_consensus_opener_price_away"] == 160
    assert row["ml_consensus_opener_price_home"] == -190


def _rows() -> pd.DataFrame:
    """Two rows as a frozen SBR day page leaves them: B has its own values."""
    return pd.DataFrame(
        {
            "game_id": ["0022600094", "0022600008"],
            "total_consensus_pct_over": [np.nan, 63.0],
            "total_consensus_pct_under": [np.nan, 37.0],
            "spread_consensus_pct_home": [np.nan, 60.0],
            "spread_consensus_pct_away": [np.nan, 40.0],
            # 0.0 with no price: the frozen page's placeholder.
            "total_consensus_opener_line_over": [0.0, 224.5],
            "total_consensus_opener_price_over": [np.nan, -110.0],
            "total_consensus_opener_line_under": [0.0, 224.5],
            "total_consensus_opener_price_under": [np.nan, -110.0],
            "spread_consensus_opener_line_away": [np.nan, 2.5],
            "spread_consensus_opener_price_away": [np.nan, -110.0],
            "spread_consensus_opener_line_home": [np.nan, -2.5],
            "spread_consensus_opener_price_home": [np.nan, -110.0],
            "ml_consensus_opener_price_away": [np.nan, 120.0],
            "ml_consensus_opener_price_home": [-526.0, -140.0],
        }
    )


def test_only_what_sbr_lacks_is_filled():
    percentages = pd.DataFrame(
        {
            "game_id": ["0022600094", "0022600008"],
            "total_consensus_pct_over": [55.0, 99.0],
            "total_consensus_pct_under": [45.0, 1.0],
            "spread_consensus_pct_home": [52.0, 99.0],
            "spread_consensus_pct_away": [48.0, 1.0],
        }
    )
    openers = covers_fill.opener_quotes(_opener_ticks())
    out, games = covers_fill.fill_from_covers(_rows(), percentages, openers)
    a, b = out.iloc[0], out.iloc[1]

    assert games == {"pct": ["0022600094"], "opener": ["0022600094"]}
    assert a["total_consensus_pct_over"] == 55.0
    assert b["total_consensus_pct_over"] == 63.0  # SBR's kept
    assert a["total_consensus_opener_line_over"] == 230.5
    assert a["spread_consensus_opener_line_home"] == -4.5
    # A market whose prices are not both there is replaced whole.
    assert a["ml_consensus_opener_price_away"] == 160
    assert a["ml_consensus_opener_price_home"] == -190
    assert b["total_consensus_opener_line_over"] == 224.5  # B's openers are SBR's


def test_nothing_from_covers_leaves_the_rows():
    rows = _rows()
    out, games = covers_fill.fill_from_covers(
        rows, pd.DataFrame(columns=["game_id"]), pd.DataFrame(columns=["game_id"])
    )
    assert games == {"pct": [], "opener": []}
    pd.testing.assert_frame_equal(out, rows)
