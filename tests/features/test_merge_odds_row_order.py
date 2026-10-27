"""The merged odds frame's row order, which the history memo digests."""

from __future__ import annotations

import pandas as pd
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds


def _sportsbook(days: int) -> pd.DataFrame:
    """Twelve games a day in game-id order, as the merge with Yahoo leaves them."""
    rows = [
        {
            "game_id": f"00225{day:03d}{game:02d}",
            "game_date": pd.Timestamp(2025, 11, 1) + pd.Timedelta(days=day),
            "season_year": 2025,
        }
        for day in range(1, days + 1)
        for game in range(12)
    ]
    return pd.DataFrame(rows)


def test_a_date_s_games_keep_their_order_whatever_later_dates_hold():
    """Each night adds games; the earlier rows must come out in the same order,
    or every finished season's memo entry misses on the next morning's build."""
    first_days = _sportsbook(60)["game_date"].max()
    orders = []
    for days in (60, 61, 75):
        sportsbook = _sportsbook(days)
        yahoo = sportsbook.assign(total_pct_bets_over=55.0)
        merged = merge_yahoo_sportsbook_odds(yahoo, sportsbook)
        earlier = merged["game_date"] <= first_days
        orders.append(merged.loc[earlier, "game_id"].tolist())
    assert orders[0] == orders[1] == orders[2]
    # Newest date first; a date's games in game-id order.
    expected = _sportsbook(60).sort_values(
        ["game_date", "game_id"], ascending=[False, True]
    )
    assert orders[0] == expected["game_id"].tolist()
