"""Tests for merge_and_validate_scheduled_odds - specifically for play-in / playoff
game behaviour where optional columns (public betting %) may be absent."""

import datetime

import pandas as pd
import pytest
from nba_ou.data_processing.odds.merge_scheduled_odds import (
    merge_and_validate_scheduled_odds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_odds_cols() -> list[str]:
    """Minimal set of columns that the historical odds DataFrame would have."""
    return [
        "game_id",
        "game_date",
        "season_year",
        "total_consensus_opener_line_over",
        "total_betmgm_line_over",
        "total_betmgm_line_under",
        "total_betmgm_price_over",
        "total_betmgm_price_under",
        "spread_betmgm_line_home",
        "spread_betmgm_line_away",
        "ml_betmgm_price_home",
        "ml_betmgm_price_away",
        # Yahoo public-betting % columns that may be absent for play-in games
        "total_pct_bets_over",
        "total_pct_bets_under",
        "total_pct_money_over",
        "total_pct_money_under",
        "spread_pct_bets_away",
        "spread_pct_bets_home",
        "spread_pct_money_away",
        "spread_pct_money_home",
        "moneyline_pct_bets_away",
        "moneyline_pct_bets_home",
        "moneyline_pct_money_away",
        "moneyline_pct_money_home",
    ]


def _make_historical_odds(n: int = 5) -> pd.DataFrame:
    """Build a small historical odds DataFrame with all columns present."""
    rows = []
    for i in range(n):
        row = {c: 0.0 for c in _base_odds_cols()}
        row["game_id"] = f"002250000{i}"
        row["game_date"] = datetime.date(2026, 1, i + 1)
        row["season_year"] = 2025
        rows.append(row)
    return pd.DataFrame(rows)


def _make_prediction_odds(
    include_public_betting_pcts: bool = True,
    include_extra_yahoo_cols: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal Yahoo + Sportsbook prediction DataFrames (one game each).

    Returns (df_yahoo, df_sportsbook) ready to be passed to
    merge_and_validate_scheduled_odds via merge_yahoo_sportsbook_odds.
    """
    # We'll pass pre-merged data by calling merge_yahoo_sportsbook_odds ourselves
    # to keep the test self-contained.  Instead, build the *merged* prediction df
    # and patch it in directly – but since we can't easily mock, we build
    # df_yahoo / df_sportsbook that when merged produce the desired rows.

    game_id = "0052500001"
    game_date = datetime.date(2026, 4, 14)

    # Sportsbook-side columns (after merge_daily_frames)
    sb_row = {
        "game_id": game_id,
        "game_date": game_date,
        "season_year": 2025,
        "team_home": "Miami Heat",
        "team_away": "Charlotte Hornets",
        "home_points": None,
        "away_points": None,
        "total_points": None,
        "total_consensus_opener_line_over": 221.5,
        "total_betmgm_line_over": 222.0,
        "total_betmgm_line_under": 222.0,
        "total_betmgm_price_over": -110.0,
        "total_betmgm_price_under": -110.0,
        "spread_betmgm_line_home": -3.5,
        "spread_betmgm_line_away": 3.5,
        "ml_betmgm_price_home": -180.0,
        "ml_betmgm_price_away": 155.0,
    }
    df_sportsbook = pd.DataFrame([sb_row])

    # Yahoo-side columns
    yahoo_row: dict = {
        "game_id": game_id,
        "game_date": game_date,
        "season_year": 2025,
        "total_line": 222.0,
        "spread_home": -3.5,
        "spread_away": 3.5,
        "moneyline_home": -180.0,
        "moneyline_away": 155.0,
    }
    if include_extra_yahoo_cols:
        yahoo_row["total_line_over"] = 222.0
        yahoo_row["total_line_under"] = 222.0

    if include_public_betting_pcts:
        yahoo_row.update(
            {
                "total_pct_bets_over": 55.0,
                "total_pct_bets_under": 45.0,
                "total_pct_money_over": 60.0,
                "total_pct_money_under": 40.0,
                "spread_pct_bets_away": 48.0,
                "spread_pct_bets_home": 52.0,
                "spread_pct_money_away": 47.0,
                "spread_pct_money_home": 53.0,
                "moneyline_pct_bets_away": 40.0,
                "moneyline_pct_bets_home": 60.0,
                "moneyline_pct_money_away": 35.0,
                "moneyline_pct_money_home": 65.0,
            }
        )
    else:
        # All public betting percentages are NaN (play-in / playoff scenario)
        for col in [
            "total_pct_bets_over",
            "total_pct_bets_under",
            "total_pct_money_over",
            "total_pct_money_under",
            "spread_pct_bets_away",
            "spread_pct_bets_home",
            "spread_pct_money_away",
            "spread_pct_money_home",
            "moneyline_pct_bets_away",
            "moneyline_pct_bets_home",
            "moneyline_pct_money_away",
            "moneyline_pct_money_home",
        ]:
            yahoo_row[col] = None

    df_yahoo = pd.DataFrame([yahoo_row])
    return df_yahoo, df_sportsbook


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_regular_season_game_passes_strict_mode_2(monkeypatch):
    """Regular-season games (all columns populated) pass strict_mode=2."""
    df_odds = _make_historical_odds()
    df_yahoo, df_sportsbook = _make_prediction_odds(
        include_public_betting_pcts=True, include_extra_yahoo_cols=True
    )

    # Patch merge_yahoo_sportsbook_odds to return a controlled merged frame
    merged = df_sportsbook.drop(
        columns=["team_home", "team_away", "home_points", "away_points", "total_points"],
        errors="ignore",
    ).copy()
    # Add yahoo columns (simulate the merge)
    for col in [
        "total_pct_bets_over",
        "total_pct_bets_under",
        "total_pct_money_over",
        "total_pct_money_under",
        "spread_pct_bets_away",
        "spread_pct_bets_home",
        "spread_pct_money_away",
        "spread_pct_money_home",
        "moneyline_pct_bets_away",
        "moneyline_pct_bets_home",
        "moneyline_pct_money_away",
        "moneyline_pct_money_home",
    ]:
        merged[col] = 50.0

    import nba_ou.data_processing.odds.merge_scheduled_odds as mod

    monkeypatch.setattr(
        mod,
        "merge_yahoo_sportsbook_odds",
        lambda y, s, **kwargs: merged,
    )

    result = merge_and_validate_scheduled_odds(df_odds, df_yahoo, df_sportsbook, strict_mode=2)

    # Today's game must be present in the combined result
    today_rows = result[result["game_id"] == "0052500001"]
    assert len(today_rows) == 1


def test_play_in_game_no_public_pcts_is_optional(monkeypatch, capsys):
    """Play-in games with missing public-betting % columns must NOT raise.

    Optional Yahoo inputs do not count against strict-mode row limits.
    """
    df_odds = _make_historical_odds()

    # Build a merged prediction frame where all 12 public-betting % cols are NaN
    # (simulating play-in / playoff games where Yahoo doesn't publish them)
    sb_row = {
        "game_id": "0052500001",
        "game_date": datetime.date(2026, 4, 14),
        "season_year": 2025,
        "total_consensus_opener_line_over": 221.5,
        "total_betmgm_line_over": 222.0,
        "total_betmgm_line_under": 222.0,
        "total_betmgm_price_over": -110.0,
        "total_betmgm_price_under": -110.0,
        "spread_betmgm_line_home": -3.5,
        "spread_betmgm_line_away": 3.5,
        "ml_betmgm_price_home": -180.0,
        "ml_betmgm_price_away": 155.0,
        # All public betting percentages are NaN
        "total_pct_bets_over": None,
        "total_pct_bets_under": None,
        "total_pct_money_over": None,
        "total_pct_money_under": None,
        "spread_pct_bets_away": None,
        "spread_pct_bets_home": None,
        "spread_pct_money_away": None,
        "spread_pct_money_home": None,
        "moneyline_pct_bets_away": None,
        "moneyline_pct_bets_home": None,
        "moneyline_pct_money_away": None,
        "moneyline_pct_money_home": None,
    }
    merged = pd.DataFrame([sb_row])

    import nba_ou.data_processing.odds.merge_scheduled_odds as mod

    monkeypatch.setattr(
        mod,
        "merge_yahoo_sportsbook_odds",
        lambda y, s, **kwargs: merged,
    )

    # Should NOT raise even though all 12 public-betting % cols are NaN (> strict_mode=2)
    result = merge_and_validate_scheduled_odds(
        df_odds, pd.DataFrame(), pd.DataFrame(), strict_mode=2
    )

    # The play-in game row must still be present in the result
    today_rows = result[result["game_id"] == "0052500001"]
    assert len(today_rows) == 1, (
        "Play-in game row was removed; pipeline would produce no predictions"
    )

    # No strict-mode exception is needed for optional Yahoo inputs.
    captured = capsys.readouterr()
    assert "exceed" not in captured.out.lower()


def test_strict_mode_drops_partial_nan_rows_keeps_clean_rows(monkeypatch):
    """When only SOME rows fail strict_mode, the bad rows are dropped but good ones kept."""
    df_odds = _make_historical_odds()

    # Two prediction rows: first has optional Yahoo gaps and required quote
    # gaps; the latter must still trigger the strict-mode row filter.
    row_bad = {
        "game_id": "0052500001",
        "game_date": datetime.date(2026, 4, 14),
        "season_year": 2025,
        "total_consensus_opener_line_over": 221.5,
        "total_betmgm_line_over": 222.0,
        "total_betmgm_line_under": 222.0,
        "total_betmgm_price_over": -110.0,
        "total_betmgm_price_under": -110.0,
        "spread_betmgm_line_home": -3.5,
        "spread_betmgm_line_away": 3.5,
        "ml_betmgm_price_home": -180.0,
        "ml_betmgm_price_away": 155.0,
        # All public betting percentages are NaN
        **{
            col: None
            for col in [
                "total_pct_bets_over",
                "total_pct_bets_under",
                "total_pct_money_over",
                "total_pct_money_under",
                "spread_pct_bets_away",
                "spread_pct_bets_home",
                "spread_pct_money_away",
                "spread_pct_money_home",
                "moneyline_pct_bets_away",
                "moneyline_pct_bets_home",
                "moneyline_pct_money_away",
                "moneyline_pct_money_home",
            ]
        },
    }
    row_good = {
        "game_id": "0022500099",
        "game_date": datetime.date(2026, 4, 14),
        "season_year": 2025,
        "total_consensus_opener_line_over": 215.0,
        "total_betmgm_line_over": 215.5,
        "total_betmgm_line_under": 215.5,
        "total_betmgm_price_over": -110.0,
        "total_betmgm_price_under": -110.0,
        "spread_betmgm_line_home": -5.0,
        "spread_betmgm_line_away": 5.0,
        "ml_betmgm_price_home": -200.0,
        "ml_betmgm_price_away": 165.0,
        **{
            col: 50.0
            for col in [
                "total_pct_bets_over",
                "total_pct_bets_under",
                "total_pct_money_over",
                "total_pct_money_under",
                "spread_pct_bets_away",
                "spread_pct_bets_home",
                "spread_pct_money_away",
                "spread_pct_money_home",
                "moneyline_pct_bets_away",
                "moneyline_pct_bets_home",
                "moneyline_pct_money_away",
                "moneyline_pct_money_home",
            ]
        },
    }
    for column in (
        "total_betmgm_price_over", "total_betmgm_price_under", "ml_betmgm_price_home"
    ):
        row_bad[column] = None
    merged = pd.DataFrame([row_bad, row_good])

    import nba_ou.data_processing.odds.merge_scheduled_odds as mod

    monkeypatch.setattr(
        mod,
        "merge_yahoo_sportsbook_odds",
        lambda y, s, **kwargs: merged,
    )

    result = merge_and_validate_scheduled_odds(
        df_odds, pd.DataFrame(), pd.DataFrame(), strict_mode=2
    )

    # The good (regular-season) row should be present
    assert len(result[result["game_id"] == "0022500099"]) == 1
    # The bad (play-in) row should have been dropped because a good row survived
    assert len(result[result["game_id"] == "0052500001"]) == 0


def test_missing_columns_in_prediction_raises():
    """A prediction DataFrame that is missing columns from the historical data must raise."""
    df_odds = _make_historical_odds()

    # Build a prediction df that is missing several historical columns
    partial_row = {
        "game_id": "0052500001",
        "game_date": datetime.date(2026, 4, 14),
        "season_year": 2025,
        "total_betmgm_line_over": 222.0,
        # Many historical cols deliberately absent
    }

    from unittest.mock import patch

    import nba_ou.data_processing.odds.merge_scheduled_odds as mod

    with patch.object(mod, "merge_yahoo_sportsbook_odds", return_value=pd.DataFrame([partial_row])):
        with pytest.raises(ValueError, match="missing columns"):
            merge_and_validate_scheduled_odds(
                df_odds, pd.DataFrame(), pd.DataFrame(), strict_mode=2
            )
