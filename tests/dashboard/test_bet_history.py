"""The bet history: its money figures, groupings and page (no database)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from nba_ou.dashboard.domain import bet_history as rules
from nba_ou.dashboard.ui import bet_history as history_ui
from nba_ou.dashboard.ui import yesterday as yesterday_ui
from nba_ou.dashboard.ui.bet_summary import bet_line, graded_bets_table
from streamlit.testing.v1 import AppTest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "streamlit/streamlit_app.py"
HISTORY_PAGE = str(ENTRYPOINT.parent / "app_pages/bet_history.py")


def _bets() -> pd.DataFrame:
    """Two slates: a win and a loss on totals, then a spread push, a pending
    bet and a postponed game's void one."""
    return pd.DataFrame(
        {
            "bet_id": [1, 2, 3, 4, 5],
            "game_id": [10, 11, 20, 21, 22],
            "game_date": [
                date(2026, 10, 30),
                date(2026, 10, 30),
                date(2026, 11, 2),
                date(2026, 11, 2),
                date(2026, 11, 2),
            ],
            "tipoff_at": pd.to_datetime(
                [
                    "2026-10-31T00:00Z",
                    "2026-10-31T02:00Z",
                    "2026-11-03T00:00Z",
                    "2026-11-03T01:00Z",
                    "2026-11-03T02:00Z",
                ]
            ),
            "home_team": ["Boston Celtics"] * 5,
            "away_team": ["New York Knicks"] * 5,
            "market": [1, 1, 2, 1, 1],
            "side": [1, -1, -1, 1, 1],
            "line": [224.5, 230.0, -3.5, 220.0, 221.0],
            "odds_decimal": [2.0, 1.9, 1.9, 1.9, 1.9],
            "stake": [10.0, 20.0, 5.0, 8.0, 4.0],
            "book": "bet365",
            "placed_at": pd.Timestamp("2026-10-30T18:00Z"),
            "result": ["win", "loss", "push", "pending", "void"],
            "profit": [10.0, -20.0, 0.0, None, 0.0],
            "model_kind": [3, 1, None, 3, 3],
            "model_dataset_type": [2, 1, None, 1, 1],
        }
    )


def test_money_counts_settled_bets_and_reports_pending_apart():
    totals = rules.performance(rules.prepare_history(_bets()))
    assert totals["bets"] == 5 and totals["settled"] == 3
    assert (totals["wins"], totals["losses"], totals["pushes"]) == (1, 1, 1)
    assert totals["staked"] == 35.0
    assert totals["profit"] == -10.0
    assert totals["returned"] == 25.0
    assert totals["roi"] == pytest.approx(-10 / 35)
    # A push is neither a win nor a loss.
    assert totals["win_rate"] == 0.5
    assert (totals["pending"], totals["pending_stake"]) == (1, 8.0)


def test_an_empty_history_has_no_rates():
    totals = rules.performance(rules.prepare_history(_bets()).iloc[:0])
    assert totals["roi"] == 0.0 and totals["win_rate"] == 0.0
    assert rules.daily_profit(rules.prepare_history(_bets()).iloc[:0]).empty


def test_the_running_profit_adds_up_slate_by_slate():
    days = rules.daily_profit(rules.prepare_history(_bets()))
    assert days["game_date"].tolist() == [date(2026, 10, 30), date(2026, 11, 2)]
    assert days["bets"].tolist() == [2, 1]
    assert days["profit"].tolist() == [-10.0, 0.0]
    assert days["total_profit"].tolist() == [-10.0, -10.0]
    assert days["total_staked"].tolist() == [30.0, 35.0]


def test_filters_keep_the_slates_and_the_market_asked_for():
    history = rules.prepare_history(_bets())
    late = rules.filter_history(history, date(2026, 11, 1), date(2026, 11, 2))
    assert late["bet_id"].tolist() == [3, 4, 5]
    spread = rules.filter_history(history, market=2)
    assert spread["bet_id"].tolist() == [3]


def test_breakdowns_by_side_model_and_month():
    history = rules.prepare_history(_bets())
    sides = rules.breakdown(history, "side_label").set_index("Group")
    assert sides.loc["Totals · Over", "W-L-P"] == "1-0-0"
    assert sides.loc["Totals · Over", "Pending"] == 1
    assert sides.loc["Totals · Under", "Profit (€)"] == -20.0
    assert sides.loc["Spread · Away", "W-L-P"] == "0-0-1"
    models = rules.breakdown(history, "model_label").set_index("Group")
    assert set(models.index) == {
        "TabPFN intermediate",
        "TabPFN closing",
        "XGBoost closing",
        rules.NO_MODEL,
    }
    assert models.loc["TabPFN intermediate", "ROI %"] == 100.0
    # Only a pending and a void bet: no settled money, no rate.
    assert models.loc["TabPFN closing", "Bets"] == 0
    assert pd.isna(models.loc["TabPFN closing", "ROI %"])
    months = rules.breakdown(history, "month")
    assert months["Group"].tolist() == ["2026-10", "2026-11"]


def test_a_bet_reads_as_one_line_with_its_result():
    bets = _bets()
    assert bet_line(bets.iloc[0]) == (
        "**Over 224.5** · 10 € at 2.00 · :green[**Win +10.00 €**]"
    )
    assert ":red[**Loss -20.00 €**]" in bet_line(bets.iloc[1])
    # The Knicks were backed: the stored home handicap -3.5 is their +3.5.
    assert bet_line(bets.iloc[2]).startswith("**New York Knicks +3.5**")
    assert ":gray[**Pending**]" in bet_line(bets.iloc[3])
    assert graded_bets_table(bets)["Bet"].tolist()[2] == "New York Knicks +3.5"


def test_the_profit_chart_draws_days_and_the_running_total():
    spec = history_ui.profit_chart(
        rules.daily_profit(rules.prepare_history(_bets()))
    ).to_dict()
    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["rule", "bar", "line"]


def test_history_page_shows_money_chart_groups_and_bets(monkeypatch, today_data):
    from nba_ou.dashboard.ui import loaders

    loaders.load_today.clear()
    monkeypatch.setattr(loaders, "read_today", lambda *args, **kwargs: today_data)
    monkeypatch.setattr(history_ui, "load_bet_history", lambda user_id: _bets())
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(HISTORY_PAGE).run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Bet history"]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Staked"] == "35.00 €"
    assert metrics["Returned"] == "25.00 €"
    assert metrics["Profit"] == "-10.00 €"
    assert metrics["W-L-P"] == "1-1-1"
    assert metrics["Pending"] == "1 · 8.00 €"
    groups, everything = [frame.value for frame in app.dataframe]
    assert "Totals · Over" in groups["Group"].tolist()
    # Newest first, every live bet.
    assert everything["Date"].tolist()[0] == date(2026, 11, 2)
    assert len(everything) == 5
    assert len(app.get("vega_lite_chart") or app.get("arrow_vega_lite_chart")) == 1
    loaders.load_today.clear()


def test_history_page_without_bets_or_store(monkeypatch, today_data):
    from nba_ou.dashboard.ui import loaders

    loaders.load_today.clear()
    monkeypatch.setattr(loaders, "read_today", lambda *args, **kwargs: today_data)
    # No fake: the account store is unreachable (conftest).
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(HISTORY_PAGE).run()
    assert not app.exception
    assert any("unavailable" in alert.value for alert in app.error)
    monkeypatch.setattr(history_ui, "load_bet_history", lambda user_id: pd.DataFrame())
    app.run()
    assert any("No bets yet" in alert.value for alert in app.info)
    loaders.load_today.clear()


def test_yesterdays_game_bets_follow_the_target_market():
    bets = _bets().assign(game_id=[10, 10, 10, 11, 11])
    totals = yesterday_ui._game_bets(bets, 10, 1)
    assert totals["bet_id"].tolist() == [1, 2]
    assert yesterday_ui._game_bets(bets, 10, 2)["bet_id"].tolist() == [1, 2]
    assert yesterday_ui._game_bets(bets, 10, 3)["bet_id"].tolist() == [3]
    assert yesterday_ui._game_bets(bets.iloc[:0], 10, 1).empty
