from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from bs4 import BeautifulSoup
from nba_ou.dashboard.repositories.predictions import DashboardDataError, TodayData
from nba_ou.dashboard.ui import loaders
from nba_ou.dashboard.ui import yesterday as yesterday_ui
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup

from .conftest import NOW, FrozenDateTime, rendered_images
from .test_history import historical_slate

ENTRYPOINT = Path(__file__).resolve().parents[2] / "streamlit/streamlit_app.py"


@pytest.fixture
def yesterday_setup(monkeypatch):
    def indices(group):
        values = group.value
        if not isinstance(values, list):
            values = [] if values is None else [values]
        return [group.options.index(group.format_func(value)) for value in values]

    monkeypatch.setattr(ButtonGroup, "indices", property(indices))
    data = historical_slate()
    reader = Mock(return_value=data)
    monkeypatch.setattr(loaders, "read_yesterday", reader)
    monkeypatch.setattr(loaders, "read_today", Mock(return_value=data))
    monkeypatch.setattr(yesterday_ui, "datetime", FrozenDateTime)
    monkeypatch.setattr(yesterday_ui, "slate_date", lambda: NOW.date())
    loaders.load_today.clear()
    loaders.load_yesterday.clear()
    yield reader, data
    loaders.load_today.clear()
    loaders.load_yesterday.clear()


def open_yesterday():
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page("app_pages/yesterday.py").run()
    return app


def tables(app):
    return [
        soup
        for node in app.get("html")
        if (soup := BeautifulSoup(node.proto.body, "html.parser")).select_one(
            ".history-comparison"
        )
    ]


def test_yesterday_defaults_to_live_with_final_scores_and_fixed_checkpoints(
    yesterday_setup,
):
    reader, _ = yesterday_setup
    app = open_yesterday()
    assert not app.exception
    assert app.title[0].value == "Yesterday's predictions"
    assert app.session_state["yesterday_source"] == "Live predictions"
    assert app.session_state["yesterday_target"] == 1
    assert (
        app.date_input(key="yesterday_date").value
        == (NOW - pd.Timedelta(days=1)).date()
    )
    assert len(rendered_images(app)) == 5
    assert any(text.value == "### 114 — 121" for text in app.markdown)
    assert any("Total points **235**" in text.value for text in app.caption)
    assert any("28 Oct 2026, 03:00 Madrid" in text.value for text in app.caption)
    cards = tables(app)
    assert len(cards) == 2
    assert all(len(card.select("tbody tr")) == 12 for card in cards)
    assert all("Closing · Final pre-game" in card.get_text() for card in cards)
    assert all("Intermediate · T−8h" in card.get_text() for card in cards)
    assert any("PUSH" in card.get_text() for card in cards)
    assert not app.info
    assert reader.call_count == 1
    assert reader.call_args.args[0] == (NOW - pd.Timedelta(days=1)).date()


def test_source_and_targets_change_without_another_aiven_read(yesterday_setup):
    reader, _ = yesterday_setup
    app = open_yesterday()
    live = [
        cell.get_text()
        for card in tables(app)
        for cell in card.select(".prediction strong")
    ]
    assert set(live) == {"238.0"}
    app.session_state["yesterday_source"] = "Snapshot replay"
    app.run()
    assert not app.exception
    assert app.info and "reconstructed after the game" in app.info[0].value
    values = [
        cell.get_text()
        for card in tables(app)
        for cell in card.select(".prediction strong")
    ]
    assert set(values) == {"238.0", "248.0"}
    app.session_state["yesterday_target"] = 2
    app.run()
    assert not app.exception
    app.session_state["yesterday_target"] = 3
    app.run()
    assert not app.exception
    assert all("Spread error" in card.get_text() for card in tables(app))
    assert reader.call_count == 1


def test_selected_date_updates_results_summary_and_refreshes_only_that_date(
    yesterday_setup,
):
    reader, data = yesterday_setup
    previous_date = (NOW - pd.Timedelta(days=2)).date()
    games = data.games.loc[data.games["game_id"].eq(1)].copy()
    games["game_date"] = previous_date
    games["tipoff_at"] -= pd.Timedelta(days=1)
    games[["away_points", "home_points"]] = [100, 110]
    predictions = data.predictions.loc[data.predictions["game_id"].eq(1)].copy()
    for column in (
        "tipoff_at",
        "tipoff_at_prediction",
        "predicted_at",
        "run_finished_at",
        "input_at",
    ):
        predictions[column] -= pd.Timedelta(days=1)
    previous_data = TodayData(games, predictions, data.fetched_at)
    reader.side_effect = lambda game_date, **_: (
        previous_data if game_date == previous_date else data
    )
    app = open_yesterday()
    app.date_input(key="yesterday_date").set_value(previous_date).run()
    assert not app.exception
    assert reader.call_args.args[0] == previous_date
    assert reader.call_count == 2
    assert len(tables(app)) == 1
    assert any(text.value == "### 100 — 110" for text in app.markdown)
    assert any("NBA slate · 26 Oct 2026" in text.value for text in app.caption)
    assert app.table[0].value["Games"].eq("1/1").all()
    assert app.table[0].value["MAE"].eq("28.00 pts").all()

    app.session_state["yesterday_target"] = 2
    app.session_state["yesterday_source"] = "Snapshot replay"
    app.run()
    assert not app.exception
    assert app.date_input(key="yesterday_date").value == previous_date
    assert reader.call_count == 2
    app.button(key="yesterday_refresh").click().run()
    assert not app.exception
    assert reader.call_count == 3
    assert reader.call_args.args[0] == previous_date

    app.date_input(key="yesterday_date").set_value(
        data.games["game_date"].iloc[0]
    ).run()
    assert not app.exception
    assert len(tables(app)) == 2
    assert reader.call_count == 3


def test_selecting_empty_date_clears_previous_cards_and_summary(yesterday_setup):
    reader, data = yesterday_setup
    app = open_yesterday()
    reader.return_value = TodayData(pd.DataFrame(), pd.DataFrame(), data.fetched_at)
    app.date_input(key="yesterday_date").set_value(
        (NOW - pd.Timedelta(days=3)).date()
    ).run()
    assert not app.exception
    assert "selected NBA slate" in app.info[0].value
    assert not tables(app)
    assert not app.table
    assert not app.toggle


def test_table_and_details_show_scores_original_outputs_and_madrid_timestamps(
    yesterday_setup,
):
    app = open_yesterday()
    app.session_state["yesterday_view"] = "Table"
    app.run()
    assert not app.exception
    detail = app.dataframe[0].value
    assert len(detail) == 24
    assert set(detail["Final score (away–home)"]) == {"114–121"}
    assert {
        "Original model line",
        "Original output",
        "Bookmaker",
        "Source",
        "Result",
    } <= set(detail.columns)
    assert detail["Available at (Madrid)"].str.endswith("Madrid").all()
    app.session_state["yesterday_view"] = "Cards"
    app.run()
    app.toggle(key="yesterday_details_1").set_value(True).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 12


def test_changing_threshold_regrades_picks_and_keeps_predictions(yesterday_setup):
    app = open_yesterday()
    before = [
        cell.get_text()
        for card in tables(app)
        for cell in card.select(".prediction strong")
    ]
    app.selectbox(key="yesterday_min_edge_xgboost").select(4.0).run()
    assert not app.exception
    after = [
        cell.get_text()
        for card in tables(app)
        for cell in card.select(".prediction strong")
    ]
    assert after == before
    assert any("NO BET" in card.get_text() for card in tables(app))


def test_daily_summary_shows_win_rate_and_mae_per_model_before_all_cards(
    yesterday_setup,
):
    app = open_yesterday()
    nodes = list(app)
    summary = next(
        i
        for i, node in enumerate(nodes)
        if node.type == "subheader" and node.value == "Daily performance"
    )
    first_card = min(
        i
        for i, node in enumerate(nodes)
        if node.type == "toggle" and node.label == "View details"
    )
    assert summary < first_card
    assert not app.metric
    assert len(app.table) == 2
    checkpoint_summary = app.table[0].value
    assert list(checkpoint_summary.index) == [
        "Closing at 6h",
        "Closing at 4h",
        "Closing at closing",
        "Intermediate at 8h",
        "Intermediate at 6h",
        "Intermediate at 4h",
    ]
    assert checkpoint_summary["Games"].eq("2/2").all()
    assert checkpoint_summary.loc["Closing at 6h", "Wins"] == "2/2"
    assert checkpoint_summary.loc["Closing at closing", "Win rate"] == "—"
    assert checkpoint_summary["MAE"].eq("3.00 pts").all()
    pd.testing.assert_frame_equal(app.table[1].value, checkpoint_summary)
    assert not app.dataframe
    assert any(text.value == "**XGBoost · 2.5**" for text in app.markdown)
    assert any(text.value == "**TabPFN · 2.5**" for text in app.markdown)
    app.session_state["yesterday_source"] = "Snapshot replay"
    app.run()
    assert not app.exception
    assert not app.metric
    for table in app.table:
        checkpoint_summary = table.value
        assert checkpoint_summary.loc["Closing at 6h", "MAE"] == "3.00 pts"
        assert checkpoint_summary.loc["Intermediate at 6h", "MAE"] == "13.00 pts"
        assert checkpoint_summary.loc["Intermediate at 6h", "Win rate"] == "0.0%"


def test_tabpfn_summary_keeps_its_separate_timepoints_without_inventing_closing_models(
    yesterday_setup,
):
    _, data = yesterday_setup
    data.predictions.drop(
        data.predictions.loc[
            data.predictions["kind"].eq(3) & data.predictions["dataset_type"].eq(1)
        ].index,
        inplace=True,
    )
    app = open_yesterday()
    assert not app.exception
    tabpfn = app.table[1].value
    assert list(tabpfn.index) == [
        "Intermediate at 8h",
        "Intermediate at 6h",
        "Intermediate at 4h",
    ]
    assert tabpfn["Games"].eq("2/2").all()
    assert tabpfn["MAE"].eq("3.00 pts").all()


def test_missing_source_rows_pending_and_void_games_remain_visible(yesterday_setup):
    reader, data = yesterday_setup
    data.games.loc[
        data.games["game_id"].eq(1), ["status", "home_points", "away_points"]
    ] = [0, None, None]
    data.games.loc[data.games["game_id"].eq(2), "status"] = 2
    data.predictions.drop(
        data.predictions.loc[data.predictions["dataset_type"].eq(2)].index, inplace=True
    )
    app = open_yesterday()
    assert not app.exception
    assert any("Result pending" in text.value for text in app.markdown)
    assert any("Postponed / void" in text.value for text in app.markdown)
    assert not any("LOSS" in card.get_text() for card in tables(app))
    assert reader.call_count == 1


def test_no_predictions_still_shows_all_recorded_game_results(yesterday_setup):
    reader, data = yesterday_setup
    reader.return_value = TodayData(data.games, pd.DataFrame(), data.fetched_at)
    app = open_yesterday()
    assert not app.exception
    assert any("No stored model predictions" in info.value for info in app.info)
    assert sum("### 114 — 121" == text.value for text in app.markdown) == 2
    app.session_state["yesterday_view"] = "Table"
    app.run()
    assert not app.exception
    assert app.dataframe[0].value["Home points"].eq(121).all()


def test_empty_day_and_safe_aiven_error_are_handled(yesterday_setup):
    reader, data = yesterday_setup
    reader.return_value = TodayData(pd.DataFrame(), pd.DataFrame(), data.fetched_at)
    app = open_yesterday()
    assert not app.exception
    assert "No games were recorded" in app.info[0].value
    reader.side_effect = DashboardDataError("Could not read Aiven.")
    app.button(key="yesterday_refresh").click().run()
    assert not app.exception
    assert app.error[0].value == "Could not read Aiven."
