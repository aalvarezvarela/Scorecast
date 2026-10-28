from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from nba_ou.dashboard.repositories.predictions import DashboardDataError, TodayData
from nba_ou.dashboard.ui import loaders
from nba_ou.dashboard.ui import today as today_ui
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup

from .conftest import NOW, FrozenDateTime

ENTRYPOINT = Path(__file__).resolve().parents[2] / "streamlit/streamlit_app.py"


@pytest.fixture
def app_setup(monkeypatch, today_data):
    # Streamlit 1.54's AppTest assumes every button group returns a list;
    # single-select segmented controls return scalars. Adapt only the test
    # serializer, preserving the widget's real frontend index representation.
    def indices(group):
        values = group.value
        if not isinstance(values, list):
            values = [] if values is None else [values]
        return [group.options.index(group.format_func(value)) for value in values]

    monkeypatch.setattr(ButtonGroup, "indices", property(indices))
    loaders.load_today.clear()
    reader = Mock(return_value=today_data)
    monkeypatch.setattr(loaders, "read_today", reader)
    monkeypatch.setattr(today_ui, "datetime", FrozenDateTime)
    monkeypatch.setattr(today_ui, "slate_date", lambda: NOW.date())
    yield reader
    loaders.load_today.clear()


def rendered_signals(app):
    return {
        text.value.split("-badge[", 1)[1].removesuffix("]")
        for text in app.markdown
        if "-badge[" in text.value
    }


def test_today_shows_both_team_images_madrid_times_and_comparisons(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert app.title[0].value == "Today's predictions"
    assert len(app.get("imgs")) == 4
    assert any("28 Oct 2026, 23:00 Madrid" in text.value for text in app.markdown)
    assert rendered_signals(app) == {"OVER"}
    assert len(app.get("arrow_vega_lite_chart")) == 2


def test_targets_update_totals_and_spread_without_loading_again(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.session_state["today_target"] = 1
    app.run()
    assert not app.exception
    assert rendered_signals(app) == {"NO EDGE"}
    app.session_state["today_target"] = 3
    app.run()
    assert not app.exception
    assert rendered_signals(app) == {"AWAY"}
    assert any(m.label == "Predicted spread error" for m in app.metric)
    assert not any(m.label == "Predicted total" for m in app.metric)
    assert app_setup.call_count == 1


@pytest.mark.parametrize(
    "filename,title",
    [
        ("yesterday.py", "Yesterday's predictions"),
        ("historical.py", "Historical analysis"),
    ],
)
def test_other_pages_contain_only_their_titles_without_reads(
    app_setup, filename, title
):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    count = app_setup.call_count
    app.switch_page(f"app_pages/{filename}").run()
    assert not app.exception
    assert [t.value for t in app.title] == [title]
    assert not app.metric
    assert not app.markdown
    assert not app.caption
    assert app_setup.call_count == count


def test_table_retains_logos_market_lines_and_madrid_times(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    table = app.dataframe[0].value
    assert len(table) == 4
    assert table["Away logo"].str.contains("espncdn.com").all()
    assert table["Tip-off (Madrid)"].str.endswith("Madrid").all()
    assert {"Bet365", "Market mean", "Market median"} <= set(table.columns)


def test_no_target_predictions_still_shows_the_games(app_setup, today_data):
    app_setup.return_value = TodayData(
        today_data.games, today_data.predictions.iloc[0:0], today_data.fetched_at
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert len(app.get("imgs")) == 4
    assert any("No live predictions" in info.value for info in app.info)


def test_connection_error_does_not_show_a_false_empty_slate(app_setup):
    app_setup.side_effect = DashboardDataError("Could not read Aiven.")
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert app.error[0].value == "Could not read Aiven."
    assert not app.info


def test_refresh_invalidates_source_cache(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.button[0].click().run()
    assert not app.exception
    assert app_setup.call_count == 2


def test_spread_reference_changes_badge_and_exposes_market_edge(app_setup, today_data):
    mask = today_data.predictions["target"].eq(3)
    today_data.predictions.loc[mask, "prediction_value"] = 2.0
    today_data.predictions.loc[mask, "bet_line_x2"] = 15
    app = AppTest.from_file(str(ENTRYPOINT))
    app.session_state["today_target"] = 3
    app.run()
    assert not app.exception
    assert rendered_signals(app) == {"AWAY"}
    assert {m.value for m in app.metric if m.label == "Predicted spread error"} == {
        "+2.0"
    }
    assert {m.value for m in app.metric if m.label == "Edge vs Bet365"} == {"-2.0"}
    app.selectbox(key="today_reference").select("Market mean").run()
    assert not app.exception
    assert rendered_signals(app) == {"HOME"}
    assert {m.value for m in app.metric if m.label == "Edge vs Market mean"} == {"+2.5"}
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    assert set(app.dataframe[0].value["Spread error"]) == {2.0}
    assert set(app.dataframe[0].value["Edge vs Market mean"]) == {2.5}
    assert set(app.dataframe[0].value["Signal"]) == {"HOME"}
    assert app_setup.call_count == 1


def test_minimum_edge_neutralizes_badges_but_preserves_values(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert rendered_signals(app) == {"OVER"}
    before = [m.value for m in app.metric if m.label == "Predicted total"]
    app.selectbox(key="today_min_edge").select(2.0).run()
    assert not app.exception
    assert rendered_signals(app) == {"NO EDGE"}
    assert [m.value for m in app.metric if m.label == "Predicted total"] == before
    assert app_setup.call_count == 1


def test_table_honors_largest_edge_sort_and_can_return_to_tipoff(app_setup, today_data):
    today_data.predictions.loc[
        today_data.predictions["game_id"].eq(1)
        & today_data.predictions["target"].eq(2),
        "prediction_value",
    ] = 240.0
    app = AppTest.from_file(str(ENTRYPOINT))
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    assert "Lakers" in app.dataframe[0].value.iloc[0]["Matchup"]
    app.selectbox(key="today_sort").select("Largest edge").run()
    assert not app.exception
    assert "Celtics" in app.dataframe[0].value.iloc[0]["Matchup"]
    assert app.dataframe[0].value.iloc[0]["Edge vs Bet365"] == 9.5
    app.selectbox(key="today_sort").select("Tip-off").run()
    assert not app.exception
    assert "Lakers" in app.dataframe[0].value.iloc[0]["Matchup"]
    assert app_setup.call_count == 1


def test_new_model_families_are_selected_without_restoring_removed_ones(
    app_setup, today_data
):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.multiselect(key="today_models_2").set_value([]).run()
    assert not app.exception
    extra = today_data.predictions.loc[today_data.predictions["target"].eq(2)].copy()
    extra["variant"] = "alt"
    app_setup.return_value = TodayData(
        today_data.games,
        pd.concat([today_data.predictions, extra], ignore_index=True),
        today_data.fetched_at,
    )
    loaders.load_today.clear()
    app.run()
    assert not app.exception
    selected = app.session_state["today_models_2"]
    assert selected and all(key.endswith(":alt") for key in selected)
