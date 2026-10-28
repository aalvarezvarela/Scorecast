from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from bs4 import BeautifulSoup
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


def card_tables(app):
    tables = []
    for node in app.get("html"):
        soup = BeautifulSoup(node.proto.body, "html.parser")
        table = soup.select_one(".prediction-comparison table")
        if table is None:
            continue
        frame = pd.read_html(StringIO(str(table)), header=[0, 1], index_col=[0, 1])[0]
        frame.index.names = [th.get_text() for th in table.select("th[rowspan]")]
        tables.append(frame)
    return tables


def rendered_signals(app):
    signals = {
        value.split("-badge[", 1)[1].removesuffix("]")
        for value in (text.value for text in app.markdown)
        if "-badge[" in value
    }
    for node in app.get("html"):
        soup = BeautifulSoup(node.proto.body, "html.parser")
        signals.update(badge.get_text() for badge in soup.select(".pick"))
    return signals


def headlines(app):
    return [
        text.value.removeprefix("### ")
        for text in app.markdown
        if text.value.startswith("### ")
    ]


def test_today_combines_totals_with_line_error_and_shows_one_consensus(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert app.title[0].value == "Today's predictions"
    assert app.session_state["today_target"] == 1
    assert len(app.get("imgs")) == 5
    assert any("28 Oct 2026, 23:00 Madrid" in text.value for text in app.caption)
    # Total points says OVER, line error says UNDER: all four predictions vote.
    assert rendered_signals(app) == {"OVER", "UNDER"}
    assert headlines(app) == [":gray[No bet: tied vote]"] * 2
    assert (
        sum(
            text.value == ":gray[**2 OVER**] · :gray[**2 UNDER**]"
            for text in app.markdown
        )
        == 2
    )
    assert {m.label: m.value for m in app.metric}["Bets"] == "0"
    assert not app.multiselect
    tables = card_tables(app)
    assert len(tables) == 2
    assert all(
        list(table.columns)
        == [
            ("Total points", "Prediction"),
            ("Total points", "Margin"),
            ("Line error", "Prediction"),
            ("Line error", "Margin"),
        ]
        for table in tables
    )
    assert all(len(table) == 2 for table in tables)
    assert all(table.index.names == ["Model", "Snapshot"] for table in tables)
    assert not app.get("arrow_vega_lite_chart")


def test_tabpfn_is_listed_before_its_first_run(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert sum("TabPFN · no prediction yet" in text.value for text in app.caption) == 2


def test_targets_update_totals_and_spread_without_loading_again(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert rendered_signals(app) == {"OVER", "UNDER"}
    app.session_state["today_target"] = 3
    app.run()
    assert not app.exception
    assert rendered_signals(app) == {"AWAY"}
    # Bet365 implied home margin 3.5: the away side gets +3.5.
    assert headlines(app) == [
        ":blue[Golden State Warriors +3.5]",
        ":blue[New York Knicks +3.5]",
    ]
    app.session_state["today_target"] = 1
    app.run()
    assert not app.exception
    assert rendered_signals(app) == {"OVER", "UNDER"}
    assert app_setup.call_count == 1


def test_three_over_votes_beat_one_under_vote(app_setup, today_data):
    mask = (
        today_data.predictions["game_id"].eq(1)
        & today_data.predictions["target"].eq(1)
        & today_data.predictions["dataset_type"].eq(1)
    )
    today_data.predictions.loc[mask, "prediction_value"] = 3.0
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert set(headlines(app)) == {":gray[No bet: tied vote]", ":red[OVER 230.5]"}
    assert any(
        text.value == ":red[**3 OVER**] · :gray[**1 UNDER**]" for text in app.markdown
    )
    assert {m.label: m.value for m in app.metric}["Tied votes"] == "1"
    assert {m.label: m.value for m in app.metric}["Bets"] == "1"


def test_line_freshness_shows_last_move_and_flags_an_old_read(app_setup, today_data):
    today_data.predictions["input_at"] = NOW - pd.Timedelta(minutes=68)
    today_data.predictions["bet_line_age_minutes"] = 58.0
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    # 18:00 UTC is 19:00 Madrid: read at 17:52, last move 58 min before.
    stale = [text.value for text in app.caption if "Last Bet365 move" in text.value]
    assert stale and all(
        value.startswith("Last Bet365 move 16:54 · :orange[line read 17:52 Madrid")
        for value in stale
    )
    assert all("68 min ago" in value for value in stale)


def test_table_retains_logos_market_lines_and_madrid_times(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    table = app.dataframe[0].value
    assert len(table) == 8
    assert set(table["Target"]) == {"Line error", "Total points"}
    assert table["Away logo"].str.contains("espncdn.com").all()
    assert table["Tip-off (Madrid)"].str.endswith("Madrid").all()
    assert {"Bet365", "Market mean", "Market median"} <= set(table.columns)


def test_no_target_predictions_still_shows_the_games(app_setup, today_data):
    app_setup.return_value = TodayData(
        today_data.games, today_data.predictions.iloc[0:0], today_data.fetched_at
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert len(app.get("imgs")) == 5
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
    assert any(":blue[New York Knicks +7.5]" == h for h in headlines(app))
    app.selectbox(key="today_reference").select("Market mean").run()
    assert not app.exception
    assert rendered_signals(app) == {"HOME"}
    assert any(":red[Boston Celtics -3.0]" == h for h in headlines(app))
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    assert set(app.dataframe[0].value["Spread error"]) == {2.0}
    assert set(app.dataframe[0].value["Edge vs Market mean"]) == {2.5}
    assert set(app.dataframe[0].value["Signal"]) == {"HOME"}
    assert app_setup.call_count == 1


def test_minimum_edge_neutralizes_badges_but_preserves_values(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert rendered_signals(app) == {"OVER", "UNDER"}
    before = [
        str(value)
        for table in card_tables(app)
        for value in table.xs("Prediction", axis=1, level=1).to_numpy().flat
    ]
    app.selectbox(key="today_min_edge_xgboost").select(2.0).run()
    assert not app.exception
    assert rendered_signals(app) == {"NO EDGE"}
    assert headlines(app) == [":gray[No bet]"] * 2
    assert [
        str(value)
        for table in card_tables(app)
        for value in table.xs("Prediction", axis=1, level=1).to_numpy().flat
    ] == before
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


def test_fixed_models_include_new_tabpfn_but_exclude_other_registry_schemas(
    app_setup, today_data
):
    app = AppTest.from_file(str(ENTRYPOINT))
    # Retired preferences cannot hide any of the fixed models.
    app.session_state["today_models_1"] = []
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    assert not app.multiselect
    extra = today_data.predictions.copy()
    extra["kind"] = 3
    extra["workflow"] = 3
    extra["variant"] = "tabpfn_v3_5"
    other_schema = today_data.predictions.copy()
    other_schema["schema_version"] = "2_6"
    app_setup.return_value = TodayData(
        today_data.games,
        pd.concat([today_data.predictions, extra, other_schema], ignore_index=True),
        today_data.fetched_at,
    )
    loaders.load_today.clear()
    app.run()
    assert not app.exception
    table = app.dataframe[0].value
    assert len(table) == 16
    assert table["Model"].str.startswith("TabPFN").sum() == 8
    assert not table["Model"].str.contains("2.6", regex=False).any()
    assert not app.multiselect


@pytest.mark.parametrize("view", ["Cards", "Table"])
def test_tonights_bets_summary_is_after_all_games(app_setup, view):
    app = AppTest.from_file(str(ENTRYPOINT))
    app.session_state["today_view"] = view
    app.run()
    assert not app.exception
    nodes = list(app)
    footer = next(
        i
        for i, node in enumerate(nodes)
        if node.type == "subheader" and node.value == "Tonight's bets"
    )
    if view == "Cards":
        last_game = max(
            i
            for i, node in enumerate(nodes)
            if node.type == "toggle" and node.label == "View details"
        )
    else:
        last_game = max(
            i for i, node in enumerate(nodes) if node.type == "arrow_data_frame"
        )
    assert footer > last_game


def test_all_targets_and_tabpfn_participate_in_the_large_summary(app_setup, today_data):
    # All four registry outputs per game agree UNDER.
    today_data.predictions.loc[
        today_data.predictions["target"].eq(2), "prediction_value"
    ] = 230.0
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert headlines(app) == [":blue[UNDER 230.5]"] * 2
    assert sum(text.value == ":blue[**4 UNDER**]" for text in app.markdown) == 2

    # The TabPFN vote is counted, but one opposing vote does not veto four UNDERs.
    extra = today_data.predictions.loc[
        today_data.predictions["game_id"].eq(1)
        & today_data.predictions["target"].eq(1)
        & today_data.predictions["model_horizon_minutes"].eq(240)
    ].copy()
    extra["kind"] = 3
    extra["workflow"] = 3
    extra["prediction_value"] = 3.0
    app_setup.return_value = TodayData(
        today_data.games,
        pd.concat([today_data.predictions, extra]),
        today_data.fetched_at,
    )
    loaders.load_today.clear()
    app.run()
    assert not app.exception
    assert headlines(app) == [":blue[UNDER 230.5]"] * 2
    assert any(
        text.value == ":gray[**1 OVER**] · :blue[**4 UNDER**]" for text in app.markdown
    )
    assert not any("Majority:" in text.value for text in app.caption)
    assert {m.label: m.value for m in app.metric}["Bets"] == "2"
    # Both model families share one table, retaining TabPFN's own snapshot.
    tables = card_tables(app)
    assert len(tables) == 2
    combined = next(
        table
        for table in tables
        if table.index.get_level_values("Model").str.startswith("TabPFN").any()
    )
    assert len(combined) == 3
    assert combined.index.names == ["Model", "Snapshot"]
    assert combined.index.get_level_values("Model").str.startswith("TabPFN").sum() == 1
    tabpfn = combined.loc[
        combined.index.get_level_values("Model").str.startswith("TabPFN")
    ]
    assert list(tabpfn.index.get_level_values("Snapshot")) == ["T−4h"]
    assert tabpfn.iloc[0][("Total points", "Prediction")] == "—"
    assert "OVER" in tabpfn.iloc[0][("Line error", "Margin")]


def test_team_names_are_full_on_cards_table_and_bet_summary(app_setup, today_data):
    today_data.games["home_team"] = ["BOS", "LAL"]
    today_data.games["away_team"] = ["NYK", "GSW"]
    today_data.predictions["home_team"] = "abbr"
    today_data.predictions["away_team"] = "abbr"
    today_data.predictions.loc[
        today_data.predictions["target"].eq(2), "prediction_value"
    ] = 230.0
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert {text.value for text in app.markdown} >= {
        "**Boston Celtics**",
        "**New York Knicks**",
        "**Los Angeles Lakers**",
        "**Golden State Warriors**",
    }
    assert any(
        "New York Knicks @ Boston Celtics" in text.value for text in app.markdown
    )
    app.session_state["today_view"] = "Table"
    app.run()
    assert not app.exception
    assert set(app.dataframe[0].value["Matchup"]) == {
        "New York Knicks @ Boston Celtics",
        "Golden State Warriors @ Los Angeles Lakers",
    }


def test_old_total_points_selection_reopens_on_combined_totals(app_setup):
    app = AppTest.from_file(str(ENTRYPOINT))
    app.session_state["today_target"] = 2
    app.run()
    assert not app.exception
    assert app.session_state["today_target"] == 1
    assert rendered_signals(app) == {"OVER", "UNDER"}
