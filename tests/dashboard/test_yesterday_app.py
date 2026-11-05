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


def open_yesterday(source="Live predictions", target=1):
    """The page on ``source`` and ``target``; None keeps the default (replay,
    Summary)."""
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page("app_pages/yesterday.py").run()
    if source is not None:
        app.session_state["yesterday_source"] = source
    if target is not None:
        app.session_state["yesterday_target"] = target
    if source is not None or target is not None:
        app.run()
    return app


def test_yesterday_defaults_to_the_replay_summary(yesterday_setup):
    reader, _ = yesterday_setup
    app = open_yesterday(source=None, target=None)
    assert not app.exception
    assert app.session_state["yesterday_source"] == "Snapshot replay"
    assert app.session_state["yesterday_target"] == 0
    target = app.button_group(key="yesterday_target")
    assert target.options[0] == "Summary"
    assert "Slate summary" in [node.value for node in app.subheader]
    assert not tables(app)
    assert reader.call_count == 1


def test_yesterday_replay_cards_show_reconstructed_outputs(yesterday_setup):
    reader, _ = yesterday_setup
    app = open_yesterday(source=None)
    assert not app.exception
    assert app.session_state["yesterday_source"] == "Snapshot replay"
    assert app.info and "reconstructed after the game" in app.info[0].value
    values = {
        cell.get_text()
        for card in tables(app)
        for cell in card.select(".prediction strong")
    }
    assert values == {"248.0"}
    assert reader.call_count == 1


def tables(app):
    return [
        soup
        for node in app.get("html")
        if (soup := BeautifulSoup(node.proto.body, "html.parser")).select_one(
            ".history-comparison"
        )
    ]


def test_live_shows_final_scores_and_fixed_checkpoints(
    yesterday_setup,
):
    reader, _ = yesterday_setup
    app = open_yesterday()
    assert not app.exception
    assert app.title[0].value == "Yesterday's predictions"
    assert app.session_state["yesterday_target"] == 1
    assert (
        app.date_input(key="yesterday_date").value
        == (NOW - pd.Timedelta(days=1)).date()
    )
    assert len(rendered_images(app)) == 5
    assert any(text.value == "### 114 — 121" for text in app.markdown)
    assert any(text.value == "Total points" for text in app.caption)
    assert any(text.value == "**235**" for text in app.markdown)
    assert any(text.value == "**Celtics by 7**" for text in app.markdown)
    assert any(text.value == "Pre-game line" for text in app.caption)
    # The final pre-game quote (235.0) equals the final total.
    assert any(text.value == "**Push**" for text in app.markdown)
    assert any("28 Oct 2026, 03:00 Madrid" in text.value for text in app.caption)
    cards = tables(app)
    assert len(cards) == 2
    assert all(len(card.select("tbody tr")) == 18 for card in cards)
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
    assert set(values) == {"248.0"}
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
    assert app.table[0].value["Games"].eq(1).all()
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
    assert len(detail) == 36
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
    assert len(app.dataframe[0].value) == 18


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
        "Closing at 8h",
        "Closing at 6h",
        "Closing at 4h",
        "Closing at closing",
        "Intermediate at 12h",
        "Intermediate at 8h",
        "Intermediate at 6h",
        "Intermediate at 4h",
        "Intermediate at 2h",
    ]
    assert checkpoint_summary["Games"].eq(2).all()
    assert checkpoint_summary.loc["Closing at 6h", "Bets"] == 2
    assert checkpoint_summary.loc["Closing at 6h", "Wins"] == 2
    assert checkpoint_summary.loc["Closing at closing", "Bets"] == 0
    assert checkpoint_summary.loc["Closing at closing", "Wins"] == 0
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
    for index, table in enumerate(app.table):
        checkpoint_summary = table.value
        assert checkpoint_summary.loc["Closing at 6h", "MAE"] == (
            "13.00 pts" if index == 0 else "—"
        )
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
        "Intermediate at 12h",
        "Intermediate at 8h",
        "Intermediate at 6h",
        "Intermediate at 4h",
        "Intermediate at 2h",
    ]
    assert tabpfn["Games"].eq(2).all()
    assert tabpfn["MAE"].eq("3.00 pts").all()


def test_summary_separates_evaluated_games_from_bets_when_a_model_abstains(
    yesterday_setup,
):
    _, data = yesterday_setup
    data.games.loc[data.games["game_id"].eq(1), "away_points"] = 119
    closing = data.predictions["target"].eq(1) & data.predictions["dataset_type"].eq(1)
    data.predictions.loc[closing & data.predictions["game_id"].eq(1), "bet_line_x2"] = (
        474
    )
    data.predictions.loc[closing & data.predictions["game_id"].eq(2), "bet_line_x2"] = (
        476
    )

    app = open_yesterday()
    assert not app.exception
    for table in app.table:
        final = table.value.loc["Closing at closing"]
        assert (final["Games"], final["Bets"], final["Wins"]) == (2, 1, 1)
        assert final["Win rate"] == "100.0%"
        assert final["MAE"] == "2.50 pts"
    assert any("Win rate = Wins / Bets" in caption.value for caption in app.caption)


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


@pytest.mark.parametrize(
    ("line", "spread", "expected"),
    [
        (None, False, []),
        (
            230.5,
            False,
            [("Pre-game line", "230.5"), ("Result", "Over by 4.5")],
        ),
        (
            240.0,
            False,
            [("Pre-game line", "240.0"), ("Result", "Under by 5.0")],
        ),
        (
            3.5,
            True,
            [
                ("Pre-game spread", "Celtics -3.5"),
                ("Result", "Celtics cover by 3.5"),
            ],
        ),
        (
            -2.0,
            True,
            [
                ("Pre-game spread", "Celtics +2.0"),
                ("Result", "Celtics cover by 9.0"),
            ],
        ),
        (
            9.5,
            True,
            [
                ("Pre-game spread", "Celtics -9.5"),
                ("Result", "Knicks cover by 2.5"),
            ],
        ),
        (
            7.0,
            True,
            [("Pre-game spread", "Celtics -7.0"), ("Result", "Push")],
        ),
    ],
)
def test_result_strip_states_total_margin_and_outcome_against_the_line(
    line, spread, expected
):
    from nba_ou.dashboard.ui.teams import result_stats

    game = {
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "home_team_id": 1610612738,
        "away_team_id": 1610612752,
        "home_points": 121,
        "away_points": 114,
    }
    assert result_stats(game, line=line, spread=spread) == [
        ("Total points", "235"),
        ("Margin", "Celtics by 7"),
        *expected,
    ]


def test_summary_shows_every_targets_slate_win_rate_without_game_cards(
    yesterday_setup,
):
    reader, _ = yesterday_setup
    app = open_yesterday()
    app.session_state["yesterday_target"] = 0
    app.run()
    assert not app.exception
    assert "Slate summary" in [node.value for node in app.subheader]
    assert not tables(app)
    assert not app.toggle
    metrics = {node.label: node.value for node in app.metric}
    assert list(metrics) == ["Line error", "Total points", "Spread"]
    assert metrics["Total points"] == "100.0%"
    assert len(app.table) == 2
    xgboost = app.table[0].value
    assert list(xgboost.columns) == ["Line error", "Total points", "Spread"]
    assert list(xgboost.index)[:2] == ["All checkpoints", "Closing at 8h"]
    assert "Intermediate at 12h" in xgboost.index
    assert xgboost.loc["All checkpoints", "Total points"] == "100.0% (16–0)"
    # The final pre-game total pushes: no bets at that checkpoint.
    assert xgboost.loc["Closing at closing", "Total points"] == "—"
    assert any(text.value == "**TabPFN · 2.5**" for text in app.markdown)
    assert app.session_state["yesterday_view"] == "Cards"
    app.session_state["yesterday_target"] = 1
    app.run()
    assert not app.exception
    assert len(tables(app)) == 2
    assert reader.call_count == 1


def test_slate_summary_tables_pool_first_then_checkpoints_by_order():
    from nba_ou.dashboard.ui.history_table import slate_summary_tables

    def summary(rows, by_checkpoint):
        frame = pd.DataFrame(rows, columns=["Kind", "Model", "Wins", "Losses"])
        if by_checkpoint:
            frame["Checkpoint"] = ["Intermediate · T−2h", "Closing · Final pre-game"]
            frame["Checkpoint order"] = [20000 - 120, 10000]
        return frame

    pooled = {
        "Line error": summary([(1, "XGBoost · 2.5", 3, 1)], False),
        "Spread": summary([(1, "XGBoost · 2.5", 0, 0)], False),
    }
    by_checkpoint = {
        "Line error": summary(
            [(1, "XGBoost · 2.5", 2, 0), (1, "XGBoost · 2.5", 1, 1)], True
        )
    }
    table = slate_summary_tables(pooled, by_checkpoint)["XGBoost · 2.5"]
    assert list(table.index) == [
        "All checkpoints",
        "Closing at closing",
        "Intermediate at 2h",
    ]
    assert list(table.columns) == ["Line error", "Spread"]
    assert table.loc["All checkpoints", "Line error"] == "75.0% (3–1)"
    assert table.loc["All checkpoints", "Spread"] == "—"
    assert table.loc["Intermediate at 2h", "Line error"] == "100.0% (2–0)"
    assert table.loc["Closing at closing", "Spread"] == "—"
