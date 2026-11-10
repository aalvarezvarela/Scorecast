"""Hourly UI: four series, source separation, cached filters and audit tables."""

from unittest.mock import Mock

import pandas as pd
import pytest
from nba_ou.dashboard.repositories.clock import ClockData
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui import historical as historical_ui
from nba_ou.dashboard.ui import loaders

from .conftest import rendered_charts
from .test_clock import (
    AS_OF,
    DAY,
    SOURCES,
    TAB_6H,
    TAB_CLOSE,
    TAB_FINAL,
    TAB_MEAN,
    XGB_6H,
    XGB_8H,
    XGB_CLOSE,
    XGB_FINAL,
    XGB_MEAN,
    clock_game,
    output,
    quote,
)
from .test_historical_app import history_setup as history_setup
from .test_historical_app import open_history

VIEW = "Betting hour · Madrid"


def clock_read(games=(1,)):
    return ClockData(
        pd.DataFrame([clock_game(game) for game in games]),
        pd.DataFrame(
            [
                output(slot, game_id=game, source=source)
                for game in games
                for source in SOURCES
                for slot in (XGB_6H, XGB_8H, XGB_CLOSE, TAB_6H, TAB_CLOSE)
            ]
        ),
        pd.DataFrame(
            [quote(game_id=game, source=source) for game in games for source in SOURCES]
        ),
        AS_OF.to_pydatetime(),
    )


@pytest.fixture
def clock_setup(request, monkeypatch):
    regular_setup = request.getfixturevalue("history_setup")
    reader = Mock(return_value=clock_read())
    monkeypatch.setattr(loaders, "read_clock_history", reader)
    monkeypatch.setattr(historical_ui, "slate_date", lambda: DAY)
    loaders.load_clock_history.clear()
    yield reader, regular_setup[0]
    loaders.load_clock_history.clear()


def hour_table(app):
    return next(
        table.value for table in app.dataframe if "Hour (Madrid)" in table.value
    )


def test_hourly_view_has_a_table_by_hour_without_an_hour_selector(clock_setup):
    reader, regular_reader = clock_setup
    app = open_history(history_view=VIEW)
    assert not app.exception
    assert reader.call_count == 1 and regular_reader.call_count == 0
    assert not any(box.key == "history_clock_hour" for box in app.selectbox)
    assert app.select_slider(key="history_clock_range").value == (12, 24)
    assert app.session_state["history_clock_source"] == SOURCES[1]
    table = hour_table(app)
    assert table.columns.tolist() == [
        "Hour (Madrid)",
        XGB_MEAN,
        XGB_FINAL,
        TAB_MEAN,
        TAB_FINAL,
    ]
    assert len(table) == 13
    assert table[XGB_MEAN].iloc[-1].startswith("100.0% · 1-0")
    headers = [node.value for node in app.subheader]
    assert "Selected-hour game detail" not in headers
    assert "Best hours and models" in headers and "Results by hour" in headers
    assert not any(toggle.key == "history_cross_target" for toggle in app.toggle)
    assert rendered_charts(app)


def test_best_cards_pick_four_hour_and_series_pairs(clock_setup):
    reader, _ = clock_setup
    reader.return_value = clock_read(games=range(35))
    app = open_history(history_view=VIEW)
    assert not app.exception
    labels = [node.label for node in app.metric]
    assert labels.count("Coin-flip chance") == 1
    assert labels.count("Game-permutation p") == 1
    assert len(labels) == 6
    assert set(labels[:4]) <= {XGB_MEAN, XGB_FINAL, TAB_MEAN, TAB_FINAL}


def test_hour_source_metric_threshold_and_kind_filters_reuse_cached_read(clock_setup):
    reader, _ = clock_setup
    app = open_history(history_view=VIEW)
    app.session_state["history_clock_source"] = SOURCES[0]
    app.session_state["history_metric"] = "ROI"
    app.run()
    assert not app.exception
    assert not any("retrospective simulation" in node.value for node in app.info)
    app.selectbox(key="history_kind").select(3).run()
    app.selectbox(key="history_min_edge_tabpfn").select(6.0).run()
    app.select_slider(key="history_clock_range").set_value((19, 20)).run()
    app.toggle(key="history_clock_common").set_value(True).run()
    assert not app.exception
    assert hour_table(app).columns.tolist() == [
        "Hour (Madrid)",
        TAB_MEAN,
        TAB_FINAL,
    ]
    assert reader.call_count == 1
    app.button(key="history_refresh").click().run()
    assert not app.exception and reader.call_count == 2


def test_hour_range_narrows_the_table(clock_setup):
    app = open_history(history_view=VIEW)
    app.select_slider(key="history_clock_range").set_value((20, 21)).run()
    assert not app.exception
    assert hour_table(app)["Hour (Madrid)"].tolist() == ["20:00", "21:00"]


def test_hourly_target_loads_separately_and_errors_are_visible(clock_setup):
    reader, _ = clock_setup
    app = open_history(history_view=VIEW)
    reader.side_effect = DashboardDataError("Hourly fixture unavailable")
    app.session_state["history_target"] = 3
    app.run()
    assert not app.exception
    assert reader.call_args.args[2] == 3
    assert app.error[0].value == "Hourly fixture unavailable"


@pytest.mark.parametrize("empty_games", [True, False])
def test_hourly_empty_games_or_predictions_do_not_crash(clock_setup, empty_games):
    reader, _ = clock_setup
    reader.return_value = ClockData(
        pd.DataFrame() if empty_games else pd.DataFrame([clock_game()]),
        pd.DataFrame(),
        pd.DataFrame(),
        AS_OF.to_pydatetime(),
    )
    app = open_history(history_view=VIEW)
    assert not app.exception
    if empty_games:
        assert any("No games were recorded" in node.value for node in app.info)
    else:
        assert hour_table(app)[XGB_MEAN].eq("—").all()
