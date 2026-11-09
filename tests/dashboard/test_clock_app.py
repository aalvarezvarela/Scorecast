"""Hourly UI navigation, source separation, cached filters and audit tables."""

from unittest.mock import Mock

import pandas as pd
import pytest
from nba_ou.dashboard.repositories.clock import ClockData
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui import historical as historical_ui
from nba_ou.dashboard.ui import loaders

from .test_clock import AS_OF, DAY, SOURCES, clock_game, clock_row
from .test_historical_app import history_setup as history_setup
from .test_historical_app import open_history


@pytest.fixture
def clock_setup(request, monkeypatch):
    regular_setup = request.getfixturevalue("history_setup")
    rows = [
        clock_row(hour=hour, kind=kind, horizon=horizon, source=source)
        for hour in (19, 20)
        for source in SOURCES
        for kind, horizon in ((1, 0), (1, 360), (3, 240))
    ]
    reader = Mock(
        return_value=ClockData(
            pd.DataFrame([clock_game()]), pd.DataFrame(rows), AS_OF.to_pydatetime()
        )
    )
    monkeypatch.setattr(loaders, "read_clock_history", reader)
    monkeypatch.setattr(historical_ui, "slate_date", lambda: DAY)
    loaders.load_clock_history.clear()
    yield reader, regular_setup[0]
    loaders.load_clock_history.clear()


def test_hourly_view_uses_separate_read_and_defaults_to_19_madrid(clock_setup):
    reader, regular_reader = clock_setup
    app = open_history(history_view="Betting hour · Madrid")
    assert not app.exception
    assert reader.call_count == 1 and regular_reader.call_count == 0
    assert app.selectbox(key="history_clock_hour").value == 19
    assert app.selectbox(key="history_clock_hour").options == [
        f"{hour:02d}:00" for hour in range(12, 24)
    ] + ["00:00 (+1 day)"]
    assert app.select_slider(key="history_clock_range").value == (12, 24)
    assert app.session_state["history_clock_source"] == SOURCES[0]
    assert len(app.metric) == 3
    assert any("Betting time (Madrid)" in table.value for table in app.dataframe)
    details = next(
        table.value for table in app.dataframe if "Betting time (Madrid)" in table.value
    )
    assert details["Series"].tolist() == ["TabPFN", "Closing", "Intermediate"]
    assert details["Betting time (Madrid)"].str.contains("19:00").all()
    assert any("Input time (Madrid)" in table.value for table in app.dataframe)
    assert not any(toggle.key == "history_cross_target" for toggle in app.toggle)


def test_hour_source_metric_threshold_and_kind_filters_reuse_cached_read(clock_setup):
    reader, _ = clock_setup
    app = open_history(history_view="Betting hour · Madrid")
    app.selectbox(key="history_clock_hour").select(20).run()
    app.session_state["history_clock_source"] = SOURCES[1]
    app.session_state["history_metric"] = "ROI"
    app.run()
    assert not app.exception
    assert any("retrospective simulation" in node.value for node in app.info)
    app.selectbox(key="history_kind").select(3).run()
    app.selectbox(key="history_min_edge_tabpfn").select(6.0).run()
    app.select_slider(key="history_clock_range").set_value((19, 20)).run()
    app.toggle(key="history_clock_common").set_value(True).run()
    assert not app.exception
    assert len(app.metric) == 1
    assert reader.call_count == 1
    app.button(key="history_refresh").click().run()
    assert not app.exception and reader.call_count == 2


def test_detail_hour_can_be_outside_comparison_window(clock_setup):
    app = open_history(history_view="Betting hour · Madrid")
    app.select_slider(key="history_clock_range").set_value((20, 20)).run()
    assert not app.exception
    assert any("outside the chart" in node.value for node in app.caption)
    assert any("Betting time (Madrid)" in table.value for table in app.dataframe)


def test_hourly_target_loads_separately_and_errors_are_visible(clock_setup):
    reader, _ = clock_setup
    app = open_history(history_view="Betting hour · Madrid")
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
        AS_OF.to_pydatetime(),
    )
    app = open_history(history_view="Betting hour · Madrid")
    assert not app.exception
    if empty_games:
        assert any("No games were recorded" in node.value for node in app.info)
    else:
        details = next(
            table.value
            for table in app.dataframe
            if "Betting time (Madrid)" in table.value
        )
        assert details["Result"].eq("MISSING").all()
