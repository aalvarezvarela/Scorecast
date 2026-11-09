"""Historical page controls, cached reads, empty/error states and chart rendering."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from nba_ou.dashboard.repositories.analysis import AnalysisData
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui import historical as historical_ui
from nba_ou.dashboard.ui import loaders
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup

from .conftest import NOW, rendered_charts
from .test_historical import analysis_row
from .test_history import historical_slate

ENTRYPOINT = Path(__file__).resolve().parents[2] / "streamlit/streamlit_app.py"


@pytest.fixture
def history_setup(monkeypatch):
    # Streamlit 1.54 AppTest single-choice segmented-control serializer workaround.
    def indices(group):
        values = group.value
        if not isinstance(values, list):
            values = [] if values is None else [values]
        return [group.options.index(group.format_func(value)) for value in values]

    monkeypatch.setattr(ButtonGroup, "indices", property(indices))
    slate = historical_slate()

    def result(start, end, target, season_type, **kwargs):
        rows = [
            analysis_row(
                game_id=game,
                horizon=horizon,
                bucket=bucket,
                source=source,
                target=target,
                kind=kind,
            )
            for game in (1, 2)
            for kind in (1, 3)
            for horizon, bucket in (
                (0, 0),
                (0, 360),
                (360, 360),
                (480, 480),
                (360, 240),
            )
            for source in ("Live predictions", "Snapshot replay")
            if source != "Snapshot replay" or horizon == 0 or horizon == bucket
        ]
        return AnalysisData(slate.games.copy(), pd.DataFrame(rows), NOW.to_pydatetime())

    reader = Mock(side_effect=result)
    today_reader = Mock(return_value=slate)
    monkeypatch.setattr(loaders, "read_today", today_reader)
    monkeypatch.setattr(loaders, "read_history_range", reader)
    monkeypatch.setattr(historical_ui, "slate_date", lambda: NOW.date())
    loaders.load_today.clear()
    loaders.load_history.clear()
    yield reader, today_reader
    loaders.load_today.clear()
    loaders.load_history.clear()


def open_history(**state):
    """The page opened directly, with ``state`` preset, in one run: every app
    run costs ~0.5 s. Navigating from Today is covered by the defaults test."""
    app = AppTest.from_file(str(ENTRYPOINT))
    app.switch_page("app_pages/historical.py")
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


def test_history_defaults_to_october_28_including_backfills(history_setup):
    reader, today_reader = history_setup
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page("app_pages/historical.py").run()
    assert not app.exception
    assert app.title[0].value == "Historical analysis"
    assert app.date_input(key="history_dates").value == (
        date(2026, 10, 28),
        date(2026, 10, 28),
    )
    assert reader.call_count == 1
    assert reader.call_args.args == (date(2026, 10, 28), date(2026, 10, 28), 1, 2)
    assert today_reader.call_count == 1  # navigation itself never re-reads Today
    assert app.session_state["history_metric"] == "Win %"
    assert app.session_state["history_kind"] is None
    assert [node.value for node in app.subheader] == [
        "Best single slots",
        "Performance overview",
        "Closing by prediction time",
        "Closing by input time · Replay",
        "Intermediate by model horizon · Replay",
        "Live model × time",
        "Own snapshot · Live vs replay",
    ]
    assert {node.label for node in app.metric} == {
        "Final games",
        "Closing · Final",
        "Closing · Replay",
        "Own snapshot · Live",
        "Live · ≥ 6h",
        "Live · < 6h",
        "Own snapshot · Replay",
        "Replay · ≥ 6h",
        "Replay · < 6h",
    }
    # Two games never reach the minimum sample for a best slot.
    assert any("No slot has 30 settled bets" in text.value for text in app.caption)
    assert len(rendered_charts(app)) == 5
    assert any("95%" in text.value for text in app.caption)
    assert any("Madrid" in text.value for text in app.caption)
    assert any("Different picks" in table.value for table in app.dataframe)
    assert any(
        "Each point represents an intermediate model" in text.value
        for text in app.caption
    )


def test_default_end_date_tracks_today(history_setup, monkeypatch):
    reader, _ = history_setup
    monkeypatch.setattr(historical_ui, "slate_date", lambda: date(2026, 11, 2))
    app = open_history()
    assert not app.exception
    assert app.date_input(key="history_dates").value == (
        date(2026, 10, 28),
        date(2026, 11, 2),
    )
    assert reader.call_args.args[:2] == (date(2026, 10, 28), date(2026, 11, 2))


def test_replay_timeline_handles_no_replay_without_using_live_rows(history_setup):
    reader, _ = history_setup
    app = open_history()
    data = reader.side_effect(*reader.call_args.args)
    reader.side_effect = None
    reader.return_value = AnalysisData(
        data.games,
        data.predictions.loc[data.predictions["source"].eq("Live predictions")],
        data.fetched_at,
    )
    app.button(key="history_refresh").click().run()
    assert not app.exception
    assert len(rendered_charts(app)) == 3
    assert any(
        "No own-snapshot intermediate replay predictions" in text.value
        for text in app.caption
    )


def test_local_filters_and_thresholds_do_not_repeat_aiven_reads(history_setup):
    reader, _ = history_setup
    app = open_history()
    app.selectbox(key="history_kind").select(3).run()
    assert not app.exception
    for table in app.dataframe:
        if "Model" in table.value:
            assert table.value["Model"].str.startswith("TabPFN").all()
    app.session_state["history_metric"] = "MAE"
    app.run()
    assert not app.exception
    app.selectbox(key="history_min_edge_tabpfn").select(6.0).run()
    assert not app.exception
    assert all(
        metric.value == "—" for metric in app.metric if metric.label != "Final games"
    )
    assert reader.call_count == 1


def test_range_target_and_season_type_load_separate_cache_entries(history_setup):
    reader, _ = history_setup
    app = open_history()
    changed_range = (date(2026, 10, 20), date(2026, 10, 26))
    app.date_input(key="history_dates").set_value(changed_range).run()
    assert not app.exception
    assert reader.call_args.args[:2] == changed_range
    assert reader.call_count == 2
    app.session_state["history_target"] = 3
    app.run()
    assert not app.exception
    assert reader.call_args.args[2] == 3
    assert reader.call_count == 3
    app.selectbox(key="history_season_type").select(4).run()
    assert not app.exception
    assert reader.call_args.args[3] == 4
    assert reader.call_count == 4
    app.button(key="history_refresh").click().run()
    assert not app.exception
    assert reader.call_count == 5
    assert reader.call_args.args == (*changed_range, 3, 4)
    # Refreshing one selection does not evict the original cached range.
    app.date_input(key="history_dates").set_value(
        (date(2026, 10, 28), date(2026, 10, 28))
    )
    app.session_state["history_target"] = 1
    app.session_state["history_season_type"] = 2
    app.run()
    assert not app.exception
    assert reader.call_count == 5


@pytest.mark.parametrize("target", [1, 2, 3])
def test_backfill_only_closing_outputs_are_visible_without_entering_live_metrics(
    history_setup, target
):
    reader, _ = history_setup
    data = reader.side_effect(date(2026, 10, 1), NOW.date(), target, 2)
    rows = data.predictions.loc[
        data.predictions["source"].eq("Snapshot replay")
        & data.predictions["dataset_type"].eq(1)
        & data.predictions["bucket"].eq(360)
    ].copy()
    reader.side_effect = None
    reader.return_value = AnalysisData(data.games, rows, data.fetched_at)
    app = open_history(history_target=target)
    assert not app.exception
    metrics = {node.label: node.value for node in app.metric}
    assert metrics["Closing · Replay"] != "—"
    assert metrics["Closing · Final"] == metrics["Own snapshot · Live"] == "—"
    assert len(rendered_charts(app)) == 1
    table = app.dataframe[0].value
    assert table["Model horizon (min)"].eq(0).all()
    assert table["Source"].eq("Snapshot replay").all()
    assert table["Input bucket (min; 0 = Final)"].eq(360).all()
    assert table["Final games with outputs"].eq(2).all()


def test_partial_range_skips_read_and_does_not_leave_old_charts(history_setup):
    reader, _ = history_setup
    app = open_history()
    app.date_input(key="history_dates").set_value((date(2026, 10, 20),)).run()
    assert not app.exception
    assert reader.call_count == 1
    assert "Choose a start and end date" in app.info[0].value
    assert not app.metric
    assert not rendered_charts(app)


def test_empty_games_empty_outputs_and_safe_errors(history_setup):
    reader, _ = history_setup
    app = open_history()
    games = (
        reader.return_value.games
        if isinstance(reader.return_value, AnalysisData)
        else historical_slate().games
    )
    reader.side_effect = None
    reader.return_value = AnalysisData(games, pd.DataFrame(), NOW.to_pydatetime())
    app.button(key="history_refresh").click().run()
    assert not app.exception
    assert "No stored predictions" in app.info[0].value
    assert app.metric[0].value == "2/2"
    assert not rendered_charts(app)
    reader.return_value = AnalysisData(
        pd.DataFrame(), pd.DataFrame(), NOW.to_pydatetime()
    )
    app.button(key="history_refresh").click().run()
    assert not app.exception
    assert "No games were recorded" in app.info[0].value
    assert not app.metric
    reader.side_effect = DashboardDataError(
        "Could not read historical predictions from Aiven."
    )
    app.button(key="history_refresh").click().run()
    assert not app.exception
    assert app.error[0].value == "Could not read historical predictions from Aiven."
    assert not app.info


def season_data(target, games=40):
    """Enough settled replay games for a best slot, its combinations and random."""
    rows, baselines, slate = [], [], []
    for game in range(1, games + 1):
        # Odd games finish 235 (OVER 230.5 wins), even games 215 (UNDER wins).
        points = 121 if game % 2 else 101
        day = date(2026, 10, 20) + pd.Timedelta(days=game % 4)
        slate.append(
            dict(
                game_id=game,
                game_date=day,
                status=1,
                home_points=points,
                away_points=114,
            )
        )
        common = dict(game_id=game, target=target, home_points=points)
        # T−6h follows the result except every tenth game: 36 of 40.
        right = (game % 2 == 1) != (game % 10 == 0)
        for horizon, value in ((480, 240), (360, 240 if right else 220), (120, 220)):
            rows.append(
                analysis_row(
                    horizon=horizon,
                    bucket=horizon,
                    source="Snapshot replay",
                    prediction_value=value if target == 2 else value - 229,
                    **common,
                )
            )
        for kind in (1, 3):
            rows.append(
                analysis_row(
                    horizon=360,
                    bucket=240,
                    kind=kind,
                    prediction_value=240 if target == 2 else 11,
                    **common,
                )
            )
        # Members to combine at T−8h and T−6h, always OVER: 20 of 40 each.
        for kind, horizon, bucket in ((3, 480, 480), (3, 360, 360), (1, 0, 360)):
            rows.append(
                analysis_row(
                    horizon=horizon,
                    bucket=bucket,
                    kind=kind,
                    source="Snapshot replay",
                    prediction_value=240 if target == 2 else 11,
                    **common,
                )
            )
        if target != 3:
            baselines.append(
                analysis_row(
                    game_id=game,
                    horizon=0,
                    bucket=0,
                    kind=2,
                    target=1,
                    home_points=points,
                    source="Random baseline",
                    variant="random_line_error_baseline",
                    prediction_value=1.0 if game % 5 else -1.0,
                )
            )
    return AnalysisData(
        pd.DataFrame(slate),
        pd.DataFrame(rows),
        NOW.to_pydatetime(),
        pd.DataFrame(baselines),
    )


@pytest.mark.parametrize("target", [1, 2, 3])
def test_best_slot_combinations_random_and_six_hour_windows(history_setup, target):
    reader, _ = history_setup
    reader.side_effect = lambda start, end, target, season_type, **kwargs: (
        season_data(target)
    )
    app = open_history(history_target=target)
    assert not app.exception
    metrics = {node.label: node.value for node in app.metric}
    # T−2h always says UNDER: 20 of 40 games.
    assert metrics["Replay · ≥ 6h"] != metrics["Replay · < 6h"]
    assert metrics["Replay · < 6h"] == "50.0%"
    assert ("Random baseline" in metrics) is (target != 3)
    # Three slot cards, the best first: T−6h follows 36 of 40 results.
    slots = [
        node.value
        for node in app.markdown
        if node.value.startswith("**") and " · " in node.value
    ]
    assert slots[0] == "**XGBoost · 2.5 · Intermediate · T−6h**"
    assert len(slots) == 3
    assert [node.value for node in app.caption if node.value.endswith(" slot")] == [
        "Best slot",
        "2nd best slot",
        "3rd best slot",
    ]
    # Slots and combinations each report the chance of their leader.
    chances = [node.value for node in app.metric if node.label == "Coin-flip chance"]
    assert len(chances) == 2 and all(value.endswith("%") for value in chances)
    permutation = [
        node.value for node in app.metric if node.label == "Game-permutation p"
    ]
    # T−6h wins 36 of 40: not luck, even as the best of several slots. For
    # totals the T−6h mean follows it; for spreads every combination's edge
    # leans HOME (+11 or −9 with +11), so the leader is a 50% coin flip.
    assert permutation[0] == "< 0.001"
    assert (permutation[1] == "< 0.001") is (target != 3)
    if target == 3:
        assert float(permutation[1]) > 0.2
    profits = [node.value for node in app.metric if node.label == "P(profitable)"]
    # Three slot cards and the leading combination.
    assert len(profits) == 4 and all(value.endswith("%") for value in profits)
    assert profits[1] == "29%"  # The 20-20 2nd slot.
    data = next(t.value for t in app.dataframe if "P(profitable) %" in t.value)
    assert data["P(profitable) %"].between(0, 100).all()
    subheaders = [node.value for node in app.subheader]
    assert subheaders.index("Best single slots") == 0
    assert "Combinations · same timepoint" in [node.value for node in app.subheader]
    # The T−6h mean follows its strongest member: it ties, never beats it.
    assert any(
        "No combination beats the best single slot" in node.value for node in app.info
    )
    # Efficacy, combinations, closing replay, intermediate replay, live
    # heatmap, own snapshot.
    assert len(rendered_charts(app)) == 6
    combinations = next(t.value for t in app.dataframe if "Strategy" in t.value)
    strategies = set(combinations["Strategy"])
    assert "XGBoost · 2.5 · Intermediate · T−6h (slot)" in strategies
    assert (
        "T−6h · mean edge · TabPFN 2.5 intermediate + XGBoost 2.5 intermediate"
        in strategies
    )
    assert not any("T−2h ·" in name for name in strategies)  # Alone at T−2h.
    assert ("Random baseline" in strategies) is (target != 3)
    tried = next(t.value for t in app.dataframe if "Method" in t.value)
    assert set(tried["Method"]) == {"Mean edge", "Majority vote"}
    assert set(tried["Timepoint (min; 0 = Final)"]) == {360, 480}
    # TabPFN live cells stay out of the model × time view and its margins.
    live_tables = [
        t.value
        for t in app.dataframe
        if "Input bucket (min; 0 = Final)" in t.value
        and t.value.get("Source", pd.Series(dtype=str)).eq("Live predictions").any()
    ]
    assert live_tables
    assert not any(t["Model"].str.startswith("TabPFN").any() for t in live_tables)
    # One read of the selected target; cross-target combinations stay off.
    assert [call.args[2] for call in reader.call_args_list] == [target]


def test_cross_target_combinations_read_the_other_totals_target(history_setup):
    reader, _ = history_setup
    reader.side_effect = lambda start, end, target, season_type, **kwargs: (
        season_data(target)
    )
    app = open_history()
    app.toggle(key="history_cross_target").set_value(True).run()
    assert not app.exception
    assert [call.args[2] for call in reader.call_args_list] == [1, 2]
    tried = next(t.value for t in app.dataframe if "Method" in t.value)
    assert any(
        "XGBoost 2.5 intermediate (line error)" in names
        and "XGBoost 2.5 intermediate (total points)" in names
        for names in tried["Slots"]
    )
