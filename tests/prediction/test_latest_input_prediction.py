"""Intermediate slots predicting from tonight's latest snapshot (LATEST_INPUT_SLOTS)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nba_ou.config.settings import Settings
from nba_ou.create_training_data.create_intermediate_line_df import (
    _keep_extra_snapshots,
    nearest_grid_horizon,
)
from nba_ou.create_training_data.intermediate_injuries import _alias_horizons
from nba_ou.data_processing.injury_status.report_state import InjuryReportState
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.prediction import prediction
from nba_ou.prediction.input_limits import (
    LATEST_INPUT_COLUMN,
    LIMITED_COUNT_COLUMN,
    limit_to_training_support,
    unsupported_news_columns,
)
from nba_ou.prediction.intermediate_frame import latest_snapshot_minutes
from nba_ou.prediction.report_requirement import (
    COVERED_COLUMNS,
    NO_REPORT_ATTR,
    reports_filed,
    requires_report,
)

import scripts.predict_nba_games as job

#: 23:00 UTC is 19:00 ET (EDT) on this date.
TIP = pd.Timestamp("2026-04-10 23:00", tz="UTC")


# ---- the latest snapshot ---------------------------------------------------


def test_latest_snapshot_is_rounded_up_and_a_tipped_game_reads_zero():
    tipoffs = pd.Series(
        {"early": TIP, "late": TIP + pd.Timedelta(hours=3), "none": pd.NaT}
    )
    as_of = TIP - pd.Timedelta(minutes=46, seconds=30)
    assert latest_snapshot_minutes(tipoffs, as_of) == {"early": 47, "late": 227}
    # Never after as_of: the snapshot falls at or before the run.
    assert TIP - pd.Timedelta(minutes=47) <= as_of
    assert latest_snapshot_minutes(tipoffs, TIP + pd.Timedelta(minutes=5)) == {
        "early": 0,
        "late": 175,
    }


def test_nearest_grid_horizon():
    grid = (0, 30, 60, 120, 1080)
    assert nearest_grid_horizon(47, grid) == 60
    assert nearest_grid_horizon(44, grid) == 30
    # A tie goes to the earlier snapshot (more minutes before tip).
    assert nearest_grid_horizon(45, grid) == 60
    assert nearest_grid_horizon(1500, grid) == 1080


def test_only_each_games_own_extra_snapshot_is_kept():
    wide = pd.DataFrame(
        [(g, m) for g in ("a", "b") for m in (30, 47, 60, 227)],
        columns=["game_id", "snapshot_minutes"],
    )
    kept = _keep_extra_snapshots(wide, (30, 60), {"a": 47, "b": 227})
    assert sorted(map(tuple, kept.to_numpy().tolist())) == [
        ("a", 30),
        ("a", 47),
        ("a", 60),
        ("b", 30),
        ("b", 60),
        ("b", 227),
    ]


# ---- the injury stage's aliases ----------------------------------------------


def _state(rows):
    """A report state with a status and a filing per (game, team)."""
    statuses = pd.DataFrame(
        [(g, t, f"p{g}", "Out") for g, t in rows],
        columns=["game_id", "team_id", "player_id", "status"],
    )
    filings = pd.DataFrame(
        [(g, t, True) for g, t in rows], columns=["game_id", "team_id", "submitted"]
    )
    ages = pd.DataFrame(
        [(g, 10.0) for g, _ in rows], columns=["game_id", "report_age_minutes"]
    )
    empty = InjuryReportState.empty()
    return InjuryReportState(
        statuses=statuses,
        filings=filings,
        report_age=ages,
        status_events=empty.status_events,
        listed_pairs=empty.listed_pairs,
    )


def test_an_alias_reads_the_earlier_games_of_its_grid_horizon():
    states = {
        60: _state([("old1", "t1"), ("old2", "t2"), ("tonight", "t3")]),
        47: _state([("tonight", "t3")]),
    }
    cutoffs = pd.DataFrame(
        {
            "game_id": ["old1", "old2", "tonight", "tonight"],
            "snapshot_minutes": [60, 60, 60, 47],
            "as_of": pd.to_datetime(["2026-01-01"] * 4, utc=True),
        }
    )
    aliased, aliased_cutoffs = _alias_horizons(
        states, cutoffs, {47: 60}, targets={"tonight"}
    )
    # Earlier games from T-60, tonight's own read at T-47 -- never its T-60 one.
    assert sorted(aliased[47].statuses["game_id"]) == ["old1", "old2", "tonight"]
    assert aliased[47].statuses["game_id"].tolist().count("tonight") == 1
    assert len(aliased[47].filings) == 3
    assert aliased[60] is states[60]
    at_47 = aliased_cutoffs[aliased_cutoffs["snapshot_minutes"].eq(47)]
    assert sorted(at_47["game_id"]) == ["old1", "old2", "tonight"]
    # The caller's frames are untouched (the memos key on them).
    assert len(cutoffs) == 4 and len(states[47].statuses) == 1


# ---- the training-support limits ---------------------------------------------


def _rows(minutes=47, **values):
    frame = pd.DataFrame(
        {
            "GAME_ID": ["g1"],
            "TIME_TO_MATCH_MIN": [minutes],
            "SNAPSHOT_TS_UTC": [TIP - pd.Timedelta(minutes=minutes)],
            "REF_CREW_FTA_TENDENCY_BEFORE": [1.5],
            "INJ_SNAP_HAS_MATERIAL_NEWS_BEFORE_TEAM_HOME": [1],
            "INJ_SNAP_NEWS_EXP_PTS_W60_BEFORE_TEAM_HOME": [np.nan],
            "x": [3.0],
        }
    )
    return frame.assign(**values)


TABLE = {
    "INJ_SNAP_HAS_MATERIAL_NEWS_BEFORE_TEAM_HOME": (480, 720),
    "INJ_SNAP_NEWS_EXP_PTS_W60_BEFORE_TEAM_HOME": (720,),
}


def test_referees_follow_the_models_nominal_snapshot():
    # 19:00 ET tip. T-360 is 13:00 ET, after the 09:00 release: kept.
    kept = limit_to_training_support(_rows(), horizon_minutes=360, unsupported={})
    assert kept["REF_CREW_FTA_TENDENCY_BEFORE"].tolist() == [1.5]
    assert kept[LIMITED_COUNT_COLUMN].tolist() == [0]
    # T-720 is 07:00 ET: its training rows never had a crew.
    masked = limit_to_training_support(_rows(), horizon_minutes=720, unsupported={})
    assert masked["REF_CREW_FTA_TENDENCY_BEFORE"].isna().all()
    assert masked[LIMITED_COUNT_COLUMN].tolist() == [1]
    # A 22:00 ET tip at T-720 is 10:00 ET: released.
    late = _rows(
        SNAPSHOT_TS_UTC=[TIP + pd.Timedelta(hours=3) - pd.Timedelta(minutes=47)]
    )
    late = limit_to_training_support(late, horizon_minutes=720, unsupported={})
    assert late["REF_CREW_FTA_TENDENCY_BEFORE"].tolist() == [1.5]


def test_news_is_zeroed_only_where_training_had_none():
    limited = limit_to_training_support(_rows(), horizon_minutes=480, unsupported=TABLE)
    assert limited["INJ_SNAP_HAS_MATERIAL_NEWS_BEFORE_TEAM_HOME"].tolist() == [0]
    # NaN stays NaN, as in training; other features are untouched.
    assert limited["INJ_SNAP_NEWS_EXP_PTS_W60_BEFORE_TEAM_HOME"].isna().all()
    assert limited["x"].tolist() == [3.0]
    assert limited[LIMITED_COUNT_COLUMN].tolist() == [1]
    supported = limit_to_training_support(
        _rows(), horizon_minutes=360, unsupported=TABLE
    )
    assert supported["INJ_SNAP_HAS_MATERIAL_NEWS_BEFORE_TEAM_HOME"].tolist() == [1]


def test_support_table_from_a_dataset():
    frame = pd.DataFrame(
        {
            "TIME_TO_MATCH_MIN": [60] * 20 + [720] * 20,
            # A value in 10/20 rows at T-60, 0/20 at T-720.
            "INJ_SNAP_NEWS_EXP_PTS_W60_BEFORE_TEAM_HOME": [1.0] * 10
            + [0.0] * 10
            + [np.nan] * 5
            + [0.0] * 15,
            # Supported everywhere: left out.
            "ODDS_SNAP_NEWS_HAS_RECENT": [1] * 40,
            "REF_X": [0] * 40,
        }
    )
    assert unsupported_news_columns(frame) == {
        "INJ_SNAP_NEWS_EXP_PTS_W60_BEFORE_TEAM_HOME": (720,)
    }


# ---- serving -----------------------------------------------------------------


def _serve(monkeypatch, horizon):
    import nba_ou.modeling.registry_store as store

    class Resolved:
        spec = SimpleNamespace(
            spec_id="s",
            identity=SimpleNamespace(
                dataset_type="intermediate_line",
                horizon_minutes=horizon,
                target="line_error",
                schema_version="2_5",
                variant="main",
            ),
            features=SimpleNamespace(feature_names=["x"]),
            cleaning=SimpleNamespace(max_na_per_row=800),
            provenance=SimpleNamespace(training_version="v"),
        )
        fit = SimpleNamespace(
            fit_id="f",
            model_name="m",
            train_date_min=pd.Timestamp("2025-01-01"),
            train_date_max=pd.Timestamp("2026-01-01"),
        )
        model_bytes = b""

    monkeypatch.setattr(store, "resolve_channel", lambda **_: Resolved())
    monkeypatch.setattr(prediction.XGBRegressor, "load_model", lambda self, raw: None)
    seen = {}

    def predict(*, df, **kwargs):
        seen.update(kwargs, rows=df)
        return df

    monkeypatch.setattr(prediction, "load_and_predict_model_for_nba_games", predict)
    return seen


def _frame():
    """Two games' grid rows plus each one's latest snapshot."""
    rows = [("g1", m) for m in (47, 60, 720)] + [("g2", m) for m in (60, 227, 720)]
    frame = pd.DataFrame(rows, columns=["GAME_ID", "TIME_TO_MATCH_MIN"])
    frame["SNAPSHOT_TS_UTC"] = TIP - pd.to_timedelta(
        frame["TIME_TO_MATCH_MIN"], unit="m"
    )
    frame["REF_CREW_FTA_TENDENCY_BEFORE"] = 1.5
    frame["x"] = 1.0
    frame[LATEST_INPUT_COLUMN] = frame["TIME_TO_MATCH_MIN"].isin([47, 227])
    return frame


def _predict(slot_horizon, **kwargs):
    return prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=ModelSlot(
            schema_version="2_5",
            target="line_error",
            horizon_minutes=slot_horizon,
            variant="main",
        ),
        df=None,
        **kwargs,
    )


def test_a_latest_input_slot_reads_each_games_latest_row_limited(monkeypatch):
    seen = _serve(monkeypatch, horizon=720)
    _predict(720, frames={"intermediate_line": _frame()}, latest_input=True)
    rows = seen["rows"]
    assert rows[["GAME_ID", "TIME_TO_MATCH_MIN"]].values.tolist() == [
        ["g1", 47],
        ["g2", 227],
    ]
    # A T-720 model of a 19:00 ET game never saw a crew.
    assert rows["REF_CREW_FTA_TENDENCY_BEFORE"].isna().all()
    assert seen["extra_summary_columns"]["INPUT_MODE"] == "latest"


def test_own_input_reads_the_models_snapshot(monkeypatch):
    seen = _serve(monkeypatch, horizon=720)
    _predict(720, frames={"intermediate_line": _frame()})
    rows = seen["rows"]
    assert rows["TIME_TO_MATCH_MIN"].tolist() == [720, 720]
    assert rows["REF_CREW_FTA_TENDENCY_BEFORE"].tolist() == [1.5, 1.5]
    assert seen["extra_summary_columns"]["INPUT_MODE"] == "own"


def test_latest_input_needs_a_frame_built_for_it(monkeypatch):
    _serve(monkeypatch, horizon=60)
    frame = _frame().drop(columns=LATEST_INPUT_COLUMN)
    with pytest.raises(ValueError, match=LATEST_INPUT_COLUMN):
        _predict(60, frames={"intermediate_line": frame}, latest_input=True)


# ---- the injury-report requirement ---------------------------------------------


def test_only_horizons_up_to_the_limit_require_a_report():
    assert requires_report(300, 300) and requires_report(30, 300)
    assert requires_report(0, 300)
    assert not requires_report(360, 300)
    # A pooled model trained on filed and unfiled rows alike; no limit, no rule.
    assert not requires_report(None, 300)
    assert not requires_report(30, None)


def test_a_report_counts_only_when_both_teams_filed():
    rows = pd.DataFrame(
        {COVERED_COLUMNS[0]: [1, 1, 0, np.nan], COVERED_COLUMNS[1]: [1, 0, 0, 1]}
    )
    assert reports_filed(rows).tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="coverage"):
        reports_filed(rows.drop(columns=COVERED_COLUMNS[1]))


def _filed_frame():
    """``_frame`` with g1 still unfiled by its away team."""
    frame = _frame()
    frame[COVERED_COLUMNS[0]] = 1
    frame[COVERED_COLUMNS[1]] = frame["GAME_ID"].ne("g1").astype(int)
    return frame


def test_a_short_horizon_slot_skips_a_game_without_both_reports(monkeypatch):
    seen = _serve(monkeypatch, horizon=60)
    out = _predict(
        60,
        frames={"intermediate_line": _filed_frame()},
        latest_input=True,
        report_required_max_horizon=300,
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["g2"]
    assert out.attrs[NO_REPORT_ATTR] == ["g1"]


def test_a_long_horizon_slot_still_predicts_it(monkeypatch):
    seen = _serve(monkeypatch, horizon=720)
    out = _predict(
        720,
        frames={"intermediate_line": _filed_frame()},
        latest_input=True,
        report_required_max_horizon=300,
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["g1", "g2"]
    assert out.attrs[NO_REPORT_ATTR] == []


def test_a_slot_with_no_filed_game_predicts_nothing(monkeypatch):
    seen = _serve(monkeypatch, horizon=60)
    frame = _filed_frame()
    frame[COVERED_COLUMNS[1]] = 0
    out = _predict(
        60, frames={"intermediate_line": frame}, report_required_max_horizon=300
    )
    assert out.empty and "rows" not in seen
    # Own input: each game's own T-60 row.
    assert out.attrs[NO_REPORT_ATTR] == ["g1", "g2"]


def test_without_a_limit_no_game_is_skipped(monkeypatch):
    seen = _serve(monkeypatch, horizon=60)
    frame = _frame()  # no coverage columns at all: never read
    _predict(60, frames={"intermediate_line": frame}, latest_input=True)
    assert seen["rows"]["GAME_ID"].tolist() == ["g1", "g2"]


# ---- configuration -------------------------------------------------------------


def _settings(tmp_path, body):
    path = tmp_path / "config.ini"
    path.write_text(
        "[PredictionModels]\n"
        "ENABLED_MODELS =\n"
        "    2_5 | line_error | 0 | main\n"
        "    2_5 | line_error | 60 | main\n"
        "    2_5 | line_error | 360 | main\n" + body
    )
    return Settings(config_path=str(path), secrets_path=str(tmp_path / "none.ini"))


def test_input_mode_defaults_to_own_and_is_validated(tmp_path):
    assert _settings(tmp_path, "").intermediate_input_mode == "own"
    assert (
        _settings(
            tmp_path, "INTERMEDIATE_INPUT_MODE = Latest\n"
        ).intermediate_input_mode
        == "latest"
    )
    with pytest.raises(ValueError, match="INTERMEDIATE_INPUT_MODE"):
        _settings(tmp_path, "INTERMEDIATE_INPUT_MODE = soon\n").intermediate_input_mode


def test_report_required_max_horizon_is_minutes_or_empty(tmp_path):
    assert _settings(tmp_path, "").report_required_max_horizon is None
    assert (
        _settings(
            tmp_path, "REPORT_REQUIRED_MAX_HORIZON =\n"
        ).report_required_max_horizon
        is None
    )
    assert (
        _settings(
            tmp_path, "REPORT_REQUIRED_MAX_HORIZON = 300\n"
        ).report_required_max_horizon
        == 300
    )
    for bad in ("5h", "-30"):
        with pytest.raises(ValueError, match="REPORT_REQUIRED_MAX_HORIZON"):
            _settings(
                tmp_path, f"REPORT_REQUIRED_MAX_HORIZON = {bad}\n"
            ).report_required_max_horizon


def test_latest_input_slots_all_or_rows(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path, "INTERMEDIATE_INPUT_MODE = latest\nLATEST_INPUT_SLOTS = all\n"
    )
    assert settings.latest_input_slots is None
    monkeypatch.setattr(job, "SETTINGS", settings)
    enabled = settings.prediction_model_slots
    assert job.latest_input_slots(enabled) == enabled
    assert job.latest_input_slots(enabled, "own") == []

    settings = _settings(
        tmp_path,
        "INTERMEDIATE_INPUT_MODE = latest\nLATEST_INPUT_SLOTS =\n"
        "    2_5 | line_error | 360\n",
    )
    monkeypatch.setattr(job, "SETTINGS", settings)
    chosen = job.latest_input_slots(settings.prediction_model_slots)
    assert [slot.horizon_minutes for slot in chosen] == [360]


# ---- building the latest snapshots alone -------------------------------------


def _tonight(monkeypatch, *, as_of):
    """``tonight_intermediate_frame`` with the live day and the builder faked;
    returns it and the builder's recorded keyword arguments."""
    from nba_ou.prediction import intermediate_frame as module

    games = pd.DataFrame(
        {"game_id": ["g1", "g2"], "tipoff_utc": [TIP, TIP + pd.Timedelta(hours=3)]}
    )
    live = SimpleNamespace(
        line_history=SimpleNamespace(games=games), game_ids=["g1", "g2"], as_of=as_of
    )
    monkeypatch.setattr(module, "fetch_live_game_day", lambda *a, **k: live)
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        wanted = kwargs.get("target_snapshots") or {}
        extra = kwargs.get("extra_target_snapshots") or {}
        rows = (
            [(g, m) for g, m in wanted.items()]
            if wanted
            else [(g, m) for g in ("g1", "g2") for m in (30, 240)] + list(extra.items())
        )
        frame = pd.DataFrame(rows, columns=["GAME_ID", "TIME_TO_MATCH_MIN"])
        tips = frame["GAME_ID"].map({"g1": TIP, "g2": TIP + pd.Timedelta(hours=3)})
        scoring = frame.assign(
            SNAPSHOT_TS_UTC=tips - pd.to_timedelta(frame["TIME_TO_MATCH_MIN"], unit="m")
        )
        return frame, scoring

    monkeypatch.setattr(module, "create_intermediate_line_df", builder)
    scheduled = {
        "scheduled_games": pd.DataFrame({"GAME_ID": ["g1", "g2"], "GAME_TIME": [1, 2]})
    }
    return module, scheduled, calls


def test_day_page_openers_are_forwarded_even_when_live_history_is_reused(monkeypatch):
    as_of = TIP - pd.Timedelta(minutes=243)
    module, scheduled, calls = _tonight(monkeypatch, as_of=as_of)
    day_odds = pd.DataFrame(
        {"game_id": ["g1"], "total_consensus_opener_line_over": [229.5]}
    )
    scheduled["df_odds_sportsbook_scheduled"] = day_odds
    live = module.fetch_live_game_day()
    module.tonight_intermediate_frame(scheduled, game_date="2026-04-10", live=live)
    assert calls[-1]["scheduled_odds_sportsbook"] is day_odds
    assert calls[-1]["live"] is live


@pytest.mark.parametrize("unavailable", [False, True])
def test_covers_current_prices_prefer_the_main_sbr_opener(monkeypatch, unavailable):
    from nba_ou.data_processing.odds.scheduled_opener import OPENER_COLUMNS
    from nba_ou.prediction.line_source import LineSource

    as_of = TIP - pd.Timedelta(minutes=243)
    module, scheduled, calls = _tonight(monkeypatch, as_of=as_of)
    source = LineSource(pd.Timestamp("2026-04-10").date(), source="covers")
    scheduled["line_source"] = source
    fallback = pd.DataFrame(
        {"game_id": ["g1", "g2"], **{c: [218.5, 230.5] for c in OPENER_COLUMNS}}
    )
    main = pd.DataFrame(
        {
            "game_id": ["g1"],
            **{c: [217.5 if "line" in c else -110] for c in OPENER_COLUMNS},
        }
    )
    scheduled["df_odds_sportsbook_scheduled"] = fallback

    def read(date, games):
        assert date == "2026-04-10" and games is scheduled["scheduled_games"]
        if unavailable:
            raise OSError("SBR unavailable")
        return main

    monkeypatch.setattr(module, "get_sportsbook_opener_data", read)
    module.tonight_intermediate_frame(scheduled, game_date="2026-04-10")
    actual = calls[-1]["scheduled_odds_sportsbook"].set_index("game_id")
    assert actual.loc["g1", OPENER_COLUMNS[0]] == (218.5 if unavailable else 217.5)
    assert actual.loc["g2", OPENER_COLUMNS[0]] == 230.5
    pd.testing.assert_frame_equal(scheduled["df_odds_sportsbook_scheduled"], fallback)


def test_latest_only_builds_each_games_latest_snapshot_alone(monkeypatch):
    as_of = TIP - pd.Timedelta(minutes=243)
    module, scheduled, calls = _tonight(monkeypatch, as_of=as_of)
    frame = module.tonight_intermediate_frame(
        scheduled, game_date="2026-04-10", latest_input=True, latest_only=True
    )
    assert calls[-1]["target_snapshots"] == {"g1": 243, "g2": 423}
    assert "extra_target_snapshots" not in calls[-1]
    assert frame[LATEST_INPUT_COLUMN].all() and len(frame) == 2

    full = module.tonight_intermediate_frame(
        scheduled, game_date="2026-04-10", latest_input=True
    )
    assert calls[-1]["extra_target_snapshots"] == {"g1": 243, "g2": 423}
    assert "target_snapshots" not in calls[-1]
    latest = full[full[LATEST_INPUT_COLUMN]].reset_index(drop=True)
    pd.testing.assert_frame_equal(latest, frame.reset_index(drop=True))


def test_latest_only_needs_latest_input(monkeypatch):
    module, scheduled, _ = _tonight(monkeypatch, as_of=TIP)
    with pytest.raises(ValueError, match="latest_only needs latest_input"):
        module.tonight_intermediate_frame(
            scheduled, game_date="2026-04-10", latest_only=True
        )


def test_a_horizon_builds_every_games_row_of_that_horizon_alone(monkeypatch):
    # g1's T-240 has passed, g2's is still ahead: both get their T-240 row,
    # and the full frame holds the same rows.
    as_of = TIP - pd.Timedelta(minutes=200)
    module, scheduled, calls = _tonight(monkeypatch, as_of=as_of)
    frame = module.tonight_intermediate_frame(
        scheduled, game_date="2026-04-10", as_of=as_of, horizon=240
    )
    assert calls[-1]["target_snapshots"] == {"g1": 240, "g2": 240}
    assert frame["TIME_TO_MATCH_MIN"].tolist() == [240, 240]
    assert frame.set_index("GAME_ID")["SNAPSHOT_PASSED"].to_dict() == {
        "g1": True,
        "g2": False,
    }

    full = module.tonight_intermediate_frame(
        scheduled, game_date="2026-04-10", as_of=as_of
    )
    assert "target_snapshots" not in calls[-1]
    pd.testing.assert_frame_equal(
        full[full["TIME_TO_MATCH_MIN"].eq(240)].reset_index(drop=True),
        frame.reset_index(drop=True),
    )


def test_a_horizon_excludes_the_latest_input(monkeypatch):
    module, scheduled, calls = _tonight(monkeypatch, as_of=TIP)
    with pytest.raises(ValueError, match="exclusive"):
        module.tonight_intermediate_frame(
            scheduled, game_date="2026-04-10", latest_input=True, horizon=240
        )
    assert calls == []


def test_target_snapshots_are_validated_before_any_io():
    from nba_ou.create_training_data.create_intermediate_line_df import (
        create_intermediate_line_df,
    )

    with pytest.raises(ValueError, match="needs target_game_ids"):
        create_intermediate_line_df(target_snapshots={"g1": 30})
    with pytest.raises(ValueError, match="non-target games"):
        create_intermediate_line_df(target_game_ids=["g1"], target_snapshots={"g2": 30})
    with pytest.raises(ValueError, match="minutes before tip"):
        create_intermediate_line_df(target_game_ids=["g1"], target_snapshots={"g1": -5})
    with pytest.raises(ValueError, match="disagree"):
        create_intermediate_line_df(
            target_game_ids=["g1"],
            target_snapshots={"g1": 47},
            extra_target_snapshots={"g1": 48},
        )
