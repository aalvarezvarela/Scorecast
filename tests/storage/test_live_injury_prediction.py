"""Today's job consumes one synchronized Aiven state for every injury family."""

from contextlib import nullcontext
from importlib import import_module

import pandas as pd
import pytest
from nba_ou.data_processing.injury_status import report_state as states
from nba_ou.postgre_db.injury_report_aiven import fetch

scheduled_module = import_module(
    "nba_ou.create_training_data.get_all_info_for_scheduled_games"
)
AS_OF = pd.Timestamp("2026-10-16T18:00:00Z")
GID = "0022600001"
HOME, AWAY = "1610612738", "1610612752"


def report_state(status="questionable", filed=True):
    state = states.InjuryReportState.empty()
    return states.InjuryReportState(
        statuses=pd.DataFrame(
            {
                "game_id": [GID],
                "team_id": [HOME],
                "player_id": ["2544"],
                "status": [status],
                "reason_category": ["injury_illness"],
                "reason_detail": ["Knee"],
                "game_date": ["2026-10-16"],
                "season_year": [2026],
            }
        ),
        filings=pd.DataFrame(
            {
                "game_id": [GID, GID],
                "team_id": [HOME, AWAY],
                "submitted": [filed, filed],
            }
        ),
        report_age=pd.DataFrame({"game_id": [GID], "report_age_minutes": [15.0]}),
        status_events=state.status_events,
        listed_pairs=state.listed_pairs,
    )


@pytest.fixture
def scheduled(monkeypatch):
    monkeypatch.setattr(
        scheduled_module, "load_live_rosters", lambda games: pd.DataFrame()
    )
    frame = pd.DataFrame(
        {
            "GAME_ID": [GID],
            "GAME_DATE_EST": ["2026-10-16"],
            "HOME_TEAM_ID": [HOME],
            "VISITOR_TEAM_ID": [AWAY],
            "GAMECODE": ["20261016/NYKBOS"],
            "GAME_TIME": [pd.Timestamp("2026-10-16T23:00Z")],
        }
    )
    monkeypatch.setattr(scheduled_module, "get_schedule_games", lambda day: frame)
    monkeypatch.setattr(
        scheduled_module,
        "process_scheduled_referee_assignments",
        lambda games: pd.DataFrame(),
    )
    monkeypatch.setattr(
        scheduled_module, "get_yahoo_prediction_data", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        scheduled_module,
        "get_sportsbook_prediction_data",
        lambda *a, **kw: pd.DataFrame(),
    )
    return frame


def test_schedule_syncs_before_reading_aiven_and_never_saves_locally(
    monkeypatch, scheduled, tmp_path
):
    calls = []
    state = report_state("out")

    def refresh(date, **kwargs):
        calls.append("sync")
        assert kwargs["scheduled_games"] is scheduled

    def load(ids, as_of):
        calls.append("read")
        assert ids == [GID]
        assert as_of.tzinfo is not None
        return state

    monkeypatch.setattr(scheduled_module, "synchronize_injury_reports", refresh)
    monkeypatch.setattr(scheduled_module, "load_prediction_report_state", load)
    destination = tmp_path / "reports"
    result = scheduled_module.get_all_info_for_scheduled_games(
        "2026-10-16", "unused", save_reports_path=destination, headless=True
    )
    assert calls == ["sync", "read"]
    assert result["injury_report_state"] is state
    assert result["injury_dict_scheduled"] == {GID: {HOME: ["2544"], AWAY: []}}
    assert not destination.exists()


def test_scheduled_out_sets_never_include_earlier_games(monkeypatch, scheduled):
    # The prediction state carries every covered game since the reports
    # began. Merged into the realized-absence history, an earlier game's
    # pre-game report would replace who actually missed it.
    state = report_state("out")
    earlier = "0022500999"
    state = states.InjuryReportState(
        statuses=pd.concat(
            [state.statuses, state.statuses.assign(game_id=earlier, player_id="1")],
            ignore_index=True,
        ),
        filings=pd.concat(
            [state.filings, state.filings.assign(game_id=earlier)], ignore_index=True
        ),
        report_age=state.report_age,
        status_events=state.status_events,
        listed_pairs=state.listed_pairs,
    )
    monkeypatch.setattr(
        scheduled_module, "synchronize_injury_reports", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        scheduled_module, "load_prediction_report_state", lambda *a: state
    )
    result = scheduled_module.get_all_info_for_scheduled_games(
        "2026-10-16", "unused", headless=True
    )
    assert result["injury_dict_scheduled"] == {GID: {HOME: ["2544"], AWAY: []}}
    assert (earlier, HOME) in result["injury_report_state"].covered


@pytest.mark.parametrize("dataset_type", ["closing_line", "intermediate_line"])
def test_rosters_are_loaded_before_injury_resolution(
    monkeypatch, scheduled, dataset_type
):
    calls = []
    rosters = pd.DataFrame({"PLAYER_ID": ["rookie"]})

    def roster_fetch(games):
        calls.append("rosters")
        return rosters

    def sync(day, **kwargs):
        calls.append("sync")
        assert kwargs["roster_snapshots"] is rosters

    def read(*args):
        calls.append("read")
        return report_state()

    monkeypatch.setattr(scheduled_module, "load_live_rosters", roster_fetch)
    monkeypatch.setattr(scheduled_module, "synchronize_injury_reports", sync)
    monkeypatch.setattr(scheduled_module, "load_prediction_report_state", read)
    result = scheduled_module.get_all_info_for_scheduled_games(
        "2026-10-16",
        "unused",
        headless=True,
        dataset_type=dataset_type,
    )
    assert calls == ["rosters", "sync", "read"]
    assert result["roster_snapshots"] is rosters


def test_refresh_failure_uses_previous_aiven_state(monkeypatch, scheduled, caplog):
    def fail(*a, **kw):
        raise OSError("NBA unavailable")

    monkeypatch.setattr(scheduled_module, "synchronize_injury_reports", fail)
    monkeypatch.setattr(
        scheduled_module, "load_prediction_report_state", lambda *a: report_state()
    )
    result = scheduled_module.get_all_info_for_scheduled_games(
        "2026-10-16", "unused", headless=True
    )
    assert not result["games_not_updated"]
    assert "reading existing Aiven state" in caplog.text


def test_unknown_filing_does_not_imply_a_healthy_roster(monkeypatch, scheduled):
    """No team filed yet: every game is left out of the closing rows, and the
    fetch itself does not fail (the intermediate slots still predict)."""
    monkeypatch.setattr(
        scheduled_module, "synchronize_injury_reports", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        scheduled_module,
        "load_prediction_report_state",
        lambda *a: report_state(filed=False),
    )
    result = scheduled_module.get_all_info_for_scheduled_games(
        "2026-10-16", "unused", headless=True
    )
    assert result["games_not_updated"] == [GID]
    assert result["games_missing_injury_reports"] == [GID]
    assert result["injury_dict_scheduled"] == {}


@pytest.mark.parametrize("missing", ["not_submitted", "no_report", "one_team"])
def test_intermediate_fetch_keeps_games_without_injury_coverage(
    monkeypatch, scheduled, missing
):
    state = (
        states.InjuryReportState.empty()
        if missing == "no_report"
        else report_state(filed=False)
    )
    if missing == "one_team":
        state.filings.loc[state.filings.team_id.eq(HOME), "submitted"] = True
    monkeypatch.setattr(
        scheduled_module, "synchronize_injury_reports", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        scheduled_module, "load_prediction_report_state", lambda *a: state
    )
    result = scheduled_module.get_all_info_for_scheduled_games(
        "2026-10-16", "unused", headless=True, dataset_type="intermediate_line"
    )
    assert result["scheduled_games"].GAME_ID.tolist() == [GID]
    assert result["games_not_updated"] == []
    assert result["games_missing_injury_reports"] == [GID]
    assert result["injury_report_state"] is state


def test_invalid_dataset_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown prediction dataset type"):
        scheduled_module.get_all_info_for_scheduled_games(
            "2026-10-16", "unused", dataset_type="intermedate"
        )


def test_prediction_state_replaces_final_statuses_with_snapshot_statuses(monkeypatch):
    from nba_ou.postgre_db import row_order
    from nba_ou.postgre_db.config import db_config

    history, current = report_state("out"), report_state("questionable")
    monkeypatch.setattr(states, "load_injury_report_state", lambda: history)
    monkeypatch.setattr(db_config, "connect_nba_db", lambda *a: nullcontext(object()))
    calls = []

    def read(query, conn, params):
        calls.append(query)
        if "LEAST" in query:
            assert pd.Timestamp(params["as_of"]) == AS_OF
            return pd.DataFrame(
                {"game_id": [GID], "snapshot_minutes": [0], "as_of": [AS_OF]}
            )
        return pd.DataFrame({"game_id": [GID], "player_id": ["2544"]})

    def snapshots(conn, cutoffs):
        assert cutoffs.as_of.tolist() == [AS_OF]
        return tuple(
            frame.assign(snapshot_minutes=0)
            for frame in (current.statuses, current.filings, current.report_age)
        )

    monkeypatch.setattr(row_order, "read_sql_canonical", read)
    monkeypatch.setattr(fetch, "report_state_at_snapshots", snapshots)
    result = states.load_prediction_report_state([GID], AS_OF)
    assert result.statuses.status.tolist() == ["questionable"]
    assert result.covered == {(GID, HOME), (GID, AWAY)}
    assert result.listed_pairs.player_id.tolist() == ["2544"]
    assert all("valid_from <" in query for query in calls if "ir_status_span" in query)
