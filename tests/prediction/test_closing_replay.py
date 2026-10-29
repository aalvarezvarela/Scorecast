"""Closing replay must use historical quotes and per-game pregame cutoffs."""

import pandas as pd
import pytest
from nba_ou.data_processing.injury_status.report_state import InjuryReportState
from nba_ou.prediction import closing_replay as replay

DAY = "2026-10-28"
GAME = "0022600001"
TIP = pd.Timestamp("2026-10-28T23:00Z")
HOME, AWAY = "1610612738", "1610612752"


def injury_state(*, filed=(HOME, AWAY), statuses=(), history=False):
    state = InjuryReportState.empty()
    return InjuryReportState(
        statuses=pd.DataFrame(
            statuses,
            columns=["game_id", "team_id", "player_id", "status"],
        ),
        filings=pd.DataFrame(
            [(GAME, team, team in filed) for team in (HOME, AWAY)]
            + ([("0022600000", HOME, True)] if history else []),
            columns=["game_id", "team_id", "submitted"],
        ),
        report_age=state.report_age,
        status_events=state.status_events,
        listed_pairs=state.listed_pairs,
    )


def schedule():
    return pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "GAME_DATE_EST": [DAY],
            "GAME_TIME": [TIP],
            "HOME_TEAM_ID": ["1610612738"],
            "VISITOR_TEAM_ID": ["1610612752"],
        }
    )


def ticks():
    rows = []
    for minute, total, spread in [
        (600, 230.5, -3.5),
        (300, 235.5, -4.5),
        (10, 240.5, -5.5),
        (-1, 999, -99),
    ]:
        for market, left, right in [
            ("totals", total, total),
            ("point_spread", spread, -spread),
            ("money_line", None, None),
        ]:
            rows.append(
                dict(
                    game_id=GAME,
                    market=market,
                    book="bet365",
                    line_ts=TIP - pd.Timedelta(minutes=minute),
                    minutes_before_tip=minute,
                    is_opener=False,
                    left_line=left,
                    right_line=right,
                    left_price=-115,
                    right_price=-105,
                )
            )
    return pd.DataFrame(rows)


def test_day_odds_use_last_known_pregame_quotes_and_keep_home_handicap_orientation():
    for horizon, total, home in [
        (480, 230.5, 3.5),
        (360, 230.5, 3.5),
        (240, 235.5, 4.5),
        (0, 240.5, 5.5),
    ]:
        odds = replay.historical_odds(schedule(), ticks(), horizon, DAY)
        assert odds.game_id.tolist() == [GAME]
        assert odds.total_bet365_line_over.tolist() == [total]
        assert odds.spread_bet365_line_home.tolist() == [home]
        assert odds.ml_bet365_price_home.tolist() == [-105]
        assert not any("score" in column for column in odds)


def test_closing_builder_receives_distinct_historical_cutoffs_and_keeps_raw_quotes(
    monkeypatch,
):
    snapshot = pd.DataFrame(
        {
            "GAME_ID": [GAME] * 5,
            "TIME_TO_MATCH_MIN": [0, 240, 360, 480, 60],
            "SNAPSHOT_TS_UTC": [
                TIP - pd.Timedelta(minutes=h) for h in [0, 240, 360, 480, 60]
            ],
            "SNAPSHOT_PASSED": True,
            "LINE_SOURCE": "covers",
            "ODDS_SNAP_TOT_BET365_RAW_LINE": [240.5, 235.5, 230.5, 230.5, 240.5],
        }
    )
    monkeypatch.setattr(
        replay,
        "fetch_pregame_ticks",
        lambda *a, **kw: pytest.fail("closing must reuse the shared ticks"),
    )
    monkeypatch.setattr(
        replay,
        "stored_crews",
        lambda *a: pytest.fail("closing must reuse the shared crews"),
    )
    report_calls, roster_calls, builder_calls = [], [], []

    def report(cutoffs):
        report_calls.extend(cutoffs.as_of.tolist())
        return injury_state()

    def roster(games, *, as_of):
        roster_calls.append(as_of)
        return pd.DataFrame()

    def builder(*, scheduled_data, **kwargs):
        cutoff = scheduled_data["roster_as_of_utc"][GAME]
        builder_calls.append(cutoff)
        assert scheduled_data["df_odds_yahoo_scheduled"].empty
        assert scheduled_data["injury_report_state"] is not None
        assert kwargs["todays_prediction"] and kwargs["target_game_ids"] == [GAME]
        assert kwargs["recent_limit_to_include"] == "2026-10-27"
        assert scheduled_data["df_odds_sportsbook_scheduled"].game_id.tolist() == [GAME]
        frame = pd.DataFrame(
            {
                "GAME_ID": [GAME],
                "GAME_TIME": [TIP],
                "ODDS_TOTAL_LINE_bet365": [231.0],
                "REF_CREW_FOO_BEFORE": [12.0],
            }
        )
        return frame, frame

    monkeypatch.setattr(replay, "load_prediction_report_state_at_cutoffs", report)
    monkeypatch.setattr(replay, "load_live_rosters", roster)
    monkeypatch.setattr(replay, "create_df_to_predict", builder)
    frame = replay.closing_replay_frame(
        schedule(), snapshot, DAY, ticks=ticks(), crews=pd.DataFrame()
    )
    expected = [TIP - pd.Timedelta(minutes=h) for h in (480, 360, 240, 0)]
    assert builder_calls == report_calls == expected
    assert roster_calls == [TIP]
    assert frame.TIME_TO_MATCH_MIN.tolist() == [480, 360, 240, 0]
    assert frame.INPUT_MODE.eq("replay").all()
    assert frame.LINE_SOURCE.eq("covers").all()
    assert frame.ODDS_TOTAL_LINE_bet365.eq(231).all()
    assert frame.ODDS_SNAP_TOT_BET365_RAW_LINE.tolist() == [230.5, 230.5, 235.5, 240.5]


@pytest.mark.parametrize("filed_at_8h", [(HOME, AWAY), (HOME,), (AWAY,), ()])
def test_closing_replay_matches_live_report_eligibility_and_out_sets_per_cutoff(
    monkeypatch, filed_at_8h
):
    horizons = (480, 360, 240, 0)
    snapshots = pd.DataFrame(
        {
            "GAME_ID": GAME,
            "TIME_TO_MATCH_MIN": horizons,
            "SNAPSHOT_TS_UTC": [TIP - pd.Timedelta(minutes=h) for h in horizons],
            "SNAPSHOT_PASSED": True,
        }
    )
    states = {
        horizon: injury_state(
            filed=filed_at_8h if horizon == 480 else (HOME, AWAY),
            statuses=[
                (GAME, HOME, "111", status),
                ("0022600000", HOME, "999", "out"),
            ],
            history=True,
        )
        for horizon, status in zip(
            horizons, ("out", "doubtful", "questionable", "available"), strict=True
        )
    }
    monkeypatch.setattr(replay, "load_live_rosters", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(replay, "historical_odds", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(
        replay,
        "load_prediction_report_state_at_cutoffs",
        lambda cutoffs: states[int(cutoffs.snapshot_minutes.iloc[0])],
    )
    observed = {}

    def builder(*, scheduled_data, **kwargs):
        cutoff = scheduled_data["roster_as_of_utc"][GAME]
        horizon = int((TIP - cutoff).total_seconds() / 60)
        observed[horizon] = scheduled_data
        # The canonical builder excludes games_not_updated at its final filter.
        frame = scheduled_data["scheduled_games"][["GAME_ID", "GAME_TIME"]].copy()
        frame = frame.loc[~frame.GAME_ID.isin(scheduled_data["games_not_updated"])]
        return frame, frame

    monkeypatch.setattr(replay, "create_df_to_predict", builder)
    result = replay.closing_replay_frame(
        schedule(),
        snapshots,
        DAY,
        ticks=pd.DataFrame({"game_id": []}),
        crews=pd.DataFrame(),
    )
    assert observed[480]["games_not_updated"] == (
        [] if len(filed_at_8h) == 2 else [GAME]
    )
    assert observed[480]["injury_dict_scheduled"] == (
        {GAME: {team: ["111"] if team == HOME else [] for team in filed_at_8h}}
        if filed_at_8h
        else {}
    )
    assert observed[360]["injury_dict_scheduled"] == {GAME: {HOME: ["111"], AWAY: []}}
    for horizon in (240, 0):
        assert observed[horizon]["injury_dict_scheduled"] == {
            GAME: {HOME: [], AWAY: []}
        }
    for horizon in horizons:
        assert observed[horizon]["injury_report_state"] is states[horizon]
        assert "0022600000" not in observed[horizon]["injury_dict_scheduled"]
    assert all(observed[h]["games_not_updated"] == [] for h in (360, 240, 0))
    assert result.TIME_TO_MATCH_MIN.tolist() == (
        [480, 360, 240, 0] if len(filed_at_8h) == 2 else [360, 240, 0]
    )
