"""Official membership drives live features without altering past games."""

import pandas as pd
import pytest
from nba_ou.data_processing.players import attach_player_features as player_features
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
    prepare_player_history,
)
from nba_ou.data_processing.players.live_roster import (
    live_roster_history,
    supplement_rosters,
)
from nba_ou.data_processing.players.players_statistics import EWMA_HALFLIFE_GAMES
from nba_ou.utils.row_cache import RowCache

TEAM, OLD_TEAM = "1610612744", "1610612738"
GAME = "0022600003"


@pytest.fixture
def inputs():
    team = pd.DataFrame(
        {
            "GAME_ID": ["0022600001", "0022600002", GAME],
            "TEAM_ID": TEAM,
            "SEASON_ID": "22026",
            "SEASON_YEAR": 2026,
            "GAME_DATE": pd.to_datetime(["2026-10-22", "2026-10-24", "2026-10-26"]),
        }
    )
    rows = []
    for game in team.itertuples():
        for player, pts in [("1", 28), ("2", 99), ("3", 12), ("4", 18)]:
            rows.append(
                {
                    "GAME_ID": game.GAME_ID,
                    "GAME_DATE": game.GAME_DATE,
                    "TEAM_ID": OLD_TEAM if player == "1" else TEAM,
                    "SEASON_ID": "22026",
                    "SEASON_YEAR": 2026,
                    "PLAYER_ID": player,
                    "PLAYER_NAME": f"Player {player}",
                    "MIN": 30.0,
                    "PTS": pts if game.GAME_ID != GAME else 999.0,
                }
            )
    injuries = pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"])
    return team, pd.DataFrame(rows), injuries


def test_official_roster_replaces_old_members_and_report_adds_missing_player(inputs):
    team, players, injuries = inputs
    members = {(GAME, TEAM): {"1": "Transferred", "3": "Stayed", "9": "Rookie"}}
    reports = {(GAME, TEAM): ["4"]}
    result, _, availability = add_player_history_features(
        team,
        players,
        injuries,
        stat_cols=["PTS", "MIN"],
        injury_dict_scheduled={GAME: {TEAM: ["4"]}},
        report_out_overrides=reports,
        current_rosters=members,
        return_availability_dict=True,
    )
    current = result.loc[result.GAME_ID.eq(GAME)].iloc[0]
    assert current.TOP1_PLAYER_ID_PTS_BEFORE == "1"
    assert current.TOP1_PLAYER_PTS_BEFORE == pytest.approx(28.0)
    assert current.TOP1_INJURED_PLAYER_ID_PTS_BEFORE == "4"
    assert current.TOP1_INJURED_PLAYER_PTS_BEFORE == pytest.approx(18.0)
    assert availability[GAME][TEAM] == {"available": ["1", "3", "9"], "injured": ["4"]}


def test_rookie_has_membership_but_no_fabricated_statistics(inputs):
    _, players, _ = inputs
    history = live_roster_history(
        players,
        {"9": "Rookie"},
        game_id=GAME,
        team_id=TEAM,
        season_id="22026",
        game_date=pd.Timestamp("2026-10-26"),
        stat_cols=["PTS", "MIN"],
    )
    assert history.PLAYER_ID.tolist() == ["9"]
    assert history[["PTS_CUM_AVG", "MIN_CUM_AVG", "MIN", "PTS"]].isna().all().all()


def test_live_override_leaves_historical_features_identical(inputs):
    team, players, injuries = inputs
    baseline, _ = add_player_history_features(
        team, players, injuries, stat_cols=["PTS", "MIN"]
    )
    changed, _ = add_player_history_features(
        team,
        players,
        injuries,
        stat_cols=["PTS", "MIN"],
        current_rosters={(GAME, TEAM): {"3": "Stayed"}},
    )
    pd.testing.assert_frame_equal(
        baseline.loc[baseline.GAME_ID.ne(GAME)], changed.loc[changed.GAME_ID.ne(GAME)]
    )


def test_row_cache_includes_roster_membership(inputs):
    team, players, injuries = inputs
    cache = RowCache()
    first, _ = add_player_history_features(
        team,
        players,
        injuries,
        stat_cols=["PTS", "MIN"],
        current_rosters={(GAME, TEAM): {"3": "Stayed"}},
        row_cache=cache,
    )
    second, _ = add_player_history_features(
        team,
        players,
        injuries,
        stat_cols=["PTS", "MIN"],
        current_rosters={(GAME, TEAM): {"1": "Transfer"}},
        row_cache=cache,
    )
    assert first.loc[first.GAME_ID.eq(GAME), "TOP1_PLAYER_ID_PTS_BEFORE"].item() == "3"
    assert (
        second.loc[second.GAME_ID.eq(GAME), "TOP1_PLAYER_ID_PTS_BEFORE"].item() == "1"
    )


def test_report_membership_only_supplements_existing_overrides():
    original = {(GAME, TEAM): {"1": "Existing"}}
    result = supplement_rosters(original, {GAME: {TEAM: ["9"]}, "past": {TEAM: ["8"]}})
    assert result == {(GAME, TEAM): {"1": "Existing", "9": "9"}}
    assert original == {(GAME, TEAM): {"1": "Existing"}}


@pytest.mark.parametrize("array_steps", [True, False], ids=["arrays", "frames"])
@pytest.mark.parametrize("status", ["out", "questionable"])
@pytest.mark.parametrize("last_row", ["appearance", "dnp_zero", "dnp_missing"])
@pytest.mark.parametrize(
    "minutes_cross_threshold",
    [False, True],
    ids=["steady-minutes", "threshold-crossing"],
)
def test_live_unavailable_statistics_match_training_with_variable_history(
    monkeypatch, array_steps, status, last_row, minutes_cross_threshold
):
    monkeypatch.setattr(player_features, "ARRAY_ROSTER_STEPS", array_steps)
    dates = pd.to_datetime(["2026-10-20", "2026-10-22", "2026-10-24", "2026-10-26"])
    ids = [f"002260000{i}" for i in range(1, 5)]
    target = ids[-1]
    team = pd.DataFrame(
        {
            "GAME_ID": ids,
            "GAME_DATE": dates,
            "TEAM_ID": TEAM,
            "SEASON_ID": "22026",
            "SEASON_YEAR": 2026,
        }
    )
    rows = []
    for game, date in zip(ids, dates, strict=True):
        for player in ("1", "2", "3"):
            if player != "3" and (
                game == target or (game == ids[2] and last_row == "appearance")
            ):
                continue  # Out/Questionable players have no target box-score row.
            minutes, points = 30.0, 9.0 if player == "2" else 15.0
            if player == "1":
                points = 10.0 if game == ids[0] else 30.0
                minutes = (
                    (11.0 if game == ids[0] else 40.0)
                    if minutes_cross_threshold
                    else 30.0
                )
            if player != "3" and game == ids[2]:
                minutes = 0.0 if last_row == "dnp_zero" else float("nan")
                points = float("nan")
            rows.append(
                dict(
                    GAME_ID=game,
                    GAME_DATE=date,
                    TEAM_ID=TEAM,
                    SEASON_ID="22026",
                    SEASON_YEAR=2026,
                    PLAYER_ID=player,
                    PLAYER_NAME=f"Player {player}",
                    MIN=minutes,
                    PTS=points,
                )
            )
    players = pd.DataFrame(rows)
    injuries = pd.DataFrame(
        {"GAME_ID": [target, target], "TEAM_ID": TEAM, "PLAYER_ID": ["1", "2"]}
    )
    prepared = prepare_player_history(players, ["PTS", "MIN"])
    kwargs = dict(
        stat_cols=["PTS", "MIN"],
        prepared_players=prepared,
        report_out_overrides={(target, TEAM): ["1", "2"] if status == "out" else []},
        report_questionable_sets=(
            {(target, TEAM): ["1", "2"]} if status == "questionable" else {}
        ),
    )
    training, _ = add_player_history_features(team, players, injuries, **kwargs)
    actual, _ = add_player_history_features(
        team,
        players,
        injuries,
        **kwargs,
        current_rosters={(target, TEAM): {p: f"Player {p}" for p in ("1", "2", "3")}},
    )
    expected = training.loc[training.GAME_ID.eq(target)].iloc[0]
    current = actual.loc[actual.GAME_ID.eq(target)].iloc[0]
    group = "INJURED" if status == "out" else "QUESTIONABLE"
    if last_row == "appearance":
        # Player 1 scored 10, then 30, then is Out/Questionable. His value
        # includes the 30-point game: the EWMA after it, not the one before it
        # (which read 10), with the latest game weighted 1 and the one before
        # it 0.5 ** (1 / halflife). With crossing minutes, that game also lifts
        # his minutes average (11 -> ~26) over the injured threshold (12), so he
        # still outranks player 2's steady 9 points.
        older = 0.5 ** (1 / EWMA_HALFLIFE_GAMES)
        assert expected[f"TOP1_{group}_PLAYER_ID_PTS_BEFORE"] == "1"
        assert expected[f"TOP1_{group}_PLAYER_PTS_BEFORE"] == pytest.approx(
            (30.0 + older * 10.0) / (1.0 + older)
        )
    pd.testing.assert_series_equal(current, expected, check_exact=True)


def test_report_only_rookie_is_kept_but_ranked_like_training(inputs):
    # The rookie stays a member (and injured) even when the roster fetch
    # failed. With no prior appearance he has no state to rank, so the injured
    # slots read exactly what a training row reads: empty, not NaN or a count.
    team, players, injuries = inputs
    kwargs = dict(
        stat_cols=["PTS", "MIN"],
        injury_dict_scheduled={GAME: {TEAM: ["9"]}},
        report_out_overrides={(GAME, TEAM): ["9"]},
        snapshot_game_ids={GAME},
        return_availability_dict=True,
    )
    result, _, availability = add_player_history_features(
        team, players, injuries, current_game_ids={GAME}, **kwargs
    )
    training, _, _ = add_player_history_features(team, players, injuries, **kwargs)
    assert availability[GAME][TEAM]["injured"] == ["9"]
    current = result.loc[result.GAME_ID.eq(GAME)]
    expected = training.loc[training.GAME_ID.eq(GAME)]
    injured = [c for c in result if "INJURED" in c]
    pd.testing.assert_frame_equal(
        current[injured].reset_index(drop=True),
        expected[injured].reset_index(drop=True),
    )
    assert current.TOP1_INJURED_PLAYER_ID_PTS_BEFORE.isna().all()


@pytest.mark.parametrize("array_steps", [True, False], ids=["arrays", "frames"])
@pytest.mark.parametrize("status", ["out", "questionable", "available"])
@pytest.mark.parametrize("same_day_row", [False, True])
@pytest.mark.parametrize("future_debut", [False, True])
def test_previous_season_member_is_not_lost_from_a_partly_known_roster(
    monkeypatch, array_steps, status, same_day_row, future_debut
):
    monkeypatch.setattr(player_features, "ARRAY_ROSTER_STEPS", array_steps)
    dates = pd.to_datetime(["2025-11-01", "2025-11-03", "2026-11-01", "2026-11-03"])
    ids = ["0022500001", "0022500002", "0022600001", "0022600002"]
    rows = []
    for game, date in zip(ids, dates, strict=True):
        year = 2025 if date.year == 2025 else 2026
        for player, points in (("1", 28.0), ("2", 12.0)):
            if player == "1" and year == 2026 and not same_day_row:
                continue  # A veteran who has not debuted this season.
            row = dict(
                GAME_ID=game,
                GAME_DATE=date,
                TEAM_ID=TEAM,
                SEASON_ID=f"2{year}",
                SEASON_YEAR=year,
                PLAYER_ID=player,
                PLAYER_NAME=f"Player {player}",
                MIN=30.0,
                PTS=points,
            )
            if player == "1" and year == 2026:
                row.update(MIN=0.0, PTS=float("nan"))
            rows.append(row)
    if future_debut:
        # A future debut must not replace the veteran's already known history.
        rows.append(
            dict(
                rows[0],
                GAME_ID="0022600009",
                GAME_DATE=pd.Timestamp("2026-12-01"),
                SEASON_ID="22026",
                SEASON_YEAR=2026,
                MIN=40.0,
                PTS=999.0,
            )
        )
    players = pd.DataFrame(rows)
    team = players.loc[
        players.PLAYER_ID.eq("2"),
        ["GAME_ID", "GAME_DATE", "TEAM_ID", "SEASON_ID", "SEASON_YEAR"],
    ]
    current_ids = ids[2:]
    injuries = pd.DataFrame({"GAME_ID": current_ids, "TEAM_ID": TEAM, "PLAYER_ID": "1"})
    kwargs = dict(
        stat_cols=["PTS", "MIN"],
        report_out_overrides={
            (g, TEAM): ["1"] if status == "out" else [] for g in current_ids
        },
        report_questionable_sets={
            (g, TEAM): ["1"] if status == "questionable" else [] for g in current_ids
        },
    )
    historical, _, availability = add_player_history_features(
        team.copy(), players, injuries, return_availability_dict=True, **kwargs
    )
    # The same membership supplied by an official roster must produce identical
    # values, even when other players already have rows in the current season.
    live, _ = add_player_history_features(
        team.copy(),
        players,
        injuries,
        **kwargs,
        current_rosters={
            (g, TEAM): {"1": "Veteran", "2": "Rotation"} for g in current_ids
        },
        current_game_ids=set(current_ids),
    )
    group = (
        "INJURED_"
        if status == "out"
        else "QUESTIONABLE_" if status == "questionable" else ""
    )
    for game in current_ids:
        row = historical.loc[historical.GAME_ID.eq(game)].iloc[0]
        assert row[f"TOP1_{group}PLAYER_ID_PTS_BEFORE"] == "1"
        assert row[f"TOP1_{group}PLAYER_PTS_BEFORE"] == pytest.approx(28.0)
        if status != "available":
            assert row[f"N_{group}PLAYERS_BEFORE"] == 1
        assert "1" in availability[game][TEAM]["injured"]
    pd.testing.assert_frame_equal(
        historical.loc[historical.GAME_ID.isin(current_ids)],
        live.loc[live.GAME_ID.isin(current_ids)],
        check_exact=True,
    )


@pytest.mark.parametrize("array_steps", [True, False], ids=["arrays", "frames"])
@pytest.mark.parametrize("same_day_row", [False, True])
def test_available_player_reads_all_prior_games_without_a_target_box_score(
    monkeypatch, array_steps, same_day_row
):
    monkeypatch.setattr(player_features, "ARRAY_ROSTER_STEPS", array_steps)
    from .test_historical_player_availability import TARGET_GAME, _players, _team_rows

    players = _players(target_star_minutes=0.0, target_star_points=None)
    star = players.PLAYER_ID.eq("star")
    players.loc[star & players.GAME_ID.eq("0022400001"), ["PTS", "MIN"]] = [10.0, 11.0]
    players.loc[star & players.GAME_ID.eq("0022400002"), ["PTS", "MIN"]] = [30.0, 40.0]
    if not same_day_row:
        players = players.loc[~(star & players.GAME_ID.eq(TARGET_GAME))]
    injuries = pd.DataFrame([dict(GAME_ID=TARGET_GAME, TEAM_ID=TEAM, PLAYER_ID="star")])
    # Helpers use a different team; both the source and target must agree.
    team = _team_rows().assign(TEAM_ID=TEAM)
    players = players.assign(TEAM_ID=TEAM)
    kwargs = dict(
        stat_cols=["PTS", "MIN"], report_out_overrides={(TARGET_GAME, TEAM): []}
    )
    historical, _ = add_player_history_features(
        team.copy(), players, injuries, **kwargs
    )
    live, _ = add_player_history_features(
        team.copy(),
        players,
        injuries,
        **kwargs,
        current_rosters={(TARGET_GAME, TEAM): {"star": "Star", "rotation": "Rotation"}},
        current_game_ids={TARGET_GAME},
    )
    row = historical.loc[historical.GAME_ID.eq(TARGET_GAME)].iloc[0]
    weight = 0.5 ** (1 / EWMA_HALFLIFE_GAMES)
    assert row.TOP1_PLAYER_ID_PTS_BEFORE == "star"
    assert row.TOP1_PLAYER_PTS_BEFORE == pytest.approx(
        (30 + weight * 10) / (1 + weight)
    )
    pd.testing.assert_frame_equal(
        historical.loc[historical.GAME_ID.eq(TARGET_GAME)],
        live.loc[live.GAME_ID.eq(TARGET_GAME)],
        check_exact=True,
    )


def test_closing_adapter_passes_official_roster_and_report_membership(inputs):
    from nba_ou.create_training_data.create_df_to_predict import (
        process_player_statistics_for_training,
    )

    team, players, injuries = inputs
    players.insert(players.columns.get_loc("MIN"), "START_POSITION", "G")
    # The closing adapter requests these statistics in addition to PTS/MIN.
    for column in ["PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT"]:
        players[column] = 100.0
    games = pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "GAME_DATE_EST": ["2026-10-26"],
            "SEASON": [2026],
            "HOME_TEAM_ID": [TEAM],
            "VISITOR_TEAM_ID": [OLD_TEAM],
        }
    )
    result, _ = process_player_statistics_for_training(
        players,
        team,
        injuries,
        ["2026-27"],
        "2026-10-25",
        scheduled_games=games,
        current_rosters={(GAME, TEAM): {"1": "Incoming", "9": "Rookie"}},
        report_listed_players={GAME: {TEAM: ["4"]}},
    )
    current = result.loc[result.GAME_ID.eq(GAME)].iloc[0]
    assert current.TOP1_PLAYER_ID_PTS_BEFORE == "1"
    assert current.TOP1_PLAYER_PTS_BEFORE == pytest.approx(28.0)


@pytest.mark.parametrize("target_only", [False, True], ids=["full", "tonight-only"])
def test_intermediate_horizons_share_the_run_roster_like_closing(
    inputs, monkeypatch, target_only
):
    # The roster fetched at 20:00 also applies to the 17:00 horizon: every
    # horizon reads the same pool the closing prediction would.
    from nba_ou.create_training_data import intermediate_injuries as pipeline
    from nba_ou.create_training_data.create_base_game_features import BaseInjuryContext
    from nba_ou.data_processing.injury_status.report_state import InjuryReportState

    _, players, missing = inputs
    for column in pipeline.PLAYER_STATS:
        if column not in players:
            players[column] = 100.0
    team = pd.DataFrame(
        {
            "GAME_ID": [GAME, GAME],
            "TEAM_ID": [TEAM, OLD_TEAM],
            "HOME": [True, False],
            "SEASON_ID": "22026",
            "SEASON_YEAR": 2026,
            "GAME_DATE": pd.Timestamp("2026-10-26"),
        }
    )
    games = pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "GAME_DATE_EST": ["2026-10-26"],
            "HOME_TEAM_ID": [TEAM],
            "VISITOR_TEAM_ID": [OLD_TEAM],
        }
    )
    roster = pd.DataFrame(
        {
            "TEAM_ID": TEAM,
            "SEASON": "2026-27",
            "PLAYER_ID": ["1", "3", "9"],
            "PLAYER_NAME": ["Incoming", "Stayed", "Rookie"],
            "POSITION": "G",
            "FETCHED_AT_UTC": pd.Timestamp("2026-10-26T20:00Z"),
            "SOURCE": "nba",
        }
    )
    context = BaseInjuryContext(
        team,
        players,
        ["2026-27"],
        ["2025-26", "2026-27"],
        games,
        roster,
        pd.Timestamp("2026-10-26T22:00Z"),
    )
    state = InjuryReportState.empty()
    state.filings = pd.DataFrame(
        {"game_id": [GAME, GAME], "team_id": [TEAM, OLD_TEAM], "submitted": True}
    )
    voting = pd.DataFrame(
        {
            "season_year": [2025, 2025],
            "player_id": ["1", "2"],
            "team_name": ["Boston Celtics", "Golden State Warriors"],
            "fan_votes": [10, 90],
            "score": [1.0, 2.0],
        }
    )
    base = pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME": [0.9],
            "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY": [0.7],
        }
    )
    snapshots = pd.DataFrame(
        {
            "GAME_ID": [GAME, GAME],
            "TIME_TO_MATCH_MIN": [360, 60],
            "SNAPSHOT_TS_UTC": pd.to_datetime(
                ["2026-10-26T17:00Z", "2026-10-26T22:00Z"]
            ),
            "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME": [0.9, 0.9],
            "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY": [0.7, 0.7],
        }
    )
    monkeypatch.setattr(
        pipeline, "_states_at_snapshots", lambda *a, **kw: {360: state, 60: state}
    )
    monkeypatch.setattr(pipeline, "_injury_history_memo", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "get_injury_data_from_db", lambda *a: missing)
    monkeypatch.setattr(pipeline, "load_player_box_history", lambda: pd.DataFrame())
    monkeypatch.setattr(pipeline, "prepare_box_and_form", lambda b: b)
    monkeypatch.setattr(pipeline, "load_all_star_voting_from_db", lambda **kw: voting)
    monkeypatch.setattr(
        pipeline, "_add_availability_effects", lambda games, *a, **kw: games
    )

    def reports(frame, *args):
        return frame.assign(INJURY_REPORT_COVERED_BEFORE=1)

    monkeypatch.setattr(pipeline, "add_injury_report_features", reports)
    monkeypatch.setattr(
        pipeline,
        "report_counter_features",
        lambda frame, state: pd.DataFrame(
            {"INJURY_REPORT_COVERED_BEFORE": 1}, index=frame.index
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "add_status_features",
        lambda frame, *a: pd.DataFrame(index=frame.index),
    )
    result = pipeline.add_snapshot_injury_features(
        snapshots, base, context, target_game_ids={GAME} if target_only else None
    ).set_index("TIME_TO_MATCH_MIN")
    assert result.TOP1_PLAYER_PTS_BEFORE_TEAM_HOME.tolist() == pytest.approx(
        [28.0, 28.0]
    )
    assert result.ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME.tolist() == pytest.approx(
        [0.1, 0.1]
    )
    assert result.ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY.tolist() == [0.7, 0.7]
    assert not any(c.startswith("_") for c in result)
