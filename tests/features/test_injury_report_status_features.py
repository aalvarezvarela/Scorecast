"""Closing-line features from the last injury report before tipoff."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.injury_status import status_history as Q
from nba_ou.data_processing.injury_status.features import (
    add_injury_report_features,
    mask_uncovered_group_columns,
)
from nba_ou.data_processing.injury_status.report_state import (
    COVERED_COL,
    REPORT_AGE_COL,
    InjuryReportState,
    apply_report_out_overrides,
    counter_col,
    nested_status_dict,
    report_counter_features,
    report_out_overrides,
    report_questionable_sets,
)
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
)

TEAM = "1610612738"
OPP = "1610612744"


# --------------------------------------------------------------------------------
# Report state: out set, overrides, counters
# --------------------------------------------------------------------------------


def _state(statuses=(), filings=(), ages=(), events=(), listed=()):
    return InjuryReportState(
        statuses=pd.DataFrame(
            list(statuses),
            columns=[
                "game_id",
                "team_id",
                "player_id",
                "status",
                "reason_category",
                "reason_detail",
                "game_date",
                "season_year",
            ],
        ),
        filings=pd.DataFrame(
            list(filings), columns=["game_id", "team_id", "submitted"]
        ),
        report_age=pd.DataFrame(list(ages), columns=["game_id", "report_age_minutes"]),
        status_events=pd.DataFrame(
            list(events),
            columns=[
                "game_id",
                "team_id",
                "player_id",
                "status",
                "game_date",
                "season_year",
                "at_last_report",
            ],
        ),
        listed_pairs=pd.DataFrame(list(listed), columns=["game_id", "player_id"]),
    )


def _status(game, team, player, status, category="injury_illness"):
    return (game, team, player, status, category, "", "2024-10-24", 2024)


def test_the_report_splits_a_filed_roster_into_three_availability_groups():
    state = _state(
        statuses=[
            _status("G1", TEAM, "a", "out"),
            _status("G1", TEAM, "b", "doubtful"),
            _status("G1", TEAM, "c", "questionable"),
            _status("G1", TEAM, "d", "probable"),
            _status("G1", TEAM, "e", "available"),
            _status("G1", OPP, "x", "out"),
        ],
        filings=[("G1", TEAM, True), ("G1", OPP, False)],
    )
    # Injured is Out + Doubtful. Questionable is its own group, and Probable and
    # Available are in neither, so they stay on the available side.
    assert report_out_overrides(state) == {("G1", TEAM): ["a", "b"]}
    assert report_questionable_sets(state) == {("G1", TEAM): ["c"]}
    # The unfiled opponent appears in neither, so its legacy set stands.
    assert ("G1", OPP) not in report_out_overrides(state)
    assert ("G1", OPP) not in report_questionable_sets(state)


def test_a_filed_team_with_nobody_questionable_is_a_fact_not_missing_data():
    state = _state(
        statuses=[_status("G1", TEAM, "a", "out")],
        filings=[("G1", TEAM, True)],
    )
    assert report_questionable_sets(state) == {("G1", TEAM): []}
    assert nested_status_dict(report_questionable_sets(state)) == {"G1": {TEAM: []}}


def test_a_filed_team_with_nobody_out_replaces_the_legacy_set():
    state = _state(filings=[("G1", TEAM, True)])
    legacy = {"G1": {TEAM: ["late_scratch"]}, "G0": {TEAM: ["old"]}}
    result = apply_report_out_overrides(legacy, report_out_overrides(state))
    assert result == {"G0": {TEAM: ["old"]}}


def test_uncovered_team_games_keep_the_legacy_set_unchanged():
    legacy = {"G1": {TEAM: ["p1"], OPP: ["p2"]}}
    overrides = {("G1", TEAM): ["q1"]}
    assert apply_report_out_overrides(legacy, overrides) == {
        "G1": {TEAM: ["q1"], OPP: ["p2"]}
    }
    assert apply_report_out_overrides(legacy, None) == legacy


def test_counters_skip_g_league_and_are_nan_without_a_report():
    state = _state(
        statuses=[
            _status("G1", TEAM, "a", "out"),
            _status("G1", TEAM, "two_way", "out", category="g_league"),
            _status("G1", TEAM, "c", "questionable"),
            _status("G1", TEAM, "d", "probable"),
        ],
        filings=[("G1", TEAM, True), ("G1", OPP, False)],
        ages=[("G1", 15.0)],
    )
    rows = pd.DataFrame({"GAME_ID": ["G1", "G1", "G2"], "TEAM_ID": [TEAM, OPP, TEAM]})
    out = report_counter_features(rows, state)

    covered = out.iloc[0]
    assert covered[COVERED_COL] == 1
    assert covered[REPORT_AGE_COL] == 15.0
    assert covered[counter_col("out")] == 1
    assert covered[counter_col("doubtful")] == 0
    assert covered[counter_col("questionable")] == 1
    assert covered[counter_col("probable")] == 1
    for row in (out.iloc[1], out.iloc[2]):
        assert row[COVERED_COL] == 0
        assert row[[REPORT_AGE_COL, counter_col("out")]].isna().all()


def test_report_derived_names_avoid_the_injury_zero_fill_substrings():
    # "injury_"/"injured_" columns are zero-filled by the missing-data policy,
    # which would turn "no report" into "fresh report, nobody listed".
    names = [REPORT_AGE_COL, *(counter_col(s) for s in ("out", "probable"))]
    names += Q.status_feature_columns()
    for name in names:
        assert "injury_" not in name.lower() and "injured_" not in name.lower()
        assert "_BEFORE" in name


# --------------------------------------------------------------------------------
# Player history features: pre-game split vs realized history
# --------------------------------------------------------------------------------

SEASON_ID = "22024"
TARGET = "0022400003"


def _team_rows():
    return pd.DataFrame(
        {
            "GAME_ID": ["0022400001", "0022400002", TARGET],
            "TEAM_ID": [TEAM] * 3,
            "SEASON_ID": [SEASON_ID] * 3,
            "SEASON_YEAR": [2024] * 3,
            "GAME_DATE": pd.to_datetime(["2024-10-20", "2024-10-22", "2024-10-24"]),
        }
    )


def _players(target_star_minutes=30.0):
    rows = []
    for game, date, star_min in [
        ("0022400001", "2024-10-20", 32.0),
        ("0022400002", "2024-10-22", 0.0),
        (TARGET, "2024-10-24", target_star_minutes),
    ]:
        for player, pts, minutes in [
            ("star", 30.0, star_min),
            ("rotation", 15.0, 24.0),
        ]:
            rows.append(
                {
                    "GAME_ID": game,
                    "TEAM_ID": TEAM,
                    "SEASON_ID": SEASON_ID,
                    "SEASON_YEAR": 2024,
                    "GAME_DATE": pd.Timestamp(date),
                    "PLAYER_ID": player,
                    "PLAYER_NAME": player,
                    "MIN": minutes,
                    "PTS": pts if minutes > 0 else None,
                    "PACE_PER40": 99.0 if minutes > 0 else None,
                    "COMMENT": "",
                }
            )
    return pd.DataFrame(rows)


def _injuries():
    # star was on the inactive list for game 2 only.
    return pd.DataFrame(
        [{"GAME_ID": "0022400002", "TEAM_ID": TEAM, "PLAYER_ID": "star"}]
    )


def _run(overrides=None, questionable=None):
    return add_player_history_features(
        _team_rows(),
        _players(),
        _injuries(),
        stat_cols=["PTS"],
        return_availability_dict=True,
        report_out_overrides=overrides,
        report_questionable_sets=questionable,
    )


def test_overrides_for_other_games_leave_the_build_identical():
    base, base_dict, base_avail = _run(None)
    other, other_dict, other_avail = _run({("0099999999", TEAM): ["someone"]})
    pd.testing.assert_frame_equal(base, other)
    assert base_dict == other_dict
    assert base_avail == other_avail


def test_an_out_star_is_out_pre_game_but_available_in_history():
    out, realized, availability = _run({(TARGET, TEAM): ["star"]})
    target = out.loc[out["GAME_ID"].eq(TARGET)].iloc[0]

    assert target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"] == "star"
    assert target["TOP1_PLAYER_ID_PTS_BEFORE"] == "rotation"
    # He played, so the history map and the realized dict must say so.
    assert "star" in availability[TARGET][TEAM]["available"]
    assert TARGET not in realized
    # Streak: out pre-game tonight, and actually out in game 2.
    assert target["TOP1_INJURED_STREAK_PTS_BEFORE"] == 2


def test_a_questionable_star_forms_the_third_group():
    # Covered team-game, nobody Out or Doubtful, the star Questionable: he
    # belongs to neither the injured nor the available group.
    out, _, availability = _run(
        {(TARGET, TEAM): []}, questionable={(TARGET, TEAM): ["star"]}
    )
    target = out.loc[out["GAME_ID"].eq(TARGET)].iloc[0]

    assert target["TOP1_QUESTIONABLE_PLAYER_ID_PTS_BEFORE"] == "star"
    assert target["TOP1_QUESTIONABLE_PLAYER_PTS_BEFORE"] == pytest.approx(30.0)
    assert target["N_QUESTIONABLE_PLAYERS_BEFORE"] == 1
    # Not subtracted as an absence...
    assert target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"] is None
    assert target["N_INJURED_PLAYERS_BEFORE"] == 0
    # ...and not counted among the available either.
    assert target["TOP1_PLAYER_ID_PTS_BEFORE"] == "rotation"
    # The streak reads like its injured twin: tonight plus game 2's absence.
    assert target["TOP1_QUESTIONABLE_STREAK_PTS_BEFORE"] == 2
    # History still records what actually happened: he played.
    assert "star" in availability[TARGET][TEAM]["available"]


def test_questionable_group_uses_the_reduced_player_profile():
    out, _ = add_player_history_features(
        _team_rows(),
        _players(),
        _injuries(),
        stat_cols=["PTS", "PACE_PER40", "MIN"],
        report_out_overrides={(TARGET, TEAM): []},
        report_questionable_sets={(TARGET, TEAM): ["star"]},
    )
    target = out.loc[out["GAME_ID"].eq(TARGET)].iloc[0]

    assert target["TOP1_QUESTIONABLE_PLAYER_ID_MIN_BEFORE"] == "star"
    assert target["TOP1_QUESTIONABLE_STREAK_MIN_BEFORE"] == 2
    assert target["QUESTIONABLE_WEIGHTED_PACE_PER40_BEFORE"] == pytest.approx(99.0)
    assert "TOP1_QUESTIONABLE_PLAYER_PACE_PER40_BEFORE" not in out.columns
    assert "TOTAL_QUESTIONABLE_PLAYER_PACE_PER40_BEFORE" not in out.columns
    assert pd.isna(out.iloc[0]["QUESTIONABLE_WEIGHTED_PACE_PER40_BEFORE"])


def test_a_covered_team_game_with_nobody_questionable_reads_zero():
    out, _, _ = _run({(TARGET, TEAM): ["star"]}, questionable={(TARGET, TEAM): []})
    target = out.loc[out["GAME_ID"].eq(TARGET)].iloc[0]
    assert target["N_QUESTIONABLE_PLAYERS_BEFORE"] == 0
    # The empty slot carries no player; only the count says "nobody at risk".
    assert pd.isna(target["TOP1_QUESTIONABLE_PLAYER_ID_PTS_BEFORE"])
    assert target["TOP1_QUESTIONABLE_PLAYER_PTS_BEFORE"] == 0
    assert target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"] == "star"


def test_an_uncovered_team_game_has_no_questionable_group_at_all():
    # The report covers some other game, so this team-game's questionable
    # columns are unknown, not zero.
    out, _, _ = _run(questionable={("0099999999", TEAM): ["someone"]})
    assert out["N_QUESTIONABLE_PLAYERS_BEFORE"].isna().all()
    assert out["TOP1_QUESTIONABLE_STREAK_PTS_BEFORE"].isna().all()


def test_without_a_questionable_set_the_roster_splits_in_two():
    out = add_player_history_features(
        _team_rows(), _players(), _injuries(), stat_cols=["PTS"]
    )[0]
    assert not [c for c in out.columns if "QUESTIONABLE" in c]


def test_available_roster_count_is_off_unless_requested():
    # A build without report features does not gain the report roster counter.
    out = add_player_history_features(
        _team_rows(), _players(), _injuries(), stat_cols=["PTS"]
    )[0]
    assert "N_AVAILABLE_ROSTER_PLAYERS_BEFORE" not in out.columns


def test_available_roster_count_ignores_the_target_game_box_score():
    played = add_player_history_features(
        _team_rows(),
        _players(30.0),
        _injuries(),
        stat_cols=["PTS"],
        include_available_roster_count=True,
    )[0]
    sat = add_player_history_features(
        _team_rows(),
        _players(0.0),
        _injuries(),
        stat_cols=["PTS"],
        include_available_roster_count=True,
    )[0]
    col = "N_AVAILABLE_ROSTER_PLAYERS_BEFORE"
    pd.testing.assert_series_equal(played[col], sat[col])
    assert played.loc[played["GAME_ID"].eq(TARGET), col].iloc[0] == 2


# --------------------------------------------------------------------------------
# Questionable history
# --------------------------------------------------------------------------------


def _box_row(game, date, player, minutes, pts, season=2024, pace=100.0, team=TEAM):
    return {
        "game_id": game,
        "team_id": team,
        "player_id": player,
        "season_year": season,
        "game_date": date,
        "MIN": minutes,
        "PTS": pts,
        "PACE_PER40": pace if minutes and minutes > 0 else np.nan,
    }


def _history():
    """Player "q" is Questionable in games 3, 5 and 7 and plays in 3 and 7.

    Every other game he plays 30 min / 20 pts; when Questionable and playing he
    plays 25 min / 15 pts. "u" is never listed and is the drift baseline.
    """
    dates = pd.date_range("2024-10-20", periods=10, freq="2D")
    box, events, filings, listed = [], [], [], []
    for i, date in enumerate(dates, start=1):
        game = f"00224000{i:02d}"
        filings += [(game, TEAM, True)]
        q_listed = i in (3, 5, 7)
        if q_listed:
            events.append((game, TEAM, "q", "questionable", date, 2024, True))
            listed.append((game, "q"))
            if i != 5:
                box.append(_box_row(game, date, "q", 25.0, 15.0))
        else:
            box.append(_box_row(game, date, "q", 30.0, 20.0))
        box.append(_box_row(game, date, "u", 28.0, 12.0))
    state = _state(filings=filings, events=events, listed=listed)
    return state, pd.DataFrame(box), dates


def _estimate(state, box, date, player="q", status="questionable"):
    box = Q.prepare_box_history(box)
    form = Q.build_player_form(box)
    events = Q.build_status_events(state, box, form)
    targets = pd.DataFrame(
        {
            "player_id": [player],
            "season_year": [2024],
            "game_date": [pd.Timestamp(date)],
        }
    )
    est = Q.status_player_estimates(targets, events, form, box, status)
    return est.iloc[0], events


def test_play_probability_uses_only_earlier_listings_with_league_prior():
    state, box, dates = _history()
    est, _ = _estimate(state, box, dates[7])  # game 8: after games 3, 5, 7
    league = 2 / 3  # fallback: all earlier last-report listings (< 200 in window)
    expected = (2 + Q.PLAYER_PLAY_PRIOR_K * league) / (3 + Q.PLAYER_PLAY_PRIOR_K)
    assert est["N_HISTORY"] == 3
    assert est["P_PLAY"] == pytest.approx(expected)


def test_a_same_day_listing_is_not_history():
    state, box, dates = _history()
    est, _ = _estimate(state, box, dates[6])  # game 7's own date
    assert est["N_HISTORY"] == 2


def test_future_outcomes_do_not_change_earlier_estimates():
    state, box, dates = _history()
    before, _ = _estimate(state, box, dates[5])
    changed = box.copy()
    game7 = changed["game_id"].eq("0022400007") & changed["player_id"].eq("q")
    changed.loc[game7, ["MIN", "PTS"]] = [0.0, 0.0]
    after, _ = _estimate(state, changed, dates[5])
    pd.testing.assert_series_equal(before, after)


def test_effect_when_playing_is_negative_and_shrunk_toward_the_prior():
    state, box, dates = _history()
    est, events = _estimate(state, box, dates[9])
    played = events.loc[events["played"] > 0]
    assert (played["d_MIN"] < 0).all()
    # Every prior level falls through to the all-statuses mean -- here only q's
    # own events -- and shrinking toward it leaves his mean unchanged.
    assert est["EFFECT_MIN"] == pytest.approx(played["d_MIN"].mean())
    assert est["EFFECT_MIN"] < 0


def test_team_rows_uncovered_are_nan_and_covered_without_questionable_are_zero():
    state, box, dates = _history()
    # Game 7 lists q at the last report; game 8 is filed with nobody Questionable.
    state.statuses = pd.DataFrame(
        [
            (
                "0022400007",
                TEAM,
                "q",
                "questionable",
                "injury_illness",
                "",
                dates[6],
                2024,
            )
        ],
        columns=state.statuses.columns,
    )
    rows = pd.DataFrame(
        {
            "GAME_ID": ["0022400007", "0022400008", "0022400008"],
            "TEAM_ID": [TEAM, TEAM, OPP],
            "GAME_DATE": [dates[6], dates[7], dates[7]],
            "SEASON_YEAR": [2024, 2024, 2024],
        },
        index=[10, 11, 12],
    )
    out = add_injury_report_features(rows, state, box)
    assert out.index.tolist() == [10, 11, 12]

    q = "questionable"
    listed = out.loc[10]
    assert 0 < listed[Q.top_col(q, 1, "P_PLAY")] < 1
    # Empty second slot: the neutral "nobody at risk" values, not NaN.
    assert listed[Q.top_col(q, 2, "P_PLAY")] == 1.0
    assert listed[Q.top_col(q, 2, "N_HISTORY")] == 0.0
    assert listed[Q.top_col(q, 2, "EFFECT_MIN")] == 0.0
    assert listed[Q.sum_exp_players_col(q)] == pytest.approx(
        listed[Q.top_col(q, 1, "P_PLAY")]
    )
    assert listed[Q.sum_exp_col(q, "MIN")] > 0
    # Nobody Probable on that team: neutral slot.
    assert listed[Q.top_col("probable", 1, "P_PLAY")] == 1.0
    assert listed[Q.sum_exp_players_col("probable")] == 0.0

    none_listed = out.loc[11]
    assert none_listed[Q.sum_exp_players_col(q)] == 0
    assert none_listed[Q.top_col(q, 1, "P_PLAY")] == 1.0
    assert none_listed[Q.top_col(q, 1, "EFFECT_PTS")] == 0.0
    assert none_listed[Q.mean_p_play_col(q)] == 1.0

    uncovered = out.loc[12]
    assert uncovered[COVERED_COL] == 0
    for status in Q.LISTED_STATUSES:
        assert pd.isna(uncovered[Q.top_col(status, 1, "FORM_MIN")])
    for status in Q.HISTORY_STATUSES:
        assert pd.isna(uncovered[Q.sum_exp_players_col(status)])
        assert pd.isna(uncovered[Q.top_col(status, 1, "EFFECT_MIN")])


def test_effect_k_is_fitted_only_on_earlier_seasons():
    rng = np.random.default_rng(0)
    rows = []
    for season in (2020, 2021):
        for p in range(40):
            shift = rng.normal(0, 2)
            for _ in range(6):
                rows.append(
                    {
                        "player_id": f"p{p}",
                        "season_year": season,
                        "d_MIN": shift + rng.normal(0, 4),
                    }
                )
    events = pd.DataFrame(rows)
    assert Q.fit_effect_k(events, "MIN", 2020) is None
    fitted = Q.fit_effect_k(events, "MIN", 2021)
    low, high = Q.EFFECT_K_BOUNDS
    assert low <= fitted <= high
    assert fitted is not None


def test_probable_gets_history_and_doubtful_only_its_players_form():
    state, box, dates = _history()
    # "p" is Probable in games 3 and 6 and plays both; "d" is never listed until
    # the target game, where he is Doubtful, so only his form can be read.
    extra_events = []
    for i, date in enumerate(dates, start=1):
        game = f"00224000{i:02d}"
        box = pd.concat(
            [
                box,
                pd.DataFrame(
                    [
                        _box_row(game, date, "p", 32.0, 18.0),
                        _box_row(game, date, "d", 34.0, 24.0),
                    ]
                ),
            ],
            ignore_index=True,
        )
        if i in (3, 6):
            extra_events.append((game, TEAM, "p", "probable", date, 2024, True))
    state.status_events = pd.concat(
        [
            state.status_events,
            pd.DataFrame(extra_events, columns=state.status_events.columns),
        ],
        ignore_index=True,
    )

    p_est, _ = _estimate(state, box, dates[7], player="p", status="probable")
    q_as_probable, _ = _estimate(state, box, dates[7], player="q", status="probable")
    assert p_est["N_HISTORY"] == 2 and p_est["P_PLAY"] == pytest.approx(1.0)
    # History is per status: q was never Probable.
    assert q_as_probable["N_HISTORY"] == 0

    state.statuses = pd.DataFrame(
        [
            ("0022400008", TEAM, "p", "probable", "injury_illness", "", dates[7], 2024),
            ("0022400008", TEAM, "d", "doubtful", "injury_illness", "", dates[7], 2024),
        ],
        columns=state.statuses.columns,
    )
    rows = pd.DataFrame(
        {
            "GAME_ID": ["0022400008"],
            "TEAM_ID": [TEAM],
            "GAME_DATE": [dates[7]],
            "SEASON_YEAR": [2024],
        }
    )
    out = add_injury_report_features(rows, state, box)
    row = out.iloc[0]
    assert row[Q.top_col("probable", 1, "N_HISTORY")] == 2
    assert row[Q.sum_exp_players_col("questionable")] == 0.0
    # Doubtful: counted, and the best listed player's own form.
    assert row[counter_col("doubtful")] == 1
    assert row[Q.top_col("doubtful", 1, "FORM_MIN")] == pytest.approx(34.0)
    assert row[Q.top_col("doubtful", 1, "FORM_PTS")] == pytest.approx(24.0)
    assert row[Q.top_col("doubtful", 1, "FORM_PACE_PER40")] == pytest.approx(100.0)
    # Probable and Questionable carry the same form columns.
    assert row[Q.top_col("probable", 1, "FORM_PTS")] == pytest.approx(18.0)
    # No play probability or effect is estimated for Doubtful: at the last
    # report it plays ~2% of the time, so those would be noise.
    assert not [
        c
        for c in out.columns
        if "_DOUBTFUL_" in c and not c.startswith(("N_REPORT_", "TOP1_"))
    ]
    assert not [c for c in out.columns if "DOUBTFUL" in c and "P_PLAY" in c]
    assert not [c for c in out.columns if "DOUBTFUL" in c and "EFFECT" in c]

    # A covered team-game with nobody Doubtful: the neutral empty slot, not NaN.
    state.statuses = state.statuses.iloc[:1]
    empty = add_injury_report_features(rows, state, box).iloc[0]
    assert empty[counter_col("doubtful")] == 0
    assert empty[Q.top_col("doubtful", 1, "FORM_PTS")] == 0.0


def test_out_set_slots_rank_by_minutes_form():
    # The slot says how much of the rotation is at risk, so the 32-minute role
    # player outranks the 22-minute scorer for Questionable and Doubtful.
    state, box, dates = _history()
    for i, date in enumerate(dates, start=1):
        game = f"00224000{i:02d}"
        box = pd.concat(
            [
                box,
                pd.DataFrame(
                    [
                        _box_row(game, date, "big_min", 32.0, 8.0),
                        _box_row(game, date, "big_pts", 22.0, 20.0),
                    ]
                ),
            ],
            ignore_index=True,
        )
    rows = pd.DataFrame(
        {
            "GAME_ID": ["0022400008"],
            "TEAM_ID": [TEAM],
            "GAME_DATE": [dates[7]],
            "SEASON_YEAR": [2024],
        }
    )

    for status in ("questionable", "doubtful"):
        state.statuses = pd.DataFrame(
            [
                (
                    "0022400008",
                    TEAM,
                    "big_min",
                    status,
                    "injury_illness",
                    "",
                    dates[7],
                    2024,
                ),
                (
                    "0022400008",
                    TEAM,
                    "big_pts",
                    status,
                    "injury_illness",
                    "",
                    dates[7],
                    2024,
                ),
            ],
            columns=state.statuses.columns,
        )
        row = add_injury_report_features(rows, state, box).iloc[0]
        assert Q.RANK_STAT[status] == "MIN"
        assert row[Q.top_col(status, 1, "FORM_MIN")] == pytest.approx(32.0)
        assert row[Q.top_col(status, 1, "FORM_PTS")] == pytest.approx(8.0)

    # Probable still ranks by points; both players are in the counters either way.
    state.statuses = pd.DataFrame(
        [
            (
                "0022400008",
                TEAM,
                "big_min",
                "probable",
                "injury_illness",
                "",
                dates[7],
                2024,
            ),
            (
                "0022400008",
                TEAM,
                "big_pts",
                "probable",
                "injury_illness",
                "",
                dates[7],
                2024,
            ),
        ],
        columns=state.statuses.columns,
    )
    row = add_injury_report_features(rows, state, box).iloc[0]
    assert Q.RANK_STAT["probable"] == "PTS"
    assert row[Q.top_col("probable", 1, "FORM_PTS")] == pytest.approx(20.0)
    assert row[counter_col("probable")] == 2


def test_a_g_league_listing_fills_no_slot_and_no_counter():
    # A two-way or assignment listing is a roster mechanic, so the slot and the
    # counter must agree that there is nobody at risk.
    state, box, dates = _history()
    for i, date in enumerate(dates, start=1):
        box = pd.concat(
            [box, pd.DataFrame([_box_row(f"00224000{i:02d}", date, "g", 20.0, 9.0)])],
            ignore_index=True,
        )
    state.statuses = pd.DataFrame(
        [
            ("0022400008", TEAM, "g", "doubtful", "g_league", "", dates[7], 2024),
            ("0022400008", TEAM, "g", "questionable", "g_league", "", dates[7], 2024),
        ],
        columns=state.statuses.columns,
    )
    rows = pd.DataFrame(
        {
            "GAME_ID": ["0022400008"],
            "TEAM_ID": [TEAM],
            "GAME_DATE": [dates[7]],
            "SEASON_YEAR": [2024],
        }
    )
    row = add_injury_report_features(rows, state, box).iloc[0]
    assert row[counter_col("doubtful")] == 0
    assert row[Q.top_col("doubtful", 1, "FORM_MIN")] == 0.0
    assert row[Q.top_col("questionable", 1, "FORM_MIN")] == 0.0
    assert row[Q.sum_exp_players_col("questionable")] == 0.0


# --------------------------------------------------------------------------------
# Availability-effect block: coverage masking
# --------------------------------------------------------------------------------


def test_uncovered_sides_lose_the_questionable_effect_block():
    # The effect builder is coverage-blind: it fills a missing aggregate with 0,
    # the shrinkage limit, and reports 0 sample games. That reading is only true
    # where the report says nobody is Questionable.
    prefix = "TOP2_QUESTIONABLE_AVAILABILITY_EFFECT"
    df = pd.DataFrame(
        {
            f"{COVERED_COL}_TEAM_HOME": [1, 0, 1],
            f"{COVERED_COL}_TEAM_AWAY": [1, 1, 0],
            f"{prefix}_HOME_MEAN_TOTAL_POINTS": [1.5, 0.0, 0.0],
            f"{prefix}_HOME_SUM_N_TOTAL_GAMES": [40, 0, 0],
            f"{prefix}_AWAY_MEAN_TOTAL_POINTS": [-2.0, 0.0, 0.0],
            f"{prefix}_AWAY_SUM_N_TOTAL_GAMES": [12, 0, 0],
            "OTHER_COLUMN": [1.0, 2.0, 3.0],
        }
    )
    out = mask_uncovered_group_columns(df.copy(), prefix)

    # Covered on both sides: untouched.
    assert out.loc[0, f"{prefix}_HOME_MEAN_TOTAL_POINTS"] == 1.5
    assert out.loc[0, f"{prefix}_AWAY_SUM_N_TOTAL_GAMES"] == 12
    # Uncovered home only: the home block goes, the away block stays.
    assert pd.isna(out.loc[1, f"{prefix}_HOME_MEAN_TOTAL_POINTS"])
    assert pd.isna(out.loc[1, f"{prefix}_HOME_SUM_N_TOTAL_GAMES"])
    assert out.loc[1, f"{prefix}_AWAY_MEAN_TOTAL_POINTS"] == 0.0
    # Uncovered away only: the mirror image.
    assert pd.isna(out.loc[2, f"{prefix}_AWAY_MEAN_TOTAL_POINTS"])
    assert out.loc[2, f"{prefix}_HOME_MEAN_TOTAL_POINTS"] == 0.0
    # Nothing outside the block is touched.
    pd.testing.assert_series_equal(out["OTHER_COLUMN"], df["OTHER_COLUMN"])


def test_the_injured_effect_block_is_never_masked():
    # It predates the report and the schema 2_3 control must reproduce it.
    injured = "TOP3_INJURED_AVAILABILITY_EFFECT"
    df = pd.DataFrame(
        {
            f"{COVERED_COL}_TEAM_HOME": [0],
            f"{COVERED_COL}_TEAM_AWAY": [0],
            f"{injured}_HOME_MEAN_TOTAL_POINTS": [0.0],
        }
    )
    out = mask_uncovered_group_columns(
        df.copy(), "TOP2_QUESTIONABLE_AVAILABILITY_EFFECT"
    )
    assert out.loc[0, f"{injured}_HOME_MEAN_TOTAL_POINTS"] == 0.0
