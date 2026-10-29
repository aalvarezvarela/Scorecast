"""Time-safe checkpoint selection, source isolation and per-quote grading."""

import pandas as pd
import pytest
from nba_ou.dashboard.domain.history import (
    grade_history,
    historical_games,
    history_summary,
    select_history,
)
from nba_ou.dashboard.repositories.predictions import TodayData

from .conftest import NOW, make_prediction


def historical_slate():
    games = []
    predictions = []
    for game_id in (1, 2):
        tipoff = NOW - pd.Timedelta(hours=16 if game_id == 1 else 14)
        games.append(
            dict(
                game_id=game_id,
                game_date=(NOW - pd.Timedelta(days=1)).date(),
                tipoff_at=tipoff,
                home_team="BOS",
                away_team="NYK",
                home_team_id=1610612738,
                away_team_id=1610612752,
                home_points=121,
                away_points=114,
                status=1,
            )
        )
        for kind in (1, 3):
            for target in (1, 2, 3):
                for horizon in (480, 360, 240, 0):
                    published = tipoff - pd.Timedelta(
                        minutes=horizon + (5 if horizon else 10)
                    )
                    row = make_prediction(
                        game_id=game_id,
                        horizon=0,
                        kind=kind,
                        target=target,
                        tipoff_at=tipoff,
                        tipoff_at_prediction=tipoff,
                        status=1,
                        predicted_at=published - pd.Timedelta(minutes=2),
                        run_finished_at=published,
                        input_at=published - pd.Timedelta(minutes=4),
                        time_to_tipoff_seconds=int((tipoff - published).total_seconds())
                        + 120,
                        bet_line_x2=480
                        if horizon in (480, 360)
                        else 460
                        if horizon == 240
                        else 470,
                        model_line=230 if target != 3 else 3.5,
                        prediction_value={1: 8, 2: 238, 3: 2}[target],
                        run_id=1000 + kind * 100 + target * 10 + horizon,
                    )
                    predictions.append(row)
                    if kind == 1:
                        replay = row.copy()
                        replay.update(
                            workflow=2,
                            input_mode=3,
                            snapshot_horizon_minutes=horizon,
                            input_at=tipoff - pd.Timedelta(minutes=horizon),
                            predicted_at=tipoff + pd.Timedelta(hours=2),
                            run_finished_at=tipoff + pd.Timedelta(hours=2, minutes=5),
                            time_to_tipoff_seconds=-7200,
                            prediction_value={1: 18, 2: 248, 3: -2}[target],
                            run_id=row["run_id"] + 10000,
                        )
                        predictions.append(replay)
                for horizon in (480, 360, 240):
                    cutoff = tipoff - pd.Timedelta(minutes=horizon)
                    row = make_prediction(
                        game_id=game_id,
                        horizon=horizon,
                        kind=kind,
                        target=target,
                        tipoff_at=tipoff,
                        tipoff_at_prediction=tipoff,
                        status=1,
                        predicted_at=cutoff + pd.Timedelta(minutes=2),
                        run_finished_at=cutoff + pd.Timedelta(minutes=5),
                        input_at=cutoff,
                        input_mode=1 if kind == 3 else 2,
                        snapshot_horizon_minutes=horizon,
                        time_to_tipoff_seconds=horizon * 60 - 120,
                        model_line=230 if target != 3 else 3.5,
                        bet_line_x2=480 if target != 3 else 7,
                        prediction_value={1: 8, 2: 238, 3: 2}[target],
                        run_id=2000 + kind * 100 + target * 10 + horizon,
                    )
                    predictions.append(row)
                    replay = row.copy()
                    replay.update(
                        workflow=2 if kind == 1 else 3,
                        input_mode=1,
                        predicted_at=tipoff + pd.Timedelta(hours=2)
                        if kind == 1
                        else cutoff + pd.Timedelta(minutes=60),
                        run_finished_at=tipoff + pd.Timedelta(hours=2, minutes=5)
                        if kind == 1
                        else cutoff + pd.Timedelta(minutes=65),
                        time_to_tipoff_seconds=-7200
                        if kind == 1
                        else horizon * 60 - 3600,
                        prediction_value={1: 18, 2: 248, 3: -2}[target],
                        run_id=row["run_id"] + 10000,
                    )
                    predictions.append(replay)
    return TodayData(
        pd.DataFrame(games), pd.DataFrame(predictions), NOW.to_pydatetime()
    )


def selected(data, source="Live predictions", target=2):
    return select_history(
        data.predictions, historical_games(data.games), target, source, NOW
    )


def test_fixed_checkpoints_keep_live_and_replay_separate_with_reconstructed_closing():
    data = historical_slate()
    live = selected(data)
    replay = selected(data, "Snapshot replay")
    assert len(live) == len(replay) == 28
    assert live["available"].all()
    assert replay.loc[
        replay["kind"].eq(1) | replay["dataset_type"].eq(2), "available"
    ].all()
    assert set(live["checkpoint"]) == {
        "Closing · T−8h",
        "Closing · T−6h",
        "Closing · T−4h",
        "Closing · Final pre-game",
        "Intermediate · T−8h",
        "Intermediate · T−6h",
        "Intermediate · T−4h",
    }
    assert set(live["prediction_value"]) == {238}
    assert set(replay.loc[replay["dataset_type"].eq(2), "prediction_value"]) == {248}
    assert set(
        replay.loc[
            replay["kind"].eq(1) & replay["dataset_type"].eq(1), "prediction_value"
        ]
    ) == {248}
    assert live.loc[live["kind"].eq(1), "workflow"].eq(1).all()
    assert (
        replay.loc[replay["kind"].eq(1) & replay["dataset_type"].eq(2), "workflow"]
        .eq(2)
        .all()
    )


def test_closing_selects_nearest_input_with_symmetric_one_hour_margin():
    data = historical_slate()
    cutoff = data.games.iloc[0]["tipoff_at"] - pd.Timedelta(hours=6)
    mask = (
        data.predictions.game_id.eq(1)
        & data.predictions.kind.eq(1)
        & data.predictions.target.eq(2)
        & data.predictions.dataset_type.eq(1)
        & data.predictions.workflow.eq(1)
    )
    index = data.predictions.loc[
        mask
        & data.predictions.input_at.between(cutoff - pd.Timedelta(minutes=30), cutoff)
    ].index[0]

    def slot():
        rows = selected(data)
        return rows.loc[
            rows.game_id.eq(1) & rows.kind.eq(1) & rows.checkpoint.eq("Closing · T−6h")
        ].iloc[0]

    # An output after the checkpoint is allowed, with the real time disclosed.
    for column in ("input_at", "predicted_at", "run_finished_at"):
        data.predictions.loc[index, column] = cutoff + pd.Timedelta(minutes=23)
    assert slot().available
    # Boundary included; more than an hour on either side is unavailable.
    for offset, available in [(60, True), (61, False), (-60, True), (-61, False)]:
        for column in ("input_at", "predicted_at", "run_finished_at"):
            data.predictions.loc[index, column] = cutoff + pd.Timedelta(minutes=offset)
        assert bool(slot().available) == available
    # Choose the closer input, not the last prediction in the window.
    original = data.predictions.loc[index].copy()
    for column in ("input_at", "predicted_at", "run_finished_at"):
        data.predictions.loc[index, column] = cutoff - pd.Timedelta(minutes=10)
        original[column] = cutoff + pd.Timedelta(minutes=45)
    original["prediction_value"] = 999
    data = TodayData(
        data.games,
        pd.concat([data.predictions, pd.DataFrame([original])]),
        data.fetched_at,
    )
    assert slot().prediction_value == 238


def test_final_pre_game_excludes_runs_finishing_after_tipoff():
    data = historical_slate()
    new = data.predictions.loc[data.predictions["dataset_type"].eq(1)].iloc[0].copy()
    new["predicted_at"] = new["tipoff_at"] - pd.Timedelta(seconds=1)
    new["run_finished_at"] = new["tipoff_at"] + pd.Timedelta(seconds=1)
    new["prediction_value"] = 999
    data = TodayData(
        data.games, pd.concat([data.predictions, pd.DataFrame([new])]), data.fetched_at
    )
    assert 999 not in set(selected(data)["prediction_value"])


def test_replay_requires_exact_own_horizon_and_not_cross_timepoint_output():
    data = historical_slate()
    mask = data.predictions["workflow"].eq(2) & data.predictions[
        "model_horizon_minutes"
    ].eq(360)
    data.predictions.loc[mask, "snapshot_horizon_minutes"] = 240
    rows = selected(data, "Snapshot replay")
    assert not rows.loc[
        rows["kind"].eq(1) & rows["checkpoint"].eq("Intermediate · T−6h"), "available"
    ].any()
    # T−6h model evaluated at T−4h must not stand in for its own live checkpoint.
    mask = data.predictions["workflow"].eq(1) & data.predictions[
        "model_horizon_minutes"
    ].eq(360)
    for index in data.predictions.loc[mask].index:
        cutoff = data.predictions.loc[index, "tipoff_at"] - pd.Timedelta(hours=4)
        data.predictions.loc[index, ["input_at", "predicted_at", "run_finished_at"]] = [
            cutoff,
            cutoff,
            cutoff,
        ]
    rows = selected(data)
    assert not rows.loc[
        rows["kind"].eq(1) & rows["checkpoint"].eq("Intermediate · T−6h"), "available"
    ].any()


def test_live_intermediate_allows_processing_delay_but_never_future_inputs():
    data = historical_slate()
    mask = data.predictions["kind"].eq(3) & data.predictions[
        "model_horizon_minutes"
    ].eq(480)
    # TabPFN can emit another game's own row before its horizon passes.
    for index in data.predictions.loc[mask].index:
        data.predictions.loc[index, "predicted_at"] = data.predictions.loc[
            index, "input_at"
        ] - pd.Timedelta(seconds=1)
    live = selected(data)
    replay = selected(data, "Snapshot replay")
    for rows in (live, replay):
        assert not rows.loc[
            rows["kind"].eq(3) & rows["checkpoint"].eq("Intermediate · T−8h"),
            "available",
        ].any()
    xgb = live.loc[live["kind"].eq(1) & live["dataset_type"].eq(2)]
    assert xgb["available"].all()
    assert (xgb["available_at"] > xgb["checkpoint_at"]).all()


def test_latest_refit_wins_once_per_game_family_and_checkpoint():
    data = historical_slate()
    extra = data.predictions.loc[data.predictions["workflow"].eq(1)].copy()
    extra["model_build_id"] += 50000
    extra["run_id"] += 50000
    extra["run_finished_at"] += pd.Timedelta(seconds=1)
    extra["prediction_value"] = 239
    data = TodayData(data.games, pd.concat([data.predictions, extra]), data.fetched_at)
    rows = selected(data)
    assert len(rows) == 28
    assert rows.loc[rows["kind"].eq(1), "prediction_value"].eq(239).all()


def test_grading_uses_each_rows_line_and_pushes_are_not_losses():
    rows = grade_history(selected(historical_slate()))
    closing = rows.loc[
        rows["kind"].eq(1) & rows["game_id"].eq(1) & rows["dataset_type"].eq(1)
    ]
    assert list(closing["reference_line"]) == [240, 240, 230, 235]
    assert list(closing["signal"]) == ["UNDER", "UNDER", "OVER", "OVER"]
    assert list(closing["result"]) == ["WIN", "WIN", "WIN", "PUSH"]
    assert rows["absolute_error"].eq(3).all()


def test_line_error_converts_to_total_using_original_anchor():
    rows = grade_history(selected(historical_slate(), target=1))
    assert rows["predicted_total"].eq(238).all()
    assert rows["predicted_error"].eq(8).all()


def test_spread_grades_implied_home_margin_not_original_error_alone():
    rows = selected(historical_slate(), target=3)
    rows["bet_line_x2"] = 12
    rows["prediction_value"] = 2
    rows["model_line"] = 3.5
    graded = grade_history(rows)
    assert graded["predicted_home_margin"].eq(5.5).all()
    assert graded["comparison_edge"].eq(-0.5).all()
    assert graded["signal"].eq("AWAY").all()
    assert graded["result"].eq("LOSS").all()  # Actual home margin is 7.


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing", "MISSING"),
        ("no_line", "NO LINE"),
        ("no_bet", "NO BET"),
        ("pending", "PENDING"),
        ("void", "VOID"),
        ("incomplete_score", "PENDING"),
    ],
)
def test_non_bets_and_unsettled_games_are_never_graded_as_losses(case, expected):
    rows = selected(historical_slate()).iloc[[0]].copy()
    if case == "missing":
        rows["available"] = False
    elif case == "no_line":
        rows["bet_line_x2"] = float("nan")
    elif case == "no_bet":
        rows["prediction_value"] = rows["bet_line_x2"] / 2
    elif case == "pending":
        rows["status"] = 0
    elif case == "void":
        rows["status"] = 2
    elif case == "incomplete_score":
        rows["home_points"] = float("nan")
    assert grade_history(rows).iloc[0]["result"] == expected


def test_daily_summary_pools_checkpoints_per_model_and_excludes_pushes():
    rows = grade_history(selected(historical_slate()))
    summary = history_summary(rows)
    assert len(summary) == 2
    assert "Checkpoint" not in summary
    assert summary["Pushes"].eq(2).all()
    assert summary["Wins"].eq(12).all()
    assert summary["Win rate"].eq(100).all()
    assert summary["MAE (pts)"].eq(3).all()
    assert summary["Scored predictions"].eq(14).all()


def test_daily_summary_uses_pooled_counts_and_errors_with_unequal_checkpoint_coverage():
    rows = grade_history(selected(historical_slate()))
    rows = rows.loc[rows["kind"].eq(1)].iloc[[0, 1, 6]].copy()
    rows["result"] = ["WIN", "LOSS", "WIN"]
    rows["absolute_error"] = [1.0, 10.0, 1.0]
    summary = history_summary(rows).iloc[0]
    assert summary["Win rate"] == pytest.approx(200 / 3)
    assert summary["MAE (pts)"] == 4


def test_daily_summary_has_no_percentage_without_settled_picks():
    rows = grade_history(selected(historical_slate()))
    rows = rows.loc[rows["result"].eq("PUSH")]
    assert history_summary(rows)["Win rate"].isna().all()


def test_checkpoint_summary_separates_closing_times_and_counts_unique_games():
    rows = grade_history(selected(historical_slate()))
    xgb = rows.loc[rows["kind"].eq(1)].copy()
    absent = xgb["game_id"].eq(2) & xgb["checkpoint"].eq("Closing · T−4h")
    xgb.loc[absent, ["available", "absolute_error", "result"]] = [
        False,
        float("nan"),
        "MISSING",
    ]
    summary = history_summary(xgb, by_checkpoint=True)
    assert list(summary["Checkpoint"]) == [
        "Closing · T−8h",
        "Closing · T−6h",
        "Closing · T−4h",
        "Closing · Final pre-game",
        "Intermediate · T−8h",
        "Intermediate · T−6h",
        "Intermediate · T−4h",
    ]
    four = summary.loc[summary["Checkpoint"].eq("Closing · T−4h")].iloc[0]
    assert four["Games"] == four["Available"] == four["Wins"] == 1
    assert four["Expected"] == 2
    pooled = history_summary(rows)
    assert pooled["Games"].eq(2).all()
    assert pooled["Scored predictions"].eq(14).all()


def test_unknown_source_is_rejected_and_empty_slate_is_supported():
    data = historical_slate()
    with pytest.raises(ValueError):
        selected(data, "unknown")
    assert select_history(pd.DataFrame(), data.games, 2, "Live predictions", NOW).empty
    assert grade_history(pd.DataFrame()).empty
    assert history_summary(pd.DataFrame()).empty


def test_live_intermediate_selects_nearest_input_on_either_side_of_checkpoint():
    data = historical_slate()
    target = data.predictions.loc[
        data.predictions.game_id.eq(1)
        & data.predictions.kind.eq(1)
        & data.predictions.target.eq(2)
        & data.predictions.model_horizon_minutes.eq(480)
        & data.predictions.workflow.eq(1)
    ].index[0]
    cutoff = data.predictions.loc[target, "input_at"]
    data.predictions.loc[target, "input_at"] = cutoff + pd.Timedelta(minutes=23)
    data.predictions.loc[target, "predicted_at"] = cutoff + pd.Timedelta(minutes=29)
    data.predictions.loc[target, "run_finished_at"] = cutoff + pd.Timedelta(minutes=30)
    row = (
        selected(data)
        .loc[
            lambda f: (
                f.game_id.eq(1) & f.kind.eq(1) & f.checkpoint.eq("Intermediate · T−8h")
            )
        ]
        .iloc[0]
    )
    assert row.available
    assert row.input_at == cutoff + pd.Timedelta(minutes=23)


def test_replay_never_fills_missing_closing_reconstruction_with_live_output():
    data = historical_slate()
    data = TodayData(
        data.games,
        data.predictions.loc[~data.predictions.input_mode.eq(3)],
        data.fetched_at,
    )
    rows = selected(data, "Snapshot replay")
    assert not rows.loc[rows.dataset_type.eq(1), "available"].any()
    assert rows.loc[rows.dataset_type.eq(2), "available"].all()
