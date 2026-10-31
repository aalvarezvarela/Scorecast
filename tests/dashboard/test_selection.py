import pandas as pd
import pytest
from nba_ou.dashboard.domain.selection import (
    live_predictions,
    market_snapshots,
    select_predictions,
    slate_games,
)

from .conftest import NOW, make_prediction


def test_each_game_gets_its_own_nearest_horizon(today_data):
    selected = select_predictions(today_data.predictions, 2, NOW)
    intermediate = selected.loc[selected["phase"].eq("Intermediate")].set_index(
        "game_id"
    )
    assert intermediate["model_horizon_minutes"].to_dict() == {1: 240, 2: 120}
    assert len(selected.loc[selected["phase"].eq("Closing")]) == 2


def test_selection_tracks_current_time_even_without_reloading_data(today_data):
    selected = select_predictions(
        today_data.predictions, 2, NOW + pd.Timedelta(hours=1)
    )
    intermediate = selected.loc[selected["phase"].eq("Intermediate")].set_index(
        "game_id"
    )
    # Three hours is equally far from T-4h and T-2h: prefer the longer horizon.
    assert intermediate.loc[1, "model_horizon_minutes"] == 240
    assert intermediate.loc[2, "model_horizon_minutes"] == 120


def test_trained_horizon_is_not_confused_with_latest_input_horizon(today_data):
    frame = today_data.predictions.copy()
    frame["snapshot_horizon_minutes"] = 47
    selected = select_predictions(frame, 2, NOW)
    assert selected.loc[
        selected["phase"].eq("Intermediate"), "model_horizon_minutes"
    ].tolist() == [120, 240]


def test_newest_output_wins_across_refits_and_closing(today_data):
    newer = make_prediction(
        horizon=0,
        fit_id="new-fit",
        prediction_value=235.0,
        predicted_at=NOW - pd.Timedelta(minutes=2),
        run_finished_at=NOW - pd.Timedelta(minutes=1),
    )
    newer_frame = pd.DataFrame([newer]).astype(today_data.predictions.dtypes)
    selected = select_predictions(
        pd.concat([today_data.predictions, newer_frame]), 2, NOW
    )
    row = selected.loc[
        selected["game_id"].eq(1) & selected["phase"].eq("Closing")
    ].iloc[0]
    assert row["fit_id"] == "new-fit"
    assert row["prediction_value"] == 235


@pytest.mark.parametrize(
    "changes",
    [
        {"workflow": 2},
        {"kind": 2},
        {"status": 2},
        {"tipoff_at_prediction": NOW - pd.Timedelta(minutes=15)},
        {"predicted_at": NOW + pd.Timedelta(seconds=1)},
        {"run_finished_at": NOW + pd.Timedelta(seconds=1)},
        {"time_to_tipoff_seconds": -1},
        {"time_to_tipoff_seconds": 0},
        {"input_mode": 0},
        {"target": 99},
    ],
)
def test_non_live_or_ineligible_outputs_are_excluded(changes):
    assert live_predictions(pd.DataFrame([make_prediction(**changes)]), NOW).empty


def test_tabpfn_own_inputs_from_live_workflow_are_allowed():
    selected = select_predictions(
        pd.DataFrame(
            [
                make_prediction(horizon=120),
                make_prediction(horizon=240, kind=3, input_mode=1),
                make_prediction(horizon=360, kind=3, input_mode=1),
            ]
        ),
        2,
        NOW,
    )
    assert selected.set_index("kind")["model_horizon_minutes"].to_dict() == {
        1: 120,
        3: 240,
    }


def test_targets_and_model_versions_are_separate(today_data):
    frame = pd.concat(
        [
            today_data.predictions,
            pd.DataFrame(
                [
                    make_prediction(schema_version="2_6", horizon=240),
                ]
            ),
        ]
    )
    selected = select_predictions(frame, 2, NOW, ["1:2_6:main"])
    assert len(selected) == 1
    assert selected.iloc[0]["schema_version"] == "2_6"
    assert set(selected["target"]) == {2}


def test_market_uses_one_latest_coherent_observation_not_mixed_non_nulls():
    frame = pd.DataFrame(
        [
            make_prediction(bet_line_x2=450, books_mean_line=226.0),
            make_prediction(
                input_at=NOW - pd.Timedelta(minutes=1),
                bet_line_x2=None,
                books_mean_line=231.0,
                books_median_line=230.5,
                books_count=4,
            ),
        ]
    )
    market = market_snapshots(frame, 2, NOW).iloc[0]
    assert pd.isna(market["bet365_line"])
    assert market["books_mean_line"] == 231
    assert market["books_median_line"] == 230.5
    assert market["books_count"] == 4


def test_market_filters_book_and_future_nominal_inputs():
    frame = pd.DataFrame(
        [
            make_prediction(bet_line_x2=461),
            make_prediction(anchor_book="fanduel", input_at=NOW),
            make_prediction(kind=3, input_mode=1, input_at=NOW + pd.Timedelta(hours=1)),
        ]
    )
    assert market_snapshots(frame, 1, NOW).iloc[0]["bet365_line"] == 230.5


def test_slate_keeps_started_and_final_games_but_excludes_other_days_and_void(
    today_data,
):
    games = today_data.games.copy()
    games.loc[0, "tipoff_at"] = NOW - pd.Timedelta(hours=1)
    games.loc[1, "status"] = 1
    assert slate_games(games, NOW.date())["game_id"].tolist() == [1, 2]
    games.loc[1, "status"] = 2
    assert slate_games(games, NOW.date())["game_id"].tolist() == [1]
    games.loc[0, "game_date"] = (NOW - pd.Timedelta(days=1)).date()
    assert slate_games(games, NOW.date()).empty


@pytest.mark.parametrize("status", [0, 1])
def test_predictions_and_pregame_markets_survive_tipoff_and_finish(today_data, status):
    frame = today_data.predictions.copy()
    frame["status"] = status
    after_tipoff = NOW + pd.Timedelta(hours=7)
    selected = select_predictions(frame, (1, 2), after_tipoff)
    assert len(selected) == 8
    intermediate = selected.loc[selected["phase"].eq("Intermediate")]
    assert set(intermediate["model_horizon_minutes"]) == {120}
    assert set(selected["minutes_remaining"]) == {0}
    assert set(market_snapshots(frame, 1, after_tipoff)["game_id"]) == {1, 2}


def test_started_games_do_not_use_post_tip_market_observations():
    frame = pd.DataFrame(
        [
            make_prediction(),
            make_prediction(input_at=NOW + pd.Timedelta(hours=5), bet_line_x2=500),
        ]
    )
    market = market_snapshots(frame, 2, NOW + pd.Timedelta(hours=7)).iloc[0]
    assert market["bet365_line"] == 230.5


def test_slate_uses_game_date_across_madrid_midnight(today_data):
    games = today_data.games.copy()
    # 00:30 in Madrid on the 29th still belongs to the NBA slate of the 28th.
    games.loc[0, "tipoff_at"] = pd.Timestamp("2026-10-28T23:30:00Z")
    assert set(slate_games(games, NOW.date())["game_id"]) == {1, 2}


def test_missing_target_and_empty_model_selection_are_empty(today_data):
    assert select_predictions(today_data.predictions, 99, NOW).empty
    assert select_predictions(today_data.predictions, 2, NOW, []).empty


def test_combined_totals_keep_closing_and_intermediate_for_each_target(today_data):
    selected = select_predictions(today_data.predictions, (1, 2), NOW)
    assert len(selected) == 8
    assert selected.groupby(["game_id", "target"]).size().to_dict() == {
        (1, 1): 2,
        (1, 2): 2,
        (2, 1): 2,
        (2, 2): 2,
    }
    intermediate = selected.loc[selected["phase"].eq("Intermediate")]
    assert intermediate.set_index(["game_id", "target"])[
        "model_horizon_minutes"
    ].to_dict() == {
        (1, 1): 240,
        (1, 2): 240,
        (2, 1): 120,
        (2, 2): 120,
    }


def test_targets_choose_independently_when_available_horizons_differ():
    raw = pd.DataFrame(
        [
            make_prediction(target=1, horizon=240),
            make_prediction(target=1, horizon=360),
            make_prediction(target=2, horizon=120),
            make_prediction(target=2, horizon=300),
        ]
    )
    selected = select_predictions(raw, (1, 2), NOW)
    assert selected.set_index("target")["model_horizon_minutes"].to_dict() == {
        1: 240,
        2: 300,
    }
