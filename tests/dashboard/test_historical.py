"""Historical bucketing, quote grading, pooled counts and paired comparisons."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from nba_ou.dashboard.domain.analysis import (
    BREAK_EVEN,
    EARLY_BUCKET,
    TIME_GRID,
    aggregate_cells,
    bucket_label,
    paired_snapshot_summary,
    prepare_analysis,
    season_start,
    time_bucket,
    wilson_interval,
)
from nba_ou.dashboard.domain.comparisons import signal
from nba_ou.dashboard.domain.history import grade_history, select_history
from nba_ou.dashboard.ui.analysis_charts import (
    closing_timeline,
    intermediate_replay_timeline,
    live_heatmap,
    snapshot_comparison,
)

from .conftest import NOW, make_prediction
from .test_history import historical_slate


def analysis_row(*, game_id=1, horizon=360, bucket=None, **changes):
    row = make_prediction(
        game_id=game_id,
        horizon=horizon,
        status=1,
        target=changes.get("target", 2),
        kind=changes.get("kind", 1),
    )
    row.update(
        bucket=horizon if bucket is None else bucket,
        home_points=121,
        away_points=114,
        source="Live predictions",
        bet_price_over_home=-110,
        bet_price_under_away=105,
        input_at=NOW - pd.Timedelta(hours=4),
        available_at=NOW - pd.Timedelta(hours=4) + pd.Timedelta(minutes=20),
    )
    row.update(changes)
    return row


@pytest.mark.parametrize(
    "minutes,bucket",
    [
        (-1, None),
        (0, None),
        (float("nan"), None),
        (float("inf"), None),
        (0.1, 30),
        (30, 30),
        (30.01, 60),
        (60, 60),
        (60.1, 120),
        (359.5, 360),
        (360, 360),
        (360.1, 420),
        (1080, 1080),
        (1080.1, EARLY_BUCKET),
    ],
)
def test_input_time_buckets_cover_right_after_each_snapshot(minutes, bucket):
    assert time_bucket(minutes) == bucket


def test_grid_matches_serving_contract_and_season_boundary():
    from nba_ou.data_processing.line_history.snapshots import DEFAULT_SNAPSHOT_GRID

    assert TIME_GRID == tuple(h for h in DEFAULT_SNAPSHOT_GRID if h > 0)
    assert bucket_label(0) == "Final"
    assert bucket_label(EARLY_BUCKET) == "T−18h+"
    assert bucket_label(30) == "T−30m"
    assert season_start(date(2026, 9, 30)) == date(2025, 10, 1)
    assert season_start(date(2026, 10, 1)) == date(2026, 10, 1)
    assert BREAK_EVEN == pytest.approx(52.38095238)


def test_wilson_intervals_and_empty_samples():
    low, high = wilson_interval(55, 45)
    assert low == pytest.approx(45.2446, abs=0.001)
    assert high == pytest.approx(64.3855, abs=0.001)
    assert all(math.isnan(value) for value in wilson_interval(0, 0))
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 0)[1] == pytest.approx(100)


def test_grading_uses_original_anchor_and_stored_quote_for_all_targets():
    frame = pd.DataFrame(
        [
            analysis_row(target=1, model_line=230, prediction_value=8, bet_line_x2=480),
            analysis_row(target=2, prediction_value=238, bet_line_x2=480),
            analysis_row(target=3, model_line=3.5, prediction_value=2, bet_line_x2=14),
        ]
    )
    graded = prepare_analysis(frame)
    assert list(graded["predicted_value"]) == [238, 238, 5.5]
    assert list(graded["signal"]) == ["UNDER", "UNDER", "AWAY"]
    assert list(graded["result"]) == ["WIN", "WIN", "PUSH"]
    assert list(graded["absolute_error"]) == [3, 3, 1.5]
    assert graded["publication_lag_minutes"].eq(20).all()


def test_roi_uses_directional_recorded_american_prices_never_assumes_odds():
    rows = [
        analysis_row(game_id=1, prediction_value=240, bet_line_x2=460),  # OVER win -110
        analysis_row(
            game_id=2, prediction_value=229, bet_line_x2=460
        ),  # UNDER loss +105
        analysis_row(game_id=3, prediction_value=240, bet_line_x2=470),  # PUSH
        analysis_row(game_id=4, prediction_value=240, bet_price_over_home=None),
        analysis_row(game_id=5, prediction_value=240, bet_price_over_home=50),
        analysis_row(game_id=6, prediction_value=240, bet_price_over_home=120),
        analysis_row(
            game_id=7, prediction_value=230, bet_line_x2=480, bet_price_under_away=105
        ),
    ]
    graded = prepare_analysis(pd.DataFrame(rows))
    assert list(graded["result"]) == ["WIN", "LOSS", "PUSH", "WIN", "WIN", "WIN", "WIN"]
    assert list(graded["priced_bet"]) == [True, True, True, False, False, True, True]
    assert graded["unit_profit"].tolist()[:3] == pytest.approx([100 / 110, -1, 0])
    summary = aggregate_cells(graded).iloc[0]
    assert summary["priced_bets"] == 5
    assert summary["roi"] == pytest.approx(100 * (100 / 110 - 1 + 1.2 + 1.05) / 5)
    graded = prepare_analysis(
        pd.DataFrame(
            [analysis_row(bet_price_over_home=None, bet_price_under_away=None)]
        )
    )
    assert math.isnan(aggregate_cells(graded).iloc[0]["roi"])


def test_pooled_counts_use_prediction_rows_and_distinct_games_and_missing_coverage():
    rows = [
        analysis_row(
            game_id=game, horizon=horizon, prediction_value=240 if game == 1 else 220
        )
        for game in (1, 2)
        for horizon in (360, 480)
    ]
    # A no-bet prediction still contributes to MAE, but not the win denominator.
    rows.append(analysis_row(game_id=3, horizon=360, prediction_value=230.5))
    rows += [
        analysis_row(game_id=4, status=0, home_points=None),
        analysis_row(game_id=5, status=2),
    ]
    graded = prepare_analysis(pd.DataFrame(rows))
    pooled = aggregate_cells(graded, by=[], settled_games_total=4).iloc[0]
    assert pooled["rows"] == 7
    assert pooled["games"] == 5
    assert pooled["settled_games"] == 3
    assert pooled["wins"] == 2 and pooled["losses"] == 2
    assert pooled["win_pct"] == 50
    assert pooled["coverage"] == 50  # 4 bets / (4 final games × 2 horizons)
    assert pooled["mae"] == pytest.approx((5 + 15 + 5 + 15 + 4.5) / 5)
    assert pooled["pending"] == 1 and pooled["void"] == 1
    assert aggregate_cells(graded)["low_sample"].all()


def test_live_replay_delta_uses_only_matched_settled_bets_and_slots():
    rows = [
        analysis_row(game_id=1, prediction_value=240),
        analysis_row(game_id=1, source="Snapshot replay", prediction_value=220),
        analysis_row(game_id=2, prediction_value=240),  # unmatched
        analysis_row(
            game_id=3, source="Snapshot replay", prediction_value=240
        ),  # unmatched
        analysis_row(game_id=4, prediction_value=230.5),  # NO BET cannot be paired
        analysis_row(game_id=4, source="Snapshot replay", prediction_value=240),
        analysis_row(
            game_id=5, horizon=480, bucket=360, source="Snapshot replay"
        ),  # off-diagonal
    ]
    summary = paired_snapshot_summary(prepare_analysis(pd.DataFrame(rows)))
    assert len(summary) == 1
    assert summary.iloc[0]["Matched games"] == 1
    assert summary.iloc[0]["Live win %"] == 100
    assert summary.iloc[0]["Replay win %"] == 0
    assert summary.iloc[0]["Replay − live (pp)"] == -100
    assert summary.iloc[0]["Different picks"] == 1


def test_vectorized_grading_preserves_signals_at_threshold_boundaries():
    edges = [
        None,
        np.nan,
        np.inf,
        -np.inf,
        0,
        0.1,
        -0.1,
        0.1 + 1e-10,
        -0.1 - 1e-10,
        0.11,
        -0.11,
    ]
    rows = [
        analysis_row(
            target=target,
            model_line=0,
            prediction_value=edge,
            bet_line_x2=0,
            available=True,
        )
        for target in (1, 2, 3)
        for edge in edges
    ]
    frame = pd.DataFrame(rows)
    graded = grade_history(frame)
    assert list(graded["signal"]) == [
        signal(edge, spread=target == 3) for target in (1, 2, 3) for edge in edges
    ]
    for threshold in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            grade_history(frame, min_edge={1: threshold})


@pytest.mark.parametrize("source", ["Live predictions", "Snapshot replay"])
def test_final_and_replay_diagonal_match_yesterday_grading(source):
    data = historical_slate()
    selected = select_history(data.predictions, data.games, 2, source, NOW)
    expected = grade_history(selected)
    reduced = selected.loc[selected["available"]].copy()
    reduced["bucket"] = reduced["checkpoint_horizon_minutes"].where(
        reduced["dataset_type"].eq(2), 0
    )
    reduced = reduced.loc[
        reduced["dataset_type"].eq(2) | reduced["checkpoint_horizon_minutes"].eq(0)
    ]
    actual = prepare_analysis(reduced)
    columns = ["game_id", "family_key", "checkpoint", "result", "absolute_error"]
    expected = (
        expected.loc[
            expected["available"] & expected["checkpoint"].isin(actual["checkpoint"])
        ]
        .sort_values(columns[:3])
        .reset_index(drop=True)
    )
    actual = actual.sort_values(columns[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[columns], expected[columns])


@pytest.mark.parametrize("metric", ["Win %", "Bets", "MAE", "ROI"])
def test_charts_have_valid_specs_and_do_not_send_raw_prediction_rows(metric):
    rows = prepare_analysis(
        pd.DataFrame(
            [
                analysis_row(horizon=0, bucket=0),
                analysis_row(horizon=0, bucket=360),
                analysis_row(horizon=360),
                analysis_row(horizon=360, source="Snapshot replay"),
            ]
        )
    )
    cells = aggregate_cells(rows)
    charts = (
        closing_timeline(cells.loc[cells["dataset_type"].eq(1)], metric),
        intermediate_replay_timeline(cells, metric),
        live_heatmap(cells.loc[cells["source"].eq("Live predictions")], metric),
        snapshot_comparison(cells.loc[cells["dataset_type"].eq(2)]),
    )
    for chart in charts:
        spec = chart.to_dict(validate=True)
        assert spec
        assert "prediction_value" not in chart.to_json()


def test_intermediate_timeline_has_one_replay_point_per_family_and_own_horizon():
    rows = prepare_analysis(
        pd.DataFrame(
            [
                analysis_row(game_id=1, horizon=360, source="Snapshot replay"),
                analysis_row(game_id=2, horizon=360, source="Snapshot replay"),
                analysis_row(game_id=1, horizon=480, source="Snapshot replay"),
                analysis_row(game_id=1, horizon=480, kind=3, source="Snapshot replay"),
                analysis_row(game_id=1, horizon=360),  # live must not enter the chart
                analysis_row(game_id=1, horizon=0, bucket=0),  # no closing/Final point
                analysis_row(
                    game_id=1, horizon=480, bucket=240, source="Snapshot replay"
                ),
            ]
        )
    )
    spec = intermediate_replay_timeline(aggregate_cells(rows), "Win %").to_dict(
        validate=True
    )
    data = next(
        dataset
        for dataset in spec["datasets"].values()
        if dataset and "bucket" in dataset[0]
    )
    assert len(data) == 3
    assert {record["source"] for record in data} == {"Snapshot replay"}
    assert {record["bucket"] for record in data} == {360, 480}
    assert all(record["bucket"] == record["model_horizon_minutes"] for record in data)
    assert next(record["rows"] for record in data if record["bucket"] == 360) == 2
    assert len({(record["family_key"], record["bucket"]) for record in data}) == 3
    assert spec["layer"][2]["mark"]["type"] == "line"
    assert spec["layer"][2]["mark"]["point"] is True
