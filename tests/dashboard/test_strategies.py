"""Best-slot selection, its selection reference and mean-of-predictions combinations."""

import math

import pandas as pd
import pytest
from nba_ou.dashboard.domain.analysis import prepare_analysis
from nba_ou.dashboard.domain.strategies import (
    candidate_rows,
    combine_predictions,
    compare_with_best,
    cumulative_win_rate,
    rank_candidates,
    selection_chance,
    sign_test,
    strategy_members,
    window_rows,
)
from nba_ou.dashboard.ui.analysis_charts import efficacy_chart, strategy_chart

from .test_historical import analysis_row

REPLAY = "Snapshot replay"


def replay_row(game_id, horizon, prediction_value, **changes):
    # Target 2 predicts the total; the stored bet line is 230.5 and the game
    # finishes 235, so a prediction above 230.5 wins and one below loses.
    changes.setdefault("bucket", horizon)
    return analysis_row(
        game_id=game_id,
        horizon=horizon,
        source=REPLAY,
        prediction_value=prediction_value,
        **changes,
    )


def test_windows_split_at_six_hours_with_six_hours_early():
    frame = pd.DataFrame({"model_horizon_minutes": [30, 300, 360, 720]})
    assert window_rows(frame, "≥ 6h")["model_horizon_minutes"].tolist() == [360, 720]
    assert window_rows(frame, "< 6h")["model_horizon_minutes"].tolist() == [30, 300]
    assert len(window_rows(frame, "All horizons")) == 4
    with pytest.raises(ValueError):
        window_rows(frame, "≥ 5h")


def test_candidates_are_own_snapshots_and_closing_final_ranked_by_win_rate():
    rows = []
    for game in range(40):
        rows.append(replay_row(game, 360, 240))  # Always OVER: 40-0.
        rows.append(replay_row(game, 480, 240 if game % 2 else 220))  # 20-20.
        rows.append(replay_row(game, 480, 240, bucket=360))  # Off-diagonal.
        rows.append(replay_row(game, 0, 240, bucket=0))  # Closing Final.
        rows.append(replay_row(game, 0, 240, bucket=240))  # Closing at T−4h.
    rows.append(replay_row(99, 120, 240))  # One bet: too small to rank.
    graded = prepare_analysis(pd.DataFrame(rows))
    candidates = candidate_rows(graded, REPLAY)
    assert set(candidates["candidate"]) == {
        "XGBoost · 2.5 · T−6h",
        "XGBoost · 2.5 · T−8h",
        "XGBoost · 2.5 · Closing Final",
        "XGBoost · 2.5 · T−2h",
    }
    assert candidate_rows(graded, "Live predictions").empty
    ranking = rank_candidates(candidates)
    assert ranking["eligible"].tolist() == [True, True, True, False]
    assert ranking.loc[0, "win_pct"] == 100
    assert ranking.loc[2, "candidate"] == "XGBoost · 2.5 · T−8h"
    assert ranking.loc[3, "candidate"] == "XGBoost · 2.5 · T−2h"


def test_selection_chance_grows_with_the_number_of_cells():
    one = selection_chance([100], 60)
    many = selection_chance([100] * 16, 60)
    assert 0.01 < one < 0.05
    assert many > 3 * one
    assert selection_chance([100] * 16, 0) == 1
    assert math.isnan(selection_chance([], 60))
    assert math.isnan(selection_chance([0], 60))


def test_sign_test_is_exact_and_two_sided():
    assert sign_test(0, 0) != sign_test(0, 0)  # NaN
    assert sign_test(5, 5) == 1
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    assert sign_test(0, 10) == sign_test(10, 0)


def test_combination_averages_edges_and_bets_at_the_latest_members_line():
    rows = pd.DataFrame(
        [
            # Edge +9.5 at T−8h against 230.5.
            replay_row(1, 480, 240),
            # Edge −1.5 at T−30m against a moved line of 236.5: the mean
            # +4 is bet OVER 236.5, which loses at 235.
            replay_row(1, 30, 235, bet_line_x2=473, bet_price_over_home=120),
            # A member without a line cannot vote or decide the bet.
            replay_row(1, 60, 200, bet_line_x2=None),
        ]
    )
    combined = combine_predictions(prepare_analysis(rows))
    assert len(combined) == 1
    row = combined.iloc[0]
    assert row["members"] == 2
    assert row["model_horizon_minutes"] == 30
    assert row["reference_line"] == 236.5
    assert row["comparison_edge"] == pytest.approx(4)
    assert row["signal"] == "OVER"
    assert row["result"] == "LOSS"
    assert row["bet_price"] == 120
    assert row["absolute_error"] == pytest.approx(5.5)


def test_combination_respects_minimum_edge_of_the_betting_member():
    rows = pd.DataFrame([replay_row(1, 480, 231), replay_row(1, 30, 230)])
    combined = combine_predictions(prepare_analysis(rows), min_edge={1: 0.5})
    assert combined.iloc[0]["result"] == "NO BET"  # Mean edge 0.0 ≤ 0.5.
    assert combine_predictions(pd.DataFrame()).empty


def test_strategy_family_is_fixed_and_adds_cross_target_only_when_given():
    own = prepare_analysis(
        pd.DataFrame(
            [replay_row(1, h, 240) for h in (60, 360)]
            + [replay_row(1, 360, 240, kind=3)]
        )
    )
    labels = list(strategy_members(own))
    assert labels == [
        "XGBoost · 2.5 · mean all horizons",
        "TabPFN · 2.5 · mean all horizons",
        "All models · mean all horizons",
        "XGBoost · 2.5 · mean ≥ 6h",
        "TabPFN · 2.5 · mean ≥ 6h",
        "All models · mean ≥ 6h",
        "XGBoost · 2.5 · mean < 6h",
        "All models · mean < 6h",
    ]
    partner = own.assign(target=1)
    crossed = strategy_members(own.loc[own["kind"].eq(1)], partner)
    assert "XGBoost · 2.5 · line error + total points · all horizons" in crossed
    assert len(crossed["XGBoost · 2.5 · line error + total points · ≥ 6h"]) == 2
    assert strategy_members(own.iloc[:0]) == {}


def test_comparison_with_best_is_paired_on_games_both_bet():
    best = prepare_analysis(
        pd.DataFrame([replay_row(g, 360, 240 if g < 3 else 220) for g in range(4)])
    )
    other = prepare_analysis(
        pd.DataFrame([replay_row(g, 480, 220 if g < 2 else 240) for g in range(5)])
    )
    table = compare_with_best({"best": best, "other": other}, best).set_index(
        "strategy"
    )
    assert table.loc["best", "matched_delta_pp"] == 0
    assert table.loc["other", "matched_games"] == 4
    # Game 3: only other won; games 0 and 1: only best won.
    assert table.loc["other", "only_strategy_won"] == 1
    assert table.loc["other", "only_best_won"] == 2
    assert table.loc["other", "matched_delta_pp"] == -25
    assert table.loc["other", "wins"] == 3


def test_cumulative_win_rate_runs_by_slate_date_with_coin_flip_band():
    frame = prepare_analysis(
        pd.DataFrame(
            [replay_row(1, 360, 240), replay_row(2, 360, 220), replay_row(3, 360, 240)]
        )
    )
    dates = pd.Series(
        pd.to_datetime(["2026-10-20", "2026-10-20", "2026-10-21"]), index=[1, 2, 3]
    )
    curve = cumulative_win_rate(frame, dates, "best")
    assert curve["bets"].tolist() == [2, 3]
    assert curve["win_pct"].round(1).tolist() == [50.0, 66.7]
    assert (curve["flip_low"] < 50).all() and (curve["flip_high"] > 50).all()
    assert cumulative_win_rate(frame.iloc[:0], dates, "best").empty
    chart = efficacy_chart(curve.assign(role="Best slot")).to_dict()
    assert chart["layer"]


def test_strategy_chart_serializes_roles():
    table = pd.DataFrame(
        {
            "strategy": ["best", "mean"],
            "role": ["Best single slot", "Combination"],
            "wins": [30, 20],
            "losses": [10, 20],
            "pushes": [0, 0],
            "win_pct": [75.0, 50.0],
            "ci_low": [60.0, 35.0],
            "ci_high": [86.0, 65.0],
            "roi": [40.0, -5.0],
            "mae": [15.0, 14.0],
            "matched_games": [40, 38],
            "matched_delta_pp": [0.0, -20.0],
            "low_sample": [False, False],
        }
    )
    spec = strategy_chart(table).to_dict()
    assert spec["layer"]
