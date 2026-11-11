"""Best-slot selection, its selection reference and same-timepoint combinations."""

import math

import numpy as np
import pandas as pd
import pytest
from nba_ou.dashboard.domain.analysis import prepare_analysis
from nba_ou.dashboard.domain.models import displayed_models
from nba_ou.dashboard.domain.strategies import (
    MAJORITY_VOTE,
    MEAN_EDGE,
    candidate_rows,
    combination_table,
    compare_with_best,
    cumulative_win_rate,
    game_permutation_pvalue,
    profit_probability,
    rank_candidates,
    rank_combinations,
    same_timepoint_combinations,
    selection_chance,
    sign_test,
    timepoint_members,
    window_rows,
)
from nba_ou.dashboard.ui.analysis_charts import efficacy_chart

from .conftest import NOW
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


def test_candidates_are_own_snapshots_and_closing_at_each_input_ranked_by_win_rate():
    rows = []
    for game in range(40):
        rows.append(replay_row(game, 360, 240))  # Always OVER: 40-0.
        rows.append(replay_row(game, 480, 240 if game % 2 else 220))  # 20-20.
        rows.append(replay_row(game, 480, 240, bucket=360))  # Off-diagonal.
        rows.append(replay_row(game, 0, 240, bucket=0))  # Closing Final.
        rows.append(replay_row(game, 0, 220, bucket=240))  # Closing at T−4h: 0-40.
    rows.append(replay_row(99, 120, 240))  # One bet: too small to rank.
    graded = prepare_analysis(pd.DataFrame(rows))
    candidates = candidate_rows(graded, REPLAY)
    assert set(candidates["candidate"]) == {
        "XGBoost · 2.5 · Intermediate · T−6h",
        "XGBoost · 2.5 · Intermediate · T−8h",
        "XGBoost · 2.5 · Closing · Final",
        "XGBoost · 2.5 · Closing · T−4h",
        "XGBoost · 2.5 · Intermediate · T−2h",
    }
    assert candidate_rows(graded, "Live predictions").empty
    ranking = rank_candidates(candidates)
    assert ranking["eligible"].tolist() == [True, True, True, True, False]
    assert ranking.loc[0, "win_pct"] == 100
    assert ranking.loc[2, "candidate"] == "XGBoost · 2.5 · Intermediate · T−8h"
    assert ranking.loc[3, "candidate"] == "XGBoost · 2.5 · Closing · T−4h"
    assert ranking.loc[4, "candidate"] == "XGBoost · 2.5 · Intermediate · T−2h"


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


def _members(rows, partner=None):
    graded = prepare_analysis(pd.DataFrame(rows))
    other = (
        None
        if partner is None
        else candidate_rows(prepare_analysis(pd.DataFrame(partner)), REPLAY)
    )
    return timepoint_members(candidate_rows(graded, REPLAY), other)


def _strategy(combined, method, timepoint):
    part = combined.loc[
        combined["method"].eq(method) & combined["timepoint"].eq(timepoint)
    ]
    assert part["strategy"].nunique() == 1
    return part


def test_combinations_only_join_slots_of_the_same_timepoint():
    rows = [
        replay_row(1, 480, 240),  # XGBoost intermediate T−8h.
        replay_row(1, 480, 240, kind=3),  # TabPFN intermediate T−8h.
        replay_row(1, 0, 240, bucket=480),  # XGBoost closing on the T−8h input.
        replay_row(1, 360, 240),  # Alone at T−6h: never combined.
        replay_row(1, 480, 240, bucket=360),  # Off-diagonal: not a slot.
    ]
    combined = same_timepoint_combinations(_members(rows))
    assert set(combined["timepoint"]) == {480}
    # Three pairs and one triple, each as a mean and a vote.
    assert combined["strategy"].nunique() == 8
    assert sorted(combined.groupby("strategy")["members"].first()) == [2] * 6 + [3] * 2
    assert "T−8h · mean edge · TabPFN 2.5 intermediate + XGBoost 2.5 closing" in set(
        combined["strategy"]
    )
    assert (
        "T−8h · majority · TabPFN 2.5 intermediate + XGBoost 2.5 closing + "
        "XGBoost 2.5 intermediate"
    ) in set(combined["strategy"])


def test_mean_edge_bets_at_the_line_of_the_member_published_last():
    late = NOW - pd.Timedelta(hours=1)
    rows = [
        # Edge +9.5 against 230.5.
        replay_row(1, 480, 240),
        # Edge −1.5 against a moved line of 236.5, published last: the mean
        # +4 is bet OVER 236.5, which loses at 235.
        replay_row(
            1,
            480,
            235,
            kind=3,
            bet_line_x2=473,
            bet_price_over_home=120,
            available_at=late,
        ),
    ]
    row = _strategy(same_timepoint_combinations(_members(rows)), MEAN_EDGE, 480).iloc[0]
    assert row["members"] == 2
    assert row["reference_line"] == 236.5
    assert row["comparison_edge"] == pytest.approx(4)
    assert (row["signal"], row["result"], row["bet_price"]) == ("OVER", "LOSS", 120)
    assert row["absolute_error"] == pytest.approx(5.5)


def test_combinations_need_every_member_with_a_line():
    rows = [
        replay_row(1, 480, 240),
        replay_row(1, 480, 240, kind=3, bet_line_x2=None),
        replay_row(2, 480, 240),
        replay_row(2, 480, 240, kind=3),
    ]
    combined = same_timepoint_combinations(_members(rows))
    assert set(combined["game_id"]) == {2}
    assert same_timepoint_combinations(_members(rows[:1])).empty


def test_mean_edge_uses_the_betting_members_minimum_edge():
    rows = [replay_row(1, 480, 231), replay_row(1, 480, 230, kind=3)]
    combined = same_timepoint_combinations(_members(rows), min_edge={1: 0.5, 3: 0.5})
    assert _strategy(combined, MEAN_EDGE, 480).iloc[0]["result"] == "NO BET"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((240, 240), ("OVER", "WIN")),  # Both agree.
        ((240, 220), ("NO EDGE", "NO BET")),  # Two members must agree.
        ((240, 230.5), ("NO EDGE", "NO BET")),  # One abstains: no majority.
        ((240, 240, 220), ("OVER", "WIN")),  # Two of three.
        ((220, 220, 240), ("UNDER", "LOSS")),
        ((240, 220, 230.5), ("NO EDGE", "NO BET")),
    ],
)
def test_majority_vote_needs_more_than_half_of_the_members(values, expected):
    kinds_horizons = [(1, 480, 480), (3, 480, 480), (1, 0, 480)]
    rows = [
        replay_row(1, horizon, value, kind=kind, bucket=bucket)
        for value, (kind, horizon, bucket) in zip(values, kinds_horizons, strict=False)
    ]
    combined = same_timepoint_combinations(_members(rows))
    votes = combined.loc[
        combined["method"].eq(MAJORITY_VOTE) & combined["members"].eq(len(values))
    ]
    assert len(votes) == 1
    assert (votes.iloc[0]["signal"], votes.iloc[0]["result"]) == expected
    assert votes["absolute_error"].isna().all()


def test_cross_target_members_name_their_target():
    rows = [replay_row(1, 480, 240)]
    partner = [replay_row(1, 480, 10, target=1)]
    combined = same_timepoint_combinations(_members(rows, partner))
    assert set(combined["member_names"]) == {
        "XGBoost 2.5 intermediate (line error) + "
        "XGBoost 2.5 intermediate (total points)"
    }
    assert set(combined["result"]) == {"WIN"}


def test_rank_combinations_puts_eligible_leaders_first():
    rows = []
    for game in range(40):
        rows.append(replay_row(game, 480, 240))  # Always OVER: wins.
        rows.append(replay_row(game, 480, 240 if game % 4 else 220, kind=3))
    ranking = rank_combinations(same_timepoint_combinations(_members(rows)))
    assert ranking["eligible"].tolist() == [True, True]
    # They agree (OVER) on 30 games and win them all. On the other 10 the
    # mean edge (+9.5 and −10.5) is −0.5: UNDER, which loses.
    lead = ranking.iloc[0]
    assert lead["method"] == MAJORITY_VOTE
    assert (lead["wins"], lead["losses"]) == (30, 0)
    assert ranking.iloc[1]["method"] == MEAN_EDGE
    assert (ranking.iloc[1]["wins"], ranking.iloc[1]["losses"]) == (30, 10)
    assert rank_combinations(pd.DataFrame()).empty


def test_combination_table_puts_both_methods_of_a_set_side_by_side():
    rows = []
    for game in range(40):
        rows.append(replay_row(game, 480, 240))
        rows.append(replay_row(game, 480, 240 if game % 4 else 220, kind=3))
    rows.append(replay_row(99, 360, 240))
    rows.append(replay_row(99, 360, 220, kind=3))  # Few games: not eligible.
    table = combination_table(same_timepoint_combinations(_members(rows)))
    assert table["timepoint"].tolist() == [480, 360]
    lead = table.iloc[0]
    assert lead["games"] == 40
    # Majority: 30 games agree, 30-0; no bet on the other 10. Mean: 30-10.
    assert (lead["vote_wins"], lead["vote_losses"], lead["vote_bets"]) == (30, 0, 30)
    assert (lead["mean_wins"], lead["mean_losses"], lead["mean_bets"]) == (30, 10, 40)
    assert lead["vote_win_pct"] == 100 and lead["mean_win_pct"] == 75
    assert lead["best_win_pct"] == 100 and lead["eligible"]
    small = table.iloc[1]
    assert not small["eligible"] and not small["vote_eligible"]
    assert small["vote_bets"] == 0 and np.isnan(small["vote_win_pct"])
    assert combination_table(pd.DataFrame()).empty


def test_thinking_high_variant_is_hidden_everywhere():
    rows = pd.DataFrame(
        [
            replay_row(1, 480, 240, kind=3, variant="tabpfn_v3_5"),
            replay_row(1, 480, 240, kind=3, variant="tabpfn_v3_5_thinking_high"),
            replay_row(1, 480, 240, kind=3, variant="tabpfn_v3_5_thinking_medium"),
        ]
    )
    assert set(prepare_analysis(rows)["variant"]) == {
        "tabpfn_v3_5",
        "tabpfn_v3_5_thinking_medium",
    }
    assert "tabpfn_v3_5_thinking_high" not in set(displayed_models(rows)["variant"])


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


def test_profit_probability_is_the_exact_beta_tail_with_a_skeptical_prior():
    # Beta(50, 50) is symmetric around a coin flip.
    assert profit_probability(0, 0, threshold_pct=50) == pytest.approx(0.5)
    # Checked against 400k Beta draws: 0.3177, 0.9851, 0.2866.
    assert profit_probability(0, 0) == pytest.approx(0.3173, abs=1e-3)
    assert profit_probability(36, 4) == pytest.approx(0.9848, abs=1e-3)
    assert profit_probability(20, 20) == pytest.approx(0.2868, abs=1e-3)
    assert profit_probability(10, 30) < profit_probability(20, 20)
    # The prior shrinks a short streak: 10-0 is far from certain.
    assert profit_probability(10, 0) == pytest.approx(0.677, abs=1e-3)
    # 55% over 2,000 bets: about 2.1 SD above break-even.
    assert profit_probability(1100, 900) == pytest.approx(0.986, abs=1e-3)


def _cells(results_by_cell):
    """One settled row per game and cell; ``True`` = WIN."""
    return pd.DataFrame(
        [
            {"cell": cell, "game_id": game, "result": "WIN" if won else "LOSS"}
            for cell, results in results_by_cell.items()
            for game, won in enumerate(results)
        ]
    )


def test_game_permutation_keeps_shared_games_shared():
    hot = [game % 10 < 7 for game in range(40)]  # 28-12: 70%.
    one = _cells({"a": hot})
    copies = _cells({name: hot for name in "abcdefghij"})
    alone = game_permutation_pvalue(one, "cell", ["a"], 70.0)
    shared = game_permutation_pvalue(copies, "cell", list("abcdefghij"), 70.0)
    # Ten copies of the same picks are one test, not ten tries ...
    assert shared == pytest.approx(alone)
    assert 0.003 < alone < 0.03
    # ... while the coin-flip chance treats them as independent.
    assert selection_chance([40] * 10, 70.0) > 3 * selection_chance([40], 70.0)


def test_game_permutation_counts_independent_cells_as_tries():
    rng = np.random.default_rng(3)
    cells = {str(i): list(rng.integers(0, 2, 40).astype(bool)) for i in range(10)}
    cells["hot"] = [game % 10 < 7 for game in range(40)]
    frame = _cells(cells)
    alone = game_permutation_pvalue(frame, "cell", ["hot"], 70.0)
    among = game_permutation_pvalue(frame, "cell", list(cells), 70.0)
    assert among > 2 * alone


def test_game_permutation_edge_cases():
    frame = _cells({"a": [True] * 40})
    assert game_permutation_pvalue(frame, "cell", ["a"], 100.0) < 0.002
    assert game_permutation_pvalue(frame, "cell", ["a"], 0.0) == 1.0
    assert math.isnan(game_permutation_pvalue(frame, "cell", ["b"], 50.0))
    assert math.isnan(game_permutation_pvalue(frame.iloc[:0], "cell", ["a"], 50.0))
    pushes = frame.assign(result="PUSH")
    assert math.isnan(game_permutation_pvalue(pushes, "cell", ["a"], 50.0))
