"""Expected overlaps: per-game shared time, and the shrunk lift as of D."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.overlap import (
    OverlapParams,
    build_overlap_game,
    expected_lift,
    expected_overlap,
)
from nba_ou.data_processing.player_graph.overlap_benchmark import overlap_errors

HOME = [f"h{k}" for k in range(6)]
AWAY = [f"a{k}" for k in range(5)]


def _stints(game="0022300001", date="2023-10-24"):
    """h4 plays the first half, h5 the second: they never share the floor."""
    return pd.DataFrame(
        {
            "game_id": game,
            "game_date": pd.Timestamp(date),
            "season_year": 2023,
            "seg_idx": [0, 1],
            "home_lineup": [np.array(HOME[:5]), np.array([*HOME[:4], HOME[5]])],
            "away_lineup": [np.array(AWAY), np.array(AWAY)],
            "seconds": [600.0, 600.0],
        }
    )


def _row(table, a, b, relation):
    a, b = sorted([a, b])
    found = table.loc[
        table.player_a.eq(a) & table.player_b.eq(b) & table.relation.eq(relation)
    ]
    assert len(found) == 1
    return found.iloc[0]


def test_every_pair_who_played_is_listed_with_shared_and_independent_time():
    table = build_overlap_game(_stints())
    # 6 home + 5 away players: C(6,2) + C(5,2) teammate pairs, 6 x 5 opponents.
    assert (table.relation == "teammate").sum() == 15 + 10
    assert (table.relation == "opponent").sum() == 30
    never = _row(table, "h4", "h5", "teammate")
    assert never.shared_seconds == 0.0
    assert never.independent_seconds == pytest.approx(600 * 600 / 1200)
    core = _row(table, "h0", "a0", "opponent")
    assert core.shared_seconds == 1200.0 and core.independent_seconds == 1200.0
    # Ids are an undirected key and the seconds follow them.
    swapped = _row(table, "h5", "a1", "opponent")
    assert swapped.player_a == "a1" and swapped.seconds_b == 600.0


def test_shared_time_sums_to_four_teammates_and_five_opponents():
    table = build_overlap_game(_stints())
    for relation, slots in (("teammate", 4), ("opponent", 5)):
        part = table.loc[table.relation.eq(relation)]
        shared = (
            pd.concat(
                [
                    part.groupby("player_a")["shared_seconds"].sum(),
                    part.groupby("player_b")["shared_seconds"].sum(),
                ]
            )
            .groupby(level=0)
            .sum()
        )
        seconds = (
            pd.concat(
                [
                    part.groupby("player_a")["seconds_a"].first(),
                    part.groupby("player_b")["seconds_b"].first(),
                ]
            )
            .groupby(level=0)
            .first()
        )
        assert np.allclose(shared, slots * seconds.loc[shared.index])


def _data(*games):
    stints = pd.concat([_stints(game, date) for game, date in games], ignore_index=True)
    return PointInTimeData.from_frames(
        stints,
        pd.DataFrame(),
        pd.DataFrame(),
        overlap_game=build_overlap_game(stints),
    )


def _pairs(*pairs):
    return pd.DataFrame(pairs, columns=["player_a", "player_b", "relation"])


def test_lift_is_shrunk_toward_the_relation_lift_with_k_per_relation():
    data = _data(("0022300001", "2023-10-24"))
    params = OverlapParams(k_teammate=300.0, k_opponent=5000.0, half_life_days=1e9)
    out = expected_lift(
        data.as_of("2023-10-25"),
        _pairs(("h5", "h4", "teammate"), ("h0", "a0", "opponent")),
        params=params,
    ).set_index(["player_a", "player_b"])
    history = build_overlap_game(_stints())
    teammates = history.loc[history.relation.eq("teammate")]
    relation_lift = teammates.shared_seconds.sum() / teammates.independent_seconds.sum()
    never = out.loc[("h4", "h5")]
    assert never.relation_lift == pytest.approx(relation_lift)
    assert never.lift == pytest.approx((0 + 300 * relation_lift) / (300 + 300))
    assert never.lift_prior_weight == pytest.approx(0.5)
    assert never.n_games == 1 and never.has_pair_history
    # Opponents: relation lift is exactly 1 (each player shares 5x his seconds).
    assert out.loc[("a0", "h0")].relation_lift == pytest.approx(1.0)


def test_history_is_strictly_before_the_cutoff():
    data = _data(("0022300001", "2023-10-24"), ("0022300002", "2023-10-26"))
    out = expected_lift(data.as_of("2023-10-26"), _pairs(("h4", "h5", "teammate")))
    assert out.iloc[0].n_games == 1
    first_day = expected_lift(
        data.as_of("2023-10-24"), _pairs(("h4", "h5", "teammate"))
    )
    row = first_day.iloc[0]
    assert row.n_games == 0 and row.lift == 1.0 and row.relation_lift == 1.0


def test_independence_provider_keeps_lift_one_and_the_evidence():
    data = _data(("0022300001", "2023-10-24"))
    out = expected_lift(
        data.as_of("2023-10-25"),
        _pairs(("h4", "h5", "teammate")),
        params=OverlapParams(provider="independence"),
    )
    assert out.iloc[0].lift == 1.0 and out.iloc[0].n_games == 1


def test_expected_overlap_never_exceeds_either_players_minutes():
    overlap = expected_overlap([36.0, 10.0, 0.0], [36.0, 40.0, 30.0], [1.6, 1.0, 1.0])
    assert overlap.tolist() == pytest.approx([36.0, 10 * 40 / 48, 0.0])


def test_benchmark_is_zero_for_a_perfect_lift():
    table = build_overlap_game(_stints())
    perfect = table.shared_seconds / table.independent_seconds
    errors = overlap_errors(table, perfect)
    assert errors["all"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    flat = overlap_errors(table, np.ones(len(table)))
    assert flat.loc[("teammate", "tvd"), "all"] > 0


def test_undefined_lifts_fall_back_to_independence_on_the_same_sample():
    table = build_overlap_game(_stints())
    half = np.where(np.arange(len(table)) % 2 == 0, np.nan, 1.0)
    errors = overlap_errors(table, half)
    independence = overlap_errors(table, np.ones(len(table)))
    # Same sample, and the undefined half scored as independence, not dropped.
    for relation in ("teammate", "opponent"):
        for metric in ("tvd", "rel_abs_error"):
            assert errors.loc[(relation, metric), "all"] == pytest.approx(
                independence.loc[(relation, metric), "all"]
            )
        assert errors.loc[(relation, "fallback"), "all"] > 0
    assert independence.loc[("teammate", "fallback"), "all"] == 0


def test_all_zero_lifts_are_not_a_perfect_score():
    table = build_overlap_game(_stints())
    zero = overlap_errors(table, np.zeros(len(table)))
    independence = overlap_errors(table, np.ones(len(table)))
    assert zero.loc[("teammate", "tvd"), "all"] == pytest.approx(
        independence.loc[("teammate", "tvd"), "all"]
    )
    assert zero.loc[("teammate", "fallback"), "all"] == 1.0
