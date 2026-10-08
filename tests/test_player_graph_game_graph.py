"""Game graphs: scenarios, expected overlaps, guard shares and the 2_6 readout."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.features import RatingBook
from nba_ou.data_processing.lineups.game_projection import (
    PlayerNight,
    project_with_counterfactual,
)
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.game_graph import (
    FULL_HEALTH,
    build_game_graphs,
    readout_2_6,
)

GAME, DATE = "0022300100", pd.Timestamp("2024-01-10")
HOME_TEAM, AWAY_TEAM = "100", "200"
HOME = [PlayerNight(f"h{k}", 36.0 - 3 * k, 0.5 if k == 0 else 0.0) for k in range(7)]
AWAY = [PlayerNight(f"a{k}", 34.0 - 2 * k, 0.0) for k in range(7)]
NIGHTS = {(GAME, HOME_TEAM): HOME, (GAME, AWAY_TEAM): AWAY}
COVERED = {(GAME, HOME_TEAM), (GAME, AWAY_TEAM)}
GAMES = pd.DataFrame(
    {
        "GAME_ID": [GAME],
        "GAME_DATE": [DATE],
        "HOME_TEAM_ID": [HOME_TEAM],
        "AWAY_TEAM_ID": [AWAY_TEAM],
    }
)


def _pair_history():
    """a0 guarded h0 heavily in an earlier game; a1 barely."""
    rows = []
    for off in ("h0", "h1"):
        for dfn in ("a0", "a1", "a2"):
            heavy = off == "h0" and dfn == "a0"
            rows.append(
                {
                    "game_id": "0022300050",
                    "game_date": DATE - pd.Timedelta(days=20),
                    "off_player_id": off,
                    "def_player_id": dfn,
                    "matchup_seconds": 600.0 if heavy else 60.0,
                    "cofloor_seconds": 1200.0,
                }
            )
    return pd.DataFrame(rows)


def _graphs(pair_game=None):
    data = PointInTimeData.from_frames(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pair_game=pair_game if pair_game is not None else _pair_history(),
    )
    return build_game_graphs(data, GAMES, NIGHTS, report_covered=COVERED)


def test_one_graph_per_scenario_plus_full_health():
    graphs = _graphs()
    scenarios = graphs.scenarios
    # h0 is a coin flip: two home scenarios x one away, then full health.
    assert len(scenarios) == 3
    assert scenarios["is_full_health"].sum() == 1
    assert scenarios.loc[~scenarios.is_full_health, "weight"].tolist() == [0.5, 0.5]
    sitting = scenarios.set_index("scenario_id")["home_sitting"]
    assert sorted(sitting.loc[[0, 1]].tolist()) == [(), ("h0",)]


def test_minutes_sum_to_240_and_absent_players_are_not_nodes():
    graphs = _graphs()
    for (_, scenario_id), nodes in graphs.nodes.groupby(["game_id", "scenario_id"]):
        assert nodes.groupby("side")["minutes"].sum().tolist() == pytest.approx(
            [240.0, 240.0]
        )
        sitting = graphs.scenarios.set_index("scenario_id").loc[
            scenario_id, "home_sitting"
        ]
        assert not set(sitting) & set(nodes["player_id"])
    # Inside a scenario minutes are not scaled by a probability: h1 has the
    # same minutes whether or not the other scenario exists.
    out = graphs.scenarios.loc[
        graphs.scenarios.home_sitting.map(lambda x: x == ("h0",)), "scenario_id"
    ]
    nodes = graphs.nodes.loc[graphs.nodes.scenario_id.eq(out.iloc[0])]
    home = nodes.loc[nodes.side.eq("home")].set_index("player_id")["minutes"]
    raw = {p.player_id: p.base_minutes for p in HOME if p.player_id != "h0"}
    assert home.loc["h1"] == pytest.approx(raw["h1"] * 240 / sum(raw.values()))


def test_overlaps_never_exceed_minutes():
    graphs = _graphs()
    minutes = graphs.nodes.set_index(["scenario_id", "player_id"])["minutes"]
    pairs = graphs.edges.loc[graphs.edges.relation.isin(["teammate", "opponent"])]
    cap = np.minimum(
        minutes.loc[list(zip(pairs.scenario_id, pairs.src, strict=True))].to_numpy(),
        minutes.loc[list(zip(pairs.scenario_id, pairs.dst, strict=True))].to_numpy(),
    )
    assert (pairs["weight"].to_numpy() <= cap + 1e-9).all()
    assert (pairs["weight"] >= 0).all()


def test_guard_shares_sum_to_one_over_tonights_defenders():
    graphs = _graphs()
    guards = graphs.edges.loc[graphs.edges.relation.eq("guards")]
    totals = guards.groupby(["scenario_id", "dst"])["weight"].sum()
    assert totals.to_numpy() == pytest.approx(1.0)
    # Shares follow r_hat * expected overlap where the rates are known.
    healthy = guards.loc[guards.scenario_id.eq(FULL_HEALTH) & guards.dst.eq("h0")]
    load = healthy["r_hat"] * healthy["expected_overlap"]
    assert healthy["weight"].to_numpy() == pytest.approx((load / load.sum()).to_numpy())
    assert healthy.set_index("src")["weight"].idxmax() == "a0"
    assert not healthy["fallback"].any()


def test_absent_defenders_attackers_are_reassigned():
    nights = dict(NIGHTS)
    nights[(GAME, AWAY_TEAM)] = [
        PlayerNight("a0", 34.0, 1.0),
        *AWAY[1:],
    ]
    data = PointInTimeData.from_frames(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pair_game=_pair_history()
    )
    graphs = build_game_graphs(data, GAMES, nights, report_covered=COVERED)
    guards = graphs.edges.loc[graphs.edges.relation.eq("guards")]
    tonight = guards.loc[guards.scenario_id.ne(FULL_HEALTH) & guards.dst.eq("h1")]
    assert "a0" not in set(tonight["src"])
    assert tonight.groupby("scenario_id")["weight"].sum().to_numpy() == pytest.approx(1)
    healthy = guards.loc[guards.scenario_id.eq(FULL_HEALTH) & guards.dst.eq("h1")]
    assert "a0" in set(healthy["src"])


def test_without_any_history_shares_follow_the_overlap():
    empty = pd.DataFrame(columns=["game_id", "game_date"])
    graphs = _graphs(pair_game=empty)
    guards = graphs.edges.loc[graphs.edges.relation.eq("guards")]
    assert guards["fallback"].all() and not guards["r_hat_known"].any()
    part = guards.loc[guards.scenario_id.eq(FULL_HEALTH) & guards.dst.eq("h0")]
    share = part["expected_overlap"] / part["expected_overlap"].sum()
    assert part["weight"].to_numpy() == pytest.approx(share.to_numpy())
    numeric = guards[["weight", "r_hat", "guard_prior_weight", "log_exposure"]]
    assert np.isfinite(numeric.to_numpy(float)).all()


def test_readout_reproduces_the_2_6_projection():
    players = [p.player_id for p in HOME + AWAY]
    rng = np.random.default_rng(0)
    ratings = pd.DataFrame(
        {
            "as_of_date": DATE,
            "player_id": players,
            "o_rating": rng.normal(0, 2, len(players)),
            "d_rating": rng.normal(0, 2, len(players)),
            "pace_rating": rng.normal(0, 1, len(players)),
            "league_ortg": 112.0,
            "league_pace": 99.0,
            "fit_max_game_date": DATE - pd.Timedelta(days=1),
        }
    )
    book = RatingBook(ratings)
    readout = readout_2_6(_graphs(), book).iloc[0]
    team_ratings, ortg, pace = book.for_date(DATE)
    expected = project_with_counterfactual(
        HOME, AWAY, team_ratings, team_ratings, ortg, pace
    )
    assert readout.raw_total == pytest.approx(expected["total"], abs=1e-9)
    assert readout.impact_points == pytest.approx(
        expected["absence_impact_points"], abs=1e-9
    )
    assert readout.possessions == pytest.approx(expected["possessions"], abs=1e-9)


def _empty_data():
    return PointInTimeData.from_frames(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pair_game=_pair_history()
    )


def test_zero_minute_roster_players_are_not_nodes_and_leave_no_nan():
    # Only coach's-decision DNPs in his window: a recent average of 0.
    nights = dict(NIGHTS)
    nights[(GAME, HOME_TEAM)] = [*HOME, PlayerNight("h_dnp", 0.0, 0.0)]
    graphs = build_game_graphs(_empty_data(), GAMES, nights, report_covered=COVERED)
    assert "h_dnp" not in set(graphs.nodes["player_id"])
    assert "h_dnp" not in set(graphs.edges["src"]) | set(graphs.edges["dst"])
    edges = graphs.edges
    shared = edges[["weight", "expected_overlap", "log_exposure"]]
    assert np.isfinite(shared.to_numpy(float)).all()
    guards = edges.loc[edges.relation.eq("guards"), ["r_hat", "guard_prior_weight"]]
    assert np.isfinite(guards.to_numpy(float)).all()
    pairs = edges.loc[edges.relation.ne("guards"), ["lift", "lift_prior_weight"]]
    assert np.isfinite(pairs.to_numpy(float)).all()
    assert (graphs.nodes["minutes"] > 0).all()


@pytest.mark.parametrize("covered", [set(), {(GAME, HOME_TEAM)}])
def test_games_without_both_injury_reports_get_no_graphs(covered):
    graphs = build_game_graphs(_empty_data(), GAMES, NIGHTS, report_covered=covered)
    assert graphs.scenarios.empty and graphs.edges.empty
    assert graphs.metadata["skipped_games"]["uncovered"] == 1


def test_report_coverage_is_required():
    with pytest.raises(TypeError):
        build_game_graphs(_empty_data(), GAMES, NIGHTS)  # type: ignore[call-arg]
