"""Oracle diagnostics: readouts R1 / R2 and the actual / observed-rate graphs."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.features import RatingBook
from nba_ou.data_processing.lineups.game_projection import PlayerNight
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.game_graph import (
    GameGraphParams,
    build_game_graphs,
    readout_2_6,
)
from nba_ou.data_processing.player_graph.oracle import (
    actual_graphs,
    readouts,
    regulation_stints,
    usage_as_of,
    with_observed_rates,
)
from nba_ou.data_processing.player_graph.overlap import OverlapParams

GAME, DATE = "0022300100", pd.Timestamp("2024-01-10")
HOME = [PlayerNight(f"h{k}", 36.0 - 3 * k, 0.5 if k == 0 else 0.0) for k in range(7)]
AWAY = [PlayerNight(f"a{k}", 34.0 - 2 * k, 0.0) for k in range(7)]
NIGHTS = {(GAME, "100"): HOME, (GAME, "200"): AWAY}
COVERED = set(NIGHTS)
GAMES = pd.DataFrame(
    {
        "GAME_ID": [GAME],
        "GAME_DATE": [DATE],
        "HOME_TEAM_ID": ["100"],
        "AWAY_TEAM_ID": ["200"],
    }
)
PLAYERS = [p.player_id for p in HOME + AWAY]


def _ratings():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "game_id": GAME,
            "player_id": PLAYERS,
            "off": rng.normal(0, 2, len(PLAYERS)),
            "def": rng.normal(0, 2, len(PLAYERS)),
            "pace": rng.normal(0, 1, len(PLAYERS)),
            "league_ortg": 112.0,
            "league_pace": 99.0,
        }
    )


def _usage():
    return pd.DataFrame(
        {
            "game_id": GAME,
            "player_id": PLAYERS,
            "usage": np.linspace(0.1, 0.3, len(PLAYERS)),
        }
    )


def _history():
    rows = []
    for off in ("h0", "h1", "h2"):
        for dfn in ("a0", "a1", "a2"):
            rows.append(
                {
                    "game_id": "0022300050",
                    "game_date": DATE - pd.Timedelta(days=20),
                    "off_player_id": off,
                    "def_player_id": dfn,
                    "matchup_seconds": 600.0 if off[1] == dfn[1] else 30.0,
                    "cofloor_seconds": 1200.0,
                }
            )
    return pd.DataFrame(rows)


def _graphs(pair_game, provider="pair_lift"):
    data = PointInTimeData.from_frames(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pair_game=pair_game
    )
    params = GameGraphParams(overlap=OverlapParams(provider=provider))
    return build_game_graphs(data, GAMES, NIGHTS, report_covered=COVERED, params=params)


def test_r1_is_the_2_6_readout():
    graphs = _graphs(_history())
    ratings = _ratings()
    book = RatingBook(
        ratings.rename(
            columns={"off": "o_rating", "def": "d_rating", "pace": "pace_rating"}
        )
        .drop(columns="game_id")
        .assign(as_of_date=DATE, fit_max_game_date=DATE - pd.Timedelta(days=1))
    )
    expected = readout_2_6(graphs, book).iloc[0]
    got = readouts(graphs, ratings, _usage()).iloc[0]
    assert got.r1_total == pytest.approx(expected.raw_total, abs=1e-9)
    assert got.r1_total - got.r1_total_healthy == pytest.approx(
        expected.impact_points, abs=1e-9
    )


def test_r2_equals_r1_when_shares_follow_floor_time():
    # Independence overlaps and no guarding history: m_ij ∝ minutes of j, as
    # long as nobody is allocated more than 48 minutes (then the overlap cap
    # min(min_i, min_j) binds; 2_6's allocate_minutes has no 48-minute cap).
    nights = {
        (GAME, "100"): [
            PlayerNight(f"h{k}", 30.0 - k, 0.5 if k == 0 else 0.0) for k in range(9)
        ],
        (GAME, "200"): [PlayerNight(f"a{k}", 28.0 - k, 0.0) for k in range(9)],
    }
    data = PointInTimeData.from_frames(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    params = GameGraphParams(overlap=OverlapParams(provider="independence"))
    graphs = build_game_graphs(
        data, GAMES, nights, report_covered=set(nights), params=params
    )
    assert graphs.nodes["minutes"].max() <= 48
    players = sorted(set(graphs.nodes["player_id"]))
    ratings = (
        _ratings().iloc[:1].merge(pd.DataFrame({"player_id": players}), how="cross")
    )
    ratings = ratings.drop(columns="player_id_x").rename(
        columns={"player_id_y": "player_id"}
    )
    ratings[["off", "def", "pace"]] = np.random.default_rng(2).normal(
        0, 2, (len(players), 3)
    )
    usage = pd.DataFrame(
        {
            "game_id": GAME,
            "player_id": players,
            "usage": np.linspace(0.1, 0.3, len(players)),
        }
    )
    out = readouts(graphs, ratings, usage).iloc[0]
    assert out.r2_total == pytest.approx(out.r1_total, abs=1e-9)
    assert out.r2_total_healthy == pytest.approx(out.r1_total_healthy, abs=1e-9)


def test_r2_moves_when_a_strong_defender_guards_the_high_usage_attacker():
    graphs = _graphs(_history())
    out = readouts(graphs, _ratings(), _usage()).iloc[0]
    assert out.r2_total != pytest.approx(out.r1_total, abs=1e-6)


def _stints(overtime=False):
    home = [
        np.array([f"h{k}" for k in range(5)]),
        np.array(["h0", "h1", "h2", "h3", "h5"]),
    ]
    away = [np.array([f"a{k}" for k in range(5)])] * 2
    rows = [
        {
            "period": period,
            "home_lineup": home[i % 2],
            "away_lineup": away[i % 2],
            "seconds": 360.0,
        }
        for period in range(1, 5)
        for i in range(2)
    ]
    if overtime:
        rows.append(
            {
                "period": 5,
                "home_lineup": home[0],
                "away_lineup": away[0],
                "seconds": 300.0,
            }
        )
    return pd.DataFrame(rows).assign(
        game_id=GAME, game_date=DATE, season_year=2023, seg_idx=lambda f: range(len(f))
    )


def _observed_rates(stints):
    """A deterministic observed rate for every opponent pair."""
    home = sorted({p for lineup in stints.home_lineup for p in lineup})
    away = sorted({p for lineup in stints.away_lineup for p in lineup})
    rows = [
        (GAME, off, dfn, 0.4 if off[1] == dfn[1] else 0.05)
        for attackers, defenders in ((home, away), (away, home))
        for off in attackers
        for dfn in defenders
    ]
    return pd.DataFrame(
        rows, columns=["game_id", "off_player_id", "def_player_id", "rate"]
    )


@pytest.mark.parametrize("overtime", [False, True])
def test_actual_regulation_minutes_sum_to_240(overtime):
    stints = regulation_stints(_stints(overtime))
    graphs = actual_graphs(stints, _observed_rates(stints), pd.Series({GAME: DATE}))
    minutes = graphs.nodes.groupby("side")["minutes"].sum()
    assert minutes.tolist() == pytest.approx([240.0, 240.0])
    shared = graphs.edges.loc[graphs.edges.relation.eq("teammate"), "weight"].sum()
    assert shared == pytest.approx(2 * 240 * 4 / 2)  # 4 teammates, each pair once


def test_observed_rates_on_actual_overlap_reproduce_observed_shares():
    stints = regulation_stints(_stints())
    rates = _observed_rates(stints)
    graphs = actual_graphs(stints, rates, pd.Series({GAME: DATE}))
    guards = graphs.edges.loc[graphs.edges.relation.eq("guards")]
    load = guards["r_hat"] * guards["expected_overlap"]
    observed = load / load.groupby(guards["dst"]).transform("sum")
    assert guards["weight"].to_numpy() == pytest.approx(observed.to_numpy())
    assert guards.groupby("dst")["weight"].sum().to_numpy() == pytest.approx(1.0)


def test_guard_oracle_swaps_rates_only():
    graphs = _graphs(_history())
    observed = pd.DataFrame(
        {
            "game_id": [GAME],
            "off_player_id": ["h3"],
            "def_player_id": ["a6"],
            "guard_rate": [0.9],
        }
    )
    swapped = with_observed_rates(graphs, observed)
    assert swapped.nodes.equals(graphs.nodes)
    guards = swapped.edges.loc[swapped.edges.relation.eq("guards")]
    hit = guards.loc[guards.dst.eq("h3") & guards.src.eq("a6"), "r_hat"]
    assert (hit == 0.9).all()
    before = graphs.edges.loc[graphs.edges.relation.eq("guards")]
    untouched = ~(guards.dst.eq("h3") & guards.src.eq("a6"))
    assert np.allclose(
        guards.loc[untouched & guards.dst.ne("h3"), "weight"],
        before.loc[untouched & before.dst.ne("h3"), "weight"],
    )
    assert guards.groupby(["scenario_id", "dst"])[
        "weight"
    ].sum().to_numpy() == pytest.approx(1)


def test_usage_is_strictly_before_the_cutoff_and_shrunk():
    box = pd.DataFrame(
        {
            "GAME_ID": ["0022300001", "0022300002"],
            "GAME_DATE": [DATE - pd.Timedelta(days=2), DATE],
            "PLAYER_ID": ["p", "p"],
            "MIN": [30.0, 30.0],
            "USG_PCT": [0.30, 0.90],
        }
    )
    data = PointInTimeData.from_frames(pd.DataFrame(), pd.DataFrame(), box)
    usage = usage_as_of(data.as_of(DATE), prior_minutes=0.0)
    assert usage.loc["p"] == pytest.approx(0.30)


def test_a_game_without_ratings_stays_missing_instead_of_zero():
    graphs = _graphs(_history())
    no_ratings = _ratings().iloc[0:0]
    out = readouts(graphs, no_ratings, _usage())
    assert out[["r1_total", "r2_total", "possessions"]].isna().all().all()
    assert out[["r1_total_healthy", "r2_total_healthy"]].isna().all().all()
