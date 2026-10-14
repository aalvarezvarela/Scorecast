"""Phase 5 feature block v1 read off hand-built game graphs."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.features import RatingBook
from nba_ou.data_processing.player_graph.game_graph import (
    FULL_HEALTH,
    GameGraphs,
    readout_2_6,
)
from nba_ou.data_processing.player_graph.pi_features import (
    FEATURES,
    LOW_SAMPLE_POSSESSIONS,
    pi_features,
)

GAME, DATE = "0021800500", pd.Timestamp("2019-01-10")
HOME = [f"h{k}" for k in range(6)]
AWAY = [f"a{k}" for k in range(6)]


def _minutes(players):
    weights = np.arange(len(players), 0, -1, dtype=float)
    return dict(zip(players, 240.0 * weights / weights.sum(), strict=True))


def _graphs():
    scenarios = pd.DataFrame(
        {
            "game_id": GAME,
            "game_date": DATE,
            "as_of_date": DATE,
            "scenario_id": [0, 1, FULL_HEALTH],
            "is_full_health": [False, False, True],
            "weight": [0.6, 0.2, 1.0],  # renormalized to 0.75 / 0.25
            "home_sitting": [(), ("h0",), ()],
            "away_sitting": [(), (), ()],
        }
    )
    rows = []
    for scenario_id, home in ((0, HOME), (1, HOME[1:]), (FULL_HEALTH, HOME)):
        for side, players in (("home", home), ("away", AWAY)):
            for player, played in _minutes(players).items():
                rows.append((GAME, scenario_id, player, side, played))
    nodes = pd.DataFrame(
        rows, columns=["game_id", "scenario_id", "player_id", "side", "minutes"]
    )
    return GameGraphs(scenarios, nodes, pd.DataFrame())


def _book():
    players = HOME + AWAY[:-1]  # a5 is missing: league average, no exposure
    rng = np.random.default_rng(3)
    rows = pd.DataFrame(
        {
            "as_of_date": DATE,
            "player_id": players,
            "o_rating": rng.normal(0, 2, len(players)),
            "d_rating": rng.normal(0, 2, len(players)),
            "pace_rating": rng.normal(0, 1, len(players)),
            # h1 and a0 are thin; the rest well sampled.
            "poss_weight": [
                LOW_SAMPLE_POSSESSIONS if p in ("h1", "a0") else 5000.0 for p in players
            ],
            "league_ortg": 110.0,
            "league_pace": 99.0,
            "fit_max_game_date": DATE - pd.Timedelta(days=1),
        }
    )
    return RatingBook(rows), rows


def test_features_are_scenario_weighted_minute_sums():
    graphs = _graphs()
    book, rows = _book()
    out = pi_features(graphs, book, rows).set_index("GAME_ID").loc[GAME]
    assert list(out.index) == list(FEATURES)
    ratings = rows.set_index("player_id")
    home = {k: _minutes(p) for k, p in ((0, HOME), (1, HOME[1:]))}
    weights = {0: 0.75, 1: 0.25}

    def expected(side_minutes, column):
        return sum(
            weights[k]
            * sum(
                m / 48 * (ratings[column].get(p, 0.0) if p in ratings.index else 0.0)
                for p, m in minutes.items()
            )
            for k, minutes in side_minutes.items()
        )

    assert out["PI_OFF_RATING_BEFORE_TEAM_HOME"] == pytest.approx(
        expected(home, "o_rating")
    )
    assert out["PI_DEF_RATING_BEFORE_TEAM_HOME"] == pytest.approx(
        expected(home, "d_rating")
    )
    away = {k: _minutes(AWAY) for k in (0, 1)}
    assert out["PI_OFF_RATING_BEFORE_TEAM_AWAY"] == pytest.approx(
        expected(away, "o_rating")
    )
    # Low sample: h1 at the threshold (inclusive); a0 thin and a5 missing.
    thin_home = sum(weights[k] * home[k]["h1"] for k in (0, 1)) / 240
    thin_away = (_minutes(AWAY)["a0"] + _minutes(AWAY)["a5"]) / 240
    assert out["PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_HOME"] == pytest.approx(thin_home)
    assert out["PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_AWAY"] == pytest.approx(thin_away)
    impact = readout_2_6(graphs, book).set_index("GAME_ID").loc[GAME, "impact_points"]
    assert out["PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE"] == pytest.approx(impact)
    assert impact != 0.0  # h0 sits in one scenario


def test_games_without_a_rating_date_get_no_row():
    graphs = _graphs()
    _, rows = _book()
    later = rows.assign(
        as_of_date=DATE + pd.Timedelta(days=5),
        fit_max_game_date=DATE + pd.Timedelta(days=4),
    )
    assert pi_features(graphs, RatingBook(later), later).empty
