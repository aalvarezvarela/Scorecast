"""Phase 4A rotation structure: expanded roster, baseline minutes, shares."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.rotation import (
    RotationParams,
    shares,
    walk_forward,
)

TEAM = "100"
REGULARS = {"x": 32.0, "a": 34.0, "b": 33.0, "c": 31.0, "d": 30.0, "e": 26.0, "f": 24.0}
BACKUP = "y"  # 12 minutes normally, takes X's minutes when X is out


def _minutes(n_games=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_games):
        date = pd.Timestamp("2018-10-16") + pd.Timedelta(days=2 * k)
        x_out = k >= 20 and k % 4 == 0
        played = dict(REGULARS)
        played[BACKUP] = 12.0
        if x_out:
            del played["x"]
            played[BACKUP] += 30.0  # nearly all of X's minutes
        total = sum(played.values())
        for p, m in played.items():
            noisy = m * 240.0 / total + rng.normal(0, 0.5)
            rows.append((f"00218{k:05d}", date, 2018, TEAM, p, noisy, p in "abcde"))
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "game_date",
            "season",
            "team_id",
            "player_id",
            "minutes",
            "started",
        ],
    )


def test_the_backup_who_always_replaces_x_gets_his_share():
    params = RotationParams(kappa=5.0, min_vacated=10.0)
    players, absorption = walk_forward(
        _minutes(), lambda d, p: np.full(3, 1 / 3), params=params, refit_monthly=False
    )
    late = absorption.loc[absorption["game_date"] > pd.Timestamp("2019-02-01")]
    assert not late.empty and set(late["absent"]) == {"x"}
    by_player = late.groupby("player_id")[["s", "proportional"]].mean()
    assert by_player["s"].idxmax() == BACKUP
    assert by_player.loc[BACKUP, "s"] > 0.6
    assert by_player.loc[BACKUP, "proportional"] < 0.1
    # Baseline is conditional on playing and net of absence gains: it tracks
    # his minutes in games X plays (~12.9), not his average including X's
    # absences (~20).
    late = players.loc[players["game_date"].gt(pd.Timestamp("2019-02-01"))]
    backup = late.loc[late["player_id"].eq(BACKUP)]
    normal = backup.loc[backup["n_absent"].eq(0), "minutes"].mean()
    assert backup["b"].mean() == pytest.approx(normal, abs=1.0)


def test_shares_are_non_negative_and_sum_to_one():
    evidence = {("x", "p1"): (-50.0, 100.0), ("x", "p2"): (90.0, 100.0)}
    out = shares(
        "x", ["p1", "p2", "p3"], evidence, np.array([0.2, 0.3, 0.5]), kappa=10.0
    )
    assert (out >= 0).all() and out.sum() == pytest.approx(1.0)
    assert out.argmax() == 1
