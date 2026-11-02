"""Participation A: availability labels and the walk-forward logistic q."""

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.participation import (
    brier,
    labelled_rows,
    walk_forward_q,
)

GAME = "0021800500"


def _players():
    base = {
        "game_id": GAME,
        "team_id": "100",
        "game_date": pd.Timestamp("2019-01-10"),
        "on_roster": True,
        "b": 20.0,
        "q": 1.0,
        "absent_event": False,
        "n_absent": 0,
        "gain_c": 0.0,
        "vacated_others": 0.0,
        "n_available": 12,
        "rank": 3,
        "streak": 0,
        "last_minutes": 20.0,
        "start_share": 1.0,
        "rest_days": 2,
        "season_game": 40,
    }
    rows = [
        base | {"player_id": "plays", "minutes": 30.0},
        base | {"player_id": "questionable_sits", "minutes": 0.0},
        base | {"player_id": "listed_out", "minutes": 0.0, "absent_event": True},
        base | {"player_id": "rested_starter", "minutes": 0.0, "absent_event": True},
        base | {"player_id": "deep_bench", "minutes": 0.0, "b": 3.0},
        base | {"player_id": "off_roster", "minutes": 12.0, "on_roster": False},
    ]
    return pd.DataFrame(rows)


P_OUT = {
    (GAME, "100", "questionable_sits"): 0.5,
    (GAME, "100", "listed_out"): 1.0,
}


def test_report_labels_keep_only_medically_available_players():
    rows = labelled_rows(_players(), P_OUT, covered={(GAME, "100")}).set_index(
        "player_id"
    )
    assert set(rows.index) == {"plays", "rested_starter", "deep_bench"}
    # Available per the report but did not play: a 0 even though the engine
    # counted him as an absence.
    assert not rows.loc["rested_starter", "played"]
    assert (rows["availability_source"] == "injury_report").all()


def test_heuristic_labels_drop_the_engine_absences():
    rows = labelled_rows(_players(), P_OUT, covered=set()).set_index("player_id")
    assert set(rows.index) == {"plays", "questionable_sits", "deep_bench"}
    assert (rows["availability_source"] == "heuristic").all()


def test_q_is_fitted_on_earlier_months_only():
    rng = np.random.default_rng(0)
    frames = []
    for month in range(1, 5):
        frame = pd.concat([_players()] * 1500, ignore_index=True)
        frame["game_date"] = pd.Timestamp(f"2019-{month:02d}-10")
        frame["b"] = rng.uniform(0, 36, len(frame))
        frame["minutes"] = np.where(
            rng.uniform(size=len(frame)) < frame["b"] / 36, 20.0, 0.0
        )
        frames.append(frame)
    rows = labelled_rows(pd.concat(frames, ignore_index=True), {}, covered=set())
    q = walk_forward_q(rows, rows, min_train_rows=1000)
    month = rows["game_date"].dt.month
    assert q[month == 1].isna().all() and q[month > 1].notna().all()
    later = month > 1
    assert brier(q[later], rows.loc[later, "played"]) < brier(
        pd.Series(rows.loc[later, "played"].mean(), index=rows.index[later]),
        rows.loc[later, "played"],
    )
