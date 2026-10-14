"""Phase 5 feature block v1: scalar ``PI_*`` columns read off the game graph.

Inputs, all as of the game's date: closing game graphs with phase 4A's v1
minutes (``minutes_provider.V1Minutes``) over the corrected pair history, and
the frozen phase 3 rating book (``rating_book``: offense and defense shrunk
toward the profile prior, pace toward zero). Stint RAPM is the defensive
rating; the rejected scalar matchup rating is not used, and neither are the
matchup-weighted defensive readouts (phase 4B: no evidence they improve team
scoring).

Per game, over tonight's availability scenarios (weights renormalized) and,
for the counterfactual, the full-health graph:

* ``PI_OFF_RATING_BEFORE_TEAM_{HOME,AWAY}``: the side's expected offense,
  ``sum_i min_i / 48 * o_i`` (points per 100 possessions above the league
  intercept; the five players on the floor add up to the team's);
* ``PI_DEF_RATING_BEFORE_TEAM_{HOME,AWAY}``: the same with ``d_i`` (positive =
  good defense, as 2_6's ``d_rating``);
* ``PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_{HOME,AWAY}``: the share of the side's
  240 minutes going to players with at most ``LOW_SAMPLE_POSSESSIONS`` decayed
  offensive possessions in the book (phase 3's thin-player threshold; a
  player missing from the book has none);
* ``PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE``: tonight's projected total minus the
  full-health one, 2_6's projection (``game_graph.readout_2_6``) with these
  ratings and minutes.

Games without graphs (no report for both teams, no rating date) get no row;
their columns are NaN in the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .game_graph import GameGraphs, readout_2_6

LOW_SAMPLE_POSSESSIONS = 1000.0
SIDES = {"home": "TEAM_HOME", "away": "TEAM_AWAY"}
FEATURES = (
    "PI_OFF_RATING_BEFORE_TEAM_HOME",
    "PI_OFF_RATING_BEFORE_TEAM_AWAY",
    "PI_DEF_RATING_BEFORE_TEAM_HOME",
    "PI_DEF_RATING_BEFORE_TEAM_AWAY",
    "PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_HOME",
    "PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_AWAY",
    "PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE",
)
#: The phase 5 primary block (decided 2026-10-14). The per-side offense and
#: defense levels are 84-91% recoverable from 2_5's team ratings on 2018-19
#: (``pi_feature_audit.py``); they stay in the files for future ablations.
PRIMARY_FEATURES = (
    "PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_HOME",
    "PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_AWAY",
    "PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE",
)
ABLATION_FEATURES = tuple(c for c in FEATURES if c not in PRIMARY_FEATURES)


def pi_features(graphs: GameGraphs, book, book_rows: pd.DataFrame) -> pd.DataFrame:
    """``GAME_ID`` and :data:`FEATURES` for every game with graphs and a
    rating date. ``book`` is the ``RatingBook`` of ``book_rows`` (the stored
    book's rows, read for ``poss_weight``)."""
    exposure = {
        day: frame.set_index("player_id")["poss_weight"]
        for day, frame in book_rows.assign(
            as_of_date=pd.to_datetime(book_rows["as_of_date"]).dt.normalize(),
            player_id=book_rows["player_id"].astype(str),
        ).groupby("as_of_date")
    }
    scenarios = graphs.scenarios
    tonight = scenarios.loc[~scenarios["is_full_health"].astype(bool)].copy()
    tonight["w"] = tonight["weight"] / tonight.groupby("game_id")["weight"].transform(
        "sum"
    )
    nodes = graphs.nodes.merge(
        tonight[["game_id", "scenario_id", "w", "as_of_date"]],
        on=["game_id", "scenario_id"],
    )
    rows = []
    for (game_id, as_of_date), part in nodes.groupby(
        ["game_id", "as_of_date"], sort=False
    ):
        rated = book.for_date(as_of_date)
        if rated is None:
            continue
        ratings = rated[0]
        thin = exposure[book.as_of(as_of_date)]
        players = part["player_id"].astype(str)
        share = part["minutes"].to_numpy(float) / 48.0
        weighted = part["w"].to_numpy(float) * share
        o = np.array([ratings.offense.get(p, 0.0) for p in players])
        d = np.array([ratings.defense.get(p, 0.0) for p in players])
        low = (thin.reindex(players).fillna(0.0) <= LOW_SAMPLE_POSSESSIONS).to_numpy()
        row = {"GAME_ID": game_id}
        for side, suffix in SIDES.items():
            mine = part["side"].eq(side).to_numpy()
            row[f"PI_OFF_RATING_BEFORE_{suffix}"] = float((weighted * o)[mine].sum())
            row[f"PI_DEF_RATING_BEFORE_{suffix}"] = float((weighted * d)[mine].sum())
            row[f"PI_LOW_SAMPLE_MIN_SHARE_BEFORE_{suffix}"] = float(
                (weighted * low)[mine].sum() * 48.0 / 240.0
            )
        rows.append(row)
    features = pd.DataFrame(rows, columns=["GAME_ID", *FEATURES[:6]])
    impact = readout_2_6(graphs, book)[["GAME_ID", "impact_points"]].rename(
        columns={"impact_points": "PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE"}
    )
    return features.merge(impact, on="GAME_ID", how="inner")[["GAME_ID", *FEATURES]]
