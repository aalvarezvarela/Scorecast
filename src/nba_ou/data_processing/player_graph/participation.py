"""Participation A of the phase 4A minutes provider (v1: logistic).

``q_i = P(player enters the rotation | medically available in this
scenario)``. Injury uncertainty stays in the scenario weights: inside a
scenario a player out plays 0 and an available one plays with probability
``q``; ``p_out`` is never applied again.

**Labels.** One row per roster player and team-game from the rotation
engine (``rotation.walk_forward``), label = played regulation minutes:

* ``availability_source = "injury_report"`` when the team filed a report by
  tip-off: only players not listed, or listed with ``p_out <= AVAILABLE``
  (probable / available), are rows. Questionable, doubtful and out are left
  out: a questionable player who sits may be a medical or a coach's decision.
  A rotation player the report calls available who does not play is a 0,
  even though the engine counts him as an absence.
* ``"heuristic"`` otherwise (no report: all of 2016-17, 2017-18 and 2018-19
  before December 17): the engine's absence events (a non-playing player
  vacating >= 10 min, e.g. any rotation player with ``b >= 15``) are taken as
  unavailable and left out; every other non-playing roster player is a
  coach's decision, a 0. These are weak labels.

**Features** describe the player in tonight's scenario (the other players'
absences): baseline minutes, depth rank among the available, the minutes C
gives him from those absences, the minutes vacated by others, how many are
available, his recent participation, his streak of games out and whether he
returns from a long one, his minutes in the last game, start share, rest and
the point of the season.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

#: p_out at or below this counts as medically available (2_6's SCENARIO_LOW).
AVAILABLE = 0.1
LONG_ABSENCE = 5  # team games

FEATURES = (
    "log_b",
    "has_baseline",
    "rank",
    "gain_c",
    "vacated_others",
    "n_available",
    "recent_participation",
    "log_streak",
    "returning",
    "last_minutes",
    "start_share",
    "rest_days",
    "back_to_back",
    "season_fraction",
)


def labelled_rows(
    players: pd.DataFrame,
    p_out: Mapping[tuple[str, str, str], float],
    covered: set[tuple[str, str]],
) -> pd.DataFrame:
    """Rows q is trained and scored on, with ``played`` and
    ``availability_source``."""
    rows = players.loc[players["on_roster"]].copy()
    key = list(zip(rows["game_id"], rows["team_id"], strict=True))
    rows["report"] = [k in covered for k in key]
    rows["p_out"] = [
        p_out.get((g, t, p), 0.0)
        for g, t, p in zip(
            rows["game_id"], rows["team_id"], rows["player_id"], strict=True
        )
    ]
    keep_report = rows["report"] & (rows["p_out"] <= AVAILABLE)
    keep_heuristic = ~rows["report"] & ~rows["absent_event"]
    rows = rows.loc[keep_report | keep_heuristic].copy()
    rows["availability_source"] = np.where(rows["report"], "injury_report", "heuristic")
    rows["played"] = rows["minutes"] > 0
    return rows


def feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    b = rows["b"]
    out = pd.DataFrame(index=rows.index)
    out["log_b"] = np.log1p(b.fillna(0.0).clip(lower=0.0))
    out["has_baseline"] = b.notna().astype(float)
    out["rank"] = rows["rank"].fillna(rows["n_available"] + 1)
    out["gain_c"] = rows["gain_c"]
    out["vacated_others"] = rows["vacated_others"]
    out["n_available"] = rows["n_available"]
    out["recent_participation"] = rows["q"].fillna(0.0)
    out["log_streak"] = np.log1p(rows["streak"])
    out["returning"] = (rows["streak"] >= LONG_ABSENCE).astype(float)
    out["last_minutes"] = rows["last_minutes"]
    out["start_share"] = rows["start_share"]
    out["rest_days"] = rows["rest_days"].fillna(3).clip(upper=5)
    out["back_to_back"] = (rows["rest_days"] == 1).astype(float)
    out["season_fraction"] = rows["season_game"] / 82.0
    return out[list(FEATURES)].astype(float)


def walk_forward_q(
    train_rows: pd.DataFrame,
    predict_rows: pd.DataFrame,
    *,
    min_train_rows: int = 5_000,
    c: float = 1.0,
) -> pd.Series:
    """Logistic ``q`` for ``predict_rows``, month by month: each month's model
    (scaler included) is fitted on the labelled ``train_rows`` of earlier
    months only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_month = pd.to_datetime(train_rows["game_date"]).dt.to_period("M")
    target_month = pd.to_datetime(predict_rows["game_date"]).dt.to_period("M")
    train_x = feature_matrix(train_rows)
    target_x = feature_matrix(predict_rows)
    out = pd.Series(np.nan, index=predict_rows.index)
    for current in sorted(target_month.unique()):
        past = train_month < current
        if past.sum() < min_train_rows:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
        model.fit(train_x.loc[past], train_rows.loc[past, "played"])
        target = target_month == current
        out.loc[target] = model.predict_proba(target_x.loc[target])[:, 1]
    return out


def brier(q: pd.Series, played: pd.Series) -> float:
    return float(((q - played.astype(float)) ** 2).mean())


def log_loss(q: pd.Series, played: pd.Series, eps: float = 1e-3) -> float:
    q = q.clip(eps, 1 - eps)
    y = played.astype(float)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
