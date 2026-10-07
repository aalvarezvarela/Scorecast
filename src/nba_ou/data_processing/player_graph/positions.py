"""Soft G/F/C positions as of a date: the v0 guarding prior's player positions.

The NBA only records a **starting** position (``START_POSITION`` in the box
score, ``off_position`` in the matchups; they agree on every 2023-24 row), so
bench players have none. Each player gets a probability per position, from
two sources, both strictly before the cutoff:

(a) **Starts**: his starting positions in the window, read from the matchups
    (they cover 2017-18, where the player box scores are missing).
(b) **Profile**: a multinomial logistic regression on per-36 box-score rates,
    trained on players with enough starts. Its prediction ``q`` is used for
    everyone, so a regular starter is still described by his starts.

They are blended like counts with a pseudo-count ``c``::

    p(x) = (starts(x) + c * q(x)) / (starts + c)

so (a) dominates with many starts and (b) with few. With no box scores either,
``q`` is the league's share of starts per position.

This is only the v0 fallback for the guarding prior. The phase 4B matchup
model should learn guarding propensity from profiles directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .as_of import AsOfView

POSITIONS = ("G", "F", "C")
PROBABILITY_COLUMNS = tuple(f"p_{position}" for position in POSITIONS)

#: Per-36 rates that separate guards, forwards and centers.
PROFILE_STATS = ("AST", "OREB", "DREB", "BLK", "STL", "FG3A", "FGA", "PF")
#: Preseason (001) and All-Star (003) games do not describe a role.
_COUNTED_GAME_PREFIXES = ("002", "004", "005", "006")


def start_counts(matchups: pd.DataFrame) -> pd.DataFrame:
    """Starts per player and position: ``player_id`` index, ``G``/``F``/``C``."""
    starts = matchups.loc[
        matchups["off_position"].isin(POSITIONS),
        ["game_id", "off_player_id", "off_position"],
    ].drop_duplicates(["game_id", "off_player_id"])
    counts = pd.crosstab(starts["off_player_id"].astype(str), starts["off_position"])
    counts.index.name = "player_id"
    return counts.reindex(columns=list(POSITIONS), fill_value=0)


def profile_rates(
    box_scores: pd.DataFrame, *, prior_minutes: float = 200.0
) -> pd.DataFrame:
    """Per-36 rates of ``PROFILE_STATS``, shrunk to the league rate.

    ``prior_minutes`` of league-average play are added to every player, so a
    player with ten minutes does not get extreme rates. Also returns
    ``minutes``.
    """
    box = box_scores.loc[
        box_scores["GAME_ID"]
        .astype(str)
        .str.zfill(10)
        .str[:3]
        .isin(_COUNTED_GAME_PREFIXES)
    ]
    box = box.loc[pd.to_numeric(box["MIN"], errors="coerce").fillna(0) > 0]
    if box.empty:
        return pd.DataFrame(columns=[*PROFILE_STATS, "minutes"])
    numeric = box[["MIN", *PROFILE_STATS]].apply(pd.to_numeric, errors="coerce")
    sums = numeric.fillna(0).groupby(box["PLAYER_ID"].astype(str)).sum()
    league = sums[list(PROFILE_STATS)].sum() / sums["MIN"].sum()
    rates = (sums[list(PROFILE_STATS)] + prior_minutes * league).div(
        sums["MIN"] + prior_minutes, axis=0
    ) * 36.0
    rates["minutes"] = sums["MIN"]
    rates.index.name = "player_id"
    return rates


def _profile_probabilities(
    rates: pd.DataFrame, counts: pd.DataFrame, min_label_starts: int
) -> pd.DataFrame:
    """``q``: the profile classifier's G/F/C probabilities for every rated player."""
    labelled = counts.loc[counts.sum(axis=1) >= min_label_starts].index
    labelled = labelled.intersection(rates.index)
    if len(labelled) == 0:
        return pd.DataFrame(index=rates.index, columns=list(POSITIONS), dtype=float)
    target = counts.loc[labelled].idxmax(axis=1)
    if target.nunique() < len(POSITIONS):
        return pd.DataFrame(index=rates.index, columns=list(POSITIONS), dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(rates.loc[labelled, list(PROFILE_STATS)], target)
    probabilities = model.predict_proba(rates[list(PROFILE_STATS)])
    return pd.DataFrame(probabilities, index=rates.index, columns=model.classes_)[
        list(POSITIONS)
    ]


def positions_as_of(
    view: AsOfView,
    *,
    window_days: int = 3 * 365,
    pseudo_starts: float = 5.0,
    min_label_starts: int = 10,
    prior_minutes: float = 200.0,
    since: Any | None = None,
) -> pd.DataFrame:
    """Soft positions of every player seen in the window before ``view.cutoff``:
    in the matchups (attacking or defending) or the box scores.

    Returns, indexed by ``player_id``: ``p_G``, ``p_F``, ``p_C`` (summing to 1),
    ``n_starts``, ``box_minutes`` and ``has_profile``.
    """
    start = since if since is not None else view.cutoff - pd.Timedelta(days=window_days)
    matchups = view.matchups(since=start)
    counts = start_counts(matchups)
    rates = profile_rates(view.box_scores(since=start), prior_minutes=prior_minutes)
    # Bench players with no box scores (all of 2017-18) appear only here.
    seen = pd.Index(
        pd.concat(
            [
                matchups[column]
                for column in ("off_player_id", "def_player_id")
                if column in matchups.columns
            ]
            or [pd.Series(dtype=str)]
        )
        .astype(str)
        .unique()
    )
    players = counts.index.union(rates.index).union(seen)
    counts = counts.reindex(players, fill_value=0)

    total_starts = counts.to_numpy(float).sum()
    league = (
        counts.sum().to_numpy(float) / total_starts
        if total_starts > 0
        else np.full(len(POSITIONS), 1.0 / len(POSITIONS))
    )
    q = _profile_probabilities(rates, counts, min_label_starts).reindex(players)
    has_profile = q.notna().all(axis=1)
    q = q.astype(float)
    q.loc[~has_profile, :] = league

    n = counts.sum(axis=1).to_numpy(float)[:, None]
    p = (counts.to_numpy(float) + pseudo_starts * q.to_numpy(float)) / (
        n + pseudo_starts
    )
    out = pd.DataFrame(p, index=players, columns=list(PROBABILITY_COLUMNS))
    out["n_starts"] = n[:, 0].astype(int)
    out["box_minutes"] = rates["minutes"].reindex(players).fillna(0.0)
    out["has_profile"] = has_profile.to_numpy()
    out.index.name = "player_id"
    return out
