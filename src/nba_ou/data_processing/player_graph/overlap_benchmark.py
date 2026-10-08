"""Benchmark for expected overlaps: how well lifts predict a game's shared time.

Given each player's **actual** seconds in the game (so minutes error is left
out), a pair's predicted shared time is
``min(lift * seconds_a * seconds_b / length, seconds_a, seconds_b)``, i.e.
``lift * independent_seconds`` clipped. Two errors, per relation:

* ``tvd``: for each player and game, how his shared time is spread over his
  teammates (or opponents), actual vs predicted, as a total variation
  distance; averaged weighted by his seconds. Scale-free, like the guard
  benchmark.
* ``rel_abs_error``: ``sum |predicted - actual| / sum actual`` in seconds.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .overlap import RELATIONS, shrink_lift


def predicted_shared(frame: pd.DataFrame, lift) -> np.ndarray:
    cap = np.minimum(frame["seconds_a"].to_numpy(float), frame["seconds_b"])
    raw = np.asarray(lift, float) * frame["independent_seconds"].to_numpy(float)
    return np.clip(raw, 0.0, cap)


def _player_rows(frame: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Each undirected pair once per player."""
    base = {
        "game_id": frame["game_id"].to_numpy(),
        "relation": frame["relation"].to_numpy(),
        "season_year": frame["season_year"].to_numpy(),
        "actual": frame["shared_seconds"].to_numpy(float),
        "predicted": predicted,
    }
    return pd.concat(
        [
            pd.DataFrame(
                base
                | {
                    "player": frame[f"player_{side}"].to_numpy(),
                    "seconds": frame[f"seconds_{side}"].to_numpy(float),
                }
            )
            for side in ("a", "b")
        ],
        ignore_index=True,
    )


def resolve_lift(lift) -> tuple[np.ndarray, np.ndarray]:
    """``(lift, fallback)``: an undefined (non-finite or negative) lift falls
    back to independence (1) and is flagged."""
    lift = np.asarray(lift, float)
    fallback = ~(np.isfinite(lift) & (lift >= 0))
    return np.where(fallback, 1.0, lift), fallback


def overlap_errors(
    frame: pd.DataFrame, lift: Iterable[float] | np.ndarray
) -> pd.DataFrame:
    """``tvd``, ``rel_abs_error`` and ``fallback`` per relation (rows) and
    season (columns).

    Every estimator is scored on the **same** player-games: those with some
    actual shared time. An undefined lift falls back to independence, and a
    player whose predicted shares sum to 0 gets independence shares. Neither is
    dropped, so an undefined prediction cannot look perfect or shrink the
    sample; ``fallback`` is the share of player-games that used either.
    """
    lift, fallback = resolve_lift(lift)
    predicted = predicted_shared(frame, lift)
    independent = predicted_shared(frame, np.ones(len(frame)))
    rows = _player_rows(frame, predicted).assign(
        independent=_player_rows(frame, independent)["predicted"].to_numpy(),
        fallback=np.concatenate([fallback, fallback]),
    )
    groups = [rows["game_id"], rows["player"], rows["relation"]]
    actual_total = rows["actual"].groupby(groups).transform("sum")
    predicted_total = rows["predicted"].groupby(groups).transform("sum")
    independent_total = rows["independent"].groupby(groups).transform("sum")
    empty = predicted_total <= 0
    shares = (rows["predicted"] / predicted_total).where(
        ~empty, rows["independent"] / independent_total
    )
    rows = rows.assign(
        diff=(rows["actual"] / actual_total - shares).abs(),
        fallback=rows["fallback"] | empty,
    ).loc[actual_total > 0]
    per_player = rows.groupby(["game_id", "player", "relation"]).agg(
        tvd=("diff", lambda d: 0.5 * d.sum()),
        seconds=("seconds", "first"),
        season_year=("season_year", "first"),
        fallback=("fallback", "any"),
    )
    pairs = frame.assign(error=np.abs(predicted - frame["shared_seconds"]))

    def summarize(part: pd.DataFrame, pair_part: pd.DataFrame) -> dict:
        return {
            "tvd": float(np.average(part["tvd"], weights=part["seconds"])),
            "rel_abs_error": float(
                pair_part["error"].sum() / pair_part["shared_seconds"].sum()
            ),
            "fallback": float(part["fallback"].mean()),
        }

    out = {}
    for relation in RELATIONS:
        part = per_player.xs(relation, level="relation")
        pair_part = pairs.loc[pairs["relation"].eq(relation)]
        columns = {"all": summarize(part, pair_part)}
        for season in sorted(part["season_year"].unique()):
            columns[int(season)] = summarize(
                part.loc[part["season_year"].eq(season)],
                pair_part.loc[pair_part["season_year"].eq(season)],
            )
        for metric in ("tvd", "rel_abs_error", "fallback"):
            out[(relation, metric)] = {
                column: values[metric] for column, values in columns.items()
            }
    return pd.DataFrame(out).T


def estimator_lifts(
    frame: pd.DataFrame, k_grid: Iterable[float]
) -> dict[str, np.ndarray]:
    """The lifts each estimator assigns to ``frame``'s pairs."""
    lifts = {
        "independence (lift 1)": np.ones(len(frame)),
        "relation lift only (k=inf)": frame["relation_lift"].to_numpy(float),
        "pair history only (k->0)": shrink_lift(
            frame["hist_shared_seconds_decayed"],
            frame["hist_independent_seconds_decayed"],
            frame["relation_lift"],
            1e-6,
        )[0],
    }
    if "lift" in frame.columns:
        lifts["v0 provider (k per relation)"] = frame["lift"].to_numpy(float)
    for k in k_grid:
        lifts[f"k={k:g}"] = shrink_lift(
            frame["hist_shared_seconds_decayed"],
            frame["hist_independent_seconds_decayed"],
            frame["relation_lift"],
            k,
        )[0]
    return lifts
