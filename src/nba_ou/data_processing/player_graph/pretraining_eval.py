"""Phase 6's own validation: per-target error on future stints and edges.

The plan's ladder, every rung scored on the same out-of-fold rows
(``pretraining_oof``): constant -> RAPM -> FM without interactions -> FM
without the matchup term -> FM with it -> GNN. The pooled loss hides what the
plan cares about, so everything is per target (points, pace, 3PA/FGA, ...).

For a target with labels ``y``, exposures ``w`` and two prediction sets, the
**gain** of ``better`` over ``worse`` is the exposure-weighted mean drop in
squared error, ``mean_w[(y - worse)^2 - (y - better)^2]``, with its standard
error clustered by game date (stints and edges of the same date share
players, games and league conditions), and as a share of ``worse``'s MSE.
Both prediction sets must cover the same rows: the comparison is paired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .rating_diagnostics import _clustered_mean


def _flatten(
    labels: np.ndarray, weights: np.ndarray, *predictions: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Rows ``(n * sides)`` per target, as ``(rows, targets)`` arrays."""
    t = labels.shape[-1]
    return tuple(a.reshape(-1, t) for a in (labels, weights, *predictions))


def target_errors(
    labels: np.ndarray,
    weights: np.ndarray,
    predictions: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    """Weighted MSE, MAE and bias per target, over rows with a label and a
    prediction."""
    y, w, p = _flatten(labels, weights, predictions)
    rows = {}
    for k, name in enumerate(names):
        mask = (w[:, k] > 0) & np.isfinite(y[:, k]) & np.isfinite(p[:, k])
        e = y[mask, k] - p[mask, k]
        ww = w[mask, k]
        rows[name] = {
            "rows": int(mask.sum()),
            "exposure": float(ww.sum()),
            "mse": float(np.average(e**2, weights=ww)) if mask.any() else np.nan,
            "mae": float(np.average(np.abs(e), weights=ww)) if mask.any() else np.nan,
            "bias": float(np.average(e, weights=ww)) if mask.any() else np.nan,
        }
    return pd.DataFrame(rows).T


def paired_gain(
    labels: np.ndarray,
    weights: np.ndarray,
    better: np.ndarray,
    worse: np.ndarray,
    dates: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    """Gain of ``better`` over ``worse`` per target (module docstring).

    ``dates`` has one entry per row of ``labels``' first axis. Raises if the
    two sets do not predict the same rows.
    """
    y, w, b, v = _flatten(labels, weights, better, worse)
    sides = int(np.prod(labels.shape[1:-1])) if labels.ndim > 2 else 1
    clusters = np.repeat(np.asarray(dates), sides)
    rows = {}
    for k, name in enumerate(names):
        has = (w[:, k] > 0) & np.isfinite(y[:, k])
        if not (np.isfinite(b[has, k]) == np.isfinite(v[has, k])).all():
            raise ValueError(f"{name}: the two prediction sets cover different rows")
        mask = has & np.isfinite(b[:, k])
        if not mask.any():
            continue
        drop = (y[mask, k] - v[mask, k]) ** 2 - (y[mask, k] - b[mask, k]) ** 2
        gain, se = _clustered_mean(drop, w[mask, k], clusters[mask])
        mse = float(np.average((y[mask, k] - v[mask, k]) ** 2, weights=w[mask, k]))
        rows[name] = {
            "rows": int(mask.sum()),
            "gain": gain,
            "se": se,
            "t": gain / se if se > 0 else np.nan,
            "gain_pct": 100 * gain / mse,
            "se_pct": 100 * se / mse,
        }
    return pd.DataFrame(rows).T


def ladder(
    labels: np.ndarray,
    weights: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    dates: np.ndarray,
    names: Sequence[str],
    reference: str,
    masks: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Every model's gain over ``reference``, per target, overall and in each
    slice of ``masks`` (boolean over the first axis). Rows any model leaves
    unpredicted are dropped for all, so every comparison is paired."""
    common = np.ones(labels.shape, bool)
    for values in predictions.values():
        common &= np.isfinite(values)
    weights = np.where(common, weights, 0.0)
    slices = {"all": np.ones(labels.shape[0], bool), **dict(masks or {})}
    out = []
    for slice_name, mask in slices.items():
        for model, values in predictions.items():
            if model == reference:
                continue
            table = paired_gain(
                labels[mask],
                weights[mask],
                values[mask],
                predictions[reference][mask],
                np.asarray(dates)[mask],
                names,
            )
            out.append(table.assign(model=model, slice=slice_name))
    if not out:
        return pd.DataFrame()
    frame = pd.concat(out).rename_axis("target").reset_index()
    return frame[["slice", "model", "target", *frame.columns[1:-2]]]
