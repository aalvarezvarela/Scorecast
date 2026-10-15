"""Monthly temporal out-of-fold folds for phase 6, and the baselines scored on them.

Plan principle 2 and phase 7's algorithm: at checkpoint ``C_k`` (the first
day of a month) a model is trained only on data dated before ``C_k`` and
predicts ``[C_k, C_(k+1))``; a later checkpoint may warm-start from the
previous one. Every phase 6 model (FM, GNN) and every baseline is scored on
the same folds, so their errors compare stint by stint.

The first checkpoint is ``FIRST_CHECKPOINT`` (2016-12-01): phase 3's profile
prior, which the RAPM baseline needs, starts there. Months without games
(the off-season) have no fold.

Baselines, at the same monthly cadence:

* **constant**: each target's exposure-weighted mean over the training
  stints, decayed with RAPM's 180-day half-life (the RAPM intercept's
  analogue for every target);
* **RAPM**: the frozen phase 3 ratings (profile prior, lambda 3,000 /
  10,000, ``rating_book.FROZEN``) as of ``C_k``: points per possession
  ``(league_ortg + sum off - sum def) / 100`` per side and pace
  ``league_pace + sum pace`` over the ten. It has no rating for the other
  four targets, which take the constant baseline.

**Daily RAPM, the residual models' base** (:func:`daily_rapm_predictions`):
the same frozen ratings refitted on every game date, each stint predicted by
the ratings fitted on stints strictly before its own date (never itself, its
day or anything later). It is the base of both the residual training targets
and the evaluation, so a model trained at ``C_k`` learns ``y - base`` with the
same base it is served. There is no prior before ``FIRST_CHECKPOINT``, so
stints before it have no base (:func:`residual_stints` gives their points and
pace no weight).

Predictions are stored per model and fold (:class:`PredictionStore`) with the
fold's training cutoff, so a run resumes and nothing is mixed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .profile_prior import HALF_LIFE_DAYS
from .rating_book import FIRST_CHECKPOINT, FROZEN, BookSpec, walk_forward_rating_book
from .stint_graph import LABELS

PTS = LABELS.index("pts_per_poss")
PACE = LABELS.index("poss_per_48")


@dataclass(frozen=True)
class Fold:
    """Train on dates before ``checkpoint``; predict ``[checkpoint, until)``."""

    checkpoint: pd.Timestamp
    until: pd.Timestamp

    @property
    def label(self) -> str:
        return self.checkpoint.strftime("%Y-%m")

    def masks(self, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(train, predict)`` boolean masks over ``dates``."""
        dates = pd.to_datetime(dates).to_numpy()
        train = dates < self.checkpoint.to_datetime64()
        predict = (dates >= self.checkpoint.to_datetime64()) & (
            dates < self.until.to_datetime64()
        )
        return train, predict


def monthly_folds(dates: Iterable, first=FIRST_CHECKPOINT, last=None) -> list[Fold]:
    """One fold per month from ``first`` that has dates to predict."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    if dates.empty:
        return []
    start = pd.Timestamp(first).normalize().replace(day=1)
    end = pd.Timestamp(last) if last is not None else dates.max()
    folds = []
    for month in pd.date_range(start, end, freq="MS"):
        until = month + pd.offsets.MonthBegin(1)
        if ((dates >= month) & (dates < until)).any():
            folds.append(Fold(month, until))
    return folds


def check_fold(fold: Fold, train_dates, predict_dates) -> None:
    """Raise unless every training date is before the checkpoint and every
    predicted date inside the fold."""
    train = pd.to_datetime(pd.Series(train_dates))
    predict = pd.to_datetime(pd.Series(predict_dates))
    if len(train) and train.max() >= fold.checkpoint:
        raise ValueError(f"Fold {fold.label} trains on {train.max().date()}")
    if len(predict) and (
        predict.min() < fold.checkpoint or predict.max() >= fold.until
    ):
        raise ValueError(f"Fold {fold.label} predicts outside its month")


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def decayed_means(
    labels: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
    cutoff: pd.Timestamp,
    half_life_days: float = HALF_LIFE_DAYS,
) -> np.ndarray:
    """Per-target exposure-weighted mean of ``labels`` (``(n, 2, L)``) dated
    before ``cutoff``, each stint's weight decayed by its age."""
    dates = pd.to_datetime(dates).to_numpy()
    before = dates < cutoff.to_datetime64()
    age = (
        (cutoff.to_datetime64() - dates[before]).astype("timedelta64[D]").astype(float)
    )
    decay = (0.5 ** (age / half_life_days))[:, None, None]
    w = weights[before] * decay
    values = np.nan_to_num(labels[before])
    total = w.sum(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (w * values).sum(axis=(0, 1)) / total, np.nan)


def constant_predictions(
    labels: np.ndarray, weights: np.ndarray, dates: np.ndarray, folds: Sequence[Fold]
) -> np.ndarray:
    """The constant baseline for every fold's stints (NaN outside the folds)."""
    out = np.full(labels.shape, np.nan)
    for fold in folds:
        _, predict = fold.masks(dates)
        if predict.any():
            out[predict] = decayed_means(labels, weights, dates, fold.checkpoint)
    return out


def rapm_predictions(
    players: np.ndarray,
    dates: np.ndarray,
    folds: Sequence[Fold],
    history: pd.DataFrame,
    prior: Callable,
    spec: BookSpec = FROZEN,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Points per possession and pace from the frozen ratings as of each
    fold's checkpoint, ``(n, 2, L)`` with NaN for the other targets; and the
    rating book it used.

    ``players``: ``(n, 10)`` node-order ids (home five, then away five);
    ``history``: stint-store rows (all seasons); ``prior``: phase 3's provider
    (``rating_book.frozen_prior_provider``).
    """
    book = walk_forward_rating_book(
        history, [fold.checkpoint for fold in folds], spec=spec, prior=prior
    )
    out = np.full((len(players), 2, len(LABELS)), np.nan)
    book["as_of_date"] = pd.to_datetime(book["as_of_date"]).dt.normalize()
    by_date = dict(tuple(book.groupby("as_of_date", sort=False)))
    for fold in folds:
        _, predict = fold.masks(dates)
        if not predict.any() or fold.checkpoint not in by_date:
            continue
        ratings = by_date[fold.checkpoint].set_index("player_id")
        ids = players[predict].astype(str)
        o, d, p = (
            _lookup(ratings[column], ids)
            for column in ("o_rating", "d_rating", "pace_rating")
        )
        ortg = float(ratings["league_ortg"].iloc[0])
        pace = float(ratings["league_pace"].iloc[0])
        home = (ortg + o[:, :5].sum(1) - d[:, 5:].sum(1)) / 100.0
        away = (ortg + o[:, 5:].sum(1) - d[:, :5].sum(1)) / 100.0
        rows = np.flatnonzero(predict)
        out[rows, 0, PTS] = home
        out[rows, 1, PTS] = away
        out[rows, :, PACE] = (pace + p.sum(1))[:, None]
    return out, book


def _lookup(ratings: pd.Series, ids: np.ndarray) -> np.ndarray:
    """``ratings`` of ``ids`` in their shape; a player without a row has 0."""
    values = ratings.reindex(ids.reshape(-1)).fillna(0.0)
    return values.to_numpy(float).reshape(ids.shape)


def daily_rapm_predictions(
    players: np.ndarray,
    dates: np.ndarray,
    history: pd.DataFrame,
    prior: Callable,
    *,
    first=FIRST_CHECKPOINT,
    until=None,
    spec: BookSpec = FROZEN,
) -> tuple[np.ndarray, pd.DataFrame]:
    """:func:`rapm_predictions` with one fold per game date: every stint dated
    in ``[first, until)`` gets the ratings fitted on stints before its date."""
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    days = days[days >= pd.Timestamp(first)]
    if until is not None:
        days = days[days < pd.Timestamp(until)]
    folds = [Fold(day, day + pd.Timedelta(days=1)) for day in days]
    return rapm_predictions(players, dates, folds, history, prior, spec=spec)


def residual_stints(stints, base: np.ndarray, targets: Sequence[str]):
    """``stints`` whose ``targets`` become ``y - base`` (a weight of 0 where
    the base is missing); other targets unchanged. ``stints`` is a
    ``PretrainingStints``; ``base`` is ``(n, 2, L)``."""
    from dataclasses import replace

    labels = stints.labels.copy()
    weights = stints.label_weight.copy()
    for name in targets:
        k = LABELS.index(name)
        known = np.isfinite(base[:, :, k])
        labels[:, :, k] = np.where(known, labels[:, :, k] - base[:, :, k], np.nan)
        weights[:, :, k] = np.where(known, weights[:, :, k], 0.0)
    return replace(stints, labels=labels, label_weight=weights)


def calibrated_residuals(
    predictions: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
    folds: Sequence[Fold],
    targets: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Walk-forward shrinkage of residual corrections (decided 2026-10-15).

    For each fold and residual target, ``lambda`` is the exposure-weighted
    least-squares slope through the origin of the realized residual
    ``y - base`` on the model's correction ``prediction - base``, over the
    out-of-fold rows dated before the fold's checkpoint (earlier folds'
    predictions, all known by then), clipped to ``[0, 1]``: 1 keeps the
    correction, 0 removes it. A fold with no earlier predictions gets 0 (no
    evidence, no correction). Returns the calibrated predictions (the other
    targets unchanged) and the ``lambda`` per fold and target.
    """
    out = predictions.copy()
    dates = pd.to_datetime(dates).to_numpy()
    rows = []
    for fold in folds:
        _, predict = fold.masks(dates)
        if not predict.any():
            continue
        earlier = dates < fold.checkpoint.to_datetime64()
        row = {"fold": fold.label}
        for name in targets:
            k = LABELS.index(name)
            c = predictions[earlier, :, k] - base[earlier, :, k]
            r = labels[earlier, :, k] - base[earlier, :, k]
            w = weights[earlier, :, k]
            ok = np.isfinite(c) & np.isfinite(r) & (w > 0)
            denominator = float((w[ok] * c[ok] ** 2).sum())
            slope = (
                float((w[ok] * c[ok] * r[ok]).sum()) / denominator
                if denominator > 0
                else 0.0
            )
            shrink = float(np.clip(slope, 0.0, 1.0))
            out[predict, :, k] = base[predict, :, k] + shrink * (
                predictions[predict, :, k] - base[predict, :, k]
            )
            row[name] = shrink
        rows.append(row)
    return out, pd.DataFrame(rows)


def with_constant(predictions: np.ndarray, constant: np.ndarray) -> np.ndarray:
    """``predictions`` with its NaN targets taken from ``constant``."""
    return np.where(np.isnan(predictions), constant, predictions)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionStore:
    """``<root>/<model>/fold=YYYY-MM.parquet`` plus ``<model>/model.json``.

    A fold file holds the stint keys (``game_id``, ``seg_idx``) and one column
    per side and target (``home_pts_per_poss``, ...).
    """

    root: Path

    def path(self, model: str, fold: Fold) -> Path:
        return self.root / model / f"fold={fold.label}.parquet"

    def has(self, model: str, fold: Fold) -> bool:
        return self.path(model, fold).exists()

    def write(
        self,
        model: str,
        fold: Fold,
        game_id: np.ndarray,
        seg_idx: np.ndarray,
        predictions: np.ndarray,
        extra: dict | None = None,
    ) -> None:
        frame = pd.DataFrame({"game_id": game_id, "seg_idx": seg_idx})
        for s, side in enumerate(("home", "away")):
            for t, name in enumerate(LABELS):
                frame[f"{side}_{name}"] = predictions[:, s, t]
        frame["checkpoint"] = fold.checkpoint
        path = self.path(model, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic: a crash never leaves half a fold
        if extra is not None:
            meta = self.root / model / "model.json"
            meta.write_text(json.dumps(extra, indent=2, default=str) + "\n")

    def read(self, model: str, game_id: np.ndarray, seg_idx: np.ndarray) -> np.ndarray:
        """``(n, 2, L)`` aligned to ``(game_id, seg_idx)``; NaN where no fold
        predicted the stint."""
        files = sorted((self.root / model).glob("fold=*.parquet"))
        if not files:
            raise FileNotFoundError(f"No predictions for {model} under {self.root}")
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        keys = pd.MultiIndex.from_arrays(
            [frame["game_id"].astype(str), frame["seg_idx"]]
        )
        if keys.duplicated().any():
            raise ValueError(f"{model}: a stint is predicted by two folds")
        wanted = pd.MultiIndex.from_arrays([np.asarray(game_id).astype(str), seg_idx])
        position = keys.get_indexer(wanted)
        out = np.full((len(wanted), 2, len(LABELS)), np.nan)
        found = position >= 0
        for s, side in enumerate(("home", "away")):
            for t, name in enumerate(LABELS):
                out[found, s, t] = frame[f"{side}_{name}"].to_numpy(float)[
                    position[found]
                ]
        return out
