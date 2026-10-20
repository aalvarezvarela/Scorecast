"""Planted-signal diagnostic: can this pipeline find a weak signal it is GIVEN?

The question this answers is not "is the model good". It is the prior question,
the one every negative result here depends on:

    line_error runs score a holdout R2 of ~0.001 and beat the closing line's MAE
    by ~0.02 points. Two very different worlds produce that number. Either the
    market is close to efficient and there is almost nothing to find, or the
    protocol -- this preprocessing, this search space, this CV, this trial
    budget -- cannot recover a weak signal even when one is definitely present.

A planted signal separates them. Inject one feature carrying a KNOWN, small
amount of target information, change nothing else, and watch whether pooled
out-of-fold performance moves. If the pipeline cannot find 1% of target variance
handed to it on a plate, then "no signal found" was never evidence about the
market.

The construction
----------------
With ``z`` the standardised target and ``e`` an independent standard normal::

    rho = sqrt(variance_explained)
    PLANTED_SIGNAL = rho * z + sqrt(1 - rho^2) * e

``Var(PLANTED_SIGNAL) = rho^2 + (1 - rho^2) = 1`` and
``Cov(PLANTED_SIGNAL, z) = rho``, so ``corr = rho`` and the feature explains
exactly ``rho^2 = variance_explained`` of the target's variance. At
``variance_explained = 0`` it degenerates to pure independent noise, which is
what makes the 0% cell a real control for "one extra random column" rather than
a different experiment entirely.

Deliberately NOT a copy of the target with noise added on top; that would be a
different quantity (unbounded variance, and R2 that depends on the target's
scale rather than on the requested fraction).

What this is not
----------------
This feature IS target-derived. That is the entire point, and it is why every
path that could turn a diagnostic run into a shipped model refuses to run --
see ``ExperimentConfig`` validation, ``run_experiment`` and
``training_pipeline.promote``. It buys exactly one thing: a known answer to
compare the pipeline's behaviour against. It says nothing about live
performance and a run carrying it must never be promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training_pipeline.config import PlantedSignalConfig

#: Marker every diagnostic experiment name must start with. Enforced in
#: ExperimentConfig validation rather than suggested in a comment, because the
#: whole risk this module carries is a run being mistaken for a real one.
DIAGNOSTIC_NAME_PREFIX = "diag_planted"


@dataclass(frozen=True)
class PlantedSignalResult:
    """What was asked for, and what the data actually ended up carrying.

    Both are recorded because they are not the same number. Rows are dropped
    after the feature is generated (cleaning's per-row NaN budget, then the
    target dropna), so the realised correlation drifts slightly from the
    requested one -- and a large drift is a finding, not a rounding error.
    """

    column: str
    #: The configured target-variance fraction.
    requested_variance_explained: float
    #: Pearson correlation between the planted feature and the target, measured
    #: on the rows that survived into the modelling frame.
    measured_correlation: float
    #: ``measured_correlation ** 2`` -- the fraction of target variance the
    #: feature actually explains on those rows.
    measured_variance_explained: float
    n_rows: int
    seed: int

    def summary(self) -> dict[str, float | str | int]:
        return {
            "planted_column": self.column,
            "planted_requested_variance_explained": (
                self.requested_variance_explained
            ),
            "planted_measured_correlation": self.measured_correlation,
            "planted_measured_variance_explained": (
                self.measured_variance_explained
            ),
            "planted_n_rows": self.n_rows,
            "planted_seed": self.seed,
        }


def build_planted_signal(
    target: pd.Series, *, config: PlantedSignalConfig
) -> np.ndarray:
    """The synthetic feature, as one array aligned to ``target``'s rows.

    Uses its own ``default_rng(config.seed)`` rather than the experiment's
    ``random_state``: the planted feature must be identical across cells that
    differ only in signal strength, and it must not shift when the model seed
    is changed to measure fit noise.
    """
    values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(
            "Cannot plant a signal against a target with no finite values."
        )

    std = float(np.nanstd(values[finite]))
    if std == 0.0:
        raise ValueError(
            "Cannot plant a signal against a constant target: standardising it "
            "would divide by zero."
        )
    mean = float(np.nanmean(values[finite]))

    # Standardised target. Non-finite targets get 0 so they contribute noise
    # only; those rows are dropped downstream anyway.
    z = np.zeros_like(values)
    z[finite] = (values[finite] - mean) / std

    rng = np.random.default_rng(config.seed)
    noise = rng.standard_normal(len(values))

    rho = float(np.sqrt(config.variance_explained))
    return rho * z + np.sqrt(1.0 - rho**2) * noise


def measure_planted_signal(
    frame: pd.DataFrame, *, target_col: str, config: PlantedSignalConfig
) -> PlantedSignalResult:
    """Measure what the planted feature actually carries in ``frame``.

    Called on the FINAL modelling frame, after cleaning and the target dropna,
    so the number describes the data the model is really given rather than the
    data the feature was generated against.
    """
    planted = pd.to_numeric(frame[config.column], errors="coerce")
    target = pd.to_numeric(frame[target_col], errors="coerce")
    usable = planted.notna() & target.notna()

    if usable.sum() < 2 or planted[usable].std() == 0:
        correlation = 0.0
    else:
        correlation = float(planted[usable].corr(target[usable]))
        if not np.isfinite(correlation):
            correlation = 0.0

    return PlantedSignalResult(
        column=config.column,
        requested_variance_explained=config.variance_explained,
        measured_correlation=correlation,
        measured_variance_explained=correlation**2,
        n_rows=int(usable.sum()),
        seed=config.seed,
    )


def planted_feature_importance(
    booster_scores: dict[str, dict[str, float]], *, column: str, n_features: int
) -> dict[str, float]:
    """Where the planted feature ranked among all features, per importance type.

    Reported as a RANK as well as a raw score because the raw numbers are not
    comparable between runs: total gain scales with the number of trees, which
    is itself a tuned hyperparameter here. "1st of 1458 by gain" transfers
    between cells; "gain 41.7" does not.

    A feature XGBoost never split on is absent from the score dict entirely --
    that is reported as a zero score and a rank of ``n_features``, never as a
    missing value, so "the model ignored it" and "the artifact lost it" cannot
    be confused.
    """
    out: dict[str, float] = {"n_features": float(n_features)}
    for kind, scores in booster_scores.items():
        value = float(scores.get(column, 0.0))
        # Rank 1 = most important. Features scoring above ours, plus one.
        better = sum(1 for other in scores.values() if other > value)
        out[f"planted_{kind}"] = value
        out[f"planted_{kind}_rank"] = float(better + 1) if value > 0 else float(
            n_features
        )
        out[f"planted_{kind}_used"] = float(value > 0)
    return out
