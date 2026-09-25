"""Recover the hyperparameters a past run found, without re-tuning.

Every run already persists its Optuna trials (``optuna_selected_trial.json``
and ``optuna_best_trial.json``), so a study that took hours never has to be
repeated just to reuse what it discovered. This module reads those artifacts
back and renders them as a YAML block you can paste into an experiment file
under ``optuna.fixed_params``, which makes the next run skip tuning entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training_pipeline.tuning import NON_XGB_TRIAL_PARAMS, USE_SAMPLE_WEIGHT_PARAM


@dataclass
class RunHyperparameters:
    """Hyperparameters recovered from a saved experiment run."""

    run_dir: Path
    #: XGBoost parameters only -- training-protocol params are split out below.
    params: dict[str, Any]
    n_estimators: int
    sample_weight_lambda: float | None
    #: "selected" (lexicographic pick) or "best" (lowest CV MAE).
    source: str
    trial_number: int | None
    #: The training window the trial selected, when it was tuned. Printed as
    #: walk_forward.train_games so a reused configuration refits on the same
    #: amount of history the hyperparameters were chosen under.
    train_games: int | None = None
    cv_mae: float | None = None
    #: Set instead of cv_mae for classifier runs, whose trial value is log loss.
    cv_logloss: float | None = None
    cv_ou_acc: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_yaml_block(self) -> str:
        """A ready-to-paste ``optuna:`` block that skips tuning."""
        lines = [
            f"# Recovered from {self.run_dir.name}"
            f" (trial {self.trial_number}, {self.source}).",
        ]
        if self.cv_logloss is not None:
            summary = f"# CV log loss {self.cv_logloss:.5f}"
            if self.cv_ou_acc is not None:
                summary += f", CV OU accuracy {self.cv_ou_acc:.2%}"
            lines.append(summary)
        elif self.cv_mae is not None:
            summary = f"# CV MAE {self.cv_mae:.4f}"
            if self.cv_ou_acc is not None:
                summary += f", CV OU accuracy {self.cv_ou_acc:.2%}"
            lines.append(summary)
        lines += [
            "optuna:",
            "  fixed_params:",
        ]
        for key, value in sorted(self.params.items()):
            lines.append(f"    {key}: {value!r}")
        lines.append(f"  fixed_n_estimators: {self.n_estimators}")
        if self.train_games is not None:
            lines.append("walk_forward:")
            lines.append(f"  train_games: {self.train_games}")
        if self.sample_weight_lambda is not None:
            lines.append(
                f"  fixed_sample_weight_lambda: {self.sample_weight_lambda!r}"
            )
        return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_run_hyperparameters(
    run_dir: str | Path, *, prefer_selected: bool = True
) -> RunHyperparameters:
    """Load the hyperparameters a run settled on.

    Prefers the lexicographically *selected* trial (best MAE within tolerance,
    then best OU accuracy) over the raw lowest-MAE trial, because that is the
    one the run actually used. Pass ``prefer_selected=False`` to force the
    lowest-MAE trial instead.
    """
    run_dir = Path(run_dir)
    selected = _read_json(run_dir / "optuna_selected_trial.json")
    best = _read_json(run_dir / "optuna_best_trial.json")

    payload: dict[str, Any] | None = None
    source = ""
    if prefer_selected and selected:
        payload, source = selected.get("selected_trial"), "selected"
    if payload is None and best:
        payload, source = best.get("best_trial"), "best"
    if payload is None and selected:
        payload, source = selected.get("selected_trial"), "selected"

    if not payload:
        raise FileNotFoundError(
            f"No Optuna trial artifacts found in {run_dir}. The run may have "
            "used optuna.fixed_params (nothing to recover), or been saved "
            "without experiment artifacts."
        )

    trial_params = dict(payload.get("params") or {})
    user_attrs = dict(payload.get("user_attrs") or {})

    params = {k: v for k, v in trial_params.items() if k not in NON_XGB_TRIAL_PARAMS}
    lambda_ = trial_params.get("sample_weight_lambda") or user_attrs.get(
        "sample_weight_lambda"
    )
    # Respect a trial that decided against weighting altogether.
    if trial_params.get(USE_SAMPLE_WEIGHT_PARAM) is False:
        lambda_ = None

    # Same order as tuning.n_estimators_from_trial, which the holdout backtest
    # uses -- a promoted model must fit the rounds the holdout measured.
    # Tuned value FIRST. A run that tuned n_estimators records it in params and
    # writes no best_iteration attrs at all. An early-stopping run records the
    # best_iteration attrs AND user_attrs["n_estimators"], but the latter is
    # only the early-stopping cap (e.g. 1000 against a median of 51), so it
    # must come last: reading it before the median refitted 20x the rounds.
    n_estimators = (
        trial_params.get("n_estimators")
        or user_attrs.get("median_best_iteration")
        or user_attrs.get("mean_best_iteration")
        or user_attrs.get("n_estimators")
    )
    if not n_estimators:
        raise ValueError(
            f"Could not determine n_estimators from {run_dir}: the trial "
            "recorded neither a tuned n_estimators nor a median/mean "
            "best_iteration."
        )

    return RunHyperparameters(
        run_dir=run_dir,
        params=params,
        # No max(50, ...) floor: it used to raise a selected 10-round model to
        # 50 rounds, silently, in 16 of the 38 runs under artifacts/experiments.
        n_estimators=int(round(float(n_estimators))),
        sample_weight_lambda=float(lambda_) if lambda_ is not None else None,
        source=source,
        trial_number=payload.get("number"),
        # No blind fallback to payload["value"]: for a classifier that value is
        # log loss, and filing it under cv_mae would print "CV MAE 0.6931" in
        # the recovered-hyperparameter block.
        train_games=(
            trial_params.get("train_games") or user_attrs.get("train_games")
        ),
        cv_mae=user_attrs.get("pooled_mae", user_attrs.get("mean_mae")),
        cv_logloss=user_attrs.get("mean_logloss"),
        cv_ou_acc=user_attrs.get("mean_ou_acc"),
        extras=user_attrs,
    )


def find_best_run_hyperparameters(
    root_dir: str | Path = Path("artifacts/experiments"),
    *,
    rank_by: str = "roi",
) -> RunHyperparameters:
    """Hyperparameters from the best run on the leaderboard.

    Defaults to ranking by ROI, i.e. "the run that actually made the most
    money", rather than by CV MAE.
    """
    from training_pipeline.leaderboard import build_leaderboard

    leaderboard = build_leaderboard(root_dir, sort_by=rank_by)
    if leaderboard.empty:
        raise FileNotFoundError(f"No runs found under {root_dir}.")

    root = Path(root_dir)
    for run_name in leaderboard["run_name"]:
        try:
            return load_run_hyperparameters(root / str(run_name))
        except (FileNotFoundError, ValueError):
            continue
    raise FileNotFoundError(
        f"No run under {root_dir} has recoverable Optuna hyperparameters."
    )
