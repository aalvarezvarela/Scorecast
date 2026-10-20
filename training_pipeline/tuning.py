"""Strategy/adapter layer resolving the asymmetry between optuna_total_points.py
and optuna_error_line.py.

Confirmed by direct inspection: optuna_total_points.py always needs `line_col`
(it pulls the betting line from X[line_col] for OU-accuracy scoring) and has
no sample-weight support at all. optuna_error_line.py never takes `line_col`
(it scores directly off the sign of the predicted error) but fully supports
recency sample-weighting (sample_weight/sample_weight_dates/
sample_weight_lambda/tune_sample_weight_lambda).

get_strategy(config) is the only branch point in this whole package -- every
other module calls strategy.<method>(...) and never imports optuna_total_points
or optuna_error_line directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling import optuna_error_line as _error_line
from nba_ou.modeling import optuna_total_points as _total_points
from nba_ou.modeling.modeling import build_recency_sample_weights
from nba_ou.modeling.scorers import (
    evaluate_error_thresholds,
    evaluate_total_points_thresholds,
    over_under_betting_accuracy_error_line,
    over_under_betting_accuracy_total_points,
)
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from training_pipeline.calibration import brier_score, log_loss
from training_pipeline.config import (
    ExperimentConfig,
    PredictionStrategy,
    SampleWeightConfig,
    SearchSpaceConfig,
    TieTolerancePolicy,
)
from training_pipeline.splits import TRAIN_GAMES_PARAM, Split, SplitProvider


class TargetFamilyStrategy(Protocol):
    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        config: ExperimentConfig,
        splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
        split_provider: SplitProvider | None = None,
        dates: pd.Series | None = None,
    ) -> optuna.Study: ...

    #: The trial user-attr selection ranks on, in whatever aggregation this
    #: strategy was built with. Named by the strategy rather than inferred from
    #: which attrs happen to exist, so a run cannot be ranked one way while a
    #: tolerance is derived from the other.
    @property
    def primary_metric_key(self) -> str: ...

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial: ...

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame: ...

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame: ...

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None,
        trial: optuna.trial.FrozenTrial | None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBRegressor: ...

    def evaluate_holdout(
        self,
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]: ...


def build_xgb_params(
    trial: optuna.Trial,
    space: SearchSpaceConfig,
    *,
    objective_name: str,
    random_state: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    """Sample XGBoost parameters from a configurable search space.

    ``space.n_estimators_range`` switches the boosting rounds from "a ceiling
    plus per-fold early stopping" to an ordinary sampled hyperparameter; see
    ``OptunaConfig.tune_n_estimators`` for the measurements behind that.

    The suggest_* calls are issued in exactly the same order, with the same
    names and distributions, as
    ``nba_ou.modeling.optuna_total_points.build_xgb_params_total_points``. That
    ordering matters: a seeded TPE sampler draws per-parameter in call order,
    so any reordering would change results even with identical ranges. A test
    asserts both builders yield identical draws under the same seed.
    """
    params: dict[str, Any] = {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": device,
        "objective": objective_name,
        "eval_metric": "mae",
        "max_depth": trial.suggest_int(
            "max_depth", space.max_depth.low, space.max_depth.high, log=space.max_depth.log
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            space.min_child_weight.low,
            space.min_child_weight.high,
            log=space.min_child_weight.log,
        ),
        "gamma": trial.suggest_float(
            "gamma", space.gamma.low, space.gamma.high, log=space.gamma.log
        ),
        "subsample": trial.suggest_float(
            "subsample", space.subsample.low, space.subsample.high, log=space.subsample.log
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            space.colsample_bytree.low,
            space.colsample_bytree.high,
            log=space.colsample_bytree.log,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space.learning_rate.low,
            space.learning_rate.high,
            log=space.learning_rate.log,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", space.reg_alpha.low, space.reg_alpha.high, log=space.reg_alpha.log
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            space.reg_lambda.low,
            space.reg_lambda.high,
            log=space.reg_lambda.log,
        ),
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if space.n_estimators_range is None:
        # Legacy: a high ceiling that each fold's own early stopping cuts down.
        params["n_estimators"] = space.n_estimators
        params["early_stopping_rounds"] = space.early_stopping_rounds
        return params

    # Tuned: one value per trial, held across every fold, and no early stopping
    # anywhere -- so the rounds the trial was scored at are exactly the rounds
    # the holdout walk-forward and the production refit will use.
    #
    # Suggested LAST so that adding this dimension leaves the draw order of
    # every parameter above unchanged; a seeded TPE sampler draws per-parameter
    # in call order, so inserting it earlier would silently move every other
    # range's samples.
    params["n_estimators"] = trial.suggest_int(
        "n_estimators",
        space.n_estimators_range.low,
        space.n_estimators_range.high,
        log=space.n_estimators_range.log,
    )
    return params


#: Optuna parameter name for the "weight at all?" decision.
USE_SAMPLE_WEIGHT_PARAM = "use_sample_weight"

#: Params belonging to the training protocol, not to XGBoost itself.
#: ``train_games`` is the size of the training window, which is a property of
#: the CV layout rather than of the booster -- passing it to XGBRegressor would
#: be silently accepted and ignored.
NON_XGB_TRIAL_PARAMS = frozenset(
    {"sample_weight_lambda", USE_SAMPLE_WEIGHT_PARAM, TRAIN_GAMES_PARAM}
)


def n_estimators_from_trial(trial: optuna.trial.FrozenTrial) -> int:
    """Boosting rounds for the final fits, straight from the trial.

    Order of preference, and why there is no ``max(50, ...)`` floor any more:

    1. ``params["n_estimators"]`` -- the tuned value. Used verbatim. The floor
       used to override this: it turned a selected 10.5 rounds into 50 in 16 of
       the 38 runs in artifacts/experiments, i.e. the CV's answer was discarded
       whenever it was small, silently and without a test covering it.
    2. ``median_best_iteration`` / ``mean_best_iteration`` -- legacy runs, where
       the rounds came from fold-level early stopping. Kept so an old trial can
       still be replayed, but read the caveat: within one selected trial folds
       stopped anywhere between 2 and 922 rounds, so this median summarises a
       quantity with a 100x spread.
    """
    tuned = trial.params.get("n_estimators")
    if tuned:
        return int(tuned)

    legacy = trial.user_attrs.get("median_best_iteration") or trial.user_attrs.get(
        "mean_best_iteration"
    )
    if not legacy:
        raise ValueError(
            f"Trial {trial.number} records no n_estimators: neither a tuned "
            "parameter nor a median/mean best_iteration from early stopping."
        )
    return int(round(float(legacy)))


def resolve_final_params(
    trial: optuna.trial.FrozenTrial, config: ExperimentConfig
) -> tuple[dict[str, Any], int, float | None]:
    """Split a tuned trial into (xgb params, boosting rounds, decay rate)."""
    params = {k: v for k, v in trial.params.items() if k not in NON_XGB_TRIAL_PARAMS}
    lambda_ = trial.params.get("sample_weight_lambda")

    # A trial that explicitly chose not to weight must not have weighting
    # reinstated by the config fallback below.
    chose_unweighted = trial.params.get(USE_SAMPLE_WEIGHT_PARAM) is False
    if chose_unweighted:
        lambda_ = None
    elif lambda_ is None and config.sample_weight.enabled:
        lambda_ = config.sample_weight.lambda_

    return params, n_estimators_from_trial(trial), lambda_


def fit_final_model(
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    params: dict[str, Any],
    n_estimators: int,
    config: ExperimentConfig,
    dates_dev: pd.Series | None = None,
    sample_weight_lambda: float | None = None,
    random_state: int | None = None,
) -> XGBModel:
    """Fit one model on fixed hyperparameters.

    Shared by both target families, which is what makes recency weighting
    available to TOTAL_POINTS: upstream's fit_best_xgb_total_points has no
    sample_weight parameter, so routing through it would silently drop the
    weights.

    ``random_state`` overrides ``config.random_state`` for this fit alone --
    the mechanism behind repeating an evaluation under several seeds. It is
    applied after ``params`` for two reasons: ``_build_static_params`` has
    already seeded from the config (so without this every "different" seed
    would fit identically), and a user-supplied ``optuna.fixed_params`` may
    legitimately contain its own ``random_state``.
    """
    final_params = {
        **_build_static_params(config),
        **params,
        "n_estimators": n_estimators,
    }
    if random_state is not None:
        final_params["random_state"] = random_state
    # A trial's params were sampled under a regression eval_metric default in
    # build_xgb_params; the static params above already hold the right one for
    # this strategy, so re-assert it after the merge.
    final_params["eval_metric"] = "logloss" if config.is_classifier else "mae"
    # No eval_set here, so early stopping cannot apply.
    final_params.pop("early_stopping_rounds", None)

    sample_weight = None
    if sample_weight_lambda is not None and dates_dev is not None:
        sample_weight = build_recency_sample_weights(
            dates_dev, lambda_=float(sample_weight_lambda)
        ).to_numpy(dtype=float)

    estimator_cls = XGBClassifier if config.is_classifier else XGBRegressor
    model = estimator_cls(**final_params)
    model.fit(X_dev, y_dev, sample_weight=sample_weight, verbose=False)
    return model


def _build_static_params(
    config: ExperimentConfig, *, random_state: int | None = None
) -> dict[str, Any]:
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": config.device,
        "objective": config.optuna.objective_name,
        # Early stopping and Optuna both track the loss that matches the model
        # class. Leaving "mae" on a classifier would early-stop on a metric
        # that ignores probability quality entirely.
        "eval_metric": "logloss" if config.is_classifier else "mae",
        "random_state": config.random_state if random_state is None else random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def _resolve_trial_sample_weight_lambda(
    trial: optuna.Trial, sample_weight: SampleWeightConfig
) -> float | None:
    """Decay rate for this trial, or None for no weighting.

    When ``allow_unweighted`` is on, a categorical decides first whether to
    weight at all; the decay rate is only sampled if it says yes. That is a
    conditional search space, which TPE handles natively, and it lets "no
    weighting" compete as a real option instead of being unreachable at the
    bottom of a log-uniform range.
    """
    if not sample_weight.enabled:
        return None
    if not sample_weight.tune_lambda:
        return sample_weight.lambda_

    if sample_weight.allow_unweighted:
        use_weighting = trial.suggest_categorical(USE_SAMPLE_WEIGHT_PARAM, [True, False])
        if not use_weighting:
            return None

    low, high = sample_weight.lambda_bounds
    return float(trial.suggest_float("sample_weight_lambda", low, high, log=True))


@dataclass
class PooledPredictions:
    """Every validation prediction of one trial, concatenated across folds."""

    y_true: np.ndarray
    y_pred: np.ndarray
    #: The betting line per game. Only total_points needs it (its OU accuracy is
    #: defined against the line); None for the other strategies.
    line: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.y_true))


@dataclass
class _PooledCollector:
    """Accumulates fold predictions so one metric can be computed over all games.

    Exists because ``mean`` over folds weights a 4-game fold the same as a
    12-game one. Under rolling_origin a fold is a few game-days and its size is
    whatever the NBA schedule decided, so the fold means quietly reweight the
    objective by calendar accident.
    """

    line_col: str | None = None
    _y_true: list[np.ndarray] = field(default_factory=list)
    _y_pred: list[np.ndarray] = field(default_factory=list)
    _line: list[np.ndarray] = field(default_factory=list)

    def add(
        self, y_valid: pd.Series, y_pred: np.ndarray, X_valid: pd.DataFrame
    ) -> None:
        self._y_true.append(pd.to_numeric(y_valid, errors="coerce").to_numpy(float))
        self._y_pred.append(np.asarray(y_pred, dtype=float))
        if self.line_col is not None:
            self._line.append(
                pd.to_numeric(X_valid[self.line_col], errors="coerce").to_numpy(float)
            )

    @property
    def n_games(self) -> int:
        return int(sum(len(chunk) for chunk in self._y_true))

    def pooled(self) -> PooledPredictions:
        return PooledPredictions(
            y_true=np.concatenate(self._y_true),
            y_pred=np.concatenate(self._y_pred),
            line=np.concatenate(self._line) if self._line else None,
        )


def _resolve_fold_set(
    trial: optuna.Trial,
    *,
    config: ExperimentConfig,
    splits: list[Split] | None,
    split_provider: SplitProvider | None,
) -> tuple[list[Split], int | None]:
    """Fold set for this trial, plus the training window it was built at.

    The window is sampled here, BEFORE any XGBoost parameter, so the draw order
    a seeded sampler sees is fixed and documented. When it is not tuned no
    ``suggest_*`` call happens at all, which is what keeps pre-existing studies
    reproducible down to the byte.
    """
    if split_provider is not None:
        train_games = split_provider.suggest_train_games(trial)
        return split_provider.splits_for(train_games), train_games
    if splits is None:
        raise ValueError("run_objective needs either splits or a split_provider.")
    return splits, config.walk_forward.train_games


def run_objective(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
    evaluate_fold: Callable[..., Any],
    splits: list[Split] | None = None,
    split_provider: SplitProvider | None = None,
    pooled_metrics: Callable[[PooledPredictions], dict[str, float]] | None = None,
    pooled_line_col: str | None = None,
    dates: pd.Series | None = None,
) -> float:
    """Validation MAE across time-aware folds, minimised by Optuna.

    Two aggregations, chosen by ``optuna.objective_aggregation``: the mean of the
    folds' own MAEs (legacy), or one MAE pooled over every validation game. Both
    record ``mean_*`` user attrs; the pooled mode additionally records
    ``pooled_*`` and returns the pooled value.

    Two fitting modes, chosen by whether ``n_estimators`` is tuned. In the legacy
    mode each fold early-stops on the very games that then score it. In the tuned
    mode there is no ``eval_set`` at all: the sampled round count is trained in
    full on every fold, which is exactly what the holdout walk-forward and the
    production refit do.

    Per-fold metrics still come from upstream's own ``evaluate_fold_*`` so those
    definitions cannot drift; the pooled metrics call the same scorer functions
    upstream calls, on the concatenated arrays.
    """
    fold_set, train_games = _resolve_fold_set(
        trial, config=config, splits=splits, split_provider=split_provider
    )
    params = build_xgb_params(
        trial,
        config.optuna.search_space,
        objective_name=config.optuna.objective_name,
        random_state=config.random_state,
        device=config.device,
    )
    weight_lambda = _resolve_trial_sample_weight_lambda(trial, config.sample_weight)
    use_early_stopping = config.uses_fold_early_stopping
    pool_objective = config.pools_objective and pooled_metrics is not None

    fold_metrics: list[Any] = []
    collector = _PooledCollector(line_col=pooled_line_col)
    for fold_num, (train_idx, valid_idx) in enumerate(fold_set, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        sample_weight = None
        if weight_lambda is not None and dates is not None:
            sample_weight = build_recency_sample_weights(
                dates.iloc[train_idx], lambda_=float(weight_lambda)
            ).to_numpy(dtype=float)

        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            # No eval_set once the rounds are tuned: there is nothing to stop
            # early, and handing the fold's own answers to fit() is precisely
            # the bias this mode removes.
            eval_set=[(X_valid, y_valid)] if use_early_stopping else None,
            verbose=False,
        )
        fold_metrics.append(
            evaluate_fold(model, X_valid, y_valid, fold=fold_num, n_train=len(X_train))
        )
        collector.add(y_valid, model.predict(X_valid), X_valid)

        running = (
            float(mean_absolute_error(*_pooled_arrays(collector)))
            if pool_objective
            else float(np.mean([m.mae for m in fold_metrics]))
        )
        trial.report(running, step=fold_num)
        if trial.should_prune():
            # Recorded so a pruned trial can be audited: "killed after 7 folds
            # / 210 games" is a judgement you can check, "pruned" is not.
            trial.set_user_attr("pruned_after_folds", fold_num)
            trial.set_user_attr("pruned_after_games", collector.n_games)
            raise optuna.TrialPruned()

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(m, attr) for m in fold_metrics]))

    mean_mae = _mean("mae")
    if weight_lambda is not None:
        trial.set_user_attr("sample_weight_lambda", float(weight_lambda))
    trial.set_user_attr("mean_mae", mean_mae)
    trial.set_user_attr("mean_rmse", _mean("rmse"))
    trial.set_user_attr("mean_r2", _mean("r2"))
    trial.set_user_attr("mean_ou_acc", _mean("ou_accuracy"))
    trial.set_user_attr("mean_ou_acc_edge_2", _mean("ou_accuracy_edge_2"))
    trial.set_user_attr("mean_ou_acc_edge_3", _mean("ou_accuracy_edge_3"))
    trial.set_user_attr("mean_ou_acc_edge_4", _mean("ou_accuracy_edge_4"))
    _record_protocol_attrs(
        trial,
        config=config,
        fold_metrics=fold_metrics,
        n_folds=len(fold_set),
        n_games=collector.n_games,
        train_games=train_games,
        params=params,
    )
    trial.set_user_attr("fold_metrics", [vars(m) for m in fold_metrics])

    if not pool_objective:
        return mean_mae

    assert pooled_metrics is not None
    pooled = collector.pooled()
    for name, value in pooled_metrics(pooled).items():
        trial.set_user_attr(f"pooled_{name}", value)
    return float(trial.user_attrs["pooled_mae"])


def _pooled_arrays(collector: _PooledCollector) -> tuple[np.ndarray, np.ndarray]:
    pooled = collector.pooled()
    return pooled.y_true, pooled.y_pred


def _record_protocol_attrs(
    trial: optuna.Trial,
    *,
    config: ExperimentConfig,
    fold_metrics: list[Any],
    n_folds: int,
    n_games: int,
    train_games: int | None,
    params: dict[str, Any],
) -> None:
    """Record what the trial actually trained, so nothing has to be re-derived.

    ``median_best_iteration`` is written ONLY in the legacy early-stopping mode.
    Once the rounds are tuned there is no best iteration -- xgboost reports none
    without an eval_set, and upstream's fold evaluator then falls back to
    ``n_estimators - 1``, which would look like a measured stopping point and be
    off by one if anything downstream read it. ``n_estimators`` lives in
    ``trial.params``, which is where the resolver looks first.
    """
    trial.set_user_attr("n_folds", int(n_folds))
    trial.set_user_attr("n_validation_games", int(n_games))
    trial.set_user_attr("n_estimators", int(params["n_estimators"]))
    if train_games is not None:
        trial.set_user_attr("train_games", int(train_games))
    if not config.uses_fold_early_stopping:
        return
    trial.set_user_attr(
        "mean_best_iteration",
        int(round(np.mean([m.best_iteration for m in fold_metrics]))),
    )
    trial.set_user_attr(
        "median_best_iteration", int(np.median([m.best_iteration for m in fold_metrics]))
    )


def _create_study(
    config: ExperimentConfig, *, n_folds: int | None = None
) -> optuna.Study:
    """One study construction path, persistent or in-memory.

    ``n_folds`` sizes the pruner's warmup. A pruning step is one fold, so a
    fixed warmup means something different at 12 folds of ~50 games than at 28
    folds of ~30 -- see ExperimentConfig.resolve_pruner_warmup_steps, which
    floors the derived value at the historical 5 so no existing config becomes
    less patient.
    """
    kwargs: dict[str, Any] = {}
    if config.optuna.persistent_storage:
        kwargs = {
            "storage": _persistent_storage_url(config),
            "load_if_exists": config.optuna.load_if_exists,
        }
    return optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=config.random_state),
        pruner=MedianPruner(
            n_warmup_steps=config.resolve_pruner_warmup_steps(n_folds)
        ),
        study_name=config.resolved_study_name,
        **kwargs,
    )


def _provider_from(
    splits: list[Split] | None,
    split_provider: SplitProvider | None,
    config: ExperimentConfig,
) -> SplitProvider:
    """Accept either shape at the tune() boundary.

    Existing callers (and several tests) pass a plain list of splits. Wrapping it
    in a provider whose window is fixed keeps one code path inside the objective
    without changing what those callers get: a provider with no
    ``train_games_choices`` issues no ``suggest_*`` call, so the sampler sees the
    identical sequence of draws it always did.
    """
    if split_provider is not None:
        return split_provider
    if splits is None:
        raise ValueError("Provide either splits or split_provider.")
    return SplitProvider(
        fold_info=pd.DataFrame(),
        default_train_games=config.walk_forward.train_games,
        fixed_splits=splits,
    )


def pooled_total_points_metrics(
    pooled: PooledPredictions,
) -> dict[str, float]:
    """Pooled metrics for a total-points regressor.

    Calls the same scorer functions ``evaluate_fold_total_points`` calls, on the
    concatenated arrays, so a pooled number and a per-fold number can never mean
    two different things.
    """
    assert pooled.line is not None, "total_points pooling needs the betting line"
    return {
        "mae": float(mean_absolute_error(pooled.y_true, pooled.y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(pooled.y_true, pooled.y_pred))),
        "r2": float(r2_score(pooled.y_true, pooled.y_pred)),
        "ou_acc": float(
            over_under_betting_accuracy_total_points(
                y_true=pooled.y_true,
                y_pred=pooled.y_pred,
                betting_line=pooled.line,
            )
        ),
        "n_games": float(len(pooled)),
    }


def pooled_line_error_metrics(pooled: PooledPredictions) -> dict[str, float]:
    """Pooled metrics for a line-error regressor."""
    return {
        "mae": float(mean_absolute_error(pooled.y_true, pooled.y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(pooled.y_true, pooled.y_pred))),
        "r2": float(r2_score(pooled.y_true, pooled.y_pred)),
        "ou_acc": float(
            over_under_betting_accuracy_error_line(
                y_true_error=pooled.y_true,
                y_pred_error=pooled.y_pred,
            )
        ),
        "n_games": float(len(pooled)),
    }


def _optimize(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    config: ExperimentConfig,
) -> optuna.Study:
    study.optimize(
        objective,
        n_trials=config.optuna.n_trials,
        timeout=config.optuna.timeout,
        n_jobs=1,
        show_progress_bar=True,
    )
    return study


def _persistent_storage_url(config: ExperimentConfig) -> str:
    """A stable (non-timestamped) location so re-running the same config
    resumes the same study (that's the point of `load_if_exists=True`) rather
    than starting a fresh study every run the way the timestamped per-run
    artifact directory does.

    The filename includes ExperimentConfig.fingerprint(), so a study can only
    be resumed by a config whose trials are actually comparable: change the
    CSV, cleaning thresholds, fold layout or XGBoost objective and you get a
    new database instead of silently appending incomparable trials to the old
    one. Renaming the experiment or changing n_trials/timeout still resumes.
    """
    db_dir = config.experiment_root_dir / "_optuna_studies"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_name = (
        f"{config.experiment_name}_{config.family.value}"
        f"_{config.fingerprint()}.db"
    )
    db_path = db_dir / db_name
    return f"sqlite:///{db_path.resolve().as_posix()}"


@dataclass
class TotalPointsStrategy:
    line_col: str
    #: Rank trials on metrics pooled over every validation game rather than on
    #: the mean of the folds' own metrics. Set from
    #: ``optuna.objective_aggregation`` in get_strategy, and carried on the
    #: strategy because select_best_trial/summarize_* take no config -- the
    #: alternative, inferring it from which user_attrs happen to be present,
    #: would silently rank a run one way or the other with nothing recording
    #: which.
    pooled: bool = False

    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        config: ExperimentConfig,
        splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
        split_provider: SplitProvider | None = None,
        dates: pd.Series | None = None,
    ) -> optuna.Study:
        def evaluate_fold(
            model: XGBRegressor,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
            *,
            fold: int,
            n_train: int,
        ) -> Any:
            return _total_points.evaluate_fold_total_points(
                model, X_valid, y_valid, self.line_col, fold=fold, n_train=n_train
            )

        provider = _provider_from(splits, split_provider, config)
        return _optimize(
            _create_study(config, n_folds=provider.n_folds),
            lambda trial: run_objective(
                trial,
                X=X,
                y=y,
                split_provider=provider,
                config=config,
                evaluate_fold=evaluate_fold,
                pooled_metrics=pooled_total_points_metrics,
                pooled_line_col=self.line_col,
                # Required, not optional: run_objective skips weighting entirely
                # when dates are absent, so omitting this would let Optuna draw
                # a sample_weight_lambda, record it on the trial, and score the
                # trial on unweighted fits -- selecting a training option it
                # never tested, which the final model would then apply.
                dates=dates,
            ),
            config,
        )

    @property
    def primary_metric_key(self) -> str:
        return "pooled_mae" if self.pooled else "mean_mae"

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial:
        if self.pooled:
            return select_best_trial_lexicographic_pooled(
                study,
                mae_tolerance_abs=mae_tolerance_abs,
                mae_tolerance_pct=mae_tolerance_pct,
            )
        return _total_points.select_best_trial_lexicographic(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame:
        if self.pooled:
            return summarize_trials_pooled(study, keys=_POOLED_REGRESSOR_KEYS)
        return _total_points.summarize_optuna_trials(study)

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame:
        if self.pooled:
            return summarize_candidates_pooled(
                study,
                primary_key="pooled_mae",
                tolerance_abs=mae_tolerance_abs,
                tolerance_pct=mae_tolerance_pct,
                keys=_POOLED_REGRESSOR_KEYS,
            )
        return _total_points.summarize_lexicographic_candidates(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None = None,
        trial: optuna.trial.FrozenTrial | None = None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBRegressor:
        selected = trial if trial is not None else study.best_trial  # type: ignore[union-attr]
        params, n_estimators, lambda_ = resolve_final_params(selected, config)
        return fit_final_model(
            X_dev=X_dev,
            y_dev=y_dev,
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev,
            sample_weight_lambda=lambda_,
        )

    def evaluate_holdout(
        self,
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        return evaluate_total_points_thresholds(
            model=model,
            X_test=X_test,
            y_test_total=y_test,
            line_col=self.line_col,
        )


@dataclass
class LineErrorStrategy:
    """Residual regression: the target IS the edge.

    Serves BOTH ``line_error_regressor`` (totals) and ``spread_error_regressor``
    (spread). Nothing in here is market-specific -- it never sees a line, because
    a residual model's prediction is already the quantity being bet -- so the two
    markets share it rather than each carrying a near-identical copy.
    """

    sample_weight: SampleWeightConfig
    #: Rank trials on metrics pooled over every validation game rather than on
    #: the mean of the folds' own metrics. Set from
    #: ``optuna.objective_aggregation`` in get_strategy, and carried on the
    #: strategy because select_best_trial/summarize_* take no config -- the
    #: alternative, inferring it from which user_attrs happen to be present,
    #: would silently rank a run one way or the other with nothing recording
    #: which.
    pooled: bool = False

    def _tune_sample_weight_kwargs(self, *, dates: pd.Series | None) -> dict[str, Any]:
        if not self.sample_weight.enabled:
            return {}
        kwargs: dict[str, Any] = {
            "sample_weight_lambda": self.sample_weight.lambda_,
            "tune_sample_weight_lambda": self.sample_weight.tune_lambda,
            "sample_weight_lambda_bounds": self.sample_weight.lambda_bounds,
        }
        if dates is not None:
            kwargs["sample_weight_dates"] = dates
        return kwargs

    def _fit_sample_weight_kwargs(self, *, dates: pd.Series | None) -> dict[str, Any]:
        if not self.sample_weight.enabled:
            return {}
        kwargs: dict[str, Any] = {"sample_weight_lambda": self.sample_weight.lambda_}
        if dates is not None:
            kwargs["sample_weight_dates"] = dates
        return kwargs

    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        config: ExperimentConfig,
        splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
        split_provider: SplitProvider | None = None,
        dates: pd.Series | None = None,
    ) -> optuna.Study:
        def evaluate_fold(
            model: XGBRegressor,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
            *,
            fold: int,
            n_train: int,
        ) -> Any:
            return _error_line.evaluate_fold_error_line(
                model, X_valid, y_valid, fold=fold, n_train=n_train
            )

        provider = _provider_from(splits, split_provider, config)
        return _optimize(
            _create_study(config, n_folds=provider.n_folds),
            lambda trial: run_objective(
                trial,
                X=X,
                y=y,
                split_provider=provider,
                config=config,
                evaluate_fold=evaluate_fold,
                pooled_metrics=pooled_line_error_metrics,
                dates=dates,
            ),
            config,
        )

    @property
    def primary_metric_key(self) -> str:
        return "pooled_mae" if self.pooled else "mean_mae"

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial:
        if self.pooled:
            return select_best_trial_lexicographic_pooled(
                study,
                mae_tolerance_abs=mae_tolerance_abs,
                mae_tolerance_pct=mae_tolerance_pct,
            )
        return _error_line.select_best_trial_lexicographic(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame:
        if self.pooled:
            return summarize_trials_pooled(study, keys=_POOLED_REGRESSOR_KEYS)
        return _error_line.summarize_optuna_trials(study)

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame:
        if self.pooled:
            return summarize_candidates_pooled(
                study,
                primary_key="pooled_mae",
                tolerance_abs=mae_tolerance_abs,
                tolerance_pct=mae_tolerance_pct,
                keys=_POOLED_REGRESSOR_KEYS,
            )
        return _error_line.summarize_lexicographic_candidates(
            study,
            mae_tolerance_abs=mae_tolerance_abs,
            mae_tolerance_pct=mae_tolerance_pct,
        )

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None = None,
        trial: optuna.trial.FrozenTrial | None = None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBRegressor:
        selected = trial if trial is not None else study.best_trial  # type: ignore[union-attr]
        params, n_estimators, lambda_ = resolve_final_params(selected, config)
        return fit_final_model(
            X_dev=X_dev,
            y_dev=y_dev,
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev,
            sample_weight_lambda=lambda_,
        )

    def evaluate_holdout(
        self,
        model: XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        return evaluate_error_thresholds(model=model, X_test=X_test, y_test_error=y_test)


@dataclass
class ClassifierFoldMetrics:
    """Per-fold classifier metrics, the analogue of upstream's fold objects."""

    fold: int
    n_train: int
    n_valid: int
    log_loss: float
    brier: float
    #: Fraction of validation games whose side the model called correctly. Every
    #: row has a definite answer here -- pushes were removed when the label was
    #: built -- so this needs no line and no push handling.
    ou_accuracy: float
    best_iteration: int


def classifier_scores(
    y: np.ndarray,
    p_over: np.ndarray,
) -> dict[str, float]:
    """Probability quality and directional accuracy for a set of games.

    Betting settlement is deliberately absent from the tuning objective. With
    the pipeline's flat symmetric odds, ROI is only a rescaling of win rate, and
    using it as another tie-breaker lets the same validation outcomes select the
    trial twice. Profit remains available in the post-hoc holdout/CV evaluator.
    """
    # Pushes were dropped when the label was built, so every row has a definite
    # OVER/UNDER answer. A probability of exactly 0.5 follows the historical
    # convention and calls OVER.
    correct = np.where(p_over >= 0.5, y == 1.0, y == 0.0)

    return {
        "log_loss": log_loss(y, p_over),
        "brier": brier_score(y, p_over),
        "ou_acc": float(correct.mean()) if len(correct) else float("nan"),
        "n_games": float(len(y)),
    }


def _classifier_fold_metrics(
    model: XGBClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    fold: int,
    n_train: int,
) -> ClassifierFoldMetrics:
    y = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=float)
    p_over = np.asarray(model.predict_proba(X_valid), dtype=float)[:, 1]
    scores = classifier_scores(y, p_over)

    best_iteration = int(getattr(model, "best_iteration", 0) or 0) + 1

    return ClassifierFoldMetrics(
        fold=fold,
        n_train=n_train,
        n_valid=int(len(y)),
        log_loss=scores["log_loss"],
        brier=scores["brier"],
        ou_accuracy=scores["ou_acc"],
        best_iteration=best_iteration,
    )


def make_pooled_classifier_metrics() -> Callable[[PooledPredictions], dict[str, float]]:
    """Pooled classifier metrics, keyed the way run_objective expects.

    ``pooled_mae`` is deliberately absent: point error against a 0/1 label is not
    a points error. The classifier objective returns pooled LOG LOSS instead, and
    reads it under its own key.
    """

    def pooled_metrics(pooled: PooledPredictions) -> dict[str, float]:
        return classifier_scores(pooled.y_true, pooled.y_pred)

    return pooled_metrics


def run_classifier_objective(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
    splits: list[Split] | None = None,
    split_provider: SplitProvider | None = None,
    dates: pd.Series | None = None,
) -> float:
    """Mean validation LOG LOSS across time-aware folds, minimised by Optuna.

    Log loss rather than accuracy because it scores the probability, not just
    the side: calling a loser at 51% is barely penalised, calling it at 95% is
    punished hard, and accuracy cannot tell those apart. The betting rule
    compares the probability against a break-even rate, so probability quality
    is exactly what needs optimising.

    Keep the scale in mind when reading trial values: on a ~50/50 outcome a
    perfectly calibrated 55% model scores 0.68814 against 0.69315 for a coin
    flip. Differences between good and bad trials live in the third decimal.

    Under ``objective_aggregation: pooled`` the returned value is log loss over
    every validation game rather than the mean of the folds' own log losses.
    """
    fold_set, train_games = _resolve_fold_set(
        trial, config=config, splits=splits, split_provider=split_provider
    )
    params = build_xgb_params(
        trial,
        config.optuna.search_space,
        objective_name=config.optuna.objective_name,
        random_state=config.random_state,
        device=config.device,
    )
    params["eval_metric"] = "logloss"
    weight_lambda = _resolve_trial_sample_weight_lambda(trial, config.sample_weight)
    use_early_stopping = config.uses_fold_early_stopping
    pool_objective = config.pools_objective

    fold_metrics: list[ClassifierFoldMetrics] = []
    collector = _PooledCollector()
    for fold_num, (train_idx, valid_idx) in enumerate(fold_set, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        sample_weight = None
        if weight_lambda is not None and dates is not None:
            sample_weight = build_recency_sample_weights(
                dates.iloc[train_idx], lambda_=float(weight_lambda)
            ).to_numpy(dtype=float)

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)] if use_early_stopping else None,
            verbose=False,
        )
        fold_metrics.append(
            _classifier_fold_metrics(
                model,
                X_valid,
                y_valid,
                fold=fold_num,
                n_train=len(X_train),
            )
        )
        # P(OVER) travels as "y_pred" so one collector serves every strategy.
        collector.add(
            y_valid, np.asarray(model.predict_proba(X_valid), dtype=float)[:, 1], X_valid
        )

        if pool_objective:
            pooled_so_far = collector.pooled()
            running = classifier_scores(
                pooled_so_far.y_true, pooled_so_far.y_pred
            )["log_loss"]
        else:
            running = float(np.mean([m.log_loss for m in fold_metrics]))
        trial.report(running, step=fold_num)
        if trial.should_prune():
            trial.set_user_attr("pruned_after_folds", fold_num)
            trial.set_user_attr("pruned_after_games", collector.n_games)
            raise optuna.TrialPruned()

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(m, attr) for m in fold_metrics]))

    mean_logloss = _mean("log_loss")
    if weight_lambda is not None:
        trial.set_user_attr("sample_weight_lambda", float(weight_lambda))
    trial.set_user_attr("mean_logloss", mean_logloss)
    trial.set_user_attr("mean_brier", _mean("brier"))
    trial.set_user_attr("mean_ou_acc", _mean("ou_accuracy"))
    _record_protocol_attrs(
        trial,
        config=config,
        fold_metrics=fold_metrics,
        n_folds=len(fold_set),
        n_games=collector.n_games,
        train_games=train_games,
        params=params,
    )
    trial.set_user_attr("fold_metrics", [vars(m) for m in fold_metrics])

    if not pool_objective:
        return mean_logloss

    for name, value in make_pooled_classifier_metrics()(collector.pooled()).items():
        trial.set_user_attr(f"pooled_{name}", value)
    return float(trial.user_attrs["pooled_log_loss"])


def _completed_classifier_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not trials:
        raise ValueError("No completed trials to select from.")
    return trials


#: Metrics reported per trial under each aggregation. The pooled names are
#: separate keys rather than a reuse of ``mean_*`` on purpose: one attribute that
#: means "average of the folds" in some runs and "over all games" in others is
#: exactly the overloading that has produced silent no-ops in this pipeline
#: before. Both are always written, so a pooled run can still be read as means.
_POOLED_REGRESSOR_KEYS: tuple[str, ...] = (
    "pooled_mae",
    "pooled_rmse",
    "pooled_r2",
    "pooled_ou_acc",
    "pooled_n_games",
)
_POOLED_CLASSIFIER_KEYS: tuple[str, ...] = (
    "pooled_log_loss",
    "pooled_brier",
    "pooled_ou_acc",
    "pooled_n_games",
)


def _pooled_metric(trial: optuna.trial.FrozenTrial, key: str) -> float:
    """A trial's pooled metric, falling back to its objective value.

    ``_completed_classifier_trials`` has already excluded trials with no value,
    so the fallback is always a real number -- but say so explicitly rather than
    letting ``float(None)`` be the thing that would raise, several frames away
    from the cause.
    """
    value = trial.user_attrs.get(key, trial.value)
    if value is None:
        raise ValueError(
            f"Trial {trial.number} has neither a {key!r} user attribute nor an "
            "objective value, so it cannot be ranked."
        )
    return float(value)


#: Studies smaller than this never raise the tie-band diagnostic. Smoke runs
#: use 1-2 trials, where a full tie set is inevitable and meaningless.
_TIE_WARNING_MIN_TRIALS = 10


@dataclass(frozen=True)
class TieToleranceResult:
    """The resolved tie band, plus everything needed to audit it afterwards."""

    policy: str
    #: Width of the band in primary-metric units.
    tolerance: float
    #: ``best + tolerance``. Trials at or under this enter the tie-break.
    cutoff: float
    best: float
    n_completed: int
    n_candidates: int
    #: Before the floor and cap were applied. Records what the data asked for.
    raw_tolerance: float
    #: Non-None when the band is wider than ``warn_fraction`` of the trials,
    #: i.e. when the secondary metric is doing the selecting.
    warning: str | None

    @property
    def fraction(self) -> float:
        return self.n_candidates / self.n_completed if self.n_completed else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "tie_policy": self.policy,
            "tie_tolerance": self.tolerance,
            "tie_raw_tolerance": self.raw_tolerance,
            "tie_cutoff": self.cutoff,
            "tie_best_metric": self.best,
            "tie_n_completed": self.n_completed,
            "tie_n_candidates": self.n_candidates,
            "tie_candidate_fraction": self.fraction,
            "tie_warning": self.warning,
        }


def resolve_tie_tolerance(
    values: Sequence[float],
    *,
    policy: TieTolerancePolicy,
    fixed_abs: float | None,
    fixed_pct: float | None,
    max_fraction: float,
    floor: float,
    cap: float,
    warn_fraction: float,
) -> TieToleranceResult:
    """Turn the completed trials' primary metric into a tie band.

    ``fixed``: the historical rule, ``best + fixed_abs`` (or ``best * (1+pct)``).

    ``quantile``: sort the completed values ascending and take the gap from the
    best to the trial at index ``floor(max_fraction * n)`` -- the width that
    spans the best ``max_fraction`` of trials -- then clamp into
    ``[floor, cap]``. Lower metric is better, which is true of every primary
    metric here (MAE, RMSE, log loss).

    The realised candidate count can exceed ``max_fraction`` for exactly two
    reasons, both intentional and both visible in the result: exact ties sitting
    on the cutoff (breaking those by trial order would be worse), and the floor
    widening a band that would otherwise be numerical dust.
    """
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        raise ValueError("No finite trial metrics to derive a tie tolerance from.")

    best = min(finite)
    n = len(finite)

    if policy == TieTolerancePolicy.FIXED:
        if fixed_pct is not None:
            tolerance = abs(best) * fixed_pct
        else:
            tolerance = float(fixed_abs or 0.0)
        raw = tolerance
    else:
        ordered = sorted(finite)
        # floor(max_fraction * n) is an INDEX, so it names the (index+1)-th best
        # trial: at n=60 and 0.10 that is ordered[6], a band spanning 7 trials.
        # Clamped to a valid index so a tiny study still produces a band.
        index = min(n - 1, max(0, int(max_fraction * n)))
        raw = ordered[index] - best
        tolerance = min(cap, max(floor, raw))

    cutoff = best + tolerance
    n_candidates = sum(1 for v in finite if v <= cutoff)

    warning = None
    # Below a handful of completed trials there is no search to speak of, and
    # "2 of 2 trials tied" is arithmetic rather than a finding. Warning there
    # would train readers to ignore the warning that matters.
    if n >= _TIE_WARNING_MIN_TRIALS and n_candidates / n > warn_fraction:
        warning = (
            f"{n_candidates} of {n} completed trials ({n_candidates / n:.0%}) "
            f"fall inside the tie band of {tolerance:.4f}, above the "
            f"{warn_fraction:.0%} diagnostic threshold. The secondary metric is "
            "selecting the model, not breaking a tie between trials the primary "
            "metric could not separate. Tighten optuna.tie_tolerance_cap or "
            "optuna.tie_max_fraction, or widen the search space so the primary "
            "metric has something to discriminate."
        )

    return TieToleranceResult(
        policy=str(policy),
        tolerance=tolerance,
        cutoff=cutoff,
        best=best,
        n_completed=n,
        n_candidates=n_candidates,
        raw_tolerance=raw,
        warning=warning,
    )


def completed_primary_values(study: optuna.Study, *, primary_key: str) -> list[float]:
    """Every completed trial's primary metric, in the units selection ranks on."""
    return [
        _pooled_metric(trial, primary_key)
        for trial in _completed_classifier_trials(study)
    ]


def select_best_trial_lexicographic_pooled(
    study: optuna.Study,
    *,
    mae_tolerance_abs: float | None = 0.10,
    mae_tolerance_pct: float | None = None,
) -> optuna.trial.FrozenTrial:
    """Regressor selection on pooled metrics.

    Identical rule to upstream's ``select_best_trial_lexicographic`` -- best MAE
    within a tolerance, then maximum OU accuracy, then RMSE and MAE as
    tiebreaks -- reading ``pooled_*`` instead of ``mean_*``. Kept separate rather
    than made conditional inside upstream, so a legacy run cannot accidentally be
    ranked on a mixture of the two scales.
    """
    trials = _completed_classifier_trials(study)
    best = min(_pooled_metric(t, "pooled_mae") for t in trials)
    if mae_tolerance_pct is not None:
        cutoff = best * (1.0 + mae_tolerance_pct)
    else:
        cutoff = best + (mae_tolerance_abs or 0.0)

    candidates = [
        t for t in trials if _pooled_metric(t, "pooled_mae") <= cutoff
    ]
    if not candidates:
        raise ValueError("No candidate trials found within the MAE tolerance.")

    return min(
        candidates,
        key=lambda trial: (
            -float(trial.user_attrs.get("pooled_ou_acc", float("-inf"))),
            float(trial.user_attrs.get("pooled_rmse", float("inf"))),
            _pooled_metric(trial, "pooled_mae"),
            trial.number,
        ),
    )


def select_best_classifier_trial_lexicographic_pooled(
    study: optuna.Study, *, logloss_tolerance_abs: float | None = 0.002
) -> optuna.trial.FrozenTrial:
    """Near-best pooled log loss, then directional accuracy.

    ROI is intentionally not a selector. Under flat symmetric odds it is a
    rescaling of win rate, and with real prices it belongs to an outer/post-hoc
    betting evaluation rather than the folds already used for model selection.
    """
    trials = _completed_classifier_trials(study)
    best = min(_pooled_metric(t, "pooled_log_loss") for t in trials)
    cutoff = best + (logloss_tolerance_abs or 0.0)

    candidates = [
        t
        for t in trials
        if _pooled_metric(t, "pooled_log_loss") <= cutoff
    ]
    if not candidates:
        candidates = trials

    return min(
        candidates,
        key=lambda trial: (
            -float(trial.user_attrs.get("pooled_ou_acc", 0.0)),
            _pooled_metric(trial, "pooled_log_loss"),
            trial.number,
        ),
    )


def summarize_trials_pooled(study: optuna.Study, *, keys: tuple[str, ...]) -> pd.DataFrame:
    """One row per trial with both aggregations and the tuned parameters.

    ``**trial.params`` is what makes ``n_estimators`` and ``train_games`` appear
    as ordinary hyperparameter columns once they are tuned -- no special-casing
    needed, and no chance of a selected value that never reaches the report.
    """
    protocol_keys = (
        "n_folds",
        "n_validation_games",
        "n_estimators",
        "train_games",
        "sample_weight_lambda",
        "pruned_after_folds",
        "pruned_after_games",
    )
    mean_keys = (
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "mean_ou_acc",
        "mean_logloss",
        "mean_brier",
    )
    rows = [
        {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            **{key: trial.user_attrs.get(key) for key in keys},
            **{
                key: trial.user_attrs.get(key)
                for key in mean_keys
                if key in trial.user_attrs
            },
            **{key: trial.user_attrs.get(key) for key in protocol_keys},
            **trial.params,
        }
        for trial in study.trials
    ]
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def summarize_candidates_pooled(
    study: optuna.Study,
    *,
    primary_key: str,
    tolerance_abs: float | None,
    tolerance_pct: float | None = None,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    """The trials inside the tolerance, in the order selection would rank them."""
    trials = _completed_classifier_trials(study)
    best = min(_pooled_metric(t, primary_key) for t in trials)
    if tolerance_pct is not None:
        cutoff = best * (1.0 + tolerance_pct)
    else:
        cutoff = best + (tolerance_abs or 0.0)

    rows = [
        {
            "trial": trial.number,
            **{key: trial.user_attrs.get(key) for key in keys},
            **trial.params,
        }
        for trial in trials
        if _pooled_metric(trial, primary_key) <= cutoff
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    ou_column = "pooled_ou_acc" if "pooled_ou_acc" in frame.columns else primary_key
    return frame.sort_values(
        [ou_column, primary_key], ascending=[False, True]
    ).reset_index(drop=True)


def select_best_classifier_trial_lexicographic(
    study: optuna.Study,
    *,
    logloss_tolerance_abs: float | None = 0.002,
) -> optuna.trial.FrozenTrial:
    """Best log loss within tolerance, then directional accuracy.

    Mirrors the regressors' lexicographic selection, and matters more here.
    Simulated at 600 validation games, log loss ranks a truly-53% trial above a
    truly-52% one only 64% of the time -- barely better than chance. Trusting
    that ordering outright would be selecting on noise, so the tolerance is
    deliberately wide relative to the metric's range and the real choice is
    made by the secondary criterion.
    """
    trials = _completed_classifier_trials(study)
    best = min(float(t.value) for t in trials)  # type: ignore[arg-type]
    cutoff = best + (logloss_tolerance_abs or 0.0)

    candidates = [
        t
        for t in trials
        if float(t.user_attrs.get("mean_logloss", t.value)) <= cutoff  # type: ignore[arg-type]
    ]
    if not candidates:
        candidates = trials

    def _key(trial: optuna.trial.FrozenTrial) -> tuple[float, float, int]:
        return (
            -float(trial.user_attrs.get("mean_ou_acc", 0.0)),
            float(trial.user_attrs.get("mean_logloss", trial.value)),  # type: ignore[arg-type]
            trial.number,
        )

    return min(candidates, key=_key)


def summarize_classifier_trials(study: optuna.Study) -> pd.DataFrame:
    rows = [
        {
            "trial": trial.number,
            "state": trial.state.name,
            "value_logloss": trial.value,
            **{
                key: trial.user_attrs.get(key)
                for key in (
                    "mean_logloss",
                    "mean_brier",
                    "mean_ou_acc",
                    "median_best_iteration",
                    "sample_weight_lambda",
                )
            },
            **trial.params,
        }
        for trial in study.trials
    ]
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def summarize_classifier_candidates(
    study: optuna.Study, *, logloss_tolerance_abs: float | None = 0.002
) -> pd.DataFrame:
    """The trials that survived the log-loss tolerance, ranked as selected."""
    trials = _completed_classifier_trials(study)
    best = min(float(t.value) for t in trials)  # type: ignore[arg-type]
    cutoff = best + (logloss_tolerance_abs or 0.0)

    rows = [
        {
            "trial": trial.number,
            "mean_logloss": trial.user_attrs.get("mean_logloss", trial.value),
            "mean_brier": trial.user_attrs.get("mean_brier"),
            "mean_ou_acc": trial.user_attrs.get("mean_ou_acc"),
            "within_tolerance": (
                float(trial.user_attrs.get("mean_logloss", trial.value)) <= cutoff  # type: ignore[arg-type]
            ),
        }
        for trial in trials
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["within_tolerance", "mean_ou_acc"], ascending=[False, False]
    ).reset_index(drop=True)


@dataclass
class OverUnderClassifierStrategy:
    """Predicts P(OVER) directly instead of a total or an error.

    Deliberately does NOT reuse upstream's optuna_total_points/optuna_error_line
    helpers: those are regression-only throughout, from the objective down to
    ``select_best_trial_lexicographic``, which reads ``mean_mae``.
    """

    line_col: str
    #: Rank trials on metrics pooled over every validation game rather than on
    #: the mean of the folds' own metrics. Set from
    #: ``optuna.objective_aggregation`` in get_strategy, and carried on the
    #: strategy because select_best_trial/summarize_* take no config -- the
    #: alternative, inferring it from which user_attrs happen to be present,
    #: would silently rank a run one way or the other with nothing recording
    #: which.
    pooled: bool = False

    def tune(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        config: ExperimentConfig,
        splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
        split_provider: SplitProvider | None = None,
        dates: pd.Series | None = None,
    ) -> optuna.Study:
        provider = _provider_from(splits, split_provider, config)
        return _optimize(
            _create_study(config, n_folds=provider.n_folds),
            lambda trial: run_classifier_objective(
                trial,
                X=X,
                y=y,
                split_provider=provider,
                config=config,
                dates=dates,
            ),
            config,
        )

    @property
    def primary_metric_key(self) -> str:
        return "pooled_log_loss" if self.pooled else "mean_logloss"

    def select_best_trial(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> optuna.trial.FrozenTrial:
        # The MAE tolerances are ignored: this strategy is selected on log loss,
        # whose scale is unrelated. The parameters stay in the signature so the
        # strategy remains interchangeable with the regressors for callers.
        del mae_tolerance_abs, mae_tolerance_pct
        if self.pooled:
            return select_best_classifier_trial_lexicographic_pooled(
                study, logloss_tolerance_abs=self._logloss_tolerance
            )
        return select_best_classifier_trial_lexicographic(
            study, logloss_tolerance_abs=self._logloss_tolerance
        )

    #: Set by get_strategy from the config so select_best_trial, which the
    #: Protocol pins to MAE-shaped arguments, can still see the right tolerance.
    _logloss_tolerance: float | None = 0.002

    def summarize_trials(self, study: optuna.Study) -> pd.DataFrame:
        if self.pooled:
            return summarize_trials_pooled(study, keys=_POOLED_CLASSIFIER_KEYS)
        return summarize_classifier_trials(study)

    def summarize_candidates(
        self,
        study: optuna.Study,
        *,
        mae_tolerance_abs: float | None,
        mae_tolerance_pct: float | None,
    ) -> pd.DataFrame:
        del mae_tolerance_abs, mae_tolerance_pct
        if self.pooled:
            return summarize_candidates_pooled(
                study,
                primary_key="pooled_log_loss",
                tolerance_abs=self._logloss_tolerance,
                keys=_POOLED_CLASSIFIER_KEYS,
            )
        return summarize_classifier_candidates(
            study, logloss_tolerance_abs=self._logloss_tolerance
        )

    def fit_best(
        self,
        *,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        study: optuna.Study | None = None,
        trial: optuna.trial.FrozenTrial | None = None,
        config: ExperimentConfig,
        dates_dev: pd.Series | None = None,
    ) -> XGBModel:
        selected = trial if trial is not None else study.best_trial  # type: ignore[union-attr]
        params, n_estimators, lambda_ = resolve_final_params(selected, config)
        return fit_final_model(
            X_dev=X_dev,
            y_dev=y_dev,
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev,
            sample_weight_lambda=lambda_,
        )

    def evaluate_holdout(
        self,
        model: XGBModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """EV-threshold sweep over the holdout.

        Returns the same (table, predictions) shape as the regressors so the
        holdout path stays uniform, but the threshold column is in EV units and
        the returned array is P(OVER), not a points prediction.
        """
        y = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
        p_over = np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
        odds = config.betting.flat_decimal_odds

        ev_over = p_over * odds - 1.0
        ev_under = (1.0 - p_over) * odds - 1.0
        bet_over = ev_over >= ev_under
        score = np.maximum(ev_over, ev_under)
        correct = np.where(bet_over, y == 1.0, y == 0.0)

        rows = []
        for threshold in config.betting.ev_thresholds:
            placed = score > threshold
            n = int(placed.sum())
            rows.append(
                {
                    "threshold_min_ev": threshold,
                    "n_bets": n,
                    "ou_betting_accuracy": (
                        float(correct[placed].mean()) if n else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows), p_over


def get_strategy(config: ExperimentConfig) -> TargetFamilyStrategy:
    if config.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        return OverUnderClassifierStrategy(
            line_col=config.line_col,
            pooled=config.pools_objective,
            _logloss_tolerance=config.optuna.logloss_tolerance_abs,
        )
    if config.strategy == PredictionStrategy.TOTAL_POINTS_REGRESSOR:
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        return TotalPointsStrategy(
            line_col=config.line_col, pooled=config.pools_objective
        )
    if config.strategy in (
        PredictionStrategy.LINE_ERROR_REGRESSOR,
        PredictionStrategy.SPREAD_ERROR_REGRESSOR,
    ):
        # One strategy object for both markets' residual regressors: the fold
        # loop, the objective and the directional scorer are identical, and the
        # only market-specific quantity (which line) never enters here because a
        # residual model's prediction already IS the edge.
        #
        # SPREAD_ERROR inherits the push convention for free, which is the
        # behaviour required of it: over_under_betting_accuracy_error_line drops
        # rows where the true side is 0 (a push) and rows where the predicted
        # side is exactly 0, so neither is silently scored as HOME or AWAY.
        return LineErrorStrategy(
            sample_weight=config.sample_weight, pooled=config.pools_objective
        )

    # Explicit rather than a fall-through: a strategy added later must fail here
    # instead of silently inheriting residual-regression semantics that may not
    # apply to it.
    raise ValueError(
        f"No tuning strategy is registered for {config.strategy.value!r}."
    )
