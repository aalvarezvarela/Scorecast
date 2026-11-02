from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.scorers import (
    over_under_betting_accuracy_total_points,
    over_under_betting_accuracy_total_points_with_min_edge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass
class FoldMetrics:
    fold: int
    mae: float
    rmse: float
    r2: float
    ou_accuracy: float
    ou_accuracy_edge_2: float
    ou_accuracy_edge_3: float
    ou_accuracy_edge_4: float
    best_iteration: int
    n_train: int
    n_valid: int


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _predict_best(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Predict using the best iteration when early stopping was used.
    """
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is not None:
        try:
            return model.predict(X, iteration_range=(0, best_iteration + 1))
        except TypeError:
            pass

    return model.predict(X)


def build_xgb_params_total_points(
    trial: optuna.Trial,
    *,
    random_state: int = 16,
    objective: str = "reg:squarederror",
) -> dict:
    """
    Conservative search space for noisy NBA totals.
    """
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": objective,
        "eval_metric": "mae",
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 5.0, 60.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 0.1, 3.0),
        "subsample": trial.suggest_float("subsample", 0.55, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.8),
        "learning_rate": trial.suggest_float("learning_rate", 0.0075, 0.06, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 50.0, log=True),
        "n_estimators": 1000,
        "early_stopping_rounds": 70,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def evaluate_fold_total_points(
    model: XGBRegressor,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    line_col: str,
    *,
    fold: int,
    n_train: int,
) -> FoldMetrics:
    """
    Compute fold metrics after training.
    """
    y_true = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=float)
    y_pred = np.asarray(_predict_best(model, X_valid), dtype=float)
    betting_line = pd.to_numeric(X_valid[line_col], errors="coerce").to_numpy(
        dtype=float
    )

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    ou_acc = float(
        over_under_betting_accuracy_total_points(
            y_true=y_true,
            y_pred=y_pred,
            betting_line=betting_line,
        )
    )
    ou_acc_edge_2 = over_under_betting_accuracy_total_points_with_min_edge(
        y_true=y_true,
        y_pred=y_pred,
        betting_line=betting_line,
        min_edge=2,
    )
    ou_acc_edge_3 = over_under_betting_accuracy_total_points_with_min_edge(
        y_true=y_true,
        y_pred=y_pred,
        betting_line=betting_line,
        min_edge=3,
    )
    ou_acc_edge_4 = over_under_betting_accuracy_total_points_with_min_edge(
        y_true=y_true,
        y_pred=y_pred,
        betting_line=betting_line,
        min_edge=4,
    )

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = model.get_params().get("n_estimators", 0) - 1

    return FoldMetrics(
        fold=fold,
        mae=mae,
        rmse=rmse,
        r2=r2,
        ou_accuracy=ou_acc,
        ou_accuracy_edge_2=(
            float(ou_acc_edge_2) if np.isfinite(ou_acc_edge_2) else 0.0
        ),
        ou_accuracy_edge_3=(
            float(ou_acc_edge_3) if np.isfinite(ou_acc_edge_3) else 0.0
        ),
        ou_accuracy_edge_4=(
            float(ou_acc_edge_4) if np.isfinite(ou_acc_edge_4) else 0.0
        ),
        best_iteration=int(best_iteration) + 1,
        n_train=int(n_train),
        n_valid=int(len(X_valid)),
    )


def objective_total_points_mae(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    line_col: str,
    objective_name: str = "reg:squarederror",
) -> float:
    """
    Optuna objective: minimize mean validation MAE across time-aware folds.
    Secondary metrics are stored in user_attrs.
    """
    if line_col not in X.columns:
        raise KeyError(f"{line_col} not found in X")

    params = build_xgb_params_total_points(
        trial,
        random_state=16,
        objective=objective_name,
    )

    fold_metrics: list[FoldMetrics] = []

    for fold_num, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        metrics = evaluate_fold_total_points(
            model=model,
            X_valid=X_va,
            y_valid=y_va,
            line_col=line_col,
            fold=fold_num,
            n_train=len(X_tr),
        )
        fold_metrics.append(metrics)

        mean_mae_so_far = float(np.mean([m.mae for m in fold_metrics]))
        trial.report(mean_mae_so_far, step=fold_num)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_mae = float(np.mean([m.mae for m in fold_metrics]))
    mean_rmse = float(np.mean([m.rmse for m in fold_metrics]))
    mean_r2 = float(np.mean([m.r2 for m in fold_metrics]))
    mean_ou_acc = float(np.mean([m.ou_accuracy for m in fold_metrics]))
    mean_ou_acc_edge_2 = float(np.mean([m.ou_accuracy_edge_2 for m in fold_metrics]))
    mean_ou_acc_edge_3 = float(np.mean([m.ou_accuracy_edge_3 for m in fold_metrics]))
    mean_ou_acc_edge_4 = float(np.mean([m.ou_accuracy_edge_4 for m in fold_metrics]))
    mean_best_iteration = int(round(np.mean([m.best_iteration for m in fold_metrics])))
    median_best_iteration = int(np.median([m.best_iteration for m in fold_metrics]))

    trial.set_user_attr("mean_mae", mean_mae)
    trial.set_user_attr("mean_rmse", mean_rmse)
    trial.set_user_attr("mean_r2", mean_r2)
    trial.set_user_attr("mean_ou_acc", mean_ou_acc)
    trial.set_user_attr("mean_ou_acc_edge_2", mean_ou_acc_edge_2)
    trial.set_user_attr("mean_ou_acc_edge_3", mean_ou_acc_edge_3)
    trial.set_user_attr("mean_ou_acc_edge_4", mean_ou_acc_edge_4)
    trial.set_user_attr("mean_best_iteration", mean_best_iteration)
    trial.set_user_attr("median_best_iteration", median_best_iteration)
    trial.set_user_attr(
        "fold_metrics",
        [
            {
                "fold": m.fold,
                "mae": m.mae,
                "rmse": m.rmse,
                "r2": m.r2,
                "ou_accuracy": m.ou_accuracy,
                "ou_accuracy_edge_2": m.ou_accuracy_edge_2,
                "ou_accuracy_edge_3": m.ou_accuracy_edge_3,
                "ou_accuracy_edge_4": m.ou_accuracy_edge_4,
                "best_iteration": m.best_iteration,
                "n_train": m.n_train,
                "n_valid": m.n_valid,
            }
            for m in fold_metrics
        ],
    )

    return mean_mae


def summarize_optuna_trials(study: optuna.Study) -> pd.DataFrame:
    """
    Build a tidy dataframe with primary and secondary metrics.
    """
    rows = []

    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue

        rows.append(
            {
                "trial": trial.number,
                "value_mae": trial.value,
                "mean_rmse": trial.user_attrs.get("mean_rmse"),
                "mean_r2": trial.user_attrs.get("mean_r2"),
                "mean_ou_acc": trial.user_attrs.get("mean_ou_acc"),
                "mean_ou_acc_edge_2": trial.user_attrs.get("mean_ou_acc_edge_2"),
                "mean_ou_acc_edge_3": trial.user_attrs.get("mean_ou_acc_edge_3"),
                "mean_ou_acc_edge_4": trial.user_attrs.get("mean_ou_acc_edge_4"),
                "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
                "median_best_iteration": trial.user_attrs.get("median_best_iteration"),
                **trial.params,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["value_mae", "mean_ou_acc", "mean_rmse"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def _get_completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]

    if not completed_trials:
        raise ValueError("No completed Optuna trials found.")

    return completed_trials


def _resolve_mae_cutoff(
    *,
    best_mae: float,
    mae_tolerance_abs: float | None,
    mae_tolerance_pct: float | None,
) -> float:
    if mae_tolerance_abs is not None and mae_tolerance_pct is not None:
        raise ValueError("Provide only one of mae_tolerance_abs or mae_tolerance_pct.")

    if mae_tolerance_abs is not None:
        return float(best_mae + mae_tolerance_abs)

    if mae_tolerance_pct is not None:
        return float(best_mae * (1.0 + mae_tolerance_pct))

    return float(best_mae)


def select_best_trial_lexicographic(
    study: optuna.Study,
    *,
    mae_tolerance_abs: float | None = 0.10,
    mae_tolerance_pct: float | None = None,
) -> optuna.trial.FrozenTrial:
    """
    Select the final trial by:
    1. Keeping trials within a small MAE tolerance of the best MAE
    2. Maximizing OU accuracy inside that candidate set
    3. Using RMSE, then MAE, as tie-breakers
    """
    completed_trials = _get_completed_trials(study)
    best_mae = min(float(trial.value) for trial in completed_trials)
    mae_cutoff = _resolve_mae_cutoff(
        best_mae=best_mae,
        mae_tolerance_abs=mae_tolerance_abs,
        mae_tolerance_pct=mae_tolerance_pct,
    )

    candidate_trials = [
        trial
        for trial in completed_trials
        if float(trial.user_attrs.get("mean_mae", trial.value)) <= mae_cutoff
    ]

    if not candidate_trials:
        raise ValueError("No candidate trials found within the MAE tolerance.")

    return min(
        candidate_trials,
        key=lambda trial: (
            -float(trial.user_attrs.get("mean_ou_acc", float("-inf"))),
            float(trial.user_attrs.get("mean_rmse", float("inf"))),
            float(trial.user_attrs.get("mean_mae", trial.value)),
            trial.number,
        ),
    )


def summarize_lexicographic_candidates(
    study: optuna.Study,
    *,
    mae_tolerance_abs: float | None = 0.10,
    mae_tolerance_pct: float | None = None,
) -> pd.DataFrame:
    """
    Summarize the final-trial candidate pool after the MAE tolerance filter.
    """
    completed_trials = _get_completed_trials(study)
    best_mae = min(float(trial.value) for trial in completed_trials)
    mae_cutoff = _resolve_mae_cutoff(
        best_mae=best_mae,
        mae_tolerance_abs=mae_tolerance_abs,
        mae_tolerance_pct=mae_tolerance_pct,
    )

    rows = []
    for trial in completed_trials:
        mean_mae = float(trial.user_attrs.get("mean_mae", trial.value))
        if mean_mae > mae_cutoff:
            continue

        rows.append(
            {
                "trial": trial.number,
                "value_mae": float(trial.value),
                "mean_mae": mean_mae,
                "mean_rmse": trial.user_attrs.get("mean_rmse"),
                "mean_r2": trial.user_attrs.get("mean_r2"),
                "mean_ou_acc": trial.user_attrs.get("mean_ou_acc"),
                "mean_ou_acc_edge_2": trial.user_attrs.get("mean_ou_acc_edge_2"),
                "mean_ou_acc_edge_3": trial.user_attrs.get("mean_ou_acc_edge_3"),
                "mean_ou_acc_edge_4": trial.user_attrs.get("mean_ou_acc_edge_4"),
                "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
                "median_best_iteration": trial.user_attrs.get(
                    "median_best_iteration"
                ),
                "mae_cutoff": mae_cutoff,
                **trial.params,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_ou_acc", "mean_rmse", "mean_mae", "trial"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )


def get_trial_n_estimators(trial: optuna.trial.FrozenTrial) -> int:
    """
    Final boosting rounds to use for the selected trial.

    The median of fold-level best iterations is more robust than the mean for
    noisy time-series folds.
    """
    final_n_estimators = trial.user_attrs.get("median_best_iteration")

    if final_n_estimators is None:
        final_n_estimators = trial.user_attrs.get("mean_best_iteration")

    if final_n_estimators is None:
        final_n_estimators = trial.params.get("n_estimators", 75)

    return max(50, int(round(float(final_n_estimators))))


def fit_best_xgb_total_points(
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    study: optuna.Study | None = None,
    trial: optuna.trial.FrozenTrial | None = None,
    objective_name: str = "reg:squarederror",
) -> XGBRegressor:
    """
    Refit the selected params on all development data.

    Pass either a study or an explicit trial. When a study is provided, the
    Optuna best-by-MAE trial is used. The final n_estimators comes from the
    median CV best iteration when available.
    """
    if (study is None) == (trial is None):
        raise ValueError("Provide exactly one of study or trial.")

    selected_trial = trial if trial is not None else study.best_trial
    best_params = selected_trial.params.copy()
    final_n_estimators = get_trial_n_estimators(selected_trial)

    final_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": objective_name,
        "eval_metric": "mae",
        "random_state": 16,
        "n_jobs": -1,
        "verbosity": 0,
        "n_estimators": final_n_estimators,
        **best_params,
    }

    model = XGBRegressor(**final_params)
    model.fit(X_dev, y_dev, verbose=False)
    return model
