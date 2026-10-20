#!/usr/bin/env python3
"""Screen cleaning policies with fixed XGBoost parameters on development CV only.

The holdout is split off but never scored. Validation is the per-fold
intersection of one reference policy's games with the rows retained by every
policy, so every cell is judged on exactly the same targets. Each cell then
trains on its own last N rows strictly before that common validation block;
this preserves the real training-history consequence of its missingness rule.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_pipeline.cli import load_config  # noqa: E402
from training_pipeline.data import prepare_dataset  # noqa: E402
from training_pipeline.splits import (  # noqa: E402
    build_holdout_split,
    build_walk_forward_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/probes/preprocessing_screen_2026_09"),
    )
    return parser.parse_args()


def betting_metrics(y_true: np.ndarray, y_pred: np.ndarray, edge: float) -> dict[str, float | int]:
    selected = np.abs(y_pred) >= edge
    decisive = selected & (y_true != 0)
    correct = np.sign(y_pred[decisive]) == np.sign(y_true[decisive])
    wins = int(correct.sum())
    bets = int(decisive.sum())
    losses = bets - wins
    return {
        f"n_bets_edge_{edge:g}": bets,
        f"win_rate_edge_{edge:g}": wins / bets if bets else float("nan"),
        f"roi_edge_{edge:g}": (wins * (1.9090909090909092 - 1) - losses) / bets
        if bets
        else float("nan"),
    }


def main() -> int:
    args = parse_args()
    configs = [(path, load_config(path)) for path in args.configs]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: establish common outer-validation games without retaining six
    # very wide feature matrices in memory at once.
    reference_validation_keys: list[list[str]] | None = None
    available_by_config: list[set[str]] = []
    for path, config in configs:
        config.cleaning.verbose = 0
        config.cleaning.keep_columns = sorted(
            set(config.cleaning.keep_columns or []) | {config.data.game_id_col}
        )
        print(f"INDEX {path.name}", flush=True)
        prepared = prepare_dataset(config)
        if config.data.game_id_col in prepared.feature_names:
            raise ValueError("Protected GAME_ID unexpectedly reached the feature matrix.")
        df_dev, _df_holdout_not_scored = build_holdout_split(prepared.df_full, config)
        splits, _ = build_walk_forward_splits(df_dev, config)
        if reference_validation_keys is None:
            reference_validation_keys = [
                df_dev.iloc[valid_idx][config.data.game_id_col].astype(str).tolist()
                for _, valid_idx in splits
            ]
        available_by_config.append(set(df_dev[config.data.game_id_col].astype(str)))
        del prepared, df_dev, splits
        gc.collect()

    assert reference_validation_keys is not None
    common_validation_keys = [
        [key for key in fold if all(key in available for available in available_by_config)]
        for fold in reference_validation_keys
    ]
    if any(not fold for fold in common_validation_keys):
        raise ValueError("At least one common validation fold is empty.")
    print(
        "Common validation games per fold: "
        + str([len(fold) for fold in common_validation_keys]),
        flush=True,
    )

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for path, config in configs:
        if not config.optuna.skip_tuning:
            raise ValueError(f"{path} must set optuna.fixed_params")

        print(f"\nPREPARE {path.name}", flush=True)
        prepared = prepare_dataset(config)
        if config.data.game_id_col in prepared.feature_names:
            raise ValueError("Protected GAME_ID unexpectedly reached the feature matrix.")
        df_dev, _df_holdout_not_scored = build_holdout_split(prepared.df_full, config)
        X_dev = prepared.X.loc[df_dev.index]
        y_dev = prepared.y.loc[df_dev.index]
        source_position = {
            str(game_id): position
            for position, game_id in enumerate(df_dev[config.data.game_id_col])
        }
        dates = pd.to_datetime(df_dev[config.data.date_col])
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_keys in common_validation_keys:
            valid_idx = np.asarray([source_position[key] for key in fold_keys], dtype=int)
            valid_start = dates.iloc[valid_idx].min()
            eligible_train = np.flatnonzero((dates < valid_start).to_numpy())
            train_idx = eligible_train[-int(config.walk_forward.train_games) :]
            if len(train_idx) != config.walk_forward.train_games:
                raise ValueError(
                    f"{path.name} has only {len(train_idx)} training rows before "
                    f"the fold starting {valid_start.date()}."
                )
            splits.append((train_idx, valid_idx))

        fold_predictions: list[pd.DataFrame] = []
        fold_maes: list[float] = []
        train_start_dates: list[pd.Timestamp] = []
        fixed_params = dict(config.optuna.fixed_params or {})
        fixed_params.update(
            booster="gbtree",
            tree_method="hist",
            objective=config.optuna.objective_name,
            eval_metric="mae",
            n_estimators=int(config.optuna.fixed_n_estimators or 0),
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0,
            device=config.device,
        )

        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            model = XGBRegressor(**fixed_params)
            model.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx], verbose=False)
            pred = np.asarray(model.predict(X_dev.iloc[valid_idx]), dtype=float)
            true = pd.to_numeric(y_dev.iloc[valid_idx], errors="coerce").to_numpy(float)
            fold_maes.append(float(mean_absolute_error(true, pred)))
            train_start_dates.append(
                pd.to_datetime(df_dev.iloc[train_idx][config.data.date_col]).min()
            )
            target_line = pd.to_numeric(
                df_dev.iloc[valid_idx][prepared.target_line_col], errors="coerce"
            ).to_numpy(float)
            fold_predictions.append(
                pd.DataFrame(
                    {
                        "config": path.stem,
                        "fold": fold,
                        "game_id": common_validation_keys[fold - 1],
                        "date": pd.to_datetime(
                            df_dev.iloc[valid_idx][config.data.date_col]
                        ).to_numpy(),
                        "y_true": true,
                        "y_pred": pred,
                        "target_line": target_line,
                    }
                )
            )

        predictions = pd.concat(fold_predictions, ignore_index=True)
        y_true = predictions["y_true"].to_numpy(float)
        y_pred = predictions["y_pred"].to_numpy(float)
        target_line = predictions["target_line"].to_numpy(float)
        if config.strategy.value == "total_points_regressor":
            actual_edge = y_true - target_line
            predicted_edge = y_pred - target_line
            market_prediction = target_line
        elif config.strategy.value == "line_error_regressor":
            actual_edge = y_true
            predicted_edge = y_pred
            market_prediction = np.zeros_like(y_true)
        else:
            raise ValueError(
                "This fixed-CV probe currently supports total_points_regressor "
                f"and line_error_regressor, not {config.strategy.value!r}."
            )
        predictions["actual_edge"] = actual_edge
        predictions["predicted_edge"] = predicted_edge
        row: dict[str, object] = {
            "config": path.stem,
            "prediction_strategy": config.strategy.value,
            "max_na_per_row": config.cleaning.max_na_per_row,
            "corr_threshold": config.cleaning.corr_threshold,
            "odds_corr_threshold": (config.cleaning.corr_threshold_overrides or {}).get(
                "ODDS_", config.cleaning.corr_threshold
            ),
            "n_clean_rows": len(prepared.df_full),
            "n_dev_rows": len(df_dev),
            "n_features": len(prepared.feature_names),
            "n_folds": len(splits),
            "n_validation_games": len(predictions),
            "train_games_min": min(len(train) for train, _ in splits),
            "train_games_max": max(len(train) for train, _ in splits),
            "earliest_train_date": min(train_start_dates).date().isoformat(),
            "cv_mae_pooled": float(mean_absolute_error(y_true, y_pred)),
            "cv_mae_fold_mean": float(np.mean(fold_maes)),
            "cv_mae_fold_std": float(np.std(fold_maes, ddof=1)),
            "cv_rmse_pooled": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "market_mae": float(mean_absolute_error(y_true, market_prediction)),
        }
        for edge in config.betting.edge_thresholds:
            row.update(betting_metrics(actual_edge, predicted_edge, edge))
        rows.append(row)
        prediction_frames.append(predictions)
        print(
            f"DONE {path.stem}: rows={row['n_clean_rows']} "
            f"features={row['n_features']} mae={row['cv_mae_pooled']:.4f} "
            f"win@0.1={row['win_rate_edge_0.1']:.3%}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values("cv_mae_pooled")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    predictions.to_parquet(output_dir / "cv_predictions.parquet", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "holdout_scored": False,
                "prediction_strategy": configs[0][1].strategy.value,
                "fixed_n_estimators": configs[0][1].optuna.fixed_n_estimators,
                "fixed_params": configs[0][1].optuna.fixed_params,
                "configs": [str(path) for path, _ in configs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"\nSaved to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
