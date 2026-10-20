"""Paired fixed-candidate temporal CV probe. Run from the repository root.

Uses production data preparation, folds, parameter sampling, selection and daily
holdout functions. No Optuna search/pruning and no early stopping. Origin-level
cache shares identical fits across cadences; standalone cost is also recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost
from nba_ou.modeling.scorers import over_under_betting_accuracy_total_points
from optuna.trial import FixedTrial
from xgboost import XGBRegressor

import training_pipeline.backtest as backtest_module
from training_pipeline.backtest import run_walk_forward_evaluation
from training_pipeline.betting import evaluate_betting
from training_pipeline.cli import load_config
from training_pipeline.data import prepare_dataset, verify_dataset_checksum
from training_pipeline.splits import build_holdout_split, build_rolling_origin_plan
from training_pipeline.tuning import (
    build_xgb_params,
    resolve_tie_tolerance,
    select_best_trial_lexicographic_pooled,
)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


@contextmanager
def daily_prediction_cache(directory, config):
    """Checkpoint the production callback, without changing fits or day splits."""
    directory.mkdir(exist_ok=True)
    original_factory = backtest_module._make_fit_and_predict
    times = []

    def factory(*args, **kwargs):
        fit = original_factory(*args, **kwargs)

        def cached(train, test):
            dates = pd.to_datetime(test[config.data.date_col]).dt.normalize()
            assert dates.nunique() == 1
            path = directory / f"{dates.iloc[0].date()}.npz"
            signature = digest(
                {
                    "train_rows": train.index.tolist(),
                    "test_rows": test.index.tolist(),
                    "params": kwargs,
                    "train_last_date": train[config.data.date_col].max(),
                }
            )
            if path.exists():
                with np.load(path) as saved:
                    assert str(saved["signature"]) == signature
                    pred, seconds = saved["prediction"], float(saved["seconds"])
            else:
                start = time.perf_counter()
                pred = fit(train, test)
                seconds = time.perf_counter() - start
                np.savez_compressed(
                    path, prediction=pred, seconds=seconds, signature=signature
                )
            assert len(pred) == len(test)
            times.append(seconds)
            return pred

        return cached

    backtest_module._make_fit_and_predict = factory
    try:
        yield times
    finally:
        backtest_module._make_fit_and_predict = original_factory


def panel(archive, config, n):
    trials = pd.read_csv(archive / "optuna_trials.csv")
    trials = trials[trials.state == "COMPLETE"].sort_values(["value", "trial"])
    selected = json.loads((archive / "optuna_selected_trial.json").read_text())[
        "selected_trial"
    ]
    ids = trials.groupby("train_games", sort=True).first().trial.astype(int).tolist()
    ids += [selected["number"]] + trials.trial.astype(int).tolist()
    ids = list(dict.fromkeys(ids))[:n]
    assert len(ids) == n
    result = []
    for number in ids:
        row = trials[trials.trial == number].iloc[0]
        keys = [
            "max_depth",
            "min_child_weight",
            "gamma",
            "subsample",
            "colsample_bytree",
            "learning_rate",
            "reg_alpha",
            "reg_lambda",
            "n_estimators",
            "train_games",
        ]
        params = {
            k: (
                int(row[k])
                if k in ["max_depth", "n_estimators", "train_games"]
                else float(row[k])
            )
            for k in keys
        }
        # Validate the archived candidate against the unchanged search space.
        build_xgb_params(
            FixedTrial(params),
            config.optuna.search_space,
            objective_name=config.optuna.objective_name,
        )
        assert params["train_games"] in config.walk_forward.train_games_choices
        result.append(
            {"id": number, "params": params, "archived_cv_mae": float(row.value)}
        )
    assert set(x["params"]["train_games"] for x in result) == set(
        config.walk_forward.train_games_choices
    )
    return result


def metrics(frame, config):
    actual = frame.actual.to_numpy()
    line = frame.line.to_numpy()
    pred = frame.predicted_total.to_numpy()
    b = evaluate_betting(
        predicted_edge=pred - line,
        actual_total=actual,
        line=line,
        min_edge=config.betting.primary_edge_threshold,
        flat_decimal_odds=config.betting.flat_decimal_odds,
    )
    return {
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "market_mae": float(np.mean(np.abs(actual - line))),
        "ou_acc": over_under_betting_accuracy_total_points(actual, pred, line),
        "n_games": len(frame),
        **b.model_dump(),
    }


def selections(rows, config):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    for row in rows:
        study.add_trial(
            optuna.trial.create_trial(
                value=row["mae"],
                user_attrs={
                    "pooled_mae": row["mae"],
                    "pooled_rmse": row["rmse"],
                    "pooled_ou_acc": row["ou_acc"],
                },
            )
        )
    o = config.optuna
    band = resolve_tie_tolerance(
        [r["mae"] for r in rows],
        policy=o.tie_tolerance,
        fixed_abs=o.mae_tolerance_abs,
        fixed_pct=o.mae_tolerance_pct,
        max_fraction=o.tie_max_fraction,
        floor=o.tie_tolerance_floor,
        cap=o.tie_tolerance_cap,
        warn_fraction=o.tie_warn_fraction,
    )
    lex = select_best_trial_lexicographic_pooled(
        study, mae_tolerance_abs=band.tolerance
    )
    return {
        "mae": rows[study.best_trial.number]["candidate"],
        "lexicographic": rows[lex.number]["candidate"],
        "tie": band.summary(),
    }


def reuse_archived_holdout(archive, folder, candidates, selected_ids, config, test):
    """Reuse an identical archived daily evaluation, recording zero NEW fits."""
    archived = json.loads((archive / "optuna_selected_trial.json").read_text())[
        "selected_trial"
    ]
    cid = archived["number"]
    path = folder / f"holdout_c{cid}.parquet"
    if cid not in selected_ids or path.exists():
        return
    assert json.loads((archive / "config.json").read_text()) == config.model_dump(
        mode="json"
    )
    candidate = next(c for c in candidates if c["id"] == cid)
    assert candidate["params"].keys() == archived["params"].keys()
    for key, value in candidate["params"].items():
        np.testing.assert_allclose(value, archived["params"][key], rtol=0, atol=1e-12)
    source = pd.read_parquet(archive / "final_test_predictions.parquet")
    np.testing.assert_array_equal(
        pd.to_datetime(source.date).to_numpy(),
        pd.to_datetime(test[config.data.date_col]).to_numpy(),
    )
    np.testing.assert_array_equal(source.actual_outcome, test[config.outcome_col])
    np.testing.assert_array_equal(source.target_line, test["ODDS_TOTAL_LINE_bet365"])
    frame = pd.DataFrame(
        {
            "date": source.date,
            "actual": source.actual_outcome,
            "line": source.target_line,
            "predicted_total": source.target_line + source.predicted_edge,
        }
    )
    archived_mae = json.loads((archive / "final_test_metrics.json").read_text())[
        "holdout"
    ]["mae"]
    np.testing.assert_allclose(
        metrics(frame, config)["mae"], archived_mae, rtol=0, atol=1e-12
    )
    daily = pd.read_csv(archive / "backtest_daily.csv")
    assert len(daily) == frame.date.nunique()
    assert (daily.train_n_games == candidate["params"]["train_games"]).all()
    frame.to_parquet(path, index=False)
    daily.to_csv(folder / f"holdout_c{cid}_daily.csv", index=False)
    dump(
        folder / f"holdout_c{cid}_cost.json",
        {
            "seconds": 0.0,
            "n_days": len(daily),
            "new_fit_days": 0,
            "reused_archive": True,
            "source": str(archive),
            "note": "Identical archived daily evaluation; zero new fits. Original isolated holdout duration unavailable.",
        },
    )
    print("REUSED HOLDOUT", folder.name, cid, str(archive), flush=True)


def run_target(target, spec, settings, out, stage):
    config = load_config(spec["yaml"])
    assert config.random_state == settings["seed"] == 16
    assert config.device == settings["device"] == "cuda"
    assert not config.uses_fold_early_stopping
    assert not config.data.exclude_overtime_from_training
    assert not config.sample_weight.enabled
    verify_dataset_checksum(
        config.data.csv_path, expected_checksum=config.data.expected_checksum
    )
    cache = out / f"prepared_{target}.pkl"
    # This pickle is generated locally by this script, never accepted externally.
    if cache.exists():
        with cache.open("rb") as handle:
            saved_config, prepared, dev, test = pickle.load(handle)
        assert saved_config.model_dump(mode="json") == config.model_dump(mode="json")
    else:
        prepared = prepare_dataset(config)
        dev, test = build_holdout_split(prepared.df_full, config)
        with cache.open("wb") as handle:
            pickle.dump((config, prepared, dev, test), handle)
    archive = Path(spec["archive"])
    schema = json.loads((archive / "feature_schema.json").read_text())
    assert prepared.feature_names == schema["feature_names"]
    candidates = panel(archive, config, settings["n_candidates"])
    # Order resolves exact ties by archived trial number, as the original selector does.
    candidates.sort(key=lambda c: c["id"])
    folder = out / target
    folder.mkdir(exist_ok=True)
    dump(folder / "candidates.json", candidates)
    dump(folder / "resolved_config.json", config.model_dump(mode="json"))
    dump(folder / "features.json", prepared.feature_names)
    manifest = {
        "config": config.model_dump(mode="json"),
        "candidates": candidates,
        "cadences": settings["cadences"],
        "xgboost": xgboost.__version__,
        "feature_hash": digest(prepared.feature_names),
    }
    fingerprint = digest(manifest)
    if (folder / "manifest.json").exists():
        assert (
            json.loads((folder / "manifest.json").read_text())["fingerprint"]
            == fingerprint
        )
    dump(folder / "manifest.json", {**manifest, "fingerprint": fingerprint})
    plans, layouts, expected = {}, [], None
    for cadence in settings["cadences"]:
        cc = config.model_copy(deep=True)
        cc.walk_forward.retrain_every_days = cadence
        plan = build_rolling_origin_plan(dev, cc)
        plan.assert_window_fits(max(config.walk_forward.train_games_choices))
        cohort = np.sort(np.concatenate([f.valid_idx for f in plan.folds]))
        assert len(np.unique(cohort)) == len(cohort)
        if expected is not None:
            np.testing.assert_array_equal(cohort, expected)
        expected = cohort
        plans[cadence] = plan
        plan.fold_info.to_csv(folder / f"folds_{cadence}d.csv", index=False)
        ages = []
        for f in plan.folds:
            assert dev.iloc[f.history_idx][config.data.date_col].max() < f.origin_date
            for age, day in enumerate(f.valid_dates):
                n = int((dev.iloc[f.valid_idx][config.data.date_col] == day).sum())
                ages.extend([(age, (day - f.origin_date).days)] * n)
        layouts.append(
            {
                "cadence": cadence,
                "folds": plan.n_folds,
                "games_per_fold": plan.fold_game_counts,
                "game_days_per_fold": [len(f.valid_dates) for f in plan.folds],
                "n_games": plan.n_validation_games,
                "n_game_days": plan.n_validation_days,
                "start": str(plan.folds[0].valid_start.date()),
                "end": str(plan.folds[-1].valid_end.date()),
                "mean_age_game_days": float(np.mean(ages, axis=0)[0]),
                "mean_age_calendar_days": float(np.mean(ages, axis=0)[1]),
            }
        )
    dump(folder / "layouts.json", layouts)
    dump(
        folder / "cohort.json",
        {
            "cv_row_indices": expected.tolist(),
            "cv_cohort_hash": digest(expected.tolist()),
            "holdout_start": str(test[config.data.date_col].min()),
            "holdout_end": str(test[config.data.date_col].max()),
            "holdout_games": len(test),
            "holdout_days": test[config.data.date_col].nunique(),
            "n_features": len(prepared.feature_names),
            "n_rows": len(prepared.df_full),
        },
    )
    if stage == "prepare":
        return
    X, y = prepared.X.iloc[: len(dev)], prepared.y.iloc[: len(dev)]
    # A common origin is fitted once and predicts the union of validation rows
    # requested by the designs. predict() has no labels and cannot update a fit.
    origins = {}
    for plan in plans.values():
        for fold in plan.folds:
            key = str(fold.origin_date.date())
            if key in origins:
                np.testing.assert_array_equal(
                    origins[key][0].history_idx, fold.history_idx
                )
                origins[key] = (fold, np.union1d(origins[key][1], fold.valid_idx))
            else:
                origins[key] = (fold, fold.valid_idx)
    cv_rows, cv_frames = [], []
    for candidate in candidates:
        cid, cp = candidate["id"], candidate["params"]
        params = build_xgb_params(
            FixedTrial(cp),
            config.optuna.search_space,
            objective_name=config.optuna.objective_name,
            random_state=16,
            device="cuda",
        )
        pred_by_origin, seconds_by_origin = {}, {}
        for key, (fold, valid_idx) in sorted(origins.items()):
            path = folder / f"origin_c{cid}_{key}.npz"
            if path.exists():
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["idx"], valid_idx)
                    predictions, seconds = data["prediction"], float(data["seconds"])
            else:
                start = time.perf_counter()
                model = XGBRegressor(**params)
                train_idx = fold.train_idx(cp["train_games"])
                assert len(train_idx) == cp["train_games"]
                model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
                device = json.loads(model.get_booster().save_config())["learner"][
                    "generic_param"
                ]["device"]
                if not device.startswith("cuda"):
                    raise RuntimeError(f"CUDA required; actual device: {device}")
                predictions = model.predict(X.iloc[valid_idx])
                seconds = time.perf_counter() - start
                np.savez_compressed(
                    path, idx=valid_idx, prediction=predictions, seconds=seconds
                )
                print(f"FIT {target} c{cid} {key} {seconds:.2f}s cuda", flush=True)
            pred_by_origin[key] = pd.Series(predictions, index=valid_idx)
            seconds_by_origin[key] = seconds
        for cadence, plan in plans.items():
            frames = []
            for fold in plan.folds:
                key = str(fold.origin_date.date())
                idx = fold.valid_idx
                raw = pred_by_origin[key].loc[idx].to_numpy(dtype=float)
                line = dev.iloc[idx][prepared.target_line_col].to_numpy(dtype=float)
                dates = dev.iloc[idx][config.data.date_col]
                frame = pd.DataFrame(
                    {
                        "row_idx": idx,
                        "date": dates.to_numpy(),
                        "actual": dev.iloc[idx][config.outcome_col].to_numpy(
                            dtype=float
                        ),
                        "line": line,
                        "predicted_total": (
                            raw + line if target == "line_error" else raw
                        ),
                        "origin": fold.origin_date,
                        "fold": fold.fold,
                        "age_calendar_days": (
                            dates - fold.origin_date
                        ).dt.days.to_numpy(),
                        "age_game_days": [
                            fold.valid_dates.index(pd.Timestamp(d)) for d in dates
                        ],
                    }
                )
                frames.append(frame)
            frame = pd.concat(frames, ignore_index=True).sort_values("row_idx")
            frame["candidate"], frame["cadence"] = cid, cadence
            cv_frames.append(frame)
            row = {
                "target": target,
                "candidate": cid,
                "cadence": cadence,
                "train_games": cp["train_games"],
                **metrics(frame, config),
                "standalone_fit_seconds": sum(
                    seconds_by_origin[str(f.origin_date.date())] for f in plan.folds
                ),
            }
            if cadence == 5:
                row["replay_mae_delta"] = row["mae"] - candidate["archived_cv_mae"]
                assert abs(row["replay_mae_delta"]) < 1e-5, row
            cv_rows.append(row)
            print("CV", target, cid, cadence, round(row["mae"], 6), flush=True)
        pd.DataFrame(cv_rows).to_csv(folder / "cv_metrics.csv", index=False)
        pd.concat(cv_frames).to_parquet(folder / "cv_predictions.parquet", index=False)
    picks = {
        str(d): selections([r for r in cv_rows if r["cadence"] == d], config)
        for d in plans
    }
    dump(
        folder / "selection.json", picks
    )  # Freeze selection BEFORE any new holdout evaluation.
    if stage == "cv":
        return
    holdout_rows = []
    selected_ids = {
        p[rule] for p in picks.values() for rule in ["mae", "lexicographic"]
    }
    if settings.get("reuse_archived_holdout", True):
        reuse_archived_holdout(archive, folder, candidates, selected_ids, config, test)
    for cid in sorted(selected_ids):
        candidate = next(c for c in candidates if c["id"] == cid)
        cp = candidate["params"]
        path = folder / f"holdout_c{cid}.parquet"
        costpath = folder / f"holdout_c{cid}_cost.json"
        if not path.exists():
            with daily_prediction_cache(
                folder / f"daily_c{cid}", config
            ) as daily_times:
                result = run_walk_forward_evaluation(
                    config,
                    prepared=prepared,
                    df_history=dev,
                    df_evaluation=test,
                    train_games=cp["train_games"],
                    xgb_params={
                        k: v
                        for k, v in cp.items()
                        if k not in ["train_games", "n_estimators"]
                    },
                    n_estimators=cp["n_estimators"],
                    show_progress=True,
                )
            frame = pd.DataFrame(
                {
                    "date": result.predictions["date"],
                    "actual": result.predictions["actual_outcome"],
                    "line": result.predictions["target_line"],
                    "predicted_total": result.predictions["target_line"]
                    + result.predictions["predicted_edge"],
                }
            )
            frame.to_parquet(path, index=False)
            result.daily_results.to_csv(
                folder / f"holdout_c{cid}_daily.csv", index=False
            )
            dump(
                costpath,
                {
                    "seconds": sum(daily_times),
                    "n_days": result.n_days,
                    "new_fit_days": result.n_days,
                    "reused_archive": False,
                    "note": "Sum of measured daily fit+prediction times; cached days retain their original timing.",
                },
            )
            # Same public scoring as the production backtest, including strict edge threshold.
            assert abs(metrics(frame, config)["mae"] - result.mae) < 1e-6
        frame = pd.read_parquet(path)
        assert len(frame) == len(test)
        row = {
            "target": target,
            "candidate": cid,
            **metrics(frame, config),
            **json.loads(costpath.read_text()),
        }
        holdout_rows.append(row)
        pd.DataFrame(holdout_rows).to_csv(folder / "holdout_metrics.csv", index=False)
        print("HOLDOUT", target, cid, round(row["mae"], 6), row["win_rate"], flush=True)
    pd.DataFrame(holdout_rows).to_csv(folder / "holdout_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="experiments/temporal_cv_probe_2026_09/probe.json"
    )
    parser.add_argument("--stage", choices=["prepare", "cv", "all"], default="all")
    parser.add_argument("--target", choices=["line_error", "total_points"])
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    settings = json.loads(Path(args.config).read_text())
    out = Path(settings["output"])
    out.mkdir(parents=True, exist_ok=True)
    if args.stage != "prepare":
        # Fail before any potentially expensive CPU fallback.
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        (out / "gpu.txt").write_text(gpu.stdout)
    dump(
        out / "environment.json",
        {
            "xgboost": xgboost.__version__,
            "optuna": optuna.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost_build": xgboost.build_info(),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    for target, spec in settings["targets"].items():
        if args.target is None or args.target == target:
            run_target(target, spec, settings, out, args.stage)


if __name__ == "__main__":
    main()
