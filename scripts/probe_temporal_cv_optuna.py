"""Two adaptive TPE proposals per design, warm-started with the common panel.

All CV choices are frozen before any additional daily holdouts. This is a
budget-limited sensitivity check, not an independent confirmation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FixedTrial, TrialState

from scripts.probe_temporal_cv import (
    daily_prediction_cache,
    digest,
    dump,
    metrics,
    selections,
)
from training_pipeline import tuning
from training_pipeline.backtest import run_walk_forward_evaluation
from training_pipeline.splits import build_split_provider


def cv_target(target, spec, settings, out):
    with (out / f"prepared_{target}.pkl").open("rb") as handle:
        config, prepared, dev, test = pickle.load(handle)
    base = out / target
    panel = json.loads((base / "candidates.json").read_text())
    previous = pd.read_csv(base / "cv_metrics.csv")
    folder = out / "optuna_micro" / target
    folder.mkdir(parents=True, exist_ok=True)
    budget = settings["optuna_micro"]
    assert budget["new_trials_per_design"] == 2 and budget["tpe_startup_trials"] == 8
    manifest = {
        "base": json.loads((base / "manifest.json").read_text())["fingerprint"],
        "budget": budget,
        "sampler": "TPESampler(seed=16,n_startup_trials=8)",
        "pruner": "NopPruner",
    }
    if (folder / "manifest.json").exists():
        assert json.loads((folder / "manifest.json").read_text()) == manifest
    dump(folder / "manifest.json", manifest)
    all_picks = {}
    for cadence in settings["cadences"]:
        c = config.model_copy(deep=True)
        c.walk_forward.retrain_every_days = cadence
        provider = build_split_provider(dev, c)
        expected = json.loads((base / "cohort.json").read_text())["cv_row_indices"]
        np.testing.assert_array_equal(
            np.sort(np.concatenate([f.valid_idx for f in provider.plan.folds])),
            expected,
        )
        sampler_path = folder / f"sampler_{cadence}d.pkl"
        sampler = optuna.samplers.TPESampler(seed=16, n_startup_trials=8)
        if sampler_path.exists():
            with sampler_path.open("rb") as handle:
                sampler = pickle.load(handle)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            storage=f"sqlite:///{(folder/f'study_{cadence}d.db').resolve()}",
            study_name=f"micro_{target}_{cadence}d",
            load_if_exists=True,
        )
        if not study.trials:
            for candidate in panel:
                cp = candidate["params"]
                fixed = FixedTrial(cp)
                fixed.suggest_categorical(
                    "train_games", list(c.walk_forward.train_games_choices)
                )
                tuning.build_xgb_params(
                    fixed,
                    c.optuna.search_space,
                    objective_name=c.optuna.objective_name,
                    device="cuda",
                )
                row = previous[
                    (previous.cadence == cadence)
                    & (previous.candidate == candidate["id"])
                ].iloc[0]
                study.add_trial(
                    optuna.trial.create_trial(
                        value=row.mae,
                        params=fixed.params,
                        distributions=fixed.distributions,
                        user_attrs={
                            "label": f"old_{candidate['id']}",
                            "cv_metrics": row.dropna().to_dict(),
                            "pooled_mae": row.mae,
                            "pooled_rmse": row.rmse,
                            "pooled_ou_acc": row.ou_acc,
                        },
                    )
                )
        assert all(
            t.state == TrialState.COMPLETE for t in study.trials
        ), "Incomplete study: inspect before resuming."
        X, y = prepared.X.iloc[: len(dev)], prepared.y.iloc[: len(dev)]

        def objective(
            trial,
            *,
            cadence=cadence,
            c=c,
            provider=provider,
            expected=expected,
            X=X,
            y=y,
        ):
            started = time.perf_counter()
            frames = []
            print("TPE START", target, cadence, trial.number, flush=True)

            def evaluate(model, Xv, yv, *, fold, n_train):
                device = json.loads(model.get_booster().save_config())["learner"][
                    "generic_param"
                ]["device"]
                assert device.startswith("cuda"), device
                raw = model.predict(Xv).astype(float)
                idx = Xv.index.to_numpy()
                line = dev.iloc[idx][prepared.target_line_col].to_numpy(dtype=float)
                frames.append(
                    pd.DataFrame(
                        {
                            "row_idx": idx,
                            "date": dev.iloc[idx][c.data.date_col].to_numpy(),
                            "actual": dev.iloc[idx][c.outcome_col].to_numpy(
                                dtype=float
                            ),
                            "line": line,
                            "predicted_total": (
                                raw + line if target == "line_error" else raw
                            ),
                            "fold": fold,
                        }
                    )
                )
                if target == "line_error":
                    return tuning._error_line.evaluate_fold_error_line(
                        model, Xv, yv, fold=fold, n_train=n_train
                    )
                return tuning._total_points.evaluate_fold_total_points(
                    model, Xv, yv, c.line_col, fold=fold, n_train=n_train
                )

            value = tuning.run_objective(
                trial,
                X=X,
                y=y,
                config=c,
                evaluate_fold=evaluate,
                split_provider=provider,
                pooled_metrics=(
                    tuning.pooled_line_error_metrics
                    if target == "line_error"
                    else tuning.pooled_total_points_metrics
                ),
                pooled_line_col=None if target == "line_error" else c.line_col,
                dates=dev[c.data.date_col],
            )
            frame = pd.concat(frames, ignore_index=True).sort_values("row_idx")
            np.testing.assert_array_equal(frame.row_idx.to_numpy(), expected)
            m = metrics(frame, c)
            np.testing.assert_allclose(value, m["mae"], rtol=0, atol=1e-10)
            frame.to_parquet(
                folder / f"cv_{cadence}d_trial{trial.number}.parquet", index=False
            )
            trial.set_user_attr("label", f"new_{cadence}d_{trial.number}")
            trial.set_user_attr("cv_metrics", m)
            trial.set_user_attr("new_cv_seconds", time.perf_counter() - started)
            print(
                "TPE DONE",
                target,
                cadence,
                trial.number,
                value,
                trial.params,
                flush=True,
            )
            return value

        remaining = len(panel) + budget["new_trials_per_design"] - len(study.trials)
        assert remaining >= 0
        for _ in range(remaining):
            study.optimize(objective, n_trials=1)
            with sampler_path.open("wb") as handle:
                pickle.dump(study.sampler, handle)
        records = [
            {
                "number": t.number,
                "label": t.user_attrs["label"],
                "params": t.params,
                "value": t.value,
                "cv_metrics": t.user_attrs["cv_metrics"],
                "new_cv_seconds": t.user_attrs.get("new_cv_seconds", 0.0),
            }
            for t in study.trials
        ]
        dump(folder / f"trials_{cadence}d.json", records)
        scores = [{"candidate": r["label"], **r["cv_metrics"]} for r in records]
        # The carried base metrics contain a numeric candidate field; restore labels.
        for score, r in zip(scores, records, strict=True):
            score["candidate"] = r["label"]
        pick = selections(scores, c)
        all_picks[str(cadence)] = pick
        dump(folder / "selection.json", all_picks)
    return config, prepared, dev, test, all_picks


def holdouts(target, context, out):
    c, prepared, dev, test, picks = context
    folder = out / "optuna_micro" / target
    rows = []
    for cadence, pick in picks.items():
        records = json.loads((folder / f"trials_{cadence}d.json").read_text())
        for rule in ["mae", "lexicographic"]:
            record = next(r for r in records if r["label"] == pick[rule])
            cp = record["params"]
            identity = digest(cp)[:16]
            if record["label"].startswith("old_"):
                cid = int(record["label"].split("_")[1])
                path = out / target / f"holdout_c{cid}.parquet"
                assert path.exists(), f"Finish the fixed-panel holdout first: {path}"
                frame = pd.read_parquet(path)
                seconds = 0.0
                source = str(path)
            else:
                path = folder / f"holdout_{identity}.parquet"
                costpath = folder / f"holdout_{identity}_cost.json"
                if not path.exists():
                    with daily_prediction_cache(
                        folder / f"daily_{identity}", c
                    ) as times:
                        result = run_walk_forward_evaluation(
                            c,
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
                            "date": result.predictions.date,
                            "actual": result.predictions.actual_outcome,
                            "line": result.predictions.target_line,
                            "predicted_total": result.predictions.target_line
                            + result.predictions.predicted_edge,
                        }
                    )
                    assert len(frame) == len(test)
                    np.testing.assert_allclose(
                        metrics(frame, c)["mae"], result.mae, rtol=0, atol=1e-10
                    )
                    frame.to_parquet(path, index=False)
                    result.daily_results.to_csv(
                        folder / f"holdout_{identity}_daily.csv", index=False
                    )
                    dump(costpath, {"seconds": sum(times), "n_days": result.n_days})
                frame = pd.read_parquet(path)
                seconds = json.loads(costpath.read_text())["seconds"]
                source = str(path)
            rows.append(
                {
                    "target": target,
                    "cadence": int(cadence),
                    "selector": rule,
                    "candidate": record["label"],
                    "identity": identity,
                    "train_games": cp["train_games"],
                    "cv_mae": record["value"],
                    "cv_win_rate": record["cv_metrics"]["win_rate"],
                    "cv_n_bets": record["cv_metrics"]["n_bets"],
                    "new_cv_minutes": sum(r["new_cv_seconds"] for r in records) / 60,
                    "holdout_new_minutes": seconds / 60,
                    "source": source,
                    **metrics(frame, c),
                }
            )
            pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
            print(
                "MICRO HOLDOUT",
                target,
                cadence,
                rule,
                record["label"],
                rows[-1]["mae"],
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="experiments/temporal_cv_probe_2026_09/probe.json"
    )
    parser.add_argument("--stage", choices=["cv", "all"], default="all")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    settings = json.loads(Path(args.config).read_text())
    out = Path(settings["output"])
    contexts = {}
    for target, spec in settings["targets"].items():
        contexts[target] = cv_target(target, spec, settings, out)
    # No adaptive proposal ever sees a new holdout metric.
    dump(
        out / "optuna_micro" / "selection_all.json",
        {t: ctx[-1] for t, ctx in contexts.items()},
    )
    if args.stage == "all":
        for target, context in contexts.items():
            holdouts(target, context, out)


if __name__ == "__main__":
    main()
