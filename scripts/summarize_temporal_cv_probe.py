"""Summarize the paired CUDA probe without feeding holdout back into selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from scripts.probe_temporal_cv import dump, metrics, selections
from training_pipeline.cli import load_config


def markdown(frame):
    """Small dependency-free Markdown table."""

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(map(fmt, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def historical_inventory(out):
    rows = []
    for path in sorted(Path("artifacts/experiments").rglob("config.json")):
        root = path.parent
        try:
            c = json.loads(path.read_text())
            m = json.loads((root / "metadata.json").read_text())
            f = json.loads((root / "final_test_metrics.json").read_text())
            b = json.loads((root / "betting_metrics.json").read_text())
        except (OSError, ValueError):
            continue
        if c.get("prediction_strategy") not in [
            "line_error_regressor",
            "total_points_regressor",
        ]:
            continue
        w = c["walk_forward"]
        cvfile = root / "cv_predictions.parquet"
        cv = pd.read_parquet(cvfile) if cvfile.exists() else pd.DataFrame()
        date = c["data"].get("date_col", "GAME_DATE")
        row = {
            "run": root.name,
            "group": c.get("comparison_group"),
            "target": c["prediction_strategy"],
            "dataset": c["data"].get("dataset_type"),
            "csv": c["data"]["csv_path"],
            "strategy": w["strategy"],
            "cadence": w.get("retrain_every_days"),
            "test_games": w["test_games"],
            "max_folds": w["max_folds"],
            "folds": m.get("cv_n_folds"),
            "cv_games": m.get("cv_n_validation_games", len(cv)),
            "cv_start": str(cv[date].min()) if date in cv else None,
            "cv_end": str(cv[date].max()) if date in cv else None,
            "cv_mae": f.get("cv", {}).get("mae"),
            "holdout_mae": f.get("holdout", {}).get("mae"),
            "holdout_start": m.get("holdout_start"),
            "holdout_end": m.get("holdout_end"),
            "holdout_games": m.get("holdout_n_games"),
            "win_rate": b.get("primary", {}).get("win_rate"),
            "n_bets": b.get("primary", {}).get("n_bets"),
            "edge": c["betting"].get("primary_edge_threshold"),
            "lex": c["refit"].get("use_lexicographic_selection"),
            "early_stopping": not c["optuna"].get("tune_n_estimators", False),
            "aggregation": c["optuna"].get("objective_aggregation", "mean"),
            "seed": c.get("random_state"),
        }
        if "fold" in cv:
            sizes = cv.groupby("fold").size()
            row.update(
                folds=len(sizes),
                min_fold_games=int(sizes.min()),
                max_fold_games=int(sizes.max()),
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "historical_inventory.csv", index=False)


def bootstrap_delta(base, other, draws=4000):
    """Paired 7-calendar-day cluster bootstrap; descriptive, conditional on picks.

    Resample blocks, pool their games (not equal-weight block MAEs). Retains
    within-block schedule/team dependence but is not an independent confirmation
    or a multiple-selection corrected interval. Seed is fixed before results.
    """
    np.testing.assert_array_equal(base.date.to_numpy(), other.date.to_numpy())
    np.testing.assert_array_equal(base.actual.to_numpy(), other.actual.to_numpy())
    np.testing.assert_array_equal(base.line.to_numpy(), other.line.to_numpy())
    delta = np.abs(other.actual - other.predicted_total) - np.abs(
        base.actual - base.predicted_total
    )
    blocks = (pd.to_datetime(base.date) - pd.to_datetime(base.date).min()).dt.days // 7
    tmp = (
        pd.DataFrame({"delta": delta, "block": blocks})
        .groupby("block")
        .delta.agg(["sum", "count"])
    )
    rng = np.random.default_rng(16)
    idx = rng.integers(0, len(tmp), (draws, len(tmp)))
    values = tmp["sum"].to_numpy()[idx].sum(axis=1) / tmp["count"].to_numpy()[idx].sum(
        axis=1
    )
    return {
        "delta_mae": float(delta.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "blocks": len(tmp),
        "draws": draws,
    }


def summarize(target, spec, out):
    folder = out / target
    config = load_config(spec["yaml"])
    cv = pd.read_csv(folder / "cv_metrics.csv")
    hold = pd.read_csv(folder / "holdout_metrics.csv").set_index("candidate")
    hold["new_fit_days"] = hold.get("new_fit_days", hold.n_days).fillna(hold.n_days)
    picks = json.loads((folder / "selection.json").read_text())
    layouts = {
        r["cadence"]: r for r in json.loads((folder / "layouts.json").read_text())
    }
    candidates = {
        c["id"]: c for c in json.loads((folder / "candidates.json").read_text())
    }
    predictions = pd.read_parquet(folder / "cv_predictions.parquet")
    rows, blocks, intervals = [], [], []
    holdframes = {
        cid: pd.read_parquet(folder / f"holdout_c{cid}.parquet")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
        for cid in hold.index
    }
    for cadence, pick in picks.items():
        d = int(cadence)
        layout = layouts[d]
        for rule in ["mae", "lexicographic"]:
            cid = pick[rule]
            cr = cv[(cv.cadence == d) & (cv.candidate == cid)].iloc[0]
            hr = hold.loc[cid]
            rows.append(
                {
                    "target": target,
                    "cadence": d,
                    "selector": rule,
                    "candidate": cid,
                    "X": candidates[cid]["params"]["train_games"],
                    "folds": layout["folds"],
                    "games_fold_min": min(layout["games_per_fold"]),
                    "games_fold_max": max(layout["games_per_fold"]),
                    "games_fold_mean": float(np.mean(layout["games_per_fold"])),
                    "cv_start": layout["start"],
                    "cv_end": layout["end"],
                    "cv_games": layout["n_games"],
                    "cv_panel_minutes": cv[cv.cadence == d].standalone_fit_seconds.sum()
                    / 60,
                    "cv_selected_minutes": cr.standalone_fit_seconds / 60,
                    "cv_mae": cr.mae,
                    "cv_market_mae": cr.market_mae,
                    "cv_ou_acc_tiebreak": cr.ou_acc,
                    "cv_win_rate": cr.win_rate,
                    "cv_n_bets": int(cr.n_bets),
                    "holdout_minutes": hr.seconds / 60,
                    "holdout_mae": hr.mae,
                    "holdout_market_mae": hr.market_mae,
                    "holdout_win_rate": hr.win_rate,
                    "holdout_n_bets": int(hr.n_bets),
                    "holdout_pushes": int(hr.n_pushes),
                    "holdout_roi_flat": hr.roi,
                    "holdout_wr_ci_low": hr.win_rate_ci_low,
                    "holdout_wr_ci_high": hr.win_rate_ci_high,
                    "tie_width": pick["tie"]["tie_tolerance"],
                    "tie_n": pick["tie"]["tie_n_candidates"],
                }
            )
            frame = holdframes[cid].copy()
            frame["block"] = (
                (pd.to_datetime(frame.date) - pd.Timestamp("2026-01-18")).dt.days // 30
            ) + 1
            for block, chunk in frame.groupby("block"):
                blocks.append(
                    {
                        "target": target,
                        "cadence": d,
                        "selector": rule,
                        "candidate": cid,
                        "block": block,
                        "start": str(chunk.date.min().date()),
                        "end": str(chunk.date.max().date()),
                        **metrics(chunk, config),
                    }
                )
            if d != 5:
                basecid = picks["5"][rule]
                intervals.append(
                    {
                        "target": target,
                        "cadence": d,
                        "selector": rule,
                        "base_candidate": basecid,
                        "candidate": cid,
                        **bootstrap_delta(holdframes[basecid], holdframes[cid]),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "comparison.csv", index=False)
    pd.DataFrame(blocks).to_csv(folder / "holdout_blocks.csv", index=False)
    pd.DataFrame(intervals).to_csv(folder / "holdout_paired_bootstrap.csv", index=False)
    ranks = cv.pivot(index="candidate", columns="cadence", values="mae").corr(
        method="spearman"
    )
    ranks.to_csv(folder / "cv_rank_correlations.csv")
    # Removing each active calendar month measures selection sensitivity on the
    # SAME fitted OOF predictions; this is not a newly trained nested CV.
    sensitivity = []
    predictions["month"] = pd.to_datetime(predictions.date).dt.strftime("%Y-%m")
    for omitted in sorted(predictions.month.unique()):
        for d, frame in predictions[predictions.month != omitted].groupby("cadence"):
            scores = [
                {"candidate": cid, **metrics(chunk, config)}
                for cid, chunk in frame.groupby("candidate", sort=True)
            ]
            pick = selections(scores, config)
            for rule in ["mae", "lexicographic"]:
                sensitivity.append(
                    {
                        "target": target,
                        "omitted_month": omitted,
                        "cadence": d,
                        "selector": rule,
                        "candidate": pick[rule],
                    }
                )
    pd.DataFrame(sensitivity).to_csv(
        folder / "cv_leave_month_out_selection.csv", index=False
    )
    # Age and season describe staleness, not independent candidate replicates.
    ages = []
    regrouping = []
    paired = []
    for (cid, d), frame in predictions.groupby(["candidate", "cadence"]):
        for age, chunk in frame.groupby("age_game_days"):
            ages.append(
                {
                    "candidate": cid,
                    "cadence": d,
                    "age_game_days": age,
                    **metrics(chunk, config),
                }
            )
        baseline = predictions[
            (predictions.candidate == cid) & (predictions.cadence == 5)
        ].set_index("row_idx")
        frame = frame.set_index("row_idx").loc[baseline.index]
        frame["delta_vs_5"] = np.abs(frame.actual - frame.predicted_total) - np.abs(
            baseline.actual - baseline.predicted_total
        )
        frame["same_origin"] = frame.origin == baseline.origin
        if frame.same_origin.any():
            np.testing.assert_allclose(
                frame.loc[frame.same_origin, "predicted_total"],
                baseline.loc[frame.same_origin, "predicted_total"],
                rtol=0,
                atol=0,
            )
        for month, chunk in frame.groupby("month"):
            paired.append(
                {
                    "candidate": cid,
                    "cadence": d,
                    "month": month,
                    "n_games": len(chunk),
                    "delta_mae_vs_5": chunk.delta_vs_5.mean(),
                    "same_origin_fraction": chunk.same_origin.mean(),
                }
            )
        # Negative control: re-grouping fixed 5d predictions into larger folds
        # preserves pooled MAE exactly (the score does not use the fold label).
        errors = np.abs(baseline.actual - baseline.predicted_total)
        grouped = errors.groupby(frame.fold).agg(["sum", "count", "mean"])
        assert len(grouped) == layouts[d]["folds"]
        original = metrics(baseline, config)["mae"]
        pooled = float(grouped["sum"].sum() / grouped["count"].sum())
        np.testing.assert_allclose(original, pooled, rtol=0, atol=1e-12)
        regrouping.append(
            {
                "candidate": cid,
                "cadence": d,
                "fixed_5d_predictions_pooled_mae": pooled,
                "unweighted_fold_mean_mae": float(grouped["mean"].mean()),
                "pooled_delta": pooled - original,
            }
        )
    pd.DataFrame(ages).to_csv(folder / "cv_model_age.csv", index=False)
    pd.DataFrame(paired).to_csv(folder / "cv_paired_months.csv", index=False)
    pd.DataFrame(regrouping).to_csv(folder / "regrouping_control.csv", index=False)
    unique_fit_seconds = 0
    for path in folder.glob("origin_*.npz"):
        with np.load(path) as cached:
            unique_fit_seconds += float(cached["seconds"])
    dump(
        folder / "cost.json",
        {
            "unique_cv_fits": len(list(folder.glob("origin_*.npz"))),
            "unique_cv_fit_minutes": unique_fit_seconds / 60,
            "unique_holdout_minutes": hold.seconds.sum() / 60,
            "unique_holdout_fits": int(hold.new_fit_days.sum()),
            "evaluated_holdout_days_including_reused": int(hold.n_days.sum()),
            "note": "Standalone CV cost sums measured origin fits; shared fits are counted once in actual cost. Includes fit+prediction, excludes preparation.",
        },
    )
    table = summary[
        [
            "cadence",
            "selector",
            "candidate",
            "X",
            "folds",
            "cv_panel_minutes",
            "cv_mae",
            "holdout_minutes",
            "holdout_mae",
            "holdout_win_rate",
            "holdout_n_bets",
        ]
    ]
    (folder / "tables.md").write_text(
        markdown(table) + "\n\n" + markdown(pd.DataFrame(intervals)) + "\n"
    )
    print(target, "\n", markdown(table), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="experiments/temporal_cv_probe_2026_09/probe.json"
    )
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    out = Path(settings["output"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    historical_inventory(out)
    for target, spec in settings["targets"].items():
        summarize(target, spec, out)


if __name__ == "__main__":
    main()
