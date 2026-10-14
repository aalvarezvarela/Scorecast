"""Phase 5 paired walk-forward: one arm, any seeds, resumable.

The frozen protocol is in ``experiments/pi_block_v1_closing_2026_10/README.md``
(and the research log, "Phase 5, step 4"). This runs one arm's config under
the given seeds with the training pipeline's own daily walk-forward
(``training_pipeline.backtest.run_walk_forward_evaluation``): every game day
of the evaluation window is predicted by a model refitted on every game
strictly before it (window capped at ``walk_forward.train_games``), with the
spec's fixed hyperparameters (``optuna.fixed_params``,
``optuna.fixed_n_estimators``) and the seed as XGBoost's ``random_state``.

The window starts on the first game day with ``walk_forward.min_train_games``
earlier games, asserted equal to the frozen ``FIRST_DAY``. It is run in
calendar months: each month is evaluated with every earlier game as history,
which is exactly what one long run does day by day (each day trains on every
row strictly before it), so a finished month is a checkpoint. Output, per
arm / seed / month, written atomically:
``<out>/<arm>/seed=<seed>/month=<YYYY-MM>.parquet`` (one row per game:
GAME_ID, date, y_true, y_pred, the target and baseline lines, the actual
total, the predicted edge and the over / under decimal odds) and a
``run.json`` per arm / seed with the config fingerprint, dataset checksum,
device, feature set and package versions. Months already written are skipped,
so an interrupted run resumes where it stopped.

    python scripts/player_graph/run_pi_walk_forward.py \\
        --config experiments/pi_block_v1_closing_2026_10/a_baseline_2_5.yaml \\
        --arm a_baseline --expect-pi none --seeds 101 202

``--max-days N`` stops each seed after N game days (smoke tests only; such
output goes to its own directory).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.pi_features import PRIMARY_FEATURES

from training_pipeline.backtest import run_walk_forward_evaluation
from training_pipeline.cli import load_config
from training_pipeline.data import prepare_dataset
from training_pipeline.decisions import collect_prices

#: Frozen with the protocol: first game day with 1,250 earlier games.
FIRST_DAY = pd.Timestamp("2021-01-21")
EXPECTED_PI = {"none": (), "primary": tuple(sorted(PRIMARY_FEATURES))}


def first_eligible_day(dates: pd.Series, min_train_games: int) -> pd.Timestamp:
    per_day = dates.value_counts().sort_index()
    before = per_day.cumsum().shift(fill_value=0)
    return pd.Timestamp(per_day.index[before >= min_train_games][0])


def attach_game_ids(prepared, config) -> pd.Series:
    """``GAME_ID`` of every row of ``prepared.df_full``.

    On the closing path cleaning drops ``GAME_ID`` and resets the index, and
    keeping it would change the cleaned frame archived runs were produced
    from. So each row is matched back to the dataset file on columns cleaning
    leaves unchanged: the date, the outcome, both lines and the three fully
    populated numeric features with the most distinct values. The match must
    be one to one; anything else stops the run.
    """
    df = prepared.df_full
    raw = pd.read_parquet(config.data.csv_path)
    raw["GAME_ID"] = raw["GAME_ID"].astype(str).str.zfill(10)
    date_col = config.data.date_col
    base = [
        c
        for c in (
            config.outcome_col,
            prepared.target_line_col,
            prepared.baseline_line_col,
        )
        if c in df.columns and c in raw.columns
    ]
    candidates = [
        c
        for c in prepared.feature_names
        if c in raw.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().all()
    ]
    spread = df[candidates].nunique().sort_values(ascending=False)
    keys = list(dict.fromkeys([date_col, *base, *spread.index[:3]]))
    left = df[keys].copy()
    left[date_col] = pd.to_datetime(left[date_col]).dt.normalize()
    right = raw[["GAME_ID", *keys]].copy()
    right[date_col] = pd.to_datetime(right[date_col]).dt.normalize()
    if left.duplicated().any() or right.duplicated(keys).any():
        raise SystemExit(f"rows are not unique on {keys}")
    merged = left.merge(right, on=keys, how="left", validate="one_to_one")
    if merged["GAME_ID"].isna().any() or merged["GAME_ID"].duplicated().any():
        raise SystemExit(f"could not match every row to one GAME_ID on {keys}")
    return pd.Series(merged["GAME_ID"].to_numpy(), index=df.index)


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def versions() -> dict:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "machine": platform.node(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--expect-pi", choices=sorted(EXPECTED_PI), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/pi_block_v1")
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Override config.device (record it; every run of one comparison "
        "must use the same device)",
    )
    parser.add_argument("--max-days", type=int, help="Smoke test only")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config = config.model_copy(deep=True)
        config.device = args.device
    prepared = prepare_dataset(config)
    pi = tuple(sorted(f for f in prepared.feature_names if f.startswith("PI_")))
    if pi != EXPECTED_PI[args.expect_pi]:
        raise SystemExit(f"arm sees PI columns {pi}, expected {args.expect_pi}")
    df = prepared.df_full
    game_ids = attach_game_ids(prepared, config)
    date_col = config.data.date_col
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    first = first_eligible_day(dates, config.walk_forward.min_train_games)
    if first != FIRST_DAY:
        raise SystemExit(
            f"first eligible day {first.date()} != frozen {FIRST_DAY.date()}"
        )
    months = sorted(dates.loc[dates >= first].dt.to_period("M").unique())
    xgb_params = dict(config.optuna.fixed_params)
    n_estimators = config.optuna.fixed_n_estimators
    train_games = config.walk_forward.train_games
    out = args.out / (args.arm if args.max_days is None else f"{args.arm}_smoke")
    print(
        f"{args.arm}: {len(df):,} rows, {len(prepared.feature_names):,} features "
        f"(PI: {list(pi) or 'none'}); window {first.date()} -> "
        f"{dates.max().date()}, {len(months)} months; device {config.device}; "
        f"n_estimators {n_estimators}; train window {train_games}",
        flush=True,
    )
    for seed in args.seeds:
        seed_dir = out / f"seed={seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "run.json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "config": str(args.config),
                    "fingerprint": config.fingerprint(),
                    "dataset_checksum": prepared.dataset_checksum,
                    "device": config.device,
                    "seed": seed,
                    "features": len(prepared.feature_names),
                    "feature_names": list(prepared.feature_names),
                    "pi_features": list(pi),
                    "xgb_params": xgb_params,
                    "n_estimators": n_estimators,
                    "train_games": train_games,
                    "first_day": str(first.date()),
                    "versions": versions(),
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        days_done = 0
        for month in months:
            path = seed_dir / f"month={month}.parquet"
            if path.exists() and args.max_days is None:
                continue
            in_month = (dates.dt.to_period("M") == month) & (dates >= first)
            evaluation = df.loc[in_month]
            if args.max_days is not None:
                keep = sorted(dates.loc[in_month].unique())[: args.max_days - days_done]
                if not keep:
                    break
                evaluation = evaluation.loc[dates.loc[in_month].isin(keep)]
            history = df.loc[dates < dates.loc[evaluation.index].min()]
            started = time.time()
            result = run_walk_forward_evaluation(
                config,
                prepared=prepared,
                df_history=history,
                df_evaluation=evaluation,
                train_games=train_games,
                xgb_params=xgb_params,
                n_estimators=n_estimators,
                show_progress=False,
                random_state=seed,
            )
            predictions = result.predictions.copy()
            rows = evaluation.iloc[predictions["row_in_test_final"].to_numpy()]
            predictions.insert(0, "GAME_ID", game_ids.loc[rows.index].to_numpy())
            if "date" not in predictions:
                predictions.insert(1, "date", dates.loc[rows.index].to_numpy())
            over, under = collect_prices(
                evaluation,
                config,
                positions=predictions["row_in_test_final"].to_numpy(),
            )
            predictions["decimal_odds_over"] = over
            predictions["decimal_odds_under"] = under
            predictions["seed"] = seed
            predictions["arm"] = args.arm
            write_atomic(predictions, path)
            days = int(result.n_days)
            days_done += days
            print(
                f"  seed {seed} {month}: {days} days, {len(predictions)} games, "
                f"MAE {result.mae:.3f} ({time.time() - started:.0f}s)",
                flush=True,
            )
            if args.max_days is not None and days_done >= args.max_days:
                break


if __name__ == "__main__":
    main()
