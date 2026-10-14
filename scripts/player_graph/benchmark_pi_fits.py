"""Benchmark one walk-forward fit on CPU and GPU before choosing the device.

Times the training pipeline's own fit-and-predict closure (the same
parameters the paired walk-forward uses) on the real data of one arm, at
training sizes spanning the evaluation window (1,250 to the 6,200-game cap),
on each available device; reports the mean seconds per fit, the peak GPU
memory (``nvidia-smi``), whether CPU and GPU predictions agree for the same
seed, and the projected time of the full run (945 game days x 10 seeds x 2
arms, each day at its real training size).

GPU XGBoost is not necessarily faster for 92 shallow trees on ~6,000 rows,
and its predictions differ slightly from the CPU's: whichever device is
chosen, every run of one comparison (both arms, all seeds) must use it.
Choosing it does not change the protocol or the hyperparameters.

    python scripts/player_graph/benchmark_pi_fits.py \\
        --config experiments/pi_block_v1_closing_2026_10/b_pi_primary.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training_pipeline.backtest import _make_fit_and_predict
from training_pipeline.cli import load_config
from training_pipeline.data import prepare_dataset

SIZES = (1250, 3000, 6200)
TEST_DAY = pd.Timestamp("2025-01-15")  # a day with more than 6,200 earlier games
FIRST_DAY = pd.Timestamp("2021-01-21")
RUNS = 20  # 10 seeds x 2 arms


class GpuMemory:
    """Peak ``memory.used`` (MiB) of GPU 0 while running, by polling."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak = 0
        self.base = self._read()
        self._stop = threading.Event()

    @staticmethod
    def _read() -> int:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else 0

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._read())
            time.sleep(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def cuda_available() -> bool:
    import xgboost

    return bool(xgboost.build_info().get("USE_CUDA")) and shutil.which("nvidia-smi")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    base = load_config(args.config)
    prepared = prepare_dataset(base)
    df = prepared.df_full
    dates = pd.to_datetime(df[base.data.date_col]).dt.normalize()
    test = df.loc[dates.eq(TEST_DAY)]
    history = df.loc[dates < TEST_DAY]
    if len(history) < max(SIZES) or test.empty:
        raise SystemExit(f"{TEST_DAY.date()} does not have {max(SIZES)} earlier games")
    devices = ["cpu"] + (["cuda"] if cuda_available() else [])
    results, predictions = {}, {}
    for device in devices:
        config = base.model_copy(deep=True)
        config.device = device
        fit = _make_fit_and_predict(
            config,
            target_col=config.target_col,
            xgb_params=dict(config.optuna.fixed_params),
            n_estimators=config.optuna.fixed_n_estimators,
            sample_weight_lambda=None,
            feature_names=prepared.feature_names,
            random_state=args.seed,
        )
        fit(history.tail(SIZES[0]), test)  # warm-up (CUDA context, caches)
        for size in SIZES:
            train = history.tail(size)
            seconds = []
            monitor = GpuMemory() if device == "cuda" else None
            if monitor:
                monitor.__enter__()
            for _ in range(args.repeats):
                started = time.perf_counter()
                predicted = fit(train, test)
                seconds.append(time.perf_counter() - started)
            if monitor:
                monitor.__exit__()
            results[(device, size)] = {
                "seconds_mean": float(np.mean(seconds)),
                "seconds_min": float(np.min(seconds)),
                "gpu_peak_mib": (monitor.peak - monitor.base) if monitor else None,
            }
            predictions[(device, size)] = predicted
            print(
                f"{device} {size:>5} rows: {np.mean(seconds):.2f}s per fit"
                + (f", GPU peak +{monitor.peak - monitor.base} MiB" if monitor else ""),
                flush=True,
            )
    agreement = {}
    if "cuda" in devices:
        for size in SIZES:
            diff = np.abs(predictions[("cpu", size)] - predictions[("cuda", size)])
            agreement[size] = float(diff.max())
        print(f"max |CPU - GPU prediction| per size: {agreement}")
    # Projection: each evaluation day at its real training size.
    per_day = dates.loc[dates >= FIRST_DAY].value_counts().sort_index()
    before = dates.value_counts().sort_index().cumsum().shift(fill_value=0)
    sizes = np.minimum(before.reindex(per_day.index).to_numpy(), max(SIZES))
    projection = {}
    for device in devices:
        times = [results[(device, s)]["seconds_mean"] for s in SIZES]
        per_fit = np.interp(sizes, SIZES, times)
        projection[device] = float(per_fit.sum() * RUNS / 3600)
        print(
            f"{device}: projected full run {projection[device]:.1f} h "
            f"({len(per_day)} days x {RUNS} runs)"
        )
    summary = {
        "config": str(args.config),
        "devices": devices,
        "features": len(prepared.feature_names),
        "test_day": str(TEST_DAY.date()),
        "results": {f"{d}/{s}": v for (d, s), v in results.items()},
        "cpu_gpu_max_abs_prediction_diff": agreement,
        "projected_hours": projection,
    }
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
