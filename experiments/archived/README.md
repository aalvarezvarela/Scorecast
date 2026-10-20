# Archived experiment definitions

Everything here ran under the **pre-rolling-origin training protocol**. Nothing
has been deleted, and every campaign still loads and still runs — but the
numbers it produced are **not comparable** with anything run after the change,
and neither the leaderboard nor the summary notebook will pretend otherwise.

## What changed, and why it breaks comparability

| | archived campaigns | current (`../_base.yaml`) |
|---|---|---|
| CV folds | 12 blocks of ~50 games stepped back through dev | 30 rolling origins: train to date T, predict the next 4 game-days |
| Validation volume | ~652 games | ~855 games |
| Boosting rounds | each fold early-stopped on the games that then scored it; downstream used `median(best_iteration)` floored at 50 | sampled per trial, identical on every fold and every downstream fit |
| Training window | fixed per experiment | an Optuna hyperparameter (2500/3000/3500/4000) |
| Objective | mean of the fold MAEs | pooled over every validation game |

The first three each change what a trial *is*, so a metric from a run here and a
metric from a run after the change are measurements of different procedures. The
gap is not a detail: the old CV metric was the minimum of a noisy curve over
~1000 candidate stopping points, which inflated Optuna's reported CV accuracy by
a measured ~7pp, and the 50-round floor silently overrode the CV's own answer in
16 of these 38 runs.

## The frozen `_base.yaml`

`_base.yaml` in *this* directory is the shared-defaults file as it stood at
commit `eeca60b`, before the protocol change.

`training_pipeline.cli.find_base_config` walks **upwards** and takes the nearest
`_base.yaml`, so every campaign below this directory inherits the frozen copy
rather than the current one. That is deliberate: re-running
`experiments/archived/window_sweep_2026_08/line_error_3850.yaml` reproduces the
run as it was, on 12 test-anchored folds with per-fold early stopping, instead
of silently re-running it under a protocol it was never designed for.

Verify at any time with:

```bash
poetry run python -m training_pipeline.cli \
  experiments/archived/window_sweep_2026_08/line_error_3850.yaml --dry-run
```

`walk_forward.strategy` should read `test_anchored` and
`optuna.tune_n_estimators` should read `false`.

## What is here

| campaign | what it measured |
|---|---|
| `strategy_window_2026_08/` | the three prediction strategies at two window sizes |
| `window_sweep_2026_08/` | window curve at 3000/3850, sample weighting, fold count |
| `window_overtime_2026_08/` | overtime exclusion, missingness budget, playoffs |
| `intermediate_line_2026_08/` | pooled intermediate line snapshots |
| `total_points/`, `line_error/` | the earliest per-target definitions, kept for parity with the notebooks they came from |
| `runners/` | the shell runners for the above, repathed for their new depth |

The runs these produced are under `artifacts/experiments/archived/`.

## Reading them alongside new runs

`training_pipeline.reporting.discovery` finds runs at any depth, so both
locations resolve without configuration:

```python
SOURCES = [
    "experiments/rolling_origin_2026_08",
    "experiments/archived/window_sweep_2026_08",
]
```

Set `RESCORE_EDGE_THRESHOLD = 0.1` when you do. Each run froze its own bet
threshold into `betting_metrics.json`, so old and new runs otherwise sit in the
same column measuring different things — and that is the *smaller* of the two
incomparabilities on this page.
