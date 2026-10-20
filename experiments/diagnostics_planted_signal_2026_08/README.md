# Planted-signal diagnostic — `line_error`

These four runs do not measure the model. They measure **this pipeline's ability
to find a weak signal it is handed**, which is the question every negative
`line_error` result quietly depends on.

`line_error` runs score a holdout R² of ~0.001 and beat the closing line's MAE by
~0.02 points. Two very different worlds produce that number:

1. the market is close to efficient and there is almost nothing to find;
2. this preprocessing / search space / CV / trial budget cannot recover weak
   signal even when it is definitely present.

A planted signal separates them. One synthetic feature, `PLANTED_SIGNAL`,
carries a known fraction of `LINE_ERROR`'s variance. Everything else — dataset,
folds, search space, seed, trial budget, preprocessing, every real feature — is
byte-identical across the four cells. If pooled out-of-fold performance does not
improve as the planted strength rises, then "no signal found" was never evidence
about the market.

## Dataset

`data/train_data/training_data_2_0_20260704.csv` (pinned:
`sha256:2fc9ed86d2f42a78`), overriding `../_base.yaml`'s default. Measured:

| | value |
|---|---|
| cleaned rows / features | 5,752 / 1,458 (+ `PLANTED_SIGNAL`) |
| dev / holdout | 5,336 / 416 |
| holdout window | 2026-02-19 → 2026-04-17 |
| rolling-origin folds | 30 (117 game-days, 850 validation games) |
| window ceiling | 4,486 — the inherited `[2500, 3000, 3500, 4000]` all fit |

**If the real `rolling_origin_2026_08` campaign runs on a different CSV, this
diagnostic's conclusion does not transfer to it.** The whole point is to
exercise the pipeline you intend to trust, and the dataset is part of that
pipeline. Keep the two campaigns on the same build, or read the diagnostic as a
statement about this build only.

## The cells

| cell | `variance_explained` | expected corr | role |
|---|---|---|---|
| `control_000` | 0.000 | 0.00 | control: `PLANTED_SIGNAL` is pure independent noise |
| `signal_005`  | 0.005 | 0.071 | weak |
| `signal_010`  | 0.010 | 0.100 | the main test |
| `signal_020`  | 0.020 | 0.141 | easy positive control |

The 0% cell runs through the *same* planted-signal machinery rather than
pointing at an unrelated historical run, so it controls for the cost of one
extra random column rather than for a different experiment.

For scale: 1% of `LINE_ERROR` variance is roughly the size of edge these models
are hunting. A pipeline that cannot see it planted cannot see it real.

## Reading the result

The question is **not** whether `PLANTED_SIGNAL` shows up in feature importance.
It is whether pooled out-of-fold performance moves. Importance is recorded as a
supporting diagnostic — a model can split on a feature and gain nothing.

Generate the comparison with:

```bash
poetry run python scripts/compare_planted_signal.py
```

Read `fold_use_rate` first. If it is 0.00 at 2% planted variance, the tree
builder never once chose a feature explaining 2% of the target across 30 folds,
and the search space — not the market — is the binding constraint.

## Safety

Every run here carries a target-derived feature. Four independent guards stop
one being mistaken for a real model:

- `experiment_name` must start with `diag_planted` (config validation);
- `refit.train_production_model` must be false (config validation);
- `run_experiment(save_model=True)` refuses;
- `training_pipeline.promote` refuses.

Each run also writes `planted_signal.json` and sets `is_diagnostic: true` in
`metadata.json`. The presence of that file is by itself the answer to "is this
run real?".
