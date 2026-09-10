# Closing schema-2.3 fixed-50 campaign

This campaign reruns only the closing-line models affected by MR #31 and the
new `training_data_2_3_20260909.csv`. Intermediate-line datasets are deliberately
out of scope because they do not contain the changed injury/availability input.

## What is held constant

- Seed 16, overtime included, playoffs excluded, regular season plus Play-In.
- History admitted from 2019 and a 90-calendar-day daily walk-forward holdout.
- Twelve latest non-overlapping CV anchors, about 50 games per fold, with starts
  separated by 60 games.
- Target-specific training windows: 4,500 games for spread, 4,000 for total
  points, and 6,250 for line error.
- At most 300 NaNs per row; correlation threshold 0.95 globally and 0.99 for
  `ODDS_` columns.
- 150 Optuna trials, pooled MAE objective, no pruning, and fold-local early
  stopping (1,000-round ceiling, patience 70).
- Lexicographic selection from approximately the best 15% by MAE, capped at
  0.04 MAE from the best trial, with directional win rate as the first tie-break.

The 2.3 file is pinned to checksum `sha256:1d74d01716ce8756`.

The campaign also preserves NaNs in the new availability `MEAN_SE` columns.
Those NaNs mean that precision cannot yet be estimated; the old generic injury
policy would have replaced the injured variants with zero, incorrectly encoding
perfect precision. With that correction, cleaning retains 8,282 rows, including
all 609 holdout games, and every requested training window remains feasible.

## Why 2.2 is not a simple column ablation

MR #31 changes roster assignment, the available/injured classification, and the
historical availability-effect estimator. Compared directly with the latest
availability-safe 2.2 CSV, 2.3 has the same 11,543 games, 16 additional columns,
and changed values in 211 shared columns (282,642 cells by pandas value hashes).
The already completed 2.2 artifacts remain the correct previous-version
reference, but they do not isolate one individual feature family.

## Experiment cells

Part 1 contains the three primary 2.3 Optuna searches:

- `a`: closing spread error;
- `b`: closing total points;
- `c`: closing line error.

Part 2 contains controls:

- `e`, `f`, `g`: fixed-parameter transfers. They apply each selected 2.2 model
  (including its final round count) to 2.3 without retuning. Comparing these to
  the existing 2.2 runs isolates the data change; comparing them to `a`/`b`/`c`
  shows what retuning adds.
- `d`: a full 150-trial rerun of the 2.0 total-points control. It is deliberately
  identical in protocol to the previous 2.0 cell and therefore doubles as a
  reproducibility check. It can be skipped if GPU time is more valuable than
  confirming deterministic reproduction.

The existing comparison artifacts are under
`artifacts/experiments/fixed50_optuna_2026_09/`.

## Run

```bash
bash experiments/runners/run_fixed50_optuna_2_3_2026_09_part1_primary.sh
bash experiments/runners/run_fixed50_optuna_2_3_2026_09_part2_controls.sh
```

The parts are independent and can be assigned to separate GPUs externally.
Within each part, cells run sequentially. Resume without repeating completed
cells with:

```bash
SKIP_EXISTING=1 bash experiments/runners/run_fixed50_optuna_2_3_2026_09_part1_primary.sh
SKIP_EXISTING=1 bash experiments/runners/run_fixed50_optuna_2_3_2026_09_part2_controls.sh
```

All cells use `--no-save-model`; this is an evaluation campaign, not model
promotion.
