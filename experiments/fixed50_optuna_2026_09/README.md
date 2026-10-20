# Fixed-50 Optuna campaign

This campaign combines the most useful historical choices with the current
cleaner pipeline. It covers closing and the six-hour (`T-360`) intermediate
snapshot for spread error, total points, and line error. The primary closing
2.2 cells use `training_data_2_2_20260907_availability_safe.csv`. Closing total
points also has a schema-2.0 control, and total points plus line error have
old-2.2 CSV controls.

## Registered protocol

- Availability-safe schema 2.2 for the three primary closing cells, schema 2.2
  unchanged for intermediate, and explicit closing controls on 2.0 plus the
  old 2.2 CSV for total points/line error.
- Seed 16, overtime included, playoffs excluded, regular season and Play-In.
- History admitted from 2019. The intermediate CSV actually starts in 2021.
- Current preprocessing: at most 300 NaNs per row, global correlation 0.95,
  and 0.99 for `ODDS_` columns.
- Ninety-calendar-day daily walk-forward holdout.
- CV `test_anchored`: 12 latest folds, 50 validation games per anchor, and a
  60-game step between starts. Whole game-days may make a fold slightly larger
  than 50; folds do not split a day.
- Pooled MAE objective. The selector admits roughly the best 15% of 150
  completed trials, never more than 0.04 MAE from the best, then uses pooled
  directional win rate, RMSE, MAE, and trial number in that order.
- Fold-local early stopping: up to 1,000 rounds with patience 70. A single
  global `n_estimators` is not tuned. The final round count is derived from the
  selected trial's fold stopping points.
- MedianPruner warmup is 13 for 12 folds, so pruning is deliberately inactive.
- No temporal sample weights and no extra evaluation seeds.

Fifty games is the best-supported fixed size available in this repository: it
reproduces the historical 12-fold protocol and falls inside the 44--81 game
range of the 10-game-day design that transferred best to the daily holdout.
There is no controlled evidence yet that 40, 60, or 75 fixed games is better.

The training window is fixed per target so every Optuna trial is scored on the
same folds. `test_anchored` deliberately rejects tuning this value inside a
study: a larger X can make early folds infeasible and would give different
trials different validation games. The registered X values are:

- closing spread 4,500; closing total points 4,000 (both schemas); closing
  line error 6,250 (the largest rounded window below the 6,293-game minimum
  available to all 12 folds, and the feasible analogue of the 6,500 winner);
- T-360 spread 4,000; T-360 total points 4,000; T-360 line error 3,000.

These are the best-supported target-specific choices from the existing fixed
or temporal panels. The T-360 spread choice is provisional because its older
window search predates removal of the rotation-depth leak. A separate fixed-
parameter X screen is the clean way to revisit these values after this run.

The availability-safe closing CSV has the same 11,543 games as the previous
2.2 file. It removes 14 rotation/availability columns and also recalculates the
historical injury/player aggregates: a direct comparison found changes in 194
shared numeric columns and 497,141 cells at tolerance `1e-12`. Therefore the
old/new total-points and line-error cells are real dataset controls, not expected
duplicates. There is no old-2.2 spread control because the old availability
representation is not a useful modelling candidate for that target.
Intermediate is unchanged because those injury/availability inputs were not
present there.

## Run

Run each part sequentially on the GPU:

```bash
bash experiments/runners/run_fixed50_optuna_2026_09_part1_closing.sh
bash experiments/runners/run_fixed50_optuna_2026_09_part2_intermediate.sh
```

To restart a part without repeating cells that already produced an artifact:

```bash
SKIP_EXISTING=1 bash experiments/runners/run_fixed50_optuna_2026_09_part1_closing.sh
SKIP_EXISTING=1 bash experiments/runners/run_fixed50_optuna_2026_09_part2_intermediate.sh
```

The two runners may be launched independently on different GPUs only if the
device assignment is controlled externally. Within each runner, cells execute
sequentially. They always use `--no-save-model`; these are research experiments,
not production-model promotion.
