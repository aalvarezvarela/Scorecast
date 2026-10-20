# Schema 2.3 CV follow-up

This follow-up investigates the two useful questions left by the first schema
2.3 campaign for closing line error and total points. Spread is omitted because
its 2.3 run did not improve holdout MAE and selected a nearly flat 12-tree model.

## Part 1: larger search budget

The `a` and `b` cells reproduce the original fixed-50 protocol exactly, changing
only `n_trials` from 150 to 300. Because persistent Optuna storage was not used
by the original run, this is a deterministic restart rather than a continuation;
with seed 16, its first 150 trials should reproduce the original search.

## Part 2: wider validation blocks

The `c` and `d` cells change CV from 12 folds of about 50 games with a 60-game
step to 6 folds of about 100 games with a 120-game step. Doubling both validation
size and step prevents overlap. Halving the fold count keeps approximately the
same total validation sample and temporal coverage, so the comparison primarily
tests whether each fold's metric becomes less noisy.

Preflight realizes 624 unique validation games (102--106 per fold), versus 643
in the original 12x50 design. The evaluation span remains almost identical and
all six line-error folds reach 6,250 training games; all totals folds reach
4,000.

Everything else is held fixed: schema 2.3, checksum, seed 16, target-specific
training window, cleaning, early stopping, Optuna search space and selector,
90-day daily walk-forward holdout, overtime included and playoffs excluded.

Important interpretation: the current protocol lets every Optuna fold choose
its own early-stopping round, while the deployable refit uses the median of those
rounds. In the first 2.3 line-error run, directional accuracy was 57.34% inside
the Optuna trial but 49.58% when CV was replayed with the final 52-round count.
The larger-budget cells intentionally preserve that behaviour for a controlled
comparison; they should not be read as fixing it. Wider validation blocks may
make stopping less noisy, but a separate fixed/tuned-round experiment would be
needed to remove the mismatch entirely.

## Run

```bash
bash experiments/runners/run_schema23_cv_followup_2026_09_part1_300trials.sh
bash experiments/runners/run_schema23_cv_followup_2026_09_part2_100game_folds.sh
```

The parts may run on separate GPUs. Within each part, line error runs before
total points and cells are sequential. To resume:

```bash
SKIP_EXISTING=1 bash experiments/runners/run_schema23_cv_followup_2026_09_part1_300trials.sh
SKIP_EXISTING=1 bash experiments/runners/run_schema23_cv_followup_2026_09_part2_100game_folds.sh
```

All experiments use `--no-save-model`.
