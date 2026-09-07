# Fixed-hyperparameter preprocessing screen

This six-cell line-error screen holds the modern 2.2 dataset, seed, target,
12x50 anchored folds, 4,000-game training window, 90-day daily walk-forward
holdout, XGBoost parameters and 28 boosting rounds fixed. Only row missingness
and correlated-column pruning change.

Cells `a`--`c` compare `max_na_per_row` 300, 80 and 150 under the current
correlation policy (0.95 globally, 0.99 for odds). Cells `d` and `e` move only
the correlation cutoff to 0.99 and 0.995. Cell `f` checks the interaction of
the two historical values (80 and 0.995).

These are CV-only screening results, not final model searches: the fixed
parameters came from the completed modern-2.2 early-stopping run. The holdout
is deliberately not scored. The winning cleaning region must be confirmed
with a small Optuna repeat before becoming a default.
The 4,000-game window is deliberate: at a threshold of 80, the earliest folds
only have about 4,140 rows. Keeping 4,500 would confound missingness with a
different effective training-window size.

```bash
bash experiments/runners/run_preprocessing_screen_2026_09.sh
```

## Screening result

The CV-only screen evaluated the same 643 validation games in every cell. The
current policy (`max_na_per_row=300`, global correlation 0.95 and odds
correlation 0.99) ranked first: pooled MAE 13.6918 and 52.52% wins at edge 0.1.
The full table is `artifacts/probes/preprocessing_screen_2026_09/summary.csv`.

Paired game bootstrap against the current policy found clear MAE degradation
for `NA=80/corr=0.95` (+0.1199, 95% CI +0.0338 to +0.2068) and
`NA=300/corr=0.995` (+0.1142, 95% CI +0.0262 to +0.2041). No tested alternative
improved the control.

An earlier implementation accidentally scored one holdout cell before the
CV-only guard was added:
`artifacts/experiments/preprocessing_screen_2026_09/prep_screen_line_2_2_na300_corr095_20260907_093105`.
It is methodologically invalid for selecting preprocessing and must not be used
or compared. The current runner writes only under `artifacts/probes/` and its
manifest explicitly records `holdout_scored: false`.
