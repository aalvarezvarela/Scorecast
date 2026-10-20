# `n_estimators` causal probe

This campaign isolates the change from fold-local early stopping to a single
Optuna-tuned number of boosting rounds.

## Modern-data controls

Cells `a`, `b`, `e`, and `f` are exact controls for the completed runs in
`legacy_cv_replication_2026_09`. They retain the modern data, 90-day holdout,
pooled objective, top-15% / 0.04 tie rule, seed 16, and historical 12x50 fold
layout. Their only intentional change is `tune_n_estimators: false`.

## Historical-data replay

Cells `c`/`d` and `g`/`h` use the July 4 training snapshot and the historical
60-day, season>=2021, max-NA 80, correlation 0.995, mean-fold objective and
fixed +0.10 selection rule. Each pair differs only in `tune_n_estimators`.

The source CSV predates the `ODDS_` naming invariant. Run
`scripts/prepare_legacy_odds_prefixed_csv.py` first. It verifies the original
checksum and changes only the header; every data row is copied byte-for-byte.

The historical line-error configuration requested 4,500 training games even
though its five earliest folds contain only 3,952--4,499 usable games. This is
the behavior of the original run, so part 1 passes the explicit preflight flag
`--allow-short-training-windows` and records a warning instead of silently
changing the historical geometry.

## Closing spread after the rotation-depth leak (cell `i`)

`i_modern22_spread_early_stop.yaml` is not about `n_estimators`. It rides along
in part 1 because it needs the same CSV and the same fold geometry, and because
it is the first `spread_error` run measured after
`clean_dataframe_for_training` started dropping `N_ACTIVE_PLAYERS_*` and
`TOTAL_NON_INJURED_PLAYER_*` (see `nba_ou.config.leakage`). Those 14 columns
aggregate over the players who logged minutes in the game being predicted; the
count correlates **+0.55 with `|HOME_MARGIN|`**.

It is an exact mirror of cell `b` apart from the target and the spread's own
price/comparison columns, so `b` and `i` are readable against each other as
totals-vs-spread on identical geometry.

**Read the cleaning report first.** `cleaning_report.json` must contain a
`rotation_leak` step naming **14** columns, and `feature_schema.json` must hold
**1,409** features rather than 1,421. (The two counts differ because the two
`TOTAL_NON_INJURED_PLAYER_PACE_PER40_*` columns were already being pruned by the
correlation step, so removing them costs the feature matrix nothing.) If the
step is absent, the run measured the old feature matrix and its win rate means
nothing. Every archived spread run
(`extended_closing_spread_2_2`, `cv15_10d_closing_spread_2_2`, the whole
`spread_error_2026_08` campaign) predates the fix and reported 62-67% for this
reason; re-running that geometry through the fixed pipeline gives 52.4%. So a
high number here is a bug report, not a result.

Run the line-error half and total-points half separately:

```bash
bash experiments/runners/run_n_estimators_probe_2026_09_part1.sh
bash experiments/runners/run_n_estimators_probe_2026_09_part2.sh
```
