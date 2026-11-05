# Data drift at fixed hyperparameters (2026-11)

**Question:** with the production hyperparameters held fixed, do the current
2_5 data train line_error models as well as the 20261016 files those
hyperparameters were tuned on? Better, worse, or within seed noise? If within
noise, Optuna is not rerun.

Evaluation only: nothing here is promoted, and no production pointer, S3
partition or `ENABLED_MODELS` row is touched.

## What changed in 2_5 since 20261016

The S3 season partitions (`train_data/2_5/seasons/<dataset_type>/`) were rebuilt
several times after the 20261016 files (`src/nba_ou/config/dataset_versions.py`):

- #64: preseason and All-Star games dropped from every history.
- #86: alternate-line ticks dropped.
- #107: repair of frozen books in the closing line.
- #112: `ACTIVE/INJURED/QUESTIONABLE_WEIGHTED_*` and
  `N_AVAILABLE_ROSTER_PLAYERS` fixed (players without minutes history).

The last rebuild of every past season ran at `9d96300` (#112) on 2026-11-04/05;
2026-27 was updated by the daily job at `93e182d`.

## Data

`export_partitions.py` reads the partitions with `SeasonPartitions.load()`, the
reader the daily refit uses, and writes them with the builders' Parquet writer
and manifest:

| file | rows | checksum |
|---|---|---|
| `closing_line_data_2_5_20261105.parquet` | 11,664 | `sha256:18f1fead982b9b98` |
| `intermediate_line_data_2_5_20261105.parquet` | 149,075 | `sha256:836224566c99a032` |
| `intermediate_line_data_2_5_20261105_scoring.parquet` | 149,075 | `sha256:b53f57498a428a6a` |

Row counts match the partition manifests season by season. The partitions do
not keep the intermediate scoring sidecar, so the script rebuilds it the way
`_build_scoring_frame` does. It takes the intermediate keys, the closing
partitions' per-game line columns renamed `ODDS_CLOSING_*`, and `TIPOFF_UTC`
from `line_history_aiven.fetch.fetch_games`. Applied to the 20261016 files,
that recipe reproduces their real sidecar exactly: all 146,815 rows and all 28
columns, timestamps included (`--check-recipe`).

**Same rows in both arms.** The `new` arm reads `*_20261105_rows20261016.parquet`
(manifests beside them). These are the export restricted to the rows the
20261016 file could train on: the same key, and a non-null bet365 total line in
the old file.

| file | rows | checksum | dropped from the export |
|---|---|---|---|
| `closing_line_data_2_5_20261105_rows20261016.parquet` | 8,935 | `sha256:23769cc98ded3cf8` | the 121 2026-27 games, and 2,608 old rows without a line (2,592 before the 2019 floor) |
| `intermediate_line_data_2_5_20261105_rows20261016.parquet` (+ `_scoring`) | 146,815 | `sha256:b980765833c22716` / `sha256:42a9c29440130078` | the 2,056 2026-27 rows, and the 2024-03-29 slate (12 games x 17 snapshots) |

The 2024-03-29 slate is the one row-level difference besides 2026-27. The old
intermediate file does not have it, and the old closing file has it without a
line, because it has since been filled from Covers. Left in, it would enter
every training window of the `new` arm and shift the fixed `train_games`
windows by 12 games. Dropping the 16 in-scope line-less rows from the closing
file removes them from cleaning's column statistics too. Rerunning cleaning
showed this does not change the closing feature list.

What remains different between the arms:

- **Values.** Seven closing games changed their bet365 line (#107): five on
  2022-03-31, one on 2024-03-28 and one holdout game on 2026-03-23. At T-480
  one game's line moved (2024-03-28, 225.5 → 224.5, no OVER/UNDER flip); at
  T-720 none. `TOTAL_POINTS` is unchanged everywhere.
- **Rows dropped by `max_na_per_row`.** At the close the cleaned dev sets match
  game for game. At T-480 the NaN budget keeps 1 game in `new` that `old` drops,
  and drops 2 that `old` keeps. At T-720 it swaps 2 for 2. All of these games
  are from 2019-20 or early 2020-21. They reach the training window of at most
  some CV folds (T-480: 2 vs 2 games; T-720: 5 vs 5, because one fold's window
  starts on a different day). They reach none of the holdout's daily training
  windows. The holdout (610 games) and the fold sizes are identical in both
  arms.
- **Feature lists.** Cleaning runs on each arm's own values, so correlation
  pruning picks partly different representatives:

  | slot | old (= production spec) | new | only old | only new |
  |---|---|---|---|---|
  | close | 1,202 | 1,206 | 1 | 5 |
  | T-480 | 2,326 | 2,334 | 6 | 14 |
  | T-720 | 2,309 | 2,311 | 11 | 13 |

  The `old` arm's list equals the production spec's, order included, at all
  three slots. The swaps are siblings above the 0.95 / 0.98 correlation
  threshold: betmgm ↔ fanatics rolling lines, one betrivers roll,
  `ODDS_n_books_total_line_present` at the close, and a few injury/questionable
  aggregates (`TOP4_INJURED_PLAYER_PTS`, `TOTAL_QUESTIONABLE_PLAYER_MIN/PTS`,
  `QUESTIONABLE_WEIGHTED_PACE_PER40`). The constant-column correlation bug is
  fixed (no `|r|=inf` drops). The lists cannot be pinned through the config.
  Excluding the new-only columns with `cleaning.exclude_cols_containing` does
  not bring back the old-only ones, which are pruned against other kept
  columns, and `training_pipeline` has no keep-only-these-features option.
  Adding one would change shared code, which this campaign does not do.
  The daily refit does pin the spec's list (`modeling/refit.py`), so
  production on the current data is "new values, spec features". The `new` arm
  is "new values, re-cleaned features": 0.3-1% of its columns differ from
  production's.

## Design

| cell | data | hyperparameters from |
|---|---|---|
| `close_{old,new}` | closing | production spec `02c92915e70a` (`retrain25_line_error_closing`, trial 77, 76 trees) |
| `t480_{old,new}` | intermediate, T-480 | production spec `aaf2a296a74e` (`retrain25_line_error_t480`, trial 131, 49 trees) |
| `t720_{old,new}` | intermediate, T-720 | production spec `6c8e96ccfa8d` (`retrain25_line_error_t720`, trial 143, 67 trees) |

The specs are read from each slot's production pointer on 2026-11-05. The
11-05 refit builds name them, and their `config` channel names the same spec.

- Each YAML is its slot's `experiments/retrain_line_error_2_5_2026_10/` recipe.
  The changes are `optuna.fixed_params` / `fixed_n_estimators` from the spec,
  `n_trials: 1`, evaluation seeds 101/202/303/404, and the data block. This is
  the setup of `report_text_last_game_2026_11`'s controls, minus their text
  exclusion. **The two arms of a slot differ only in `data`** (`diff close_old.yaml close_new.yaml`).
- Everything else is the production recipe: the 6 x 100-game test-anchored CV
  (step 120); the 90-day daily walk-forward holdout; the NaN budget (300 close,
  800 intermediate); correlation pruning (0.95, `ODDS_` 0.99 / 0.98); the fixed
  window (6,200 / 6,175 / 5,700 games); and no recency weighting.
- **Seeds:** 5 per arm, paired. Each main cell runs its primary seed (17 close,
  16 intermediate) plus evaluation seeds 101-404 on the holdout. CV is scored
  with the primary seed only, so each arm also has four seed-replicate cells
  (`<slot>_<arm>_s<seed>.yaml`: `random_state` = 101..404, no evaluation
  seeds). These give CV, and every seed's holdout predictions, a per-seed
  value. A replicate's holdout must reproduce the main cell's evaluation seed;
  `analyze.py` checks it.
- Runner: `experiments/runners/run_data_drift_fixed_params_2026_11.sh` (main
  cells, then replicates; it accepts cell names to split the work across
  parallel runners).

## Pre-registered reading (written before any result)

`analyze.py` computes, for each slot, on CV and on the holdout separately:

1. **Noise floor first.** Each arm's MAE and win rate across its five seeds
   (range and SD).
2. **Paired difference, new - old,** per seed, for MAE and for the directional
   win rate at every minimum edge (0, **0.1** = primary, 0.5, 1, 2), and ROI at
   -110. A 95% t interval over the five seeds, plus the per-game |error|
   difference averaged over seeds with its game-level SE.
3. **Decision rule.** A slot counts as **better** or **worse** only if the MAE
   difference has the same sign in CV and in the holdout, and its seed interval
   excludes 0 in both. A slot counts as **within noise** if the MAE difference
   is inside the old arm's seed range in both splits, and the win-rate interval
   at 0.1 contains 0. The three slots share most games, so a single slot moving
   alone is not read as a data effect.
4. **Action.** Within noise at every slot: no Optuna rerun; the production
   specs stay valid on the current data. Worse at two or more slots: re-tune on
   the current data. Better: no re-tune is needed either.
5. **Expectation:** within noise. The target is nearly unchanged (OVER/UNDER
   flips on 2 of 11,543 closing games), and the changed features are ~1-3% of
   the columns. Bias: CV folds are the folds these hyperparameters were tuned
   on with the old values and the old feature list. CV therefore favours `old`
   slightly by construction; the holdout is the cleaner read.

## Results (2026-11-05): within noise; no re-tune

All 30 cells ran OK (runner logs `artifacts/logs/data_drift_fixed_params_2026_11_20261105_*`).
Full tables, including every minimum edge and ROI, are in `results/summary.md` (from
`analyze.py`; raw numbers in `results/summary.json`).

**Checks that passed:**

- Each `old` main cell reproduces its `report_text_last_game_2026_11` control bit
  for bit: all five seeds, and the feature list. The close cell's primary-seed
  holdout MAE (14.404704575753604) is the production spec's own vouching
  `holdout_mae`. The `old` arm is the run production was promoted from.
- Every seed replicate's holdout equals the main cell's evaluation seed
  (|ΔMAE| ≤ 2e-15).
- The win rates computed here match the framework's `betting_sweep.csv`.
- Holdout 610 games and CV 624 games, identical in both arms. One closing
  holdout game (2026-03-23) has a different line, from #107.

**new - old at the primary 0.1-point threshold**, five paired seeds. The
"seeds" CI is a t interval over the five per-seed differences. The game-level
SE is for the seed-averaged per-game difference.

| slot | split | MAE old (seed range) | MAE new (seed range) | ΔMAE, seeds [95% CI] | ΔMAE (game SE) | win% old | win% new | Δwin pp, seeds [95% CI] | Δwin pp (game SE) |
|---|---|---|---|---|---|---|---|---|---|
| close | CV | 14.146 (14.131-14.158) | 14.171 (14.157-14.193) | +0.025 [-0.003, +0.053] | +0.025 (0.017) | 51.7 | 49.3 | -2.3 [-3.5, -1.1] | -2.3 (1.0) |
| close | holdout | 14.356 (14.301-14.405) | 14.361 (14.298-14.426) | +0.005 [-0.056, +0.067] | +0.005 (0.017) | 55.3 | 54.6 | -0.7 [-1.7, +0.4] | -0.6 (0.9) |
| T-480 | CV | 14.360 (14.278-14.407) | 14.373 (14.298-14.442) | +0.013 [-0.104, +0.130] | +0.013 (0.022) | 53.4 | 51.7 | -1.7 [-6.4, +3.1] | -1.7 (1.0) |
| T-480 | holdout | 14.440 (14.390-14.511) | 14.404 (14.375-14.436) | -0.036 [-0.105, +0.034] | -0.036 (0.024) | 56.2 | 56.1 | -0.1 [-1.6, +1.4] | -0.1 (1.0) |
| T-720 | CV | 14.468 (14.343-14.592) | 14.483 (14.446-14.518) | +0.015 [-0.106, +0.137] | +0.015 (0.027) | 51.6 | 50.8 | -0.8 [-4.9, +3.3] | -0.8 (1.0) |
| T-720 | holdout | 14.381 (14.362-14.405) | 14.363 (14.330-14.407) | -0.018 [-0.077, +0.041] | -0.018 (0.028) | 57.4 | 57.4 | -0.0 [-3.3, +3.3] | +0.0 (1.0) |

The arms pick a different side on 22-33% of games: the colsample reshuffle a
handful of changed features causes. The other minimum edges (0, 0.5, 1, 2)
show the same picture: every holdout interval contains 0.

**Reading, by the pre-registered rule:**

- **No slot is better or worse.** No MAE difference has a seed interval that
  excludes 0 in either split, let alone in both with the same sign. Every MAE
  difference is inside the `old` arm's own seed range, in CV and holdout alike.
- **One flag: close CV win rate**, -2.3 pp (seed CI [-3.5, -1.1], game-level
  t ≈ 2.3). By the letter of the rule, close is not "within noise" in CV on
  that criterion. It is not confirmed where it would matter:
  - The holdout gives -0.7 [-1.7, +0.4]; its MAE difference is +0.005.
  - The CV MAE difference (+0.025) is inside the old arm's seed range.
  - In CV, every slot leans slightly toward `old` (ΔMAE +0.013 to +0.025;
    Δwin -0.8 to -2.3 pp). In the holdout, MAE leans toward `new` at T-480 and
    T-720. That split is what the pre-registered bias predicts: the
    hyperparameters were selected on these CV folds with the old values and the
    old feature list.
  - Across ~30 slot x split x threshold readings, one 2-SE result is
    expected by chance.
- **Action:** the production specs stay valid on the current 2_5 data. Optuna is
  not rerun for data drift. A re-tune on the current data would remove the CV
  selection bias above, but the holdout shows nothing for it to recover.

**What this does not cover:**

- total_points and spread_error slots, and the other 14 intermediate horizons.
  T-480 and T-720 bracket the horizons that serve best, and the horizons are
  highly correlated (`intermediate-promotion-sweep`).
- Production's exact feature list on the new data. The `new` arm re-cleans, so
  0.3-1% of its columns differ from the spec's (see Data). Production keeps the
  spec's list, a smaller change than this arm tests.

No code bug was found in the shared pipeline. Two harness notes:

- `load_config` resolves `_base.yaml` from the config's own folder, so a config
  copied outside `experiments/` silently loses the base defaults.
- Seed-replicate cells (no evaluation seeds) write no `seed_stability.csv`.
