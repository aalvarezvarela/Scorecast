# Report text, last game (2026-11)

**Question:** do the report embeddings of each team's previous game add anything
to line_error, at the close, T-480 and T-720?

Research only: the text columns come from a scratchpad build, not from a schema
layer, and nothing here is promoted.

## Features

For each team, the newest NBA.com report (Qwen3-Embedding-8B, 4096 dims)
**published before the snapshot time**: before tip-off for the close, before the
snapshot timestamp for T-480 and T-720, at most 30 days old. In 98-99% of games
this is the team's previous game. The exceptions are mostly back-to-backs at
T-720, where last night's report is not out yet, so the one before it is used.
Report publication runs from 2.1 to 45 hours after tip-off, with a median of 3.8.

| block | vector | reduction | per team |
|---|---|---|---|
| `TXT_TGT64_PCnn` | target_view (the team rewritten as "Target Team") | PCA | 64 |
| `TXT_TGT16R_PCnn` | target_view | PCA of the V4 residual: the game's box score, both franchises and the shrunk running mean of the team and of its opponent are removed first | 16 |
| `TXT_RAW64_PCnn` | raw report (one vector per game, no perspective) | PCA | 64 |
| `TXT_RAW16R_PCnn` | raw report | PCA of the residual on both teams' box score, both franchises and both teams' running means | 16 |

PCA bases and residual ridges are fitted **once**, on report seasons 2018-19 to
2024-25, and applied to every row. Every season of a fold therefore sees the same
axes. The CV folds and the holdout are in 2025-26.

The sizes come from a reduction study on the same vectors. It used a forward
ridge probe for a next-game new absence, scored over stats and franchise
controls:

- PCA 64 of the raw target_view keeps pairwise similarities at Spearman 0.98
  and gives the best probe gain (+0.023 AUC).
- PCA 16 of the residual gives nearly the same gain (+0.020) in a quarter of the
  columns.
- Truncation to the first k dimensions performed no better than a random
  projection.

Coverage is 97.6% of games in 2019-20 and about 98.9% from 2020-21 on; the
missing games are mostly season openers.

## Data

- **Close:** `closing_line_research_2_5_txt_v1_20261104.parquet`. This is
  `closing_line_data_2_5_20261016.parquet` with the 320 text columns appended;
  the parent's values are untouched.
- **T-480 and T-720:** `intermediate_line_research_2_5_txt_v1_t480_t720_20261104.parquet`.
  This is a slice of `intermediate_line_data_2_5_20261016.parquet` holding its
  T-480 and T-720 rows only, plus the text columns. 231 of its 17,381 rows have
  no scoring snapshot, so their text columns are NaN.

Both files have a `.provenance.json` alongside them.

The 20261016 files were chosen deliberately over the S3 season partitions that
the daily refit reads. The partitions were rebuilt on 2026-10-27 and carry later
odds fixes, but their `ACTIVE_WEIGHTED_*` player features are 0 in 39-65% of
each season's first 107 games, against 2-4% in the 20261016 files. That pattern
looks like season-scoped history at season start; it is unconfirmed and is
reported separately. Text columns could look useful merely by filling that hole.
The production hyperparameters below were also tuned on the 20261016 files.

## Design

| cell | data | hyperparameters from |
|---|---|---|
| `close_*` | close | production spec `02c92915e70a`, from `retrain25_line_error_closing` (trial 77, 76 trees) |
| `t480_*` | T-480 slice | production spec `aaf2a296a74e`, from `retrain25_line_error_t480` (trial 131, 49 trees) |
| `t720_*` | T-720 slice | production spec `6c8e96ccfa8d`, from `retrain25_line_error_t720` (trial 143, 67 trees) |

There are five arms per horizon. All of them read the same file, and they differ
only in `cleaning.exclude_cols_containing`:

| arm | text columns kept |
|---|---|
| `a_control` | none (equivalent to production) |
| `b_tgt64` | `TXT_TGT64_` (128) |
| `c_tgt16r` | `TXT_TGT16R_` (32) |
| `d_raw64` | `TXT_RAW64_` (128) |
| `e_raw16r` | `TXT_RAW16R_` (32) |

- **No tuning.** `optuna.fixed_params` uses the served hyperparameters. They
  were tuned without text columns, so any bias favours the control.
- **Seeds:** `random_state` (17 at the close, 16 at the intermediate
  horizons), plus evaluation seeds 101, 202, 303 and 404.
- **Everything else:** the production recipe — the 6 x 100-game test-anchored
  CV, the 90-day daily walk-forward holdout, the NaN budget (300 at the close,
  800 at the intermediate horizons) and the correlation pruning.

## Pre-registered reading (written before any run)

1. **Holdout first.** Compare each arm with the control on the paired per-game
   |error| difference, averaged over the five seeds, with a game-level standard
   error. MAE and win rate at the 0.1-point threshold, with a Wilson interval,
   come second. The seed range within each arm is the noise floor.
2. **CV second.** It is read only as a check on direction: the CV folds are the
   folds the hyperparameters were tuned on.
3. **Decision rule.** An arm counts only if it improves the paired |error| in
   both the holdout and CV, at two of the three horizons. The horizons share
   most of their games, so they are not independent tests.
4. **Expectation.** Null at the close. If anything shows, it should be target
   over raw, and at T-720 rather than at the close. Twelve arm-vs-control
   comparisons means about one false positive is expected by luck.
5. **What follows.**
   - If an arm passes: per-arm Optuna (150 trials), then evolution features
     (EWMA, drift, last-game shock).
   - If nothing passes: stop the text-feature line, or check whether a schema
     layer would differ at all before investing more.

Runner: `experiments/runners/run_report_text_last_game_2026_11.sh`.

## Result (2026-11-04): no arm passes; the 64-dim arms hurt in CV

The holdout has 610 games (608 at T-720: two games with an identical date,
total and line cannot be paired and are dropped). CV has 624 games. Paired
values are arm |error| minus control |error| on the primary seed, so negative
means the arm is better. SE is the game-level standard error. MAE is the mean
over the five seeds.

| horizon | arm | holdout MAE | paired holdout (SE) | paired CV (SE) | side changed | arm right when changed |
|---|---|---|---|---|---|---|
| close | control | 14.356 | | | | |
| close | tgt64 | 14.394 | -0.042 (0.038) | +0.083 (0.042) | 25% | 51.7% |
| close | tgt16r | 14.360 | +0.001 (0.014) | -0.014 (0.016) | 8% | 56.2% |
| close | raw64 | 14.377 | -0.076 (0.040) | +0.049 (0.039) | 27% | 56.3% |
| close | raw16r | 14.361 | -0.010 (0.016) | -0.015 (0.016) | 10% | 55.0% |
| T-480 | control | 14.440 | | | | |
| T-480 | tgt64 | 14.473 | -0.015 (0.048) | +0.094 (0.051) | 27% | 51.2% |
| T-480 | tgt16r | 14.424 | -0.037 (0.022) | -0.007 (0.024) | 10% | 56.5% |
| T-480 | raw64 | 14.465 | +0.056 (0.050) | +0.139 (0.051) | 31% | 47.6% |
| T-480 | raw16r | 14.435 | -0.027 (0.022) | +0.006 (0.023) | 12% | 58.3% |
| T-720 | control | 14.381 | | | | |
| T-720 | tgt64 | 14.360 | +0.017 (0.060) | +0.163 (0.057) | 27% | 44.7% |
| T-720 | tgt16r | 14.377 | -0.070 (0.030) | +0.062 (0.029) | 15% | 50.0% |
| T-720 | raw64 | 14.418 | +0.122 (0.059) | +0.212 (0.059) | 25% | 41.7% |
| T-720 | raw16r | 14.374 | -0.001 (0.034) | +0.052 (0.030) | 16% | 42.7% |

- **No arm passes the pre-registered rule.** That rule needs paired
  improvement in both the holdout and CV at two of three horizons; the best arm
  is `tgt16r`, which does it only at T-480 (-0.037 / -0.007, both within
  noise).
- **The 64-dim arms (128 columns) are worse in CV at every horizon** (t from
  +1.3 to +3.6). They flip the bet side on 25-31% of games without winning
  those games more often. At T-720 `raw64` is also worse in the holdout
  (t = +2.1). Part of the CV gap is expected: the control's hyperparameters were
  selected on these folds with its own features.
- **The 16-dim residual arms are neutral.** Every paired difference is within
  about 2 SE, and they change the bet side on only 8-16% of games.
- **The single holdout gain beyond 2 SE**, `tgt16r` at T-720 (-0.070, t = -2.3),
  is contradicted by CV (+0.062, t = +2.1). With 12 comparisons it is
  consistent with luck.
- **Win rates.** Every arm's holdout win rate is inside every other arm's Wilson
  interval. The seed range of holdout MAE within one arm (0.05-0.15) is wider
  than every arm-vs-control MAE gap.
- **Target vs raw.** Target is not clearly better than raw. At 64 dims raw is
  the worst arm at T-480 and T-720.

**Reading:** with the served hyperparameters, the previous game's report adds no
detectable value to line_error at the close, T-480 or T-720. The compact
residual versions do no harm; the PCA-64 versions add noise.

Analysis script: scratchpad `analyze_txt_campaign.py` (session 2026-11-04);
numbers in `analyze_txt_campaign.json` there.
