# Report text with Optuna (2026-11)

**Question:** once the hyperparameters are tuned for them, do each team's
report embeddings (the last game, plus the mean of the last 5) add anything to
line_error at T-480 and T-720?

This is the last attempt on the text line, after
[report_text_last_game_2026_11](../report_text_last_game_2026_11/README.md)
found nothing at the production hyperparameters. That campaign's caveat was
that the hyperparameters had been tuned without text columns, which favours the
control. Here every arm is tuned with the same budget. Research only: the text
columns come from a scratchpad build, not from a schema layer, and nothing here
is promoted. The close is left out: it was the most null horizon, and team and
injury information is fully priced at the close.

## Features

Both blocks use the **target_view** vectors: each report rewritten with the
team as "Target Team" (Qwen3-Embedding-8B), so a team's vector always describes
that team, whether it played the reported game at home or away. The 16
components are the PCA of the V4 residual: the game's box score, both
franchises and the shrunk season running means of the team and its opponent are
removed first. This is the `TXT_TGT16R_` block of the previous campaign,
recomputed and checked equal to it (max abs difference 0).

| block | per team (home and away) | columns |
|---|---|---|
| `TXT_TGT16R_PCnn` | the newest report published strictly before the snapshot, at most 30 days old | 16 x 2 |
| `TXT_TGT16R_L5_PCnn` | the mean of the same 16 components over the team's 5 newest reports published before the snapshot, same season | 16 x 2 |

- The last-5 mean covers fewer reports when fewer exist that season (1-4 in
  about 5% of rows); it equals the last-game block when there is one.
- The residual already removes the season running mean, so the last-5 mean
  reads as how the team's recent reports deviate from its season so far:
  repeated absences, trades, playoff stakes.
- PCA basis and residual ridge fitted once on report seasons 2018-19 to
  2024-25, applied to every row.
- Coverage: 98.7% of rows have both blocks for each team; no report used is the
  game's own.

## Data

`intermediate_line_research_2_5_txt_v2_l5_t480_t720_20261105.parquet`: the
v1 research slice (`intermediate_line_research_2_5_txt_v1_t480_t720_20261104`,
the T-480 and T-720 rows of `intermediate_line_data_2_5_20261016`) with the 32
`TXT_TGT16R_L5_` columns appended. Its `.provenance.json` holds the checks.
The 20261016 parent is kept over the S3 season partitions for the reason given
in the previous campaign's README (`ACTIVE_WEIGHTED_*` zeros at season start).

## Design

| cell | text columns | random_state |
|---|---|---|
| `t{480,720}_a_control` | none (production features) | 16 |
| `t{480,720}_a2_control` | none | 17 |
| `t{480,720}_t_text` | `TXT_TGT16R_` + `TXT_TGT16R_L5_` (64) | 16 |

- **Recipe:** the production tuning recipe
  (`experiments/retrain_line_error_2_5_2026_10/intermediate_line_error_t{480,720}.yaml`),
  unchanged except for the data file, the text exclusions and the trial count.
- **Optuna:** 60 trials per cell (production used 150), a fixed count rather
  than a timeout, so every cell gets the same budget whatever else the machine
  is running. The TPE sampler is seeded by `random_state`, so `a` and `t` start
  from the same random trials.
- **`a2_control`** repeats the control with another seed. The gap between `a`
  and `a2` is the tuning noise floor: two tunings of the same features.
- **Seeds:** `random_state` plus evaluation seeds 101 and 202.
- **Everything else** as production: the 6 x 100-game test-anchored CV, the
  90-day daily walk-forward holdout, NaN budget 800, correlation pruning.

## Pre-registered reading (written before any run)

1. **Holdout only.** Optuna selects on the CV folds in every cell, so CV is
   not a measurement here. Read the paired per-game |error| difference with a
   game-level standard error, primary seed first, then the seed mean. MAE and
   the win rate at 0.1 points, with Wilson intervals, come second.
2. **Decision rule.** `t_text` counts only if all three hold:
   - its paired |error| beats **both** `a_control` and `a2_control` at T-480
     **and** at T-720;
   - at both horizons the gain is larger than the `a` vs `a2` gap;
   - at least one horizon is beyond 2 SE.
3. **Expectation.** Null. If anything shows, it should be at T-720.
4. **What follows.**
   - If it passes: an ablation that drops the last-game and the last-5 blocks
     one at a time.
   - If not: the text-feature line is closed.

Runner: `experiments/runners/run_report_text_optuna_2026_11.sh`.

## Result (2026-11-05): t_text does not pass; the text line is closed

All six cells finished (60 trials each, 87-103 min per cell). Every holdout has
610 games (608 at T-720: two games share a date, total and line and cannot be
paired). Paired values are the first cell's |error| minus the second's on the
primary seed, so negative means the first cell is better. SE is the game-level
standard error. All 64 text columns survive cleaning in `t_text`.

| horizon | cell | holdout MAE, primary | MAE over 3 seeds [range] | win rate over 3 seeds [range] |
|---|---|---|---|---|
| T-480 | a_control | 14.449 | 14.449 [14.443, 14.454] | 55.5% [53.8, 57.8] |
| T-480 | a2_control | 14.376 | 14.468 [14.376, 14.526] | 55.4% [53.4, 57.2] |
| T-480 | t_text | 14.458 | 14.441 [14.411, 14.458] | 55.3% [53.2, 56.5] |
| T-720 | a_control | 14.407 | 14.397 [14.391, 14.407] | 57.6% [55.9, 59.0] |
| T-720 | a2_control | 14.317 | 14.370 [14.317, 14.402] | 57.3% [55.5, 60.1] |
| T-720 | t_text | 14.427 | 14.439 [14.421, 14.468] | 54.8% [53.1, 56.5] |

| horizon | comparison | paired holdout (SE, t) | side changed | first right when changed | paired CV (SE), reference only |
|---|---|---|---|---|---|
| T-480 | t_text - a_control | +0.010 (0.042, +0.2) | 24% | 49.0% | -0.065 (0.042) |
| T-480 | t_text - a2_control | +0.083 (0.059, +1.4) | 25% | 40.9% | -0.076 (0.060) |
| T-480 | a2_control - a_control | -0.073 (0.058, -1.2) | 24% | 58.6% | +0.012 (0.061) |
| T-720 | t_text - a_control | +0.017 (0.057, +0.3) | 27% | 47.0% | -0.014 (0.054) |
| T-720 | t_text - a2_control | +0.109 (0.061, +1.8) | 26% | 37.5% | -0.002 (0.059) |
| T-720 | a2_control - a_control | -0.093 (0.047, -2.0) | 27% | 59.1% | -0.012 (0.046) |

- **No pass.** `t_text` is worse than both controls in the holdout at both
  horizons; it fails every condition of the rule.
- **The tuning noise floor is large.** Two tunings of the same features
  (`a2` vs `a`) differ by 0.07-0.09 MAE in the holdout and change the bet side
  on 24-27% of games. That is as large as any text effect seen in either
  campaign, so a single Optuna run per arm cannot resolve effects this small.
- **CV leans slightly towards text at T-480** (-0.065 and -0.076, about 1.5 SE),
  but CV is Optuna's selection set here and the holdout points the other way.
- Win rates of all cells sit inside each other's ranges.

Conclusion: with tuning, as without it, the report embeddings (last game and
last-5 mean) add nothing detectable to line_error. The text-feature line is
closed, as pre-registered. Analysis script: scratchpad `analyze_txtopt.py`
(session eb774800).
