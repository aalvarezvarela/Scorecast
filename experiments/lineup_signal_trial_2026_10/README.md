# Lineup signal trial: 2_5 vs 2_6 at closing and T-360 (fixed hyperparameters)

A small, cheap check of whether the 25 columns schema 2_6 adds over 2_5
(8 `STARTER_*` and 17 `LU_*`) carry signal for `line_error`. Not a promotion
campaign: nothing is tuned.

| Cell | Dataset | Hyperparameters |
|---|---|---|
| `a_closing_2_5` | `closing_line_data_2_5_20261003` | production spec `d055949f99db` (92 trees) |
| `b_closing_2_6` | `closing_line_data_2_6_20261003` | same |
| `c_t360_2_5` | `intermediate_line_data_2_5_20261003`, T-360 | production spec `573805aacb73` (61 trees) |
| `d_t360_2_6` | `intermediate_line_data_2_6_20261003`, T-360 | same |

Each config is generated from the promoted 2_5 configuration
(`closing_prod_rerun_2_5_2026_10/line_error.yaml`,
`intermediate_prod_rerun_2_5_2026_10/line_error_t360.yaml`); a pair differs
only in its dataset file. Device `cpu`, seeds 17/16 + [101, 202, 303].

## Cohorts (measured with `prepare_dataset`, 2026-10-05)

- Closing: identical 8,298 games and targets in both arms; 2_6 keeps every
  2_5 feature and adds 24 of the 25 new columns (`LU_ABSENCE_IMPACT_POSS_BEFORE`
  is removed by correlation pruning). `LU_PROJ_TOTAL_BEFORE` is populated for
  99-100% of every season from 2019-20.
- T-360: 8,271 vs 8,269 games; the 2_6 arm's extra NaNs push two training
  games (2019-12-21, 2021-05-16) over `max_na_per_row: 800`. The 610-game
  holdout is identical. Same feature additions; `LU_*` is populated for
  74-83% of rows per season (both teams' reports must be filed by T-360).

## Caveats

- The hyperparameters were tuned on the 2_5 feature set, which favours the
  control. `colsample_bytree` 0.20 / 0.31 means each tree sees a fifth to a
  third of ~1,250 / ~2,350 columns: 24 new columns are a small share.
- Feature selection for the lineup family inspected 2025-26, which overlaps
  the 90-day holdout, so the holdout is not an untouched test of the family.

## Pre-registration (written before any result)

Expected: null at closing and T-360. Read in the order of the experiments
skill's gate. A difference is not signal unless it exceeds the seed range of
both arms; MAE differences under ~0.05 points and ROI differences inside the
seed range are a null. Four cells, two comparisons: one apparent win is
within what luck produces.

## Results (2026-10-05, runs `artifacts/experiments/lineup_signal_trial_2026_10/`)

Holdout = 610 games, last 90 days; four passes (config seed + 101/202/303).
CV = 624 pooled validation games, config seed only. Paired = same games, both
arms; intervals are 95% bootstrap.

| | Closing 2_5 | Closing 2_6 | T-360 2_5 | T-360 2_6 |
|---|---|---|---|---|
| Holdout MAE, mean of 4 seeds (range) | 14.342 (14.29-14.40) | 14.318 (14.26-14.43) | 14.354 (14.32-14.38) | 14.384 (14.33-14.42) |
| Holdout ROI, mean (range) | +4.8% (+0.7..+7.9) | +4.2% (+3.3..+5.4) | +5.7% (+3.1..+9.0) | +4.6% (+1.2..+6.8) |
| CV MAE / ROI / win | 14.175 / -5.0% / 49.7% | 14.111 / -0.7% / 52.0% | 14.347 / +3.2% / 54.1% | 14.341 / -0.4% / 52.2% |
| Paired holdout abs-error, 2_6 - 2_5 | | -0.040 [-0.20, +0.12] | | +0.049 [-0.03, +0.13] |
| Paired CV abs-error, 2_6 - 2_5 | | -0.064 [-0.22, +0.08] | | -0.006 [-0.08, +0.08] |

Line MAE on the holdout: 14.439 (closing), 14.452 (T-360).

**Verdict: null at both horizons.** Every difference is inside the seed range
(gate question 1), so the gate stops there. Direction is mixed: closing is
slightly better on MAE in both CV and holdout, T-360 slightly worse on the
holdout; ROI moves the opposite way between CV and holdout in both pairs.

Observations worth carrying forward:

- **The arms disagree a lot for a 2% feature change.** Prediction correlation
  between arms is 0.54-0.56 at closing and 0.63-0.71 at T-360; 25-32% of games
  flip side. Adding columns reshuffles `colsample_bytree` draws, so arm-to-arm
  noise is of the same order as seed noise. A per-game comparison needs seed
  ensembles (or many more seeds) to resolve effects of ~0.05 MAE.
- **The new columns do not predict what the 2_5 model leaves.** Spearman rho of
  `LU_ABSENCE_IMPACT_{PTS,DEF,PACE}` and `LU_PROJ_TOTAL` with the 2_5 model's
  residual is -0.03..+0.05 (all p >= 0.10) on ~1,140 CV+holdout games at both
  horizons.
- **Exploratory, post hoc (one of several looks):** at closing, games with
  |absence impact| > 3 points (n 352) show 2_6 abs-error lower by 0.24
  (SE 0.12); the rest -0.01 (SE 0.07). Not seen at T-360 (-0.02, SE 0.06).
  Consistent with the pre-registered "concentrated in absence games", but a
  ~2 SE subgroup after multiple looks is a hypothesis, not a finding.
