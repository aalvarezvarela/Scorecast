---
name: experiments
description: How models are trained and evaluated in this repo - the training_pipeline package, campaigns under experiments/, and runs under artifacts/experiments/. Use when creating or editing experiment configs, running a campaign, comparing runs, adding a metric/filter/threshold to the pipeline, judging whether a result is real, or before concluding that one configuration beats another.
---

# How models are trained here

The task is predicting a market that is close to efficient. The edge being
chased is small relative to the noise in any single measurement, so the
pipeline is built around **measuring the same thing several ways** and refusing
to rank on the weakest of them. Most of what follows exists for that reason.

Use this for: campaign design, config edits, running campaigns, comparing runs,
extending `training_pipeline/`, and any moment you are about to say
"configuration A beats configuration B". Not for feature engineering in
`src/nba_ou/data_processing/`, DB/ingestion work, the Streamlit app, or
prediction serving.

## What one run does

`training_pipeline.pipeline.run_experiment(config)`, in order:

1. **`prepare_dataset`** — load the CSV, verify its sha256 against
   `data.expected_checksum`, apply `season_year_floor`, filter season types via
   the **GAME_ID prefix** (not the `SEASON_TYPE` text column, which mislabels
   play-in games), derive the target, then clean.
2. **`build_holdout_split`** — the last `holdout.test_days` (60) become the
   holdout. Everything earlier is *dev*.
3. **`build_walk_forward_splits`** — time-aware folds over dev, anchored on
   dates. `exclude_test_months = (5, 6)` keeps playoff months out of every
   validation window. Training-row filters are applied here to **TRAIN indices
   only** (`apply_training_filter`).
4. **`strategy.tune`** — Optuna over the fold set. Since 2026-08-26 the
   selected trial is simply the **lowest primary metric** (`refit.
   use_lexicographic_selection: false` in `_base.yaml`). Runs before that date
   used a lexicographic pick — best primary metric within a tolerance band,
   then the best betting outcome — on the argument that the primary metric
   alone cannot rank trials this close together. The band was doing too much:
   it often handed the choice to pooled OU accuracy, which carries ~1.7pp of
   binomial noise on ~850 games. The `tie_*` metadata and
   `optuna_lexicographic_candidates.csv` are still written, but now describe a
   decision nothing acts on. **A new run is not selector-comparable with an
   archived one**; set the flag back to `true` to reproduce one.
5. **Holdout evaluation** — by default `daily_walk_forward`: retrain once per
   test game-day on everything strictly earlier (dev plus already-played test
   days), predict only that day, pool. This mirrors production. `single_shot`
   fits once and is for smoke runs.
6. **`evaluate_cv_betting`** — refit at the chosen hyperparameters on each
   fold and score the pooled validation games for profit.
7. **Seed loop** — repeat step 5 for each of `evaluation_seeds`.
8. **Production refit** — only if `refit.train_production_model`; fitted on the
   tail of *all* data, never scored.
9. **Save artifacts** to `artifacts/experiments/<name>_<timestamp>/`.

## The three strategies

All answer the same question — OVER or UNDER this line — but differ in how much
model capacity goes to the decision versus to reproducing the line:

| strategy | target | `predicted_edge` units | has MAE | has probabilities |
|---|---|---|---|---|
| `total_points_regressor` | game total | points vs line | yes | no |
| `line_error_regressor` | total − line | points vs line | yes | no |
| `over_under_classifier` | P(OVER) | **EV difference** | no | yes |

Consequences that bite:

- **ROI, win rate and bet volume are the only measures comparable across all
  three.** MAE is regressor-only; log-loss/Brier/calibration classifier-only.
- **Bet thresholds are not in the same units.** Regressors select on
  `|edge|` in points (`primary_edge_threshold`); the classifier on expected
  value (`primary_ev_threshold`). The same number means different things.
  `bet_rate` is the only selectivity measure comparable across strategies.

## Two measurements, deliberately

Every run reports both, and they answer different questions:

- **Cross-validation** (~650 pooled validation games) — the folds that *chose*
  the hyperparameters, so optimistically biased. Use it to **rank**
  configurations; the bias applies roughly equally to each.
- **Holdout** (~416 games) — never used for selection, so honest, but small
  enough to be lucky. Use it to **estimate**, never to rank alone.

The gap between them is itself a result: a large `cv_minus_holdout_roi` is the
signature of selection overfitting.

Both are always compared against **trusting the bookmaker's line**, plus a
harder "line + its historical drift" null. A model that only rediscovers a
league-wide over/under tendency beats the first and not the second.

## The gate

Answer in order. Stop at the first failure — later questions are meaningless
once an earlier one fails.

1. **Is the difference bigger than one run's own seed noise?** Refitting the
   same config with a different seed moves ROI by several points. Read
   `seed_roi_range` first, always.
2. **Is there enough volume to conclude anything?** At −110 break-even is
   **52.38%**, and a Wilson interval around a genuine 55% win rate does not
   clear it at 114, 600, or even 1200 bets. Compute the interval before
   believing a rate.
3. **Do CV and the holdout agree?** A large gap in *either* direction is a
   warning, not just a negative one.
4. **Is the comparison legitimate?** Same holdout cohort, same threshold in the
   same units, and nothing scored against a column the model was given.
5. Only then, the result — as a hypothesis for the next campaign.

## The failure mode to expect: silent no-ops

Nearly every real bug found in this pipeline **did nothing quietly** rather than
erroring. When a change appears to have no effect, suspect these before
concluding the effect is absent.

| symptom | pattern |
|---|---|
| every seed gave byte-identical output | override merged *before* the value it must beat |
| Optuna tuned a knob it never applied | optional arg whose absence disables a feature (`dates=None`) |
| a trial's "no weighting" was reinstated | `None` overloaded to mean two opposite things |
| one code path ignored a config flag | filter applied at N−1 of N fit sites |
| runs vanished from a comparison | loader skips on a missing column and names nothing |
| a threshold changed nothing | knob preempted by another mechanism (`exclude_cols_containing`) |
| window larger than the data | `tail(n)` reads as "require n"; folds shrink, no error |
| a filter matched zero rows | dtype inference dropped a leading zero from `GAME_ID` |
| spectacular ROI against another line | scored against a column the model has as a feature |
| old and new runs disagree wildly | an analysis choice was frozen into artifacts at run time |

**Two habits that catch these.** Mutation-test every new test: revert the fix,
confirm the test fails, restore — a test that passes both ways is worse than
none. And verify rather than assert: grep the real call sites, check the column
exists, measure the row count. Estimates here have been wrong more than once,
always in the direction of "looks fine".

## Configs and campaigns

Layering is a deep merge: `experiments/_base.yaml` → the campaign YAML →
pydantic defaults in `training_pipeline/config.py`. A campaign is a folder of
YAMLs under `experiments/<campaign>/`; runners live in `experiments/runners/`
and resolve the repo root with `cd "$(dirname "$0")/../.."`.

`ExperimentConfig.fingerprint()` keys persistent Optuna storage. Data, cleaning,
folds, search space and objective are **in** it; labels, output paths and
post-hoc `betting.*` settings are **out**, so changing a bet threshold does not
fork a study. One exception to remember: `primary_ev_threshold` does reach
classifier trial selection through `mean_ou_acc`/`mean_roi`.

Designing a campaign:

1. **Measure feasibility first.** Window ceilings depend on the row count
   *after* cleaning and on how date-anchored folds reach back, so they cannot be
   inferred from the CSV or from arithmetic.
2. **One deliberate difference per cell.** If a cell changes two things, say so
   in its `hypothesis` and only read it against the single-change cells.
3. **Pin `expected_checksum`.** A regenerated dataset must not pass silently; the builders print the checksum of the Parquet file they write.
4. **Set `evaluation_seeds`** — without them nothing has an error bar.
5. **Fix `n_trials`, disable `timeout`.** A timeout makes trial count a function
   of the machine, silently un-matching runs meant to be comparable.
6. **Pre-register expectations** in the config header with a
   multiple-comparisons budget: at ~10 cells against this noise floor, expect
   about one to look good by luck.
7. **Run the pre-flight**, then commit the GPU time:

```bash
poetry run python scripts/preflight_campaign.py experiments/<campaign>
nohup bash experiments/runners/run_<campaign>.sh > /dev/null 2>&1 &
tail -f artifacts/logs/<campaign>_*.log
```

The pre-flight parses every config, verifies checksums, prints the design
matrix, and — the reason it exists — builds the real splits and reports the
**actual per-fold training size**, catching a window that would otherwise
silently shrink. `--skip-data` skips exactly that check, so it is not a
substitute. Runners use `set -uo pipefail` *without* `-e`, so one failed run
does not cancel the rest; order cells so single-change ones run first.

## Reading results

`experiments/notebooks/survey_experiments.ipynb`. `SOURCES` accepts any mix of:
an artifacts folder, a campaign *config* folder (resolved to the runs it
produced), a single config file, or one run directory.

Set `RESCORE_EDGE_THRESHOLD` / `RESCORE_EV_THRESHOLD` to re-score every run at
one common threshold from raw predictions. **Do this whenever runs span a config
change** — each run freezes its own threshold into `betting_metrics.json`, so
otherwise old and new runs sit in the same column measuring different things.

`experiments/notebooks/summary_experiments.ipynb` is the cross-campaign view:
every run's CV and holdout win rate on one page, then **one variable at a
time**. It reduces each `config.json` to a factor vector and only compares runs
agreeing on every factor but the one being read, so a cell that moved two knobs
appears in no comparison. Use it for "what does changing X do?"; use
`survey_experiments.ipynb` for "what is wrong with this run?".

Helpers live in `training_pipeline/reporting/` (`discovery`, `loaders`,
`charts`, `factors`, `narrative`, `rescore`, `theme`). Extend those rather than
growing notebook code. If a campaign varies a knob not in
`factors.FACTOR_SOURCES`, add it — otherwise its runs are matched as though
they were identical, which is exactly the silent no-op this table warns about. Chart conventions there are deliberate: dumbbells rather than
truncated bars when values differ by ~0.1 on a base of ~14; small multiples
rather than overlaying runs that share a strategy colour; a plain rule for the
seed range, because it is an observed min–max and not a confidence interval.

## Extending the pipeline

- **Training-only filters** (e.g. `exclude_overtime_from_training`) must be
  applied at *every* fit site: CV folds via `splits.apply_training_filter`, the
  daily walk-forward, the production refit, **and** the single-shot path.
  Evaluation must keep those rows — excluding them from scoring measures a world
  that does not exist.
- **Post-hoc scoring** belongs in `betting.*` so it stays out of the
  fingerprint and can be re-derived at read time instead of frozen per run.
- Run `ruff`, `mypy`, and the full `pytest` suite. Some failures pre-date
  current work; confirm by stashing before blaming a change for one.

## Current measured findings

**Snapshot — regenerate after each campaign; do not treat as fixed.** Produce
these from `survey_experiments.ipynb` (§0 and §3) and
`scripts/preflight_campaign.py`.

*As of the `strategy_window_2026_08` campaign, 9 runs, re-scored at a common
0.1-point / 0.0-EV threshold:*

- `line_error_regressor` leads: CV 53.9% → holdout 56.0% win rate, ROI +6.8% on
  ~400 bets. `total_points_regressor` +1.7%; `over_under_classifier` −0.4%.
- Seed ROI range, same config: **4.9–12.0 points**. Nothing smaller than that is
  a result.
- Only **3 of 7** runs agreed even on the *sign* of their edge between CV and
  holdout.
- **Predicted edge does not predict winning.** Rank correlation with win rate is
  **+0.02** in CV; no run reaches significance. The former `min_edge: 2.0`
  discarded **64% of volume** for **+0.2 points** of win rate, which is why the
  default is now 0.1.
- **Optuna's reported CV accuracy is inflated ~7 points** (11 of 12 folds),
  because each fold early-stops on the fold it is then scored on. `cv_win_rate`
  is the conservative, holdout-comparable number.
- Overtime games: 5.2% of games, **+21.2 points** vs the line, **85.5% OVER**.
- Keeping playoffs collapses the 60-day holdout to **89 games** (85 playoff),
  because the data ends mid-playoffs. CV is unaffected.
- `nan_threshold` is **inert** on the 2.0 build — 25/50/80/100 all keep 1890
  columns. `max_na_per_row` is the real lever: 80 → 200 recovers ~390 rows, most
  from season 2023, whose consensus-percentage columns are 62–85% missing.
- Measured window ceilings at 12 folds (2.0 build): ~3950 regressors, ~3880
  classifier at `max_na=80`; ~4180 at 120; ~4390 at 200; 3750 on the old build.
