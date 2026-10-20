# Line error without the global market-regime family (schema 2.5)

Five cells, each the **exact twin** of a horizon in
`experiments/early_line_error_window_2_5_2026_09/` with one deliberate
difference: `cleaning.exclude_cols_containing: ["GLOBAL_"]`.

| this campaign | pairs with | horizon | max train games |
| --- | --- | ---: | ---: |
| `ng_t0_line_error.yaml` | `l_t0_line_error.yaml` | 0 | 4,788 |
| `ng_t30_line_error.yaml` | `l_t30_line_error.yaml` | 30 | 4,788 |
| `ng_t360_line_error.yaml` | `l_t360_line_error.yaml` | 360 | 4,778 |
| `ng_t720_line_error.yaml` | `l_t720_line_error.yaml` | 720 | 4,627 |
| `ng_t1080_line_error.yaml` | `l_t1080_line_error.yaml` | 1080 | 3,775 |

Everything else is byte-identical to the twin: same CSV and checksum, same
90-day holdout, same eight `test_anchored` folds, same `train_games_choices`,
same 150-trial search space, same `evaluation_seeds: [101, 202]`, same
`random_state: 16`. `comparison_group` is deliberately left at
`early_line_error_window_2_5_2026_09` so the leaderboard reads the two sets
against each other.

## What is being dropped

`GLOBAL_*` is the league-wide market-regime family from
`src/nba_ou/data_processing/merged_home_away_data/global_market_features.py`:
rolling bias, MAE, error volatility, tail-miss rates, over/push rates,
scoring-regime levels, open-to-close movement and correction-success, all
computed per game **date** over 15/30/75/150-game and 3/7/14-day windows from
games that finished strictly earlier. 82 of them survive cleaning at T-30.

The substring is safe. All 109 columns in the CSV containing "GLOBAL" start
with `GLOBAL_`; no target, line or metadata column matches, so the pattern and
the family are the same set.

## Pre-registered expectation: no detectable change

Stated before running, because the result is otherwise unreadable.

Two measurements on 2026-09-20, both on the T-30 arm:

* permuting the 82 columns at predict time moved the holdout win rate by
  **+0.24pp** (12 repetitions);
* deleting them and re-running the full daily walk-forward gave **57.17%**
  against a 55.17% baseline, i.e. **+2.00pp**.

The seed range on this config is 0.9pp of win rate, and a Wilson interval at
~600 decided bets is about +/-4pp. **Both numbers sit inside the noise band.**

So the honest reading of a five-cell result:

* if the five twins move by less than ~4pp, the family is **not load-bearing**
  — which is the question worth answering, and the one this can answer;
* the campaign **cannot** establish the sign of the effect. Five cells against
  a ~2pp effect at this noise floor is underpowered, and the cells are not
  independent of each other either: all five score the same 610 holdout games,
  so five agreeing cells are much weaker evidence than five independent ones.

A related caution from the same review: across the parent campaign's 18
horizons the arms are statistically **one population** (homogeneity chi2
p=0.897 on CV, p=0.446 on holdout), and CV rank does not predict holdout rank
(Spearman +0.134, p=0.60). Do not read a horizon ranking out of these five
either.

## Running it

**The pre-flight has not been run yet.** Do that first — it builds the real
splits and reports the actual per-fold training size, which is the check that
catches a window silently shrinking:

```bash
poetry run python scripts/preflight_campaign.py experiments/no_global_line_error_2_5_2026_09
```

`train_games_choices` was inherited from each twin rather than recomputed.
That is safe in the one direction that matters: dropping 82 columns can only
lower a row's NaN count, so `max_na_per_row: 500` can only retain more rows,
never fewer, and the ceiling cannot fall below the inherited value. (It was
already inert at T-30 — the twin's cleaning report removed 0 rows.) If the
pre-flight reports a *higher* ceiling, the largest choice is merely
conservative, not wrong.

Then, ~70 min per cell, ~6 h sequential on one GPU:

```bash
nohup bash experiments/runners/run_no_global_line_error_2_5_2026_09.sh > /dev/null 2>&1 &
tail -f artifacts/logs/no_global_line_error_2_5_2026_09_single_*.log
```

## Reading it

`training_pipeline/reporting/factors.py` gained a `drop_global_market` factor
for this campaign, and `theme.py` renders the arm as `no-global` rather than
`std-cols`. Without both, `summary_experiments.ipynb` would match each cell
with its twin as an *identical* run and the contrast would vanish silently —
the same failure the `referee_features` factor exists to prevent.

Read the pair in `summary_experiments.ipynb` (one variable at a time), not in
`survey_experiments.ipynb`, which is for diagnosing a single run.
