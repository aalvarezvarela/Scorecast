# total_points and spread_error with the line_error recipe (2026-10)

The 17 `line_error` slots were promoted on 2026-10-05 from
`prod_rerun_2_5_2026_10`. This campaign gives `total_points` and
`spread_error` the same treatment: one cell per slot, closing plus T-30 ..
T-1080, each built from the line_error config of the same slot with **only the
target changed**. Nothing here is promoted; the runs are read first.

## Design: 2 targets x 17 slots = 34 cells

| | configs | derived from |
|---|---|---|
| intermediate, T-30 .. T-1080 | `intermediate_{total_points,spread_error}_t<N>.yaml` | `prod_rerun_2_5_2026_10/intermediate_line_error_t<N>.yaml` |
| closing | `closing_{total_points,spread_error}_na800.yaml` | `prod_rerun_2_5_2026_10/closing_{total_points,spread_error}.yaml` + the intermediate cleaning, window 6,275 |

**Intermediate cells** differ from their line_error source only in
`prediction_strategy` (plus `line_col: ODDS_TOTAL_LINE_bet365` for
total_points), the CLV comparison column for spread
(`ODDS_CLOSING_SPREAD_LINE_HOME_bet365`), and labels. The recipe:

- data `intermediate_line_data_2_5_20261003.parquet` (`sha256:7e41757f62bc49f4`), one snapshot per cell;
- 2019 floor, `max_na_per_row` 800, correlation 0.95 (`ODDS_` 0.98);
- `train_games` fixed at the slot's longest feasible window (6,275 at T-30 down to 4,400 at T-1080);
- 6 x 100-game test-anchored folds, step 120; 150 Optuna trials with early stopping (1000 cap, 70 rounds); lexicographic selection; no time decay;
- 90-day daily walk-forward holdout (610 games), seed 16 plus evaluation seeds 101 / 202.

All three targets settle against bet365 lines (`ODDS_TOTAL_LINE_bet365`,
`ODDS_SPREAD_LINE_HOME_bet365`), so every cell scores the same holdout games as
its line_error source.

**Closing cells are not a rerun of the closing line_error recipe.** That recipe
(`max_na` 300, `ODDS_` 0.99, window 6,200) was already run for both targets on
the same data in `prod_rerun_2_5_2026_10`:

| | holdout seed-mean win rate | CV win rate | MAE vs line |
|---|---|---|---|
| `prodrerun25_closing_total_points` | 51.3% (49.6-52.5) | 52.5% | +0.059 |
| `prodrerun25_closing_spread_error` | 49.5% (49.0-49.8) | 47.6% | -0.007 |

Running it again would be a replicate. The closing cells here instead take the
intermediate cleaning (`max_na` 800, `ODDS_` 0.98, longest window). That is the
one line_error-derived recipe not yet tried at closing.

**Expect them to be close to a replicate.** Measured on 2026-10-05: on the
rebuilt closing data, `max_na` 300 and 800 keep the same 8,310 rows, and the
smallest fold pool is 6,294 games either way. So the only real differences are:

- `ODDS_` correlation 0.98 instead of 0.99;
- a window of 6,275 instead of 6,200 games.

Read each closing cell against the `prodrerun25_closing_*` run of the same
target. A gap smaller than the seed spread is noise, and then the two together
give the noise floor at closing.

## What to expect

September's intermediate totals/spread runs (`intermediate_horizons_2_5_2026_09`,
`early_totals_spread_window_2_5_2026_09`) were mostly at or below break-even:

- total_points: seed-mean 48-54%;
- spread: 49-54%;
- MAE above the line at most horizons.

Those runs used older data, a 2021 floor, `max_na` 500 and tuned windows. This
campaign tests whether the recipe that lifted line_error to ~55.8% transfers.
Expect most cells to be null. With 32 intermediate cells on one holdout, two or
three will look good by luck.

## Running

```bash
poetry run python scripts/preflight_campaign.py experiments/totals_spread_2_5_2026_10

# The two parts at the same time, in separate terminals
bash experiments/runners/run_totals_spread_2_5_2026_10_part1_total_points.sh
bash experiments/runners/run_totals_spread_2_5_2026_10_part2_spread.sh
# SKIP_EXISTING=1 resumes a part without rerunning finished cells.
```

Each part runs the closing cell (~1.5-2h) and then the 16 horizons (~2.5-3.5h
each with the two parts sharing the GPU): about 45-55h per part. The horizons
are interleaved (T-240, T-720, T-60, T-480, T-960, ...), so the first five runs
already span the range. If none of them is above break-even on every seed,
stopping early loses little. Each run reads one horizon of the Parquet file
(~2-3GB), so both parts fit in memory together. All runs use `--no-save-model`.

## How to read it

The gate is the same as in every campaign, checked per cell:

1. **Seeds.** Does every seed clear break-even (52.4%) on the holdout?
2. **Line.** Does the model beat the line on MAE?
3. **CV and holdout.** Do they agree in sign?

Then:

- **Compare with line_error at the same slot.** Same holdout games, so the
  paired script (`compare.py`-style join on date + target_line +
  actual_outcome) applies. For spread, the truth is `sign(SPREAD_ERROR)`.
- **Horizons are correlated.** line_error wins correlated r=0.35 across
  horizons, so the 16 cells of one target are about 2-3 independent tests,
  not 16. Do not pick horizons by rank.
- **CLV is not edge.** Positive CLV against the bet365 close does not mean the
  bets cover.

Promotion is out of scope here. If a cell passes the gate, it goes through the
same `training_pipeline.promote --to-s3` + `promote_build.py` path as the
line_error slots (see `prod_rerun_2_5_2026_10/README.md`). Intermediate slots
still cannot serve or refit daily until live snapshot features exist.
