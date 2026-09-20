# Schema 2.5 intermediate lines: T-360, T-720, T-960

The first training runs on `intermediate_line_data_2_5_20260613.csv` (built
2026-09-18, pinned to `sha256:5bda016d4c872e0d`). This build is also the first
to contain the market-dynamics columns from `feat/intermediate-market-dynamics`
(G1-G4 in `docs/intermediate_market_dynamics_plan.md`). This campaign measures
the full feature set by timepoint. The group ablations are a separate campaign
(plan section 6).

## Design: 3 targets x 3 timepoints

| | T-360 (6 h) | T-720 (12 h) | T-960 (16 h) |
|---|---|---|---|
| spread error | `s_t360_spread` | `s_t720_spread` | `s_t960_spread` |
| line error | `l_t360_line_error` | `l_t720_line_error` | `l_t960_line_error` |
| total points (bet365) | `t_t360_total_points` | `t_t720_total_points` | `t_t960_total_points` |

Each cell is one model trained on one snapshot (`snapshot_minutes`). All 9
cells share every other setting, so each row reads as a curve of one target
across timepoints:

- seasons from 2021 (`season_year_floor: 2021`), regular season plus play-in,
  overtime included;
- `max_na_per_row: 500`; column NaN threshold 50% (the `_base.yaml` default);
  correlation pruning at 0.95 (0.99 for `ODDS_`);
- **3,500 training games for every target and timepoint** (see Feasibility);
- CV of 6 x 100-game folds with a 120-game step, fold-local early stopping;
- 150 Optuna trials with no timeout; pooled MAE objective; lexicographic
  selection;
- 90-day daily walk-forward holdout; seed 16 plus `evaluation_seeds: [101, 202]`;
- bets at |edge| >= 0.1, settled at a flat -110 against the snapshot line;
- CLV is read from the `ODDS_CLOSING_*_bet365` columns in the scoring sidecar.
  These columns never enter the feature matrix.

Why these choices:

- **Floor 2021 and 500 NaNs.** Measured on this file, 2019-20 rows carry
  400-1,100 NaNs each, because about 275 book, consensus and snapshot columns
  do not exist before 2021-22. From 2021 on, a cutoff of 500 keeps 98.4-100%
  of games at every timepoint. A cutoff of 300 would lose up to about 400 games
  at T-960.
- **6 x 100 folds with early stopping.** This led on schema 2.3 and is cell `b`
  of `schema25_closing_2026_09`. If that campaign shows tuned rounds (cell `c`)
  are better, rerun these cells with `tune_n_estimators: true` as a
  one-change follow-up.
- **One window for all cells.** The closing campaign uses per-target windows
  (4,500 / 6,200 / 4,000), but here a common 3,500 keeps timepoints and
  targets comparable.

## Feasibility (measured by the preflight)

| | cleaned rows | dev | holdout | largest window every fold supports |
|---|---|---|---|---|
| T-360 | 6,155 | 5,545 | 610 | > 4,000 |
| T-720 | 6,001-6,004 | ~5,392 | 610 | ~3,985 |
| T-960 | 5,572-5,575 | ~4,963 | 610 | ~3,567 |

With 4,000 games, the earliest T-960 folds would have quietly trained on less.
3,500 fits every cell.

## Pre-registered reading

Use the same gate as every campaign: seed noise first, then Wilson intervals
against 52.38%, then CV/holdout agreement, and only then comparisons.

For intermediate lines, also check:

1. **Error at T against the line at T.** Compare MAE against the snapshot
   line's own MAE. The line is weaker further out, so beating it is easier at
   T-960 and means less.
2. **CLV against the bet365 close.** Do the model's bets move with the market?
   CLV is not edge: in the market-dynamics research, bets with positive CLV
   still covered only 52-53%.
3. **Different games per timepoint.** Games without a line 16 h out are absent
   at T-960 (5,572 vs 6,155 rows at T-360). A horizon curve compares slightly
   different game sets, so check the cohort before reading a horizon trend.
4. **Execution.** The settlement line is a normalised even-price line at
   -110. A win rate at T-960 assumes you could bet that line at that time.

Expectations: error vs the snapshot line should improve with distance from
tip. The broad review (2026-09-17) found that fundamentals explain the
open-to-close move at T-720 but are absorbed by T-240. Win rate may not follow:
a model that anticipates the move earns CLV, not necessarily covers.
Multiple-comparisons budget: 9 cells, so expect about one to look good by luck.

## Run

```bash
poetry run python scripts/preflight_campaign.py experiments/intermediate_horizons_2_5_2026_09
bash experiments/runners/run_intermediate_horizons_2_5_2026_09_part1_line_error.sh
bash experiments/runners/run_intermediate_horizons_2_5_2026_09_part2_total_points.sh
bash experiments/runners/run_intermediate_horizons_2_5_2026_09_part3_spread.sh
```

The three parts are independent. To resume without repeating finished cells,
prefix any of them with `SKIP_EXISTING=1`. All cells use `--no-save-model`.

Each run loads the whole 4.6 GB CSV before filtering to its snapshot. Peak
memory has not been measured. On the 15 GB machine, run one part at a time
unless `free -g` shows room for two.
