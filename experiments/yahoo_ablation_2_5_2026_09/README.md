# Yahoo ablation, schema 2.5 (2026-09)

**Question:** if the Yahoo feed disappeared, would line_error lose anything?

Yahoo's only unique contribution is public-betting splits (`pct_bets`,
`pct_money`: share of tickets and of money per side). In the final data they
exist only as rolling features over past games: 156 columns in the closing CSV,
144 in the intermediate one; 145 / 129 survive cleaning. Yahoo also fills gaps in
BetMGM lines, but only for 2020 (all games) and 46 games of 2021, so that path
is not tested here.

| cell | data | hyperparameters from | features |
|---|---|---|---|
| `a_close_with` | closing | `close6x100_25_ref_20260923_173750` (trial 78, 51 trees) | 1,348 |
| `a_close_without` | closing | same | 1,203 |
| `b_t240_with` | T-240 slice | `promo25_t240_line_error_20260925_051339` (trial 58, 43 trees) | 2,445 |
| `b_t240_without` | T-240 slice | same | 2,316 |

- **No tuning.** `optuna.fixed_params` holds the model constant, so the
  features are the only difference within each pair. The parameters were tuned
  *with* the Yahoo columns, so any bias favours the `with` arm, and the CV
  folds are the ones they were tuned on: read the holdout first.
- **Seeds:** `random_state` 16 plus evaluation seeds 101, 202, 303, 404.
- **Same games.** Validation folds and the 610 holdout games are identical within
  each pair. Dropping the columns lets 66 (closing) / 14 (T-240) extra 2019-20
  games through `max_na_per_row`; they only reach the oldest end of CV folds 1-5.
- **T-240 slice:** `intermediate_line_data_2_5_20260613_t240_slice.csv` holds the
  T-240 rows of `intermediate_line_data_2_5_20260613.csv`, values untouched.
  It reproduces the promote run (2,445 features, 8,285 games) and loads in a
  few GB, not 14.

Runner: `experiments/runners/run_yahoo_ablation_2_5_2026_09.sh`.

## Result (2026-09-25): no detectable value from the Yahoo features

Holdout = 610 games, daily walk-forward, mean of 5 seeds. CV = 624 games, seed 16
only, on the folds the hyperparameters were tuned on (with Yahoo).

| pair | holdout MAE with / without | holdout win with / without | CV MAE with / without | CV win with / without |
|---|---|---|---|---|
| closing | 14.360 / 14.342 | 53.6% / 55.0% | 14.154 / 14.184 | 52.0% / 49.0% |
| T-240 | 14.324 / 14.335 | 54.7% / 55.3% | 14.302 / 14.324 | 50.8% / 52.6% |

- Paired per-game |error| difference (without − with) is within ±0.03 points,
  with SE ~0.05, everywhere: holdout seed 16 −0.011 (closing) / +0.008 (T-240);
  CV +0.030 / +0.022. No p-value below 0.5.
- Seed-to-seed MAE spread *within* an arm (up to 0.14 at T-240) is larger than
  any between-arm difference.
- Where the two arms bet opposite sides (23-30% of games), neither is right
  significantly more often (48-54%).
