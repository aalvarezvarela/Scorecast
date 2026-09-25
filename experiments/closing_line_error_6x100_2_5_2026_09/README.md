# Closing-line line_error, 6×100, schema 2.5 (2026-09)

Seven cells on `training_data_2_5_20260704.csv`, all `test_anchored` 6 folds ×
100 games (120-game step), early stopping, 150 Optuna trials, `train_games`
fixed (not tuned), evaluation seeds 101/202, same 610-game holdout
(2026-01-18…04-17). Each cell changes **one** thing from its control.

| cell | NaN budget | window | ODDS_ corr | seed | read against |
|---|---|---|---|---|---|
| `c_ref` | 300 | 6,200 | 0.99 | 16 | — |
| `c_ref_rep` | 300 | 6,200 | 0.99 | **17** | `c_ref` → **noise floor** |
| `c_tg4000` | 300 | **4,000** | 0.99 | 16 | `c_ref` |
| `c_tg2500` | 300 | **2,500** | 0.99 | 16 | `c_ref` |
| `c_na150_tg4000` | **150** | 4,000 | 0.99 | 16 | `c_tg4000` |
| `c_na80_tg4000` | **80** | 4,000 | 0.99 | 16 | `c_tg4000` |
| `c_corr0995_tg4000` | 300 | 4,000 | **0.995** | 16 | `c_tg4000` |

`c_ref` is the best closing protocol measured so far (schema 2.5, 6×100, early
stopping), rerun on the corrected cleaner.

## Measured before configuring (2026-09-23)

| NaN budget | games kept | features | window ceiling |
|---|---|---|---|
| 80 | 6,125 | 1,348 | 4,109 |
| 150 | 6,479 | 1,348 | 4,514 |
| 300 | 8,232 | 1,348 | 6,216 |
| 600 / 1000 | 8,310 | 1,348 | 6,294 |

- The NaN budget does not change the feature set here, so the NaN cells are
  clean single-factor changes. They cannot reach 6,200, which is why they run at
  4,000 against `c_tg4000`.
- 600 and 1000 are identical and add only 78 games to 300 — a near-replicate,
  not an arm, so neither is run.
- ODDS_ corr 0.995 keeps more market columns, and because `max_na_per_row`
  counts NaNs over *surviving* columns, 380 more games fall under the budget
  (8,232 → 7,852; ceiling ~5,836). The pre-flight caught 6,200 as infeasible —
  a shortfall that would not have raised at runtime, just trained two folds on
  less. That cell runs at 4,000.

## Reading it

**`c_ref` vs `c_ref_rep` first.** Two identical cells, different seed. Whatever
separates them is noise; nothing smaller than that gap is a result. For scale:
the 25 earlier closing `line_error` runs on this holdout had a median of 53.28%
against a 52.38% break-even, and two identical intermediate cells differed by
3.2pp on the primary seed. Use 3-seed means, and pair comparisons on the shared
holdout games.

## Running

```bash
bash experiments/runners/run_closing_line_error_6x100_2_5_2026_09.sh
```

No lock — the closing CSV is 222MB, so cells peak at a few GB and can run
alongside an intermediate campaign. Reference and replicate run first. Budget
~1–1.5h per cell, ~8–10h in total.
