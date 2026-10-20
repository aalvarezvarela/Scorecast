# Closing-line total_points, 6×100, schema 2.5 (2026-09)

Five cells on `training_data_2_5_20260704.csv`, all `test_anchored` 6 folds ×
100 games, early stopping, 150 Optuna trials, `train_games` fixed, evaluation
seeds 101/202, same 610-game holdout (2026-01-18…04-17). Each cell changes one
thing from `t_ref`.

| cell | NaN budget | window | seed |
|---|---|---|---|
| `t_ref` | 300 | 4,000 | 16 |
| `t_ref_rep` | 300 | 4,000 | **17** — noise floor |
| `t_tg6200` | 300 | **6,200** | 16 |
| `t_tg2500` | 300 | **2,500** | 16 |
| `t_na150` | **150** | 4,000 | 16 |

Ceilings measured 2026-09-23: 6,216 games at NaN budget 300, 4,514 at 150.
The NaN budget does not change the feature set, so every cell is a clean
single-factor change.

**Read `t_ref` vs `t_ref_rep` first**; nothing smaller than that gap is a
result. Use 3-seed means and pair comparisons on the shared holdout games.

```bash
bash experiments/runners/run_closing_total_points_6x100_2_5_2026_09.sh
```

No lock. Budget ~1–1.5h per cell, ~6–7h in total.
