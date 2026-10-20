# Early totals + spread windows (schema 2.5)

The totals/spread counterpart to `early_line_error_window_2_5_2026_09`. It
takes the same schema-2.5 intermediate dataset, CV design and tuned
game-history windows, but swaps the line-error target for the two other
markets — `total_points_regressor` and `spread_error_regressor` — and turns
time decay on for **every** arm rather than only the T-360 line-error arm.

## What differs from the line-error campaign

- **Two markets, nine horizons each.** Only half the 17 snapshots are covered:
  0, 60, 180, 300, 420, 540, 660, 840, and 1080 minutes before tip-off.
- **Time decay on every arm.** `sample_weight` is enabled with a tuned
  exponential `lambda_` in `[0.0005, 0.005]` and `allow_unweighted: true`, so
  Optuna may still choose no weighting.
- **Optuna picks the window.** `train_games_choices` offers 2,500, 3,000,
  3,500 and the per-horizon maximum; the trial selection chooses one.

Everything else matches the line-error campaign: 90-day holdout, `max_na_500`
cleaning, and eight latest nonoverlapping `test_anchored` folds of at least 80
games each (`step_games_between_tests: 1`, folds do not overlap). The
validation games are built once from full prior histories; each Optuna trial
takes the last `train_games` from those same histories, so the window is tuned
without changing which games score a trial.

`train_games_choices` maximum per horizon (cleaned development games before the
earliest of the eight CV folds, mirrored from the line-error campaign):

| Minutes before tip-off | Maximum train games |
| ---: | ---: |
| 0 | 4,788 |
| 60 | 4,788 |
| 180 | 4,788 |
| 300 | 4,788 |
| 420 | 4,778 |
| 540 | 4,755 |
| 660 | 4,699 |
| 840 | 4,436 |
| 1080 | 3,775 |

These ceilings were computed for the line-error target. The totals and spread
targets drop rows differently, so confirm the true per-fold size from the
pre-flight before trusting a maximum; lower it if a horizon shrinks.

## Running

Run the pre-flight, then the two parts concurrently in separate terminals or
processes — part 1 is total_points, part 2 is spread:

```bash
poetry run python scripts/preflight_campaign.py experiments/early_totals_spread_window_2_5_2026_09
bash experiments/runners/run_early_totals_spread_window_2_5_2026_09_part1_total_points.sh
bash experiments/runners/run_early_totals_spread_window_2_5_2026_09_part2_spread.sh
```

Each part runs its nine configs sequentially. Set `SKIP_EXISTING=1` to skip
configs with an existing experiment artifact when resuming.
