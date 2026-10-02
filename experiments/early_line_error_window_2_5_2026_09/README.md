# Intermediate line-error windows (schema 2.5)

Eighteen experiments cover all 17 snapshots in the training CSV, from T-0
(tip-off) through T-1080: 0, 30, 60, 120, 180, 240, 300, 360, 420, 480,
540, 600, 660, 720, 840, 960, and 1080 minutes. T-360 also has a separate
arm that lets Optuna choose exponential time decay, including an unweighted
option. All other arms have time decay disabled. Every arm keeps
`max_na_per_row: 500` and uses the same 90-day holdout and line-error target
as the previous schema-2.5 campaign.

CV uses the eight latest nonoverlapping `test_anchored` folds, each containing
at least 80 games. A fold may contain slightly more than 80 because a game day
is never split. One complete game day is skipped between validation blocks.
The validation games are built once from full prior histories;
each Optuna trial takes the last `train_games` from those same histories. Thus
the training window is tuned without changing which games score a trial.

Optuna chooses `train_games` from **2,500, 3,000, 3,500, and the maximum for
that horizon**. Each maximum is the number of cleaned development games before
the earliest of the eight CV validation folds, with the 500-NaN row limit and
the configured 90-day holdout applied:

| Minutes before tip-off | Maximum train games |
| ---: | ---: |
| 0 | 4,788 |
| 30 | 4,788 |
| 60 | 4,788 |
| 120 | 4,788 |
| 180 | 4,788 |
| 240 | 4,788 |
| 300 | 4,788 |
| 360 (both arms) | 4,778 |
| 420 | 4,778 |
| 480 | 4,764 |
| 540 | 4,755 |
| 600 | 4,727 |
| 660 | 4,699 |
| 720 | 4,627 |
| 840 | 4,436 |
| 960 | 4,209 |
| 1080 | 3,775 |

Game coverage can differ by horizon because older lines are not available for
every game. The two T-360 arms use the same game cohort.

Run the two parts in separate terminals or processes:

```bash
bash experiments/runners/run_early_line_error_window_2_5_2026_09_part1.sh
bash experiments/runners/run_early_line_error_window_2_5_2026_09_part2.sh
```

Each part runs its own configs sequentially. Set `SKIP_EXISTING=1` to skip
configs with an existing experiment artifact when resuming.
