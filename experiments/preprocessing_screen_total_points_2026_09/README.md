# Total-points preprocessing screen

This repeats the six-cell line-error preprocessing screen for
`total_points_regressor`, using the selected modern-2.2 hyperparameters and
125 boosting rounds from trial 39 of `nest_probe_modern22_total_early_stop`.

Only preprocessing varies:

- row missingness: 80, 150, or 300 allowed NaNs;
- global correlation threshold: 0.95, 0.99, or 0.995;
- the 0.95 cells keep the current `ODDS_ = 0.99` override; 0.995 cells do not.

The protocol is held fixed at 12 latest anchored folds, 50 validation games,
60-game starts, 4,000 training games, seed 16, and a 90-day unscored holdout.
Each cell is evaluated on the same intersection of validation `GAME_ID`s.
Wins are computed from `predicted total - Bet365 total line`; the total-points
MAE and the bookmaker-line MAE stay in absolute total-points space.

Run:

```bash
bash experiments/runners/run_preprocessing_screen_total_points_2026_09.sh
```

Results are written to
`artifacts/probes/preprocessing_screen_total_points_2026_09/`. The holdout is
split off before CV and is deliberately never scored.

## Result

All six cells were evaluated on the same 643 validation games. The two best
raw results were:

- `NA=300 / corr=0.995`: MAE 13.8717, 55.07% wins at edge 0.1;
- `NA=80 / corr=0.95 + ODDS=0.99`: MAE 13.8756, 54.76% wins at edge 0.1.

The current policy (`NA=300 / corr=0.95 + ODDS=0.99`) scored MAE 13.8838 and
54.02% wins. Neither apparent improvement is resolved from noise by a paired
7-calendar-day cluster bootstrap. Relative to the current policy, the MAE
deltas and 95% intervals were `-0.0121 [-0.0609, +0.0357]` for the 0.995
correlation cell and `-0.0082 [-0.0429, +0.0271]` for the 80-NaN cell. Their
win-rate gains at edge 0.1 were respectively `+1.04 pp [-1.76, +3.87]` and
`+0.73 pp [-1.79, +3.25]`.

Therefore this fixed-model screen does not justify changing the global
preprocessing default. It does justify carrying the current policy plus these
two neighboring candidates into a small repeated Optuna confirmation. The
combination `NA=80 / corr=0.995` should not be treated as the union of both
possible gains: its interaction was worse (MAE 13.8929, 52.41% wins).
