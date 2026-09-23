# Experiment runners

Shell entry points that validate and run groups of experiment YAML files live
here. Run them from anywhere; each script resolves the repository root before
accessing configs, datasets, logs, or artifacts.

## Available campaigns

- `run_early_line_error_window_2_5_2026_09_part1.sh` and
  `run_early_line_error_window_2_5_2026_09_part2.sh`: parallel halves of the
  schema-2.5 line-error campaign covering every T-0 through T-1080 snapshot,
  including T-30.
  Both let Optuna choose 2,500, 3,000, 3,500, or the maximum for each horizon
  (3,775–4,788 games) on eight test-anchored folds of at least 80 games.
  Part 2 also runs the optional time-decay arm at T-360.
- `run_train_window_line_error_2_5_2026_09_part1_t60.sh` and
  `run_train_window_line_error_2_5_2026_09_part2_t720.sh`: does older data help?
  Six cells per horizon on ONE identical dataset (2019 floor, max_na 800 in
  every arm), with `train_games` FIXED at 1,500 / 2,500 / 3,500 / 4,500 / max
  rather than tuned, plus a same-window replicate under a different seed that
  measures the noise floor. Replaces the history-depth campaign below, which
  varied the season floor and so varied the surviving feature set along with it.
  Neither part locks, so both can be launched together.
- `run_rolling_origin_campaign.sh`: the current protocol -- rolling-origin CV
  (train to a date, predict the next 4 game-days), with the training window and
  the boosting rounds both tuned by Optuna. 2 runs, one per regression target.
- `run_time_decay_2026_09_part1.sh` and `run_time_decay_2026_09_part2.sh`:
  parallel halves of the regenerated no-decay campaign. Together they run the
  three closing targets on normalized 2.2 data, spread error at T-30, T-360,
  and T-720, plus a closing spread-error comparison on schema 2.1.
- `run_totals_schema_lexicographic_2026_09.sh`: six sequential CUDA runs that
  compare `total_points` and `line_error` across closing schemas 2.0, 2.1 and
  2.2. Selection is lexicographic: best pooled O/U win rate inside the best 15%
  by pooled MAE, with an absolute MAE gap capped at 0.04.
- `run_extended_history_cv_2026_09_part1.sh` and
  `run_extended_history_cv_2026_09_part2.sh`: the extended-history protocol.
  Part 1 runs all three closing targets on schema 2.2 plus a schema 2.0
  line-error control; part 2 runs the three targets at T-360. Both use a
  90-day holdout, five-game-day rolling-origin folds, seed 16 and the
  lexicographic top-15% / +0.04-MAE selector.

## Archived campaigns

Everything that ran under the previous training protocol lives in
`../archived/runners/`, alongside the configs it drives and a frozen
`_base.yaml` that keeps those runs reproducing as they originally did. See
`../archived/README.md` for what changed and why the two sets of numbers are not
comparable.

Run a campaign in the foreground:

```bash
bash experiments/runners/run_rolling_origin_campaign.sh
```

Run it detached:

```bash
nohup bash experiments/runners/run_rolling_origin_campaign.sh > /dev/null 2>&1 &
```

Each runner documents its log paths and any supported resume options at the top
of the script.
