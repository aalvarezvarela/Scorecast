# Closing production configurations, rerun on the rebuilt 2_5 data (2026-10)

The three closing-line configurations promoted to production on 2026-09-24,
rerun **unchanged** on the rebuilt closing dataset, then re-promoted.

| config | target | source config | source run (promoted 2026-09-24) |
|---|---|---|---|
| `line_error.yaml` | line_error | `closing_line_error_6x100_2_5_2026_09/c_ref_rep.yaml` | `close6x100_25_ref_rep_20260923_190421` |
| `total_points.yaml` | total_points | `closing_total_points_6x100_2_5_2026_09/t_tg6200.yaml` | `close6x100_25_total_points_tg6200_20260924_034258` |
| `spread_error.yaml` | spread_error | `closing_spread_6x100_2_5_2026_09/s_tg6200.yaml` | `close6x100_25_spread_tg6200_20260924_022125` |

The source runs come from each production bundle's `training_code_tag`
(`models/<target>/6200_games/*.meta.json`). Only the data block and labels
differ from the source configs: `csv_path`, `data_version`,
`expected_checksum`, `experiment_name`, `training_version`,
`comparison_group`, `hypothesis` and a `prod_rerun` tag. Seeds, the 6x100
folds, `train_games: 6200`, the NaN budget of 300, the correlation thresholds,
150 Optuna trials and the evaluation seeds are as promoted.

## What changed in the data

`closing_line_data_2_5_20261003.parquet` against `training_data_2_5_20260704.csv`:

- written as Parquet, with identifier columns declared by exact name (no
  feature is read as text any more);
- no missing-value imputation (`apply_missing_policy` removed): on the old
  data that cost 30 rows at a NaN budget of 300;
- Yahoo reduced to the 36-column compact block, which no longer counts toward
  `max_na_per_row` -- this can add rows back, mainly 2019-20 games without Yahoo.

No game after the 2025-26 season ends exists yet, so the games are the same and
the 90-day holdout covers the same period. The row count after cleaning is not
the same, so read the pre-flight's window ceiling before trusting 6,200.

## Steps

```bash
# 1. Build the dataset (prints data.csv_path and data.expected_checksum)
poetry run python scripts/create_train_data/create_train_data.py --limit 2026-10-03

# 2. Pin its checksum in the three configs
CHECKSUM="sha256:..."   # the value the builder printed
sed -i "s|expected_checksum: null|expected_checksum: \"$CHECKSUM\"|" \
    experiments/closing_prod_rerun_2_5_2026_10/*.yaml

# 3. Pre-flight alone (the runner repeats it): checksum, row keys, and the
#    per-config window ceiling against train_games 6200
poetry run python scripts/preflight_campaign.py experiments/closing_prod_rerun_2_5_2026_10

# 4. Run the three experiments (~1-1.5h each on CUDA)
bash experiments/runners/run_closing_prod_rerun_2_5_2026_10.sh
```

If the pre-flight reports a window ceiling below 6,200, the configurations can
no longer be run unchanged: lower `train_games` to the ceiling in all three and
note it here rather than letting the early folds train on fewer games silently.

## Promoting

Read each run against its source run first (holdout MAE and win rate on the
same holdout period). Then, per target -- `promote` refits the selected
hyperparameters on the run's own dataset and stages the build; `promote_build`
switches production to it:

```bash
for target in line_error total_points spread_error; do
  run=$(ls -d artifacts/experiments/closing_prod_rerun_2_5_2026_10/prodrerun25_${target}_2* | tail -1)
  poetry run python -m training_pipeline.promote "$run" --to-s3 --replace-config --overwrite --dry-run
done
# then the same without --dry-run, and:
poetry run python scripts/promote_build.py                                   # dry run
poetry run python scripts/promote_build.py --slot 2_5/line_error/t0000/main --execute
poetry run python scripts/promote_build.py --slot 2_5/total_points/t0000/main --execute
poetry run python scripts/promote_build.py --slot 2_5/spread_error/t0000/main --execute
```

`--replace-config` is required because the three 2_5 slots already have a
configuration. `--overwrite` replaces the local bundle copy under `models/`,
whose name carries the last training game date (`..._17_04_26`), the same as
the current one because the games are the same; the S3 build it replaces stays
in the registry as a rollback target (`promote_build.py --rollback`). The `ENABLED_MODELS` rows in `src/nba_ou/config.ini` already
name these slots and need no change. Until `promote_build` runs, production
keeps serving the current builds; those were trained on the full 156-column
Yahoo family and cannot score the new prediction frame, so do not run the
daily predictor between rebuilding the data and promoting.
