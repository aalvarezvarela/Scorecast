# Intermediate production configurations, rerun on the rebuilt 2_5 data (2026-10)

The sixteen intermediate `line_error` configurations promoted to the registry on
2026-09-27 (`2_5/line_error/t0030` .. `t1080`), rerun **unchanged** on the
rebuilt intermediate dataset, then re-promoted. The closing slots are in
`closing_prod_rerun_2_5_2026_10`.

`line_error` is the only intermediate target that was ever promoted.
Intermediate `total_points` and `spread_error` were measured
(`intermediate_horizons_2_5_2026_09`, `early_totals_spread_window_2_5_2026_09`)
but never promoted, so there is nothing to rerun for them.

`line_error_t<N>.yaml` is a copy of
`promote_intermediate_line_error_2_5_2026_09/p_t<N>_line_error.yaml`, the config
behind the bundle in `models/line_error/t<NNNN>/` (its `training_code_tag`
names the run). Only the data block and labels differ: `csv_path`,
`scoring_csv_path`, `data_version`, `expected_checksum`, `experiment_name`,
`training_version`, `comparison_group`, `hypothesis` and a `prod_rerun` tag.
The 2019 floor, NaN budget 800, per-horizon `train_games` (6,275 at T-30 down
to 4,400 at T-1080), 6x100 folds, ODDS_ correlation 0.98, 150 trials and the
seeds are as promoted.

## What changed in the data

`intermediate_line_data_2_5_20261003.parquet` against
`intermediate_line_data_2_5_20260613.csv`:

- seven market-dynamics features reach the model for the first time
  (`ODDS_LINE_HIST_RIDGE_EXPECTED_{TOTAL,SPREAD,ML}_MOVE_TO_CLOSE`,
  `ODDS_SNAP_NEWS_{TOT,SPR,ML}_BET365_REACTION_RESIDUAL`,
  `ODDS_SNAP_XMKT_TOTAL_MOVE_WITHOUT_SIDE_MOVE_60`) -- the old loader read
  their names as identifiers and cleaning dropped them;
- no missing-value imputation: on the old T-240 data that cost 16 rows;
- Yahoo reduced to the 24 historical columns, outside the row-NaN count;
- data through 2026-10-03 rather than 2026-06-13 (no new regular-season games).

Same games and holdout period, but a different row count after cleaning, so
the pre-flight re-measures each horizon's window ceiling. It fails a horizon
whose `train_games` no longer fits; lower that one to its new ceiling and
note it here.

## Steps

```bash
# 1. Build (prints data.csv_path and data.expected_checksum; also writes _scoring)
poetry run python scripts/create_train_data/create_intermediate_line_train_data.py --recent-limit 2026-10-03

# 2. Pin its checksum in the sixteen configs
CHECKSUM="sha256:..."   # the value the builder printed for the main file
sed -i "s|expected_checksum: null|expected_checksum: \"$CHECKSUM\"|" \
    experiments/intermediate_prod_rerun_2_5_2026_10/*.yaml

# 3. Run. The two parts can run at the same time (each ~1-1.5h per horizon)
bash experiments/runners/run_intermediate_prod_rerun_2_5_2026_10_part1_t30_t420.sh
bash experiments/runners/run_intermediate_prod_rerun_2_5_2026_10_part2_t480_t1080.sh
# SKIP_EXISTING=1 resumes a part without rerunning finished horizons.
```

## Promoting

Read each horizon against its source run first. Then:

```bash
for h in 30 60 120 180 240 300 360 420 480 540 600 660 720 840 960 1080; do
  run=$(ls -d artifacts/experiments/intermediate_prod_rerun_2_5_2026_10/prodrerun25_t${h}_line_error_2* | tail -1)
  poetry run python -m training_pipeline.promote "$run" --to-s3 --replace-config --overwrite --dry-run
done
# then the same without --dry-run, and per slot:
poetry run python scripts/promote_build.py                                    # dry run
poetry run python scripts/promote_build.py --slot 2_5/line_error/t0240/main --execute   # etc.
```

`--replace-config`: the sixteen slots already have a configuration.
`--overwrite`: the local bundle copy under `models/line_error/t<NNNN>/` carries
the same last-game date as the current one.

These slots are **not** in `ENABLED_MODELS` and are not served: live
intermediate prediction is not wired up yet (snapshot features at prediction
time). Promoting keeps the registry current for when it is; it changes nothing
the daily jobs do.
