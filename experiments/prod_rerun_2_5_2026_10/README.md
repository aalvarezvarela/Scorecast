# Every 2_5 production configuration, rerun on the rebuilt data (2026-10)

The production registry (`models/2_5/.../channels/production.json`) holds 19
slots, all trained on 2_5 data with the loader and cleaning bugs fixed on
2026-10-03. This campaign reruns each slot's configuration **unchanged** on the
rebuilt datasets, so the results can be read against the current builds and,
if they hold up, promoted over them.

| slots | configs | source campaign (current production build) |
|---|---|---|
| `line_error/t0000`, `total_points/t0000`, `spread_error/t0000` | `closing_{line_error,total_points,spread_error}.yaml` | `closing_line_error_6x100_2_5_2026_09/c_ref_rep`, `closing_total_points_6x100_2_5_2026_09/t_tg6200`, `closing_spread_6x100_2_5_2026_09/s_tg6200` (promoted 2026-09-24) |
| `line_error/t0030` .. `t1080` (16) | `intermediate_line_error_t<N>.yaml` | `promote_intermediate_line_error_2_5_2026_09/p_t<N>_line_error` (promoted 2026-09-27) |

Only `line_error` was ever promoted for intermediate horizons. Intermediate
`total_points` and `spread_error` were measured but never promoted, so they
are not here.

Each source was read from the production spec's `provenance.source_run`.
Diffing each config against that run's `config.json` shows that only these
fields differ: `csv_path`, `scoring_csv_path`, `data_version`,
`expected_checksum`, `experiment_name`, `training_version`,
`comparison_group`, `hypothesis` and `tags`. Seeds, folds, `train_games`, NaN
budgets, correlation thresholds, Optuna trials and evaluation seeds are as
promoted.

## What changed in the data

| | closing | intermediate |
|---|---|---|
| file | `closing_line_data_2_5_20261003.parquet` (`sha256:7fa3465bcde6d8ec`) | `intermediate_line_data_2_5_20261003.parquet` (`sha256:7e41757f62bc49f4`) |
| replaces | `training_data_2_5_20260704.csv` | `intermediate_line_data_2_5_20260613.csv` |

- **Identifier columns are declared by exact name.** The old loader read every
  column with "ID" in its name as text and cleaning dropped it. On closing data
  that was 212 `..._line_mid_...` columns. On intermediate data it was seven
  market-dynamics features, which now reach the model for the first time.
- **No missing-value imputation.** `apply_missing_policy` is gone. On the old
  data, `max_na_per_row` then drops 30 closing rows and 16 T-240 rows.
- **Yahoo is reduced to its compact block** (36 closing columns, 24
  intermediate) and is left out of the row-NaN count.

The games are the same (nothing has been played since the 2025-26 season
ended), and so is the 90-day holdout period. The row counts after cleaning
differ, so the pre-flight re-measures every slot's window ceiling.

## Running

The checksums are already pinned. The datasets were built after the Yahoo
commit. If either dataset is rebuilt, re-pin its checksum:
`sed -i 's|expected_checksum: ".*"|expected_checksum: "<new>"|' experiments/prod_rerun_2_5_2026_10/closing_*.yaml`
(or `intermediate_*.yaml`).

```bash
# Pre-flight for the whole campaign (each runner repeats it for its part)
poetry run python scripts/preflight_campaign.py experiments/prod_rerun_2_5_2026_10

# The two parts, at the same time, ~27h each
bash experiments/runners/run_prod_rerun_2_5_2026_10_part1_closing_t30_t360.sh
bash experiments/runners/run_prod_rerun_2_5_2026_10_part2_t420_t1080.sh
# SKIP_EXISTING=1 resumes a part without rerunning finished slots.
```

Part 1 runs the three closing slots first (~1.5-2h each), then T-30..T-360.
Part 2 runs T-420..T-1080. The intermediate runs took ~2.5-3.5h each in
September, with two campaigns sharing the GPU.

If the pre-flight reports a window ceiling below a slot's `train_games`, that
slot can no longer run unchanged. Lower `train_games` to the ceiling and record
it here; do not let the early folds train on fewer games silently.

## Analysing

Every run lands in `artifacts/experiments/prod_rerun_2_5_2026_10/`. Compare
each one with its source run (the production spec's `provenance.source_run`) on
the same holdout period: holdout MAE, O/U accuracy and win rate. September's
noise floor between replicates was 1.1-2.6pp of accuracy per intermediate
horizon.

## Promoting (only after the analysis)

`promote` refits the selected hyperparameters on the run's own dataset and
stages the build. `promote_build` then points the slot's production channel at
it. The current build is not deleted: it stays in the registry as the rollback
target (`promote_build.py --rollback --slot ... --execute`).

```bash
C=artifacts/experiments/prod_rerun_2_5_2026_10
runs=()
for t in line_error total_points spread_error; do runs+=("$(ls -d $C/prodrerun25_closing_${t}_2* | tail -1)"); done
for h in 30 60 120 180 240 300 360 420 480 540 600 660 720 840 960 1080; do
  runs+=("$(ls -d $C/prodrerun25_t${h}_line_error_2* | tail -1)")
done
for run in "${runs[@]}"; do
  poetry run python -m training_pipeline.promote "$run" --to-s3 --replace-config --overwrite --dry-run
done
# then the same loop without --dry-run, and:
slots=(2_5/line_error/t0000 2_5/total_points/t0000 2_5/spread_error/t0000)
for h in 30 60 120 180 240 300 360 420 480 540 600 660 720 840 960 1080; do
  slots+=("2_5/line_error/t$(printf %04d "$h")")
done
args=(); for s in "${slots[@]}"; do args+=(--slot "$s/main"); done
poetry run python scripts/promote_build.py "${args[@]}"            # dry run
poetry run python scripts/promote_build.py "${args[@]}" --execute
```

- `--replace-config`: every slot already has a configuration.
- `--overwrite`: the local bundle copy under `models/` carries the same
  last-game date as the current one.
- **Promote the closing slots before running the daily predictor again.** The
  current closing builds were trained on the full Yahoo family (~156 columns)
  and cannot score a prediction frame built with the compact block.
- The intermediate slots are not in `ENABLED_MODELS` and are not served, because
  live intermediate prediction is not wired up yet. Promoting them only keeps
  the registry current.
