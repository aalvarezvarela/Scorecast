# Retraining the corrected 2_5 data: line_error

17 evaluation-only cells: closing and T-30, 60, 120, 180, 240, 300, 360,
420, 480, 540, 600, 660, 720, 840, 960 and 1080. The other two campaigns are
`retrain_total_points_2_5_2026_10` and `retrain_spread_error_2_5_2026_10`.
No 2_6/2_7 model is included; the intermediate dataset's T-0 is not another
model, because the closing cell already owns the horizon-0 registry slot.

The source recipe is named in each YAML's first line. line_error comes from
`prod_rerun_2_5_2026_10`; total_points and spread_error come from
`totals_spread_2_5_2026_10`, including their closing NaN-budget-800 cells.
The deliberate change is the rebuilt data after MR #40. Source recipes keep
their cleaning, training window, seeds, 6 x 100-game test-anchored CV folds
(step 120), 90-day daily walk-forward holdout, no time decay, early stopping
(1000-tree cap, 70 rounds), lexicographic trial selection, 150 Optuna trials,
and evaluation seeds 101/202. Primary seed is 17 for closing line_error and
16 elsewhere. No timeout, persistent Optuna study or automatic promotion.

## Production verification

`production_reference.json` in each campaign records the actual S3 production
pointer, its fit and the spec that fit names, read on 2026-10-16. All 51 match
their new campaign's identity, dataset type, required line column, NaN
thresholds, correlation thresholds/overrides, train window, sample-weight lambda
and refit strategy. These are the served specs, not a possibly different
configuration for the next refit.

Original run `config.json` files remain on the GPU host. CV/search settings are
retained from the matching checked-in campaign recipes; the registry spec does
not store that protocol in full. The selected production hyperparameters are
recorded as a reference, but the new runs re-tune them with 150 trials.

## Before launching any target

Finish both 20261016 builders, the injury-store reload and input-cache refresh.
Only the new closing file was available when these campaigns were prepared;
the intermediate checksum is deliberately unpinned until its build finishes.
Never substitute the 20261003 files. Copy both Parquet files, the intermediate
`_scoring.parquet` and the two builder `.manifest.json` files to the GPU host.

Use this branch's checkout and its Python environment. From the repo root:

The closing checksum is already pinned. When the intermediate builder finishes,
use its checksum and pin the three input files (no additional Python scripts):

```bash
INTERMEDIATE_CHECKSUM=$(PYTHONPATH=src:. .venv/bin/python -c \
  'from training_pipeline.data import compute_file_checksum; print(compute_file_checksum("data/train_data/intermediate_line_data_2_5_20261016.parquet"))')
for target in line_error total_points spread_error; do
  sed -i "s|^  expected_checksum:.*|  expected_checksum: '$INTERMEDIATE_CHECKSUM'|" \
    experiments/retrain_${target}_2_5_2026_10/intermediate_*.yaml
done
sha256sum \
  data/train_data/closing_line_data_2_5_20261016.parquet \
  data/train_data/intermediate_line_data_2_5_20261016.parquet \
  data/train_data/intermediate_line_data_2_5_20261016_scoring.parquet \
  | tee experiments/retrain_line_error_2_5_2026_10/datasets.sha256 \
        experiments/retrain_total_points_2_5_2026_10/datasets.sha256 \
        experiments/retrain_spread_error_2_5_2026_10/datasets.sha256

bash experiments/runners/run_retrain_line_error_2_5_2026_10.sh --preflight-only
bash experiments/runners/run_retrain_total_points_2_5_2026_10.sh --preflight-only
bash experiments/runners/run_retrain_spread_error_2_5_2026_10.sh --preflight-only
```

Check that both builder manifests exist and match the completed files before
pinning; a checksum alone does not prove a build is healthy. Review and commit
the pins once ready. Runners only verify them, never silently repin. Full
preflight checks row identity, each horizon's presence and actual fold sizes.
Data preparation also checks the dataset/scoring join. Bash calls the existing
preflight and training CLI directly.
The source windows are provisional until all 51 preflights pass. If a ceiling
falls below a requested window, reduce that YAML's `train_games` explicitly,
document the change and rerun preflight. Do not allow silent short windows.

## Three target processes in parallel

After the preflights succeed, launch these in separate terminals (or under
tmux). Each process trains its 17 models sequentially. Each runner repeats
full preflight for all its cells before starting training.

```bash
bash experiments/runners/run_retrain_line_error_2_5_2026_10.sh --skip-existing
bash experiments/runners/run_retrain_total_points_2_5_2026_10.sh --skip-existing
bash experiments/runners/run_retrain_spread_error_2_5_2026_10.sh --skip-existing
```

The runners prefer `.venv/bin/python`, else `poetry run python`. Set
`PYTHON_BIN` to choose another interpreter. CUDA is tested by inspecting the
trained booster's actual device; a silent CPU fallback aborts. Set
`CUDA_VISIBLE_DEVICES` per terminal to allocate separate GPUs, if available.
One shared GPU may be slower with three jobs and must have enough free VRAM;
the campaign does not assume three simultaneous jobs are faster. Inspect
GPU/host memory during the first three cells before leaving jobs unattended.
Optionally set `CPUSET` to different CPU ranges per terminal to constrain each
process and its children (e.g. `CPUSET=0-7 bash ...` on a host with those CPUs).
The scripts do not assume hardware-specific CPU ranges.

Logs go to `artifacts/logs/retrain_<target>_2_5_2026_10_<timestamp>/`; runs go under
`artifacts/experiments/retrain_<target>_2_5_2026_10/`. One lock prevents two
copies of the same target from running together. The order is closing, then
interleaved horizons (240, 720, 60, 480, 960, ...), so early results span the
range. Failures are logged and do not cancel later models; final exit is
nonzero if any cell failed.

`--skip-existing` skips only a successful run with a completion receipt,
matching YAML/shared-base/data/scoring pins and complete evaluation artifacts.
An empty or partially written run directory is not a success. A failed or
interrupted cell starts a fresh 150-trial search on retry; persistent storage
is off, so no old study or old data is reused.

## Read first, promote later

Read CV, holdout MAE against the bookmaker baseline, directional win rate,
bet volume/Wilson intervals and seed spread for each slot. Compare horizons
on their common eligible games too: their row availability differs, especially
T-960/T-1080. Use a common edge threshold (0.1 points here); spread uses the
spread baseline and spread CLV column, not the totals line. Old features were
incorrect, so the old model results are descriptive references rather than a
guaranteed fair paired comparison. Compare saved predictions on matching rows.

The 90-day holdout has already been inspected in older campaigns; corrected
data does not make it untouched again. No NBA games have been added merely
by rebuilding during the offseason. Use future games for independent forward
confirmation. There are 51 correlated cells, not 51 independent discoveries;
do not select a lucky horizon by its holdout rank or treat every promoted
prediction model as a profitable bet signal.

The training recipes intentionally retain the old early-stopping and
lexicographic choices. The CV early-stopping metric is optimistic; inspect
the subsequent refit CV betting results and holdout too. Changing the tuning
protocol would be a separate experiment, not this data-fix rerun.

Only after reviewing results should selected runs be refitted and staged with
`training_pipeline.promote`, smoke-tested in staging, and selected production
pointers updated with `scripts/promote_build.py`. Do not promote all enabled
slots implicitly. Existing builds remain rollback targets. The daily automatic
refit for intermediate horizons is still unavailable in
`nba_ou.modeling.refit.resolve_training_frame`; serving a promoted fixed build
and automatically refreshing its training data are separate operations.
