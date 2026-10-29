# Historical predictions with past-only refits

Status: Reference

`scripts/backfill_past_predictions.py` fills missing closing/intermediate predictions
for past dates of the current NBA season. It trains a new model even when no
model existed on that date. It never loads the current production booster.

## Preview locally

Use the prediction CUDA environment (`requirements/predict-cuda.txt`, which
also includes the refit dependencies), with Supabase/Aiven read credentials, S3
credentials and the GPU required by the promoted configurations:

```bash
PYTHONHASHSEED=0 PYTHONPATH=.:src .venv/bin/python scripts/backfill_past_predictions.py \
  --start-date 2026-10-21 --end-date 2026-10-27 \
  --slot 2_5/line_error/t0360/main --dry-run
```

`--dry-run` is the default. It performs real refits and predictions and writes
CSV/JSON reports under `data/backfill_predictions/<execution-id>/`. It writes
no predictions, model builds, partition updates or feature memos to Aiven/S3.
Local feature and input caches may be populated. No database ingestion runs.

Omit `--end-date` for one day. Repeat `--slot` to select several models; omit
it to select every enabled intermediate slot. Add `--include-closing` to
include enabled closing models, evaluated at the shared T-240 timepoint.
Pooled models and layered schemas are not supported in this first version.

## Closing and TabPFN at one shared timepoint

Add `--tabpfn` to also fit **standard** TabPFN for all three targets
(`line_error`, `total_points`, `spread_error`), at **T-240** (four hours) only.
Add `--include-closing` to also fit the three closing TabPFN targets at that
same T-240 timepoint (six standard fits per date). There are no thinking modes.
`--snapshot-minutes` (alias `--tabpfn-horizon-minutes`) changes the shared
closing/TabPFN timepoint; it never expands TabPFN over other horizons.

Closing models train on the **closing_line** dataset through the historical
cutoff. Their evaluated game's inputs are rebuilt with the closing feature
pipeline, using only stored quotes, injury reports and rosters available at
T-240. Stored final quotes and untimestamped Yahoo inputs are discarded.
The model horizon stays 0; the input snapshot is explicitly stored as 240.
A closing prediction at another timepoint does not count as coverage.

For just TabPFN, without any XGBoost refits:

```bash
PYTHONHASHSEED=0 PYTHONPATH=.:src .venv/bin/python scripts/backfill_past_predictions.py \
  --start-date 2026-10-21 --end-date 2026-10-27 \
  --tabpfn-only --include-closing --snapshot-minutes 240 --dry-run
```

Use `requirements/predict-tabpfn-cuda.txt` (or the CPU variant) and
`TABPFN_ACCESS_TOKEN`, as in the existing TabPFN workflow. A TabPFN-only run
does not require a local training GPU. **Dry run still performs hosted
TabPFN fits/predictions and consumes service credits**; it disables the
Aiven/S3 persistence, rather than the computation.

Each target follows its own enabled closing or T-240 configuration. Only the current
configuration is read, never the production booster or the live TabPFN job's
unfiltered training input. Its training rows are cut to the historical limit
before selecting the window, and the final training-set date is checked
before calling TabPFN. The training checksum is recorded in the run manifest.
Historical injury/roster snapshots and the configured injury-report coverage
requirement apply to these fits too.

TabPFN rows share `workflow = 4` with the backfill but have `kind = 3` and
the standard `tabpfn_v3_5` variant. Registry predictions and thinking variants
cannot count as existing TabPFN coverage. `--persist` stores these predictions
and the run archive; hosted TabPFN has no local booster to archive.

## Insert predictions

```bash
PYTHONHASHSEED=0 PYTHONPATH=.:src .venv/bin/python scripts/backfill_past_predictions.py \
  --start-date 2026-10-21 --end-date 2026-10-27 \
  --slot 2_5/line_error/t0360/main --persist
```

This reruns training and saves the missing predictions in the existing Aiven
prediction tables, plus the input/run archive and historical models in S3.
`runs.workflow = 4` identifies these reconstructions. They appear under
**Snapshot replay** in the dashboard, and are excluded from live predictions.
Execution timestamps remain real; the input's snapshot horizon and tipoff
record the simulated prediction time.

Models are written under
`models/historical_backfills/<execution-id>/<game-date>/<slot>/`. No config,
staging or production pointer is changed. Failures in one date/horizon/model
are reported and do not prevent other valid predictions from being saved;
the process exits nonzero for incomplete coverage. Rerun the same command to
retry missing rows. Runs are skipped only for the same specification, exact
model horizon, input timepoint, and a model trained within the historical cutoff (at most D-1,
or the earlier conservative limit for overnight snapshots).
Existing runs are retained, including invalid older reconstructions; a
prediction produced by a model trained on the evaluated date or later does
not count as coverage.

The manual **Predictions - Historical Backfill** Action exposes the same date
range, slots and `persist` toggle (off by default), and uploads local reports
even when prediction fails. It shares the predictor's queue and runs one job
per date/horizon, sequentially, each with a fresh one-hour AWS session.
The Action includes standard TabPFN (`include_tabpfn`) and closing models
(`include_closing`) at T-240 by default. `tabpfn_only` selects just TabPFN
fits; `tabpfn_horizon_minutes` changes the shared timepoint. Closing and
intermediate T-240 models are grouped in the same date/timepoint job. The API token is read from `secrets.TABPFN_ACCESS_TOKEN`.
Feature/input caches survive between jobs on the runner. Up to 256 jobs can
be selected per dispatch; split larger ranges. Do not run a local insertion at
the same time as another backfill of the same dates/specs: coverage checks
are read-before-write and do not lock independent invocations.

## Time and data rules

1. The current configuration specifications are pinned once for the range.
   Features, hyperparameters, seed, device and rolling window are reused;
   fitted trees are always new. This evaluates today's fixed system, rather
   than reproducing the code/configuration that existed historically. If a
   configuration was chosen using these same evaluation dates, that model
   selection bias still exists.
2. Training partitions are read as stored. Rows are cut to at most D-1
   **before** cleaning, sample weights and window selection. A fitted model
   whose recorded training end exceeds that cutoff is rejected. The data
   checksum identifies the filtered rows. No production-model fallback exists.
3. Target rows are rebuilt from archived SBR/Covers ticks and stored injury
   timelines through the corresponding closing or intermediate feature pipeline. Historical
   crews are read from storage; the existing 09:00 ET release gate applies.
   Injury reports require publication strictly before each snapshot.
4. Official roster observations are filtered per game and horizon **before**
   building any features, including roster-dependent base features. A missing
   eligible roster uses the existing previous-game inference, supplemented
   by the injury report available at the snapshot. Current rosters and the
   target game's eventual absences are not a substitute.
5. The stores do not have reliable historical final-buzzer timestamps. For
   snapshots before 06:00 ET, both training and target feature history use
   an extra day's lag, to leave the preceding overnight slate out. Thus an
   early T-1080 row can use D-2 while a later T-360 row uses D-1. This is a
   conservative daily-data convention, not an exact reconstruction of when
   every previous result first became available. All games of one rebuilt
   horizon use the earliest missing game's cutoff.
6. Missing stored lines/snapshots or unusable model inputs are reported, not
   replaced with current quotes. Existing partitions are not refreshed by
   this command; update them separately if the training history is incomplete.

The source data is today's stored historical version. Later statistical
corrections cannot be undone unless the original observations were archived.
