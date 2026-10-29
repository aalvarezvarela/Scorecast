# Historical predictions from the saved retrain dataset

Status: Reference

`scripts/backfill_past_predictions.py` fills missing predictions for past dates
of the current NBA season. **It never creates, rebuilds or updates a dataset.**
There is no reconstruction mode or fallback to a feature builder.

For each game date D, it reads the existing retrain season partitions, trains a
new model on earlier rows, and predicts D's saved rows at the requested horizon.
The date filter runs before cleaning, weighting and training-window selection.
Actual outcomes, overtime and scoring-only closing values are removed from the
prediction inputs. A current production booster is never loaded.

If a partition, row, required feature or safe timestamp is missing, the failure
is reported. Other valid slots/dates can continue; the process exits nonzero
for incomplete coverage. Missing inputs are never generated automatically.

## Closing and TabPFN

`--include-closing` evaluates closing models at T-240 by default. Their training
rows come from the saved closing_line dataset; their prediction inputs come
from the saved intermediate T-240 row. The evaluated game's final closing row
is never used to fill inputs.

The closing adapter applies the existing row-wise odds formulas to the saved
snapshot quotes. It does not calculate team/player history, fetch injury reports,
read rosters, or run either dataset builder. Untimestamped Yahoo/consensus
percentages and unavailable opener prices remain NaN. All required feature
names must be supplied, and both injury reports must be filed at that snapshot.

`--tabpfn` adds standard TabPFN for all three targets at T-240. With closing,
there are six standard fits per date: three intermediate and three closing.
`--tabpfn-only` omits XGBoost; `--snapshot-minutes` (alias
`--tabpfn-horizon-minutes`) changes that one shared timepoint. Thinking modes,
pooled slots and layered schemas are not included; base schema 2_5 is supported.

## Run

```bash
PYTHONHASHSEED=0 PYTHONPATH=.:src .venv/bin/python scripts/backfill_past_predictions.py \
  --start-date 2026-10-20 --end-date 2026-10-25 \
  --tabpfn --include-closing --dry-run
```

Dry run is the default: real fits/predictions and local CSV/JSON reports under
`data/backfill_predictions/<execution-id>/`, with no Aiven/S3 writes.
**Hosted TabPFN still consumes service credits in dry run.** Use the normal
prediction/refit environment and credentials, plus `TABPFN_ACCESS_TOKEN` for
TabPFN. TabPFN-only does not require a local GPU; XGBoost follows its spec's device.

Omit `--end-date` for one day. Repeat `--slot schema/target/tXXXX/main` to choose
registry slots; otherwise all enabled intermediate slots are included, plus
closing when requested. Replace `--dry-run` with `--persist` to insert missing
predictions into Aiven and archive runs/isolated XGBoost builds in S3. Builds
live under `models/historical_backfills/<execution-id>/<game-date>/<slot>/`;
production/staging pointers and training partitions are unchanged.

The manual **Predictions - Historical Backfill** Action uses this same stored-
data path. It defaults to all XGBoost slots, standard TabPFN and closing at
T-240, with persistence off. Jobs run sequentially by date/horizon with fresh
AWS credentials and local partition caches. Local reports are uploaded even
on failure. The optional scheduled-prediction dependency waits on a hosted
runner before taking the NBA runner.

Saved predictions use workflow 4 (`historical_backfill`) and appear in
**Snapshot replay**, not live predictions. TabPFN uses kind 3 and the standard
`tabpfn_v3_5` variant. Coverage matches the spec, kind, target, variant, model
horizon, input timepoint and historical training end. Reruns retry missing rows.
Independent local insertions should not overlap the same dates/specs.

## One-off cron: October 30, 2026 at 06:20 Madrid

`scripts/cronjobs/schedule_historical_backfill.py` prepares a single cron-job.org
dispatch for **2026-10-30 06:20 Europe/Madrid (05:20 UTC)**. Its defaults fill
**October 20–25 inclusive**, using all enabled XGBoost slots plus standard
TabPFN for all three targets, intermediate and closing at T-240. The eventual
workflow inserts missing predictions and archives its isolated models. It uses
only saved retrain partitions; it never rebuilds the dataset.

The last scheduled past-snapshot dispatch is October 30 at 06:00 Madrid. Before
taking the NBA runner, the backfill waits on a GitHub-hosted runner for that run
to finish and for all three prediction workflows to have no active or queued
runs. If this takes more than two hours, it fails without starting the backfill.
A completed prediction run can have any conclusion, including failure.

Offline preview (no credentials, service calls, fitting or cron creation):

```bash
PYTHONPATH=.:src .venv/bin/python scripts/cronjobs/schedule_historical_backfill.py
```

Only when scheduling is authorized, append `--apply` to create/update the
one-off cron. This books a future dispatch; it does not start the backfill
immediately. It requires `CRONJOB_API_KEY`, `CRONJOB_GITHUB_DISPATCH_TOKEN`, and
the backfill workflow published on `main`. The script checks that GitHub has
registered the workflow before changing any cron, verifies the booked time,
and refuses past times or re-arming a cron that has already fired. The cron
expires at 05:20:59 UTC that day, so it cannot repeat the following year.

`--dry-run-predictions` changes the eventual workflow to real fits with local
reports only, without inserting predictions or archiving models. This is
independent of the scheduler's default offline preview. Hosted TabPFN consumes
credits if that workflow is later run, including in prediction dry run.

## Time assumptions

Training stops at D-1, except snapshots before 06:00 Eastern, which require an
extra overnight lag. Saved daily-history rows cannot guarantee that stricter
feature cutoff, so those inputs are reported as unavailable; they are never
rebuilt. Injury information comes from the saved snapshot, not the final injury
state. Reports record the input partition checksum/file and training cutoff.

The partitions are today's stored historical version. Filtering dates does not
undo later corrections or change their original point-in-time conventions.
Today's fixed configurations are reused, so any model-selection bias from those
same evaluation dates remains even though fitting uses only earlier rows.
