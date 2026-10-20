# Prediction storage: a new Aiven table

Updated: 2026-10-17. Status: design proposal. No tables have been created;
this change only updates the document.

## Direction and requirements

Start a new prediction store in **Aiven**, independent of the existing
Supabase table. Keep the old history in place; importing it is outside this
proposal. Keep **every output from every model in every run**, including
repeated predictions for the same earlier snapshot. A new run adds history;
a persistence retry must neither duplicate a row nor overwrite a conflict.

The frequent filters must be directly available in SQL:

- Actual time remaining before the game when the prediction was made.
- Absolute prediction timestamp in UTC, and potentially UTC hour of day.
- Feature/data schema version, such as `2_5` or `2_6`.
- Dataset type: `closing_line` or `intermediate_line`.
- Trained model horizon in minutes: T−0, T−30, T−360, etc.
- Model/target, game/date, outcome, and potentially minimum predicted edge.

SHAP is not required in this store. Optional explanations, complete inputs,
and detailed metadata belong in S3. Do not compute SHAP solely to save a
prediction.

## Findings that affect the design

The existing Supabase table has 47 columns. The 2026-10-16 read-only audit
found 9,835 predictions for 354 games. Its two totals lines were identical
throughout, and all registry identity/horizon fields were empty. We do not
need to preserve that schema or its compatibility defaults in the new store.

The 2026-10-17 read-only Aiven check reported PostgreSQL 18.6 and a database
size of 469,399,231 bytes, approximately 448 MiB. Line-history partitions
account for most of that space. This is database size, not free disk, WAL
usage, or verification of the service quota. The line-history schema comments
mention a 1 GB cap; confirm the actual limit before sizing retention.

Two repository contracts are particularly important:

1. `SpecIdentity.dataset_type` is independent of `horizon_minutes`: T−0 can
   use either dataset. Do not infer closing/intermediate from the horizon
   or the actual time remaining.
2. Intermediate totals **and spread** targets use a normalized anchor line.
   It can differ from the bookmaker's raw line and is not necessarily a
   half-point value. Preserve both references to distinguish model error
   from correctness against an observed quoted line.

## Recommended structure

Use one main fact table, **`prediction.predictions`**, containing the frequent
filters and the fields needed for next-day grading. Add a small immutable
**`prediction.model_builds`** catalog so each row references an `INTEGER`
instead of repeating long model names, fit identifiers, and S3 paths.

This replaces the earlier three-table proposal. A SQL run table is not
required initially: identify each run with a UUID and use a deterministic S3
manifest key, e.g. `prediction_snapshots/runs/{run_id}/manifest.json`.

Keep schema version, dataset type, target, and trained horizon in the fact
table even though they also describe the model. This deliberate duplication
supports the requested filters without joining metadata. Validate them
against the immutable model record on insertion.

## Proposed fields and PostgreSQL types

Fields are required unless marked nullable. Types and codes below describe
the proposed contract; they are not deployed DDL.

### Identity and timing

| Field | Type | Meaning and purpose |
| --- | --- | --- |
| `prediction_id` | `BIGINT GENERATED ALWAYS AS IDENTITY` | Compact primary key and stable output identifier. |
| `run_id` | `UUID` | Generated once per run, shared by its outputs and reused for persistence retries. |
| `game_id` | `TEXT` | NBA identifier, preserving leading zeros. No foreign key to a finished-only box score. |
| `game_date` | `DATE` | Schedule date using the existing pipeline convention; not necessarily the UTC date of tip-off. |
| `predicted_at` | `TIMESTAMPTZ` | Actual inference completion time for this output/batch, supplied by the application in UTC. Not insertion time or a timestamp captured before all models run. |
| `tipoff_at_prediction` | `TIMESTAMPTZ` | Scheduled tip-off known then. Preserve it after schedule corrections. |
| `time_to_tipoff_seconds` | `INTEGER` | Signed whole seconds, `floor(extract(epoch FROM tipoff_at_prediction - predicted_at))`. Stored for direct range filtering. |
| `inputs_observed_at` | `TIMESTAMPTZ` | Observation/as-of time of the frozen input bundle. Distinguishes later reconstructions of earlier snapshots. |
| `snapshot_at` | `TIMESTAMPTZ`, nullable | Nominal snapshot cutoff. Required for intermediate rows; null for a current closing frame without a formal historical cutoff. |
| `snapshot_horizon_minutes` | `SMALLINT`, nullable | Horizon of the input row; required for intermediate rows, including a future pooled model. Null for a current closing frame. |

Use aware timestamps ending in `Z` or `+00:00` and UTC database sessions.
`TIMESTAMPTZ` stores an instant and displays it in the session time zone; it
does not retain the original zone label. Avoid text timestamps and `TIME`.
See [PostgreSQL date/time types](https://www.postgresql.org/docs/current/datatype-datetime.html).

Positive lead time means before tip-off; negative means after it. Unknown
tip-off must produce a data-quality/persistence failure, not a substituted
zero. For minute filters, convert query bounds to seconds: 30–60 minutes is
`time_to_tipoff_seconds >= 1800 AND time_to_tipoff_seconds < 3600`.
Rounding every output to a whole minute loses boundary precision.

For intermediate rows validate:

```text
snapshot_at = tipoff_at_prediction - snapshot_horizon_minutes * 60 seconds
snapshot_at <= inputs_observed_at <= predicted_at
fixed-horizon model: snapshot_horizon_minutes = model_horizon_minutes
```

The nominal cutoff is not proof that every feature is historically
reconstructible. The current roster path uses the latest same-day roster
observation. Preserve the actual input bytes in S3.

### Model and data identity

| Field | Type | Meaning and purpose |
| --- | --- | --- |
| `model_build_id` | `INTEGER` | Foreign key to an immutable concrete build, not a mutable production pointer. |
| `data_schema_version` | `TEXT` | Served model's declared feature schema (`spec.identity.schema_version`), e.g. `2_5`; a version is not a decimal number. |
| `dataset_type` | `SMALLINT` | Checked code: `1 = closing_line`, `2 = intermediate_line`. A view exposes readable labels. |
| `model_target` | `SMALLINT` | Checked code: `1 = line_error`, `2 = total_points`, `3 = spread_error`. Identifies the market for current targets. |
| `model_horizon_minutes` | `SMALLINT`, nullable | Trained horizon; zero means T−0. Null is reserved for declared pooled models, not unknown metadata. |

Dataset type, trained horizon, input horizon, and actual time remaining are
different dimensions. A 19:00 run for a 23:00 game may run a T−360 model on
its 17:00 snapshot:

```text
predicted_at              = 19:00 UTC
time_to_tipoff_seconds    = 14400       # actual T−240
dataset_type             = 2           # intermediate_line
model_horizon_minutes    = 360
snapshot_horizon_minutes = 360
snapshot_at              = 17:00 UTC
```

The common live frame may contain newer additive columns than a model needs;
do not label every model with the newest frame version. Record the actual
input frame version in the manifest as well as the model's declared version.
Check dataset/target codes and horizon ranges on write; the registry permits
0–9,999 minutes, which fits `SMALLINT`.

Schema version, training dataset generation, and live input generation also
differ. Keep the model's declared schema version in SQL; keep training
build/checksum with model metadata and the exact live input identity/checksum
in the run manifest.
Add an input-generation SQL key only if it becomes a frequent filter.

The model catalog needs a numeric ID, unique immutable `model_key`, model
identity fields, and an S3 metadata reference. Registry identity includes
slot + fit and resolves its spec; an isolated name or fit ID is not assumed
globally unique. Keep variant, spec/fit IDs, training dates, training-code
version, and training dataset identity once there or in referenced S3
metadata. Baselines need explicit versioned identities; random seeds belong
in run artifacts.

### Prediction and market reference

| Field | Type | Meaning and purpose |
| --- | --- | --- |
| `prediction_value` | `DOUBLE PRECISION` | Original finite model output, without display rounding. |
| `model_reference_line` | `DOUBLE PRECISION` | Anchor used to interpret the output; it may be normalized. Required for current targets. |
| `reference_book_id` | `SMALLINT` | Stable book/source mapping. Reuse Aiven's book catalog where appropriate; never silently change the target anchor. |
| `betting_line_x2` | `SMALLINT`, nullable | Raw quoted line × 2, e.g. 224.5 → 449. Spread uses implied home margin. Validate half-point granularity instead of rounding other values. |
| `pick_side` | `SMALLINT` | Actual emitted decision: `+1 = OVER/HOME`, `-1 = UNDER/AWAY`, `0 = no pick`. |

`DOUBLE PRECISION` retains ordinary double-precision inference outputs;
`REAL` saves four bytes but loses precision. Reject NaN/infinity on input.
Use small integers for exact raw half-point lines and final scores, not
normalized lines. See PostgreSQL's [numeric types](https://www.postgresql.org/docs/current/datatype-numeric.html)
for storage and precision tradeoffs.

Store the original output rather than every equivalent output:

| Target | Predicted total or home margin | Model edge against its anchor |
| --- | --- | --- |
| `line_error` | `model_reference_line + prediction_value` | `prediction_value` |
| `total_points` | `prediction_value` | `prediction_value - model_reference_line` |
| `spread_error` | `model_reference_line + prediction_value` | `prediction_value` |

Spread follows `config/market_columns.py`: a positive reference means the
home team is expected to win by that margin. Convert negative home-handicap
quotes before storage.

Comparing the implied prediction with the raw quoted line gives a separate
**betting edge**. Current intermediate picks use the normalized residual;
preserve that emitted side instead of silently changing it on read/grading.
If a new policy selects against the raw line or applies thresholds, record
its version in the manifest and persist the side it actually selected.
`pick_side = 0` is no directional pick, not a later result push.

### Quotes: recommended for returns analysis

| Field | Type | Meaning |
| --- | --- | --- |
| `odds_positive` | `SMALLINT`, nullable | American odds for OVER/HOME at the stored raw line. |
| `odds_negative` | `SMALLINT`, nullable | American odds for UNDER/AWAY from the same paired quote. |
| `quote_at` | `TIMESTAMPTZ`, nullable | Source timestamp of that raw line/price observation. |
| `quote_basis` | `SMALLINT`, nullable | `1 = observed live for this run`, `2 = historical snapshot quote`. |

Line, prices, book, and timestamp must describe the same observation. Validate
raw spread sides are mirrored before storing one canonical line. Null
invalid/off-board prices using existing ingestion rules.

A T−240 rerun using a T−360 snapshot does not establish that its old quote
was still available at execution. Preserve quote basis/time. Returns against
snapshot quotes are retrospective scoring, not observed execution-time
opportunities. Fresh run-time quotes, if collected, are a distinct basis.

Keep the raw line even without returns analysis: it defines wins and pushes.
Prices may remain null. A winning pick with missing odds has unknown return,
not zero; a unit-stake loss is −1, and pushes/no-picks return zero regardless
of price. Never fill gaps with later prices or assumed −110. Normalized
references alone cannot establish executable returns.

### Results: updated after the game

| Field | Type | Meaning |
| --- | --- | --- |
| `home_points` | `SMALLINT`, nullable | Official final home score, including overtime for current full-game markets. |
| `away_points` | `SMALLINT`, nullable | Official final away score under the same convention. |
| `settlement_status` | `SMALLINT NOT NULL DEFAULT 0` | `0 = pending`, `1 = graded`, `2 = void`, `3 = final but ungradable`, e.g. missing raw line. |
| `pick_result` | `SMALLINT`, nullable | `+1 = win`, `-1 = loss`, `0 = push`, `2 = no pick`. Null for pending, void, or ungradable. |
| `result_updated_at` | `TIMESTAMPTZ`, nullable | Latest result/grade write time. |
| `evaluation_version` | `SMALLINT`, nullable | Grading-rule version, set during grading for explicit regrading. |

Expose `is_correct` in a view: true for win, false for loss, null for everything
else. Filter stored `pick_result` for wins/losses. Pushes, missing results,
and abstentions must not silently become losses.

Derive total points, home margin, signed/absolute prediction error, model
edge, betting edge, and unit-stake return. Do not duplicate totals or store
fixed-threshold booleans such as “error under 3 points.” Changing a threshold
should not require rewriting history.

## Next-day update contract

1. Read distinct pending game IDs using a partial index and fetch official
   results once per game. Revisit older pending games, not just yesterday:
   postponed games and failures must remain eligible for retry.
2. Require official final status and both scores. Avoid arbitrary total-score
   cutoffs. The old updater filters totals below 140; do not carry that
   heuristic into the new store. Pending scores remain null.
3. Update all rows for the game, but grade **each row using its stored raw
   line and emitted side**. Write scores/status/grade/version/timestamp
   together in transactional batches sized to limit lock/WAL bursts.
4. For totals compare `2 * (home_points + away_points)` with `betting_line_x2`;
   for spread compare `2 * (home_points - away_points)` with the canonical
   implied-home-margin line. Equality is push; otherwise matching the
   recorded side means win. A recorded no-pick is graded separately and
   does not require a market line.
5. Missing raw line prevents directional bet grading but not model-error
   analysis; mark the directional row ungradable. Missing odds alone do not
   prevent win/loss/push grading. Postponement stays pending unless an
   explicit void rule applies; void is never a loss.
6. Repeat safely, updating only changed facts/grades. Recheck a bounded recent
   final-game window for corrections and support targeted regrading by game
   ID. Scores/grades may change; original prediction facts must not.

Persist these facts in Aiven. Ordinary SQL cannot directly join the Supabase
database: transfer results in the job rather than requiring a remote lookup
for each dashboard query.

Point accuracy and pick correctness are separate: predicted total 226.2, raw
line 224.5, and actual total 225 gives absolute error 1.2 and an OVER win.
Actual total 224 makes that pick lose. Raw line 225 with actual total 225 is
a push regardless of point error.

## Keys and indexes

Primary key: `prediction_id`. Persistence identity:

```sql
UNIQUE NULLS NOT DISTINCT (run_id, game_id, model_build_id, snapshot_at)
```

Aiven's checked PostgreSQL version supports this null treatment; see
[unique constraints](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-UNIQUE-CONSTRAINTS).
It allows multiple snapshots from a pooled model in one run and one null-snapshot
closing output per model/game/run. Every new run UUID retains a new row.
Retries reuse the UUID, timestamp, and frozen payload; rerunning inference
is a new output/run. Compare immutable fields on conflict and reject a
mismatch rather than overwriting prior prediction facts.

Start with these B-tree candidates in addition to the primary/unique keys.
Retain or reorder them based on measured query plans and storage cost:

```sql
-- Absolute prediction-time windows, across all versions and horizons.
CREATE INDEX predictions_time_idx
    ON prediction.predictions (predicted_at);

-- Version/type/model-horizon equality, then a prediction-time range.
CREATE INDEX predictions_model_window_idx
    ON prediction.predictions
       (data_schema_version, dataset_type, model_horizon_minutes, predicted_at);

-- Actual lead-time ranges, independently of the trained horizon.
CREATE INDEX predictions_lead_time_idx
    ON prediction.predictions (time_to_tipoff_seconds, predicted_at);

-- Game history and result updates.
CREATE INDEX predictions_game_time_idx
    ON prediction.predictions (game_id, predicted_at DESC);

-- Small settlement queue; rows leave it when grading finishes.
CREATE INDEX predictions_pending_idx
    ON prediction.predictions (game_id) WHERE settlement_status = 0;
```

Put equality filters before a range in the intended composite access path.
Two range predicates do not become equally efficient just because both
columns are indexed. Measure lead-time ranges combined with timestamp ranges.
The model-window index is strongest when version, type, and model horizon
are specified; queries omitting one may need a different order. Do not create
every permutation. See PostgreSQL's [multicolumn indexes](https://www.postgresql.org/docs/current/indexes-multicolumn.html)
and [partial indexes](https://www.postgresql.org/docs/current/indexes-partial.html).

Example with the requested dimensions:

```sql
SELECT model_build_id, count(*) AS predictions,
       count(*) FILTER (WHERE pick_result = 1) AS wins,
       count(*) FILTER (WHERE pick_result = -1) AS losses,
       count(*) FILTER (WHERE pick_result = 0) AS pushes
FROM prediction.predictions
WHERE predicted_at >= TIMESTAMPTZ '2026-10-20 00:00:00+00'
  AND predicted_at <  TIMESTAMPTZ '2026-10-27 00:00:00+00'
  AND data_schema_version = '2_5'
  AND dataset_type = 2
  AND model_horizon_minutes = 360
  AND time_to_tipoff_seconds >= 1800
  AND time_to_tipoff_seconds < 3600
GROUP BY model_build_id;
```

This counts prediction rows, not independent games or actual bets. Statistical
evaluation needs an explicit per-game selection policy, grouping, or treatment
of repeated observations; keep all history regardless.

For absolute hours use half-open UTC timestamp ranges. For hour of day across
dates use `EXTRACT(HOUR FROM predicted_at AT TIME ZONE 'UTC')` plus a bounded
date range. Do not initially add an index for just 24 hour values; measure
an expression index if this becomes a bottleneck.

Do not index every outcome, book, target, and edge by default. The pending
index has an update cost when settlement changes, which measurements must
include. Add a generated edge/expression index only if threshold queries
justify it, after choosing model edge versus raw-line betting edge.
Partitioning/BRIN are later volume/retention decisions, not automatic speedups.

## Data kept outside the fact table

- SHAP values, feature explanations, SHAP confidence, and missing-feature lists.
- Full model parameters, feature lists, training/checksum/code metadata:
  reference the immutable model catalog and S3.
- Full input frames, injury/roster provenance, raw payloads, diagnostics, and
  decision-policy metadata: keep them in the run manifest.
- Repeated team names, matchup labels, and duplicate timestamp strings.
- Closing lines/CLV unless separately requested for postgame market comparison.
  Never replace the original raw line with a closing line.

Team IDs (`INTEGER`), season year (`SMALLINT`), and season type (`SMALLINT`)
are reasonable small additions if those filters are required. For live display
use schedule context in artifacts or a small shared game table, not finished-
only box scores. These are outside the minimum requested filter set.

Archive both closing **and intermediate** input frames. The current uploader
only receives the closing frame and must change. Use the UUID manifest path
rather than a second-resolution timestamp as the sole identity. Record upload
failures explicitly: a deterministic key does not prove an object exists.
Prediction persistence must not depend on successful optional SHAP work.

## Capacity and rollout

The current configuration enables 51 slots: three targets at T−0 and 16
intermediate horizons each. An illustrative upper bound is 1,230 games × 51
models × 30 runs per game = **1,881,900 rows per regular season**, before
baselines/playoffs. Actual eligibility reduces this; never create fake rows
for models that did not execute.

Small scalar types and a numeric model key reduce repetition, but heap/index
overhead and updates still matter. Scores and some derived fields are
deliberately repeated for direct queries. If capacity is too tight, first
evaluate a small local Aiven results table with one row per game and a join.
Do not silently discard runs; agree any SQL retention/S3 archival separately.

Implementation sequence:

1. Finalize quote basis, raw/normalized references, and whether returns or
   team filters are needed immediately.
2. Prepare reviewed DDL, the small model catalog, views, and explicit
   `connect_nba_db(env="aiven")`. Do not change global `DB_ENV` and reroute
   unrelated stores.
3. Propagate actual inference timestamps, snapshot metadata, and observation
   times; separate inference, optional SHAP, and persistence. Freeze retries.
4. Create the new Aiven store without dropping or importing the old table,
   and connect the new writer and run manifests.
5. Adapt and schedule the finished-match job. Its current workflow targets
   Supabase and has commented-out cron schedules.
6. Check new runs vs write retries, pooled/closing snapshots, refits, UTC
   midnight/DST, rescheduling, normalized/raw differences, spread signs,
   wins/losses/pushes/no-picks, missing prices, pending games, corrections,
   and batch failures.
7. Load representative data; measure `EXPLAIN (ANALYZE, BUFFERS)`, heap/index
   size, writes, and next-day grading. Inspect estimated plans before heavy
   analyses on the shared service. Check actual disk/WAL headroom before
   selecting indexes and retention.
8. Adapt consumers to each row's line, explicit dataset/horizon filters,
   and nullable correctness. Keep optional SHAP access independent of SQL.

## Repository evidence

- `src/nba_ou/postgre_db/predictions/create/create_ou_predictions_db.py`
- `src/nba_ou/postgre_db/predictions/update/update_total_points_predictions.py`
- `src/nba_ou/prediction/prediction.py`
- `src/nba_ou/prediction/intermediate_frame.py`
- `src/nba_ou/create_training_data/create_intermediate_line_df.py`
- `src/nba_ou/create_training_data/live_game_day.py`
- `src/nba_ou/config/market_columns.py`
- `src/nba_ou/modeling/registry_models.py`
- `src/nba_ou/modeling/registry_paths.py`
- `src/nba_ou/postgre_db/line_history_aiven/schema.py`
- `src/nba_ou/postgre_db/config/db_config.py`
- `src/nba_ou/utils/s3_prediction_snapshots.py`
- `scripts/predict_nba_games.py`
- `.github/workflows/update_finished_matches_daily.yml`
- `streamlit/streamlit_ou_app.py`
