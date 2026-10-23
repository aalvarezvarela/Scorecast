# Prediction storage: Aiven tables + S3 run archive

Status: **Implemented** (checked 2026-10-22) — merged in PR #49; the Aiven
schema was created on 2026-10-21 (see "Implementation status").

Updated 2026-10-21. It replaces the 2026-10-17 draft (one wide fact table with
results and settlement columns).

## Goals

1. Keep every model output needed for later analysis in a small, queryable
   **Aiven** store, independent of the old Supabase table (which stays as is;
   importing it is out of scope).
2. Archive the exact inputs of every run in **S3**, so any prediction can be
   reproduced, re-scored or explained later.
3. Distinguish the two ways an intermediate model is served (below), so the
   analysis can compare a model on its own horizon with the same model on a
   later snapshot -- e.g. whether a T−360 model predicts at T−60 better than
   the T−60 model does.
4. Make it obvious which **line** each prediction was betting against.

The SQL filters that must be direct columns of the analysis view (no JSON):

- actual time remaining before tip-off when the prediction was made;
- absolute prediction timestamp (UTC);
- data schema version (`2_5`, later `2_6`, …);
- target (`line_error`, `total_points`, `spread_error`);
- the horizon the model was **trained** for, and the horizon of the **input
  row** it predicted;
- input mode (closing / own snapshot / latest snapshot), game and date.

Out of scope: SHAP values, SHAP confidence scores and any other explanation
output. They are removed from the prediction path and not stored; see
"SHAP: removed, recomputable".

## How predictions are produced

Two workflows write predictions
([nba_predictor_daily.yml](../../.github/workflows/nba_predictor_daily.yml),
[nba_predict_past_snapshots.yml](../../.github/workflows/nba_predict_past_snapshots.yml);
details in [docs/intermediate/live_prediction.md](../intermediate/live_prediction.md)):

| Run | When | Closing slots | Intermediate slots |
| --- | --- | --- | --- |
| Daily predictor, `INTERMEDIATE_INPUT_MODE = latest` | Several times a day (planned: one run per grid time) | Tonight's closing frame, one row per game | Every slot reads each game's **latest** snapshot -- the run's own moment, off the grid in general (e.g. T−47) -- limited per model by `input_limits.limit_to_training_support` |
| Past-snapshot run, `own` | Once, after the date's games have tipped | Not run (post-tip odds are not the closing line) | Every slot reads the row of **its own** horizon: the exact simulation of what it was trained on |

Consequences for storage:

- A run produces **at most one row per (game, model build)**: the latest
  row, the own row, or the closing row. No run re-predicts a snapshot an
  earlier run already predicted, so no "first prediction wins" rule is
  needed.
- In `latest` mode the 48 intermediate models read the same base row, but
  each sees its own limited copy: referees masked at the model's nominal
  snapshot `tip − H`, unsupported news columns set to 0. The limits are a
  deterministic function of (row, model horizon, support table), so the base
  row is archived once and the per-model copies are recomputed when needed.
  `INPUT_LIMITED_COUNT` (cells changed) is stored per prediction.
- `latest` rows are what was known at that moment. The `own` run rebuilds
  past snapshots at the end of the day: lines, injury reports and referees
  are cut at the snapshot, but the roster is the latest same-day observation
  (`LiveGameDay`), and ticks or reports ingested late may differ from what a
  live run would have seen. Analyses that need strictly live information
  filter on `input_mode`.

## Capacity

Per game, with the current 51 enabled slots, two baselines and ~17 daily
runs before tip (estimate; the cron schedule is not final):

| Source | Rows per game |
| --- | --- |
| `latest` runs: (48 intermediate + 3 closing + 2 baselines) × ~17 runs | ~900 |
| `own` run: 48 intermediate | 48 |
| **Total** | **~950** |

A regular season (1,230 games) is **~1.2M prediction rows**. With the
narrow layout below (per-model values only, no B-tree key) that is an
estimated **~75 MB per season** including `inputs`; measure it in rollout
step 6.

Aiven: the free plan is **1 GB**. A read-only check on 2026-10-21 found
448 MiB in use, almost all line history: 395 MiB over seasons 2019–2025,
of which the 2025 season alone is 128 MiB (862k quotes, about half of it
index). If the new season grows like the last one, line history adds
~130 MiB and predictions ~75 MB: ~650 MB at season end. The plan can be
upgraded to 10 GB cheaply; do that when the measured growth (step 6) says
the 1 GB limit will be reached, rather than pruning data. If space is ever
needed without upgrading, the largest lever is archiving closed line-history
seasons to S3 (one partition each, exporter in
`line_history_aiven/backup.py`), which needs the historical dataset builder
to read those seasons from Parquet.

Statistical note: one live season is ~1,230 games per cell, a standard error
of about 1.4 pp on a 55% accuracy, and earlier work found the 16 on-horizon
models statistically interchangeable (chi² p = 0.999, noise floor
1.1–2.6 pp). Expect the live comparison to separate only large effects. The
same model-horizon × snapshot-horizon matrix can be computed offline on the
historical intermediate dataset, where every game has all snapshots; run
that as a `training_pipeline` evaluation in parallel with live collection.

## S3 run archive

Replace the current layout (`prediction_snapshots/{date}/{Madrid timestamp}/`
with one Parquet per model and only the closing frame archived) with one
folder per run:

```text
prediction_snapshots/v2/{game_date}/{run_uuid}/
  manifest.json                # run metadata, see below
  inputs_closing_line.parquet       # the closing rows the models read
  inputs_intermediate_line.parquet  # latest run: each game's latest row;
                                    # own run: every passed snapshot row
  predictions.parquet          # every output of this run, long format
```

- Each input row is written **once per run, not once per model**. All models
  of a frame read the same base row.
- `predictions.parquet` mirrors the SQL rows with their input row's lines,
  the game's date and teams, the slot label and `NA_COLUMNS_NAMES`; it is the
  complete record if SQL is ever pruned.
- `manifest.json`: run UUID, UTC start/finish, workflow, status, git SHA,
  main book, the schema version of each frame (a 2_5 model may read a frame
  built at a later version), a hash of the input-support table, row counts,
  the upload status of each file, schedule/frame errors, and **one entry per
  slot**: predicted, skipped or failed, the reason, rows, and the games its
  NaN budget (`max_na_per_row`) left out -- "the model did not predict" and
  "the data was missing" stay distinguishable.
- Every file is uploaded on its own and the manifest is written last, even
  when a file failed. Paths and timestamps are UTC. An upload failure is
  recorded in the manifest and in `runs.status`; a deterministic key does not
  prove an object exists.

### File size: measured

Measured on `intermediate_line_data_2_5_20261016.parquet` (3,082 columns),
one game date with 15 games × 17 horizons = 255 rows, zstd:

| Case | Size |
| --- | --- |
| 255 rows, written as today (`DataFrame.to_parquet`) | 3.37 MB |
| … of which the footer (per-column metadata + pandas schema JSON) | 2.38 MB |
| 1 row, same columns | 1.06 MB |
| The 255 rows repeated 48 times **in one file** | 3.65 MB (+8%) |
| 255 rows, no pandas metadata, no column statistics | 1.72 MB |
| … and features as float32 | 1.52 MB |
| Long-format predictions, 12,240 rows × 5 columns | 0.12 MB |

What this means:

- Parquet already absorbs duplicated values **within a file** (dictionary +
  RLE per column chunk): 48 copies cost 8% more, not 48×.
- The waste is **many small files**: with ~3,000 columns, every file pays
  ~1–2 MB of footer whatever its row count, and duplicates across files are
  not shared. Hence one input file per frame per run, never one per model.
- Data proper is ~4 KB per row compressed.

Writer settings: drop the pandas schema metadata
(`table.replace_schema_metadata(None)`, −1.4 MB per file) and write without
column statistics (−0.26 MB). Optionally store float64 features as float32:
XGBoost converts its input to float32 anyway, so predictions and SHAP are
reproduced exactly; keep identifiers, timestamps and integer columns as they
are.

Volume: ~17 runs × 2 input files × ~1 MB plus the `own` run is ~35 MB per
day, ~6 GB per season, cents per month in S3. A monthly compaction into one
file per month would bring it under 0.5 GB; it is not needed for cost.

## SQL schema (Aiven, schema `prediction`)

Five tables, sized so that the one large table carries only what differs
per model:

| Table | One row per | Rows per season |
| --- | --- | --- |
| `predictions` | model output | ~1.2M |
| `inputs` | input row a run fed to its models | ~60k |
| `games` | game | ~1.3k |
| `runs` | run | a few thousand |
| `model_builds` | concrete model build | a few hundred |

In `latest` mode the 48 intermediate models read the same base row, so the
lines, prices and timing of that row live once in `inputs` instead of 48
times in `predictions`. Everything is queried through one view
(`predictions_full`, below), so the common filters stay direct columns.

Write order and idempotency: a run writes everything in **one transaction
at its end** -- upsert `games` and `model_builds`, insert the `runs` row,
its `inputs`, then its `predictions`. `runs.run_uuid` is unique, so a retry
of a run that was already stored fails on that key and writes nothing; a run
that failed before committing stored nothing. Slots that failed inside a run
are left out and the run is stored as `partial`. A run that predicted
nothing still stores its `runs` row (and its S3 manifest), so coverage can be
measured. Only `games`, `runs` and `model_builds.s3_build_prefix` (see
"Archived builds") are ever updated; `inputs` and `predictions` are
append-only.

### `prediction.games`

Upserted by every prediction run **before** its results exist (schedule
fields; `status = 0`, scores null), and completed by the finished-games job.

| Column | Type | Meaning |
| --- | --- | --- |
| `game_id` | `INTEGER` PK | NBA game id without leading zeros (`0022500001` → `22500001`); format back with `lpad(game_id::text, 10, '0')`. |
| `game_date` | `DATE` | The NBA's date for the game (`GAME_DATE_EST` of the schedule), not the run's date. |
| `season_year` | `SMALLINT` | |
| `season_type` | `SMALLINT` | Regular / playoffs / play-in / pre-season codes. |
| `home_team_id`, `away_team_id` | `INTEGER` | |
| `home_team`, `away_team` | `TEXT` | Team abbreviations for display. |
| `tipoff_at` | `TIMESTAMPTZ` | Latest known scheduled tip-off. |
| `home_points`, `away_points` | `SMALLINT`, nullable | Official final score incl. overtime; null until final. |
| `status` | `SMALLINT` | `0` scheduled, `1` final, `2` postponed/void. |
| `result_updated_at` | `TIMESTAMPTZ`, nullable | |

### `prediction.runs`

| Column | Type | Meaning |
| --- | --- | --- |
| `run_id` | `INTEGER` identity PK | |
| `run_uuid` | `UUID` unique | Also the S3 folder name. |
| `workflow` | `SMALLINT` | `1` daily predictor, `2` past-snapshot run, `3` TabPFN run (`scripts/predict_tabpfn.py`). |
| `started_at`, `finished_at` | `TIMESTAMPTZ` | UTC. |
| `git_sha` | `TEXT` | |
| `status` | `SMALLINT` | `1` ok, `2` partial, `3` S3 upload failed. |

### `prediction.model_builds`

| Column | Type | Meaning |
| --- | --- | --- |
| `model_build_id` | `INTEGER` identity PK | |
| `kind` | `SMALLINT` | `1` registry model, `2` baseline, `3` TabPFN (fitted per run on a registry slot's spec, which `spec_id` names; `s3_build_prefix` is the training frame's key). |
| `schema_version` | `TEXT` | e.g. `2_5`. |
| `dataset_type` | `SMALLINT` | `1 = closing_line`, `2 = intermediate_line`. |
| `target` | `SMALLINT` | `1 = line_error`, `2 = total_points`, `3 = spread_error`. |
| `horizon_minutes` | `SMALLINT`, nullable | Trained horizon; `0` = closing; null only for a declared pooled model. |
| `variant` | `TEXT` | |
| `spec_id`, `fit_id` | `TEXT` | Registry identity; baselines get an explicit versioned name. |
| `train_date_min`, `train_date_max` | `DATE`, nullable | |
| `training_code_tag` | `TEXT`, nullable | Used by the dashboard filter. |
| `anchor_book` | `TEXT` | Book of the stored lines. |
| `s3_build_prefix` | `TEXT`, nullable | |

Unique on `(schema_version, target, horizon_minutes, variant, fit_id)`.
Closing slots refit daily, so this grows by a few rows per day.

### `prediction.inputs`

One row per input row a run fed to its models: per game, the closing row
and the latest intermediate row of a daily run, or the 16 snapshot rows of
the past-snapshot run. Both markets are stored, since the three targets of a
game read the same row.

| Column | Type | Meaning |
| --- | --- | --- |
| `input_id` | `INTEGER` identity PK | |
| `run_id` | `INTEGER` → `runs` | |
| `game_id` | `INTEGER` → `games` | |
| `tipoff_at` | `TIMESTAMPTZ` | Tip-off known at prediction time (`games.tipoff_at` may change later). |
| `input_mode` | `SMALLINT` | `0 = closing` (closing frame), `1 = own` (each model's own snapshot: exact horizon simulation), `2 = latest` (the game's latest snapshot, read by every model: cross-snapshot). |
| `snapshot_horizon_minutes` | `SMALLINT`, nullable | `TIME_TO_MATCH_MIN` of the row; off the grid for `latest`; null for closing. |
| `total_model_line`, `spread_model_line` | `REAL`, nullable | Anchor the outputs are relative to: the normalized line for intermediate rows, the main-book line for closing rows. Spread as implied home margin. |
| `total_bet_line_x2`, `spread_bet_line_x2` | `SMALLINT`, nullable | **The line being bet**: raw quoted main-book line × 2 (224.5 → 449). Spread as implied home margin. |
| `total_price_over`, `total_price_under` | `SMALLINT`, nullable | American odds at `total_bet_line_x2`, same quote. |
| `spread_price_home`, `spread_price_away` | `SMALLINT`, nullable | American odds at `spread_bet_line_x2`, same quote. |
| `total_line_age_minutes`, `spread_line_age_minutes` | `REAL`, nullable | Minutes since that quote was last observed, at the row's time. |

Unique `NULLS NOT DISTINCT (run_id, game_id, input_mode,
snapshot_horizon_minutes)`.

### `prediction.predictions`

Only what differs per model. Columns are ordered 4-byte then 2-byte to avoid
alignment padding.

| Column | Type | Meaning |
| --- | --- | --- |
| `input_id` | `INTEGER` → `inputs` | The row the model read (gives run, game, mode, snapshot, lines). |
| `model_build_id` | `INTEGER` → `model_builds` | Immutable concrete build (slot + fit), never a channel pointer. |
| `time_to_tipoff_seconds` | `INTEGER` | `inputs.tipoff_at − predicted_at`, taken after this model's inference. Positive = before tip. Unknown tip-off fails the write; never 0. |
| `prediction_value` | `REAL` | Raw model output. XGBoost predicts in float32, so `REAL` is lossless; reject NaN/infinity. |
| `pick_side` | `SMALLINT` | Emitted decision: `+1` OVER/HOME, `−1` UNDER/AWAY, `0` no pick. |
| `na_feature_count` | `SMALLINT` | Missing model features in the input row (names go to S3). |
| `input_limited_count` | `SMALLINT` | Feature cells changed by the training-support limits; 0 for `own` and closing. |

No primary key: a B-tree on `(input_id, model_build_id)` would cost about as
much as the table itself, and the run transaction already prevents
duplicates. One BRIN index finds a run's or a date's rows; it stays a few
kilobytes because rows arrive in `input_id` order:

```sql
CREATE INDEX predictions_input_brin
    ON prediction.predictions USING brin (input_id);
```

Each slot's output is validated when the run receives it
(`PredictionRunRecorder.add_predictions`), and an output that breaks the
contract is rejected **alone** -- reported as that slot's failure, with the
reason in the manifest -- while the rest of the run is stored. The contract:
every game is scheduled with a tip-off; every row is a row of its frame (the
own or latest snapshot `INPUT_MODE` names); an `input_mode = 1` row is the
model's own horizon (`snapshot_horizon_minutes = horizon_minutes`); the
output's line is that row's line; values are finite; and no (input, build)
pair is predicted twice in a run. A schedule or frame the store cannot read
rejects the outputs that need it, never the run.

Size, measured on PostgreSQL 16 (2026-10-21, 1.2M synthetic rows): 52.2
bytes per `predictions` row, 59.7 MiB for a season's ~1.2M rows, 48 KiB of
BRIN index; `inputs` takes ~133 bytes per row with its keys, ~8 MB for ~60k
rows. About 68 MiB per season in all; the first draft (one wide row per
prediction with a B-tree key) was ~180 MB.

### The bet line

Intermediate rows carry the main book's quote at the snapshot (`bet365`
today, `get_main_book()`):

| `inputs` column | Intermediate input column |
| --- | --- |
| `total_bet_line_x2` / `spread_bet_line_x2` | `ODDS_SNAP_TOT_BET365_RAW_LINE` / `ODDS_SNAP_SPR_BET365_RAW_LINE` × 2 |
| prices | `ODDS_SNAP_{TOT,SPR}_BET365_PRICE_LEFT` / `_PRICE_RIGHT` |
| `*_line_age_minutes` | `ODDS_SNAP_{TOT,SPR}_BET365_LINE_AGE_MINUTES` |

- The input limits only touch referee and news columns, so every model that
  reads a `latest` row sees the same lines and prices.
- In the line history `left` is OVER for totals and the AWAY side for spread
  (`data_processing/line_history/snapshots.py`), and the snapshot's raw
  spread line is already the implied home margin (checked on the 2_5
  dataset: it equals `ODDS_SPREAD_LINE_HOME_bet365 − NORM_MINUS_RAW`).
- The closing frame normalizes the line **and** prices in place and rounds
  the line back to a half-point, so it keeps no raw quote. A closing row's
  bet line, prices and age come from the game's latest intermediate row of
  the same run (`LATEST_INPUT`, the run's own moment): the quote that could
  be bet when the run predicted. A run without intermediate rows stores a
  null bet line for its closing rows.
- Line, prices and age describe the same observation. Never fill a missing
  price with a later one or an assumed −110.
- Today's intermediate pick is taken against the normalized line (the model
  line); where it differs from the raw line (`..._NORM_MINUS_RAW`), the side
  against the raw line can differ or land exactly on it. Store the side
  actually emitted; the view also exposes the side implied against the bet
  line.
- With `input_mode = 2` the quote is the one live at prediction time
  (staleness in the line age); with `input_mode = 1` it is historical, so
  returns against it are retrospective, not executable.

## The analysis view

All analysis reads `predictions_full`: the small tables are joined once, the
market columns are picked by target, and grading is computed. Nothing in it
is stored, so changing a grading rule never rewrites rows.

```sql
CREATE OR REPLACE VIEW prediction.predictions_full AS
WITH base AS (
    SELECT p.*,
           i.run_id, i.game_id, i.input_mode, i.snapshot_horizon_minutes,
           i.tipoff_at - make_interval(secs => p.time_to_tipoff_seconds)
                                                     AS predicted_at,
           i.tipoff_at                               AS tipoff_at_prediction,
           r.run_uuid, r.workflow, r.status          AS run_status,
           r.started_at                              AS run_started_at,
           m.kind, m.schema_version, m.dataset_type, m.target,
           m.horizon_minutes                         AS model_horizon_minutes,
           m.variant, m.fit_id, m.training_code_tag,
           g.game_date, g.season_year, g.season_type, g.tipoff_at,
           g.home_team, g.away_team, g.status,
           g.home_points, g.away_points,
           CASE WHEN m.target = 3 THEN i.spread_model_line
                ELSE i.total_model_line END          AS model_line,
           CASE WHEN m.target = 3 THEN i.spread_bet_line_x2
                ELSE i.total_bet_line_x2 END         AS bet_line_x2,
           CASE WHEN m.target = 3 THEN i.spread_price_home
                ELSE i.total_price_over END          AS bet_price_over_home,
           CASE WHEN m.target = 3 THEN i.spread_price_away
                ELSE i.total_price_under END         AS bet_price_under_away,
           CASE WHEN m.target = 3 THEN i.spread_line_age_minutes
                ELSE i.total_line_age_minutes END    AS bet_line_age_minutes,
           CASE WHEN m.target = 3 THEN g.home_points - g.away_points
                ELSE g.home_points + g.away_points END AS actual_value
    FROM prediction.predictions p
    JOIN prediction.inputs i       USING (input_id)
    JOIN prediction.model_builds m USING (model_build_id)
    JOIN prediction.games g        USING (game_id)
    JOIN prediction.runs r         ON r.run_id = i.run_id
), implied AS (
    SELECT base.*,
           -- The model's implied total or home margin.
           CASE WHEN target = 2 THEN prediction_value
                ELSE model_line + prediction_value END AS predicted_value
    FROM base
)
SELECT implied.*,
       predicted_value - model_line                        AS model_edge,
       predicted_value - bet_line_x2 / 2.0                 AS bet_edge,
       sign(2 * predicted_value - bet_line_x2)::smallint   AS bet_side,
       (model_horizon_minutes = snapshot_horizon_minutes)  AS on_horizon,
       CASE
         WHEN status <> 1 OR pick_side = 0 OR bet_line_x2 IS NULL THEN NULL
         WHEN 2 * actual_value = bet_line_x2 THEN 0
         WHEN sign(2 * actual_value - bet_line_x2) = pick_side THEN 1
         ELSE -1
       END::smallint                                       AS pick_result
FROM implied;
```

Rules:

- `pick_result`: `1` win, `−1` loss, `0` push, null for pending, void,
  no-pick or missing bet line. Pushes, pending games and no-picks never count
  as losses.
- The emitted side is graded against the bet line; it is never recomputed
  on read. `bet_side` is the side the prediction implies against the bet
  line, for analyses that want that policy instead.
- Unit-stake return (a view column when needed): win → profit from the stored
  American odds, null if the odds are missing; loss −1; push 0.
- Point error and pick correctness are separate: predicted 226.2, line 224.5,
  actual 225 → error 1.2 and an OVER win.
- Thresholds (min edge, "error under 3") are query parameters, never stored
  booleans.

Example: T−360 models predicting around T−60 versus the T−60 models on their
own snapshot.

```sql
SELECT model_horizon_minutes, input_mode,
       count(*) FILTER (WHERE pick_result = 1)  AS wins,
       count(*) FILTER (WHERE pick_result = -1) AS losses
FROM prediction.predictions_full
WHERE schema_version = '2_5' AND target = 1
  AND (  (input_mode = 2 AND model_horizon_minutes = 360
          AND time_to_tipoff_seconds BETWEEN 45 * 60 AND 75 * 60)
      OR (input_mode = 1 AND model_horizon_minutes = 60))
GROUP BY 1, 2;
```

This counts prediction rows, not independent games: a game predicted in
several runs must be reduced to one row per game (e.g. the run closest to a
target time) before any significance test. Re-dispatching the `own` run
creates a second run; analyses take the latest `own` run per game.

## Finished-games job

Only `games` is updated:

1. Select games with `status = 0` and `tipoff_at < now()`, not just
   yesterday's, so postponed games and failed fetches are retried.
2. Require official final status and both scores; no total-points
   heuristics (the old updater's `< 140` filter is not carried over).
   A game is final only when both teams have a result **and** points; any
   other game is left for the next run.
3. Upsert scores, status and `result_updated_at`. Re-check the games
   finalised in the last `--recheck-days` (default 3) for corrections;
   `--game-id` re-reads named games whatever their status.
4. Results come from the NBA API (`LeagueGameFinder`, one call per season),
   written into Aiven, so dashboard queries never cross databases.
5. Postponed and void games: a postponed game normally keeps its id and is
   played on a new date; it stays pending, is looked up again every run, and
   the prediction run of its new date retimes it. Games still pending
   `--stale-days` (default 3) after their stored tip-off are reported. There
   is no reliable automatic signal for void, so `status = 2` is a manual
   update.

## Archived builds

`scripts/archive_registry_builds.py` moves superseded builds (and specs no
build left in the slot uses) to `models/retired/{YYYYMMDD}/...`. Protecting
every build a prediction references would leave nothing to archive -- every
production build predicts -- so the move is recorded instead: once the
originals are deleted, the script rewrites the moved builds'
`s3_build_prefix` in `prediction.model_builds` to the retired path. If Aiven
cannot be reached it prints the `UPDATE` statements to run later. A retired
build's spec, when moved too, is under the same retired date.

## SHAP: removed, recomputable

SHAP is removed from the prediction path: `compute_shap_values` runs for
every batch today and its output only feeds the old Supabase columns and the
dashboard's reasoning tab. `shap` can then leave the predictor's requirements.

SHAP can be recomputed exactly for any stored prediction, because the
archive keeps everything it depends on:

1. **The input row**: the run's `inputs_*.parquet`; for `input_mode = 2`,
   re-apply `limit_to_training_support(row, horizon_minutes=H)` with the
   support table of the manifest's git SHA.
2. **The model**: `builds/{fit_id}/` in the registry is immutable and its
   spec holds the feature list and order. `model_builds.s3_build_prefix`
   says where it is now (see "Archived builds").
3. **The feature matrix**: the same preparation as serving
   (`_prepare_required_features_for_prediction` and the `max_na_per_row`
   filter in `load_and_predict_model_for_nba_games`), factored into one
   function that both serving and the offline tool call.
4. **The same XGBoost version** (`2.1.4`). Native TreeSHAP needs no `shap`
   package: `booster.predict(DMatrix(X), pred_contribs=True)`.

Check: the recomputed prediction must equal `prediction_value` in SQL
(both float32); if it does, the SHAP values are the ones the model
would have produced that day.

## What the Streamlit dashboard needs

Checked against `streamlit/streamlit_ou_app.py`:

| Dashboard use | Source in the new store |
| --- | --- |
| Teams, tip-off time (Madrid), season type | `games` |
| Latest prediction per model before tip / before Madrid midnight | `predicted_at`, `games.tipoff_at` |
| Timing buckets (≤1h, 2–4h, 4–6h) | `time_to_tipoff_seconds` |
| "Move vs final" and pick agreement vs final | closing rows of every run |
| Accuracy by min edge (0, 0.5 … 2.0) | `model_edge` / `bet_edge` in the view |
| OVER/UNDER accuracy, confusion matrix | `pick_side`, `pick_result` |
| Consensus variants (mean, vote, 5y3y, bold contrarian) | computed in the app from per-model rows |
| `training_code_tag` filter | `model_builds` |
| Missing-feature count / names | `na_feature_count` / S3 |
| Baselines | `model_builds.kind = 2` |
| TabPFN | `model_builds.kind = 3` |
| SHAP reasoning tab | removed |

New dashboard work, outside this store: the dashboard has no spread views
and groups models by a `model_type` string, not by target and horizon. It
needs target / horizon / input-mode filters and a model-horizon ×
snapshot-horizon matrix view (accuracy, MAE, n).

## Implementation status

Implemented (branch `feat/prediction-storage-aiven`, 2026-10-21):

| Piece | Where |
| --- | --- |
| DDL, codes, view | `src/nba_ou/postgre_db/predictions_aiven/schema.py` |
| Run writer, results updater | `src/nba_ou/postgre_db/predictions_aiven/store.py` |
| Final scores from the NBA API | `src/nba_ou/postgre_db/predictions_aiven/results.py` |
| Run rows from frames and outputs, `persist_prediction_run` | `src/nba_ou/prediction/prediction_store.py` |
| S3 run archive (v2) | `src/nba_ou/utils/s3_prediction_runs.py` |
| Wiring | `scripts/predict_nba_games.py`, `scripts/predict_past_snapshots.py` |
| Create the schema (`--print` shows the SQL) | `scripts/prediction_store/create_prediction_store.py` |
| Results job | `scripts/prediction_store/update_game_results.py`, step in `update_finished_matches_daily.yml` |
| Tests (the writer ones need `NBA_OU_TEST_PG_DSN`) | `tests/test_prediction_store.py` |

- SHAP is removed from `prediction.py` and from the predictor's requirements
  (the old Supabase columns are written null).
- Each model's output carries `PREDICTED_AT_UTC` (taken right after its
  inference), `MODEL_DATASET_TYPE` and `MODEL_BUILD_PREFIX`.
- The old per-model S3 snapshots are replaced by the v2 archive. The old
  Supabase table is still written, in parallel, until it is removed
  (rollout 3) -- but no longer from inside the predictor: the scripts call
  it separately after the store has the output, so a Supabase failure is
  reported (the run exits non-zero) without costing the store anything, and
  it does not make the store's run `partial`.
- A storage failure does not lose the run's other output: it is reported
  with the slot failures and the run exits non-zero.
- Baselines: the historical average's fit id is `1.0-<last history date>`,
  a build per history window, so each value keeps the calculation that
  produced it; the random baseline stays `1.0`. A baseline's inference time
  is taken when it is added to the run.
- `predictions_full` also exposes `season_year`, `season_type`, the game's
  current `tipoff_at`, `tipoff_at_prediction`, and the run's `run_uuid`,
  `workflow`, `run_status`, `run_started_at`.

Not done yet:

- Factoring the feature-matrix preparation into one function shared with an
  offline SHAP tool (only needed when SHAP is first recomputed).
- `TIME_TO_MATCH_MINUTES` still zero-fills a missing `GAME_TIME`, but only in
  the old Supabase rows; the new store computes its own time to tip and
  refuses a game without a tip-off.
- Deleting the (now unused by the scripts) Supabase upload path inside
  `load_and_predict_model_for_nba_games` and the scripts' parallel write,
  after the parallel week.

## Rollout

1. ~~Review the DDL (`create_prediction_store.py --print`) and create it on
   Aiven.~~ Done 2026-10-21.
2. Dry-run one game day (`--no-persist`) for both workflows.
3. Turn on in both workflows; keep the old Supabase write running in
   parallel for a week, then remove it.
4. Schedule the finished-games job (its workflow is dispatch-only today).
5. Recompute SHAP offline for a handful of stored predictions to confirm the
   archive is sufficient (prediction values must match).
6. After ~2 weeks: measure table size per game, write time, Aiven growth
   including line history, and the dashboard queries
   (`EXPLAIN (ANALYZE, BUFFERS)`); add indexes only where measured (a
   B-tree on `predictions (model_build_id)` is the likely first one);
   upgrade the Aiven plan if the 1 GB limit is in sight.
7. Adapt the dashboard (target / horizon / input-mode filters, horizon
   matrix, no SHAP).

## Open decisions

- Final cron schedule (runs per day) — drives the `latest` and closing
  volume.
- Whether `games` lives in the prediction schema or reuses an existing
  schedule table in Aiven.

## Repository evidence

- `src/nba_ou/prediction/prediction.py`
- `src/nba_ou/prediction/intermediate_frame.py`
- `src/nba_ou/prediction/input_limits.py`
- `src/nba_ou/prediction/baseline_predictions.py`
- `src/nba_ou/create_training_data/live_game_day.py`
- `src/nba_ou/utils/s3_prediction_snapshots.py`
- `scripts/predict_nba_games.py`, `scripts/predict_past_snapshots.py`
- `.github/workflows/nba_predictor_daily.yml`,
  `.github/workflows/nba_predict_past_snapshots.yml`,
  `.github/workflows/update_finished_matches_daily.yml`
- `src/nba_ou/postgre_db/predictions/create/create_ou_predictions_db.py`
- `src/nba_ou/postgre_db/predictions/update/update_total_points_predictions.py`
- `src/nba_ou/config/market_columns.py`
- `src/nba_ou/modeling/registry_models.py`, `registry_paths.py`
- `src/nba_ou/postgre_db/line_history_aiven/schema.py`
- `src/nba_ou/postgre_db/config/db_config.py`
- `streamlit/streamlit_ou_app.py`
