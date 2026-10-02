# Model registry reorganisation: target × horizon, spec-driven refits

Status: **implemented except the daily dataset build (item 10) and the first
real promotion (item 13).** See section 7 for what each item landed as. The
retirement sweep (item 12) ran on 2026-09-21: all 751 objects of the old
tree now live under `models/retired/20260921/`.

| what | where |
|---|---|
| paths, slots, targets | [registry_paths.py](src/nba_ou/modeling/registry_paths.py) |
| spec / fit / pointer documents | [registry_models.py](src/nba_ou/modeling/registry_models.py) |
| S3 reads, writes, resolution | [registry_store.py](src/nba_ou/modeling/registry_store.py) |
| refit from a spec | [refit.py](src/nba_ou/modeling/refit.py) |
| build promotion and rollback | [build_promotion.py](src/nba_ou/modeling/build_promotion.py) |
| run -> spec | [training_pipeline/registry.py](training_pipeline/registry.py) |
| promote a configuration | [training_pipeline/promote.py](training_pipeline/promote.py) `--to-s3` |
| daily refit | [scripts/retrain_prediction_models.py](scripts/retrain_prediction_models.py) |
| daily promotion | [scripts/promote_build.py](scripts/promote_build.py) |
| retirement sweep | [scripts/retire_legacy_models.py](scripts/retire_legacy_models.py) |
| training-data layout in S3 | [train_data_store.py](src/nba_ou/create_training_data/train_data_store.py) |
| training-data sweep | [scripts/organize_s3_train_data.py](scripts/organize_s3_train_data.py) |
| tests | [test_model_registry_paths.py](tests/test_model_registry_paths.py), [test_model_registry_store.py](tests/test_model_registry_store.py), [test_prediction_targets_and_horizon.py](tests/test_prediction_targets_and_horizon.py), [test_train_data_store.py](tests/test_train_data_store.py) |

Two things are being asked of the registry that it was not built for:

1. **Three targets** — `line_error`, `total_points`, `spread_error` — where the
   serving path today knows only two, and infers which one it is by looking for
   substrings in a name.
2. **One model per horizon**, roughly hourly, where the concept of "horizon"
   appears nowhere in S3, nowhere in `ModelInfo`, and nowhere in the
   predictions table.

and one thing it *does* do is worth preserving unchanged: a daily refit that
promotes a fresh fit into service and keeps the one it replaces.

The design below separates **the configuration** (chosen once, by hand, from an
experiment you trust) from **the fit** (recomputed every day, unattended), and
makes both immutable behind a pointer. That split is the whole plan; the
directory layout follows from it.

### This is a greenfield build, not a migration

Nothing currently in S3 will serve again. The six production bundles date from
2026-06-13, predate the schema they would need to be, and are being retired
along with the season-window variants. So:

- **No back-compatibility layer.** No legacy substring-sniffing branch in the
  serving path, no deprecated `prediction_model_prefixes` shim. The old code
  paths are deleted, not wrapped.
- **The old tree is retired wholesale** (§3), not converted.
- **Nothing is blocked on the experiments finishing.** Every item in §7 can be
  built against an empty tree; the first promotion is what creates the first
  slot. The implementation and the remaining campaigns can run in parallel.

---

## 1. What exists today — verified, not assumed

### 1.1 Layout

```
s3://adrian-nba-model-registry-eu-west-1/models/
  <family>/
    production/   <name>.json + <name>.meta.json
    staging/
    archive/      mix of loose bundles and <UTC-timestamp>/ dirs
```

Ten families, flat. Six live, all last promoted 2026-06-13:

| family | bundle | meta size |
|---|---|---|
| `line_error_full_dataset` | `all_seasons_xgb_line_error_10_06_26` | 88 KB |
| `line_error_last_5_seasons` | `five_seasons_xgb_line_error_10_06_26` | 149 KB |
| `line_error_last_3_seasons` | `three_seasons_xgb_line_error_10_06_26` | 149 KB |
| `total_points_full_dataset` | `full_xgb_total_points_10_06_26` | 87 KB |
| `total_points_last_5_seasons` | `five_seasons_xgb_total_points_10_06_26` | 149 KB |
| `total_points_last_3_seasons` | `three_seasons_xgb_total_points_10_06_26` | 149 KB |

Four dead: `xgboost_model_full_dataset`, `xgboost_model_recent_games`,
`xgboost_full_dataset_total_points`, `xgboost_recent_games_total_points`. All
`.joblib` from March 2026. A grep over `*.py`, `*.ini`, `*.yml`, `*.yaml`
returns **zero** references; they are also unloadable by the serving path, which
calls `XGBRegressor.load_model` on JSON bytes
([prediction.py:660-661](src/nba_ou/prediction/prediction.py#L660-L661)).

Archive across the six live families: **348 bundles, 82.3 MB of models and
42.7 MB of metadata**. Essentially all of that 42.7 MB is the same 1,366-entry
`feature_names` list written 348 times.

### 1.2 The axes already in play, and where each one lives

| axis | values | where it is recorded today |
|---|---|---|
| target | line_error, total_points, spread_error | folder name; re-derived by substring match |
| schema version | 2_0 … 2_5 | `TRAINING_DATA_SCHEMA_VERSION`; reaches artifacts only via CSV filenames |
| dataset build | one CSV per rebuild | `data.data_version` label + `expected_checksum` |
| dataset type | closing_line, intermediate_line | experiment YAML; **not** in the bundle |
| horizon | 0 … 1080 min before tip | experiment YAML `snapshot_minutes`; **not** in the bundle |
| training window | full / 3 / 5 seasons | folder name *and*, separately, `training_metrics.train_games` |

Only the first and last are in S3 at all, and the last is in two places that are
not checked against each other.

### 1.3 The refit is a chain, not a rebase

`retrain_prediction_models.py` loads the **current production metadata** and
derives the next model from it
([retraining_utils.py:455-492](src/nba_ou/modeling/retraining_utils.py#L455-L492)):
`train_games`, `nan_threshold`, `max_na_per_row` and `feature_names` all come
from yesterday's bundle, and the new name is yesterday's name with the date
regex-stripped and re-appended
([retraining_utils.py:717-723](src/nba_ou/modeling/retraining_utils.py#L717-L723)).

There is no fixed reference anywhere in the loop. Every refit inherits from the
refit before it, so any value that was once wrong stays wrong indefinitely, and
"what is this model supposed to be" has no answer other than "whatever the last
one was". This is the single most important thing the plan changes.

### 1.4 The two promotion paths never meet

- [`training_pipeline/promote.py`](training_pipeline/promote.py) already does
  the hard part: load a run's Optuna-selected hyperparameters, refit them on
  fresher data, save a bundle. It writes to **local disk**
  (`models/<family>/<window_dir_label>/`, via
  [naming.py:25-30](training_pipeline/naming.py#L25-L30)) and has no idea S3
  exists.
- [`model_registry.promote_prediction_models`](src/nba_ou/modeling/model_registry.py#L129)
  moves S3 objects between `staging/`, `production/` and `archive/`, and has no
  idea where they came from.

Nothing bridges them. Getting an experiment into production is a manual copy
today. Confirmed: the only code that uploads a model to S3 is
[`save_retrained_bundle_to_staging`](src/nba_ou/modeling/retraining_utils.py#L756).

### 1.5 Defects to delete rather than carry forward

Because nothing in the bucket is being preserved, each of these is removed
outright rather than deprecated:

- **Silent target misidentification.**
  [`_infer_prediction_target_from_metadata`](src/nba_ou/prediction/prediction.py#L272-L287)
  checks for `"total_points"`, then `"line_error" | "error_line" |
  "diff_from_line"`, and **returns `PRED_LINE_ERROR` for anything else**. A
  spread bundle falls through to that default and is served as a line-error
  model with no warning.
- **Spread has no serving path at all.** `PredictionTarget` is only
  `PRED_LINE_ERROR` / `TOTAL_POINTS`
  ([prediction.py:25-26](src/nba_ou/prediction/prediction.py#L25-L26)), and the
  predictions table carries `pred_line_error`, `pred_total_points`,
  `total_over_under_line`, `total_bet365_line_at_prediction` — all totals
  ([create_ou_predictions_db.py:270-310](src/nba_ou/postgre_db/predictions/create/create_ou_predictions_db.py#L270-L310)).
- **Two incompatible discovery contracts for one layout.**
  `get_latest_model_bundle_from_prefix` sorts candidates by S3 `LastModified`
  ([s3_models.py:115](src/nba_ou/utils/s3_models.py#L115)); `extract_single_bundle`
  raises unless the prefix holds *exactly* two objects
  ([model_registry.py:45-78](src/nba_ou/modeling/model_registry.py#L45-L78)).
  `LastModified` is set by the copy, so restoring an archived bundle reshuffles
  which one is "latest".
- **Model and metadata are two independent writes.** `save_retrained_bundle_to_staging`
  clears the prefix, then uploads the model, then uploads the metadata
  ([retraining_utils.py:783-803](src/nba_ou/modeling/retraining_utils.py#L783-L803)).
  A reader between those calls sees an empty prefix, or a model with no
  metadata. §2.2 removes the possibility rather than narrowing the window.
- **Archive layout is inconsistent in every family.** The first archived bundle
  lands loose, later ones get a timestamp directory
  ([model_registry.py:169-172](src/nba_ou/modeling/model_registry.py#L169-L172)).
  Live counts: 49–65 timestamp dirs plus 2–3 loose objects per family.
- **Four naming vocabularies for three windows.** Folders say
  `full_dataset` / `last_3_seasons` / `last_5_seasons`; filenames say
  `all_seasons` / `full` / `three_seasons` / `five_seasons`. All six retire
  together with the season-window axis.

---

## 2. The design

### 2.1 Layout: immutable artifacts, mutable pointers

```
models/{schema_version}/{target}/{horizon}/{variant}/
    specs/{spec_id}.json              # immutable
    builds/{fit_id}/model.json        # immutable
    builds/{fit_id}/fit.json          # immutable
    channels/config.json              # -> spec_id the NEXT refit must use
    channels/staging.json             # -> fit_id awaiting promotion
    channels/production.json          # -> fit_id currently serving
    channels/history/{UTC-ts}.json    # one object per production flip
```

Concretely:

```
models/2_5/line_error/t0000/main/channels/production.json
models/2_5/line_error/t0060/main/channels/production.json
models/2_5/spread_error/t0360/main/channels/production.json
models/2_5/total_points/t0720/main/channels/production.json
```

Nothing that has been written is ever rewritten. A spec is content-addressed by
`spec_id`; a build is a directory named by `fit_id` holding exactly the two
files that belong together. The only mutable objects in the tree are three
small JSON pointers, and each is a single `PutObject` — so every state change is
atomic and every reader sees a coherent pair.

**`{schema_version}`** — `TRAINING_DATA_SCHEMA_VERSION` from
[dataset_versions.py](src/nba_ou/config/dataset_versions.py), currently `"2_5"`,
used verbatim so the registry and the CSV filenames say the same string. What it
does and does not promise is in §2.5 — it is a contract family, not a guarantee
of comparability, and the plan previously overstated it.

**`{target}`** — `line_error | total_points | spread_error`, exactly
`TargetFamily` ([config.py:37-46](training_pipeline/config.py#L37-L46)).

**`{horizon}`** — `t` + minutes-before-tip, zero-padded to four digits. `t0000`
is the close, `t0060` is T-60, `t1080` is T-18h. Padding makes `aws s3 ls` of a
target read in chronological order. The number is `data.snapshot_minutes`
verbatim. **`tpool` is reserved** for a model trained across every snapshot at
once, which
[slice_intermediate_snapshot.py](scripts/create_train_data/slice_intermediate_snapshot.py)
argues is the model you actually bet with; the current campaigns are all
single-snapshot, but the layout should not have to change if that reverses.

**`{variant}`** — `main` everywhere for now. The season-window axis is not being
carried over, so nothing uses it at launch. It stays because it is the only
thing that lets a candidate configuration run against the incumbent at the same
target and horizon: same slot, second variant, both serving, distinct
`model_name`s in the predictions table. One path segment now; a re-promotion of
every slot later.

### 2.2 Why pointers rather than moving objects

The earlier draft had `spec/current.json` plus `production/` and `staging/`
directories whose contents were copied around. Three problems, all removed by
the layout above:

1. **A configuration promotion took production down.** Writing a new
   `current.json` while production still held the previous fit made the two
   disagree, and serving was specified to refuse on exactly that disagreement.
   The rare, deliberate act of promoting a new configuration would have caused an
   immediate outage. Now `channels/config.json` says what the *next refit* must
   use and nothing else reads it; production resolves its spec through the fit
   it is actually serving (§2.3), so changing the configuration cannot affect
   what is being served.
2. **Torn reads.** Model and fit were separate writes to a shared prefix, so a
   reader could catch a new model beside an old fit. A build directory is
   written once, in full, before any pointer names it.
3. **Promotion was a copy of megabytes; rollback was undefined.** Promotion is
   now one `PutObject` of a few hundred bytes, and rollback is the same
   operation naming an earlier `fit_id`. The `archive/` concept disappears
   entirely — old builds simply stay in `builds/` and are never referenced.

### 2.3 Resolution order

Serving, for each enabled slot:

```
channels/production.json   -> fit_id
builds/{fit_id}/fit.json   -> spec_id, model_name, train dates
specs/{spec_id}.json       -> features, cleaning, identity, provenance
builds/{fit_id}/model.json -> the booster
```

The spec a served model is interpreted by is **the one its own fit names**,
never `channels/config.json`. Both spec and build are immutable, so the four
reads cannot disagree, and both are safe to cache by id in
`SETTINGS.s3_local_model_cache_dir`.

Refit, for each enabled slot:

```
channels/config.json -> spec_id -> specs/{spec_id}.json -> fit -> builds/{new fit_id}/ -> channels/staging.json
```

Build promotion:

```
assert staging fit.spec_id == config.json.spec_id    # the staged build is current
PUT channels/production.json {fit_id}                # atomic flip
PUT channels/history/{ts}.json {from, to, spec_id, actor}
```

A failed refit leaves `config.json` pointing at the new spec with no staged
build for it; production keeps serving its own build under its own spec, and the
next day's refit retries. No state in between is broken.

### 2.4 What is in each file

**`specs/{spec_id}.json`** — written by a configuration promotion, read by
refits and by serving. `spec_id` is the first 12 hex of the sha256 of the
canonical body with `provenance` excluded, so re-promoting an identical
configuration is a no-op that produces the same id.

```jsonc
{
  "spec_id": "a3f9c21d4b0e",
  "identity": {
    "schema_version": "2_5",
    "target": "line_error",
    "prediction_strategy": "line_error_regressor",
    "horizon_minutes": 60,
    "variant": "main",
    "dataset_type": "intermediate_line"
  },
  "features": { "feature_names": ["..."], "n_features": 1366 },
  "cleaning": { "nan_threshold": 5.0, "max_na_per_row": 500,
                "corr_threshold": 0.95, "corr_threshold_overrides": {"ODDS_": 0.99} },
  "training": {
    "params": { "max_depth": 4, "learning_rate": 0.0334, "...": "..." },
    "n_estimators": 82,
    "sample_weight_lambda": 0.008536,
    "train_games": 3500,
    "refit_strategy": "rolling_window",
    "required_line_col": "ODDS_CLOSING_TOTAL_LINE_bet365",
    "minimum_line_value": 100.0
  },
  "provenance": {
    "experiment_name": "early25_t60_line_error_window",
    "source_run": "artifacts/experiments/early_line_error_window_2_5_2026_09/...",
    "source_dataset_build_id": "intermediate_line_data_2_5_20260613",
    "source_dataset_checksum": "sha256:5bda016d4c872e0d",
    "vouching_metrics": { "cv_mae": 12.92, "holdout_ou_acc": 0.545 },
    "promoted_at": "2026-09-21T14:02:11Z",
    "promoted_by": "panchojasen"
  }
}
```

**`builds/{fit_id}/fit.json`** — written once per refit.
`fit_id = {UTC timestamp}-{spec_id[:8]}`, e.g. `20260921T154203Z-a3f9c21d`:
sortable, unique, and it names its spec on sight.

```jsonc
{
  "fit_id": "20260921T154203Z-a3f9c21d",
  "spec_id": "a3f9c21d4b0e",
  "model_name": "line_error_t0060_main_2_5_21_09_26",
  "train_date_min": "2021-03-02T00:00:00",
  "train_date_max": "2026-09-21T00:00:00",
  "n_train_games": 3500,
  "n_features": 1366,
  "dataset_build_id": "intermediate_line_data_2_5_20260921",
  "dataset_checksum": "sha256:9c1e77b0a4d3f215",
  "fitted_at": "2026-09-21T15:42:03Z",
  "fitted_by": "github-actions"
}
```

Storing `feature_names` once per spec rather than once per fit is what takes
archive metadata from 42.7 MB to a few hundred KB; a `fit.json` is under 1 KB.

**Retention.** Delete `builds/` directories that no channel references, that no
`history/` entry from the last N days names, and that are not among the most
recent M. History entries are kept even when their bytes are gone — the record
of what served when is smaller and more useful than the booster.

### 2.5 Three identifiers, not one

The plan previously said that sharing a path segment meant comparable columns.
That is false, and [dataset_versions.py](src/nba_ou/config/dataset_versions.py)
says so itself: under the unchanged `2_5` label, intermediate builds added the
`ODDS_SNAP_TOT_<ANCHOR>_*` columns and redefined `DEVIATION_FROM_CONSENSUS` and
friends to measure against a leave-one-book-out peer median, with the docstring
stating that *"intermediate 2_5 files built before and after this change are not
directly comparable"*. Columns were added and semantics changed without a bump,
which is also contrary to the module's own stated policy.

So three names, with three different jobs:

| identifier | example | meaning |
|---|---|---|
| `schema_version` | `2_5` | contract *family*. The path segment. Different values are definitely incomparable; the same value is **not** a guarantee. |
| `dataset_build_id` | `intermediate_line_data_2_5_20260921` | which concrete generation of the CSV |
| `dataset_checksum` | `sha256:9c1e…` | exact identity of the bytes. The only guarantee. |

The spec records the build and checksum it was *tuned* on; each fit records the
build and checksum it was *trained* on. When a silent semantic change lands
under an unchanged `schema_version`, those two fields are what make it visible
after the fact — the feature-name list will still resolve, so nothing else
would notice.

Deliberately **not** named `data_version`: `ExperimentConfig.data.data_version`
already exists and means something else — its own docstring calls it a "human
label for the dataset snapshot… purely descriptive; the checksum below is what
actually identifies the bytes"
([config.py:468-470](training_pipeline/config.py#L468-L470)). Reusing the name
for the schema version would collide with a field whose values look like
`"20260918-intermediate-line-2.5-t180"`.

A consequence worth adopting separately: if `schema_version` is to mean what the
path implies, it must be bumped on **semantic** changes too, not only on
columns gained or lost. That is a change to the discipline around
`TRAINING_DATA_SCHEMA_VERSION`, not to this plan, but the registry is what will
make the cost of not doing it visible.

### 2.6 Nothing is pre-created

S3 has no directories. A prefix exists because an object has that prefix, and
`models/2_5/line_error/t0060/main/` comes into existence when the first
promotion writes a spec under it.

So the horizon grid is not a decision to take up front — it is the accumulated
record of what you have chosen to promote. One caveat, from §5: the promotable
horizons are bounded by the snapshot grid frozen into the daily dataset build,
so `promote-config` must reject a horizon that the daily build does not sample.

### 2.7 Naming convention for `model_name`

`{target}_t{horizon:04d}_{variant}_{schema_version}_{DD_MM_YY}` —
e.g. `line_error_t0060_main_2_5_21_09_26`. Version-then-date, matching the CSV
convention; the date is the trailing eight characters, so it parses
unambiguously.

This matters beyond tidiness: the predictions table upserts on
`(game_id, model_name, prediction_datetime, prediction_value_type)`, so two
horizons refit to the same date **must** produce different names or they
collide. The horizon in the name is load-bearing.

---

## 3. Retiring what is there

Nothing in the bucket will serve again, so this is a sweep, not a migration.

**Move all ten families to `models/retired/{YYYYMMDD}/`** in one pass —
server-side copy then delete, the operation `move_bundle` already performs
([model_registry.py:94](src/nba_ou/modeling/model_registry.py#L94)). Roughly 700
objects. Afterwards `models/` contains `retired/` and `2_5/`, and the live tree
is the one without a date in its name.

**Why retire rather than reshape.** Converting the six bundles to spec/build
form means assigning each a `schema_version`, and that information does not
exist: their metadata carries `training_code_tag: "1.0"` and no schema field,
and they predate `TRAINING_DATA_SCHEMA_VERSION`. Any value written there would
be a guess, in the one field whose job is to say whether two models are
comparable. Since nothing will read them, they keep their original shape under a
prefix whose name says they are out of service.

The 348 archived bundles travel with them. Deleting outright is also defensible
— nothing reads it and it is 82 MB — but moving now and deleting later is
reversible by default.

### 3.1 The `train_data/` prefix gets the same treatment

The bucket's other prefix holds six generated datasets, flat, named
`historical_training_data_until_{YYYYMMDD}.parquet`. Its only writer is
[create_and_upload_historical_train_data.py](scripts/create_train_data/create_and_upload_historical_train_data.py),
and **nothing in the repository reads it** — the training pipeline reads local
`data/train_data/` CSVs instead.

The inconsistency worth fixing: that script and
[create_train_data.py](scripts/create_train_data/create_train_data.py) call the
same `create_df_to_predict`, but only the local one writes
`TRAINING_DATA_SCHEMA_VERSION` into the filename. The same frame was recorded
two different ways depending on where it landed.

So the prefix gains the version as its first segment, as `models/` does:

```
train_data/1_0/historical_training_data_until_20260215.parquet   <- the six existing
train_data/2_5/historical_training_data_2_5_20260215.parquet     <- from now on
```

`1_0` is a label, not a recovered value — those builds predate
`TRAINING_DATA_SCHEMA_VERSION`, so the same reasoning applies as for the retired
models. New filenames put the version immediately before an 8-digit date, so
`training_pipeline.registry.parse_schema_version` parses them exactly as it
parses the local CSVs.

The sweep moves **only objects sitting loose at the root of the prefix**. That
is narrower than "anything not named like a version", deliberately: the
injury-report storage helper offers `train_data/injury_reports` as a destination
([storage.py:80](src/nba_ou/fetch_data/injury_reports/archive/storage.py#L80)),
and a version-name rule would have swept it on whichever machine had run that
import. Re-running is therefore a no-op.

Local `data/train_data/` is **left flat**: 328 files reference those paths, most
of them experiment configs recording what past campaigns trained on, and half
the filenames already carry their schema version.

---

## 4. Changes in `nba_ou` (serving and registry)

### 4.1 New module: `src/nba_ou/modeling/registry_paths.py`

The one place that knows the layout.

```python
@dataclass(frozen=True)
class ModelSlot:
    schema_version: str       # "2_5"
    target: str               # "line_error" | "total_points" | "spread_error"
    horizon_minutes: int      # 0 == close; None-equivalent sentinel for "tpool"
    variant: str = "main"

    def spec_key(self, spec_id: str) -> str: ...
    def build_key(self, fit_id: str, name: str) -> str: ...
    def channel_key(self, channel: Channel) -> str: ...
    @classmethod
    def parse(cls, prefix: str) -> "ModelSlot": ...
```

`derive_staging_prefix` / `derive_archive_prefix`
([model_registry.py:21-35](src/nba_ou/modeling/model_registry.py#L21-L35)) are
deleted along with the string surgery on `/production/`.

### 4.2 Config

Replace the six hand-written prefixes at
[config.ini:57-66](src/nba_ou/config.ini#L57-L66) with a table that carries the
schema version **per row**. A single global version cannot express a gradual
rollout — the whole point of putting the version in the path is that `2_6` slots
can appear one at a time while `2_5` slots keep serving, and a global would
force a flag day. It also could not express deliberately promoting an older
build (§6.2).

```ini
[PredictionModels]
; Default for CLI tools only; never an implicit part of an enabled slot.
DEFAULT_SCHEMA_VERSION = 2_5
; schema_version | target | horizon_minutes | variant  -- one row per promoted slot
ENABLED_MODELS =
```

`SETTINGS.prediction_model_slots` returns `list[ModelSlot]`, each fully
specified. The old `prediction_model_prefixes` property and its callers are
removed in the same change.

An empty table is a valid state and the daily jobs must treat it as "nothing to
do" rather than raising, so the system is runnable before the first promotion.
Today all three entry points raise
([retrain_prediction_models.py:75-80](scripts/retrain_prediction_models.py#L75-L80),
[predict_nba_games.py:103-108](scripts/predict_nba_games.py#L103-L108),
[model_registry.py:130-133](src/nba_ou/modeling/model_registry.py#L130-L133)).

### 4.3 Target handling

- Add `PREDICTION_TARGET_SPREAD_ERROR: PredictionTarget = "PRED_SPREAD_ERROR"`.
- `_infer_prediction_target_from_metadata` is **deleted**. The target is read
  from `spec.identity.target` and validated against `TargetFamily`; an unknown
  value raises. No substring matching survives anywhere.
- `load_and_predict_model_for_nba_games` gains a spread branch: the predicted
  value is the residual against the anchor spread
  (`ODDS_CLOSING_SPREAD_LINE_HOME_bet365`), and `PRED_PICK` is home/away rather
  than over/under.

### 4.4 Horizon guard at serving time

`time_to_match_minutes` in the predictions table is when the prediction *ran*.
Nothing records what the model was *trained for*. Without both you cannot notice
that the T-720 model served every game at T-60 for a month.

At predict time, compare `spec.identity.horizon_minutes` against the row's
actual minutes-to-tip and skip (or warn, configurably) outside a band — ±45 min
is a reasonable default for an hourly grid, with `t0000` treated as "at or after
the last snapshot" and `tpool` exempt.

### 4.5 Predictions table

New columns, all nullable, added once:

| column | why |
|---|---|
| `model_target` | replaces inferring it from `model_name` |
| `model_horizon_minutes` | the slot; distinct from `time_to_match_minutes` |
| `model_schema_version` | which contract family produced its features |
| `model_variant` | A/B identification |
| `spec_id` | joins a prediction to the exact configuration |
| `fit_id` | joins it to the exact booster |
| `pred_spread_error` | the third target's value |
| `spread_line_at_prediction` | its anchor line, mirroring the totals column |

`model_horizon_minutes` alongside `time_to_match_minutes` is what makes the
post-hoc question answerable: *did the T-360 model actually beat the T-0 model
when both were run at T-360?*

### 4.6 Build promotion

`promote_build` replaces `promote_prediction_models`: validate that the staged
build's `spec_id` equals `channels/config.json`'s, run the smoke test against
the staged build (load it, score one day, compare its prediction spread against
the incumbent's), then flip `channels/production.json` and write the history
entry. Plus `--rollback [fit_id]`, which is the same flip and needs no new
machinery.

---

## 5. The daily dataset build — its own problem

`df = prepared_frame_for(spec)` hid the largest unbuilt piece of this plan, so
it gets a section and a work item of its own, ahead of `retrain_from_spec`.

**Today** the retrain job calls `create_df_to_predict` once, in-process, and
hands the same closing-line frame to every configured model
([retrain_prediction_models.py:83](scripts/retrain_prediction_models.py#L83)).
That works because all six models are closing-line models over one frame.

**Hourly horizons need the intermediate dataset**, which is a different thing in
every respect:

- It is built by a separate entry point with no shared state,
  [create_intermediate_line_train_data.py](scripts/create_train_data/create_intermediate_line_train_data.py),
  writing its own CSV plus a `_scoring.csv` companion.
- The current one, `intermediate_line_data_2_5_20260613.csv`, is **4.4 GB**.
- **The snapshot grid is frozen at build time.** `--snapshot-grid` defaults to
  `(0, 30, 60, 120, 180, 240, 300, 360, 480, 720)`, and
  [snapshots.py:63-67](src/nba_ou/data_processing/line_history/snapshots.py#L63-L67)
  states that removing a horizon is a filter but *adding one back means
  regenerating the whole dataset*. The current campaigns train at 420, 540, 660,
  840, 960 and 1080 — none of which are in that default — so the file in use was
  built with a custom grid.

Decisions this forces, none of which the rest of the plan can make for it:

1. **Full rebuild or incremental append?** A nightly 4.4 GB rebuild is a
   different operational proposition from appending the day's games.
2. **Where does it live, and is it versioned?** Local `data/train_data/` will
   not hold many 4.4 GB generations; S3 with a `dataset_build_id` is the obvious
   alternative, and it is what lets a fit record which build it used.
3. **How does a spec choose its build?** By `dataset_type` plus
   `schema_version`, resolving to the newest compatible build — with the
   checksum recorded in `fit.json` either way.
4. **What guards against mixing semantics?** The `DEVIATION_FROM_CONSENSUS`
   redefinition (§2.5) landed under an unchanged `2_5`, so "same schema_version"
   is not enough. Comparing a build's checksum against the spec's
   `source_dataset_checksum` and warning on a change is the cheap version.
5. **What is the promotable-horizon contract?** The daily build's grid bounds
   which slots can exist. `promote-config` must reject a horizon the daily build
   does not sample, or the slot silently never refits.
6. **What happens when the build is missing or stale?** Serve yesterday's
   model — which is correct and should be explicit — and alert, rather than
   failing the whole job.

Until these are settled, the horizon slots cannot actually be refit daily, which
makes this the critical path for everything except the first promotion.

---

## 6. Changes in `training_pipeline` (configuration promotion)

### 6.1 Naming the two promotions apart

- **`promote-config`** — rare, manual, human judgement. An experiment run →
  `specs/{spec_id}.json` + `channels/config.json` + a first build in staging.
  **This is what creates a slot.**
- **`promote-build`** — daily, unattended. Flip `channels/production.json`.

`training_pipeline/promote.py` becomes the former (keeping
`python -m training_pipeline.promote` working);
`scripts/promote_prediction_models.py` becomes `scripts/promote_build.py`.

### 6.2 `training_pipeline/promote.py`

It already loads a run's config, recovers the Optuna-selected trial, refuses
diagnostic and classifier runs, refits and saves a bundle. What it needs:

- **Emit a spec.** `build_spec_from_run(run_dir, slot) -> ModelSpec`, lifting
  identity from the run's own config — `prediction_strategy` → `target`,
  `data.snapshot_minutes` → `horizon_minutes`, `data.dataset_type`,
  `schema_version` parsed from `data.csv_path` — and the training block from
  `RunHyperparameters` ([reuse.py:22-46](training_pipeline/reuse.py#L22-L46)).
- **Take `schema_version` from the data, not from HEAD.** A run trained on
  `training_data_2_3_20260909.csv` is a 2_3 model even when
  `TRAINING_DATA_SCHEMA_VERSION` says `2_5`. Parse it from the filename and warn
  — do not fail — on disagreement; promoting an older build deliberately is
  legitimate, and §4.2's per-row version is what lets the result be enabled
  alongside 2_5 slots.
- **Record `source_dataset_build_id` and `source_dataset_checksum`** from the
  run, per §2.5.
- **Reject horizons the daily build does not sample** (§5, item 5).
- **Infer the slot, then let it be overridden.** `--slot 2_5/line_error/t0060/main`
  is available; the default comes from the run's config, so the usual case is
  `python -m training_pipeline.promote <run_dir> --to-s3`. A mismatch between
  inferred and given is an error, not a silent override.
- **Write in dependency order**: spec, then build, then `config.json`, then
  `staging.json`. Nothing points at an object that is not there yet.
- **Print the `ENABLED_MODELS` row** to add to config.ini.
- **Refuse to replace a slot's config without `--replace-config`.**
- **Keep the diagnostic and classifier guards**
  ([promote.py:103-155](training_pipeline/promote.py#L103-L155)). They matter
  more, not less, when the destination is S3.

### 6.3 A campaign-level promotion helper

`scripts/promote_campaign_to_registry.py`:

```
python -m scripts.promote_campaign_to_registry \
    artifacts/experiments/early_line_error_window_2_5_2026_09 \
    --select best-roi --dry-run
```

Maps each campaign cell to the slot its config implies, picks the winning run
per cell, prints the full table (slot → run → CV/holdout → params), writes
nothing without `--execute`. `find_best_run_hyperparameters`
([reuse.py](training_pipeline/reuse.py)) already does the per-run selection.
This is where the horizon grid actually gets decided — by which cells you
select, bounded by §5.

### 6.4 Feature-list consistency across horizons

Check rather than assume during the first campaign promotion: whether every
horizon of a target resolves to the same feature list after correlation pruning.
If it does, `features` could eventually be hoisted. If it does not — likely,
since pruning is data-dependent and the intermediate frames differ by snapshot —
per-slot storage is correct and this stays as written.

---

## 7. Work items

Items 1–9 build against an empty tree. Item 10 is the critical path for daily
horizon refits; item 13 validates the whole chain.

1. **`registry_paths.py`** — *DONE*. `ModelSlot`, key builders, `parse()`, round-trip
   tests. *Small.*
2. **`ModelSpec` / `FitRecord` / channel models** — *DONE*, plus `spec_id` and
   `fit_id` derivation. Landed in its own
   [registry_models.py](src/nba_ou/modeling/registry_models.py) rather than
   beside `ModelBundleMetadata` in `modeling.py`, which imports xgboost,
   sklearn and tqdm: reading a pointer should not require a training stack.
   *Small.*
3. **Registry I/O** — *DONE*. `read_spec`, `write_spec`, `read_build`, `write_build`,
   `read_channel`, `set_channel`, plus the resolution order in §2.3 as one
   function with its own test. *Medium.*
4. **Config table + empty-state handling** — *DONE*. Per-row `schema_version`,
   `prediction_model_slots`, remove `prediction_model_prefixes`, make all three
   entry points no-op on an empty table. *Small.*
5. **`build_spec_from_run`** — *DONE*, with a test per target family over real campaign
   configs, including the `schema_version` parse and the build/checksum capture.
   *Medium.*
6. **`promote.py --to-s3`** — *DONE*. Slot inference, write ordering, `--replace-config`.
   *Medium.*
7. **`promote_build`** — *DONE*. Spec agreement check, smoke test, atomic flip, history,
   `--rollback`. *Small.*
8. **Serving** — *DONE*. Spread end to end, target from spec, horizon guard, resolution
   through `channels/production.json`. *Medium.*
9. **Predictions-table migration** — *DONE (code); the ALTER still needs
   running*. The eight columns in §4.5, plus widening the
   `prediction_value_type`, target-present and `pred_pick` CHECK constraints,
   which would otherwise reject every spread row. Run
   `python -m nba_ou.postgre_db.predictions.create.create_ou_predictions_db
   --add-columns`. *Small, but it is a DB write you will want
   to run yourself.*
10. **Daily dataset build** — *NOT DONE; the one real gap*. Settle §5 and implement it: rebuild strategy,
    storage, build selection, grid contract, missing-build behaviour. *Large;
    everything about daily horizon refits waits on it.*
11. **Refit from spec** — *DONE for closing_line; intermediate waits on 10*.
    `refit_from_spec` replacing
    `build_retraining_settings_from_artifacts`, with the game-ID-counted window
    ([promote.py:191-198](training_pipeline/promote.py#L191-L198)). Depends on
    10. *Medium.*
12. **Retirement sweep** — *DONE, 2026-09-21*. All 751 objects of the ten
    families moved to `models/retired/20260921/`; copies verified before any
    delete, byte sizes unchanged, and a second dry run finds nothing (it does
    not nest `retired/` inside itself). `models/` now holds only `retired/`.
    *Small.*
13. **First real promotion** — *WAITING on the experiments*. One slot end to
    end once they settle:
    `promote-config` → smoke test → `promote-build` → one day of predictions →
    check the row. Do this for a single slot before the campaign helper touches a
    dozen. *Small, and it is what validates items 1–9.*

---

## 8. Open questions

**8.1 Do the retired bundles need the new shape?**
§3 argues no, on the grounds that assigning them a `schema_version` would be a
fabrication. Overrule if you want the bucket uniform anyway.

**8.2 Should `TRAINING_DATA_SCHEMA_VERSION` bump on semantic changes?**
§2.5 shows it currently does not, and that the module's own docstring records an
incomparability under an unchanged label. The registry works either way — the
checksum carries exact identity — but the path segment means less than it looks
like it means until this is decided.

**8.3 One refit job or one per horizon?**
A dozen horizons × three targets is a different runtime profile from today's
six. If they share one prepared frame per `(dataset_type, build)` the marginal
cost per slot is one `fit`. Worth measuring on the first two slots (item 13),
and it partly depends on how §5 resolves.

**8.4 Does `t0000` mean the closing dataset or the intermediate T-0 slice?**
Both are legitimate and they are different models. Whichever you promote first
takes `variant: main`; the other is a second variant. The spec records
`dataset_type` either way, so this need not be settled now — only noticed, so
that two different things do not land in one slot on different days.
