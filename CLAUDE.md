# NBA Over/Under Predictor

## Project overview

This repo predicts NBA betting markets. It trains XGBoost regressors, stores
them in an S3 model registry and serves predictions daily via GitHub Actions
(local runner) and a Streamlit dashboard. Three targets:

1. **`TOTAL_POINTS`** — the actual combined score of a game.
2. **`LINE_ERROR`** (a.k.a. `DIFF_FROM_LINE` / "error_line" in older code) —
   the bookmaker's totals error: `TOTAL_POINTS - ODDS_TOTAL_LINE_<book>`.
   Positive means the game went OVER the total.
3. **`SPREAD_ERROR`** — the spread analogue:
   `HOME_MARGIN - ODDS_SPREAD_LINE_HOME_<book>`.

Each target is modelled at the **closing** line (horizon 0) and at
**intermediate** snapshots (30–1080 minutes before tip-off), one model per
`(schema_version, target, horizon, variant)` slot.

The market is close to efficient and the edge being chased is small relative
to measurement noise. Treat any "A beats B" claim with suspicion until it has
survived the evaluation discipline in the `experiments` skill.

## Repo map

- `src/nba_ou/` — installable package (`from nba_ou...`): data, features,
  registry and serving.
  - `create_training_data/` — dataset builders. `create_df_to_predict()`
    (closing line, one row per game) and `create_intermediate_line_df()`
    (one row per game × snapshot) are the canonical entry points, used for
    both historical training data and tonight's prediction rows.
    `schema_layers/` holds the layered schema versions (see below).
  - `data_processing/` — feature-engineering submodules: `team/`, `players/`,
    `odds/`, `line_history/`, `referees/`, `travel/`, `past_injuries/`,
    `injury_status/`, `lineups/`, `player_graph/`, `all_star_voting/`,
    `historic_games/`, `merged_home_away_data/` (incl.
    `select_train_columns.py`), `statistics/`, `missing_data/`,
    `scheduled_games/`.
  - `modeling/` — model-side library code:
    - `modeling.py` — CV splitters, recency sample weighting,
      `ModelBundleMetadata`.
    - `optuna_total_points.py` / `optuna_error_line.py` — XGBoost search
      spaces and objectives; called through `training_pipeline/tuning.py`,
      not directly.
    - `scorers.py` — betting-accuracy metrics.
    - `registry_paths.py` / `registry_models.py` / `registry_store.py` — the
      S3 slot registry (see "Model registry" below).
    - `refit.py` / `retraining_utils.py` — daily refit of a slot from its spec.
    - `build_promotion.py` — flips a slot's production pointer to a build.
    - `meta_learner_training_data.py` — stacked dataset builder. **Not
      migrated to the slot registry**; it only reads old-style prefixes under
      `models/retired/<date>/` and raises otherwise.
  - `prediction/` — serving: `prediction.py` (loads enabled slots and
    predicts), `intermediate_frame.py` (tonight's snapshot rows, every horizon),
    `baseline_predictions.py`, `prediction_tabpfn_client.py`.
  - `config/` — `settings.py` (`SETTINGS`, reads `config.ini` +
    `config.secrets.ini`), `odds_columns.py` (`get_main_book()`,
    `total_line_col()`, `spread_line_home_col()`), `market_columns.py`,
    `leakage.py`, `identity_columns.py`, `dataset_versions.py`.
  - `postgre_db/`, `fetch_data/` — DB layer (Supabase + Aiven) and raw-data
    ingestion. Line history has four Aiven stores with the same tables: SBR
    in `line_history` (`line_history_aiven/`, read by the pipeline),
    OddsLogic/WagerTalk in `line_history_oddslogic` (`line_history_oddslogic/`,
    2026-27 on, no bet365, unused and unmaintained — `docs/odds/oddslogic_line_history.md`),
    MoviOdds in `line_history_moviodds` (`line_history_moviodds/`, bet365
    only, 2026-27 on, unused and unmaintained — `docs/odds/moviodds_line_history.md`),
    and Covers in `line_history_covers` (`line_history_covers/`, SBR's feed
    for the nine US books incl. bet365 plus `consensus_opener` = BetMGM's
    opener, and `lh_consensus` = each game's Covers Consensus pick counts,
    the source of SBR's `*_consensus_pct_*`; 2026-27 on plus backfilled past
    SBR gaps. SBR's per-game fallback in the stored (training) line-history
    reads and in tonight's prediction —
    `docs/odds/covers_line_history.md`). Every tick read, stored or live,
    drops the alternate-line quotes the SBR/Covers feed files as the main
    line, in every season (`line_history_aiven/alt_lines.py`,
    `docs/odds/alternate_line_ticks.md`).
    OddsLogic and MoviOdds automatic updates are disabled as of 2026-10-24:
    too slow and unstable for the value they add. Keep their implementations
    and existing data for possible future reuse; no maintenance or integration
    into training/prediction is planned.
- `training_pipeline/` — second installable package: the experiment framework
  (config, data prep, walk-forward CV, Optuna tuning, holdout, betting
  evaluation, leaderboards, promotion). **Read the `experiments` skill before
  touching it or interpreting a run.**
- `experiments/` — campaign definitions (YAML, deep-merged over
  `experiments/_base.yaml`). Runs land in `artifacts/experiments/`.
- `scripts/` — thin CLI entry points: `create_train_data/` (incl.
  `build_schema_version.py`), `predict_nba_games.py`,
  `retrain_prediction_models.py`, `promote_build.py`,
  `smoke_test_s3_model_load.py`, `preflight_campaign.py`, plus ingestion and
  DB-maintenance folders (`update_databases/`, `line_history/`, `lineups/`,
  `injury_reports/`, `player_graph/`, `backup/`, ...).
- `lab/` — exploration only, not production. `lab/old_train_setups/` holds
  deprecated AutoGluon/TabPFN/classifier attempts (historical reference).
- `models/` — legacy git-tracked artifacts. Do not add to it; models live in
  the S3 registry.
- `data/train_data/` — generated datasets, **Parquet only** for current work:
  `closing_line_data_<schema>_<date>.parquet` and
  `intermediate_line_data_<schema>_<date>.parquet` (+ `_scoring`), each with a
  `.manifest.json` for layered versions. The `training_data_*.csv` files are
  legacy. Campaign configs name the exact file they read.
- `docs/` — topic folders (`pipeline/`, `odds/`, `intermediate/`, `injuries/`,
  `lineups/`, `serving/`); `docs/README.md` indexes every document with its
  status (Reference / Implemented / Unmaintained / Partial / Plan / Research / Superseded).
  `pipeline/training_data_processing.md` is the detailed reference for the
  closing-line feature pipeline; `pipeline/feature_engineering_overview.md` is
  the map of data sources, storage and which module builds each feature family.
  Keep both references in sync when features change. A document keeps its path
  when its status changes: update its `Status:` line and the index instead.
- `.claude/skills/` — `experiments` (repo-specific training and evaluation)
  and four portable skills (`feature-engineering`, `sports-model-pipeline`,
  `odds-data-architecture`, `sports-data-architecture`).
- `tests/` — pytest, grouped by the module under test: `core/`, `datasets/`,
  `features/`, `ingestion/`, `training/`, `prediction/`, `storage/`, and
  `reporting/`. Shared builders and fakes live in `support/`, reference
  implementations in `support/legacy/`. See `tests/README.md` for ownership
  rules and `tests/MIGRATION.md` for the file mapping. Run the full suite with
  `python -m pytest`; CI uses `.github/workflows/tests.yml`. Do not import
  another test module for helpers; use `tests.support`. Resolve repository
  paths through `tests.support.paths.REPO_ROOT`.
- `streamlit/`, `.github/workflows/` — dashboard and the operational
  lifecycle (daily predict, DB updates, retrain → smoke test → promote,
  backups).

## Data pipeline essentials

- Naming conventions that matter for any new feature or model input:
  - `_BEFORE` = leakage-safe, pre-game feature; what final column selection
    in `select_train_columns.py` filters on.
  - `_TEAM_HOME` / `_TEAM_AWAY` = post-merge side suffix; `_DIFF_BEFORE` =
    home minus away.
  - `ODDS_` prefix = every odds-derived column.
  - `ODDS_TOTAL_LINE_<book>` = the totals line; `ODDS_SPREAD_LINE_HOME_<book>`
    = the canonical spread (market-implied home margin). The raw per-book
    spread is a home *handicap* with the opposite sign — use
    `spread_line_home_col()`, never the raw column, as a line.
  - The main book comes from `get_main_book()` (`MAIN_SPORTSBOOK` in
    `config.ini`, currently `bet365`).
- Leakage is enforced in code: `select_training_columns()` raises if a raw,
  non-`_BEFORE` column would leak, and `config/leakage.py` names column
  families that carry `_BEFORE` but still depend on who played tonight. A
  correct `_BEFORE` suffix is not proof of no leakage.
- **Schema versions are layers.** The builders produce the frozen base `2_5`;
  every later version (`2_6`, `2_7`, …) is a `SchemaLayer` in
  `src/nba_ou/create_training_data/schema_layers/` that only *adds* columns to
  its parent, computed from a 2_5 frame or file without rebuilding it
  (`scripts/create_train_data/build_schema_version.py`). New columns go in a new
  layer, never in the base builders; changing an existing column is a new base
  (`3_0`). Inspect a file with `python -m nba_ou.create_training_data.schema_layers
  describe <file>`. Rules and history: `nba_ou.config.dataset_versions`.
- Targets: `TOTAL_POINTS` and `HOME_MARGIN` are in the pipeline output (as
  outcome columns, blocked from feature matrices). `LINE_ERROR` is derived
  downstream by `_ensure_line_error_column()` in
  `meta_learner_training_data.py` — the single canonical derivation, reused by
  `training_pipeline/data.py`. `SPREAD_ERROR` is derived by
  `ensure_spread_error_column()` in `training_pipeline/data.py` (the
  intermediate dataset ships it per snapshot).
- **Naming inconsistency:** prediction-side code (`prediction.py`,
  `prediction_tabpfn_client.py`, `baseline_predictions.py`) calls the totals
  residual `DIFF_FROM_LINE` / `PRED_LINE_ERROR`; modeling-side code calls it
  `LINE_ERROR` / `line_error`. Same quantity.

## Training and evaluation

All new training work goes through `training_pipeline/` driven by YAML
campaigns in `experiments/`; the `experiments` skill describes the run
sequence, the gate, the known silent-no-op failure modes and how to read
results. In short:

- Run: `poetry run python -m training_pipeline.cli <config.yaml>`
  (`--dry-run` prints the resolved config); check a campaign first with
  `scripts/preflight_campaign.py experiments/<campaign>`.
- Time-aware CV and recency weighting come from `nba_ou.modeling.modeling`
  (`make_walk_forward_last_n_seasons_splits`,
  `make_test_anchored_walk_forward_splits`, `split_latest_dates_holdout`,
  `validate_time_splits` / `assert_valid_time_splits`,
  `build_recency_sample_weights`, `TemporalDecaySampleWeightRegressor`).
- Evaluation is betting-aware (directional OVER/UNDER accuracy, min-edge
  variants), not just MAE/RMSE. Rank only after measuring seed noise; CV and
  holdout are separate measurements.
- Run artifacts are written by `training_pipeline/tracking.py` (a copy of
  `lab/utils/experiment_tracking.py`, deliberately not importing `lab/`).
- Changes to shared logic (`modeling.py`, `scorers.py`, `training_pipeline/`)
  must keep parity across all three targets.

## Model registry and serving

- S3 bucket `adrian-nba-model-registry-eu-west-1`, one subtree per slot:
  `models/{schema}/{target}/{horizon}/{variant}/` with immutable `specs/` and
  `builds/{fit_id}/`, plus mutable pointers `channels/{config,staging,
  production}.json`. Promotion and rollback are a single pointer write.
  Design: `docs/serving/model_registry.md`.
- Which slots are live: `ENABLED_MODELS` under `[PredictionModels]` in
  `src/nba_ou/config.ini` (`schema | target | horizon | variant` rows).
- Adopting a new *configuration* (human decision):
  `python -m training_pipeline.promote <run_dir> --to-s3`, which prints the
  `ENABLED_MODELS` row to add.
- Daily *builds*: `scripts/retrain_prediction_models.py` refits each slot from
  its spec (not from yesterday's model) and stages it;
  `smoke_test_s3_model_load.py --channel staging` checks it;
  `scripts/promote_build.py --execute` flips production. Closing and
  one-horizon intermediate slots are refitted; pooled or layered intermediate
  specs keep their build.
- Training rows for the daily refit (and TabPFN) come from the season
  partitions (`create_training_data/season_partitions.py`): each dataset in S3
  one file per season, `train_data/2_5/seasons/<dataset_type>/season=YYYY.parquet`.
  Past seasons stay as stored; the daily job builds only the games since the
  last cutoff, in target mode, and rewrites the season they belong to. It also
  rebuilds the last 3 days and reports any stored row that came back
  different. Manage them with `scripts/create_train_data/build_season_partitions.py`
  (`status`, `split`, `update`, `rebuild`).
- Serving: `scripts/predict_nba_games.py` builds tonight's closing rows and,
  when any enabled slot has a horizon, tonight's intermediate rows for every
  horizon, the closing rows in a spawned side process meanwhile
  (`prediction/side_build.py`); a build that starts computing a past season
  waits for the other (`utils/warm_gate.py`), so only warm builds overlap
  (`PARALLEL_BUILDS`, `--sequential-builds`). Every enabled slot predicts on
  every run, whatever the time to tip: closing slots every game,
  intermediate slots the row of their own horizon, marked `SNAPSHOT_PASSED`
  (False = built from the ticks so far, not the trained snapshot). A game without both teams' injury report is skipped by
  the closing slots and by intermediate slots of horizon
  `REPORT_REQUIRED_MAX_HORIZON` (300) or shorter
  (`prediction/report_requirement.py`). A failing slot does not stop the
  others; the run exits 1 at the end. `--no-persist` predicts without storing.
  The predictor and TabPFN log numbered steps (`utils/run_log.py`): in GitHub
  Actions each step is a collapsible group, warnings and failures are
  annotations, and the timings go to the job summary.
- Scheduling: cron-job.org dispatches the workflows. The two daily times
  (update + retrain, booking of the day's runs) are `[Schedule]` in
  `config.ini`, applied with `scripts/cronjobs/setup_cronjobs.py --apply`. The
  repository variables `DAILY_UPDATE_ENABLED` and `PREDICTIONS_ENABLED`
  (`false` = off) skip the update/refit and the prediction workflows.
  Details: `docs/serving/snapshot_triggers.md`.
- SBR fallback (`prediction/line_source.py`): when SBR's feed is frozen or
  serves no odds, tonight's lines (closing day-page odds and live line history)
  come from Covers, with SBR's exact columns and books (consensus pick % from
  the Covers Consensus page, SBR's own source). A game SBR does not
  serve, or serves without spread or moneyline, is filled from Covers. Each stored input row records its source in
  `prediction.inputs.line_source` (1 SBR, 2 Covers); frames carry
  `LINE_SOURCE`. Only both feeds frozen stops the run.
- Intermediate input mode (`INTERMEDIATE_INPUT_MODE` under
  `[PredictionModels]`, `--intermediate-input` per run): in `latest` (the
  configured default) the slots in `LATEST_INPUT_SLOTS` (default `all`)
  predict each game's latest snapshot, the run's own moment, limited to what
  their training rows carried (`nba_ou.prediction.input_limits`); in `own`
  every slot reads the row of its own horizon. `scripts/predict_past_snapshots.py`
  is the end-of-day `own` run. Details: `docs/intermediate/live_prediction.md`.
- TabPFN runs on its own (`scripts/predict_tabpfn.py`, dispatch-only workflow
  `nba_predictor_daily_tabpfn_client.yml`): one or more targets (`--targets`)
  and one line per run (default `line_error` and `total_points` at the closing line;
  `--line intermediate` follows a fixed horizon, T-360 by default); the targets
  share the database update, tonight's rows and the stored run, and each gets
  its own fit. TabPFN keeps no model, so each run fits it
  with `tabpfn-client` on the rows, features, NaN budget and window of that
  slot's production spec, read from the season partitions, and predicts
  tonight's rows. Stored as `model_builds.kind` 3 / `runs.workflow` 3.

## Adding or changing features

- Follow the "Adding Or Modifying Features" checklist in
  `docs/pipeline/training_data_processing.md` and the `feature-engineering`
  skill.
- New columns go in a new schema layer; check them against `config/leakage.py`
  as well as the `_BEFORE` convention.
- A feature that is cheap to compute historically may differ in the
  tonight-only prediction build (it loads less history). Check training vs
  prediction parity for any expanding/season-level feature.

## Environment / tooling

- Poetry-managed (`pyproject.toml`, packages `nba_ou` and
  `training_pipeline`), Python 3.11–3.13.
- Dependencies are split into Poetry groups: main (core: pandas, pyarrow,
  scikit-learn, psycopg, boto3, nba_api, ...), `cuda` / `cpu` (`xgboost==2.1.4`
  or `xgboost-cpu`, never both; `cpu` is optional), `scrape` (HTTP scrapers,
  PDFs), `browser` (Playwright, manual scrapers only; no job installs it),
  `train` (optuna, shap, yaml), `tabpfn`, `dashboard`, `lab` (autogluon),
  `timeseries`, `dev`.
  `poetry install` gives the full local environment.
- `pyproject.toml` is the single source. Workflows install a per-job file from
  `requirements/`, generated by `scripts/export_requirements.py` and checked in
  CI; after changing dependencies run `poetry lock` and that script
  (`requirements/README.md`). Never edit `requirements/*.txt` by hand.
- `ruff` / `black` / `isort` / `mypy` configured (line-length 88, py311
  target). Tests via `pytest`.
- `config.secrets.ini` holds DB/S3/odds-API credentials — never print or
  quote its contents.
