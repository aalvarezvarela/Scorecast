# Documentation index

Start with [pipeline/training_data_processing.md](pipeline/training_data_processing.md)
(how the closing-line dataset is built) and
[pipeline/feature_engineering_overview.md](pipeline/feature_engineering_overview.md)
(where every data source and feature family lives). Training and evaluation
are described in the `experiments` skill (`.claude/skills/experiments/`), not here.

## Statuses

Every document has one `Status:` line under its title:

| Status | Meaning |
|---|---|
| **Reference** | Describes how the code works today; keep it in sync when the code changes |
| **Implemented** | A design document whose implementation exists; the body may still read as a plan |
| **Unmaintained** | Implemented but unused; automatic updates are disabled and the code is retained for possible future reuse without maintenance |
| **Partial** | Part of the design is implemented; the status line says what remains |
| **Plan** | Not implemented |
| **Research** | Measurements or findings, not a design |
| **Superseded** | Kept as evidence; the status line names what replaced it |

**Maintenance.** Folders are by topic, so a document keeps its path when its
status changes: update the `Status:` line and this index, do not move or
rename the file. Code docstrings link to these paths. A new document goes in
the folder of its topic, with a status line, and gets a row here.

## pipeline/ — dataset builders and training data

| Document | Status | Covers | Main code |
|---|---|---|---|
| [training_data_processing.md](pipeline/training_data_processing.md) | Reference | Closing-line pipeline step by step: rolling windows, players, injuries, referees, travel, column selection, naming | `create_training_data/create_df_to_predict.py` |
| [feature_engineering_overview.md](pipeline/feature_engineering_overview.md) | Reference | Map of data sources, storage and the module behind each feature family | `data_processing/` |
| [yahoo_features.md](pipeline/yahoo_features.md) | Reference | The compact 36-column Yahoo percentage contract | `config/yahoo_features.py` |
| [season_type_mixing.md](pipeline/season_type_mixing.md) | Plan | Preseason/All-Star games in player EWMAs and playoff games in cross-season team windows: measurements, judgement, fixes | `data_processing/players/`, `data_processing/team/rolling.py` |
| [defect_fixes_2_5.md](pipeline/defect_fixes_2_5.md) | Implemented | The 2_5 defects found in October 2026 and how each was fixed | `data_processing/`, `postgre_db/` |
| [exhibition_games.md](pipeline/exhibition_games.md) | Partial | Preseason and All-Star games dropped from every feature input (2_5 in place); rebuild and retrain pending | `utils/seasons.py`, `postgre_db/config/db_loader.py`, `injury_report_aiven/fetch.py` |
| [speedup.md](pipeline/speedup.md) | Partial | Making the 2_5 builds faster without changing a value: caches, target mode, season partitions | `scripts/perf/`, `create_training_data/season_partitions.py` |
| [prediction_input_cache_s3.md](pipeline/prediction_input_cache_s3.md) | Partial | Past seasons of every input table read from S3, the current season from the DB | `postgre_db/season_tables.py`, `scripts/refresh_input_cache.py` |
| [prediction_runtime_plan.md](pipeline/prediction_runtime_plan.md) | Partial | Shorter daily prediction runs (A–E and G done, F deferred): trust frozen 2_5 seasons, latest-only rows, shared loads, model cache, one intermediate injury memo for retrain and predictor | `postgre_db/season_tables.py`, `create_training_data/history_memo.py`, `create_intermediate_line_df.py` |

## odds/ — odds and line-history data

| Document | Status | Covers | Main code |
|---|---|---|---|
| [line_history_updating.md](odds/line_history_updating.md) | Reference | Daily and backfill updates of the Aiven `line_history` store | `scripts/update_databases/update_line_history_database.py` |
| [line_history_phase0_findings.md](odds/line_history_phase0_findings.md) | Research | Tipoff source, timezones, match rates and in-play rows behind the store | `postgre_db/line_history_aiven/` |
| [odds_source_alternatives.md](odds/odds_source_alternatives.md) | Partial | Odds APIs, OddsTrader and Covers as alternatives to SBR; OddsLogic and MoviOdds implemented but unused and unmaintained | `fetch_data/odds_oddstrader/` |
| [oddslogic_line_history.md](odds/oddslogic_line_history.md) | Unmaintained | OddsLogic (WagerTalk) ingestion implemented but unused; automatic updates disabled due to slowness, instability and limited value; retained for possible reuse | `fetch_data/odds_oddslogic/`, `postgre_db/line_history_oddslogic/` |
| [moviodds_line_history.md](odds/moviodds_line_history.md) | Unmaintained | MoviOdds bet365 ingestion implemented but unused; automatic updates disabled due to slowness, instability and limited value; retained for possible reuse | `fetch_data/odds_moviodds/`, `postgre_db/line_history_moviodds/` |
| [covers_line_history.md](odds/covers_line_history.md) | Implemented | Covers line history (SBR's feed, nine US books incl. bet365): endpoints, parsing rules, `consensus_opener` = BetMGM's opener, the `line_history_covers` store (incl. the Covers Consensus pick counts, `lh_consensus`), its daily update, parity with SBR, the past-gap audit and backfill, and its use as SBR's per-game fallback (stored training reads and the live prediction) | `fetch_data/odds_covers/`, `postgre_db/line_history_covers/` |
| [alternate_line_ticks.md](odds/alternate_line_ticks.md) | Implemented | Alternate-line quotes the SBR/Covers feed files as the main line (burst 2026-10-21/22, low level since April 2026): the audit and the reader-side filter for every season | `postgre_db/line_history_aiven/alt_lines.py` |

## intermediate/ — snapshot (pre-close) datasets and serving

| Document | Status | Covers | Main code |
|---|---|---|---|
| [line_dataset.md](intermediate/line_dataset.md) | Implemented | The one-row-per-game-×-snapshot dataset and its features | `create_training_data/create_intermediate_line_df.py` |
| [training_pipeline_integration.md](intermediate/training_pipeline_integration.md) | Implemented | Running snapshot data through `training_pipeline`: game-counted windows, pooled vs single-horizon | `training_pipeline/` |
| [feature_engineering_round2.md](intermediate/feature_engineering_round2.md) | Partial | Round-2 market features ranked by measured evidence | `data_processing/line_history/` |
| [market_dynamics.md](intermediate/market_dynamics.md) | Implemented | G1–G4: injury news, news reaction, cross-market coherence, move-to-close Ridges | `create_training_data/intermediate_market_dynamics.py` |
| [live_prediction.md](intermediate/live_prediction.md) | Implemented | Serving the intermediate slots from tonight's data, without storing a partial game | `prediction/intermediate_frame.py` |

## injuries/ — injury reports, availability and rosters

| Document | Status | Covers | Main code |
|---|---|---|---|
| [report_archive.md](injuries/report_archive.md) | Implemented | The NBA injury-report PDF archive: URL eras, coverage and the collector | `fetch_data/injury_reports/archive/` |
| [report_db.md](injuries/report_db.md) | Implemented | The Aiven `injury_report` schema: reports, status spans, filings | `postgre_db/injury_report_aiven/` |
| [status_tiers.md](injuries/status_tiers.md) | Implemented | Report statuses in the closing dataset: the three availability groups of 2_5 | `data_processing/injury_status/` |
| [availability_improvement.md](injuries/availability_improvement.md) | Superseded | Earlier availability redesign; its evidence (§1) still holds | — |
| [live_reports.md](injuries/live_reports.md) | Reference | How a prediction run refreshes and reads the injury reports | `fetch_data/injury_reports/sync.py` |
| [live_rosters.md](injuries/live_rosters.md) | Reference | The player pool of a live prediction: official roster plus report-listed players | `create_training_data/live_game_day.py` |

## lineups/ — stints, matchups and player interactions

| Document | Status | Covers | Main code |
|---|---|---|---|
| [data_inventory.md](lineups/data_inventory.md) | Reference | The stint and who-guarded-whom data on disk | `fetch_data/nba_lineups/` |
| [lineup_projection.md](lineups/lineup_projection.md) | Implemented | Projected minutes and lineups: the 2_6 `STARTER_*` / `LU_*` layer | `data_processing/lineups/`, `schema_layers/v2_6.py` |
| [player_interactions_2_7.md](lineups/player_interactions_2_7.md) | Plan | Schema 2_7: player graph and interaction features (`PI_*`) | — |

## serving/ — model registry, predictions and operations

| Document | Status | Covers | Main code |
|---|---|---|---|
| [model_registry.md](serving/model_registry.md) | Implemented | The S3 slot registry: specs, builds, channels, daily refit and promotion | `modeling/registry_*.py`, `modeling/refit.py` |
| [prediction_storage.md](serving/prediction_storage.md) | Implemented | The Aiven prediction store and the S3 run archive | `prediction/prediction_store.py`, `postgre_db/predictions_aiven/` |
| [streamlit_dashboard.md](serving/streamlit_dashboard.md) | Implemented | Aiven-only prediction dashboard, per-game horizons, team logos and Madrid times | `streamlit/streamlit_app.py`, `dashboard/` |
| [historical_analysis.md](serving/historical_analysis.md) | Partial | Historical closing timeline, live model/time heatmap, own-snapshot replay comparison and pooled metrics; combined strategies pending | `dashboard/domain/analysis.py`, `dashboard/repositories/analysis.py`, `dashboard/ui/historical.py` |
| [runner_timings.md](serving/runner_timings.md) | Research | Prediction and retrain durations on the self-hosted runner, cold and warm; what slows them | `scripts/predict_nba_games.py`, `scripts/retrain_prediction_models.py` |
| [snapshot_triggers.md](serving/snapshot_triggers.md) | Implemented | Daily cron-job.org booking of predictor runs at every snapshot time and before tip-off | `scripts/cronjobs/schedule_snapshot_triggers.py`, `scheduling/cronjob_org.py` |
| [ops_watchlist.md](serving/ops_watchlist.md) | Reference | Open checks on the daily jobs and their external sources | `.github/workflows/` |
