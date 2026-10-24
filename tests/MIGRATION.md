# Test migration map

This table records the primary module used to assign every existing file.
The move phase preserves test filenames, assertions and parametrizations;
shared builders move into support without importing test modules.

| Original file | Destination | Primary module |
| --- | --- | --- |
| `tests/test_all_star_voting_dataset.py` | `tests/storage/test_all_star_voting_dataset.py` | `nba_ou.postgre_db.all_star_voting.process_all_star_voting_data` |
| `tests/test_all_star_voting_features.py` | `tests/features/test_all_star_voting_features.py` | `nba_ou.data_processing.all_star_voting.attach_all_star_voting_features` |
| `tests/test_all_star_voting_scripts.py` | `tests/ingestion/test_all_star_voting_scripts.py` | `scripts.all_star_voting.manage_all_star_voting` |
| `tests/test_anchor_total_path.py` | `tests/features/test_anchor_total_path.py` | `nba_ou.data_processing.line_history.anchor_total_path` |
| `tests/test_archive_registry_builds.py` | `tests/storage/test_archive_registry_builds.py` | `scripts.archive_registry_builds` |
| `tests/test_availability_shrinkage.py` | `tests/features/test_availability_shrinkage.py` | `nba_ou.data_processing.past_injuries.injury_effects` |
| `tests/test_backup_core.py` | `tests/storage/test_backup_core.py` | `nba_ou.postgre_db` |
| `tests/test_book_combination.py` | `tests/features/test_book_combination.py` | `nba_ou.data_processing.odds.book_combination` |
| `tests/test_cleaning_dataset_shape.py` | `tests/datasets/test_cleaning_dataset_shape.py` | `nba_ou.data_processing.missing_data.clean_df_for_training` |
| `tests/test_cleaning_hygiene.py` | `tests/datasets/test_cleaning_hygiene.py` | `nba_ou.data_processing.missing_data.clean_df_for_training` |
| `tests/test_closing_line_repair.py` | `tests/features/test_closing_line_repair.py` | `nba_ou.data_processing.line_history.normalization` |
| `tests/test_column_redundancy.py` | `tests/datasets/test_column_redundancy.py` | `nba_ou.data_processing.missing_data.clean_df_for_training` |
| `tests/test_crossbook_total_std.py` | `tests/features/test_crossbook_total_std.py` | `nba_ou.data_processing.merged_home_away_data.global_market_features` |
| `tests/test_db_config_multi_env.py` | `tests/storage/test_db_config_multi_env.py` | `nba_ou.postgre_db.config` |
| `tests/test_extend_history_season_gated.py` | `tests/training/test_extend_history_season_gated.py` | `training_pipeline.config` |
| `tests/test_feature_schema_reduction.py` | `tests/features/test_feature_schema_reduction.py` | `nba_ou.data_processing.team.rolling` |
| `tests/test_fetch_nba_schedule.py` | `tests/ingestion/test_fetch_nba_schedule.py` | `nba_ou.fetch_data.nba_schedule` |
| `tests/test_fresh_absence_features.py` | `tests/features/test_fresh_absence_features.py` | `nba_ou.data_processing.merged_home_away_data.add_features_after_merging` |
| `tests/test_game_projection.py` | `tests/features/test_game_projection.py` | `nba_ou.data_processing.lineups.game_projection` |
| `tests/test_game_time_index_sync.py` | `tests/storage/test_game_time_index_sync.py` | `nba_ou.postgre_db.game_time_index.sync_game_time_index` |
| `tests/test_get_schedule_games.py` | `tests/ingestion/test_get_schedule_games.py` | `nba_ou.fetch_data.scheduled_game.get_schedule_games` |
| `tests/test_global_market_speedups.py` | `tests/features/test_global_market_speedups.py` | `nba_ou.data_processing.merged_home_away_data` |
| `tests/test_head_to_head_speedup.py` | `tests/features/test_head_to_head_speedup.py` | `nba_ou.data_processing.historic_games.historic_games_statistics` |
| `tests/test_high_value_features.py` | `tests/features/test_high_value_features.py` | `nba_ou.data_processing.merged_home_away_data.add_features_after_merging` |
| `tests/test_historical_player_availability.py` | `tests/features/test_historical_player_availability.py` | `nba_ou.data_processing.players.attach_player_features` |
| `tests/test_historical_ridge_movement.py` | `tests/datasets/test_historical_ridge_movement.py` | `nba_ou.create_training_data.historical_ridge_movement` |
| `tests/test_history_memo.py` | `tests/datasets/test_history_memo.py` | `nba_ou.create_training_data` |
| `tests/test_history_only_books.py` | `tests/core/test_history_only_books.py` | `nba_ou.config.odds_columns` |
| `tests/test_identity_columns_and_text_cleaning.py` | `tests/datasets/test_identity_columns_and_text_cleaning.py` | `nba_ou.data_processing.missing_data.clean_df_for_training` |
| `tests/test_injury_report_archive.py` | `tests/ingestion/test_injury_report_archive.py` | `nba_ou.fetch_data.injury_reports.archive` |
| `tests/test_injury_report_db.py` | `tests/storage/test_injury_report_db.py` | `nba_ou.postgre_db.injury_report_aiven.parse` |
| `tests/test_injury_report_game_dimension.py` | `tests/storage/test_injury_report_game_dimension.py` | `nba_ou.postgre_db.injury_report_aiven.parse` |
| `tests/test_injury_report_parsing.py` | `tests/ingestion/test_injury_report_parsing.py` | `nba_ou.fetch_data.injury_reports.get_latest_injury_report` |
| `tests/test_injury_report_status_features.py` | `tests/features/test_injury_report_status_features.py` | `nba_ou.data_processing.injury_status` |
| `tests/test_injury_report_sync.py` | `tests/ingestion/test_injury_report_sync.py` | `nba_ou.fetch_data.injury_reports` |
| `tests/test_injury_stage_caching.py` | `tests/features/test_injury_stage_caching.py` | `nba_ou.data_processing.past_injuries.injury_effects` |
| `tests/test_input_cache.py` | `tests/storage/test_input_cache.py` | `nba_ou.postgre_db` |
| `tests/test_intermediate_base_lookback.py` | `tests/datasets/test_intermediate_base_lookback.py` | `nba_ou.create_training_data.create_intermediate_line_df` |
| `tests/test_intermediate_column_leakage.py` | `tests/datasets/test_intermediate_column_leakage.py` | `nba_ou.create_training_data.select_intermediate_columns` |
| `tests/test_intermediate_injuries.py` | `tests/datasets/test_intermediate_injuries.py` | `nba_ou.create_training_data` |
| `tests/test_intermediate_lineup_features.py` | `tests/features/test_intermediate_lineup_features.py` | `nba_ou.data_processing.lineups.intermediate_features` |
| `tests/test_intermediate_market_dynamics.py` | `tests/features/test_intermediate_market_dynamics.py` | `nba_ou.data_processing.line_history.market_dynamics` |
| `tests/test_intermediate_movement_features.py` | `tests/features/test_intermediate_movement_features.py` | `nba_ou.data_processing.line_history.movement_features` |
| `tests/test_intermediate_pipeline_coexistence.py` | `tests/datasets/test_intermediate_pipeline_coexistence.py` | `nba_ou.create_training_data.create_df_to_predict` |
| `tests/test_intermediate_prediction.py` | `tests/storage/test_intermediate_prediction.py` | `nba_ou.modeling.registry_paths` |
| `tests/test_intermediate_referees.py` | `tests/datasets/test_intermediate_referees.py` | `nba_ou.create_training_data.intermediate_referees` |
| `tests/test_intermediate_target_mode.py` | `tests/datasets/test_intermediate_target_mode.py` | `nba_ou.create_training_data` |
| `tests/test_intermediate_wide_merge.py` | `tests/datasets/test_intermediate_wide_merge.py` | `nba_ou.create_training_data.create_intermediate_line_df` |
| `tests/test_is_overtime_retained_in_training_data.py` | `tests/features/test_is_overtime_retained_in_training_data.py` | `nba_ou.data_processing.merged_home_away_data.add_features_after_merging` |
| `tests/test_latest_input_prediction.py` | `tests/prediction/test_latest_input_prediction.py` | `nba_ou.prediction.intermediate_frame` |
| `tests/test_layered_dataset.py` | `tests/datasets/test_layered_dataset.py` | `nba_ou.create_training_data.schema_layers` |
| `tests/test_legacy_injury_report.py` | `tests/ingestion/test_legacy_injury_report.py` | `nba_ou.fetch_data.injury_reports.legacy_injury_report` |
| `tests/test_line_history_backup.py` | `tests/storage/test_line_history_backup.py` | `nba_ou.postgre_db` |
| `tests/test_line_history_book_merge.py` | `tests/features/test_line_history_book_merge.py` | `nba_ou.data_processing.line_history.book_merge` |
| `tests/test_line_history_ingest.py` | `tests/ingestion/test_line_history_ingest.py` | `nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history` |
| `tests/test_line_history_normalization.py` | `tests/features/test_line_history_normalization.py` | `nba_ou.data_processing.line_history.normalization` |
| `tests/test_line_history_scrape.py` | `tests/ingestion/test_line_history_scrape.py` | `nba_ou.fetch_data.odds_sportsbook` |
| `tests/test_line_history_snapshots.py` | `tests/features/test_line_history_snapshots.py` | `nba_ou.data_processing.line_history.snapshots` |
| `tests/test_line_history_speedups.py` | `tests/features/test_line_history_speedups.py` | `nba_ou.data_processing.line_history` |
| `tests/test_line_history_transform.py` | `tests/storage/test_line_history_transform.py` | `nba_ou.postgre_db.line_history_aiven` |
| `tests/test_line_history_update.py` | `tests/storage/test_line_history_update.py` | `nba_ou.postgre_db.line_history_aiven` |
| `tests/test_lineup_archive.py` | `tests/ingestion/test_lineup_archive.py` | `nba_ou.fetch_data.nba_lineups.archive` |
| `tests/test_lineup_availability.py` | `tests/features/test_lineup_availability.py` | `nba_ou.data_processing.lineups.availability` |
| `tests/test_lineup_campaign_configs.py` | `tests/training/test_lineup_campaign_configs.py` | `training_pipeline.cli` |
| `tests/test_lineup_client.py` | `tests/ingestion/test_lineup_client.py` | `nba_ou.fetch_data.nba_lineups.client` |
| `tests/test_lineup_features.py` | `tests/features/test_lineup_features.py` | `nba_ou.data_processing.lineups` |
| `tests/test_lineup_leakage.py` | `tests/features/test_lineup_leakage.py` | `nba_ou.data_processing.lineups` |
| `tests/test_lineup_manifest.py` | `tests/ingestion/test_lineup_manifest.py` | `nba_ou.fetch_data.nba_lineups.manifest` |
| `tests/test_lineup_stint_store.py` | `tests/features/test_lineup_stint_store.py` | `nba_ou.data_processing.lineups.stint_store` |
| `tests/test_lineup_stints.py` | `tests/features/test_lineup_stints.py` | `nba_ou.data_processing.lineups.stints` |
| `tests/test_lineup_synergy.py` | `tests/features/test_lineup_synergy.py` | `nba_ou.data_processing.lineups.synergy` |
| `tests/test_lineup_update.py` | `tests/ingestion/test_lineup_update.py` | `nba_ou.fetch_data.nba_lineups.client` |
| `tests/test_live_game_day_rosters.py` | `tests/datasets/test_live_game_day_rosters.py` | `nba_ou.create_training_data.live_game_day` |
| `tests/test_live_games.py` | `tests/ingestion/test_live_games.py` | `nba_ou.fetch_data.live_games` |
| `tests/test_live_injury_ingest.py` | `tests/storage/test_live_injury_ingest.py` | `nba_ou.postgre_db.injury_report_aiven` |
| `tests/test_live_injury_prediction.py` | `tests/storage/test_live_injury_prediction.py` | `nba_ou.postgre_db.injury_report_aiven.live` |
| `tests/test_live_injury_reports.py` | `tests/ingestion/test_live_injury_reports.py` | `nba_ou.fetch_data.injury_reports.archive` |
| `tests/test_live_line_history.py` | `tests/storage/test_live_line_history.py` | `nba_ou.postgre_db.line_history_aiven.live` |
| `tests/test_live_replay.py` | `tests/datasets/test_live_replay.py` | `nba_ou.create_training_data.live_replay` |
| `tests/test_live_roster_features.py` | `tests/features/test_live_roster_features.py` | `nba_ou.data_processing.players.attach_player_features` |
| `tests/test_live_rosters.py` | `tests/ingestion/test_live_rosters.py` | `nba_ou.fetch_data` |
| `tests/test_matchups_archive.py` | `tests/ingestion/test_matchups_archive.py` | `nba_ou.fetch_data.nba_lineups.client` |
| `tests/test_merge_scheduled_odds.py` | `tests/features/test_merge_scheduled_odds.py` | `nba_ou.data_processing.odds.merge_scheduled_odds` |
| `tests/test_meta_learner_baselines.py` | `tests/training/test_meta_learner_baselines.py` | `lab.meta_learner.meta_learner_baselines` |
| `tests/test_meta_learner_classifier_tsplit_utils.py` | `tests/training/test_meta_learner_classifier_tsplit_utils.py` | `lab.meta_learner.meta_learner_classifier_tsplit_utils` |
| `tests/test_meta_learner_feature_utils.py` | `tests/training/test_meta_learner_feature_utils.py` | `lab.meta_learner.meta_learner_feature_utils` |
| `tests/test_meta_learner_training_data.py` | `tests/training/test_meta_learner_training_data.py` | `nba_ou.modeling.meta_learner_training_data` |
| `tests/test_minutes_model.py` | `tests/features/test_minutes_model.py` | `nba_ou.data_processing.lineups.minutes_model` |
| `tests/test_model_registry_paths.py` | `tests/storage/test_model_registry_paths.py` | `nba_ou.modeling.registry_paths` |
| `tests/test_model_registry_store.py` | `tests/storage/test_model_registry_store.py` | `nba_ou.modeling.registry_store` |
| `tests/test_moneyline_data_readiness.py` | `tests/features/test_moneyline_data_readiness.py` | `nba_ou.data_processing.odds.canonical_markets` |
| `tests/test_moviodds_line_history.py` | `tests/ingestion/test_moviodds_line_history.py` | `nba_ou.fetch_data.odds_moviodds` |
| `tests/test_nba_api_session.py` | `tests/ingestion/test_nba_api_session.py` | `nba_ou.fetch_data.nba_api_session` |
| `tests/test_normalize_spread_lines.py` | `tests/features/test_normalize_spread_lines.py` | `nba_ou.data_processing.odds.normalize_spread_lines` |
| `tests/test_normalize_total_lines.py` | `tests/features/test_normalize_total_lines.py` | `nba_ou.data_processing.odds.normalize_total_lines` |
| `tests/test_odds_column_prefix_invariant.py` | `tests/core/test_odds_column_prefix_invariant.py` | `nba_ou.config.odds_columns` |
| `tests/test_odds_feature_books.py` | `tests/features/test_odds_feature_books.py` | `nba_ou.data_processing.merged_home_away_data.odds_feature_engeneer` |
| `tests/test_oddslogic_line_history.py` | `tests/ingestion/test_oddslogic_line_history.py` | `nba_ou.fetch_data.odds_oddslogic` |
| `tests/test_oddstrader_line_history.py` | `tests/ingestion/test_oddstrader_line_history.py` | `nba_ou.fetch_data.odds_oddstrader` |
| `tests/test_overtime_adjustment.py` | `tests/features/test_overtime_adjustment.py` | `nba_ou.data_processing.team.totals` |
| `tests/test_pbp_archive.py` | `tests/ingestion/test_pbp_archive.py` | `nba_ou.fetch_data.nba_lineups.pbp_archive` |
| `tests/test_player_average_speedups.py` | `tests/features/test_player_average_speedups.py` | `nba_ou.data_processing.players` |
| `tests/test_player_feature_profile.py` | `tests/features/test_player_feature_profile.py` | `nba_ou.data_processing.players.attach_player_features` |
| `tests/test_player_features_season_opener.py` | `tests/features/test_player_features_season_opener.py` | `nba_ou.data_processing.players.attach_player_features` |
| `tests/test_player_lookup_speedups.py` | `tests/features/test_player_lookup_speedups.py` | `nba_ou.data_processing.past_injuries.past_injuries` |
| `tests/test_player_ratings.py` | `tests/features/test_player_ratings.py` | `nba_ou.data_processing.lineups.player_ratings` |
| `tests/test_predict_past_snapshots.py` | `tests/prediction/test_predict_past_snapshots.py` | `scripts.predict_past_snapshots` |
| `tests/test_predict_tabpfn.py` | `tests/prediction/test_predict_tabpfn.py` | `nba_ou.prediction.prediction_tabpfn_client` |
| `tests/test_predict_tonight_only.py` | `tests/prediction/test_predict_tonight_only.py` | `nba_ou.prediction.prediction` |
| `tests/test_prediction_store.py` | `tests/prediction/test_prediction_store.py` | `nba_ou.prediction.prediction_store` |
| `tests/test_prediction_targets_and_horizon.py` | `tests/prediction/test_prediction_targets_and_horizon.py` | `nba_ou.prediction.prediction` |
| `tests/test_preflight_campaign.py` | `tests/training/test_preflight_campaign.py` | `scripts.preflight_campaign` |
| `tests/test_projected_lineup.py` | `tests/features/test_projected_lineup.py` | `nba_ou.data_processing.lineups.projected_lineup` |
| `tests/test_rating_gate.py` | `tests/features/test_rating_gate.py` | `nba_ou.data_processing.lineups.rating_gate` |
| `tests/test_read_injury_report.py` | `tests/ingestion/test_read_injury_report.py` | `nba_ou.fetch_data.injury_reports.get_latest_injury_report` |
| `tests/test_ref_assignments_archive.py` | `tests/storage/test_ref_assignments_archive.py` | `nba_ou.postgre_db.injuries_refs` |
| `tests/test_referee_assignment_store.py` | `tests/ingestion/test_referee_assignment_store.py` | `nba_ou.fetch_data.referees` |
| `tests/test_referee_campaign_configs.py` | `tests/training/test_referee_campaign_configs.py` | `training_pipeline.cli` |
| `tests/test_referee_speedups.py` | `tests/features/test_referee_speedups.py` | `nba_ou.data_processing.referees.add_referee_features` |
| `tests/test_referee_tendencies.py` | `tests/features/test_referee_tendencies.py` | `nba_ou.data_processing.referees.referee_tendencies` |
| `tests/test_refit_rows.py` | `tests/training/test_refit_rows.py` | `nba_ou.modeling.refit` |
| `tests/test_refit_schema_layers.py` | `tests/training/test_refit_schema_layers.py` | `nba_ou.modeling.refit` |
| `tests/test_reporting_alt_line.py` | `tests/reporting/test_reporting_alt_line.py` | `training_pipeline.reporting` |
| `tests/test_reporting_coverage.py` | `tests/reporting/test_reporting_coverage.py` | `training_pipeline.reporting` |
| `tests/test_reporting_discovery.py` | `tests/reporting/test_reporting_discovery.py` | `training_pipeline.reporting.discovery` |
| `tests/test_reporting_factors.py` | `tests/reporting/test_reporting_factors.py` | `training_pipeline.reporting` |
| `tests/test_reporting_labels.py` | `tests/reporting/test_reporting_labels.py` | `training_pipeline.reporting` |
| `tests/test_retraining_utils_sample_weight.py` | `tests/training/test_retraining_utils_sample_weight.py` | `nba_ou.modeling.retraining_utils` |
| `tests/test_rolling_speedups.py` | `tests/features/test_rolling_speedups.py` | `nba_ou.data_processing.statistics` |
| `tests/test_roster_continuity.py` | `tests/features/test_roster_continuity.py` | `nba_ou.data_processing.merged_home_away_data.merge_home_away` |
| `tests/test_rotation_rebuild.py` | `tests/ingestion/test_rotation_rebuild.py` | `nba_ou.fetch_data.nba_lineups.rotation_rebuild` |
| `tests/test_row_order.py` | `tests/storage/test_row_order.py` | `nba_ou.postgre_db.row_order` |
| `tests/test_sbr_close_provenance.py` | `tests/storage/test_sbr_close_provenance.py` | `nba_ou.postgre_db.odds_sportsbook.repair_closes` |
| `tests/test_sbr_feed_freshness.py` | `tests/storage/test_sbr_feed_freshness.py` | `nba_ou.postgre_db.line_history_aiven.feed_freshness` |
| `tests/test_sbr_json_engine.py` | `tests/ingestion/test_sbr_json_engine.py` | `nba_ou.fetch_data.odds_sportsbook` |
| `tests/test_schedule_snapshot_triggers.py` | `tests/prediction/test_schedule_snapshot_triggers.py` | `scripts.cronjobs.schedule_snapshot_triggers` |
| `tests/test_scheduled_player_rows.py` | `tests/features/test_scheduled_player_rows.py` | `nba_ou.data_processing.players.players_statistics` |
| `tests/test_scheduled_referees.py` | `tests/datasets/test_scheduled_referees.py` | `nba_ou.create_training_data.intermediate_referees` |
| `tests/test_schema_layer_2_6.py` | `tests/datasets/test_schema_layer_2_6.py` | `nba_ou.create_training_data.schema_layers` |
| `tests/test_schema_layers_contract.py` | `tests/datasets/test_schema_layers_contract.py` | `nba_ou.create_training_data.schema_layers` |
| `tests/test_schema_layers_registry.py` | `tests/datasets/test_schema_layers_registry.py` | `nba_ou.create_training_data.schema_layers` |
| `tests/test_season_partitions.py` | `tests/datasets/test_season_partitions.py` | `nba_ou.create_training_data` |
| `tests/test_season_queries.py` | `tests/storage/test_season_queries.py` | `nba_ou.postgre_db.season_queries` |
| `tests/test_season_tables.py` | `tests/storage/test_season_tables.py` | `nba_ou.postgre_db` |
| `tests/test_season_ticks_s3.py` | `tests/storage/test_season_ticks_s3.py` | `nba_ou.postgre_db` |
| `tests/test_season_utils.py` | `tests/core/test_season_utils.py` | `nba_ou.utils.seasons` |
| `tests/test_shrinkage_prior.py` | `tests/features/test_shrinkage_prior.py` | `nba_ou.data_processing.past_injuries.injury_effects` |
| `tests/test_snapshot_grid_and_windows.py` | `tests/features/test_snapshot_grid_and_windows.py` | `nba_ou.data_processing.line_history.cross_book` |
| `tests/test_snapshot_scoring.py` | `tests/training/test_snapshot_scoring.py` | `training_pipeline.betting` |
| `tests/test_spread_backward_compatibility.py` | `tests/training/test_spread_backward_compatibility.py` | `training_pipeline.cli` |
| `tests/test_spread_betting_settlement.py` | `tests/training/test_spread_betting_settlement.py` | `training_pipeline.betting` |
| `tests/test_spread_directional_accuracy.py` | `tests/training/test_spread_directional_accuracy.py` | `nba_ou.modeling.scorers` |
| `tests/test_spread_error_strategy.py` | `tests/training/test_spread_error_strategy.py` | `training_pipeline.config` |
| `tests/test_spread_prediction_end_to_end.py` | `tests/prediction/test_spread_prediction_end_to_end.py` | `nba_ou.prediction` |
| `tests/test_spread_review_regressions.py` | `tests/training/test_spread_review_regressions.py` | `training_pipeline.betting` |
| `tests/test_spread_snapshot_target.py` | `tests/datasets/test_spread_snapshot_target.py` | `nba_ou.create_training_data.create_intermediate_line_df` |
| `tests/test_spread_target_algebra.py` | `tests/core/test_spread_target_algebra.py` | `nba_ou.config.market_columns` |
| `tests/test_spread_upstream_gates.py` | `tests/datasets/test_spread_upstream_gates.py` | `nba_ou.create_training_data.select_intermediate_columns` |
| `tests/test_starter_history.py` | `tests/features/test_starter_history.py` | `nba_ou.data_processing.players.starter_history` |
| `tests/test_style_matchup.py` | `tests/features/test_style_matchup.py` | `nba_ou.data_processing.lineups.game_projection` |
| `tests/test_style_matchup_features.py` | `tests/features/test_style_matchup_features.py` | `nba_ou.data_processing.team.rolling` |
| `tests/test_team_one_hot_features.py` | `tests/features/test_team_one_hot_features.py` | `nba_ou.data_processing.merged_home_away_data.team_one_hot_features` |
| `tests/test_team_records.py` | `tests/features/test_team_records.py` | `nba_ou.data_processing.team.records` |
| `tests/test_train_data_store.py` | `tests/datasets/test_train_data_store.py` | `nba_ou.create_training_data.train_data_store` |
| `tests/test_training_data_parquet.py` | `tests/training/test_training_data_parquet.py` | `training_pipeline.parquet_dataset` |
| `tests/test_training_pipeline_backtest.py` | `tests/training/test_training_pipeline_backtest.py` | `training_pipeline.backtest` |
| `tests/test_training_pipeline_baseline.py` | `tests/training/test_training_pipeline_baseline.py` | `training_pipeline.baseline` |
| `tests/test_training_pipeline_betting.py` | `tests/training/test_training_pipeline_betting.py` | `training_pipeline.betting` |
| `tests/test_training_pipeline_classifier.py` | `tests/training/test_training_pipeline_classifier.py` | `training_pipeline` |
| `tests/test_training_pipeline_cli_config.py` | `tests/training/test_training_pipeline_cli_config.py` | `training_pipeline.cli` |
| `tests/test_training_pipeline_config.py` | `tests/training/test_training_pipeline_config.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_cv_betting_and_seeds.py` | `tests/training/test_training_pipeline_cv_betting_and_seeds.py` | `training_pipeline` |
| `tests/test_training_pipeline_data.py` | `tests/training/test_training_pipeline_data.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_evaluation.py` | `tests/training/test_training_pipeline_evaluation.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_flow.py` | `tests/training/test_training_pipeline_flow.py` | `training_pipeline` |
| `tests/test_training_pipeline_leaderboard.py` | `tests/training/test_training_pipeline_leaderboard.py` | `training_pipeline.leaderboard` |
| `tests/test_training_pipeline_naming.py` | `tests/training/test_training_pipeline_naming.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_overtime_filter.py` | `tests/training/test_training_pipeline_overtime_filter.py` | `training_pipeline` |
| `tests/test_training_pipeline_pipeline.py` | `tests/training/test_training_pipeline_pipeline.py` | `training_pipeline` |
| `tests/test_training_pipeline_planted_signal.py` | `tests/training/test_training_pipeline_planted_signal.py` | `training_pipeline` |
| `tests/test_training_pipeline_promote.py` | `tests/training/test_training_pipeline_promote.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_reuse.py` | `tests/training/test_training_pipeline_reuse.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_rolling_origin.py` | `tests/training/test_training_pipeline_rolling_origin.py` | `training_pipeline` |
| `tests/test_training_pipeline_sample_weight_wiring.py` | `tests/training/test_training_pipeline_sample_weight_wiring.py` | `training_pipeline` |
| `tests/test_training_pipeline_search_space.py` | `tests/training/test_training_pipeline_search_space.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_snapshot_grain.py` | `tests/training/test_training_pipeline_snapshot_grain.py` | `training_pipeline.config` |
| `tests/test_training_pipeline_tuning_dispatch.py` | `tests/training/test_training_pipeline_tuning_dispatch.py` | `training_pipeline.config` |
| `tests/test_travel_processing.py` | `tests/features/test_travel_processing.py` | `nba_ou.data_processing.merged_home_away_data.add_features_after_merging` |
| `tests/test_trend_slope_season_fallback.py` | `tests/features/test_trend_slope_season_fallback.py` | `nba_ou.data_processing.team.rolling` |
| `tests/test_unservable_slots_do_not_fail_daily_jobs.py` | `tests/prediction/test_unservable_slots_do_not_fail_daily_jobs.py` | `scripts.retrain_prediction_models` |
| `tests/test_yahoo_feature_profile.py` | `tests/features/test_yahoo_feature_profile.py` | `nba_ou.data_processing.odds.yahoo_features` |
| `tests/fake_s3.py` | `tests/support/fake_s3.py` | `Shared test support` |
| `tests/frame_identity.py` | `tests/support/frame_identity.py` | `Shared test support` |
| `tests/legacy_anchor_total_path.py` | `tests/support/legacy/legacy_anchor_total_path.py` | `Shared test support` |
| `tests/legacy_player_lookup.py` | `tests/support/legacy/legacy_player_lookup.py` | `Shared test support` |
| `tests/legacy_referee_features.py` | `tests/support/legacy/legacy_referee_features.py` | `Shared test support` |
| `tests/legacy_top_n_averages.py` | `tests/support/legacy/legacy_top_n_averages.py` | `Shared test support` |

New support modules extracted during the migration:

| Module | Extracted from |
| --- | --- |
| `support/paths.py` | Shared repository-root resolution |
| `support/line_history.py` | `test_line_history_snapshots.py` |
| `support/model_registry.py` | `test_model_registry_store.py` |
| `support/lineup_world.py` | `test_lineup_leakage.py` |
| `support/style_matchup.py` | `test_style_matchup.py` |
| `support/player_availability.py` | `test_historical_player_availability.py` |
| `support/injury_states.py` | `test_injury_stage_caching.py` |

Migration validation: the sorted normalized node IDs (including duplicate
multiplicities) match all 3,012 cases collected before the move. Every original
`assert` statement was checked unchanged. A full run found 3,004 passing cases,
5 PostgreSQL skips and 3 failures in `TestFlushing`; the same failures were
reproduced from the original file in Git. Its fake ingest result lacked the
recently added `stale_status_games` field. That fixture correction belongs to
the subsequent reinforcement change, not the organizational move.
