# Feature engineering overview

This page maps the feature engineering in this repository: what data exists,
where it is stored, how it reaches each dataset, and which module builds each
feature family. It is a map, not the full reference, so it links to the
detailed documents below rather than repeating them.

Snapshot of the code as of 2026-09-17, schema version `2_5`
(`src/nba_ou/config/dataset_versions.py`).

## Where the detail lives

| Document | Scope |
|---|---|
| `docs/README_Training Data Processing.md` | Closing-line pipeline step by step: rolling windows, player and injury features, referees, travel, column selection, naming |
| `src/nba_ou/config/dataset_versions.py` | What every schema version (`2_0` to `2_5`) added or changed |
| `docs/intermediate_line_dataset_plan.md`, `docs/intermediate_line_pipeline_plan.md` | Intermediate (snapshot) dataset design |
| `docs/intermediate_line_feature_engineering_plan.md` | Round-2 market features for snapshots, with the evidence behind them |
| `docs/intermediate_market_dynamics_plan.md` | G1–G4: injury news, news reaction, cross-market coherence, move-to-close Ridges |
| `docs/injury_report_db_plan.md`, `docs/injury_report_archive_plan.md`, `docs/injury_status_tiers_plan.md`, `docs/injury_availability_improvement_plan.md` | Official injury-report archive and the availability features built on it |
| `docs/line_history_phase0_findings.md`, `docs/line_history_updating.md` | Line-history store: data findings and daily update |
| `.claude/skills/feature-engineering/` | Portable, sport-agnostic rules and families; no repo paths |
| `.claude/skills/odds-data-architecture/`, `.claude/skills/sports-data-architecture/` | Portable ingestion and storage design |

## 1. Data sources and storage

There are two Postgres instances, selected with `DB_ENV` or the `[Database]`
section of `config.secrets.ini` (`src/nba_ou/postgre_db/config/db_config.py`):

- **Supabase** is the default environment. It holds the game, player, odds,
  referee, injury and All-Star schemas.
- **Aiven** holds the two time-stamped stores: `line_history` and
  `injury_report`. They are reached with `connect_line_history_db()` or
  `connect_nba_db("aiven")`.

| Data | Source | Fetch / ingest | Storage | Read by features through |
|---|---|---|---|---|
| Team game logs, player box scores | NBA stats API (`nba_api` box scores) | `fetch_data/fetch_nba_data`, `postgre_db/games/update_games` | Supabase `nba_games`, `nba_players` | `postgre_db.load_all_nba_data_from_db` |
| Closing and opening odds, per book | Sportsbook Review | `fetch_data/odds_sportsbook/scrape_sportsbook.py` | Supabase `odds_sportsbook` | `postgre_db/odds/merge_odds_data.load_and_merge_odds_yahoo_sportsbookreview` |
| Closing odds | Yahoo Sports | `fetch_data/odds_yahoo` | Supabase `odds_yahoo` | same merge function |
| Line history: every pre-game price change, per book, for totals, spread and moneyline | Sportsbook Review line history | `fetch_data/odds_sportsbook/scrape_sportsbook_line_history.py`, `scripts/update_databases/update_line_history_database.py` | Aiven `line_history` (`lh_game`, `lh_line`) | `postgre_db/line_history_aiven/fetch.py` (`fetch_pregame_ticks`, `fetch_games`) |
| Historical injuries and referee crews | NBA stats API box-score summaries | `postgre_db/injuries_refs/update_ref_injuries_database` | Supabase `nba_injuries`, `nba_refs` | `get_injury_data_from_db`, `get_refs_data_from_db` |
| Official injury reports, time-stamped | NBA injury-report PDFs (`ak-static.cms.nba.com`) | `fetch_data/injury_reports/archive`, `scripts/injury_reports/` | Aiven `injury_report` (reports, status spans, filing spans) | `postgre_db/injury_report_aiven/fetch.py` |
| Same-day referee assignments | `official.nba.com/referee-assignments` | `fetch_data/referees/fetch_refs_data.py` | used live; `ref_assignments_archive.py` defines an archive table (`nba_ref_assignments`) that does not exist in the Supabase database yet | same-day prediction path only |
| All-Star voting | NBA voting results | `scripts/all_star_voting/manage_all_star_voting.py` | Supabase `nba_all_star_voting` | `load_all_star_voting_from_db` |
| Schedule, tip-off times | NBA schedule (`ScoreboardV2`) | `fetch_data/nba_schedule`, `fetch_data/scheduled_game` | `lh_game.tipoff_utc`, Supabase `nba_game_time_index` | line-history games; injury-report game resolution |

Supabase also has schemas that neither dataset builder reads: `nba_odds`,
`nba_odds_mgm`, `sportsbook_matchup_history` and the per-market
`odds_sportsbook_*_line_history` schemas. Line history for features comes from
Aiven.

The daily refresh is `scripts/update_databases/update_all_databases.py` (games,
Sportsbook Review, Yahoo, refs and injuries) plus
`update_line_history_database.py` (Aiven line history).

## 2. The two datasets

| | Closing-line dataset | Intermediate-line dataset |
|---|---|---|
| Grain | one row per game | one row per (game, snapshot) |
| Builder | `create_training_data/create_df_to_predict.py` | `create_training_data/create_intermediate_line_df.py` |
| Script | `scripts/create_train_data/create_train_data.py` | `scripts/create_train_data/create_intermediate_line_train_data.py` |
| Moment the bet is placed | at close | `TIME_TO_MATCH_MIN` before tip, from `DEFAULT_SNAPSHOT_GRID` = 0, 30, 60, 120, 180, 240, 300, 360, 480, 720 minutes |
| Line the target is defined against | closing line of the main book | anchor book's (bet365) centred line **at the snapshot** |
| Market features | wide closing and opening columns per book | snapshot panel read from line-history ticks |
| Injuries | last report before tip, with fallback to realised absences | last report strictly before the snapshot timestamp |
| Referees | always visible | masked before 09:00 ET on game day |
| Same-day serving | yes (`todays_prediction=True`) | not implemented |
| Leakage gate | `merged_home_away_data/select_train_columns.py` | `create_training_data/select_intermediate_columns.py` |
| Closing lines | the target line | moved to a scoring sidecar CSV (`ODDS_CLOSING_*`), never in X |

The intermediate builder reuses the closing pipeline for everything that does
not depend on the snapshot. `create_base_game_features.py` builds the per-game
base once. The snapshot-dependent families are then built per snapshot and
joined on `(GAME_ID, TIME_TO_MATCH_MIN)`.

Targets: `TOTAL_POINTS`, `LINE_ERROR` (total minus line), `HOME_MARGIN` and
`SPREAD_ERROR`. They survive into the CSV but are blocked from every feature
matrix by `training_pipeline.data.assert_no_leaking_features`.

## 3. How the data flows

```text
nba_games + nba_players ──► team cleaning, records, rolling stats (team rows, 2 per game)
nba_injuries ─────────────► player history + injured/available sets ─┐
injury_report (Aiven) ────► report state at tip / at snapshot ───────┤
nba_all_star_voting ──────► star-quality of injured players ─────────┤
                                                                      ▼
                                              merge home/away ──► one game row
odds_sportsbook + odds_yahoo ──► closing/opener odds, book combination ─┤
nba_refs ──► referee legacy + crew tendencies ─────────────────────────┤
schedule/arenas ──► travel, rest, calendar ────────────────────────────┤
                                                                      ▼
                                         column gate ──► CLOSING DATASET
                                                                      │ same per-game base, rebuilt by create_base_game_features;
                                                                      │ current-game closing odds go to the scoring sidecar
line_history (Aiven) ticks ──► snapshot panel ──► movement, consensus, path ──┐
injury_report (Aiven) ──► 2_5 injury families per snapshot, G1 news ──────────┤
ticks + news timeline ──► G2 news reaction, G3 cross-market, G4 Ridges ───────┤
                                                                               ▼
                                   referee mask, column gate ──► INTERMEDIATE DATASET
                                                               + scoring sidecar
```

## 4. Feature families

In the "Datasets" column, **C** is the closing dataset and **I** is the
intermediate dataset.

### Team form and matchup

| Family | Example columns | Module | Inputs | Datasets |
|---|---|---|---|---|
| Rolling team stats (windows, expanding season mean and std, trends, home/away) | `PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME`, `PTS_SEASON_BEFORE_AVG_TEAM_HOME` | `data_processing/team/rolling.py`, `team/totals.py` | team games | C, I |
| Records, rest, playoffs last season, overtime history | `TEAM_RECORD_BEFORE_GAME`, `REST_DAYS_BEFORE_MATCH`, `IS_OVERTIME_LAST_GAME_BEFORE` | `team/records.py`, `team/rolling.py` | team games | C, I |
| Style matchups (offence against opposing defence, pace) | `EXPECTED_POSS_FROM_PACE_BEFORE`, `OFFDEF_MISMATCH_*_BEFORE` | `team/style_matchups.py`, `merged_home_away_data/add_features_after_merging.py` | team games | C, I |
| Travel and schedule | `TOTAL_KM_IN_LAST_<N>_DAYS_*`, `JETLAG_HOURS_FROM_LAST_GAME_*`, `BACK_TO_BACK_BEFORE` | `travel/travel_processing.py` | schedule, arenas | C, I |
| Calendar and team context | `IS_WEEKEND_BEFORE`, `SAME_DIVISION_BEFORE`, team one-hot | `add_features_after_merging.add_game_date_features`, `team_one_hot_features.py` | schedule | C, I |

### Players, injuries and availability

| Family | Example columns | Module | Inputs | Datasets |
|---|---|---|---|---|
| Player profile: top players, injured, available and questionable groups | `TOP1_PLAYER_*_BEFORE_TEAM_*`, `TOTAL_INJURED_PLAYER_PTS_BEFORE_*`, `N_QUESTIONABLE_PLAYERS_*` | `players/attach_player_features.py`, `players/feature_profile.py` | box scores, `nba_injuries`, report state | C, I (per snapshot) |
| Report state (coverage, report age, status counts, P_PLAY, effects) | `INJURY_REPORT_COVERED_*`, `LAST_STATUS_REPORT_AGE_MIN_*`, `N_REPORT_OUT_PLAYERS_*` | `injury_status/features.py`, `report_state.py`, `status_history.py` | Aiven `injury_report` | C (last report before tip), I (per snapshot) |
| Fresh absences | `INJ_FRESH_OUT_<stat>_BEFORE_*`, `INJ_KEY_PLAYER_FIRST_GAME_OUT_*` | `players/fresh_absence.py`, `add_fresh_absence_sums` | box scores, injuries | C, I |
| Availability effects (empirical-Bayes with/without deltas) | `TOP3_AVAILABILITY_EFFECT_*`, `TOP3_INJURED_*`, `TOP2_QUESTIONABLE_*` | `past_injuries/injury_effects.py` | box scores, closing lines of **earlier** games | C, I (per snapshot) |
| Roster continuity | `ROSTER_MINUTES_CONTINUITY_*_PCT_BEFORE_*`, `ROSTER_NET_MINUTES_*` | `players/roster_continuity.py` | box scores | C, I |
| All-Star voting | `ALL_STAR_*_INJURED_*`, `ALL_STAR_*_QUESTIONABLE_*` | `all_star_voting/attach_all_star_voting_features.py` | `nba_all_star_voting` | C, I (per snapshot) |
| **G1** injury news up to the snapshot (change in expected missing points) | `INJ_SNAP_*_BEFORE_TEAM_{HOME,AWAY}` | `injury_status/news.py` | Aiven `injury_report`, player form | I |

In the intermediate dataset, the families marked "per snapshot" are recomputed
at each snapshot by `create_training_data/intermediate_injuries.py`.

### Officials

| Family | Example columns | Module | Inputs | Datasets |
|---|---|---|---|---|
| Legacy referee deltas | `REF_{AVG,STD,SUM}_*_DIFF_BEFORE` | `referees/add_referee_features.py` | `nba_refs`, closing lines of earlier games | C, I (masked) |
| Crew tendencies and interactions | `REF_CREW_*_TENDENCY_BEFORE`, `REF_CREW_*_X_*_BEFORE` | `referees/referee_tendencies.py`; `intermediate_referees.py` for snapshots | `nba_refs`, team games, odds | C, I (masked, interactions against the snapshot spread) |

### Market: closing dataset

| Family | Example columns | Module | Inputs |
|---|---|---|---|
| Per-book closing and opening lines, prices, normalised lines | `ODDS_TOTAL_LINE_<book>`, `ODDS_SPREAD_LINE_HOME_<book>`, `ODDS_ML_PROB_*` | `postgre_db/odds/merge_odds_data.py`, `data_processing/odds/*` (normalisation, book combination) | `odds_sportsbook`, `odds_yahoo` |
| Engineered odds (consensus, dispersion, opener move) | `ODDS_SPREAD_CONSENSUS_*`, `ODDS_SPREAD_CROSS_BOOK_*` | `merged_home_away_data/odds_feature_engeneer.py` | same |
| Rolling team results against the line | `DIFF_FROM_ODDS_LINE_bet365_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME` | `add_features_after_merging.add_betting_stats_differences` | team games, closing lines of earlier games |
| League-wide market regime | `GLOBAL_MARKET_BIAS_30G_BEFORE`, `GLOBAL_MARKET_MAE_7D_BEFORE` | `merged_home_away_data/global_market_features.py` | closing lines of earlier games |
| Implied team points | `IMPLIED_PTS_{HOME,AWAY}_BEFORE` | `add_features_after_merging` | lines |

### Market: intermediate dataset (all from Aiven line-history ticks)

| Family | Example columns | Module |
|---|---|---|
| Snapshot panel per book: raw and centred line, prices, de-vigged probabilities, line age | `ODDS_SNAP_{TOT,SPR,ML}_<BOOK>_*` | `line_history/snapshots.py`, `normalization.py`, `book_merge.py` |
| Movement: from open, trailing windows, velocity, reversals, position in range | `ODDS_SNAP_ML_BET365_MOVE_FROM_OPEN`, `..._MOVE_LAST_<w>` with `..._HAS_WINDOW_<w>` | `line_history/movement_features.py` |
| Cross-book consensus and leave-one-out deviation | `ODDS_SNAP_{MKT}_CONSENSUS_*`, `ODDS_SNAP_{MKT}_<BOOK>_DEVIATION_FROM_CONSENSUS` | `line_history/cross_book.py` |
| Anchor total path | `ODDS_SNAP_TOT_BET365_MINUTES_SINCE_LAST_LEVEL_MOVE`, `..._PEERS_MOVED_ANCHOR_STILL_60` | `line_history/anchor_total_path.py` |
| Prior-game line dynamics (how this team's earlier games re-priced) | `ODDS_TOT_N_MOVES_OPEN_TO_CLOSE_LAST_5_BEFORE_TEAM_HOME`, `ODDS_{SPR,ML}_*` | `line_history/history_features.py` |
| **G2** market reaction to injury news | `ODDS_SNAP_NEWS_{TOT,SPR,ML}_*`, `ODDS_SNAP_NEWS_HAS_RECENT` | `line_history/market_dynamics.py` |
| **G3** cross-market coherence | `ODDS_SNAP_XMKT_*` | `line_history/market_dynamics.py` |
| **G4** walk-forward Ridge expected move to close | `ODDS_LINE_HIST_RIDGE_EXPECTED_{TOTAL,SPREAD,ML}_MOVE_TO_CLOSE` | `create_training_data/historical_ridge_movement.py`, `line_history/walk_forward.py` |

`create_training_data/intermediate_market_dynamics.py` wires G1–G3 into the
builder.

## 5. How the time cutoffs are enforced

| Rule | Where |
|---|---|
| `_BEFORE` in the name means it can be computed before the game; the column gates keep only `_BEFORE`, allow-listed or odds-shaped columns and raise on leftovers | `select_train_columns.py`, `select_intermediate_columns.py` |
| Historical aggregates use `shift(1)` before rolling | `team/rolling.py` and the other rolling builders |
| Snapshot ticks: only `minutes_before_tip >= horizon`, with a 5-minute margin before tip | `line_history/snapshots.py`, `line_history_aiven/fetch.py` |
| Injury reports: only reports observed strictly before the snapshot timestamp | `injury_report_aiven/fetch.py` |
| Referees: NaN for snapshots before 09:00 ET on game day | `intermediate_referees.mask_referees_before_release` |
| Walk-forward fits (G2 β, G4 Ridges): train only on games that tipped off before the row's timestamp, from the current and previous season | `line_history/walk_forward.py` |
| Closing lines are kept out of X by physically separate files; the builder raises if kept features can reconstruct the close | `create_intermediate_line_df._build_scoring_frame`, `audit_closing_line_reconstruction` |
| Post-game membership leaks, such as rotation-depth counts, are named centrally and blocked | `src/nba_ou/config/leakage.py` |
| Missing history: season-to-date, then previous regular season, then a neutral value; structural absence uses a flag plus 0 | `feature-engineering` skill, rule 3 |

## 6. Open work

- G1–G4 are implemented on `feat/intermediate-market-dynamics`, but the full
  intermediate dataset has not been rebuilt with them yet and they have not
  been through an ablation. See `docs/intermediate_market_dynamics_plan.md` §6.
- A fixed-clock prediction time (such as 09:05 ET) was analysed but not
  implemented. The snapshot panel and the training pipeline key rows on a
  shared minutes-before-tip value, while injuries, referees and G1–G3 already
  cut on an absolute timestamp.
