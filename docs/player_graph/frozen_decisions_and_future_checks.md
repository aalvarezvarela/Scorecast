# Player graph: frozen decisions and future checks

Research log for the 2_7 player graph ([plan](../player_interactions_2_7_plan.md),
code in `src/nba_ou/data_processing/player_graph/`). It records choices that
were **deliberately frozen** for v0 instead of being tuned, with the evidence
behind them, and checks worth doing later.

This is not a TODO list and nothing here blocks implementation. Its purpose is
to remember why each value was picked, so it can be revisited on purpose rather
than drift. Add an entry whenever a value is frozen, and move a check to
"Done" with its result when it is run.

**Evaluation hygiene.** The 2_7 XGBoost walk-forward starts in 2019-20. Any
stage 1 hyperparameter is chosen on **2018-19 or earlier** and then not touched
while 2019-20 onward is being evaluated. Record the selection seasons with every
tuned value.

## Frozen for v0

| Component | Decision | Value | Why / evidence | Revisit when |
| --- | --- | --- | --- | --- |
| `as_of` | Cutoff for game data | `game_date < D`, time of day ignored | Same rule as 2_6; a game finished earlier on D is excluded | Only if intraday snapshots need same-day games |
| `as_of` | Injury cutoff | Timestamp, done upstream in SQL | `injury_report_aiven.fetch` already returns the last report before tip / snapshot | — |
| Data | First season | **2016-17** (`as_of.FIRST_SEASON`); `PointInTimeData.load` refuses earlier seasons | Stints exist from 2012-13 (audited 2026-10-08, same standard), but before 2016-17 there is no matchup tracking at all and adding them means infrastructure work before the encoder has shown any signal; 2016-25 is enough volume to test the approach first | Phase 6 (see checks: older seasons) |
| Data | Player box scores | DB from 2018-19 on; **2016-17 and 2017-18 from the season CSVs** (`data/season_games_data/nba_players_YYYY_YY.csv`, the files the DB was loaded from). Whole seasons only: a season the DB holds is never mixed with CSV rows; every row carries `BOX_SOURCE` (`db` / `csv`) | The DB deletes box scores before 2018-19 (`delete_old_data.py`), the CSVs do not. On 2018-19, which both hold, the CSV equals the DB row for row (35,845 rows; minutes, every stat, `USG_PCT`, `START_POSITION`). The only difference is the CSV's turnover column name, `TO` | — |
| Data | Box scores for game-graph rosters | **DB rows only** (`PointInTimeData.box_scores_2_6`), as 2_6 reads them; positions, node profiles and usage read all rows | v0 must reproduce 2_6 exactly; with CSV rows, a player who missed 2018-19 would get 2017-18 recent minutes. Re-verified after the change (see the integration check) | Phase 4A (the minutes provider may use the full history) |
| Data | 2016-17 stint points | **Store rebuilt 2026-10-08** (`build_lineup_stints.py --season 2016 --force`; old store in `data/lineup_stints_backup/season=2016_pre_rebuild_20261008`) | It had been built from an archive version with placeholder "0" scores, rebuilt later without rebuilding the stints: 6,211 stints with negative points and 6,919 impossible ones in 1,034 of 1,304 games (-131 … +262), yet every game's total reconciled. After: same 39,421 stints, boundaries, lineups, FGA / FG3A / FTA / OREB / DREB / TOV and possessions (100%); points changed in 26,144 stints, totals identical, no stint below 0 | — |
| Data | Stint point validation | `validate_game_stints` also rejects `points_anomaly`: a stint below -10 points, more than 10 points above `3·FGA + FTA`, or more than 3 negative stints in a game (`POINT_ANOMALY_LIMITS`); `scripts/lineups/audit_stint_points.py` reports the stored games | Opposite errors cancel in the game total. Small negative stints (1-2 in ~0.1% of games, down to -7) pass. Audit 2012-25 after the change: 0 games outside the limits. The 2 stored games that were outside (built before the check) were rebuilt from the current PBP: same values, and **not** feed corrections but stale scores on non-scoring events (`0021800927`: period-start events carrying 17-30 after a 26-37 quarter; `0022101075`: an instant-replay "support ruling" carrying 20-21 at 73-62), so the validator now rejects both and the store excludes them (`read_stints` reads status `ok` only). Stored 2_7 tables built before (pair_game, expected_guard, node profiles, oracle) still include them; negligible, refreshed on the next full rebuild | See checks |
| Data | 2_6 ratings | **Not rebuilt**: 2_6 stays frozen and reproducible from its cache | Its cache was fitted on the corrupted 2016-17 points, so its offense / defense ratings of 2017-18 and 2018-19 inherit the corruption; with the 180-day half-life the 2016-17 weight is ≈ 0.03 by 2019-20, where 2_6 is evaluated. Pace is unaffected (possessions do not use the score) | If 2_6 is ever rebuilt |
| Data | Preseason and All-Star rows | Excluded from position profiles | Do not describe a role | — |
| `pair_game` | Guarding rate denominator | Our stints' co-floor seconds | NBA `pct_total_time_both_on` uses ≈ 0.39 × co-floor time (SD 0.056, 2023-24) and is rounded to 3 decimals | — |
| `pair_game` | Matchup rows with no shared stint | Kept, `cofloor_seconds` 0, rate NaN | 3-151 rows a season (< 0.06%) | — |
| `pair_game` | Rate above 1 | Clipped at 1, raw seconds kept | 10-31 rows a season, up to 34 s over | — |
| `pair_game` | Shared floor, no matchup row | Real zero | ~6.5% of directed pairs | — |
| Positions | Representation | Soft G/F/C probabilities | Avoids a hard threshold between starts and profile | Phase 4B |
| Positions | Blend | `(starts + c·q) / (n_starts + c)`, `c = 5` pseudo-starts | Not tuned | See checks |
| Positions | Profile classifier | Multinomial LR on per-36 AST, OREB, DREB, BLK, STL, FG3A, FGA, PF; trained on players with ≥ 10 starts; rates shrunk with 200 league-average minutes | Out of sample (next-60-day start position): 80-93% accuracy, 60-84% for < 10 prior starts | See checks |
| Positions | No box scores and no starts | League share of starts per position | Only debuts and the first games of 2016-17 since the CSV backfill | — |
| Positions | Window | 3 seasons before D | Same as the guarding history | — |
| `expected_guard` | Shrinkage strength `k` | **300 co-floor seconds** | Chosen on **2018-19 only**. Re-swept 2026-10-08 after the box-score backfill changed the 2018-19 position prior: weighted TVD 0.2743 (k=150), 0.2737 (200), **0.2736 (300)**, 0.2741 (400), 0.2760 (600), 0.2827 (1200); 300 is now the minimum outright and stays frozen. First sweep (no box scores before 2018-19): 0.2755 (200), 0.2756 (300), 0.2786 (600), 0.2860 (1200) | After the main out-of-sample evaluation |
| `expected_guard` | Temporal decay | Half-life 365 days, applied to matchup and co-floor seconds before summing | Not tuned | See checks |
| `expected_guard` | History window | 3 seasons (1,095 days) | Not tuned | See checks |
| `expected_guard` | Position prior | `p_i' R p_j`, `R` = decayed 3×3 rate by attacker × defender position, positions as of D | Diagonal-dominant as expected (C on C 0.22, G on G 0.10 as of 2024-06-06) | Phase 4B |
| `expected_guard` | Pair history across teams | All history counts equally, whatever team the defender was on | Simplest | See checks |
| `expected_guard` | No history at all | `r_prior` and `r_hat` NaN; graphs use uniform shares | First days of 2017-18 only | — |
| `expected_guard` | Stored output | Unnormalized `r_hat` plus evidence columns; `m_ij` computed in the graph | Keeps magnitude and confidence | — |
| `expected_guard` | Stored parameters | One `season=YYYY.json` next to each `season=YYYY.parquet`; readers refuse to mix seasons built with different parameters | A partial rebuild or `--reshrink` cannot leave a global record that misdescribes other seasons | — |
| Benchmark | Undefined predictions | An attacker whose rates are not all finite, or sum to 0, falls back to the constant rate (shares ∝ co-floor time), counted in a `fallback` column | Mirrors the graph's uniform fallback; otherwise an undefined prediction scored a perfect 0 (k = 0: 66.5% of attacker-games fall back, TVD 0.3175) | — |
| `expected_guard` | Which pairs get a row | Every pair that shared a stint, from the stints (not `pair_game`) | An expected rate needs no tracking of the game itself, so 2016-17 and stint-only games are covered | — |
| Stint graph | Representation | Fixed-shape arrays per batch (10 nodes, 20 teammate, 25 opponent, 50 guard edges), no graph objects | Every stint has the same shape; PyG vs plain tensors is still open (phase 6) | Phase 6 |
| Stint graph | Guard weight | `m_ij = r_hat_ij / Σ r_hat_il` over the **five defenders on the floor** | Plan phase 2 | — |
| Stint graph | Unknown rates | Uniform 0.2 for an attacker unless all five rates are known; flagged `guard_fallback` | Only when there is no history at all (2016-17, first days of 2017-18) | If partial gaps appear |
| Stint graph | Guard-edge attributes | `r_hat`, `prior_weight`, `log1p(hist_cofloor_seconds_decayed)`, `has_pair_history`, `fallback` | So the encoder can tell long history from prior | See checks |
| Stint graph | Unknown attributes | All attribute arrays finite: unknown `r_hat` stored as 0 with `guard_r_hat_known` False, `prior_weight` 1, exposure 0 | Arrays go into a network as they are; the mask keeps "unknown" distinct from "zero" | — |
| Stint graph | Message passing | Templates hold each undirected pair once; `message_passing_edges()` lists teammate and opponent edges both ways, with `source_edge` to gather weights; guards stay defender → attacker, `guarded_by` optional | Undirected relations must pass messages both ways | Phase 6 (whether to use `guarded_by`) |
| Stint graph | Teammate / opponent weight | Stint seconds (equal within a stint) | On a training graph every pair shares the whole stint | Familiarity attribute pending (node/pair tables) |
| Stint graph | Stint duration | **Exposure / loss weight only, never an encoder input**. Short stints are kept, not dropped; their duration decides how much they count | Duration describes how much of the outcome was observed, not who played. The smoke test feeds the encoder profiles, `is_home` and guard attributes only (teammate / opponent messages are means, not seconds-weighted) | Phase 6 (pace weights, see checks) |
| Stint graph | Labels | Per offensive side: pts/poss, TOV/poss, poss/48, 3PA/FGA, FTA/FGA, OREB/(OREB + opp DREB); weight = possessions; NaN when the denominator is 0 | Plan phase 6 | Phase 6 |
| Stint graph | Non-positive possessions | Weight 0 and all labels NaN for a side with possessions ≤ 0 | The estimate goes negative in stints of a few seconds (an offensive rebound of the previous stint's miss is subtracted): 8-29 sides a season, down to -1.12; 5-6.5% of sides have exactly 0 | See checks |
| Stint graph | Input contract | Reject any `pair_game` observed column or betting-named column (`TOTAL_LINE`, `SPREAD`, `MONEYLINE`, `ODDS`, `LINE_ERROR`) | Plan principle 7 and stage 1 rule | — |
| Overlap | Provider | **Pair lift** (v0); independence (`min_i·min_j/48`) kept as the baseline / ablation | Overlap weights teammate/opponent edges and enters every guard share (`m_ij ∝ r_hat·E[overlap]`) | See checks |
| Overlap | Per-game table | Every pair of players who both played a game, including pairs that never shared the floor (shared 0); independence uses the game's real length (overtime) | The zeros are what a lift below 1 must learn. Exact invariants: shared time sums to 4× (teammates) and 5× (opponents) the player's seconds | — |
| Overlap | Shrinkage target | The **relation's** decayed league lift, not 1 | Teammate overlap under independence is biased high (only 4 teammate slots): actual / independent = 0.903 (2018-19), 0.906 (2023-24). Opponents are 1.000 by construction | — |
| Overlap | Shrinkage strength | **k_teammate = 150, k_opponent = 1,200** independent-overlap seconds | Chosen on **2018-19 only**. Teammates: TVD 0.218 (independence) → 0.172, flat for k = 0-300. Opponents: 0.162 → 0.1560 at 1,200 (0.1563 at 2,400), pair history alone is worse (0.176): opponents meet a few times a season | After the main evaluation |
| Overlap benchmark | Undefined predictions | Same sample for every estimator (player-games with actual shared time); an undefined lift falls back to independence and a player whose predicted shares sum to 0 gets independence shares, counted in `fallback` (share of player-games) | Undefined predictions used to be dropped, which could score 0 on a shrunken sample. v0 and all baselines: fallback 0, numbers unchanged | — |
| Overlap | Decay and window | Half-life 365 days, 3 seasons | Same as `expected_guard`; not tuned | See checks |
| Overlap | Cap | `E[overlap] = min(lift·min_i·min_j/48, min_i, min_j)` | Plan check: never above either player's minutes | — |
| Game graph | Rosters, availability, minutes, scenarios | 2_6 unchanged: `game_nights` / `build_player_nights`, `chance_out`, `enumerate_scenarios` (≤ 3 uncertain per team), `allocate_minutes` (recent average rescaled to 240); scenarios where a team has nobody to play are dropped | Reproduces 2_6 exactly (see below) | Phase 4A |
| Game graph | Injury report coverage | Required argument; a game without both teams' reports before the cutoff gets **no graphs** (features NaN), counted in `metadata["skipped_games"]` | As 2_6: everybody "available" without a report is a falsely certain projection. Oct-Dec 2018: 435 games skipped, exactly the 435 the 2_6 file leaves NaN | — |
| Game graph | Zero-minute roster players | Not nodes, no edges | A recent average of 0 (only coach's-decision DNPs) gave 0/0 guard shares; 132 such nodes in December 2023 alone. They add nothing to 2_6's sums | — |
| Game graph | Edge attributes per relation | `lift`, `lift_prior_weight` on teammate/opponent; `r_hat`, `r_hat_known`, `guard_prior_weight` on guards; non-applicable columns NaN; `weight`, `expected_overlap`, `log_exposure` always finite | One edge table for all relations | Phase 6 adapter |
| Game graph | Full-health counterfactual | Its own graph (`scenario_id = -1`), every roster player available | Absence features compare tonight with it | — |
| Game graph | Guard shares | `m_ij = r_hat_ij·E[overlap_ij] / Σ_l r_hat_il·E[overlap_il]` over the defenders playing in the scenario; unknown rates → shares ∝ overlap (`fallback`) | Same form as the guard benchmark's implied share; an absent defender leaves the denominator, which reassigns his attackers with no extra rule | Phase 4B |
| Game graph | Storage | Tables: `scenarios`, `nodes`, `edges` (teammate and opponent stored once, guards directed) | Node count varies per game, unlike stints | Phase 6 |
| Node profiles | Content | Per-36 rates (PTS, FGA, FG3A, FTA, OREB, DREB, AST, TOV, STL, BLK, PF), TS%, 3P%, `USG_PCT`, recent minutes (last 10 games played), start share, soft G/F/C | Plan phase 2 node features; initial inputs of the player embeddings | Phase 6 |
| Node profiles | Temporal rule | Box scores strictly before the as-of date (`as_of` view); preseason and All-Star excluded; window 2 seasons, half-life 180 days | Profiles should track the current role; not tuned | See checks |
| Node profiles | Shrinkage | Toward the league: 200 minutes for rates and usage, 100 shooting attempts for TS%, 50 three-point attempts for 3P% | Low-minute and replacement players are the noisiest and matter most for the absence counterfactual | See checks |
| Node profiles | Confidence | `minutes_window`, `games_window`, `minutes_decayed` (effective sample), `prior_weight = 200 / (minutes_decayed + 200)`, `days_since_last_game`, `games_in_data`, `has_box_history` | So the encoder can tell a measured profile from a prior | — |
| Node profiles | No box scores | League rates, `prior_weight` 1, `has_box_history` False (debuts, and the first games of 2016-17 where the data starts) | A graph can always be built | — |
| Node profiles | Storage | `data/player_graph/node_profiles/season=YYYY.parquet` keyed `(as_of_date, player_id)`, for every game date and the day before (intermediate history dates); parameters in `season=YYYY.json` | Joins to any graph through its `as_of_date` | — |
| Node profiles | Player set | Everyone seen in the window in counted box scores (no preseason / All-Star, positive minutes), matchups or stints, **plus the date's own stint players** (debuts, first games in the data), whose profile is still read strictly before the date | Preseason-only players would only add prior rows; without same-day players 1.8% of 2016-17 stint nodes had no profile. Coverage of stint nodes: 100% every season; with box history 98.3% (2016-17), 99.5-99.7% (2017-25). Before the CSV backfill: 0% in 2016-17 and 2017-18, 98.1% in 2018-19 | — |
| Oracle | Purpose | **Diagnostic only**: the game's own minutes, rotation and guarding never produce a feature or train a model | Plan principle 7 | — |
| Oracle | Factor decomposition | `m_ij ∝ r_ij·overlap_ij`. Oracle rotation (4A) replaces node minutes **and** overlap with actual values, keeping `r_hat`; oracle guards (4B) replaces only `r_hat` with the observed rate, keeping projected minutes and expected overlap; oracle both replaces all | Keeps 4A (who plays, how much, with whom) and 4B (given they share the floor, who guards whom) apart; observed overlap in the guard oracle would mix them | — |
| Oracle | Actual minutes and overlap | **Regulation only** (periods 1-4) from the stints | Minutes sum to 240 per team and overlaps are consistent with them; an overtime game does not reveal its overtime | — |
| Oracle | Observed rate | `pair_game.guard_rate` (whole game: matchups have no timestamps); a pair without one keeps `r_hat` | With both oracles the shares are exactly the observed matchup distribution in games without overtime; with overtime, whole-game rates times regulation overlap only approximate it | — |
| Oracle | Full-health counterfactual | No actual version, so it keeps projected minutes and expected overlap: v0's for the rotation oracle; the guard oracle's (observed rates where a pair has one, `r_hat` for an absent defender) for the guard and both oracles | An absent player has no observed assignment; each impact changes only the factors its oracle changes | — |
| Oracle | Readouts | R1 = 2_6 projection (only minutes move it); R2 = R1 with each team's defense replaced by `5·Σ_i w_i Σ_j m_ij·def_j`, `w_i ∝ min_i·usage_i` | With shares proportional to floor time R2 equals R1 (tested), so R2 − R1 is who guards whom | — |
| Oracle | Usage | As-of `USG_PCT`, minutes-weighted over 365 days, shrunk to the league with 200 minutes; identical in every version | Only the studied weights may change between versions | — |
| Oracle | Measures | Level calibrated with 2_6's `walk_forward_offset` (2018-19 built only to warm it up); total MAE; edge vs `LINE_ERROR` (corr, slope, hit rate); absence impact vs `LINE_ERROR`; all games, \|v0 R1 impact\| > 3, key defender out (top-2 RAPM defender of his team's ≥ 20-minute rotation, expected absent ≥ 50%) | Plan phase 2 | — |
| Game graph | Prediction times | Closing and the 17 intermediate horizons (T-0 … T-1080) | — | — |
| Snapshots | History date | As 2_6: the earlier of the game date and the cutoff's Eastern date, a cutoff before 05:00 ET counting as the previous day (`snapshot_history_dates`); rosters, ratings, guard and overlap history and positions are all read as of it; only availability is read at the UTC cutoff | Box scores and stints have dates, not publication times. At T-960 / T-1080 most snapshots read the previous day | — |
| Snapshots | Rates per date | Expected lifts and guarding rates computed once per history date for every horizon's rosters | They depend only on the date and the rosters; ~17× less work | — |
| Phase 3 | Zero-prior control | **Shared `lambda_offdef` = 3,000, `lambda_pace` = 10,000** (half-life 180 d, stints from 2016-17 on the rebuilt store, 2_6's solver unchanged). Phase 3's profile prior must beat this, not only 2_6 | Chosen on **2018-19 only** by weighted squared error on the next stints, over a log grid 100 … 10⁶ plus infinity (2_6's grid was 10 … 1,000); both optima interior. A first run on the corrupted 2016-17 store had picked 10,000 / 10,000. 2_6 itself keeps 1,000 / 30,000 | After the main evaluation |
| Phase 3 | What the profile prior must show | **Incremental value over a well-regularized RAPM** (the 3,000 / 10,000 control), not the repair of a failure: on clean data thin players do not degrade the zero-prior projection. Where it can add: profile information for thin players and debutants (who are below average on offense), and the level of pace for lineups with thin players | Step 1 and 2a on the rebuilt store; the earlier "RAPM fails with thin samples" came from the corrupted 2016-17 points | — |
| Phase 3 | Profile prior `f` (v0) | Three weighted linear ridges (offense, defense, pace) on 22 standardized profile features (per-36 rates, TS%, 3P%, usage, recent minutes, start share, soft G/F/C, log games in data, profile `prior_weight`, `has_box_history`); target `t = rating / s`, weight `s`, `s ≥ 0.02`; refit monthly on every checkpoint so far (scaler included); ridge strength **o 0.1, d 100, pace 0.1** | Strength chosen by player-grouped 5-fold CV on pairs before 2018-10-01 only, then frozen (o and pace are flat for 0.01-1). The rating's exposure is a weight, never a feature. Age, draft position, listed position, height / weight would be added if a source appears | Step 3 |
| Phase 3 | Debut prior (v0) | A player with no box score before the date gets, per rating, the reliability-weighted mean `t` of earlier debutants (first game after 2016-12-01) in their first 20 games; as soon as he has a profile, `f` | No position before a first game; one value per rating | Step 3 |
| Phase 3 | Profile-prior penalties | **Shared `lambda_offdef` = 3,000, `lambda_pace` = 10,000** toward `beta0`, the same as the zero-prior control | Retuned on **2018-19 only** with the same log grid and criterion; both optima interior and flat nearby (efficiency 10,000: -0.43 vs 3,000; pace 30,000: -0.05). Separate offense / defense penalties add +0.21 ± 0.34 | After the main evaluation |
| Phase 3 | **Final pipeline** (closed 2026-10-08) | Offense and defense: profile prior, lambda 3,000; pace: zero prior, lambda 10,000; `f` 0.1 / 100 / 0.1, monthly expanding refit, debut prior, 0 only without a profile | 2019-25 out of sample: efficiency +4.27 ± 0.48 over the zero prior (all seasons, larger for low-sample players and debuts, MAE better); pace -0.11 ± 0.05. The split was chosen after seeing 2019-25 (see the methodological note) | Never on 2019-25 |
| Phase 4A | Redistribution shares C (**frozen**) | **v1 structural features, share half-life 41 team games, kappa 30 vacated minutes, rotation threshold 15 min, no sharpening (temperature 1)**; baseline half-life 15 team games | Chosen on **2018-19 only** by a small predefined one-at-a-time search on the minutes MSE of every player who played (same rows for every configuration); only the share half-life moved it (37.59 → 37.15). v2 features tie and stay optional. Any temperature above 1 is worse, monotonically (step 2c) | After the main evaluation |
| Phase 4A | **Minutes provider v1 frozen** (2026-10-09) | Expanded roster (rule below), B, C (v1, 41, 30, 15, tau 1), A (logistic), reconciliation 0-48 / 240, injury-report scenario integration; see "Phase 4A: minutes provider v1 frozen" | Development numbers in steps 1-6 | Never on 2019-25 |
| Phase 4A | Expanded roster after long absences | **Kept beyond 10 missed team games only if the team's report listed him in one of its last 5 games** (as of the game; tonight's pre-game report included) | Removes ~87% of the minutes v1 gave to waived / unlisted players and improves every comparable error on 2018-19 (report scenarios: player MAE 4.751 → 4.656 on the same rows); costs 34 of 140 real long returns, never listed | If a roster / transactions feed appears |
| Phase 3 | Pseudo-target protection | Weighted ridge with `w = s`, pairs with **`s` < 0.02 left out** (13% of the pairs, 0.4% of the weight); no robust loss | On the clean store `s · Var(t)` is flat and the only extremes are one-game players; a robust loss on `sqrt(s) · (t - f(x))` is added only if strong outliers reappear | If outliers reappear |
| Smoke-test GNN | Library | **PyTorch only for the phase 2 smoke test**: optional Poetry group `graph` (`torch 2.9.1`, the version the lock already resolved through `timeseries`; `poetry install --with graph`), no PyTorch Geometric | Stint graphs always have 10 nodes and fixed edge templates, so dense tensors suffice; dependency and code stay minimal while only the plumbing is tested | **PyG decision deferred until the phase 6 architecture is defined** (game graphs: variable node counts, scenarios, edge types and attributes, batching) |
| Smoke-test GNN | Scope | Plumbing only: tables → tensors → message passing → node embeddings → pooling → prediction head → loss / backprop → checkpoint save / reload. Not the phase 6 architecture, no claim about signal | Plan phase 2 | Phase 6 |
| Smoke-test GNN | Data | 2018-19 only (train 2018-10-16 → 2019-01-31, held out → 2019-03-31); the script refuses dates from 2019-07-01 | Evaluation hygiene: 2019-20 on is the 2_7 evaluation | — |

## v0 benchmark (the reference phase 4B must beat)

`python scripts/player_graph/evaluate_expected_guard.py --seasons 2019-2025`
(provider v0, k = 300, half-life 365 d, window 3 seasons). Weighted TVD between
expected and observed in-game guarding shares, 188,238 attacker-games:

| Estimator | All | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constant rate (co-floor only) | 0.3605 | 0.3947 | 0.3809 | 0.3561 | 0.3561 | 0.3581 | 0.3452 | 0.3390 |
| Position prior only | 0.2994 | 0.3256 | 0.3123 | 0.2972 | 0.2990 | 0.3004 | 0.2863 | 0.2799 |
| Pair history only | 0.2713 | 0.2882 | 0.2807 | 0.2737 | 0.2685 | 0.2658 | 0.2617 | 0.2638 |
| **v0 (k = 300)** | **0.2571** | 0.2751 | 0.2666 | 0.2576 | 0.2553 | 0.2542 | 0.2470 | 0.2471 |

Since the box-score backfill (2026-10-08). Before it, 2019-20 read 0.3267
(position prior), 0.2883 (pair history) and 0.2753 (v0), and the position prior
was 0.3125 in 2020-21 and 0.2996 overall; nothing else moved. That is the
expected footprint: positions use a 3-season window, so only seasons whose
window reaches 2016-18 can change. Pair by pair, every history column (`n_games`,
raw and decayed co-floor and matchup seconds, `has_pair_history`) and
`prior_weight` are identical before and after in every season; only `r_prior`
and `r_hat` move, by a mean 0.015 / 0.010 (2017-18), 0.0035 / 0.0019
(2018-19), 0.0009 / 0.0005 (2019-20), 0.0002 / 0.0001 (2020-21), and are
bit-identical from 2021-22 on. (The pair-history-only baseline moves too
because it shrinks with k = 1e-6, so a pair with no history still gets
`r_prior`.)

Error by the attacker's share-weighted `prior_weight` rises monotonically, so
the evidence columns carry information:

| `prior_weight` | ≤ 0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TVD | 0.205 | 0.234 | 0.250 | 0.267 | 0.283 | 0.295 | 0.322 |

Part of this error can never be removed: observed shares react to the game
(foul trouble, a hot scorer drawing a different defender), so 0 is not the
target.

## Overlap benchmark (v0 reference)

`python scripts/player_graph/evaluate_overlap.py --seasons 2019-2025 --k-grid 150 1200`:
lifts as of each game, scored against the time pairs actually shared given
the players' actual seconds, 8,854 games. Lower is better.

| Relation | Metric | Independence | Pair history only | **v0 (k per relation)** |
| --- | --- | --- | --- | --- |
| Teammate | TVD | 0.1976 | 0.1647 | **0.1644** |
| Teammate | Relative absolute error | 0.4081 | 0.3334 | **0.3335** |
| Opponent | TVD | 0.1499 | 0.1679 | **0.1462** |
| Opponent | Relative absolute error | 0.2998 | 0.3326 | **0.2929** |

The lift matters for teammates (-17% TVD) and helps opponents a little (-2.5%);
pair history alone is worse than independence for opponents.

## Integration check: 2_6 reproduced from the game graph

`python scripts/player_graph/check_game_graph_readout.py --from … --to …` reads
the 2_6 projection off the v0 graphs' node minutes and compares it with 2_6's
`project_lineup_games` on the same rosters and with the 2_6 closing file
(`closing_line_data_2_6_20261003`):

| Window | Games | vs `project_lineup_games` (max \|diff\|) | vs file `LU_ABSENCE_IMPACT_PTS_BEFORE`, `LU_PROJ_POSS_BEFORE` |
| --- | --- | --- | --- |
| 2023-12-01 → 2023-12-31 | 208 | 1.1e-13 | 100% equal |
| 2019-10-22 → 2019-11-15 (season start) | 174 | 0 | 100% equal |
| 2018-10-16 → 2018-12-15 (no reports yet) | 435, all uncovered | — | the same 435 games NaN in the file and without graphs |
| 2019-01-01 → 2019-02-15 | 318 | 1.1e-13 | 100% equal |
| 2018-12-17 → 2019-01-31 (after the box-score backfill; first covered games) | 327 | 1.1e-13 | 100% equal |

Intermediate (`--intermediate …`), against `intermediate_line_data_2_6_20261003`
with each snapshot's report state and history date; games without a graph
are exactly the file's NaN rows at every horizon:

| Window | Horizons | Snapshots with graphs | vs `project_lineup_games` | vs file |
| --- | --- | --- | --- | --- |
| 2023-12-01 → 2023-12-31 | 0, 360, 720, 1080 | 208 / 168 / 152 / 150 | ≤ 1.4e-13 | 100% equal |
| 2019-10-22 → 2019-11-10 | all 17 | 139 (T-0) … 89 (T-1080) | ≤ 5.7e-14 | 100% equal |
| 2019-10-22 → 2019-11-05 (after the box-score backfill) | 0, 360, 1080 | 102 / 88 / 68 | ≤ 5.7e-14 | 100% equal |

The injury report digest matched the build's (`293dfbebdb5b2820`).

## Oracle: how to read it

The oracle is **not a strict mathematical ceiling**:

- It can **overstate** what is attainable: actual minutes and assignments react to the game itself (blowouts and garbage time, foul trouble, a hot scorer drawing a different defender).
- It can **understate** what matchups are worth: R2 uses 2_6's stint RAPM `def_j`, which credits the five defenders equally, not the matchup-adjusted defensive rating of phase 4B.
- The absence counterfactual has no observed version: for an absent defender the full-health graph keeps his expected `r_hat`.

### v0 oracle result (2026-10-08)

`python scripts/player_graph/oracle_ceiling.py summarize --seasons 2019-2025`:
8,759 closing games with all four versions (2018-19 only warms up the level
calibration). Paired difference vs v0 on the same games, mean ± SE; a negative
absolute error and a positive hit rate (side of the closing line) are better.

| Slice | Readout | Oracle | \|error\| | Hit rate |
| --- | --- | --- | --- | --- |
| All (8,759) | R1 | rotation (4A) | **-0.090 ± 0.026** | **+1.40 ± 0.47 pp** |
| All | R2 | guards (4B) | +0.001 ± 0.006 | -0.10 ± 0.24 pp |
| All | R2 | both | -0.082 ± 0.027 | +1.06 ± 0.47 pp |
| All | R2 − R1 | v0 | -0.003 ± 0.006 | |
| \|v0 impact\| > 3 (2,281) | R1 | rotation | -0.094 ± 0.053 | +0.35 ± 0.90 pp |
| \|v0 impact\| > 3 | R2 | guards | -0.016 ± 0.013 | 0.00 ± 0.49 pp |
| Key defender out (4,005) | R1 | rotation | **-0.172 ± 0.039** | **+1.60 ± 0.70 pp** |
| Key defender out | R2 | guards | -0.008 ± 0.010 | +0.47 ± 0.37 pp |

v0 itself: total MAE 14.53, hit rate 50.6% against the closing line.

The stored oracle tables predate the box-score backfill and were not rebuilt.
For 2019-25 the backfill only moves `r_prior` in 2019-20 and 2020-21 (mean
|Δ| ≤ 0.0009, see the benchmark) and the 2018-19 warm-up of the level
calibration; the as-of usage window (365 days) never reaches the CSV seasons
from 2019-20 on. Rebuild them with the 4B rerun.

Reading:

- **Rotation (4A) has measurable headroom**, about twice as large when a key
  defender is out. It is an upper bound that includes reactive information
  (garbage time, foul trouble), so a projection model will recover only part
  of it; the target is who replaces whom in absence games.
- **Guard weights (4B) have none in this readout**: even the observed
  assignments do not move R2, and R2 ≈ R1 in v0. With stint RAPM `def_j`,
  reweighting the same five defenders changes little. Any 4B value must come
  from new information, i.e. the **matchup-adjusted defensive rating**, not
  from better assignment weights. v0 guard weights stay as they are.

Decisions taken from it (2026-10-08):

- 4A focuses on **absence redistribution** (who absorbs an absent player's
  minutes, and how rotation and overlap change), not on fine-tuning minutes in
  normal games.
- No more work on `expected_guard` weights (k, half-life, window) until
  player-vs-player information is shown to add signal. What was tested is
  narrow: guard weights add nothing **when they only weight a scalar RAPM
  defensive rating**. The next 4B test is the matchup-adjusted defensive
  rating, with R2 and the oracle rerun on it. Guard edges stay as structure for
  the FM / GNN, where they can interact with attacker and defender embeddings
  and the rest of the floor.

## Smoke-test GNN (phase 2, 2026-10-08)

`python scripts/player_graph/smoke_test_gnn.py` (module
`player_graph/smoke_gnn.py`, tests `tests/test_player_graph_smoke_gnn.py`).
Model: input MLP, 2 relational layers (mean teammate and opponent messages, guard
messages conditioned on the edge attributes and weighted by `m_ij`), mean pooling
of each five, one head per offensive side; 15,366 parameters, dim 32. Inputs:
27 node features (the as-of profile on the game date, counts as `log1p`, plus
`is_home`) and 7 guard attributes, standardized on the training stints.

| Check | Result |
| --- | --- |
| Tables → tensors | 23,730 train / 11,666 held-out stints; nodes (n, 10, 27), guards (n, 50, 7), labels (n, 2, 6); every input finite, every node with a profile |
| Label coverage | 94.9% (per-possession labels, pace); 84.9% (3PA/FGA, FTA/FGA); 66.2% (OREB rate); 5.1% of sides have no possessions (weight 0) |
| Guard inputs | fallback 0.0%, pair history 75.9% of guard edges (2018-19) |
| Training | 8 epochs in 13 s on CPU; epoch loss 0.9995 → 0.9923 (noisy) |
| Held-out loss / constant | 0.995 (pts/poss), 0.994 (TOV/poss), 1.000 (pace), 0.957 (3PA/FGA), 0.995 (FTA/FGA), 0.993 (OREB rate) |
| Player order within a side | max \|diff\| 2.4e-07 after reordering the home lineups |
| Checkpoint | `data/player_graph/smoke_gnn/checkpoint.pt` (73 KiB): reloaded predictions identical |
| Unit tests | profile alignment, missing profile raises, order invariance, masked labels never enter the loss, every parameter gets a gradient, overfits 8 stints, checkpoint round trip |

Stint-level labels are very noisy, so a ratio just under 1 is all a tiny model
on three months can show; 3PA/FGA is the one label the node profiles clearly
inform. The pace label has a problem of its own (see the stint graph checks).

## Phase 3, step 1: how 2_6's RAPM does with thin samples (2018-19)

`python scripts/player_graph/rapm_exposure_diagnostic.py --ratings …` (module
`player_graph/rating_diagnostics.py`). Diagnostic only: ratings scored on every
2018-19 stint as 2_6 projects them, against each player's **as-of exposure**
(2_6's own `poss_weight`, decayed offensive possessions, reproduced to 1e-10;
never the scored stint). Data share of a rating: `s = e / (e + lambda)`.
Reference: the same ridge with lambda → ∞ (no player information; the intercept
is the decayed historical mean). `mse_gain` = drop in squared error the ratings
achieve; SE clustered by game. 76,434 offensive stint-sides, 39,759 stints for
pace (2018-19 without the game now rejected as `points_anomaly`).

**Correction (2026-10-08).** The first run scored 2_6's stored cache, whose
ratings were fitted on the corrupted 2016-17 stints (see the Data rows): the
ratings then *lost* to no information (-33 ± 5, -0.9%), worst whenever one
player had less than ~1,000 possessions. That measured the corruption, not the
sample size. The numbers below refit 2_6's penalties (1,000 / 30,000) on the
rebuilt store, into a separate file
(`data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet`); 2_6's own
cache is untouched.

| Efficiency, by the least-exposed of the ten (possession deciles) | Share | MSE gain ± SE |
| --- | --- | --- |
| q1-q2 [0, 299] | 20% | +19.1 ± 7.0, +14.8 ± 8.1 |
| q3-q5 [299, 856] | 30% | +12.2, +25.5, +23.9 (SE ≈ 8) |
| q6 [856, 1,063] | 10% | +0.3 ± 8.2 |
| q7-q10 [1,063, 4,011] | 40% | +26.3, +28.1, +24.9, +21.3 (SE ≈ 8) |
| **All** | 100% | **+19.7 ± 2.5 (+0.54%)** |

| Efficiency, players of the ten with exposure ≤ 1,000 | 0 | 1 | 2 | 3 | 4+ |
| --- | --- | --- | --- | --- | --- |
| Share of possessions | 43% | 28% | 14% | 7.5% | 6.7% |
| MSE gain ± SE | +23.0 ± 3.7 | +20.4 ± 4.7 | +4.2 ± 6.9 | +34.8 ± 8.7 | +11.3 ± 9.3 |

Reading (development season only; nothing here looked at 2019-25):

- On clean data 2_6's ratings beat no information overall (+0.54%) and the
  gain is roughly **flat across exposure**: thin players do not make the
  projection worse than the intercept. Thin offensive fives score ~3-5
  pts/100 less than established ones (bias of the reference -1 to -2.4 vs
  +2.2 / +3.1), and the ratings capture that on average (bias after the
  ratings +0.56 overall, flat).
- **Pace** does not depend on the score and is unchanged: ratings help
  everywhere (+2.0%), but lineups with thin players play faster than predicted
  (bias +3.3 ± 1.4 unrated, +2.2 ± 0.4 at ≤ 100, +0.85 at 100-300, vs +0.5
  established), a level error a profile prior can fix.
- Low-sample slices kept for phase 3: ≥ 1 player ≤ 1,000 (57% of possessions),
  ≥ 1 player ≤ 300 (20%), the count, and overall; offense, defense and pace
  reported separately.

## Phase 3, step 2a: zero-prior control (penalties retuned on 2018-19)

`python scripts/player_graph/rapm_lambda_sweep.py --ratings data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet`
(module `player_graph/rapm_sweep.py`): 2_6's walk-forward ridge accumulated
once and solved for every penalty on each 2018-19 date, scored on the next
stints exactly as in step 1. It reproduces the clean refit at 2_6's penalties
(max |diff| 3e-14) and, at infinity, the decayed mean. 2018-19 without the
game rejected as `points_anomaly`. Rerun on the rebuilt
2016-17 store; the first run (corrupted history) had picked 10,000, the
corruption asking for more shrinkage.

Drop in squared error from lambda → ∞, SE clustered by game:

| Shared `lambda_offdef` | 100 | 300 | 1,000 (2_6) | **3,000** | 10,000 | 30,000 | 100,000 | 10⁶ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Efficiency, all | -0.6 | +10.9 | +19.7 ± 2.5 | **+20.8 ± 1.8** (+0.57%) | +15.7 | +9.0 | +3.7 | +0.4 |
| ≥ 1 player ≤ 1,000 | -12.0 | +5.1 | +17.2 | **+19.1 ± 2.3** | +14.1 | +7.9 | +3.1 | +0.4 |
| MAE (∞: 46.96) | 46.94 | 46.84 | 46.77 | **46.76** | 46.82 | 46.89 | 46.93 | 46.96 |

| `lambda_pace` | 1,000 | 3,000 | **10,000** | 30,000 (2_6) | 100,000 | 10⁶ |
| --- | --- | --- | --- | --- | --- | --- |
| Pace, all | +12.7 | +13.9 | **+14.4 ± 0.9** (+2.13%) | +13.6 | +11.1 | +3.9 |
| MAE (∞: 14.65) | 14.46 | 14.42 | 14.38 | **14.36** | 14.39 | 14.55 |

Best against 2_6's penalties and against no information, per slice (paired):

| Slice | Share | Efficiency 3,000 vs 1,000 | vs ∞ | Pace 10,000 vs 30,000 | vs ∞ |
| --- | --- | --- | --- | --- | --- |
| All | 100% | +1.14 ± 0.82 | +20.8 ± 1.8 | +0.79 ± 0.20 | +14.4 ± 0.9 |
| ≥ 1 player ≤ 1,000 | 57% | +1.92 ± 1.17 | +19.1 ± 2.3 | +0.95 ± 0.31 | +14.3 ± 1.2 |
| ≥ 1 player ≤ 300 | 20% | +0.39 ± 1.97 | +17.4 ± 3.7 | +1.54 ± 0.60 | +15.6 ± 2.2 |
| 0 players ≤ 1,000 | 43% | +0.11 ± 1.09 | +23.1 ± 2.8 | +0.57 ± 0.24 | +14.5 ± 1.3 |
| 1 / 2 / 3 players | 28 / 14 / 7.5% | +1.0 / +7.0 / -4.2 | +21.4 / +11.3 / +30.7 | | |
| 4+ players ≤ 1,000 | 6.7% | +1.82 ± 3.93 | +13.1 ± 5.8 | +2.66 ± 1.23 | +15.8 ± 3.8 |

Secondary, separate offense / defense penalties (10 × 10 grid, infinity drops
the block): the best pair is (3,000, 3,000), the shared value; offense only
+11.8, defense only +7.2, both +20.8: defense adds **+9.0 ± 1.2** on top of
offense.

Reading:

- The efficiency optimum is flat between 1,000 and 3,000 (+1.14 ± 0.82); the
  pre-declared criterion picks 3,000. 2_6's lambda_offdef was about right once
  the history is clean.
- Against no information, offense and defense both carry signal (+11.8 and
  +9.0 marginal); pace ratings remove 2.1% of their target's squared error vs
  0.57% for efficiency.
- Pace: squared error prefers 10,000, MAE 30,000 (2_6's, chosen by MAE); the
  difference is small and the pre-declared criterion is kept.

## Phase 3, step 2b: pseudo-targets of the profile prior

`python scripts/player_graph/build_prior_pairs.py` (module
`player_graph/profile_prior.py`; nothing fitted yet). Pairs: every rated player
at the first game date of each month, 2016-12 → 2019-06 (25 checkpoints,
14,786 pairs, 744 players), zero-prior ratings at 3,000 / 10,000 and the as-of
node profile. 234 rated player-dates are left out: their last game is beyond
the 2-season profile window (median residual exposure 10 possessions).

Pseudo-target `t = rating / s` with `s = e / (e + lambda)`, a **diagonal
approximation** of the ridge's shrinkage (RAPM's columns are correlated, so `t`
is a pseudo-target for `f`, not the player's true rating). With weight `w = s`
each pair contributes `x · rating` to the normal equations whatever `s` is, so
tiny shares cannot destabilize the fit by themselves.

- `s · Var(t)` is flat across share buckets (offense 1.6-2.5, defense 1.6-2.5,
  pace 1.0-1.8): on clean data the approximation is consistent. (On the
  corrupted store it was 37-166 for offense, with -29 / +24 ratings for a
  single player.)
- Extremes sit only at `s` < 0.005 (one-game players, 0.0% of the weight);
  the largest standardized offensive target is Curry's (t ≈ 10.5 at s ≈ 0.55).
- The median `t` rises with exposure (offense -7.8 at s < 0.005, +1.3 at
  s > 0.5): thin players are below average, which `f` has to learn through
  its experience features.

## Phase 3, step 2c: the profile prior `f` on its own (2018-19, out of time)

`python scripts/player_graph/fit_profile_prior.py`. Each 2018-19 checkpoint is
predicted by an `f` fitted on the checkpoints strictly before it; weighted MSE
of the pseudo-targets (`s ≥ 0.02`), against zero (the ridge's current prior),
the training mean, and a position-only model. Not yet connected to the solver.

| Rating | Pairs | MSE `f` | MSE zero | `f` vs zero | Calibration slope | Mean `t - f` |
| --- | --- | --- | --- | --- | --- | --- |
| Offense, all | 5,126 | 4.42 | 7.67 | **-42%** | 0.93 | -0.12 |
| Offense, exposure ≤ 1,000 | 2,121 | 13.36 | 19.78 | -33% | 1.49 | +0.15 |
| Offense, exposure > 1,000 | 3,005 | 2.91 | 5.61 | -48% | 0.91 | -0.16 |
| Defense, all | 5,127 | 5.25 | 6.21 | **-15%** | 1.01 | +0.05 |
| Defense, exposure ≤ 1,000 | 2,121 | 15.51 | 16.58 | -6% | 1.34 | +0.27 |
| Pace, all | 5,942 | 1.84 | 2.74 | **-33%** | 1.01 | -0.41 |

- The prior has the right **scale** (slopes 0.93, 1.01, 1.01; above 1 for thin
  players, whose targets are noisiest) and is **centered** for offense and
  defense. Pace sits 0.4 poss/48 above 2018-19's targets, a level shift that a
  free intercept absorbs when every player carries it.
- Position alone predicts almost nothing (offense 7.65 vs 7.67); the profile
  rates do the work (offense mainly points per shot, assists and turnovers;
  defense steals, blocks and defensive rebounds). Features are correlated, so
  single coefficients are a sanity check only.
- Debut prior as of 2018-10-01 vs the 2018-19 debutants' realized targets in
  their first 20 games: offense -2.49 vs -0.78, defense -0.42 vs -1.81, pace
  +0.14 vs +1.00. Debutants are below average and faster, as the prior says,
  but the sizes are noisy.

## Phase 3, step 3: RAPM shrunk toward the profile prior (same penalties)

`python scripts/player_graph/rapm_profile_prior.py`: the zero-prior control's
walk-forward ridge with only the target of the shrinkage changed,
`(X'X + lambda I) beta = X'y + lambda beta0`, at the frozen 3,000 / 10,000.
`beta0` comes from the latest monthly `f` (checkpoints up to the date). In the
solver 93.6% of player-dates get `f`, 0.1% the debut prior and 6.3% the 0
fallback, all of them players without a profile (last game beyond the
2-season window); none of those is on the floor in a scored stint (763,142
on-floor player appearances from `f`, 1,198 from the debut prior). Unrated
players are scored at their prior. Scored as in steps 1 and 2a; SE clustered
by game. The solver's prior path is tested: a constant prior is absorbed by
the free intercept (identical predictions), fitted or fixed.

| Efficiency slice | Share | Zero prior vs ∞ | Profile prior vs ∞ | Profile only vs ∞ | **Profile − zero** | Prior on offense only − zero | Prior on defense only − zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 100% | +20.8 ± 1.8 | +23.0 ± 2.6 | +17.7 ± 2.3 | **+2.15 ± 1.16** | +0.60 ± 1.11 | +1.10 ± 0.48 |
| ≥ 1 player ≤ 1,000 | 57% | +19.1 | +20.2 | +16.2 | +1.11 ± 1.73 | -0.62 ± 1.65 | +1.37 ± 0.69 |
| ≥ 1 player ≤ 300 | 20% | +17.4 | +14.0 | +6.4 | -3.37 ± 3.65 | -4.77 ± 3.59 | -0.00 ± 1.44 |
| 0 players ≤ 1,000 | 43% | +23.1 | +26.6 | +19.6 | +3.52 ± 1.40 | +2.21 ± 1.37 | +0.75 ± 0.62 |
| 1 / 2 / 3 / 4+ players ≤ 1,000 | 28 / 14 / 7.5 / 6.7% | | | | +4.41 / -2.80 / +1.09 / -4.27 | | |
| ≥ 1 unrated | 1.2% | +17.6 | +37.5 | +29.2 | +19.9 ± 16.9 | +22.3 ± 18.9 | -7.2 ± 6.7 |
| ≥ 1 debut (≤ 10 games) | 15% | +18.2 | +17.4 | +9.6 | -0.87 ± 4.02 | -2.75 ± 3.97 | -0.04 ± 1.61 |

| Pace slice | Zero prior vs ∞ | Profile prior vs ∞ | Profile only vs ∞ | **Profile − zero** |
| --- | --- | --- | --- | --- |
| All | +14.4 ± 0.9 | +14.7 ± 1.0 | +7.7 ± 1.0 | **+0.27 ± 0.13** |
| ≥ 1 player ≤ 300 | +15.6 | +16.2 | +9.2 | +0.51 ± 0.41 |
| ≥ 1 debut (≤ 10 games) | +15.5 | +15.4 | +8.5 | -0.15 ± 0.55 |

MAE: efficiency 46.764 (zero prior) vs 46.770 (profile prior); pace 14.376 vs
14.401.

Reading (2018-19 only):

- With the same penalties the profile prior is better on the pre-declared
  criterion, squared error, but only by about 2 SE (efficiency +2.15 ± 1.16,
  pace +0.27 ± 0.13), and not on MAE. The defensive prior is the steadier part
  (+1.10 ± 0.48).
- Against expectation, the gain sits in lineups of established players (0 or
  1 thin player: +3.5, +4.4), not in thin ones (≥ 1 player ≤ 300: -3.4 ± 3.7;
  debuts: -0.9 ± 4.0; small samples, wide errors). At lambda 3,000 even a
  regular keeps only half of his effect, so the shrinkage target matters for
  established players too; for thin players both `f` and the data are noisy
  (`f`'s calibration slope there is 1.3-1.5).
- `f` alone, with no RAPM data, recovers +17.7 of the zero-prior RAPM's +20.8
  for efficiency (pace: +7.7 of +14.4): the profile carries most of what the
  stints say about a player, so the best penalty toward it may well differ
  from 3,000.

## Phase 3, step 3b: penalty toward the prior, retuned on 2018-19

`python scripts/player_graph/rapm_profile_prior.py --sweep`: the zero-prior
sweep's log grid with every block shrunk toward `beta0` (infinity = the profile
alone), paired against the zero-prior control in the same run.

| Penalty toward the prior | 100 | 300 | 1,000 | **3,000** | 10,000 | 30,000 | 100,000 | ∞ (profile alone) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Efficiency vs ∞ (no info) | -1.7 | +9.7 | +19.2 | **+23.0 ± 2.6** | +22.5 | +20.6 | +18.9 | +17.7 |
| vs zero-prior control | -22.5 | -11.2 | -1.6 | **+2.15 ± 1.16** | +1.72 ± 1.24 | -0.21 | -1.93 | -3.12 |

| Penalty toward the prior | 1,000 | 3,000 | **10,000** | 30,000 | 100,000 | 10⁶ | ∞ (profile alone) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pace vs ∞ (no info) | +12.6 | +13.8 | **+14.7 ± 1.0** | +14.6 | +13.5 | +9.9 | +7.7 |
| vs zero-prior control | -1.77 | -0.58 | **+0.27 ± 0.13** | +0.22 ± 0.17 | -0.92 | -4.52 | -6.76 |

Secondary (separate offense / defense penalties toward the prior): best pair
offense 10,000, defense 3,000, +23.16 vs no information, +0.21 ± 0.34 over the
shared 3,000.

**Phase 3 on the development season.** The best penalty toward the profile
prior equals the best toward zero, so the final 2018-19 comparison is step 3's:
best profile prior vs best zero prior, **+2.15 ± 1.16 (pts/100)²**
(efficiency) and **+0.27 ± 0.13 (poss/48)²** (pace) on squared error; no gain on
MAE (46.770 vs 46.764; 14.401 vs 14.376). The gain sits in lineups of
established players; low-sample and debut slices are not better (small
samples, wide errors). Against the plan's "Done when" (at least as good as the
2_6 ratings, better for low-sample players): the first part holds on 2018-19,
the second does not.

## Phase 3: frozen before the 2019-25 evaluation (2026-10-08)

Everything below was decided on 2018-19 (and earlier) only, and is committed
before any 2019-25 result is generated. Nothing changes during or after the
evaluation; looking at individual seasons does not reopen any of it.

- **Main comparison:** clean zero-prior RAPM (shared `lambda_offdef` 3,000,
  `lambda_pace` 10,000) vs clean profile-prior RAPM (same penalties, shrunk
  toward `beta0`). Lambda → ∞ (no player information) and 2_6 as stored are
  context only; 2_6 is not the control, since its older ratings carry the
  2016-17 contamination.
- **Frozen:** zero-prior and profile-prior penalties 3,000 / 10,000; `f`
  ridge strengths 0.1 (offense) / 100 (defense) / 0.1 (pace); `f`'s 22
  features and preprocessing (`log1p` games in data, standardization fitted on
  each training window); `s_min` = 0.02; pseudo-targets `t = rating / s` with
  weight `s`; debut prior; monthly expanding refit of `f`; half-life 180 d;
  stints from 2016-17.
- **Walk-forward:** for a date D, every `f_C`, profile, RAPM target and prior
  depends only on information before D; seasons already observed join the
  training as the evaluation advances (that is the walk-forward), but no
  hyperparameter or design decision changes.
- **Criteria:** primary, weighted squared error on the next stints (profile
  prior − zero prior, SE clustered by game); secondary, MAE and the slices
  (≥ 1 player ≤ 1,000; ≥ 1 ≤ 300; unrated players; debutants ≤ 10 games;
  count of low-sample players), offense, defense and pace separately, pooled
  2019-25 and per season.
- **Interpretation, fixed in advance:**
  - consistent out-of-sample gain in squared error without material MAE
    degradation → the profile prior adds value;
  - plus a gain in the low-sample / debut slices → confirms the original
    hypothesis;
  - only a marginal global gain and none in low-sample slices → a secondary
    improvement / ablation, not an essential component;
  - no out-of-sample gain → the profile prior does not enter the main
    pipeline, and the negative result is documented.

## Phase 3 out-of-sample result, 2019-25 (one pass, 2026-10-08)

`python scripts/player_graph/evaluate_profile_prior.py` (committed before it
ran), configuration as frozen above. 513,152 offensive stint-sides and 267,379
pace stints on 1,419 dates. Players on the floor took `beta0` from `f`
5,121,883 times and from the debut prior 9,637 times, never from the 0
fallback. Drop in squared error, SE clustered by game.

| Efficiency, pooled | Share | Zero prior vs ∞ | Profile prior vs ∞ | 2_6 stored vs ∞ | **Profile − zero** | Prior on offense − zero | Prior on defense − zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 100% | +21.8 ± 0.7 | +26.0 ± 1.0 | +20.4 ± 1.0 | **+4.27 ± 0.48** | +1.98 ± 0.46 | +1.68 ± 0.22 |
| ≥ 1 player ≤ 1,000 | 59% | +21.9 | +26.4 | +20.2 | +4.52 ± 0.72 | +1.95 ± 0.67 | +1.97 ± 0.32 |
| ≥ 1 player ≤ 300 | 22% | +19.2 | +23.8 | +18.6 | +4.63 ± 1.51 | +1.32 ± 1.50 | +0.71 ± 0.66 |
| 0 players ≤ 1,000 | 41% | +21.6 | +25.5 | +20.8 | +3.90 ± 0.56 | +2.04 ± 0.54 | +1.28 ± 0.26 |
| 1 / 2 / 3 players ≤ 1,000 | 26 / 16 / 8.3% | | | | +3.08 / +6.29 / +2.61 | | |
| 4+ players ≤ 1,000 | 9.0% | +20.3 | +27.7 | +20.0 | +7.47 ± 2.62 | +0.63 ± 2.95 | -2.12 ± 1.25 |
| ≥ 1 unrated | 1.3% | +17.6 | +33.4 | +19.9 | +15.7 ± 7.0 | +14.7 ± 7.4 | -6.5 ± 3.0 |
| ≥ 1 debut (≤ 10 games) | 16% | +22.3 | +29.6 | +22.3 | +7.33 ± 1.74 | +3.81 ± 1.77 | +0.37 ± 0.81 |

| Efficiency by season | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Profile − zero | +3.85 ± 1.35 | +7.24 ± 1.38 | +1.58 ± 1.22 | +4.48 ± 1.18 | +4.15 ± 1.25 | +4.16 ± 1.27 | +4.69 ± 1.24 |
| MAE zero → profile | 48.250 → 48.224 | 46.890 → 46.828 | 47.821 → 47.808 | 46.992 → 46.935 | 47.273 → 47.232 | 48.338 → 48.292 | 49.563 → 49.499 |

| Pace, pooled | Zero prior vs ∞ | Profile prior vs ∞ | **Profile − zero** |
| --- | --- | --- | --- |
| All | +11.29 ± 0.31 | +11.18 ± 0.34 | **-0.11 ± 0.05** |
| ≥ 1 player ≤ 300 | +10.07 | +9.63 | -0.44 ± 0.16 |
| 4+ players ≤ 1,000 | +10.75 | +9.72 | -1.03 ± 0.30 |
| ≥ 1 debut (≤ 10 games) | +10.66 | +10.09 | -0.57 ± 0.21 |
| ≥ 1 unrated | +9.62 | +11.23 | +1.61 ± 0.83 |

Pace by season, profile − zero: -0.28, -0.49, -0.20, -0.03, +0.05, +0.09,
+0.04 (SE 0.11-0.20). MAE pooled: efficiency 47.886 (zero) → 47.842 (profile),
2_6 stored 47.875, no information 48.090; pace 14.474 → 14.498, 2_6 stored
14.468.

Reading against the rules fixed in advance:

- **Efficiency (offense and defense): the profile prior adds value.** The gain
  in squared error is consistent (positive in all 7 seasons, 6 of them beyond
  2 SE; pooled +4.27 ± 0.48, about 20% more than the zero prior's gain over no
  information), and MAE improves in every season. It is **larger in the
  low-sample and debut slices** (debuts +7.33 ± 1.74, 4+ thin players +7.47 ±
  2.62, unrated +15.7 ± 7.0 vs +3.90 with none thin), which confirms the
  original hypothesis out of sample, although the development season did not
  show it. Both blocks contribute (offense +1.98, defense +1.68).
- **Pace: no out-of-sample gain.** Slightly worse in squared error
  (-0.11 ± 0.05) and MAE, worst in the thin and debut slices, negative in the
  first seasons and about 0 later; only unrated players gain. The pace prior
  does not enter the main pipeline; pace keeps the zero prior.
- The phase's "Done when" holds for efficiency (better than 2_6 as stored and
  than the clean zero prior, and better for low-sample players) and not for
  pace.
- Hypothesis, not shown: the out-of-sample gain is larger than on 2018-19
  because `f` has more history as the walk-forward advances (about 20 months
  of pairs when 2018-19 starts; the per-season gain settles near +4 from
  2022-23).

**Methodological note.** The final split (profile prior for efficiency, zero
prior for pace) was decided **after** observing the 2019-25 results. They
validate separately that the prior works for efficiency and not for pace, but
2019-25 is no longer an untouched holdout for the resulting hybrid pipeline:
it must not be used again to evaluate that combination as if it were.

**Phase 3 closed (2026-10-08).** Frozen pipeline, never re-selected on
2019-25: offense and defense shrunk toward the profile prior at
lambda 3,000; pace shrunk toward zero at lambda 10,000; `f` ridge strengths
0.1 / 100 / 0.1; monthly expanding refit; debut and no-profile fallbacks as
defined above.

## Phase 4A, step 1: where 2_6's projected minutes fail (2018-19)

`python scripts/player_graph/minutes_diagnostic.py` (module
`player_graph/minutes_diagnostics.py`). Diagnostic only, development season.
Projected minutes are 2_6's closing projection, the scenario-weighted
`allocate_minutes` the v0 game graph carries; actual minutes are regulation
minutes from the stints. 868 games whose two teams had filed a report and that
have validated stints (1,736 team-games, 24,855 player rows); both sides sum to
240 per team-game.

| Per-player MAE (min) | Nobody out | ≥ 1 listed out |
| --- | --- | --- |
| Available, projected starters | 4.74 | 5.28 |
| Available, projected bench | 4.75 | 5.84 |
| Listed uncertain (0.1 < p_out < 0.9) | 11.16 (they play 48% of the time) | |
| Not on the projected roster | 15.5 (projected 0, play 15.5 on average) | |

| Misallocated minutes per team-game (Σ \|proj − actual\| / 2) | 0 out | 1 out | 2 out | 3+ out |
| --- | --- | --- | --- | --- |
| Total | 34.7 | 36.6 | 36.7 | 43.3 |
| Available players who did not play | 7.3 | 7.5 | 6.2 | 6.5 |
| Players not on the roster | 1.2 | 1.3 | 1.4 | 1.1 |
| Everyone else (ordinary variation) | 25.4 | 26.7 | 28.1 | 34.8 |

By the absent player's full-health minutes: 33.5 (< 15), 37.8 (15-25),
39.7 (25-32), 41.5 (> 32). By final margin, ordinary variation rises with
blowouts (26.5 below 10 points, 30.6 at 20+).

**Redistribution** (1,110 team-games with someone listed out; 37,497 absent
full-health minutes):

| Who absorbs the absent minutes | `allocate_minutes` | Reality |
| --- | --- | --- |
| Projected starters | 52% | 36% |
| Projected bench | 48% | 56% |
| Players not on the projected roster | 0% | 7% |
| Least / middle / most position-similar third | 31 / 38 / 31% | 20 / 36 / 36% |

- The absent minutes go mostly to **one** player: with a single ≥ 20-minute
  player out (302 team-games), the top actual absorber takes a median **53%**
  of them, where `allocate_minutes` gives him 8%. He is a projected bench
  player 79% of the time, with a median full-health rank of 8 (outside the
  usual rotation), and the most position-similar available player only 15%
  of the time. Across available players, corr(projected change, actual
  change) is 0.17 (slope 0.45).
- **Returning players**: 87% of the minutes played by players off the
  projected roster belong to players with ≥ 20 games in the data (18.7 min on
  average): starters back from an absence longer than the 10-game roster
  window, who drop off the roster.
- **Available players who do not play** (14.1 projected min per team-game): 55%
  of it is deep bench (< 10 full-health min, 43% of them play at all), 24%
  rotation players (≥ 15 min) who sit 1.9% of the time unannounced.
- **Above 48 minutes**: 1.6% of team-games have a scenario over 48 (max 82.5),
  9.2% with 3+ players out; expected minutes over 48 in 0.9%.
- The 2018-19 oracle tables are too small to split the rotation oracle's gain
  by game type (+0.12 ± 0.17 overall; 88% of games have someone listed out on
  either side).

Reading: most of the misallocation is ordinary in-game variation (blowouts,
fouls, matchups), which no pre-game projection recovers. What a pre-game model
can recover is concentrated and specific: who takes an absent player's
minutes (one backup, usually from the end of the rotation, not every teammate
in proportion), keeping returning players on the roster, the deep bench's
low chance of playing, and the 0 ≤ minutes ≤ 48 / sum 240 constraint.

## Phase 4A, step 2: expanded roster, baseline minutes and redistribution shares (2018-19)

`python scripts/player_graph/rotation_diagnostic.py` (module
`player_graph/rotation.py`). Pieces B and C of the v1 minutes provider and the
expanded roster, walk-forward over 2016-17 → 2018-19 (each team-game from
earlier games only), checked on 2018-19; participation (A) and the final
0-48 / 240 reconciliation are not connected yet. Absences here are the
realized ones (on the roster, did not play, vacating ≥ 10 min), so this
measures the structure, not the injury report.

Definitions (v1): `b_Y = E[min_Y | Y plays, full-health]`, a decayed mean
(half-life 15 team games) of his played minutes net of the gains attributed to
that game's absences (clipped at 0); `V_X = q_X·b_X` with `q_X = 1` for rotation
players (`b ≥ 15`, whose missed games are unavailability) and his recent
participation rate otherwise, pending the participation model; `s(X→Y) ≥ 0`,
`Σ_Y s = 1`: pair evidence (decayed, half-life 82 team games; simultaneous
absences split EM-style in proportion to `V_X·s`) shrunk toward a structural
softmax prior `s0` (depth rank, rank gap, minutes, position similarity, start
share, participation; refit monthly on single-absence events), clipped at 0
and renormalized over tonight's available teammates.

| Check (2018-19) | Result |
| --- | --- |
| Expanded roster: players who played, on the roster | 98.8%; 1.68 min per team-game off the roster (2_6's 10-game window: 2.70); left: 235 player-games new to the team (mid-season moves without report evidence), 94 first games in the data |
| B in team-games without absence events (4,026 player-games) | MAE 4.57, bias -0.49 (mean of the last 10 games played: 4.72, +1.19) |

Predicted minute gains of available teammates who played (`Σ_X V_X·s(X→Y)`)
against their actual gain over `b` (23,439 teammate-games in 2,248 team-games
with absences; mean squared error):

| Rule | All | One absence | 2+ absences | Corr (all) |
| --- | --- | --- | --- | --- |
| Predict no gain (variance of the gain) | 46.6 | 35.0 | 49.6 | — |
| Proportional (`allocate_minutes`) | 55.8 | 36.1 | 65.4 | 0.14 |
| Structural prior `s0` | 39.6 | 34.0 | 42.3 | 0.40 |
| **`s0` + pair history** | **37.9** | **32.8** | **40.4** | **0.44** |

Ordinary variation (minutes − `b` without absences): 35.6. Absences add about
11 to the variance of a teammate's minutes; the shares remove about 8.7 of
them, while the proportional split adds 9. Top absorber (332 single absences
of ≥ 20 min): he takes a median 43% of the vacated minutes; the proportional
rule picks him 1.5% of the time, `s0` 17%, `s` 26%, and `s` gives him a median
10%: the shares find the right group of teammates but stay too diffuse to
name the one backup.

## Phase 4A, step 2b: ranking vs concentration, next-man-up features, small search (2018-19)

`rotation_diagnostic.py` now reports, on single absences vacating ≥ 20 min,
whether a share rule **ranks** the real top absorber high and how much it
**concentrates** on the teammates who really take the minutes, and scores the
v2 structural features ("next man up": who replaced X last time, Y's minutes
and start in the last game, his trend, whether he sits just outside the
rotation) on the same rows as the v1 engine. v2 must not feed back into the
baselines: used inside the engine it changed `b` and the events and worsened
the minutes (s0's sharper wrong picks attribute gains to the wrong players).

| Single absences ≥ 20 min (332) | `s0` v1 | `s0` v2 | `s` v1 | `s` v2 | Real |
| --- | --- | --- | --- | --- | --- |
| Top-1 / top-2 / top-3 recall of the real top absorber | 17 / 29 / 37% | 30 / 43 / 48% | 26 / 38 / 47% | 27 / 41 / 52% | |
| Median rank of the real top absorber (≈ 12 candidates) | 5 | 4 | 4 | 3 | |
| Share given to the real top (median) | 8% | 9% | 10% | 11% | 43% |
| … when ranked first | 19% | 26% | 26% | 28% | |
| Effective number of absorbers (median) | 12.5 | 12.1 | 9.3 | 9.0 | 5.6 |

Reading: the ranking is reasonable (the real top absorber is in the top 3 about
half of the time, median rank 3-4) and v2 improves the prior's ranking
clearly, the final shares' only at top-3; the shares are **too flat**
(9 effective absorbers vs 5.6 real; even correctly ranked first, the top
absorber gets 26-28% of the minutes where the real one takes 43%). A sharper
prior alone does not help the minutes: `s0` v2 has a worse squared error with
2+ absences (50.0 vs 42.3), where several concentrated bags land on the same
backup.

Small predefined search (one parameter at a time around the defaults; MSE of
the minutes, `b` + predicted gains, of every player who played in 2018-19,
27,465 player-games, the same rows for every configuration):

| Configuration | `s` v1 | `s` v2 |
| --- | --- | --- |
| Default (kappa 30, share half-life 82, rotation 15 min) | 37.59 | 37.59 |
| Share half-life 41 | **37.15** | 37.18 |
| Share half-life 164 | 37.88 | 37.95 |
| kappa 10 / 100 | 37.88 / 37.52 | 38.07 / 37.49 |
| Rotation threshold 12 / 18 min | 37.57 / 37.62 | 37.65 / 37.62 |
| Proportional split, for reference | 52.8 | |
| `s0` alone, for reference | 39.0 | 42.8 |

The only change worth more than a tenth or two is the shorter share half-life
(-0.44); kappa and the rotation threshold are flat. v1 and v2 shares tie on
the criterion (v2 0.01 worse in MSE, 0.01 better in MAE).

## Phase 4A, step 2c: concentration calibration (temperature), 2018-19

`python scripts/player_graph/rotation_temperature.py`: `s' = s^tau / Σ s^tau`
over the same eligible teammates, on the frozen run's tables (baselines, events
and targets unchanged). Criterion and decision rules fixed before running
(minutes MSE of every player who played; v1 on a tie; `tau > 1` only for a
clear gain; no small global gain bought with worse multi-absence games).

| Shares | tau | Minutes MSE | Gain MSE, one absence | Gain MSE, 2+ absences | Share to real top (median) | Effective absorbers | Multi-absence games with a stacked player | Largest predicted gain / its actual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | **1** | **37.15** | 32.57 | 39.88 | 10.0% | 9.7 | 10% | 10.1 / 8.9 |
| v1 | 1.5 | 38.62 | 33.06 | 42.21 | 9.7% | 7.4 | 23% | 13.0 / 8.7 |
| v1 | 2 | 41.54 | 34.25 | 46.74 | 9.0% | 5.7 | 31% | 15.6 / 8.8 |
| v1 | 3 | 48.72 | 37.30 | 57.78 | 6.7% | 3.7 | 38% | 19.4 / 8.7 |
| v2 | 1 | 37.18 | 32.23 | 40.10 | 10.8% | 9.5 | 19% | 12.0 / 9.6 |
| v2 | 1.5-3 | 40.5-56.3 | 32.8-37.5 | 45.6-71.0 | 10.9-7.1% | 7.1-3.5 | 32-46% | 15.7-23.4 / 9.5 |

Reading: sharpening concentrates the minutes on the model's own top pick, which
is the real top absorber only about a quarter of the time, so the share given
to the real top does not rise (it falls) while errors grow, worst with several
absences, where more bags stack on one player (10% → 38% of those games). At
`tau = 1` the model's largest predicted gain is already close to what that
player really gains (10.1 vs 8.9 min). **The flat shares are the error-optimal
hedge given the ranking uncertainty**; concentrating more requires better
ranking, not a calibration. Decision (by the rules fixed in advance): `tau = 1`,
v1. C is frozen.

## Phase 4A, step 3: participation A (v1 logistic), 2018-19

`python scripts/player_graph/participation_diagnostic.py` (module
`player_graph/participation.py`). `q = P(enters the rotation | medically
available in this scenario)`; injury uncertainty stays in the scenario weights
and `p_out` is never applied again inside a scenario. Labels: where the team
filed a report (`availability_source = injury_report`, from 2018-12-17) only
players not listed or with `p_out ≤ 0.1` are rows (questionable, doubtful and
out are left out; a rotation player the report calls available who sits is a
0); elsewhere (`heuristic`, weak labels) the engine's absence events are left
out and every other non-playing roster player is a coach's decision. Logistic
regression on 14 scenario features (baseline, depth rank among the available,
C's gain, minutes vacated by others, number available, recent participation,
streak out and return from ≥ 5 games out, last game's minutes, start share,
rest and back-to-back, point of the season), refit monthly on earlier months
(scaler included). 111,665 labelled rows (86k heuristic, 26k report); 38,989
scored in 2018-19. The scenario is the engine's realized absences.

| 2018-19 | Rows | Played | Brier logistic | Brier recent participation | Brier v0 (q = 1) | Log-loss logistic / recent |
| --- | --- | --- | --- | --- | --- | --- |
| All | 38,989 | 70% | **0.078** | 0.176 | 0.302 | 0.256 / 0.819 |
| Rotation (b ≥ 15) | 19,379 | 95% | 0.025 | 0.069 | 0.052 | 0.101 / 0.341 |
| Bench (b < 15) | 19,271 | 46% | 0.131 | 0.269 | 0.542 | 0.411 / 1.199 |
| Returning (≥ 5 games out) | 6,723 | 8% | 0.071 | 0.438 | 0.921 | 0.260 / 2.416 |
| C gain ≥ 3 min | 13,159 | 77% | 0.064 | 0.154 | 0.227 | 0.210 / 0.709 |
| Depth rank 1-5 / 6-8 / 9-10 / 11+ | | 96 / 89 / 70 / 29% | 0.020 / 0.046 / 0.118 / 0.145 | | | |
| Report labels / heuristic labels | 25,693 / 13,296 | 70 / 70% | 0.078 / 0.076 | 0.173 / 0.181 | 0.301 / 0.305 | |

Calibration is good at the extremes (q ≤ 0.1: 3.7% predicted, 5.9% observed;
q > 0.95: 98.9% vs 98.6%) and **over-confident in the middle**, more so on report
labels: q 0.5-0.75 → 63% predicted, 54% observed (report) vs 57% (heuristic);
q 0.75-0.9 → 84% vs 76% (report), 79% (heuristic). Three quarters of the
training rows are heuristic.

Raw minutes before any reconciliation, per 2018-19 team-game (2,620), over
tonight's available players:

| | Mean | SD | 5% | Median | 95% |
| --- | --- | --- | --- | --- | --- |
| Σ (b + C) | 274.2 | 24.8 | 246.9 | 269.2 | 322.6 |
| Σ q·(b + C) | 237.4 | 12.4 | 219.5 | 236.8 | 256.6 |
| What q takes off | 36.9 | 21.7 | 14.6 | 31.6 | 78.1 |

B + C alone over-allocates by about 34 minutes (it assumes every available
player plays); with q the raw total is nearly right on average (median scale to
240: 1.014) with a spread of ±12 minutes for the reconciliation to absorb. The
available players' actual minutes sum to 238.3 (players off the roster take
the rest).

## Phase 4A, step 4: reconciliation and v1 vs v0 minutes (2018-19, controlled)

`python scripts/player_graph/minutes_provider_diagnostic.py` (module
`player_graph/minutes_provider.py`). `raw_i = q_i·(b_i + C_i)` over tonight's
available players, reconciled per team to `0 ≤ m ≤ 48`, `Σ m = 240` with
`m_i = min(48, c·raw_i)` and `c` by bisection (the weighted least-squares
projection onto the capped simplex; proportional rescaling until the cap
binds). Every version sees the same scenario, the engine's realized absences:
v0 is 2_6's `game_nights` + `allocate_minutes` with those players sitting.
2,620 team-games, 40,654 player rows (anyone projected or playing).

| Reconciliation, per team-game | Median | 95% | 99% | Max |
| --- | --- | --- | --- | --- |
| Raw sum (min) | 236.8 | 256.5 | 271.5 | 311.4 |
| Scale `c` | 1.014 | 1.093 | 1.179 | 1.463 |
| Minutes moved, Σ \|m − raw\| | 7.2 | 24.6 | 43.0 | 75.9 |
| Largest single change (min) | 1.0 | 3.6 | 6.3 | 14.9 |

0 infeasible team-games; the 48-minute cap binds in 0.04%; `c` falls outside
0.85-1.15 in 2.0% (15 team-games beyond 0.8 / 1.25, mostly raw totals near 186-191
for teams carrying players without a baseline yet).

| Player MAE (min) | v0 | v1 raw | **v1** |
| --- | --- | --- | --- |
| All | 4.66 | 4.08 | **4.07** |
| Starters / bench | 5.59 / 4.22 | 4.80 / 3.73 | 4.78 / 3.73 |
| No absence / 1 / 2+ | 4.40 / 4.21 / 4.98 | 3.84 / 3.68 / 4.36 | 3.86 / 3.65 / 4.36 |
| Absent player's baseline < 15 / 15-25 / 25-32 / > 32 | 4.09 / 4.43 / 4.98 / 5.20 | 3.50 / 3.86 / 4.43 / 4.42 | 3.48 / 3.82 / 4.43 / 4.42 |
| Returning (≥ 5 games out) | 1.87 | 1.56 | 1.55 |
| Deep bench (rank 11+) / rank ≤ 10 | 3.50 / 5.30 | 2.56 / 4.91 | 2.56 / 4.90 |

Misallocated minutes per team-game: **v0 36.2 → v1 31.6** (no absence 41.6 →
36.5; one 34.0 → 29.4; two or more 35.9 → 31.4; absent player above 32 min
38.9 → 33.1). C survives the reconciliation: for teammates of the absentees
who played, corr(projected − b, actual − b) is 0.318 (v0), 0.437 (raw), 0.436
(reconciled), and their minutes MSE 51.6, 39.7, 39.4.

Reading: v1 cuts the player error by 13% and the misallocated minutes by
4.6 per team-game, in every slice; the reconciliation only rescales (median
1.4%, about 7 minutes per team-game) and almost never needs the cap, so it
does not hide a calibration problem of A + B + C.

## Phase 4A, step 5: true pre-game evaluation with the report's scenarios (2018-19)

`python scripts/player_graph/pregame_minutes_evaluation.py`. Both providers
project minutes for the closing report's scenarios (`p_out`,
`enumerate_scenarios`) and the expected minutes are the scenario-weighted mean.
v1 is `minutes_provider.ScenarioProvider`, fed by a pre-game callback of the
rotation engine (its as-of state only) with q models fitted on earlier months;
its full-health graph is the same provider with nobody sitting. 868 games with
both reports, 1,736 team-games. Integration check: v0's scenario graphs
reproduce 2_6's closing file (absence impact and possessions) to 1e-13.

| Player MAE (min) | v0 realized | v1 realized | **v0 report** | **v1 report** |
| --- | --- | --- | --- | --- |
| All | 4.58 | 4.09 | 5.13 | **4.75** |
| Starters / bench | 5.47 / 4.11 | 4.67 / 3.78 | 5.61 / 4.87 | 4.95 / 4.64 |
| No uncertainty / questionable or doubtful listed | 4.56 / 4.67 | 4.05 / 4.20 | 4.96 / 5.68 | 4.54 / 5.44 |
| 0 / 1 / 2+ listed out | 4.01 / 4.11 / 4.94 | 3.51 / 3.74 / 4.38 | 4.59 / 4.64 / 5.49 | 4.19 / 4.33 / 5.08 |
| Key player out (b ≥ 25) / not | 5.04 / 4.24 | 4.51 / 3.78 | 5.61 / 4.78 | 5.20 / 4.41 |
| Most likely scenario matched / missed | 4.54 / 4.72 | 4.04 / 4.23 | 4.98 / 5.59 | 4.58 / 5.25 |
| Deep bench (rank 11+) | 3.56 | 3.04 | 3.62 | 3.09 |
| Returning (≥ 5 games out) | 1.63 | 1.38 | **2.06** | **2.24** |

Misallocated minutes per team-game: v0 36.8 → v1 34.0 with the report
(realized: 32.8 → 29.3). Going from realized absences to the report's scenarios
costs v0 +0.55 and v1 +0.66 player MAE: availability uncertainty, not the
rotation model.

Total (R1 read off each provider's scenario graphs, clean ratings for both,
`walk_forward_offset` calibration, 764 calibrated games): total MAE v0 14.415,
v1 14.333, **v1 better by +0.082 ± 0.054**. Absence impact against each
provider's own full-health graph: mean |impact| v0 1.69, v1 1.80, correlation
0.85.

**Integration failure found: long absences not on the report.** The only slice
where v1 is worse than v0 is players out ≥ 5 games. They are not real returns:
not listed on the report, out 11-30 games (1,357 rows, play 8%, 0.8 min) or 31+
(980 rows, play 2%, 0.2 min), v1 projects 1.7 and 1.3 min where v0 (10-game
window) projects 0.7 and 0.1. They are most likely waived players and G League
assignments the report does not list, kept by the expanded roster because they
have not appeared elsewhere, with a q that does not reach 0.

## Phase 4A, step 6: roster membership after long absences (2018-19)

Rule (`RotationParams.roster_absence_games = 10`, `report_recency_games = 5`): a
player who has missed more than 10 team games stays on the expanded roster only
while his team's injury report has listed him in one of its last 5 games
(tonight's pre-game report included), strictly as of the game. Before reports
existed nobody is listed, which is 2_6's 10-game window. B, C and A rebuilt on
the new rosters and every evaluation rerun; the old rule is
`roster_absence_games = None`. Comparisons on the same rows (union of both
runs, 0 where a version projects nothing) or per team-game:

| 2018-19 | Old rule (no limit) | **New rule** |
| --- | --- | --- |
| Report scenarios, v1 player MAE (same 24,867 rows; v0 5.132) | 4.751 | **4.656** |
| Realized absences, v1 player MAE (same rows) | 4.089 | 4.035 |
| Report scenarios, misallocated min per team-game (v0 36.75) | 34.02 | **33.35** |
| Realized absences, misallocated (all 2,620 team-games; v0 36.2) | 31.56 | 31.45 |
| Minutes given to players out > 10 games who do not play (v1, report; v0 ≈ 805) | ≈ 3,190 | **≈ 400** |
| True returns (play after > 10 games out, 140): kept on the roster | 98.6% | 75.7% |
| True returns: MAE (v0 9.06) | 7.38 | 7.54 |
| Expanded-roster coverage of players who played (all of 2018-19) | 98.8% | 98.6% |
| R1 total, v1 better than v0 by | +0.082 ± 0.054 | +0.091 ± 0.053 |
| C after reconciliation, corr(projected − b, actual − b) | 0.436 | 0.449 |

The rule removes about 87% of the ghost minutes (now fewer than v0's) and
improves every comparable error. Its cost is 34 of 140 real long returns
falling off the roster (mostly marginal players back from G League stints or
long healthy scratches, never listed), whose error grows slightly but stays
well below v0's; and, where no reports exist (2016-17 to mid-December 2018),
long injuries drop off after 10 games like in v0, so C stops modelling them
there. B improves (MAE 4.54 vs 4.57 in team-games without absences); A's Brier
(0.081 vs 0.078) and C's MSE rows change with the population, not comparable
one to one; the raw A + B + C total moves closer to 240 (median scale 1.009).

## Phase 4A: minutes provider v1 frozen, before the 2019-25 evaluation

Frozen on 2018-19 (and earlier) only, committed before any 2019-25 result:

- **Expanded roster**: last team until he appears for another; carry-over of
  last season's players for 10 team games without phantom absences; beyond 10
  missed team games only with a report listing in the team's last 5 games.
- **B**: `b = E[min | plays, full health]`, decayed mean (half-life 15 team
  games) of played minutes net of attributed absence gains.
- **C**: v1 structural features, share half-life 41, kappa 30, rotation
  threshold 15 min, no sharpening (temperature 1); `V_X = q_X·b_X` with `q = 1`
  for rotation players.
- **A**: logistic participation `q = P(enters the rotation | medically
  available)` on 14 scenario features, report labels where filed (`p_out ≤ 0.1`),
  heuristic labels elsewhere, refit monthly on earlier months.
- **Reconciliation**: `min(48, c·q·(b + C))`, `c` by bisection, 240 per team and
  scenario.
- **Scenario integration**: the closing report's `p_out` and exclusions,
  `enumerate_scenarios` as in v0, full-health graph from the same provider.

Evaluation of 2019-25 (`pregame_minutes_evaluation.py --evaluation --seasons
2019-2025`), one walk-forward pass with history from 2016-17; for a date D every
roster, baseline, share, q model and report state reads only information
before D (tonight's pre-game report included); no architecture, rule or
hyperparameter changes after it runs.

- **Main comparison**: v1 vs v0 expected minutes from the closing report's
  scenarios.
- **Primary criteria**: player MAE of expected vs actual regulation minutes
  (on the union of both providers' rows) and misallocated minutes per
  team-game, pooled 2019-25 and per season.
- **Secondary**: the same under realized absences, the cost of availability,
  the slices of step 5 (role, uncertainty, players out, key player out, most
  likely scenario matched, returns, deep bench); R1 total MAE (walk-forward
  offset, paired SE) and absence impact, each provider against its own
  full-health graph, with 2_6's rating cache for both (its 2016-17
  contamination weighs about 0.04 by 2019-20).
- **Reading, fixed in advance**: a consistent minutes improvement over seasons
  → v1 replaces v0 as the minutes provider of the game graph; a significant R1
  gain → it also translates to totals; minutes better but R1 within noise →
  v1 is kept for its structure (graph weights, phase 5-6) without a claim on
  totals; no minutes improvement → v1 does not replace v0, and the negative
  result is documented.

## Phase 4A out-of-sample result, 2019-25 (one pass, 2026-10-09)

`python scripts/player_graph/pregame_minutes_evaluation.py --evaluation
--seasons 2019-2025` (committed before it ran, 06a757a), configuration as
frozen above. 8,845 games with both teams' closing report, 17,690 team-games,
255,764 player rows (union of both providers and actual players). v0's
scenario graphs reproduce 2_6's closing file on all 8,845 games (max
difference 1.4e-13 in absence impact, 7.1e-14 in possessions).

| Pooled 2019-25 | v0 | v1 | v1 − v0 |
| --- | --- | --- | --- |
| **Player MAE, report scenarios** | 5.423 | 4.733 | **-0.690** |
| **Misallocated minutes per team-game** | 39.20 | 34.22 | **-4.98** |
| Player MAE, realized absences | 5.040 | 4.291 | -0.749 |
| Cost of availability (realized → report) | +0.383 | +0.442 | |
| R1 total MAE (8,744 calibrated games) | 14.540 | 14.513 | **+0.027 ± 0.016** better |
| Mean absolute absence impact | 2.157 | 2.374 | corr 0.841 |

| By season | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Player MAE v0 → v1 | 5.526 → 4.851 | 5.418 → 4.879 | 5.563 → 4.889 | 5.412 → 4.675 | 5.219 → 4.500 | 5.389 → 4.611 | 5.450 → 4.759 |
| Misallocated v0 → v1 | 39.86 → 34.99 | 39.76 → 35.81 | 39.89 → 35.06 | 39.01 → 33.70 | 38.07 → 32.83 | 38.80 → 33.20 | 39.18 → 34.21 |
| R1, v1 better by | +0.082 ± 0.051 | +0.085 ± 0.045 | -0.057 ± 0.042 | +0.048 ± 0.035 | +0.004 ± 0.039 | +0.032 ± 0.045 | +0.012 ± 0.043 |

Slices (player MAE, report scenarios, v0 → v1): starters 6.021 → 5.041, bench
5.107 → 4.570; teams with questionable / doubtful players 6.076 → 5.513,
without 5.331 → 4.624; 0 / 1 / 2+ out 4.936 → 4.047, 4.922 → 4.253, 5.638 →
4.969; key player out 5.761 → 5.072; most likely scenario missed 5.844 →
5.156; returning after ≥ 5 out 3.639 → 3.171; deep bench 4.173 → 3.159. Long
absences: 1,792 true returns (play after > 10 team games out), 81.7% on v1's
roster, MAE v0 10.67 → v1 8.64; but the 400 returns after 31+ games out get
less from v1 (2.82 projected vs 5.01 for v0, actual 13.26), and players who
do not play after 11-30 / 31+ games out get slightly more ghost minutes from
v1 (1.21 / 1.10 vs 1.11 / 0.99).

Reading against the rules fixed in advance:

- **Minutes: consistent improvement → v1 replaces v0 as the game graph's
  minutes provider.** Better in every season on both primary criteria (player
  MAE -0.54 to -0.78, misallocated -4.0 to -5.6 per team-game), in every
  slice, under realized absences and under the report's scenarios. Larger
  than on 2018-19 (5.13 → 4.66) mostly because v0 is worse in these seasons
  (5.42 vs 5.13) while v1 changes little (4.73 vs 4.66). (Corrected
  2026-10-14: this line first compared with 4.75, the 2018-19 figure before
  the roster rule of step 6.)
- **Totals: R1 within noise → no claim on totals.** Pooled +0.027 ± 0.016
  (1.7 SE; 0.017 clustered by date), positive in 6 of 7 seasons but beyond 2
  SE in none. v1 is kept for its structure (graph weights for phases 5-6)
  without a claim that it improves the total.
- Descriptive, after the reading (not a selection): the R1 gain does not grow
  where the providers disagree most (terciles of |impact v1 − impact v0|:
  +0.027, +0.039, +0.015) nor with a key player out (+0.025 vs +0.033 without),
  so the small gain is not concentrated where the minutes changed.
- v1 loses a little more than v0 from the realized absences to the report's
  scenarios (+0.442 vs +0.383), and its gain is smaller on teams with
  questionable / doubtful players; consistent with the over-confident q in the
  middle range (future check).

**Phase 4A closed (2026-10-09).** The frozen v1 provider replaces v0 as the
game graph's minutes provider; its value is a more realistic rotation for
phases 5-6, with no claim that better minutes alone improve the total. 2019-25
is no longer an untouched holdout for the minutes provider: any later change to
it is developed on 2018-19 or earlier and must not be re-judged on these
seasons as if they were.

## Phase 4B, step 1: what a matchup-adjusted defensive effect can measure (2017-18, 2018-19)

`python scripts/player_graph/matchup_defense_diagnostic.py`
(`player_graph/matchup_defense.py`). Diagnostic only, nothing frozen. The
`pair_game` rows (attacker × defender × game) are labels and history only. For
a target `y` per exposure `w` each row gets the attacker's as-of norm (his
decayed rate, half-life 365 days, shrunk toward his position's rate with an
empirical-Bayes weight from 2017-18) and a residual `r = y − w·norm`; a
defender's effect is `D_j = Σr / Σw` (positive = his attackers beat their
norms). Signal by split-half covariance (each defender's games alternate
between halves; the covariance across defenders is the variance of the
persistent effect), empirical-Bayes shrinkage `k = φ/σ²` from 2017-18;
walk-forward on 2018-19 with as-of estimates centered by date, scored as the
drop in squared error of each defender-game's residual at slope 1 (nothing
fitted on 2018-19). 2017-18 counts from December, after the attacker norms'
burn-in.

**Data defect: doubled matchup exposure.** 65 late-season 2018-19 games
(`0021801152`-`0021801219`), one 2020-21 game (`0022000870`) and two 2021-22
playoff games (`0042100131`, `0042100235`) carry every row's matched seconds
and partial possessions at exactly twice their value (same rows, normal
points); `0022200861` (2022-23) has zero points on every row and inflated
seconds. All come from the archive. Game-level `Σ matchup_seconds / Σ
cofloor_seconds` is 0.075-0.087 in every season and about 0.16 in these games,
so `doubled_exposure_games` flags them at 0.12. All phase 4B history is read
through `read_matchup_history`, which drops them; `read_pair_game` is unchanged,
so they are still in the pair history of `expected_guard` v0, which stays as
built (see future checks).

**The NBA's credit measures defensive role, not effectiveness.** Points and
shots are credited to whoever is matched when the shot goes up, while partial
possessions follow guarding time. Per unit of guarding time, centers are
credited 1.42× the average attempts (1.37× the points) and guards 0.87×; a
center against guards sits +40 attempts per 100 partial possessions above the
guards' norms (help at the rim credited as the matchup). The resulting
"effects" are large and stable (attempts per possession: SD 5.3 per 100,
split-half r 0.94, season to season 0.91; walk-forward gain 32%) but correlate
+0.71 with the defender's blocks and point the wrong way against stint RAPM:
the better rim protector looks worse. Shot-credited volume is not a
defensive-skill target. The one exception is efficiency on credited shots
(points per attempt): small (SD 0.04, split-half r 0.19-0.23, season to season
0.34) but consistent with RAPM (corr 0.38) and adding to it jointly
(t 8.2 and 5.3); walk-forward +0.60 ± 0.07%.

**Time attribution.** Each attacker's whole-game residual is spread over his
defenders by their share of his partial possessions (seconds for the
per-minute target), so it no longer matters who was nearest when the shot
went up. No position structure is left (defender × attacker position cells
within ±1.1 per 100), and the season effect has the right sign against stint
RAPM `d` (corr −0.14; −0.26 among centers, −0.21 among guards). 2018-19:

| Time-attributed target | Defender SD | Split-half r | Season → season r | Within-team SD | Team SD | Walk-forward: defender | Team | Team + within |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Points per 100 poss. | 0.87 | 0.43 | 0.28 | 0.64 | 0.61 | **+1.05 ± 0.18%** | +0.72 ± 0.21% | **+1.21 ± 0.23%** |
| Attempts per 100 poss. | 0.59 | 0.46 | 0.32 | 0.54 | 0.19 | +1.07 ± 0.20% | +0.26 ± 0.17% | +1.03 ± 0.21% |
| 3PA per 100 poss. | 0.43 | 0.64 | 0.31 | 0.25 | 0.35 | +1.91 ± 0.26% | +1.23 ± 0.32% | +1.81 ± 0.34% |
| FTA per 100 poss. | 0.37 | 0.48 | 0.28 | 0.21 | 0.30 | +0.35 ± 0.27% | +1.01 ± 0.22% | +1.20 ± 0.23% |
| Turnovers per 100 poss. | 0.24 | 0.48 | 0.37 | 0.07 | 0.22 | +1.18 ± 0.15% | +1.27 ± 0.18% | +1.34 ± 0.19% |
| Points per attempt | 0.024 | 0.29 | 0.25 | 0.008 | 0.025 | +0.64 ± 0.14% | +0.85 ± 0.21% | +0.88 ± 0.21% |
| Points per matched minute | 0.054 | 0.49 | 0.36 | 0.035 | 0.045 | +1.94 ± 0.24% | +1.57 ± 0.28% | +2.06 ± 0.29% |

- Small and slow: one game's reliability is about 0.03 and a season's
  0.6-0.7 (`k` ≈ 1,700 partial possessions for points, about 35 games of a
  defender's matched possessions). The empirical-Bayes shrinkage is
  calibrated (walk-forward slope 1.02 ± 0.08 for points).
- Efficiency allowed is a **team** property: within-team signal is about 0
  (SD 0.008, split-half r 0.04-0.07). The individual part is in volume:
  attempts, 3PA and turnovers.
- Nearly orthogonal to stint RAPM: `−d` predicts none of the time-attributed
  residual (slope 0.02 ± 0.02, corr 0.005; joint t −0.4 next to `D_j`'s 12.7).

**Does the effect depend on the attacker?** Interaction cells (defender ×
attacker position, usage tercile or 3PA-share tercile, cut on 2017-18
exposure), net of the defender's own effect and the type's mean:

- Shot credit: large and stable (position: interaction SD 10 per 100, about 4×
  the main effect's variance) and the same role structure as above.
- Time attribution: some split-half covariance (interaction SD 0.4-0.8 per 100,
  t 2-9) but a reliability of 0.1-0.4 at a cell's median exposure. Walk-forward,
  the defender's own cells add +0.04 ± 0.03% (position), +0.03 ± 0.02% (usage)
  and −0.01 ± 0.02% (3PA share) to points; generic type means shared by all
  defenders slightly hurt (usage −0.48 ± 0.13%). Defender × attacker-type
  effects are not measurable per cell at this volume.

Reading (development seasons only):

- The target for the matchup-adjusted defensive effect is the
  **time-attributed** residual, not the NBA's shot credit (efficiency on
  credited shots is the only shot-credited piece that behaves like skill).
- The data support a **scalar** defender effect (plus the team's), not a
  per-attacker-type one; attacker-dependent effects are left to phase 6, where
  the FM / GNN can share parameters across attacker and defender profiles
  instead of estimating cells.
- The effect is small, persistent, calibrated with empirical-Bayes shrinkage
  and nearly orthogonal to stint RAPM `d`. Whether it adds to points allowed
  and totals beyond RAPM is the next question (the plan's own validation:
  future stints' points allowed, then R2 and the oracle rerun).

## Phase 4B, step 2: the scalar matchup-adjusted rating against stint RAPM (2018-19)

**Model** (`player_graph/matchup_rating.py`). One row per attacker and game:
his points per partial possession against all his defenders, weighted by his
partial possessions, explained by a walk-forward ridge
`μ + h·home + a_i + t_T + Σ_j s_ij·d_j`. Here `a_i` is the attacker, `t_T` the
defending team, `d_j` the defender entering with `s_ij`, his share of the
attacker's partial possessions (time attribution, step 1). There are no
defender × attacker-type terms. The ridge pulls each block toward 0 with its
own penalty in partial possessions; the normal equations decay with a
365-day half-life and each date is solved on earlier games. History comes
from 2017-18 through `read_matchup_history`, with the 65 corrupt games out.
In 2_6's units the rating is `d_rating = −100·(t_T + d_j)`: points per 100
possessions, positive = good defense. It splits into `d_team` and
`d_individual`.

**Penalties** (`matchup_rating_sweep.py`): walk-forward weighted MSE of each
2018-19 attacker-game's points per partial possession. The first grid's best
set sat on its lower edge, and its tie-break (largest team penalty) let the
defenders absorb team defense, so the grid was extended downward and the
team tie-break reversed. Both changes were made on 2018-19 and are recorded
in the script.

- Second grid: interior optimum at attacker 100 and defender 300. The team
  penalty barely moves the criterion: in an attacker-game a team shift is the
  same as shifting all its defenders. Following the design (team defense
  belongs to the team term), the smallest team penalty among ties: **100 / 100 /
  300**.
- At that point the between-team share of the defenders' `d_j` variance is
  0.058, about chance level for 30 teams of ~13 defenders. With a team penalty
  of 30,000 and a defender penalty of 1,000-3,000 it rises to 0.08-0.11.
- Out of sample, the defender terms cut the attacker-game MSE by
  1.30 ± 0.17 (of 119.9, about 1.1%). The ridge beats step 1's estimator on the
  same attacker-games (119.87 vs 120.60).

**Future stints** (`evaluate_matchup_rating.py`). Each book keeps the clean
RAPM book's offense, pace and league level and changes only the defense.
Scored on 76,434 stint-sides of 2018-19 (points per 100 possessions, weighted
by possessions):

| Defense | Gain over the league mean | Paired vs RAPM | Calibration slope of the shift |
| --- | --- | --- | --- |
| Clean stint RAPM `d` | +19.7 ± 2.5 | — | — |
| Matchup `d_team + d_individual` | −1.4 ± 3.0 | **−21.1 ± 2.1** | 0.02 ± 0.05 |
| Matchup `d_team` only | +8.0 ± 2.6 | −11.7 ± 1.9 | 0.11 ± 0.06 |
| RAPM + matchup `d_individual` | −5.5 ± 3.6 | **−25.1 ± 2.2** | −0.06 ± 0.05 |
| Step 1's simple estimator | −9.5 ± 3.5 | −29.1 ± 2.9 | 0.22 ± 0.06 |

- Worse than RAPM in every band of the defense's lowest RAPM exposure,
  including the thin ones (≤ 300 possessions: matchup −27.0 ± 6.4).
- Joint regression of `actual − league − Σ off` on the defense's summed
  components (a calibrated component has coefficient −1): RAPM `d`
  −0.82 ± 0.07, matchup `d_team` +0.04 ± 0.08, matchup `d_individual`
  +0.01 ± 0.05. This is no information, not a wrong scale: with that SE, even
  a fifth of the nominal effect would show.

**R2 / oracle rerun on 2018-19.** `oracle_ceiling.py build --seasons 2018
--ratings NAME=PATH` builds the four graph versions once and reads them out
with each book. Only 32 of 2018-19's graph games have a closing line, so
`summarize --without-line` scores total MAE on all of them. 764 games have a
calibrated level in every book; paired differences on the same games, slices
defined by RAPM:

| Defense book | R1 v0 | R2 v0 (expected guards) | R2 observed guards | Observed − expected guards | R2 observed guards − RAPM's R1 |
| --- | --- | --- | --- | --- | --- |
| Clean stint RAPM | 14.430 | 14.475 | 14.471 | −0.004 ± 0.021 | +0.04 ± 0.02 |
| Matchup `d_team + d_individual` | 14.718 | 14.770 | 14.686 | **−0.084 ± 0.035** | +0.26 ± 0.13 |
| Matchup `d_team` only | 14.727 | 14.727 | 14.727 | 0 (same rating for a whole team) | +0.30 ± 0.16 |
| RAPM + matchup `d_individual` | 15.114 | 15.169 | 15.140 | −0.029 ± 0.040 | +0.71 ± 0.18 |

With a key defender out (288 games): matchup observed − expected guards
−0.137 ± 0.062; RAPM +0.007 ± 0.038.

- With stint RAPM the observed assignments still add nothing, as in the
  2019-25 oracle.
- With the matchup rating they do add something (2.4 SE in one development
  season). But they only win back what the expected guard weights lose with
  that rating: R2 is worse than R1 with expected guards (+0.05), and observed
  guards bring it to −0.03 ± 0.04 against the same book's R1.
- Every matchup book stays clearly worse than plain RAPM (R1 +0.29 ± 0.12),
  as the stint test predicts.

**Reading.**

- A defender's persistent effect on the attacker he guards does not move his
  team's points. This fits step 1, where the individual part is volume
  suppression: his man shoots less, and the shots go to teammates.
- The matchup-adjusted rating predicts the matchup ledger (who scores), not
  points allowed. As a replacement for stint RAPM `def_j`, or on top of it, it
  adds nothing to future stints or to the total on 2018-19.
- Guard assignments only matter in the readout when the defensive rating is
  matchup-specific, which is weak support for guard edges as structure.
  Their value would have to come from a model that also redistributes the
  offense (who takes the shots the defender takes away): the FM / GNN of
  phase 6, not a scalar readout.

**Decision (2026-10-14): the scalar matchup-adjusted rating is closed as a
negative development result.** Stint RAPM stays `def_j` in R1 / R2 and
phase 5; the matchup rating does not enter the main pipeline;
`expected_guard` stays the guard-edge structure for phase 6. The
implementation and results stay as a documented experiment
(`matchup_rating.py`, `matchup_rating_sweep.py`,
`evaluate_matchup_rating.py`). No 2019-25 evaluation: the model is rejected
on development data. That guarding interactions carry information a single
additive defender rating cannot express (who takes the shots, the rest of
the floor) is a phase 6 hypothesis, not a result.

## Phase 4B: sensitivity of `expected_guard` v0 to the corrupt matchup games (2026-10-14)

`PointInTimeData.without_corrupt_matchups` (or `load(...,
exclude_corrupt_matchups=True)`) drops the 69 games of
`doubled_exposure_games` from the pair history, whole games at a time. The
numerator (matched seconds) and the denominator (co-floor seconds) of every
pair rate leave together, along with the game counts and the position prior.
The raw matchups stay: positions read only their start-position labels and
node profiles only their players, never the exposure. `expected_guard` is
the graph's only reader of pair history.

`build_expected_guard.py --exclude-corrupt-matchups` rebuilt 2018-19 to
2025-26 into `expected_guard/v0_without_corrupt_matchups/` with v0's
parameters (k 300, half-life 365 days, 3-season window); nothing was
retuned. `expected_guard_sensitivity.py tables` compares the two:

- Every as-of date through 2019-04-01, the first corrupt game's date, is
  identical; the first difference is on 2019-04-02.
- Two games (`0021800927`, `0022101075`) have rows only in the stored v0.
  Their stints were rejected by the points validator after v0 was built, a
  stint-store change unrelated to matchups. The comparison uses the
  2,050,334 shared pairs.

| Season | Mean abs. Δ `r_hat` (relative) | p99 | Guard-share TVD between versions (weighted) | p99 | Attacker-games > 0.01 | Benchmark TVD vs observed: v0 → corrected |
| --- | --- | --- | --- | --- | --- | --- |
| 2018-19 | 5.7e-4 (0.5%) | 9.1e-3 | 0.0017 | 0.072 | 2.2% | 0.27470 → 0.27470 |
| 2019-20 | 2.2e-3 (2.3%) | 3.4e-2 | 0.0108 | 0.148 | 14.4% | 0.27506 → 0.27436 |
| 2020-21 | 1.0e-3 (1.2%) | 1.4e-2 | 0.0057 | 0.086 | 9.7% | 0.26661 → 0.26630 |
| 2021-22 | 4.9e-4 (0.6%) | 6.1e-3 | 0.0028 | 0.049 | 5.4% | 0.25773 → 0.25776 |
| 2022-23 | 1.3e-4 (0.1%) | 1.5e-4 | 0.0006 | 0.0066 | 0.9% | 0.25521 → 0.25515 |
| 2023-24 | 5.7e-5 | 1.2e-4 | 0.0003 | 0.0045 | 0.6% | 0.25421 → 0.25418 |
| 2024-25 | 2.9e-5 | 5.5e-5 | 0.0002 | 0.0028 | 0.4% | 0.24700 → 0.24699 |
| 2025-26 | 5.6e-6 | 1.5e-5 | 0.00003 | 0.00003 | 0.07% | 0.24709 → 0.24709 |

All seasons: guard-share TVD between the versions 0.0026; benchmark TVD
0.25934 → 0.25921. `r_hat` averages about 0.08, so "relative" is mean
`|Δ r_hat| / r_hat`.

- **Roll-off.** The monthly mean `|Δ r_hat|` is 5e-3 in April 2019, about 2e-3
  in 2019-20, 1e-3 in 2020-21, 5e-4 in 2021-22 and about 1e-4 from 2022-23. It
  reaches 0 from March 2026, three years after the last corrupt game. One
  local spike: June 2023 (1.7e-2), the Finals between the two teams of the
  zero-points game `0022200861` (Heat vs Nuggets, 2023-02-13).
- **Game graphs** (`expected_guard_sensitivity.py graphs`: closing v0
  graphs of 2018-19 to 2020-21, 2_6's rating cache). R1 is identical.
  - Guard-edge TVD per attacker and scenario: 0.004 / 0.010 / 0.005 (p99 0.13 /
    0.15 / 0.08).
  - R2 moves by 0.026 / 0.072 / 0.033 points on average (max 2.26 / 1.06 /
    0.46).
  - For scale, R2's own guard effect `|R2 − R1|` is 1.10 / 0.53 / 0.45 points,
    and R2 itself adds nothing to total MAE (−0.003 ± 0.006, 2019-25).

**Reading: negligible.** The change is small and transient: concentrated in
2019-20 and 2020-21, fading with the window. The benchmark against observed
shares does not move (where it does, the corrected history is marginally
better). The frozen settings stay (k 300, half-life 365 days, 3-season
window); nothing is retuned because of the correction.

**Two pair histories, kept apart (decision 2026-10-14).**

| Matchup history | `PointInTimeData` | Stored `expected_guard` | Used by |
| --- | --- | --- | --- |
| `original` | `load(...)` (default) | `expected_guard/v0/` (unchanged since phase 2) | Phase 2, the v0 oracle, the guard and overlap benchmarks: the reproducible reference |
| `without_corrupt_matchups` | `load(..., exclude_corrupt_matchups=True)` | `expected_guard/v0_without_corrupt_matchups/` | Phase 5 graphs and every representation-learning experiment after it |

`PointInTimeData.matchup_history` carries the label and every artifact
records it: game-graph metadata (`matchup_history`) and the stored
`expected_guard` JSONs (`matchup_history`, plus `excluded_matchup_games` for
the corrected one). `read_seasons` refuses to mix seasons with different
metadata, and `stored_version(matchup_history)` maps a history to its
directory, so the phase 5 pipeline must name its history explicitly.

## Phase 5, steps 1-3: inputs, graphs and the feature block (2026-10-14)

**Phase 3 rating book** (`player_graph/rating_book.py`,
`build_rating_book.py`). The frozen pipeline (offense and defense toward the
profile prior at 3,000, pace toward zero at 10,000, half-life 180 days, `f`
refit monthly with the frozen alphas, the debut prior) emitted as a
walk-forward book in 2_6's cache format:
`data/player_graph/rating_book/phase3_profile_prior.parquet`, 1,829,400 rows
on 1,656 dates (2018-10-16 to 2026-06-13), every fit ending before its date.
A player without data carries exactly his prior; one without a prior and
without exposure is omitted (a lookup reads 0, his exact value). On phase 3's
2019-25 evaluation rows (513,152 stint-sides, 267,379 pace stints, same rows
and order) the book reproduces the stored walk-forward predictions to
3.2e-6 (solver tolerance; mean 4e-9).

**Minutes-provider hook** (`game_graph.TeamMinutes`, `MinutesProvider`,
`V0Minutes`; `minutes_provider.V1Minutes`, `run_scenario_provider`). The game
graph takes each team's scenarios, weights and minutes from a provider and
records `minutes_provider` next to `matchup_history`; v0 (2_6) stays the
default. `check_v1_game_graphs.py` on 2018-19 (corrected history):

- v0 graphs still reproduce the 2_6 closing file (870 games, max difference
  1.1e-13);
- v1 graphs' scenario-weighted minutes equal the 4A provider's expected
  minutes to 7e-15 (868 games, 22,506 rows);
- against the frozen provider's stored pre-game table
  (`pregame_minutes_roster10_season=2018.parquet`) they agree to the
  provider's run-to-run spread (below) and reproduce its printed figures
  (player MAE 5.001 on its 23,152 rows; misallocated minutes 36.75 → 33.35 per
  team-game). The step 6 "4.656" is on the union of the roster rules' rows
  (24,867), not this table's;
- every graph reads history strictly before its game date.

**Determinism.** The 4A provider was not reproducible across processes:
runs with different `PYTHONHASHSEED` differ by up to 0.003 minutes (97% of
rows), because sums over absentee sets followed hash order and two iterative
fits (C's structural share, q's logistic) amplify the float differences.
Fixed by ordering only (`participation.ROW_ORDER`; absentees iterated
sorted in `rotation` and `minutes_provider`), no modelling change: two runs
with different hash seeds now give identical minutes (0 of 23,152 rows
differ), within 5.2e-4 of the stored table, and every published 4A figure is
unchanged at its printed precision.

## Phase 5, step 4: evaluation protocol (decided 2026-10-14)

Fixed before anything is run on the evaluation window; the first comparison
of 2_5 and the PI block on 2019-25 is the single pre-registered pass of step
5.

- **Data.** A research file, `closing_line_research_2_6_pi_v1_<date>.parquet`:
  the pinned 2_6 closing file (20261003) plus the `PI_*` columns joined by
  `GAME_ID`, NaN where a game has no graphs (`build_pi_research_file.py`,
  with a provenance record). Not a registered schema layer: a 2_7 layer would
  become every dataset builder's default and run the 4A engine on every
  build; it is registered only if the block earns it.
- **Model.** The promoted 2_5 closing `LINE_ERROR` spec (d055949f99db), its
  hyperparameters fixed (`n_estimators` 92, max depth 4, colsample 0.204, ...),
  its cleaning, `season_year_floor` 2019, regular season and play-in,
  playoffs out, no sample weights. No tuning per arm.
- **Feature block (decided).** The primary block is the three mostly new
  columns, `pi_features.PRIMARY_FEATURES`: `PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_{HOME,AWAY}`
  and `PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE`. The four rating levels stay in the
  files (`ABLATION_FEATURES`) for future ablations, not in this experiment.
- **Arms.** A: 2_5 (excludes `LU_`, `STARTER_`, `PI_`). B: 2_5 + the primary
  block (excludes `LU_`, `STARTER_`, `PI_OFF_RATING`, `PI_DEF_RATING`); the
  runner refuses an arm whose `PI_*` features after cleaning are not exactly
  its expected set. Same file, same rows (8,298 after
  cleaning in both, verified with the block entirely missing as the worst
  case); the runner asserts identical training and evaluation games per day.
- **Walk-forward.** Retrain on every game strictly before each refit date,
  window capped at the spec's 6,200 most recent games. Warm-up: no
  prediction until 1,250 earlier games exist (the spec's minimum), so the
  first eligible game day is 2021-01-21 and 2019-20 and early 2020-21 only
  train. Evaluation: 945 game days, 7,042 games, 2021-01-21 to 2026-04-17.
- **Seeds.** Ten paired seeds (101, 202, ..., 1010), the same for both arms.
- **Primary loss, fixed:** absolute error of the `LINE_ERROR` prediction.
  Primary statistic: the mean paired difference per game,
  `|error_A| − |error_B|` (positive = the block helps), each arm's
  prediction being the mean of its ten seeds.
- **Two kinds of uncertainty, reported separately.** Training randomness:
  the ten per-seed paired differences (mean, SD, min-max) and the 2_5 seed
  band, the spread of MAE across arm A's ten seeds and its 45 seed-pair
  differences (measured first, before arm B runs). Statistical uncertainty
  over games: the SE of the ensemble difference clustered by game date, and a
  95% interval.
- **Decision rule, fixed:** the block improves the 2_5 baseline if the
  date-clustered 95% interval of the ensemble difference excludes 0 in its
  favour and at least 8 of the 10 seed pairs agree in sign. Otherwise null.
  One primary test.
- **Secondary (no decisions):** squared error; the total implied by line +
  predicted error; calibration (slope of the actual line error on the
  prediction, per arm); per season; fixed slices: games with
  |`LU_ABSENCE_IMPACT_PTS_BEFORE`| > 3, games without PI coverage (a placebo:
  should be about 0), the first 30 days of each season, after each trade
  deadline; betting at the spec's 0.1-point edge against the actual odds
  (win rate with a Wilson interval, ROI) as an economic diagnostic only.
- **Not an untouched holdout.** 2019-25 already served the phase 3 and 4A
  out-of-sample evaluations and the 2_6 trial, and the 2_5 hyperparameters
  were selected on folds inside it (a bias both arms share). The pass is the
  first downstream comparison under this frozen specification, not a clean
  holdout for every upstream component.
- **Cadence (decided): daily**, as the promoted protocol: 945 refits x 20 runs
  = 18,900 fits (about 42 h on this machine at 8 s per fit).
- **Where it runs (decided).** Not on this machine: on the machine with the
  6 GB GPU. The runner checkpoints every month of every seed, so the run can
  stop and resume. The device (CPU or CUDA) is chosen with
  `benchmark_pi_fits.py` on that machine and is the same for every run of
  the comparison; changing the protocol or the hyperparameters for hardware
  reasons needs a decision first.

**Prepared, not run (2026-10-14).** Features for 2019-20 to 2025-26
(`build_pi_features.py`, commit 138f0d0; 98.6-99.9% of each season's games
covered) and the research file
`closing_line_research_2_6_pi_v1_20261014.parquet` (`sha256:32fcdc51ef979b53`,
with its provenance record). The campaign
(`experiments/pi_block_v1_closing_2026_10/`: the two configs, the README with
the pre-registration and the transfer instructions), the resumable runner
(`run_pi_walk_forward.py`, monthly checkpoints), the pre-registered report
(`report_pi_walk_forward.py`), the CPU / GPU benchmark (`benchmark_pi_fits.py`)
and `experiments/runners/run_pi_block_v1.sh`. Smoke-tested on two game days and
two seeds of each arm: arm A 1,239 features, arm B the same plus exactly the
three primary columns, identical games. The closing dataset has no over /
under prices, so the betting diagnostic settles at -110.

**Feature block v1** (`player_graph/pi_features.py`,
`build_pi_features.py`), from v1 graphs and the phase 3 book, scenario
weighted: `PI_OFF_RATING_BEFORE_TEAM_{HOME,AWAY}` and
`PI_DEF_RATING_BEFORE_TEAM_{HOME,AWAY}` (sum of `min/48 × rating`, stint
RAPM, not the rejected matchup rating), `PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_{HOME,AWAY}`
(share of the side's 240 minutes for players with ≤ 1,000 decayed offensive
possessions in the book, phase 3's threshold), `PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE`
(tonight − full health, 2_6's projection with these ratings and minutes). The
matchup-weighted defensive family is excluded (phase 4B); guard-share
dispersion stays a future structural ablation. 2018-19: 868 of 1,312 games
(both teams' reports needed, as `LU_*`).

**Redundancy audit, 2018-19** (`pi_feature_audit.py`; thresholds fixed before
running: `|r| ≥ 0.9` with any 2_5 or 2_6 column, or 5-fold ridge R² from all
1,803 2_5 columns ≥ 0.8):

| Column | Strongest 2_5 correlate | R² from 2_5 (CV ridge) | Max \|r\| with 2_6 | Redundant |
| --- | --- | --- | --- | --- |
| Offense, home / away | team season `OFF_RATING` +0.82; `ACTIVE_WEIGHTED_OFF_RATING` +0.81 (away) | 0.91 / 0.91 | 0.50 / 0.53 (`LU_PROJ_MARGIN`) | **yes** |
| Defense, home / away | team season `DEF_RATING` −0.72 / −0.78 | 0.86 / 0.84 | 0.52 / 0.53 | **yes** |
| Low-sample share, home / away | injured players' minutes +0.60 / +0.64 | 0.51 / 0.54 | 0.24 / 0.28 | no |
| Absence impact (prior, v1) | injured / first-game-out points −0.31 | 0.35 | 0.69 (`LU_ABSENCE_IMPACT_POSS`, `_PACE_PTS`) | no |

The rating levels are mostly 2_5's team ratings again; the low-sample shares
and the absence impact carry mostly new information.

## Phase 6, step 1: the stint-graph pretraining dataset and edge targets (2026-10-14)

**Decisions (2026-10-14).**

- Pretraining starts in 2016-17.
- 2016-17 is used for stint targets and teammate / opponent interactions. Its guard edges are position-based estimates, never observations, and they carry their provenance.
- Edge-level supervision uses real matchups from 2017-18.
- Edge targets are time-attributed attacker volume: FGA and 3PA per partial possession. Credited-shot efficiency is a diagnostic only.
- Embedding dimension is 16, with 8 as a possible predeclared capacity ablation.
- The phase 5 protocol stays as frozen.

**Dataset** (`player_graph/pretraining_data.py`, `build_pretraining_dataset.py`;
`data/player_graph/pretraining/v1/`, with a `manifest.json` that records
inputs, parameters, exclusions and checks). It holds 389,095 stints in 10
seasons (2016-17 → 2025-26), stored as three tables per season:

- **`stints`:** the ten players in node order, the counts, the targets and their exposures.
- **`nodes`:** one row per `(as_of_date, player_id)`.
- **`guard`:** one row per `(game, attacker, defender)` with the expected rate and its source.

The encoder's loader (`to_arrays` / `load_stints`) reads only these tables. Every input table passes `check_encoder_inputs`: no same-game observed column and no betting column.

- **Node features:** the stored v0 node profiles with positions recomputed
  from box-score starts as well (`positions_as_of(..., box_starts=True)`),
  plus `position_starts`.
  - v0 positions read starts only from the matchups, so all of 2016-17 was
    1/3 G/F/C.
  - The box scores' `START_POSITION` agrees with the matchups' on every
    2018-19 start both hold (13,050). It also adds the 2016-17 starts and 70
    regular-season starts missing from the matchups.
  - The option is off by default: `expected_guard` v0 and the stored node
    profiles keep the positions they were built with.
  - In 2016-17 the mean top-position probability goes from 0.43 (October) to
    0.85 (January) as starts accumulate.
- **Guard edges**, by source:
  - `pair_history` / `position_prior` (2017-18 on): `expected_guard` v0 on the
    corrected history (`v0_without_corrupt_matchups`, frozen k / half-life /
    window). 2016-17 and 2017-18 were built for it and are identical to v0
    (no corrupt game is in their window). Every season's JSON now lists the
    69 games found over the whole stored history, so the ten seasons read
    together.
  - `structural`: where `expected_guard` has no rate at all (all of 2016-17,
    the first date of 2017-18), the same-position rule
    `r = Σ_x p_i(x) p_j(x)` on the as-of positions. It has no parameter and
    reads nothing after the date.
  - `uniform`: the remaining fallback.
- **Targets:** each is weighted by its own exposure, the denominator of the rate.
  - Points and turnovers per possession: possessions.
  - Pace: the stint's seconds.
  - 3PA/FGA and FTA/FGA: FGA.
  - OREB rate: rebound chances.

  Pace is the stint's own (mean of both sides' possessions per 48, phase 3's
  pace rows), stored in both side slots with half the seconds each. This
  carries out the "pace by exposure seconds" check below.
- **Exclusions:**
  - The two stint-rejected games are absent (checked).
  - The 69 corrupt matchup games keep their stint graphs, since their stints
    are sound, but are out of the guard history and of every observed label.
- **Checks per season** (all passed):
  - ten distinct players;
  - a node row as of every stint date;
  - a guard row for every pair on the floor;
  - structural rates only on dates without any matchup history;
  - stored targets equal to the counts;
  - no stint-rejected game.

  As of the date, on three dates per season, the node features (rebuilt
  from scratch by `node_profiles`) and the expected guarding rates (rebuilt
  by `expected_guard` from the as-of view) equal the stored rows to 1e-9.
  A unit test changes box scores on and after the date and gets identical
  rows.

**The structural rule against observed shares** (`structural_guard_check.py`;
weighted TVD, lower is better; development seasons, nothing tuned):

| Guard rate | 2017-18 | 2018-19 |
| --- | --- | --- |
| Constant (uniform, the fallback it replaces) | 0.424 | 0.410 |
| **Structural rule** (2016-17's edges) | **0.383** | **0.373** |
| Learned position prior | 0.361 | 0.335 |
| `expected_guard` v0 (k = 300) | 0.310 | 0.275 |

In October 2017, with a few days of matchup history, the rule (0.379) beats
both the learned prior (0.431) and v0 (0.423). It is worse than uniform only
in the playoffs (May 0.406 vs 0.354).

**Edge targets** (`player_graph/edge_targets.py`, `build_edge_targets.py`;
`pretraining/v1/edges/`, labels only, never read by the input loader):

- **Time attribution.** For each attacker-game, the targets are the whole
  game's FGA and 3PA per partial possession: `y_ig = Σ_j count_ij / Σ_j pp_ij`.
  - A model predicts `f(i, j)` per defender and is scored on
    `Σ_j s_ij f(i, j)`, weighted by `pp_ig`.
  - `s_ij` is the defender's share of the attacker's partial possessions. The
    observed shares enter only as these exposure weights.
- **Other columns.** Credited efficiency (points per credited attempt) is
  kept per pair as a diagnostic. The observed guarding share is kept as a
  possible auxiliary label.
- **Size.** 2017-18 → 2025-26: 241,673 attacker-games. Mean FGA per partial
  possession is about 0.20, 3PA 0.061 → 0.073.
- **Excluded games.** 71 games have no rows: the 69 corrupt games and the 2
  stint-rejected ones. 145 attacker-games (0.06%) are dropped because a
  player lacks node features on that date.
- **Limitation.** These are noisy observational proxies, not causal
  defensive effects. Assignments react to the game, help defense and
  switches blur the attribution, and one game's reliability is about 0.03
  (phase 4B).

## Phase 6, step 2: monthly out-of-fold harness, baselines, FM, hardware benchmark (2026-10-14)

**Folds** (`player_graph/pretraining_oof.py`). Checkpoint `C_k` is the first
of each month from 2016-12-01, the first checkpoint of phase 3's prior. Fold
`k` trains on everything dated before `C_k` and predicts `[C_k, C_(k+1))`.

- There are 88 folds with games, 25 of them before 2019-07.
- `check_fold` refuses a training date at or after the checkpoint.
- Predictions are stored per model and fold (`pretraining/v1/oof/<model>/`),
  written atomically. A stint predicted by two folds is refused.

**Baselines at the same cadence** (`pretraining_baselines.py`, all 88 folds
stored):

- **constant:** each target's exposure-weighted mean, decayed with the
  180-day half-life.
- **RAPM:** the frozen phase 3 ratings as of each checkpoint, for points per
  possession and pace; its other four targets take the constant.

Development folds (2016-12 → 2019-06, 110,147 stints), RAPM over constant:

| Target | Gain in MSE | t |
| --- | --- | --- |
| Points per possession | +0.57% | 13.1 |
| Pace | +1.72% | 20.9 |

Each of 2016-17, 2017-18 and 2018-19 gives 0.54-0.58% for points and
1.3-2.2% for pace. Stint noise dominates, so gains on future stints are small
in MSE terms by construction.

**Evaluation** (`player_graph/pretraining_eval.py`). Per target, the gain is
the exposure-weighted drop in squared error, with the SE clustered by game
date and expressed as a share of the reference's MSE. Every comparison is
paired: rows any model leaves unpredicted are dropped for all.
`ladder(...)` runs the plan's ladder, overall and by slice.

**Factorization machine, step 6a** (`player_graph/fm.py`, plain PyTorch;
`fm_oof.py` runs it on the folds with warm starts):

- **Interaction terms:**
  - offensive and defensive chemistry: pair means over the five;
  - team style: `Σu · Σv / 25`;
  - matchups: `Σ m_ij a_i·b_j / 5` over the **expected** guard shares.
- **Per-player blocks:** every block is `A·x + δ`. `x` holds the as-of node
  features. `δ` is penalized (300 pseudo-stints) and is zero for a player
  unseen at the checkpoint.
- **Targets and embeddings:** each target reads a per-target diagonal
  readout over embeddings shared by all targets. The dimension is 16.
- **Edge head:** `Σ_j s_ij f(i, j)` with the same `a`, `b` vectors.
- **Ladder variants:** `additive`, `no_matchup`, `matchup`, `matchup_edges`.
- **Hyperparameters** (penalties, learning rate 3e-3, 3 epochs, batch 1,024)
  are untuned starting values.
- **Synthetic tests:**
  - the output does not depend on the order within a side;
  - the pair-sum identity holds;
  - a planted additive effect is recovered (r > 0.8);
  - a planted attacker × defender interaction acting only through guard
    edges is learned by the matchup term (R² > 0.5) and not by the additive
    model (R² < 0.1);
  - the edge head equals the hand-computed `Σ s f`;
  - warm starts move `δ` to the grown vocabulary;
  - checkpoints round-trip.
- **CPU smoke runs** (no claim about signal):
  - 2016-12 → 2017-02, 3 folds, 2 epochs, 20,000-stint cap: about 1 s per
    fold. The edge loss is 0, since 2016-17 has no edge labels.
  - 2017-11 → 2017-12, with edges: the standardized edge loss is 0.65 → 0.57.
  - Rerunning skips the stored folds.

**Hardware benchmark** (`benchmark_pretraining.py`). It uses random tensors
with the dataset's shapes, so it runs on the GPU machine with no data. It
measures training throughput and peak GPU memory for the FM (dimension
8/16/32) and a relational GNN (`SmokeGNN`, dimension × layers 32×2 → 128×3;
the phase 6b architecture is still open) at batches of 1,024-16,384, and
projects 3 epochs over all 88 folds (17.4 M stint passes per epoch). A
configuration that runs out of GPU memory is recorded. This machine, CPU only
(10 threads):

| Model | Throughput (stints/s) | 3 epochs over all folds |
| --- | --- | --- |
| FM, dimension 16 | 69k-93k | 0.16-0.21 h |
| GNN 32×2 | about 13k-15k | about 1.0-1.1 h |
| GNN 64×3 | about 4k-5k | about 2.9-3.7 h |
| GNN 128×3 | about 1.6k-2k | about 7-9 h |

The FM needs no GPU; the GNN is where the 6 GB card matters. On the GPU
machine:

```bash
poetry install --with graph
poetry run python scripts/player_graph/benchmark_pretraining.py --out artifacts/experiments/pretraining_benchmark_gpu.json
```

## Phase 6, step 3: the FM ladder on the development folds (2026-10-15)

**Design, declared before the runs** (`experiments/runners/run_fm_ladder.sh`,
report `fm_ladder_report.py`; outputs in `artifacts/experiments/fm_ladder_dev/`,
predictions and per-fold checkpoints and logs under `pretraining/v1/{oof,fm,edges_oof}/`):

- **Variants:** `additive`, `no_matchup` (chemistry and style), `matchup`, and
  `matchup_edges` (time-attributed edge loss, weight 1.0: provisional, not
  optimized).
- **Training:** `FMConfig` defaults (dimension 16, `δ` penalty 300), 3 epochs,
  batch 1,024, learning rate 3e-3, no subsampling, CPU.
- **Folds:** the same 25 monthly folds (2016-12 → 2019-06) for every variant;
  each trains on every earlier stint.
- **Losses:** the stint loss is the mean over targets of the exposure-weighted
  MSE. The edge loss is the same per attacker-game, weighted by partial
  possessions only.
- **Seeds:** 0, 1 and 2 per variant, each fold seeded from (seed, fold), so
  reruns are bit-identical and a resumed run equals an uninterrupted one. The
  primary predictions are each variant's mean over its seeds.
- **Scoring:** every model is scored on the same 110,147 stints. Gains are
  paired, with SEs clustered by date.
- **Edge baselines:** a constant (365-day half-life) and an attacker norm,
  his decayed rate shrunk toward the constant with a fixed 100 partial
  possessions (a reference value, not tuned).
- Nothing was tuned or changed after the runs started.

**Stint targets, gain in % of the reference's MSE (t):**

| Comparison | Points / poss. | Turnovers | Pace | 3PA/FGA | FTA/FGA | OREB rate |
| --- | --- | --- | --- | --- | --- | --- |
| RAPM over constant | +0.57 (13.1) | — | +1.72 (20.9) | — | — | — |
| Additive FM over constant | +0.59 (12.8) | +0.65 (13.3) | +1.22 (17.5) | +5.35 (36.7) | +0.84 (16.4) | +1.45 (18.4) |
| Additive FM over RAPM | +0.02 (0.7) | +0.65 | **−0.51 (−9.0)** | +5.35 | +0.84 | +1.45 |
| Matchup FM over RAPM | +0.03 (1.0) | +0.64 | −0.26 (−4.6) | +5.39 | +0.86 | +1.44 |
| Chemistry / style increment | +0.004 (0.2) | −0.001 | **+0.23 (8.0)** | +0.05 (2.3) | +0.01 | −0.05 (−2.1) |
| Matchup increment | +0.009 (1.2) | −0.013 | +0.021 (1.8) | −0.008 | +0.006 | +0.037 (3.0) |
| Edge-supervision increment | −0.028 (−1.3) | +0.004 | +0.008 | +0.012 | −0.066 (−2.6) | −0.017 |

**By season.**

- The additive FM's pace deficit against RAPM is there in every season
  (−0.75, −0.47, −0.38).
- The chemistry / style pace increment is positive in every season (+0.28,
  +0.10, +0.33) and in 18 of 25 folds. Seed by seed it is +0.21, +0.20 and
  +0.05.
- The matchup increment on points is positive in 18 of 25 folds but tiny
  (+0.009 ± 0.007).
- Per seed, the matchup and edge increments change sign.

**Training noise.** The spread of MSE across the three seeds of a variant is
0.04-0.22% of the constant's MSE, larger than the matchup and edge
increments.

**Edge targets** (51,345 attacker-games, 2017-11 → 2019-06, gain in % of the reference's MSE):

| Comparison | FGA / pp | 3PA / pp |
| --- | --- | --- |
| Attacker norm over constant | +35.8 ± 0.6 | +47.7 ± 0.6 |
| FM (matchup + edges) over constant | +38.1 ± 0.6 | +49.9 ± 0.6 |
| FM over attacker norm | **+3.5 ± 0.2** | **+4.3 ± 0.3** |

The FM's edge gain over the attacker norm is the same in both seasons. It
mixes attacker profile, defender terms and the matchup product; this run
cannot attribute it.

**Training.** 0.8 (additive) to 1.8 (edges) minutes per seed for all 25
folds. No non-finite loss. The training loss rose over the epochs in at most
2 folds per variant, by under 0.001. The gap between training and
out-of-fold standardized loss is 0.012-0.014 for every variant. Its fold
pattern (largest for May-June and October, negative in mid-season) is the
same across variants, so it reflects the months, not overfitting by the
interaction terms.

**Reading (development only).**

1. **Learned interactions add nothing measurable on points per possession
   beyond additive RAPM.** No variant beats RAPM: +0.005 to +0.033%, t ≤ 1.0.
   The matchup term's increment is +0.009 ± 0.007, below the seed band.
2. **The FM's large gains are additive, not interactions.** They come from
   the four targets RAPM does not model (3PA/FGA +5.4%, OREB +1.4%, FTA/FGA
   +0.8%, turnovers +0.65%). Even 3PA/FGA, where a non-additive lineup effect
   had been measured, gains only +0.05% from interactions.
3. **Pace:** chemistry / style interactions add a small, consistent amount
   (+0.23%). The FM's additive backbone still predicts pace worse than RAPM
   (−0.5%). RAPM's pace model is a separate symmetric sum over all ten
   players, fitted with seconds weights by ridge to convergence; the FM's
   pace comes from the side-specific `off − def` structure with 3 epochs.
4. **Edge supervision with weight 1.0 does not help the stint targets.** It
   gives −0.03% on points and −0.07% on FTA/FGA, within or near noise. The
   edge head itself learns real structure: +3.5-4.3% over the attacker norm.

## Phase 6, step 4: residual FM over daily RAPM and the edge-head ablation (2026-10-15)

**Decisions (2026-10-15).**

- Points per possession and pace are learned as a **residual over a frozen,
  as-of RAPM base**; the other four targets keep their own heads.
- The ladder distinguishes:
  1. RAPM alone;
  2. + additive residual;
  3. + chemistry / style;
  4. + matchup;
  5. + edge supervision (optional).
- An edge-head ablation (attacker; + defenders; + matchup product) comes
  before any edge loss in the GNN objective.
- At least three paired seeds per comparison.
- No betting line in stint pretraining: it enters downstream in XGBoost, with
  the market-conditioned encoder kept as a future ablation.
- Edge-loss weights are not tuned on post-2019 results.

**Base** (`pretraining_daily_rapm.py`,
`pretraining_oof.daily_rapm_predictions`).

- The frozen phase 3 ratings are refitted on every game date (592 fits,
  2016-12-01 → 2019-06). Each stint gets the ratings fitted on stints before
  its own date: never itself, its day or later. A test checks that a collapse
  on day D is invisible on D and seen on D+1.
- Training residuals and evaluation use the same base.
- Stints before 2016-12-01 have no base (no prior), so their points and pace
  carry no residual weight. In the first fold no residual has been trained,
  so its prediction is the base alone; this is logged per fold.
- Daily RAPM is a stronger rung 1 than the monthly one: over the constant,
  +0.64% on points and +2.11% on pace (monthly: +0.57%, +1.72%).

**Design** (declared in `experiments/runners/run_fm_residual_ladder.sh`).

- Training as before: `FMConfig` defaults, 3 epochs, the 25 monthly folds,
  seeds 0-2 paired; nothing tuned.
- Edge-only models train on the edge loss alone, over the 17 folds with
  training edges (2017-11 on).
- Report: `fm_ladder_report.py --ladder residual`
  (`artifacts/experiments/fm_residual_ladder_dev/`). The refactored report
  reproduces step 3's report byte for byte.

**Stint targets** (110,147 stints; gain in % of the reference's MSE, t):

| Comparison | Points / poss. | Pace | 3PA/FGA | OREB | FTA/FGA | Turnovers |
| --- | --- | --- | --- | --- | --- | --- |
| 1. RAPM (daily) over constant | +0.64 (14.8) | +2.11 (24.6) | — | — | — | — |
| 2. + additive residual, over RAPM | **−0.18 (−6.0)** | **−0.24 (−7.2)** | +5.35 | +1.45 | +0.84 | +0.65 |
| 3. + chemistry / style, over RAPM | −0.22 | −0.28 | +5.39 | +1.41 | +0.82 | +0.65 |
| 4. + matchup, over RAPM | −0.24 | −0.29 | +5.42 | +1.40 | +0.80 | +0.65 |
| 5. + edges, over RAPM | −0.25 | −0.31 | +5.43 | +1.39 | +0.75 | +0.67 |
| All interactions (4 vs 2) | −0.05 (−4.0) | −0.05 (−3.8) | +0.07 (3.2) | −0.05 | −0.05 | 0.00 |
| Residual vs from scratch, additive | −0.13 (−4.5) | +0.66 (10.9) | 0 | 0 | 0 | 0 |

- **Residuals hurt.** The residual corrections make points and pace worse
  than RAPM alone in every season and in 21-23 of 25 folds. Interactions
  make it slightly worse again: points −0.06 to −0.17 seed by seed.
- **Why: over-dispersed corrections.** Regressing the actual residual on the
  model's correction gives a slope of +0.31 for points (+0.28 with matchups)
  and +0.09 for pace. The corrections point the right way for points but are
  about 3× too large; for pace they are almost noise.
- **Not overfitting by the interaction terms.** The train / out-of-fold gap
  is 0.014-0.016 (from-scratch FMs 0.012-0.013).
- **A plausible source: base drift.** A training stint's residual is taken
  against the base as it was on its date. A player's per-player correction
  `δ` therefore learns past errors of RAPM, which RAPM has corrected by the
  time it is served. This is the cost of a strictly as-of base.
- **The four targets RAPM lacks** keep the FM's additive gains (3PA/FGA +5.4%).

**Edge-head ablation** (51,345 attacker-games, 2017-11 → 2019-06, edge-only
models; gain %, every seed positive unless noted):

| Comparison | FGA / pp | 3PA / pp |
| --- | --- | --- |
| Edge-only attacker over attacker norm | −0.3 (2017-18 −3.8, 2018-19 +2.8) | −5.2 (−14.9, +3.3) |
| + defenders (`Σ s_ij de_j`) over attacker only | **+0.47 ± 0.06** | **+0.87 ± 0.12** |
| + matchup product over + defenders | +1.09 ± 0.21 | +3.93 ± 0.32 |
| Joint (residual stint model + edges) over edge-only matchup | +2.3 (2018-19 −0.1) | +4.6 (2018-19 +0.2) |

- **The defenders' additive term is a real defender contribution.** It holds
  in both seasons and all seeds, and matches phase 4B's time-attributed
  defender effects on attempts and 3PA.
- **The matchup product is not yet shown to be an interaction.** The
  attacker-only FM loses to the simple attacker norm in 2017-18, a sign its
  attacker effects are over-shrunk there. `a_i · Σ s_ij b_j` with
  near-constant defender vectors adds attacker capacity, so part of its gain
  can be less shrinkage, not who-guards-whom. A capacity control (the product
  with a learned defender-independent vector) is needed to tell them apart.
- **Joint stint pretraining helps the edge head only in 2017-18**, when edge
  history is short; in 2018-19 it changes nothing.

**Reading (development only).** On the stint targets, interactions add
nothing beyond RAPM in either formulation:

- from scratch they were at noise level;
- as residuals over RAPM they cost 0.05%.

The residual heads themselves are over-dispersed (slopes 0.3 and 0.1). The
next design step is their regularization, not more model capacity.

## Phase 6, step 5: interaction-only residuals, calibration, edge capacity control; FM closed (2026-10-15)

**Decisions (2026-10-15).**

- Remove the per-player additive residual for points and pace (RAPM holds
  the individual contributions); keep profiles as inputs.
- Add a walk-forward calibration that can shrink the corrections to zero,
  and report raw and calibrated results.
- Add a defender-independent capacity control to the edge head.
- One controlled run, then move to the GNN. A null FM result is kept as a
  baseline.

**Design** (declared in `experiments/runners/run_fm_interaction_ladder.sh`;
report `fm_ladder_report.py --ladder interaction`, in
`artifacts/experiments/fm_interaction_ladder_dev/`).

- **Interaction-only (`int_*`).** `FMConfig.no_additive` switches off `off`
  and `def` for points and pace. The interaction vectors are **centered** on
  a running mean over players (`center_interactions`), so an interaction
  term averages to about 0 over random partners.
  - Without centering, interaction-only terms rebuild a planted additive
    rating almost entirely in a synthetic test (R² 0.93 chemistry, 0.96
    style). With centering: 0.05 for style, and 0.20 for chemistry, the
    remainder coming from the 40-player synthetic roster (a player's
    teammates average −1/39 of his own vector).
  - So step 3's chemistry / style pace gain (+0.23%) may have been additive
    leakage.
- **Variants:** `int_control` (global intercept and home term only),
  `int_chemistry_style`, `int_matchup`. The other four targets keep their
  additive heads.
- **Calibration** (`pretraining_oof.calibrated_residuals`,
  `calibrate_residuals.py`). Per fold and target, the slope through the
  origin of `y − base` on the correction, over the run's own earlier
  out-of-fold rows, clipped to [0, 1]; 0 with nothing earlier.
- **Edge capacity control.** `edge_capacity` adds `a_i · g` with one learned
  defender-independent vector: the matchup product's attacker capacity,
  without the defenders. Then come the defenders' terms (`+ defenders`) and
  the matchup product (`+ matchup`) on top of it.
- **Training:** as in steps 3-4. Three paired seeds, 25 development folds,
  nothing tuned.

**Calibration factors (mean over folds):**

| Model | Points | Pace |
| --- | --- | --- |
| `int_control` | 0.66 | about 0.05 |
| Interaction models | 0.40-0.42 | 0.17-0.36 |
| Step 4's additive residual | 0.17-0.22 | 0-0.10 |

**Stint targets** (110,147 stints; gain % over daily RAPM, t):

| Model | Points / poss. raw | Calibrated | Pace raw | Calibrated |
| --- | --- | --- | --- | --- |
| 2. + intercept / home (control) | +0.038 (4.2) | +0.034 (5.4) | −0.025 (−3.7) | −0.002 |
| 3. + chemistry / style | −0.038 (−1.8) | +0.023 (2.6) | −0.025 | +0.018 (1.5) |
| 4. + matchup | −0.038 (−1.9) | +0.027 (3.2) | −0.039 | +0.014 (1.7) |
| Step 4's additive residual | −0.18 | +0.028 (4.5) | −0.24 | +0.005 |

| Increment | Points raw | Points calibrated | Pace raw | Pace calibrated |
| --- | --- | --- | --- | --- |
| Chemistry / style (3 vs 2) | −0.076 (−4.2) | −0.011 (−1.4) | 0.000 | +0.020 (1.7) |
| Matchup (4 vs 3) | −0.001 | +0.005 (1.1) | −0.014 | −0.004 |
| All interactions (4 vs 2) | −0.077 (−4.2) | −0.006 (−0.8) | −0.014 | +0.017 (1.9) |

- **Interactions add nothing on points.** The calibrated increments are
  −0.011 and +0.005, below 2 SE and changing sign across seasons. Seed by
  seed, the calibrated chemistry / style increment on points is negative for
  all three seeds; the pace increment is positive for all three (+0.013 to
  +0.023) but at t ≈ 1.7 and 0 in 2016-17.
- **What beats RAPM is not interactions.** It is the global intercept and
  home correction (RAPM has no home term): +0.034% on points, positive in
  20 of 25 folds. Calibration itself is worth +0.07% on points and +0.05% on
  pace over the raw corrections.
- **The four auxiliary targets** keep the additive gains (3PA/FGA +5.4%,
  OREB +1.4%, FTA/FGA +0.8%, turnovers +0.65%); interactions change them by
  under ±0.06%.
- **Training was stable:** no non-finite loss, train / out-of-fold gap
  0.013-0.014.

**Edge head** (51,345 attacker-games; gain %, every seed positive):

| Comparison | FGA / pp | 3PA / pp |
| --- | --- | --- |
| Capacity control over attacker only | +0.75 ± 0.13 (2018-19 0.0) | +2.66 ± 0.24 (2018-19 −0.08) |
| + defenders over capacity control | +0.37 ± 0.07 | +0.94 ± 0.11 |
| + matchup product over + defenders | **+0.62 ± 0.18** (2017-18 +0.95, 2018-19 +0.30) | **+1.58 ± 0.17** (+2.46, +0.73) |
| Defender-dependent terms together over the control | +0.99 ± 0.18 | +2.51 ± 0.23 |

- **Part of step 4's matchup gain was attacker capacity.** The control alone
  gives +0.75% / +2.66%, almost entirely in 2017-18, when edge history is
  short.
- **Against the control, defender-dependent terms still help** in both
  seasons and all seeds. The defenders' main effect gives +0.4 / +0.9%. The
  attacker × defender product adds +0.6 / +1.6% on top (2018-19: +0.3 /
  +0.7%).
- **Who guards whom therefore predicts an attacker's shot volume beyond his
  own profile and his defenders' average.** That is an individual,
  time-attributed volume result. It does not imply better team scoring
  prediction, which the stint results do not show.

**Conclusions for phase 6 (development only).**

1. **FM closed with a negative stint-level result.** Learned player
   interactions (chemistry, style, matchups) add no measurable information
   beyond as-of RAPM for points per possession, from scratch, as residuals,
   or calibrated. For pace there is at most +0.02% (t < 2). This is the
   baseline the GNN must beat.
2. Calibrated, interaction-only residuals over RAPM are the safe form: they
   cost nothing when the interactions carry no information.
3. The FM's real stint gains are individual effects on the four targets
   RAPM does not model.
4. The edge head shows defender and attacker × defender structure in shot
   volume. It is not yet tied to team scoring.

## Phase 6, step 6: the GNN (step 6b), implemented and benchmarked, not yet run (2026-10-15)

**Decisions (2026-10-15).**

- The GNN is an interaction-only residual over the strictly as-of daily RAPM
  for points and pace, with separate heads for the four auxiliary targets.
- Two message-passing layers of width 32, chosen in advance.
- The same monthly folds, three paired seeds and walk-forward calibration;
  raw and calibrated results kept apart.
- Centered interaction representations, with tests for additive leakage.
- Edge supervision is a separate later experiment.
- The ladder:
  - A: RAPM;
  - B: + intercept / home;
  - C: + capacity-matched node-only neural model;
  - D: + GNN with teammate / opponent edges;
  - E: + full GNN with guard edges.
- Benchmark on this CPU and prepare the 6 GB GPU benchmark. No long training
  yet, and no architecture tuning on 2019-25.
- Afterwards, assess the representations as pregame embeddings for XGBoost
  rather than optimizing stint accuracy.

**Model** (`player_graph/gnn.py`). Nodes start from `MLP(x_i) + e_i`: the
standardized as-of profile plus a penalized ID embedding, zero when unseen.
Each layer is
`h ← LayerNorm(h + ReLU(W_self h + B_tm(teammates' mean) + B_opp(opponents' mean) + B_g(Σ_l m_kl [h_l, e_kl]) + B_gb(defender's share-weighted attackers)))`:

- `m` is the **expected** guard share;
- `e_kl` holds the guard edge's attributes (expected rate, prior weight,
  history, provenance one-hot).

Observed guarding never enters.

**Capacity matching.** A relation switched off keeps its branch and
parameters but receives the node's own state instead of messages. C, D and E
therefore have **identical parameter counts**, inputs, heads, penalties and
training budget; only the information flowing between players differs
(tested). B is the same code with the interaction readout off.

**Readout.** As the FM's, over the centered final embeddings: offensive and
defensive pair means and the cross-side product, with per-target readouts.
Per-player additive scalars serve the four auxiliary targets.

**Leakage found and fixed: the additive absorber.** Centering does not stop
message passing from rebuilding an additive rating.

| Synthetic test (planted additive rating, 400 players) | R² of the interaction-only output |
| --- | --- |
| Node-only, not centered | 0.95 |
| Node-only, centered | 0.05 |
| Full GNN, centered | **0.87** |
| Full GNN, centered, with absorber | 0.00 |

- **Why:** each node sees its lineup's and the opponents' totals through
  messages, and the product readout can form functions of those totals that
  track a linear rating.
- **Fix:** the residual targets keep per-player additive terms **while
  training** and drop them **at prediction** (`absorb_additive`). This
  partials the individual effects out (including the RAPM base's drift)
  instead of leaving the interaction part an incentive to rebuild them.
- **Cost:** the absorber slows learning of an interaction whose player types
  exist only in IDs. With the type visible in a profile feature, as in real
  data, the full GNN still learns a planted who-guards-whom signal well above
  the guard-blind model (tested).
- **Ladder:** every rung B-E uses the absorber.

**Tests** (`tests/test_player_graph_gnn.py`, 11):

- C, D and E have equal parameter counts.
- Guard features are expected quantities only.
- Node-only embeddings cannot see other players or edges.
- The teammate / opponent model reads neighbours, not guard edges.
- The full model changes when who guards whom changes.
- The output does not depend on the order within a side.
- Centered leakage stays bounded, and the absorber stops the
  message-passing leak.
- Guard edges carry a planted matchup signal.
- Gradients reach every parameter block.
- A stint's prediction depends only on its own graph, not on other stints in
  the batch; nothing crosses dates.
- Checkpoints round-trip, and warm starts move IDs to the grown vocabulary.

The fold-level temporal checks are the harness's (`check_fold`).

**Pipeline.** `fm_oof.py` variants `gnn_control`, `gnn_node`,
`gnn_no_guard` and `gnn_full`, with guard tensors built from the
pretraining dataset. The full GNN was smoke-fitted on 3 real folds (2017-11
→ 2018-01, full training data, 3 epochs): 14-25 s per fold, and rerunning
skipped them. Calibration: `calibrate_residuals.py`. Report:
`fm_ladder_report.py --ladder gnn`. Runner:
`experiments/runners/run_gnn_ladder.sh [cpu|cuda]` (declared, **not run**).

**CPU benchmark** (`benchmark_pretraining.py`, random tensors with the
dataset's shapes; this machine, 10 threads; 3 epochs):

| Model | Throughput (stints/s) | One run, 25 development folds | Development ladder (12 runs) | One run, all 88 folds |
| --- | --- | --- | --- | --- |
| FM 16 (closed) | 57k-89k | about 1 min | 0.2 h | 0.2 h |
| GNN 32×2 node-only / no guard | 13k-44k | 2-6 min | 0.4-1.2 h | 0.3-1.1 h |
| **GNN 32×2 full** | **14k-22k** | **4-6 min** | **0.7-1.2 h** | **0.7-1.1 h** |
| GNN 64×2 full | 8k-10k | about 10 min | about 2 h | about 1.8 h |
| GNN 128×3 full | 2k-4k | 22-40 min | 4-8 h | 4-7 h |

On real data a fold also prepares tensors and writes checkpoints. The smoke
fit ran at an effective 5k-12k stints/s, so the development GNN ladder is
about 1-2 h on this CPU.

On the 6 GB GPU machine (no data needed):

```bash
poetry install --with graph
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
poetry run python scripts/player_graph/benchmark_pretraining.py --out artifacts/experiments/pretraining_benchmark_gpu.json
```

It reports throughput and peak VRAM per model and batch (1,024-32,768),
records out-of-memory configurations instead of failing, and projects the
same run and ladder times.

## Phase 6, step 7: the GNN ladder A-E on the development folds (2026-10-15)

**Run** (`experiments/runners/run_gnn_ladder.sh cpu`, as declared in step 6, nothing changed).

- Variants: B `gnn_control`, C `gnn_node`, D `gnn_no_guard`, E `gnn_full`.
- Seeds 0-2; the 25 folds 2016-12 → 2019-06; width 32, 2 layers, 3 epochs.
- The process was pinned to 11 of the 12 logical CPUs (`taskset -c 0-10`,
  10 threads), leaving one core free.
- Training per seed for all 25 folds: B 2.3 min, C 3.7, D 2.6, E 5.2. The
  whole ladder took about 45 minutes, including calibration and the report.
- No non-finite loss; the training loss rose over the epochs in 3 of 25 folds
  per variant, by under 0.001.
- Report: `artifacts/experiments/gnn_ladder_dev/`.
- Diagnostics: `gnn_ladder_diagnostics.py` (`diagnostics.json` there).
- Checkpoints, per-fold logs and predictions: under `pretraining/v1/{fm,oof}/fm_gnn_*`.

**Confirmed.**

- **Absorber on in training, off at prediction.** `absorb_additive` is set
  in every variant. Re-predicting fold 2018-12 from its checkpoint
  reproduces the stored predictions exactly with the absorbed terms dropped.
  Including them would have moved points by SD 0.037 and pace by SD 1.18.
- **Calibration from earlier folds only.** Every calibration factor (4
  variants × 3 seeds × 25 folds × 2 targets) recomputes exactly from the
  rows dated before its fold's checkpoint; the first fold's are all 0.

**A defect: the centering mean lags in the first fold.** The running mean
that centers the embeddings starts at 0. Fold 2016-12 trains for about 27
steps (8.5k stints), after which the mean has covered about 75% of the way.
The stale centering offsets every product of the interaction readout, so in
that fold C, D and E over-predict 3PA/FGA by 0.14-0.16 (the control by
0.01).

- Points and pace are protected: their first-fold calibration factor is 0.
- The four auxiliary targets are not calibrated, so over all folds they look
  1.4-2.8% worse than B, with a seed spread of 1-4% of the constant's MSE.
- Smaller lags of the same kind at the end of later folds add seed noise.
- The fix for any rerun: center with the exact mean of the training
  embeddings, computed after each fold's training. That is an implementation
  correction, not a tuning choice; it was not applied to this frozen run.
- Below, "excl. 2016-12" re-scores the same stored predictions without that
  fold.

**Points per possession and pace** (gain % over the reference, t; mean of three seeds):

| Comparison | Points raw | Points calibrated | Pace raw | Pace calibrated |
| --- | --- | --- | --- | --- |
| A. RAPM over constant | +0.641 (14.8) | | +2.110 (24.6) | |
| B over A | +0.038 (4.3) | +0.033 (5.2) | −0.018 (−2.8) | −0.003 (−3.5) |
| **C vs B** (neural capacity) | −0.085 (−6.0) | −0.020 (−4.6) | −0.062 (−4.4) | +0.001 (0.9) |
| **D vs C** (teammate / opponent messages) | −0.120 (−6.2) | −0.007 (−3.8) | −0.179 (−5.6) | +0.001 (0.9) |
| **E vs D** (guard edges) | +0.090 (4.5) | +0.003 (1.9) | +0.081 (2.3) | −0.002 (−1.4) |
| E vs C (all message passing), calibrated | | −0.004 (−2.4) | | −0.001 (−1.3) |
| E calibrated over A | | +0.009 (1.6) | | −0.003 (−1.8) |
| E calibrated vs the FM's calibrated matchup residual | | −0.018 (−2.2) | | −0.017 (−2.0) |

- Excluding 2016-12, the raw increments are similar: C vs B −0.091 / −0.066,
  D vs C −0.128 / −0.192, E vs D +0.096 / +0.086.
- **Seeds:** calibrated, C vs B on points is negative for all three seeds
  (−0.016 to −0.034), D vs C for all three (−0.004 to −0.012), and E vs D
  is −0.002 to +0.009.
- **Folds:** E vs D calibrated on points is positive in 18 of 25 folds, the
  others in 6-12.

**Size and calibration of the raw corrections** (seed mean, folds after the first):

| Variant | Points SD | Slope of realized on correction | Pace SD | Slope |
| --- | --- | --- | --- | --- |
| B control | 0.013 | 0.91 | 0.19 | −0.49 |
| C node-only | 0.021 | 0.39 | 0.64 | −0.19 |
| D teammate / opponent | 0.029 | 0.25 | 1.48 | 0.09 |
| E full | 0.022 | 0.36 | 0.96 | −0.12 |
| FM matchup (step 5) | 0.028 | 0.40 | 0.98 | 0.37 |

Mean walk-forward factors: points 0.69 (B), 0.23 / 0.15 / 0.21 (C / D / E);
pace 0.06 and 0.01-0.03. Calibration keeps a quarter of the GNN's points
corrections and switches its pace corrections off, which is why the
calibrated rungs sit within ±0.02% of each other.

**Auxiliary targets**, gain % over the constant (B and A: RAPM has none, so
B's are the additive heads):

| Variant | Turnovers | 3PA/FGA | FTA/FGA | OREB |
| --- | --- | --- | --- | --- |
| B | +0.65 | +5.35 | +0.84 | +1.45 |
| E, all folds | −1.70 | +3.03 | −1.42 | −0.32 |
| E over A, excl. 2016-12 | +0.62 | +5.37 | +0.91 | +1.47 |
| E vs D raw, excl. 2016-12 | −0.02 | +0.04 | **+0.14 (8.4)** | **+0.09 (4.3)** |

Excluding the defective fold, the GNN's auxiliary heads match the additive
ones. Message passing changes them by under ±0.15%. Guard edges help FTA/FGA
and OREB against the teammate / opponent model, but only by about what that
model lost against node-only (E vs C ≈ +0.03 / +0.02).

**How to read it.**

- **What the absorber means.** This ladder measures interactions **beyond
  the best additive fit**. The absorber removes everything a per-player
  additive term can explain, including signal that is correlated with player
  identity but would be produced by interactions. If a player's
  "interaction" value is consistent across partners, it counts as additive
  here.
- **On that definition, the GNN finds nothing for team scoring.**
  - Neural capacity, teammate / opponent context and guard edges each move
    calibrated points and pace by at most ±0.02%.
  - Uncalibrated corrections are mostly noise (slopes 0.25-0.39 on points,
    about 0 on pace), and calibration shrinks them to near zero.
  - Guard edges consistently undo part of what the teammate / opponent
    model loses: raw E vs D is positive in 17 of 25 folds. They do not beat
    the node-only model.
  - The full GNN is not better than the FM matchup residual; both are worth
    about +0.01-0.03% over RAPM after calibration, and all of that is the
    intercept / home control's.
- **Stint level, together with the FM:**
  - additive player effects carry the signal (RAPM, and the auxiliary
    additive heads);
  - learned interactions, from FM or GNN, add nothing measurable for
    points or pace;
  - who guards whom matters for an individual's shot volume (step 5), not
    for team efficiency in these models.
- **Not an end-to-end test.** Representations may still carry game-level
  information (style, shot distribution, absences) that XGBoost can use
  against the line. That has to be tested in XGBoost, not inferred from
  stint MSE.

## Phase 7 preparation: the centering fix, the pregame adapter, out-of-fold outputs, campaign v1 (2026-10-15)

**Decisions (2026-10-15).**

- Stop optimizing stint-level points and pace.
- Fix the centering defect.
- Build pregame outputs: style rates per side, their absence shifts, and one
  16-dimensional game embedding.
- Define a stint-to-game adapter instead of feeding whole rotations to the
  encoder.
- Keep a node-only representation control.
- Prepare the monthly out-of-fold generation and a separate pre-registered
  XGBoost follow-up with arms A-E.
- Launch nothing large.

**Centering fix** (`fm.finalize_centering`). After each fold's training, the
centering means are set to the **exact** mean of the fold's training-stint
embeddings (FM blocks or GNN embeddings). `fm_oof.py` applies it to every
centered model, and fold logs record `"centering": "exact training mean"`.

- Tests: after a 27-step "first fold" the running mean leaves the training
  embeddings' mean off by more than 0.05, and the exact mean brings it under
  1e-5.
- The mean reads only the given training stints; prediction never moves it.
- The temporal-isolation tests still pass.
- The step 7 ladder stays as run and documented (running-mean centering).

**Stint-to-game adapter** (`player_graph/pregame_encoder.py`). The encoders
saw only 10-node stints: means over 4 teammates and 5 opponents, guard shares
over 5 defenders. A pregame graph has 20-26 players with fractional minutes,
so it is not fed directly.

- **Lineups.** Each scenario and team becomes **48 one-minute lineups** by
  systematic (Madow) sampling on the expected minutes. Players are ordered by
  minutes, then id; inclusion probability is `minutes / 48`, summing to 5;
  grid points are `(k + ½) / 48`. Every slot has five distinct players, the
  team gets exactly 240 minutes, and each player gets his expected minutes
  within one minute. Deterministic and untuned.
- **Graphs.** Home slot k plays away slot k: 48 ordinary stint graphs. Guard
  shares are recomputed over the five defenders on the floor from the
  expected rates (as in training); node features are the as-of profiles on
  the game date.
- **Aggregation.** Outputs are averaged over slots, then probability-weighted
  over the availability scenarios (the v1 provider's, questionable /
  doubtful included). The full-health graph is the counterfactual.
- **Tests** (`test_player_graph_pregame_encoder.py`, 8):
  - five a slot, 240 minutes, each player within one minute;
  - deterministic and independent of input order;
  - impossible minutes are refused, and a 48-minute player is in every slot;
  - sides stay in place and guard shares are normalized;
  - swapping home and away swaps every output;
  - scenario weights and the absence shift are right;
  - the projection is orthonormal and Procrustes undoes a rotation.

**Outputs** (`generate_pregame_outputs.py`).

- **Style, `GE_STYLE_*` (16):** turnovers per possession, 3PA/FGA, FTA/FGA
  and offensive rebound rate per offensive side, tonight and the absence
  shift.
- **Embedding (16):** each side's pooled centered embedding on 8 principal
  directions. These are fitted per fold on its last 20,000 training stints
  and Procrustes-aligned to the previous fold, so the coordinates keep their
  meaning across checkpoints.
- **Out of fold.** Each game uses **its own month's checkpoint**, trained on
  earlier stints only, for training and predicted rows alike. A checkpoint
  not dated at the game's month start stops the run.
- **Bookkeeping.** Every row records its fold and checkpoint date. The
  manifest records coverage, missing checkpoints, missingness, the encoder's
  configuration and seed, the adapter and the projection. Seasons resume.
- **Two encoders keep the control.**
  - `gnn_node` gives the style outputs (C) and the node-only embedding (D).
  - `gnn_full` gives the graph embedding (E).
  - So E vs D isolates graph relations from neural capacity, and the style
    block does not depend on the graph question.

**Pregame store** (`build_pregame_graph_store.py`). Phase 5's v1 closing game
graphs (v1 minutes, corrected expected-guard history) plus node features as of
each game date, written per season with a manifest.

- Checked: 240 minutes per team and scenario, nobody above 48, features for
  every node, a guard row for every opposing pair, no observed or betting
  column.
- 2018-19 built in 11 minutes: 868 of 1,312 games (both teams' reports, as
  phase 5), 2.02 scenarios per game, 13.4 players per team.

**Smoke test** (2018-19, the development checkpoints of step 7 with
running-mean centering, so their values are not used for anything).

- 40 games with `gnn_full` and 20 with `gnn_node`, in 0.1 s per game after
  the projections (25 folds, 30-50 s).
- Every checkpoint is on or before its game's date; no missing values.
- Style rates are plausible: turnovers 0.138, 3PA/FGA 0.345, FTA/FGA
  0.22-0.24, offensive rebounds 0.21-0.23. Absence shifts are small and
  centered.
- A rerun skips the season and reuses the cached projections.

**Campaign v1** (`experiments/pregame_graph_v1_closing/README.md`,
pre-registered; `prepare_pregame_campaign.py`;
`report_pregame_walk_forward.py`; staged runner
`experiments/runners/run_pregame_graph_v1.sh`, **not run**).

- **Arms:**
  - A: 2_5;
  - B: + PI primary;
  - C: + `GE_STYLE_`;
  - D: C + `GE_NODE_EMB_`;
  - E: C + `GE_GRAPH_EMB_`.
- **Protocol:** phase 5's exactly (spec, fixed hyperparameters, daily refits,
  window, ten paired seeds, decision rule).
- **Comparisons:** B vs A, C vs B, D vs C, **E vs D**, and E vs C, read as a
  family; each pair must differ by exactly its column family.
- **Research file:** phase 5's pinned file plus the `GE_` columns, with
  provenance and a temporal check of every encoder checkpoint.
- **Phase 5 untouched:** its runner is reused unchanged.
- **Cost:** stages up to `prepare` take a few hours on this CPU (encoders
  about 40 + 55 min; the store needs the database). The campaign is about
  2.5× phase 5's run.
- **Known limits:** one encoder seed (training noise of the representation
  is not averaged); the 2019-25 window is not untouched; the encoder never
  sees the market (the market-conditioned encoder stays a future ablation).

## Future checks

### Guarding weights (`expected_guard`)

- [ ] Re-evaluate `k` beyond the v0 value once the main out-of-sample evaluation is complete.
- [ ] Tune or ablate the temporal half-life (365 days). Needs a rebuild (~25 min).
- [ ] Compare the 3-season window against 1, 2 and 4 seasons, and all available history with decay.
- [ ] Give pair history from the defender's current team more weight than history from previous teams (a scheme changes with the team).
- [ ] Compare using raw `r_hat` information as an edge attribute in addition to normalized `m_ij`.
- [ ] Decide which confidence variables become guard-edge attributes in the final encoder: `prior_weight`, effective co-floor exposure (`hist_cofloor_seconds_decayed` or its log), `has_pair_history`, `n_games`.
- [ ] Revisit whether position-based priors remain useful once phase 4B predicts guarding directly from player profiles and team context.
- [ ] Check whether a per-team prior (scheme / switching tendency) beats the league-wide position prior.

### Overlap

- [ ] Compare independence vs pair lift end to end (oracle, phase 5 readouts, guard benchmark on game graphs), not only on shared time.
- [ ] Revisit the lift shrinkage (k per relation) beyond v0, and whether the prior should depend on role (starter-starter vs bench) instead of only the relation.
- [ ] Check whether teammates and opponents need further different treatment (e.g. opponent lift driven by both teams' rotation patterns rather than the pair).
- [ ] Pairwise overlaps need not add up to a realizable five-man rotation over 48 minutes. Fine as expected edge weights in v0; check whether global consistency (e.g. iterative proportional fitting so each player shares exactly 4× / 5× his minutes) is needed.
- [ ] Tune / ablate half-life and window for the lift, as for `expected_guard`.
- [ ] The ~0.90 teammate prior corrects a structural limit of `min_i·min_j/48`: with a player on the floor only four teammate slots remain. Try a same-team null model that encodes the four slots explicitly (e.g. `min_i·min_j·4 / (5·48 − min_i)` symmetrized, or a fitted rotation model) instead of correcting independence with a mean lift.

### Game graph

- [ ] In the encoder ablations (phases 6-8), measure what teammate, opponent and guard edges each add on their own. The overlap gain is much larger for teammates than for opponents, so uniform-ish opponent edges may add little once guard edges are present.

- [ ] Intermediate snapshots (each snapshot's injury state).
- [ ] Guard benchmark on game graphs: expected overlap instead of the actual co-floor time, to measure how much projected minutes and overlaps degrade guard shares (part of the oracle).
- [ ] Doubtful players beyond 3 per team are folded into their most likely state (2_6 rule); revisit if games with many doubts matter.
- [ ] **Phase 4A requirement, not a v0 change.** `allocate_minutes` (2_6) has no 48-minute cap: with absences a player can be allocated more than 48 minutes (5.4% of v0 games 2019-25, 3-7% a season), and then the overlap cap `min(min_i, min_j)` binds and guard shares stop being proportional to floor time. v0 keeps it on purpose, because v0 must replicate 2_6 exactly. The 4A / v1 minutes provider must satisfy `0 <= minutes <= 48` per player and `sum = 240` per team, and we measure whether fixing it adds anything beyond being physically correct.
- [ ] Rerun the oracle with phase 4B's matchup-adjusted defensive rating in R2 (the v0 oracle uses stint RAPM `def_j`).

### Stint graph

- [x] Add node features (as-of profile, soft position, games played) once the node table exists: the phase 6 pretraining dataset joins them per stint date.
- [ ] Add a teammate familiarity attribute (historical shared minutes) once the co-play table exists.
- [x] 2016-17 stints have uniform guard shares (no matchup history): decided to keep 2016-17 with structural (position-based) shares from box-score starts, flagged by source (phase 6, step 1).
- [x] Observed guarding labels from `pair_game` (matched seconds, guard rates, shares) must skip the 69 corrupt matchup games when phase 6 uses them as targets: `edge_targets` drops them (and the two stint-rejected games).
- [ ] Very short stints (median ~73 s) give noisy labels; check whether to drop stints under some possession count or rely on possession weights.
- [ ] Possession estimate at stint boundaries: an offensive rebound credited to a stint whose miss was in the previous one. Check whether attributing it to the miss's stint fixes the negative estimates.
- [x] **Pace targets: stint duration as exposure (decided for phase 6; done in the pretraining dataset).** `poss_per_48` is weighted by possessions like the other labels, so sub-second stints (free throws around a substitution: the clock is stopped, possessions are credited) dominate it: possession-weighted SD 136 around a mean of 112 (2018-10-16 → 2019-01-31), max 28,800; the smoke test's held-out loss / constant is 1.000. Keep those stints, but weight pace targets by the stint's **seconds** (exposure), so a sub-second stint counts next to nothing against one of several minutes and the weighted mean becomes Σ poss / Σ time (equivalently, model possessions with log-duration as an offset). Per-possession targets keep possession weights. Duration stays out of the encoder's inputs.

### Phase 4A (minutes)

- [ ] Better ranking of the main substitute is what would let the shares concentrate (step 2c): coach-specific rotation patterns, lineup co-occurrence with the absent player, the substitute's minutes in the absent player's slot of the rotation.
- [ ] v2 "next man up" features improve the prior's ranking (top-1 17% → 30%) but tie on minutes; revisit them together with any ranking improvement.
- [ ] A saturation term for one player absorbing several absences (multi-absence games are where concentrated shares fail).
- [ ] Incorporate a historical roster / transactions feed (signings, waivers, trades) for the expanded roster.
- [ ] G League and two-way assignments: long returns never listed on the report fall off the expanded roster after 10 games (34 of 140 in 2018-19).
- [ ] Revisit the periods before injury reports existed (2016-17 to mid-December 2018), where long injuries drop off after 10 games and C stops modelling them, if they become important for training.
- [ ] Questionable / doubtful scenarios: v1 loses more than v0 from realized absences to the report's scenarios (+0.442 vs +0.383 player MAE, 2019-25) and gains less on teams with questionable or doubtful players; look at how scenario weights and q combine for them.
- [ ] Very long absences: returns after 31+ team games out get less from v1 than from v0 (2.82 vs 5.01 projected, actual 13.26; 400 rows 2019-25), and players still out after 11+ games get slightly more ghost minutes from v1.
- [ ] Participation q is over-confident in the middle (0.5-0.9), more on report labels: recalibrate on report-labelled data (e.g. Platt on recent months) or weight report labels up once enough exist; GBM is the challenger only if it clearly improves Brier / log-loss with good calibration.

### Phase 4B (matchups)

- [x] **Doubled matchup exposure (69 archive games, 2018-19 to 2022-23):** sensitivity check done (2026-10-14, section above): excluding them changes the guard weights negligibly; v0's settings stay. Phase 4B reads matchups through `read_matchup_history`; the pair history without them is `PointInTimeData.without_corrupt_matchups`.
- [ ] Repair the matchup store itself (re-fetch the 69 games from `BoxScoreMatchupsV3`, or halve the exposure columns once a re-fetch confirms the factor) and rebuild `pair_game`, if the exposure of those games is ever needed (e.g. phase 6 labels).
- [ ] Attacker-dependent defense (defender × attacker type) through shared parameters in the FM / GNN (phase 6), not per-cell estimates; revisit per-cell effects only if more seasons of history change their reliability.
- [ ] Efficiency on shots the NBA credits to the defender (the one shot-credited measure consistent with RAPM) as a separate component of the matchup-adjusted rating, if the time-attributed rating proves useful.
- [ ] Attacker norms: position prior and 365-day half-life were not tuned; the usage-tier type means did not transfer from 2017-18 to 2018-19.
- [ ] The matchup rating's 365-day half-life was not tuned; its penalties were chosen on the matchup ledger (attacker-game MSE), not on points allowed.
- [ ] A matchup model that also redistributes the offense: when a defender suppresses his man, whose shots replace them (attacker usage shares, the guarded attacker's teammates). That is the missing piece between a persistent matchup effect and team points, and the natural place for guard edges in phase 6.
- [ ] Guard edges with a matchup-specific defensive signal: the 2018-19 oracle hint (observed − expected guards −0.084 ± 0.035 with the matchup rating) is one season; recheck in phase 6's ablations rather than with scalar readouts.

### Phase 3 (RAPM with a profile prior)

- [ ] Decide before the solver (step 3): the prior **replaces** 2_6's ridge toward 0, `(X'X + lambda I) beta = X'y + lambda beta0`, or a base penalty toward 0 is kept and one toward `beta0` added, `(X'X + (lambda_zero + lambda_prior) I) beta = X'y + lambda_prior beta0`.
- [ ] Prior `f(profile, position) -> rating` (step 2): train on as-of (profile, rating) pairs available before each date, **weighted by the rating's reliability** (exposure / data share) or restricted to player-dates with enough history, so it learns what players of a profile are like rather than the noise of thin ratings. Rookies: earlier rookies at the same position.

### Smoke-test GNN → phase 6

- [ ] Revisit PyTorch Geometric (`HeteroData`, batching of graphs with variable node counts) when the phase 6 architecture is defined, especially for game graphs and scenarios.
- [ ] Node features currently join the stint graphs in `smoke_gnn.node_inputs`; move the join into the stint-graph loader when phase 6 fixes the inputs.
- [ ] A CPU-only PyTorch wheel (or a separate source) if the CUDA build's size matters; the machine has no GPU.
- [ ] **Hardware before buying it** (`benchmark_pretraining.py` ready with the step 6b GNN; CPU measured 2026-10-15, GPU pending on the 6 GB machine). With the first representative phase 6 GNN, benchmark CPU and GPU training time, peak GPU memory and the largest workable batch size. Start on the GPU at hand (6 GB of VRAM, on another machine; this one has none) and consider other hardware only after measuring it with the actual architecture.

### Phase 6 / phase 8 (encoder and embeddings)

- [ ] **Market-conditioned graph encoder (future ablation; no change to the current pretraining architecture or the phase 6 plan).** The current design keeps the encoder market-blind and brings betting lines in only downstream, in XGBoost. Test whether giving the encoder the market's expectation helps it learn how player relationships interact with the expected scoring environment, beyond what XGBoost already learns from the embeddings and the line separately:
  1. Baseline: market-blind GNN embeddings + the betting line and existing features in XGBoost.
  2. Market-conditioned encoder: the pregame total line as a global graph-level feature, so it can shape the graph representations.
  3. Optional: a market-blind encoder with the line entering a small neural readout after pooling.

  Only the line actually available at prediction time T, never a later closing line for an earlier snapshot. Strictly temporal out-of-fold features; judged by the increment over the existing line-aware XGBoost baseline. A later ablation can add the spread / market-implied team totals as further global context.
- [ ] Guarding interactions as a phase 6 hypothesis (phase 4B, step 2): a single additive defender rating cannot express how defensive pressure changes team scoring; test whether the FM / GNN learns attacker × defender × teammate interactions, including the redistribution of offensive opportunities.

### Node profiles

- [ ] Tune / ablate the profile window and half-life (2 seasons, 180 days) and the shrinkage priors, judged by what the embeddings or phase 3's prior gain, not on their own.
- [ ] Add size (height, weight) if a source appears; position is only inferred.
- [ ] Rookie prior from earlier rookies at the same inferred position (plan phase 3) instead of the league average.
- [ ] Opponent-adjusted rates (a player's per-36 depends on the defenses he faced).

### Positions

- [ ] Evaluate calibration of the G/F/C probabilities (reliability curve, Brier score), not only classification accuracy.
- [ ] Tune the pseudo-start count `c` and the minimum starts used as classifier labels.

### Data

- [ ] **Transient score glitches in V3.** Most score decreases are not corrections: 26 games 2012-25 have a negative stint, and in nearly all of them the score drops on a non-scoring or misordered event and recovers at the next score (instant-replay rulings, period starts, technical free throws, made shots logged out of order). The builder keeps them as signed deltas, moving 1-3 points between stints; the 24 games within the limits stay in the store. Fix: ignore a decrease that the next scoring event reverses (keep only persistent ones, e.g. a rescinded basket), then rebuild the affected games.
- [ ] Burn-in of 2016-17: `load` reads the CSVs only for requested seasons, so profiles and positions in the first weeks of 2016-17 are priors although 2015-16 box scores exist. Loading 2015-16 as box-score context only (no stints) is a possible improvement; test it in phase 6 if starting 2016-17 with little history hurts pretraining.
- [ ] Check whether the matchup store can cover Play-In (`005`) and NBA Cup final (`006`) games.
- **Two games rejected by the stint points validator (separate from the corrupt matchup games).** `0021800927` (2019-02-28) and `0022101075` (2022-03-21) failed the `points_anomaly` check added in d2d9c04 (stale scores on non-scoring events) and are no longer in the stint store, so stint-based data (RAPM history, stint graphs, overlaps) does not contain them. They remain where the defect does not reach: `pair_game` (built before the rejection; their co-floor time is a full regulation game and their tracking is unaffected, so they stay in both pair histories) and the closing 2_6 dataset (their totals are valid). The stored `expected_guard` v0 still has rows for their own pairs; the corrected tables do not. Any rebuild of stints from the archive must keep them rejected; they must not come back into stint-based training data.
- [ ] Measure matchup revisions: morning-after fetch vs later refetch for a sample of games.

### Older seasons (2012-13 → 2015-16)

Audited 2026-10-08 (raw PBP V3 and GameRotation archive, rotations rebuilt from PBP V2, stints built). Kept out of 2_7 for now; recorded so the extension can be decided on evidence.

- Stint quality is practically the same standard as 2016-25: validated stints for 98.2% (2012-13), 99.5%, 99.8%, 99.2% (2015-16) of regular-season and playoff games, every rotation rebuilt from PBP V2 (as for 2016-17); summed team PTS, FGA, FG3A, FTA, OREB, DREB and TOV equal the box score in 100% of team-games. Gaps: 2012-13's final night (16 games, no PBP V3 in the archive, the API serves it) and 3-10 games a season that failed the V2 rebuild self-check.
- Box scores are complete in the CSVs (100% of games, `START_POSITION` and `USG_PCT` populated). The DB games table starts in 2014-15, so 2012-13 and 2013-14 would also need game dates from the games CSVs.
- **No observed matchup edges before 2017-18**: NBA matchup tracking starts that season (the archive starts there too; a live API probe on 2026-10-08 hit a rate-limit block, so not re-verified at the source). Guard edges would be position-prior only (`prior_weight` 1, `has_pair_history` False) or masked. RAPM ratings would need a separate fit for as-of dates before 2017-10 (refitting from 2012 would change the ratings 2_6 uses). No official injury reports (they start 2018-12-17), so no game graphs.
- [ ] These seasons could extend the stage 1 pretraining of **teammate / opponent interactions** (stint graphs: nodes, teammate and opponent edges, stint labels).
- [ ] In phase 6, compare pretraining from 2012-13 against the current history (2016-17 on) before taking on the extra cost. Note the era shift: pace in the stints rises from 93.7 possessions per 48 (2012-13) to ~102 (2018-25).

## Done

- **k for v0** (2026-10-07): swept on 2018-19 only; 300 frozen (see table above).
- **Phase 3 closed** (2026-10-08): profile prior for offense and defense, zero prior for pace (out-of-sample 2019-25 result above).
- **Phase 4A closed** (2026-10-09): minutes provider v1 replaces v0 in the game graph; better minutes in all 7 out-of-sample seasons, R1 within noise (result above).
- **Scalar matchup-adjusted defensive rating rejected** (2026-10-14): no information on future stints' points beyond RAPM on 2018-19; kept as a documented experiment (phase 4B, step 2).
- **Phase 4B closed** (2026-10-14): guard-edge provider `expected_guard` v0 with its frozen settings (k 300, half-life 365 days, 3-season window), insensitive to the corrupt matchup games; phase 5 onward reads the corrected pair history (`without_corrupt_matchups`), the original stays the phase 2 reference; the scalar matchup rating rejected; stint RAPM stays `def_j`. Guarding interactions move to phase 6 as a hypothesis; phase 6 guarding labels exclude the same 69 games.
- **2016-17 stint store rebuilt** (2026-10-08): placeholder-score corruption removed; stint point validation and audit added; phase 3 steps 1 and 2a rerun on clean data (zero-prior control re-frozen at 3,000 / 10,000).
- **Box-score backfill for 2016-17 and 2017-18** (2026-10-08): `as_of` falls back to the season CSVs for whole seasons the DB lacks; positions, node profiles and `expected_guard` rebuilt; game-graph rosters keep the DB rows only, and 2_6 is still reproduced exactly. k re-swept on 2018-19 only: 300 stays (see the frozen table and the benchmark).
