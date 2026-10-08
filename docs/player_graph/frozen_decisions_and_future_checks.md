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

- [ ] Add node features (as-of profile, soft position, games played) once the node table exists.
- [ ] Add a teammate familiarity attribute (historical shared minutes) once the co-play table exists.
- [ ] 2016-17 stints have uniform guard shares (no matchup history): start pretraining in 2017-18, or keep them (plan phase 6 decision).
- [ ] Very short stints (median ~73 s) give noisy labels; check whether to drop stints under some possession count or rely on possession weights.
- [ ] Possession estimate at stint boundaries: an offensive rebound credited to a stint whose miss was in the previous one. Check whether attributing it to the miss's stint fixes the negative estimates.
- [ ] **Pace targets: stint duration as exposure (decided for phase 6).** `poss_per_48` is weighted by possessions like the other labels, so sub-second stints (free throws around a substitution: the clock is stopped, possessions are credited) dominate it: possession-weighted SD 136 around a mean of 112 (2018-10-16 → 2019-01-31), max 28,800; the smoke test's held-out loss / constant is 1.000. Keep those stints, but weight pace targets by the stint's **seconds** (exposure), so a sub-second stint counts next to nothing against one of several minutes and the weighted mean becomes Σ poss / Σ time (equivalently, model possessions with log-duration as an offset). Per-possession targets keep possession weights. Duration stays out of the encoder's inputs.

### Phase 4A (minutes)

- [ ] Better ranking of the main substitute is what would let the shares concentrate (step 2c): coach-specific rotation patterns, lineup co-occurrence with the absent player, the substitute's minutes in the absent player's slot of the rotation.
- [ ] v2 "next man up" features improve the prior's ranking (top-1 17% → 30%) but tie on minutes; revisit them together with any ranking improvement.
- [ ] A saturation term for one player absorbing several absences (multi-absence games are where concentrated shares fail).
- [ ] Participation q is over-confident in the middle (0.5-0.9), more on report labels: recalibrate on report-labelled data (e.g. Platt on recent months) or weight report labels up once enough exist; GBM is the challenger only if it clearly improves Brier / log-loss with good calibration.

### Phase 3 (RAPM with a profile prior)

- [ ] Decide before the solver (step 3): the prior **replaces** 2_6's ridge toward 0, `(X'X + lambda I) beta = X'y + lambda beta0`, or a base penalty toward 0 is kept and one toward `beta0` added, `(X'X + (lambda_zero + lambda_prior) I) beta = X'y + lambda_prior beta0`.
- [ ] Prior `f(profile, position) -> rating` (step 2): train on as-of (profile, rating) pairs available before each date, **weighted by the rating's reliability** (exposure / data share) or restricted to player-dates with enough history, so it learns what players of a profile are like rather than the noise of thin ratings. Rookies: earlier rookies at the same position.

### Smoke-test GNN → phase 6

- [ ] Revisit PyTorch Geometric (`HeteroData`, batching of graphs with variable node counts) when the phase 6 architecture is defined, especially for game graphs and scenarios.
- [ ] Node features currently join the stint graphs in `smoke_gnn.node_inputs`; move the join into the stint-graph loader when phase 6 fixes the inputs.
- [ ] A CPU-only PyTorch wheel (or a separate source) if the CUDA build's size matters; the machine has no GPU.

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
- **2016-17 stint store rebuilt** (2026-10-08): placeholder-score corruption removed; stint point validation and audit added; phase 3 steps 1 and 2a rerun on clean data (zero-prior control re-frozen at 3,000 / 10,000).
- **Box-score backfill for 2016-17 and 2017-18** (2026-10-08): `as_of` falls back to the season CSVs for whole seasons the DB lacks; positions, node profiles and `expected_guard` rebuilt; game-graph rosters keep the DB rows only, and 2_6 is still reproduced exactly. k re-swept on 2018-19 only: 300 stays (see the frozen table and the benchmark).
