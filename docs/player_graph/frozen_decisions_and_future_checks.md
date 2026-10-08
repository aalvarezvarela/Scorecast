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
| Data | Player box scores | Start in 2018-19 (DB has none before) | Measured 2026-10-07: 0 player rows for 2014-15 to 2017-18 | If pretraining should start in 2016-17 / 2017-18 |
| Data | Preseason and All-Star rows | Excluded from position profiles | Do not describe a role | — |
| `pair_game` | Guarding rate denominator | Our stints' co-floor seconds | NBA `pct_total_time_both_on` uses ≈ 0.39 × co-floor time (SD 0.056, 2023-24) and is rounded to 3 decimals | — |
| `pair_game` | Matchup rows with no shared stint | Kept, `cofloor_seconds` 0, rate NaN | 3-151 rows a season (< 0.06%) | — |
| `pair_game` | Rate above 1 | Clipped at 1, raw seconds kept | 10-31 rows a season, up to 34 s over | — |
| `pair_game` | Shared floor, no matchup row | Real zero | ~6.5% of directed pairs | — |
| Positions | Representation | Soft G/F/C probabilities | Avoids a hard threshold between starts and profile | Phase 4B |
| Positions | Blend | `(starts + c·q) / (n_starts + c)`, `c = 5` pseudo-starts | Not tuned | See checks |
| Positions | Profile classifier | Multinomial LR on per-36 AST, OREB, DREB, BLK, STL, FG3A, FGA, PF; trained on players with ≥ 10 starts; rates shrunk with 200 league-average minutes | Out of sample (next-60-day start position): 80-93% accuracy, 60-84% for < 10 prior starts | See checks |
| Positions | No box scores and no starts | League share of starts per position | Affects 2017-18 bench players | — |
| Positions | Window | 3 seasons before D | Same as the guarding history | — |
| `expected_guard` | Shrinkage strength `k` | **300 co-floor seconds** | Chosen on **2018-19 only**: weighted TVD 0.2755 (k=200), 0.2756 (300), 0.2786 (600), 0.2860 (1200). 200 and 300 tie; 300 kept as the slightly more conservative value | After the main out-of-sample evaluation |
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

## v0 benchmark (the reference phase 4B must beat)

`python scripts/player_graph/evaluate_expected_guard.py --seasons 2019-2025`
(provider v0, k = 300, half-life 365 d, window 3 seasons). Weighted TVD between
expected and observed in-game guarding shares, 188,238 attacker-games:

| Estimator | All | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constant rate (co-floor only) | 0.3605 | 0.3947 | 0.3809 | 0.3561 | 0.3561 | 0.3581 | 0.3452 | 0.3390 |
| Position prior only | 0.2996 | 0.3267 | 0.3125 | 0.2972 | 0.2990 | 0.3004 | 0.2863 | 0.2799 |
| Pair history only | 0.2713 | 0.2883 | 0.2807 | 0.2737 | 0.2685 | 0.2658 | 0.2617 | 0.2638 |
| **v0 (k = 300)** | **0.2571** | 0.2753 | 0.2666 | 0.2576 | 0.2553 | 0.2542 | 0.2470 | 0.2471 |

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

Intermediate (`--intermediate …`), against `intermediate_line_data_2_6_20261003`
with each snapshot's report state and history date; games without a graph
are exactly the file's NaN rows at every horizon:

| Window | Horizons | Snapshots with graphs | vs `project_lineup_games` | vs file |
| --- | --- | --- | --- | --- |
| 2023-12-01 → 2023-12-31 | 0, 360, 720, 1080 | 208 / 168 / 152 / 150 | ≤ 1.4e-13 | 100% equal |
| 2019-10-22 → 2019-11-10 | all 17 | 139 (T-0) … 89 (T-1080) | ≤ 5.7e-14 | 100% equal |

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

### Positions

- [ ] Evaluate calibration of the G/F/C probabilities (reliability curve, Brier score), not only classification accuracy.
- [ ] Tune the pseudo-start count `c` and the minimum starts used as classifier labels.
- [ ] Add a profile for 2017-18 bench players (no box scores) from matchup production, or backfill the 2016-17 / 2017-18 box scores.

### Data

- [ ] Check whether the matchup store can cover Play-In (`005`) and NBA Cup final (`006`) games.
- [ ] Measure matchup revisions: morning-after fetch vs later refetch for a sample of games.

## Done

- **k for v0** (2026-10-07): swept on 2018-19 only; 300 frozen (see table above).
