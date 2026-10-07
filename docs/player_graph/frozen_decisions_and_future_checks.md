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

### Positions

- [ ] Evaluate calibration of the G/F/C probabilities (reliability curve, Brier score), not only classification accuracy.
- [ ] Tune the pseudo-start count `c` and the minimum starts used as classifier labels.
- [ ] Add a profile for 2017-18 bench players (no box scores) from matchup production, or backfill the 2016-17 / 2017-18 box scores.

### Data

- [ ] Check whether the matchup store can cover Play-In (`005`) and NBA Cup final (`006`) games.
- [ ] Measure matchup revisions: morning-after fetch vs later refetch for a sample of games.

## Done

- **k for v0** (2026-10-07): swept on 2018-19 only; 300 frozen (see table above).
