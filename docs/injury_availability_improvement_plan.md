# Injury availability: improvement plan

**Status:** plan only — nothing implemented.
**Depends on:** the Aiven `injury_report` store from branch
`lab/injury-report-intermediate-times` (see `injury_report_db_plan.md` there).
**Touches:** `data_processing/past_injuries/`, `data_processing/players/`,
`create_training_data/create_df_to_predict.py`, the intermediate-line dataset,
and the "Player And Injury Features" / "Availability Effect Features" sections of
`README_Training Data Processing.md`.

---

## 0. Summary

The current injury family measures the wrong thing, at the wrong time, with an
estimator that has no usable signal.

- **Wrong estimator.** The `TOP3_*AVAILABILITY_EFFECT_*` columns are team
  with-minus-without means per player. Against the market they carry no forward
  signal, the spread-error effects are **identically zero** since 2020 (the
  empirical-Bayes fit finds no between-player variance), and the total-points
  effect mostly encodes team scoring level (r = +0.20 with the total line, even for
  players who are *playing*).
- **Wrong source.** Training reads the post-game inactive list and box-score
  comments; production reads the pre-game report. At T-3h, only 79% of the rotation
  players training marks injured were reported `Out`; 23% of team-games differ by
  20 or more rotation minutes.
- **Wrong population.** Half the injured rows are low-minute players — mostly
  G League / two-way assignments — which inflates `N_INJURED_PLAYERS` and the
  `TOTAL_INJURED_*` sums.

What does work is simpler: **expected absent rotation minutes known at decision
time**, weighted by the report status. It predicts line error and spread error
at r ≈ 0.05 against the decision-time line, survives line/month/season controls,
and is as strong from the pre-game report as from the post-game list.

This plan replaces the family in four phases, each gated by a campaign.

---

## 1. Evidence

All numbers are regular season, 2021-22 to 2025-26 (the hourly-report era),
unless stated. "Rotation player" = season-to-date average of at least 20 minutes
before the game. Lines are SBR cross-book consensus; T-3h means 180 minutes
before tip. Analysis scripts were exploratory (not committed).

### 1.1 Current features against the market (2019-2025, 2.3 closing dataset)

| Feature (game level) | vs line error | vs spread error | per-season sign |
|---|---|---|---|
| Injured effect on total points (sum) | +0.006 | — | flips |
| Injured effect on line error (sum) | +0.009 | — | flips |
| Active-player effect on line error (sum) | +0.013 | — | flips |
| Injured effect on spread error (diff) | — | −0.016 | column is 0 in 2020-2025 |
| `TOTAL_INJURED_PLAYER_MIN` (sum / diff) | +0.034 | +0.029 | mostly positive |

Adding the effect features to injured points did not improve forward prediction
of either error.

### 1.2 Post-game injured set vs the pre-game report

Rotation players that training marks injured, by report status:

| As of | Out | Doubtful | Questionable | Not listed | Team not filed |
|---|---|---|---|---|---|
| T-6h | 66% | 4% | 13% | 4% | 14% |
| T-3h | 79% | 4% | 14% | 3% | 0% |
| T-30m | 90% | 2% | 7% | 1% | 0% |

P(did not play | status at T-3h), rotation players: Out 99.9%, Doubtful 97%,
Questionable 44% (stars 38.5%), Probable 4%, not listed 2%. The report's `Out` is
almost never wrong; `Questionable` is the information the pipeline ignores.

A probability-weighted absence tracks actual missing minutes better than a binary
`Out` flag (r = 0.94 vs 0.91).

### 1.3 The market prices injury news fast

2,862 events where a rotation player (24+ mpg) moved to `Out` during the day:

| Consensus move (points) | Before the report shows it | First 30 min after | 30 min → close |
|---|---|---|---|
| Spread (affected team) | −0.36 | −0.13 | −0.03 |
| Total | −0.27 | −0.08 | +0.05 |

Upgrades are symmetric; a placebo is flat. News arriving after T-3h moves the line
strongly (r = 0.58 for spread) but leaves nothing against the close (r ≈ −0.01).
**There is no edge in reacting to news faster than the market.**

### 1.4 What carries signal

Rotation minutes expected absent at T-3h, against the T-3h line (5,721 games):

| Absence measure × value weight | line error | spread error |
|---|---|---|
| Report probability × minutes (production-usable) | **+0.052** [+0.025, +0.082] | **+0.046** [+0.020, +0.072] |
| Post-game injured set × minutes (training today) | +0.044 | +0.056 |
| Report probability × points / usage-minutes | +0.046 / +0.045 | +0.043 / +0.042 |
| Any absence × plus-minus per game | ≈ 0 | ≈ 0 |

- **Controls:** partial r with T-3h line, month and season controlled: 0.039 (totals), 0.045 (spread). Spread is positive in 5/5 seasons.
- **Where it lives:** concentrated in heavy-absence games. The top quintile goes over 54% (mean error +2.0); strongest February-April.
- **Fresh vs persistent:** today's absence minus the team's recent absence. Totals load more on **fresh** absence (0.019 vs 0.011 per minute); spread loads more on **persistent** absence (0.014 vs 0.007).

### 1.5 Star-specific history

642 fresh star `Out` events (18+ ppg, 28+ mpg, played the previous game; 116 from
"not listed"/`Probable`, 526 from `Questionable`):

| Forward r (fit earlier seasons) | spread move | total move | ATS error | total error |
|---|---|---|---|---|
| Star value only | 0.26 | 0.32 | −0.03 | +0.02 |
| + team on/off history | 0.26 | 0.38 | +0.01 | −0.03 |
| + market's past reaction to this star | **0.33** | **0.41** | −0.01 | +0.01 |

History predicts **how far the line moves**, not whether the market ends up wrong.
One weak, unconfirmed hint: stars whose earlier news dropped the total a lot saw
the next game go over the close more often (partial r −0.08, ~80 comparisons run).
`Questionable` stars at T-3h are priced fairly (spread error −0.15 [−0.79, +0.44]).

---

## 2. Goals and non-goals

**Goals**

1. One availability definition, computed from information published before the
   decision time, shared by training and prediction.
2. Features that express *expected* absence, its uncertainty and its novelty,
   weighted by player value.
3. Remove columns that are constant, confounded or unmeasurable.
4. Prove each change in-model for `total_points_regressor`,
   `line_error_regressor` and `spread_error_regressor`.

**Non-goals**

- Beating the market to injury news (§1.3 shows there is nothing there).
- Per-player on/off or past-market-error estimates as model features (§1.1, §1.5).
- A line-movement model. Possible later (§8), separate from the three targets.

---

## 3. Design principles

- **As-of, strictly before.** Every availability value is read from report spans
  with `valid_from < as_of <= valid_to`; never fall forward to a later report.
- **Absence is not health.** "Not listed" counts as available only when the team
  had filed by `as_of`; otherwise emit an explicit unknown flag.
- **Same decision time as the odds.** For the closing dataset use a fixed horizon
  (default T-30m, configurable); for the intermediate dataset use each row's
  `snapshot_minutes`. A row's injury view and its line must describe the same instant.
- **Probabilities are fitted walk-forward.** Nothing measured in §1 (e.g. 44% for
  `Questionable`) is hard-coded; it is fitted on seasons strictly before the row.
- **Value weights are pre-game.** Minutes, points and usage are season-to-date
  means over strictly earlier dates, with the existing previous-season fallback.

---

## 4. Phase 0 — prune (ablation first, deletion after the campaign)

| Change | Reason | Where |
|---|---|---|
| Drop `TOP3_*AVAILABILITY_EFFECT_*_SPREAD_ERROR` and `*_MEAN_SE_SPREAD_ERROR` | constant 0 since 2020 | `past_injuries/injury_effects.py` |
| Drop `*_MAX_ABS_*` | discards sign | same |
| Drop `*_MEAN_TOTAL_POINTS` effects | encode team level, not player impact | same |
| Keep `*_DIFF_FROM_LINE` effects for one campaign only | weakest-but-nonzero member; decide by ablation | same |
| Restrict `TOP*_INJURED_*`, `AVG_INJURED_*`, `TOTAL_INJURED_*`, `N_INJURED_PLAYERS` to rotation players | half the rows are two-way / G League noise | `players/attach_player_features.py`, `past_injuries/past_injuries.py` |
| Exclude roster-mechanics reasons (report category `g_league`, which covers two-way and on-assignment players) from "injured" | not basketball absences | same |

**Deliverable:** a pruned build behind a flag (`availability_feature_set="pruned"`)
so the campaign can compare it with the current set on the same CSV.

---

## 5. Phase 1 — point-in-time availability

### 5.1 Data access

- Merge the Aiven injury-report reader (`postgre_db/injury_report_aiven/fetch.py`:
  `status_as_of`, `coverage_at`) from `lab/injury-report-intermediate-times`.
- Add a bulk reader: one query per season returning spans + filings for a set of
  games, then resolve as-of in pandas (per-game SQL is too slow for training).
- New module `data_processing/availability/report_asof.py`:
  - `player_status_asof(spans, filings, games, minutes_before_tip) -> frame`
    with `status`, `reason`, `report_age_minutes`, `team_filed`.
  - Leakage tests mirroring `fetch.py`'s rules: strict `<`, no fall-forward,
    unfiled team → unknown, span clamped at tip.

### 5.2 Absence probability model

New module `data_processing/availability/absence_probability.py`:

```text
P(absent | status, reason_category, minutes_to_tip bucket, report era, player value tier)
```

- Empirical rates with additive smoothing toward the status-level rate, fitted on
  **seasons strictly earlier** than the row (walk-forward), refit each season.
- Player value tier is required: `Questionable` stars sit 38.5% vs 44% overall.
- `Out` and `Doubtful` rates are near 1; `not listed` with a filed team near 0.
- Unknown (team not filed) → the league prior for "not listed at this horizon",
  plus the unknown flag.
- Output per player-game: `p_absent`.

### 5.3 Player value

Pre-game, per player (season-to-date over strictly earlier dates, previous-season
fallback, zero if neither):

- `avg_min`, `avg_pts`, `usg_min = mean(USG_PCT × MIN)`.
- Rotation flag: `avg_min >= 20` and at least 5 games played (both configurable).

Plus-minus weighting is explicitly **not** used (§1.4).

### 5.4 Team features (before home/away merge)

For each team-game, over rotation players on the roster at `as_of`:

| Feature | Definition |
|---|---|
| `AVAIL_EXP_ABSENT_MIN_BEFORE` | Σ p_absent × avg_min |
| `AVAIL_EXP_ABSENT_USG_MIN_BEFORE` | Σ p_absent × usg_min |
| `AVAIL_ABSENT_MIN_VARIANCE_BEFORE` | Σ p(1−p) × avg_min² (uncertainty) |
| `AVAIL_QUESTIONABLE_MIN_BEFORE` | Σ avg_min over `Questionable`/`Doubtful` players |
| `AVAIL_TOP_PLAYER_P_ABSENT_BEFORE` | p_absent of the team's highest `usg_min` player |
| `AVAIL_RECENT_ABSENT_MIN_BEFORE` | mean *actual* absent rotation minutes over the team's previous 5 games |
| `AVAIL_FRESH_ABSENT_MIN_BEFORE` | `EXP_ABSENT_MIN − RECENT_ABSENT_MIN` |
| `AVAIL_TEAM_FILED_BEFORE` | team had filed by `as_of` (0/1) |
| `AVAIL_REPORT_AGE_MIN_BEFORE` | minutes since the report read |

`RECENT_ABSENT_MIN` uses post-game truth for games **already played**, which is
leakage-safe (strictly earlier games) and matches what production knows.

### 5.5 Game features (after merge)

- Totals: `_SUM_BEFORE` of `EXP_ABSENT_MIN`, `EXP_ABSENT_USG_MIN`,
  `FRESH_ABSENT_MIN`, `RECENT_ABSENT_MIN`, `ABSENT_MIN_VARIANCE`.
- Spread: `_DIFF_BEFORE` as away − home of the same (positive favours home),
  consistent with the `SPREAD_ERROR` convention in `config/market_columns.py`.

### 5.6 Coverage before the hourly era

| Seasons | Report data | Policy |
|---|---|---|
| 2021-22 onward | hourly / 15-min | report as-of |
| Dec 2019 - 2020-21 | ~3 reports/day | report as-of, with `AVAIL_REPORT_AGE_MIN` making staleness visible |
| before Dec 2019 | none | post-game set with `AVAIL_SOURCE_IS_REPORT_BEFORE = 0` |

Campaigns already floor at 2019; the ablation in §7 decides whether pre-report
rows help or should be dropped for this family.

### 5.7 Wiring

- `create_df_to_predict()` gains `availability_as_of_minutes` (closing dataset)
  and calls the new builders after player statistics, before the home/away merge.
- The intermediate-line builder resolves `as_of` per snapshot row.
- Production (`predict_nba_games.py`) replaces `injury_dict_scheduled` with the
  same as-of reader on the latest archived report, so training and serving share
  one code path. The live PDF parser remains only as a fallback when the archive
  is behind, with a warning.

---

## 6. Phase 2 — optional hypotheses (one campaign cell each)

These showed weak or unconfirmed evidence; they enter only as isolated cells.

1. **Star's previous total move.** Mean total-line move at the same star's
   earlier `Out` news, shrunk by count. Hint only (§1.5).
2. **Heavy-absence indicator.** `EXP_ABSENT_MIN_SUM` above the season's 80th
   percentile (fitted walk-forward), since the signal concentrates there.
3. **Legacy `DIFF_FROM_LINE` effect** kept alongside the new family, to confirm
   it adds nothing in-model before deleting it.

---

## 7. Validation campaign

Dataset: one schema bump (next free `TRAINING_DATA_SCHEMA_VERSION`) carrying the
current family, the pruned family and the Phase 1 family together, so cells
differ only in `cleaning.exclude_cols_containing`.

| Cell | Injury columns kept | Read against |
|---|---|---|
| A | none | control |
| B | current family (as today) | A |
| C | Phase 0 pruned | B |
| D | Phase 1 point-in-time family | B, C |
| E | D + Phase 2 hypothesis cells (one at a time) | D |

Run for all three strategies, on both the closing and the intermediate (T-3h)
datasets. Protocol as in the latest schema campaigns (fixed folds, seeds
17/42/91, `--no-save-model`). Add a `tests/test_*_campaign_configs.py` pinning
each cell's kept columns, and a reporting factor for the injury feature set (the
referee campaign showed exclusion-only campaigns otherwise pool as replicates).

**Gate** (in addition to the standard gate in the experiments skill):

- Adopt D only if it beats B by more than the seed range on CV win rate for at
  least one strategy, without worsening the others.
- Adopt C over B if C ≥ B within noise: fewer columns at equal performance wins.
- Spread: require the per-season direction to agree in at least 4 of 6 seasons.

---

## 8. Phase 3 — production and documentation

- Delete the losing families from the build; keep one schema bump per removal.
- Update `README_Training Data Processing.md` (availability sections, leakage
  controls) and the feature-engineering skill reference for family 4.
- Rebuild the meta-learner dataset (`scripts/build_meta_learner_training_data.py`)
  and retrain through staging.
- Monitoring: log per prediction day the share of games with `team_filed = 0` and
  the report age, so a stalled archive is visible.

**Possible follow-up (separate project):** a per-star "expected line move if
ruled out" model (forward r 0.33 spread / 0.41 total, §1.5). It predicts market
movement, not market error, so it belongs to bet-timing decisions for
unresolved `Questionable` stars, not to the three prediction targets.

---

## 9. Tests

- **As-of reader:** strict `<` at a report stamp; no fall-forward; unfiled team
  → unknown; span clamped at tip; DST fall-back ordering in UTC.
- **Probability model:** fitted only on earlier seasons (mutate a later season's
  outcomes, earlier rows unchanged); smoothing toward the status prior with few
  samples; value tier respected.
- **Features:** rotation filter excludes a two-way player; fresh = expected −
  recent, where recent uses only earlier games; home/away swap negates every
  `_DIFF_BEFORE` and leaves `_SUM_BEFORE` unchanged.
- **Parity:** the same game built for training and for same-day prediction yields
  identical availability features.
- **Leakage gate:** every new column ends in `_BEFORE`; no target-game minutes are
  read (extend `test_target_game_minutes_cannot_move_a_player_between_buckets`).
- Mutation-test each guard (revert, confirm failure, restore).

---

## 10. Risks and open questions

- **Small effects.** r ≈ 0.05 is a supporting signal, comparable to the referee
  free-throw tendency; expect in-model gains near seed noise.
- **Short report history.** Hourly data covers five seasons; the probability model
  has few `Doubtful`/`removed` cases per value tier.
- **Timestamp granularity.** Hourly reports before Dec 2025 blur "before vs after
  the report"; the 15-minute era should be used to re-check §1.3.
- **Decision time.** The closing dataset's horizon (T-30m default) must match when
  bets are actually placed; if production bets earlier, the intermediate dataset
  is the honest benchmark.
- **Name resolution.** Unresolved report names (`ir_unresolved`, up to 477
  occurrences in 2024) silently remove players; the builder should fail above a
  threshold, as the referee coverage check does.
- **Open:** keep pre-2019 rows with a source flag, or floor the family at the
  report era? Decided by the §7 ablation.
