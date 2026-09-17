# Intermediate snapshots: injury news, market reaction, continuation and cross-market features

Status: **implemented** on `feat/intermediate-market-dynamics` (G1–G4; §7 lists where the
code differs from this plan). Not yet ablated (§6). Schema stays `2_5`. The measurements below
come from exploratory scripts run on 2026-09-16. They cover the line-history
store's seven seasons (`season_year` 2019–2025: 8,884 games, all three markets,
all books) and the injury-report store (2018 onward).

**Revised 2026-09-16 after reviewing `feat/historical-ridge-line-movement`.**
That branch already implements part of this plan:
- the total-line continuation expectation (G4 for totals), as a walk-forward Ridge;
- five path features for the anchor total;
- a leave-one-book-out book deviation.

The plan now assumes that branch is merged first. G4 reuses its Ridge engine
instead of the cached XGBoost models proposed originally (§1.6, §3 G4). §1.8
measures the branch's features.

This plan builds on [intermediate_line_feature_engineering_plan.md](intermediate_line_feature_engineering_plan.md).
Its Tier 6, cross-market coherence, was never built; §3 G3 below is that tier,
now backed by measurements.

---

## 0. Scope and what is not duplicated

At each snapshot, schema 2_5 already provides:

- **Injury state as of the snapshot:** availability groups, report counters,
  coverage, report age, and per-player `P_PLAY` / `EFFECT_*` / `FORM_*` for
  Questionable and Probable players.
- **Injury aggregates:** `SUM_*_EXP_{MIN,PTS}` and the availability effects.
- **Market state and movement:** per-book and consensus levels, move from open,
  `MOVE_LAST_{15..720}` per book, velocity, acceleration, steam counts,
  cross-book std/range, book deviation from consensus, reversals, path shape
  and line age.
- **From `feat/historical-ridge-line-movement`:**
  - `ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE`;
  - the anchor-total path features `ODDS_SNAP_TOT_<ANCHOR>_{MINUTES_SINCE_LAST_LEVEL_MOVE, PEERS_MOVED_ANCHOR_STILL_60, ABS_LEVEL_PATH_60, SIGNED_MOVE_STREAK_60, LAST_TWO_LEVEL_MOVES_GAP_MIN}`;
  - book deviation measured against the other books' median, not a consensus that includes the book itself.

This plan adds only information about **how the situation evolved up to the
snapshot**:

| Group | What it answers |
|---|---|
| G1 injury news | What changed in the reports recently, and how important were the players? |
| G2 news-anchored reaction | How far has the market moved since the news, against what that news usually moves? |
| G3 cross-market | Are total, spread and moneyline telling a consistent story? |
| G4 continuation expectation | Historically, where does a line in this state go before the close? |

No level of expected missing minutes or points, and no play probability, is
re-emitted. Those already exist.

---

## 1. What the data says

Eras used throughout:

| Era | Seasons | Report cadence | Games |
|---|---|---|---|
| 3/day | 2019–2020 | 3 reports/day, bubble included | 2,309 |
| hourly | 2021–2024 | every hour at :30 | 5,263 |
| 15-min | 2025 | every 15 min | 1,312 |

SEs are clustered by game and demeaned by season × horizon unless stated
otherwise. **About 250 regressions were run across both exploration rounds;
treat any single t≈2 as unconfirmed.**

### 1.1 Report timestamps are publication times

- The HTTP `Last-Modified` header of each PDF sits a median of **6 seconds** after
  its label time in every season, so the strict `valid_from < T` rule in 2_5 is correct.
- Around clean single-player status changes, the share of the eventual spread move
  (−240 → +120 min) already done at each offset is:

| Era | −120 | −60 | −15 | report stamp | +15 | +60 |
|---|---|---|---|---|---|---|
| 3/day | 0.36 | 0.44 | 0.55 | 0.64 | 0.81 | 0.91 |
| hourly | 0.17 | 0.24 | 0.50 | 0.57 | 0.91 | 0.96 |
| 15-min | 0.10 | 0.16 | 0.26 | 0.47 | 0.78 | 0.95 |

- **Much of the move precedes the report,** because news breaks elsewhere first
  (most in the 3/day era).
- **The rest arrives within ~15 min after it.** That is market latency, not a
  publication delay.
- **2019–20 timestamps line up with market reactions,** so there is no timezone problem in the new seasons.

### 1.2 Defining "news": fixed time windows, not "previous report"

- **"Change since the previous report" means a different thing per era:** a
  multi-hour window in 3/day, 60 min in hourly, 15 min in 15-min.
- **A fixed time window means the same thing everywhere.**
  - "The latest report at least W minutes earlier" is the same quantity, because
    the state only changes at reports.
- **Share of snapshots with material news** (|spread-signed news| ≥ 3 expected points):

| Window | 3/day | hourly | 15-min |
|---|---|---|---|
| W15 | 0.000 | 0.000 | 0.034 |
| W60 | 0.065 | 0.103 | 0.139 |
| W240 | 0.191 | 0.246 | 0.315 |
| since previous game | 0.627 | 0.726 | 0.749 |

- **W15 is degenerate outside 2025. Use W60 and W240, plus "since the previous game".**
- **Market price of news is similar across eras:** the same-window consensus move
  per expected missing point, material rows only:

| Market | 3/day | hourly | 15-min |
|---|---|---|---|
| Spread (pts) | 0.027 | 0.032 | 0.035 |
| Total (pts) | 0.019 | 0.028 | 0.030 |
| Moneyline (prob) | 0.0007 | 0.0009 | 0.0009 |

- **Walk-forward by season, spread:** 0.025, 0.027, 0.032, 0.036, 0.031, 0.030, 0.035.
- **Raw counts drift badly:** rotation-player transitions per game went from 1.4
  (2021) to 2.8 (2025).
  - About 15% of 2023+ transitions are Out→Out reason-only changes.
  - **Do not emit counts.**

### 1.3 Encoding who changed: change in chance of sitting × `FORM_PTS`

- **Candidate weightings compared.** R² of the market response to clean single-player events:

| Weighting | Spread | Moneyline | Total |
|---|---|---|---|
| Δp × `FORM_PTS`, p from 2_5 `P_PLAY` | 0.279 | 0.282 | 0.072 |
| Δp × `FORM_PTS`, flat status map | 0.283 | 0.289 | 0.071 |
| Δp × `FORM_MIN` | 0.208 | 0.212 | 0.039 |

- Earlier rounds found usage×minutes and PIE×minutes are close to points, and a
  starter share is worse.
- **2_5 `P_PLAY` is as good as a flat map, so reuse it rather than adding
  anything.** P(play | Questionable held at T−h) is flat by horizon (0.56–0.60),
  so the any-time `P_PLAY` is valid at every snapshot.
- **Using Δp instead of the transition label handles types automatically.**
  Market move per point of `FORM_PTS`:

| Transition | n | mean Δp | Spread / pt | Total / pt |
|---|---|---|---|---|
| new → Out | 404 | 1.00 | −0.057 | −0.090 |
| Q → Out | 234 | 0.55 | −0.044 | −0.079 |
| new → Q | 422 | 0.50 | −0.038 | −0.076 |
| Q → Available | 121 | −0.48 | +0.020 | +0.057 |
| Out last game → Q | 84 | −0.51 | +0.002 | +0.002 |

- The last row matters: without carrying a player's status from the team's
  previous game, every long-term absence would read as fresh news each game.

### 1.4 Market reaction to news

- **Spread under-reacts to injury news; totals over-react.**
- **Spread:**
  - The injury-explained part of the bet365 spread move keeps moving toward the
    close in every era. Future-move slope +0.28 (t 3.0) in 3/day, +0.15 (t 4.0)
    hourly, +0.11 (t 2.3) 15-min.
  - The unexplained part does not.
  - Following it gives CLV (closing-line value) of **+0.10 pts** (95% CI 0.02–0.18), but covers only 52.1%.
- **Totals:**
  - The injury-explained part of the bet365 total move reverses in the final score:
    −5.0 (t −2.1) in 3/day, −3.0 (t −3.1) hourly, but +0.9 (t 0.6) in 15-min.
    Pooled W240: −2.1 (t −3.0).
  - The close does not correct it: fading has CLV ≈ 0, covers 53.1% (n=783, by
    season 62.9 / 53.5 / 54.0 / 51.7 / 50.2%).
  - It is fading, which is consistent with the market learning.
- **Unexplained total moves continue into the close** (t 9.9 hourly, 4.2 3/day,
  1.1 15-min). CLV +0.19, but they only cover 50.7%.
- **"News happened, line hasn't moved" does not lead to a catch-up.** In most such
  cases the market priced the news before the report.
- **Anchor choice.** Correlation between actual and expected move, measured from
  `a` minutes before the news:

| Anchor a | 0 | 30 | 60 | 120 |
|---|---|---|---|---|
| Spread | 0.43 | 0.54 | 0.56 | 0.58 |
| Total | 0.29 | 0.36 | 0.38 | 0.38 |
| Moneyline | 0.44 | 0.53 | 0.56 | 0.57 |

- **Use a = 60.** It sits on the plateau without reaching back into unrelated earlier movement.

### 1.5 Cross-market

- **The moneyline-implied home margin minus the spread** (13.5 × Φ⁻¹(p_home) − spread,
  where Φ⁻¹ is the inverse normal CDF) predicts the bet365 spread's move to close
  in **all seven seasons**: t = 4.3, 2.2, 2.7, 3.5, 4.2, 2.5, 5.0; pooled +0.088 (t 9.0).
  - It also predicts spread error: +0.36 (t 2.3) pooled, +0.55 (t 3.0) controlling
    for the spread level.
  - The error effect is **~0 in 2019–2022 and positive in 2023–2025**.
  - bet365's own gap carries it; the consensus gap adds nothing.
  - CLV +0.17 (CI 0.04–0.32) at a top-decile threshold, cover 50.7% (n=442).
- **The gap's mean drifts by season** (0.37 in 2019 → 0.07 in 2025), which makes
  it a possible season proxy (see §5).
- **A total that moves with no spread/moneyline move** predicts total error more
  strongly than one moving alongside the side markets: +0.60 (t 2.9) vs +0.21
  (t 1.2), in the 2021–25 round.

### 1.6 Continuation expectation (G4)

- **Setup:** XGBoost on a dense 15-minute panel (30–720 min before tip), target =
  bet365 move to close, trained only on seasons before the test season, 3 seeds.
- **Out-of-sample correlation with the realised bet365 move** (seed std ≤ 0.005):

| Model | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| Spread, own market | 0.070 | 0.091 | 0.077 | 0.064 | 0.159 |
| **Spread, own + cross-market** | **0.118** | **0.097** | **0.172** | **0.137** | **0.183** |
| Spread, + injury news | 0.112 | 0.085 | 0.167 | 0.151 | 0.144 |
| Spread, lookup table | 0.030 | 0.029 | 0.016 | 0.016 | 0.035 |
| **Total, own market** | **0.104** | **0.138** | **0.195** | **0.133** | **0.130** |
| Total, + cross + news | 0.102 | 0.115 | 0.152 | 0.115 | 0.096 |
| **Moneyline, own market** | **0.105** | **0.112** | **0.127** | **0.087** | **0.159** |

- **Calibration:** realised move per unit predicted is 0.55–0.80, so the models are over-confident.
- **Injury-news inputs never help the continuation models.** Keep news in G1/G2, not in G4.
- **Beyond raw moves, the prediction explains bet365 error:**
  - Spread own +2.7 (t 2.9); own+cross +1.5 (t 2.0). Total own +1.75 (t 2.4).
    Moneyline own +1.6 (t 1.4).
  - **The spread effect is mostly 2025:** +8.5 (t 5.4) in 15-min vs +0.7 (t 0.7) hourly.
- **One bet per game at the top-decile threshold (earlier seasons only):**

| Signal | n | CLV (95% CI) | Cover |
|---|---|---|---|
| Spread continuation (own+cross) | 1,523 | +0.46 pts (0.39–0.54) | 52.9% |
| Total continuation (own) | 3,396 | +0.49 pts (0.42–0.55) | 51.8% |
| Moneyline continuation (own) | 1,677 | +0.009 prob (0.008–0.011) | 53.1% |

- **CLV is consistent in every season; cover rates are not significant.**

**The branch's Ridge matches XGBoost for totals.** I ran the branch's own code
(`build_snapshot_panel` → `add_movement_features` → `add_book_deviation` →
`add_historical_ridge_movement`) on the 2019–2025 ticks at the default grid,
30–720 min. The label is the normalised bet365 total at the 0-minute snapshot
minus the line at T, clipped at ±8.

| Season | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Ridge: correlation with realised move | 0.080 | 0.118 | 0.107 | 0.153 | 0.205 | 0.122 | 0.154 |
| XGBoost total (own market), same rows | – | – | 0.099 | 0.146 | 0.164 | 0.127 | 0.157 |
| Correlation between the two predictions | – | – | 0.27 | 0.39 | 0.54 | 0.60 | 0.53 |
| Mean of the two (standardised) | – | – | 0.129 | 0.179 | 0.210 | 0.139 | 0.178 |

- **Ridge is well calibrated:** realised move per unit predicted is 0.96 (t 14.9), against 0.55–0.80 for XGBoost.
- **Ridge is fast and needs no stored models:** about 1 s, and in 2019 it only gives 0 for the first 100 games (9% of 2019 rows).
- **The two are complementary.** In a joint regression each stays significant in every season (Ridge t 2.6–8.4, XGBoost t 4.3–6.0). Averaging them adds about 0.01–0.03 correlation.
- **No edge beyond the close.** (TOTAL − close) regressed on the Ridge prediction gives +1.0 (t 1.4). TOTAL minus the line at T gives +1.45 (t 2.1), which is what a move reaching the close would produce anyway.

### 1.7 Warning: CLV is not edge

Fading a spread move that only bet365 made earns **+0.30 pts of CLV and covers
45.7%** (n=478). The line reverts, but the original move was informed.

- Any CLV-based evaluation must also report cover or error against the line at T.
- CLV here is line-only. Price (vig) changes were not included (§6 P2).

### 1.8 Anchor-total path features (from the branch)

Pooled 2019–2025, 30–720 min. Each is regressed on the future move (FM) and on
the error against the line at T (ERR), controlling for the Ridge prediction and
`MOVE_LAST_60`:

| Feature | FM | ERR | Notes |
|---|---|---|---|
| `PEERS_MOVED_ANCHOR_STILL_60` | +0.097 (t 5.0) | +0.20 (t 0.8) | non-zero on 5–12% of rows |
| `SIGNED_MOVE_STREAK_60` | +0.035 (t 3.2) | +0.16 (t 1.3) | |
| `ABS_LEVEL_PATH_60` | +0.006 (t 0.7) | +0.15 (t 1.5) | p99 3.5, max 34 (feed glitches) |
| `MINUTES_SINCE_LAST_LEVEL_MOVE` | ~0 (t −0.8) | −0.001/min (t −3.0) | p99 1,330 min, max 87,668 min (61 days) |
| `LAST_TWO_LEVEL_MOVES_GAP_MIN` | ~0 (t −1.5) | ~0 (t 0.4) | capped at 720 |

- **"Peers moved, bet365 didn't" predicts that bet365 catches up.**
  - The signed future move averages +0.22 pts; bet365 follows by at least 0.5 pts 43% of the time.
  - But (TOTAL − close) on its sign is +0.21 (t 0.9): **CLV, not edge** (§1.7).
  - Keep it as a feature; do not use it as a betting rule.
- **The minutes-since-move ERR effect is a single t≈3 among many tests.** Treat it as unconfirmed.

---

## 2. Definitions

All times are UTC. T = tip − `TIME_TO_MATCH_MIN`.

**Chance a player sits, at time t.** For each (game, player), read the span with
`valid_from < t ≤ valid_to`, and map its status:

| Status | p_out |
|---|---|
| Out | 1.0 |
| Doubtful | 0.98 |
| Questionable | 1 − 2_5 `P_PLAY` (Questionable) |
| Probable | 1 − 2_5 `P_PLAY` (Probable) |
| Available | 0.02 |
| Dropped from report (NULL status) | 0.0 |

- **Before the player's first span in this game,** use their p_out on the **last
  report of the team's previous game**, or 0 if they were not listed there.
- **Same status, new reason:** Δp = 0, so it is not news.

**Importance** = 2_5 `FORM_PTS` (`player_form_slots`: EWM with half-life 10,
strictly earlier games, previous-regular-season fallback).

**Expected missing points** per team: E(t) = Σ p_out(t) × `FORM_PTS`.

**News:**
- Per side: `NEWS_W` = E(T) − E(T − W), and `NEWS_SINCE_PREV_GAME` = E(T) − Σ baseline.
- Spread-signed = NEWS_away − NEWS_home (positive = home relatively stronger).
- Total-signed = −(NEWS_home + NEWS_away).

**Material news:** a team-report change with |ΔE| ≥ 3.

**t_news:** the latest material news of either team with `valid_from < T`,
considered only within 8 h of T.

**Anchor:** the line at t_news − 60 min. Read as-of, ticks ≤ anchor time.

**β:** one coefficient per market and window, no intercept. It is fitted on
material rows by regressing the consensus move over the same window on the news
signed for that market (spread-signed for spread and moneyline, total-signed for
totals).
- **Same walk-forward rule as the branch's Ridge:** a row at T uses only games that tipped off before T, from the current and previous seasons.
- It needs ≥ 300 material training rows; below that β = 0, which is the neutral value, so the expected move is 0.
- β stays stable across seasons (§1.2), so the short window costs little.

**Closing line (every "to close" label):** the anchor book's `norm_line` at
the 0-minute snapshot of the same tick series. This is the branch's rule; the
separate closing-odds table disagrees with the tick series on some games. For
spread and moneyline use the same snapshot's `level`.

**Moneyline-implied margin** = 13.5 × Φ⁻¹(p_home), with p_home the de-vigged home
probability (`fair_right`).

**Levels:** use `level` from `snapshots.market_level` (raw total, raw
expected-home-margin spread, de-vigged home probability) for every move, so the
new features are consistent with the existing movement family.

---

## 3. Features (~38 new columns)

Naming follows the gate:
- Odds-derived columns start with `ODDS_SNAP_`, as required by `assert_odds_columns_prefixed`.
- Injury-only columns start with `INJ_SNAP_`, carry `_BEFORE`, and use the
  2_5 side suffix `_TEAM_HOME` / `_TEAM_AWAY`.
- `INJ_SNAP_` is added to `SNAPSHOT_COLUMN_PREFIXES` so the gate recognises the
  family explicitly, not only through `_BEFORE`.

### G1 — injury news (12 columns, per side)

| Column | Definition |
|---|---|
| `INJ_SNAP_NEWS_EXP_PTS_W60_BEFORE_TEAM_{HOME,AWAY}` | E(T) − E(T−60) |
| `INJ_SNAP_NEWS_EXP_PTS_W240_BEFORE_TEAM_{HOME,AWAY}` | E(T) − E(T−240) |
| `INJ_SNAP_NEWS_EXP_PTS_SINCE_PREV_GAME_BEFORE_TEAM_{HOME,AWAY}` | E(T) − baseline |
| `INJ_SNAP_MAX_PLAYER_NEWS_EXP_PTS_W240_BEFORE_TEAM_{HOME,AWAY}` | signed Δp×FORM_PTS of the largest single-player change in W240 |
| `INJ_SNAP_MIN_SINCE_MATERIAL_NEWS_BEFORE_TEAM_{HOME,AWAY}` | capped at 1440 |
| `INJ_SNAP_HAS_MATERIAL_NEWS_BEFORE_TEAM_{HOME,AWAY}` | flag |

- **Uncovered team at T** (no filing): NaN for that side, flags 0. This mirrors
  the 2_5 coverage rule.
- **Not built:** W15/W30, transition counts, a per-type split, any P_PLAY or
  level column.

### G2 — news-anchored market reaction (18 columns)

MKT ∈ {TOT, SPR, ML}.

| Column | Definition |
|---|---|
| `ODDS_SNAP_NEWS_{MKT}_EXPECTED_MOVE_W240` | β(MKT, W240) × signed news W240 |
| `ODDS_SNAP_NEWS_{MKT}_{ANCHOR,CONSENSUS}_MOVE_SINCE_PRE_NEWS` | level(T) − level(t_news − 60) |
| `ODDS_SNAP_NEWS_{MKT}_ANCHOR_REACTION_RESIDUAL` | anchor move since pre-news − β × news since pre-news |
| `ODDS_SNAP_NEWS_{MKT}_N_BOOKS_MOVED_SINCE_PRE_NEWS` | books whose level changed since the anchor |
| `ODDS_SNAP_NEWS_{TOT,SPR}_ANCHOR_UNEXPLAINED_MOVE_W180` | anchor `MOVE_LAST_180` − expected move |
| `ODDS_SNAP_NEWS_HAS_RECENT` | material news (one report step ≥ 3 expected points) within 8 h (either team) |

- ANCHOR = the dataset's anchor book (bet365).
- **Two notions of news.** `EXPECTED_MOVE_W240` and `UNEXPLAINED_MOVE_W180` use all news in their window, however small, so they can be non-zero while `HAS_RECENT` is 0. `HAS_RECENT` and the since-pre-news and residual columns are anchored on the latest *material* change only.
- **Without recent material news:** the since-pre-news moves, residuals and book counts are 0.0, and `HAS_RECENT` = 0 says so. This is the `has_window_<w>` / `move_last_<w>` convention. NaN would put 58–77% of rows (by season) above the cleaning `nan_threshold` of 50%, so the family would never reach the model. NaN is kept where news exists but a quote at one of the two instants is missing.
- Before β has enough training rows (early 2019 only): β = 0, so β-derived columns equal the raw moves (§5 R1).

### G3 — cross-market (6 columns)

| Column | Definition |
|---|---|
| `ODDS_SNAP_XMKT_ANCHOR_ML_MARGIN_MINUS_SPREAD` | ML-implied margin − spread, anchor book |
| `ODDS_SNAP_XMKT_ANCHOR_ML_MARGIN_MINUS_SPREAD_MOVE_60` | its change over 60 min |
| `ODDS_SNAP_XMKT_ANCHOR_ML_MARGIN_MINUS_SPREAD_FROM_OPEN` | its change since the opener (removes the per-game structural gap) |
| `ODDS_SNAP_XMKT_ANCHOR_ML_MARGIN_MINUS_SPREAD_VS_LEAGUE` | gap − rolling league mean of the same gap over the previous 30 game days (strictly earlier dates), each game weighted once (its snapshots averaged first); the season-drift guard |
| `ODDS_SNAP_XMKT_TOTAL_MOVE_WITHOUT_SIDE_MOVE_60` | consensus total move over 60 min where \|spread move\| < 0.25 and \|ML move\| < 0.01; 0 when either side market is seen moving; NaN when a side market is unobserved and the other is not seen moving |
| `ODDS_SNAP_XMKT_ABS_SPREAD_MOVE_60` | \|consensus spread move over 60 min\| |

Optional, untested, from the round-2 Tier 6: snapshot-implied team totals. Add
only as a separate ablation.

### G4 — continuation expectation (2 new columns; totals already done)

**Engine:** generalise `historical_ridge_movement.add_historical_ridge_movement`
with a `market` argument. Everything else about it stays the same:
- walk-forward over events, using only games that tipped off before the snapshot;
- current and previous seasons only;
- each game weighted one;
- fixed input scales, 100-game warm-up returning 0, prediction exactly 0 at horizon 0;
- labels and predictions clipped.

| Column | Status | Design matrix |
|---|---|---|
| `ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE` | **exists** (branch) | branch design: anchor total moves, line age, deviation, move count, books quoting, log-horizon interactions |
| `ODDS_LINE_HIST_RIDGE_EXPECTED_SPREAD_MOVE_TO_CLOSE` | new | same design on the anchor **spread** `level` (expected home margin), **plus cross-market inputs:** G3 `ML_MARGIN_MINUS_SPREAD` and its `_MOVE_60`, anchor total `MOVE_LAST_60`, anchor moneyline `MOVE_LAST_60` |
| `ODDS_LINE_HIST_RIDGE_EXPECTED_ML_MOVE_TO_CLOSE` | new | same design on the anchor moneyline `level` (de-vigged home probability); scales divided by ~25, clip ±0.15 |

- **Cross-market inputs for the spread** because own+cross beat own-only in every season in §1.6 (0.10–0.18 vs 0.06–0.16).
  - Totals and moneyline gained nothing from cross inputs.
- **No injury inputs** (§1.6).
- **No `MODEL_TRAIN_SEASONS` column and no model cache:** the warm-up rule replaces both.
- **Clips:** spread label and prediction ±5; totals keep the branch's ±8.
- **XGBoost continuation is not built by default.** It adds ~0.01–0.03 correlation over Ridge (§1.6) at the cost of a GPU-optional fit and a model cache. It stays an ablation arm (§6); build it only if that arm passes the keep rule.

---

## 4. Implementation

### Phase A — data access (no feature changes)

1. `postgre_db/injury_report_aiven/fetch.py`: add `status_spans(conn, season_years)`.
   - Returns every pre-tip span with `game_id, team_id (nba id), player_id,
     valid_from, valid_to, status, reason_category, game_date, season_year, tipoff_utc`.
   - Filter on `valid_from < tipoff`.
   - Load from one season before the first line-history season, for the previous-game baseline.
2. `data_processing/injury_status/news.py`:
   - `listing_estimates(spans, events, box)`: `P_PLAY` (Questionable, Probable) and
     `FORM_PTS` for every (game, player) listed, via `status_player_estimates` and
     `player_form_slots`. There is no new estimator.
   - `team_timelines(spans, estimates, team_games)`: p_out per span, the
     previous-game baseline, Δp, and per-team change events with cumulative E.
   - `news_at(timelines, keys, windows)` → G1 columns, via `merge_asof(...,
     allow_exact_matches=False)`.

### Phase B — walk-forward estimators (no dense panel, no model cache)

Prerequisite: `feat/historical-ridge-line-movement` is merged.

3. `create_training_data/historical_ridge_movement.py`: refactor rather than copy.
   - Split the event-accumulation loop (sort events by tip-off; per-season Gram
     matrix and right-hand side; refit when the version changes) into a private
     helper, `_walk_forward_ridge(x, y, frame, alpha, min_games)`.
   - Add `market` to `add_historical_ridge_movement` and a design function per market (§3 G4).
   - Keep the public total column name and behaviour unchanged; the branch tests must still pass.
   - Spread and moneyline closing lines come from the same 0-minute snapshot
     (`_resolve_target_spread_line` exists already; add a moneyline `level` equivalent).
4. `data_processing/line_history/news_beta.py`:
   - `add_news_beta(frame, ...)` fits β with the same walk-forward helper (no
     intercept, no penalty, one weight per game, ≥ 300 material rows).
   - Log the final β per market and season.
5. **Pre-news anchor levels** (G2) are read as of `t_news − 60` directly from the
   ticks with `merge_asof(..., direction="backward")` per (game, market, book).
   This needs no dense panel.
   - The 30-day league mean for G3 `_VS_LEAGUE` comes from the snapshot rows of earlier dates, averaged per game first so a game quoted at more horizons does not weigh more.

### Phase C — wiring into `create_intermediate_line_df`

6. After `snapshot_wide` and `SNAPSHOT_TS_UTC` exist:
   - G1 from Phase A.
   - G3 from the snapshot panel levels. The anchor book's `level` for SPR and ML
     is already in the panel; the 30-day league mean comes from snapshot rows of
     strictly earlier dates.
   - G2 needs G1 timelines, β and the ticks.
   - G4 (spread, moneyline) runs next to the branch's total Ridge call on the same
     narrow frame, after G3 exists (the spread design uses G3).
   - Every block merges on `(GAME_ID, TIME_TO_MATCH_MIN)` with
     `validate="one_to_one"` and raises on column overlap, as
     `add_snapshot_injury_features` does.
7. `select_intermediate_columns.py`: add `INJ_SNAP_` to `SNAPSHOT_COLUMN_PREFIXES`.
   Confirm `audit_closing_line_reconstruction` still passes on both markets.
8. `scripts/create_train_data/create_intermediate_line_train_data.py`: flag
   `--no-market-dynamics` (G1–G3 and the spread/moneyline G4). **On by default.**
   The branch's `--no-ridge-movement` keeps controlling the total Ridge.
9. `config/dataset_versions.py`: add a 2_5 history note describing G1–G4. The same
   note must record the branch's changes, which it does not add itself:
   - the Ridge total column and the five path columns;
   - **the change in meaning of the existing `DEVIATION_FROM_CONSENSUS` / `DEVIATION_Z` / `IS_OUTLIER_BOOK` columns** (leave-one-book-out, capped).

   Intermediate 2_5 builds from before and after the merge are therefore not like-for-like. Schema stays 2_5.

### Phase D — tests

| Test | Asserts |
|---|---|
| report exactly at T | excluded from E(T) and from t_news |
| baseline carry | a player Out last game and Out now produces no news; a new Out does; an Out→Out reason change produces 0 |
| uncovered team | G1 side NaN, flags 0 |
| future shuffle | permuting every span with `valid_from ≥ T` and every tick after T leaves all G1–G4 values unchanged |
| β guard | β at T is unchanged when any game tipping off at or after T changes, and when a season older than S−1 changes (same pattern as `tests/test_historical_ridge_movement.py`) |
| Ridge spread/moneyline guard | the branch's three causal tests, parametrised over `market` |
| Ridge refactor | the total column is bit-identical before and after the helper extraction |
| anchor | the pre-news level reads ticks ≤ t_news − 60, never later |
| orientation | home favourite → positive spread level; moneyline-implied margin increases with p_home; spread-signed news positive when the away team loses a scorer |
| gate | every new column survives `select_intermediate_training_columns`; `assert_odds_columns_prefixed` passes; no new column reconstructs the closing line |
| season proxy | `find_season_gated_columns` flags none of the new columns |

---

## 5. Risks

- **R1 First season.** The branch's warm-up rule resolves this.
  - The first 100 games of 2019 get 0 (β-based columns: until 300 material rows).
  - The zeros are neutral values, not NaN, so the season is not identifiable from missingness.
  - They are a slightly worse estimate early in 2019 only (Ridge correlation 0.08 in 2019 vs 0.11–0.21 later).
- **R2 Season proxies.** The ML–spread gap mean drifts 0.37 → 0.07. The
  `_FROM_OPEN` and `_VS_LEAGUE` variants exist for this; if the raw level is
  flagged, drop it.
- **R3 CLV ≠ edge** (§1.7). CLV is line-only; price changes are not included.
- **R4 Era dependence.** The totals over-reaction is absent in 2025, and the
  spread-error effects of G3/G4 sit mostly in 2023–25. 2025 is the only 15-min season.
- **R5 Multiple testing and a worn holdout.** The 2025–26 window has been viewed
  many times (see the schema 2_3 review). Pre-register the decision rule below.
- **R6 Calibration.** Ridge totals is calibrated (0.96). The XGBoost arm is not
  (0.55–0.80); never read its value as points.
- **R7 Stale books.** "Did not move" can be staleness. G2 book counts, and the
  branch's `PEERS_MOVED_ANCHOR_STILL_60`, should ignore books whose quote is older
  than the window (line age is available).
- **R8 Path-feature tails (branch).** Neither is a leak, but the tails are glitches rather than information.
  - `MINUTES_SINCE_LAST_LEVEL_MOVE` reaches 61 days (openers posted far ahead), so cap it at 1,440 like the gap feature.
  - `ABS_LEVEL_PATH_60` reaches 34 pts from single-tick feed glitches. Ignore level jumps larger than `PEER_GAP_LIMITS` (10 pts) when summing.
- **R9 Cross-fold contamination of walk-forward features.** In a *random* game
  split, a training row's Ridge or β value can include a test game's closing line.
  - The effect is negligible: it is one game among hundreds, and the close is not the game model's target.
  - It is zero under walk-forward evaluation, which the campaign uses anyway.

---

## 6. Experiments

### Before implementation (cheap, scratch only)

- **P1 Ridge for spread and moneyline.** Before wiring, run the generalised
  engine on the snapshot panel.
  - Target: out-of-sample correlation within 0.02 of the §1.6 XGBoost numbers
    (spread own+cross, moneyline own) in each 2021–2025 season.
  - If the spread falls short, add the §1.6 inputs Ridge lacks (books up/down over 60 min, share of the move in the last 15 min) before considering XGBoost.
  - *(The original P1, first-season policy, is resolved by the warm-up rule; see R1.)*
- **P2 Price-inclusive CLV.** Redo §1.6/§1.7 CLV in de-vigged probability for
  the anchor book (`fair_left` / `fair_right`), so line moves absorbed by price
  changes are counted.
- **P3 G3 league-mean window.** 14 vs 30 vs 60 days for `_VS_LEAGUE`: stability
  of the spread future-move slope by season.

### After implementation (training pipeline, group ablations)

The campaign lives under `experiments/intermediate_market_dynamics_2026_09/`.

- **Arms:**
  - `baseline−branch` (2_5 without the branch's Ridge total and path columns; `--no-ridge-movement` plus dropping the five path columns). This measures the merged branch, which has not been ablated yet.
  - `baseline` (2_5 including the branch)
  - `+G1`
  - `+G1+G2`
  - `+G3`
  - `+G4` (spread and moneyline Ridge)
  - `+all`
  - `+all −G4`
  - `+all +XGB-continuation` (optional, only if P1 shows Ridge short)
- **Targets:** line error (total) and spread error at the intermediate horizons,
  each with `evaluation_seeds` ≥ 5.
- **Drop features as whole groups,** never one at a time (correlated families; see
  the rotation-leak lesson).
- **Report per horizon and per era:**
  - error metrics against the anchor line at T
  - CLV of the model's bets (line and price)
  - cover rate with Wilson CI
  - feature-group importance
- **Pre-registered decision rule:** a group is kept only if all three hold:
  1. it improves the per-horizon error metric in ≥ 4 of 5 walk-forward test
     seasons, pooled across seeds;
  2. model bets at the pre-registered edge threshold have positive price-inclusive CLV;
  3. its importance is not concentrated in `TIME_TO_MATCH_MIN`-like or season-proxy columns.
- **Expected outcome:** tenths of a percentage point of win rate at best.
  Detecting that from cover rates alone needs far more bets than exist. Decisions
  lean on error metrics and CLV, as in §1.6.

---

## 7. As implemented (2026-09-16)

**Where the code lives:**

| Piece | File |
|---|---|
| Span, filing and previous-game reads | `postgre_db/injury_report_aiven/fetch.py`: `status_spans`, `filing_spans`, `team_game_schedule` |
| G1 timeline and features | `data_processing/injury_status/news.py` |
| G2, G3 and the as-of tick reader | `data_processing/line_history/market_dynamics.py` |
| Walk-forward fit shared by the Ridge and β | `data_processing/line_history/walk_forward.py` |
| G4 (Ridge per market) | `create_training_data/historical_ridge_movement.py` |
| Loading and attaching G1–G3 | `create_training_data/intermediate_market_dynamics.py` |
| Wiring, flag | `create_intermediate_line_df(include_market_dynamics=True)`, `--no-market-dynamics` |
| Tests | `tests/test_intermediate_market_dynamics.py` |

**Differences from §2–§3, each deliberate:**

- **Baseline carry.** A player's previous-game status carries only until their first listing in this game. A player the game never lists is left out.
  - Also counting such players as returning at the team's first filing was tried. It tracked the market worse: spread move from open, correlation 0.30 vs 0.34 at 120 min (0.17 vs 0.20 for totals), on 2023–24. So it was dropped.
- **G League listings** have p_out 0, like the 2_5 counters.
- **One β per market,** fitted at W60 (consensus move over 60 min against news over 60 min) and applied to every window (W180, W240, since pre-news). The per-window βs were not estimated. β is a price per expected point, and on 2023–24 it matches §1.2: spread 0.031–0.032, total 0.020–0.025, moneyline 0.0008.
- **Book names in column names:** `ANCHOR` becomes the book (`ODDS_SNAP_NEWS_SPR_BET365_MOVE_SINCE_PRE_NEWS`, `ODDS_SNAP_XMKT_BET365_ML_MARGIN_MINUS_SPREAD`), matching `ODDS_SNAP_TOT_BET365_*`.
- **`_VS_LEAGUE`** uses the previous **30 game days,** not calendar days, so season openers read the end of the previous season instead of an empty off-season window.
- **`HAS_MATERIAL_NEWS`** means material news within 240 min. `MIN_SINCE_MATERIAL_NEWS` is 1,440 when there is none.
- **"Recent news" for G2** means material news within 8 h, as in §2. The research prototype had no limit, so its "has news" flag covered more rows.
- **Moneyline-implied margin** uses the repo's `MARGIN_SIGMA` (13.46), not 13.5.
- **R7:** `N_BOOKS_MOVED_SINCE_PRE_NEWS` counts movers only, so a stale book cannot lower it.
- **G4 current line:** the spread Ridge predicts the move of the anchor's `norm_line`; the moneyline Ridge predicts the move of its `level`.

**Checks run:**

- The refactored total Ridge is bit-identical to the branch version on 88,474 real rows.
- G1 values match the research prototype on 2023–24: W60 correlation 0.99, 97% identical.
- The anchor move since pre-news matches at 0.98.
- The pre-implementation experiment P1 (§6), Ridge for spread and moneyline, was run on 2019–2025 with the implemented code. The table gives out-of-sample correlation with the realised bet365 move to close, at 30–720 min:

| Estimator | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Calibration |
|---|---|---|---|---|---|---|---|---|
| Total Ridge (branch) | 0.080 | 0.118 | 0.107 | 0.153 | 0.205 | 0.123 | 0.155 | 0.93 |
| Moneyline Ridge | 0.082 | 0.096 | 0.114 | 0.134 | 0.149 | 0.114 | 0.132 | 0.96 |
| Moneyline XGBoost (§1.6, same rows) | – | – | 0.084 | 0.103 | 0.127 | 0.105 | 0.164 | |
| Spread Ridge, linear gap only | 0.091 | 0.056 | 0.061 | 0.096 | 0.090 | 0.072 | 0.172 | 0.87 |
| **Spread Ridge, with gap terms (implemented)** | 0.109 | 0.066 | 0.075 | 0.107 | 0.112 | 0.102 | 0.187 | |
| Spread XGBoost own+cross (§1.6, same rows) | – | – | 0.097 | 0.068 | 0.116 | 0.105 | 0.143 | |

  - **The linear spread design fell short of XGBoost** by 0.026–0.036 in 2021, 2023 and 2024.
    - Adding more linear inputs changed nothing (±0.005): consensus steam, 15-minute move, gap from open or against the league, cross-market moves from open, book deviation.
    - What closed the gap was non-linear terms of the gap: |gap|, gap clipped to ±1, gap × |spread| / 7, and |spread| / 7.
  - **Result:** the spread Ridge averages 0.117 over 2021–25 against 0.106 for XGBoost. It is within 0.02 in every season except 2021 (0.022 short).
  - **The XGBoost arm stays optional.**
- **Season behaviour of the new columns on 2019–2025:**
  - No column's mean moves by more than 0.6 pooled SD across seasons.
  - Before the fix below, the G2 "since pre-news" columns were NaN on 77% of 2019 rows and 58% of 2025 rows. That follows how often recent material news exists (`HAS_RECENT` 0.24 → 0.42), which grows with report cadence.
  - **Fixed after review (2026-09-17):** at that NaN rate the default `nan_threshold: 50` would have dropped the family. Rows without recent material news now hold 0.0 (see G2 above). The ablation should still check the built CSV with `find_season_gated_columns`.
  - **Also fixed then:** `TOTAL_MOVE_WITHOUT_SIDE_MOVE_60` treated an unobserved side-market move as quiet, and `_VS_LEAGUE` weighted snapshots rather than games. The values quoted above predate both fixes.
  - β by season (spread, per expected point): 0.024, 0.027, 0.030, 0.032, 0.033, 0.031, 0.032.
- **Real-data bug found and fixed:** numpy season integers could not be sent as a Postgres array alongside Python ints (`fetch._season_params` now casts).
- **Cost:** the injury timeline takes about 40 s, G1–G3 about 20 s, and each Ridge about 1 s, all on 2019–2025.
- **The full dataset build was not run.** A build with the new flags is the next step before the §6 ablation campaign.
