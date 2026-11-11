# Historical analysis

Status: **Partial** — main historical views implemented 2026-10-28; ≥/< 6h
windows, best slot, mean-of-predictions combinations and the random baseline
2026-11-02; three best slots and same-timepoint mean/majority combinations
2026-11-05; combinations shown as a heatmap table, also by Madrid hour,
2026-11-11. Consensus/agreement views across all slots and a full-season
performance measurement remain.

The **Betting hour · Madrid** view compares XGBoost and TabPFN, intermediate and
closing, around each clock hour (selection reworked 2026-11-10). Its selection
and grading rules are documented below; the model/snapshot views retain their
existing definitions.

## Context

The dashboard's third page (`streamlit/app_pages/historical.py`) was initially
only a title. This design adds season-level analysis of stored predictions. The main
question is: **what is each model's win % at each time before tip-off, with live
predictions kept separate from the generated (replayed) snapshots?** For closing
models, that becomes win % by the time the prediction was made. A secondary
question is how to combine the predictions made at different times. The page
has three views, one per target.

Decision already taken with the user: replay is **diagonal only**. Workflow 2
predicts each intermediate model on its own snapshot only, so "generated
snapshots" fill just the model-horizon = time diagonal. No predictor changes.

---

## 1. Data facts the design rests on

- Source: `prediction.predictions_full` (+ `runs.finished_at`, `model_builds.anchor_book`),
  as in `repositories/predictions.py`.
- **Live** = workflow 1 (daily predictor) and workflow 3 (TabPFN), with
  input, prediction and publication before both the stored and current tip.
  Closing slots read `input_mode 0`; intermediate
  slots normally read `input_mode 2` (`latest`: every model reads the run's
  current snapshot, so **every intermediate model predicts at every run time**).
  Already-passed own inputs (`input_mode 1`) can also enter the live sample.
- **Replay intermediate** = workflows 2/4, `input_mode 1`: each model on its own snapshot only,
  rebuilt after the games (late ticks or reports can differ from live). TabPFN
  own-snapshot rows (workflow 3, `input_mode 1`) count in both only when
  genuinely available before tip-off, as Yesterday does.
- **Replay closing** = workflow 2 with `input_mode 3`, or Historical Backfill
  (workflow 4) with `input_mode 0`, using the explicit stored snapshot horizon.
  These have a separate closing replay KPI, timeline and data table. An exact
  T−0 input maps to replay Final; a T−4h input stays at T−4h. Refits and reruns
  reduce to one output per game/slot/snapshot, independently of live outputs.
- Runs fire at every game's `tip − h` (clustered ≤30 min, `snapshot_triggers.md`),
  and every run predicts every game still to start. A game therefore gets many
  runs (roughly 40 × 51 slots ≈ 2k rows, ~2.5M per season). The page must
  reduce rows in SQL. It cannot pull them raw.

## 2. Core definitions (one place, used by every view)

| Concept | Definition |
| --- | --- |
| Slot | `(kind, schema_version, target, dataset_type, model_horizon_minutes, variant)`. A refit replaces the build in the same slot; never group by `fit_id`. |
| Input time to tip `r` | Intermediate and reconstructed closing: `snapshot_horizon_minutes`. Live closing: `tipoff_at_prediction − run_started_at`. This measures what the model knew. The publication lag (`available_at − input_at`, ~20 min on a warm runner) is shown, not bucketed. |
| Time bucket | Snapshot grid `G = 30, 60, 120 … 720, 840, 960, 1080` plus `T−18h+`. `bucket(r) = min{h ∈ G : h ≥ r}`, the last snapshot that had passed, covering `(previous h, h]`. Closing also gets a **Final** column: the last output available before tip, the same rule as Yesterday. |
| One row per cell | Per `(game, slot, source, bucket)` keep the prediction with `r` closest to `h` (right after the snapshot passed). This makes the live diagonal comparable to replay. |
| Grading | Same as `grade_history` (`domain/history.py`): the signal requires \|edge\| > per-kind min edge, measured against **that row's stored Bet365 quote**. Results are WIN/LOSS/PUSH/NO BET/NO LINE/VOID. |
| Win % | `W / (W + L)` from pooled counts, with a per-cell Wilson 95% CI and a break-even reference at 52.38% (−110). Coverage = bets / expected settled-game opportunities, including missing outputs. Pooled margins use one opportunity per observed slot/time cell. Secondary metrics: MAE (all available predictions on final games) and unit ROI from the stored prices (null when the price is missing, never an assumed −110). |

## 3. Page layout

A shared filter bar: date range (default most recent 28 October through today's NBA slate), season type
(regular), model family (XGBoost 2.5 / TabPFN), per-kind min edge
(`minimum_edge_select`), and a metric switch (Win % / Bets / MAE / ROI).
Below it, a **target selector**: Line error | Total points | Spread error.
These are the three views, each with the same sections; TabPFN appears only
where it has outputs (no spread).

The best single slots and their combinations (1b–1c) open the page, above
the KPI strip.

1. **KPI strip**: settled games, pooled win % for closing Final and closing
   replay, and the random baseline (totals only). Below, one row per source
   (live, replay): own snapshot pooled over every horizon, then split into
   **≥ 6h** (models trained at T−6h or earlier) and **< 6h**.
1b. **Best single slots** (source switch, replay by default): the three
   highest win % among the slots with at least 30 settled bets. A slot is an
   intermediate model at its own snapshot, or a closing model at one input
   timepoint (Final included). Each card shows the slot, win %, bets and ROI.
   Each card also gives **P(profitable)**: the posterior probability that the
   slot's true win rate beats 52.38%, from its wins and losses and a skeptical
   Beta(50, 50) prior (a coin flip worth 100 bets; 32% with no bets), exact
   via the binomial identity, no scipy. It does not know the slot was picked
   as a leader. Beside the cards, two selection-aware checks of the best one:
   the **game-permutation p** (each simulation flips every game's result at
   random, the same flip for every slot that bet it, so shared games stay
   shared; the share of 2,000 simulations whose best slot reaches the observed
   win %, +1 corrected, shown as `< 0.001` below its resolution) and the
   **coin-flip chance**: the share of
   simulations in which the best of the same number of coin-flip cells, with
   the same bet counts, reaches that win %. Cells share games, so treating
   them as independent overstates the chance; it is a conservative guide, not
   a p-value. The chart is the cumulative win % by slate date of the three
   slots (Wilson band for the best), pooled own snapshots, the random
   baseline, the leading combination, and the grey band where a coin flip
   with the same bets lands 95% of the time. The slots are chosen on the
   games they are scored on, so the curves flatter them.
1c. **Combinations · same timepoint**: §4.
2. **Closing by prediction time** (main closing answer): a line chart with
   x = time bucket (`T−18h+` → `T−30m` → Final), y = win %, a CI band, a
   break-even rule and bet-count labels. The table goes in an expander. Pooled
   KPI cards do not show a CI, because their repeated game observations are
   correlated.
   **Closing by input time · Replay** adds a separate chart and detail table
   for reconstructed closing outputs, including Historical Backfill. The
   overview's Closing · Replay metric pools these outputs; it never enters
   Closing · Final or any live metric.
3. **Model × time heatmap: live** (the main answer). XGBoost only: TabPFN
   predicts only at its scheduled runs and its sparse cells added nothing.
   Rows are Closing plus the 16 intermediate horizons (longest at the top). Columns are the time
   buckets, far to near. The **diagonal is outlined**: cells to its left are
   a model predicting before its trained snapshot (`SNAPSHOT_PASSED` false),
   cells to its right are a model fed a fresher snapshot than it was trained
   on. Color diverges around break-even and is capped. Cells with
   n < 30 are greyed and show only their numbers. The tooltip shows W-L-P,
   n games, CI, MAE, coverage and the median publication lag. Row and column
   margins are pooled counts, labelled as rows, not independent games.
4. **Intermediate by model horizon · Replay**: a closing-style timeline with
   one point per model family and trained horizon, pooling that model's
   own-snapshot replay predictions across the selected games. It uses the same
   metric selector, Wilson band, bet-count labels and break-even reference as
   the closing chart. Replay only; no live fallback or extra database read.
5. **Own snapshot: live vs replay**: a dot + CI per model horizon, with two
   series (live diagonal vs replay) and a delta table. It shows whether the
   replay reproduces live.
6. **Combined across time** (secondary, collapsed, loaded on demand). See §4.
7. **Method notes** expander: the definitions above, the exclusions and the
   noise warning.

## Betting hour · Madrid

Select **Betting hour · Madrid** in the Historical analysis view selector. It
answers one question: **what results did the original predictions closest to
12:00, 13:00 … 00:00 Madrid get?** The hour only chooses a prediction. It shares
the inclusive NBA slate range, season type, target, model-family filter, minimum
edges and chart metric. The hours are **12:00–23:00 on each NBA slate date, then
00:00 at the following midnight**, inclusive. Hours use `Europe/Madrid`,
including daylight-saving changes, not a fixed UTC offset. The **Hours to
compare** range narrows the page and common-game cohort locally, without another
database read.

The page opens with the **four best hour and series pairs** by win %, among
those with the minimum of settled bets (each card gives record, ROI and
coverage), beside the coin-flip chance and the game-permutation p of the leader
over all pairs tried. Below it is a **table with one row per hour and one column
per series**, each cell giving win %, wins-losses and coverage, for example
`58.5% · 55-39 · 94/120 (78.3%)`: green above break-even at −110, red below, and
grey text for a small sample. Below it, **Combinations by hour** combines the
series chosen at each hour exactly as §4 combines the slots of one timepoint
(every subset of 2–4 series, on the games where all of them have a chosen
prediction with a line, as a mean edge and as a majority vote), with the
leader's win %, P(profitable), coin-flip chance and game-permutation p over all
combinations, and the same heatmap table as §4 (one row per hour and set of
series). The chart and the long-form hourly results follow.

There are four series: **XGBoost · Intermediate**, **XGBoost · Closing**,
**TabPFN · Intermediate** and **TabPFN · Closing**. The model-family filter
keeps the two of one family. Two sources are selectable, Snapshot replay by
default, and never fill each other's gaps. Each prediction has a **reference
time**:

- **Live predictions**: when the prediction was generated (`predicted_at`).
- **Snapshot replay**: the time of the reconstructed snapshot (`input_at`,
  tip-off minus the snapshot horizon), never when the backfill ran. Only builds
  with `train_date_max < game_date` enter this view. This is retrospective:
  saved features can include later corrections, and the existing daily-training
  cutoff is not a guarantee that every previous-day game had finished at the
  snapshot time.

**Selection.** For each game, hour and series, the candidates are the
predictions whose reference time is within **one hour before or after** the hour
(both ends included), not after their tip-off, with an output and a stored bet
line. Exactly one is chosen, the closest to the hour. Ties take the earlier side
of the hour; within one run (live intermediate slots predict together), the slot
trained nearest the time to tip; then the latest stored run. A series with
several slots is therefore a choice among them, never an average. Nothing
outside the window fills a gap. Hidden variants are excluded. A prediction can
serve two neighbouring hours, so neighbouring rows share bets.

**Grading.** The chosen prediction keeps its own output, line, prices, side and
result, through the same `grade_history` as the checkpoint views: the stored
`bet_line_x2` of that input (the quoted Bet365 line of that snapshot) and the
stored odds. Output of a line-error or spread model is added to the
`model_line` it was made on, and the line-error prediction is never moved to
another line. The per-kind minimum edge can turn it into a no-bet; another
prediction does not replace it. ROI stakes one unit per priced bet, pushes
return zero, and missing prices are excluded from ROI. Win % is W/(W+L), with
per-cell Wilson 95% intervals. MAE includes available predictions on final
games even without a bet.

**Coverage.** A game is eligible for an hour when it finished and that hour's
window opened before its tip-off and before the read. Coverage is eligible games
with a valid prediction in the window / eligible games, no-bet signals included
in the numerator. Games without a prediction are missing, never losses. The
**Compare common games only** filter requires a prediction for every displayed
hour and series, without requiring a directional bet. Closing predictions exist
only near tip-off, so a wide hour range can leave an empty cohort.

Reads (`repositories/clock.py`) return the eligible outputs of both sources, each
with its own line and prices, and the selection runs locally (`domain/clock.py`).
The first version crossed every output with the thirteen hours in PostgreSQL; it
ran for more than 170 s on a two-week range with only about 100,000 stored
predictions, so the query shape, not the data volume, was the limit. No index or
precomputed table is needed. Reads are read-only, bounded to 200,000 rows with a
45-second timeout, and cached for 15 minutes per range, target and season type.
Exceeding the bound fails without displaying partial results. Source, metric,
edge and family changes reuse the read; Refresh invalidates the selected entry.
No new predictions, remote inference, backfills or database writes are run.

The best pairs are exploratory: cells share games, available horizons change,
and selecting the maximum across hours introduces selection bias. Counts,
coverage and uncertainty accompany them; small samples are labelled. No
automatic claim of a profitable/best future hour is made.

## 4. Combining predictions across time (secondary)

Every combined pick is graded at **the line available when it would be
placed** (time T), never at a later line.

- **Same-timepoint combinations** (implemented 2026-11-05,
  `domain/strategies.py`): only slots available at the same timepoint are
  combined, e.g. at T−8h the XGBoost and TabPFN T−8h intermediate models and
  the closing models on the T−8h input; with the Settings toggle, also the
  other totals target's slots (cached read). Every subset of 2–4 members (at
  most 8 members per timepoint, those with the most settled bets) is scored on
  the games where all of them have a line, two ways:
  - **Mean edge**: members are averaged as **edges** (prediction minus the
    line that member was graded at), so models, phases and totals targets
    share a scale; bet with the minimum edge of the member published last.
  - **Majority vote**: each member's own signal at its own minimum edge;
    bet when more than half of the members take one side (with two, both
    must agree), otherwise no bet. A vote has no point prediction, so no MAE.
  Both bet at the line and price of the member published last, when the
  whole combination is known. The leader is picked from all combinations
  tried, so cards report its win %, P(profitable), and its coin-flip chance
  and game-permutation p over all of them. Every combination is then listed
  in a **heatmap table** (2026-11-11, `ui/combination_table.py`), one row per
  timepoint and set of slots: final games where all of them had a line, and
  per method (mean, majority) the win % (green above 52.38%, red below, grey
  text under 30 wins + losses) and the settled bets placed (shaded blue by
  count; for the vote, the games where a majority agreed). Rows with an
  eligible method come first, by their best eligible win %. An expander pairs
  the ten best, the three best single slots and the random baseline with the
  best slot on the games both bet (W/L only), with an exact two-sided sign
  test on the games only one of them won and P(profitable) per row. The
  forest chart of those rows was replaced by the table. The previous fixed family
  (per-model means across horizons, 2026-11-02) is replaced: it mixed
  snapshots that are never available together at a betting time.
  First read of that family (2026-11-02, 92 games, replay): no combination
  clearly beat the best slot; line error + total points did not help (pooled
  47.7% vs 48.7% for line error alone, edge correlation 0.29), and the stored
  random baseline stood at 29-20.
- **Hidden variants**: TabPFN thinking-high outputs
  (`*_thinking_high`, `HIDDEN_VARIANT_SUFFIXES` in `dashboard/config.py`) are
  deprecated and dropped from every page.
- **Random baseline** (implemented): stored `kind=2` random picks
  (`random_line_error_baseline`, from 2026-10-26), one per game, preferring the
  past-snapshot workflow's T−0 row; graded like any closing Final with no
  minimum edge. Line error and total points show the same OVER/UNDER bets;
  spread has none.

- **Consensus at T** (implement first): for each game and bucket T, a simple
  majority of every slot's signal in column T, with the same equal-vote and
  tie = no bet rule as `domain/picks.game_pick`. Output: a win % vs T curve
  with coverage. It answers "if I bet the models' majority at T−x, how
  often do I win?".
- **Agreement filter**: at a chosen T, win % against coverage as the minimum
  agreement share rises from 50% to 100%. Fewer bets should mean a higher
  hit rate if agreement carries signal.
- **Own-horizon path vote** (optional): at T, the vote of every model at its
  own snapshot that has already passed (diagonal up to T), with a
  live/replay switch.
- **Pick stability** (diagnostic): the number of side flips in a game's
  closing-model timeline, and win % for stable vs flipped games.

## 5. Statistical guardrails (shown on the page, not just in the doc)

- One regular season is ~1,230 games, an SE of ~1.4 pp at 55% for independent
  game outcomes. With many cells and only an early-season sample, some cells
  will look good by chance. Earlier homogeneity results do not establish
  horizon interchangeability when predictions share games.
- Do not report an independent-sample χ² homogeneity test: rows and columns
  share games. Formal comparisons require a paired or game-clustered method.
  The heatmaps never highlight a "best" cell. The best-slot and combination
  sections (§3, 1b–1c) and the best hour and series pairs are the only places a
  best is picked, at the user's
  request; they always show the coin-flip chance and state that the pick
  flatters itself.
- Cells in one column share games, so they are correlated. The CI holds per
  cell only.

## 6. Implementation outline

| Piece | Where |
| --- | --- |
| `read_history_range(start, end, target, season_type)`: one SQL query computing `r`, source and bucket, a window-rank reduction per `(game, slot, source, bucket)` (plus the Final row), narrow columns only, `SET LOCAL statement_timeout` raised for this read; a separate small query returns game counts | `src/nba_ou/dashboard/repositories/analysis.py` (bucket `CASE` generated from the grid constant) |
| Grid constant, bucket labels, Wilson CI, cell aggregation, consensus/agreement | new `src/nba_ou/dashboard/domain/analysis.py` |
| Best slots, coin-flip chance, cumulative win %, same-timepoint mean/majority combinations, paired sign test | `src/nba_ou/dashboard/domain/strategies.py` |
| Random baseline read (`kind=2`, totals only) | `BASELINE_QUERY` in `src/nba_ou/dashboard/repositories/analysis.py` |
| Vectorized grading shared with Yesterday (replaces `iterrows` in `grade_history`, same outputs) | `src/nba_ou/dashboard/domain/history.py` |
| Cached loader (TTL 15 min, keyed by range + target + season type) | `src/nba_ou/dashboard/ui/loaders.py` |
| Page rendering, Altair heatmap / CI charts | new `src/nba_ou/dashboard/ui/historical.py`, with chart specifications in `ui/analysis_charts.py`, sharing `config.SIGNAL_HEX_COLORS` |
| Page entry | `streamlit/app_pages/historical.py` → `render_historical()` |
| Tests (synthetic fixtures) | `tests/dashboard/test_historical.py`, `test_historical_repository.py`, `test_historical_app.py` |

Phases: (1) data + domain + tests; (2) sections 1–4 for all three targets;
(3) combined view; (4) measure with `EXPLAIN (ANALYZE, BUFFERS)` on season to
date. If the read exceeds ~5 s or ~200k rows per target, add a cell table
refreshed by the finished-games job. That is out of scope until measured.

## 7. Verification (for the implementation phase)

- `poetry run pytest tests/dashboard -q`; `ruff check` on dashboard paths.
- Parity checks on a single date range = yesterday: closing Final and the
  replay diagonal at 480/360/240 must match Yesterday's summary win % and counts.
- Bucketing tests: clustered runs, off-grid `latest` rows, `T−18h+`, refits
  in one slot, and pushes, no-line and void excluded from win %.
- Manual: `poetry run streamlit run streamlit/streamlit_app.py`, then all three targets.

---

## Review decisions — 2026-10-28

- This document is now the design reference; the current user request authorizes
  implementation. The former document-only deliverable is superseded.
- Keep the existing serving folder: documentation is organized by topic, not
  implementation status. No new plans folder is necessary.
- A live row must have valid input, prediction and run-completion timestamps and
  be fully available before both the stored and current tip-off. Future own
  snapshots must not enter the live sample. Replay never silently fills live gaps.
- Time buckets describe **input** time; publication lag remains explicit. They
  are not a claim that the output was tradable at that bucket's boundary.
- The live/replay comparison includes a matched-game table: unpaired headline
  percentages may have different coverage and are not an estimate of replay bias.
  Live diagonal inputs fall within the trained-time bucket and can be slightly
  fresher than the exact replay snapshot; this is not an exact-input equality test.
- Do not report ordinary chi-square homogeneity p-values for these repeated,
  correlated games. That independence assumption is violated. Display sample
  sizes and per-cell Wilson intervals; formal comparisons need a paired or
  game-clustered method in a later statistical pass.
- Consensus must use outputs **available by the betting cutoff**, re-evaluated
  at one common, already-known Bet365 quote. Input-bucket cells alone cannot
  provide that as-of sample; implement a separate on-demand reduction.
- Prices are stored as American odds. ROI uses only valid recorded prices,
  counts pushes at zero return, and reports its priced-bet denominator.
- Season type is stored numerically (regular = 2), and season dates use the
  existing October-to-September NBA convention. Filters must bind SQL values.
- The first delivery covers the reduced data read, shared vectorized grading,
  cell statistics and the main historical charts. Combined strategies and
  production query-performance measurements are tracked separately below.

## Implementation progress and validation

Implemented:

- The Historical analysis navigation page now renders the actual dashboard.
- Inclusive date range (most recent 28 October through today by default), season type (regular
  by default), model family, three targets, per-kind minimum edges and the
  Win % / Bets / MAE / ROI switch. Family, metric and edge changes reuse source
  data; range, target and season type are cache keys. Refresh evicts that key.
  Historical backfills are included in the default range, and reconstructed
  outputs remain separate from live. The end follows the current Pacific NBA slate date.
- Final-game and pooled win-rate KPIs, closing input-time timeline with Final,
  confidence bands and bet counts, live slot/time heatmap with outlined
  diagonal and low-sample shading, and own-horizon live/replay intervals.
- Closing-style intermediate replay timeline: one aggregate point per model
  family and trained horizon, using cached own-snapshot replay predictions.
- Closing replay KPI, timeline and counts for workflows 2/4, including exact
  T−0 snapshots. Past games awaiting scores are disclosed and cannot contribute
  wins/losses or MAE until their final results are stored.
- Matched-game win-rate deltas and pick disagreements, cell detail tables,
  pooled row/column margins without misleading confidence intervals, and
  visible method/sample-size notes.
- SQL reduces refits and clustered runs per slot/source/bucket before returning
  data. Replay chooses the latest own-horizon reconstruction. Final chooses
  the last available live closing row. Threshold grading is shared and
  vectorized with Yesterday. There are no predictor or database-schema changes.
- Read-only repeatable-read transaction, bound filters, two SELECTs (small
  game list and reduced outputs), 45-second statement timeout, 15-minute TTL
  and 8 cache entries. A hard 200,000-cell limit raises an explicit error;
  the page never displays silently truncated statistics.

Measured against Aiven on 2026-10-28, NBA slates 2026-10-01 through
2026-10-27, regular season, Line error:

| Measure | Observed |
| --- | --- |
| Games returned | 13 |
| Reduced prediction rows | 2,131 |
| Live / replay rows | 1,904 / 227 |
| End-to-end two-query read | 1.446 s |
| PostgreSQL planning / execution (`EXPLAIN ANALYZE, BUFFERS`) | 21.781 ms / 124.429 ms |
| Shared buffer hits / reads | 108 / 0 |

This is a small current-season sample, not a full-season load test. Measure
a mature season before deciding whether to add precomputed cell storage.
The SQL is also tested against synthetic JSON/CTE rows using SELECT only,
covering exact/off-grid inputs, refits, early buckets, post-tip availability,
future own snapshots, source isolation and the Final selection. The optional
pytest PostgreSQL test runs with the local `NBA_OU_TEST_PG_DSN` fixture.

For the single-date range 2026-10-27, all three targets matched Yesterday's
closing Final and replay 8h/6h/4h selections and result counts: 28 rows for
Line error, 28 for Total points and 16 for Spread error. The two-query reads
took 1.786 s, 2.020 s and 1.752 s respectively. The dashboard test suite passed
184 tests, with the optional local-PostgreSQL test skipped; its synthetic SQL
assertions were also executed successfully against Aiven with a read-only
CTE/SELECT query. Ruff checks and formatting passed.

Remaining:

- Combined-as-of consensus and agreement curves need their own on-demand
  availability-based reduction and common quote at the actual betting cutoff.
  The first page does not present input-bucket votes as tradable consensus.
- Optional own-horizon path voting and pick-stability diagnostics.
- Paired or game-clustered formal hypothesis tests if needed.
- Full-season query/load measurement; a precomputed cell table only if the
  observed time or row budget justifies it.
