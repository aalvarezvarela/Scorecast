# Historical analysis

Status: **Partial** — main historical views implemented 2026-10-28; combined
strategies and a full-season performance measurement remain.

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
- **Replay** = workflow 2, `input_mode 1`: each model on its own snapshot only,
  rebuilt after the games (late ticks or reports can differ from live). TabPFN
  own-snapshot rows (workflow 3, `input_mode 1`) count in both only when
  genuinely available before tip-off, as Yesterday does.
- Runs fire at every game's `tip − h` (clustered ≤30 min, `snapshot_triggers.md`),
  and every run predicts every game still to start. A game therefore gets many
  runs (roughly 40 × 51 slots ≈ 2k rows, ~2.5M per season). The page must
  reduce rows in SQL. It cannot pull them raw.

## 2. Core definitions (one place, used by every view)

| Concept | Definition |
| --- | --- |
| Slot | `(kind, schema_version, target, dataset_type, model_horizon_minutes, variant)`. A refit replaces the build in the same slot; never group by `fit_id`. |
| Input time to tip `r` | Intermediate: `snapshot_horizon_minutes`. Closing: `tipoff_at_prediction − run_started_at`. This measures what the model knew. The publication lag (`available_at − input_at`, ~20 min on a warm runner) is shown, not bucketed. |
| Time bucket | Snapshot grid `G = 30, 60, 120 … 720, 840, 960, 1080` plus `T−18h+`. `bucket(r) = min{h ∈ G : h ≥ r}`, the last snapshot that had passed, covering `(previous h, h]`. Closing also gets a **Final** column: the last output available before tip, the same rule as Yesterday. |
| One row per cell | Per `(game, slot, source, bucket)` keep the prediction with `r` closest to `h` (right after the snapshot passed). This makes the live diagonal comparable to replay. |
| Grading | Same as `grade_history` (`domain/history.py`): the signal requires \|edge\| > per-kind min edge, measured against **that row's stored Bet365 quote**. Results are WIN/LOSS/PUSH/NO BET/NO LINE/VOID. |
| Win % | `W / (W + L)` from pooled counts, with a per-cell Wilson 95% CI and a break-even reference at 52.38% (−110). Coverage = bets / expected settled-game opportunities, including missing outputs. Pooled margins use one opportunity per observed slot/time cell. Secondary metrics: MAE (all available predictions on final games) and unit ROI from the stored prices (null when the price is missing, never an assumed −110). |

## 3. Page layout

A shared filter bar: date range (default 2026-10-27 through today's NBA slate), season type
(regular), model family (XGBoost 2.5 / TabPFN), per-kind min edge
(`minimum_edge_select`), and a metric switch (Win % / Bets / MAE / ROI).
Below it, a **target selector**: Line error | Total points | Spread error.
These are the three views, each with the same sections; TabPFN appears only
where it has outputs (no spread).

1. **KPI strip**: settled games, then pooled win % for closing Final, live
   intermediate on-horizon (pooled) and replay on-horizon (pooled).
2. **Closing by prediction time** (main closing answer): a line chart with
   x = time bucket (`T−18h+` → `T−30m` → Final), y = win %, a CI band, a
   break-even rule and bet-count labels. The table goes in an expander. Pooled
   KPI cards do not show a CI, because their repeated game observations are
   correlated.
3. **Model × time heatmap: live** (the main answer). Rows are Closing plus the
   16 intermediate horizons (longest at the top). Columns are the time
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

## 4. Combining predictions across time (secondary)

Every combined pick is graded at **the line available when it would be
placed** (time T), never at a later line.

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
  Never highlight a "best" cell.
- Cells in one column share games, so they are correlated. The CI holds per
  cell only.

## 6. Implementation outline

| Piece | Where |
| --- | --- |
| `read_history_range(start, end, target, season_type)`: one SQL query computing `r`, source and bucket, a window-rank reduction per `(game, slot, source, bucket)` (plus the Final row), narrow columns only, `SET LOCAL statement_timeout` raised for this read; a separate small query returns game counts | `src/nba_ou/dashboard/repositories/analysis.py` (bucket `CASE` generated from the grid constant) |
| Grid constant, bucket labels, Wilson CI, cell aggregation, consensus/agreement | new `src/nba_ou/dashboard/domain/analysis.py` |
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
- Inclusive date range (2026-10-27 through today by default), season type (regular
  by default), model family, three targets, per-kind minimum edges and the
  Win % / Bets / MAE / ROI switch. Family, metric and edge changes reuse source
  data; range, target and season type are cache keys. Refresh evicts that key.
  Earlier days had dataset errors and are excluded by default, not prohibited
  from manual selection. The end follows the current Pacific NBA slate date.
- Final-game and pooled win-rate KPIs, closing input-time timeline with Final,
  confidence bands and bet counts, live slot/time heatmap with outlined
  diagonal and low-sample shading, and own-horizon live/replay intervals.
- Closing-style intermediate replay timeline: one aggregate point per model
  family and trained horizon, using cached own-snapshot replay predictions.
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
