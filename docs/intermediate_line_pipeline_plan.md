# Running the intermediate-line dataset through `training_pipeline`

Status: **plan, not implemented.**

The dataset exists: `data/train_data/intermediate_line_data_20260412.csv`,
36,570 rows x 2,430 columns, 6,122 games, six snapshots each
(30/60/120/240/480/720 minutes before tip), seasons 2021-2025.

The question this document answers: the pipeline assumes one row per game, and
now there are six. What changes, and what deliberately does not.

The stated constraint is **keep the evaluation as it is**. Everything below is
built around honouring that literally — not by patching the evaluator to
understand groups, but by arranging for the thing being evaluated to be one row
per game again.

---

## 1. The measurement that decides the design

Within a single game, across its six snapshots:

| quantity | value |
|---|---|
| std of `TOTAL_LINE_bet365` across the 6 snapshots (median over games) | **0.75 points** |
| std of `LINE_ERROR` at every individual snapshot | **~18.0 points** |
| mean `LINE_ERROR` by snapshot (T=30 -> T=720) | 0.660, 0.682, 0.688, 0.703, 0.604, 0.563 |

The six rows of a game carry **one game's worth of information about the
outcome**. They differ only by a small perturbation of the line — a
three-quarter-point wobble against eighteen points of irreducible noise. What
they add is not six independent observations; it is variation in the *question*
(what line am I being offered) attached to a single realisation of the *answer*
(what the game did).

Every consequence below follows from that one fact.

---

## 2. What already works, unchanged — verified, not assumed

These were checked against the built CSV and the pipeline source, so they need
no work:

**Split boundaries are date-aligned.**
`make_test_anchored_walk_forward_splits` (`modeling.py:843-895`) builds test
windows out of whole dates — `temp["_date"].isin(test_dates)`, with train being
strictly `_date < test_start_date`. All six snapshots of a game share one
`GAME_DATE`, so a game can never straddle a fold boundary. The same holds for
the calendar-based `holdout.test_days`. **There is no train/test contamination
from the replicated grain.** This was the failure mode most likely to
invalidate the whole exercise, and it is genuinely absent.

**`GAME_ID` survives as a string.**
`load_raw_training_csv` forces every column with "ID" in its name to `str`, so
`0022100001` keeps its leading zeros and `resolve_season_type` maps the prefix
correctly. `exclude_playoffs` and `allowed_season_types` work as they do today.

**The target is already snapshot-local.**
The CSV sets `TOTAL_LINE_bet365` to the *normalized snapshot* line and
`LINE_ERROR = TOTAL_POINTS - that line`. A `line_error_regressor` run with
`line_col` omitted therefore predicts exactly the right quantity with no
config gymnastics.

**The opener baseline is available.**
`betting.comparison_line_cols: ["TOTAL_LINE_consensus_opener"]` is present in
the CSV (it is whitelisted in `SAFE_ODDS_COLUMNS`), so "beat the opener"
scoring works out of the box.

**Post-filter size is comparable to the closing dataset.**
After `season_year_floor: 2021` + `exclude_playoffs`: 34,686 rows / **5,808
games**, versus 5,980 games in `all_odds_training_data_until_20260318.csv`.
Within 3%. A per-snapshot slice is a structural drop-in.

---

## 3. What breaks silently

None of these raise. All of them produce a plausible-looking number that is
wrong.

### 3.1 Row-counted windows shrink the training set 6x -- FIXED

Every `*_games` setting used to count **rows** (`train_pool.tail(train_games)`).
Left at base defaults on a six-snapshot dataset each fold trained on ~417 games
while the closing-line model trained on 2,500, so any accuracy difference
measured dataset *size* rather than dataset *quality*. The campaign worked
around it by multiplying every knob by the snapshot count in a YAML comment --
a correction no test could check, and one that stops being right the moment the
grid changes (it did: the grid is ten horizons now, not six).

They count **games** now. `training_pipeline.splits` resolves a per-row game key
from `GAME_ID` and every window, span and floor is expressed over distinct
games; `nba_ou.modeling.modeling.tail_n_games` does the same for the daily
walk-forward, and `training_pipeline.data.rolling_window_index` for the refit.
Whole games only -- a window never keeps four of a game's ten snapshots.

`train_games: 3500` therefore means the same history on both datasets, and the
pooled arm and its single-horizon control land on identical fold layouts. On a
one-row-per-game frame the game key resolves to `None` and the original tail
slice runs unchanged, so the closing-line path is untouched by construction.

`holdout.test_days` never needed rescaling: it is calendar-based, which is what
gives both datasets the identical evaluation window.

A pooled frame may no longer be split by `test_anchored` or `last_n_seasons` at
all. Those come from `nba_ou.modeling` and describe a fold as a block of N rows;
the config layer refuses the combination rather than silently scoring a tenth of
what the numbers say.

### 3.2 Confidence intervals become ~2.4x too narrow -- FIXED

`evaluate_betting` and `wilson_interval` (`betting.py:48-71`, `132-260`) are
purely row-wise. Nothing in `training_pipeline` groups by `GAME_ID` anywhere.

Six bets on one game are one bet's worth of evidence: they share an outcome and
their lines differ by three quarters of a point. Feeding them to a binomial
interval as six independent trials understates the width by roughly `sqrt(6)`
= 2.4x.

This matters more here than it would elsewhere, because `_base.yaml` already
documents the honest position: *"At -110 a true 55% win rate needs ~1400 bets
before its interval clears break-even."* A 6x-inflated `n_bets` would appear to
clear that bar on a sixth of the real evidence. It would make an inconclusive
result look conclusive -- the single most expensive error available here.

**Handled, and no longer opt-in.** `training_pipeline.snapshot_scoring` regroups
finished predictions one horizon at a time, where there really is one row per
game, and blanks the pooled row's interval and significance verdict rather than
printing a corrected one it cannot justify (the true factor is not `sqrt(N)`:
horizon coverage varies and the min-edge filter selects differently at each).

`run_experiment` now builds that report itself for any frame with several rows
per game, writes `snapshot_{cv,holdout}_metrics.csv` beside the other
artifacts, and the CLI prints it under a header saying the pooled row is not a
bet count. It used to require launching a separate wrapper script; the plain
CLI printed the inflated number with nothing saying so.

The same applies to `betting.evaluate_cv_folds` and to every `n_bets` on the
leaderboard.

### 3.3 The two datasets do not end on the same date

`holdout.test_days: 60` anchors on each dataset's last date, so:

| dataset | last regular-season date | 60-day holdout |
|---|---|---|
| intermediate | 2026-04-12 | Feb 11 – Apr 12 |
| `..._until_20260318.csv` | 2026-03-18 | Jan 17 – Mar 18 |
| `..._until_20260408.csv` | 2026-04-08 | Feb 07 – Apr 08 |

A 25-day (or even 4-day) shift means the two models are scored on different
games in different market conditions. `_base.yaml` chose `test_days` precisely
so that *"two datasets then get the IDENTICAL calendar window"* — that only
holds when both end on the same day, and here they do not.

### 3.4 Optuna selects on an inflated effective n -- STILL OPEN

The objective averages fold metrics over rows. With ten correlated rows per
game, `subsample`/`colsample` bootstraps draw near-duplicates, and early
stopping tunes its round count against a validation fold whose effective size
is a tenth of its row count. The search will systematically prefer less
regularisation than the data supports.

This is the one item in §3 that is **not** fixed, and it is worth being precise
about what changed around it. The fold *layout* is now game-aware: `train_games`
and `min_validation_games` count distinct games, and the pooled arm and its
single-horizon control score identical cohorts. But under
`objective_aggregation: pooled` the objective still weights each ROW equally, so
a game with ten horizons contributes ten times and one with six contributes six.
That is not merely an inflated `n` -- it is an uneven weighting across games.

The obvious correction is a per-row weight of `1/snapshots`. That was built once,
shipped in the scoring sidecar as `SNAPSHOT_WEIGHT`, read by nothing, and has now
been deleted rather than wired in: the pipeline's `sample_weight` machinery is
recency-only, so using it means multiplying two weight sources, and it would fix
only the training side -- `evaluate_betting` has no weight parameter at all.

Two honest ways to live with it:

- **Read the single-horizon control as the primary number.** One row per game,
  so the objective weighting question does not arise. This is the strongest
  argument for running that arm.
- **Treat pooled CV as a ranking device only**, which is what §2 already says
  about CV generally, and let the per-horizon holdout tables carry the estimate.

---

## 3.5 Snapshot coverage and staleness (measured)

Coverage at 12h is not a constraint: **97.6%** of post-filter games have a
720-minute row (5,670 of 5,808), and `SNAP_TOT_BET365_HAS_QUOTE` is 100% on all
of them — no row is fabricated from an absent market. By season: 97.9 / **90.3**
/ 99.5 / 100.0 / 99.9 for 2021-2025. Only 2022 is meaningfully short.

Staleness is the more interesting number. Line age for the bet365 total:

| snapshot | median age | p75 | p90 |
|---|---|---|---|
| 30m | 57 min | 211 | 436 |
| 60m | 78 | 269 | 458 |
| 120m | 184 | 331 | 477 |
| 240m | 146 | 276 | 401 |
| 480m | 94 | 178 | 302 |
| 720m | 86 | 291 | 518 |

Two things follow. At 12h, 10% of snapshots read a line last changed ~20.6
hours before tip — effectively the opener. And even at 30 minutes out the
median line was last changed 57 minutes earlier: the store's tick cadence is
coarse. This is the same fact behind the 0.75-point median within-game line
spread in §1. **The six snapshots are less distinct from one another than the
grid implies**, which argues for pooling them into one model rather than
fitting six.

---

## 4. Recommended approach

The betting use case is "I place bets at whatever time I happen to be free",
so the deliverable is **one model that conditions on time-to-tip**. Six
separate models would require picking one by clock-watching and could not
interpolate to a time not on the grid. Design B below is therefore the product;
Design A is retained only as a control.

### Design B — pooled model, per-snapshot evaluation (primary)

Train one model on all 34,686 rows with `TIME_TO_MATCH_MIN` as a feature, so it
can learn how the mapping changes with time to tip. Then **score one snapshot at
a time**: filter the held-out predictions to a single `TIME_TO_MATCH_MIN` and
call the existing `evaluate_betting` on each slice.

That split is what makes the whole thing work. Pooling on the training side is
what you want to ship; slicing on the evaluation side is what keeps `n_bets`
equal to games, so §3.2 never arises and the evaluator itself is never touched.

What it needs:

- The §3.1 window rescale (config only, no code).
- `TIME_TO_MATCH_MIN` present in the feature matrix — it is a plain CSV column,
  so it is a feature by default. Confirm it survives cleaning rather than
  assuming.
- `GAME_ID` and `TIME_TO_MATCH_MIN` carried into `predictions_df` so the
  per-snapshot breakdown is possible at all (see work item 3).
- §3.4 stays partly unaddressed: correlated rows inside a fold still inflate
  Optuna's effective n. Snapshot weighting (work item 5) is the mitigation, and
  is worth running as an A/B rather than assuming it helps.

### Design A — single snapshot, as a control (one run, not six)

Filter to **T=720** only: 5,670 games, one row per game, structurally identical
to the closing-line dataset. Everything in §3 disappears — no rescale, honest
intervals, no correlated folds.

Its job is not to be the product. It is the answer to "is pooling earning its
complexity?" If the pooled model cannot beat a model trained solely on 12h rows
*at 12h*, the pooling is costing more than it returns and the design should be
reconsidered. One run, and it also gives a clean read on whether there is signal
at the longest horizon at all.

Set `data.snapshot_minutes: 720` and run it exactly like the closing-line
dataset. This is a config knob, not a file transform: the arm and its control
read the same bytes, so one `expected_checksum` covers both and they cannot
drift apart. (It replaces `slice_intermediate_snapshot.py`, which made the
control by writing a second ~187MB CSV.)

What it buys, all of it as a measuring instrument rather than a deliverable:

- **§3.1 is moot here** -- and fixed for the pooled arm too, so this is no
  longer a reason to prefer the control. `train_games: 2500` means 2,500 games
  on both.
- **§3.2 disappears.** One row per game, so `n_bets` is games and the Wilson
  interval is honest without any slicing step.
- **§3.4 disappears.** No correlated rows inside a fold.
- `TIME_TO_MATCH_MIN` is constant within the slice, so the cleaner drops it
  automatically. No accidental snapshot indicator.

Read it two ways. Against the **pooled model scored at T=720**, it says whether
pooling helped or hurt at the horizon that matters. Against a **closing-line
run on the same window**, it says whether there is exploitable signal 12 hours
out at all — like-for-like, same games, same window sizes, same evaluator, one
deliberate difference: which line the model is betting into.

T=720 is the right control because it is both the hardest case and the real use
case. If 12 hours out works, the shorter horizons are strictly easier.

### Why not "just weight the rows 1/6"

Tried and removed. The sidecar used to carry `SNAPSHOT_WEIGHT = 1/snapshots`,
and nothing ever read it. Wiring it in would mean extending the config and
multiplying two weight sources together, because the pipeline's `sample_weight`
machinery is recency-only (`build_recency_sample_weights`). More importantly it
fixes only the *training* side: `evaluate_betting` has no weight parameter at
all, so the pooled-bet-count problem would survive untouched.

The correction that shipped instead is `training_pipeline.snapshot_scoring`,
which regroups finished predictions one horizon at a time. Within a horizon
there is exactly one row per game, so the binomial maths is correct again
without weighting anything.

---

## 5. Work items

Ordered. Items 1-4 are what the primary design needs.

**1. `data.date_max`** — ~8 lines in `prepare_dataset`, beside the existing
`apply_season_year_floor` / `filter_allowed_season_types` calls at
`data.py:372-385`.
Truncate both datasets to a common last date so the 60-day holdout covers the
identical calendar window (§3.3). Pin it to the earlier of the two ends.
Alternative without code: regenerate a closing-line CSV through 2026-04-12 —
more compute, and it has to be redone every time either dataset is rebuilt.

**2. `data.snapshot_minutes` filter** — ~15 lines, same place.
Optional field on `DataConfig`; `None` = no filter, so every existing config is
unaffected. Needed for the Design A control run, and useful for any later
single-horizon diagnostic. Preferred over writing per-snapshot CSVs: the
campaign config states which snapshot it is on, and the checksum stays one
value across every run.

**3. Carry `GAME_ID` and `TIME_TO_MATCH_MIN` into `predictions_df`** — ~2 lines
at `evaluation.py:295-316`.
Today that frame holds `y_true`, `y_pred`, `target_line`, `predicted_edge`,
`TOTAL_POINTS` and `GAME_DATE`, but no game key and no snapshot. Both are
sitting in `df_test_full`. Without them the per-snapshot breakdown that the
whole design rests on cannot be computed from the saved
`final_test_predictions.parquet` at all. Purely additive — nothing reads that
frame positionally.

**4. Per-snapshot scoring script + campaign configs.**
- A standalone script that loads `final_test_predictions.parquet`, groups by
  `TIME_TO_MATCH_MIN`, and calls the **existing, unmodified** `evaluate_betting`
  on each slice. This is the whole of "keep the evaluation as it is": same
  function, applied to one-row-per-game inputs.
- `experiments/intermediate_line_2026_08/pooled.yaml` — all snapshots,
  `prediction_strategy: line_error_regressor`, `line_col` omitted, the §3.1
  rescaled windows, `csv_path` + `expected_checksum` (changes with the fanatics
  rebuild), `date_max`.
- `t720_control.yaml` — `snapshot_minutes: 720`, base windows unrescaled
  (one row per game, so the defaults are already correct).
- `closing_reference.yaml` — the closing-line dataset with the *same*
  `date_max` and `season_year_floor`, so the comparison has a controlled arm.
- Shared `comparison_group`; distinct `hypothesis` per file.
- Then `poetry run python scripts/preflight_campaign.py experiments/intermediate_line_2026_08`
  before spending any tuning time.

**5. Snapshot weighting** — extends `SampleWeightConfig` to compose a per-row
weight with the recency weight. Mitigates §3.4. Run as an A/B against the
unweighted pooled model rather than assumed to help.

**6. CLV scoring against the true close** (optional, highest analytical value).
`intermediate_line_data_20260412_scoring.csv` holds `CLOSING_*` joined on
`GAME_ID` + `TIME_TO_MATCH_MIN`. `training_pipeline` never joins it, so "did the
model's pick beat the closing line" is not currently measurable. This is the
metric that most directly tests the premise — that betting early captures value
the close later confirms — and it is unavailable today.

---

## 6. Open question worth settling before running anything

**What counts as success?** The closing-line model reads the closing line, the
single most informative feature either dataset has. On MAE against
`TOTAL_POINTS`, the intermediate model should be *expected* to lose, and that
would say nothing about whether the idea works.

The comparable quantity is directional OU accuracy **each against its own
executable line** — the closing model against the close, the intermediate model
against the snapshot line it could actually have bet. Both runs must share a
`date_max` and `season_year_floor` for that to mean anything, which is what
work items 2 and 3 exist to guarantee.
