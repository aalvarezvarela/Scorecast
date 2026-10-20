# Intermediate-line feature engineering — round 2

Status: **proposal, nothing implemented.** Follows
`docs/intermediate_line_dataset_plan.md`, which is built and describes the
existing feature set. Everything here is measured on the as-built dataset
(seasons 2023–24, 14,670 rows) rather than proposed from intuition.

---

## 1. What the data says, before proposing anything

Four measurements taken specifically to prioritise this work. They point
somewhere quite different from where I would have guessed.

### (a) There is real headroom — the line keeps moving after every snapshot

| snapshot | mean abs. move from T to close | moves at all | std |
|---|---|---|---|
| 30 min | 0.52 | 56.9% | 0.82 |
| 60 min | 0.71 | 66.4% | 1.07 |
| 2 h | 0.96 | 75.8% | 1.39 |
| 4 h | 1.08 | 80.3% | 1.51 |
| 8 h | 1.40 | 84.5% | 1.93 |
| 12 h | 1.84 | 89.9% | 2.42 |

Betting at 12h means betting into a line that will move ~1.8 points nine times
out of ten. Predicting the *direction* of that move is the tradeable quantity.

### (b) The closing line really is sharper — but only slightly

Mean |error| against the actual total: **14.094 for the closing line** versus
14.156 (30 min) rising to 14.244 (12 h) for the snapshot line. So perfect
foresight of the closing line buys ~0.06–0.15 points of MAE. That is a genuine
edge and a small one; it should temper expectations for everything below.

### (c) Cross-sectional signal beats time-series signal, by ~3x

Correlation with the subsequent move from T to close:

| signal | T=30 | T=240 | T=720 | vs `LINE_ERROR` |
|---|---|---|---|---|
| **anchor deviation from consensus** | **−0.247** | **−0.156** | −0.083 | −0.024 |
| **consensus steam net** | **+0.160** | +0.099 | +0.072 | +0.019 |
| position in realised range | +0.059 | +0.099 | +0.074 | +0.019 |
| move from open | +0.012 | +0.070 | **+0.092** | +0.027 |
| move last 60 | +0.088 | +0.039 | +0.037 | +0.014 |
| line age | +0.011 | +0.006 | −0.023 | −0.015 |
| n moves so far | −0.029 | −0.027 | +0.023 | −0.012 |

Three things follow:

1. **Where a book sits relative to other books matters far more than its own
   history.** Deviation from consensus is the strongest signal found, at −0.247.
   The sign says convergence: a book above consensus subsequently moves down.
2. **Signal strength is strongly horizon-dependent, and not in one direction.**
   Deviation decays with horizon (−0.247 → −0.083) while move-from-open *grows*
   (+0.012 → +0.092). Different features matter at different lead times.
3. **Correlations against `LINE_ERROR` itself are all ≤ 0.03.** Game outcome is
   noise-dominated (σ ≈ 14). The learnable structure is in the *line*, not the
   game — which drives the architecture proposal in §4.

### (d) The "find the sharp book" idea is weak — a negative result

Correlation of each book's deviation at T with the *consensus* move from T to
30 min (positive ⇒ consensus moves toward that book ⇒ it leads):

| book | T=720 | T=240 | T=120 | mean abs. deviation |
|---|---|---|---|---|
| FanDuel | 0.053 | 0.061 | **0.096** | 0.193 |
| DraftKings | −0.005 | 0.020 | 0.085 | 0.152 |
| BetMGM | 0.018 | 0.062 | 0.059 | 0.112 |
| bet365 | 0.041 | 0.009 | 0.026 | 0.119 |
| Caesars | 0.014 | 0.021 | 0.023 | **0.352** |

No book leads meaningfully. Caesars is a striking case: by far the largest
deviations (0.352, ~3x the others) and essentially no lead score — it is the
*noisy* book, not the sharp one.

Note the asymmetry against (c): a book's deviation predicts **its own**
reversion strongly (−0.247) but the **consensus's** movement weakly (≤0.096).
The signal is *"this book is stale and will catch up"*, not *"this book knows
something"*. That distinction determines what is worth building.

---

## 2. What already exists

So that round 2 does not rebuild it: snapshot state (raw/normalised line,
prices, devigged probabilities, overround, line age, books quoting); cross-book
consensus, std, range, per-book deviation, steam counts over 60 min; movement
from open, windowed moves/velocity over 15/30/60/120/180/360/720 with `HAS_`
flags,
acceleration; counts (moves, price-only ticks, distinct levels, reversals,
moves per hour); path shape (max/min/range/position-in-range); and prior-game
open→close dynamics rolled up per team and league-wide.

---

## 3. Proposed families, ranked by measured evidence

### Tier 1 — staleness and convergence (evidence: r = −0.247)

The strongest signal in the data, and the current implementation understates it.

**A methodological fix first.** The consensus currently *includes* the anchor
book, so the anchor's deviation is mechanically shrunk toward zero — the book is
partly compared against itself. **Leave-one-out consensus** (median of all books
except the anchor) is strictly better and should be the benchmark for every
deviation feature. This alone should strengthen the −0.247 measured above.

Then:

- `deviation_z` = deviation / cross-book std — scale-free, so "0.5 points off"
  means something different on a tight market than a scattered one.
- `deviation_rank`, `is_most_extreme_book`, `n_books_above` / `n_books_below` —
  ordinal position is robust where the raw gap is noisy.
- `deviation_persistence_minutes` — how long the anchor has sat on the same side
  of consensus. A book briefly offside is noise; one offside for six hours is
  stale.
- `consensus_moved_anchor_did_not` — the sharpest case of all: the market has
  moved and the anchor has not yet followed, i.e. a price still available at the
  old number. Combines tiers 1 and 2 and is the single feature I would build
  first.
- Same family for spread and moneyline, since staleness is a property of the
  book's operation rather than of the market.

### Tier 2 — steam and agreement dynamics (evidence: r = +0.160)

- Multi-window steam (15 / 30 / 120 / 240 min), not just 60.
- **Steam magnitude**, not only counts: sum and median of the books' moves.
  Five books moving 0.5 is a different event from five moving 2.0.
- `steam_unanimity` (every quoting book moved the same way).
- `time_since_last_consensus_move` — quiet markets and active ones behave
  differently.
- `anchor_participated_in_steam` — pairs with tier 1.

### Tier 3 — horizon interactions (evidence: signal varies ~3x by horizon)

Measurement (c) shows features reverse their relative importance across
horizons, so the model must be able to condition on lead time.

**This is not something the model can be relied on to learn.** The tuned search
space allows `max_depth: 1`
([`experiments/_base.yaml`](experiments/_base.yaml)) — stumps cannot represent
an interaction at all — and even at depth 4 with `colsample_bytree` as low as
0.05, the chance of the right pair landing in one tree is small. So build them
explicitly:

- `deviation_z × TIME_TO_MATCH_MIN`, `steam_net × TIME_TO_MATCH_MIN`, and the
  same for move-from-open.
- `expected_remaining_move_std` — the empirical σ of future movement given the
  horizon, from table (a), estimated on prior seasons only. Tells the model how
  much is still unresolved.

An honest alternative worth one campaign cell: **fit per-horizon models** instead
of one pooled model with interactions. Six models on ~2,400 games each is thin,
but it sidesteps the interaction problem entirely.

### Tier 4 — market-hours context (untested, cheap, plausible)

Not yet measured, but the mechanism is well established and `tipoff_utc` is
already available:

- Snapshot local time in US Eastern, and an `is_overnight` flag. A 12h snapshot
  at 03:00 ET is a very different market from 12h at 10:00 ET — overnight lines
  are posted thin and move on low volume.
- Slate size that day, and the game's position within it.
- Days into the season: early-season lines are softer, before the market has
  priced roster changes.

Worth measuring before building — a single correlation table like (c) would
settle it.

### Tier 5 — price microstructure (cheap, partly free)

- **`norm_line − raw_line`** — the half-tick the book has priced but not yet
  taken. Both columns exist already; their difference does not, and it is the
  cleanest available read on pressure that has not reached the line.
- Vig level, and vig *change* over each window. A widening vig signals a book
  reducing exposure.
- `fair_left − 0.5`: how far the devigged price sits from even.

### Tier 6 — cross-market coherence

- **Implied team totals from the snapshot total and spread.** Note this is the
  same construction as the leaky `IMPLIED_PTS_*_BEFORE` (§4 L2 of the dataset
  plan) — but rebuilt from the *snapshot* line it is entirely legitimate. Worth
  stating in the code, since the name will otherwise look alarming.
- Total-move versus spread-move coherence: a spread moving hard while the total
  holds is a different state from both drifting.
- Moneyline-implied versus spread-implied win probability gap — a staleness
  detector across markets within one book.

### Tier 7 — matchup and level context

- Head-to-head prior meetings this season: their lines, closing lines, actual
  totals and errors. All finished, so all known at T.
- Snapshot line as a z-score against the two teams' recent line history — is
  this an unusual number for these teams?
- Line percentile within the season-to-date distribution.
- Proximity to round numbers (220, 230). Weak for totals — key numbers matter far
  more on spreads (3, 7), so apply this to the spread market first.

### Not worth building

- **Sharp-book identification / per-book lead scores.** Measured at ≤ 0.096
  (§1d). The convergence story explains the data better.
- ~~**Sub-30-minute windows.**~~ **Built anyway, deliberately.** The original
  reasoning still holds on the data -- a book moves in the trailing 60 minutes
  on only 32% of (row, book), and 37% of rows have no book moving at all, so
  `move_last_15` is zero far more often than not. It was added because "mostly
  zero" is not the same as "no signal": the rows where a book *does* move
  fifteen minutes out are exactly the late-money rows, and they were previously
  indistinguishable from a quiet market. `has_window_15` separates "did not
  move" from "no history". Judge it on measured importance, and drop it if it
  earns nothing -- windows are cheap to remove and expensive to add back.

  One consequence worth knowing: steam is **pinned** to `STEAM_WINDOW_MINUTES`
  (60) rather than "the shortest configured window", because cross-book
  agreement needs books to have moved. At 15 minutes steam would be a
  near-constant zero, and adding a window would silently have redefined an
  existing feature family.
- **Full feature families for every book.** Width for little gain; the reduced
  per-book set plus a leave-one-out consensus captures the cross-sectional
  signal.

---

## 4. The architecture change I would make first

Measurement (c) is stark: the best signal correlates **−0.247 with future line
movement** and **−0.024 with `LINE_ERROR`**. Ten times the signal, against a
target with roughly a tenth the noise (σ 0.8–2.4 versus σ ≈ 14).

So: **train a CLV model as well as an error model.**

- **Auxiliary target** `CLOSING_MOVE = closing_line − snapshot_line`. Already
  computable — `CLOSING_TOTAL_LINE_<book>` is carried in the dataset for exactly
  this and is excluded from features.
- **Two-stage option:** predict the closing line from the snapshot, then feed
  `predicted_close` (or `predicted_close − snapshot_line`) as a feature to the
  existing `line_error_regressor`. The first stage learns something genuinely
  learnable; the second keeps the target the thing you are paid on.

Two cautions, both real:

1. The ceiling is bounded by measurement (b): perfect foresight of the closing
   line is worth ~0.15 points of MAE at 12h. The two-stage model cannot exceed
   that through this channel.
2. `CLOSING_MOVE` must never become a feature, and the stage-1 model must be
   fitted **inside** the walk-forward, not once on the whole dataset. Fitting it
   globally would be the "spectacular ROI against another line" failure mode with
   extra steps.

---

## 5. Leakage rules specific to these families

The dataset's existing gate handles column-shaped leakage. These families
introduce a different kind.

- **Fitted features are target encodings.** `expected_remaining_move_std`,
  per-team drift priors and any stage-1 prediction are estimated *from data* and
  must be fitted on strictly prior games — expanding window, inside each fold.
  Fitting them once on the full dataset leaks the future into every training row
  and is invisible to a column-name gate.
- **Leave-one-out consensus must exclude the anchor at every use**, including
  inside the deviation-persistence and steam features, or the anchor is being
  compared to itself.
- **Head-to-head features must use strictly earlier meetings** — the same-date
  hazard as the league rollups.
- Everything in tiers 1–7 is otherwise computable from ticks at or before T, and
  should go through the same as-of machinery rather than a parallel path.

---

## 6. Sequencing, and how to know if any of it worked

1. Leave-one-out consensus + `consensus_moved_anchor_did_not` (tier 1). Smallest
   change with the best measured evidence.
2. Measure tier 4 with a correlation table before building it.
3. Tiers 2, 3, 5 as separate campaign cells.
4. The CLV auxiliary target (§4) — arguably worth doing before 2–3, since it
   changes what the model is asked to learn rather than adding columns.
5. Tiers 6, 7 last.

Evaluation discipline, from `.claude/skills/experiments` and unchanged here:

- **One deliberate difference per cell.** These families are individually small;
  bundling them makes the result unreadable.
- **Read `seed_roi_range` first.** Same-config seed ROI spans **4.9–12.0 points**.
  Nothing smaller than that is a result, and every effect proposed here is
  plausibly smaller than that.
- At ~10 cells against this noise floor, **expect about one to look good by
  luck**. Pre-register expectations in the config header.
- Rank on pooled CV, estimate on holdout, and treat a large `cv_minus_holdout_roi`
  as its own finding.
- Remember the row-counted windows need rescaling by the snapshot multiplier
  (§7 of the dataset plan) — the pre-flight reports the real per-fold size.

A useful intermediate check that avoids burning GPU time: because
`CLOSING_MOVE` is a low-noise target, a feature family can be screened by
whether it improves *that* regression first. If a family cannot predict line
movement, it is very unlikely to predict game outcome.

---

## 7. Serving implications

Out of scope for now — the dataset is training-only — but worth knowing while
choosing what to build. Most of what is proposed is servable from a **single live
multi-book scrape**, which the daily job already performs in substance:

- Tiers 1, 5, 6 need only the current cross-book snapshot.
- Tiers 3, 4, 7 need only the clock, the schedule and finished games.
- Tier 2 and the persistence features in tier 1 need intra-day tick history,
  which is the one genuine gap.

That reinforces the recommendation already in the dataset plan: **start
persisting timestamped daily odds scrapes now.** It cannot be backfilled, and it
is what makes tier 2 servable later.
