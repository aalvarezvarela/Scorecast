# Intermediate-line training dataset — implementation plan

Status: **implemented.** Companion to `docs/line_history_phase0_findings.md`,
which established the line-history store this dataset reads from.

Build it with:

```bash
poetry run python scripts/create_train_data/create_intermediate_line_train_data.py
```

By default the training CSV includes
`ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE`: the walk-forward Ridge
estimate, in points, of the anchor total line at the final pregame tick minus
its value at this snapshot. Use `--no-ridge-movement` to omit the column and
skip its calculation, or pass `include_ridge_movement=False` to
`create_intermediate_line_df`. Each historical row uses only games that tipped
off before that row's snapshot, from its current and immediately previous
seasons. Each completed game has total training weight one across its snapshots.
The first 100 eligible games receive a neutral value of zero; 0-minute rows
also receive exactly zero. This feature is computed from the same tick series
as the snapshot lines, since the separate closing-odds table does not always
agree with that series.

It emits **two files**: the training CSV, and a `_scoring.csv` sidecar holding
the closing lines, the per-row snapshot weight and raw timestamps. See §10 for
why they are physically separate rather than distinguished by a prefix.

As-built figures on seasons 2023–24 (2 seasons, 6 snapshots): **14,670 rows,
2,446 games**, NaN load flat across snapshots (134.7–135.0), and line accuracy
improving monotonically toward tip (mean |LINE_ERROR| 14.24 at 12h → 14.16 at
30m). Two plan revisions came out of building it: a second and worse instance of
the `_BEFORE` leakage trap (§4, L2) and a spread sigma calibrated from data
rather than assumed (§6).

**§10 records six defects found in review after the first build.** One of them
(closing lines reaching the feature matrix) would have invalidated any result
produced before it was fixed.

Goal: a second, independent training dataset at grain
**(game, pre-game snapshot)** so a model can be trained to bet at whatever time
before tip-off we happen to be betting, rather than only at close.
`create_df_to_predict()` and everything it feeds is left untouched.

**Scope: historical training data only.** Same-day prediction serving is
deliberately out of scope and will be added later — see §9.

---

## 1. How intermediate lines are stored today

### Source data

Scraped SBR line-history CSVs under `data/sbr_line_history/<season>/line_history/*.csv`.
**The scraper that produced these is not in this repo** — `lab/scrape_odds/scrape_sportsbook.py`
and `src/nba_ou/fetch_data/odds_sportsbook/scrape_sportsbook.py` both scrape the
*current* odds grid (plus the consensus opener), not the line-history popup. This
matters a great deal for production and is picked up in §9.

### Two stores exist; use the newer one

| Store | Location | Verdict |
|---|---|---|
| Legacy per-market schemas | [process_sportsbook_line_history_data.py](src/nba_ou/postgre_db/odds_sportsbook_line_history/process_sportsbook_line_history_data.py), schemas `odds_sportsbook_{totals,spread,moneyline}_line_history` | superseded — no tip-off join, so no leakage filter |
| **Aiven line-history store** | [line_history_aiven/](src/nba_ou/postgre_db/line_history_aiven/), schema `line_history` | **use this** — commit `f011598`, tip-off-aware |

### Aiven schema ([schema.py](src/nba_ou/postgre_db/line_history_aiven/schema.py))

```
lh_game (game_id PK, game_date, season_year, tipoff_utc, event_id, team_home, team_away)
lh_line (game_id, season_year, market_id, book_id, line_ts, mins_to_tip,
         is_pregame, is_opener, left_line, left_price, right_line, right_price)
         PARTITION BY LIST (season_year)
lh_book (book_id, slug, name)     lh_market (1=totals, 2=point_spread, 3=money_line)
lh_load_meta (per-season timezone + confidence + drop counts)
```

Three encoding facts that any reader must honour:

1. **Lines are stored doubled as `SMALLINT`** — `224.5` is stored as `449`. Every
   read must divide by 2. This is a silent-corruption landmine, not a leakage one.
2. **`mins_to_tip` is negative pre-game.** `is_pregame = mins_to_tip < 0`.
3. **Ticks are *changes*, not samples.** There is no row for "the line at 14:00";
   there is a row each time a book moved. A snapshot is therefore an **as-of /
   last-observation-carried-forward** read, never an equality match.

Loaded state, measured against Aiven for this plan:

| season_year | rows | pre-game rows | games |
|---|---|---|---|
| 2021 | 281,161 | 228,983 | 1,284 |
| 2022 | 201,433 | 193,246 | 1,171 |
| 2023 | 264,200 | 255,184 | 1,198 |
| 2024 | 325,869 | 316,281 | 1,248 |
| 2025 | 584,188 | 565,698 | 1,227 |

**6,128 games total. 2019-20 and 2020-21 are deliberately not loaded** (Phase 0
found no clean DST step; timezone confidence "low"). Books present: `fanduel`,
`draftkings`, `caesars`, `betmgm`, `bet365`, plus `fanatics_sportsbook` from 2025
only.

---

## 2. Measurements that should drive the snapshot design

These were run against the live store specifically to choose the grid.

**Coverage — how far back does history reach?** Median earliest pre-game totals
tick is **1,200–1,600 minutes (20–27 h) before tip**, per game per book, in every
season. So a 12 h snapshot is comfortably inside the data; 24 h is not.

**Snapshot availability and staleness** (totals, 2023+, 19,339 game×book pairs):

| snapshot | game×book pairs with a line | median staleness | p90 staleness |
|---|---|---|---|
| 30 min | 19,339 (100%) | 32 min | 293 min |
| 60 min | 19,339 (100%) | 37 min | 330 min |
| 2 h | 19,338 (100%) | 96 min | 380 min |
| 3 h | 19,338 (100%) | 103 min | 363 min |
| 4 h | 19,336 (100%) | 91 min | 340 min |
| 6 h | 19,323 (100%) | 78 min | 305 min |
| 12 h | 19,247 (99.5%) | 88 min | 540 min |
| 24 h | 11,627 (**60%**) | 83 min | 375 min |

"Staleness" = how old the carried-forward line already is at the snapshot instant.

Three conclusions follow, and two of them change the proposal:

- **24 h is not viable** as a standard snapshot — 40% of game×book pairs have no
  line yet. Drop it, or carry it only with an explicit availability flag.
- **30 min and 60 min are near-duplicates.** Median staleness 32 vs 37 min means
  that for most games *the very same tick* is the answer at both snapshots. Two
  rows, one piece of information. Same objection applies to 3 h vs 4 h.
- **p90 staleness is ~5 h.** For a tenth of cases the "current" line is hours old.
  That is not a defect — it is genuine information (the market has not moved) —
  but the model must be told, hence a mandatory `LINE_AGE_MIN` feature (§6).

### Recommended grid

Roughly geometric — equal spacing in *log* time, which is how line-movement
information actually decays:

```
{30 min, 60 min, 2 h, 4 h, 8 h, 12 h}          → 6 snapshots
```

versus the proposed `{30m, 1h, 2h, 3h, 4h, 6h, 12h}`. This keeps 30m and 60m
(they bracket the most common betting window even if often equal), and drops the
redundant 3 h/4 h pairing in favour of an 8 h point that is currently unsampled.

At 6,128 games × 6 snapshots ≈ **36,800 rows** (≈22,000 for 2023+ alone).

**Superseded — the shipped grid is denser:**

```
{0, 30m, 1h, 2h, 3h, 4h, 5h, 6h, 8h, 12h}      → 10 snapshots
```

The parsimony argument above is sound for a grid that must be chosen once, but
it does not apply here: **snapshots are rows**. An unwanted horizon is removed
by filtering `TIME_TO_MATCH_MIN`, at no cost and with no rebuild, while adding
one back means regenerating the entire dataset. So the asymmetry favours
over-sampling, and the 3 h/5 h points this section dropped are cheap to carry
until measured importance says otherwise.

`0` is new in kind rather than in density: it is the **closing snapshot**, the
same bet the closing-line dataset makes. It is not literally tip-off —
`fetch_pregame_ticks` refuses anything inside its five-minute safety margin — so
it resolves to the last tick at least 5 minutes out. It is what makes the two
datasets directly comparable at the same decision point. Note that
closing-line value is ~0 by construction on those rows, so CLV measured over a
pooled dataset is diluted by them; compute it per snapshot.

A variant worth benchmarking later, but not for v1: sample at *actual tick times*
rather than a fixed grid, with `time_to_match` as a continuous feature. More
faithful, but it weights heavily-ticked games more and complicates production.

---

## 3. Row grain and dataset structure

One row per `(GAME_ID, TIME_TO_MATCH_MIN)`.

| block | contents | varies by snapshot? |
|---|---|---|
| **A. Keys** | `GAME_ID`, `GAME_DATE`, `SEASON_YEAR`, `TIPOFF_UTC`, `SNAPSHOT_TS_UTC`, `TIME_TO_MATCH_MIN`, team names/ids | only `TIME_TO_MATCH_MIN`, `SNAPSHOT_TS_UTC` |
| **B. Snapshot line state** | per-book line/prices at T, consensus, availability flags | **yes** |
| **C. Movement** | open→T, windowed moves, velocity, volatility, reversals, steam | **yes** |
| **D. Cross-book** | dispersion and per-book deviation at T | **yes** |
| **E. Base pre-game team features** | the existing `_BEFORE` rolling/team/travel features | no — constant within a game |
| **F. Targets** | `TOTAL_POINTS`, `SNAPSHOT_LINE_ERROR`, optional CLV target | `SNAPSHOT_LINE_ERROR` **yes** |

Block E being constant across a game's 6 rows is exactly why §7's grouped CV is
non-negotiable.

### Targets

- `TOTAL_POINTS` — unchanged, constant per game.
- `SNAPSHOT_LINE_ERROR = TOTAL_POINTS − TOTAL_LINE_SNAP_<book>`. **This is the
  one that matters** and it is *not* the existing `LINE_ERROR`: it is defined
  against the line you could actually have bet at T, not the closing line. It
  differs per snapshot for the same game.
- Optional auxiliary: `CLOSING_MOVE = closing_line − snapshot_line` — a
  "beat the closing line" (CLV) target. Legitimate **as a target**; catastrophic
  as a feature. Worth training as a separate head because CLV is a lower-variance
  signal than game outcome.

### How the target reaches `training_pipeline` without changing it

This needs care, and the obvious approach does not work. For
`prediction_strategy: line_error_regressor`, `experiments/_base.yaml` states
that **`line_col` must be omitted — the main book's line is used** — and
`_ensure_line_error_column`
([meta_learner_training_data.py:207](src/nba_ou/modeling/meta_learner_training_data.py#L207))
hardcodes `total_line_col()`, i.e. `TOTAL_LINE_bet365`. So the target is *not*
freely parameterisable, and left alone the pipeline would define the target
against the **closing** line while we intended the snapshot line.

Two facts make this solvable with zero pipeline changes:

1. `_ensure_line_error_column` returns early if a `LINE_ERROR` column already
   exists. Emitting `LINE_ERROR` precomputed against the snapshot line makes the
   pipeline adopt it verbatim.
2. Bets are still *settled* against the main book column, so that column must
   hold the same line the target was defined against.

**Therefore: in this dataset, `TOTAL_LINE_bet365` holds the bet365 line as of T,
and `LINE_ERROR` is precomputed against it.** This is not a hack — it is the
bet365 line, just quoted at T rather than at close, and T is the moment this
model bets. The true closing line is carried only under a distinct name
(`CLOSING_TOTAL_LINE_<book>`), used solely for optional CLV scoring, and must be
in the feature exclusion list. Document this at the top of the generator, because
a reader who assumes `TOTAL_LINE_bet365` means "closing" will misread everything
downstream.

---

## 4. Leakage register

The central risk, and the reason this plan exists. Each item is a concrete,
named thing to exclude or rebuild.

**L1 — The closing line itself.** `TOTAL_LINE_<book>`, `SPREAD_<book>`,
`MONEYLINE_<book>` as merged by `merge_total_spread_moneyline_by_game_id` are
close-time values for the current game. Excluded outright.

**L2 — `_BEFORE` is NOT a sufficient filter here.** This is the trap, and
implementation found it is worse than first documented. **Four** columns carry a
`_BEFORE` name while being computed from the current game's closing prices:

| column | source | severity |
|---|---|---|
| `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` | [global_market_features.py:554](src/nba_ou/data_processing/merged_home_away_data/global_market_features.py#L554), from lines 296–312 | dispersion of this game's closing lines |
| `THIS_GAME_CROSSBOOK_TOTAL_RANGE_BEFORE` | same | same |
| `IMPLIED_PTS_HOME_BEFORE` | [add_features_after_merging.py:349](src/nba_ou/data_processing/merged_home_away_data/add_features_after_merging.py#L349) | **critical** |
| `IMPLIED_PTS_AWAY_BEFORE` | same | **critical** |

The `IMPLIED_PTS_*` pair was not in the original plan and is the more dangerous
find. They are `line/2 ∓ spread/2`, so **they sum to the closing line exactly**:
measured over 2,626 games the reconstruction error is `0.0`. A model handed both
can add them and read off the number it is meant to be predicting.

Neither a name-based filter nor a single-column correlation screen catches this
— each column alone correlates only ~0.8 with the line, which is unremarkable
next to legitimate rolling features at ~0.68. It was found by measurement, which
is why `audit_closing_line_reconstruction` now runs on every build and searches
*pairs* among high-correlation candidates rather than trusting the list above to
stay complete.

The rolled-up cousins are safe and are kept: `GLOBAL_CROSSBOOK_TOTAL_STD_AVG_*G_BEFORE`
goes through `_rolling_game_agg`, which reads `values[start - window : start]` —
strictly earlier dates.

**L3 — the entire `odds_*` family.** `engineer_odds_features` is, per its own
docstring, "focused on **close-time** TOTAL/SPREAD/ML data"
([odds_feature_engeneer.py:80](src/nba_ou/data_processing/merged_home_away_data/odds_feature_engeneer.py#L80)).
Every `odds_`-prefixed column is close-time for the current game. All excluded;
the useful ones (consensus, dispersion, vig, no-vig probability) get **rebuilt at
snapshot time** in block D.

**L4 — in-play ticks.** SBR labels them `row_kind="history"` identically to
pre-game rows; 2.7–3.7% of recent-season rows land at or after tip. An in-play
totals line is conditioned on points already scored. Filter is
`is_pregame AND mins_to_tip <= -T` — and note the second clause is the real one:
`is_pregame` alone would let a 5-minutes-before-tip line into a 12 h snapshot.

**L5 — same-slate contamination.** A feature aggregated over *other games on the
same date* leaks: at 12 h before tip those games have not closed either. The
existing global-market features use `_ewm_game_mean_strict_prior_dates`
(strictly prior dates) and are safe on this axis — but the audit must be explicit
per feature, not assumed from the function name.

**L6 — injuries and referees.** Excluded by requirement. Independently correct:
neither has trustworthy timestamped history, so train/inference would disagree.

**L7 — Yahoo/consensus betting percentages.** `total_pct_bets_over`,
`total_consensus_pct_over` and siblings are scraped at or near close and carry no
timestamp. Exclude unless and until a timestamped source exists.

**L8 — line normalisation.** `normalize_total_lines=True` converts asymmetrically
priced totals to estimated 50/50 lines. Must be recomputed from **snapshot-time
prices only**, never inherited from the closing-line version.

**L9 — correlated rows across the split. Already handled, but only by
accident, so it must be pinned.** Six rows per game share all of block E, so a
random split would put the same game in train and validation. Verified: it
cannot happen with the existing splitters. `split_latest_days_holdout` cuts on
the date ([splits.py:19-49](training_pipeline/splits.py#L19-L49)), and
`make_test_anchored_walk_forward_splits` builds test windows from whole
`unique_dates` and takes the train pool as `_date < test_start_date`
([modeling.py:843-895](src/nba_ou/modeling/modeling.py#L843-L895)). Rows sharing
a `GAME_DATE` always land on the same side.

**The condition this rests on:** `data.date_col` must stay `GAME_DATE` — the
*game's* date. If anyone ever points it at the snapshot timestamp, a 12 h
snapshot falls on the previous calendar day and the guarantee silently
evaporates. Add a test asserting no `GAME_ID` appears on both sides of any split.

**Safe by construction:** the consensus **opener** (100% game coverage, median
~25 h before tip, p10 ~20.5 h — i.e. before every proposed snapshot), and all
historical closing-line-derived features from *previous* games, which were fully
known at T.

---

## 5. Architecture — three layers, zero edits to existing files

```
  Layer 1: base per-game features        Layer 2: snapshot line panel
  (existing modules, new composition)    (new, reads Aiven line_history)
  one row per game                       one row per (game, snapshot)
                    └──────── as-of join ────────┘
                                  ↓
                    Layer 3: assemble + strict column filter
                                  ↓
                 data/train_data/intermediate_line_<date>.csv
```

**Layer 1** re-uses the existing feature functions (`clean_team_data`,
`compute_all_rolling_statistics`, `merge_home_away_data`,
`compute_travel_features`, `add_style_matchup_features`, …) in a *new orchestrator*,
skipping the player/injury/referee/all-star stages. Note the subtlety: the team
pipeline merges closing odds *before* computing rolling stats, and rolling stats
over **prior games'** closing lines are legitimate and valuable. So we keep that
step and drop the current-game closing columns at the end, via a new stricter
filter rather than by pruning the pipeline.

**Layer 2** reads pre-game totals (and spread/ML) ticks per game×book, and for
each snapshot T does an as-of read plus the movement aggregations of §6.

**Layer 3** joins, derives targets, applies `select_intermediate_training_columns()`
— an **allowlist** that fails closed, mirroring how `select_training_columns`
raises on unexpected raw columns.

### Files to add (all new; no existing file modified)

```
src/nba_ou/postgre_db/line_history_aiven/fetch.py           read pre-game ticks
src/nba_ou/data_processing/line_history/__init__.py
src/nba_ou/data_processing/line_history/snapshots.py        as-of snapshot builder
src/nba_ou/data_processing/line_history/movement_features.py
src/nba_ou/data_processing/line_history/cross_book.py
src/nba_ou/create_training_data/create_intermediate_line_df.py     entry point
src/nba_ou/create_training_data/select_intermediate_columns.py     strict allowlist
scripts/create_train_data/create_intermediate_line_train_data.py   CLI
tests/test_line_history_snapshots.py
tests/test_intermediate_movement_features.py
tests/test_intermediate_column_leakage.py
```

`connect_line_history_db()` already exists in
[db_config.py:284](src/nba_ou/postgre_db/config/db_config.py#L284) — no config
change needed.

### Coexistence guarantees

- No edits to `create_df_to_predict.py` or any module it calls — new code
  *imports* existing functions, never alters them.
- Existing test suite must pass unchanged.
- Add a golden-schema test pinning `create_df_to_predict`'s output column set, so
  any accidental drift is caught rather than discovered in production.

---

## 6. Feature catalogue

Every item below is computable from ticks with `mins_to_tip <= -T`.

**Snapshot state** (per book `b`, and consensus): `TOTAL_LINE_SNAP_<b>`,
over/under prices, vig, no-vig implied probability, `SPREAD_SNAP_<b>`,
`MONEYLINE_SNAP_<b>`, `N_BOOKS_AVAILABLE_SNAP`, per-book availability flags, and
**`LINE_AGE_MIN_<b>`** — minutes since that book last moved. Given p90 staleness
of ~5 h this is one of the more informative columns in the set, not bookkeeping.

**Cross-book at T:** consensus median/mean, `CROSSBOOK_STD_SNAP`,
`CROSSBOOK_RANGE_SNAP`, and `BOOK_DEVIATION_<b>` = book line − consensus (an
outlying book is often the stale one, or the sharp one).

As built, the per-book `deviation_from_consensus`, `abs_deviation_from_consensus`,
`deviation_z` and `is_outlier_book` columns keep their names but use a
**leave-one-book-out** reference: the median of the *other* books, after
dropping peer quotes more than 10 points (totals/spread) or 0.15 probability
(moneyline) from the all-book median. The gap is then capped at that same limit,
and `deviation_z` divides by the peer std floored at 0.5 pt / 0.01. Including the
evaluated book in its own median shrank its deviation toward zero; one corrupt
feed value could inflate it by tens of points. A book with no quoted peers gets
0. The headline `consensus_line` (all books) is unchanged.

**Anchor-total path state** (bet365 totals only, to avoid dozens of redundant
per-book columns; built in `line_history/anchor_total_path.py`). All use raw
line *level* changes; a price-only reprice is not a move:

- `ODDS_SNAP_TOT_<ANCHOR>_MINUTES_SINCE_LAST_LEVEL_MOVE` — minutes since the
  level last changed (since open if it never did). Differs from
  `line_age_minutes`, which resets on any tick, including price-only ones.
- `ODDS_SNAP_TOT_<ANCHOR>_PEERS_MOVED_ANCHOR_STILL_60` — +1/−1 when the median of
  the other books (quoted both now and 60 min ago, at least two, outliers
  removed) rose/fell ≥ 0.5 pt in the last 60 min while the anchor level did
  not; 0 otherwise. Signed so it gives the direction the anchor would follow.
- `ODDS_SNAP_TOT_<ANCHOR>_ABS_LEVEL_PATH_60` — sum of absolute level changes in
  the last 60 min; separates a round trip from no activity, which share
  `move_last_60 = 0`.
- `ODDS_SNAP_TOT_<ANCHOR>_SIGNED_MOVE_STREAK_60` — consecutive moves in the
  latest direction within the last 60 min, signed (+ up, − down, 0 none).
- `ODDS_SNAP_TOT_<ANCHOR>_LAST_TWO_LEVEL_MOVES_GAP_MIN` — minutes between the
  last two level changes, capped at 720 (also 720 with fewer than two moves).

If `--windows` omits 60, the 60-minute peer lookback is resolved internally
without exporting a 60-minute window family.

**Movement from the opener:** `MOVE_FROM_OPEN` (signed), `ABS_MOVE_FROM_OPEN`,
`PCT_MOVE_FROM_OPEN`, `MOVE_DIRECTION` (sign), `MINUTES_SINCE_OPEN`.

**Windowed movement** for windows {1 h, 3 h, 6 h, 12 h}, each computed only where
history reaches back that far (else null + a `_HAS_` flag): `MOVE_LAST_<w>`,
`VELOCITY_<w>` (points/hour), `N_MOVES_<w>`, `LINE_STD_<w>`.

**Momentum / shape:** acceleration (`MOVE_LAST_1H − MOVE_LAST_3H/3`), total
`N_MOVES_SO_FAR`, `N_DISTINCT_LEVELS`, running `MAX`/`MIN`/`RANGE_SO_FAR`, and
`POSITION_IN_RANGE` (where the current line sits within its own realised range).

**Reversals:** count of direction changes, and whether current direction opposes
the opener→first-move direction — a reversal after a strong initial move is a
different market state from a steady drift of the same size.

**Steam proxy:** number and fraction of books that moved in the *same* direction
within the last 30/60 min. Cross-book agreement in a short window is the classic
sharp-money signature and is fully available at T.

**Price-only movement:** change in vig and in no-vig probability while the line
itself held. Books frequently price a half-tick before taking it — this captures
pressure that the line level misses entirely.

**Historical, closing-line-based (legitimate — prior games only):** all existing
`_BEFORE` rolling team features, plus new ones this dataset makes natural — each
team's historical average open→close move, and the historical *predictiveness* of
movement (does this team's line typically keep moving in the direction it started?).

### Movement counts

Two different quantities, and both are wanted:

- **At the snapshot** — `N_MOVES_SO_FAR` and `N_MOVES_<w>` above: how many times
  the line has changed between the opener and T. Bounded by what has happened so
  far, so it grows as T approaches tip and is partly a proxy for elapsed time —
  pair it with `N_MOVES_PER_HOUR` so the model can separate "busy market" from
  "more hours elapsed".
- **Over prior games' full open→close history** — `N_MOVES_OPEN_TO_CLOSE`, rolled
  up per team and league-wide over the usual `_BEFORE` windows. These count the
  *complete* movement history of games that have already finished, so they are
  fully known at T and carry no look-ahead. This is a genuinely new signal: a team
  whose lines are habitually re-priced many times is one the market is uncertain
  about, which is different information from the size of the average move.

Both are computed per market from the tick stream (`COUNT(*)` over ticks where
the line level actually changed — consecutive identical lines with only a price
change are *not* line moves and are counted separately as price-only ticks).

### Markets: totals, spread and moneyline all get the full treatment

`lh_line` carries all three markets (`market_id` 1 = totals, 2 = point_spread,
3 = money_line), and the snapshot, movement, cross-book, and movement-count
families above are built for **each** of them, not for totals alone. Spread and
moneyline movement is not decoration: a total moves with pace expectations while
a spread moves with strength expectations, and a game whose spread is moving hard
while its total holds is a different market state from one where both drift.

Per-market encoding notes from the store:

- **Totals** — a valid quote has `left_line == right_line` (the same number
  quoted both sides); `left` is over, `right` is under.
- **Spread** — mirrored, `left_line == -right_line`. Note the repaired
  price-bleed rows have NULL lines with valid prices, so spread line features
  must tolerate a present price with an absent line.
- **Moneyline** — prices only, no line. Movement is movement *in price*, so the
  natural feature is the change in devigged implied probability.

### Normalising prices to −110, as in the existing training

The current pipeline centres asymmetrically priced totals onto an estimated 50/50
line at −110/−110 via `normalize_total_lines_inplace`
([normalize_total_lines.py:169](src/nba_ou/data_processing/odds/normalize_total_lines.py#L169)),
controlled by `create_df_to_predict(normalize_total_lines=True)`. The snapshot
dataset does the same so that a line at T means the same thing as a line in the
existing dataset, and so that lines are comparable across books and across
snapshots rather than confounded with how each book happened to be pricing.

The maths generalises; the wrapper does not. `estimate_fair_line` (line ~110) is
market-agnostic given a `sigma`, but `_total_market_prefixes` discovers only
`total_*_line_over` column quartets. So per market:

- **Totals** — reuse directly, `sigma = DEFAULT_TOTAL_POINTS_SIGMA` (15.7).
- **Spread** — same estimator, but `sigma` is the *margin* distribution's, not
  the total's. **Calibrated, not assumed:** over 6,112 games from 2021-22 onward
  the residual of (home margin − closing spread) has std **13.46**, against a raw
  margin std of 15.57 and a totals sigma of 15.70. Using the totals sigma here
  would bias every centered spread. One concrete trap: `round_to_increment`
  rejects negative values
  ([line 67](src/nba_ou/data_processing/odds/normalize_total_lines.py#L67)), and
  spreads are signed — round the magnitude and reapply the sign, which also
  keeps the mirror symmetric (a fair ±3.25 must land on ∓3.5, not on −3.0/+3.5
  as half-up would give).
- **Moneyline** — there is no line to shift, so "normalisation" here is
  two-way devigging only, via `remove_vig_two_way`, yielding fair home/away
  probabilities. Do not invent a −110 line for it.

**A correctness trap worth stating loudly:** `lh_line.left_price`/`right_price`
are **American** odds, while `normalize_total_lines_inplace` defaults to
`odds_format="decimal"`. Passing American prices under the default silently
produces garbage rather than raising — `odds_to_decimal` only rejects decimal
odds `<= 1.0`, and American prices are mostly negative, so they *would* raise —
but positive American odds like `+105` pass the check as a "decimal" price and
convert wrongly. Pass `odds_format="american"` explicitly, and unit-test a
`+105`/`−125` pair against a hand-computed expectation.

Keep both the raw and the normalised line: the raw one is what you could actually
bet, and the normalised one is what is comparable. Only the normalised line should
feed cross-book dispersion.

---

## 7. `training_pipeline` integration — the 6× row-multiplier problem

Per `.claude/skills/experiments`, every config knob below was checked against the
code rather than assumed. **This section is the one most likely to produce a
silent, wrong result**, because a 6× denser dataset breaks the meaning of every
row-counted setting in `experiments/_base.yaml` without raising anything.

### Windows are counted in rows, not days

`walk_forward.train_games` is applied as `train_pool.tail(train_games)`
([modeling.py:894](src/nba_ou/modeling/modeling.py#L894)) and `test_games`
accumulates `date_counts` per date. Despite the name, these count **rows**. At 6
snapshots per game, the inherited defaults would silently cover one sixth of the
calendar history they were reasoned for:

| setting | base value | effective span on this dataset | rescaled |
|---|---|---|---|
| `walk_forward.train_games` | 2500 | ~417 real games | **15000** |
| `walk_forward.min_train_games` | 1250 | ~208 real games | **7500** |
| `walk_forward.test_games` | 50 | ~8 real games | **300** |
| `walk_forward.step_games_between_tests` | 60 | ~10 real games | **360** |
| `backtest.test_games` | 300 | ~50 real games | **1800** |
| `holdout.test_days` | 60 | 60 days — **correct as-is** | 60 |

`holdout.test_days` is the exception precisely because it is calendar-based, which
is why `_base.yaml` prefers it. Everything else must be multiplied by the snapshot
count. This is the documented `tail(n)` failure mode — folds shrink, no error.

The measured window ceilings in the skill (~3950 at 12 folds) are row counts, so
they scale by the same 6×; the total row budget scales identically (6,128 games ×
6 ≈ 36,800 rows). **The pre-flight is not optional here** — it exists to report
actual per-fold training size, which is exactly the thing at risk:

```bash
poetry run python scripts/preflight_campaign.py experiments/<campaign>
```

### `max_na_per_row: 80` will preferentially delete long-horizon snapshots

Windowed movement features are null by design where history does not reach back
far enough — and that is *systematically* the 8 h and 12 h snapshots. Under
`cleaning.max_na_per_row: 80` those rows carry the most NaNs and get dropped
first, quietly biasing the dataset toward short horizons and destroying the very
comparison the design exists to make.

Mitigation, in the generator rather than the config: emit an explicit
`HAS_<window>` flag plus a **sentinel-filled** value instead of a bare NaN, so a
missing 12 h window costs one informative column rather than N missing ones. Then
verify row retention per snapshot after cleaning — equal retention across
snapshots is the acceptance criterion. The skill notes `nan_threshold` is inert
on the 2.0 build and `max_na_per_row` is the real lever, so this is the knob that
will bite.

### Other config interactions, verified

- **`exclude_cols_containing: ["fanatics_sportsbook"]`** is already in
  `_base.yaml`, so the 2025-only book is handled — my earlier concern is moot.
  It is a substring match, so name snapshot columns accordingly.
- **`GAME_ID` must keep its leading zeros.** Season-type filtering resolves from
  the GAME_ID prefix, not the `SEASON_TYPE` text column. `lh_line.game_id` is
  `TEXT` in the DB, so the risk is purely CSV round-tripping —
  `load_raw_training_csv` forces ID columns to `str`, but the generator must not
  emit them as ints. "A filter matched zero rows" is a listed failure mode.
- **Carry `IS_OVERTIME`**, or `exclude_overtime_from_training` raises. Overtime
  is 5.2% of games at +21.2 points and 85.5% OVER, so it is worth being able to
  toggle.
- **`season_year_floor: 2021`** already matches the line-history store's coverage
  exactly. No change needed.
- **`comparison_line_cols`**: `TOTAL_LINE_consensus_opener` remains valid and is
  leakage-safe. Adding `CLOSING_TOTAL_LINE_<book>` here would measure CLV
  directly — but only if it is genuinely excluded from features, or you get the
  listed "spectacular ROI against another line" artefact.
- **Pin `data.expected_checksum` and set `data_version`** for the new CSV.

### Statistical power — per-snapshot evaluation is underpowered

I earlier suggested evaluating per snapshot. That is directionally right but the
volume math says it cannot carry a conclusion on the holdout: 60 days is ~417
games and ~166 bets, so slicing by 6 snapshots leaves **~28 bets each**. At −110
a Wilson interval around a true 55% win rate fails to clear break-even at 114,
600 and even 1200 bets. So:

- Read per-snapshot results from **pooled CV folds** (~5× the volume), never the
  holdout alone, and treat them as directional only.
- Report seed range first — same-config seed ROI spans **4.9–12.0 points**, so
  nothing smaller than that is a result.
- The one per-snapshot check that *is* worth trusting is qualitative and is the
  main leakage assay: **accuracy should degrade monotonically as
  `TIME_TO_MATCH_MIN` grows.** If a 12 h model matches a 30 min model, something
  leaked.

### Grain notes

- **Sample weighting**: each game contributes 6× its natural weight. Either weight
  rows `1/n_snapshots` or treat the snapshot as a conditioning variable and accept
  it. Worth an A/B — but only as a single deliberate difference per cell.
- **`TIME_TO_MATCH_MIN` as a feature**, so one model serves all bet times and at
  inference you pass your actual minutes-to-tip. This is the whole point.
- Betting scorers in `scorers.py` apply unchanged, but `primary_edge_threshold`
  is now measured against the **snapshot** line. Base is 0.1 because measured win
  rate does not rise with predicted edge — do not re-impose a wide filter without
  re-measuring it on this dataset.

---

## 8. Suggested build order

1. `fetch.py` + snapshot as-of builder **for all three markets** + tests
   (halve-the-`SMALLINT` handling, in-play filter, staleness, the totals /
   mirrored-spread / price-only-moneyline encodings). Verify against a handful of
   hand-checked games.
2. Price normalisation to −110: totals via the existing estimator, spread with a
   calibrated margin `sigma` and signed-value handling, moneyline as devig only.
   Test the American-vs-decimal `odds_format` trap explicitly with a `+105` price.
3. Movement, movement-count and cross-book features + tests, per market.
4. Layer 1 base-feature orchestrator + the golden-schema test on
   `create_df_to_predict`.
5. Strict allowlist filter + a leakage test asserting no `odds_*`, no
   `THIS_GAME_CROSSBOOK_*`, no bare closing `TOTAL_LINE_<book>` survives.
6. Assembly + CLI, emit the CSV (with `LINE_ERROR` precomputed against the
   snapshot line, `GAME_ID` zero-padded, `IS_OVERTIME` carried,
   `exclude_fanatics=True`).
7. **Campaign scaffolding before any GPU time**: a new `experiments/<campaign>/`
   with every row-counted window rescaled by the snapshot multiplier (§7),
   `expected_checksum` pinned, `evaluation_seeds` set, `n_trials` fixed and
   `timeout` disabled. Then run `scripts/preflight_campaign.py` and read the
   **actual per-fold training size** — that number, not arithmetic, is what
   confirms the rescale worked.
8. Verify row retention per snapshot after cleaning is even across snapshots
   (the `max_na_per_row` trap), before believing any model result.
9. Sanity model. First real check: does accuracy degrade monotonically with
   `TIME_TO_MATCH_MIN`? If not, something leaked.

Serving today's games is **not** in this build order — see §9.

Per the skill's guidance, mutation-test each leakage test: revert the guard,
confirm the test fails, restore. A leakage test that passes both ways is worse
than none.

---

## 9. Open questions and risks

**Scope: historical training data only.** This work builds the historical
dataset and trains against it. **Serving predictions for today's games is
explicitly out of scope for now** and will be added later. Nothing here should
grow a same-day prediction path, and `create_df_to_predict`'s
`todays_prediction=True` branch is not touched or mirrored.

**The production inference gap — deferred, not solved.** There is no live
line-history fetcher in this repo: the daily scrape gets the current grid and the
consensus opener, but not the intra-day tick history. Under the training-only
scope this blocks nothing today, so the full windowed feature set can be built
now. But it is the reason the *later* serving work is not just "call the model":

- Opener→now movement, cross-book dispersion and line age will be computable
  live from a single scrape.
- Anything windowed ("moved 2 points in the last hour") will need either a live
  line-history scrape at bet time, or our own timestamped scrapes accumulated
  from now on.

**One thing worth doing early despite the deferral:** start persisting timestamped
daily odds scrapes now. Tick history cannot be reconstructed retroactively, so
every day this is postponed is a day of live-comparable history permanently lost.
It is cheap, independent of everything else in this plan, and it is what makes
the eventual serving path cheap instead of blocked.

When serving is picked up, expect the model to need a declared feature subset it
can actually compute live — the honest check will be whether the live-computable
subset alone still beats the snapshot line.

**Training window is 2021-2025** (~6,128 games) because 2019-20/2020-21 were held
back for timezone confidence. Recovering them is possible but is separate work.

**Which book to anchor the target on.** `bet365` is the configured main book, but
it has the *fewest* pre-game totals ticks of the five long-running books (73,796
vs FanDuel's 134,481). A consensus-median anchor may be both more stable and
better sampled — but it is not a line you can actually bet. Worth deciding
explicitly rather than defaulting.

**`fanatics_sportsbook` exists only from 2025** and must not become an implicit
season indicator — a book that is present exactly when `season_year == 2025` lets
a model identify the season from book availability alone.

Excluded, but **behind an explicit generator flag rather than silently**:

```python
create_intermediate_line_df(..., exclude_fanatics: bool = True)
```

Default `True`, so the safe behaviour is what you get by not thinking about it,
and the book can still be switched on deliberately for a 2025-only experiment.
Two reasons the flag beats relying on `cleaning.exclude_cols_containing:
["fanatics_sportsbook"]` in `_base.yaml`, even though that is already there:

1. It keeps the columns out of the CSV entirely, so they cannot inflate
   `max_na_per_row` counts for pre-2025 rows and drag those rows over the
   cleaning threshold (§7).
2. `exclude_cols_containing` is a substring match applied at read time and is
   listed in the skill's failure table as a knob that silently preempts others.
   Depending on it alone means the exclusion is invisible from the dataset.

When the flag *is* set to `False`, the book must be availability-flagged
(`HAS_FANATICS_SNAP`) rather than left as bare NaN, for the same reason as the
windowed features.

**How many snapshots is the right multiplier?** Six is a judgement call, and it
is also the constant that every window in §7 must be rescaled by. Fewer snapshots
means less row correlation and less config surgery; more means better coverage of
betting times. If the first campaign shows accuracy is flat across
`TIME_TO_MATCH_MIN`, the grid can be thinned without redesigning anything.

---

## 10. Defects found in review after the first build

Six issues, found by an independent review of the shipped code and each
verified against real data before being fixed. Recorded because most of them
were *silent* — the dataset built cleanly and looked reasonable throughout.

### 10.1 Closing lines reached the feature matrix (critical)

Closing lines were kept in the training frame behind a `CLOSING_` prefix, on the
assumption that the `feature_columns()` helper would filter them. **It does
not.** `training_pipeline.data.build_feature_matrix` builds `X` by dropping only
the *configured* exclusions ([data.py:341](training_pipeline/data.py#L341)) and
never consults that helper, so all seven closing columns would have entered `X`.
Any result from the first build would have been contaminated by the outcome the
model is meant to predict.

This is the same class of mistake as the `assert_no_bare_closing_odds` no-op
found earlier: **a guard that lives outside the path it is guarding.**

Fixed by physical separation. `create_intermediate_line_df(return_scoring=True)`
returns `(training, scoring)`, the CLI writes two files, and `is_kept_column`
now *rejects* `CLOSING_*` rather than admitting it. Absence from the file is the
only guarantee that cannot be forgotten at a call site.

### 10.2 Spread normalisation was directionally inverted (high)

`center_two_way_line` adds `sigma · Φ⁻¹(fair_left)`, which is correct when the
left side wins *above* the line — true for totals (OVER), false for spreads. The
left side is the AWAY team and the resolved line is the expected HOME margin, so
the away side covers when the margin lands *below* it.

A quote of away +4.5 at −130 / home −4.5 at +110 was centered to **6.0** where
**~3.0** is correct: wrong by three points and in the wrong direction, on
roughly 29% of spread snapshots. Symmetric −110/−110 quotes hide it entirely,
and the original test used exactly that case. Fixed with an explicit
`left_wins_above` flag, plus asymmetric-price and mirror-symmetry tests.

### 10.3 The moneyline market was inert (high)

`resolve_line` returns NaN for the moneyline by design, so `line_delta` was NaN
on every moneyline row. Everything derived from it silently became zero:
`n_moves_so_far`, `move_from_open`, reversals, window flags, `n_books_quoting`
and cross-book dispersion. Only the probability-movement family worked.

Fixed by introducing `snapshots.market_level`, the one quantity each market
*moves in*: the raw line for totals and spread, the devigged home win
probability for the moneyline. Every movement, dispersion and path feature is
now computed from it.

### 10.4 Movement features mixed raw and normalised scales (high)

The opener, running min/max and path std came from raw lines while the current
value came from the centered line. Measuring a centered "now" against a raw
"then" put **946 spread rows outside their own realised range**
(`position_in_range` beyond [0,1]). Fixed by computing the whole family from
`level`. The pricing correction is not lost — it is carried explicitly as
`norm_minus_raw`, the half-tick a book has priced but not yet taken.

### 10.5 The consensus opener was destroyed (medium)

The closing-line rename swept up `TOTAL_LINE_consensus_opener`, so the
`betting.comparison_line_cols` baseline configured in
[`experiments/_base.yaml`](experiments/_base.yaml) silently matched nothing. The
opener is *safe* at every snapshot — openers land a median ~25h before tip,
outside the whole grid — so it should never have been treated as a closing line.
It now survives under its own name and is asserted by test.

### 10.6 Non-positive look-back windows were accepted (medium)

`windows` was unvalidated. A negative window reads the panel at
`snapshot − |w|`, i.e. a *later* moment than the snapshot: a direct look-ahead
that no column-name check downstream would catch. Now rejected.

### Also fixed

Prior-game line dynamics ran on totals only and now cover all three markets;
`exclude_fanatics` now also strips the base-data Fanatics closing column;
`normalize_total_lines` is plumbed through to the snapshot panel instead of
being hardcoded; raw side prices and `deviation_z` are exported; the
`anchor_book="consensus"` path is removed rather than left crashing.

---

## 11. Two things the dataset does NOT give you

Both are deliberate, and both will mislead if forgotten.

### Reported ROI is not executable

The target is defined against the **normalised** (−110-equivalent) line, not the
number on the board. That is the right *modelling* choice — it is comparable
across books and snapshots instead of confounded with each book's pricing — but
it is not a price anyone can take. On the default bet365 anchor the two differ
on only a handful of rows, because its totals are almost always −110/−110; on
other books the gap is material. Raw lines and both side prices are exported as
features so the difference stays visible, and bets settle at flat −110.

**So: treat ROI from this dataset as a ranking signal, not a profit forecast.**

### Six snapshots are not six independent observations

Date-based splitting keeps a game whole (asserted by test), but everything
counted in *rows* still treats the snapshots as independent. Measured on this
data, adjacent snapshots resolve to the **identical tick**:

| pair | identical |
|---|---|
| 30m vs 60m | 65.7% |
| 60m vs 120m | 56.3% |
| 120m vs 240m | 64.5% |
| 240m vs 480m | 39.2% |
| 480m vs 720m | 27.2% |

The decay with separation is the geometric grid working as designed, but the
short pairs carry little independent information. Consequences:

- `n_unique_games` in CV betting counts **row positions**, not games
  ([cv_betting.py:304](training_pipeline/cv_betting.py#L304)), so it reports up
  to 6× the real figure.
- Any Wilson interval computed on bet counts is correspondingly too narrow.

Handled dataset-side only, deliberately leaving `training_pipeline` untouched so
the closing-line model is unaffected:

- A `SNAPSHOT_WEIGHT` column was tried here and removed. It was emitted by
  the builder and read by nothing, and a correction nobody applies is worse
  than an acknowledged limitation: it reads as though the problem is handled.
  The correction that shipped is `training_pipeline.snapshot_scoring`, which
  regroups finished predictions one horizon at a time.
- `GAME_ID` and `TIME_TO_MATCH_MIN` stay in the training frame so predictions
  can be regrouped by game.

**Declare a one-bet-per-game policy when evaluating.** Pick one snapshot per
game (or the model's most confident), score that, and read the resulting bet
count as the real sample size. If six bets per game are genuinely intended, they
are a correlated portfolio position, not six independent wagers.
