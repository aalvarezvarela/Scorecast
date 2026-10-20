---
name: feature-engineering
description: Feature-engineering philosophy, temporal-correctness rules and the proven feature families for a sports-betting model - rolling team form, team-vs-team matchup, schedule/rest/travel, injuries and availability, roster continuity, market/line-movement, officials, and league-wide market regime. Use when adding or changing any model feature, designing the feature set for a new sport, auditing a feature for look-ahead leakage, or deciding what history a feature needs before it is trustworthy.
---

# Feature engineering for sports betting

Distilled from an NBA totals/spread system with ~2,000–3,200 engineered columns
across two datasets. Companions: `sports-data-architecture` (entities and
storage), `odds-data-architecture` (market ingestion).

**Per-family detail — what each family computes, why it may be predictive, how
temporal correctness is maintained, and the MLB translation — is in
[references/feature-families.md](references/feature-families.md). Read this file
for the rules; read that one when building a specific family.**

---

## The three rules

Everything below is downstream of these.

### 1. A feature's name declares its temporal status, and the name is enforced

| suffix / prefix | meaning |
|---|---|
| `_BEFORE` | leakage-safe, computable strictly before the game |
| `_TEAM_HOME` / `_TEAM_AWAY` | post-merge side |
| `_DIFF_BEFORE` | home minus away |
| `ODDS_` | derived from bookmaker market data |

Final column selection keeps a column **only if** it contains `_BEFORE`, is on a
short explicit allowlist (ids, dates, season, team info), or is odds-shaped. It
then **raises** if any raw, non-`_BEFORE` source column survived.

Two more guards sit behind it: `ODDS_` is asserted as an invariant (a column
*named* like a market that arrives without the prefix raises, because it would be
invisible to every consumer selecting on the prefix), and the training pipeline
independently drops and then **asserts absent** a hard list of outcome-derived
columns before the feature matrix is built.

**Why belt and braces.** `exclude_cols` is a config field a caller can overwrite,
and the columns only appear in *some* dataset snapshots — exactly the combination
that yields a silent, spectacular-looking result rather than an error.

**Reusable principle.** Encode the temporal contract in the **column name**, then
enforce it at a **gate every dataset passes through**, then assert it again at
the feature matrix. Naming conventions that are not machine-checked are decoration.

### 2. Every historical aggregate is `shift(1)` first, then aggregate

Never `rolling(...).mean()` on a series that includes the current row.

```python
series.groupby(keys).transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
```

Where a `shift` would cross a group boundary, avoid it. Pre-game win/loss record
uses **cumulative sum minus the current row's own result** rather than
`groupby().cumsum().shift(1)` — the latter leaks the previous group's final value
into the first row of the next group. That is a real bug class, not a
hypothetical.

### 3. Missing history gets a documented fallback chain, never a bare NaN

Season openers are the problem: a rolling window resets, `min_periods` is not
met, and ~1,100 columns go NaN — enough for a row-level NaN limit to discard the
start of every season. So the model never sees a season begin.

The chain, used identically for means, standard deviations, trends and player
averages:

```
this season to date  →  the same quantity from the previous REGULAR season  →  a defined neutral value
```

Three details that make it correct:

- **Previous *regular* season only.** A playoff run is a different competitive
  regime and only some teams have one, so sourcing "last season's average" from
  whatever a team happened to play last makes it mean something different for a
  finalist than for a lottery team.
- **Keyed the same way as the column being filled.** A home/away split falls back
  to the previous season's home/away value, not its overall one.
- **The neutral value equals the estimator's own limit.** For a shrunk-toward-zero
  effect, "no evidence" is 0 — the same thing one weak observation gives. Leaving
  NaN made the feature discontinuous at exactly the point the shrinkage was
  designed to smooth.

No leakage: the previous season is complete before the current one starts. *This
assumption is sport-specific — check it for any sport with overlapping
seasons, winter leagues or mid-year competitions.*

---

## The philosophy

**Compute at the accumulation grain, pivot at the end.** Team-level history is
built on team-game rows where a rolling feature is one `groupby`. The
home/away wide form is produced once, by one merge, and interaction features are
built after it.

**Rates, not counts.** Store and roll `FG3A / FGA`, `TOV / POSS`, `points per 100
possessions` — scale-free quantities that can be recombined against a specific
opponent. Raw counts are inputs to rates, not features.

**Volume × efficiency.** A totals model is
`expected_events × expected_value_per_event`. Keep the two factors separate so a
matchup can be built by crossing one team's offensive rate with the other's
allowed rate, then scaling by expected tempo.

**Differences and interactions are worth their own columns.** Trees can in
principle learn `home − away`, but they must spend depth on it. ~200 explicit
`_DIFF_BEFORE` columns are generated mechanically for every betting-related
rolling statistic.

**Redundancy is pruned automatically, with a market-aware threshold.** Identical
columns are grouped and one representative kept; correlated columns are dropped
at |r| > 0.95, **except** odds-derived columns at 0.99. Measured on 1,578 numeric
columns: a single 0.995 threshold dropped 104 columns of which 83 were
odds-derived — the block the problem is actually about. The split threshold drops
212 while keeping the market features.

**Over-generate, then filter.** Dozens of rolling variants per statistic are
produced and a downstream limit prunes them, on the reasoning that adding a
variant back means regenerating the entire dataset while removing one is a
filter. Know that this is the chosen tradeoff — the cost is build time and a
correlated feature space.

**Every feature carries its own missingness story.** Structural absence gets an
explicit flag (`has_quote`, `has_window_60`) so "no data" is a value the model
reads rather than a NaN a cleaning pass may act on.

---

## The families

Detail for each is in [references/feature-families.md](references/feature-families.md).

| # | Family | Core idea | MLB status |
|---|---|---|---|
| 1 | **Rolling team performance** | multi-window means, WMA, expanding season mean/std, regression-slope trends, home/away splits as *deltas* | direct |
| 2 | **Team-vs-team matchup** | offence × opposing defence, expected tempo, style-rate crossing, head-to-head history | direct, plus handedness splits |
| 3 | **Schedule, rest, travel** | rest days, back-to-backs, rolling km over 5 windows, timezone change, travel compression ratio | reframe: series, not single games |
| 4 | **Injuries and availability** | who is out, weighted by an empirical-Bayes-shrunk per-player effect on outcomes | **highest-value and highest-risk** |
| 5 | **Roster continuity** | minutes-weighted share of the roster retained over two horizons, plus incoming and net | direct, plus in-season acquisitions |
| 6 | **Market / line movement** | snapshot as-of T, movement over trailing windows, cross-book dispersion, steam | direct; see odds skill |
| 7 | **Officials** | per-official with-minus-without delta on outcome, foul rate and market error | **umpire — stronger in MLB** |
| 8 | **External importance** | vote-share proxy for star quality, independent of own box scores | substitute projections/WAR |
| 9 | **League-wide market regime** | rolling market bias, MAE, tail-miss rate, over/under rate at league level | direct |
| 10 | **Calendar and context** | weekend, month, holiday, conference/division, playoff proximity | direct, plus day-game-after-night |
| 11 | **Prior-game line dynamics** | how much the market re-priced this team's *previous* games | direct |

---

## Leakage: the audit

### Safeguards that exist

| Guard | Where | Catches |
|---|---|---|
| `_BEFORE` name + raise on raw survivors | final column selection | a raw box-score column reaching training |
| `ODDS_` prefix invariant, asserted | pipeline exit | an unprefixed market column becoming invisible to filters |
| hard outcome list, dropped **and** asserted | feature-matrix construction | `TOTAL_POINTS`, `LINE_ERROR`, `IS_OVERTIME`, per-team finals |
| `mins_to_tip` / `is_pregame` NOT NULL | tick store schema | in-play market observations |
| negative horizon / non-positive window raise | snapshot + movement builders | reading the market after the snapshot |
| closing lines in a **separate file** | intermediate dataset | closing line entering X |
| `shift(1)` before every aggregation | statistics helpers | the current game in its own history |
| cumsum-minus-self for records | pre-game record | group-boundary bleed |
| opponent value only for valid 2-team games | style sources | a scheduled game's blank box score entering history |
| `find_season_gated_columns` | cleaning | a column whose *availability* identifies the season |

### Known risks and open assumptions

Ordered most to least concerning. Each is a thing to handle **differently**
when building for a new sport.

1. **Historical availability comes from the settled post-game inactive list;
   production uses the pre-game report.** Not strictly look-ahead — the inactive
   list is determined before tip — but the training signal is *better* than the
   serving one, so backtests are optimistic and CV cannot see it. Additionally,
   players are added to the historical injured set by matching `injur` in the
   box score's `COMMENT` field, which is written after the game.
   → **Archive the pre-game report daily from day one.** For MLB this is more
   than optimism: **IL stints are backdated**, so an after-the-fact query returns
   a player as "on the IL" on a date when nobody knew. That is genuine
   look-ahead.
2. **Roster membership at a season opener is read from the current game's own box
   score.** Documented and argued safe (who dresses is known pre-tip; every
   attached value is a prior-season average), but it *is* reading the target
   game's record. Same for selecting active players by `GAME_DATE == date`.
   → Prefer an archived roster feed.
3. **Unmatched injury-report players are logged, not raised** (the `raise` is
   commented out). A star whose name fails to resolve silently vanishes from that
   day's features. → Make it fatal.
4. **The referee previous-season fallback does not filter `GAME_DATE < current`**
   — it takes the whole prior season, relying on that season being complete.
   True for NBA. → Verify for any sport with overlapping competitions.
5. **Rolling windows cross the offseason by default** (`group_by_season=False`),
   so "last 5 games" at a season's start includes last season's playoffs.
   Deliberate — it fills the window — but it means early-season form is partly
   last year's team. Trend *slopes* are grouped per season precisely because a
   slope spanning an offseason is not a trend.
6. **Availability features are shrunk toward zero and then zero-filled.** Fine,
   but "no effect" and "no evidence" are the same number in the output. The
   sample-size columns are the only way to tell them apart — keep them.
7. **`normalize_total_lines` restates a quote at −110/−110.** Better modelling
   target, but ROI computed against it is **not executable**. Report it as such.
8. **Snapshot lines can be hours stale** (p90 ~5h). Real, not a defect — which is
   why `line_age_minutes` is a feature — but do not read a snapshot as "the
   market at T" without it.
9. **Book availability by season.** A book present only in recent seasons lets a
   model recover the year from column availability. Handled by merging the
   discontinued book into its successor. → Audit availability-by-season for every
   column, not just odds.

### The audit question

For every feature, answer in one sentence:

> **At the moment I would place this bet, what is the exact source that carried
> this value, and was it published?**

"It could have been computed" is not enough. "It was in the report I archived at
09:00" is.

---

## Missing-data policy

Not one imputation strategy — five, assigned by column category:

| Category | Policy | Why |
|---|---|---|
| Required market columns (main spread, both moneylines) | **drop the row** | missing means a pipeline failure, not a fact |
| Market / book / public-betting columns | **keep NaN + `__is_missing` flag** | a book not quoting is information; trees handle NaN |
| Injury / availability effects | **zero-fill, no flag** | 0 is the estimator's own "no evidence" value |
| Rolling-window features | **infer from the season average** of the same quantity | a defined weaker estimate of the same thing |
| Everything else | training-set median, optional | last resort |

Two operational rules: **the same policy must run at training and prediction
time**, and for cross-validation **medians must be computed within each fold**.

Row-level and column-level NaN limits are separate levers, and which one bites
was measured: the column threshold turned out **inert** on one build (25/50/80/100
all kept 1,890 columns) while the row limit was the real lever (80 → 200 recovered
~390 rows). Measure yours.

---

## Building the feature set for a new sport

1. **Team-game rows first.** Everything else is downstream.
2. **Adopt `_BEFORE` (or equivalent) and enforce it with a raising gate** before
   writing the second feature.
3. **Family 1 (rolling form) on your sport's volume and efficiency rates.** Build
   the fallback chain immediately — not later, when season openers turn out to
   be missing.
4. **Family 3 (schedule/rest/travel).** Cheap, no extra data source.
5. **Family 6 (market).** The line is the strongest single feature; get it in
   early to calibrate what "beating the market" even looks like.
6. **Family 2 (matchup).** Needs 1 to exist.
7. **Family 4 (availability)** — highest value, most care. Do not start until you
   have an archived point-in-time source, or you will build a backtest you cannot
   reproduce live.
8. **Families 7–11** as data allows.
9. **Prune redundancy** with a market-aware threshold once the space is large.
10. **Audit** every family against the question above before trusting a result.
