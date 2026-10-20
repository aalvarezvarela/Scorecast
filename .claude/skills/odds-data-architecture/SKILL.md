---
name: odds-data-architecture
description: How betting-odds data is acquired from SportsbookReview (SBR), stored, backfilled and kept current, and how it is shaped so it can later become model features. Use when building or changing odds ingestion for any sport, designing an odds/market schema, backfilling historical lines, adding line-movement or snapshot data, or deciding what a "market observation" record should contain.
---

# Odds data architecture

Transferable design decisions from an NBA totals/spread/moneyline system built
on SportsbookReview. Written to be reused for another sport (MLB is the
motivating case). Sport-specific parts are marked; the underlying pattern is
stated separately each time.

The one sentence to carry over:

> An odds record is **(game, sportsbook, market, side, line, price, timestamp)**.
> Everything else — books as columns, closing lines, opening lines, consensus —
> is a *view* over that record. Build the tick first; derive the views.

## Two stores, one source, different grains

SBR is scraped twice, into two stores that answer different questions. Do not
collapse them; they have different failure modes, different refresh needs, and
different leakage surfaces.

| | **Wide game table** | **Tick store** |
|---|---|---|
| Grain | one row per game | one row per (game, market, book, minute) |
| Question | "what was the line?" | "what was the line *at T*, and how did it get there?" |
| Source | rendered HTML of the daily odds pages | `__NEXT_DATA__` JSON on the line-history page |
| Transport | Playwright (headless browser) | plain `requests.get` |
| Requests | 3 per **day** (totals, spread, moneyline) | 1 per **game** (all books × all 3 markets) |
| Where | `odds_sportsbook` schema, main Postgres | `line_history` schema, separate Postgres |
| Rows | ~1 per game | ~1.9M and growing |
| Feeds | the closing-line dataset | the intermediate/snapshot dataset |

**Reusable principle.** A per-day wide scrape is cheap and gets you a training
set immediately; a per-game tick scrape is ~1,300 requests per season and gets
you everything the wide table can never reconstruct (movement, timing,
cross-book disagreement at a horizon). Build the wide one first to unblock
modelling, then the tick store — but design the wide table knowing it will be
superseded, and do not let feature code depend on its column shape.

**MLB adaptation.** Same two-tier structure. Note the daily volume difference:
MLB is ~15 games/day over ~186 days vs NBA's ~11 over ~170, so the per-game
tick scrape is roughly 2.5× the request budget per season. Budget for it, and
lean harder on the "only fetch what is missing" planner below.

---

## Historical acquisition

### Decision: parse the page's embedded JSON, not the rendered DOM

**Reasoning.** The rendered line-history table is drawn client-side in the
*browser's local timezone* and carries no offset. That is a silent,
machine-dependent bug: the same scraper produces different timestamps on a
Madrid laptop and a UTC CI runner, and nothing errors. This actually happened
here — `Europe/Madrid` had to be recovered *after the fact* from daylight-saving
steps in the data (see `docs/line_history_phase0_findings.md`), and two COVID-era
seasons could never be pinned confidently and remain excluded from the store.

The same page ships a Next.js `__NEXT_DATA__` payload where `oddsDate` carries
an explicit UTC offset, and which also contains the game's own `startDate`.

**Current implementation.** `fetch_data/odds_sportsbook/scrape_sportsbook_line_history.py`
regex-extracts `<script id="__NEXT_DATA__">`, `json.loads` it, and walks
`props.pageProps.lineHistoryModel.lineHistory.{gameView, oddsViews}`.

Three consequences, all wins:

1. **Server-rendered → no browser.** No Playwright, no cookie-consent clicking,
   no per-book/per-market tab clicking. One HTTP GET returns every sportsbook
   across totals, spread and moneyline.
2. **Timezone question never arises.** Parsing refuses a naive datetime outright
   rather than assuming one — a naive value would reintroduce exactly the
   ambiguity the module exists to remove.
3. **Tipoff comes from the same payload as the ticks**, so `minutes_to_tip` is
   internally consistent even if an external schedule feed disagrees.

**Reusable principle.** Before writing a DOM scraper, check for an embedded
state blob (`__NEXT_DATA__`, `__NUXT__`, `window.__INITIAL_STATE__`) or an XHR
the page itself calls. It is almost always more complete, more stable across
redesigns, and carries explicit types the DOM has already lossily formatted
away. **A rendered timestamp with no offset is a bug, not data.**

### Decision: truncate tick timestamps to the minute

SBR polls books on a fixed cadence and stamps every tick with the same seconds
value (`:20` at time of writing), so seconds carry zero information. Dropping
them makes a re-scrape reproduce byte-identical keys, which is what makes loads
idempotent. Within a minute, the **latest** quote wins, so truncation can never
produce two rows on one key.

**Reusable principle.** Round the timestamp to the coarsest unit that loses no
information, then make it part of the primary key. Idempotent re-fetch is worth
more than the discarded precision.

### Getting the complete history from scratch

The order matters, because each step is the reference for the next:

1. **Games first.** Ingest the sport's own game table (stable `game_id`, date,
   home/away). Odds have no useful identity until they can be joined to it.
2. **Tipoffs.** One request per season from the league schedule feed. Do this
   before any tick work — `minutes_to_tip` is the leakage filter and cannot be
   backfilled cheaply.
3. **Wide odds, day by day.** Iterate dates over the seasons you want; skip a
   date the moment the page says "No odds available at this time for this
   league" (an explicit no-slate signal, cheaper and safer than inferring from
   an empty table).
4. **Ticks, day by day.** For each date, one request to the day's odds page to
   *discover* the event ids on the slate, then one request per event. Do not
   guess event ids.
5. **Resolve, encode, insert.** Below.

Both scrapers sleep a uniform random 0.4–1.1 s between requests, and the tick
fetcher retries 3× with linear backoff (`2.0 * attempt`) on any
`RequestException` or missing payload.

**Reusable principle.** Discovery and detail are separate requests. A slate
listing tells you what exists; never construct detail URLs from a guessed id
range.

### Decision: a scrape failure on one date must not end the backfill

`scrape_events(on_error="warn")` prints and continues; `"raise"` is the strict
mode for tests and small targeted runs. The per-date loop in the updater wraps
each date in `try/except` and records `failed_dates` on the result object.

**Reasoning.** A 6-season backfill that dies on hour four because one game
404'd is worse than one with a reported gap — and the gap-finder below will
pick that game up on the next run anyway. Make partial success a first-class
outcome that the caller can read, not an exception.

---

## Identity: the hard part

Every odds row arrives keyed by the *provider's* event id, which means nothing
to the rest of the system. Resolution is:

```
SBR event_id
  → (game_date, team_home, team_away)     # from the payload
  → standardise both team names            # TEAM_NAME_STANDARDIZATION
  → look up internal game_id               # from the games table
```

Four decisions inside that:

**`game_date` is the Eastern-time date of tipoff.** A 00:30 UTC tipoff belongs
to the previous day's slate. SBR uses this convention and so does the games
table — which is *the only reason they join at all*. Get this wrong and roughly
the late third of every slate silently fails to match.
*MLB: same idea, same `America/New_York` convention for MLB's own game dates.*

**Team names are standardised through one hand-maintained map that raises on an
unknown name.** `constants.TEAM_NAME_STANDARDIZATION`. A relocation, a rebrand
or a new provider spelling should stop the pipeline, not silently drop a game.
The line-history loader tries several capitalisation/`LA`→`L.A.` variants
before giving up.

**An unmatched game is diagnosed, not dropped blindly.** Here, essentially all
misses are preseason, which the games table does not carry. That is a *known,
named* reason recorded in `IngestStats.unmatched_games`. If your miss rate has
no such explanation, you have a name-mapping bug.

**Keep the provider's id.** `lh_game.event_id` stores the SBR id alongside the
internal `game_id`. It cannot be re-derived later — a date holds many games —
and you will need it to re-fetch one game.

**Reusable principle.** Provider id ↔ internal id mapping is a *stored
dimension row*, written at the moment both are on the same record. Never a
join you plan to redo.

**MLB adaptation.** Harder, in three specific ways:
- **Doubleheaders.** `(date, home, away)` is not unique. MLB game ids carry a
  game number; your join key must include it, and the odds provider must expose
  something (start time, "Game 2" label) to disambiguate. **Design the key as
  `(date, home, away, game_number)` from day one** rather than discovering this
  in season.
- **Suspended/resumed games** get a second date.
- **Probable-pitcher-conditional lines**: some books void or re-hang a total when
  the announced starter changes. That is a market event worth recording, not
  noise to smooth over.

---

## Storage model

### The tick store: a narrow star schema

```
lh_book    (book_id SMALLINT PK, slug UNIQUE, name)
lh_market  (market_id SMALLINT PK, code UNIQUE)     -- totals | point_spread | money_line
lh_game    (game_id TEXT PK, game_date, season_year, tipoff_utc,
            event_id, team_home, team_away)
lh_line    (game_id, season_year, market_id, book_id, line_ts,
            mins_to_tip, is_pregame, is_opener,
            left_line, left_price, right_line, right_price,
            PRIMARY KEY (game_id, market_id, book_id, line_ts, season_year))
           PARTITION BY LIST (season_year)
lh_load_meta (season_year PK, timezone, confidence, source_rows,
              loaded_rows, dropped_rows JSONB, loaded_at)
```

Every column is a join key, a filter, or a price. Everything derivable from
`game_id` (team names, matchup URLs) or from the parsed values (`timestamp_raw`,
`left_value_raw`) was dropped.

**Why so narrow:** the instance has a 1 GB cap covering heap, indexes *and* WAL.
That constraint produced better decisions than an unlimited budget would have.

### Decision: `left` / `right`, not `over` / `under` / `home` / `away`

One pair of columns serves all three markets:

| market | `left` | `right` | invariant |
|---|---|---|---|
| totals | OVER | UNDER | `left_line == right_line` |
| point_spread | AWAY | HOME | `left_line == -right_line` (mirrored) |
| money_line | AWAY | HOME | both line columns NULL by nature |

**Reasoning.** A two-way market always has two sides. Naming them positionally
lets one table and one set of feature functions cover every market. The cost is
that "left" means different things per market, and getting it backwards
silently inverts every spread and moneyline feature.

**So it is verified against the data, not assumed, and pinned by a test.** Two
independent measurements: the residual of (home margin − left spread) has
std 13.46 vs 19.96 for the opposite orientation; devigged moneyline
probabilities average 0.551 on the right side vs 0.449 on the left, consistent
with home advantage. `tests/test_line_history_snapshots.py` locks it.

**Reusable principle.** Any sign or side convention that is not enforced by a
type must be (a) measured against realised outcomes and (b) pinned by a test.
"Verified against realised margins at correlation +0.460" is the level of
evidence to demand of yourself here. It is the single easiest place to lose
months of work to a silent sign flip.

### Decision: encode lines and prices as SMALLINT

- **Lines are always half-points**, so store them **doubled**: `224.5 → 449`.
  Exact, 2 bytes, versus ~12 for `NUMERIC`.
- **Prices are American odds** and fit natively once the "off the board"
  sentinel is nulled.
- **Decode in exactly one place** — the read module divides by 2 on the way out
  — so no feature module ever has to remember the encoding.

**MLB adaptation.** The doubling trick relies on a half-point quote increment.
MLB run lines are ±1.5 and totals move in halves, so it still works — but
**alternate run lines and quarter-point Asian-style totals do not.** Verify the
increment on real data before committing to the encoding, and prefer ×4 or a
plain `NUMERIC(5,2)` if the increment is finer. Getting a lossy encoding into
1.9M rows is expensive to undo.

### Decision: null the "off the board" sentinel at load

SBR writes `-100000` where a book had no price up. Left in, it is a
catastrophic outlier that survives into every mean, std and devig. It is
converted to NULL once, in `_encode_price`.

**Reusable principle.** Provider sentinels (`-100000`, `0`, `-1`, `999`) become
NULL at the storage boundary. Never downstream, and never in the feature layer,
where each consumer would have to remember.

### Decision: `mins_to_tip` and `is_pregame` are NOT NULL and are not conveniences

SBR records **in-play** ticks with the same shape as pre-game ones. Nothing else
in the row separates a legitimate feature from direct target leakage. So the
distance to tipoff is computed at load, stored, and enforced non-null.

Sign convention: `mins_to_tip` is **negative** pre-game (stored), and the read
layer flips it to a positive `minutes_before_tip` so the sign convention never
reaches feature code. Reads additionally refuse anything inside a 5-minute
safety margin of tipoff, because a per-game tipoff that is a few minutes stale
would otherwise admit an in-play tick.

**Reusable principle.** If your source mixes pre-game and in-play observations,
**the discriminator is a NOT NULL stored column, not a runtime filter.** A
filter can be forgotten at one call site; a column cannot be null.

**MLB adaptation.** Same hazard, worse: MLB in-play markets are extremely
liquid and first-inning ticks look exactly like late pre-game ticks. Also
account for **rain delays** — a tick 20 minutes "after" a scheduled first pitch
may still be pre-game. Consider storing both scheduled and actual first pitch.

### Decision: LIST-partition the fact table by season

- A season can be dropped instantly if the storage cap is reached.
- Season-filtered reads prune to one partition and need no secondary index.
- The write path stages one season at a time, keeping the WAL off a single spike.

`season_year` appears **last** in the primary key only because Postgres requires
the partition key inside any unique constraint; `game_id` stays leading so
"all lines for game X" still uses the index prefix.

### Decision: record how each load was performed

`lh_load_meta` stores per season the timezone assumed, a **confidence label**,
row counts in and out, and a JSONB breakdown of *why* rows were dropped.

**Reasoning.** The 2019-20 and 2020-21 seasons could not have their timezone
pinned. Rather than loading them under a guess, they are held back — and the
fact that they are held back, and why, is queryable rather than tribal
knowledge. `available_seasons()` reads the store instead of a hardcoded range,
so "what can we train on" has an honest answer.

**Reusable principle.** Store provenance and confidence alongside the data.
"We are not sure about these two seasons" is a fact the modelling layer needs.

### The wide game table

`odds_sportsbook`, one row per game, PK `game_id`, plus a unique index on
`(game_date, team_home, team_away)` — the natural key, enforced so a
double-insert under a different provider id cannot happen.

Columns are generated programmatically from three book lists, one triple per
book per market: `total_<book>_line_over/_price_over/_line_under/_price_under`,
`spread_<book>_line_home/_away/_price_home/_price_away`,
`ml_<book>_price_home/_away`, plus consensus percentages and the consensus
**opener** line.

**This is a denormalised view, and it shows.** Adding a book means a schema
migration; a book that only exists in recent seasons becomes a *season
indicator* the model can exploit (see the leakage section). The tick store's
`book_id` dimension has neither problem. **If you are starting fresh, build the
tick/long form as the source of truth and generate the wide table from it.**

---

## Avoiding duplicates and unnecessary writes

### Decision: every odds write is insert-only

Both stores use `ON CONFLICT DO NOTHING`. Never upsert a tick, and never
replace a game's rows wholesale.

**Reasoning, concretely.** SBR has since dropped Caesars. There are ~270k
historical Caesars rows that **cannot be refetched**. A "delete then reload this
game" refresh would destroy them permanently. Insert-only makes re-fetching cost
one HTTP request and nothing else, which in turn makes it safe to re-fetch
aggressively.

The game *dimension* is deliberately insert-only too, not an upsert: the
`mins_to_tip` on already-stored ticks was computed against the stored
`tipoff_utc`, so silently moving the tipoff would desynchronise rows the run is
not touching.

**Reusable principle.** For any append-only observation stream from a source you
do not control, **the source can lose data.** Insert-only is the only refresh
strategy that is safe against that. Make idempotency a property of the key
(truncated timestamp + natural composite PK), not of a diffing step.

### Decision: presence is not the same as finality

The daily updater re-fetches a rolling window of recent dates **unconditionally**
rather than diffing.

**Reasoning.** A game fetched on the morning it is played is *present* but not
*final* — its lines keep moving until tipoff. Presence-based gap-filling would
never bring it back, so the store would permanently hold a partial history for
every game first seen early. Three days covers a skipped run plus the gap
between a morning fetch and the close.

### Decision: three independent reasons to fetch a date

`plan_update()` unions:

1. **Refresh window** — dates with games in the last 3 days. Unconditional.
2. **Gaps** — games the games table knows about that the store does not. The
   games table is the reference for what *should* exist; anything present there
   and absent here is a gap. Bounded below by the store's own earliest game
   (before that there is no history to be missing, only history never collected).
3. **Partial coverage** — stored games missing a book that *most games on their
   own date* carry.

Reason 3 needed two guards to be usable at all, both learned the hard way:

- **Discontinued books are excluded outright.** A missing Caesars must never
  mark a game partial, or those games get re-fetched forever waiting for data
  the source no longer has.
- **A book is "expected" on a date only if it priced ≥50% of that date's games.**
  Without the threshold, a book's launch day (Fanatics covered 1 of 11 games on
  2025-11-05) marks the other ten partial permanently. Books also legitimately
  skip individual games.

`dry_run=True` reports the whole plan and stops before the first request.

**Reusable principle.** "What should I fetch?" is a query against your own
store, not a date range someone types. Write it as a planner that can run
without fetching. And a coverage-based completeness check needs both a
*discontinued* list and a *share threshold*, or it will chase data that does not
exist.

### Bulk loading

`COPY` into an `UNLOGGED` staging table, then one
`INSERT … SELECT … ON CONFLICT DO NOTHING` into the partitioned target, staged
per season. `executemany` is one round trip per row — untenable for 1.9M rows
over a WAN — and `UNLOGGED` keeps the copy itself out of the WAL.

---

## Snapshots: turning ticks into "the market at time T"

This is where the tick store pays for itself, and the design generalises
completely.

### Decision: a snapshot is last-observation-carried-forward, never equality

**The store records line *changes*, not samples.** There is no row for "the line
at 14:00" — only a row each time a book moved. So a snapshot at T is: *the last
tick with `minutes_before_tip >= T`, per (game, market, book)*.

`>=` is the leakage filter, and it is inclusive of the boundary only: a tick
exactly at the horizon was observable, a tick one minute later was not.

**The age of the carried line is itself a feature.** Measured here, the median
carried line is 30–100 minutes old and the p90 is around five hours. So
`line_age_minutes` is emitted next to every quote rather than treated as
bookkeeping — "this book's number is five hours stale" is real information about
that book at that moment.

Availability is explicit too: `has_quote` is a 0/1 column, so "this book had no
price at T" is a value the model can read rather than a NaN that row-level
cleaning may act on.

### Decision: a roughly geometric snapshot grid, over-sampled

`(0, 30, 60, 120, 180, 240, 300, 360, 480, 720)` minutes before tip.

- **Geometric, not uniform**, because line-movement information decays in log
  time.
- **Stops at 720** because coverage is measured at ~100% of game-book pairs out
  to 12h and collapses to ~60% at 24h. A snapshot that only exists for
  well-covered games is a biased sample, not a longer lead time.
  `snapshot_coverage()` exists as an acceptance check for exactly this.
- **`0` is the closing snapshot** — "bet as late as the market allows", which
  given the 5-minute safety margin is the last tick ≥5 min out. This is what
  puts the intermediate dataset on the same footing as the closing dataset:
  same bet, same moment, different feature construction. Note that closing-line
  value is ~0 by construction on those rows, so CLV over a pooled dataset is
  diluted by them.
- **Deliberately denser than any model needs.** Snapshots are *rows*, so an
  unwanted horizon is removed with a filter; adding one back means regenerating
  the dataset. Over-sampling is the cheap direction.

**MLB adaptation.** The horizon shape should change. NBA lines are hung the day
before; MLB totals are heavily conditioned on the **announced starting
pitchers**, which land at very different times, and lines can be off the board
or heavily juiced until then. A grid anchored on *hours before first pitch*
alone will mix "pitchers known" and "pitchers unknown" states into the same
horizon. Consider adding an explicit `pitchers_announced` flag to the panel, or
snapshotting relative to the announcement rather than only to first pitch.
Measure the coverage cliff on real MLB data before fixing a grid.

### Decision: normalise each quote onto its −110/−110 equivalent

A book can move its **price** without moving its **line**. Two quotes of "224.5"
at −105/−115 and at −120/+100 are not the same market view. Comparing them
across books, or across snapshots of one book, is only meaningful once each is
restated as the line it would carry at symmetric pricing.

Implementation (`data_processing/line_history/normalization.py`):
- Devig the two-way prices to fair probabilities; keep the `overround` as its own
  feature.
- **Half-point lines**: closed form, `line + sigma * Phi^-1(fair)` — no push is
  possible, so the quantile maps straight across.
- **Integer lines**: a push *is* possible, so the quoted probability is
  conditional on not pushing; solved by vectorised bisection (monotone in the
  center, so plain bisection is valid).
- **Per-market sigma, calibrated not assumed.** Margin sigma is 13.46, measured
  as the std of (home margin − closing spread) over 6,112 games, against a raw
  margin std of 15.57. The totals sigma differs by over two points. Using one
  for the other is a real error.
- **`left_wins_above` differs by market.** For totals OVER wins above the line;
  for spread the left (away) side covers when the home margin lands *below* it.
  Getting this wrong does not round — it inverts the correction. A quote of
  away +4.5 at −130 implies a fair margin *below* 4.5 (~3.0); treating it like a
  total pushes it to 6.0, wrong by 3 points in the wrong direction. **Symmetric
  −110/−110 quotes hide the bug entirely**, which is why the tests use
  asymmetric prices.
- Rounding is half-away-from-zero for signed lines, so a fair −3.25 and +3.25
  land on −3.5 and +3.5 rather than −3.0 and +3.5.
- Moneyline gets no centered line — there is no line to shift, so
  "normalisation" there is devigging only.

**The raw line is always kept alongside the normalised one.** The raw one is
what you could actually have bet; the normalised one is what is comparable.
ROI measured against a normalised line is comparable but **not executable** —
state that plainly wherever it is reported.

**Reusable principle.** Price and line are two dials on the same market view.
Any cross-book or cross-time comparison of lines must first collapse the price
dial, and the collapse needs a per-market outcome-distribution parameter you
have measured on your own data.

### Decision: one canonical `level` per market, and one "up" probability

Every movement, dispersion and path feature is computed from a single series
per market so they stay comparable within a market and cannot mix scales
across one:

- **totals / spread** → the **raw** line, the number actually on the board.
  Deliberately raw: a move is a thing that visibly happened, and mixing a
  centered "now" against a raw "then" put 946 spread rows outside their own
  realised range. The pricing correction is carried separately as
  `norm_minus_raw` and by the probability-movement family.
- **moneyline** → the devigged HOME win probability. Without this the market has
  no level at all, and `line_delta` was NaN on every row — silently zeroing every
  move count, reversal, window flag and dispersion figure for the entire market.

Paired with it, `fair_up` = the devigged probability of *the side that wins when
the level goes up*: `fair_left` for totals, `fair_right` for spread and
moneyline. Using `fair_left` everywhere reversed the sign against the level on
two of three markets — reporting one event with two opposite signs.

**Reusable principle.** Define "the number this market moves in" once per
market, in one function, and build every path feature on it. Then define "the
side that benefits when that number rises" as a second function. These two are
the whole of orientation, and centralising them is what stops sign bugs from
being rediscovered per feature.

### Movement, cross-book and history features

Full detail is in the `feature-engineering` skill. The architecturally relevant
points:

- **Windowed moves reuse the snapshot machinery.** "How far has the line moved in
  the last hour, as of T" is exactly `level(T) − level(T + 60)` — a second as-of
  read, not a separate code path with its own leakage surface. Cost scales as
  `len(grid) × len(windows)`.
- **Missing windows are flagged, not left NaN.** Long horizons are systematically
  the ones whose history does not reach far enough back, so bare NaNs would let a
  row-level NaN limit delete precisely the 8h/12h rows the dataset exists to
  compare. Emit `move_last_W = 0.0` plus `has_window_W = 0/1`.
- **A move is a change in level; a re-price at the same number is counted
  separately** (`is_price_only`). Books routinely price a half-tick before taking
  it, and that pressure is invisible in the line. On the moneyline the two
  collapse by construction, kept as a structural zero so columns stay comparable.
- **Cross-book consensus is a median, not a mean.** One stale book sitting three
  points off the market is common; a mean drags toward it. Dispersion is reported
  separately so disagreement stays visible rather than averaged away.
- **"Steam" is pinned to a 60-minute window**, not "the shortest configured".
  Steam is cross-book *agreement*, which needs enough movers to mean anything —
  measured here a book moves in the trailing 60 min on only 32% of (row, book),
  and 37% of rows have no book moving at all. Letting the shortest window define
  it meant that merely *adding* a shorter window silently redefined an existing
  feature.
- **Closing lines of *previous* games are legitimate features.** Those games
  finished before the snapshot, so their complete open-to-close history was known
  at T. This is the one place a closing line may be used.

---

## Data-quality repairs, at load time

Two are worth copying as *patterns*:

**Structural repair using an invariant.** On a pick'em the SBR cell holds only a
price ("−110") with no spread number, and a `([+-]\d+(?:\.\d+)?)` pattern matches
the price as the line. Such rows are recognisable because **a genuine spread is
mirrored** (`left == -right`) while these carry complementary *price* pairs
(−110/−110, −115/−105). The value is relabelled as the price it demonstrably is;
the spread is left NULL rather than inferred as 0, since the source never said so.

**Bounds that respect state.** Pre-game totals outside 150–300 and spreads
outside ±30 are nulled — a dropped decimal turns 228.5 into 2285. **In-play rows
are exempt**: a live spread legitimately blows out past 30 during a rout. Only
the impossible value is cleared; the price and the row survive.

**Reusable principle.** Prefer repairs justified by a *structural invariant of
the market* over magnitude heuristics; where you must use bounds, condition them
on the row's state. And clear the bad field, not the row.

Every drop and repair is counted by reason into an `IngestStats` object and
persisted, so "we lost 4% of season 2023" is answerable.

---

## Leakage rules specific to odds

1. **In-play ticks are indistinguishable from pre-game ones except by
   `mins_to_tip`.** Enforced as a NOT NULL column plus a 5-minute safety margin
   on read. 2.7–3.7% of recent-season rows land at or after tipoff.
2. **Negative horizons are rejected.** `build_snapshot_panel` raises on a
   negative grid entry and `add_movement_features` raises on a non-positive
   window — both would read the market *after* the snapshot and would pass every
   column-name check downstream.
3. **Closing lines are physically separated, not filtered.** In the snapshot
   dataset, every current-game closing/consensus column across all three markets
   is renamed to `ODDS_CLOSING_*` and routed to a **separate scoring file** keyed
   by `(GAME_ID, TIME_TO_MATCH_MIN)`. The earlier design kept them in the frame
   behind a prefix, assuming the feature selector would drop them — it did not,
   because it drops only *configured* exclusions. **Physical separation is the
   only version of this that cannot be forgotten at the call site.**
4. **The opening line is safe at every horizon** (openers land a median ~25h
   before tip) and is deliberately *not* swept up with the closing columns — it
   is the baseline the betting evaluation compares against.
5. **Column availability can encode the season.** A book present only from 2025
   lets a model recover the year from which columns are non-null. Handled three
   ways: fold the discontinued book into the new one so there is one
   continuously-covered book (`merge_caesars_into_fanatics_ticks` at the tick
   level, mirrored at the column level), or exclude either outright. This is a
   real, easily-missed leak — **audit availability-by-season for every odds
   column.**
6. **Per-book coverage is uneven at the old end of the history.** BetMGM is 0% in
   2020 and 37.5% in 2019 against ~100% from 2022. The first season with usable
   odds was *measured* (2019, with 2018 holding 32 rows from one postseason
   fragment), not assumed from the games table's range.
7. **The scrape-time tipoff wins over the schedule feed's**, with a 15-minute
   tolerance check that *reports* disagreements rather than silently trusting
   either. Consistency with the ticks matters more than agreeing with a
   third party.

---

## Checklist for a new sport

1. Games table with stable ids and an unambiguous `game_date` convention — before
   any odds work. For MLB, settle doubleheaders in the key now.
2. Tipoff / first-pitch times, one request per season if the league offers a
   schedule feed.
3. Find the provider's embedded JSON payload. Confirm its timestamps carry an
   explicit offset. If they do not, stop and find another endpoint.
4. Decide the `left`/`right` convention per market, then **measure it against
   realised outcomes** and pin it with a test.
5. Verify the quote increment before choosing an integer encoding.
6. Star schema: book dim, market dim, game dim, partitioned fact. Insert-only.
   `mins_to_tip` + `is_pregame` NOT NULL.
7. Load-time repairs and per-reason drop counters from the first load.
8. Backfill day by day with polite sleeps, retries, and warn-not-raise.
9. Build the update planner (refresh window ∪ gaps ∪ partial coverage) with a
   dry-run mode.
10. Calibrate the per-market sigma on your own data before normalising anything.
11. Measure snapshot coverage by horizon, then choose the grid.
12. Separate closing lines into a scoring sidecar from day one.
