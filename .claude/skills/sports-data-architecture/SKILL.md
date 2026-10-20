---
name: sports-data-architecture
description: How non-odds sports data is modelled, ingested and kept current - game/team/player/official entities, stable identity across seasons and providers, availability and injury records, and the contextual datasets that proved worth collecting. Use when designing the schema for a new sports-betting repo, adding a data source, resolving entity identity across providers, or deciding what history is worth storing before any feature work starts.
---

# Sports data architecture

The data model underneath an NBA totals/spread system, stated as decisions
rather than as a column list. Companion skills: `odds-data-architecture` for
market data, `feature-engineering` for what gets built on top.

The single most consequential decision, stated first because everything else
follows from it:

> **Store one row per team per game, not one row per game.**

---

## Decision: the team-game is the atomic row

**Current implementation.** `nba_games` has primary key
`(game_id, team_id, season_id, season_year)` — two rows per game. Home/away is a
boolean column, not a column *suffix*. The same for players
(`game_id, team_id, player_id, season_year`), inactives
(`player_id, game_id`) and officials (`official_id, game_id`).

The game-level, `_TEAM_HOME` / `_TEAM_AWAY` wide form is produced **once, at the
end**, by a single merge step.

**Reasoning.** Every historical feature a betting model wants is "this team's
recent history". In team-game form that is
`groupby("TEAM_ID")[col].shift(1).rolling(n).mean()` — one line, obviously
leakage-safe, and identical code for every statistic. In game-level form the
same feature needs a union of two differently-named column families, and every
new statistic doubles the code. It also makes the two sides silently
inconsistent the first time someone updates one branch.

**Reusable principle.** Model the grain at which *history accumulates*, not the
grain at which *predictions are made*. Pivot to prediction grain as the last
step. This holds for any sport where an entity plays repeatedly.

**MLB adaptation.** Directly applicable, and you probably want **three**
accumulation grains, not two:
- team-game (as here),
- player-game (as here — batters),
- **pitcher-appearance**, which is the grain MLB totals actually turn on. A
  starting pitcher's history is the single strongest team-independent signal for
  a run total, and it accumulates on its own clock (every ~5 games, not every
  game). Treat it as a first-class entity with its own rolling history rather
  than as "a player who happened to start".

---

## Core entities and identity

| Entity | Key | Notes |
|---|---|---|
| Game | `game_id` TEXT | the provider's id, adopted as canonical |
| Team | `team_id` TEXT | provider id; names mapped, ids never |
| Player | `player_id` TEXT | provider id |
| Official | `official_id` TEXT | provider id |
| Season | `season_year` INT | **start** year: 2025 = the 2025-26 season |
| Book | `book_id` SMALLINT | surrogate, assigned first-seen (see odds skill) |

### Decision: adopt the primary provider's ids as canonical

No surrogate keys for games, teams, players or officials.

**Reasoning.** One provider (here, the league's own API) supplies games, box
scores, inactives and officials. Making its ids canonical means those four
sources join with no mapping layer at all. A surrogate layer only earns its keep
when *no* source is dominant.

**Cost, and how it is paid.** Every *other* source — the odds provider, the
injury-report PDF, the all-star voting scrape — must be mapped in. That mapping
is by **name**, and it is the single most fragile part of the system.

### Decision: one hand-maintained name map that raises on the unknown

`config/constants.TEAM_NAME_STANDARDIZATION` maps every spelling any source has
ever produced onto one canonical team name. Unknown name → `RuntimeError`, not a
dropped row.

**Reasoning.** A relocation, a rebrand or a provider changing "LA Clippers" to
"L.A. Clippers" must stop the pipeline. Silently dropping unmatched games gives
you a training set that is quietly missing a team for a season, and nothing
surfaces it. Loud failure costs an hour; silent drop costs a season of results.

Supporting maps live beside it: `TEAM_ID_MAP` (name → id),
`TEAM_NAME_CONFERENCE_MAP`, `TEAM_NAME_DIVISION_MAP`, `CITY_TO_LATLON`,
`CITY_TO_TIMEZONE`.

**Reusable principle.** Cross-provider entity resolution is a **data artifact you
own and version**, not an algorithm. Fuzzy matching belongs in a one-off script
that *proposes additions to the map*, never in the pipeline.

### Player identity is harder and is resolved differently

Player names on the injury report arrive as `"Last, First"` and must reach a
`player_id`. `manage_injury_data.py` reverses the name, calls the provider's
static player index, and disambiguates duplicates by querying each candidate's
current team.

Unmatched players are enumerated with team and date in a detailed error message —
though **the raise is currently commented out**, so an unmatched player is
reported and then silently skipped. That is a live gap: a star player whose name
fails to resolve simply vanishes from the injury features for that day, and only
a log line says so.

**Reusable principle.** Name-based player resolution needs three things: a
canonical direction for the name format, a disambiguator (team, position, or
birth date), and a **hard failure** on a miss. If you only build two of the
three, build the third before you trust an availability feature.

**MLB adaptation.** Somewhat easier — MLBAM ids are widely propagated, and
Chadwick's register maps MLBAM ↔ Retrosheet ↔ Baseball-Reference ↔ FanGraphs.
**Use it.** Adopt MLBAM as canonical and store a mapping table for the rest,
rather than resolving by name at all. Where you must resolve by name (an odds
page listing probable pitchers), disambiguate by team and handle accents and
suffixes explicitly.

### Decision: derive season type from the game id, not the text column

`SEASON_TYPE_MAP` keys off the first three characters of the `game_id`
(`001` preseason, `002` regular, `004` playoffs, and so on). The text
`SEASON_TYPE` column exists but **mislabels play-in games**, so filters use the
prefix.

**Reusable principle.** Where a provider encodes structure in an id, prefer the
id. Text label columns are maintained by hand somewhere upstream and drift.

### Decision: `season_year` is the season's **start** year, everywhere

One helper computes it (`d.year if d.month >= 10 else d.year - 1` for NBA) and
everything uses it. Season boundaries appear in dozens of places — rolling
resets, previous-season fallbacks, partition keys, roster windows — and two
conventions in one codebase is a whole class of off-by-one bugs.

**MLB adaptation.** Trivial by comparison: an MLB season is contained in one
calendar year, so `season_year` is just the year. Keep the named helper anyway
so the concept is one function.

### Decision: `game_date` is the **Eastern-time date of tipoff**

A 00:30 UTC tipoff belongs to the previous day's slate. This is the join key
between the games table, both odds stores, the schedule feed and the injury
report. It is the convention every US-sports data source already uses.

Separately, `tipoff_utc` is stored as a real timestamp where a precise instant is
needed (the whole snapshot/leakage machinery depends on it).

**Reusable principle.** Keep both: a **local slate date** for joining across
sources, and a **UTC instant** for anything temporal. Never try to make one do
both jobs.

---

## What is stored per game

Rather than the NBA column list, the *categories* worth reproducing:

| Category | Why it earns its place |
|---|---|
| **Outcome** | points for/against, win/loss, margin. The targets, and the input to every rolling feature. |
| **Volume / tempo** | possessions, pace. Separates "scored a lot" from "scored efficiently" — for a totals model this is the primary decomposition. |
| **Efficiency rates** | points per 100 possessions on offence and defence, true-shooting. Scale-free, so comparable across opponents and eras. |
| **Component counts** | attempts, makes, rebounds, turnovers, fouls. Needed to build *style* rates (below); mostly not features themselves. |
| **Context flags** | overtime, season type, home/away, venue. |
| **Officials** | one row per official per game. |
| **Availability** | who did not play. |

**The tempo/efficiency split is the transferable idea.** A totals model is
fundamentally `expected_events × expected_value_per_event`. Store both factors
separately rather than only their product, and store them as **rates**, so they
can be recombined against a specific opponent.

**MLB adaptation.** The same decomposition, different names:
- *volume/tempo* → plate appearances, innings, baserunners allowed, pitches
- *efficiency* → wOBA, wRC+, FIP/xFIP, K% and BB%
- *component counts* → the batted-ball mix (GB/FB/LD), hard-hit rate, barrel rate
- *context* → **park, weather, wind, umpire, DH rules, extra innings, and the
  bullpen state** (how many innings the relief corps threw in the last 3 days).
  The bullpen is MLB's closest analogue to NBA fatigue and has no NBA equivalent
  feature — it is genuinely new work.

A data-quality guard worth copying: the games table has
`pts INTEGER NOT NULL CHECK (pts > 40)`. A basketball team does not score 40.
Cheap constraints at the schema boundary catch broken box scores before they
reach a rolling mean. *MLB equivalent: not a floor on runs (0 is legal) but a
ceiling, and a check that innings pitched are plausible.*

---

## Availability and injuries

This is the subtlest part of the model, and the part with the most real leakage
risk. Read this section twice before building the MLB equivalent.

### Two different sources for the same concept

| | Historical (training) | Same-day (serving) |
|---|---|---|
| Source | `BoxScoreSummaryV3` **inactive list**, per game | league **injury-report PDF**, scraped and parsed |
| Table | `nba_injuries` (`player_id`, `game_id`, `team_id`, `game_date`, …) | in-memory dict `{game_id: {team_id: [player_id]}}` |
| Semantics | who *actually did not play* | who is *listed Out or Doubtful* pre-game |
| Statuses | binary (on the inactive list or not) | Out / Doubtful / Questionable / Probable / Available |
| Timing | settled after the game | published and revised repeatedly before tip |

Both are funnelled into the same nested dict shape
`{game_id: {team_id: [player_id, ...]}}` and consumed by one code path.

### The train/serve skew, stated plainly

**The historical availability signal is strictly better than the one production
has.** The inactive list is the settled truth and includes late scratches; the
pre-game report at prediction time is a forecast, and `Questionable` players
resolve either way.

Strictly, this is not target leakage — the inactive list is determined before
tip-off, so the information *did* exist. But it is **optimistic**: a model
trained on "who was actually out" and served "who is listed Out or Doubtful"
will underperform its backtest, and the gap is invisible in cross-validation
because both sides of the split use the historical source.

There is a second, sharper issue. The historical injured set is built from *two*
sources: the inactive list, **and** any player whose box-score `COMMENT` field
matches `injur|injry` (e.g. "DND — Injury/Illness"). That comment field is
written **after** the game. It is being used as a proxy for a pre-game state,
and for most players it is a fair one — but it is a post-game field feeding a
pre-game feature, and it should be named as such.

**Reusable principle, and the thing to actually do differently for MLB.**
*Reconstruct the historical availability signal from the same source production
will use.* Archive the pre-game report daily from the first day of the project;
in a year you will have a point-in-time-correct history that no post-hoc
reconstruction can match. Until then, use the settled source, **measure the
disagreement** between it and the archived reports, and treat backtest
availability features as an upper bound.

*Concretely for MLB:* start archiving the daily lineup card, the IL transaction
feed, and the probable-pitcher listing **on day one**, timestamped at fetch. The
retrospective sources (a box score's absence of a player, the IL stint's
official start date) are exactly the "settled truth" trap described above — an
IL stint is often *backdated*, so an after-the-fact query returns a player as
"on the IL" on a date when nobody knew it yet. **That one is genuine
look-ahead leakage, not merely optimism.**

### Decision: availability is a *team roster* question, not a player-list question

Knowing who is out is useless without knowing who *would otherwise have played*.
`create_player_lookup` builds, per (season, team), a timeline of every player's
appearances, and answers "who was on this team's roster as of this date" by
finding each candidate's **last game strictly before** the date and checking it
was for this team. Mid-season trades are handled by that rule alone.

One cross-check worth copying: if a player appears on *another* team's injury
list for the same game, they are treated as no longer active for this team — a
traded player often shows up in both places for a few days.

**Two acknowledged holes in this** (both documented in the code, both worth
re-examining for a new sport):
- At a team's **season opener** nobody has an earlier game that season, so the
  lookup returns nothing and ~190 columns go missing — enough for the row to be
  discarded. The fallback reads the roster from **the game's own box score
  rows**. The argument for safety is that who dresses is known before tip from
  the injury report, and every *value* attached to those players is a strictly
  prior-season average. That is reasonable but it *is* reading the current
  game's record.
- Selecting active players for a historical row filters on `GAME_DATE == date`,
  i.e. the current game's box score. Same argument, same caveat.

**Reusable principle.** Roster membership at time T should ideally come from a
roster *feed* archived at T. Where it is reconstructed from appearances, you
need an explicit season-opener strategy, and you should write down which side of
the leakage line the fallback sits on.

### Decision: injury streaks are counted in team games, not calendar days

`create_injury_streak_lookup` walks a team's game list backwards from the current
game and counts consecutive games the player was listed out, capped at two
seasons. "Out for the last 8 team games" is more predictive than "out for 19
days", because it says how long the team has been adapting.

**MLB adaptation.** Days *and* games diverge much more (off-days, IL minimums).
Count both; the IL stint length in days is directly available and is itself a
severity proxy.

---

## Contextual datasets that proved worth collecting

For each: what it is, the semantic purpose, and what the MLB analogue would be.

### Officials
One row per official per game. Used to build a per-official "games with them
minus games without them" delta on total points, on total fouls, and on the
market's error. Position within the crew is ignored — the crew is sorted and
de-duplicated into deterministic slots before anything is computed.

*Purpose:* individual arbiters measurably shift scoring rate.
*MLB:* **home-plate umpire.** The strongest direct analogue in any sport — umpire
strike-zone size moves K rate, walk rate and therefore run totals. Collect
umpire assignments from day one; they are published pre-game.

### All-Star fan voting
Scraped season-by-season; converted into per-team-game features: the team's
share of league-wide fan votes among its available players, the maximum vote
share among its *injured* players, candidate counts.

*Purpose:* an **externally-sourced, market-independent proxy for player
importance and star power** that does not depend on your own box-score
aggregates. Star absence is not linear in minutes or points.

*MLB:* there is a literal All-Star ballot, but the better analogues are
**projected WAR / preseason projections** (Steamer, ZiPS, PECOTA) or **contract
value**. The reusable point is not "all-star votes" — it is *have at least one
importance signal that is not a function of your own rolling stats*, so that
"team's best player is out" means something at the start of a season when the
rolling stats are empty.

### Venue geography
`CITY_TO_LATLON` and `CITY_TO_TIMEZONE`, used for great-circle travel distance
and timezone-change hours.

*MLB:* the same, **plus the park itself as a first-class entity** — park factors
are far more consequential for MLB run totals than travel is, and altitude,
dimensions, roof state and prevailing wind all belong on the venue record.
Attach weather at first pitch.

### Schedule feed
One request per season returns every game's id, tipoff in UTC and ET, arena, and
team tricodes. This is the tipoff source. A per-game box-score endpoint exists
(`game_time_index`, 36 columns plus JSONB payloads, including arena, attendance
and status) and is scriptable, but needs ~1,400 requests per season instead of
one, and was not the source used.

*Reusable principle:* prefer the bulk schedule endpoint. Also keep the raw
payload in a JSONB column when you do fetch per-game detail — reparsing beats
re-fetching.

### Standings — **not stored as such**
Worth being explicit: there is no standings table. Win/loss record before each
game is *derived* from the team-game rows by a cumulative sum minus the current
row's result. That derivation is strictly better than a scraped standings
snapshot, because it is automatically point-in-time correct.

*Reusable principle:* **derive point-in-time state from an event log rather than
storing snapshots**, wherever the event log is complete. Snapshot tables are
where "as of" bugs live.

### Coaching — not collected
No coaching data. Noted only so it is not assumed present.

---

## Ingestion and keeping current

### Decision: gap-driven, not date-driven

Every updater asks the database what is missing rather than being told a range:

1. Load games from the games DB for the target season.
2. Optionally restrict to the latest N games.
3. Query which of those ids are **already present** in the target table.
4. Fetch only the dates covering the missing ids.
5. Merge back to resolve the canonical `game_id`, filter to the missing set,
   insert.

**Reusable principle.** The reference for "what should exist" is your own games
table. The updater is then idempotent, safe to run on any cadence, and cheap
when there is nothing to do.

### Decision: exclude live and same-day games from ingestion

The daily orchestrator collects live game ids plus the ids for an excluded date
and refuses to upload them.

**Reasoning.** A box score fetched mid-game is a *partial* box score that looks
complete. Written insert-only, it poisons that game permanently.

**Reusable principle.** Never ingest an in-progress event. Explicitly exclude
live ids; do not rely on a status field you have not checked.

### Retries, and where they stop

Odds updates retry twice with a 2 s pause and then **raise**, aborting the
backfill. Provider rate limiting raises immediately with a "re-run later"
message.

**Reasoning.** Distinguish *transient* (retry) from *throttled* (stop and come
back). Retrying through a rate limit gets you banned. Because everything is
gap-driven and insert-only, stopping early is free — the next run resumes.

### Operational cadence (GitHub Actions)

Daily DB update → daily predict → daily finished-match settlement; weekly
retrain; twice-monthly S3 backup of the DB and of the line-history store;
weekly idle-connection cleanup; a model smoke test. Each workflow uses a
`concurrency` group so a slow run cannot overlap itself.

**Reusable principle.** The three daily jobs are *distinct*: update data,
predict, settle results. Keeping settlement separate is what lets you measure
realised performance without touching the prediction path.

---

## Storage layout

One Postgres database, one **schema per domain** (`nba_games`, `nba_players`,
`nba_injuries`, `nba_refs`, `odds_sportsbook`, `all_star_voting`,
`ou_predictions`, …), plus a **second, separate Postgres** for the high-volume
line-history store.

**Reasoning for the split.** The tick store is ~1.9M rows with completely
different access patterns, growth rate and cost profile. Isolating it means its
size cannot threaten the operational database, and it can be hosted on cheaper,
differently-tuned infrastructure.

Predictions are stored in the database too, with a separate evaluation updater
that settles them against final scores. Model artifacts live in an S3 registry
with `staging` / `production` / `archive` prefixes and structured metadata
(feature schema, training metrics, NaN policy, train date range) — **not** in
git, though legacy artifacts remain there.

**Reusable principle.** Separate by *access pattern and growth rate*, not by
subject matter. And store predictions as data from the first day — you cannot
retroactively measure a model you did not record.

---

## Blueprint order for a new sport

Each step is a prerequisite for the next.

1. **Games table.** Stable id, `season_year` (start year), local slate date, UTC
   start instant, home/away, venue, status, final score. Add plausibility CHECKs.
   *For MLB, settle doubleheader keying now.*
2. **Team-game table.** Two rows per game. Volume/tempo, efficiency rates,
   component counts.
3. **Name standardisation map.** Raises on unknown. Before any second source.
4. **Player-game table.** Plus, for MLB, pitcher-appearance as its own grain.
5. **Schedule feed** for start times.
6. **Start archiving daily point-in-time snapshots** of availability, lineups and
   probable starters. Day one. This is the one thing that cannot be backfilled.
7. **Officials / umpires.**
8. **Odds.** See `odds-data-architecture`.
9. **Venue geography and, for MLB, park + weather.**
10. **One external importance signal** independent of your own box scores.
11. **Gap-driven updaters** for each source, then the daily orchestrator.
12. **Predictions table and a settlement job**, before you believe any result.
