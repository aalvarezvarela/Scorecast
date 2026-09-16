# Injury-report database plan (Aiven `injury_report` schema)

**Status:** plan only — nothing implemented.
**Companion:** [`injury_report_archive_plan.md`](injury_report_archive_plan.md) covers
collecting the PDFs. This document covers turning them into queryable state.

---

## 0. What this is for

One question, asked at prediction time:

> For game *G*, at instant *T*, what was each player's officially reported status,
> and how stale was that information?

The existing `nba_injuries` table cannot answer it. Its primary key is
`(player_id, game_id)` with **no timestamp** — one row per player-game, written
after the fact. It is post-hoc truth, not a time series, and §8.6 of the archive
plan already rules it out for horizon features. Given this repo's history with the
rotation-depth availability leak, the two must not be mixed in one feature set.

The design goal is therefore not "store the injury reports". It is **store the
minimum from which any historical snapshot can be reconstructed exactly**, with
the timestamp as a first-class key.

---

## 1. The central decision: change-log, not snapshots

The naive schema is one row per (report × player). It is 20–50× larger than
necessary, because consecutive reports are almost identical — a player listed
`Out` stays `Out` for dozens of consecutive reports.

Measured on two complete game-days, every published report downloaded and parsed:

| | 2025-01-29 (24/day era) | 2026-02-11 (96/day era) |
|---|---:|---:|
| Reports that day | 24 | 96 |
| Games covered | 16 | 15 |
| Rows per report (mean) | 121 | 139 |
| **Snapshot rows** (report × player) | **2,902** | **13,302** |
| Distinct (game, player) pairs | 160 | 174 |
| Status transitions observed | 28 | 70 |
| **Change-log rows** (first sighting + transitions) | **188** | **244** |
| **Compression** | **15.4×** | **54.5×** |

Nothing is lost. A change-log with validity intervals reconstructs any snapshot
by range containment, and it answers the *actually interesting* question —
"when did this player become questionable?" — directly, which the snapshot form
can only answer with a self-join.

The compression grows with cadence, which matters because the 96/day era is
already 38% of all reports and every future season is 96/day.

Two further observations from the same runs, both load-bearing:

* **Players never dropped off a report before tipoff** — 0 of 334 (game, player)
  pairs across 120 consecutive reports. Games do leave the report, but only
  ~4.5 h *after* tipoff. Section 7 works through what this means.
* **`NOT YET SUBMITTED` is frequent** (220 and 887 rows respectively) and is
  *not* noise. It means "this team has not filed yet", i.e. *unknown*, which is
  categorically different from "nobody is injured". It gets its own table.

---

## 2. Schema

Schema name `injury_report`, table prefix `ir_`, on **Aiven** — mirroring the
`line_history` / `lh_` module, which solves the same problem under the same 1 GB
cap and is the working precedent in this repo.

### 2.1 Dimensions (tiny, static)

**`Current Status` is a closed set of exactly five values.** Verified over 404
reports sampled every 4 days across the whole in-scope period (2019-12-18 →
2026-06), **34,505 status rows**:

| `status_id` | code | rows | share |
|---:|---|---:|---:|
| 1 | `available` | 2,829 | 8.2% |
| 2 | `probable` | 1,445 | 4.2% |
| 3 | `questionable` | 3,504 | 10.2% |
| 4 | `doubtful` | 520 | 1.5% |
| 5 | `out` | 26,207 | 76.0% |

No sixth value, no `Game Time Decision`, no case or spelling variants. Checked a
second, independent way — harvesting every short line of raw PDF text and
subtracting everything attributable to teams, names, headers and reasons — which
found the same five and nothing else. The two passes agree to within 4 rows in
34,505 (0.01%), so a status cannot be hiding behind a parser bug.

Because it is closed and verified, it is worth constraining rather than trusting:

```sql
-- Ordinal on purpose: severity increases with the id, so "at least doubtful"
-- is `status_id >= 4` and no CASE expression is needed anywhere downstream.
CREATE TABLE injury_report.ir_status (
    status_id  SMALLINT PRIMARY KEY,
    code       TEXT NOT NULL UNIQUE      -- available, probable, questionable, doubtful, out
);
-- A sixth status would be a real change in what the NBA publishes, and should
-- stop the load rather than land silently as an unmapped row.
ALTER TABLE injury_report.ir_status_span
  ADD CONSTRAINT ir_status_known CHECK (status_id BETWEEN 1 AND 5);

**Reason categories are the opposite: an open, slightly dirty set.** The same
scan found ~16 of them, and they must not be modelled as an enum:

| Category | rows | note |
|---|---:|---|
| `Injury/Illness` | 23,479 | |
| `G League` | 8,292 | ` - Two-Way`, ` - On Assignment` |
| `Health and Safety Protocols` | 892 | COVID era, now dormant |
| `Not With Team` | 463 | **also appears as `Not with Team`** |
| `Personal Reasons` | 360 | |
| `-` | 304 | no reason given |
| `Rest` | 197 | |
| `League Suspension` | 120 | |
| `Reconditioning` / `Return to Competition Reconditioning` | 197 | |
| `Trade Pending` | 60 | |
| `Concussion Protocol` | 52 | |
| `Ineligible To Play` | 36 | verified genuine, 2022-01-14 |
| `Coach's Decision` | 30 | |
| `Team Suspension` | 14 | |
| `Non-NBA Team` | 2 | verified genuine, 2022-02-20 |

**Keep the reason. It is not optional metadata — it changes what the status
means.** Status × category over 109 reports spanning 2020-2026, 10,469 rows:

| Category | Out | Doubtful | Questionable | Probable | Available |
|---|---:|---:|---:|---:|---:|
| Injury/Illness | 4,890 | 120 | 957 | 367 | 667 |
| **G League** | **2,393** | 19 | 100 | 14 | 90 |
| Health and Safety Protocols | 218 | 0 | 4 | 0 | 6 |
| Not With Team | 134 | 0 | 0 | 0 | 2 |
| Personal Reasons | 90 | 1 | 8 | 1 | 8 |
| Rest | 75 | 1 | 3 | 0 | 0 |
| Trade Pending | 44 | 0 | 3 | 0 | 4 |
| League Suspension | 37 | 0 | 0 | 0 | 0 |
| everything else | 94 | 0 | 10 | 4 | 15 |

Of 7,975 `Out` rows, **only 61.3% are `Injury/Illness`. 30.0% are `G League`** —
`Two-Way` (1,957) and `On Assignment` (659) — players who are not hurt at all,
just not with the NBA club tonight. Adding `Rest`, `Trade Pending`,
`League Suspension`, `Coach's Decision` and the rest, **38.7% of `Out` is not an
injury.**

A feature built on status alone would count a two-way player's routine G League
assignment as identical to a starter tearing an ACL. `reason_id` is therefore
`NOT NULL` in `ir_status_span`, and any "how depleted is this team" feature must
filter on category rather than on `status_id` alone.

Three consequences for the loader:

* **Case-fold before matching.** `Not With Team` and `Not with Team` are the same
  category printed two ways — the NBA is inconsistent, not us.
* **Insert unseen categories, never reject them.** New ones do appear
  (`Health and Safety Protocols` arrived in 2020, `Ineligible To Play` in 2022).
  Unlike statuses, an unknown category is normal and must not stop a load.
* **Watch for wrapped detail masquerading as a category.** A reason that wraps
  onto a second line puts its tail on a line of its own, which is how
  `Return to Competition Reconditioning` shows up looking like a category when it
  is really the detail of `Not with Team`. Split on the *first* ` - ` only.

```sql
CREATE TABLE injury_report.ir_reason_category (
    category_id SMALLINT PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,    -- case-folded: not_with_team, g_league, ...
    label       TEXT NOT NULL            -- first spelling seen, for display
);

-- The free-text tail ("Right Knee; Soreness"). Normalised because it repeats
-- endlessly: 145 parsed reports produced 18,383 rows but only 691 distinct
-- strings, mean length 32 chars. Inline TEXT would cost ~36 B/row for nothing.
CREATE TABLE injury_report.ir_reason (
    reason_id   INTEGER PRIMARY KEY,
    category_id SMALLINT NOT NULL REFERENCES injury_report.ir_reason_category,
    detail      TEXT NOT NULL DEFAULT '',
    UNIQUE (category_id, detail)
);

CREATE TABLE injury_report.ir_team (
    team_id      SMALLINT PRIMARY KEY,   -- surrogate; nba_team_id kept for joins
    nba_team_id  BIGINT NOT NULL UNIQUE,
    tricode      TEXT NOT NULL UNIQUE,
    report_name  TEXT NOT NULL UNIQUE    -- exactly as printed: "LA Clippers"
);
```

### 2.2 Report coverage

```sql
-- One row per report that was successfully parsed and loaded. This is what
-- separates "the player was not listed" from "we have no report for that hour"
-- -- without it, every gap is ambiguous and every as-of query is a guess.
CREATE TABLE injury_report.ir_report (
    report_id    INTEGER PRIMARY KEY,
    observed_at  TIMESTAMPTZ NOT NULL UNIQUE,  -- report_datetime_utc, the ONLY ordering key
    season_year  SMALLINT NOT NULL,
    era          SMALLINT NOT NULL,            -- legacy_3 / bubble_3 / hourly_24 / quarter_96
    n_rows       SMALLINT NOT NULL,
    parse_ok     BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX ON injury_report.ir_report (season_year, observed_at);

-- Local copy of the game dimension, mirroring line_history's lh_game. ~10,000
-- rows, well under 1 MB, and it is what lets every read join locally instead of
-- reaching across to Supabase. tipoff_utc is also what section 7 clamps spans
-- against, so it has to be here regardless.
CREATE TABLE injury_report.ir_game (
    game_id     TEXT PRIMARY KEY,
    game_date   DATE        NOT NULL,
    season_year SMALLINT    NOT NULL,
    tipoff_utc  TIMESTAMPTZ NOT NULL,
    team_home   TEXT        NOT NULL,
    team_away   TEXT        NOT NULL
);
CREATE INDEX ON injury_report.ir_game (game_date);
```

`observed_at` is `report_datetime_utc` from the archive manifest. The S3 key is
**not** stored — `urls.s3_key()` derives it from that instant, so storing it
would be 70 bytes of duplicated state per row.

### 2.3 The fact table

```sql
CREATE TABLE injury_report.ir_status_span (
    game_id      TEXT        NOT NULL,
    player_id    BIGINT      NOT NULL,   -- ids reach 1.97e9, near the INTEGER limit
    team_id      SMALLINT    NOT NULL REFERENCES injury_report.ir_team,
    season_year  SMALLINT    NOT NULL,

    valid_from   TIMESTAMPTZ NOT NULL,   -- observed_at of the report that first showed it
    -- LEAST(next observation, tipoff_utc). NOT NULL on purpose: no span may
    -- outlive its own game. See section 7.
    valid_to     TIMESTAMPTZ NOT NULL,

    status_id    SMALLINT    REFERENCES injury_report.ir_status,   -- NULL = removed from report
    -- NOT NULL: 38.7% of `Out` is not an injury (section 2.1), so a status
    -- without its reason is not merely less informative, it is misleading.
    reason_id    INTEGER     NOT NULL REFERENCES injury_report.ir_reason,

    first_report_id INTEGER  NOT NULL REFERENCES injury_report.ir_report,
    mins_to_tip  INTEGER     NOT NULL,   -- (tipoff_utc - valid_from); negative = after tip
    is_pregame   BOOLEAN     NOT NULL,   -- mins_to_tip > 0

    PRIMARY KEY (game_id, player_id, valid_from, season_year)
) PARTITION BY LIST (season_year);
```

Design notes, each with a reason:

* **`valid_to` is precomputed at load.** The backfill is immutable history, so
  closing the interval at write time is free and turns every as-of read into an
  indexable range containment instead of a `DISTINCT ON ... ORDER BY DESC`.
* **`season_year` trails the primary key** for the same reason as `lh_line`:
  Postgres requires the partition key inside the unique constraint, but `game_id`
  must stay the leading column so "everything for game X" uses the index prefix.
* **LIST partition by season** so a season can be dropped instantly if the cap is
  ever approached, and season-filtered reads prune without a secondary index.
* **`mins_to_tip` / `is_pregame` are NOT NULL and denormalised on purpose.** This
  is exactly the `lh_line` precedent: they are the only cheap guard against
  building a feature from a report published *after* tip-off. Recomputing them at
  query time is one join away from being forgotten once.
* **`status_id NULL` means "no longer listed"**, closing the previous span. It is
  defensive: pre-tipoff removal was never observed (section 7), but the schema
  must be able to express it if it ever happens.
* **`valid_to` is clamped at `tipoff_utc`** so no span can outlive its game.
  This makes "never read a report published after tip" a property of the data
  rather than a rule every query has to remember.

```sql
-- "which teams had not filed yet, as of T" -- same interval treatment.
CREATE TABLE injury_report.ir_filing_span (
    game_id     TEXT        NOT NULL,
    team_id     SMALLINT    NOT NULL REFERENCES injury_report.ir_team,
    season_year SMALLINT    NOT NULL,
    valid_from  TIMESTAMPTZ NOT NULL,
    valid_to    TIMESTAMPTZ NOT NULL,    -- clamped at tipoff, as above
    submitted   BOOLEAN     NOT NULL,
    PRIMARY KEY (game_id, team_id, valid_from, season_year)
) PARTITION BY LIST (season_year);
```

### 2.4 Resolution audit

```sql
-- Persisted so name -> player_id resolution is reproducible and reviewable,
-- rather than a fuzzy match re-run differently on every load.
CREATE TABLE injury_report.ir_player_alias (
    raw_name    TEXT     NOT NULL,       -- "Gilgeous-Alexander, Shai", verbatim
    team_id     SMALLINT NOT NULL,
    season_year SMALLINT NOT NULL,
    player_id   BIGINT   NOT NULL,
    method      TEXT     NOT NULL,       -- exact | normalised | manual
    PRIMARY KEY (raw_name, team_id, season_year)
);

-- Anything that did not resolve. A load that silently drops players is the
-- failure mode that matters most here, so it is recorded, counted and surfaced.
CREATE TABLE injury_report.ir_unresolved (
    raw_name        TEXT NOT NULL,
    raw_team        TEXT NOT NULL,
    season_year     SMALLINT NOT NULL,
    occurrences     INTEGER NOT NULL,
    first_report_id INTEGER NOT NULL,
    PRIMARY KEY (raw_name, raw_team, season_year)
);
```

### 2.5 What the detail field costs, and why it stays

The category is settled above. The free-text tail (`Right Knee; Soreness`) is a
separate judgement, because unlike the category it has no proven use yet.

| | |
|---|---|
| Distinct detail strings | 1,474 over 9,624 rows (109 reports) |
| Mean length | 18.7 chars |
| Projected full-archive cardinality | ~10-20k distinct |
| **Marginal cost of keeping it** | **~1.5 MB** (dimension + index) |

Keep it, for two reasons:

1. **1.5 MB against ~700 MB of headroom** is not a trade-off worth thinking about.
2. Dropping it is not truly reversible in practice. The PDFs are in S3, so nothing
   is *lost* — but recovering it means re-parsing ~45,000 PDFs, hours of work, at
   whatever future moment someone wants body-part features. Paying 1.5 MB now to
   avoid that is obviously right.

**Do not parse it further yet.** The structure is clearly `site; pathology`
(`Left Ankle; Sprain`, `Right Hamstring; Strain`, `Left Knee; Surgery`), so
splitting it later is easy. Splitting it *now* would bake in a normalisation
nobody has validated against a real feature requirement. Store the raw string,
defer the taxonomy.

---

---

## 3. Size estimate

Inputs, all measured rather than assumed:

| Quantity | Value | Source |
|---|---:|---|
| Games 2018-19 → 2025-26 | 10,850 | `nba_games`, live query |
| Games covered by the archive (from 2018-12-17) | ~10,300 | archive plan §9 |
| ...**in scope**, i.e. from 2019-12-18 (section 6) | **~9,100** | excludes ~1,240 legacy-era games |
| Players listed per game | 10.0 / 11.6 | measured, two full days |
| Transitions per (game, player) | 0.18 (24/day), 0.40 (96/day) | measured |
| Distinct players ever | ~2,300 | `nba_players` ∪ `nba_injuries` |
| Reports in the archive | ~47,000 | archive plan §9 |
| ...**in scope** | **~44,700** | excludes ~2,300 legacy-era reports |

| Table | Rows | Heap | +Indexes | Total |
|---|---:|---:|---:|---:|
| `ir_status_span` | ~114,000 | 9 MB | 8 MB | **17 MB** |
| `ir_report` | ~44,700 | 2 MB | 2 MB | **4 MB** |
| `ir_filing_span` | ~27,000 | 2 MB | 1 MB | **3 MB** |
| `ir_player_alias` | ~5,000 | <1 MB | <1 MB | **1 MB** |
| `ir_game` | ~10,300 | <1 MB | <1 MB | **1 MB** |
| `ir_reason` + dims | ~5,000 | <1 MB | <1 MB | **1 MB** |
| **Total** | | | | **≈ 27 MB** |

Row width for `ir_status_span`: 23 B tuple header + 1 B null bitmap + `game_id`
(~11 B) + 8 + 2 + 2 + 8 + 8 + 2 + 4 + 4 + 4 + 1, padded ≈ **80 B** (`player_id` is BIGINT).

**Growth: ~3 MB per future season** (~1,320 games × ~11 players + ~24,000 reports).

Aiven currently holds 236 MB of a 1 GB cap that covers heap, indexes *and* WAL.
27 MB is comfortably inside the headroom, and ~20 seasons of growth still fits.

**The snapshot alternative, for contrast:** ~5.8 M rows ≈ 440 MB heap + ~250 MB
indexes ≈ **690 MB**, taking Aiven to ~930 MB. It does not fit. The change-log is
not merely tidier — it is the only form that fits the instance we have.

**Decision: Aiven.** Measured, not assumed:

| | Supabase | Aiven |
|---|---:|---:|
| Used | **383 MB** | 236 MB |
| Largest tenant | `nba_players` 267 MB | `line_history` 228 MB |
| Growth | **~33 MiB/season** (`nba_players`, ~37k rows/season at 973 B/row) | ~static |

Supabase is both the tighter database *and* the one growing, so its headroom is
the scarce resource. The only argument for putting the schema there — being
co-located with `nba_games` / `nba_players` for resolution — is answered for
~1 MB by `ir_game` (section 2.2), copying the game dimension across exactly as
`lh_game` already does. No query in this repo joins across databases anyway.

---

## 4. What is deliberately not stored

The PDFs stay in S3 and are the system of record, so anything recoverable from
them or derivable from a key is left out:

| Not stored | Why |
|---|---|
| Raw PDF / extracted text | S3 holds it; `urls.s3_key()` addresses it |
| S3 key | derived from `observed_at` |
| Player name, team name, matchup, game date, game time | derivable from `player_id` / `team_id` / `game_id` |
| Full reason string per row | normalised into `ir_reason`; the FK stays (see 2.1) |
| Unchanged repeat observations | the whole point of §1 |
| `Previous Status` (legacy 9-col era) | the change-log *is* the previous status |
| Per-report row counts per team | derivable |

The one deliberate redundancy is `mins_to_tip` / `is_pregame`, kept for the
leakage reason in §2.3.

---

## 5. Pipeline

Three phases, each independently re-runnable, mirroring the two-phase structure
already used by the archive downloader.

**Phase A — parse.** S3 PDF → tidy rows. Reuses `read_injury_report`.
Output is a per-season Parquet file in S3 (`injury_reports/parsed/season=.../`),
*not* a direct DB write, so re-resolution never requires re-downloading and the
DB load stays a pure function of the Parquet.

Phase A **skips any report whose page text contains `Previous Status`** — the
legacy 9-column layout, excluded by section 6. Keying the skip on the text rather
than on the 2019-12-18 cutover date means a stray legacy-format file can never be
silently mis-parsed into the modern column order.

**Phase B — resolve.** Attach `game_id`, `team_id`, `player_id`.

* `game_id` ← (`Game Date`, `Matchup` tricodes) against `nba_game_time_index`.
  Both sides are ET calendar dates, so they compare directly, and `game_time_utc`
  from the same row is the `tipoff_utc` that section 7 clamps spans against.

  **Prerequisite:** that table does not exist in the database yet — the schema is
  defined in `game_time_index/create_db/` but is unpopulated, and `nba_games`
  carries no tipoff time at all. Populating it (or reading tipoffs from
  `fetch_nba_schedule.py`, which archive plan section 8.5 verifies has explicit
  UTC and 100% coverage) has to happen before Phase B can run.
* `team_id` ← the `Team` column verbatim, via `ir_team.report_name`. A fixed
  30-row map; needs manual aliases for the forms the NBA actually prints
  (`LA Clippers`, `Philadelphia 76ers`).
* `player_id` ← `"Familyname, Firstname"` + team, resolved **against that game's
  own roster** rather than against all players. Once `game_id` and `team_id` are
  known, the candidate set collapses from ~2,300 players to ~15 per team, which
  makes the match near-trivial and mis-matches nearly impossible.

  The candidate set is `nba_players` **∪** `nba_injuries` *for that `game_id`*.
  The union is what makes it work: `nba_players` holds box-score rows, so a
  player who is `Out` — precisely the ones the injury report is about — never
  appears there. `nba_injuries` carries that game's inactive list and covers
  exactly that gap. Together they are the roster that was actually present.

  Three fallbacks, in order, each recorded in `ir_player_alias.method`:

  1. exact normalised match within the game's roster;
  2. same, widened to the team's roster for the whole season (catches a player
     who is listed on the report but is on neither list for that game — a G
     League assignee, or a just-signed player);
  3. otherwise `ir_unresolved`, never a guess.

  Normalisation must handle suffixes (`Jr.`, `III`), accents, apostrophes and
  hyphens. Ambiguity — two players normalising identically on one roster — is an
  **error, not a coin flip**, and also goes to `ir_unresolved`.

**Phase C — load.** Sort each (game_id, player_id) by `observed_at`, emit a span
whenever the (status, reason) pair changes, close the previous span, compute
`mins_to_tip` against `tipoff_utc`. Idempotent: `ON CONFLICT DO NOTHING` on the
primary key, so a re-run of an overlapping window is safe.

Ordering rule, inherited from archive plan §8.4: **order and join on
`observed_at` (UTC) only.** Never an ET wall-clock string, never an ET date.
On a fall-back date the ET clock repeats 01:00–01:59, and the UTC offset in the
canonical key (`-0400` first, `-0500` second) is what disambiguates. This is why
`ir_report.observed_at` is TIMESTAMPTZ and why the ET date appears nowhere in
this schema.

---

## 6. Scope: the legacy era is excluded

> **Superseded (2026-09): the legacy era is now parsed.** It holds betting-data
> games the feature pipeline needs (406 in Oct–Dec 2019 and the 2019 playoffs),
> so `fetch_data/injury_reports/legacy_injury_report.py` reads it by column
> position under each page header instead of by token pattern. A scan of all 788
> archived reports up to 2019-12-18 found **four** legacy layouts, not one:
>
> | Layout | Columns after Player Name | Reports | Dates |
> |---|---|---|---|
> | A | Category, Reason, Current Status, Previous Status | 685 | 2018-12-17 → 2019-11-14 |
> | B | Reason, Current Status, Previous Status | 16 | 2019-11-13, 11-15 → 11-19 |
> | C | Current Status, Reason, Previous Status | 80 | 2019-11-20 → 12-16 |
> | D | Current Status, Reason, Previous Status, Previous Reason | 4 | 2019-12-16 → 12-17 |
>
> Validation: every status parsed is in the closed vocabulary; on 60 modern
> reports the geometric reader matches the modern reader on 98.5% of rows (every
> difference is reason text the modern reader truncates); and against box scores,
> 0.1% of 9,355 legacy `Out` rows played (Questionable 42–59%, Probable 84–91%),
> the same profile as the modern era. Layout A's `Category` is folded into
> `Reason` as `Category - Reason`, and `G League Team` maps to `G League`.
> Previous-status columns are dropped. The original decision is kept below for
> the record.

**Decision: the archive loads from 2019-12-18 onward.** The 9-column era
(2018-12-17 → 2019-12-17) is left in S3, parsed by nobody, and is not in scope.

It is excluded because the parser cannot read it and because it would not earn
its keep if it could. Measured against 5 real reports from that era, 450 player
rows:

| | current parser |
|---|---:|
| Players dropped | 14 (3.1%) |
| Rows with a wrong status | 216 (48%) |
| Rows with a corrupted game date | 223 |

```
legacy (→2019-12-17):  Game Date | Game Time | Matchup | Team | Player Name | Category | Reason | Current Status | Previous Status
modern (2019-12-18→):  Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason
```

Two extra columns, and status/reason in the opposite order.

**Cost of the exclusion:** ~2,300 reports and roughly one and a half seasons of
history, at 3 reports/day with median staleness ~11 h. Archive plan section 2
already shows T-12h coverage there is 0.1%, so it could never have supported the
short-horizon features the modern era does.

**Two things keep the door open**, at no cost now:

* The PDFs are already in S3, so this is a parsing decision, not a collection
  one. Nothing expires.
* `ir_reason_category` exists as a separate dimension from `ir_reason.detail`
  specifically because the legacy `Category` column maps straight onto it. If the
  era is ever loaded, it fits the schema as-is rather than forcing a migration.

When the branch is built it should key on `"Previous Status" in page_text`, not
on the date — the text is self-describing and cannot drift.

---

## 7. Absence, disappearance and "healthy"

The worry: a player is `Doubtful` an hour before tip, vanishes from the report
later, and an as-of query keeps reporting `Doubtful` forever. Measured on 120
consecutive reports covering 31 games and 334 (game, player) pairs:

| | 2025-01-29 (24/day) | 2026-02-11 (96/day) |
|---|---:|---:|
| Pairs observed before tipoff | 160 | 174 |
| **Ever removed while the game was still upcoming** | **0** | **0** |
| Game stays listed until | tip **+270 min** | tip **+285 min** |

**Pre-tipoff disappearance did not happen once.** What does happen is that the
whole game drops off the report about 4.5 h *after* it started — which is the
disappearance you noticed, and it is never inside a prediction window.

So the stale-`Doubtful` risk is real but it comes from the *open interval*, not
from disappearance: a span with `valid_to = NULL` reads as "still doubtful,
indefinitely". Hence the fix in section 2.3 — **`valid_to` is clamped at
`tipoff_utc` and is `NOT NULL`**. A query at any instant after tip returns no
span at all, rather than a confidently wrong one, and the invariant is enforced
by the data instead of by everyone remembering a rule.

### Statuses mostly do resolve, but not entirely

The last status strictly before tipoff:

| Last pre-tip status | 2025-01-29 | 2026-02-11 |
|---|---:|---:|
| Out | 130 | 134 |
| Available | 15 | 33 |
| Questionable | 12 | 4 |
| Doubtful | 0 | 3 |
| Probable | 3 | 0 |
| **Still unresolved at tip** | **15 (9.4%)** | **7 (4.0%)** |

Teams do convert most `Questionable`/`Doubtful` into `Out` or `Available` before
tip, and the 15-minute cadence resolves more of them than the hourly one did.
But **4–9% are genuinely still unresolved at tipoff.** That residue is not a data
defect to be cleaned up — it is real uncertainty that existed at prediction time,
and flattening it into a binary would be inventing information the NBA never
published. `Questionable at T-15m` should reach the model as its own state.

### Do not add a `healthy` status

Absence should be **derived at read time, not stored**. Two reasons:

1. It would be the largest table in the schema for no information: every rostered
   player × every game is ~350,000 rows against ~128,000 real ones, all of them
   reconstructible.
2. **"Not listed" does not mean healthy.** It means *no designation was filed*.
   For a rostered player that implies available; for someone on a two-way, in the
   G League, or not on the roster at all, it implies nothing.

The read must therefore distinguish three cases, and the schema exists to keep
them apart:

| Situation | Meaning | Value |
|---|---|---|
| A span covers *T* | reported status | that status |
| No span, **and** `ir_report` has a report at *T*, **and** `ir_filing_span.submitted` is true for that team | no designation filed for a listed team | **available** |
| No span, **and** no report at *T*, **or** the team had not filed | nothing was known | **NULL** |

Case 3 is the trap. `NOT YET SUBMITTED` is frequent — 220 and 887 rows on the two
sampled days — and treating "we have no report" as "everyone is healthy" is
exactly the shape of the rotation-depth availability leak: an unknown silently
recoded as a favourable known. It must stay `NULL` and propagate as `NULL`.

This is why `ir_report` and `ir_filing_span` are not bookkeeping. They are the
only things that make case 2 distinguishable from case 3.

### A related trap in the PDF itself

The report's `Game Time` column prints `10:30 (ET)` with **no AM/PM**. It is not
a usable tipoff: decoding it needs the outside knowledge that NBA games tip
between 11:00 and 22:30 ET. I got this wrong on the first pass of the measurement
above and it moved every number in the table. **Take `tipoff_utc` from the
schedule, never from the PDF** — which is also what archive plan section 8.5
requires, since only the schedule carries explicit UTC.

---

## 8. Reading it back

```sql
-- Every player's officially reported status for game G as known at instant T.
SELECT s.player_id, s.team_id, st.code AS status, rc.code AS category, r.detail,
       EXTRACT(EPOCH FROM (%(t)s - s.valid_from)) / 60 AS report_age_minutes
FROM   injury_report.ir_status_span s
JOIN   injury_report.ir_status          st ON st.status_id = s.status_id
LEFT   JOIN injury_report.ir_reason      r ON r.reason_id  = s.reason_id
LEFT   JOIN injury_report.ir_reason_category rc ON rc.category_id = r.category_id
WHERE  s.game_id = %(game_id)s
  AND  s.valid_from <  %(t)s                      -- strictly <, never <=
  AND  s.valid_to  >= %(t)s                      -- always closed; never outlives tipoff
  AND  s.status_id IS NOT NULL;
```

Three rules carried over from archive plan §8, and they apply here unchanged:

1. **Strictly `<`.** A report stamped exactly at the cutoff is published *at* that
   instant. 67.3% of tipoffs are on the hour, so at T-30m two thirds of the
   sample lands exactly on a report stamp — `<=` leaks on all of them.
2. **Always emit `report_age_minutes`.** It is ~30 min in the 24/day era, ≤15 min
   in the 96/day era and up to ~11 h in the legacy era. A model that cannot see
   the age cannot learn that early-era features are stale.
3. **Never fall forward.** No span before the cutoff means `NULL`, not the next
   report. And check `ir_report` before concluding a player was healthy: no row
   may mean no report was published, which is not the same thing.

---

## 9. Open questions

1. **Player-resolution accuracy is the main unknown.** Scoping to the game's own
   roster (section 5) should make it easy, but the rate is unmeasured until
   Phase B runs. `ir_unresolved` is designed to make every failure visible and
   counted; the thing to agree is what rate is acceptable before loading, and
   whether fallback 2 (season-wide) should be auto-accepted or reviewed.
2. **Does the change-log need materialised snapshots for convenience?** A view at
   fixed horizons (T-24h, T-6h, T-30m) would be larger but simpler to join in
   training. Deferred — cheap to add later, and it should be driven by what the
   feature code actually wants rather than guessed now.
3. **Team tricode stability** across 2019-2026 is assumed, not verified.
4. **The 4–9% of players still unresolved at tipoff** (section 7) need a
   modelling decision, not a data one: `Questionable` has to reach the model as
   its own state rather than being collapsed to available or out.

---

## 10. What was built, and how to run it

Implemented on Aiven, schema created and verified end-to-end.

| File | Phase |
|---|---|
| `postgre_db/injury_report_aiven/schema.py` | DDL, dimensions, season partitions |
| `postgre_db/injury_report_aiven/parse.py` | A — PDF bytes -> tidy rows |
| `postgre_db/injury_report_aiven/resolve.py` | B — game / team / player ids |
| `postgre_db/injury_report_aiven/spans.py` | C — observations -> validity spans |
| `postgre_db/injury_report_aiven/load.py` | D — dimension upserts, fact inserts |
| `postgre_db/injury_report_aiven/ingest.py` | orchestration, one season at a time |
| `postgre_db/injury_report_aiven/fetch.py` | as-of reads |
| `scripts/injury_reports/load_injury_reports_to_aiven.py` | CLI |
| `tests/test_injury_report_db.py` | 28 pure tests |

```bash
# one-off
python scripts/injury_reports/load_injury_reports_to_aiven.py --create-schema

# what is available to load
python scripts/injury_reports/load_injury_reports_to_aiven.py --list-seasons

# shape a season without writing (still hits network for dimensions)
python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2023-24 --dry-run

# load it
python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2023-24
```

### Verified end-to-end

Two complete game-days staged as a local archive and ingested for real:

| | 2025-01-29 | 2026-02-11 |
|---|---:|---:|
| Reports | 24 | 96 |
| Observations | 2,902 | 13,302 |
| Spans | **187** | **243** |
| Compression | 15.5x | 54.7x |
| Player resolution | **100.00%** | **100.00%** |

Matching the section 1 predictions (15.4x / 54.5x) almost exactly. As-of reads
behaved correctly at every horizon: 0 rows at T-24h (no report yet, no
fall-forward), 3 questionable at T-6h resolving to available by T-1h, and **0
rows at T+3h** -- the tipoff clamp.

**The fact tables were then reset to empty on purpose.** Spans may only be built
from a contiguous run of reports, and those two days covered only one of the two
report-days each game appears in, so their earliest spans started later than the
truth. Leaving them would have been a trap: `ON CONFLICT DO NOTHING` would
preserve the wrong rows through a later full load. The schema and its seeded
dimensions remain in place.

### Resolution accuracy, measured

Over 140 reports / 18,383 rows spanning 2019-12 to 2026-04:

| Tier | Rows |
|---|---:|
| `game_roster` (full name, that game's roster) | 17,483 |
| `game_roster_initial` (surname + initial) | 819 |
| `season_roster` (widened to the season) | 48 |
| **unresolved** | **1** (99.99%) |

The single failure is `Kanter, Enes` in 2019 -- who legally changed his name --
and it was refused rather than guessed, which is the intended behaviour.

The `game_roster_initial` tier is not a nicety. `nba_players` stores
`familyname` **only for season 2025**; every earlier season has it null and
carries an abbreviated `player_name` ("B. Adebayo"). Without that tier, stars
including Adebayo, Herro and Kuzma failed to resolve.

### Two prerequisites still open

1. **Nothing has been downloaded.** `injury_reports/raw/` in S3 is empty; the
   manifests record discovery only (2023-24: 5,441 reports found, `sha256` null
   on all 6,736 rows). `backfill_injury_reports.py` must complete its download
   phase before this loader has anything to read.
2. **Playoff tipoffs are missing.** `fetch_schedules` returns the regular season
   only -- verified: season 2025 ends 2026-04-12, no playoff or play-in games.
   `nba_games` has the 590 playoff and 37 play-in games but no tipoff column.
   The fix is to run `sync_game_time_index`, which fetches `gameTimeUTC` per game
   from the NBA CDN; `load_game_dimension` already reads that table when it
   exists and falls back cleanly when it does not. Until then, playoff rows are
   counted as `games_without_tipoff` and skipped rather than loaded unclamped.

---

## 11. Parser verified against the real corpus

**44,626 PDFs are downloaded** across all 8 seasons. Verified two ways: a
stratified sample of **2,322 reports / 139,964 player rows** read straight from
S3, and a full-season dry run over all **5,823** reports of 2023-24.

| Audit check | Before | After |
|---|---:|---:|
| Fully clean reports | 99.3% | **100.0%** (2,322/2,322) |
| Bad player name | 3 | **0** |
| Null status | 14 | **0** |
| Bad status | 17 | **0** |
| Bad game date | 254 | **0** |
| Bad time / matchup | 36 / 36 | **0 / 0** |
| **Bad team** | *not checked* | **0** |

| 2023-24 dry run | Before | After |
|---|---:|---:|
| Reports parsed / failed | 5,817 / 6 | **5,823 / 0** |
| Observations | 328,152 | **350,081** |
| Spans | 16,843 | **17,924** |
| Player resolution | 93.71% | **99.97%** |

### Five bugs, all pre-existing, all found only against the real corpus

1. **The `len == 7` positional shortcut fired on rows that were not 7-column
   rows.** A row starting a new game carries Game Time, Matchup and Team inline,
   so a six-field row with a wrapped reason also has seven elements; mapping it
   by position shifted every column left. Guarded by `is_positional_row`.
2. **A long reason spills into its own block.** The old `joinwith_next` handled
   only 2-element blocks; it now fires whenever a block's last element is a bare
   status, since a row always ends with its reason.
3. **The classification loop overwrote already-filled columns.** A trailing
   reason fragment that is one capitalised word (`Reconditioning`) matches the
   status pattern and clobbered the real status. Only reachable after (1), and
   it briefly regressed bad-status from 17 to 916 before being caught.
4. **A reason wrapping across a page break was orphaned.** `joinwith_next` was
   reset per page, so the tail arrived alone on the next page and parsed as a
   phantom player -- `"Knee Bone Bruise, Left Heel"` matches the name pattern and
   `"Contusion"` then matches the status pattern. The join now carries across
   pages, and a first-of-page block with no status is returned to the row above.
5. **Reason categories were filed as the Team.** `team_pattern` matches any two
   capitalised words, so a bare reason (`Personal Reasons`, `League Suspension`,
   `Not With Team`, `Concussion Protocol`, `Trade Pending`) became the Team, and
   `ffill` propagated it down the column. This alone accounted for **6.3% of
   rows** and was the whole of the 93.71% resolution rate. Fixed by extending the
   sequence rule: once the status is seen, everything after it is the reason.

### Lessons about the verification itself

* **Sampling evening reports was not sampling the corpus.** Bugs 1-4 only occur
  in block shapes that dense evening reports do not produce. The first
  20-report check and the 404-report status scan both passed while all five bugs
  were live.
* **The audit had a blind spot that hid the largest bug.** It validated name,
  status, date, time and matchup, and reported "100% clean" while 6.3% of rows
  carried a reason category as their team. An audit only proves what it checks;
  the Team column is now checked against the 31 known names.
* **The `CHECK` on `status_id` did its job.** Bugs 3 and 4 surfaced as a load
  *failure*, not as corrupt rows. A schema that refuses the unexpected is what
  turned a silent data problem into a visible one.

### Corrections to earlier claims in this document

* **`Current Status` is closed at five values, and that still holds** -- but the
  evidence in section 2.1 did not. `Return to Competition`, `Reconditioning` and
  `Contusion` all appeared as apparent statuses; every one was a parser artifact
  of bugs 1, 3 and 4, not an NBA value.
* **Two manifest classes are not parser problems.** `date_mismatch` (164
  reports) were validated, rejected and *deliberately never uploaded*
  (`download_status = invalid`) -- absent from S3 by design, not lost.
  `image_only` (188) parse fine. Reports yielding zero player rows (~8%) are
  legitimate: every team is `NOT YET SUBMITTED`, which the filing table captures.

### Still open

* **104 unresolved names in 2023-24** (0.03%), all genuine mid-season trades --
  the report lists a player under his new club before he has appeared for them.
  Fallback 2 (season roster) catches most; these are the residue.
* **1 game dropped for want of a tipoff**, pending `sync_game_time_index`.

---

## 12. Tipoffs, removals and filings -- fixed before the first real load

### Tipoffs: the gap was 131 games, not "the playoffs"

Section 10 said the schedule feed returns the regular season only. **That was
wrong.** `fetch_schedules` carries playoffs and play-in for every season from
2019-20 to 2024-25, with zero null tipoffs. Only the **2025-26 feed is frozen**,
and it is broken in two ways:

* **122 games are absent**: 85 playoff, 6 play-in, 30 NBA Cup knockout-week and
  the Cup final.
* **9 games carry a stale date** because they were rescheduled after the feed
  froze -- DEN@MEM reads 2026-01-25 15:30 but was played 2026-03-18. Loaded
  as-is, `ir_game` would have stored a wrong tipoff and clamped every span
  against it. 2019-20 to 2024-25 have zero such disagreements.

`sync_game_time_index` cannot fill this from the development machine: **every
`cdn.nba.com` URL returns 403** there, including the static schedule, and
`stats.nba.com` times out. It would also write each game's full boxscore JSON
into Supabase, the tighter database.

Instead, those 131 games take their tipoff from the **injury reports
themselves**, tagged `ir_game.tipoff_source = 'injury_report'`. The report
prints a 12-hour clock with no AM/PM, decodable because NBA games tip between
11:00 and 22:30 ET. Validated against the 2024-25 schedule: **1,317 of 1,320
games agree exactly, including all 83 playoff and 6 play-in games.** The three
misses were rescheduled starts -- POR@DAL read `08:30 (ET)` until 13:30 on game
day and `07:30 (ET)` afterwards -- so the tipoff is taken from the **latest**
report listing the game. Schedule rows whose date disagrees with `nba_games`
are discarded first.

### Removals: the stale-Doubtful case does happen

Section 7 reported zero pre-tipoff removals, from two regular-season days.
Over 28 sampled game-days, **a player drops off a filed team's list before tip
in ~2% of games, usually 1-3 players** (MIA@CHI went from 15 to 12 listed
players on game morning). The tipoff clamp alone left those players reading
their last status until tip. The loader now emits a NULL-status "not listed"
observation at the first report he is missing from -- only when his team has
filed, since absence under `NOT YET SUBMITTED` is unknown, not removal.

### Filings: "submitted" was never recorded

The loader recorded only `NOT YET SUBMITTED` observations, so every filing span
stayed open until tip even after the team filed -- collapsing section 7's case 2
("filed, nobody listed") into case 3 ("unknown"). It now records, for every
report and every listed game, `submitted` for both teams. A game can also vanish
from the report once both teams file empty lists (seen in the 2026 East Finals);
its first unlisted report marks both teams as filed.

Every parsed report is now registered in `ir_report`, including reports listing
no players, since they still prove the feed was live.

### "Games dropped for want of a tipoff" was counting non-games

The first dry runs reported 87 (2023-24) and 97 (2025-26). Almost all are not
NBA games: **82 per season are Summer League**, plus postponed dates (GSW@UTA
and DAL@GSW in January 2024) and "if necessary" playoff games never played
(PHX@MIN, 2024-04-30). The count is now split into real NBA games lacking a
tipoff and listed non-games.

### Also changed

* `player_id` is **BIGINT**: NBA person ids already reach 1,966,938,209, within
  9% of the INTEGER ceiling. Applied as an in-place migration.

---

## 12. Loader hardening before the first full load

Four gaps found while trying to obtain playoff tipoffs, all fixed and tested.

### 12.1 Tipoffs: the game-time sync cannot run from here

`sync_game_time_index` fetches `cdn.nba.com` boxscores, and **every
`cdn.nba.com` URL returns 403 from this machine** -- boxscores, today's
scoreboard, even the static schedule. `stats.nba.com` times out and other
`data.nba.com` paths 403 as well. It would also have written full boxscore JSONB
into Supabase, the tight database.

It is not needed. The earlier claim that the schedule feed lacks playoffs was
wrong: **`fetch_schedules` carries playoffs, play-in, preseason and the Cup for
2019-20 through 2024-25 with zero null tipoffs.** Only the 2025-26 file froze,
leaving two problems:

| 2025-26 schedule problem | Games | Fix |
|---|---:|---|
| Missing (play-in, playoffs, Cup knockout week) | 122 | tipoff from the latest injury report listing the game |
| **Stale date** -- rescheduled after the feed froze | 9 | dropped against `nba_games`' played date, then as above |

The report prints `08:00 (ET)` with no AM/PM; NBA games tip 11:00-22:30 ET, so
hour 11 is morning and 1-12 afternoon or evening. **Validated against the
2024-25 schedule: 1,317/1,320 exact (99.77%), and 100% of 83 playoff and 6
play-in games.** The 3 misses were rescheduled starts -- the report itself shows
the change (POR@DAL read 08:30 until 13:30, then 07:30) -- which is why the
*latest* report is used. `ir_game.tipoff_source` records `schedule` vs
`injury_report` for every game.

Stale rows were checked across all seasons: 9 in 2025-26 (e.g. MIA@CHI listed
2026-01-08, played 2026-01-29), **zero** elsewhere. Kept, they would have stored
a wrong tipoff and dropped every span for the game.

### 12.2 Removals: the stale-Doubtful case was real after all

Section 7's "0 pre-tip disappearances" came from two regular-season days. Across
a larger sample, **players drop off a filed team's list before tipoff in ~2% of
games, usually 1-3 players** (e.g. MIA@CHI going from 15 listed to 12 on game
morning). The loader never emitted anything for them, so their last span ran to
tipoff -- exactly the "listed Doubtful, then gone" failure. Now a NULL-status
observation is derived at each report where a listed player is absent from a
**filed** team's list; absence while the team is `NOT YET SUBMITTED` is left as
unknown, not removal. Verified end to end in Aiven: a player read `out` one
minute before his removal and was absent one minute after, with both teams
marked filed.

### 12.3 Filing spans recorded only "not submitted"

`ir_filing_span` was fed only from `NOT YET SUBMITTED` rows, so every span ran to
tipoff even after the team filed -- collapsing section 7's case 2 into case 3.
Now every team of every listed game gets an observation per report: submitted
unless marked `NOT YET SUBMITTED`. A game can also **vanish once both teams file
empty lists** (2026 East Finals Games 2-3 left the report ~28 h before tip);
its first unlisted report marks both teams submitted. Every parsed report is
now registered in `ir_report`, including all-`NOT YET SUBMITTED` ones, since
they still prove the feed was live.

### 12.4 Listed "games" that are not games

Reports list fixtures that never become NBA games. They are skipped and counted
separately from genuine tipoff gaps:

* **Summer League** -- 82 in each of 2023-24 and 2025-26.
* **Postponed original dates** -- GSW@UTA and DAL@GSW (Jan 2024, after the death
  of assistant coach Dejan Milojevic); four January 2026 fixtures.
* **"If necessary" playoff games never played** -- PHX@MIN 2024-04-30 after a
  4-0 sweep.

### 12.5 Also changed

* `player_id` is **BIGINT** in `ir_status_span` and `ir_player_alias`. NBA ids
  already reach 1,966,938,209, within 9% of the INTEGER ceiling. Applied as an
  in-place migration.
* The write path was exercised for real (400 reports into Aiven, verified, then
  reset): removal spans, both filing states, BIGINT ids, zero invariant
  violations.
