# Lineup stints, projected minutes and a bottom-up game projection

Status (2026-09-23): **A-F implemented; phase G features are built behind a
default-off `lineup_features` flag, with leakage tests; a feature audit (§8.6)
located the signal in the defense and pace channels and fixed three defects.
The with/without campaign waits for the 2019-2020 backfill.** The paragraph below is the original
2026-09-18 status, kept for history. Written 2026-09-18 on
`feat/pregame-rotation-features` for the agent that will implement it. Every
API fact below was checked against the live `stats.nba.com` API that day (§2).
Anything not checked is marked **(unverified)**.

Implementation checkpoint (2026-09-19): the paced raw archive, manifest,
stint builder, validation, local status report and compact database schema
are implemented. A pilot archive is running; the first 112 complete games
passed stint validation after handling zero-duration rotation rows, rebound
descriptions, same-clock substitutions and corrected PBP actions. The 112 pilot games
were loaded into the default database and read back; their stint aggregates
reconcile **exactly** (100% of 224 team-games) with the `nba_players` box score
for points, FGA, 3PA, FTA, OREB, DREB and TOV. The walk-forward ridge
solver has synthetic leakage tests, a real-data smoke fit, a walk-forward
penalty-tuning command, an atomic Parquet cache builder and the ratings-only
go/no-go evaluator. The daily S3-backed fetch/build/load command is wired into
the finished-game workflow. The 112-game pilot is only a smoke test and does not
replace the specified 2021-2025 gate. Audit (2026-09-20) fixed three defects:
a restated cumulative rebound counter double-counted OREB (~4% of team-games),
the Phase C gate projected minutes for players long absent from the roster, and
the daily updater never retried a game stored as `failed`.

Ingestion checkpoint (2026-09-21): **play-by-play is complete.** All 10,191
finished games from 2018-19 on were imported from the `shufinskiy/nba_data`
season archive instead of being fetched: 162 MB on disk, ~10 minutes, and the
rebuilt payloads produce byte-identical stints in all 437 of the 440
API-fetched games that pass rotation validation (§4.7). That halves the API
backfill, which now owes **9,701 `GameRotation` calls and nothing else**. Two
client defects surfaced while diagnosing it: an HTTP 5xx with an empty body was
being counted as a throttle, costing 570 s per game and risking the circuit
breaker, and the endpoint class discarded the status code before it could be
read (§2.3, §2.4). `GameRotation` coverage is **not** complete before 2021
(§2.4). The rotation backfill, lambda selection, go/no-go C, and D-G remain
open.
Backfill checkpoint (2026-09-22): rotation is complete for **2021, 2022, 2023
and 2024**; 2025 is in progress and **2019, 2020 and 868 games of 2018 have not
been started**. 4,430 `GameRotation` calls remain. Play-by-play stays complete
at 10,191/10,191. Two operational facts measured that day: seasons 2022-2024
fetched at a clean **3.0 s/call** (pacer-bound), but the 2025 run degraded to
**~38 s/call** — a bimodal gap distribution with nothing between 10 s and 20 s,
which is the 30 s read timeout followed by the session reset, the 5 s backoff
and a fast retry, not a gradual slowdown. `--timeout 8` is the knob; the
circuit breaker never trips on this failure mode because every call eventually
succeeds. Stint building then ran for every archived season: **5,332 games and
160,359 stints**, at a 99.31-99.86% validation pass rate per season (2018-19
112, 2021-22 1,292, 2022-23 1,310, 2023-24 1,305, 2024-25 1,313), clearing the
99% phase-B threshold. The three failure modes are `bad_rotation_interval`,
`overlapping_player_intervals` and `points_mismatch`. `build_lineup_stints.py`
takes no run lock, so it is safe to run while a backfill is fetching. Storage sizing and free
space re-measured (§4.3); the feature-family switch was specified (§8.2).

See `scripts/lineups/README.md` for commands and current data layout.

Read these first:

- `CLAUDE.md`, for repo conventions.
- `docs/README_Training Data Processing.md`. Its "Adding Or Modifying Features"
  section is the checklist for L6.
- `docs/feature_engineering_overview.md`, for where feature families live.
- `src/nba_ou/data_processing/players/starter_history.py`. It follows the same
  temporal contract this plan needs.

---

## 0. Goal and non-goals

**Goal.** Build a per-game projection from players and lineups:

1. Decide who is likely to play, using the pre-game injury report.
2. Predict each player's minutes.
3. Rate each five-man lineup that will share the floor, using real on-court
   history (who played with whom, and against whom).
4. Combine these into expected possessions and points per team (and per player).
5. Emit the projection, its uncertainty and its components as `_BEFORE`
   features for both targets (`TOTAL_POINTS`, `LINE_ERROR`).

**Why.** Schema 2_6 already has starter-stability features and injury-status
features. What it lacks is any measurement of how **combinations** of players
perform, and any explicit minutes projection that redistributes an absent
player's minutes. The owner's working hypothesis, which is not established, is
that edge sits in fresh-news situations:

- a key player's first game out (a live lead: 53.2% OVER, CI [50.6, 55.7]);
- untested replacements;
- minutes restrictions after a return.

On a quiet night the projection will mostly reproduce the line.

**Non-goals**

- No new starter-stability features. Codex's `starter_history.py` covers that,
  and a "step 1" built only from box-score aggregates was explicitly rejected as
  adding no new information.
- Intermediate (per-snapshot) dataset support is **phase 2** (§9). Build for the
  closing-line dataset first.
- No change to model training code (`training_pipeline/`, `src/nba_ou/modeling/`)
  beyond what a new schema version needs.

---

## 1. Decisions already made (do not re-open without the owner)

| Decision | Choice | Reason |
|---|---|---|
| Lineup data source | Stints we build ourselves from `GameRotation` + `PlayByPlayV3` | Full control of timestamps (auditable leakage). Enables pairs, on/off and matchups. `LeagueDashLineups` is only a cross-check (§4.6). |
| Injury-report cut-off | **The repo's existing convention:** closing dataset reads the last injury report before tip-off (schema 2_4, `injury_status/report_state.py`); intermediate dataset reads it per snapshot | Keeps training and serving consistent with every other injury feature |
| Actual starters and minutes of the target game | **Never an input**, only a training target | They are outcomes. The starting five is announced about 30 min before tip, but using it would still differ between training and serving. |
| First version of on-floor time | **Minutes-weighted player ratings** (§7.1). The rotation template (§7.2) is optional and comes later. | Simplest thing that can be falsified |
| Backfill range | **2018-19 → present**, regular season + playoffs, **accepting that 2018-2020 rotation coverage is incomplete** | 2018-19 gives the ratings a prior season before the 2019 training start. **Correction 2026-09-21:** the original "verified available" rested on a handful of early-season games, which is the part that works. `GameRotation` is missing for scattered games in 2018-2020 (§2.4). Play-by-play is complete for all 10,191 games. |
| Play-by-play acquisition | **Imported from the `shufinskiy/nba_data` season archive; only `GameRotation` is fetched from the API** | Decided 2026-09-21. The archive republishes the same `PlayByPlayV3` payloads: for the 440 games already fetched, the rebuilt payloads give byte-identical stints in all 437 that pass rotation validation. Halves the backfill (~20.4k calls → ~10.2k, ~29 h → ~14 h) and halves rate-limit exposure. The archive has no rotation data, so the API is still required for that half. |
| Stint storage | **Local Parquet (`data/lineup_stints/`) is the default; Postgres is opt-in (`--load-db`)** | Decided 2026-09-20. Until go/no-go C and G pass, the data may never be used; Parquet keeps it reproducible from the raw archive with no database cost. Revisit when the features enter the pipeline. |
| Eventual database | **Aiven, if the gates pass.** Parquet stays the default until then | Decided 2026-09-22 with both databases measured (§4.3). Supabase is the default `DB_ENV` but has the least room; Aiven has ~550 MB under its hard cap for a load sized at ~75-110 MB. Ongoing maintenance (daily load, partition retention) is an open question to settle **after** go/no-go G, not before. |
| Feature-family switch | **A `lineup_features` flag on the dataset builders, default `False`** | Decided 2026-09-22. The family is unproven, so the campaign needs the same builder to emit a dataset with and without it (§8.2). Follows the existing `injury_report_features` precedent: same schema version, distinct filename suffix. |

---

## 2. What the API actually returns (verified 2026-09-18)

### 2.1 `GameRotation(game_id)` → `[HomeTeam, AwayTeam]` DataFrames

Columns: `GAME_ID, TEAM_ID, TEAM_CITY, TEAM_NAME, PERSON_ID, PLAYER_FIRST,
PLAYER_LAST, IN_TIME_REAL, OUT_TIME_REAL, PLAYER_PTS, PT_DIFF, USG_PCT`.

- One row per player **stint**.
- `IN_TIME_REAL` / `OUT_TIME_REAL` are in **tenths of a second from tip-off**.
  Regulation ends at `28800` (48 min × 600). Each overtime adds 3000.
- Stints **can span a period break**. Example: `12450 → 18340` runs from Q2
  into Q3. Cut segments at period boundaries yourself (multiples of 7200 up to
  28800, then +3000 per overtime).
- About 25–40 rows per team per game.
- Checked on `0022400061` (2024-25), `0021800001` (2018-19 opener) and
  `0041800406` (2019 Finals Game 6, final 110–114, which matches the real result).

### 2.2 `PlayByPlayV3(game_id)` → one DataFrame

Columns: `gameId, actionNumber, clock, period, teamId, teamTricode, personId,
playerName, playerNameI, xLegacy, yLegacy, shotDistance, shotResult,
isFieldGoal, scoreHome, scoreAway, pointsTotal, location, description,
actionType, subType, videoAvailable, shotValue, actionId`.

- `clock` is an ISO duration of time remaining in the period, e.g.
  `PT06M35.00S`.
- `scoreHome` / `scoreAway` are **strings, empty on non-scoring rows**. Forward
  fill them within the game.
- `location` is `h` / `v`.
- **There is no `possession` column** (the nba_api build here doesn't include
  it). Possessions must be estimated (§4.3).
- `actionType` values seen: `Made Shot, Missed Shot, Rebound, Substitution, Foul,
  Free Throw, Turnover, Timeout, period, Violation, Jump Ball, Instant Replay`,
  plus rows with a blank `actionType`.
- **Substitution rows carry only the outgoing player** in `personId`. The
  incoming player appears only by name in `description`
  (`"SUB: Kornet FOR Holiday"`). **Do not parse lineups from substitution rows.**
  Take lineups from `GameRotation` intervals. Use substitution rows only to
  order events that share a clock value (§4.2).
- About 400–520 rows per game.

### 2.3 Rate limiting (live tests, `GameRotation`)

| Gap between calls | Result |
|---|---|
| 0.05–0.1 s | blocked at request 16 |
| 0.6–1.2 s | blocked at request 74. A same-session retry, a session reset and a **brand-new process** all failed. The block cleared on its own after ~60–85 s. |
| **3.0 s** | **400/400 ok**, 0 blocks, one isolated timeout that an immediate retry fixed |

- A block shows up as a `ReadTimeout`, or as a response that isn't valid JSON
  (`JSONDecodeError`).
- **But not every `JSONDecodeError` is a block (measured 2026-09-21).** An
  HTTP 5xx with an empty body fails to parse in exactly the same way while
  meaning the opposite: the game is permanently unavailable, not throttled.
  Constructing an nba_api endpoint parses the body inside `__init__`, so the
  status code is discarded before anything can read it; the client now defers
  the request (`get_request=False` + `send_api_request`) to keep it. See §2.4.
- The limit is **rate-based**, on the server side and probably per IP. Resetting
  the session does not get past it.
- A 1.2–3 s gap is untested. Don't go below 3 s without measuring first.

### 2.4 `GameRotation` coverage is not complete before 2021 (measured 2026-09-21)

Some games return **HTTP 500 with a zero-byte body**, in under a second. This
is not throttling, and three separate tests say so:

- **It is deterministic per game.** Three rounds, new session each time:
  `0021800445` → 500, 500, 500 while the control `0021800444` → 200, 200, 200
  with an identical 5,923-byte body.
- **Games that have data never return it.** Ten games already archived with
  valid rotations: seven `ReadTimeout`, three 200, **zero 500**. A throttle
  would have hit these too.
- **A healthy game answers between two failures.** `0021800444` returned 200 in
  0.2 s interleaved with three consecutive 500s on `0021800445`.

Coverage sample, six games per season at the 2/20/40/60/80/98 % marks:

| Season | Result |
|---|---|
| 2018 | 200 200 200 **500** 200 **500** |
| 2019 | 200 200 200 200 timeout **500** |
| 2020 | 200 200 200 **500** 200 200 |
| 2021-2025 | **30/30 → 200** |

The holes are **scattered, not a cut-off**: in 2018-19 the game at the 80 %
mark answers 200 while `0021800900` and `0021801100` return 500. The 98 %
picks for 2018 and 2019 are playoff games and both fail, which hints that old
playoffs are worse, but six samples per season cannot carry that claim.

**The exact hole rate for 2018-2020 is unmeasured**; establishing it means
probing thousands of games, which is the backfill itself. The manifest's
`empty` count after each season is the measurement.

Consequences:

- A missing game costs 0.4 s and is recorded `empty`, not `failed`, and does
  **not** count towards the circuit breaker. Before this was separated from
  throttling it cost 570 s per game, and four such games in a row would have
  ended the run.
- 2018-19 exists to warm the ratings up before the 2019 training start. With
  holes in it that warm-up is weaker. The walk-forward ridge simply sees fewer
  games, but **if go/no-go C fails narrowly, rule this out before blaming the
  method.**

**Related fix (uncommitted in the working tree, commit it with phase A):**
`src/nba_ou/fetch_data/nba_api_session.py` and `tests/test_nba_api_session.py`.
The old `reset_nba_http_session()` was a no-op: it cleared `NBAHTTP._session`,
but endpoints cache the session on `NBAStatsHTTP`. Two duplicate copies in
`fetch_nba_data.py` and `update_refs_injuries_database_utils.py` now import
the shared helper.

---

## 3. Architecture

```
            stats.nba.com (3 s pacing)
                    │
   A  fetch ────────┴──► raw JSON archive (S3 / local mirror)  + manifest
                    │
   B  build stints ─┴──► lineup DB: lineup dimension, stints, per-game status
                    │
        ┌───────────┼──────────────────┬────────────────────┐
   C  player ratings    D  lineup & pair      E  minutes model
      (walk-forward        synergy (shrunk       (walk-forward XGBoost,
       ridge on stints)    residuals)             p_play × minutes | play)
        └───────────┬──────────────────┴────────────────────┘
   F  game projection: scenarios × minutes × ratings → possessions, points
                    │
   G  features (_BEFORE), schema 2_7 → experiments campaign vs 2_6
```

New code:

| Path | Contents |
|---|---|
| `src/nba_ou/fetch_data/nba_lineups/` | `client.py` (paced fetch + backoff), `archive.py` (raw JSON storage, reusing `fetch_data/injury_reports/archive/storage.py`'s `Storage` / `LocalStorage` / S3 pattern), `manifest.py` |
| `src/nba_ou/postgre_db/lineups/` | `schema.py`, `load.py`, `fetch.py`, `update.py` (copy the `line_history_aiven/` layout) |
| `src/nba_ou/data_processing/lineups/` | `stints.py`, `player_ratings.py`, `synergy.py`, `minutes_model.py`, `game_projection.py`, `features.py` |
| `scripts/lineups/` | `backfill_lineup_raw.py`, `build_lineup_stints.py`, `update_lineups.py` (thin CLIs) |
| `tests/` | `test_lineup_client.py`, `test_lineup_stints.py`, `test_player_ratings.py`, `test_lineup_synergy.py`, `test_minutes_model.py`, `test_game_projection.py`, `test_lineup_features.py`, `test_lineup_leakage.py` |

---

## 4. Phase A + B: fetching, archiving, stints

### 4.1 Fetch client (`fetch_data/nba_lineups/client.py`)

- One module-level pacer. It enforces **≥3.0 s between any two calls**, across
  both endpoints.
- Retry protocol per call:
  1. On `ReadTimeout`, `ConnectionError` or `JSONDecodeError`, wait 5 s and
     retry once.
  2. If the retry fails, treat it as a block. Call `reset_nba_http_session()`
     (cheap, harmless), then wait **90 s → 180 s → 300 s**, escalating.
  3. After 4 consecutive blocks, stop and exit non-zero. This is a circuit
     breaker; the run is resumable.
- Store the **raw JSON** (`endpoint.get_json()`), gzipped, under
  `nba_api_raw/{gamerotation,playbyplayv3}/{season_year}/{game_id}.json.gz`,
  in the S3 bucket from `config.ini [S3] BUCKET` or a local mirror under `data/`.
  Keeping the raw response lets stints be rebuilt without refetching.
- Manifest (parquet or small table): `game_id, endpoint, fetched_at, status
  (ok/empty/failed), bytes`. The backfill skips anything `ok`. **An empty or
  failed fetch is never recorded as "no data"** (same lesson as the 403
  ambiguity in `docs/injury_report_archive_plan.md` §0 row 2).
- Game list: from `nba_games.nba_games`, finished games only, 2018-19 onward.
- Backfill cost: ~10.5k games × 2 calls × ~3.1 s ≈ **18 h**. Run it overnight
  and resume as needed.
- Tests: mock the endpoint classes and assert the pacing, the retry/backoff
  sequence and the circuit breaker. A fake clock avoids real sleeps.

### 4.2 Stint builder (`data_processing/lineups/stints.py`)

`build_game_stints(rotation_home, rotation_away, pbp) -> pd.DataFrame`. Pure
function, no I/O.

1. **Boundaries.** Collect every `IN_TIME_REAL` / `OUT_TIME_REAL` from both
   teams, plus the period boundaries. Consecutive boundaries define segments.
   In each segment, each team must have exactly 5 players on the floor.
2. **Event times.** Convert each play-by-play `(period, clock)` to tenths of a
   second from tip-off, on the same scale as the rotation data.
3. **Events at a boundary.** An event whose time equals a boundary is ambiguous.
   Example: free throws shot after a substitution at the same clock. Resolve by
   `actionNumber` order relative to that team's `Substitution` rows at that
   clock:
   - events before the first substitution belong to the **outgoing** segment;
   - events after the last substitution belong to the **incoming** segment.

   If a boundary has no substitution rows (a period start or end), assign by
   period.
4. **Per segment and per team,** aggregate:
   - points, from `scoreHome`/`scoreAway` deltas (forward-filled);
   - FGA, 3PA, FTA, OREB, DREB and TOV, from `actionType` / `subType` /
     `isFieldGoal` / `shotValue`, attributed by `teamId`;
   - seconds.
5. **Possessions per team:** `FGA + 0.44·FTA − OREB + TOV`. Also store the raw
   counts so the formula can be revisited later.

**Validation.** Run per game. A game that fails any check is *rejected* and gets
a reason code in the per-game status table:

- every segment has 5 + 5 distinct players;
- segment seconds sum to the game length;
- stint points sum to the final score in `nba_games` (exact);
- each player's rotation seconds are within ±60 s of the box-score `MIN` in
  `nba_players`;
- no player appears in two overlapping intervals.

Target: **≥99% of games pass.** Look at the failures before loosening any check.

Tests should use small hand-built fixtures covering:

- a sub mid-period;
- a stint spanning halftime;
- free throws around a substitution at the same clock;
- overtime;
- a game whose points don't reconcile, which must be rejected.

### 4.3 Storage

**Current default: local Parquet.** `data/lineup_stints/season=YYYY/` holds one
file per validated game (carrying `game_date`) plus `game_status.parquet`.
`data_processing/lineups/stint_store.py` reads it. Measured: 112 games = 2.2 MB
on disk, so the full 2018-to-present backfill is ~200 MB of Parquet.

The Postgres schema below is implemented and stays available behind
`--load-db`; it is not populated by the daily job. Measured in Supabase: 112
games occupy 1.22 MB, so a linear extrapolation gives ~120 MB.

**Sizing re-derived 2026-09-22 (bottom-up, from the built games).** The linear
extrapolation overstates, because the lineup dimension saturates: at 112 games
there are 2,324 distinct lineups for 3,711 stints (0.63 per stint), but the
distinct-lineup curve grows as roughly `G^0.75` (639 → 1,076 → 1,434 → 1,808 →
2,140 → 2,324 over the first 112 games), which extrapolates to ~70k lineups for
~330k stints (0.21 per stint). Measured 33.1 segments per game.

| Table | Rows | Heap | Indexes | Total |
|---|---|---|---|---|
| `lu_stint` | ~330k | ~31 MB | ~11 MB (PK) | ~42 MB |
| `lu_lineup` | ~70k | ~6 MB | ~8 MB (PK + the 6-BIGINT UNIQUE) | ~14 MB |
| `lu_game_status` | ~10k | ~1 MB | ~0.3 MB | ~1 MB |
| | | | | **~57 MB** |

Calibrating against the real 1.22 MB the schema occupies today (~1.3× the
per-row arithmetic, from page overhead and per-partition minimums) gives
**~75-95 MB**. Postgres does **not** create indexes for the two foreign keys;
adding them for joins costs ~19 MB more, so budget **~110 MB** if they are
wanted. The lineup dimension is earning its place: storing `lineup_id` instead
of ten `BIGINT`s per stint row saves ~25 MB.

**Free space, measured 2026-09-22** (`pg_database_size`):

| Database | Size | Largest schemas |
|---|---|---|
| Supabase (default `DB_ENV`) | 449 MB | `nba_players` 267 MB, `odds_sportsbook` 72 MB, `nba_injuries` 31 MB |
| Aiven | 447 MB | `line_history` 396 MB, `injury_report` 41 MB |

Aiven's hard 1 GB cap covers heap, indexes **and WAL**, so ~550 MB is free and
the load fits with room to spare. Supabase is the riskier target despite being
the default: **confirm the plan's storage limit before loading there** — the
free tier caps at 500 MB, which this would exceed. Hence the decision in §1:
Aiven is the eventual destination, and nothing is loaded until go/no-go G
passes.

#### Postgres schema (opt-in, `postgre_db/lineups/schema.py`)

Sizing: ~10.5k games × ~30 segments ≈ 315k stint rows. Use a **lineup
dimension** so a stint row stores two small integers instead of ten player IDs:

```
lu_lineup  (lineup_id INTEGER PK, team_id BIGINT, p1..p5 BIGINT  -- sorted,
            UNIQUE (team_id, p1, p2, p3, p4, p5))
lu_stint   (game_id TEXT, seg_idx SMALLINT, season_year SMALLINT,
            period SMALLINT, start_ds INTEGER, end_ds INTEGER,      -- tenths of a second
            home_lineup_id INTEGER, away_lineup_id INTEGER,
            home_pts SMALLINT, away_pts SMALLINT,
            home_fga, home_fg3a, home_fta, home_oreb, home_dreb, home_tov SMALLINT,
            away_... same,
            PRIMARY KEY (game_id, seg_idx, season_year)) PARTITION BY LIST (season_year)
lu_game_status (game_id TEXT PK, season_year, game_date DATE, tipoff_utc TIMESTAMPTZ,
                status TEXT, reason TEXT, built_at TIMESTAMPTZ)
```

- Person IDs need `BIGINT`. They already reach ~1.97e9 (see the comment in
  `injury_report_aiven/schema.py`).
- **Before choosing Aiven vs the default DB (`DB_ENV = supabase`), measure free
  space.** Aiven has a hard 1 GB cap covering heap, indexes and WAL, and already
  holds `line_history` and `injury_report`. Run
  `SELECT pg_database_size(current_database())` on both.
- Write the schema so a season can be dropped, following `line_history_aiven`.

### 4.4 Daily update

- Add a step to `.github/workflows/update_finished_matches_daily.yml`, or
  `update_database_daily.yml` if that fits better. It runs
  `scripts/lineups/update_lineups.py`: fetch the finished games missing from
  the manifest, build stints, load.
- ≤15 games/day × 2 calls × 3 s is under 2 min.

### 4.5 Phase A/B acceptance

- The backfill leaves **no `failed`** entries in the manifest. `empty` is not a
  failure of ours: since §2.4 it also means the NBA has no rotation for that
  game, and no amount of retrying will change it.
- **Per season, of the games the NBA does serve, ≥99.5% are `ok`.** The old
  flat ≥99.5% `ok` target is unreachable for 2018-2020 and was written before
  the coverage gap was known; the 2021-2025 seasons should still meet it
  outright, since 30/30 sampled games answered 200.
- The stint build is ≥99% `status=ok` **among games with both endpoints
  archived**.
- The report script prints pass rates per season and failure reason counts, and
  now also the `empty` share, which is the measurement of §2.4 that a sample
  could not provide.

### 4.6 Cross-check (optional, cheap)

For 3 team-seasons, compare our 5-man minutes, net rating and pace with
`LeagueDashLineups(group_quantity=5, season=..., team_id_nullable=...)`.
Expect agreement within rounding and small differences in possession estimates.
A large gap means a bug in 4.2.

### 4.7 Play-by-play comes from the season archive, not the API (2026-09-21)

`shufinskiy/nba_data` republishes the same `PlayByPlayV3` payloads one season
per file (1996-2025, regular season + playoffs, ~8 MB each).
`scripts/lineups/import_pbp_archive.py` rewrites them into the ordinary raw
archive and records them `ok`, so nothing downstream changes;
`backfill_lineup_raw.py --endpoints gamerotation` then skips the half we
already hold. **The archive has no rotation data**, which is why the API is
still needed for the other half.

Verified against the 440 games already fetched from the API: **byte-identical
stints in all 437 that pass rotation validation** (the other 3 fail with API
data too). Three details the importer must get right, each found by that
comparison rather than assumed:

- The archive **omits `shotValue`**, which §4.2 reads to separate threes from
  twos. It is rebuilt from the `3PT` marker in the description, which matched
  the API exactly on all 78,050 field goals in 2018-19. Block rows get 0; the
  parser only reads `shotValue` on made and missed field goals.
- **Corrected actions share an `actionNumber`** with the row they amend, so the
  sort must be stable. A quicksort swapped a turnover with its steal and
  reattributed both to the wrong team.
- The archive **strips diacritics from `playerName`** and pads some text
  columns. Players are identified by `personId`, so this never reaches a stint.

The importer takes the same `lineup_run_lock` as the fetcher, because both
write the per-season manifest.

---

## 5. Phase C: player ratings (`player_ratings.py`)

A regularized adjusted plus-minus style ridge regression on stints, fitted
**walk-forward**.

- **Offense/defense rows.** Each segment gives two observations, one per team
  on offense. The target is points per 100 possessions minus the league
  average. The design has `+1` for each offensive player's O-column and `+1` for
  each defensive player's D-column. Weight each observation by possessions.
- **Pace rows.** A separate ridge on the same design (all 10 players) with
  target possessions per 48 minutes minus the league average, weighted by
  seconds.
- **Recency.** Exponential decay by game date, half-life ≈ 1 season (a
  hyperparameter).
  - With pure exponential decay the normal equations update **incrementally**
    per game date: `A ← γ^Δ·A + XᵀWX`, `b ← γ^Δ·b + XᵀWy`. Then solve
    `(A + λI)β = b`, a dense solve over ~1–2k active players. This is cheap
    enough to refit **every game date**.
  - Use `scipy.sparse` for X (scipy is already a transitive dependency via
    scikit-learn).
- **Temporal contract.** It is the same as
  `data_processing/line_history/walk_forward.py`: ratings used on date D come
  only from stints of games on dates **< D**. Same-day games are held back
  together, as in `starter_history.py`.
- **λ:** choose by walk-forward CV on next-day stint prediction error, one value
  per rating type. Record the chosen values in the module as constants with a
  comment on how they were chosen, like `KEY_PLAYER_THRESHOLDS` in
  `fresh_absence.py`.
- **Players with no history** get a rating of 0, i.e. league average. A
  replacement-level prior (e.g. slightly negative for players with few minutes)
  is a v2 option; don't build it first.
- **Output:** a long table `(as_of_date, player_id, o_rating, d_rating,
  pace_rating, poss_weight)`. Cache it as parquet under `data/` so the feature
  build doesn't refit.
- **Tests:**
  - synthetic stints where one player adds +10 per 100 → the recovered rating
    has the right sign and roughly the right size;
  - the incremental update gives the same ratings as a batch refit;
  - ratings on date D are unchanged when stints dated ≥ D are perturbed.

**RESULT (2026-09-22): go/no-go C PASSES.** Lambdas tuned on 2022-23/2023-24
and gated on 2024-25, disjoint windows. The README's grids were **truncated** —
both penalties selected their grid maximum with a monotone curve. Widening them
found interior minima at **`lambda_offdef=1000`** (stint MAE 47.126) and
**`lambda_pace=30000`** (14.222); the offdef value was right by luck, the pace
value was not (10000 was a boundary artefact).

| Window | n | ratings MAE | baseline MAE | improvement | 95% CI |
|---|---|---|---|---|---|
| 2024-25, **out of sample** | 1,314 | 14.716 | 15.880 | **+1.163** | [0.707, 1.606] |
| 2021-2025 | 5,251 | 14.846 | 15.835 | **+0.989** | [0.769, 1.210] |

Stable every season (+0.95, +0.85, +1.00, +1.16), and the out-of-sample season
is the *best* of the four, so the tuning did not overfit. The ratings beat the
rolling team-average baseline on 54.5% of games.

**But the phase-G dry run on the same projection is a null.** Against the
`ODDS_TOTAL_LINE_bet365` closing line on the same games, the ratings-only
projection is **worse than the market** (MAE 14.82 vs **14.11**), and
regressing `LINE_ERROR` on `proj_total - line` gives a slope of **-0.037, 95%
CI [-0.133, +0.053]** (corr -0.011). Every subset tested also spans zero,
including the ones this plan is built on:

| Subset | n | slope | 95% CI |
|---|---|---|---|
| all | 5,236 | -0.037 | [-0.133, +0.048] |
| best absentee >= 20 pts | 2,138 | +0.023 | [-0.122, +0.164] |
| first game out (streak <= 1) | 2,729 | -0.020 | [-0.136, +0.108] |
| key player >= 15 pts **and** first game out | 1,652 | -0.050 | [-0.213, +0.115] |
| \|proj - line\| >= 8 | 677 | +0.037 | [-0.081, +0.160] |

**Do not read this as falsifying the hypothesis.** The ratings-only projection
uses minutes = each player's recent five-game average and carries **no injury
information at all**, so on a key player's first game out it still assigns that
player his recent minutes. The fresh-absence subsets are therefore the cases
this particular projection is *least* able to model, and §0 already predicted
the quiet-night result ("the projection will mostly reproduce the line").

What it does establish: **the unconditional path is closed, and phase E is
load-bearing.** The bet now rests entirely on the minutes model redistributing
minutes correctly on news nights, not on the ratings. Weigh that before
spending the effort on D-F, and re-run this dry run the moment F v1 exists.

Artifacts: `data/lineup_ratings/{cv,cv_wide,gate_2024,gate_all}/`, cache
`data/lineup_ratings/player_ratings.parquet` (741,335 rows, as-of 2021-10-19 to
2025-06-22).

**Market-calibration check (2026-09-22), run instead of the standalone
comparison above.** The dry run asked "is this projection better than the
bookmaker?", which is not the question a feature has to answer: the model
already sees ~3,000 columns, so what matters is incremental information. The
owner redirected this. Measured on the 2_3 closing-line dataset, 8,829 non-push
games, 2018-19 to 2025-26, `LINE_ERROR = TOTAL_POINTS - ODDS_TOTAL_LINE_bet365`:

| Situation | n | OVER | 95% CI | mean LINE_ERROR |
|---|---|---|---|---|
| all games (reference) | 8,829 | 50.7% | [49.6, 51.7] | +0.52 |
| best absentee >= 18 pts | 4,333 | 51.1% | [49.6, 52.5] | +0.54 |
| best absentee >= 25 pts | 1,316 | 49.3% | [46.6, 52.0] | -0.10 |
| **>= 18 pts AND first game out** | **1,498** | **53.4%** | **[50.9, 55.9]** | **+1.46** |
| >= 18 pts AND >= 5 games out (control) | 2,017 | 49.6% | [47.4, 51.8] | -0.11 |

**The size of the absence carries nothing; the freshness carries everything.**
The settled-absence control is a clean null, which is what an overreaction
story predicts and a data-mining artefact would not. Stable in 7 of 8 seasons
(2025-26 is the exception at 48.5%, n=231, whose CI still covers the other
seasons). This reproduces the owner's existing lead.

**It is not yet a bankable edge.** Break-even at -110 is 52.38% and the
interval's lower bound is 50.9%, so this is a real but marginal signal, not a
proven profitable one.

**The lineup projection adds nothing to it.** Regressing `LINE_ERROR` on
`proj_total - line` *within* the fresh-absence subset gives slope **-0.117, 95%
CI [-0.340, +0.105]** (n=924) — indistinguishable from zero and, if anything,
negative. Same null as the unconditional test.

**The bar this sets for D-F.** The inefficiency is already detectable from two
columns that exist today, `TOP1_INJURED_PLAYER_PTS_BEFORE` and
`TOP1_INJURED_STREAK_PTS_BEFORE`. Phases D-F do not have to find the effect —
they have to **beat features already in the dataset**. The phase-C projection
does not, though it is also the version least able to (no injury inputs at
all). Any decision to build E should be made against that bar, and the first
thing E's output should be tested on is this subset.

**Go/no-go C.** Build `PROJ_TOTAL` from ratings alone, with minutes = each
player's recent average (no minutes model yet). On 2021–2025 walk-forward,
compare its total-points MAE with a rolling team-average baseline.
- If it is not better, stop and report back before building D–F.
- **Run it on what is built, not on the full backfill** (§10). As of 2026-09-22
  that is 2021-22 through 2024-25: tune the lambdas on 2022-23/2023-24 and gate
  on **2024-25, out of sample**. Tuning and gating on the same window, as the
  README's example does, flatters the verdict on a gate whose whole purpose is
  to be failable.

---

## 6. Phase D + E

### 6.1 Synergy (`synergy.py`)

For any five, or any pair, as of date D:

- **Residual:** observed points per 100 and pace, minus what the §5 ratings
  predict for those same stints. Shrink toward 0 by possessions:
  `w = n / (n + k)`, with k fitted empirically. Reuse the empirical-Bayes idea
  from `past_injuries/injury_effects.py::_expanding_empirical_bayes_weights`
  rather than a hand-set k.
- **Five-level value** = exact-five residual if the five has meaningful
  possessions, otherwise the sum of the shrunk residuals of its 10 pairs. Emit
  both and let the model choose.
- **Minutes together:** minutes for the exact five and for each pair, this
  season and in the last 10 games.
- Same temporal contract as §5.

### 6.1b Phase D as built, and its null result (2026-09-23)

`data_processing/lineups/synergy.py`, 22 tests. Residual per stint against the
phase-C prediction, accumulated per pair and per exact five, decayed with a
180-day half-life and shrunk by `w = n / (n + k)` with `k = 200` possessions.
`k` is an explicit parameter rather than the expanding empirical-Bayes fit this
section specifies: same idea, one fewer moving part while it is being tested.

**Two scale traps, both hit before they were fixed.** Written naively the
synergy term averaged **+14 points per 100** and took the projection's MAE from
14.5 to **24.7**:

- A pair inherits the residual of every lineup it appeared in, so its mean is a
  *lineup-level* quantity. Summing a five's ten pairs without dividing by ten
  inflates the term tenfold.
- The league's own mean residual is **+2.75 points per 100** and is not synergy
  at all — it is the same estimated-possessions calibration gap `total_offset`
  already removes (7.1c). Every pair inherited it, ten times over.

Both are fixed by reporting every value as a deviation from the running league
mean, divided by the pairs on the floor. The term then has mean +0.28 and sd
1.54, which are plausible magnitudes.

**And it does not help.** On 2021-22 to 2024-25, 5,235 games:

| | MAE |
|---|---|
| projection without synergy | **14.518** |
| projection with synergy | 14.612 |

Regressing `LINE_ERROR` on the synergy term alone: slope **-0.024, 95% CI
[-0.325, +0.305]** — null, and the term makes the projection slightly worse.

**Where that leaves the three levers.** Phase E moved the projection by 0.004
MAE, phase D by -0.094, and only the absence counterfactual (7.1d) separates
from zero. The structural reading from 6.3 survives: the team aggregate is
dominated by the ratings and the possession count, and neither redividing the
fixed 240 minutes (E) nor adding a shrunk lineup residual (D) disturbs it much.

This is one parameterisation — `k = 200`, a 180-day half-life, pair-level
rather than exact-five weighting — and a linear addition to the projection, so
it does not prove synergy is worthless. It does mean the column should go into
the campaign to be judged there rather than be argued for here.

### 6.1c Calibration of the hand-set parameters (2026-09-23)

Three values had been chosen by judgement. Each now has an objective criterion
and a sweep; two of the three answers are "it does not matter", which is worth
knowing.

**`ROSTER_APPEARANCE_WINDOW`: flat, set to 5.** Criterion: how well the
projected five matches the **actual** starting five -- an observable fact, not
a betting metric. Swept over {3, 5, 8, 10, 15, 20, 30, 60}: mean overlap 4.4699
at 3 against 4.4675 at 60, out of 5. The false replacements the bound removes
are ~25 cases in 10,514 team-games, far too few to move it.

**The D0 machinery itself was never compared to the trivial alternative**, and
should have been. Against simply reusing the latest starting five:

| | projected | latest five |
|---|---|---|
| mean overlap | **4.4700** | 4.3096 |
| exact 5/5 | **58.91%** | 51.77% |
| overlap where a starter is out (n=2,663) | **4.098** | 3.467 |
| exact 5/5 there | **28.13%** | 0.04% |

The gain grows with the number of starters out: +0.444, +1.084, **+1.870** for
one, two and three. D0 earns its place.

**Synergy `k` and half-life: the calibration confirms the null, and corrects an
earlier claim.** Swept k in {50, 200, 1000, 5000, 25000} against half-lives of
{90, 180, 365}, fitted on 2021-2023 and scored on 2024-25 (fit/test MAE):

| half-life | k=50 | k=200 | k=1000 | k=5000 | k=25000 |
|---|---|---|---|---|---|
| 90 | 14.749/14.460 | 14.663/**14.428** | 14.583/14.429 | 14.548/14.450 | 14.538/14.462 |
| 180 | 14.741/14.483 | 14.665/14.453 | 14.590/14.442 | 14.552/14.453 | 14.539/14.462 |
| 365 | 14.733/14.494 | 14.664/14.470 | 14.595/14.452 | 14.555/14.456 | 14.540/14.463 |

No synergy at all: **14.535 fit / 14.466 test**.

On the fitting window **nothing beats leaving synergy out**, and the curve is
monotone toward `k -> infinity`, where the term vanishes and the MAE converges
back to 14.535. Three settings edge the baseline on the test season by ~0.04
MAE, which the fitting window does not corroborate; adopting one would be
selecting on the evaluation set.

**Correction to 6.1b.** The `k = 200`, 180-day setting used there was a *poor*
one -- 14.665 on the fitting window against a 14.535 baseline. The reported
"synergy makes the projection 0.094 worse" was partly an artefact of that
choice. Properly calibrated, synergy is **neutral rather than harmful**. Still
null, but the distinction matters.

`k` is set to 1000 as a near-neutral compromise, and **the projection no longer
adds the synergy term at all**; it is emitted as its own column for the
campaign to judge, which is where section 8.4 says this belongs.

**Exact-five synergy was specified and not emitted.** This section asks for the
exact-five residual *and* the pair-based value, "emit both and let the model
choose". `five_value` was being computed and discarded, and the pair weighting
picked unilaterally. `projected_five_synergy` now returns the exact five's
residual with the possessions behind it, so a consumer can tell the sharp case
from the empty one.

### 6.1d The recent-minutes window, and a problem with go/no-go E

Swept 2026-09-23 on projection MAE, fitted on 2021-2023 and scored on 2024-25.
The window turned out to control **two** things -- how many games the minutes
average spans, and how long a player may go unseen before leaving the roster --
so they were separated (`recent_games` and `roster_games`) before anything was
concluded from a sweep of either.

**The two criteria are anti-correlated.** A 3-game average gives the best
individual minutes (MAE 5.73, against 6.70 at 15 games) and the **worst** game
total; a long one is the reverse:

| window | minutes MAE | projection MAE (test) |
|---|---|---|
| 3 | **5.732** | 14.4871 |
| 5 | 5.967 | 14.4677 |
| 15 | 6.695 | **14.4072** |

A short window tracks a changing role but overreacts to one blowout or one
foul-trouble night, and that noise accumulates in the team aggregate rather
than averaging out.

**This makes go/no-go E the wrong instrument for a projection decision.**
Section 6.2 asks the minutes model to beat the last-5 average on **minutes
MAE**, and it does, by 23.8%. But winning on that criterion can actively hurt
the total, which is what the pipeline is for. The gate remains a fair test of
the minutes model; it is not evidence about what the projection wants. Read it
that way, and note that the phase-E result (6.3) is measured against the same
confusion.

**Separated, the two windows behave very differently** (fit/test MAE):

| average \ roster | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| 3 | 14.5409/14.4528 | 14.5355/14.4379 | 14.5332/14.4589 | 14.5205/14.5230 |
| 5 | 14.5357/14.4677 | 14.5311/14.4583 | 14.5269/14.4813 | 14.5155/14.5504 |
| 10 | 14.5149/14.4514 | **14.5089/14.4368** | 14.5050/14.4613 | **14.4947**/14.5195 |
| 20 | 14.5194/14.4099 | 14.5115/**14.3961** | 14.5092/14.4252 | 14.5005/14.4775 |
| 40 | 14.5381/14.4167 | 14.5287/14.4018 | 14.5303/14.4297 | 14.5262/14.4833 |

- **The average carries a consistent signal**: both windows agree 5 is too
  short. The fitting window's optimum is 10, the test season's is 20, so
  **10** is taken; 20 would be chosen on the evaluation set.
- **The roster window carries none**: the fitting window prefers 40 and the
  test season prefers 10, in every row. That is noise, not an optimum. **10**
  is kept as the only value that is never poor on either.

`RECENT_GAMES` moves 5 -> 10, worth about 0.03 MAE on both windows.

### 6.2 Minutes model (`minutes_model.py`)

**Target:** each player's actual minutes in each team-game (0 if they didn't
play). The population is players on the roster as of the game; reuse
`injury_status/status_history.py` and `players/roster_continuity.py` for roster
membership.

**Structure:** `E[min] = P(play) × E[min | play]`.

- **`P(play)`:** start from `injury_status/news.py::chance_out()`, which already
  turns report status into a probability of being out. For unlisted players,
  learn it: coach's DNPs are common for bench players.
- **`E[min | play]`:** an XGBoost regressor.

**Inputs** (all known before tip-off):

- recent minutes, weighted toward the last 3 / 5 / 10 games (see
  `status_history.FORM_HALFLIFE_GAMES`);
- share of the team's minutes;
- starter in the latest five (from `starter_history` events);
- games started this season;
- games since the player's last absence (captures a minutes restriction);
- report status and `P_PLAY`;
- expected minutes missing from teammates at the same `START_POSITION` group;
- whether this player was the historical replacement for a missing starter:
  who started the last times that player was out;
- back-to-back and rest days;
- closing spread size, since blowouts cut starters' minutes (use the **as-of**
  line available at the cut-off);
- season phase;
- days since a trade.

**Team constraint:** rescale so each team's expected minutes sum to
`240 × (1 + expected OT share)`. Players with `P(play) ≈ 0` are excluded from
the rescale.

**Training:** walk-forward, expanding window, **refit monthly**. Predictions for
month M come from a model trained on games before month M. Fixed seed. No
artifact goes to the model registry: it is a feature-pipeline component, refit
on the fly (per player-game data ≈ 300k rows, seconds per fit).

**Tests:**

- a month-M prediction is unchanged when data from month ≥ M changes;
- the team sum holds;
- a player listed Out gets 0.

**Go/no-go E.** Beat the last-5 average on minutes MAE, both overall and in
games immediately after a teammate's absence. Report both numbers. If it loses,
use the last-5 average with the §6.2 team constraint as the minutes input, and
say so.

---

### 6.3 Phase E as built, and what it did not buy (2026-09-23)

`data_processing/lineups/minutes_model.py`, 18 tests.
`E[min] = P(play) * E[min | play]`, the regressor trained only on games the
player actually played so an Out listing cannot be learned as "few minutes"
instead of "no minutes". Twelve features; the plan's closing-spread size and
days-since-trade are **absent**, the first because it needs an odds join this
module avoids and the second because there is no transactions feed.

**Go/no-go E passes decisively**, on 2021-22 to 2024-25, 149,541 player-games:

| Subset | model MAE | last-5 baseline | gain |
|---|---|---|---|
| all | 5.227 | 6.858 | **+1.631 (23.8%)** |
| >= 15 min of teammates missing (n=87,594) | 5.424 | 7.555 | +2.131 |
| >= 30 min missing (n=59,118) | 5.546 | 8.098 | **+2.552** |

The gain **grows** with the size of the absence, which is the criterion this
gate exists for. The last-5 average degrades badly when players are missing
(6.86 to 8.10) because it reallocates blindly; the model holds (5.23 to 5.55)
because it knows who actually absorbs the minutes.

**And it barely moves anything downstream.** Feeding those minutes into phase F
instead of recent averages:

| | recent averages | phase-E minutes |
|---|---|---|
| Projection MAE | 14.519 | **14.515** |
| `absence_impact_points` slope | +0.291 [+0.100, +0.472] | **+0.318 [+0.101, +0.534]** |
| Peak directional accuracy | 52.73% at \|impact\|>=2 | 53.26% at \|impact\|>=2 |
| sd of the impact column | 2.58 | 2.32 |

A 23.8% better minutes model buys **0.004 of game MAE** and leaves the slope
inside the old interval. The accuracy still falls above \|impact\| >= 2 rather
than concentrating, so the shape that was worrying did not invert.

**Why, and it is structural.** The team aggregate is
`sum_i (min_i / 48) * rating_i` with `sum_i min_i` pinned at 240. Reshuffling
minutes between players of similar rating leaves that sum almost unchanged, so
per-player minute errors largely cancel at team level. Minutes accuracy and
team-total accuracy are much more weakly coupled than the plan assumed -- and
than the agent predicted when it argued phase E was where the remaining value
sat. It was not.

**What follows.** Keep phase E: it passes its gate, it is the plan's
specification, and better minutes feed other section 8.1 columns
(`LU_PROJ_MIN_CHANGE_BEFORE`, the replacement columns). But stop treating it as
the lever on the totals signal. Either minutes variant produces nearly the same
feature, so the campaign can use either. The remaining untried lever is
**phase D synergy and the 7.1b shared-minutes weighting**, which change the
aggregate itself rather than how the fixed 240 is divided.

---

## 7. Phase F: game projection (`game_projection.py`)

### 7.1 v1: minutes-weighted ratings

**Scenarios.** Players with `0.1 < P(out) < 0.9` define scenarios.
- For up to 3 such players per team, enumerate all 2³ = 8 combinations per team.
- Beyond 3 players, sample.
- Weight each scenario by the product of its probabilities.
- Re-run the minutes model's team constraint inside each scenario.

**Per scenario, per team:**

```
# Σ_i min_i = 240 per team, so Σ_i (min_i/48) = 5 = players on court: each
# sum is the average on-court lineup's summed rating, same scale as §5.
off_T  = Σ_i (min_i / 48) · o_rating_i        # pts/100 poss above league avg
def_T  = Σ_i (min_i / 48) · d_rating_i        # pts/100 poss allowed below avg
pace_T = Σ_i (min_i / 48) · pace_rating_i     # poss/48 above league avg

poss   = (league_pace + pace_H + pace_A) · game_minutes / 48
ptsH   = poss / 100 · (league_ortg + off_H − def_A) + home_court
ptsA   = poss / 100 · (league_ortg + off_A − def_H)
```

The §5 pace ridge has all 10 players in each row, so `pace_H + pace_A` is
already the full game's pace deviation. `game_minutes` = 48 + 5 × expected
overtime periods.

- **Starter-vs-starter synergy.** Take the expected minutes the two projected
  starting fives share. Estimate it from each team's history: the average
  minutes its most frequent starting five played in Q1 + Q3 openings. Add the
  §6.1 synergy for those minutes only.
- **Player points** (for diagnostics and player-level features):
  - each player's per-minute scoring rate × projected minutes;
  - absent players' usage is redistributed to the others in proportion to
    their usage;
  - scaled so the players sum to the team points. **The team total from the
    ratings is the fixed number; player points only split it.**
- **Aggregate over scenarios:** mean and standard deviation of the total, and
  each team's mean.
- `league_ortg`, `league_pace` and the home-court term are walk-forward
  estimates too (the league-average intercepts of §5).

### 7.1b Shared-minutes weighting (owner's direction, 2026-09-22)

**The v1 formula above throws away the combination information.** It sums
individual ratings weighted by each player's own minutes, so two stars who
share 30 minutes and two who never overlap give the identical team number.
Who plays *with* whom is the entire reason this project collects stints
instead of box scores, and v1 does not use it.

Between v1 and the full rotation template of §7.2 there is a cheaper step that
captures most of it: an **expected shared-minutes matrix**.

- For each pair of available players, estimate the minutes they will share,
  from their shared-minutes history **conditioned on tonight's availability** —
  a pair's past overlap is only informative among the players actually
  expected to play.
- Normalise so each player's row sums to his projected minutes and the team
  total holds at 240.
- Weight the §6.1 pair synergy by **expected shared minutes**, not by whether
  both are merely in the rotation. The exact-five term stays as in §7.1 for
  the projected starting five.

New column, alongside §8.1:

| Column | Meaning |
|---|---|
| `LU_PROJ_PTS_COMINUTES_BEFORE` | expected team points with synergy weighted by expected shared minutes, i.e. what the available players are expected to score **given how they actually share the floor** |

Emit it **beside** `LU_PROJ_PTS_BEFORE`, not instead of it. The difference
between the two is itself informative: it is the part of the projection that
combination information accounts for, and it is cheap to let the model choose
between them.

Same temporal contract as §5: shared-minutes history comes only from games
before the target date.

### 7.1c Phase F v1 as built (2026-09-23)

`data_processing/lineups/game_projection.py`, 25 unit tests. Two deviations
from 7.1 above, both deliberate:

**Home court splits rather than adds.** 7.1 writes `+ home_court` on the home
side only, which raises every projected *total* by that amount. Venue cannot
move a total in aggregate -- every game has one home side and one away side --
so it is applied as `+h/2` and `-h/2`, keeping the margin and leaving the total
alone.

**A calibration term was needed, and it is explicit.** Projecting straight from
the fitted intercepts under-predicts every total by **2.75 points**. The cause
is measurement, not modelling: the stint possession count is *estimated*
(`FGA + 0.44*FTA - OREB + TOV`, unattributed team rebounds unclassified), it
runs high, so points per 100 come out low -- the fitted `league_ortg` averages
**111.37** against a game-level implied **113.39**. `project_totals` therefore
takes a `total_offset` the caller estimates walk-forward. Correcting it here
keeps the distortion out of the player ratings.

**Validation.** With everyone available (`p_out = 0`), F v1 reproduces the
phase-C gate almost exactly on 2024-25: **MAE 14.710 against 14.716**, bias
-0.32 after calibration. That is the expected agreement -- with no absences the
scenarios collapse to one and the arithmetic is the same -- and it checks the
new code against the existing implementation.

**This has not yet tested the hypothesis.** `p_out` was zero throughout, so no
scenario branched and the counterfactual was always null. Wiring `chance_out()`
from the injury report is the next step and the first real test of F. The
phase-G dry run on this absence-free version is unchanged from phase C: slope
-0.082, 95% CI [-0.269, +0.110].

**A harness trap worth recording.** The first measurement made F v1 look worse
than phase C (15.054 vs 14.716). The cause was the evaluation driver, not the
module: it never expired players from a team's roster, so long-departed players
kept diluting the minutes share. `rating_gate.ratings_only_game_projections`
applies a `last_seen > team_games_played - recent_games` cutoff; anything
comparing against it must apply the same rule or it is not comparing like with
like.

### 7.1d p_out wired: the first non-null result (2026-09-23)

`data_processing/lineups/availability.py` builds the `PlayerNight` rosters from
box-score history and `injury_status.news.chance_out`, with the same temporal
contract as `rating_gate`. Measured on **2021-22 to 2024-25**, 5,235 games; the
rotation backfill for 2025-26 is irrelevant to all of it, since the rating
cache stops at `last_season: 2024`.

**Availability information improves the projection.**

| | MAE on the same games |
|---|---|
| Phase-C gate | 14.716 |
| F v1, `p_out = 0` | 14.710 |
| **F v1 with the real report** | **14.519** |

73.3% of team-games carry at least one certain absence, so the counterfactual
has content nearly everywhere. Only **7.7%** branch on a genuinely doubtful
player: the report holds 42,615 `out` listings against 1,374 `questionable` and
149 `doubtful`, so scenario spread is rare and `total_sd` is 0 at the median.
**The counterfactual, not the scenario spread, is the payload.**

**`absence_impact_points` is the first column in this project that is not
null.** Regressing `LINE_ERROR` on it, against the plan's own gate-G quantity:

| Regressor | n | slope | 95% CI |
|---|---|---|---|
| `proj_total - line` (the plan's gate G) | 5,156 | +0.077 | [-0.034, +0.185] |
| **`absence_impact_points`** | 5,156 | **+0.291** | **[+0.100, +0.472]** |

A slope near 0.29 says roughly **29% of the modelled point impact of tonight's
absences is not in the closing line**. The sign is positive in all four seasons
(+0.26, +0.10, +0.28, +0.54) and the strongest season, 2024-25, is the one
**outside** the window the lambdas were tuned on: +0.536, 95% CI [+0.195,
+0.889], significant on its own.

**It does not yet clear the vig, and one diagnostic points the wrong way.**
Betting OVER on a positive impact:

| Filter | n | Directional accuracy | 95% CI |
|---|---|---|---|
| all | 5,156 | 51.59% | [50.23, 52.95] |
| \|impact\| >= 1 | 2,776 | 52.34% | [50.48, 54.20] |
| \|impact\| >= 2 | 1,667 | 52.73% | [50.33, 55.13] |
| \|impact\| >= 3 | 1,008 | 51.09% | [48.01, 54.18] |
| \|impact\| >= 4 | 608 | 50.49% | [46.52, 54.47] |

Break-even at -110 is 52.38%; no interval's lower bound clears it. Worse, the
accuracy **falls** as the threshold rises. A real edge should concentrate where
the signal is strongest, not dilute, so the slope is likelier carried by the
bulk of small-impact games than by the large ones. Read it as a genuine
correlation that is not yet an exploitable one.

**Two things this changes.**

- The right feature is the **counterfactual**, not the level. The plan named
  `LU_PROJ_TOTAL_MINUS_LINE_BEFORE` the probable key input for `LINE_ERROR`; on
  this evidence it is not, and `LU_ABSENCE_IMPACT_PTS_BEFORE` is.
- The fresh-absence subset is **not** where the effect lives. On the impact
  column the non-fresh games carry it (+0.403, CI [+0.176, +0.645]) while the
  fresh ones do not separate (+0.264, CI [-0.116, +0.609]). That is a different
  story from the 53.4% OVER finding in section 5, which was about freshness.
  Both can be true; they are not the same effect, and neither should be used to
  argue for the other.

**How to read the accuracy table (owner, 2026-09-23).** It is a *univariate
linear* betting rule, which is not how the column will be used. The model is
gradient-boosted and will see this alongside ~3,000 other columns, including
interactions the rule cannot express, and the edge needed is small. The table
is therefore a weak lower bound, not a verdict: the real test is the campaign
in section 8.4 with `lineup_features` off and on. Do not cite it to argue the
column is worthless, and do not cite the slope to argue it is valuable --
settle it in the campaign.

**What it justifies.** Phases D and E are worth building. The impact column is
computed from a crude proportional minutes redistribution; a real minutes model
(E) is exactly what would sharpen it, and this is the first evidence that
sharpening it is worth the effort.

### 7.2 v2 (optional): rotation template

Model which fives share the floor from recent substitution patterns:
- who starts each quarter;
- the typical time of the first substitution;
- bench staggering.

This distributes minutes across specific lineup pairs so exact-five synergy
applies beyond the starting units.

**Gating relaxed 2026-09-22 (owner).** This was "only do this if F v1 clears
go/no-go G", which had the ordering backwards: v1 is the version that cannot
use combination information at all, so gating the combination-aware version
behind v1's result risks discarding the idea on a test that could not have
shown it. §7.1b is the cheap way to get most of the way there; keep the full
rotation template gated behind §7.1b showing something, not behind v1.

---

## 8. Phase G: features, schema, evaluation

### 8.1 Columns

All end in `_BEFORE`; team-level columns get `_TEAM_HOME` / `_TEAM_AWAY` from
the merge. **Keep the set small and add more only on evidence.**

**Team level (in `lineups/features.py`, attached where `add_starter_history_features` is):**

| Column | Meaning |
|---|---|
| `LU_PROJ_PTS_BEFORE` | expected team points (§7.1 mean) |
| `LU_PROJ_OFF_BEFORE`, `LU_PROJ_DEF_BEFORE`, `LU_PROJ_PACE_BEFORE` | minutes-weighted ratings |
| `LU_PROJ_MIN_CHANGE_BEFORE` | Σ\|projected min − last-10 avg min\| over the roster: how unusual tonight's rotation is |
| `LU_MISSING_RATING_BEFORE` | minutes-weighted rating of expected-absent players (what is lost) |
| `LU_REPLACEMENT_RATING_BEFORE` | minutes-weighted rating of the minutes that absorb it (what replaces it) |
| `LU_PROJ_STARTERS_REPLACED_BEFORE` | number of the latest starting five with P(out) ≥ 0.5 |
| `LU_REPL_MIN_STARTS_SEASON_BEFORE` | fewest starts this season among the projected replacements |
| `LU_STARTING_FIVE_SYNERGY_BEFORE`, `LU_STARTING_FIVE_MIN_TOGETHER_BEFORE` | §6.1 for the projected starting five |

**Game level (in `merged_home_away_data/add_features_after_merging.py`, like
`add_fresh_absence_sums`):**

| Column | Meaning |
|---|---|
| `LU_PROJ_TOTAL_BEFORE` | expected total |
| `LU_PROJ_TOTAL_SD_BEFORE` | spread across scenarios |
| `LU_PROJ_POSS_BEFORE` | expected possessions |
| `LU_PROJ_TOTAL_MINUS_LINE_BEFORE` | projected total − closing total line (`total_line_col()`); probably the key input for `LINE_ERROR` |
| `LU_STARTERS_VS_STARTERS_NET_BEFORE`, `LU_STARTERS_VS_STARTERS_PACE_BEFORE` | the two projected starting fives against each other |

#### Added 2026-09-22 (owner's direction)

Two additions, and a caveat on how the null results above should be read.

**Caveat first.** The dry-run and calibration tests in §5 are **linear** slopes.
A flat linear slope does not rule out the non-linear structure and interactions
XGBoost is there to find, so treat those nulls as weak evidence in one
direction, not a verdict. They do establish the bar (the effect is already
reachable from two existing injury columns), not the impossibility.

**A. Counterfactual absence impact.** The question worth asking is not "what
will the total be" but **"did the market react correctly to this news"**. Make
that directly measurable: run the §7.1 projection **twice** — once with
projected availability, once with the same roster at full health — and emit the
difference.

| Column | Meaning |
|---|---|
| `LU_ABSENCE_IMPACT_PTS_BEFORE` | projected total with absences − projected total at full health. The model's estimate of what tonight's absences are worth, in points |
| `LU_ABSENCE_IMPACT_PACE_BEFORE` | the same difference in projected possessions |
| `LU_ABSENCE_IMPACT_MINUS_LINE_BEFORE` | `LU_ABSENCE_IMPACT_PTS_BEFORE` − (closing line − the team's recent typical line level). A crude read on whether the line moved more or less than the absence is worth |

This is the closing-line version. The sharper form — impact versus the line's
*actual* move since the news broke — needs snapshots and belongs to phase 2
(§9), where `INJ_SNAP_*` already exists.

**B. Replacement-specific synergy.** §6.1 gives synergy for the projected
starting five as a whole. The owner's sharper question is about **the
replacement in particular**: the player expected to take an absent starter's
place, and how he plays *with the other four*.

| Column | Meaning |
|---|---|
| `LU_REPL_SYNERGY_WITH_STARTERS_BEFORE` | sum of the shrunk §6.1 pair residuals between the projected replacement and each of the four remaining projected starters |
| `LU_REPL_MIN_WITH_STARTERS_BEFORE` | minutes the replacement has actually played alongside those four, season and last 10 games |
| `LU_REPL_EXACT_FIVE_MIN_BEFORE` | minutes the resulting exact five has ever played together |

`LU_REPL_MIN_WITH_STARTERS_BEFORE` is the one to watch. It is the "untested
replacement" hypothesis made measurable, and it cuts both ways: when it is near
zero **neither we nor the market have a basis**, which is precisely the
condition under which a market overreaction is plausible. Expect it to earn its
place as an interaction term, not as a main effect.

**Open question: there is no roster feed.** Building D0 surfaced this. Nothing
in the repo says who is on a team tonight. Box scores say who has already
played, and the injury report lists about five players per team-game, not a
squad. Two consequences, both measured on 2024-25:

- A mid-season trade is handled by bounding roster membership to the last 10
  in-season appearances, which cut projected replacements who never appear for
  that team again from **5.4% to 2.8%** (52 to 27 of ~960). The residual is
  mostly season-ending injuries, where the player *is* still rostered.
- **A summer departure cannot be detected at all.** At a season opener there is
  no in-season appearance evidence, and marking the unlisted as departed would
  invent departures for four fifths of the returning five. The opener therefore
  projects last season's five as-is, and `LU_PROJ_GAMES_THIS_SEASON_BEFORE`
  (0 there) is the flag that says so.

Acquiring a roster or transactions feed would close both. Until then, treat
early-season `LU_*` values as weakly founded rather than wrong.

**Sequencing consequence.** Both additions need a **projected starting five**,
not the full §6.2 minutes model. That is a much lighter prerequisite —
`starter_history.py` already supplies the latest five, and §6.2 already plans
"who started the last times that player was out" as an input, which is a
replacement-identification rule that can stand on its own. Build the projected
starting five first, emit group A and B against it, and test on the 1,498-game
fresh-absence subset before committing to the full minutes model. If the
replacement columns move nothing there, E is unlikely to rescue them.

- **Leakage gate.** `select_training_columns()` must accept every new column
  unchanged. Don't add exemptions.
- **Minus-line column.** `LU_PROJ_TOTAL_MINUS_LINE_BEFORE` uses the closing line
  of the target game, which is legitimate because the line is known before tip.
  Check it passes the `training_pipeline.config` `LEAKING_TARGET_COLUMNS` /
  `OUTCOME_ONLY_COLUMNS` gates.

### 8.2 Wiring and schema

**The family is switchable, and off by default.** Decided 2026-09-22. The point
of the campaign in §8.4 is to find out whether these columns are worth
anything, so one builder has to be able to emit the dataset with and without
them. Two existing flags are the precedent to copy; do not invent a new
mechanism:

- `injury_report_features` (`create_train_data.py` → `create_df_to_predict`) —
  a boolean that **keeps the schema version** and distinguishes the build by a
  **filename suffix** (`_without_injury_reports`).
- `include_same_season_referee_variants` — a default-`False` additive flag
  threaded from an `argparse` `store_true` down into the builder.

Thread `lineup_features: bool = False` the same way:

| Layer | Change |
|---|---|
| `create_df_to_predict.py` | `lineup_features: bool = False` parameter; guard the `add_lineup_features(...)` call with it, attached where `add_starter_history_features` is (line ~696) |
| `create_base_game_features.py` | same parameter and guard (line ~424) |
| `create_train_data.py` (`main`) | `lineup_features: bool = False`, passed straight through |
| `create_train_data.py` (CLI) | `--lineup-features`, `action="store_true"` — opt-in, because the default is off |
| filename | `variant` suffix `_with_lineup_features` when on, alongside the existing `_without_injury_reports` logic |

**Schema version: stay on `2_6` while the family is experimental.** The owner's
call, 2026-09-22. With the flag off the builder must emit a file **identical**
to today's 2_6, which means the campaign's control arm is the existing 2_6 file
and needs no rebuild — only the treatment arm is generated. This follows
`injury_report_features`, whose CLI help says as much: *"The current schema
version is retained with a distinct filename suffix."*

Bump `dataset_versions.py` to **`2_7`** and add a History entry **only if
go/no-go G passes and the flag's default flips to `True`** — at that point the
columns are part of the standard dataset and the version contract in the
`dataset_versions` docstring ("bumped whenever a regenerated CSV gains or loses
columns") applies. Until then a 2_6 file with the suffix is the honest label:
same pipeline, one extra opt-in family.

- Attach in **both** `create_training_data/create_base_game_features.py` and
  `create_df_to_predict.py`, the same way commit `73eef65` wired starter history.
- **Test the flag, not just the features.** Add to `tests/test_lineup_features.py`:
  with `lineup_features=False` no `LU_*` column exists and the frame's columns
  equal the current 2_6 set; with it `True` every column in §8.1 is present and
  no other column changes value. That second half is what stops the flag from
  silently perturbing the control arm.
- **Serving path uses the same flag.** `predict_nba_games.py` must pass whatever
  the production model was trained with; a model trained without `LU_*` columns
  and served with them (or the reverse) is a schema mismatch the bundle's
  feature list should catch, but the flag should not be left to chance.
- Update `docs/feature_engineering_overview.md` (the feature table and §6 "Open
  work") and `docs/README_Training Data Processing.md`.
- **Serving path.** For scheduled games, `create_df_to_predict` needs:
  - today's latest injury report, which already exists;
  - ratings and synergy fitted through yesterday;
  - a minutes model fitted on everything before today.

  Make sure none of these read the scheduled game itself.

### 8.3 Leakage tests (`tests/test_lineup_leakage.py`), mandatory

1. **Perturbation.** Build features for game G. Change G's own stints,
   rotation and box score, and any same-day game's. Rebuild and assert every
   `LU_*` column is identical.
2. **Fit windows.** Every rating, synergy and minutes-model fit used for game G
   has a max training game date < G's date.
3. **Injury cut-off.** Injury inputs for G come from the last report before G's
   tip-off. Reuse the existing `report_state` helpers; don't re-derive them.

### 8.4 Evaluation (load the `experiments` skill first)

1. **Go/no-go G (cheap).** On 2021–2025, walk-forward, regress `LINE_ERROR`
   on `LU_PROJ_TOTAL_MINUS_LINE_BEFORE`.
   - Slope ≈ 0 means the market already prices it. Report that and stop; don't
     tune your way past it. The owner has had null results like this before;
     see the memory note on team line-error history.
2. If the slope is clearly positive: run a campaign under
   `experiments/lineup_projection_2026_09/`:
   - control vs treatment, both targets, multiple seeds. The two arms are the
     same builder run with `lineup_features` off and on (§8.2), so the control
     is the existing `2_6` file and only the treatment arm is generated.
   - Report overall results and the subsets:
     - games with ≥1 projected starter replaced;
     - a key player's first game out (`INJ_KEY_PLAYER_FIRST_GAME_OUT_PTS_BEFORE == 1`);
     - games with a Questionable starter.
3. Report confidence intervals. The known trap: 2021 alone carries the base
   model's win rate (58% vs 50.8% excluding it). Show results with and without
   2021.
4. **Meta-learner.** If the lineup family changes production base models,
   `scripts/build_meta_learner_training_data.py` must be re-run.
5. **Only then** flip the `lineup_features` default to `True` and bump the
   schema version to `2_7` (§8.2).

### 8.5 Phase G as built, and the 2025-26 holdout (2026-09-23)

**New data.** The 2025-26 rotation backfill finished: stints for 1,303 of
1,315 games (99.1%; 7 `bad_rotation_interval`, 2 `overlapping_player_intervals`,
1 `points_mismatch`, 2 unclassified `ValueError`). The rating cache was rebuilt
through 2026-06-30 with the same lambdas; it agrees with the previous cache
on every shared date (max difference 1e-6). 2025-26 is the first season no
lambda, window or `k` was tuned on.

**Reproducibility.** The 7.1d / 6.3 / 6.1b numbers came from drivers that were
never committed. `scripts/lineups/evaluate_game_projection.py` now regenerates
them from the modules. A rebuild matches the 2021-24 slope (+0.288 against
+0.291) but not every game (`impact` differs by more than 0.01 on ~25% of
games, mostly early-season; the old calibration scheme could not be
recovered). From here, the script's output is the reference.

**Result** (`evaluate_game_projection.py`, `RECENT_GAMES = 10`, 6,588 games):

| Window | n | `LINE_ERROR ~ impact` slope | 95% CI (date-clustered) |
|---|---|---|---|
| 2021-2024 | 5,266 | +0.289 | [+0.102, +0.474] |
| **2025-26 holdout** | 1,322 | **+0.206** | [-0.113, +0.515] |
| 2021-2025 | 6,588 | +0.269 | [+0.115, +0.423] |

Positive in all five seasons (+0.40, +0.00, +0.25, +0.52, +0.21). The holdout
agrees in sign and size but is not significant on its own. `proj - line` stays
null (+0.071, [-0.020, +0.164]). Controlling for `TOP1_INJURED_PLAYER_PTS` and
the fresh-absence flag (OLS, date-clustered SEs) the impact slope **rises** to
+0.35 [+0.17, +0.53], so the column is not a restatement of the injury
features already in the dataset -- the bar §5 set.

**The accuracy shape was partly a setting.** With the 5-game minutes average,
accuracy fell as |impact| grew (7.1d's worrying sign). With the calibrated
10-game default it rises: 51.2% at any size, 52.9% at >= 2, 53.3% at >= 4
(2021-24); 2025-26 is 51.6% to 53.0%. No interval clears the 52.38% break-even.

Two robustness checks: dropping the `DND`/`NWT` injury rows that enter the box
score with 0 minutes (see open issue below) gives +0.254 pooled; a 5-game
window gives +0.272.

**What was built.** `data_processing/lineups/features.py`:

| Column | Meaning |
|---|---|
| `LU_PROJ_TOTAL_BEFORE` | F v1 total, calibrated on the previous 200 games |
| `LU_PROJ_POSS_BEFORE` | projected possessions |
| `LU_PROJ_TOTAL_SD_BEFORE` | spread across availability scenarios |
| `LU_ABSENCE_IMPACT_PTS_BEFORE` | total with absences minus full health |
| `LU_ABSENCE_IMPACT_PACE_BEFORE` | the same in possessions |

Deviations from §8.1-8.2, each deliberate:

- **Five columns, not the §8.1 list.** Synergy and `LU_PROJ_TOTAL_MINUS_LINE`
  measured null; the team-level and replacement columns have no evidence yet.
  Add them on evidence.
- **Closing builder only.** `create_base_game_features` is the intermediate
  dataset's base, whose injury status is per snapshot -- phase 2 (§9).
- **Serving is not wired.** No production model has these columns, and the
  rating cache has no daily refresh. Both are prerequisites for G′.
- **Calibration window in games (200), not an expanding mean.** An expanding
  mean depends on how many seasons the build loads, so a training build and
  same-day serving would disagree. Swept {100, ..., 800}: flat MAE, 200 has
  the least bias.

**Campaign constraint.** Ratings start in 2021-22. On any window starting
earlier, `find_season_gated_columns` drops the whole family (100% NaN before,
~0% after), so the campaign must train and evaluate on 2021-22+.

**Open issue, not fixed.** `availability.build_player_nights` appends 0 minutes
for `DND - Injury/Illness` / `NWT` box-score rows (700-1,400 a season) and
refreshes `last_seen`, contrary to its docstring. The check above shows it does
not drive the result; fix it before the family is promoted.

### 8.6 Feature audit: where the signal is, and what was wrong (2026-09-23)

The campaign was paused to check the calculations first. Everything below is
measured with `evaluate_game_projection.py` on the committed code, 2021-22 to
2025-26, 6,584 games, date-clustered 95% intervals.

**The absence impact splits by channel, and only two channels carry it.**
`project_with_counterfactual` now separates the impact into the points the
absentees would have *scored* (offense), *prevented* (defense) and the
possession change (pace); the three add up to the total exactly.

| Channel | `LINE_ERROR` slope | Line move (open to close) per point |
|---|---|---|
| offense | -0.082 [-0.310, +0.166] | +0.37 |
| **defense** | **+0.446** [+0.193, +0.705] | +0.25 |
| **pace** | **+0.499** [+0.224, +0.769] | +0.52 |
| defense + pace | **+0.451** [+0.268, +0.630]; 2025-26 alone +0.398 [+0.035, +0.730] | |

The market prices an absence's **offense** fully and under-prices its
**defense and pace**. That is also why the absentees' raw points per game
predicts the error with a *positive* sign alongside the rating impact: the
line over-reacts to "a scorer is out" and under-reacts to "a defender is out".
Positive in all five seasons (+0.57, +0.26, +0.34, +0.68, +0.40).

The effect concentrates where it should. Betting the sign of defense + pace:
57.9% at |x| >= 3 (n=859, 2021-24) and 54.9% (n=257) in 2025-26; the top
decile's mean line error is +4.2. It survives removing the COVID window
(Dec 2021 - Jan 2022), April and the playoffs (58.0% [54.6, 61.4] at |x| >= 3),
and is mostly one-sided: an absent defender -> OVER.

**Honesty about the search.** About fifteen candidate quantities were tried
before this split was chosen, on all five seasons including 2025-26. 2025-26
is therefore **not** an untouched test of this particular hypothesis. The
seasons that are: 2019-20 and 2020-21, being backfilled now.

**Pre-registered for 2019-20 and 2020-21** (written before those stints exist):

- slope of `LINE_ERROR` on `LU_ABSENCE_IMPACT_DEF_PTS_BEFORE + LU_ABSENCE_IMPACT_PACE_PTS_BEFORE` > 0;
- the offense channel's interval covers 0;
- sign accuracy at |defense + pace| >= 3 above 52.38%.

- *(added after the lineup-style probe below)* the absence-driven shift in
  **3PA per FGA** has a positive `LINE_ERROR` slope while the line does not
  move with it; the shift in **pace** has a positive slope.
- *(added after the bench probe below)* `LU_ABSENCE_IMPACT_BENCH_DP_PTS_BEFORE`
  has a positive `LINE_ERROR` slope **alongside** the rest of the defense +
  pace impact (both in one regression), and its per-point slope is at least
  the rest's.

Caveats set in advance: 2020-21 rotation coverage is under 45% and ends
around game 400, so ratings are thin, and the 14-day staleness rule leaves the
rest of that season NaN; 2019-20 includes the bubble.

**Home and away.** The per-side impacts are symmetric (+0.31 home, +0.24
away) and their interaction -- both teams depleted -- is null (-0.006). A
matchup term (each offense against the other defense) is null (-0.006), and
the summed pace ratings marginal (+0.17 [-0.01, +0.35]). For the spread, the
projected margin against the spread line has a small positive slope, +0.112
[+0.025, +0.199]; the margin impact alone does not separate (+0.130
[-0.014, +0.265]). The per-side and margin columns are emitted because the
spread target and tree interactions need them, not on strong evidence.

**Lineup-vs-lineup style interactions: absent.** Each player gets a style
profile from earlier seasons' stints only (his lineups' on-court 3PA/FGA,
FTA/FGA, offensive rebound %, turnovers per possession, points per possession
and pace, on offense and on defense, shrunk to league average); a lineup's
traits are the mean of its five. On 319,742 stint-directions in 2022-2025 the
additive model is strong (1 SD of offensive 3PA trait moves 3PA/FGA by 3.0
points; pace traits move pace by 2.7 possessions per 48 each), while the
offense x defense interaction adds at most +0.003% out-of-sample, and the 5x5
residual grids show only noise. A lineup with traits A against one with traits
B plays like A + B. Do not build lineup-matchup interaction features; the
earlier synergy and matchup nulls say the same.

**Matching similar past matchups: also null.** The owner's variant of the
same question: give each actual starting five a vector of how that exact five
has played together before (11 rates, offense and defense, shrunk to its
players' profiles; median 94 prior possessions together), and predict what
happens in starters-vs-starters time (9.6 minutes a game on average) from the
**joint** vector, after removing what adding the two vectors predicts. Out of
sample (train 2022-23, test 2024-25, ~1,900 games): K=150 nearest past
matchups correlate +0.017 / +0.010 / +0.011 with the residual in points per 48
/ pace / points per possession, and a depth-3 tree -0.003 / +0.022 / +0.001;
both make the out-of-sample fit *worse*. The noise floor on a correlation
there is about +/-0.045, so any interaction is below what this data can see.

**Borrowing from similar vector pairs across all teams: one exception.**
The same question with a far larger pool: every 2022-23 stint (155,808
offense-five vs defense-five pairs, any teams) described by the joint vector
[offense five's 6 offensive traits, defense five's 6 defensive traits], and the
K most similar pairs predicting the non-additive residual of 60,000 2024-25
stints. Points per possession: null (game-level correlation -0.01 to -0.02).
Pace: null and harmful. **3PA per FGA: a small, real non-additive component**
-- stint correlation +0.016 to +0.019, game-level +0.067 to +0.079 (K = 200 to
3,000; ~0.02 standard error), out-of-sample R2 gain +0.4-0.5% at K >= 1000.
Caveat: the neighbours are averaged unweighted, so very short stints add
noise; that biases every row toward null, which makes the 3PA result more
credible rather than less. It is the same stat whose absence-driven shift the
line ignores (below). Not yet tested against the line; pre-register it with
the 3PA shift for 2019-20 and 2020-21.

**Implemented as `lineups/style_matchup.py` (four `LU_*` columns).** Walk-forward
player style traits (decayed, 365-day half-life, shrunk), a pool of every
past stint direction with its joint vector as it stood on its date, a monthly
refit of the additive 3PA model and an FGA-weighted K=1,000 neighbour index
over its residuals, queried with tonight's minutes-weighted rotations -- two
queries per game, ~2 minutes for 2021-2026. Validated with
`evaluate_game_projection.py --style`:

| Check | Result |
|---|---|
| interaction vs the game's non-additive 3PA rate | **+0.066** (6,521 games; by season -0.05, +0.07, +0.12, +0.00, +0.08) |
| 3PA-rate MAE, additive -> with interaction | 0.0370 -> 0.0363 |
| `LINE_ERROR` ~ interaction | +0.10 per SD [-0.33, +0.53] -- null |
| `LINE_ERROR` ~ absence-driven 3PA shift | +0.17 per SD [-0.27, +0.61] -- null |

The matchup effect on three-point **volume** is real and survives the move to
rotation averages. It does not reach the totals line, and the probe's
"3PA shift the line ignores" (+0.55 below) does **not** reproduce under this
walk-forward construction -- most likely a product of testing six stats at
once. The columns stay behind the flag so the pre-registered 2019-20 / 2020-21
test judges this exact code. A 180-day evidence guard leaves games NaN when
the stint archive has a hole (it had projected 2019-20 from 2018 stints).

**Player-vs-player matchups: not tested, data identified.** Stints carry team
counts only, so "does defender X change what player Y does" needs
`BoxScoreMatchupsV3` (in the installed `nba_api`): per game and per
offensive-player x defender pair, matchup minutes, partial possessions, the
attacker's points, FGA, 3PA, FTA, shooting fouls and switches. One call per
game, so ~1.1 h per season at the 3 s pacing, and it competes with the
rotation backfill for the same rate limit. Two expectations to test against:
for a **total**, a suppressed star's usage moves to teammates, the same
dilution that made the minutes model irrelevant (6.3); the realistic payoff is
a sharper **defensive** rating -- the channel the market under-prices -- and
player-level projections. Season coverage of the endpoint is unverified.

**What the market does with style shifts.** Computing, per game, how much the
expected absences shift each stat (tonight's rotations against full health,
through the additive model), per 1 SD of the shift:

| Shift in | Line moves with it (open to close) | `LINE_ERROR` slope |
|---|---|---|
| points per possession | +0.51 | -0.20 [-0.71, +0.32] -- priced |
| pace | +0.49 | **+0.60** [+0.13, +1.08] -- under-priced |
| 3PA per FGA | -0.06 | **+0.44** [-0.02, +0.90]; jointly **+0.55** [+0.08, +1.02] -- ignored |
| FTA per FGA | +0.12 | +0.05, null |
| offensive rebound % | -0.02 | +0.09, null |
| turnovers per possession | -0.16 | +0.04, null |

The pace row confirms the channel result independently. The 3PA row is a new
hypothesis, not a finding: six shifts were tested and its interval barely
clears zero. Both are pre-registered above. Exploratory scripts only
(scratchpad, not committed); if the 2019-2020 test holds, the style traits
belong in a walk-forward module fitted like the ratings (ridge per factor
rather than on-court averages, which are confounded by teammates).

**Defects fixed** (projection MAE 14.612 -> **14.459**, and `proj - line`
went from null to +0.145 [+0.041, +0.243]):

| Defect | Size | Fix |
|---|---|---|
| Calibration window mixed playoffs into season starts | playoff games score 6-15 below a regular-season projection; Oct - mid-Nov under-projected by ~2.8 | `walk_forward_offset(phases=game_phase(...))`: each phase calibrates on its own games |
| G-League assignments projected to play | 11,596 listings on a projected roster since 2021-22, 1,103 regulars | `availability.roster_exclusions`: off the roster for that game, in both projections, so not counted as an absence either |
| `DND`/`NWT` rows read as 0-minute appearances | 700-1,400 rows a season | only a coach's decision (or a blank comment) is a zero-minute appearance |

**Checked and left alone.** Projections over-shoot actual totals less when
many minutes go to unrated players: +2.6 points of bias above 20% unrated
minutes (419 games). A replacement-level prior would fix the level, but the
`LINE_ERROR` slope on the unrated share is null, so it is deferred. Players
returning from an absence (a "change versus the recent rotation" column):
null (-0.002).

**The bench (2026-09-23).** The starting five is presumably what the market
watches, so the probe asked whether the rest of the rotation carries line
error the starters do not. Each team's rotation was split into the projected
five and the bench (6+ by minutes), and the defense + pace impact into the
absentee's own minutes (*lost*) and the change in everyone else's (*replaced*),
with the bench's share of the latter separated. Pre-registered before running;
fit 2021-24, replication 2025-26, slope per point of projected impact:

| Quantity | 2021-24 | 2025-26 |
|---|---|---|
| lost (the absentee) | +0.47 [+0.23, +0.70] | +0.31 [-0.05, +0.68] |
| replaced (everyone absorbing his minutes) | +0.59 [+0.23, +0.95] | +0.70 [+0.17, +1.24] |
| of which absorbed by the healthy bench | +0.65 [+0.00, +1.29] | **+1.29** [+0.34, +2.24] |
| bench level at full health, given the impact | -0.04 [-0.22, +0.14] | +0.29 [-0.04, +0.62] |
| starters' level, given the impact | +0.00 | -0.02 |

Null, and not pursued: bench and starter *offense*; the deep bench (9+) times
the closing spread (garbage time); each team's empirical second-unit pace from
stints with <= 2 of that night's starters on the floor. The closing spread's
own size leans OVER (+0.63 per SD in both periods), but that is a market
effect on a column the model already has.

So the bench matters only through absences: the market prices a missing
player by who he is, less by who plays his minutes, and least when a reserve
does. The lost/replaced slopes are not yet distinguishable (p = 0.22 pooled),
so this is carried as **one** column, `LU_ABSENCE_IMPACT_BENCH_DP_PTS_BEFORE`
(`game_projection.bench_replacement`), pre-registered above. Through the
evaluation script it reproduces the probe (r = 0.997): +0.50 [-0.11, +1.11]
before the holdout, +1.05 [+0.26, +2.00] on it, positive in 4 of 5 seasons
(2023-24 -0.65). Around 15 coefficients were read in the probe, so expect
about one to look good by luck; 2019-20 and 2020-21 decide.

**Columns now** (`LINEUP_FEATURE_COLUMNS`, 17 -- the 13 below plus the four
three-point matchup columns of `style_matchup.py`): the level
(`LU_PROJ_TOTAL_BEFORE`, `LU_PROJ_POSS_BEFORE`, `LU_PROJ_TOTAL_SD_BEFORE`,
`LU_PROJ_MARGIN_BEFORE`), the impact (`LU_ABSENCE_IMPACT_PTS_BEFORE`,
`LU_ABSENCE_IMPACT_POSS_BEFORE` -- renamed from `..._PACE_BEFORE`, which was
in possessions), its channels (`LU_ABSENCE_IMPACT_{OFF,DEF,PACE}_PTS_BEFORE`), the bench's
share of the replacement (`LU_ABSENCE_IMPACT_BENCH_DP_PTS_BEFORE`),
its sides (`LU_ABSENCE_IMPACT_PTS_BEFORE_TEAM_{HOME,AWAY}`) and its margin
(`LU_ABSENCE_IMPACT_MARGIN_BEFORE`).

---

## 9. Phase 2 (after G passes)

- **Intermediate dataset:** recompute the scenarios per snapshot, with injury
  status as of each snapshot. Ratings and the minutes model don't change within
  a day, so only the §7 aggregation reruns.
- Rotation template (§7.2).
- Replacement-level prior for players with little history.
- Player-level projections as their own features, e.g. the top-3 players'
  projected points change versus their form.

---

## 10. Order of work and hand-back points

| Step | Deliverable | Stop and report to the owner if |
|---|---|---|
| A | client + archive + manifest; the session fix committed | — |
| A′ | backfill running (overnight, resumable) | manifest ok < 99.5% |
| B′ | build stints for every archived game, not just the 112 pilot | stint pass rate < 99% |

**The backfill is not on the critical path to the decision (2026-09-22).** It
was being treated as one, which is wrong. Three questions need very different
amounts of data:

| Question | What it needs | Status |
|---|---|---|
| Write the feature code | Almost no data — a rating cache is enough | Unblocked now |
| **Decide whether the family is worth anything** (gates C and G) | The 2021-2025 window this plan already specifies | **5,257 of 6,572 games (80%) built; only 2025-26 missing** |
| Train production models with the family | Stints back to the training start | ~54% of training rows would be NaN |

Four complete consecutive seasons (2021-22 through 2024-25) are ample for an
MAE comparison and for a single-regressor slope. Run the gates on what is
built; do not wait for the backfill.

The third row matters less than it looks. Of the six production prefixes,
`last_3_seasons` (two models) is **fully** covered by what is already built and
`last_5_seasons` (two models) misses only 2020-21; only `full_dataset` carries
the 2017-2020 hole, and XGBoost takes NaN natively, so the feature simply does
not inform those older rows.

Revised order, cheapest decision first:

1. Gate C on 2021-2024. **Tune and gate on disjoint windows** — tune the lambdas
   on 2022-23/2023-24 and gate on 2024-25 out of sample. The README's example
   does both on 2021-2025, which flatters the result.
2. If C passes: build D, E, F and `features.py`; run gate G on 2021-2024.
3. **Only if G shows a clearly positive slope**, spend the 3,175 calls on
   2019-20, 2020-21 and the rest of 2018-19 to fill in `full_dataset`.

2025-26 (1,027 calls) is worth finishing regardless of the gates, because it is
needed to serve predictions next season either way.

**Warm-up caveat.** The plan assumed 2018-19 and 2019-20 as rating history
before the evaluation window. Neither is available: 2019 and 2020 have no
rotation at all, and 2018-19 has only 112 games from October, which a 180-day
half-life decays to ~1.5% weight three years later. **2021-22 is therefore the
warm-up season and the ratings start cold there.** Read any gate result that
includes late 2021 with that in mind.
| B | stint builder + validation + DB load | stint pass rate < 99% |
| C | walk-forward player ratings | **go/no-go C** fails |
| D0 | projected starting five + replacement identification (no minutes model) | — |
| D1 | counterfactual absence impact + replacement synergy (§8.1 additions), tested on the fresh-absence subset | report the result, but **do not stop** — see below |
| D, E | synergy; minutes model | **go/no-go E** fails (continue with the fallback, but report it) |
| F | projection v1 **and §7.1b shared-minutes weighting** | — |

**D0/D1 are an ordering, not a reduction (owner, 2026-09-22).** They are the
cheapest falsifiable slice of the hypothesis and worth running first, but D, E
and F are all in scope and a weak D1 does not cancel them. The agent proposed
narrowing twice; the owner declined twice. Implement the plan.
| G | features behind the `lineup_features` flag (§8.2), leakage tests, docs, regression test, campaign | **go/no-go G** slope ≈ 0 |
| G′ | flip the flag's default and bump to schema 2_7 | only after G passes |

Work in small commits per step, with tests. Run `pytest` (the whole suite, ~2.5
min) and `ruff check` before each commit.
