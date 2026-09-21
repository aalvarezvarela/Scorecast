# Lineup stints, projected minutes and a bottom-up game projection

Status: **phases A and B implemented; C solver implemented, calibration and
acceptance gates pending.** Written 2026-09-18 on
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
games occupy 1.22 MB, so the full backfill would be ~120 MB.

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

**Go/no-go C.** Build `PROJ_TOTAL` from ratings alone, with minutes = each
player's recent average (no minutes model yet). On 2021–2025 walk-forward,
compare its total-points MAE with a rolling team-average baseline.
- If it is not better, stop and report back before building D–F.

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

### 7.2 v2 (optional): rotation template

Model which fives share the floor from recent substitution patterns:
- who starts each quarter;
- the typical time of the first substitution;
- bench staggering.

This distributes minutes across specific lineup pairs so exact-five synergy
applies beyond the starting units. **Only do this if F v1 clears go/no-go G.**

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

- **Leakage gate.** `select_training_columns()` must accept every new column
  unchanged. Don't add exemptions.
- **Minus-line column.** `LU_PROJ_TOTAL_MINUS_LINE_BEFORE` uses the closing line
  of the target game, which is legitimate because the line is known before tip.
  Check it passes the `training_pipeline.config` `LEAKING_TARGET_COLUMNS` /
  `OUTCOME_ONLY_COLUMNS` gates.

### 8.2 Wiring and schema

- Attach in **both** `create_training_data/create_base_game_features.py` and
  `create_df_to_predict.py`, the same way commit `73eef65` wired starter history.
- Bump `src/nba_ou/config/dataset_versions.py` to **`2_7`** and add a History
  entry.
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
   - `2_6` control vs `2_7`, both targets, multiple seeds.
   - Report overall results and the subsets:
     - games with ≥1 projected starter replaced;
     - a key player's first game out (`INJ_KEY_PLAYER_FIRST_GAME_OUT_PTS_BEFORE == 1`);
     - games with a Questionable starter.
3. Report confidence intervals. The known trap: 2021 alone carries the base
   model's win rate (58% vs 50.8% excluding it). Show results with and without
   2021.
4. **Meta-learner.** If 2_7 changes production base models,
   `scripts/build_meta_learner_training_data.py` must be re-run.

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
| B | stint builder + validation + DB load | stint pass rate < 99% |
| C | walk-forward player ratings | **go/no-go C** fails |
| D, E | synergy; minutes model | **go/no-go E** fails (continue with the fallback, but report it) |
| F | projection v1 | — |
| G | features, schema 2_7, leakage tests, docs, regression test, campaign | **go/no-go G** slope ≈ 0 |

Work in small commits per step, with tests. Run `pytest` (the whole suite, ~2.5
min) and `ruff check` before each commit.
