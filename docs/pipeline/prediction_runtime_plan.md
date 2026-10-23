# Prediction runtime plan: shorter daily runs on the self-hosted runner

Status: **Plan** (2026-10-23, branch `perf/prediction-runtime`)

## Goal

On the self-hosted runner a warm `nba_predictor_daily` run takes ~19–23 min,
and the first run after any builder change takes ~70 min
(`docs/serving/runner_timings.md`). The snapshot scheduler books up to ~20 runs
a day, so the warm run sets how dense the schedule can be. The cold run sets how
many booked runs a deploy day loses.

Targets:
- a warm run of ~8–10 min;
- no cold runs from code changes that leave 2_5 values unchanged.

## Acceptance rule

**Every change must leave the outputs exactly as the current code produces
them.** Nothing below is allowed to move a value:

- Frames: tonight's closing and intermediate rows, and the scoring sidecar, are
  bit-identical to the current code's (`scripts/perf/compare_builds.py --exact`).
- Predictions: the 51 slots' predictions are identical.
- Comparisons run on a fixed moment, a replayed date with an explicit `as_of`,
  so the two builds read the same ticks and reports. Two dates:
  - 2026-10-22 (2 games): today's slate;
  - 2026-04-10 (15 games): the reference date of the earlier perf work.
- The reference outputs come from the current code (step 0). Each change is
  compared against them, not against the previous change.

Schema 2_5 is frozen: its finished seasons do not change. Where today's code
spends time re-checking that, the plan trusts what is stored instead. A
correction to a past season's data then needs an explicit rebuild of the
caches; a command for that is part of A.

## How "nothing changes" is tested

Two levels, both required before a change is merged.

**1. End-to-end parity: old code against new code, same inputs.** A new script,
`scripts/perf/prediction_runtime_parity.py`, runs what `predict_nba_games.py`
runs for one replayed moment and dumps the result:
- the closing frame, the intermediate frame and its scoring sidecar;
- the predictions of all 51 slots.

The run is made the same on both sides:
- **Inputs:** the replayed date and `as_of` come from
  `live_replay.replay_live_game_day` / `replay_scheduled_data`. Every database
  read is recorded once with `db_tape.record`, and the candidate replays the
  tape (`db_tape.replay`). So the databases' daily changes cannot move a value.
- **Models:** the reference records the `fit_id` each slot resolved to, and the
  candidate predicts with those exact builds. A promotion in between cannot
  move a prediction.
- **Code:** the reference is a worktree pinned to the commit before the plan
  (`--code-root`, as the existing perf scripts do).
- **Hash seed:** `PYTHONHASHSEED=0`, because the `ROSTER_*_PCT` columns vary
  without it.
- **No shared writes:** persistence is off, and the memo and input cache are
  local directories.

The two dumps are compared with `compare_builds.compare(exact=True)` for frames
and exact equality for predictions. The result must be `RESULT: EQUAL`.

Dates:
- 2026-04-10 (15 games, regular season, a full 2025-26 season of history behind
  it);
- 2026-10-22 (2 games, start of season: the current season is almost empty).

Each date is run at two moments: a morning `as_of` (long horizons, referees
masked) and one ~1 h before the first tip. For the caches (A2, F), each
candidate is run twice in a row, so both the run that fills a cache and the
run that reads it are compared with the reference.

For B, the latest rows are also compared with the latest rows of a full build
from the *new* code (`--full-intermediate`), so B is tested on its own as well
as through the end-to-end comparison.

**2. Unit tests in `tests/` (run in CI).** Each change adds tests that fail if
the shortcut ever returns something different from the slow path, on small
synthetic frames:
- A1: an expired but stored finished season is served from S3, and its frame
  equals the database read. The current season is always queried.
- A2: a frozen-key hit returns the same table as a validated recompute. The
  current season is never frozen. A new base schema or main book misses. The
  escape hatch restores validated keys.
- B: `latest-only` gives the same latest rows as the full build, on a fixture
  with games at two tip times. It falls back to the full build when any slot
  reads `own`.
- C: the in-process load cache returns frames equal to a fresh load, and
  changing a returned frame does not change the next one.
- D: a cached model with a matching sha256 is used, and a mismatched one is
  downloaded again.
- F: a day-cache hit equals a recompute. Changed current-season inputs (a
  corrected game) miss. A new day misses.

The existing parity scripts (`live_replay_parity.py`,
`closing_prediction_parity.py`, `point_in_time_parity.py`, `target_parity.py`)
must still report `EQUAL`.

## Step 0: reference and profile (no code change)

- Write `scripts/perf/prediction_runtime_parity.py` (see above). Build and dump
  the reference outputs of both dates and both moments with the current code,
  recording the database tape and the `fit_id` of every slot.
- Profile one warm run per stage and per slot. In particular, split the
  2–4 s per slot into model loading, input limits and prediction, so D is aimed
  at the real cost.
- Record the stage timings of the current code. Every later step reports the
  same table.

### Step 0 results (2026-10-23)

References recorded with the code at `a6cd8ce` (worktree
`../NBA_ou_runtime_ref_a6cd8ce`), in `artifacts/perf/runtime/ref_*` (tape,
inputs, builds, frames, predictions, `timings.json`). The memo read the S3
memo and wrote to a shared local directory. The input cache was off, as every
taped run needs, so the database reads are the uncached ones.

| Run | Games | Inter. rows | Closing rows (s) | Intermediate rows (s) | 51 slots (s) | Memo | Fork timeout |
|---|---:|---:|---:|---:|---:|---|---|
| 2026-04-10 18:00 ET | 15 | 261 | 669 | 3009 | 127 | missed | yes |
| 2026-04-10 11:00 ET | 15 | 261 | 507 | 765 | 84 | hit | no |
| 2025-10-22 18:00 ET | 12 | 207 | 644 | 2143 | 102 | missed | yes |
| 2025-10-22 11:00 ET | 12 | 207 | 470 | 796 | 88 | hit | no |
| Self-replay of the first | 15 | 261 | 522 | 2213 | 132 | missed | yes |

- **The harness works.** The self-replay (same code, the reference's tape,
  inputs and builds) is `EQUAL` on both frames and all 765 predictions. Only
  `PREDICTED_AT_UTC`, a wall-clock stamp, is left out.
- **Inside the intermediate rows, warm (11:00 ET runs):** base features
  320–390 s; referees ~180 s; injury report states ~95 s; injury horizons and
  market dynamics 125–165 s. Cold, the last stage takes 25–30 min: the forked
  pool times out at 600 s and every horizon reruns in turn.
- **Slots:** 1.9 s median per slot, of which ~0.8 s is resolving the build in
  S3 and ~1 s predicting. D saves ~0.5–1.5 min at most.
- **Why the memo missed with unchanged data.** Two runs a few minutes apart
  loaded identical odds values for 2018–2024, in a different row order. The
  odds query sorts by `game_date DESC` only
  (`fetch_data_from_odds_sportsbook_db.py`), so games on the same date come
  back in whatever order Postgres returns them. The memo's per-season digest
  depends on row order, so it reports "inputs changed" and recomputes. The
  self-replay missed for this reason: the 2025-10-22 runs between the two
  overwrote the per-season entries with digests from another row order. While
  the S3 input cache is expired, every production run reads odds from the
  database, which explains the misses seen on every run. A2's frozen key
  removes the problem for finished seasons. F needs an order-insensitive
  digest of the current season (rows sorted by a stable key before hashing),
  which changes no value.

## Changes, in order

### A. Trust the finished seasons of 2_5 (expected −5 to −6 min per run; no cold runs)

**A1. S3 input cache: serve finished seasons whatever their age.** Today a
season is read from S3 only if `refresh_input_cache.py` validated it within
36 h. Otherwise all its rows are queried again
(`season_tables.MAX_VALIDATION_AGE`, also read by `season_queries` and
`line_history_aiven.fetch`). For seasons before the current one, the plan
serves the stored entry whenever it exists. The current season is always
queried, as today. The refresher then only matters when a season finishes.
- One-time check that nothing changes: for every table and finished season,
  compare the S3 entry with a fresh database read (read-only). A mismatch is
  reported, and that entry is refreshed before A1 is enabled. That refresh
  writes to S3, so it needs a go-ahead.
- Measured effect (P3 → P5 in the timings): closing rows 457 → 185 s; retrain
  partition update 309 → 178 s.

**A2. History memo: key finished seasons on the schema, not on code and
input digests.** Today each memo entry is keyed on the source of its module
closure (`history_memo.code_key`) and checked against a chained digest of the
season's inputs. So:
- any change in that closure makes every entry unusable. That is the ~70 min
  cold run after a deploy;
- some entries miss on every run: `intermediate_report_states 2019..2025:
  inputs changed`, and `intermediate_injury_effect_priors 2025`. They are
  recomputed each time, ~1–2 min.

For seasons up to `max_season`, the plan keys entries on a frozen key: base
schema version (`2_5`), memo name, season and main book. It skips the digest
comparison, and with it the hashing of every input frame.

**As implemented (de2f5ef).** Step 0 showed that the misses came from the
digests changing with row order, not from validation being slow. So the
digests stay as a safety net, and their cause is fixed instead:
- `history_memo.code_key` is the frozen key: `frozen-2_5-<main book>-<roots
  hash>`. `NBA_OU_HISTORY_MEMO_FROZEN=0` returns to the source key.
  `NBA_OU_HISTORY_MEMO_REBUILD=1` reads no entry and rewrites every one it
  computes. That is the rebuild for a deliberate correction.
- `merge_scheduled_odds` sorts by date with a stable sort. Past games keep the
  loader's canonical order, whatever tonight's rows are. The odds digests of
  finished seasons then repeat from day to day.
- A1 is `season_tables.servable`, used by the three readers.
  `NBA_OU_TRUST_FINISHED_SEASONS=0` restores the 36 h rule.
  `refresh_input_cache.py --check` compares without writing. On 2026-10-23 all
  146 stored seasons (15 tables) matched the server, so nothing needed
  refreshing.
- E's fork timeout is in the same commit (600 → 3600 s), because every cold
  test run paid for it.
- A1 cannot be replayed from a tape, because a candidate reading S3 sends
  narrower queries than the tape holds. Its evidence is the `--check` above,
  plus the existing equality tests of S3 assembly. It changes which stored
  seasons are served, not how they are read.
- Kept as today: the current season is never memoised. A new base schema
  (`3_0`) or a different main book gives new keys. An escape hatch
  (`NBA_OU_HISTORY_MEMO_FROZEN=0`) returns to validated keys.
- A rebuild command recomputes the frozen entries from scratch, for when a
  past season's data is corrected on purpose.
- Seeding: the frozen entries are written once from a validated build. These
  are writes to the memo prefix in S3, so they need a go-ahead.
- Check: a build reading only frozen entries is bit-identical to the reference
  on both dates. Also find out why the report-state digests change between runs
  (nondeterministic input order?). The frozen key makes it harmless, but it may
  signal nondeterminism elsewhere.

### B. Build only tonight's latest snapshot when every slot reads it (expected −3 to −5 min)

With `INTERMEDIATE_INPUT_MODE = latest` and `LATEST_INPUT_SLOTS = all`, every
intermediate slot predicts from each game's latest row (`LATEST_INPUT`).
Today the run builds all 16 grid horizons + T-0 for every game, then discards
them. The expensive part is per horizon: report states, availability effects
and shrinkage fits over ~39k history rows (169 s + ~212 s warm).
- New builder option: tonight's target rows only at each game's latest
  snapshot. The injury stage then computes only the grid horizons those rows
  alias to (`nearest_grid_horizon` / `_alias_horizons`). That means one horizon
  per distinct tip-time group instead of 16. The history cutoffs, priors and
  market dynamics are restricted to the horizons needed.
- Used only when every enabled intermediate slot reads latest input. The
  past-snapshots workflow (`own`) keeps the full build. TabPFN's intermediate
  run can use the same option for its single horizon.
- Side effect: the run archive (`recorder.add_frame`) stores the latest rows
  only, and the per-horizon `SNAPSHOT_PASSED` counts in the log go. No stored
  prediction changes.
- Check: the latest rows and every intermediate prediction are bit-identical to
  the reference on both dates. Run the existing intermediate parity scripts
  (`point_in_time_parity.py`, `live_replay_parity.py`) too.

### F. Day cache: the current season up to yesterday, and tonight's crews (expected: measured in step 0)

A2 covers finished seasons. The current season is never memoised, because it
gains games every day. **Within one day** it does not change: last night's
games are in after the morning update, and the runs during the day add none.
- Keep the same per-season memo entries for the current season, keyed on the
  game date and a digest of the current season's inputs as loaded (games, box
  scores, odds after repair, injuries, reports). The first run of the day
  computes and stores them, and later runs read them. If a run's database
  update corrects a past game of the season, the digest changes and the entry is
  recomputed. The current season is small, so the digest is cheap.
- Referee features of tonight's games: reuse the first run's result when the
  crew of a game and the referee history are the same.
- Stored locally on the runner (`~/.cache/nba_ou/day`), removed after the day.
  No S3 writes.
- The gain grows through the season: almost none in October, more by March.
  Step 0's profile at 2026-04-10 sizes it. Skip F if it is small.
- Check: end-to-end parity on the first and second run of the day, and the
  unit tests above.

**Results (2026-10-23).**

Parity against the step 0 references (`artifacts/perf/runtime/`):

| Candidate | Memo | vs reference | Intermediate rows (s), ref → candidate |
|---|---|---|---|
| A, 2026-04-10 18:00 ET | empty, filled here | `EQUAL` | 3009 → 1838 |
| A, 2026-04-10 11:00 ET | filled | `EQUAL` | 765 → 410 |
| A, 2025-10-22 18:00 ET | filled from another date | `EQUAL` | 2143 → 979 |
| A, 2026-04-10 18:00 ET again | after the other date | `EQUAL` | 3009 → 1404 |
| B, 2026-04-10 18:00 ET | filled | `EQUAL` (latest rows) | 3009 → 317 |
| B, 2026-04-10 11:00 ET | filled | `EQUAL` (latest rows) | 765 → 316 |
| B, 2025-10-22 18:00 ET | from another date | `EQUAL` (latest rows) | 2143 → 591 |
| B, 2025-10-22 11:00 ET | filled | `EQUAL` (latest rows) | 796 → 420 |

- **Across dates, the closing memo hits.** The odds order fix made that
  possible. In the 2025-10-22 run, closing rows took 644 → 190 s.
- **The intermediate injury memo misses across these two dates, as it should.**
  Its digest includes the whole All-Star voting table
  (`_injury_history_memo`), and the 2025-10-22 build lacks the January 2026
  votes. Entries are one per season, so the two dates overwrite each other.
  In production every build of a day sees the same votes. They change only
  when new votes arrive.
- **B's tape replay.** B reads tonight's injury reports at fewer cutoffs than
  the reference recorded. `db_tape.replay(subset_cutoffs=True)` serves such a
  read from the recorded rows of those cutoffs. It applies only to the
  cutoff-array report reads, which resolve each cutoff on its own.

Live timings on the self-hosted runner, with the harness of
`docs/serving/runner_timings.md`: 2026-10-23, 12 closing games, 2 games with
intermediate rows (SBR frozen).

| Run | Step 2 | Closing | Intermediate | Slots | Total |
|---|---:|---:|---:|---:|---:|
| P1 before, cold (2026-10-22) | 314 | 717 | 2926 | 196 | 4156 |
| P3 before, warm as CI (2026-10-22) | 113 | 457 | 489 | 88 | 1154 |
| M1 A, frozen memo empty | 251 | 422 | 1468 | 144 | 2304 |
| M2 A, warm | 147 | 189 | 582 | 89 | 1027 |

M2 still missed `intermediate_report_states` 2019–2025 and
`intermediate_injury_effect_priors` 2025, as every run did before. Their
digests chain `report_season_fingerprints`, whose first link hashes the
dimension tables (`ir_reason`, `ir_status`, ...). An injury report ingested
between M1 and M2 (17:30 UTC) is the likely trigger. This is unconfirmed; it
is being watched across the next ingestion.

### C. Load each input table once per run (expected −0.5 to −1 min)

The closing and intermediate builds are independent pipelines by design.
Reusing the closing base in the intermediate build was considered and
**rejected**: it loads a different season span, so expanding features
(`*_SEASON_BEFORE`) would change. What they can share without changing a
value is the loading:
- games, players, odds, referees and injuries are each loaded twice per run;
- an in-process cache of the season-table and season-query loads, keyed on
  (spec, seasons), returns copies, so a caller can never alter another's frame.
- Check: both frames bit-identical.

### D. Cache model builds locally (expected −1 to −2.5 min, if step 0 confirms)

A build is immutable per `fit_id` (`registry_store.read_model_bytes`,
`resolve_build`). Keep `model.json`, `fit.json` and the spec per `fit_id`
under `~/.cache/nba_ou/models` (`NBA_OU_MODEL_CACHE_DIR`), checked by sha256.
Only the channel pointer is read from S3 each run. Scope it to whatever step 0
shows dominates the 2–4 s per slot. Check: predictions identical.

### E. Workflow settings that change no value

- Predictor, past-snapshots and TabPFN workflows: set `NBA_OU_INPUT_CACHE_DIR`
  outside the checkout. `actions/checkout` removes `data/cache/inputs` every
  run. This matters mostly while the S3 cache is expired (−3 min P3 → P4).
- `FORKED_TIMEOUT_SECONDS`: 600 s on the whole pool is too short for 4 cores
  when a horizon has to be computed from scratch, and the serial fallback
  repeats all the work. Make it per horizon, or raise it. After A2 this mostly
  protects a new season's first day.
- Retrain partition update: drop the 3-day overlap rebuild (`OVERLAP_DAYS`).
  It exists to compare stored rows with a rebuild, which a frozen 2_5 does not
  need. Small gain, retrain only. Optional.

## Not in this plan (they would change values)

- Reusing the closing base frame in the intermediate build (see C).
- Dropping the Yahoo scrape (~45 s per run). Yahoo columns feed the current
  models.
- Skipping the database update after the day's first run. It changes which
  data a run sees.

## Expected result

| | Warm run today (P3) | After A–E (estimate) |
|---|---:|---:|
| Step 2 (schedule, reports) | 113 s | ~100 s |
| Closing rows | 457 s | ~150–180 s |
| Intermediate rows | 489 s | ~150–250 s |
| 51 slots | 88–189 s | ~30–60 s |
| **Build and predict** | **~19 min** | **~7–10 min** |
| First run after a builder change | ~70 min | ~same as warm, while 2_5 values do not change |

Re-measure with the timing harness after each change, and on a 10+ game night
once the SBR feed is back.

## Blockers outside this plan

- SBR's live feed has been frozen since 2026-10-20. Every real run stops at the
  feed check.
- The 09:00 database update fails daily at "Repair SBR closing lines" on
  `main`, so `refresh_input_cache.py` never runs. A1 removes the dependence for
  finished seasons, but the failure still needs fixing.
