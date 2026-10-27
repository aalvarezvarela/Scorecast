# Runner timings: prediction and retrain on the self-hosted runner

Status: **Research** (measured 2026-10-22, code at `feat/daily-cronjobs`
80e10f0, on top of dev 451b3ee; production runs re-read 2026-10-27, see
"2026-10-27: warm runs in production")

How long the two heavy daily workflows take on the self-hosted runner, cold
and warm, so the daily schedule (`docs/serving/snapshot_triggers.md`) can be
sized. The measurements run the workflows' own scripts on the runner machine
with every write to shared systems blocked, plus the real step durations of
the 09:00 database update from the runner's own logs.

## Summary

| Workflow | Typical (warm) | Cold | Biggest single cost |
|---|---:|---:|---|
| `nba_predictor_daily` (one run, all 51 slots) | **~19–23 min** | **~70 min** | intermediate rows |
| `retrain_prediction_models` (DB update + refit + smoke test) | **~15–18 min** + unmeasured DB steps | ~30 min | closing partition update, then the 51 fits |

- A warm prediction run fits comfortably in the hourly cadence the scheduler
  books. Runs 30 min apart, such as a T-30 run between two hourly runs, also
  fit, but with only ~10 min to spare.
- A cold prediction run (after any change to the feature-building code) takes
  ~70 min. It swallows the next one or two booked runs: GitHub keeps one
  pending run per concurrency group and drops older pending ones.
- The retrain at 09:00 finishes well before the 10:00 scheduler, unless the
  injury-report step repeats its 55-min run of 2026-10-20 (see below).
- Five findings below cost time, or block the schedule outright. Three are cheap
  to fix.

## 2026-10-27: warm runs in production

Read from the workflow logs of 2026-10-26/27 (commits 1bcfe35 and 0f35ef6),
not a separate measurement. A warm predictor run takes **~13.9 min**, with 1
game or with 9: the cost is the history each run rebuilds and the 51 slots,
not tonight's games.

| Phase (run of 2026-10-27 00:01 UTC) | Time |
|---|---:|
| Job setup (checkout, venv, pip) | 0:51 |
| Database update, SBR feed, schedule and reports | 1:22 |
| Closing rows | 3:24 |
| Intermediate rows (base 1:46, join 0:39, report states 0:54, market dynamics 0:23, injuries 0:25) | 4:35 |
| 51 slots (~3.7 s each) | 3:26 |
| Storing | 0:13 |

A warm TabPFN run at one horizon is ~8 min, of which ~1.3 min is its own
(spec, the two fits, storing); the rest repeats what the covering predictor
run built two minutes earlier. Validating the history memo costs seconds
(digests ~0.7 s, the injury-report fingerprint ~1.1 s, measured locally); a
memo miss costs minutes.

Since the same date every self-hosted job logs unbuffered
(`PYTHONUNBUFFERED`), so each log line carries the time it was printed, and
the predictor and TabPFN print a `[time] <phase>: <duration>; peak RSS ...`
line as each phase ends (`nba_ou.utils.phase_timer`), one per slot (with its
model read; until 2026-10-27 also its Supabase write, ~2.5 s, since removed),
and a `Timings (wall clock)` summary before
their own summary. The job venv is reused across runs
(`.github/actions/job-venv`): setup drops from ~50–60 s to a few seconds
except after a requirements change.

## Machine and slate

- Runner: GTX 1060 6 GB, 4 cores, 15 GB RAM. This is the machine the
  `github-runner` service runs on. The earlier timings in
  `docs/pipeline/speedup.md` came from a 10-core i7-1365U laptop. This machine
  is ~3–4x slower on the CPU-bound stages.
- Slate: 2026-10-22, 2 games (CLE@PHI 23:00 UTC, DEN@OKC 01:30 UTC). A 10–15
  game night has more target rows. How the build stages scale with the number of
  games is **not measured here**, so check the first busy night.

## Method

The scripts are the same ones the workflows run: `predict_nba_games.py` and
`retrain_prediction_models.py`. They ran in the project's Poetry environment
through a guard that blocks every shared write:

- Prediction: `persist=False`, so no prediction store rows and no S3 archive.
  Also blocked:
  - the database update (step 1);
  - the injury-report sync;
  - the roster and referee snapshot uploads;
  - every season-store write.
  
  It ran with `allow_stale_lines=True`, see finding 1.
- Retrain: `--no-stage`, so no build is written and no channel moves. The
  season-partition update computes everything, but its write and touch are
  blocked.
- History memo: either an empty local directory ("cold"), or an overlay that
  reads the production memo in S3 and writes new entries to a local directory
  only.
- The S3 input cache was tested as it is today (expired, see finding 2). It was
  also tested as a working daily refresh would leave it (`MAX_VALIDATION_AGE`
  raised in memory; read-only).
- Job setup used a fresh venv from the runner's own Python 3.12.15, with
  `pip install -r requirements/<job>.txt` and `pip install --no-deps -e .`,
  first with an empty pip cache and then a filled one.

Not measured, because they write:
- the database update steps;
- the injury-report sync inside the prediction;
- prediction persistence.

The database steps are taken from the runner's logs instead
(`_diag/Worker_*.log`).

## Prediction (`nba_predictor_daily`)

Seconds per stage of `predict_nba_games.py`. Step 2 covers the schedule,
referees, rosters and reports. Step 3 covers the closing rows and the
intermediate rows for every horizon, latest input. Step 4 is the 51 slots
plus the baselines.

| Run | History memo | Local input cache | S3 input cache | Step 2 | Closing rows | Intermediate rows | 51 slots | **Total** |
|---|---|---|---|---:|---:|---:|---:|---:|
| P1 cold | empty | empty | expired | 314 | 717 | 2926¹ | 196 | **4156** (69 min) |
| P2 | S3, missing this code's entries | empty | expired | 119 | 726 | 2819¹ | 92 | **3763** (63 min) |
| P3 warm memo, as CI | complete | empty | expired | 113 | 457 | 489 | 88 | **1154** (19 min) |
| P4 all local caches kept | complete | kept | expired | 94 | 285 | 485 | 89 | **959** (16 min) |
| P5 warm memo, cache refreshed | complete | empty | valid | 98 | 185 | 535 | 189 | **1012** (17 min) |

¹ The forked injury horizons hit `FORKED_TIMEOUT_SECONDS` (600 s) and were
rerun one by one, see finding 3.

On a real run, add:
- job setup: ~50 s (venv and pip from the runner's warm pip cache, Playwright
  0.5 s, checkout and OIDC a few seconds);
- step 1, the database update: `update_all_databases` took 90–104 s in the
  09:00 runs of 2026-10-21/22;
- the injury-report sync: unmeasured.

**A warm run is therefore ~19–23 min end to end**, P3 being what CI sees today.
A run is cold whenever the code a memo is keyed on has changed, as after
merging builder changes to `main`, or when the memo lacks the day's entries.

The prediction of the 51 slots took 88–196 s across identical runs,
2–4 s per slot. Most of that is loading each booster from S3.

## Retrain (`retrain_prediction_models`)

Seconds per stage of `retrain_prediction_models.py --no-stage`. Partition
update = bringing the season partitions up to today, which built 14 closing
games. The intermediate update took 3–5 s, because there is no 2026-27 line
history yet.

| Run | Partition cache | S3 input cache | Closing partition update | Load frames | 51 fits (GPU) | **Total** |
|---|---|---|---:|---:|---:|---:|
| R1 cold | empty | expired | 602 | ~490 (download) | ~310 | **1428** (24 min) |
| R2 warm | kept | expired | 309 | <5 | ~309 | **645** (11 min) |
| R3 warm | kept | valid | 178 | <5 | ~308 | **515** (9 min) |

The smoke test of all 51 slots (`smoke_test_s3_model_load.py`, read-only on
the production channel) took **65 s**. Job setup takes ~45 s with a warm pip
cache. The workflow keeps the partition cache in `~/.cache/nba_ou`, so R1
happens once per runner.

### The database update (first job of the retrain)

From the runner's logs of the 09:00 `update_database_daily` runs:

| Run (UTC) | Install | `update_all_databases` | Line history | Repair closes | Injury reports | Input-cache refresh |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-20 13:14 | 11 s | 21 s | 21 s | 4 s | **55 min, failed** | skipped |
| 2026-10-21 07:01 | 12 s | 90 s | 18 s | **failed** | skipped | skipped |
| 2026-10-22 07:01 | 13 s | 104 s | 41 s | **failed** | skipped | skipped |

The measured steps add up to ~3 min. The last two steps have not completed
in CI yet (ops watchlist item 5), so the full update time is unknown.
**Retrain workflow estimate: ~3 min + injury reports + refresh, then
~1 min setup + 9–11 min refit + 1 min smoke test + promotion.** That is
~15–18 min plus the two unmeasured steps.

### Pip cache

A fresh venv with an **empty** pip cache took **450 s** for
`predict-cuda.txt`. With a filled cache it took 47 s, and 43 s for
`retrain-cuda.txt`. The runner's pip cache persists (`~github-runner/.cache/pip`),
so the slow install happens only when the requirements change.

## Findings

1. **SBR's line-history feed is frozen. Every prediction run would stop.**
   `check_live_feed` scrapes SBR live. The latest tick for tonight's games was
   2026-10-20 16:46 ET, two days earlier. A real run exits 1 after the database
   update and predicts nothing, unless it is run with `--allow-stale-lines`.
   This is the freeze seen on opening night (it still had not recovered on
   2026-10-22). It also explains why the intermediate partitions built no
   2026-27 rows. It blocks the whole snapshot schedule, so resolve it before
   applying the cron jobs.

2. **The S3 input cache has expired, because the 09:00 database update fails
   before refreshing it.** A season is read from S3 only if
   `refresh_input_cache.py` validated it within 36 h
   (`season_tables.MAX_VALIDATION_AGE`). Since 2026-10-16 every table has been
   read from the database, every season. A valid cache saves **~4.5 min per
   prediction** (closing rows 457 → 185 s) and **~2 min per retrain**
   (partition update 309 → 178 s). The cause is the 09:00 job, which is live and
   failing daily at "Repair SBR closing lines" on `main`'s code. The refresh
   step runs last and is skipped on any earlier failure. Fix the failure, and
   consider `if: ${{ !cancelled() }}` on the refresh step (ops watchlist item 5
   already suggests it).

3. **The forked injury horizons time out on this runner whenever the memo is
   cold. The fallback then pays the work twice.** `intermediate_injuries`
   forks 3 workers here (cores − 1) with a 600 s timeout on the whole pool. A
   cold memo needs more than 600 s on 4 cores, so the pool is abandoned at
   600 s and all 16 horizons rerun serially. In P1 the abandoned pool wasted
   10 min, then the serial rerun took ~20 min (2770 → 3958 s). Raising
   `FORKED_TIMEOUT_SECONDS`, or making it per horizon, saves at least the
   10 wasted minutes. It likely saves more, since the serial rerun replaces
   3-way parallel work. A warm memo stays well under the limit (P3–P5).

4. **The local input cache lives inside the checkout, so CI loses it every
   run.** The default `data/cache/inputs` is under `data/` (gitignored), and
   `actions/checkout` removes ignored files. Setting `NBA_OU_INPUT_CACHE_DIR`
   to `~/.cache/nba_ou/inputs` in the predictor workflows (as the retrain
   already does for partitions) keeps it. That saves ~3 min per run while the S3
   cache is expired (P3 → P4), and little once it is valid.

5. **Superseded: the memo key is now frozen.** Since the frozen code key
   (`history_memo.frozen_code_key`: base schema, root modules and main book),
   a deploy keeps every memo entry; only a change to the inputs refuses one.
   What follows describes the source-keyed memo of 2026-10-22.
   **Every code change made the next prediction cold (~70 min, less with
   finding 3 fixed).** The memo keys include the source of the modules that
   build each entry (`history_memo.code_key`). A merge to `main` that touches
   the builders therefore invalidates the memo, and the first booked run after
   it takes over an hour. Merge such changes in the morning and dispatch one
   warm-up run by hand (`persist` off) before the first booked snapshot. The
   09:00 retrain warms the closing memos only, since its partition update builds
   the closing rows.

## What this means for the schedule

- On 2026-10-21 (11 games), the scheduler books 21 runs. At ~20 min each,
  that is ~7 h of runner time between 11:00 and 04:00 Madrid, plus TabPFN and
  past snapshots on the same runner. That fits on one runner.
- Grouping snapshots at ≤ 30 min is the right granularity. A finer one would
  queue runs, since each takes ~20 min, and the queue keeps only one pending
  run.
- The weakest point is a cold run. Findings 3 and 5 decide whether a deploy
  day loses one booked run or three.
- Re-measure after fixing findings 1 and 2, on a busy night (10+ games), with
  the real database update and injury sync included.

Raw logs of this measurement were kept in the session's scratch directory
only. They are not in the repository.
