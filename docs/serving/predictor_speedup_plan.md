# Faster prediction runs on the self-hosted runner

Status: **Partial** (updated 2026-10-28) — P0 and P3 are done (PR #84); P2
(a, b and c), P4 and P5 are merged (PR #87), and P8 is on branch
`perf/prediction-logging`, all to be confirmed on a warm runner run. P1, P6
and P7 are open. The progress log at the end records
each step.

Where the time of the prediction workflows goes and what to change to cut it,
ordered by saving. The measurements behind it are in
[runner_timings.md](runner_timings.md); this document is the working plan and
is updated as each item lands.

**Ground rule: no item may change a feature value or a prediction.** Each one
is checked by comparing rows and predictions with the current code.

Data: logs of the runner `panchojasen-HP` on 2026-10-26/27 (commits 1bcfe35,
0f35ef6, then 2b27fd1 for the per-slot timings) and the schedule of
2026-10-27 (4 games). Every figure says whether it is measured or estimated.

## Summary

- **~13.9 min per predictor run**, with 1 game or with 9 (measured,
  2026-10-27 00:01 UTC).
- **~6.2 h of runner time** on a 4-game day: 20 predictor runs and 12 TabPFN
  runs (estimated).
- **103 min**: the longest queue wait of a TabPFN run on 2026-10-26
  (measured).

The bottleneck is the single runner: every workflow waits in the same queue.
Each run pays ~14 fixed minutes, almost all of it rebuilding history and
predicting 51 slots. TabPFN redoes 83 % of what the predictor computed two
minutes earlier. Validating the caches costs seconds; what costs minutes are
cache misses that should not happen.

With P1–P3 a day like 2026-10-27 drops from 6.2 h to ~3.9 h of runner time;
with P5 as well, to ~2.9 h. T-30 predictions would be ready around T-22
instead of T-16.

## Diagnosis

### Where a run's time goes

Wall-clock minutes from the moment the runner picks the run up (run of
2026-10-27 00:01 UTC, before PR #84):

| Phase | Time |
|---|---:|
| Job setup (checkout, venv, pip) | 0:51 |
| Database update, SBR feed, schedule and reports | 1:22 |
| Closing rows | 3:24 |
| Intermediate rows (base 1:46, join 0:39, report states 0:54, market dynamics 0:23, injuries 0:25) | 4:35 |
| 51 slots (~3.7 s each) | 3:26 |
| Storing | 0:13 |

### D1 The runner is saturated

Predictor, TabPFN, retrain and past snapshots share one runner. On
2026-10-27 there are 20 predictor runs (hourly, every 30 min close to tip-off)
and 12 TabPFN runs: ~6.2 h of runner time. On 2026-10-26 (9 games) the TabPFN
runs created at 21:02 started at 21:50 and 22:45, and the one of 22:02 at
23:09. In the half hour before each tip-off the predictor takes almost half
the runner, and a T-30 run that waits in the queue can land after tip-off.

### D2 The cost is fixed, not per game

The 01:30 run (1 game) took 13:18 and the 22:56 run of 2026-10-26 (9 games)
13:08. The time goes into the history each run rebuilds and into the 51
slots, not into tonight's games.

### D3 TabPFN repeats almost all of the predictor's work

Of the ~8.0 min of a TabPFN run (with the horizon fix), only ~1.3 min is its
own: loading the spec (0:13), the two fits (0:56) and storing (0:11). The
rest (setup, data, base, join, reports, market dynamics) is what the predictor
run covering that snapshot computed two minutes earlier.

### D4 The prediction loop costs ~3.8 s per slot

Measured per slot since PR #84 (run 40578707960, 2026-10-27 20:30 UTC, 4
games; 51 slots in 3:16):

| Per slot | Time | Total |
|---|---:|---:|
| Model and spec read from S3 | 0.2–0.4 s | ~15 s |
| Cleaning, prediction, prediction-store rows | ~1.1 s | ~55 s |
| **Write to the old Supabase predictions table** | **2.0–3.2 s (mean 2.5)** | **~2:05** |

The Supabase write (`upload_predictions_to_postgre`) opens two TLS
connections per slot and repeats three schema queries (table, columns, upsert
constraint) before its `executemany`; the baselines go through the same path.
It is the parallel write that [prediction_storage.md](prediction_storage.md)
(rollout 3) planned to remove after a week; the Aiven store has written every
run since 2026-10-26 14:58 UTC. The S3 reads, which the original plan targeted,
are ~15 s in total.

### D5 Every run reinstalled everything — fixed

Creating the venv and running `pip install` cost 50–60 s per run, ~30 min a
day, with the same versions every time. Since PR #84 the job venv is reused
(`.github/actions/job-venv`): setup takes ~10 s (run of 2026-10-27 20:30).

### D6 Validating the caches is cheap; missing them is expensive

Measured locally: the game, player and injury digests take 0.7 s and the
report fingerprint 1.1 s (twice per intermediate build), 5–15 s per run in
all. The misses that should not happen are what cost:

- Report states rebuilt for every season: +1:25. The chained fingerprint
  starts with the dimension tables, so a new row in them (a new injury
  reason) invalidates every season. Seen in 6 runs over 2026-10-26/27.
  Fixed (P4).
- Shrinkage priors: +0:29. Their key holds the same report fingerprint, so
  they missed with the report states. Fixed with them (P4).
- Closing rows of the first retrain of the day: +24 min (27:40 instead of
  3:40), on both days observed. Cause found (P4): the merged odds frame was
  sorted with an unstable sort, so each night's new games reshuffled the rows
  of earlier seasons and their memo entries missed. Fixed.
- 2026-10-27 15:30–20:30 UTC: `closing_line_repair`,
  `intermediate_snapshot_history` and other memos missed for 2018–2025 on
  several runs. Not a key fault: the line history of every past season was
  rewritten that afternoon (closing ticks re-uploaded to the input cache at
  14:39–14:44 for the alternate-line filter, PR #85/#86), and the memos
  rebuilt as they should. From the 21:30 run on they hit again.

### D7 The runner is 2.5–3x slower than the development laptop

The closing rows take 3:24 on the runner against 71 s warm on pc1641; the
intermediate rows 4:35 against ~2 min. The runner is a 4-core machine with
15 GB RAM and a GTX 1060 6 GB ([runner_timings.md](runner_timings.md)); swap,
disk and S3 bandwidth during a run are not measured.

## Plan

Ordered by saving.

| Item | Estimated saving | Effort | Risk | State |
|---|---|---|---|---|
| P0 · Measure before optimizing | — | low | low | **done** (PR #84) |
| P1 · TabPFN inside the predictor | ≈6.2 min per TabPFN run | medium | low, if the fits go last | open |
| P2 · Prediction loop | ≈3 min per run (from measured parts) | low–medium | low | **done** (`perf/prediction-loop`), to confirm on the runner |
| P3 · Persistent venv | ≈0.7 min per run | low | low | **done** (PR #84) |
| P4 · False cache misses | ~24 min each morning retrain; ~1:54 per new injury reason | low | one cold run on deploy | **done** (`perf/prediction-loop`), to confirm on the runner |
| P5 · Closing and intermediate in parallel | ≈3 min per run | medium | memory, only warm builds overlap | **done** (`perf/prediction-loop`), to confirm on the runner |
| P6 · Runner hardware | depends on the result | low to measure | — | open |
| P7 · Streamlit on Aiven | — (unblocks the Supabase clean-up) | medium | dashboard history | open |
| P8 · Step 2 (schedule, referees, injury reports, odds) | ≈15–25 s per run | low | low | **done** (`perf/prediction-loop`), to confirm on the runner |
| P9 · TabPFN inside the predictor run | ≈2.5–3 min and a queue slot per TabPFN run (phase 1); most of its ~6:42 rebuild with phase 2 | medium | medium | phase 1 **done** (PR #89), not booked; phase 2 levels 1–2 **done** (PR #92), validated on the runner; not booked |

### P0 · Measure before optimizing — done

`PYTHONUNBUFFERED=1` on every self-hosted workflow, and a
`[time] <phase>: <duration>; peak RSS …` line as each phase of the predictor
and TabPFN ends (`nba_ou.utils.phase_timer`, shared with the retrain), one per
slot with its S3 read and Supabase write, and a `Timings (wall clock)` summary.
Merged in PR #84 (f97e28b).

### P1 · TabPFN inside the predictor run

**What.** The scheduler stops booking TabPFN separately. The predictor run
covering each TabPFN snapshot (T-720/480/360/240, the TabPFN horizon and its
games) also builds that horizon's row for each game, in the same builder call,
and fits TabPFN once the XGBoost slots are done.

**Saving.** A TabPFN run goes from ~8.0 min on its own to ~1.8 min added (one
more horizon in the injury step, the spec, the fits, storing). On 2026-10-27
that is 12 × 6.2 min ≈ 74 min; on a 9-game night ~2.5 h. The TabPFN queue
disappears.

**How.** The builder accepts one snapshot per game in `target_snapshots`
today; it must accept the latest one and the horizon's. The predictor needs
`tabpfn-client` and the token. The fits run after the XGBoost predictions are
stored and are stored as a separate run (`workflow` 3), so an API failure
breaks nothing else. The standalone workflow stays for manual runs.

**Correctness.** The covering predictor run starts at or after the snapshot,
so the row of the game that triggers it is the exact one. TabPFN already
starts 2 min after that predictor run today, so nothing is delayed.

**Check.** Compare its rows with those of the standalone TabPFN run at the
same cutoff.

### P2 · Prediction loop — done (to confirm on the runner)

Reordered on 2026-10-27 after the per-slot timings (D4): the Supabase write
is two thirds of the loop.

**P2a · Stop writing to Supabase — done** (`perf/prediction-loop`). Only the
Aiven store (and its S3 run archive) keeps the predictions. The `persist`
argument of `load_and_predict_model_for_nba_games` and
`load_registry_model_and_predict`, and `upload_to_postgres` of the baselines,
are gone: they only return their output.

- `scripts/predict_nba_games.py`: drop `legacy_supabase_upload` for the slots
  and the baselines.
- `scripts/predict_past_snapshots.py`: the same.
- `prediction/prediction.py`: drop the `persist` → `upload_predictions_to_postgre`
  path inside `load_and_predict_model_for_nba_games` (no script uses it).
- `prediction/baseline_predictions.py`: drop `upload_to_postgres`.
- The workflows keep `DB_ENV: supabase` and the Supabase secrets: the
  predictor still *reads* games, players, injuries and odds from Supabase.
- Kept until P7: `upload_predictions_to_postgre`, the readers in
  `postgre_db/predictions/update/` (Streamlit uses them), and the "Update
  finished matches" step of `update_finished_matches_daily.yml`, so the last
  Supabase rows get their results.
- Tests that fake the Supabase write and the docs
  (`prediction_storage.md` rollout 3, `runner_timings.md`) are updated.
- Saving: ~2 min per run, from the measured write times; to confirm on the
  first runs after merge. Changes no prediction.

**P2b · Clean each frame once, predict from an array — done**
(`perf/prediction-loop`). The profile of the ~1.1 s per slot left after P2a
(locally 0.43 s a slot, 21.9 s for the 51 slots, on the archived frames of the
2026-10-27 21:54 UTC run) put it in two places, neither the model itself:

- **Cleaning** (~45 %): every slot cleaned the same frame again.
  `clean_prediction_frame` is now that cleaning on its own; the predictor, the
  past-snapshot run and TabPFN clean each frame once
  (`clean_prediction_frames`), and each slot chooses its rows on the frame as
  before, then reads them from the clean one (`clean_frames=`). Cleaning keeps
  or drops each row on its own values and keeps every non-text column, so the
  rows are the ones per-slot cleaning kept; the `latest` limits are applied to
  the clean rows. A frame that fails to clean, or has a repeated index label,
  falls back to per-slot cleaning.
- **XGBoost reading a DataFrame** (~55 %): its pandas path inspects each of
  the 1,100–2,300 columns' dtype on every call. An all-numeric `X` now goes in
  as one float64 array (`prediction._predict`), after checking the booster's
  feature names and order, which the DataFrame path checked.

Measured locally with pinned builds (scratch harness replaying the archived
input frames through `load_registry_model_and_predict`):

| Frames | Before | After | Predictions |
|---|---:|---:|---|
| `latest` run, 51 slots (204 rows) | 21.9 s | 3.4 s (0.3 s of it cleaning) | identical, all 44 columns |
| `own` rows T-240/360/480/720, 51 slots (60 rows) | 6.5 s | 0.9 s | identical, all 43 columns |

On the runner (~2.5x slower) that is ~55 s → ~9 s for the compute part; the
S3 reads (~15 s) are what is left of the loop. TabPFN saves one cleaning per
extra target (~1 s on the runner).

**P2c · Read the builds from S3 side by side — done** (`perf/prediction-loop`).
The spec, fit and model of a build are immutable and already come from the
local model cache (`~/.cache/nba_ou/models`); what each slot still waited for
was its channel pointer, one S3 read of 0.2–0.4 s, in turn.
`read_slot_builds` reads every slot's build with 8 threads before the loop
(the predictor's `model reads` phase; the past-snapshot run too), and each
slot is handed its build (`resolved=`). A failed read is kept for its slot,
which raises it in its own turn, so it fails alone as before. Locally, warm
cache, 51 slots: 6.5 s in turn → ~2 s, the same builds (fit, spec, model
bytes). Expected on the runner: ~15 s → ~5 s.

With P2a–c, the loop is expected to go from 3:16 (`51 slots`) to roughly 0:15
plus ~5 s of `model reads` and well under 1 s of `cleaning`.

### P3 · Persistent venv — done

Self-hosted jobs install through `.github/actions/job-venv`: one venv per
runner, requirements file and Python, under `~/.cache/nba_ou/venvs`, reused
while its packages match and rebuilt otherwise. Merged in PR #84 (f97e28b).
Setup went from 50–60 s to ~10 s.

### P4 · False cache misses — done (to confirm on the runner)

Read from the memo reports of every predictor and retrain run of 2026-10-26/27
(31 runs). Three kinds of miss, beyond the current season's expected one:

**Morning retrain (+24 min) — fixed.** On both days the first retrain missed
`closing_line_repair`, `player_history_rows` and `availability_effects` from
one season on (2022–2025 on 2026-10-26, 2020–2025 on 2026-10-27), and later
runs hit again. No past-season input had changed: the 07:00 refresh found
every input-cache season unchanged, and no job ran in between that writes
those memos. The cause was `merge_yahoo_sportsbook_odds` sorting the merged
odds frame by `game_date` with pandas' default quicksort, which is not stable:
a date's games came out in an order that depended on every other row, so the
games each night added reshuffled earlier seasons' rows. The memo digests rows
in order, so each season from the first reshuffled one missed — the repair
memo directly, the chained closing memos (`odds` is one of their inputs) from
there on. Measured on the real frame: leaving out the games from 2026-10-24,
-25 or -26 on changed the row digest of 2020–2025 with quicksort and of no
season with the stable sort. The sort is now `kind="stable"`, as
`merge_scheduled_odds` already was for the same reason: a date's games keep
the merge's game-id order. `game_id` is unique in the merged frame (9,003
rows, 2011–2026) and every consumer merges on it, so only the row order
changes, never a value. `test_merge_odds_row_order.py` fails on the old sort.
Checked end to end: the closing rows of 2026-10-26's nine games, built in
target mode (as the retrain builds them) on an empty local memo, are equal
bit for bit (`compare_builds.py --exact`, 1,670 columns) with the old sort
patched back in; built again on that memo, every season hits (8 repair, 17
player-history tables) and the rows are equal again, in 90 s.

**New injury reason (report states +1:25, priors +0:29) — fixed.**
`report_season_fingerprints` started its chain with the whole dimension tables
(`ir_team`, `ir_reason`, `ir_reason_category`, `ir_status`); `ir_reason`
(5,364 free-text rows) gains rows with new reports, so every season's report
states and the effect priors missed. Each season's fingerprint now holds only
the dimension rows its spans, filings and games join. Checked on Aiven by
taking the last 20 reasons out: the old fingerprint changes for 2018–2026, the
new one for 2026 only (the one season using them). The query takes ~1.6 s
instead of ~1.05 s.

**Misses now name their cause.** Every memo stores what its digest is made of
(`HistoryMemo(parts=...)`: each input table for the chained memos, odds rows
and closing ticks for the repair, settings and ticks for the snapshot history,
reports and cutoffs for the report states and priors). A miss reports
`inputs changed (odds)` or `(an earlier season)` instead of a bare `inputs
changed`; the digests themselves are unchanged. Entries written before this
report as before until rewritten.

**Not done: `values_only` for the closing memo.** The plan proposed it (as #79
did for the intermediate injuries). The logs show no disagreement between the
retrain and the predictor — the 12:55 retrain hits the predictor's entries —
so it would cost a cold rebuild for nothing. Revisit if a miss report names a
dtype or column change.

**Deploy.** The stable sort reorders the odds frame once and the report
fingerprints change once, so the first runs after merge rebuild the closing
memos (~24 min on the runner) and the report states and priors (~2 min).
Merge outside game hours and dispatch one predictor run with `persist` off.
Then check the next morning's retrain: the closing memos should hit, and any
miss should name its part.

### P5 · Closing and intermediate in parallel — done (to confirm on the runner)

**What.** The two builds are independent until the prediction phase and most
of their work is single-threaded. Build the closing rows in a separate
process while the intermediate rows build.

**Saving.** Up to 3:24 per run; ~3 min allowing for CPU contention.

**Risk.** In the retrain the closing build peaks at 3.1 GB and the
intermediate at 5.1 GB; together ~8 GB on a 15 GB runner. A cold build has
already run out of memory on pc1641 with 8 workers. So only warm builds may
overlap. Warm predictor runs on the runner peak at 3.0 GB (closing) and
5.0 GB (intermediate).

**Done.** `predict_nba_games.py` spawns the closing build
(`nba_ou.prediction.side_build`) and builds the intermediate rows meanwhile.
Whether a run is warm is not known in advance: every warm run still misses
the current season (report states, injury effects of 2026), and computing
the digests is the build's own first step. So the decision is made during the
build, by the warm gate (`nba_ou.utils.warm_gate`):

- Each build holds a file lock, shared.
- A build that starts computing a past season takes the lock exclusive. It
  waits there until the other build has finished, then goes on alone. Two
  builds turning cold at once take turns; they do not deadlock.
- Computing a past season means one of two things. Either the memo refused a
  stored entry (inputs changed, corrupt, read failed, rebuild), and the miss
  is reported as the entry loads. Or a table stored nowhere gains its first
  entry, which happens after a code-key change. A table that stays empty is
  no work: the injury memo asks for season 2000 on every run and stores
  nothing for it.
- Misses of the current season pass: they are part of every warm run.

The closing build's log is printed once the intermediate rows are done, in a
step of its own: what it holds so far, then live, each line marked `[closing
rows]`, with a `... still running` line after two quiet minutes. (Until
2026-10-28 it came as one block at the end, and a cold closing build left the
job log silent for 28 minutes.) The run then prints `closing rows (side
process)`, with the side process's own memory peak, and `closing and
intermediate rows, side by side` in its `[time]` lines, plus a warning naming
the cold memo when a build computed past seasons. The side process is spawned, not forked, so it shares no S3 or
database connection. If it dies without a result (e.g. killed for memory),
the closing rows are built again in the main process; if it raises, the run
fails as a sequential one would. It runs sequentially without a usable
history memo (`NBA_OU_HISTORY_MEMO=0`, no backend, `REBUILD=1`), with no
closing rows to build, or with `[PredictionModels] PARALLEL_BUILDS = false` /
`--sequential-builds`.

**Validation** (2026-10-28, local, replay of 2026-04-10 at 22:30 UTC, 15
games, local memo, `PYTHONHASHSEED=0`). Every run gave the closing rows (15 ×
1,672), their layer inputs (11,437 × 5) and the intermediate rows (15 ×
3,086) equal bit for bit to the sequential build:

| Run | Intermediate | Closing | Total |
|---|---:|---:|---:|
| Sequential, warm | 2:56 | 4:36 | 7:32 |
| Side by side, warm (no wait) | 4:16 | 5:42 | 5:43 |
| Side by side, closing cold (one season's digest made stale) | 3:24 | 8:35, of which 2:06 waiting | 8:36 |

On the laptop the closing rows are the longer build; on the runner the
intermediate rows are (3:24 closing, 4:35 intermediate), so the expected run
saving is ~2–3 min of contention-adjusted overlap. Memory side by side: ~3.0
plus ~4.7 GB. The first runner runs should confirm the phase times and the
peak.

### P6 · Runner hardware

**What.** Run `lscpu`, `free -h` and `vmstat 5` during a run and measure the
S3 download speed. That checks that P5 fits in memory and on 4 cores, whether a second runner on the
same machine would help, or whether the bottleneck is disk or network.

**Note.** A second runner shortens the queue, not the total work. After P1 it
is much less needed.

### P7 · Streamlit on Aiven

`streamlit/streamlit_ou_app.py` and `streamlit/streamlit_app.py` read the old
Supabase table (`postgre_db/predictions/update/`). After P2a that table stops
growing: its history covers 2026-03-06 to 2026-10-27 (16,012 rows), while the
Aiven store starts on 2026-10-26 14:58 UTC. The dashboard needs to read Aiven
for new predictions and either Supabase or an import for the older ones.
[prediction_storage.md](prediction_storage.md) ("What the Streamlit dashboard
needs") maps each dashboard use to the store. Once done, remove the Supabase
writer, the readers and the "Update finished matches" step.

### P8 · Step 2: scheduled games, referees, injury reports and odds — done (to confirm on the runner)

**Measured.** Step 2 takes 45–58 s on the runner (52 s on 2026-10-28 02:31
UTC). Of those, ~37 s printed nothing: referees, live rosters, the
injury-report sync and the report state. The odds (Yahoo, SBR) took ~12 s
after them. Local timings, read-only, 12 games:

| Part | Local |
|---|---:|
| Schedule, referees, league rosters | 0.5 + 0.8 + 0.3 s |
| Sync: team dimension (`DISTINCT` over `nba_players` since 2019) | 4.6 s cold, 0.2 s warm |
| Sync: game dimension (the season's schedule request, 6.0 s, plus Supabase) | ~7 s |
| Sync: rosters | 0.9–1.9 s |
| Sync: download, validate and parse one new PDF | ~0.3 s plus the 0.6 s pause |
| Report state (history from S3 and Aiven, today from Aiven) | 4.6–6 s |
| Odds | ~16 s |

There is no retry backlog: the 125 publication slots of yesterday and today
were all ingested, so each run fetches only the 1–4 reports published since
the previous one. Parsing is negligible. What costs is the dimensions each
ingest reads and the odds waiting behind everything else.

**Done.**

- The odds are fetched in a thread from the start of step 2, beside the
  referees, rosters, sync and report state. Nothing in between reads them,
  and a failed fetch still fails the step when they are collected.
- The sync reads the three dimensions (teams, games, rosters) in a thread
  from its start, beside the slot registration and the downloads. They are
  used only when the parsed reports' seasons are exactly the candidates'
  seasons; otherwise, or when the prefetch failed, they are read again as
  before. Same queries, same values, read a few seconds earlier, after the
  step 1 database update.
- Each part prints a `[time] step 2 ...` line, and the sync prints its
  counts (pending, ingested, unavailable, failed).

**Not done.**

- Caching the dimensions across runs: the input cache's rule is that a
  cached read is checked against the server, and checking `nba_players`
  costs the scan itself.
- Reading the report history before the sync: it reads the current season
  from Aiven, which the sync may have just updated (a late report for one of
  yesterday's games).

**Validation.** Local, read-only (sync and roster archive stubbed), 12
games: 30.1 s → 17.7 s, with the same odds rows. The dimension prefetch is
covered by unit tests; its saving shows in the runner's `[time] step 2
injury report sync` line.

### Dropped

- Speeding up cache validation: 5–15 s per run.
- Loading less history: it changes feature values, and no optimization may.
- More workers in the injury step: under 1 min on normal runs.

### P9 · TabPFN inside the predictor run — phase 1 done, not booked yet

A TabPFN run (`scripts/predict_tabpfn.py`, about 11:30 of runner time at
T-720 on 2026-10-28) repeats most of the predictor run that fired two minutes
before it: job start and venv, SBR check (0:22), step 2 (1:35), then its own
rows (6:42) and fits (line_error 1:42, total_points 0:18). It also waits in
the runner's queue as a run of its own.

**Phase 1 (done on `feat/tabpfn-thinking`).** The TabPFN core (specs and
training rows, tonight's rows, fits, store) is `nba_ou.prediction.tabpfn_run`,
used by both scripts. `predict_nba_games.py --tabpfn JOBS` runs each job once
the predictor's own predictions are stored, after freeing their frames, so a
TabPFN failure or slow fit never costs them; a failed job fails the run at the
end, as a slot does. A job reads the run's schedule, referees, reports and
line source and rebuilds its rows at its snapshot (`as_of`), exactly as a
separate run does: a scheduled run's inputs all stop at its snapshot, so data
fetched later changes nothing, except a report stamped before the snapshot and
published after the step-2 sync. `--tabpfn-refresh` (workflow
`tabpfn_refresh`, off by default) fetches step 2 again and builds the rows at
that moment instead, for predicting on the latest data. The scheduler books it
with `[Schedule] TABPFN_RUNS = joined` (default `separate`, today's runs). The
predictor workflow now installs `requirements/predict-tabpfn-cuda.txt`, has
`TABPFN_ACCESS_TOKEN`, and takes its OIDC session right before the run.

Saving: job start, SBR check and step 2 per TabPFN run, plus its queue slot;
the fit comes after the predictions (safer for memory; it could overlap the
builds later). A predictor run can carry up to three jobs (2026-10-28 dry
run: 23 jobs in 10 predictor runs), each ~9 min in phase 1, so such a run
takes ~40 min warm: within the hourly gap, not on a cold day.

**Phase 2.** Keep the history-only parts of the predictor's intermediate
build in the process, and recompute only what depends on the job's `as_of`.
Profiled locally on 2026-04-10 (predictor at 22:30Z, then TabPFN T-720 at
12:00Z, warm memo, under cProfile): the TabPFN build took 187 s after the
predictor's in the same process, 232 s alone. Of the 187 s, ~60 s depend on
the `as_of` (injuries per snapshot, market dynamics, roster continuity); the
rest repeats loads of stored history: the closing odds with their repair
(25 s, read twice per build), the database connections (~20 s), the team
rolling statistics (18 s), the injury-report history tables (~15 s, some read
two or three times), the memo fingerprints (~10 s) and games and players
(7 s).

- Level 1 (**done**, `perf/tabpfn-history-cache`):
  `nba_ou.utils.history_cache`. Inside `history_cache()` each `@cached` load
  of stored history reads once per set of arguments and every caller gets a
  copy. Marked: the closing odds, games and players, referees, injuries,
  all-star votes, the injury-report history tables, the box-score history and
  the two memo fingerprints. The odds load leaves out the seasons before the
  odds tables begin (`FIRST_SEASON_WITH_STORED_ODDS`, 2018), so the
  referees' span from 2011 shares the base features' read. The predictor run
  holds the scope for the whole run and clears it after step 2 and after a
  TabPFN refresh. `predict_tabpfn.py` holds it for its build. Measured: the
  predictor build 183 → 162 s and the TabPFN build after it 187 → 138 s;
  both frames identical bit for bit to the uncached builds.
- Level 2 (**done**, same branch): the steps computed from those frames
  alone, a frame argument keyed on its content (a SHA-256 of
  `hash_pandas_object` per column, ~0.2 s for the 266k player rows):
  `process_team_statistics_for_training` (the team rolling statistics), and,
  shared rather than copied as their callers only read them,
  `prepare_player_history`, `prepare_all_star_inputs` and
  `prepare_box_and_form`; plus `load_games_from_db` (the referees' games). A
  loader's `None` (a failed read) is not kept. Measured: the TabPFN build
  after the predictor's 138 → 93 s (base features 52 → 24 s, injuries 64 →
  47 s); the predictor build 162–173 s across runs (database latency); both
  frames identical bit for bit; peak 5.3 GB.
- Level 3 (open): reused database connections, ~15–20 s more.

**Validation (to do before booking `joined`).** A manual predictor dispatch
with one `tabpfn` job and `persist` off, compared with a separate TabPFN run
at the same `as_of`: same rows bit for bit, and the phase times.

## Expected result

| Scenario | Predictor | TabPFN | Runner per day (2026-10-27) | T-30 ready |
|---|---|---|---|---|
| Before PR #84, with the horizon fix | 13.9 min | 8.0 min separately | 6.2 h | T-16 |
| With P1–P3 | 10.7 min | +1.8 min inside | 3.9 h | T-19 |
| With P1–P3 and P5 | ≈7.7 min | +1.8 min inside | ≈2.9 h | T-22 |

Estimates from the measured phases. The T-30 column assumes no queue; there
is one on busy nights today. P4 is left out because it depends on when the
caches miss.

## Order of work

1. ~~P0 and P3: phase timings and persistent venv.~~ Done (PR #84).
2. ~~P2a: stop the Supabase write. Then P2b, checked with identical
   predictions; P2c if it is worth it.~~ Done on `perf/prediction-loop`;
   confirm the phase times on the first runs after merge.
3. P1: TabPFN inside the predictor, checked against the standalone run's rows
   at the same cutoff.
4. ~~P4: memo fixes, with a cold run planned outside game hours.~~ Done on
   `perf/prediction-loop`; deploy with a warm-up run, then check the next
   morning retrain's memo report.
5. ~~P5: closing and intermediate side by side.~~ Done on
   `perf/prediction-loop`, warm builds only; confirm phase times and memory on
   the runner. P6 still measures the runner itself.
6. P7: Streamlit on Aiven, then the Supabase clean-up.
7. ~~P8: step 2 in parallel.~~ Done (`perf/prediction-logging`, left out of
   PR #87); confirm the `[time] step 2 ...` lines on the runner.
8. P9: TabPFN inside the predictor run. Phase 1 done on
   `feat/tabpfn-thinking` (not booked: `TABPFN_RUNS = separate`); phase 2
   levels 1–2 merged (PR #92) and validated on the runner; next: book
   `joined` or level 3.

## Progress log

- 2026-10-27 — P0 and P3 merged (PR #84). Per-slot timings show the Supabase
  write at ~2.5 s of ~3.8 s per slot; P2 reordered (P2a first). Snapshot
  history memo misses since 18:30 UTC added to P4.
- 2026-10-27 — P2a done on `perf/prediction-loop`: no Supabase write left in
  the predictor, the past-snapshot run, the baselines or TabPFN; full test
  suite passes (3359). Streamlit still reads the frozen Supabase table (P7).
- 2026-10-27 — P2b done on `perf/prediction-loop`: cleaning once per frame and
  an array into XGBoost take the slot loop from 21.9 s to 3.4 s locally, with
  identical predictions in `latest` and `own` mode; applied to TabPFN too.
  To confirm on the runner after merge (expected `51 slots` ~3:16 → ~0:25).
- 2026-10-27 — P2c done on `perf/prediction-loop`: the 51 builds are read
  with 8 threads before the loop (locally 6.5 s → ~2 s, same builds).
- 2026-10-28 — P4 done on `perf/prediction-loop`. Morning retrain miss
  (+24 min): unstable sort of the merged odds frame, now stable; closing rows
  identical bit for bit. Report-state and prior misses on each new injury
  reason: fingerprint now per-season dimension rows. Every memo miss names the
  inputs that changed. The 2026-10-27 afternoon misses were the line-history
  rebuild, not a key fault. `values_only` for the closing memo not needed.
- 2026-10-28 — P5 done on `perf/prediction-loop`. The closing rows build
  in a spawned side process beside the intermediate rows. A warm gate (a
  shared file lock) makes a build that starts computing a past season wait for
  the other and finish alone, so only warm builds overlap.
  `PARALLEL_BUILDS` / `--sequential-builds` turn it off. On a replay of
  2026-04-10: 7:32 → 5:43 warm; a cold closing build waited 2:06 then ran
  alone; every frame equal bit for bit.
- 2026-10-28 — P8 done on `perf/prediction-loop`. Step 2's odds are fetched
  beside the injury reports, and the sync's team, game and roster reads beside
  its downloads. Every part of step 2 prints its time. Locally 30.1 s → 17.7 s.
- 2026-10-28 — PR #87 merged (P2, P4, P5; P8 was not pushed in time). The
  first run after it (40677865380) was the expected cold one: every memo entry
  refused (`inputs changed`, new code keys). Both builds turned cold; the gate
  let the intermediate rows go first (11:12) and the closing rows waited
  10:05, then computed alone (~28 min); 42:16 in all, peak 4.8 GB. It looked
  stuck because the side log printed only at the end. On
  `perf/prediction-logging`, with P8: the predictor and TabPFN log numbered
  steps (`nba_ou.utils.run_log`), each a collapsible group in GitHub Actions,
  the 51 slots one line each in one step, the side log live, warnings and
  failures as annotations, and the timings in the job summary. CI progress
  bars, the cleaning banner and the memo's per-season miss list are shorter.

- 2026-10-28 — TabPFN thinking modes (`--modes`, `thinking_medium`,
  `thinking_high`, each its own variant) and P9 phase 1 on
  `feat/tabpfn-thinking`: TabPFN jobs inside the predictor run, after its
  stored predictions; refresh optional and off; not booked yet.
- 2026-10-28 — P9 phase 2 level 1 on `perf/tabpfn-history-cache`: an
  in-process cache for the loads of stored history. Predictor build 183 →
  162 s, TabPFN build in the same run 187 → 138 s (local, 2026-04-10 replay),
  frames identical bit for bit. Level 2 (computed steps keyed on their
  input frames): the TabPFN build 138 → 93 s, still identical.
- 2026-10-28 — Phase 2 validated on the runner (PR #92 merged). Predictor
  dispatch 40768878728 (`persist` off, one job T-720 at 14:30Z): tonight's
  TabPFN rows 5:35 → 2:21, intermediate rows 6:44 → 5:18, the run 15:02 →
  10:15, peak 6.9 → 7.3 GB. Its 24 predictions equal, to the last digit, a
  separate TabPFN run at the same cutoff right after (40770941257, no cache,
  rows 4:27). Both differ from the 15:36Z runs in NOP@DEN and DAL@OKC (and
  SAS@UTA from the 14:46Z one): the stored data changed during the day, not
  the cache.

## Sources

- Predictor measured: run 40423624682 (2026-10-27 00:01 UTC); same structure
  in 40420258268, 40426879572 and 40433003704.
- Per-slot timings: run 40578707960 (2026-10-27 20:30 UTC); memo misses in
  40560672200 and 40569778408, none in 40541650525.
- TabPFN measured: 40409268749 (2026-10-26 23:09 UTC, 17 horizons).
- Queues: 40400900759, 40400900995 and 40409268749. Retrain: 40464529559.
  Schedule: 40474550984.
- Cache timings measured locally on real Supabase, Aiven and S3 data.
- Store ranges: read-only queries of `prediction.runs` (Aiven) and the
  Supabase predictions table, 2026-10-27.
- P4 memo reports: every predictor and retrain run of 2026-10-26/27 (31 runs),
  the 2026-10-27 07:00 database update (input-cache refresh: every past season
  unchanged) and the past-snapshot run 40452911022; input-cache upload times
  from S3; report fingerprints and dimension tables read-only on Aiven.
