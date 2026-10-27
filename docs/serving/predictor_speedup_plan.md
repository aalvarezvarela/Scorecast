# Faster prediction runs on the self-hosted runner

Status: **Partial** (updated 2026-10-27) — P0 and P3 are done (PR #84); P2a
is done on branch `perf/prediction-loop`; P2b is next. The progress log at the end records
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
  starts with the dimension tables, so a new row in them (probably a new
  injury reason) invalidates every season.
- Shrinkage priors: +0:29.
- Closing rows of the first retrain of the day: +24 min (27:40 instead of
  3:40), on both days observed. The past seasons' input tables did not
  change; the cause is unknown.
- New on 2026-10-27, from the 18:30 UTC run on: `intermediate_snapshot_history`
  reports "inputs changed" for 2019–2025 on every run and is rebuilt
  ("1 season(s) stored, built [2019 … 2025]"), so the intermediate rows take
  10–11 min instead of ~5–7. The same run (18:30) spent 27:08 on the closing
  rows. Cause not identified; the line-history partitions were rebuilt for the
  alternate-line filter that day (PR #85, PR #86 merged 21:42 UTC).

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
| P2 · Prediction loop | ≈3 min per run (2 min measured) | low–medium | low | **in progress** (P2a done) |
| P3 · Persistent venv | ≈0.7 min per run | low | low | **done** (PR #84) |
| P4 · False cache misses | 1.5–5 min when they happen; 24 min in the morning | low | one cold run on deploy | open |
| P5 · Closing and intermediate in parallel | ≈3 min per run | medium | memory | open |
| P6 · Runner hardware | depends on the result | low to measure | — | open |
| P7 · Streamlit on Aiven | — (unblocks the Supabase clean-up) | medium | dashboard history | open |

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

### P2 · Prediction loop — in progress

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

**P2b · Clean each frame once.** Clean the closing and intermediate frames
once and let each slot select its columns from the clean frame. Profile the
~1.1 s per slot locally first. Estimated 30–45 s per run. Check: the same
predictions, value for value, in a `--no-persist` run before and after.

**P2c · Read models and specs from S3 in parallel.** Threads at the start of
the loop. At most ~10 s per run; optional.

### P3 · Persistent venv — done

Self-hosted jobs install through `.github/actions/job-venv`: one venv per
runner, requirements file and Python, under `~/.cache/nba_ou/venvs`, reused
while its packages match and rebuilt otherwise. Merged in PR #84 (f97e28b).
Setup went from 50–60 s to ~10 s.

### P4 · False cache misses

- **Reports.** Each season's fingerprint includes only the dimension rows that
  season uses, not the whole table.
- **Closing.** Digest the closing memo with `values_only`, as was done for the
  injuries in #79.
- **Retrain.** Diagnose the miss of the first retrain of the day by comparing
  each input's and season's digest between the first and the second retrain.
- **Snapshot history (new, D6).** Find why `intermediate_snapshot_history`
  rebuilds 2019–2025 on every run since 2026-10-27 18:30 UTC and stores only
  one season; check it again after PR #86.
- **Cost.** Changing a digest invalidates those entries once: deploy it and
  dispatch a cold run outside game hours.

### P5 · Closing and intermediate in parallel

**What.** The two builds are independent until the prediction phase and most
of their work is single-threaded. Build the closing rows in a separate
process while the intermediate rows build.

**Saving.** Up to 3:24 per run; ~3 min allowing for CPU contention.

**Risk.** In the retrain the closing build peaks at 3.1 GB and the
intermediate at 5.1 GB; together ~8 GB on a 15 GB runner. A cold build has
already run out of memory on pc1641 with 8 workers. Decide after P6.

### P6 · Runner hardware

**What.** Run `lscpu`, `free -h` and `vmstat 5` during a run and measure the
S3 download speed. That decides whether P5 fits, whether a second runner on the
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

### Dropped

- Speeding up cache validation: 5–15 s per run.
- Loading less history: it changes feature values, and no optimization may.
- More workers in the injury step: under 1 min on normal runs.

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
2. P2a: stop the Supabase write. Then P2b, checked with identical
   predictions; P2c if it is worth it.
3. P1: TabPFN inside the predictor, checked against the standalone run's rows
   at the same cutoff.
4. P4: memo fixes, with a cold run planned outside game hours.
5. P6, then P5 only if the runner's memory allows it.
6. P7: Streamlit on Aiven, then the Supabase clean-up.

## Progress log

- 2026-10-27 — P0 and P3 merged (PR #84). Per-slot timings show the Supabase
  write at ~2.5 s of ~3.8 s per slot; P2 reordered (P2a first). Snapshot
  history memo misses since 18:30 UTC added to P4.
- 2026-10-27 — P2a done on `perf/prediction-loop`: no Supabase write left in
  the predictor, the past-snapshot run, the baselines or TabPFN; full test
  suite passes (3359). Streamlit still reads the frozen Supabase table (P7).

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
