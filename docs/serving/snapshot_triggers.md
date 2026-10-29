# Snapshot triggers: running the predictor at each snapshot time

Status: **Implemented** (2026-10-22). The daily jobs (08:00 and 09:30 Madrid by
default, see "The day's schedule") are built by `scripts/cronjobs/setup_cronjobs.py`
from `[Schedule]` in `config.ini` (2026-10-26). Two repository variables switch
the workflows off ("Switching it off").

## What it does

The intermediate slots are trained on line snapshots at fixed horizons before
tip-off (`DEFAULT_SNAPSHOT_GRID`: 30, 60, 120 … 720, 840, 960, 1080 min). Once a
day, `schedule_snapshot_triggers.yml` runs
`scripts/cronjobs/schedule_snapshot_triggers.py`, which:

1. reads the date's games and their tip-off times from the NBA scoreboard
   (`fetch_tipoffs_for_date`: ScoreboardV3 `gameTimeUTC` plus the tip-off
   corrections). Games marked TBD or postponed are skipped.
2. computes the runs to book, each predicting every enabled slot (closing and
   intermediate), at the snapshot times of every game: `tip − h` for each
   horizon `h` of `ENABLED_MODELS`.
   - Snapshots at most 30 min apart share one run, fired at the last of them
     (`cluster_times`), so every snapshot of the group has passed. A group
     starts at its earliest snapshot and never spans more than 30 min.
   - The **T-30** snapshot is never grouped: it gets a run at its exact time.
     A group ending on the same minute is the same run.
   - There is no other run before tip-off.

   It leaves out:
   - times already past;
   - times outside the game date in **US/Pacific**. The predictor predicts the
     slate of its run's Pacific date (`predict_nba_games.py`), so a trigger
     before Pacific midnight (03:00 ET) would predict the previous day's games.
     Examples: an early tip's T-1080, a noon game's T-720.
3. adds **TabPFN** runs (`nba_predictor_daily_tabpfn_client.yml`) at each
   game's **T-720, T-480, T-360 and T-240** (12 h, 8 h, 6 h and 4 h before
   tip-off). Each run predicts `line_error total_points` with the production
   specs and rows of that exact horizon (`line=intermediate`, `update_db=false`),
   for **every game of the slate not yet started** when it runs, as the closing
   line does: the games whose snapshot triggered it at that snapshot, the later
   ones from the lines so far (`SNAPSHOT_PASSED` false) and the earlier ones at
   their own, already passed snapshot. Only that horizon's rows are built, so
   the injury stage computes one horizon, not the whole grid.
   Games sharing a tip-off share a horizon's run; different tip-offs are never
   moved or grouped. Each fires 2 min (`TABPFN_DELAY_MINUTES`) after the
   predictor run that covers its snapshot, which can be up to 30 min after the
   snapshot when the predictor groups it, so that run enters the runner's
   queue first. Coinciding TabPFN dispatches are then spread at least one
   minute apart, in snapshot order (22:02, 22:03, 22:04, …), pushing on any
   later run they would land on. Only dispatch times move; each job keeps its
   own snapshot cutoff.
   The payload pins `game_date` and `as_of` (the snapshot itself), so a queue
   delay cannot switch it to another slate or to the latest input; it carries
   no `game_ids` filter, only `snapshot_game_ids`: the games whose snapshot it
   is, as metadata. Every input stops at this cutoff, even when the run starts
   later: line history and archived roster observations, and for a snapshot
   still ahead the injury reports, the injury news timeline and the referee
   release, all read as published before the cutoff (the report age is still
   counted to the snapshot). Each row is then the one a live run at the cutoff
   builds, though the store already holds what was published since. The cutoff
   comes from the NBA scoreboard's tip-off, the rows from the line-history
   page's. When a `snapshot_game_ids` row's snapshot is within 15 min of it
   (`TIPOFF_TOLERANCE_MINUTES`) but not on it, the run waits for that snapshot
   to pass and rebuilds at it, so no row of the triggering games is cut before
   the snapshot it is labelled with. Another game's snapshot a few minutes away
   has its own run and triggers nothing; a manual run without
   `snapshot_game_ids` takes the snapshot nearest the cutoff as its own. A
   moved game is just another row from the lines so far.
   `[Schedule] TABPFN_RUNS` (`--tabpfn` per booking) decides how they are
   booked: `separate` as above; `joined` (the setting since 2026-10-29) as jobs of the
   predictor run that covers their snapshot, in its `tabpfn` input
   (`HORIZON@AS_OF@GAME_IDS`, `;`-separated, `nba_ou.scheduling.tabpfn_jobs`),
   run once that run's own predictions are stored, each stored as a TabPFN run
   of its own (`nba_ou.prediction.tabpfn_run`); `none` books none. A joined job
   reads the predictor run's schedule, referees, reports and line source, and
   its rows are cut at its snapshot as a separate run's are. A predictor run
   carrying jobs gets a pool job of its own (its payload differs), and a
   snapshot no predictor run covers keeps its own TabPFN dispatch. The
   predictor's `tabpfn_refresh` input (off by default) fetches the schedule,
   reports and odds again before the jobs and builds their rows at that
   moment instead of at their snapshot; `tabpfn_modes` is their `modes`.
   `--no-tabpfn` books none of these runs. Snapshots already past when the
   scheduler runs (normally 09:30 Madrid) are lost, including early T-720s;
   there is no catch-up or previous-day booking.
4. after the last game, whose end is estimated as its tip-off + 150 min
   (`GAME_LENGTH_MINUTES`), adds:
   - the date's past-snapshot predictions (`nba_predict_past_snapshots.yml`,
     `game_date` passed) 30 min after that end;
   - the final scores, finished matches and lineups
     (`update_finished_matches_daily.yml`) 2 h after it.
5. books cron-job.org jobs that dispatch those workflows at those times.

The TabPFN workflow's `target` input is a list separated by spaces or commas,
`line_error total_points` by default (also the script's default and what the
scheduler requests). From the CLI, use for example
`python scripts/predict_tabpfn.py --targets line_error` (`--target` remains an
alias). Each target uses its own production spec and
training window; database updates, the schedule and tonight's feature rows are
shared, and all outputs are stored in one prediction run. If a target's inference
fails, the other targets are still attempted and saved, and the run fails with
a partial result. `--feature-slot` requires a single target.

Its `modes` input (`--modes` in the script) lists the fits of each target, `standard` by
default, which is today's model and what the scheduler requests. The other
modes are TabPFN's thinking at medium or high effort (`thinking_medium`,
`thinking_high`): the server searches the fit within its default time budget,
which costs more time and inference. Each mode is stored as its own variant
(`tabpfn_v3_5`, `tabpfn_v3_5_thinking_high`), so a run with
`modes="standard thinking_high"` stores both side by side. No scheduled run
uses thinking yet; dispatch one by hand to try it.

## The day's schedule

All on the one self-hosted runner. The two fixed jobs come from `setup_cronjobs.py`,
at the times of `[Schedule]` in `src/nba_ou/config.ini` (Madrid time, defaults
below); the rest are booked by the scheduler from the day's games.

| `[Schedule]` key | Default | Job |
|---|---|---|
| `TIMEZONE` | `Europe/Madrid` | zone of the two times below |
| `UPDATE_AND_RETRAIN_AT` | `08:00` | `retrain_prediction_models.yml` |
| `SCHEDULE_PREDICTIONS_AT` | `09:30` | `schedule_snapshot_triggers.yml` |

To change a time, edit it and run `python scripts/cronjobs/setup_cronjobs.py
--apply`: cron-job.org holds the schedule, so the edit alone changes nothing.
`setup_cronjobs.py` refuses a `SCHEDULE_PREDICTIONS_AT` that falls before
US/Pacific midnight on any day of this year or the next (earlier than 09:00
Madrid), because that run would book the previous night. Keep the booking after
the retrain: both share the runner, and a booking that waits for the refit
skips the snapshots that pass meanwhile.

| When | Workflow | What |
|---|---|---|
| 08:00 | `retrain_prediction_models.yml` (`update_database`, `promote`) | database update; on a game day, then refit, smoke test, promotion |
| 09:30 | `schedule_snapshot_triggers.yml` | books everything below |
| each snapshot | `nba_predictor_daily.yml` | every enabled slot |
| each game at tip − 12 h, − 8 h, − 6 h, − 4 h; 2 min after the predictor run covering it | `nba_predictor_daily_tabpfn_client.yml` | TabPFN line_error + total_points, matching snapshot slots, only that tip-off group |
| last end + 30 min | `nba_predict_past_snapshots.yml` | the date's intermediate slots on their own snapshots |
| last end + 2 h | `update_finished_matches_daily.yml` | final scores (prediction store), finished matches, lineups |

On 2026-10-22 (last tip 01:30 UTC), the post-game runs land at 06:30 and 08:00
Madrid.

**Retrain on a day without games.** `scripts/check_games_today.py` reads
today's Eastern date on the scoreboard. Without games, the workflow runs only
the database update; the refit, smoke test and promotion are skipped. An
unreadable scoreboard counts as a game day. `refit_without_games` forces the
refit on a manual run.

The standalone 09:00 `update_database_daily.yml` job is retired: the retrain
workflow runs the same update as its first job, and refits only after it
succeeds. `--apply` disables the old job (`RETIRED_TITLES`). If the refit
outlasts 09:30, the scheduler waits for the runner and skips the times that
pass meanwhile.

**Post-game runs depend on the scheduler.** On a day without games, or when
the 09:30 scheduler fails, no past-snapshot or final-score run is booked. The
final-score update retries every tipped game without a score on its next run,
so a day lost is caught up on the next game day. An overtime game still in
progress at +2 h is left for that next run too.

## Switching it off

Two repository variables (GitHub → Settings → Secrets and variables → Actions →
Variables) act as off switches. Unset, or any value other than `false`, means
on.

| Variable | `false` skips |
|---|---|
| `DAILY_UPDATE_ENABLED` | `retrain_prediction_models.yml` (DB update, refit, smoke test, promotion), `update_database_daily.yml`, `update_finished_matches_daily.yml` |
| `PREDICTIONS_ENABLED` | `schedule_snapshot_triggers.yml` (no new bookings), `nba_predictor_daily.yml`, `nba_predictor_daily_tabpfn_client.yml`, `nba_predict_past_snapshots.yml` |

    gh variable set PREDICTIONS_ENABLED --body false   # off
    gh variable set PREDICTIONS_ENABLED --body true    # on again

Each workflow checks its variable in its job's `if:`, so a run that is off
still starts (cron-job.org sees a successful dispatch and sends no failure
mail) and shows its job as skipped, without taking the runner. The check reads
the variable when the run starts: a switch flipped mid-day stops the runs
already booked for the rest of the day, and a run already going finishes.

The switch skips manual dispatches too, unless **force** is ticked: every
switched workflow has a boolean `force` input (default off) that runs it
whatever the variable says. cron-job.org never sends it, so the booked runs
stay off. The retrain passes its `force` on to the database update it calls.

With `PREDICTIONS_ENABLED` off, the final-score run is not booked either; the
next booked one catches up every tipped game without a score.

## Which snapshot each run predicts from

Every predictor run predicts every enabled slot (policy of 2026-10-20). With
the current `INTERMEDIATE_INPUT_MODE = latest` and `LATEST_INPUT_SLOTS = all`:

- **All 48 intermediate slots read each game's *latest* row**: minutes to tip
  at the moment the run builds its rows (`as_of`, rounded up; the
  `LATEST_INPUT` rows), limited to what each slot's training rows carried. They
  do **not** read the row of their own horizon. A trigger at `tip − 120` gives
  rows at `tip − (120 − lag)`. The lag is the time from dispatch to the row
  build: runner start, dependency install, database update and closing frame.
  It is not measured yet, because no daily-predictor run is in the prediction
  store. Once runs are stored, `runs.started_at` against the `input_mode = 2`
  rows' `tipoff_at − snapshot_horizon_minutes` gives it, and
  `--offset-minutes −<lag>` would land the build on the snapshot.
- The 3 closing slots read the closing frame with the odds current at the run.
- With `INTERMEDIATE_INPUT_MODE = own`, each intermediate slot reads its own
  horizon's row instead. A row whose snapshot has passed by `as_of`
  (`SNAPSHOT_PASSED`) is the row the slot was trained on. A later horizon's
  row is built from the ticks so far.

## Do the times agree?

- **Tip-off.** The scheduler takes the NBA scoreboard's `gameTimeUTC`. The
  predictor times its snapshots (`SNAPSHOT_TS_UTC = TIPOFF_UTC − h`) on the SBR
  page's `startDate` tip-off. On 2026-10-20 to 2026-10-22 (16 games), the two
  gave the same set of whole-minute UTC tip-offs, as did the season schedule feed.
- **Cron.** Each job's schedule is in UTC, built from the same UTC trigger
  times. After `--apply`, the script reads every job's `nextExecution` back
  from `GET /jobs` and exits 1 if one is not the planned next time or the job
  is disabled.

## Why a pool of jobs

The cron-job.org API allows 100 requests a day, and job creation at most once a
second and five times a minute. One job per game × horizon would need about 170
jobs on a 10-game night. Instead:

- For the normal predictor, **one job per (UTC date, minute of the hour)**, firing at that
  minute of several hours (`minutes=[m]`, `hours=[…]`, `mdays=[d]`,
  `months=[mo]`). It fires at exactly the planned times and no others. Runs
  mostly fall on :00 and :30, so a night needs about 4–8 jobs.
- TabPFN needs **one job per (snapshot time, horizon)** because its cutoff
  changes per run. Two targets share that job; games with the same
  tip-off share it too. Different horizons at the same minute remain separate.
- **A fixed pool**, titled `Scorecast - snapshot trigger NN`. Each day the
  script makes one `GET /jobs`, then patches jobs 01..n with the day's
  schedules. It creates a job only when the pool is too small, waiting 13 s
  between creates, and disables pool jobs it does not need. Each active job
  needs one write, plus writes for jobs being disabled and two GETs. It never
  touches jobs with other titles. TabPFN increases the pool size: at most
  `MAX_POOL=90` jobs, leaving room for both GETs under the 100-request allowance.
  The scheduler has a 30-minute timeout to allow the initial pool growth at
  one creation per 13 s; later days reuse the jobs.
- Job titles append their purpose, for example
  `Scorecast - snapshot trigger 02 | TabPFN T-240 CLE@PHI | snapshot 26/10 21:30 Madrid`.
  TabPFN titles show up to two matchups whose snapshot it is (plus the
  remaining game count; the run predicts the whole slate), the horizon and the
  snapshot's Madrid time; the execution time remains in the
  console's schedule column. A predictor job lists every Madrid time it fires
  (`Predictor 22:00, 23:00, 00:00 Madrid`), since the console shows only the
  next one. The post-game jobs read `Past snapshots <date>` and `Final scores`.
  The numbered prefix identifies the reusable pool slot: both legacy and
  descriptive names are recognized and updated in place, so applying the new
  plan does not create duplicate jobs. A job disabled for the day goes back to
  the bare `Scorecast - snapshot trigger NN` title. Disabled jobs are never
  deleted: they wait for a day that needs more jobs, which saves the rate-limited
  creations.
- Schedules are in **UTC**, so a daylight-saving change cannot skip or repeat
  a trigger. Each schedule expires at the end of its UTC date, so a job left
  over from a day when the scheduler failed never fires again.

If the scoreboard is unavailable, the script exits 1 and leaves the pool as it
is. On a day with no games, it disables the whole pool.

## Running it

    python scripts/cronjobs/schedule_snapshot_triggers.py                        # dry run, no secrets
    python scripts/cronjobs/schedule_snapshot_triggers.py --apply                # book today
    python scripts/cronjobs/schedule_snapshot_triggers.py --game-date 2026-10-23 # another date

Options: `--offset-minutes` (default 0; normal predictor only, negative fires
earlier) and `--no-tabpfn` (leave all four TabPFN horizons out).

A scheduled TabPFN run is equivalent to:

```bash
python scripts/predict_tabpfn.py --targets line_error total_points \
  --line intermediate --horizon-minutes 720 --skip-db-update \
  --game-date 2026-10-26 --as-of 2026-10-26T11:00:00Z \
  --snapshot-game-ids 0022600101 0022600102
```

Manual runs can omit `game_date` (current Pacific date) and `as_of` (live), and
can add `--game-ids` to predict only those games (the scheduler never does). A cutoff requires an explicit timezone and intermediate mode.

The workflow needs the repository secrets `CRONJOB_API_KEY` and
`CRONJOB_GITHUB_DISPATCH_TOKEN`. The second is a fine-grained PAT on this repo
with "Actions: Read and write"; see `nba_ou.scheduling.cronjob_org`. Run the
scheduler once a day, at `SCHEDULE_PREDICTIONS_AT` (09:30 Madrid). That is always
after Pacific midnight: normally 00:30 Pacific, 01:30 when only one side has
changed its clocks. Snapshots before the run are past and are skipped, e.g.
T-1080 and T-960 of a 19:00 ET game.

## One firing per run, and daylight saving

- **Each planned run fires once.** A job's schedule names one UTC date (`mdays`
  and `months`) and the hours of that date. It also expires at 23:59:59 UTC of
  that date (`expiresAt`), so it can never repeat on the same date a month or a
  year later. The next day's scheduler run reuses the same pool jobs with that
  day's date and hours. If the scheduler fails one day, nothing fires that day:
  yesterday's times are never replayed.
- **Daylight saving cannot shift a run.** The tip-offs come in UTC, the
  snapshots are computed in UTC and the jobs are scheduled in UTC. Madrid time
  appears only in the printout. The predictor's Pacific date (the window above)
  is computed with `zoneinfo` for that date, so it follows PDT/PST. The
  scheduler's own daily job (`setup_cronjobs.py`) is in `[Schedule] TIMEZONE`
  (`Europe/Madrid`), so it stays at the same Spanish time all year.

## Caveats

- Both prediction workflows use `cancel-in-progress: false` and `queue: max`:
  an active run finishes and up to 100 pending runs per concurrency group are
  retained in FIFO order (by entry into the group). Only a full queue cancels
  new arrivals. The normal predictor and TabPFN have separate groups; their
  jobs wait for the same self-hosted runner when they coincide. TabPFN fires
  2 min after the predictor run covering its snapshot so that one is queued first,
  but cross-workflow execution order is not guaranteed. See [GitHub's concurrency reference](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency).
- More TabPFN runs increase runner time and inference usage: every tip-off
  group needs two independent fits at each of its four horizons. Queuing
  preserves work but does not guarantee execution at the trigger's minute.
- The T−30 run can finish close to tip-off. The predictor never
  stores a game that is in progress.
