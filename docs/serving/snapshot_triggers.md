# Snapshot triggers: running the predictor at each snapshot time

Status: **Implemented** (2026-10-22). The daily jobs (08:00 and 09:35 Madrid by
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
3. adds **four TabPFN jobs** at **T-600, T-480, T-360 and T-240** (10 h, 8 h,
   6 h and 4 h) of the **last scheduled tip-off of the date**. Postponed, cancelled
   and TBD games do not select the last tip-off. Games sharing the last tip-off
   share its anchor metadata; earlier tip-offs do not book extra jobs.

   Each job predicts the targets and modes of `TABPFN_FITS`
   (`schedule_snapshot_triggers.py`), table below. The first three
   run intermediate, using each target's production spec and training rows
   of that exact horizon. Intermediate
   reconstructs **every game of the slate, including started/final games**, at
   its own pregame snapshot. For example, at T-6 of a 03:00 last game (21:00
   the previous evening), a 01:00 game's row reads its own 19:00 snapshot.
   Only the selected horizon is built. The full slate and crews are read for
   TabPFN alone; normal predictions retain their live schedule.

   **T-600, T-480 and T-360 also predict the closing line**,
   inside the same GitHub job, with that job's API key.
   **T-240 runs closing only**, without an intermediate T-240 prediction.

   | Run | Intermediate fits | Closing fits | Key |
   |---|---|---|---|
   | T-600 | standard: `line_error`, `total_points` | standard: `line_error`, `total_points` | normal |
   | T-480 | standard: all three; `thinking_high`: `line_error`, `total_points` | standard: all three | `TABPFN_API_KEY_ALT` |
   | T-360 | standard: `line_error`, `total_points` | standard: `line_error`, `total_points` | normal |
   | T-240 | — | standard: all three | `TABPFN_API_KEY_ALT` |

   Since 2026-10-29 `spread_error` is left out of T-600 and T-360, and
   `thinking_high` runs only for the totals targets on T-480's intermediate line.
   Each job carries its own fits, `MODE=TARGET[,TARGET]` items separated by
   spaces (`nba_ou.scheduling.tabpfn_jobs`, `--fits` of `predict_tabpfn.py`):
   a joined job as the `FITS` field of its `tabpfn` item, a standalone
   dispatch as its `fits` and `closing_fits` inputs.
   Joined dispatches append a `0` closing
   job after the intermediate job; standalone dispatches set `include_closing`
   and run a second script step, without repeating the database update.
   T-240 carries only `0` when joined, or `line=closing` when standalone;
   standalone closing has no intermediate horizon, cutoff or anchor-game metadata.
   Closing uses the line and closing features available at execution time,
   not a future final market line. It retains the normal closing requirements:
   only pending games with both teams' injury reports filed are eligible.
   If no game qualifies, only the closing job is skipped.

   `game_date` and `as_of` retain the slate and snapshot cutoff. The
   `snapshot_game_ids` metadata identifies the last-tip games, not a game
   filter. Each earlier game's inputs stop at its own snapshot: line ticks,
   archived rosters, injury reports/news and referee releases. The scoreboard
   and line-history tip-offs can differ by a few minutes; when an anchor
   game's snapshot is within 15 min of the cutoff, the run waits for that
   snapshot if necessary and rebuilds there. Earlier games do not move the
   shared cutoff.

   `[Schedule] TABPFN_RUNS` (`--tabpfn` per booking) selects:
   - `joined` (current): each job runs inside the predictor run covering the
     last game's snapshot, after its normal predictions are stored. It reuses
     the run's data/history and connections, and stores a separate TabPFN run.
   - `separate`: a standalone TabPFN workflow fires 2 min after that covering
     predictor run (or the snapshot itself when none covers it), keeping the
     snapshot cutoff. Coinciding dispatches are spaced at least one minute.
   - `none` / `--no-tabpfn`: no TabPFN jobs.

   A joined job gets a pool job with its own `tabpfn` input
   (`HORIZON@AS_OF@GAME_IDS@FITS`); a snapshot without a covering predictor run
   retains a standalone dispatch. Normal predictor times do not change.
   The fits are those of the table above, the same in both booking modes.
   **T-480 and T-240 use the repository secret
   `TABPFN_API_KEY_ALT` for every fit**;
   T-600 and T-360 use the normal `TABPFN_ACCESS_TOKEN`.
   The scheduler sets `tabpfn_use_alt_key=true` on the joined T-480 and T-240
   dispatches, or `use_alt_key=true` on their standalone fallbacks. The workflow exposes
   `TABPFN_USE_ALT_KEY=true` and the client selects the alternate secret.
   A missing alternate key fails explicitly, without falling back to the
   normal account. Add `TABPFN_API_KEY_ALT` before enabling this policy.
   The optional manual
   `tabpfn_refresh` fetches the data again and predicts at that moment instead
   of replaying the booked cutoff. Snapshots already past at booking (normally
   09:35 Madrid) are skipped, with no catch-up or previous-day booking.
4. after the last game, whose end is estimated as its tip-off + 150 min
   (`GAME_LENGTH_MINUTES`), adds:
   - the date's past-snapshot predictions (`nba_predict_past_snapshots.yml`,
     `game_date` passed) and daily baselines 30 min after that end;
   - the final scores and finished matches
     (`update_finished_matches_daily.yml`) 2 h after it, **capped at 07:50
     the following morning** in `[Schedule] TIMEZONE`. An earlier dispatch
     stays earlier. This caps the planned dispatch; runner queues can delay
     its actual start. Lineup updates remain disabled.

   A separate daily job retries final-score updates at **11:11**, including
   days without games or when the game-day scheduler fails.

   The final prediction run computes `random_line_error_baseline` and
   `historical_average_line_error_baseline` once per game, using the T-0
   pregame snapshot already built for the whole slate. Daytime runs no longer
   compute either baseline. Random OVER/UNDER picks are seeded by slate date
   and game ID, so reruns, row reordering and partial coverage retain the same
   choices. The mean uses complete games from the current and previous season,
   strictly before the slate date, excluding preseason and All-Star games.
   Both are saved as retrospective closing outputs (`kind=2`, `workflow=2`),
   with their baseline names in `variant` and their actual inference times.
   Missing T-0 lines or a failed baseline mark the run partial and fail it
   after saving the available outputs. This remains one scheduled run per
   slate; a manual rerun produces a separate run with reproducible random picks.
5. books cron-job.org jobs that dispatch those workflows at those times.

The TabPFN workflow's `target` input is a list separated by spaces or commas,
`line_error total_points spread_error` by default (also the script's default and what the
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
`modes="standard thinking_high"` stores both side by side. `fits` (`--fits`)
gives each mode its own targets instead of `target` and `modes`, e.g.
`standard=line_error,total_points,spread_error thinking_high=line_error,total_points`.
The scheduled runs fit (table in "What it does"): T-600 and T-360 two standard
intermediate and two standard closing fits each; T-480 five intermediate fits
(three standard, two high-thinking) and three standard closing fits; T-240
three standard closing fits.
Closing fits run only when games are eligible. All fits of a line share its preparation
and are stored in the same TabPFN run, with a variant per target and mode.

## The day's schedule

All on the one self-hosted runner. The three fixed jobs come from `setup_cronjobs.py`,
at the times of `[Schedule]` in `src/nba_ou/config.ini` (Madrid time, defaults
below); the rest are booked by the scheduler from the day's games.

| `[Schedule]` key | Default | Job |
|---|---|---|
| `TIMEZONE` | `Europe/Madrid` | zone of the daily times and the morning cap |
| `UPDATE_AND_RETRAIN_AT` | `08:00` | `retrain_prediction_models.yml` |
| `SCHEDULE_PREDICTIONS_AT` | `09:35` | `schedule_snapshot_triggers.yml` |
| `FINISHED_MATCHES_RETRY_AT` | `11:11` | independent daily `update_finished_matches_daily.yml` |
| `FINISHED_MATCHES_LATEST_AT` | `07:50` | latest post-game final-score dispatch the following morning |

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
| 09:35 | `schedule_snapshot_triggers.yml` | books everything below |
| each snapshot | `nba_predictor_daily.yml` | every enabled slot |
| last tip − 10 h, − 8 h, − 6 h | joined inside `nba_predictor_daily.yml` | TabPFN intermediate: `line_error` + `total_points` at T-10/T-6; all three + `thinking_high` for the totals at T-8; every game at its own snapshot |
| last tip − 10 h, − 8 h, − 6 h, in the same job | TabPFN closing | standard, eligible pending games: `line_error` + `total_points` at T-10/T-6, all three at T-8; alternate key at T-8, normal key otherwise |
| last tip − 4 h | joined inside `nba_predictor_daily.yml`, standalone fallback | TabPFN closing only, all three targets, standard mode, alternate key, eligible pending games |
| last end + 30 min | `nba_predict_past_snapshots.yml` | intermediate slots on their own snapshots; registry closing models on reconstructed T-8h/T-6h/T-4h/T-0; random and historical-average baselines on T-0 |
| earlier of last end + 2 h and next morning 07:50 | `update_finished_matches_daily.yml` | final scores (prediction store), finished matches |
| 11:11 every day | `update_finished_matches_daily.yml` | retry pending final scores and finished matches |

On 2026-10-22 (last tip 01:30 UTC), the post-game runs land at 06:30 and 07:50
Madrid (the final-score dispatch is capped).

**Retrain on a day without games.** `scripts/check_games_today.py` reads
today's Eastern date on the scoreboard. Without games, the workflow runs only
the database update; the refit, smoke test and promotion are skipped. An
unreadable scoreboard counts as a game day. `refit_without_games` forces the
refit on a manual run.

The standalone 09:00 `update_database_daily.yml` job is retired: the retrain
workflow runs the same update as its first job, and refits only after it
succeeds. `--apply` disables the old job (`RETIRED_TITLES`). If the refit
outlasts 09:35, the scheduler waits for the runner and skips the times that
pass meanwhile.

**Post-game runs depend on the scheduler.** A day without games books no
past-snapshot or post-game final-score run. The independent 11:11 retry still
runs and looks up every tipped game lacking a final score. It also catches
results unavailable at the morning cap or missed when the scheduler failed.

**Applying a policy change.** After merging to `main`, run
`setup_cronjobs.py --apply` to create/update the daily jobs (including the
11:11 retry), and dispatch `schedule_snapshot_triggers.yml` with `apply=true`
to replace the remaining game-day reservations. A dry run (`apply=false`)
prints the plan. Editing or merging the files alone does not update existing
cron-job.org reservations. Neither operation replays times already passed.

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

With `PREDICTIONS_ENABLED` off, no post-game final-score run is booked. The
independent 11:11 retry remains active unless `DAILY_UPDATE_ENABLED` is off.

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
  from `GET /jobs`. A non-null prediction must equal the planned next time.
  If the prediction is null, the job must be enabled and its entire stored
  schedule must match the plan exactly, including timezone, date, times and
  expiry; otherwise the script exits 1. Disabled or missing jobs always fail.
  The API documents null as an unavailable prediction, not a failed booking.
  On 2026-10-31 this caused a false failure for three correctly booked Nov 1
  jobs: the provider's PHP predictor adds a month before resetting the day,
  so Oct 31 overflows to Dec 1 and skips November. This fallback verifies the
  saved booking without changing its execution times or making extra API calls.
  See the [API contract](https://docs.cron-job.org/rest-api.html#job) and
  [provider predictor](https://github.com/pschlan/cron-job.org/blob/master/api/lib/ExecutionPredictor.php).

## Why a pool of jobs

The cron-job.org API allows 100 requests a day, and job creation at most once a
second and five times a minute. One job per game × horizon would need about 170
jobs on a 10-game night. Instead:

- For the normal predictor, **one job per (UTC date, minute of the hour)**, firing at that
  minute of several hours (`minutes=[m]`, `hours=[…]`, `mdays=[d]`,
  `months=[mo]`). It fires at exactly the planned times and no others. Runs
  mostly fall on :00 and :30, so a night needs about 4–8 jobs.
- TabPFN needs **one job per (snapshot time, horizon)** because its cutoff
  changes per run. Three targets share each job, anchored only on the date's last tip-off.
  There are at most four TabPFN jobs per slate.
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
  `Scorecast - snapshot trigger 02 | TabPFN T-360 CLE@PHI | snapshot 26/10 19:30 Madrid`.
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
earlier) and `--no-tabpfn` (leave all four TabPFN times out).

A scheduled TabPFN run is equivalent to:

```bash
python scripts/predict_tabpfn.py --targets line_error total_points spread_error \
  --line intermediate --horizon-minutes 600 --skip-db-update \
  --game-date 2026-10-26 --as-of 2026-10-26T11:00:00Z \
  --snapshot-game-ids 0022600101 0022600102
```

Manual runs can omit `game_date` (current Pacific date) and `as_of` (live), and
can add `--game-ids` to predict only those games (the scheduler never does). A cutoff requires an explicit timezone and intermediate mode.

The workflow needs the repository secrets `CRONJOB_API_KEY` and
`CRONJOB_GITHUB_DISPATCH_TOKEN`. The second is a fine-grained PAT on this repo
with "Actions: Read and write"; see `nba_ou.scheduling.cronjob_org`. Run the
scheduler once a day, at `SCHEDULE_PREDICTIONS_AT` (09:35 Madrid). That is always
after Pacific midnight: normally 00:35 Pacific, 01:35 when only one side has
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
- More TabPFN runs increase runner time and inference usage: the slate
  needs two standard intermediate fits at T-600 and T-360, and three standard
  plus two high-thinking intermediate fits at T-480. Closing adds two standard
  fits at T-600 and T-360 and three at T-480 and T-240 when games are eligible. Queuing
  preserves work but does not guarantee execution at the trigger's minute.
- The T−30 run can finish close to tip-off. The predictor never
  stores a game that is in progress.
