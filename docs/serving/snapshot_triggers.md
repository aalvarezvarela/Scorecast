# Snapshot triggers: running the predictor at each snapshot time

Status: **Implemented** (2026-10-22). The scheduler's daily cron-job.org entry
(a row in `scripts/cronjobs/setup_cronjobs.py`) has not been added yet.

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
3. books cron-job.org jobs that dispatch `nba_predictor_daily.yml` at those
   times.

On 2026-10-21 (11 games), booked at 10:00 Madrid: 176 snapshots became 21 runs
on 4 cron jobs. The early games' T-1080 and T-960 had already passed. Tips on
:00 and :30 put snapshots 30 min apart, which pair up, so the runs come every
hour from 11:00 Madrid, plus each tip time's T-30.

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

## Why a pool of jobs, one per minute of the hour

The cron-job.org API allows 100 requests a day, and job creation at most once a
second and five times a minute. One job per game × horizon would need about 170
jobs on a 10-game night. Instead:

- **One job per (UTC date, minute of the hour)**, firing at that
  minute of several hours (`minutes=[m]`, `hours=[…]`, `mdays=[d]`,
  `months=[mo]`). It fires at exactly the planned times and no others. Runs
  mostly fall on :00 and :30, so a night needs about 4–8 jobs.
- **A fixed pool**, titled `Scorecast - snapshot trigger NN`. Each day the
  script makes one `GET /jobs`, then patches jobs 01..n with the day's
  schedules. It creates a job only when the pool is too small, waiting 13 s
  between creates, and disables pool jobs it does not need. That is about
  10–15 requests a day, plus one `GET /jobs` to check the booking. It never
  touches jobs with other titles.
- Schedules are in **UTC**, so a daylight-saving change cannot skip or repeat
  a trigger. Each schedule expires at the end of its UTC date, so a job left
  over from a day when the scheduler failed never fires again.

If the scoreboard is unavailable, the script exits 1 and leaves the pool as it
is. On a day with no games, it disables the whole pool.

## Running it

    python scripts/cronjobs/schedule_snapshot_triggers.py                        # dry run, no secrets
    python scripts/cronjobs/schedule_snapshot_triggers.py --apply                # book today
    python scripts/cronjobs/schedule_snapshot_triggers.py --game-date 2026-10-23 # another date

Option: `--offset-minutes` (default 0; negative fires earlier).

The workflow needs the repository secrets `CRONJOB_API_KEY` and
`CRONJOB_GITHUB_DISPATCH_TOKEN`. The second is a fine-grained PAT on this repo
with "Actions: Read and write"; see `nba_ou.scheduling.cronjob_org`. Run the
scheduler once a day, e.g. at 10:00 Madrid. That is always after Pacific
midnight: 01:00 PDT in summer, 01:00 PST in winter, and 02:00 PDT in the weeks
when only Europe has changed its clocks. Snapshots before the run are past and
are skipped, e.g. T-1080 and T-960 of a 19:00 ET game.

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
  scheduler's own daily job (`setup_cronjobs.py`) is in `Europe/Madrid`, so it
  stays at 10:00 Spanish time all year.

## Caveats

- A predictor run takes about 13 min or more. GitHub's concurrency group
  `daily-nba-predictor` keeps one run going and **one** pending: when triggers
  bunch up, a newer queued run replaces the older pending one. Since every run
  predicts every slot, only that run's time point is lost.
- The T−30 run can finish close to tip-off. The predictor never
  stores a game that is in progress.
