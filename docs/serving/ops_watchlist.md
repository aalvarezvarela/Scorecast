# Ops watchlist

Status: **Reference** (checked 2026-10-22) — a living list; items are removed
once resolved.

Things to check on the daily jobs and their external sources, mostly around
the 2026-27 season start. Each item: what was seen, why it matters, and how to
close it. Remove an item once it is resolved; move anything that turns into
real work into its own doc.

Defects in the training data live in `docs/pipeline/defect_fixes_2_5.md`, not here.

## Open

### 1. Line history finds no 2026-27 dates yet (seen 2026-10-20)

`update_line_history_database.py` reported `Season 2026: 0 date(s) to fetch`.
Expected for now: line history only stores finished games from `nba_games`, and
on the 2026-10-20 morning run the 2026-27 season held only preseason games
(filtered out). The regular season opens the evening of 2026-10-20 (DET–BOS,
NYK–PHI, SAS–OKC).

**Check:** from 2026-10-21, the step should report recent dates and inserted
ticks. If it still says 0 after a few game days, check whether the games
reached `nba_games` first (`update_all_databases.py`: `Found N existing games`).

### 2. Legacy schedule feed returns 403 for 2026-27 (seen 2026-10-20) — fixed in code

**Status:** `fetch_season_schedule` now reads `ScheduleLeagueV2` first and
falls back to the legacy feed only if stats.nba.com fails
(`tests/test_fetch_nba_schedule.py`). Validated against the legacy feed for
2016-17 through 2025-26:
- 2021-22, 2023-24, 2024-25: identical in every column.
- 2016-17, 2019-20, 2020-21, 2022-23: three regular-season openers tip one
  minute later in V2 (actual vs scheduled tip), plus seven 2016 preseason games;
  V2 has no 2019-20 preseason (112 games, unused by the stores).
- 2025-26: V2 adds 129 games the frozen feed lacks (play-in, playoffs, NBA Cup);
  with `tipoff_corrections` applied to both, every shared tip-off agrees.

A full rebuild of the line history or injury store can therefore move those
three openers' tip-offs by a minute. **Close when** the fix is merged and the
next `update_database_daily` run gets through the injury step.

`https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{season}/league/00_full_schedule.json`
(`src/nba_ou/fetch_data/nba_schedule/fetch_nba_schedule.py`), with the same
browser headers the code sends: season `2025` returns 200 and season `2026`
returns 403. The NBA appears not to publish this legacy feed for 2026-27.

Not a "season not started yet" effect: the 2026-27 schedule has been public
since August, and `resolve.load_game_dimension` already documents that the
2025-26 file stopped updating on 2026-04-12. The feed looks abandoned.

**Impact:**
- **Fatal for the injury-report ingest.** `daily_injury_reports.py` →
  `injury_report_aiven/ingest.py:ingest_season` →
  `resolve.load_game_dimension` calls `fetch_schedules` with no fallback and
  crashes with the 403 (2026-10-20 run, after the IAM fix, once the 132 PDFs
  were stored). The game dimension supplies the tip-offs that clamp injury spans.
- **Loses every intermediate prediction.**
  `create_training_data/live_game_day.py` passes `fetch_schedules([season])`
  to the live line-history scrape with no fallback. `predict_nba_games.py`
  catches it (`Intermediate rows unavailable: ...`), so the closing slots still
  run but every intermediate slot predicts nothing. `live_replay.py` (parity
  scripts) fails the same way.
- **Silent for line history.** Both updaters
  (`scripts/update_databases/update_line_history_database.py`,
  `scripts/line_history/update_line_history.py`) catch the error and continue,
  but lose the tip-off cross-check that catches moved games (how in-play SBR
  lines end up stored as closing lines).

**Replacement, verified 2026-10-20:** `nba_api`'s `ScheduleLeagueV2`
(stats.nba.com, `season="2026-27"`) returns the full season: 1,274 games (1,206
regular season, 67 preseason, 1 NBA Cup final placeholder) with
`gameDateTimeUTC`, `gameDateTimeEst`, home/away tricodes, arena and
`gameStatus`. That covers every column of `SCHEDULE_COLUMNS`. Safest change:
keep the legacy feed for past seasons (their data is validated) and fall back to
`ScheduleLeagueV2` when it fails. Check that both give the same tip-offs on
2024-25 before relying on it.

### 3. cdn.nba.com returns 403 (seen 2026-10-20)

From the dev machine, with browser User-Agent, Referer and Origin headers,
every path tried returned 403:

- `liveData/scoreboard/todaysScoreboard_00.json` (live scoreboard)
- `liveData/boxscore/boxscore_0022500001.json` (a past box score)
- `staticData/scheduleLeagueV2_1.json` (current schedule)

The CI runner hit it too: `live ScoreBoard live-game lookup failed: Expecting
value` in `update_database_daily` (a non-JSON body, consistent with a 403 page).
It may be an IP- or bot-level block rather than a removed feed.

**Consumers:**
- `fetch_data/live_games/live_games.py` falls back to `ScoreboardV2` and then
  `ScoreboardV3` (stats.nba.com), so it degrades gracefully.
- `postgre_db/game_time_index/sync_game_time_index.py` reads box scores from
  the CDN with no fallback. It is run by hand
  (`scripts/create_databases/sync_nba_game_index_for_season.py`), and the
  injury-report resolver reads its table and tolerates it being empty.

**Check:** whether the runner gets JSON from the CDN on a game night. If not,
`game_time_index` cannot be filled for 2026-27 from the CDN.

### 4. CI role cannot write to `injury_reports/` in S3 (seen 2026-10-20)

`update_database_daily`, step "Store injury reports": S3 preflight failed with
`AccessDenied` on `s3:PutObject`. The OIDC role is `S3ModelsFullAccess`
(`assumed-role/S3ModelsFullAccess/GitHubActions`), and the bucket uses a
per-prefix allowlist. The policy JSON is in
`docs/injuries/report_archive.md` §17.2.1. It was written for the
`adrian-personal-cli` user, but the role needs it too. Added to the role as an
inline policy on 2026-10-20; the next run stored 132 PDFs and the 2026-27
manifest under `injury_reports/`, so that prefix is fixed.

**Also check on the role:**
- `train_data/input_cache/*` PutObject. "Refresh the S3 input cache" writes
  there and has never run in CI. `season_store.py` only claims the role can
  *read* `train_data/`.
- `prediction_snapshots/*`. `predict_nba_games.py` uploads snapshots there,
  and failures are caught and only printed (`Failed to upload ... snapshot`),
  so a missing grant loses them silently.

**Close when:** a full `update_database_daily` run passes both steps.

### 5. The new `update_database_daily` steps have never completed in CI

"Store injury reports and finished games' spans" and "Refresh the S3 input
cache" were added after the last green run (2026-06-15). The 2026-10-20 runs
failed before reaching them (OIDC on the branch run, the S3 preflight on
main). Problems further in (PDF download from the runner, DB writes) are still
unknown.

**Also:** the cache refresh runs last and is skipped whenever an earlier step
fails, so one failing source leaves the S3 input cache stale for the predictor.
Consider `if: always()` (or `!cancelled()`) on that step.

### 6. `requirements_predict.txt` not yet tested in CI (2026-10-20)

`nba_predictor_daily.yml` now installs `requirements_predict.txt`, a minimal
set traced from the imports of `predict_nba_games.py` (no TabPFN, no dev or
Jupyter tools). It was checked in a clean Python 3.12 venv (all module-level
imports load), but no real prediction has run with it. The last
`nba_predictor_daily` run on GitHub is from June.

**Check:** dispatch it once by hand and read the log. Note that on the
self-hosted runner `setup-python` installs into a shared tool-cache
interpreter, so packages left by `requirements.txt` are still present and a
missing dependency could stay hidden. A dedicated venv would make the test real.

### 7. `update_database_daily` installs every Poetry dependency (~10 min)

"Install dependencies" ran 12:47–12:58 on 2026-10-20. Besides
`requirements.txt`, the step runs `pip install -e .`, which also installs every
dependency in `pyproject.toml`, including `autogluon` (and torch), `streamlit`,
`optuna` and `ipywidgets`. The job uses none of them. `pip install -e .
--no-deps` should be enough. Part of the planned requirements reorganisation
(per-job files under a `requirements/` folder, `xgboost-cpu` for serving).

## Season-rollover sources to keep an eye on

External sources the daily jobs depend on. A new season is when URLs, IDs and
formats change.

| Source | Used for | Code | Watch for |
|---|---|---|---|
| stats.nba.com (`nba_api`): `LeagueGameFinder`, `BoxScoreSummaryV3`, `CommonTeamRoster`, `ScoreboardV2/V3` | games, players, refs/injuries, live rosters, live games | `fetch_data/fetch_nba_data`, `postgre_db/injuries_refs`, `fetch_data/live_rosters.py`, `fetch_data/live_games` | rate-limit blocks (back off ≥90 s); `ScoreboardV2` deprecation |
| data.nba.com legacy schedule | tip-off cross-check | `fetch_data/nba_schedule` | item 2 |
| cdn.nba.com live data | live scoreboard, game time index | `fetch_data/live_games`, `postgre_db/game_time_index` | item 3 |
| official.nba.com/referee-assignments | tonight's referees | `fetch_data/referees/fetch_refs_data.py` | page layout (parsed with `pd.read_html`, needs `lxml`) |
| ak-static.cms.nba.com injury report PDFs | injury reports | `fetch_data/injury_reports` | file naming / timestamps; item 4 |
| sportsbookreview.com (JSON in `__NEXT_DATA__`) | closing odds, line history | `fetch_data/odds_sportsbook` | new or renamed books (a rename creates a new `lh_book` slug); book order (closing table reads only the first 7) |
| sports.yahoo.com (Playwright) | Yahoo odds and betting % | `fetch_data/odds_yahoo` | page changes; Yahoo features are droppable |
| RapidAPI odds (`ODDS_API_KEY`) | odds | `postgre_db/odds/update_odds` | key quota / expiry |
| basketball-reference All-Star voting | all-star features | `postgre_db/all_star_voting/scrape_all_star_voting.py` | hard-coded `NBA_2026_voting.html` / `END_YEAR = 2026`; 2026-27 voting (January 2027) is `NBA_2027_voting.html` |
| github.com/shufinskiy/nba_data | play-by-play archive | `fetch_data/nba_lineups/pbp_archive.py` | when the 2026-27 files appear |
