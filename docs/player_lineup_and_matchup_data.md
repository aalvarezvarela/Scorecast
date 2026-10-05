# Player lineup and matchup data: what we hold

This is an inventory of the **player-interaction data** in the repo: who was on
the floor together, and who guarded whom.

All figures were measured on the local stores on **2026-10-05**. Season labels
follow the repo convention: `season_year` is the **start** year, so `2024` is
2024-25.

---

## 1. Summary

| | Players on court together (**stints**) | Who guarded whom (**matchups**) |
|---|---|---|
| Question it answers | Which five played against which five, for how long, and what happened | For each offensive player, which defenders guarded him and what he did against each |
| Grain | One row per **segment**, the interval between two lineup changes, in one game | One row per **offensive player × defender** in one game |
| Time resolution | Tenth of a second; ordered within the game | **None.** A whole game is one aggregate per pair |
| Seasons | 2016-17 → 2025-26 | 2017-18 → 2025-26 (tracking starts in 2017-18) |
| Volume | ~39k segments and ~13k–20k distinct five-man units per season | ~245k pair rows per season |
| Coverage | 98.9–100% of regular-season + playoff games | 99.4–100% of games, except 2019-20 (91.9%) |
| Location | `data/lineup_stints/season=YYYY/` | `data/matchups/season=YYYY/` |

---

## 2. The three sources

Three NBA Stats endpoints feed this data. Two different providers serve them.

| # | Endpoint | What it contributes | Provider in practice | Seasons |
|---|---|---|---|---|
| 1 | **`GameRotation`** | Each player's on-court intervals (`IN_TIME_REAL` → `OUT_TIME_REAL`). It defines **who is on the floor** at every moment. | `stats.nba.com` API, one paced call per game (≥3 s apart) | 2016-17 → 2025-26, with gaps before 2021-22 (see §3.3) |
| 2 | **Play-by-play**, `PlayByPlayV3` and `PlayByPlayV2` | **V3** holds the events (shots, free throws, rebounds, turnovers, the running score), which are counted into each segment. **V2** names both players in every substitution by ID, which lets a missing rotation be **rebuilt**. | The [`shufinskiy/nba_data`](https://github.com/shufinskiy/nba_data) season archive (it republishes the API payloads), so no per-game API calls are needed | V3: every season. V2: through 2024-25 only |
| 3 | **`BoxScoreMatchupsV3`** | **Who guarded whom**: time, partial possessions and box stats for every offensive player against every defender | `shufinskiy/nba_data` (`matchups_YYYY`, `matchups_po_YYYY`). The API fills the games the archive lacks | 2017-18 → 2025-26 |

**Supporting input:** the box scores (`nba_players` in the DB, or
`data/season_games_data/nba_players_YYYY_YY.csv` before 2018-19, which the DB
no longer holds). They are not a source of interaction data. They are the
**validation reference**: final points and each player's minutes.

How the pieces combine:

```
GameRotation (API) ─────────────┐
   or, when missing/corrupt:    ├─► who is on court ─┐
PlayByPlayV2 rebuild (archive) ─┘                    ├─► stint builder ─► validated against box score ─► data/lineup_stints/
PlayByPlayV3 (archive) ─────────► what happened ─────┘

BoxScoreMatchupsV3 (archive + API fill) ──────────────────────────────────────────────────────────► data/matchups/
```

The two products are **independent**. A matchup row cannot be placed inside a
stint, because matchups carry no timestamps.

---

## 3. Dataset A: five-on-five stints (players playing together)

### 3.1 What one row is

One **segment**: an interval of one game during which neither team changed its
five, cut at every substitution and period boundary. Each row holds both fives
and what both teams did in that interval.

| Column | Meaning |
|---|---|
| `game_id`, `season_year`, `game_date` | Game identity. `game_date` is stored, so no DB is needed to fit on it |
| `seg_idx`, `period` | Order within the game; quarter (5+ = overtime) |
| `start_ds`, `end_ds`, `seconds` | Interval in tenths of a second from tip-off; duration in seconds |
| `home_team_id`, `away_team_id` | Team IDs (stored as strings) |
| `home_lineup`, `away_lineup` | **The five player IDs on court for each side**, as a sorted array of strings |
| `home_pts`, `away_pts` | Points scored by each side in the segment |
| `{home,away}_fga`, `_fg3a`, `_fta`, `_oreb`, `_dreb`, `_tov` | Team counts in the segment |
| `{home,away}_poss` | **Estimated** possessions: `FGA + 0.44·FTA − OREB + TOV` |

Median segment length is about 73 s (mean ~95 s).

**What a stint does *not* say:** which player took the shot or got the
rebound. The counts are per team. The player detail is still in the raw
play-by-play (§5), so it can be added.

### 3.2 Volume and coverage

Coverage = validated games ÷ scheduled regular-season + playoff games.

| Season | Validated games | Coverage | Segments | Distinct five-man units | Players |
|---|---|---|---|---|---|
| 2016-17 | 1,304 | 99.6% | 39,421 | 13,315 | 486 |
| 2017-18 | 1,298 | 98.9% | 38,985 | 14,179 | 541 |
| 2018-19 | 1,311 | 99.9% | 40,277 | 15,203 | 530 |
| 2019-20 | 1,142 | 100% | 36,269 | 14,249 | 530 |
| 2020-21 | 1,159 | 99.5% | 34,466 | 15,318 | 540 |
| 2021-22 | 1,314 | 99.8% | 39,585 | 17,799 | 606 |
| 2022-23 | 1,312 | 99.8% | 39,389 | 17,130 | 541 |
| 2023-24 | 1,310 | 99.8% | 38,759 | 16,218 | 572 |
| 2024-25 | 1,314 | 100% | 39,887 | 18,351 | 569 |
| 2025-26 | 1,303 | 99.1% | 42,115 | 19,872 | 582 |

**Not covered:** preseason (`001`), All-Star (`003`), **Play-In (`005`)** and
the **NBA Cup final (`006`)**. The last two are real competitive games that
the game list skips. The 22 rejected games are listed with a `reason` in
`game_status.parquet` (mostly `points_mismatch`; 7 of the 12 in 2025-26 are
corrupt NBA rotations, `bad_rotation_interval`).

### 3.3 Where each season's lineups come from

`GameRotation` is unreliable before 2021-22. The NBA builds it on demand behind
a ~30 s server cap and caches failures for ~3 h, so many old games never come
back. Those rotations were **rebuilt offline from PlayByPlayV2** instead. Count
of games by rotation source (manifest `source` column; empty = API):

| Season | From `GameRotation` API | Rebuilt from PlayByPlayV2 |
|---|---|---|
| 2016-17 | 0 | 1,304 |
| 2017-18 | 14 | 1,286 |
| 2018-19 | 1,174 | 137 |
| 2019-20 | 1,034 | 108 |
| 2020-21 | 494 | 666 |
| 2021-22 → 2024-25 | 1,306–1,313 per season | 1–9 per season (replacing corrupt API rotations) |
| 2025-26 | 1,315 | 0 |

**How far to trust a rebuilt rotation.** Every rebuilt game had to pass a
check before it was written: five players a side for every second, and every
player within 60 s of his box-score minutes. Against ~8,900 games that also
have an API rotation (2018–2024), 99.3% rebuild and pass, and 99.4–99.9% of
those agree with the API on ≥99% of seconds. Nearly every disagreement is a
corrupt API rotation. Rebuilt rotations carry no per-stint points, plus-minus
or usage, which are null in the rebuilt JSON. The stint builder never reads
those fields anyway.

### 3.4 Validation every stored game passed

A game enters the store only if **all** of these hold. Otherwise it is
recorded `failed` with a reason:

- exactly five players a side at every moment, with no overlapping intervals;
- segment durations sum to the game length;
- summed segment points equal the final score for both teams;
- every player's on-court time is within 60 s of his box-score minutes.

### 3.5 Known limits

- **Possessions are estimated**, not counted. PlayByPlayV3 has no possession
  column, and team rebounds without a player ID are not classified.
- **2025-26 onward depends on the `GameRotation` API.** PlayByPlayV2 is dead for
  new games, both in the API and in the archive. A V3-only rebuild was
  prototyped on 2025-26: 89% of games self-validate, and of those, 1,164 of
  1,172 match the API. It was **never committed** (scratchpad only). So today
  a corrupt or unavailable rotation in a new season has no fallback.
- **Daily refresh is not scheduled.** `update_lineups.py` is a step in
  `.github/workflows/update_finished_matches_daily.yml` (self-hosted runner),
  but that workflow's cron is commented out, so it runs only on manual dispatch.
- **No roster feed.** Who is on a team tonight is not in this data. Box scores
  show only who has already played, so summer departures are invisible at a
  season opener.

---

## 4. Dataset B: who guarded whom (primary defenders)

### 4.1 What one row is

One **offensive player against one defender in one game**, aggregated over
the whole game. Files: `data/matchups/season=YYYY/matchups.parquet` (rows) and
`games.parquet` (per-game status `ok`/`empty` and source `archive`/`api`).

**Direction is fixed by the column prefix:** `off_*` is the player on offence,
`def_*` the defender. The raw archive calls them `person_id` /
`matchups_person_id`. nba_api's label `positionDef` is misleading: it is the
**offensive** player's position. The direction was verified against 2020-21 box
scores. Summed per offensive player, points correlate 0.97 and assists 0.96
with his own box score.

| Column group | Columns | Populated? |
|---|---|---|
| Identity | `game_id`, `season_year`, `season_type` (Regular Season / Playoffs), `home_team_id`, `away_team_id` | yes |
| Offensive player | `off_team_id`, `off_player_id`, `off_first_name`, `off_family_name`, `off_name_i`, `off_position` | yes |
| Defender | `def_team_id`, `def_player_id`, `def_first_name`, `def_family_name`, `def_name_i` | yes |
| Exposure | `matchup_seconds`, `partial_possessions` | yes |
| Shares of time | `pct_defender_total_time`, `pct_offensive_total_time`, `pct_total_time_both_on` | yes |
| Offensive output against this defender | `player_points`, `team_points`, `matchup_fgm/fga`, `matchup_fg3m/fg3a`, `matchup_ftm/fta`, `matchup_assists`, `matchup_turnovers` | yes |
| Defensive events | `matchup_blocks`, `shooting_fouls` | yes (sparse: ~3% and ~8% of rows non-zero) |
| **Switches / help defence** | `switches_on`, `help_blocks`, `help_fgm`, `help_fga` | **NO: zero in every row of every season** |
| Potential assists | `matchup_potential_assists` | **NO: zero everywhere** |
| Provenance | `source` (`archive` or `api`) | yes |

> **The switch and help columns are empty at the source, not lost in import.**
> A live `BoxScoreMatchupsV3` call on 2026-10-05 for `0022400196` and
> `0022000451` returned 0 for all five columns while `shootingFouls` and
> `matchupBlocks` were populated.

### 4.2 Volume and coverage

| Season | Games held | Coverage | Pair rows |
|---|---|---|---|
| 2017-18 | 1,304 | 99.4% | 246,443 |
| 2018-19 | 1,305 | 99.5% | 247,601 |
| 2019-20 | 1,050 | **91.9%** | 195,911 |
| 2020-21 | 1,165 | 100% | 217,255 |
| 2021-22 | 1,316 | 99.9% | 245,213 |
| 2022-23 | 1,314 | 100% | 243,002 |
| 2023-24 | 1,309 | 99.8% | 244,303 |
| 2024-25 | 1,313 | 99.9% | 246,961 |
| 2025-26 | 1,315 | 100% | 256,748 |

All held games currently come from the archive. About 112 games are owed, 88
of them the **2020 bubble seeding games**, which the archive lacks but the API
serves. `import_matchups.py --fill-missing --no-import` would fetch them.
**There is no daily matchup update**: nothing in the workflows refreshes this
store for new games.

### 4.3 "Primary defender" is something you derive

The data lists **every** defender a player faced, not one assigned defender.
A "primary defender" has to be defined, typically as the defender with the most
`matchup_seconds` or `partial_possessions` against that player in that game.
How dominant that defender is (2024-25, offensive player-games with ≥10 matched
minutes, n = 12,116):

- median **9 different defenders** per offensive player per game;
- the top defender accounts for a median **36%** of matched time
  (interquartile range 29–45%).

So the primary defender guards a player only about a third of the time, and
the remaining exposure is distributed across the other defenders.

### 4.4 Scale checks and quirks

- `partial_possessions` summed per team-game is ≈ 490, i.e. ≈ 5 × the team's
  possessions. Each possession is split across the five offensive players.
- `player_points` summed per team-game runs slightly **above** the actual team
  score: median +4, 5th–95th percentile −8 to +16 (2024-25). Use it for
  relative comparisons, not as a points ledger.
- **Revisions.** The NBA reprocesses tracking after the fact. 2020-21 and
  2024-25 archive games match today's API exactly. The 2025-26 archive was
  captured in-season and differs by a median 1.9 s per pair (totals within
  ~1%). A game fetched the morning after is preliminary in the same way.
- The matchups and stints can disagree on whether a game exists (each store has
  its own small set of missing games). Join on `game_id` and expect a few
  one-sided misses.

---

## 5. Raw archive: detail omitted from the processed stores

`data/nba_api_raw/{gamerotation,playbyplayv3}/{season_year}/{game_id}.json.gz`
holds the original responses (gzip), with a per-season resume manifest under
`data/nba_api_raw/manifest/`. It is the source of truth. Re-parsing it costs no
API calls. The archive contains things the stints do not keep:

| In the raw archive | Kept in stints? | Additional detail |
|---|---|---|
| `personId` on every play-by-play event | no (team totals only) | Player ID associated with each event |
| `xLegacy`, `yLegacy`, `shotDistance` | no | Shot coordinates and distance |
| `PLAYER_PTS`, `PT_DIFF`, `USG_PCT` per rotation interval | no; API rotations only, null when rebuilt | Points, point differential and usage percentage |
| Fouls, violations, timeouts, jump balls | no | Event-level stoppages and violations |

## 6. Reading the data

```python
from nba_ou.data_processing.lineups.stint_store import read_stints, read_game_statuses
from nba_ou.fetch_data.nba_lineups.matchups import MatchupStore

stints = read_stints([2023, 2024])            # validated games only, sorted by date
status = read_game_statuses([2024])           # ok / failed + reason per game
matchups = MatchupStore().read(2024)          # off_* x def_* rows for 2024-25
held = MatchupStore().statuses(2024)          # per-game ok / empty, archive / api
```

Rebuilding and refreshing (commands, flags, run lock, S3 mirroring):
`scripts/lineups/README.md`.
