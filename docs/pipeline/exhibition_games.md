# Preseason and All-Star games in the features

Status: **Partial** (2026-10-25). The code drops preseason (`001`) and
All-Star (`003`) games from every feature input. It changes existing `2_5`
values, so `2_5` is rebuilt in place, as for the bug fixes in
[defect_fixes_2_5.md](defect_fixes_2_5.md). Still to do: the rebuild of both
datasets, the check below and the retrain of every slot (no re-tuning).

## The problem (found 2026-10-25)

`filter_valid_games()` (`data_processing/team/filters.py`) dropped `001` and
`003` from the **team** frame only. The other inputs of the builders never went
through it, and each consumer decided for itself:

| Consumer | Season types it read |
|---|---|
| Referee tendencies (`EXCLUDED_SEASON_TYPES`) | all but `001`/`003` |
| Roster continuity (`_ROSTER_SEASON_TYPES`) | regular, playoffs, play-in |
| Player previous-season fallback (`PLAYER_FALLBACK_SEASON_TYPES`) | regular only |
| Team trend fallback (`TREND_FALLBACK_SEASON_TYPES`) | regular only |
| **Player in-season EWMA** (`precompute_cumulative_avg_stat`) | **everything** |
| Player bench / prior-roster lookups (`attach_player_features.py`) | everything |
| Injury history (`df_injuries`), injury-report history (Aiven) | everything |
| Team `LAST_ALL_5` / `LAST_ALL_10` (`group_by_season=False`) | `002`, `004`, `005`, `006`, across seasons |

Training and prediction computed these the same way, so there was no
train/serve skew and no leakage. The problem was what the features meant.

### Measurements

**Preseason in the player EWMA.** The EWMA groups by `(SEASON_YEAR,
PLAYER_ID)`, and preseason games share the season's year, so they opened every
player's season average (`adjust=True`, 15-game half-life). `df_players` for
2024-25 held 2,762 preseason rows. `PTS_CUM_AVG` on regular-season rows of
2023-24 and 2024-25, with and without `001`/`003`:

| Month | Mean \|diff\| (pts) | Rows with \|diff\| > 1 pt |
|---|---|---|
| Oct | 3.18 | 76% |
| Nov | 1.16 | 41% |
| Dec | 0.48 | 12% |
| Jan–Apr | 0.16–0.28 | 3–5% |

89% of rows differed by more than 0.01, and 15% by more than a point. The
value feeds the TOP-N player columns. A rookie with preseason minutes also had
a value where he would otherwise be NaN.

**All-Star.** Mostly Rising Stars mini-games (games to 40 points, 70–75 total
points), and since 2024-25 the All-Star Game is a mini-game tournament too.
2022-23 to 2024-25: 17 games and 181 players. 3% of regular-season
`PTS_CUM_AVG` values changed, by −0.42 points on average.

**Injury reports.** The Aiven store holds 1,283 preseason status spans
(2019-20 to 2021-22), 593 preseason games and 23 All-Star games.

**Playoffs, play-in and the Cup final** in the player EWMA are negligible.
Within a season, playoffs and play-in only reach later playoff rows, which
training leaves out, and the previous-season fallback reads the regular season
only. The Cup final (December) changes 2.3% of regular-season rows, none by
more than a point.

**Playoffs in the cross-season team windows.** For a playoff team, the first
games of the next season average playoff games: 3.0% of regular-season rows
for `LAST_ALL_5` and 6.0% for `LAST_ALL_10` (2022-23 to 2024-25). Kept on
purpose (below).

## What changed

One definition: `EXHIBITION_SEASON_TYPES = ("Preseason", "All Star")` in
`config/constants.py`, and `drop_exhibition_games()` in `utils/seasons.py`. It
classifies by the game id's prefix (padded to ten digits), never by the
`SEASON_TYPE` text, which labels play-in games "Playoffs".

The filter runs in the loaders the builders read, right after each read, so
closing, intermediate, the `2_6` layer and tonight's prediction get it from the
same code:

- `load_all_nba_data_from_db` (`postgre_db/config/db_loader.py`): games and
  player box scores. Only the builders call it.
- `get_injury_data_from_db`: the injury history.
- `injury_report_aiven/fetch.py`: the history reads behind
  `load_injury_report_state` and the intermediate injury and market-dynamics
  features (`last_status_before_tip`, `filing_at_tip`, `report_age_at_tip`,
  `listed_status_events`, `listed_pairs`, `status_spans`, `filing_spans`). The
  per-snapshot reads ask for given games and are unchanged.
- The two places that load games themselves: the intermediate referee
  builder (`intermediate_referees.py`) and the extra referee-history seasons of
  the closing builder (`_load_referee_history_inputs`). The intermediate one
  mattered: its legacy `REF_*` estimator averaged every crew game it was given,
  and the referee store holds preseason and All-Star crews (157 such games with
  a box score in 2023-24 and 2024-25). The closing estimator reads its own rows,
  which never had them. So the intermediate `REF_*` values move too.
- `season_partitions`: an exhibition is never a game to build or to wait for.

Already restricted, so unchanged: the box history behind the per-horizon
injury features (`load_player_box_history`), the lineup data (regular season
and playoffs), the line history (SBR and Covers store no exhibition), the odds
tables, referee tendencies and roster continuity. `filter_valid_games`
now drops exhibitions before its two-rows-per-game check, so its warning no
longer lists preseason games against non-NBA clubs. Its output is the same.

### Kept as it was

- **Playoffs in the features.** `LAST_ALL_*` windows still cross seasons with
  their playoff games. Whether to skip them is a modelling question for an
  ablation, not a fix.
- **Playoffs out of the training rows.** `training_pipeline` drops them by
  default (`data.exclude_playoffs: true`, the season types in
  `TRAINING_SEASON_TYPES`: regular season and play-in), and so do the daily
  refit and TabPFN (`modeling/refit.resolve_training_frame`). A campaign with
  `exclude_playoffs: false` trains on them.

### Caches

- **The S3 input caches stay valid.** Their key is the season query
  (`season_tables` `code_key`), and the filter runs after the read.
- **The history memo invalidates itself.** Its code key is frozen
  (`frozen-2_5-…`), but each season's digest is taken over the frames as
  loaded: games, players, injuries and report state for the closing builder;
  team games, players and injuries for the intermediate injury memo. Those
  frames lose their exhibition rows, so every digest changes and no old entry
  is read. The memos that do not depend on these inputs (line-history
  snapshots, the odds repair, per-snapshot report reads) stay valid. Do not
  set `NBA_OU_HISTORY_MEMO_REBUILD=1`: it would recompute those too and reuse
  nothing between season rebuilds.

## Rebuild and rollout

Models and features must switch together, or the old models predict on the new
features. In one window, after the day's predictions and before the 09:00
retrain:

1. Pause the prediction triggers and the retrain.
2. Merge the change.
3. `build_season_partitions.py update` for both datasets, then `rebuild` every
   season: closing 2017–2025, intermediate 2019–2025 (about 5–6 hours).
4. `scripts/retrain_prediction_models.py`, then
   `smoke_test_s3_model_load.py --channel staging`, then
   `scripts/promote_build.py --execute`.
5. Resume the triggers.

The full files under `data/train_data/` that campaigns read are not updated by
this; rebuild them for experiments on the new values.

## Check before the rebuild

`scripts/create_train_data/compare_season_partition.py` at the stored cutoff,
for one closing and one intermediate season (2024). Expected:

- player columns differ (TOP-N, `*_CUM_AVG`, availability), mostly in October
  and November;
- team, odds and line-history columns are identical;
- intermediate legacy `REF_*` columns move (preseason crews are gone), closing
  ones do not;
- see whether the row or NaN counts move. A
  rookie without preseason minutes is NaN now, and `max_na_per_row` can drop a
  row it kept before.
