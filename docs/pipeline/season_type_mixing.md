# Non-regular-season games inside history features

Status: **Plan** (found 2026-10-25) — measured, nothing changed. Every fix
below changes existing `2_5` columns, so it needs a new base (`3_0`) or an
in-place `2_5` rebuild, and an ablation before it is adopted.

## How the games get in

`filter_valid_games()` (`data_processing/team/filters.py`) drops Preseason
(`001`) and All-Star (`003`) games from the **team** frame only. The raw frame
loaded in `create_df_to_predict()` also feeds `df_players`, `df_injuries`,
`df_games_for_referee_history` and `df_team_player_context`, which never pass
through it. Each consumer decides for itself:

| Consumer | Season types it reads |
|---|---|
| Referee tendencies (`referee_tendencies.py`, `EXCLUDED_SEASON_TYPES`) | all but `001`/`003` |
| Roster continuity (`roster_continuity.py`, `_ROSTER_SEASON_TYPES`) | regular, playoffs, play-in |
| Player previous-season fallback (`players_statistics.py`, `PLAYER_FALLBACK_SEASON_TYPES`) | regular only |
| Team trend fallback (`team/rolling.py`, `TREND_FALLBACK_SEASON_TYPES`) | regular only |
| **Player in-season EWMA** (`precompute_cumulative_avg_stat`) | **everything**, `001`–`006` |
| Player bench lookup / prior-roster lookup (`attach_player_features.py`) | everything |
| **Team `LAST_ALL_5` / `LAST_ALL_10` windows** (`team/rolling.py`, `group_by_season=False`) | everything left after `filter_valid_games` (`002`, `004`, `005`, `006`), across seasons |

Training and prediction compute these the same way, so there is no
train/serve skew and no leakage. The problem is what the features mean.

## Measurements

### 1. Preseason and All-Star in the player EWMA

`df_players` for 2024-25 holds 2,762 preseason rows and 119 All-Star rows.
The EWMA groups by `(SEASON_YEAR, PLAYER_ID)`, and preseason games share the
season's `SEASON_YEAR`, so they open every player's season average. The EWMA
uses `adjust=True` with a 15-game half-life, so their weight decays slowly.

`PTS_CUM_AVG` on regular-season rows (2023-24 + 2024-25), with versus without
`001`/`003`:

| Month | Mean \|diff\| (pts) | Rows with \|diff\| > 1 pt |
|---|---|---|
| Oct | 3.18 | 76% |
| Nov | 1.16 | 41% |
| Dec | 0.48 | 12% |
| Jan–Apr | 0.16–0.28 | 3–5% |

Overall 89% of rows differ by more than 0.01, and 15% by more than 1 point.
The value feeds the TOP-N player columns. A rookie with preseason minutes also
gets a value where he would otherwise be NaN (`seen_before`).

All-Star on its own (2022-23 to 2024-25). The `003` rows are mostly Rising
Stars mini-games (rookies, sophomores and G League players; games to 40
points, about 70–75 total points), and since 2024-25 the All-Star Game is a
mini-game tournament too. Only 17 games, 307 rows and 181 players are
involved, so the net effect *lowers* averages:

| Measure | Value |
|---|---|
| Regular-season rows whose `PTS_CUM_AVG` changes | 2,915 / 96,237 (3%), 97 players |
| Mean diff on those rows | −0.42 pts (max 2.6); 323 rows > 1 pt |
| Injured-player state whose last row is a `003` game | 141 cases, 15 players, mean −0.07 pts |

Only the player features are affected. The team frame drops `003`, referees
and roster continuity filter it, and `df_injuries` has no `003` rows. Roster
membership for stored games comes from box score plus inactive list, so a
star injured after the break stays on his team even though his timeline holds
an All-Star team id.

### 2. Playoffs, play-in and the Cup final in the player EWMA

These are negligible. Playoffs and play-in come after the regular season, so
within a season they only reach later playoff rows, which training excludes.
The previous-season fallback reads regular season only. The Cup final (`006`,
December) is the exception. Dropping `004`/`005`/`006` changes 2.3% of
regular-season rows, none of them by more than 1 point (2022-23 to 2024-25).

### 3. Playoffs in the team windows that cross seasons

The `LAST_ALL_5` and `LAST_ALL_10` windows group by `TEAM_ID` only. For a team
that made the playoffs, the first games of the next season average playoff
games. Over 2022-23 to 2024-25 (an undercount, since 2021-22 was not loaded):

| Window | Regular-season rows with ≥1 `004`/`005`/`006` game in it |
|---|---|
| Last 5 | 220 / 7,380 (3.0%) |
| Last 10 | 440 / 7,380 (6.0%) |

Almost all of those games are playoff games (`004`). Preseason never enters
these windows.

### 4. Not checked

- `df_injuries` has 182 preseason rows in 2024-25. It is unverified whether
  missed-game streaks or other injury history count them.
- The effect on the bench lookup, the prior-roster lookup and the
  `latest_player_states(injured=True)` "after game" state, which reads a
  player's last row and can be a preseason or playoff game at a season opener.

## Basketball judgement

- **All-Star (`003`):** exhibitions (All-Star and Rising Stars mini-games)
  with no defence and nonstandard formats. Exclude them; there is no argument
  for keeping them. The effect is small (see above).
- **Preseason (`001`) in raw counting averages:** starters play 15–25 minutes,
  camp invitees and end-of-bench players play a lot, and coaches experiment
  with rotations. A raw points average built on that underrates stars and
  overrates fringe players exactly when the in-season sample is smallest.
  Preseason does carry *some* information (new players' roles, per-minute
  rates, pace under a new coach). If it is kept, it belongs in a separate
  per-minute prior or with low weight, not in the regular-season EWMA as
  equal games.
- **Playoffs (`004`) in cross-season team windows:** this case is debatable.
  Playoff games are slower and lower scoring (about 11.6 fewer total points)
  against elite opponents, and only good teams carry them. The window
  therefore drags early-season pace and points down in a way that correlates
  with team quality. The bigger issue is that any window crossing a
  five-month offseason (trades, free agency, coaching changes) is stale,
  playoff games or not. The options are to skip `004` in cross-season
  windows, as the trend fallback already does, or to leave it and let an
  ablation decide.
- **Cup final (`006`):** a competitive game between two regular-season-form
  teams. Keeping it is fine.
- **Play-in (`005`):** already in `TRAINING_SEASON_TYPES`. It only reaches
  next-season windows. Treat it like playoffs.

## Proposed changes (in order)

1. Drop `001` and `003` from `df_players` (and from `df_injuries`, once item
   4 above is checked) right after loading, in `create_df_to_predict()`. Use
   the same filter in the intermediate builder.
2. Optionally, exclude `004`/`005` from the cross-season part of the
   `LAST_ALL_*` team windows.
3. Rebuild and run a fixed-parameter ablation (`experiments` skill) at
   closing and at one intermediate horizon. Expect any effect to concentrate
   in October–November games; evaluate that slice separately, because the
   full-season effect will be within noise.

## Related noise

The `⚠️ Filtering out N games with incomplete team entries` warning in
`filter_valid_games()` lists preseason games against non-NBA clubs
(FC Barcelona, Melbourne United, …) and one NBA API gap (`0011700012`, no
Bulls row). The row-count check runs before the preseason filter, so these
games would be dropped anyway. Moving the season-type filter first would
leave the warning listing only games that matter.
