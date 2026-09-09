# Training Data Processing

This document describes how this project builds the training and prediction feature
set for NBA over/under models. It is written as an implementation reference for
future agents, so it focuses on the actual pipeline in the codebase rather than
only the modeling intent.

The main entry point is
`src/nba_ou/create_training_data/create_df_to_predict.py`, specifically
`create_df_to_predict()`. Despite the function name, it is used for both
historical training data and same-day prediction data.

## Entry Points

Historical training data is usually created by calling:

```python
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict

df_train = create_df_to_predict(
    todays_prediction=False,
    recent_limit_to_include="2026-03-04",
    older_season_limit=None,
)
```

Common scripts that call this path:

- `scripts/create_train_data/create_train_data.py`
- `scripts/create_train_data/create_and_upload_historical_train_data.py`
- `scripts/retrain_prediction_models.py`
- The `__main__` block in `create_df_to_predict.py`

Same-day prediction data is created by first collecting scheduled-game context
with `get_all_info_for_scheduled_games()`, then passing that dictionary into
`create_df_to_predict(todays_prediction=True, scheduled_data=...)`.

```python
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict

scheduled_data = get_all_info_for_scheduled_games(
    date_to_predict="2026-04-12",
    nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
)

df_to_predict = create_df_to_predict(
    todays_prediction=True,
    scheduled_data=scheduled_data,
    strict_mode=30,
    normalize_total_lines=True,
)
```

## High-Level Pipeline

The pipeline builds one final row per NBA game. Most intermediate processing is
team-level, with two rows per game, then the home and away rows are merged into
one game-level row.

At a high level, `create_df_to_predict()` does this:

1. Determine the historical cutoff date.
2. Determine which seasons to load.
3. Optionally collect extra historical matchup games for same-day prediction.
4. Load team game logs and player boxscores from PostgreSQL.
5. Load and merge Yahoo plus Sportsbook Review odds.
6. Clean and enrich team-level rows.
7. Load injury data and attach player/injury plus roster-continuity features.
8. Merge home and away team rows into one game row.
9. Merge remaining game-level odds.
10. Add all-star voting, team identity, betting, market-regime, referee,
    odds-engineered, injury-effect, travel, and date features.
11. Select final training columns and return a DataFrame.

## Date And Season Selection

The cutoff is controlled by `recent_limit_to_include`.

- If `todays_prediction=False` and no cutoff is provided, the cutoff defaults to
  yesterday in US/Pacific time.
- If `todays_prediction=True` and no cutoff is provided, the cutoff is inferred
  as the day immediately before the scheduled games. This prevents current-day
  games from entering historical rolling statistics.

Season selection is based on the cutoff date:

- With `older_season_limit=None`, historical training defaults to all seasons
  from the 2017-18 season onward.
- With `older_season_limit=N`, the pipeline includes the current season and the
  previous `N - 1` seasons.
- For same-day prediction, the default is two seasons unless
  `older_season_limit` is provided.

All date-based season utilities use the same boundary: January through July
belong to the season that started in the previous calendar year, while August
begins the next season bucket. Therefore, a July 23, 2026 cutoff belongs to
`2025-26`; requesting two seasons produces exactly `["2024-25", "2025-26"]`
and cannot introduce the future `2026-27` bucket. The delayed 2020-21 season is
the explicit exception: dates through October 2020 remain assigned to
`2019-20`.

The team rows returned by the pipeline still follow that requested range, but
player and injury loading includes one additional earlier season. That extra
season is context-only: it supplies the March-15 roster baseline and prior-year
minute averages for continuity in the earliest output season. It does not add
games to the returned training/prediction DataFrame.

Filtering is performed by
`filter_by_seasons_with_extra_game_ids()`, which keeps rows from the selected
season years and applies the date cutoff. Extra game IDs can be preserved even
when they fall outside the ordinary season filter, as long as they respect the
upper date cap.

## Historical Inputs

The historical path loads these main data sources:

- Team game logs from `load_all_nba_data_from_db()`.
- Player boxscores from `load_all_nba_data_from_db()`.
- Yahoo odds from `load_odds_yahoo_from_db()`.
- Sportsbook Review odds from `load_odds_sportsbook_from_db()`.
- Injury rows from `get_injury_data_from_db()`.
- Referee rows from `get_refs_data_from_db()`.
- All-Star voting rows from `load_all_star_voting_from_db()`.

The active sportsbook is configured through `nba_ou.config.odds_columns`.
`get_main_book()` returns `SETTINGS.main_sportsbook` when configured, otherwise
it falls back to `consensus_opener`. The configured book controls canonical
column names such as:

- `ODDS_TOTAL_LINE_<book>`
- `ODDS_SPREAD_<book>`
- `ODDS_MONEYLINE_<book>`

`fanatics_sportsbook` only exists from the 2025 season, and the odds-fetching
pipeline no longer scrapes the now-discontinued Caesars, so left alone
fanatics_sportsbook is a de facto season indicator. `create_df_to_predict()`
and `create_intermediate_line_df()` both take `exclude_caesars`,
`exclude_fanatics`, and `combine_fanatics_and_caesars` to reconcile this -- see
`nba_ou.data_processing.odds.book_combination` (wide, per-book columns) and
`nba_ou.data_processing.line_history.book_merge` (tidy Aiven ticks).

Combining is the **default**: it folds Caesars into fanatics_sportsbook
(fanatics values kept where present, Caesars fills the gap), giving one
continuously-covered book instead of two disjoint, season-correlated ones. It
is the only option that fixes the season-leak without discarding a book's data.

`combine_fanatics_and_caesars` is tri-state so that "default on" cannot turn an
explicit exclusion into an error (`resolve_combine_books`):

- `None` (default) -- combine, unless `exclude_caesars` or `exclude_fanatics`
  was explicitly asked for, in which case the exclusion wins.
- `True` -- combine, and reject a simultaneous exclusion as the genuine
  contradiction it is: there is no standalone book left to exclude.
- `False` -- leave both books as they are.

The intermediate CLI follows the same default: `--exclude-fanatics` and
`--exclude-caesars` are escape hatches, and running the script with no flags
produces the merged book.

## Same-Day Prediction Inputs

Same-day prediction requires a `scheduled_data` dictionary created by
`get_all_info_for_scheduled_games()`. It contains:

- `scheduled_games`: games from the NBA schedule endpoint.
- `df_referees_scheduled`: processed scheduled referee assignments.
- `injury_dict_scheduled`: game/team/player injury lookup for scheduled games.
- `df_odds_yahoo_scheduled`: Yahoo odds for the scheduled games.
- `df_odds_sportsbook_scheduled`: Sportsbook Review odds for the scheduled games.
- `games_not_updated`: scheduled games whose injury report status was not usable.

Scheduled team rows are appended to the historical team table by
`standardize_and_merge_scheduled_games_to_team_data()`. Scheduled player
placeholder rows are appended by
`standardize_and_merge_scheduled_games_to_players_data()` so roster and player
cumulative features can be computed consistently for future games.

The scheduled odds data is merged with historical odds by
`merge_and_validate_scheduled_odds()`. That function checks that scheduled odds
columns are compatible with historical odds columns, drops extra scheduled-only
columns, logs null-count diagnostics when `strict_mode >= 0`, then concatenates
historical and scheduled odds.

For scheduled games, the pipeline may also include extra historical game IDs for
the exact home/away matchups being predicted. These are discovered through
`get_historical_game_ids_for_home_away_matchups()` and help matchup-specific
features have more history.

## Team-Level Processing

Team-level processing is handled by `process_team_statistics_for_training()`.
At this point the data has one row per team per game.

The major steps are:

1. `clean_team_data()`
   - Converts `GAME_DATE` to datetime.
   - Drops rows with missing points.
   - Drops duplicate `(GAME_ID, TEAM_ID)` rows.
   - Removes zero-minute rows and rows with implausibly low points.
   - Converts `TEAM_ID` to string.
   - Attempts to fix home/away parsing issues from `MATCHUP`.

2. `adjust_overtime()`
   - Creates `IS_OVERTIME`.
   - Preserves raw `PTS`, so `TOTAL_POINTS` remains the actual final score,
     including overtime.
   - Creates `PTS_PER_40` as `PTS * 200 / MIN` only when
     `0 < MIN < 240`. For every other minutes value—including missing, zero,
     negative, non-finite, regulation, and overtime minutes—it keeps raw `PTS`
     and performs no division.
   - For overtime games only, overwrites additive counting statistics with their
     48-minute regulation equivalents using `value * 240 / MIN`.
   - The normalized allowlist is `FGM`, `FGA`, `FG3M`, `FG3A`, `FTM`, `FTA`,
     `OREB`, `DREB`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `PLUS_MINUS`,
     and `POSS`.
   - Does not normalize `MIN`, identifiers, metadata, percentages, ratios,
     ratings, pace fields, `PIE`, or the raw `PTS` target source.
   - Keeps normalized counts as fractional estimates rather than rounding them.

3. `merge_total_spread_moneyline_by_game_id()`
   - Merges the selected book's spread and moneyline by `GAME_ID`.
   - Merges total lines for all known total sources when `total_lines_mode="all"`.
   - Creates canonical columns such as `ODDS_TOTAL_LINE_betmgm`,
     `ODDS_TOTAL_LINE_bet365`, `ODDS_SPREAD_<book>`, and `ODDS_MONEYLINE_<book>`.
   - Assigns spread and moneyline from the current team's perspective.

4. `compute_total_points_features()`
   - Creates `TOTAL_POINTS` as home plus away points, repeated on both team rows.
   - For each `ODDS_TOTAL_LINE_*` column, creates `DIFF_FROM_LINE_<book>` as
     actual total points minus that line.

5. `filter_valid_games()`
   - Keeps only games with exactly two team entries.
   - Classifies season type.
   - Drops preseason and All-Star games.
   - Adds integer `SEASON_YEAR`.

6. `add_overtime_history_features()`
   - Adds the previous completed game's overtime indicator for each team.
   - Adds the fraction of a team's previous five completed games that went to
     overtime.
   - Adds the fraction of a team's prior completed games in the current
     `SEASON_YEAR` that went to overtime.
   - Excludes the current game from all three calculations. Scheduled rows with
     no result consume the latest history but do not enter it.

7. `add_last_season_playoff_games()`
   - Counts each team's playoff games from the previous season.

8. `add_team_record_before_game()`
   - Adds `GAME_NUMBER`, `WINS_BEFORE_THIS_GAME`, and
     `TEAM_RECORD_BEFORE_GAME`.

9. `compute_rest_days_before_match()`
   - Adds `REST_DAYS_BEFORE_MATCH` within team and season.
   - Uses seven days for the first game of each team-season, where no
     within-season previous game exists.

10. `merge_odds_percentages_and_prices_by_game_id()`
   - Merges public betting percentages, consensus percentages, and price columns
     before rolling features are computed.
   - Total-market prices are game-level.
   - Spread and moneyline prices are converted to the current team's side.

11. `compute_all_rolling_statistics()`
    - Adds the bulk of team-form, betting-form, trend, weighted-average, and
      season-to-date features.

## Rolling And Trend Features

Rolling features are implemented in `src/nba_ou/data_processing/team/rolling.py`
and `src/nba_ou/data_processing/statistics/statistics.py`.

Most rolling features are explicitly shifted by one game, so they represent
information available before the current game.

Overtime history is also strictly pre-game and creates these team-level source
features:

- `IS_OVERTIME_LAST_GAME_BEFORE`: `1` when the team's latest completed game
  went to overtime, otherwise `0`.
- `OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE`: overtime games divided by the
  number of available completed games among the previous five.
- `OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE`: overtime games divided by completed
  games already played by that team in the current `SEASON_YEAR`; this is `0`
  for the team's first game of the season.

Last-game and last-five history continue across season boundaries. Only the
season-year frequency resets. A scheduled game with no `IS_OVERTIME` result is
not counted, so adding multiple future games cannot dilute or replace the most
recent completed-game history. After the home/away merge, each feature receives
the corresponding `_TEAM_HOME` or `_TEAM_AWAY` suffix.

Core team stats rolled over recent games include:

- Points and total points.
- `PTS_PER_40`, using only shifted historical values. It receives the full
  `PTS`-style rolling family: last-five all-games and same-location relative
  averages, short 1/2/3/10 windows, 5-game weighted moving average,
  season-before average and standard deviation, and 5- and 10-game trend
  slopes (with the 5-minus-10 relative slope).
- Offensive, defensive, and net rating.
- Effective field goal percentage and true shooting percentage.
- Pace, possessions, PIE.
- Field goals, threes, free throws, and personal fouls.

Betting-related rolling inputs include:

- Main and per-book total lines.
- Spread and moneyline columns.
- `DIFF_FROM_*` columns.
- Public betting percentages from Yahoo.
- Consensus percentages.
- Price columns for spreads and moneylines.

Raw total-market over/under prices remain available as current-game features,
but are intentionally excluded from rolling windows and season averages. This
avoids creating historical price derivatives such as last-five and
season-before price features.

Feature patterns include:

- `*_LAST_ALL_5_MATCHES_BEFORE`: previous five games, any location.
- `*_LAST_HOME_AWAY_5_MATCHES_BEFORE`: same-location rolling average minus the
  all-games rolling average.
- `*_LAST_ALL_10_MATCHES_BEFORE`: previous ten games for selected features.
- `*_LAST_5_MINUS_LAST_10_MATCHES_BEFORE`: short-window minus longer-window form.
- `*_LAST_1_MATCHES_BEFORE`, `*_LAST_2_MATCHES_BEFORE`,
  `*_LAST_3_MATCHES_BEFORE`: short windows for line-difference features.
- `*_LAST_5_WMA_BEFORE`: weighted moving average with more weight on recent games.
- `*_SEASON_BEFORE_AVG`: season-to-date average before the current game, with a
  previous-season fallback.
- `*_SEASON_BEFORE_STD`: expanding season-to-date standard deviation before the
  current game, with a previous-season fallback.
- `*_TREND_SLOPE_LAST_5_GAMES_BEFORE`: linear trend slope over prior games.
- `*_TREND_SLOPE_LAST_10_GAMES_BEFORE` and
  `*_TREND_SLOPE_LAST_5_MINUS_LAST_10_GAMES_BEFORE`: trend comparison features.

`compute_all_rolling_statistics()` dynamically discovers new `ODDS_TOTAL_LINE_*`,
`DIFF_FROM_*`, percentage, consensus, and eligible spread/moneyline price
columns, so adding a new book can automatically expand the rolling feature set
when the column naming convention matches the existing patterns. Total-market
over/under prices are the deliberate exception. Trend slopes for total lines and
`DIFF_FROM_*` inputs are calculated only from the original source columns;
rolling averages, weighted averages, and season statistics are never fed back
into trend calculation.

`PTS_PER_40` is treated as a first-class `PTS`-like source: it receives the
full rolling family listed above. Its raw same-game home/away columns are
excluded by the final feature-selection safety rules; only `_BEFORE`
derivatives can reach the model.

### Style-matchup features

`add_team_style_source_features()` derives attempt and possession rates from
completed team box scores already present in the database:

- Three-point attempts and free-throw attempts per field-goal attempt.
- Turnovers and offensive rebounds per possession.
- Field-goal attempts per possession, used to convert attempt rates into
  expected counts.
- The corresponding opponent values observed in each game, representing what
  that team's defense allowed or forced.

These same-game values are history sources, not prediction features. The
rolling stage creates a season-to-date-before-game estimate for each source. A
previous-five-games value is used internally as an early-season fallback and
then removed. It deliberately does not create style-specific home/away deltas,
weighted averages, standard deviations, or trends.

After home and away rows are merged, `add_style_matchup_features()` combines
each team's season-before offensive tendency with the opponent's season-before
allowed tendency. It creates expected home and away rates for threes, free
throws, turnovers, and offensive rebounds; game-level expected counts scaled
by `EXPECTED_POSS_FROM_PACE_BEFORE`; and a free-throw-rate interaction with
`REF_AVG_TOTAL_PF_DIFF_BEFORE`.

The home/away historical source columns are removed after those interactions
are built. Only the 13 compact matchup features remain in the final model
table, rather than exposing the 20 intermediate home/away source columns as
additional predictors.

All model-facing style columns end in `_BEFORE`. Every rolling calculation uses
`shift(1)`, the referee calculation uses games with dates strictly earlier than
the game being scored, and scheduled rows retain missing current-game box-score
sources. Therefore neither a historical game's result nor a scheduled game's
unknown statistics can enter that game's style-matchup features.

## Player And Injury Features

Player processing is handled by `process_player_statistics_for_training()` and
`add_player_history_features()`.

The player table is first cleaned by `clear_player_statistics()`:

- Game dates and season fields are merged from the team table.
- `MIN` is parsed from `MM:SS` into decimal minutes.
- Duplicate rows are dropped.
- Rows are filtered to the selected seasons and cutoff date.

For scheduled prediction, synthetic player rows are appended from each player's
latest known row. This lets the feature code identify likely roster membership
for a future game without using future boxscore data.

The pipeline computes player features for these stats:

- `PTS`
- `PACE_PER40`
- `DEF_RATING`
- `OFF_RATING`
- `TS_PCT`
- `MIN`

For each stat, `precompute_cumulative_avg_stat()` creates a recency-weighted
player estimate using an EWMA with a 10-game halflife. It is shifted so the
current game's stat is excluded. The lookup is grouped by `SEASON_YEAR` and
`PLAYER_ID`.

For each team-game, the pipeline identifies active and injured players using:

- Historical injury database rows.
- Player boxscore comments containing injury wording.
- Scheduled injury-report data for same-day predictions.
- Roster membership inferred from each player's last team before the game date.

Historical box-score rows establish roster membership only for later games, not
for the game recorded by that row. This matches same-day prediction, where the
current game's box score does not exist yet. A current injury report is valid
pregame assignment evidence and overrides the player's previously known team,
including on their first game after a trade. Scheduled player placeholders are
also safe same-day evidence because they are generated from prior player history
and contain no current-game statistics. Every numeric player feature remains
shifted, so current-game box-score values are excluded from those calculations.
No target-game box-score quantity takes part in the roster split at all.

Availability is the roster minus the injured/inactive set, and that set is built
from reasons rather than from minutes. Two sources feed it: the inactive-players
table published with the box score, and the box-score `COMMENT` field whenever it
names an absence reason that the pre-game NBA injury report would also have
carried. `UNAVAILABLE_COMMENT_PATTERN` holds that list: Injury/Illness, Rest,
Personal Reasons, suspensions, Trade Pending, and G League assignments.

`DNP - Coach's Decision` is deliberately excluded and leaves the player
available. This is the selection rule for the whole family: a category counts
only if the same absence would have been visible on the pre-game report, because
anything else teaches the model to react to information that is not available at
prediction time.

Target-game `MIN` is not consulted. Minutes cannot distinguish a late scratch
from a coach's decision, so a minutes rule buys label coverage at the cost of
making roster membership a function of the game being predicted -- the failure
recorded in `nba_ou.config.leakage` and re-checked by
`test_target_game_minutes_cannot_move_a_player_between_buckets`.

> The remaining gap is the source, not the rule: training reads the box score
> while production reads the pre-game injury report PDF. Archiving those reports
> is what closes it, and it is the prerequisite for using `Questionable` and
> `Doubtful` statuses rather than the current `Out`/`Doubtful` set.

Feature families added at the team row level include:

- Top six active-player IDs, names, and prior averages by stat.
- Top four injured-player IDs, names, and prior averages by stat.
- Average top-injured-player value by stat.
- Total injured-player prior value by stat.
- Number of injured players.
- Injury streak features for injured players by points.
- Bench scoring and pace features from players averaging roughly 7 to 21 minutes.

After home/away merging, player columns are suffixed by side, for example:

- `TOP1_PLAYER_PTS_BEFORE_TEAM_HOME`
- `TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY`
- `TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME`
- `N_INJURED_PLAYERS_BEFORE_TEAM_AWAY`

> **Removed from newly generated data: `N_ACTIVE_PLAYERS_*` and
> `TOTAL_NON_INJURED_PLAYER_*`.** In legacy datasets both
> aggregated over the *non-injured* player set, which
> `get_top_n_averages_with_names` resolves as `df[df["GAME_DATE"] == date]` for
> a game already played — that is, the players who logged minutes in the game
> being predicted. The count therefore tracked rotation depth: it correlates
> **+0.55 with `|HOME_MARGIN|`** (bench-emptying in blowouts) and runs from 9.7
> for games decided by ≤5 points to 12.3 for games decided by >30. A
> `spread_error` regressor given these 14 columns scored 66.7% against the
> closing spread on a 609-game holdout; without them, 52.4%. The per-player
> *values* were correctly lagged — only the membership of the set leaked, which
> is why the `_BEFORE_` naming looked right. `TOTAL_INJURED_PLAYER_*` and
> `N_INJURED_PLAYERS_*` are unaffected: they resolve each player's last game
> *strictly before* this one. The individual top-player and bench families now
> use roster-minus-injured membership and are safe. Old CSVs stay usable —
> `clean_dataframe_for_training` drops the legacy aggregate columns if it finds
> them (see `nba_ou.config.leakage`).

Each top-N statistic also produces an id and a name column
(`TOP1_PLAYER_ID_PTS_BEFORE`, `TOP1_PLAYER_NAME_PTS_BEFORE`). **These are
bookkeeping, not features.** They exist so
`add_top3_availability_effect_features_for_columns()` can locate each team's
leading players; an id is meaningless as a numeric input (player 1610612747 is
not "greater than" player 201939) and a name is a string. Both are removed at the
end of the pipeline by `drop_player_identifier_columns()`, which must run *after*
the availability-effect features have consumed the id columns.

### Season Openers

Three subsystems used to reset at the season boundary and leave a team's opening
games empty. All three now follow the same fallback chain — **current season →
previous REGULAR season → a defined no-history value** — restricted to regular
season because that is the regime these models target and because only 16 teams
play a postseason, which would otherwise make the carried-over value mean
something different per team:

- `compute_trend_slope()` (`team/rolling.py`) grouped by `(TEAM_ID, SEASON_YEAR)`
  with `min_periods=2`, leaving every slope column empty for the first two games
  of a season. It now falls back to the trend the team ended its previous regular
  season with, then to `0` (which is also what `calculate_slope` returns for
  insufficient data).
- `precompute_cumulative_avg_stat()` (`players/players_statistics.py`) grouped its
  EWMA by `(SEASON_YEAR, PLAYER_ID)`. It now seeds from the player's previous
  regular season average, then `0`.
- `create_player_lookup()` (`past_injuries/past_injuries.py`) resolves a roster by
  asking who last played for the team *earlier in the same season*, so it returned
  nobody at an opener and every top-N player column stayed missing.
  `add_player_history_features()` now falls back to the game's own roster in that
  case — no new lookahead, since the primary path already selects on
  `GAME_DATE == date`, and every value attached is a prior-season average.

The availability-effect aggregates are the one place a plain `0` is correct:
`_shrink_effect` computes `raw * n/(n+k)`, whose value at `n=0` is `0`, so a row
with no evidence has a fully shrunk estimate of zero rather than an unknown. The
fill is applied to the aggregates only, never to per-player values, so rows where
some players do have evidence keep the mean over those players.

Filling the *level* features (ratings, TS%, pace, minutes, counts) with `0` would
be wrong, not merely lossy: `OFF_RATING` runs 87–165 and `TS_PCT` 0.38–1.33 in
real data, so `0` is an extreme that never occurs and a tree would read a season
opener as the worst offense on record. That is why they get a previous-season
value instead.

### Minutes-Weighted Roster Continuity

`add_roster_continuity_feature()` adds two team-level continuity horizons before
the home/away merge. Both measure lost roster value in minutes rather than
counting players, because losing one 35-minute starter should matter more than
losing a low-minute end-of-bench player.

For a game in season `S`, the roster-membership window is:

- March 15 during the final months of season `S - 1` through the target game.
- Regular-season and postseason games only; preseason and All-Star assignments
  are ignored.
- Both player rows and injury-report rows count as team assignments, so injured
  and active players are treated identically for roster membership.

Every player assigned to the target team during that window is a candidate. A
candidate is considered lost only if their latest known assignment before the
game is to a different NBA team. If no later assignment exists, the player is
not assumed to have left; this avoids treating an unobserved status as a trade.
A player who moved away and later returned is retained because the latest
assignment wins.

Every observed roster candidate receives a target-team-specific minute weight:

1. Total positive `MIN` played for the target team in season `S`, strictly
   before the target game, divided by the number of target-team games in that
   period.
2. If the player has no target-team appearance in `S`, use the equivalent
   target-team contribution from season `S - 1`.
3. If neither season has target-team minute history, the weight is zero.

This prevents minutes played after moving to another team from inflating the
value of the departure. Dividing by all target-team games also gives short-term
players their true season contribution instead of treating their per-appearance
average as if they filled that role for every game.

Continuity is the retained share of the complete observed candidate pool:

```text
candidate_minutes = sum(target-team minutes per team game for all candidates)
lost_minutes = sum(candidate minute weights for players now assigned elsewhere)
ROSTER_MINUTES_CONTINUITY_PCT_BEFORE =
    1 - lost_minutes / candidate_minutes
```

For example, if the observed roster candidate pool represents 240 normalized
team minutes and departed players account for 110, continuity remains
`1 - 110 / 240 = 0.5417`, or 54.17%. Unlike the former fixed-denominator
calculation, sequential short stays cannot push the lost total past an unrelated
240-minute denominator and produce an artificial zero.

Leakage handling is deliberately stricter for historical boxscores than pregame
sources. A historical boxscore assignment becomes available on the following
calendar day, so the current game's player rows and minutes cannot affect their
own feature. Current-game injury reports are allowed because they are pregame
information. Scheduled synthetic player rows are also available on game day
when both `MIN` is missing and their `GAME_ID` belongs to the explicit scheduled
game set. This distinction is necessary because the scheduled injury dictionary
contains only OUT players; a healthy player on a new team must be represented by
the active-roster placeholder. Future trades cannot alter earlier historical
rows.

Allowed competition types come from the shared `SEASON_TYPE_MAP`, not a local
`SEASON_ID` prefix rule. Regular Season, Playoffs, and Play-In Tournament are
included. The repository maps `005` to Play-In Tournament and `006` to In-Season
Final Game; the latter remains excluded from this roster calculation.

If no late-previous-season membership exists in the loaded data for a team, the
value is `NaN` rather than an artificial 100%. After the home/away merge, the
two model parameters are:

- `ROSTER_MINUTES_CONTINUITY_PCT_BEFORE_TEAM_HOME`
- `ROSTER_MINUTES_CONTINUITY_PCT_BEFORE_TEAM_AWAY`

The `_BEFORE` tag ensures both columns survive final leakage-safe column
selection.

The immediate-trade horizon applies the identical membership, minute weighting,
and leakage rules with a shorter start date:

- Normally, the window begins two calendar months before the game.
- If that calculated start lands during the June-through-September offseason,
  the start is moved back to March 1 of that calendar year. For example, an
  October 22 game would ordinarily start on August 22, so it instead uses
  March 1 and can observe the roster that existed before summer transactions.
- Unlike the season-continuity horizon, it does not require a prior-season
  observation when the ordinary two-month window lies fully within the current
  season. It returns `NaN` only when the team has no known assignment in the
  effective window.

After home/away merging, the two additional immediate-trade parameters are:

- `ROSTER_MINUTES_CONTINUITY_2M_PCT_BEFORE_TEAM_HOME`
- `ROSTER_MINUTES_CONTINUITY_2M_PCT_BEFORE_TEAM_AWAY`

### Minutes Brought In By New Players

The same season and immediate windows also measure incoming roster value. A
player is newly incorporated when their latest known assignment is the target
team and their preceding distinct assignment inside the effective window was a
different team. Repeated appearances for the new team do not count as multiple
incorporations; the most recent different team is the previous team.

For multiple observed moves such as `A -> B -> C`, A and B both treat the
player as lost because the latest assignment is C. Team C treats the player as
incoming from B and therefore uses only the player's B-minute average. A team
that never appears in a boxscore or injury assignment cannot be inferred as an
intermediate stop because this feature does not use a transaction feed.

Each new player is weighted only by minutes played for that previous team:

1. Mean positive `MIN` for the previous team in season `S`, before the game.
2. If unavailable, mean positive `MIN` for that same team in season `S - 1`.
3. If neither exists, zero.

```text
new_player_minutes = sum(previous-team average minutes of incoming players)
ROSTER_NEW_PLAYER_MINUTES_PCT_BEFORE = clip(new_player_minutes / 240, 0, 1)
```

Unlike continuity, this is not subtracted from one: zero means no observed
incoming minute value, while a larger value means the team recently added more
previous-team playing time. The final four columns are:

- `ROSTER_NEW_PLAYER_MINUTES_PCT_BEFORE_TEAM_HOME`
- `ROSTER_NEW_PLAYER_MINUTES_PCT_BEFORE_TEAM_AWAY`
- `ROSTER_NEW_PLAYER_MINUTES_2M_PCT_BEFORE_TEAM_HOME`
- `ROSTER_NEW_PLAYER_MINUTES_2M_PCT_BEFORE_TEAM_AWAY`

### Net Roster Minutes

The loss and incoming measures are also combined into a directional net value.
Lost share is `1 - continuity`, so the equivalent formulas are:

```text
ROSTER_NET_MINUTES_PCT_BEFORE =
    ROSTER_NEW_PLAYER_MINUTES_PCT_BEFORE
    + ROSTER_MINUTES_CONTINUITY_PCT_BEFORE
    - 1

ROSTER_NET_MINUTES_2M_PCT_BEFORE =
    ROSTER_NEW_PLAYER_MINUTES_2M_PCT_BEFORE
    + ROSTER_MINUTES_CONTINUITY_2M_PCT_BEFORE
    - 1
```

A positive value means the team brought in more previous-team minutes than it
lost, zero means the minute shares balance, and a negative value means it lost
more than it brought in. This is deliberately a difference rather than a sum:
a sum would measure total roster churn but would not preserve its direction.
If continuity is unavailable because there is no valid prior-season baseline,
the corresponding net value is also `NaN`.

After the home/away merge, the four final columns are:

- `ROSTER_NET_MINUTES_PCT_BEFORE_TEAM_HOME`
- `ROSTER_NET_MINUTES_PCT_BEFORE_TEAM_AWAY`
- `ROSTER_NET_MINUTES_2M_PCT_BEFORE_TEAM_HOME`
- `ROSTER_NET_MINUTES_2M_PCT_BEFORE_TEAM_AWAY`

## All-Star Voting Features

The pipeline adds all-star fan-vote features before the home/away merge. The
pipeline computes required All-Star voting season years from team `GAME_DATE`
values with `all_star_season_year_for_game_date()`, then loads those years from
PostgreSQL through `load_all_star_voting_from_db()`.

The season-year mapping is intentionally calendar-sensitive:

- Games before March 1 use the All-Star voting season from two NBA season starts
  earlier.
- Games on or after March 1 use the prior NBA season start year.

For example, February 20, 2026 maps to All-Star `season_year=2024`, while
March 2, 2026 maps to `season_year=2025`.

`add_all_star_voting_features()` combines:

- Team-game rows.
- Cleaned player rows, including roster movement inferred from the latest known
  team on or before the game date.
- All-Star voting rows with `season_year`, `player_id`, `team_name`,
  `fan_votes`, and optional `score`.
- The same injury dictionary used by player features.

The function validates that every required voting season has usable rows. If a
required season is missing or has zero total fan votes, the pipeline raises a
`ValueError` and asks for the Basketball Reference voting data to be scraped,
rebuilt, and uploaded.

All-Star votes follow the player when they change teams. The old team stops
receiving the player's votes and the new team starts receiving them on the
first game date where the new assignment is known. Same-game player rows are
used only for `PLAYER_ID`, `TEAM_ID`, and `GAME_DATE`; minutes, points, and
other game outcomes are not used. A current pregame injury-report assignment
can establish the new team before the player records a box score there.

Team-level all-star columns initially include:

- `ALL_STAR_FAN_VOTE_SHARE_BEFORE`
- `ALL_STAR_MIN_SCORE_BEFORE`
- `ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE`
- `ALL_STAR_MIN_INJURED_SCORE_BEFORE`
- `ALL_STAR_FAN_VOTES_BEFORE`
- `ALL_STAR_CANDIDATE_COUNT_BEFORE`
- `ALL_STAR_SEASON_YEAR_BEFORE`

After `merge_home_away_data()`, `create_df_to_predict()` drops the audit/count
columns and keeps only the side-specific predictive columns:

- `ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME`
- `ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY`
- `ALL_STAR_MIN_SCORE_BEFORE_TEAM_HOME`
- `ALL_STAR_MIN_SCORE_BEFORE_TEAM_AWAY`
- `ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_HOME`
- `ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY`
- `ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_HOME`
- `ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_AWAY`

The all-star vote share denominator is league-wide total fan votes for the
selected All-Star voting season. Traded players are handled by checking the
player's last known team before the game date and by also adding current roster
players who appear in that season's All-Star voting rows.

## Home/Away Merge

`merge_home_away_data()` converts the two team rows per game into one row per
game.

It separates `HOME == True` and `HOME == False`, then merges on game-level keys:

- `SEASON_ID`
- `GAME_ID`
- `GAME_DATE`
- `SEASON_TYPE`
- `SEASON_YEAR`
- `IS_OVERTIME`
- `GAME_TIME` for same-day prediction, when available

Home-side columns get `_TEAM_HOME`. Away-side columns get `_TEAM_AWAY`.

During this stage, the pipeline also:

- Renames top-player and average-injured columns with `_BEFORE`.
- Creates star contribution features, such as
  `STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE` and
  `STAR_PTS_PERCENTAGE_BEFORE`.
- Deduplicates game-level `ODDS_TOTAL_LINE_*` columns that were duplicated by
  the home/away merge.
- Creates game-level `TOTAL_POINTS` and `TOTAL_PF`.
- Adds `IS_PLAYOFF_GAME_BEFORE`.
- Computes home/away points-conceded features and historical matchup features.
- Creates `TEAMS_DIFFERENCE_OVER_UNDER_LINE_BEFORE` when both side-specific
  season average total-line columns are available.

Immediately after the home/away merge, `create_df_to_predict()` calls
`add_team_one_hot_features()`. By default it adds 60 binary identity features:

- `TEAM_HOME_<TEAM_SLUG>_BEFORE`
- `TEAM_AWAY_<TEAM_SLUG>_BEFORE`

There is one home and one away column for each team in `TEAM_ID_MAP`.

If `create_df_to_predict(categorical_team_encoding=True)` is used, the pipeline
adds two pandas categorical columns instead:

- `TEAM_HOME_CATEGORY_BEFORE`
- `TEAM_AWAY_CATEGORY_BEFORE`

This mode is intended for models that can consume native categorical features,
such as XGBoost with categorical handling enabled.

## Game-Level Odds Features

Odds are used in several ways:

1. Canonical line, spread, and moneyline columns are merged before rolling stats.
2. Percentages and prices are merged before rolling stats so team-level market
   behavior can be rolled.
3. Remaining raw game-level odds are merged after the home/away merge.
4. `engineer_odds_features()` creates robust market summaries.

The raw Yahoo and Sportsbook Review data is merged in
`merge_yahoo_sportsbook_odds()`:

- Yahoo and Sportsbook Review are outer-joined by `game_id`.
- Missing BetMGM total, spread, and moneyline values are filled from Yahoo when
  available.
- American odds prices are converted to decimal odds.
- By default, asymmetrically priced total markets are converted in place to an
  estimated 50/50 line and decimal `-110/-110` prices. No columns are added.
- Yahoo public betting percentages are retained when available.

Total-line normalization is controlled by the `normalize_total_lines` argument
of `create_df_to_predict()` and defaults to `True` for both historical data and
today's scheduled games. The conversion removes the two-way vig, assumes NBA
total points have a standard deviation of `15.7`, and rounds the estimated fair
center to the nearest `0.5`. For example, a `209.5` line priced at `1.62/2.20`
becomes `212.5` priced at `1.91/1.91`. Total prices are rounded to two decimal
places before comparison so insignificant price differences are ignored. Quotes
where either side is already priced at `-110` (`1.91` decimal) are treated as
main-market-like and are not normalized. Quotes that are already symmetric,
incomplete, invalid, or whose over and under sides use different lines remain
unchanged. Set `normalize_total_lines=False` to preserve the source values; the
live prediction script exposes the same choice as
`--no-normalize-total-lines`.

`engineer_odds_features()` creates `ODDS_*` features, including:

- Per-book total line midpoints.
- Cross-book total line mean, median, standard deviation, range, IQR, and MAD.
- Total-price log ratios: over price divided by under price.
- No-vig total probability differences.
- Total-market vig by book and aggregate vig summaries.
- Counts and ratios of books with total lines and total prices present.
- Spread home-line summaries, favorite absolute spread, and home-favorite flags.
- No-vig moneyline home probabilities and moneyline vig.
- Favorite win probability and moneyline probability gap.
- Interactions such as total line times probability skew, total line times
  favorite spread, and line disagreement times vig.
- Implied home and away points from total and spread.
- `ODDS_close_total_consensus`, based on cross-book close total median or mean.

## Betting Difference Features

`add_betting_stats_differences()` creates home-minus-away differences for
team-specific betting and market-form columns.

It looks for home/away pairs that are both:

- Betting-related, such as total lines, spreads, moneylines, percentages, prices,
  consensus values, `DIFF_FROM_*`, and `TOTAL_POINTS`.
- Rolling or derived, such as last-game windows, weighted averages,
  season-before averages/stds, and trend slopes.

The resulting columns generally end in `_DIFF_BEFORE`. These features directly
encode relative home-vs-away market context.

## Global Market Regime Features

`add_global_market_features()` adds league-wide, game-date-level features. These
features are computed from games strictly before the current calendar date, so
same-day games cannot influence each other.

The function resolves the active close total line via
`resolve_main_total_line_col()` and compares it with actual `TOTAL_POINTS`.

Feature families include:

- Rolling global market bias: actual total minus close total.
- Rolling global MAE.
- Rolling global market error standard deviation.
- Median error and median absolute error.
- Tail miss rates for errors above 10, 15, and 20 points.
- Over, under, and push rates.
- League-wide close-total averages.
- League-wide actual-total averages.
- Actual-minus-close average gaps.
- Short-window versus long-window regime ratios.
- Bias, MAE, close-total, and actual-total acceleration features.
- League activity counts over recent days and games.
- Open-to-close move features when a consensus opener exists.
- Whether the close line has recently beaten the open line.
- Cross-book total-line disagreement for the current game and rolling league
  disagreement.

Column names are suffixed with `_BEFORE`, for example:

- `GLOBAL_MARKET_BIAS_30G_BEFORE`
- `GLOBAL_MARKET_MAE_7D_BEFORE`
- `GLOBAL_MARKET_OVER_RATE_75G_BEFORE`
- `GLOBAL_CLOSE_TOTAL_AVG_DIFF_15G_75G_BEFORE`
- `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE`

## Referee Features

Referee processing is handled by
`add_referee_features_to_training_data()`.

Historical referee rows are transformed into one row per game with deterministic
`REF_1`, `REF_2`, and `REF_3` slots. Names are canonicalized and sorted so crew
order does not affect the features.

For each current game's referees, `compute_referee_features()` compares games
with a given referee to games without that referee, using same-season history
when possible and previous-season history as fallback.

Metrics used:

- `TOTAL_POINTS`
- `DIFF_FROM_LINE`
- `TOTAL_PF`

Features include:

- `REF_AVG_<METRIC>_DIFF_BEFORE`
- `REF_STD_<METRIC>_DIFF_BEFORE`
- `REF_SUM_<METRIC>_DIFF_BEFORE`

Exact trio features also exist in the implementation, but
`create_df_to_predict()` currently calls referee processing with
`include_ref_trio_features=False`.

For same-day prediction, scheduled referee assignments are appended before
feature computation. If a scheduled referee has no historical match in the
database, the pipeline raises an error rather than silently producing unknown
referee features.

## Availability Effect Features

After the initial player and injury features are created,
`add_top3_availability_effect_features_for_columns()` estimates how important
specific player availability has historically been for a team.

It is called twice:

1. For top active/home-away player columns, producing
   `TOP3_AVAILABILITY_EFFECT_*`.
2. For top injured/home-away player columns, producing
   `TOP3_INJURED_AVAILABILITY_EFFECT_*`.

For each player, team, and game date, the function looks at the current and
previous season for that team before the game date. Only games where the local
roster split classified that player as available or injured are included. This
excludes games before the player joined the team or after they left it.

It compares:

- Team `TOTAL_POINTS` when the player was present versus injured.
- Team `DIFF_FROM_LINE` when the player was present versus injured.
- Team-oriented `SPREAD_ERROR` when the player was present versus injured. For
  both home and away teams, a positive value means the team beat the Bet365
  spread by that many points.

Each effect is `mean(available) - mean(injured)`, so a positive spread effect
means the team historically performed better with the player available.

Team win rate was previously a fourth effect. It was dropped: a binary outcome
over a handful of games is the noisiest of the four and measures nearly what
`SPREAD_ERROR` measures on a continuous scale.

Raw effects are shrunk toward zero by an empirical-Bayes weight that is fitted
rather than assumed:

```text
effect_shrunk = effect_raw * tau2 / (tau2 + se2)
```

The observed spread of player effects is real player-to-player variation plus
sampling noise, `var(effect) = tau2 + mean(se2)`, so `tau2 = var(effect) -
mean(se2)` is the signal variance and `tau2 / (tau2 + se2)` is the posterior
weight on one player's estimate. This is the same shape as `n_eff / (n_eff + k)`
-- `se2` falls like `1/n` -- but the constant is measured instead of guessed, it
is measured separately per metric, and it uses each player's own precision
rather than their game count alone. A player with one game on either side has no
standard error, so `se2` is imputed as `sigma2 * (1/n_present + 1/n_injured)`
from the pooled within-team variance.

Both variances are estimated only from games strictly earlier than the row being
shrunk. A single global fit would let a March game set the shrinkage applied to
an October row.

`shrinkage_k` survives as the fallback prior for rows too early in the history to
fit anything (fewer than `MIN_SHRINKAGE_FIT_SAMPLES` prior observations), and
`shrinkage_k=0` still disables shrinkage entirely. `fit_shrinkage=False` forces
the old fixed-`k` behaviour for comparison.

Each call prints the fitted `tau2`, `sigma2` and the implied `k = sigma2 / tau2`
per metric. A `tau2` of zero is a real answer, not a failure: it says the spread
between players is fully explained by sampling noise, and the honest estimate of
every effect for that metric is then zero. Watch that line -- it is the fastest
way to see whether a metric carries any player-level signal at all.

Every effect carries a Welch standard error, `sqrt(var_present/n_present +
var_injured/n_injured)`, which needs two valid games in each group and is `NaN`
otherwise. The group-level value is the mean of the per-player standard errors,
and stays `NaN` when no selected player has evidence: no evidence means unbounded
uncertainty, so filling it with zero would assert the opposite. Only the effect
aggregates are filled with zero, where that is the correct limit of a shrinkage
estimator.

The standard error replaced a Welch/Fisher p-value and its Bonferroni-corrected
group summary. A p-value is a monotone function of the effect size and the two
group sizes, all already emitted, so it added little the model could not
reconstruct -- while implying a significance test that a handful of games per
player cannot support, across a family of comparisons that invites false
positives. The standard error keeps magnitude and precision on separate axes and
leaves the trust rule to the model.

Aggregate outputs include:

- Home and away mean effects on total points.
- Home and away mean effects on difference from line.
- Home and away mean effects on spread error.
- Home and away mean standard errors for all three effects.
- Home and away max absolute effects.
- Home and away total historical sample sizes.

This compact default creates 18 columns per call (36 across active and injured
players). The redundant injured/present count split, player counts, and boolean
flags are omitted. They remain available for diagnostics through
`include_detailed_sample_size_features=True`, but the training and prediction
pipeline explicitly keeps the compact schema.

## Travel And Schedule Features

`compute_travel_features()` converts the game-level table back into a team-game
travel log and uses city coordinates to estimate travel distance.

For each team, it identifies the city where the current game is played and the
city of the previous game. It then computes great-circle distance with the
Haversine formula. Two consecutive home games are treated as zero travel.

Rolling travel sums are computed over calendar windows ending at the current
game:

- 1 day
- 2 days
- 5 days
- 7 days
- 14 days

The trip from the previous game location to the current game location is
included because it has already occurred before tipoff and is therefore valid
pregame information. Trips exactly on the left edge of each window are also
included.

The final game-level columns are:

- `TOTAL_KM_IN_LAST_<N>_DAYS_HOME_TEAM`
- `TOTAL_KM_IN_LAST_<N>_DAYS_AWAY_TEAM`

`create_df_to_predict()` calls this with `log_scale=True`, so the final values
are `log1p(kilometers)`.

`add_high_value_features_for_team_points()` later derives fatigue and rest
compression features such as:

- `TRAVEL_RECENCY_RATIO_HOME_2D_OVER_14D_BEFORE`
- `TRAVEL_RECENCY_RATIO_AWAY_2D_OVER_14D_BEFORE`
- `REST_DAYS_DIFF_HOME_MINUS_AWAY_BEFORE`

## Date And Calendar Features

`add_game_date_features()` adds simple calendar indicators:

- `IS_WEEKEND_BEFORE`
- `MONTH_BEFORE`
- `IS_US_HOLIDAY_BEFORE`

These are derived from `GAME_DATE`.

## Matchup And Team Context Features

The merged feature set includes several team and matchup context families:

- Team identity columns for home and away teams.
- Conference and division features:
  - `SAME_CONFERENCE_BEFORE`
  - `SAME_DIVISION_BEFORE`
  - `IS_HOME_WEST_CONFERENCE_BEFORE`
  - `IS_AWAY_WEST_CONFERENCE_BEFORE`
- Playoff context:
  - `IS_PLAYOFF_GAME_BEFORE`
  - `PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME`
  - `PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY`
- Team record and rest:
  - `WINS_BEFORE_THIS_GAME`
  - `TEAM_RECORD_BEFORE_GAME`
  - `REST_DAYS_BEFORE_MATCH`
- Historical matchup features from
  `get_last_5_matchup_excluding_current()`.
- Points-conceded features from historical games, including differences between
  expected scoring and home/away conceded averages.

## Derived High-Value Features

`add_derived_features_after_computed_stats()` and
`add_high_value_features_for_team_points()` create compact interaction features
from the larger feature set.

Examples include:

- `TOTAL_PTS_SEASON_AVG_BEFORE`
- `TOTAL_PTS_LAST_GAMES_AVG_BEFORE`
- `BACK_TO_BACK_BEFORE`
- `DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE`
- `DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE`
- `IMPLIED_PTS_HOME_BEFORE`
- `IMPLIED_PTS_AWAY_BEFORE`
- `EXPECTED_POSS_FROM_PACE_BEFORE`
- `EXPECTED_PTS_HOME_FROM_OFFR_PACE_BEFORE`
- `EXPECTED_PTS_AWAY_FROM_OFFR_PACE_BEFORE`
- `OFFDEF_MISMATCH_HOME_OFF_MINUS_AWAY_DEF_BEFORE`
- `OFFDEF_MISMATCH_AWAY_OFF_MINUS_HOME_DEF_BEFORE`
- `PTS_FORM_Z_HOME_LAST5_VS_SEASON_BEFORE`
- `PTS_FORM_Z_AWAY_LAST5_VS_SEASON_BEFORE`
- `PTS_TREND_SLOPE_DIFF_HOME_MINUS_AWAY_BEFORE`
- `PTS_TREND_SLOPE_SUM_HOME_PLUS_AWAY_BEFORE`
- `INJURY_PTS_SHARE_HOME_BEFORE`
- `INJURY_PTS_SHARE_AWAY_BEFORE`
- `STAR_PTS_PCT_DIFF_HOME_MINUS_AWAY_BEFORE`
- `POSS_X_TSPCT_HOME_BEFORE`
- `POSS_X_TSPCT_AWAY_BEFORE`

These features are intentionally numeric and are designed to be useful for tree
models such as XGBoost.

The trend difference compares the direction and strength of the teams' recent
scoring trends. The trend sum captures whether their combined scoring form is
rising or falling. Both use the shifted
`PTS_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME/AWAY` inputs, so the current
game's points are not involved.

## Final Column Selection

Final column selection is handled by `select_training_columns()`.

The selection rules intentionally favor features that are known before the game:

- Team metadata columns with `_TEAM_HOME` and `_TEAM_AWAY` suffixes are kept.
- Static game columns are kept.
- Any column containing `BEFORE` is kept.
- Team identity columns are kept because both the default one-hot columns and
  optional categorical columns include `_BEFORE`.
- All-star voting columns that survive the post-merge auxiliary-column drop are
  kept because they include `_BEFORE`.
- Known odds columns are kept.
- Columns beginning with `ODDS_` are kept (the unified odds-derived marker;
  see Naming Conventions below).
- Columns beginning with `ODDS_TOTAL_LINE_` are kept.
- `GAME_TIME` is kept only for same-day prediction mode.
- `TOTAL_POINTS` is kept as the target when present.

The function drops explicitly forbidden columns:

- `DIFFERENCE_FROM_LINE`
- `DIFF_FROM_LINE`
- `TOTAL_PF`
- `IS_OVER_LINE`

It also drops columns containing `DIFF_FROM` when they do not contain
`_BEFORE`, to avoid direct target leakage from same-game outcomes.

As a final safety check, if an original raw team column appears in the final
training DataFrame without `_BEFORE` and is not explicitly allowed, the function
raises a `ValueError`.

## Target

The primary target for total-points regression is:

- `TOTAL_POINTS`

It is the actual combined score for the game. The codebase also creates many
features based on differences from betting lines, but current final selection is
careful to retain only prior-game or otherwise pre-game versions of those values.
Overtime points remain part of this target. `TOTAL_POINTS` is always calculated
from the untouched raw `PTS` columns, never from `PTS_PER_40` or another
overtime-normalized statistic.

For same-day prediction rows, `TOTAL_POINTS` may be missing or unavailable for
future games. Downstream prediction code should treat it as absent target data
and use the feature columns only.

## Naming Conventions

Important conventions for future agents:

- `_BEFORE` means the feature is intended to use only information available
  before the game.
- `_TEAM_HOME` and `_TEAM_AWAY` identify side-specific columns after home/away
  merging.
- `_DIFF_BEFORE` usually means home-side feature minus away-side feature.
- `TEAM_HOME_<SLUG>_BEFORE` and `TEAM_AWAY_<SLUG>_BEFORE` are one-hot team
  identity features.
- `TEAM_HOME_CATEGORY_BEFORE` and `TEAM_AWAY_CATEGORY_BEFORE` are optional
  categorical team identity features.
- `ALL_STAR_*_BEFORE_TEAM_HOME` and `ALL_STAR_*_BEFORE_TEAM_AWAY` are
  side-specific All-Star fan-vote and score features.
- `ODDS_` is the unified marker for every odds-derived column (mirroring
  `_BEFORE` for leakage safety) so odds features can be selected as a group,
  e.g. `[c for c in df.columns if is_odds_column(c)]`
  (`nba_ou.config.odds_columns.is_odds_column`). It sits in front of any other
  odds-specific tag, e.g. `ODDS_CLOSING_TOTAL_LINE_<book>` in the
  intermediate-line dataset.
- **This is an enforced invariant, not a convention.** Each pipeline entry
  point ends with `apply_odds_prefix()` followed by
  `assert_odds_columns_prefixed()`, which raises if any column named like a
  bookmaker market reached the output without the marker. The recognised
  shapes are `ODDS_SHAPED_PREFIXES` in `nba_ou/config/odds_columns.py`:
  `TOTAL_LINE_`, `SPREAD_`, `MONEYLINE_`, `total_`, `spread_`, `ml_`,
  `moneyline_`. A new odds feature named in any of those shapes fails the
  build rather than silently disappearing from every odds-based selection.
  `DIFF_FROM_` and `IS_OVER_` are deliberately outside the guard — they are
  named after the target relationship rather than the market, and prefixing
  them would be a separate rename.
- **Where the marker is applied matters.** It goes on *last*, after
  `engineer_odds_features()` and `add_high_value_features_for_team_points()`,
  because both resolve their inputs by raw market name
  (`total_<book>_price_over`, `spread_consensus_opener_line_home`). Prefixing
  earlier hides those inputs and silently drops every vig / no-vig /
  price-dispersion feature. Do not move the rename into
  `select_training_columns`.
- `ODDS_TOTAL_LINE_<book>` is the canonical total line for a book.
- `ODDS_SPREAD_<book>` and `ODDS_MONEYLINE_<book>` are canonical selected-book
  team-side columns before the home/away merge.
- Raw game-level odds columns use lowercase prefixes like `total_`, `spread_`,
  and `ml_` throughout the pipeline; the final pass prepends `ODDS_`.
- Odds-engineered features use the `ODDS_` prefix (see
  `engineer_odds_features(prefix=...)`).

When adding new features, prefer using `_BEFORE` for any historical, rolling,
estimated, or pre-game-derived column that should be eligible for training
selection.

## Leakage Controls

The pipeline uses several controls to avoid target leakage:

- Rolling team features use `shift(1)` to exclude the current game.
- Player EWMA features use shifted prior appearances.
- Referee features use only games before the current game date.
- Global market features aggregate games from strictly earlier calendar dates.
- Injury availability effects use only games before the current game date.
- Same-day prediction uses a cutoff date before the scheduled games.
- Final selection drops raw `DIFF_FROM_*` columns unless they carry `_BEFORE`.
- Raw original boxscore columns are blocked from final selection unless they are
  static, explicitly allowed, or transformed into `_BEFORE` features.

## Adding Or Modifying Features

Use these guidelines when extending the training data:

1. Decide whether the feature is team-level or game-level.
   - Team-level features should usually be added before `merge_home_away_data()`.
   - Game-level features should usually be added after `merge_home_away_data()`.

2. Preserve pre-game semantics.
   - Rolling, cumulative, and historical aggregations should exclude the current
     game.
   - For same-day prediction, make sure scheduled rows do not use future
     boxscore results.

3. Follow naming conventions.
   - Add `_BEFORE` to any feature that should be selected automatically.
   - Use `_TEAM_HOME` and `_TEAM_AWAY` only after the home/away merge.
   - Use `ODDS_` for features created by odds engineering.

4. Check final selection.
   - If the feature is not selected, verify whether it contains `BEFORE`, starts
     with `ODDS_`, or is in the explicit odds/static allow lists.

5. Be careful with new sportsbook columns.
   - If the raw column follows existing naming conventions, many rolling and odds
     engineering functions can discover it dynamically.
   - If a book has a new naming pattern, update the relevant inference logic in
     odds merging, rolling stats, and odds engineering.

6. Validate historical and scheduled modes.
   - Historical training can pass even when scheduled mode breaks because
     scheduled mode needs compatible odds columns, scheduled referee matches, and
     injury-report mappings.

7. Validate all-star voting coverage.
   - The pipeline treats All-Star voting as a required source. If a new date range
     maps to a voting season that is not in the All-Star voting database, the
     training-data build will fail before the home/away merge.

8. Choose team encoding intentionally.
   - Default one-hot encoding is broad model compatible but adds 60 columns.
   - Categorical encoding is more compact, but downstream model training must
     preserve pandas categorical dtypes and enable native categorical support.

## Output Shape

The final DataFrame returned by `create_df_to_predict()` is game-level:

- One row per game.
- Side-specific home and away feature columns.
- Static game identifiers and dates.
- Pre-game features, mostly selected by `_BEFORE`.
- Team identity features, as one-hot columns by default or as categorical
  columns when requested.
- Side-specific All-Star fan-vote and score features.
- Raw and engineered odds features.
- `TOTAL_POINTS` target for completed historical games.

The exact number of columns changes as books, source schemas, and dynamic
feature discovery change.
