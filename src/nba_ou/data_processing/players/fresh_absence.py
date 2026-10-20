"""Fresh-absence features: a key player's *first* game out.

The pipeline already emits, separately, how good each injured player is
(``TOP1_INJURED_PLAYER_PTS_BEFORE``) and how long they have been out
(``TOP1_INJURED_STREAK_PTS_BEFORE``). This module crosses the two, because the
interaction -- not either factor alone -- is what the market appears to misprice.

Measured on ``training_data_2_3_20260909`` (regular season 2019-2025, bet365
closing totals, 8,232 games), splitting each game into how far the closing total
moved from the two teams' recent baseline and how far actual scoring moved:

    a key player's FIRST game out   line -1.47   scoring -0.13   -> 53.2% OVER
    the same player's SECOND game   line -1.15   scoring -0.98   -> 48.4% OVER

On the first game out the market cuts the total by about a point and a half
while scoring barely falls: usage is redistributed and the pace does not drop.
By the second game the adjustment is right. The effect held in 6 of 7 seasons
and is stronger for the away team (54.8% OVER).

**This is a hypothesis carried into the feature set, not a settled result.**
The 95% interval on that 53.2% is [50.6, 55.7] and does not clear the 52.38%
break-even; an explicit-feature walk-forward moved the model the right way
(+0.7pp on the affected games) with an interval spanning zero. It is here so a
proper campaign can measure it; see ``experiments/``.

Temporal correctness: every input is a ``_BEFORE`` column. The streak counts
team games up to and including the current one, which is knowable pre-tip -- a
player is on the inactive list before the game starts. The caveat is the one
already documented for availability data in general: history is built from the
settled post-game inactive list while production reads the pre-game report, so
late scratches and questionable-then-rested players are labelled differently
than they would have been at bet time. Both of those mislabellings make the
market look like it *under*-reacted, so they cannot manufacture this effect.
"""

import pandas as pd

from nba_ou.data_processing.past_injuries.past_injuries import N_TOP_PLAYERS_INJURED
from nba_ou.utils.general_utils import _with_before_suffix

#: Statistics whose top-N injured list gets a consecutive-games-missed streak.
#: PTS ranks a player by scoring, MIN by how much of the game they are on the
#: floor -- a different player on most rosters, and a different kind of absence
#: (a high-minute low-scoring starter still changes the possession count).
STREAK_STAT_COLS = ("PTS", "MIN")

#: "Key player" cut-offs, per statistic. Chosen by sweeping thresholds on the
#: measurement above: PTS 14/16/18/20/22/24 peaked at 18 (53.2% OVER, 6/7
#: seasons above 50%) and MIN 26/28/30/32/34 peaked at 30 (53.3%, 6/7 seasons).
#: They are cut-offs on a noisy curve, not sharp boundaries -- neighbouring
#: values behave similarly, which is the reason the continuous
#: ``INJ_TOP1_FIRST_GAME_OUT_*`` columns are emitted alongside the flags.
KEY_PLAYER_THRESHOLDS = {"PTS": 18.0, "MIN": 30.0}

#: A streak of 1 is the player's first game out: the re-pricing the market has
#: to do tonight. 2 is the follow-up game, kept as its own flag because that is
#: where the measurement says the market has already corrected -- so the model
#: can tell the two apart instead of reading "recently out" as one state.
FIRST_GAME_OUT_STREAK = 1
SECOND_GAME_OUT_STREAK = 2


def fresh_absence_feature_names(stat_col: str) -> list[str]:
    """Team-level ``_BEFORE`` column names emitted for ``stat_col``."""
    return [
        _with_before_suffix(f"INJ_FRESH_OUT_{stat_col}"),
        _with_before_suffix(f"INJ_TOP1_FIRST_GAME_OUT_{stat_col}"),
        _with_before_suffix(f"INJ_KEY_PLAYER_FIRST_GAME_OUT_{stat_col}"),
        _with_before_suffix(f"INJ_KEY_PLAYER_SECOND_GAME_OUT_{stat_col}"),
    ]


def add_fresh_absence_features(df_team: pd.DataFrame) -> pd.DataFrame:
    """Add first-game-out interaction features to a **team-level** frame.

    Emitted per statistic in :data:`STREAK_STAT_COLS`:

    - ``INJ_FRESH_OUT_<stat>_BEFORE`` -- summed stat of every injured player in
      the top-N list whose first game out this is. The size of tonight's news.
    - ``INJ_TOP1_FIRST_GAME_OUT_<stat>_BEFORE`` -- the leading injured player's
      stat when tonight is their first game out, else 0. The continuous form of
      the flag below, so the model is not forced through one cut-off.
    - ``INJ_KEY_PLAYER_FIRST_GAME_OUT_<stat>_BEFORE`` -- 1 when that leading
      player clears :data:`KEY_PLAYER_THRESHOLDS`.
    - ``INJ_KEY_PLAYER_SECOND_GAME_OUT_<stat>_BEFORE`` -- the same for their
      second game out.

    A missing streak or stat means "no such injured player", which is an
    absence of news rather than an unknown, so 0 is the correct fill -- the same
    reasoning the availability-effect aggregates use. Columns are created even
    when the source columns are absent (an early pipeline stage, or a stat whose
    streaks are not computed), so downstream schema checks see a stable set.
    """
    out = {}
    for stat_col in STREAK_STAT_COLS:
        (fresh_sum_col, top1_col, first_flag_col, second_flag_col) = (
            fresh_absence_feature_names(stat_col)
        )

        fresh_sum = pd.Series(0.0, index=df_team.index)
        for i in range(1, N_TOP_PLAYERS_INJURED + 1):
            value_col = _with_before_suffix(f"TOP{i}_INJURED_PLAYER_{stat_col}")
            streak_col = _with_before_suffix(f"TOP{i}_INJURED_STREAK_{stat_col}")
            if value_col not in df_team.columns or streak_col not in df_team.columns:
                continue
            value = pd.to_numeric(df_team[value_col], errors="coerce").fillna(0.0)
            streak = pd.to_numeric(df_team[streak_col], errors="coerce").fillna(0)
            is_first_game_out = streak == FIRST_GAME_OUT_STREAK
            fresh_sum = fresh_sum + value.where(is_first_game_out, 0.0)
            if i == 1:
                threshold = KEY_PLAYER_THRESHOLDS[stat_col]
                out[top1_col] = value.where(is_first_game_out, 0.0)
                out[first_flag_col] = (is_first_game_out & (value >= threshold)).astype(
                    int
                )
                out[second_flag_col] = (
                    (streak == SECOND_GAME_OUT_STREAK) & (value >= threshold)
                ).astype(int)

        out[fresh_sum_col] = fresh_sum
        for col in (top1_col, first_flag_col, second_flag_col):
            if col not in out:
                out[col] = pd.Series(0, index=df_team.index)

    existing = [col for col in out if col in df_team.columns]
    if existing:
        df_team = df_team.drop(columns=existing)
    return pd.concat([df_team, pd.DataFrame(out, index=df_team.index)], axis=1)
