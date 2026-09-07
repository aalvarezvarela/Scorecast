import numpy as np
import pandas as pd

from nba_ou.config.constants import SEASON_TYPE_MAP

EWMA_HALFLIFE_GAMES = 10

#: Season types the previous-season player fallback is allowed to read.
#: Regular season only, matching the team-level trend fallback in
#: ``data_processing.team.rolling``: this model targets regular-season games, and
#: only 16 teams play a postseason, so sourcing the carried-over average from
#: whatever a player happened to play last would make it mean something different
#: depending on how far their team went.
PLAYER_FALLBACK_SEASON_TYPES: frozenset[str] = frozenset({"Regular Season"})


def _is_fallback_season_type(game_id) -> bool:
    """Season type from the game id, using the repository's canonical mapping."""
    if pd.isna(game_id):
        return False
    return (
        SEASON_TYPE_MAP.get(str(game_id).zfill(10)[:3], "Unknown")
        in PLAYER_FALLBACK_SEASON_TYPES
    )


def _previous_regular_season_player_average(out, stat_col, valid_mask):
    """Each player's average ``stat_col`` in their PREVIOUS regular season.

    Mirrors the ``_SEASON_BEFORE_AVG`` fallback in
    ``statistics.compute_rolling_stats`` and the trend-slope fallback in
    ``team.rolling``: a player-season that has no history yet inherits what the
    player actually did last year rather than starting from nothing.

    Returns a Series aligned to ``out``'s row order (positional), so a duplicated
    index cannot misalign the fill.
    """
    if "GAME_ID" not in out.columns:
        # Nothing to derive the season type from; the caller falls through to the
        # documented "no prior valid game" value instead of guessing.
        return pd.Series(np.nan, index=out.index)

    regular = valid_mask & out["GAME_ID"].map(_is_fallback_season_type)
    season_means = (
        out.loc[regular]
        .assign(_stat=pd.to_numeric(out.loc[regular, stat_col], errors="coerce"))
        .groupby(["PLAYER_ID", "SEASON_YEAR"], as_index=False)["_stat"]
        .mean()
    )
    if season_means.empty:
        return pd.Series(np.nan, index=out.index)

    # Carry each season's average into the season that follows it.
    season_means["SEASON_YEAR"] = season_means["SEASON_YEAR"] + 1
    season_means = season_means.rename(columns={"_stat": "_previous_season_stat"})

    merged = out[["PLAYER_ID", "SEASON_YEAR"]].merge(
        season_means, on=["PLAYER_ID", "SEASON_YEAR"], how="left"
    )
    return pd.Series(merged["_previous_season_stat"].to_numpy(), index=out.index)


def get_top_n_averages_with_names(
    df, date, stat_col="PTS", injured=False, lowest=False, n_players=3, min_minutes=15
):
    """
    Returns a list of tuples (Player Name, <CUM_AVG>) for the top n (or bottom n) players
    by {stat_col}_CUM_AVG.

    Args:
        df (pd.DataFrame): DataFrame with player stats including cumulative averages
        date (datetime or str): The target date (usually the current game date)
        stat_col (str): The stat column (e.g., "PTS") for cumulative average lookup
        injured (bool): If False, consider every player in the available-roster
                       frame and use their latest state at or before `date`.
                       Same-day rows carry shifted, pre-game cumulative values;
                       current-game MIN/PTS are never used for selection.
                       If True, consider the last state strictly before `date`.
        lowest (bool): If False (default), return highest averages (descending).
                      If True, return lowest averages (ascending)
        n_players (int): Number of players to return
        min_minutes (int): Minimum average minutes threshold

    Returns:
        list: List of tuples (player_id, player_name, cumulative_average)
    """
    if stat_col == "DEF_RATING":
        lowest = True

    if injured:
        min_minutes = min_minutes * 0.8

    if df.empty:
        return []

    if injured:
        # For injured players: last game *before* `date`
        df_inj = df[df["GAME_DATE"] < date].sort_values(["PLAYER_ID", "GAME_DATE"])
        df_last = df_inj.groupby("PLAYER_ID", as_index=False).tail(1).copy()

    else:
        # Availability is roster membership minus the injury/inactive set.  Take
        # one state per available player instead of selecting only players who
        # logged minutes in this game.  The same-day cumulative value is safe:
        # precompute_cumulative_avg_stat shifts the raw stat before calculating it.
        df_available = df[df["GAME_DATE"] <= date].sort_values(
            ["PLAYER_ID", "GAME_DATE"], kind="mergesort"
        )
        df_last = df_available.groupby("PLAYER_ID", as_index=False).tail(1).copy()

    if df_last.empty:
        return []

    # Check if MIN_CUM_AVG already exists (e.g., when stat_col="MIN")
    if "MIN_CUM_AVG" not in df_last.columns:
        current_season = (
            df_last["SEASON_YEAR"].iloc[0] if "SEASON_YEAR" in df_last.columns else None
        )
        df_prior = df[df["GAME_DATE"] < date]
        if current_season is not None and "SEASON_YEAR" in df_prior.columns:
            df_prior = df_prior[df_prior["SEASON_YEAR"] == current_season]
        df_cum_min = (
            df_prior.groupby("PLAYER_ID", as_index=False)["MIN"]
            .mean()
            .rename(columns={"MIN": "MIN_CUM_AVG"})
        )

        # Merge the cumulative average minutes into the selected game rows
        df_last = df_last.merge(df_cum_min, on="PLAYER_ID", how="left")

    cum_col = f"{stat_col}_CUM_AVG"

    # Create extra variable to check if player meets the minimum threshold
    df_last.loc[:, "MEETS_MIN_THRESHOLD"] = (
        df_last["MIN_CUM_AVG"].fillna(0) >= min_minutes
    ).astype(int)

    # Sort by the cumulative average column
    df_sorted = df_last.sort_values(
        by=["MEETS_MIN_THRESHOLD", cum_col], ascending=[False, lowest]
    )

    # Extract the top n (or bottom n) players
    chosen = df_sorted.head(n_players)

    top_or_bottom_n = list(
        zip(chosen["PLAYER_ID"], chosen["PLAYER_NAME"], chosen[cum_col], strict=True)
    )

    return top_or_bottom_n


def precompute_cumulative_avg_stat(
    df_players: pd.DataFrame,
    stat_col: str = "PTS",
    ewm_halflife_games: int = EWMA_HALFLIFE_GAMES,
) -> pd.DataFrame:
    """
    Recency-weighted average (EWMA) of `stat_col` per (SEASON_YEAR, PLAYER_ID),
    using only prior valid appearances (MIN > 0) and excluding the current game.

    - Excludes the current game's stat from the estimate (via shift).
    - Gives slightly higher weight to recent games.
    - If a player has no prior valid game, the value is 0.
    - Only considers games where MIN > 0 as valid appearances.
    - Groups by both SEASON_YEAR and PLAYER_ID.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        stat_col (str): Column name for the statistic to compute average for
        ewm_halflife_games (int): EWMA halflife in games (higher = less reactive)

    Returns:
        pd.DataFrame: Updated DataFrame with recency-weighted average columns
    """
    out = df_players.copy()

    # Defensive type conversions
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    out[stat_col] = pd.to_numeric(out[stat_col], errors="coerce")
    out["MIN"] = pd.to_numeric(out["MIN"], errors="coerce")

    # Sort to make shift/expanding meaningful
    out.sort_values(
        ["SEASON_YEAR", "PLAYER_ID", "GAME_DATE"], ascending=True, inplace=True
    )

    # Keep only valid appearances for the signal; invalid games are ignored.
    valid_mask = out["MIN"].fillna(0) > 0
    valid_stat = out[stat_col].where(valid_mask)

    # Use only prior games (shift) to avoid target leakage.
    shifted_valid_stat = valid_stat.groupby(
        [out["SEASON_YEAR"], out["PLAYER_ID"]]
    ).shift(1)

    # Recency-weighted estimate by player-season.
    in_season = shifted_valid_stat.groupby(
        [out["SEASON_YEAR"], out["PLAYER_ID"]],
        group_keys=False,
    ).transform(
        lambda s: s.ewm(
            halflife=ewm_halflife_games,
            adjust=False,
            min_periods=1,
            ignore_na=True,
        ).mean()
    )

    # Fill order, matching _SEASON_BEFORE_AVG in statistics.compute_rolling_stats:
    #   1) this season to date (preferred)
    #   2) the player's previous REGULAR season average
    #   3) 0, for a player with no prior season at all
    # Without step 2 the grouping by (SEASON_YEAR, PLAYER_ID) left every player at
    # 0 for their season opener, which then propagated into every TOP-N player
    # column as a missing value.
    previous_season = _previous_regular_season_player_average(out, stat_col, valid_mask)
    out[f"{stat_col}_CUM_AVG"] = in_season.fillna(previous_season).fillna(0)

    return out
