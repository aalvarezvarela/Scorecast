"""``get_top_n_averages_with_names`` before its split into state and ranking, verbatim.

Reference for tests/test_player_lookup_speedups.py only.
"""


def get_top_n_averages_with_names_before(
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
