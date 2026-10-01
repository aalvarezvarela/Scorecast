import numpy as np
import pandas as pd


def compute_differences_in_points_conceeded_annotated(df: pd.DataFrame) -> pd.DataFrame:
    """Average prior conceded-versus-expected differences by team and season.

    Args:
        df (pd.DataFrame): Game rows with TEAM_ID_TEAM_HOME, TEAM_ID_TEAM_AWAY,
            SEASON_YEAR, GAME_DATE, and
            DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME and
            DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME. GAME_ID is an
            optional sort tie-breaker.

    Returns:
        pd.DataFrame: A copy sorted by descending date with
            AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_HOME and
            AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_AWAY. Each is an
            expanding mean excluding the current row, with missing values set to 0.
    """
    out = df.copy()

    sort_home = ["TEAM_ID_TEAM_HOME", "SEASON_YEAR", "GAME_DATE"]
    sort_away = ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR", "GAME_DATE"]
    if "GAME_ID" in out.columns:
        sort_home.append("GAME_ID")
        sort_away.append("GAME_ID")

    home_col = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_HOME"
    away_col = "AVG_DIFFERENCE_CONCEDED_VS_ANNOTATED_BEFORE_GAME_TEAM_AWAY"

    out.sort_values(sort_home, ascending=True, inplace=True)
    out[home_col] = (
        out.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
            "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    out.sort_values(sort_away, ascending=True, inplace=True)
    out[away_col] = (
        out.groupby(["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"])[
            "DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)

    out.sort_values("GAME_DATE", ascending=False, inplace=True)
    return out


def get_last_5_matchup_excluding_current(row, df):
    """Return total points from the five latest prior home/away matchups.

    Matchups must have the same home and away team orientation and a GAME_DATE
    strictly before the reference date. Missing matchups are padded with NaN;
    the mean uses only available total-point values.

    Args:
        row (pd.Series): Reference game with TEAM_ID_TEAM_HOME,
            TEAM_ID_TEAM_AWAY, and GAME_DATE.
        df (pd.DataFrame): Historical games with those columns and TOTAL_POINTS.

    Returns:
        dict: Six entries: LAST_{1..5}_GAMES_TOTAL_POINTS_BEFORE, ordered from
            most recent to oldest, and LAST_5_GAMES_TOTAL_POINTS_BEFORE_MEAN.
    """
    home_team = row["TEAM_ID_TEAM_HOME"]
    away_team = row["TEAM_ID_TEAM_AWAY"]
    current_date = row["GAME_DATE"]

    df_matchups = df[
        (df["TEAM_ID_TEAM_HOME"] == home_team)
        & (df["TEAM_ID_TEAM_AWAY"] == away_team)
        & (df["GAME_DATE"] < current_date)
    ].copy()

    df_matchups.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    df_matchups = df_matchups.head(5)

    totals_list = df_matchups["TOTAL_POINTS"].tolist()

    # Pad with NaN if fewer than 5 real matchups found
    while len(totals_list) < 5:
        totals_list.append(np.nan)

    real_values = [v for v in totals_list if not np.isnan(v)]
    mean = sum(real_values) / len(real_values) if real_values else np.nan
    return {
        "LAST_1_GAMES_TOTAL_POINTS_BEFORE": totals_list[0],
        "LAST_2_GAMES_TOTAL_POINTS_BEFORE": totals_list[1],
        "LAST_3_GAMES_TOTAL_POINTS_BEFORE": totals_list[2],
        "LAST_4_GAMES_TOTAL_POINTS_BEFORE": totals_list[3],
        "LAST_5_GAMES_TOTAL_POINTS_BEFORE": totals_list[4],
        "LAST_5_GAMES_TOTAL_POINTS_BEFORE_MEAN": mean,
    }


def compute_home_points_conceded_avg(df):
    """Compute season-to-date points conceded at home and away.

    Exclude the current game and group by team and SEASON_YEAR, separately for
    home and away appearances. Rows with no prior games retain NaN averages.

    Args:
        df (pd.DataFrame): Game rows containing TEAM_ID_TEAM_HOME,
            TEAM_ID_TEAM_AWAY, SEASON_YEAR, GAME_DATE, PTS_TEAM_HOME, and
            PTS_TEAM_AWAY. GAME_ID is used as a sort tie-breaker when present.

    Returns:
        pd.DataFrame: A copy sorted by descending date, with
            AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME and
            AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME.
    """

    out = df.copy()

    sort_home = ["TEAM_ID_TEAM_HOME", "SEASON_YEAR", "GAME_DATE"]
    sort_away = ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR", "GAME_DATE"]
    if "GAME_ID" in out.columns:
        sort_home.append("GAME_ID")
        sort_away.append("GAME_ID")

    # Sort DataFrame so previous games come first
    out = out.sort_values(sort_home, ascending=True)

    # Define the new column name
    col_name = "AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME"

    # Compute rolling average of points conceded at home (excluding current game)
    out[col_name] = out.groupby(["TEAM_ID_TEAM_HOME", "SEASON_YEAR"])[
        "PTS_TEAM_AWAY"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Sort again for away calculations
    out = out.sort_values(sort_away, ascending=True)

    # Compute rolling average of points conceded away by the away team
    out["AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME"] = out.groupby(
        ["TEAM_ID_TEAM_AWAY", "SEASON_YEAR"]
    )["PTS_TEAM_HOME"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    final_sort = ["GAME_DATE"]
    if "GAME_ID" in out.columns:
        final_sort.append("GAME_ID")
    out = out.sort_values(final_sort, ascending=False)
    return out
