import numpy as np
import pandas as pd
from nba_ou.config.market_columns import HOME_MARGIN_COL
from nba_ou.config.odds_columns import spread_col, total_line_col
from nba_ou.data_processing.historic_games.historic_games_statistics import (
    compute_differences_in_points_conceeded_annotated,
    compute_home_points_conceded_avg,
    get_last_5_matchup_excluding_current,
)
from tqdm import tqdm

MAIN_TOTAL_LINE_COL = total_line_col()
MAIN_SPREAD_COL = spread_col()


def deduplicate_game_level_columns(df_merged):
    """
    Deduplicate columns that are game-level (same for both teams) after home/away merge.

    These columns get duplicated with _TEAM_HOME and _TEAM_AWAY suffixes but contain
    the same value since they're game-level, not team-level data.

    Only deduplicates the actual total line values, not derived statistics like
    rolling averages or trends which are team-specific.

    Args:
        df_merged (pd.DataFrame): Merged home/away dataframe

    Returns:
        pd.DataFrame: Dataframe with deduplicated game-level columns
    """
    # Keywords that indicate team-specific derived statistics (not game-level)
    team_specific_keywords = [
        "_BEFORE_",
        "_AVG",
        "_TREND_",
        "_STD",
        "_LAST_",
        "_WEIGHTED_",
        "_SEASON_",
    ]

    # Find all pairs of duplicated columns
    home_suffix = "_TEAM_HOME"
    away_suffix = "_TEAM_AWAY"

    columns_to_deduplicate = []
    for col in df_merged.columns:
        if col.endswith(home_suffix):
            base_col = col[: -len(home_suffix)]
            away_col = base_col + away_suffix

            # Check if this should be deduplicated
            # 1. Must be a total line column
            is_total_line = base_col.startswith("ODDS_TOTAL_LINE_")

            # 2. Must NOT contain team-specific keywords
            is_team_specific = any(
                keyword in base_col for keyword in team_specific_keywords
            )

            # Deduplicate only if it's a total line AND not a derived team-specific stat
            if is_total_line and not is_team_specific and away_col in df_merged.columns:
                columns_to_deduplicate.append((col, away_col, base_col))

    # Deduplicate: keep home version, drop away version, rename to base name
    for home_col, away_col, base_col in columns_to_deduplicate:
        df_merged[base_col] = df_merged[home_col]
        df_merged = df_merged.drop(columns=[home_col, away_col])

    if columns_to_deduplicate:
        print(f"Deduplicated {len(columns_to_deduplicate)} game-level column(s)")

    return df_merged


def merge_home_away_data(df, todays_prediction=False):
    """
    Merge home and away team data and prepare final training features.

    Args:
        df (pd.DataFrame): Team statistics DataFrame with injury data attached
        todays_prediction (bool): If True, includes GAME_TIME in static columns. Default: False

    Returns:
        pd.DataFrame: Final training-ready DataFrame
    """
    # Rename columns that start with 'TOP' by adding '_BEFORE' at the end
    df.rename(
        columns=lambda x: (
            f"{x}_BEFORE" if x.startswith("TOP") and not x.endswith("_BEFORE") else x
        ),
        inplace=True,
    )

    # Add '_BEFORE' suffix to specified AVG_INJURED_* columns
    avg_injured_cols = [
        "AVG_INJURED_PTS",
        "AVG_INJURED_MIN",
        "AVG_INJURED_PACE_PER40",
        "AVG_INJURED_DEF_RATING",
        "AVG_INJURED_OFF_RATING",
        "AVG_INJURED_TS_PCT",
    ]

    df.rename(
        columns={
            col: f"{col}_BEFORE"
            for col in avg_injured_cols
            if col in df.columns and not col.endswith("_BEFORE")
        },
        inplace=True,
    )

    # The per-slot OFF_RATING value only exists under the legacy player profile;
    # the reduced profile replaces rate slots with minutes-weighted aggregates
    # (see players/feature_profile.py), so the ratio is derived only when its
    # input was emitted.
    if "TOP1_PLAYER_OFF_RATING_BEFORE" in df.columns:
        den = df["OFF_RATING_SEASON_BEFORE_AVG"].replace(0, np.nan)
        df["STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE"] = (
            df["TOP1_PLAYER_OFF_RATING_BEFORE"] / den
        ).fillna(0)

    df["STAR_PTS_PERCENTAGE_BEFORE"] = (
        df["TOP1_PLAYER_PTS_BEFORE"] / df["PTS_SEASON_BEFORE_AVG"]
    )

    # Fill NaNs in injured player columns with zeros
    injured_player_cols = [col for col in df.columns if "INJURED_PLAYER" in col]
    df[injured_player_cols] = df[injured_player_cols].fillna(0)

    # Merge home and away stats
    static_columns = [
        "SEASON_ID",
        "GAME_ID",
        "GAME_DATE",
        "SEASON_TYPE",
        "SEASON_YEAR",
        "IS_OVERTIME",
    ]

    # Include GAME_TIME for today's predictions
    if todays_prediction and "GAME_TIME" in df.columns:
        static_columns.append("GAME_TIME")

    df_home = df[df["HOME"]].copy().drop(columns="HOME")
    df_away = df[~df["HOME"]].copy().drop(columns="HOME")

    key = static_columns
    if df_home.duplicated(key).any() or df_away.duplicated(key).any():
        raise ValueError(
            "Home/Away tables have duplicate keys; merge would be many-to-many."
        )

    df_merged = pd.merge(
        df_home,
        df_away,
        on=static_columns,
        how="inner",
        suffixes=("_TEAM_HOME", "_TEAM_AWAY"),
    )

    # Deduplicate game-level columns (same value for both teams)
    df_merged = deduplicate_game_level_columns(df_merged)

    # Set unified main-book spread column (home team's perspective is standard)
    main_spread_team_home = f"{MAIN_SPREAD_COL}_TEAM_HOME"
    if main_spread_team_home in df_merged.columns:
        df_merged[MAIN_SPREAD_COL] = df_merged[main_spread_team_home]

    # Compute Totals
    df_merged["TOTAL_POINTS"] = df_merged.PTS_TEAM_HOME + df_merged.PTS_TEAM_AWAY
    df_merged["TOTAL_PF"] = df_merged.PF_TEAM_HOME + df_merged.PF_TEAM_AWAY

    # The spread market's outcome, derived at the one point where both teams'
    # final scores exist side by side. PTS_TEAM_HOME/PTS_TEAM_AWAY themselves are
    # carried onward from here by select_training_columns so this stays
    # reproducible downstream; both they and HOME_MARGIN are outcome facts and
    # are blocked from every feature matrix (see training_pipeline.config
    # LEAKING_TARGET_COLUMNS).
    df_merged[HOME_MARGIN_COL] = df_merged.PTS_TEAM_HOME - df_merged.PTS_TEAM_AWAY

    # IS_PLAYOFF_GAME based on SEASON_TYPE
    df_merged["IS_PLAYOFF_GAME_BEFORE"] = (
        df_merged["SEASON_TYPE"].str.contains("Playoff", case=False, na=False)
    ).astype(int)

    # Apply function
    df_merged = compute_home_points_conceded_avg(df_merged)
    df_merged["DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_HOME_GAME"] = (
        df_merged["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
        - df_merged["AVG_POINTS_CONCEDED_AT_HOME_BEFORE_GAME"]
    )
    df_merged["DIFERENCE_POINTS_CONCEDED_VS_EXPECTED_BEFORE_AWAY_GAME"] = (
        df_merged["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        - df_merged["AVG_POINTS_CONCEDED_AWAY_BEFORE_GAME"]
    )

    df_merged = compute_differences_in_points_conceeded_annotated(df_merged)
    main_total_home = f"{MAIN_TOTAL_LINE_COL}_SEASON_BEFORE_AVG_TEAM_HOME"
    main_total_away = f"{MAIN_TOTAL_LINE_COL}_SEASON_BEFORE_AVG_TEAM_AWAY"
    if main_total_home in df_merged.columns and main_total_away in df_merged.columns:
        df_merged["TEAMS_DIFFERENCE_OVER_UNDER_LINE_BEFORE"] = (
            df_merged[main_total_home] - df_merged[main_total_away]
        )

    tqdm.pandas()
    # Apply row-by-row, returning a Series of dictionaries
    results_series = df_merged.progress_apply(
        lambda row: get_last_5_matchup_excluding_current(row, df_merged), axis=1
    )

    # Convert that Series of dicts into a DataFrame
    results_df = pd.DataFrame(results_series.tolist(), index=df_merged.index)

    # Finally, concatenate the new columns onto df_merged
    df_merged = pd.concat([df_merged, results_df], axis=1)
    df_merged.sort_values(["TEAM_ID_TEAM_HOME", "GAME_DATE"], ascending=True)

    df_merged = df_merged.sort_values(by="GAME_DATE", ascending=False)

    return df_merged
