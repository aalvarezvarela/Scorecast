import numpy as np
import pandas as pd

from nba_ou.config.constants import (
    OVERTIME_THRESHOLD_MINUTES,
    REGULATION_GAME_MINUTES,
    TEAM_MINUTES_PER_40,
)

# Only additive box-score statistics are eligible for overtime normalization.
# A positive allowlist prevents identifiers, ratings, percentages, ratios, and
# other numeric metadata from being rescaled accidentally when schemas change.
OVERTIME_NORMALIZED_COUNTING_STATS = (
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
    "POSS",
)

PTS_PER_40_COLUMN = "PTS_PER_40"


def adjust_overtime(df):
    """
    Add points per 40 minutes and normalize additive overtime statistics.

    This function:
    - Preserves raw ``PTS`` so game targets remain actual final scores,
      including overtime.
    - Creates ``PTS_PER_40`` with an explicit valid-minutes guard. For
      ``0 < MIN < 240``, it uses ``PTS * TEAM_MINUTES_PER_40 / MIN``. For
      every other minutes value, including missing, zero, negative, non-finite,
      regulation, and overtime minutes, it retains raw ``PTS``.
    - Creates ``IS_OVERTIME`` using the shared overtime threshold.
    - Normalizes only explicitly allowlisted additive counting statistics in
      overtime games to a 48-minute regulation equivalent via ``240 / MIN``.
    - Leaves percentages, ratios, ratings, pace fields, identifiers, metadata,
      and ``MIN`` unchanged.

    Args:
        df (pd.DataFrame): Team game statistics with aggregate player-minutes
            in ``MIN``.

    Returns:
        pd.DataFrame: DataFrame with raw points preserved, ``PTS_PER_40``
            added, and eligible overtime counts normalized as floats.
    """
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    minutes = pd.to_numeric(df["MIN"], errors="coerce")
    points = pd.to_numeric(df["PTS"], errors="coerce")
    valid_minutes = minutes.gt(0) & np.isfinite(minutes)
    overtime_mask = valid_minutes & minutes.ge(OVERTIME_THRESHOLD_MINUTES)

    df["IS_OVERTIME"] = overtime_mask.astype(int)

    points_per_40 = points.astype(float).copy()
    below_regulation_mask = valid_minutes & minutes.lt(REGULATION_GAME_MINUTES)
    points_per_40.loc[below_regulation_mask] = (
        points.loc[below_regulation_mask]
        * TEAM_MINUTES_PER_40
        / minutes.loc[below_regulation_mask]
    )
    df[PTS_PER_40_COLUMN] = points_per_40

    regulation_factor = REGULATION_GAME_MINUTES / minutes.loc[overtime_mask]
    normalized_columns = [
        column for column in OVERTIME_NORMALIZED_COUNTING_STATS if column in df.columns
    ]
    for column in normalized_columns:
        values = pd.to_numeric(df[column], errors="coerce").astype(float)
        values.loc[overtime_mask] = values.loc[overtime_mask] * regulation_factor
        df[column] = values

    print(
        "Overtime adjustments completed: raw PTS preserved; "
        f"{len(normalized_columns)} additive statistic(s) normalized."
    )
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)

    return df


def fix_home_away_parsing_errors(df_team: pd.DataFrame) -> pd.DataFrame:
    """Repair HOME flags for games with duplicate home/away assignments.

    For rows sharing the same GAME_ID and HOME value, infer the home team from
    MATCHUP and compare it with TEAM_ABBREVIATION. Column lookup accepts uppercase
    or lowercase names. The input is returned unchanged when GAME_ID or HOME is
    absent; team IDs and points are not modified.

    Args:
        df_team (pd.DataFrame): Team game rows with GAME_ID and HOME. MATCHUP and
            TEAM_ABBREVIATION are needed to repair the flagged rows.

    Returns:
        pd.DataFrame: The input frame with repaired HOME values where possible.
    """

    # Helper function to get column name regardless of case
    def get_col(name_upper: str) -> str:
        if name_upper in df_team.columns:
            return name_upper
        elif name_upper.lower() in df_team.columns:
            return name_upper.lower()
        else:
            return name_upper  # fallback to uppercase

    game_id_col = get_col("GAME_ID")
    home_col = get_col("HOME")
    matchup_col = get_col("MATCHUP")
    team_abbr_col = get_col("TEAM_ABBREVIATION")

    # Check if required columns exist
    if home_col not in df_team.columns or game_id_col not in df_team.columns:
        return df_team

    problematic_rows = df_team[df_team[[game_id_col, home_col]].duplicated(keep=False)]
    if not problematic_rows.empty:
        print(
            f"Found {len(problematic_rows)} rows with potential home/away parsing errors."
        )

        # Fix HOME column for problematic rows based on MATCHUP
        if matchup_col in df_team.columns and team_abbr_col in df_team.columns:
            for idx in problematic_rows.index:
                matchup = df_team.loc[idx, matchup_col]
                team_abbr = df_team.loc[idx, team_abbr_col]

                if "@" in matchup:
                    # Team after @ is home team
                    home_team = matchup.split("@")[1].strip()
                    df_team.loc[idx, home_col] = team_abbr == home_team
                elif "vs." in matchup:
                    # Team before vs. is home team
                    home_team = matchup.split("vs.")[0].strip()
                    df_team.loc[idx, home_col] = team_abbr == home_team

            print(f"Fixed HOME column for {len(problematic_rows)} problematic rows.")

    return df_team


def clean_team_data(df):
    """
    Clean team game data by removing invalid rows.

    This function:
    - Converts GAME_DATE to datetime
    - Drops rows with missing PTS values
    - Removes duplicate game/team entries
    - Filters out rows with 0 minutes or PTS <= 10

    Args:
        df (pd.DataFrame): Team game statistics DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")
    df.dropna(subset=["PTS"], inplace=True)
    df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="first", inplace=True)
    df = df[df["MIN"] != 0]
    df = df[df["PTS"] > 10]
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    df = fix_home_away_parsing_errors(df)
    return df
