"""
NBA Over/Under Predictor - Travel Processing Module

This module computes travel-related features for NBA teams, including:
- Distance traveled between consecutive games
- Rolling 7-day and 14-day travel distances
- Home vs away game logic

Uses geographic coordinates to calculate great-circle distances between cities.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nba_ou.config.constants import CITY_TO_LATLON, CITY_TO_TIMEZONE


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth using Haversine formula.

    Args:
        lat1 (float): Latitude of first point in degrees
        lon1 (float): Longitude of first point in degrees
        lat2 (float): Latitude of second point in degrees
        lon2 (float): Longitude of second point in degrees

    Returns:
        float: Distance in kilometers
    """
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def build_team_game_log(df):
    """
    Convert game-level dataframe into team-centric game log.

    Each game row is split into two rows (one for home team, one for away team).
    CITY represents where the game is PLAYED (not team's home city).

    Args:
        df (pd.DataFrame): Game-level dataframe with columns:
            - GAME_ID
            - GAME_DATE
            - TEAM_ID_TEAM_HOME
            - TEAM_CITY_TEAM_HOME
            - TEAM_ID_TEAM_AWAY
            - TEAM_CITY_TEAM_AWAY

    Returns:
        pd.DataFrame: Team-centric game log with columns:
            - GAME_ID
            - GAME_DATE
            - TEAM_ID
            - CITY (where game is played)
            - IS_HOME
    """
    # Home team: plays in their own city
    home = df[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID_TEAM_HOME",
            "TEAM_CITY_TEAM_HOME",
        ]
    ].copy()

    home.columns = ["GAME_ID", "GAME_DATE", "TEAM_ID", "CITY"]
    home["IS_HOME"] = True

    # Away team: plays in HOME TEAM'S city (travels TO opponent's city)
    away = df[
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID_TEAM_AWAY",
            "TEAM_CITY_TEAM_HOME",  # Changed: away team plays in home team's city!
        ]
    ].copy()

    away.columns = ["GAME_ID", "GAME_DATE", "TEAM_ID", "CITY"]
    away["IS_HOME"] = False

    team_log = pd.concat([home, away], ignore_index=True)
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

    return team_log.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def add_travel_distance(team_log, city_coords=None):
    """
    Compute travel distance for each game based on previous game location.

    Rules:
    - If team name not in city_coords, distance = 0
    - If previous city not available (first game), distance = 0
    - If both current and previous games are home games, distance = 0
    - Otherwise, distance = great-circle distance between cities

    Args:
        team_log (pd.DataFrame): Team-centric game log with TEAM_ID, CITY, IS_HOME
        city_coords (dict, optional): Dictionary mapping city names to (lat, lon).
                                     Defaults to CITY_TO_LATLON from constants

    Returns:
        pd.DataFrame: Team log with TRAVEL_KM column added
    """
    if city_coords is None:
        city_coords = CITY_TO_LATLON

    team_log = team_log.copy()

    # Get previous game information
    team_log["PREV_CITY"] = team_log.groupby("TEAM_ID")["CITY"].shift(1)
    team_log["PREV_IS_HOME"] = team_log.groupby("TEAM_ID")["IS_HOME"].shift(1)

    def compute_distance(row):
        """Compute travel distance for a single game."""
        # First game for team or missing data
        if pd.isna(row["PREV_CITY"]) or pd.isna(row["CITY"]):
            return 0.0

        # Both games at home - no travel
        if row["IS_HOME"] and row["PREV_IS_HOME"]:
            return 0.0

        # Check if cities exist in coordinates dictionary
        if row["PREV_CITY"] not in city_coords or row["CITY"] not in city_coords:
            return 0.0

        # Calculate distance
        lat1, lon1 = city_coords[row["PREV_CITY"]]
        lat2, lon2 = city_coords[row["CITY"]]

        return haversine_km(lat1, lon1, lat2, lon2)

    team_log["TRAVEL_KM"] = team_log.apply(compute_distance, axis=1)

    return team_log


def _timezone_offset_hours(city, game_date, city_timezones):
    """Return the UTC offset in hours for a city's timezone on a game date."""
    if pd.isna(city) or pd.isna(game_date) or city not in city_timezones:
        return None

    timestamp = pd.Timestamp(game_date)
    timezone_name = city_timezones[city]

    try:
        timezone = ZoneInfo(timezone_name)
    except Exception:
        return None

    if timestamp.tzinfo is not None:
        local_dt = timestamp.tz_convert(timezone_name).to_pydatetime()
    else:
        local_dt = datetime.combine(timestamp.date(), time(12), tzinfo=timezone)

    utc_offset = local_dt.utcoffset()
    if utc_offset is None:
        return None

    return utc_offset.total_seconds() / 3600


def add_jetlag_hours(team_log, city_timezones=None, max_lookback_days=4):
    """
    Compute timezone-change hours from the team's most recent game location.

    Only the immediately previous game is considered. If it is more than
    max_lookback_days before the current game, jetlag hours are set to 0.
    """
    if city_timezones is None:
        city_timezones = CITY_TO_TIMEZONE

    team_log = team_log.copy()
    team_log = team_log.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])

    if "PREV_CITY" not in team_log.columns:
        team_log["PREV_CITY"] = team_log.groupby("TEAM_ID")["CITY"].shift(1)

    team_log["PREV_GAME_DATE"] = team_log.groupby("TEAM_ID")["GAME_DATE"].shift(1)

    def compute_jetlag_hours(row):
        if pd.isna(row["PREV_CITY"]) or pd.isna(row["PREV_GAME_DATE"]):
            return 0.0

        days_since_prev_game = row["GAME_DATE"] - row["PREV_GAME_DATE"]
        if days_since_prev_game > pd.Timedelta(days=max_lookback_days):
            return 0.0

        current_offset = _timezone_offset_hours(
            row["CITY"], row["GAME_DATE"], city_timezones
        )
        previous_offset = _timezone_offset_hours(
            row["PREV_CITY"], row["GAME_DATE"], city_timezones
        )

        if current_offset is None or previous_offset is None:
            return 0.0

        return abs(current_offset - previous_offset)

    team_log["JETLAG_HOURS_FROM_LAST_GAME"] = team_log.apply(
        compute_jetlag_hours, axis=1
    )

    return team_log


def add_rolling_distances(team_log):
    """
    Add rolling travel distance sums for multiple time windows.

    Each window includes the trip to the current game's location because that
    travel has already occurred before tipoff. It also retains trips exactly on
    the left edge of the calendar window.

    For each game, computes:
    - Total kilometers traveled in last 1 day
    - Total kilometers traveled in last 2 days
    - Total kilometers traveled in last 5 days
    - Total kilometers traveled in last 7 days
    - Total kilometers traveled in last 14 days

    Args:
        team_log (pd.DataFrame): Team log with GAME_DATE, TEAM_ID, and TRAVEL_KM

    Returns:
        pd.DataFrame: Team log with KM_LAST_*_DAYS columns for windows: 1, 2, 5, 7, 14
    """
    team_log = team_log.copy()

    # Sort by TEAM_ID and GAME_DATE before rolling operations
    team_log = team_log.sort_values(["TEAM_ID", "GAME_DATE"])
    team_log = team_log.set_index("GAME_DATE")

    # Define rolling windows
    windows = [1, 2, 5, 7, 14]

    for window in windows:
        col_name = f"KM_LAST_{window}_DAYS"
        team_log[col_name] = (
            team_log.groupby("TEAM_ID")["TRAVEL_KM"]
            .rolling(f"{window}D", closed="both")
            .sum()
            .reset_index(level=0, drop=True)
        )
        # Fill NaN values (first games with no history) with 0
        team_log[col_name] = team_log[col_name].fillna(0)

    return team_log.reset_index()


def merge_travel_features(df, team_log, log_scale=True):
    """
    Merge travel features back to original game-level dataframe.

    Creates columns for multiple rolling windows (1, 2, 5, 7, 14 days), recent
    jetlag hours and raw-kilometer travel-recency ratios for both teams. Ratios
    are computed before the optional log transform; zero 14-day travel leaves
    the ratio undefined (NaN).

    Args:
        df (pd.DataFrame): Original game-level dataframe
        team_log (pd.DataFrame): Team log with travel features
        log_scale (bool): Whether to apply log transformation to travel features

    Returns:
        pd.DataFrame: Game-level dataframe with travel features added
    """
    # Define rolling windows
    windows = [1, 2, 5, 7, 14]

    # Build column lists dynamically
    travel_cols = ["GAME_ID", "TEAM_ID", "JETLAG_HOURS_FROM_LAST_GAME"] + [
        f"KM_LAST_{w}_DAYS" for w in windows
    ]

    # Filter team_log for home teams only and select relevant columns
    home_feats = team_log[team_log["IS_HOME"]][travel_cols].copy()

    # Rename columns for home team
    rename_dict_home = {
        "TEAM_ID": "TEAM_ID_TEAM_HOME",
        "JETLAG_HOURS_FROM_LAST_GAME": "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM",
    }
    for w in windows:
        rename_dict_home[f"KM_LAST_{w}_DAYS"] = f"TOTAL_KM_IN_LAST_{w}_DAYS_HOME_TEAM"
    home_feats.rename(columns=rename_dict_home, inplace=True)

    # Filter team_log for away teams only and select relevant columns
    away_feats = team_log[~team_log["IS_HOME"]][travel_cols].copy()

    # Rename columns for away team
    rename_dict_away = {
        "TEAM_ID": "TEAM_ID_TEAM_AWAY",
        "JETLAG_HOURS_FROM_LAST_GAME": "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM",
    }
    for w in windows:
        rename_dict_away[f"KM_LAST_{w}_DAYS"] = f"TOTAL_KM_IN_LAST_{w}_DAYS_AWAY_TEAM"
    away_feats.rename(columns=rename_dict_away, inplace=True)

    # Merge home features
    df = df.merge(home_feats, on=["GAME_ID", "TEAM_ID_TEAM_HOME"], how="left")

    # Merge away features
    df = df.merge(away_feats, on=["GAME_ID", "TEAM_ID_TEAM_AWAY"], how="left")

    # Build list of all travel columns
    travel_columns = []
    for w in windows:
        travel_columns.append(f"TOTAL_KM_IN_LAST_{w}_DAYS_HOME_TEAM")
        travel_columns.append(f"TOTAL_KM_IN_LAST_{w}_DAYS_AWAY_TEAM")

    travel_columns.extend(
        [
            "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM",
            "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM",
        ]
    )

    # Missing travel keeps its existing zero-distance fallback.
    for col in travel_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # A concentration is a fraction of kilometers, independent of their scale.
    # Compute it while these columns still carry raw distances.
    for side in ("HOME", "AWAY"):
        recent = df[f"TOTAL_KM_IN_LAST_2_DAYS_{side}_TEAM"]
        total = df[f"TOTAL_KM_IN_LAST_14_DAYS_{side}_TEAM"].replace(0, np.nan)
        df[f"TRAVEL_RECENCY_RATIO_{side}_2D_OVER_14D_BEFORE"] = recent / total

    if log_scale:
        for col in travel_columns:
            if col.startswith("TOTAL_KM_"):
                df[col] = np.log1p(df[col])

    return df


def compute_travel_features(df, log_scale=True):
    """
    Main function to compute all travel-related features for training data.

    This is the main entry point that orchestrates:
    1. Building team-centric game log
    2. Computing travel distances between consecutive games
    3. Computing timezone-change hours from the last game in the past 4 days
    4. Calculating rolling sums for 1, 2, 5, 7, and 14 day windows
    5. Merging features back to original dataframe
    6. Computing travel-recency ratios from raw kilometers
    7. Optionally applying log transformation to travel distances

    Args:
        df (pd.DataFrame): Game-level dataframe with required columns:
            - GAME_ID
            - GAME_DATE
            - TEAM_ID_TEAM_HOME
            - TEAM_CITY_TEAM_HOME
            - TEAM_ID_TEAM_AWAY
            - TEAM_CITY_TEAM_AWAY
        log_scale (bool): Whether to apply log1p transformation to travel distances

    Returns:
        pd.DataFrame: Original dataframe with added travel features for both home and away teams:
            - TOTAL_KM_IN_LAST_{1,2,5,7,14}_DAYS_HOME_TEAM
            - TOTAL_KM_IN_LAST_{1,2,5,7,14}_DAYS_AWAY_TEAM
            - JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM
            - JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM
            - TRAVEL_RECENCY_RATIO_{HOME,AWAY}_2D_OVER_14D_BEFORE (raw km ratio)
    """
    print("Computing travel features...")

    # Step 1: Build team-centric game log
    team_log = build_team_game_log(df)

    # Step 2: Compute travel distances
    team_log = add_travel_distance(team_log)

    # Step 3: Compute jetlag hours from the last game in the past 4 days
    team_log = add_jetlag_hours(team_log)

    # Step 4: Compute rolling sums
    team_log = add_rolling_distances(team_log)

    # Step 5: Merge back to original dataframe
    df = merge_travel_features(df, team_log, log_scale=log_scale)

    print("Travel features computed successfully.")

    return df
