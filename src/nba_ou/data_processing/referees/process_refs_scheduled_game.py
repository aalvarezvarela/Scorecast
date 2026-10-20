import re

import pandas as pd
from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.referees.load_assignments import load_referee_assignments


def _canonicalize_referee_name(name: str) -> str:
    if pd.isna(name):
        return pd.NA
    name = str(name).strip()
    name = re.sub(r"\s*\(#\d+\)", "", name)
    name = name.replace(".", "")
    name = re.sub(r"\s+", " ", name)
    return name or pd.NA


def process_scheduled_referee_assignments(
    df_scheduled_games_original: pd.DataFrame,
) -> pd.DataFrame:
    """Attach available crews; retain unassigned games with missing REF slots."""
    df_scheduled_games = df_scheduled_games_original.copy()

    # Create reverse mapping from team ID to team name
    id_to_team_name = {v: k for k, v in TEAM_ID_MAP.items()}

    # Convert team IDs to standard team names
    df_scheduled_games["home_team"] = (
        df_scheduled_games["HOME_TEAM_ID"].astype(str).map(id_to_team_name)
    )
    df_scheduled_games["away_team"] = (
        df_scheduled_games["VISITOR_TEAM_ID"].astype(str).map(id_to_team_name)
    )

    # Extract GAME_DATE from GAME_DATE_EST and keep GAME_ID
    df_scheduled_games["GAME_DATE"] = pd.to_datetime(
        df_scheduled_games["GAME_DATE_EST"]
    ).dt.date

    dates = df_scheduled_games["GAME_DATE"].dropna().unique()
    if len(dates) != 1:
        raise ValueError("Scheduled referee assignments require exactly one game date")
    df_refs = load_referee_assignments(dates[0].isoformat())

    if df_refs.empty:
        # Splitting an empty Game column cannot produce two team columns.
        # Keep every scheduled game so closing and intermediate builders both
        # receive missing assignments and leave their referee features NaN.
        return df_scheduled_games[
            ["GAME_ID", "GAME_DATE", "home_team", "away_team"]
        ].assign(REF_1=float("nan"), REF_2=float("nan"), REF_3=float("nan"))

    df_refs[["away_team", "home_team"]] = (
        df_refs["Game"].str.split("@", expand=True).apply(lambda c: c.str.strip())
    )

    # Create lowercase version of TEAM_NAME_STANDARDIZATION for case-insensitive mapping
    team_name_std_lower = {k.lower(): v for k, v in TEAM_NAME_STANDARDIZATION.items()}

    df_refs["home_team"] = df_refs["home_team"].str.lower().map(team_name_std_lower)
    df_refs["away_team"] = df_refs["away_team"].str.lower().map(team_name_std_lower)

    # Create referee columns and remove (#XX) notation
    df_refs["REF_1"] = df_refs["Crew Chief"].map(_canonicalize_referee_name)
    df_refs["REF_2"] = df_refs["Referee"].map(_canonicalize_referee_name)
    df_refs["REF_3"] = df_refs["Umpire"].map(_canonicalize_referee_name)

    # Merge referee data with scheduled games on home_team and away_team
    df_merged = df_scheduled_games[
        ["GAME_ID", "GAME_DATE", "home_team", "away_team"]
    ].merge(
        df_refs[["home_team", "away_team", "REF_1", "REF_2", "REF_3"]],
        on=["home_team", "away_team"],
        how="left",
    )

    return df_merged
