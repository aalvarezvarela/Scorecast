import re
import warnings

import pandas as pd
from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION
from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.referees.fetch_refs_data import (
    fetch_nba_referee_assignments_today,
)
from nba_ou.postgre_db.injuries_refs.ref_assignments_archive import (
    archive_referee_assignments,
)


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
    archive: bool = True,
) -> pd.DataFrame:
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

    df_refs = fetch_nba_referee_assignments_today(SETTINGS.nba_official_assignments_url)

    if archive:
        # Best effort: the archive exists to accumulate role-labelled crews and
        # must never block the day's predictions.
        try:
            archived = archive_referee_assignments(df_refs)
            print(f"Archived {archived} referee assignment rows")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not archive referee assignments: {exc}", stacklevel=2)

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
