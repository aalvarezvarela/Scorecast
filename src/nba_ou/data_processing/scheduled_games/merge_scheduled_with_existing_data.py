import pandas as pd
from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION


def _normalize_scheduled_season_fields(games: pd.DataFrame) -> pd.DataFrame:
    """Align scheduled-game season keys with the historical team/player tables."""
    out = games.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out["SEASON_YEAR"] = pd.to_numeric(
        out["SEASON"].astype(str).str[:4], errors="coerce"
    ).astype("Int64")
    out["SEASON_ID"] = out["GAME_ID"].str[2] + out["SEASON_YEAR"].astype(str)
    return out


def standardize_and_merge_scheduled_games_to_team_data(df, scheduled_games):
    """Append scheduled games as home and away team rows.

    Normalize game and season keys, attach the latest historical team metadata,
    and keep the historical columns plus GAME_TIME when provided. Scheduled
    outcome statistics are null. Deduplicate by TEAM_ID and GAME_DATE, keeping
    the last row, and sort by descending date.

    Args:
        df (pd.DataFrame): Historical team statistics and team metadata.
        scheduled_games (pd.DataFrame): Scheduled rows with GAME_ID,
            GAME_DATE_EST, SEASON, HOME_TEAM_ID, VISITOR_TEAM_ID, and optional
            GAME_TIME.

    Returns:
        pd.DataFrame: Historical and scheduled team rows.
    """
    # Ensure column names match
    games_renamed = scheduled_games.rename(
        columns={
            "GAME_DATE_EST": "GAME_DATE",
            "HOME_TEAM_ID": "TEAM_ID",
            "VISITOR_TEAM_ID": "TEAM_ID_AWAY",  # Temporarily rename to avoid conflict
        }
    )
    games_renamed = _normalize_scheduled_season_fields(games_renamed)
    # set string to Team_ID of games
    games_renamed["TEAM_ID"] = games_renamed["TEAM_ID"].astype(str)
    games_renamed["TEAM_ID_AWAY"] = games_renamed["TEAM_ID_AWAY"].astype(str)
    games_renamed["TEAM_ID_AWAY"] = games_renamed["TEAM_ID_AWAY"].astype(str)
    games_renamed["GAME_DATE"] = pd.to_datetime(games_renamed["GAME_DATE"])
    games_renamed["SEASON_ID"] = games_renamed["SEASON_ID"].astype(str)
    # Create separate DataFrames for home and away teams
    cols_to_keep = ["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON_ID", "SEASON_YEAR"]
    if "GAME_TIME" in games_renamed.columns:
        cols_to_keep.append("GAME_TIME")

    home_games = games_renamed[cols_to_keep].copy()
    home_games["HOME"] = True  # Mark as home team

    cols_to_keep_away = [
        "GAME_ID",
        "TEAM_ID_AWAY",
        "GAME_DATE",
        "SEASON_ID",
        "SEASON_YEAR",
    ]
    if "GAME_TIME" in games_renamed.columns:
        cols_to_keep_away.append("GAME_TIME")

    away_games = games_renamed[cols_to_keep_away].copy()
    away_games.rename(columns={"TEAM_ID_AWAY": "TEAM_ID"}, inplace=True)
    away_games["HOME"] = False  # Mark as away team

    # Concatenate both home and away records
    games_expanded = pd.concat([home_games, away_games], ignore_index=True)
    team_info_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "TEAM_CITY"]
    team_info_source = df[team_info_cols].copy()
    team_info_source["TEAM_ID"] = team_info_source["TEAM_ID"].astype(str)

    # Pick the most recent metadata per TEAM_ID to avoid reviving historical names
    # (e.g., "New Jersey Nets" instead of "Brooklyn Nets") on scheduled rows.
    if "GAME_DATE" in df.columns:
        team_info_source = team_info_source.join(
            pd.to_datetime(df["GAME_DATE"], errors="coerce").rename("_GAME_DATE_SORT")
        )
        sort_cols = ["TEAM_ID", "_GAME_DATE_SORT"]
        ascending = [True, False]
        if "GAME_ID" in df.columns:
            team_info_source = team_info_source.join(
                df["GAME_ID"].rename("_GAME_ID_SORT")
            )
            sort_cols.append("_GAME_ID_SORT")
            ascending.append(False)
        team_info_source = team_info_source.sort_values(
            by=sort_cols, ascending=ascending, kind="mergesort"
        )

    team_info_from_df = team_info_source.drop_duplicates(
        subset=["TEAM_ID"], keep="first"
    )[team_info_cols].copy()

    id_to_name = {team_id: name for name, team_id in TEAM_ID_MAP.items()}
    team_info_from_df["TEAM_NAME"] = (
        team_info_from_df["TEAM_NAME"]
        .map(TEAM_NAME_STANDARDIZATION)
        .fillna(team_info_from_df["TEAM_NAME"])
        .fillna(team_info_from_df["TEAM_ID"].astype(str).map(id_to_name))
    )
    games_expanded = games_expanded.merge(team_info_from_df, on="TEAM_ID", how="left")

    # Merge with `df`, keeping columns from both dataframes
    # Preserve GAME_TIME if it exists in games_expanded
    combined_df = pd.concat([df, games_expanded], ignore_index=True, join="outer")
    columns_to_keep = list(df.columns)
    if "GAME_TIME" in combined_df.columns and "GAME_TIME" not in columns_to_keep:
        columns_to_keep.append("GAME_TIME")
    df_merged = combined_df[columns_to_keep]

    # remove duplicated rows based on TEAM_ID and GAME_DATE, keeping the last one
    df_merged = df_merged.drop_duplicates(
        subset=["TEAM_ID", "GAME_DATE"], keep="last"
    ).reset_index(drop=True)
    df_merged = df_merged.sort_values(by=["GAME_DATE"], ascending=False).reset_index(
        drop=True
    )

    return df_merged


def standardize_and_merge_scheduled_games_to_players_data(
    games_original, df_players_original
):
    """Build scheduled player placeholders from each player's latest team row.

    Assign players to scheduled games using their latest loaded TEAM_ID. Preserve
    columns preceding START_POSITION and clear the remaining box-score values,
    including MIN. Replace the game-date and season keys with the scheduled
    values. Null MIN marks these rows as scheduled roster evidence.

    Args:
        games_original (pd.DataFrame): Scheduled rows with GAME_ID, GAME_DATE_EST,
            SEASON, HOME_TEAM_ID, and VISITOR_TEAM_ID.
        df_players_original (pd.DataFrame): Player history with PLAYER_ID,
            TEAM_ID, GAME_DATE, and box-score columns in their original order.

    Returns:
        pd.DataFrame: Placeholder rows only, ready to append to player history;
            an empty frame if no team-game placeholders can be built.
    """
    games = _normalize_scheduled_season_fields(games_original)
    df_players = df_players_original.copy()
    games = games.rename(columns={"GAME_DATE_EST": "GAME_DATE"})
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    cols_to_keep = ["SEASON_ID", "SEASON_YEAR", "GAME_DATE", "GAME_ID"]

    games_home = games[cols_to_keep + ["HOME_TEAM_ID"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    games_away = games[cols_to_keep + ["VISITOR_TEAM_ID"]].rename(
        columns={"VISITOR_TEAM_ID": "TEAM_ID"}
    )

    # Combine them into a single DataFrame for "all participating teams in each game"
    games_teams = pd.concat([games_home, games_away], ignore_index=True)
    games_teams["TEAM_ID"] = games_teams["TEAM_ID"].astype(str)

    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], format="%Y-%m-%d")
    df_players = df_players.sort_values(by=["PLAYER_ID", "GAME_DATE"])

    # 3) Group by PLAYER_ID and grab the last row in each group
    df_last_game = df_players.groupby("PLAYER_ID", as_index=False).tail(1).copy()
    df_last_game["TEAM_ID"] = df_last_game["TEAM_ID"].astype(str)

    cols_to_keep = []
    for col in df_last_game.columns:
        if col == "START_POSITION":
            break
        cols_to_keep.append(col)

    next_game_parts = []
    for row in games_teams.itertuples():
        df_temp = df_last_game[df_last_game["TEAM_ID"] == row.TEAM_ID].copy()
        # set all to null except cols to keep
        for col in df_temp.columns:
            if col not in cols_to_keep:
                df_temp[col] = None

        df_temp["GAME_ID"] = row.GAME_ID
        df_temp["SEASON_ID"] = row.SEASON_ID
        df_temp["SEASON_YEAR"] = row.SEASON_YEAR
        df_temp["GAME_DATE"] = row.GAME_DATE
        next_game_parts.append(df_temp)

    if not next_game_parts:
        return pd.DataFrame(columns=df_last_game.columns)

    df_next_game = pd.concat(next_game_parts, ignore_index=True, sort=False)
    if "SEASON_YEAR" in df_players.columns:
        df_next_game["SEASON_YEAR"] = df_next_game["SEASON_YEAR"].astype(
            df_players["SEASON_YEAR"].dtype
        )

    # create_player_lookup identifies scheduled placeholder rows by MIN.isna(); breaking
    # this contract would cause those rows to be silently dropped from the roster.
    assert "MIN" not in df_next_game.columns or df_next_game["MIN"].isna().all(), (
        "Scheduled placeholder player rows must have NaN MIN to be recognized by "
        "create_player_lookup; update create_player_lookup if the contract changes."
    )

    return df_next_game
