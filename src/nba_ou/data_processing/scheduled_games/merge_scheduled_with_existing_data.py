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


def _season_bucket(players: pd.DataFrame) -> pd.Series:
    """The lookup's season key: ``SEASON_ID`` where present, else game type + year.

    ``0022501171`` in 2025 -> ``"22025"``, the same key a stored row carries.
    """
    derived = players["GAME_ID"].astype(str).str[2] + pd.to_numeric(
        players["SEASON_YEAR"], errors="coerce"
    ).astype("Int64").astype(str)
    if "SEASON_ID" not in players.columns:
        return derived
    season_id = players["SEASON_ID"].astype(str)
    return season_id.where(players["SEASON_ID"].notna(), derived)


def _has_row_in_bucket(placeholders: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """Whether each placeholder's player already has a row in its season bucket."""
    seen = set(
        zip(_season_bucket(history), history["PLAYER_ID"].astype(str), strict=True)
    )
    return pd.Series(
        [
            (bucket, str(player)) in seen
            for bucket, player in zip(
                _season_bucket(placeholders), placeholders["PLAYER_ID"], strict=True
            )
        ],
        index=placeholders.index,
        dtype=bool,
    )


def standardize_and_merge_scheduled_games_to_players_data(
    games_original, df_players_original
):
    """Build scheduled player placeholders from each player's latest team row.

    Assign players to scheduled games using their latest loaded TEAM_ID. Preserve
    columns preceding START_POSITION and clear the remaining box-score values,
    including MIN. Replace the game-date and season keys with the scheduled
    values. Null MIN marks these rows as scheduled roster evidence.

    Only players who already have a row in the scheduled game's own season
    bucket (``SEASON_ID``: ``22025`` for the 2025-26 regular season, ``52025``
    for its play-in) get a placeholder. That is the bucket
    ``create_player_lookup`` reads a stored game's roster from, so the roster a
    prediction sees is the one training saw for the same game:

    * Without it, every player whose latest row *anywhere in the loaded seasons*
      is with a team reappeared on that team tonight -- players gone for two
      seasons, camp cuts from the preseason. On 2026-04-10 Charlotte got 37
      players instead of training's 19 (Davis Bertans, Ish Smith, ...), the
      Lakers 30 instead of 18 (Cam Reddish, Christian Wood, ...); none is on an
      injury report, so all counted as available.
    * When nobody has a row in the bucket yet -- opening night, a first play-in
      game -- no placeholder is built, and the lookup falls back to the season
      year and then the previous season exactly as it does in training. A
      placeholder there would have blocked that fallback.

    Checked with the real lookup, training vs prediction rosters: identical for
    4/4 team-games on the 2025-10-21 opener, 24/24 on 2025-10-22, 24/24 on
    2025-10-24 and 30/30 on 2026-04-10.

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
    df_next_game = df_next_game[
        _has_row_in_bucket(df_next_game, df_players)
    ].reset_index(drop=True)
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
