from nba_ou.config.constants import EXHIBITION_SEASON_TYPES
from nba_ou.utils.seasons import classify_season_type, is_exhibition_game_id


def filter_valid_games(df):
    """
    Filter to games with exactly 2 team entries and exclude invalid game types.

    This ensures that only complete games (both teams present) are included
    in the dataset, removing any incomplete or malformed game records.
    Also excludes Preseason and All Star games (``EXHIBITION_SEASON_TYPES``),
    first, so the incomplete-entry warning does not list preseason games
    against non-NBA clubs.

    Args:
        df (pd.DataFrame): Team game statistics DataFrame with SEASON_TYPE column

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid games
    """
    df = df[~is_exhibition_game_id(df["GAME_ID"]).to_numpy()]

    # Filter to games with exactly 2 team entries
    valid_games = df["GAME_ID"].value_counts()
    invalid_games = valid_games[valid_games != 2]
    if len(invalid_games) > 0:
        print(
            f"\n⚠️ Filtering out {len(invalid_games)} games with incomplete team entries:"
        )
        for game_id, count in invalid_games.items():
            game_rows = df[df["GAME_ID"] == game_id]
            if not game_rows.empty:
                teams = (
                    game_rows["TEAM_NAME"].unique()
                    if "TEAM_NAME" in game_rows.columns
                    else game_rows["TEAM_ID"].unique()
                )
                date = (
                    game_rows["GAME_DATE"].iloc[0]
                    if "GAME_DATE" in game_rows.columns
                    else "Unknown"
                )
                print(
                    f"  - {game_id}: {count} team(s) - {', '.join(map(str, teams))} on {date}"
                )

    valid_games = valid_games[valid_games == 2].index

    df = df[df["GAME_ID"].isin(valid_games)]
    df.loc[:, "SEASON_TYPE"] = df["GAME_ID"].apply(classify_season_type)
    df.loc[:, "SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)
    df = df[~df["SEASON_TYPE"].isin(EXHIBITION_SEASON_TYPES)]

    return df
