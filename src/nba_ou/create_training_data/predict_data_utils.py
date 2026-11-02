import pandas as pd


def normalize_game_ids(game_ids) -> list[str]:
    """Normalize GAME_ID values into unique, non-empty strings."""
    if game_ids is None:
        return []

    normalized = []
    seen = set()
    for game_id in game_ids:
        if pd.isna(game_id):
            continue
        game_id_str = str(game_id).strip()
        if not game_id_str or game_id_str in seen:
            continue
        seen.add(game_id_str)
        normalized.append(game_id_str)
    return normalized


def extract_home_away_pairs_from_scheduled_games(
    scheduled_games: pd.DataFrame | None,
) -> list[tuple[str, str]]:
    """Extract unique (home_team_id, away_team_id) tuples from scheduled games."""
    if scheduled_games is None or scheduled_games.empty:
        return []

    required_cols = {"HOME_TEAM_ID", "VISITOR_TEAM_ID"}
    if not required_cols.issubset(set(scheduled_games.columns)):
        missing = sorted(required_cols - set(scheduled_games.columns))
        print(
            f"Could not derive extra matchup game IDs; missing scheduled columns: {missing}"
        )
        return []

    pairs_df = (
        scheduled_games[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]]
        .dropna(subset=["HOME_TEAM_ID", "VISITOR_TEAM_ID"])
        .drop_duplicates()
    )

    return [
        (str(home_team_id), str(away_team_id))
        for home_team_id, away_team_id in pairs_df.itertuples(index=False, name=None)
    ]


def filter_by_seasons_with_extra_game_ids(
    df: pd.DataFrame,
    seasons: list[str],
    recent_limit_to_include=None,
    extra_game_ids=None,
) -> pd.DataFrame:
    """
    Filter DataFrame to rows whose `SEASON_YEAR` matches the given seasons list,
    with an optional upper date cap and explicit extra `GAME_ID`s.

    Seasons are expressed as "YYYY-YY" strings (e.g., "2024-25"), and the
    DataFrame is expected to contain a `SEASON_YEAR` column with the integer
    season start year (e.g., 2024).

    Args:
        df: DataFrame with a `SEASON_YEAR` column.
        seasons: List of season strings like ["2024-25", "2023-24"].
        recent_limit_to_include: Optional upper date cap (inclusive).  Applied to
            `GAME_DATE` when the column is present to exclude unplayed games.
        extra_game_ids: Explicit `GAME_ID`s to always include (respects date cap).
    """
    if df.empty:
        return df

    # Derive the integer start-years for every season string
    season_years = {int(s[:4]) for s in seasons}

    season_mask = (
        df["SEASON_YEAR"]
        .apply(lambda v: int(str(v)) if pd.notna(v) else -1)
        .isin(season_years)
    )

    filtered_df = df[season_mask].copy()

    # Apply upper date cap when GAME_DATE is available
    if recent_limit_to_include is not None and "GAME_DATE" in filtered_df.columns:
        cutoff = pd.to_datetime(recent_limit_to_include)
        if hasattr(cutoff, "tz") and cutoff.tz is not None:
            cutoff = cutoff.tz_localize(None)
        filtered_df = filtered_df[
            pd.to_datetime(filtered_df["GAME_DATE"], errors="coerce") <= cutoff
        ]

    # Also preserve explicit extra GAME_IDs (with date cap)
    normalized_extra = normalize_game_ids(extra_game_ids)
    if normalized_extra and "GAME_ID" in df.columns:
        extra_mask = df["GAME_ID"].astype(str).isin(set(normalized_extra))
        extra_rows = df.loc[extra_mask].copy()
        if recent_limit_to_include is not None and "GAME_DATE" in extra_rows.columns:
            cutoff = pd.to_datetime(recent_limit_to_include)
            if hasattr(cutoff, "tz") and cutoff.tz is not None:
                cutoff = cutoff.tz_localize(None)
            extra_rows = extra_rows[
                pd.to_datetime(extra_rows["GAME_DATE"], errors="coerce") <= cutoff
            ]
        filtered_df = pd.concat([filtered_df, extra_rows], ignore_index=True)
        filtered_df = filtered_df.drop_duplicates(keep="first")

    sort_cols = [c for c in ["GAME_DATE", "GAME_ID"] if c in filtered_df.columns]
    if sort_cols:
        filtered_df = filtered_df.sort_values(sort_cols).reset_index(drop=True)

    return filtered_df
