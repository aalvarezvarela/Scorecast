from pathlib import Path

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION

TOTAL_BOOKS: list[str] = [
    "consensus_opener",
    "betmgm",
    "fanduel",
    "caesars",
    "bet365",
    "draftkings",
    "fanatics_sportsbook",
    # History only until admitted as a feature: see HISTORY_ONLY_BOOKS.
    "betrivers",
]


def build_games_home_away_df(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.copy()
    df["team_name"] = (
        df["team_name"].map(TEAM_NAME_STANDARDIZATION).fillna(df["team_name"])
    )
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    if "home" not in df:
        return pd.DataFrame(columns=["game_id", "game_date", "team_home", "team_away"])

    home = df[df["home"] == True]
    away = df[df["home"] == False]

    home = home.rename(columns={"team_name": "team_home"})[
        ["game_id", "game_date", "team_home"]
    ]
    away = away.rename(columns={"team_name": "team_away"})[
        ["game_id", "game_date", "team_away"]
    ]

    merged = home.merge(away, on=["game_id", "game_date"], how="inner")
    return merged


def merge_sportsbook_with_games(
    master_df: pd.DataFrame, games_df: pd.DataFrame | None
) -> pd.DataFrame:
    if master_df.empty:
        return master_df

    df = master_df.copy()
    df = df.drop(columns=["game_id"], errors="ignore")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    if games_df is None or games_df.empty:
        print("No games data provided; returning master_df without game_id.")
        return df

    games_ha = build_games_home_away_df(games_df)
    if games_ha.empty:
        print("No home/away games mapping found; returning master_df without game_id.")
        return df

    merged = df.merge(
        games_ha,
        left_on=["game_date", "team_home", "team_away"],
        right_on=["game_date", "team_home", "team_away"],
        how="left",
    )
    return merged


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_one_game_row_from_totals_group(g: pd.DataFrame) -> dict:
    """
    g contains 2 rows for one event_id (away row_index=0, home row_index=1)
    and (typically) 1 row for Over and 1 row for Under identified by consensus_opener_side in {"O","U"}.
    """
    g = g.copy()

    # Identify teams and scores using row_index rule
    away_row = g.loc[g["row_index"].astype("Int64") == 0].head(1)
    home_row = g.loc[g["row_index"].astype("Int64") == 1].head(1)

    team_away = away_row["team_name"].iloc[0] if len(away_row) else pd.NA
    team_home = home_row["team_name"].iloc[0] if len(home_row) else pd.NA

    score_away = _to_num(away_row["score"]).iloc[0] if len(away_row) else pd.NA
    score_home = _to_num(home_row["score"]).iloc[0] if len(home_row) else pd.NA

    total_points = (
        (0 if pd.isna(score_away) else float(score_away))
        + (0 if pd.isna(score_home) else float(score_home))
        if (not pd.isna(score_away) or not pd.isna(score_home))
        else pd.NA
    )

    # Identify Over and Under rows
    over_row = g.loc[g["consensus_opener_side"].astype(str).str.upper() == "O"].head(1)
    under_row = g.loc[g["consensus_opener_side"].astype(str).str.upper() == "U"].head(1)

    out = {
        "game_date": g["date"].iloc[0],
        "game_id": g["event_id"].iloc[0],
        "season_year": g["season"].iloc[0],
        "team_home": team_home,
        "team_away": team_away,
        "home_points": score_home,
        "away_points": score_away,
        "total_points": total_points,
        "total_consensus_pct_over": _to_num(over_row["consensus_pct"]).iloc[0]
        if len(over_row)
        else pd.NA,
        "total_consensus_pct_under": _to_num(under_row["consensus_pct"]).iloc[0]
        if len(under_row)
        else pd.NA,
    }

    # Split line and price per book into over and under
    for book in TOTAL_BOOKS:
        line_col = f"{book}_line"
        price_col = f"{book}_price"

        out[f"total_{book}_line_over"] = (
            _to_num(over_row[line_col]).iloc[0]
            if (len(over_row) and line_col in g.columns)
            else pd.NA
        )
        out[f"total_{book}_price_over"] = (
            _to_num(over_row[price_col]).iloc[0]
            if (len(over_row) and price_col in g.columns)
            else pd.NA
        )

        out[f"total_{book}_line_under"] = (
            _to_num(under_row[line_col]).iloc[0]
            if (len(under_row) and line_col in g.columns)
            else pd.NA
        )
        out[f"total_{book}_price_under"] = (
            _to_num(under_row[price_col]).iloc[0]
            if (len(under_row) and price_col in g.columns)
            else pd.NA
        )

    return out


def load_one_day_totals_csv(day_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    """
    Reads one totals CSV (for a given date) or accepts a pre-loaded DataFrame.
    If `day_csv` is a DataFrame it will be used directly (copied); otherwise the
    path will be read as CSV. Returns one row per game (event_id).
    """
    if isinstance(day_csv, pd.DataFrame):
        df = day_csv.copy()
        src_name = "dataframe_input"
    else:
        day_csv_path = Path(day_csv)
        df = pd.read_csv(day_csv_path)
        src_name = day_csv_path.name

    required = [
        "date",
        "season",
        "event_id",
        "row_index",
        "team_name",
        "score",
        "consensus_pct",
        "consensus_opener_side",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in totals input {src_name}: {missing}"
        )

    # Normalize types
    df["row_index"] = pd.to_numeric(df["row_index"], errors="coerce").astype("Int64")
    df["event_id"] = df["event_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    # Build one row per event_id
    rows = [
        build_one_game_row_from_totals_group(g)
        for _, g in df.groupby("event_id", sort=False)
    ]
    out = pd.DataFrame(rows)

    # Optional: enforce convenient ordering
    first_cols = [
        "game_date",
        "game_id",
        "season_year",
        "team_away",
        "team_home",
        "home_points",
        "away_points",
        "total_points",
        "total_consensus_pct_over",
        "total_consensus_pct_under",
    ]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + sorted(other_cols)]

    return out


if __name__ == "__main__":
    # Example usage (one day):
    totals_game_df = load_one_day_totals_csv(
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/sbr_totals_full_game/2024/csv/2024-10-23.csv"
    )
    totals_game_df
