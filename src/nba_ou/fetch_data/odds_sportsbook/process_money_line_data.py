from pathlib import Path

import pandas as pd

ML_BOOKS: list[str] = [
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _off_board_to_na(price):
    """SBR writes -10000 for an off-the-board price.

    A book missing from the page arrives as ``pd.NA``, and ``pd.NA == -10000``
    raises rather than returning False, so check for a value first.
    """
    if pd.isna(price) or price == -10000:
        return pd.NA
    return price


def build_one_game_row_from_moneyline_group(g: pd.DataFrame) -> dict:
    """
    g contains 2 rows for one event_id:
      - row_index == 0 is away
      - row_index == 1 is home
    Moneyline columns are *_price (no line column).
    """
    g = g.copy()

    away_row = g.loc[g["row_index"].astype("Int64") == 0].head(1)
    home_row = g.loc[g["row_index"].astype("Int64") == 1].head(1)

    team_away = away_row["team_name"].iloc[0] if len(away_row) else pd.NA
    team_home = home_row["team_name"].iloc[0] if len(home_row) else pd.NA

    score_away = _to_num(away_row["score"]).iloc[0] if len(away_row) else pd.NA
    score_home = _to_num(home_row["score"]).iloc[0] if len(home_row) else pd.NA

    out = {
        "game_date": g["date"].iloc[0],
        "game_id": g["event_id"].iloc[0],
        "season_year": g["season"].iloc[0],
        "team_away": team_away,
        "team_home": team_home,
        "home_points": score_home,
        "away_points": score_away,
    }

    # Split moneyline prices into away/home per book
    for book in ML_BOOKS:
        price_col = f"{book}_price"

        # Get away price and replace -10000 with NA
        away_price = (
            _to_num(away_row[price_col]).iloc[0]
            if (len(away_row) and price_col in g.columns)
            else pd.NA
        )
        out[f"ml_{book}_price_away"] = _off_board_to_na(away_price)

        # Get home price and replace -10000 with NA
        home_price = (
            _to_num(home_row[price_col]).iloc[0]
            if (len(home_row) and price_col in g.columns)
            else pd.NA
        )
        out[f"ml_{book}_price_home"] = _off_board_to_na(home_price)

    return out


def load_one_day_moneyline_csv(day_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    """
    Reads one moneyline CSV (for a given date) or accepts a pre-loaded DataFrame.
    If `day_csv` is a DataFrame it will be used directly (copied); otherwise the
    path will be read as CSV.
    Returns one row per game (event_id).
    """
    # Accept either a DataFrame or a path-like object
    if isinstance(day_csv, pd.DataFrame):
        df = day_csv.copy()
        src_name = "dataframe_input"
    else:
        day_csv_path = Path(day_csv)
        df = pd.read_csv(day_csv_path)
        src_name = day_csv_path.name

    required = ["date", "season", "event_id", "row_index", "team_name", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in moneyline input {src_name}: {missing}"
        )

    # Normalize types
    df["row_index"] = pd.to_numeric(df["row_index"], errors="coerce").astype("Int64")
    df["event_id"] = df["event_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    rows = [
        build_one_game_row_from_moneyline_group(g)
        for _, g in df.groupby("event_id", sort=False)
    ]
    out = pd.DataFrame(rows)

    # Optional: ordering
    first_cols = [
        "game_date",
        "game_id",
        "season_year",
        "team_home",
        "team_away",
        "home_points",
        "away_points",
    ]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + sorted(other_cols)]

    return out


if __name__ == "__main__":
    # Example usage:
    ml_df = load_one_day_moneyline_csv(
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/sbr_totals_full_game/2024/csv_moneyline/2024-10-23.csv"
    )
    ml_df
