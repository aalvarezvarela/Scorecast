from pathlib import Path

import pandas as pd

SPREAD_BOOKS: list[str] = [
    "consensus_opener",
    "betmgm",
    "fanduel",
    "caesars",
    "bet365",
    "draftkings",
    "fanatics_sportsbook",
    # From the 2021-22 season only.
    "betrivers",
]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_one_game_row_from_spread_group(g: pd.DataFrame) -> dict:
    """
    g contains 2 rows for one event_id:
      - row_index == 0 is away
      - row_index == 1 is home

    Spread columns are typically:
      - consensus_opener_spread_line / consensus_opener_spread_price
      - {book}_spread_line / {book}_spread_price for books (betmgm, fanduel, ...)

    We output one row per game with away/home splits.
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
        "team_home": team_home,
        "team_away": team_away,
        "home_points": score_home,
        "away_points": score_away,
        "spread_consensus_pct_away": _to_num(away_row["consensus_pct"]).iloc[0]
        if len(away_row)
        else pd.NA,
        "spread_consensus_pct_home": _to_num(home_row["consensus_pct"]).iloc[0]
        if len(home_row)
        else pd.NA,
    }

    # Consensus opener special naming differs from the rest
    # consensus_opener_spread_line / consensus_opener_spread_price
    if "consensus_opener_spread_line" in g.columns:
        out["spread_consensus_opener_line_away"] = (
            _to_num(away_row["consensus_opener_spread_line"]).iloc[0]
            if len(away_row)
            else pd.NA
        )
        out["spread_consensus_opener_line_home"] = (
            _to_num(home_row["consensus_opener_spread_line"]).iloc[0]
            if len(home_row)
            else pd.NA
        )
    else:
        out["spread_consensus_opener_line_away"] = pd.NA
        out["spread_consensus_opener_line_home"] = pd.NA

    if "consensus_opener_spread_price" in g.columns:
        out["spread_consensus_opener_price_away"] = (
            _to_num(away_row["consensus_opener_spread_price"]).iloc[0]
            if len(away_row)
            else pd.NA
        )
        out["spread_consensus_opener_price_home"] = (
            _to_num(home_row["consensus_opener_spread_price"]).iloc[0]
            if len(home_row)
            else pd.NA
        )
    else:
        out["spread_consensus_opener_price_away"] = pd.NA
        out["spread_consensus_opener_price_home"] = pd.NA

    # Other books: {book}_spread_line / {book}_spread_price
    for book in [b for b in SPREAD_BOOKS if b != "consensus_opener"]:
        line_col = f"{book}_spread_line"
        price_col = f"{book}_spread_price"

        out[f"spread_{book}_line_away"] = (
            _to_num(away_row[line_col]).iloc[0]
            if (len(away_row) and line_col in g.columns)
            else pd.NA
        )
        out[f"spread_{book}_price_away"] = (
            _to_num(away_row[price_col]).iloc[0]
            if (len(away_row) and price_col in g.columns)
            else pd.NA
        )

        out[f"spread_{book}_line_home"] = (
            _to_num(home_row[line_col]).iloc[0]
            if (len(home_row) and line_col in g.columns)
            else pd.NA
        )
        out[f"spread_{book}_price_home"] = (
            _to_num(home_row[price_col]).iloc[0]
            if (len(home_row) and price_col in g.columns)
            else pd.NA
        )

    return out


def load_one_day_spread_csv(day_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    """
    Reads one spread CSV (for a given date) or accepts a pre-loaded DataFrame.
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
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in spread input {src_name}: {missing}"
        )

    # Normalize types
    df["row_index"] = pd.to_numeric(df["row_index"], errors="coerce").astype("Int64")
    df["event_id"] = df["event_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    rows = [
        build_one_game_row_from_spread_group(g)
        for _, g in df.groupby("event_id", sort=False)
    ]
    out = pd.DataFrame(rows)

    first_cols = [
        "game_date",
        "game_id",
        "season_year",
        "team_home",
        "team_away",
        "home_points",
        "away_points",
        "spread_consensus_pct_home",
        "spread_consensus_pct_away",
        "spread_consensus_opener_line_home",
        "spread_consensus_opener_line_away",
        "spread_consensus_opener_price_home",
        "spread_consensus_opener_price_away",
    ]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + sorted(other_cols)]

    return out


if __name__ == "__main__":
    # Example usage:
    spread_game_df = load_one_day_spread_csv(
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/sbr_totals_full_game/2024/csv_spread/2024-10-23.csv"
    )
    spread_game_df
