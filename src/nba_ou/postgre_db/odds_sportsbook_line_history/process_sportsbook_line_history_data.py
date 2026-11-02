
import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_cleaned_games_for_odds,
)

MARKET_TOTALS = "totals"
MARKET_MONEYLINE = "money_line"
MARKET_SPREAD = "point_spread"

LINE_HISTORY_MARKETS = [MARKET_TOTALS, MARKET_MONEYLINE, MARKET_SPREAD]
LINE_HISTORY_COLUMNS = [
    "game_date",
    "season_year",
    "event_id",
    "start_time",
    "matchup_url",
    "line_history_url",
    "team_away",
    "team_home",
    "bookmaker",
    "bookmaker_slug",
    "market",
    "row_kind",
    "change_order",
    "timestamp_raw",
    "timestamp",
    "left_label",
    "right_label",
    "left_value_raw",
    "right_value_raw",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]

MARKET_ALIASES = {
    "total": MARKET_TOTALS,
    "totals": MARKET_TOTALS,
    "over_under": MARKET_TOTALS,
    "over-under": MARKET_TOTALS,
    "moneyline": MARKET_MONEYLINE,
    "money_line": MARKET_MONEYLINE,
    "money-line": MARKET_MONEYLINE,
    "ml": MARKET_MONEYLINE,
    "spread": MARKET_SPREAD,
    "point_spread": MARKET_SPREAD,
    "point-spread": MARKET_SPREAD,
    "pointspread": MARKET_SPREAD,
}


def _normalize_sbr_team_name(value: object) -> object:
    if pd.isna(value):
        return value

    name = " ".join(str(value).strip().split())
    candidates = [
        name,
        name.title(),
        name.upper(),
        name.replace("LA ", "L.A. ", 1),
        name.title().replace("La ", "L.A. ", 1),
    ]

    for candidate in candidates:
        mapped = TEAM_NAME_STANDARDIZATION.get(candidate)
        if mapped is None and candidate in TEAM_NAME_STANDARDIZATION:
            raise RuntimeError(f"Team name maps to None: {candidate}")
        if mapped:
            return mapped

    raise RuntimeError(f"Unrecognized SBR team name: {name}")


def build_games_home_away_for_line_history(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "game_season_year",
                "team_home",
                "team_away",
            ]
        )

    df = games_df.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["team_name"] = (
        df["team_name"].map(TEAM_NAME_STANDARDIZATION).fillna(df["team_name"])
    )

    is_home = df["home"].astype("boolean")
    home = df[is_home.fillna(False)].rename(
        columns={"team_name": "team_home", "season_year": "game_season_year"}
    )
    away = df[(~is_home).fillna(False)].rename(columns={"team_name": "team_away"})

    home_cols = ["game_id", "game_date", "game_season_year", "team_home"]
    away_cols = ["game_id", "game_date", "team_away"]
    games_ha = home[home_cols].merge(away[away_cols], on=["game_id", "game_date"])
    return games_ha.drop_duplicates(
        subset=["game_date", "team_home", "team_away"],
        keep="first",
    )


def load_games_for_line_history_creation() -> pd.DataFrame:
    games_df = load_cleaned_games_for_odds(season_year=None)
    if games_df is None:
        return pd.DataFrame()
    return games_df
