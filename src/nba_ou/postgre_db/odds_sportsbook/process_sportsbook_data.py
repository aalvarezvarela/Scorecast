from pathlib import Path

import pandas as pd

# Team name standardization map
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_sportsbook.process_money_line_data import (
    load_one_day_moneyline_csv,
)
from nba_ou.fetch_data.odds_sportsbook.process_spread_data import (
    load_one_day_spread_csv,
)
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    load_one_day_totals_csv,
    merge_sportsbook_with_games,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from tqdm import tqdm


def _safe_read_day(
    loader_fn,
    csv_path: Path,
    label: str,
) -> pd.DataFrame:
    try:
        return loader_fn(csv_path)
    except Exception as e:
        raise RuntimeError(f"[{label}] Failed on file: {csv_path}") from e


def _drop_duplicate_core_cols(
    df: pd.DataFrame, cols_to_drop: list[str]
) -> pd.DataFrame:
    drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=drop) if drop else df


def merge_daily_frames(
    totals_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    ml_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the 3 one-row-per-game daily frames into a single daily frame.
    De-duplicate shared core columns and keep all betting columns.
    """
    core_cols = [
        "game_date",
        "season_year",
        "team_home",
        "team_away",
        "home_points",
        "away_points",
        "total_points",
    ]

    # Ensure game_id exists
    for name, d in [("totals", totals_df), ("spread", spread_df), ("moneyline", ml_df)]:
        if "game_id" not in d.columns:
            raise ValueError(f"{name} df missing 'game_id'")

    # Use totals as the "base" because it already has total_points
    out = totals_df.copy()

    # Merge spread, but drop overlapping core cols to avoid duplicate columns
    spread_trim = _drop_duplicate_core_cols(spread_df, cols_to_drop=core_cols)
    out = out.merge(spread_trim, on="game_id", how="outer", validate="one_to_one")

    # Merge moneyline, drop overlapping core cols
    ml_trim = _drop_duplicate_core_cols(ml_df, cols_to_drop=core_cols)
    out = out.merge(ml_trim, on="game_id", how="outer", validate="one_to_one")

    return out


def build_master_lines_df(
    seasons_root_dir: str | Path,
    season_dir_glob: str = "*",
    strict_triplet: bool = True,
) -> pd.DataFrame:
    """
    Iterates seasons root directory.
    For each season folder:
      - expects csv/, csv_spread/, csv_moneyline/
      - expects matching filenames across those folders (YYYY-MM-DD.csv)
      - loads per-day totals/spread/moneyline frames (one row per game) and merges on game_id
    Returns:
      - master_df: concatenated across all seasons/days
      - audit_df: per-file audit (missing files, row counts, etc.)

    strict_triplet:
      - True: only process dates that exist in ALL 3 folders
      - False: process union of dates; missing ones become empty frames for that leg
    """
    seasons_root_dir = Path(seasons_root_dir)
    if not seasons_root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {seasons_root_dir}")

    all_daily_merged: list[pd.DataFrame] = []

    season_dirs = sorted(
        [p for p in seasons_root_dir.glob(season_dir_glob) if p.is_dir()]
    )

    for season_dir in season_dirs:
        print(f"Processing season folder: {season_dir.name}")

        totals_dir = season_dir / "csv"
        spread_dir = season_dir / "csv_spread"
        ml_dir = season_dir / "csv_moneyline"

        if not totals_dir.exists() or not spread_dir.exists() or not ml_dir.exists():
            continue

        totals_files = {p.name: p for p in totals_dir.glob("*.csv")}
        spread_files = {p.name: p for p in spread_dir.glob("*.csv")}
        ml_files = {p.name: p for p in ml_dir.glob("*.csv")}

        if strict_triplet:
            day_names = sorted(
                set(totals_files).intersection(spread_files).intersection(ml_files)
            )
        else:
            day_names = sorted(set(totals_files).union(spread_files).union(ml_files))

        for day_name in tqdm(day_names, desc="  Processing days", unit="day"):
            t_path = totals_files.get(day_name)
            s_path = spread_files.get(day_name)
            m_path = ml_files.get(day_name)

            if strict_triplet and (t_path is None or s_path is None or m_path is None):
                continue

            totals_df = (
                _safe_read_day(load_one_day_totals_csv, t_path, "totals")
                if t_path is not None
                else pd.DataFrame(columns=["game_id"])
            )
            spread_df = (
                _safe_read_day(load_one_day_spread_csv, s_path, "spread")
                if s_path is not None
                else pd.DataFrame(columns=["game_id"])
            )
            ml_df = (
                _safe_read_day(load_one_day_moneyline_csv, m_path, "moneyline")
                if m_path is not None
                else pd.DataFrame(columns=["game_id"])
            )

            merged = merge_daily_frames(totals_df, spread_df, ml_df)

            all_daily_merged.append(merged)

    if not all_daily_merged:
        master_df = pd.DataFrame()
    else:
        master_df = pd.concat(all_daily_merged, ignore_index=True)

    # -------------------------
    # REQUIRED POST-PROCESSING
    # -------------------------

    # game_id as string + prefix "00"
    master_df["game_id"] = "00" + master_df["game_id"].astype(str)

    # remove rows with missing team names first so we don't raise on NaNs
    master_df = master_df.dropna(subset=["team_home", "team_away"])

    # normalize team names using TEAM_NAME_STANDARDIZATION
    def _normalize_team(name: object) -> str:
        if pd.isna(name):
            return name
        n = str(name).strip()
        if n in TEAM_NAME_STANDARDIZATION:
            mapped = TEAM_NAME_STANDARDIZATION[n]
            if mapped is None:
                print(f"Team name maps to None in TEAM_NAME_STANDARDIZATION: {n}")
                raise RuntimeError(f"Team name maps to None: {n}")
            return mapped
        # not found — print and raise
        print(f"Unrecognized team name: {n}")
        raise RuntimeError(f"Unrecognized team name: {n}")

    master_df["team_home"] = master_df["team_home"].map(_normalize_team)
    master_df["team_away"] = master_df["team_away"].map(_normalize_team)

    print(
        f"Final merged dataframe: {master_df.shape[0]} rows × {master_df.shape[1]} columns"
    )

    return master_df


def nan_percentage_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage of NaN values per column.

    Returns a dataframe with:
      - column
      - nan_count
      - nan_pct (0–100)
    """
    total_rows = len(df)

    nan_count = df.isna().sum()
    nan_pct = (nan_count / total_rows * 100).round(2)

    out = (
        pd.DataFrame(
            {
                "column": nan_count.index,
                "nan_count": nan_count.values,
                "nan_pct": nan_pct.values,
            }
        )
        .sort_values("nan_pct", ascending=False)
        .reset_index(drop=True)
    )

    return out


if __name__ == "__main__":
    # -------------------------
    # Example usage
    # -------------------------
    seasons_root = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/sbr_totals_full_game"
    master_df = build_master_lines_df(
        seasons_root_dir=seasons_root,
        strict_triplet=True,
    )

    games_df = load_games_from_db()
    master_df = merge_sportsbook_with_games(master_df, games_df)

    nan_stats = nan_percentage_by_column(master_df)
    print("NaN percentage by column:")
    nan_stats
