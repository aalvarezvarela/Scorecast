from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_yahoo.process_yahoo_day import yahoo_one_row_per_game
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_yahoo,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.odds_sportsbook.process_sportsbook_data import (
    merge_sportsbook_with_games,
)
from psycopg import sql
from tqdm import tqdm


def schema_exists(schema_name: str | None = None) -> bool:
    """Check if the schema exists in the database."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_odds_yahoo()

        conn = connect_nba_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            exists = cur.fetchone()
        conn.close()
        return exists is not None
    except Exception as e:
        print(f"Error checking schema existence: {e}")
        return False


def create_odds_yahoo_schema_if_not_exists(
    conn: psycopg.Connection, schema: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def _yahoo_columns() -> list[tuple[str, str]]:
    return [
        ("game_id", "TEXT NOT NULL"),
        ("game_date", "DATE NOT NULL"),
        ("season_year", "INTEGER NOT NULL"),
        ("team_home", "VARCHAR(100) NOT NULL"),
        ("team_away", "VARCHAR(100) NOT NULL"),
        ("team_home_abbr", "VARCHAR(10)"),
        ("team_away_abbr", "VARCHAR(10)"),
        ("spread_home", "NUMERIC(8, 4)"),
        ("spread_away", "NUMERIC(8, 4)"),
        ("moneyline_home", "NUMERIC(10, 4)"),
        ("moneyline_away", "NUMERIC(10, 4)"),
        ("total_line", "NUMERIC(8, 4)"),
        ("total_pct_bets_over", "NUMERIC(8, 4)"),
        ("total_pct_bets_under", "NUMERIC(8, 4)"),
        ("total_pct_money_over", "NUMERIC(8, 4)"),
        ("total_pct_money_under", "NUMERIC(8, 4)"),
        ("spread_pct_bets_away", "NUMERIC(8, 4)"),
        ("spread_pct_bets_home", "NUMERIC(8, 4)"),
        ("spread_pct_money_away", "NUMERIC(8, 4)"),
        ("spread_pct_money_home", "NUMERIC(8, 4)"),
        ("moneyline_pct_bets_away", "NUMERIC(8, 4)"),
        ("moneyline_pct_bets_home", "NUMERIC(8, 4)"),
        ("moneyline_pct_money_away", "NUMERIC(8, 4)"),
        ("moneyline_pct_money_home", "NUMERIC(8, 4)"),
    ]


def create_odds_yahoo_table(drop_existing: bool = False) -> bool:
    """Create the odds_yahoo table inside schema SCHEMA_NAME_ODDS_YAHOO."""
    try:
        schema = get_schema_name_odds_yahoo()
        table = schema
        conn = connect_nba_db()

        create_odds_yahoo_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            column_defs = _yahoo_columns()
            column_sql = ",\n".join(
                [f"{name} {dtype}" for name, dtype in column_defs]
                + ["PRIMARY KEY (game_id)"]
            )

            create_table_query = sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS {{}}.{{}} (
                    {column_sql}
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(create_table_query)

            cur.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{}(game_date, team_home, team_away)"
                ).format(
                    sql.Identifier("idx_odds_yahoo_unique_game"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_odds_yahoo_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.{table}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating odds_yahoo table: {e}")
        return False


def _normalize_team(name: object) -> object:
    if pd.isna(name):
        return name
    n = str(name).strip()
    if "all-star" in n.lower():
        return None
    if n in TEAM_NAME_STANDARDIZATION:
        mapped = TEAM_NAME_STANDARDIZATION[n]
        if mapped is None:
            print(f"Team name maps to None in TEAM_NAME_STANDARDIZATION: {n}")
            raise RuntimeError(f"Team name maps to None: {n}")
        return mapped
    print(f"Unrecognized team name: {n}")
    raise RuntimeError(f"Unrecognized team name: {n}")


def _iter_csv_paths(
    seasons_root_dir: str | Path, season_dir_glob: str = "*"
) -> list[Path]:
    root = Path(seasons_root_dir)
    if not root.exists():
        return []

    paths: list[Path] = []
    for season_dir in sorted([p for p in root.glob(season_dir_glob) if p.is_dir()]):
        csv_dir = season_dir / "csv"
        if not csv_dir.exists():
            continue
        paths.extend(sorted(csv_dir.glob("*.csv")))

    return paths


def build_master_yahoo_df(
    seasons_root_dir: str | Path,
    season_dir_glob: str = "*",
) -> pd.DataFrame:
    csv_paths = _iter_csv_paths(seasons_root_dir, season_dir_glob=season_dir_glob)
    if not csv_paths:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for csv_path in tqdm(csv_paths, desc="Processing Yahoo odds CSVs", unit="file"):
        df = yahoo_one_row_per_game(csv_path)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    master_df = pd.concat(dfs, ignore_index=True)

    master_df = master_df.drop_duplicates()

    # Keep ingestion filtering aligned with DB schema (`total_line`).
    if "total_line" not in master_df.columns:
        raise ValueError("Expected column 'total_line' in Yahoo master dataframe")
    master_df = master_df.dropna(subset=["total_line"])

    master_df = master_df.dropna(subset=["team_home", "team_away"])
    master_df["team_home"] = master_df["team_home"].map(_normalize_team)
    master_df["team_away"] = master_df["team_away"].map(_normalize_team)

    return master_df.reset_index(drop=True)


def load_all_games_from_db() -> pd.DataFrame:
    df = load_games_from_db(seasons=None)
    if df is None:
        return pd.DataFrame()

    if "season_type" in df.columns:
        season_type = df["season_type"].astype("string").str.lower()
        mask_pre = season_type.str.contains("pre", na=False)
        mask_allstar = season_type.str.contains("all", na=False)
        df = df[~(mask_pre | mask_allstar)]

    return df


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def upsert_odds_yahoo_df(
    odds_df: pd.DataFrame, conn: psycopg.Connection | None = None
) -> int:
    if odds_df.empty:
        return 0

    schema = get_schema_name_odds_yahoo()
    table = schema
    close_conn = False
    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    odds_df = odds_df.copy()
    odds_df["game_id"] = odds_df["game_id"].astype(str)
    odds_df["game_date"] = pd.to_datetime(odds_df["game_date"], errors="coerce").dt.date

    odds_df = odds_df.dropna(
        subset=[
            "game_id",
            "game_date",
            "team_home",
            "team_away",
            "season_year",
        ]
    )

    col_defs = _yahoo_columns()
    cols = [c for c, _ in col_defs]
    odds_df = _ensure_columns(odds_df, cols)

    rows = [tuple(row) for row in odds_df[cols].itertuples(index=False, name=None)]

    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            {cols}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT (game_id)
        DO NOTHING
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in cols),
    )

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, rows)
        conn.commit()
        return len(rows)
    finally:
        if close_conn:
            conn.close()


def build_and_load_odds_yahoo(
    seasons_root_dir: str | Path,
) -> dict[str, object]:
    odds_df = build_master_yahoo_df(seasons_root_dir=seasons_root_dir)
    games_df = load_all_games_from_db()
    merged_df = merge_sportsbook_with_games(odds_df, games_df)

    # Print null counts per column before dropping rows
    null_counts = merged_df.isnull().sum()
    print("Null counts per column in merged_df:")
    total_rows = len(merged_df)
    for col, cnt in null_counts.items():
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0
        print(f"  {col}: {cnt} ({pct:.2f}%)")

    null_game_id_count = null_counts.get("game_id", 0)
    print(f"Number of rows with null game_id in merged_df: {null_game_id_count}")
    merged_df = merged_df.dropna(subset=["game_id"])

    inserted = upsert_odds_yahoo_df(merged_df)

    return {
        "odds_rows": len(odds_df),
        "merged_rows": len(merged_df),
        "inserted_rows": inserted,
    }


if __name__ == "__main__":
    seasons_root = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/yahoo_odds"
    )

    print("\nStep 1: Creating schema + odds_yahoo table...")
    if not create_odds_yahoo_table(False):
        raise SystemExit(1)

    print("\nStep 2: Building yahoo df and merging with games...")
    results = build_and_load_odds_yahoo(seasons_root_dir=seasons_root)
    print(f"Loaded yahoo rows: {results['odds_rows']}")
    print(
        f"Merged yahoo rows: {results['merged_rows']} ({results['merged_rows']/results['odds_rows']*100:.2f}% of loaded)"
    )
    print(
        f"Inserted rows: {results['inserted_rows']} ({results['inserted_rows']/results['merged_rows']*100:.2f}% of merged)"
        if results["merged_rows"] > 0
        else f"Inserted rows: {results['inserted_rows']}"
    )
