from pathlib import Path

import pandas as pd
import psycopg
from nba_ou.config.odds_columns import is_book_column
from nba_ou.fetch_data.odds_sportsbook.process_money_line_data import ML_BOOKS
from nba_ou.fetch_data.odds_sportsbook.process_spread_data import SPREAD_BOOKS
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import TOTAL_BOOKS
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_sportsbook,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.odds_sportsbook.process_sportsbook_data import (
    build_master_lines_df,
    merge_sportsbook_with_games,
)
from psycopg import sql


def schema_exists(schema_name: str | None = None) -> bool:
    """Check if the schema exists in the database."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_odds_sportsbook()

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


def create_odds_sportsbook_schema_if_not_exists(
    conn: psycopg.Connection, schema: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def _sportsbook_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("game_id", "TEXT NOT NULL"),
        ("game_date", "DATE NOT NULL"),
        ("season_year", "INTEGER NOT NULL"),
        ("team_home", "VARCHAR(100) NOT NULL"),
        ("team_away", "VARCHAR(100) NOT NULL"),
        ("home_points", "NUMERIC(8, 2)"),
        ("away_points", "NUMERIC(8, 2)"),
        ("total_points", "NUMERIC(8, 2)"),
        ("total_consensus_pct_over", "NUMERIC(8, 4)"),
        ("total_consensus_pct_under", "NUMERIC(8, 4)"),
        ("spread_consensus_pct_away", "NUMERIC(8, 4)"),
        ("spread_consensus_pct_home", "NUMERIC(8, 4)"),
        ("spread_consensus_opener_line_away", "NUMERIC(8, 4)"),
        ("spread_consensus_opener_line_home", "NUMERIC(8, 4)"),
        ("spread_consensus_opener_price_away", "NUMERIC(10, 4)"),
        ("spread_consensus_opener_price_home", "NUMERIC(10, 4)"),
    ]

    for book in TOTAL_BOOKS:
        columns.append((f"total_{book}_line_over", "NUMERIC(8, 4)"))
        columns.append((f"total_{book}_price_over", "NUMERIC(10, 4)"))
        columns.append((f"total_{book}_line_under", "NUMERIC(8, 4)"))
        columns.append((f"total_{book}_price_under", "NUMERIC(10, 4)"))

    for book in [b for b in SPREAD_BOOKS if b != "consensus_opener"]:
        columns.append((f"spread_{book}_line_away", "NUMERIC(8, 4)"))
        columns.append((f"spread_{book}_price_away", "NUMERIC(10, 4)"))
        columns.append((f"spread_{book}_line_home", "NUMERIC(8, 4)"))
        columns.append((f"spread_{book}_price_home", "NUMERIC(10, 4)"))

    for book in ML_BOOKS:
        columns.append((f"ml_{book}_price_away", "NUMERIC(10, 4)"))
        columns.append((f"ml_{book}_price_home", "NUMERIC(10, 4)"))

    return columns


def book_columns(books: list[str] | tuple[str, ...]) -> list[str]:
    """Every stored column belonging to ``books`` (totals, spread and moneyline)."""
    return [
        name
        for name, _ in _sportsbook_columns()
        if any(is_book_column(name, book) for book in books)
    ]


def _add_missing_columns(cur: psycopg.Cursor, schema: str, table: str) -> None:
    """Bring an existing table up to ``_sportsbook_columns``.

    ``CREATE TABLE IF NOT EXISTS`` never touches a table that is already there,
    so a book added to the scraper would otherwise have nowhere to be written.

    Only columns that are actually missing are altered: every ``ALTER TABLE``
    takes an ACCESS EXCLUSIVE lock even when ``IF NOT EXISTS`` makes it a no-op,
    so issuing one per column on every run would queue behind any open reader.
    A lock timeout makes a blocked migration fail loudly instead of hanging.
    """
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s",
        (schema, table),
    )
    existing = {row[0] for row in cur.fetchall()}
    missing = [(n, t) for n, t in _sportsbook_columns() if n not in existing]
    if not missing:
        return

    print(f"Adding {len(missing)} column(s) to {schema}.{table}")
    cur.execute("SET LOCAL lock_timeout = '30s'")
    for name, dtype in missing:
        cur.execute(
            sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} {}").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(name),
                sql.SQL(dtype),
            )
        )


def create_odds_sportsbook_table(drop_existing: bool = False) -> bool:
    """Create the odds_sportsbook table inside schema SCHEMA_NAME_ODDS_SPORTSBOOK."""
    try:
        schema = get_schema_name_odds_sportsbook()
        table = schema
        conn = connect_nba_db()

        create_odds_sportsbook_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            column_defs = _sportsbook_columns()
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
            _add_missing_columns(cur, schema, table)

            cur.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{}(game_date, team_home, team_away)"
                ).format(
                    sql.Identifier("idx_odds_sportsbook_unique_game"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_date)").format(
                    sql.Identifier("idx_odds_sportsbook_game_date"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_home)").format(
                    sql.Identifier("idx_odds_sportsbook_team_home"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_away)").format(
                    sql.Identifier("idx_odds_sportsbook_team_away"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.{table}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating odds_sportsbook table: {e}")
        return False


def load_all_games_from_db() -> pd.DataFrame:
    df = load_games_from_db(seasons=None)
    if df is None:
        return pd.DataFrame()

    if "season_type" in df.columns:
        df = df[~df["season_type"].isin(["All Star", "Preseason"])]

    return df


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def build_upsert_query(
    schema: str, table: str, cols: list[str], fill_columns: list[str] | None = None
) -> sql.Composed:
    """INSERT for new games; optionally fill-only UPDATE for existing ones.

    Without ``fill_columns`` an existing game is left untouched (the original
    behaviour). With them, an existing row gets *only* those columns, and only
    where they are still NULL -- so backfilling a new book can never overwrite a
    value another book already stored.
    """
    if fill_columns:
        unknown = sorted(set(fill_columns) - set(cols))
        if unknown:
            raise ValueError(f"fill_columns not in the table: {unknown}")
        conflict = sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(
                sql.SQL("{col} = COALESCE({tbl}.{col}, EXCLUDED.{col})").format(
                    col=sql.Identifier(col), tbl=sql.Identifier(table)
                )
                for col in fill_columns
            )
        )
    else:
        conflict = sql.SQL("DO NOTHING")

    return sql.SQL(
        """
        INSERT INTO {}.{} (
            {cols}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT (game_id)
        {conflict}
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in cols),
        conflict=conflict,
    )


def upsert_odds_sportsbook_df(
    odds_df: pd.DataFrame,
    conn: psycopg.Connection | None = None,
    *,
    fill_columns: list[str] | None = None,
) -> int:
    if odds_df.empty:
        return 0

    schema = get_schema_name_odds_sportsbook()
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

    col_defs = _sportsbook_columns()
    cols = [c for c, _ in col_defs]
    odds_df = _ensure_columns(odds_df, cols)

    # Convert pd.NA to None for database compatibility
    odds_df = odds_df.where(pd.notna(odds_df), None)

    rows = [tuple(row) for row in odds_df[cols].itertuples(index=False, name=None)]

    insert_query = build_upsert_query(schema, table, cols, fill_columns)

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, rows)
        conn.commit()
        return len(rows)
    finally:
        if close_conn:
            conn.close()


def build_and_load_odds_sportsbook(
    seasons_root_dir: str | Path,
    *,
    strict_triplet: bool = True,
) -> dict[str, object]:
    odds_df = build_master_lines_df(
        seasons_root_dir=seasons_root_dir,
        strict_triplet=strict_triplet,
    )
    games_df = load_all_games_from_db()
    merged_df = merge_sportsbook_with_games(odds_df, games_df)

    null_game_id_count = merged_df["game_id"].isnull().sum()
    print(f"Number of rows with null game_id in merged_df: {null_game_id_count}")
    merged_df = merged_df.dropna(subset=["game_id"])

    inserted = upsert_odds_sportsbook_df(merged_df)

    return {
        "odds_rows": len(odds_df),
        "merged_rows": len(merged_df),
        "inserted_rows": inserted,
    }


if __name__ == "__main__":
    seasons_root = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/"
        "sbr_totals_full_game"
    )
    print("\nStep 1: Creating schema + odds_sportsbook table...")
    if not create_odds_sportsbook_table(False):
        raise SystemExit(1)

    print("\nStep 2: Building sportsbook df and merging with games...")
    results = build_and_load_odds_sportsbook(
        seasons_root_dir=seasons_root,
        strict_triplet=True,
    )
    print(f"Loaded sportsbook rows: {results['odds_rows']}")
    print(f"Merged sportsbook rows: {results['merged_rows']}")
    print(f"Inserted rows: {results['inserted_rows']}")
