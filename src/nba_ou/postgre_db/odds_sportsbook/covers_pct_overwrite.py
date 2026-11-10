"""Replace a season's stored consensus pick percentages with Covers'.

``covers_fill`` only fills the percentage cells SBR left empty. For 2023-24
that is not enough: SBR's day pages of that season carry the percentages for
about 40% of the games, and those they carry are shares of a handful of picks
(14% are exactly 0 or 100), not the Covers Consensus the other seasons hold.
This module overwrites the four columns of every game of a season that has
pick counts in ``line_history_covers.lh_consensus``.

The untouched SBR row is kept in ``odds_sportsbook_sbr_raw`` (copied there
first if a game has none yet), as the close repair does. A later
``repair_closes`` run, ``--redo`` included, does not bring SBR's percentages
back: it restores only the per-book closes from the raw rows.
"""

from __future__ import annotations

import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_line_history_db,
    connect_nba_db,
    get_schema_name_odds_sportsbook,
)
from nba_ou.postgre_db.line_history_covers import store as covers_store
from nba_ou.postgre_db.odds_sportsbook.covers_fill import PCT_COLUMNS, pick_percentages
from nba_ou.postgre_db.odds_sportsbook.repair_closes import (
    RAW_TABLE,
    _column_types,
    _native,
    _read,
    ensure_repair_tables,
)
from psycopg import sql

PCT_COLUMN_NAMES: list[str] = list(PCT_COLUMNS)


def plan_overwrite(rows: pd.DataFrame, percentages: pd.DataFrame) -> pd.DataFrame:
    """The rows whose percentages differ from Covers', with Covers' values.

    ``rows`` are ``odds_sportsbook`` rows (``game_id`` and the four columns),
    ``percentages`` the output of ``pick_percentages``. A game without Covers
    counts is left out, and so is one that already holds Covers' values.
    Returns ``game_id``, the four columns and ``had_sbr_value`` (whether SBR
    had any of them). Pure.
    """
    stored = rows[["game_id", *PCT_COLUMN_NAMES]].copy()
    stored["game_id"] = stored["game_id"].astype(str)
    for column in PCT_COLUMN_NAMES:
        stored[column] = pd.to_numeric(stored[column], errors="coerce")
    covers = percentages.drop_duplicates("game_id").copy()
    covers["game_id"] = covers["game_id"].astype(str)
    for column in PCT_COLUMN_NAMES:
        covers[column] = pd.to_numeric(covers[column], errors="coerce")

    both = stored.merge(covers, on="game_id", suffixes=("_sbr", ""))
    old = both[[f"{c}_sbr" for c in PCT_COLUMN_NAMES]].to_numpy(dtype="float64")
    new = both[PCT_COLUMN_NAMES].to_numpy(dtype="float64")
    same = (old == new) | (pd.isna(old) & pd.isna(new))
    out = both.loc[~same.all(axis=1), ["game_id", *PCT_COLUMN_NAMES]].copy()
    out["had_sbr_value"] = (~pd.isna(old)).any(axis=1)[~same.all(axis=1)]
    return out.reset_index(drop=True)


def overwrite_season_percentages(
    season_year: int,
    *,
    dry_run: bool = True,
    conn: psycopg.Connection | None = None,
    covers_conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    """Write Covers' percentages over the stored ones of ``season_year``.

    Returns the plan (``plan_overwrite``); with ``dry_run`` nothing is written.
    """
    schema = get_schema_name_odds_sportsbook()
    table = schema
    if covers_conn is not None:
        consensus = covers_store.fetch_consensus(covers_conn, [season_year])
    else:
        with connect_line_history_db() as owned_covers:
            consensus = covers_store.fetch_consensus(owned_covers, [season_year])
    percentages = pick_percentages(consensus)

    owned = conn is None
    conn = conn or connect_nba_db()
    try:
        rows = _read(
            conn,
            sql.SQL("SELECT game_id, {} FROM {}.{} WHERE season_year = %s").format(
                sql.SQL(", ").join(map(sql.Identifier, PCT_COLUMN_NAMES)),
                sql.Identifier(schema),
                sql.Identifier(table),
            ),
            (int(season_year),),
        )
        plan = plan_overwrite(rows, percentages)
        if dry_run or plan.empty:
            return plan

        ensure_repair_tables(conn, schema, table)
        game_ids = plan["game_id"].tolist()
        with conn.cursor() as cur:
            columns = sql.SQL(", ").join(
                map(sql.Identifier, _column_types(cur, schema, table))
            )
            cur.execute(
                sql.SQL(
                    "INSERT INTO {schema}.{raw} ({cols}) SELECT {cols} "
                    "FROM {schema}.{tbl} WHERE game_id = ANY(%s) "
                    "ON CONFLICT (game_id) DO NOTHING"
                ).format(
                    schema=sql.Identifier(schema),
                    raw=sql.Identifier(RAW_TABLE),
                    tbl=sql.Identifier(table),
                    cols=columns,
                ),
                (game_ids,),
            )
            cur.executemany(
                sql.SQL("UPDATE {}.{} SET {} WHERE game_id = %s").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.SQL(", ").join(
                        sql.SQL("{} = %s").format(sql.Identifier(c))
                        for c in PCT_COLUMN_NAMES
                    ),
                ),
                [
                    tuple(_native(v) for v in values) + (game_id,)
                    for game_id, values in zip(
                        game_ids,
                        plan[PCT_COLUMN_NAMES].itertuples(index=False, name=None),
                        strict=True,
                    )
                ],
            )
        conn.commit()
        return plan
    except Exception:
        conn.rollback()
        raise
    finally:
        if owned:
            conn.close()
