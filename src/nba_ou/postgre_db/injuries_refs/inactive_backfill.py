"""Move inactive-player rows already stored in ``nba_players`` to ``nba_injuries``.

Games stored before ingestion split the inactive list out of the box score
(``fetch_data.fetch_nba_data.inactive_rows``) carry it as ``Inactive - ...``
rows of ``nba_players``. This copies them to ``nba_injuries`` in its layout,
checks every one landed, and only then deletes them from ``nba_players``: a
failure in between leaves a row in both tables, never in neither.

Entry point: ``scripts/update_databases/move_inactive_box_rows.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from nba_ou.fetch_data.fetch_nba_data.inactive_rows import (
    inactive_rows_to_injuries,
    is_inactive_comment,
)
from nba_ou.postgre_db.config.db_config import (
    connect_games_db,
    connect_injuries_db,
    connect_players_db,
    get_schema_name_games,
    get_schema_name_injuries,
    get_schema_name_players,
)
from nba_ou.postgre_db.injuries_refs.creation.create_injuries_db import (
    upload_injuries_data_to_db,
)
from psycopg import sql

# nba_players column -> the box-score name inactive_rows_to_injuries reads.
_STORED_TO_BOX = {
    "player_id": "PLAYER_ID",
    "firstname": "firstName",
    "familyname": "familyName",
    "jerseynum": "jerseyNum",
    "team_id": "TEAM_ID",
    "team_city": "TEAM_CITY",
    "team_name": "TEAM_NAME",
    "team_abbreviation": "TEAM_ABBREVIATION",
    "game_id": "GAME_ID",
    "comment": "COMMENT",
}


@dataclass
class InactiveBackfill:
    """The stored inactive rows of a season and their ``nba_injuries`` rows."""

    season_year: int
    player_rows: pd.DataFrame
    injuries: pd.DataFrame

    def describe(self) -> str:
        rows = self.player_rows
        if rows.empty:
            return f"Season {self.season_year}: no inactive rows in nba_players."
        team_games = rows[["GAME_ID", "TEAM_ID"]].drop_duplicates()
        lines = [
            f"Season {self.season_year}: {len(rows)} inactive rows in nba_players, "
            f"{rows['GAME_ID'].nunique()} games, {len(team_games)} team-games "
            f"({len(rows) / len(team_games):.2f} per team-game)",
            "By reason:",
        ]
        for reason, count in rows["COMMENT"].value_counts().items():
            lines.append(f"  {count:5d}  {reason}")
        unmatched = len(rows) - len(self.injuries)
        if unmatched:
            lines.append(f"{unmatched} rows have no game in nba_games and stay put.")
        return "\n".join(lines)


def stored_rows_to_injuries(
    player_rows: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(inactive player rows, their nba_injuries rows)`` from stored frames.

    ``player_rows`` and ``games`` are ``nba_players`` / ``nba_games`` rows with
    the database's lowercase column names.
    """
    rows = player_rows.rename(columns=_STORED_TO_BOX)
    rows = rows.loc[is_inactive_comment(rows["COMMENT"])].reset_index(drop=True)
    games = games.rename(columns=str.upper)
    return rows, inactive_rows_to_injuries(rows, games)


def load_backfill(season_year: int) -> InactiveBackfill:
    """Read the season's inactive ``nba_players`` rows and their games."""
    columns = sql.SQL(", ").join(sql.Identifier(col) for col in _STORED_TO_BOX)
    with connect_players_db() as conn:
        player_rows = pd.read_sql_query(
            sql.SQL(
                "SELECT {} FROM {}.nba_players "
                "WHERE season_year = %s AND comment ILIKE 'inactive%%'"
            )
            .format(columns, sql.Identifier(get_schema_name_players()))
            .as_string(conn),
            conn,
            params=(season_year,),
        )
    with connect_games_db() as conn:
        games = pd.read_sql_query(
            sql.SQL(
                "SELECT DISTINCT game_id, season_id, game_date FROM {}.nba_games "
                "WHERE season_year = %s"
            )
            .format(sql.Identifier(get_schema_name_games()))
            .as_string(conn),
            conn,
            params=(season_year,),
        )
    rows, injuries = stored_rows_to_injuries(player_rows, games)
    return InactiveBackfill(season_year, rows, injuries)


def _stored_injury_keys(game_ids: list[str]) -> set[tuple[str, str]]:
    with connect_injuries_db() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT player_id, game_id FROM {}.nba_injuries "
                "WHERE game_id = ANY(%s)"
            ).format(sql.Identifier(get_schema_name_injuries())),
            (game_ids,),
        )
        return {(str(player), str(game)) for player, game in cur.fetchall()}


def already_in_injuries(backfill: InactiveBackfill) -> int:
    """How many of the rows to move ``nba_injuries`` holds already."""
    if backfill.injuries.empty:
        return 0
    keys = _stored_injury_keys(sorted(backfill.injuries["GAME_ID"].unique()))
    pairs = zip(
        backfill.injuries["PLAYER_ID"], backfill.injuries["GAME_ID"], strict=True
    )
    return sum(pair in keys for pair in pairs)


def execute_backfill(backfill: InactiveBackfill) -> int:
    """Copy the rows to ``nba_injuries``, verify, delete them from ``nba_players``.

    Only rows with a game in ``nba_games`` move. Returns the rows deleted.
    """
    injuries = backfill.injuries
    if injuries.empty:
        return 0
    if not upload_injuries_data_to_db(injuries, if_exists="append"):
        raise RuntimeError("Upload to nba_injuries failed; nba_players untouched.")

    keys = _stored_injury_keys(sorted(injuries["GAME_ID"].unique()))
    moved = list(zip(injuries["PLAYER_ID"], injuries["GAME_ID"], strict=True))
    missing = [pair for pair in moved if pair not in keys]
    if missing:
        raise RuntimeError(
            f"{len(missing)} rows are not in nba_injuries after the upload "
            f"(e.g. {missing[:3]}); nba_players untouched."
        )

    deleted = 0
    with connect_players_db() as conn, conn.cursor() as cur:
        statement = sql.SQL(
            "DELETE FROM {}.nba_players WHERE season_year = %s AND player_id = %s "
            "AND game_id = %s AND comment ILIKE 'inactive%%'"
        ).format(sql.Identifier(get_schema_name_players()))
        for player_id, game_id in moved:
            cur.execute(statement, (backfill.season_year, player_id, game_id))
            deleted += cur.rowcount
        conn.commit()
    return deleted
