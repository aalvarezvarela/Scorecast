from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg
import requests
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_games
from nba_ou.postgre_db.game_time_index.create_db.create_game_time_index_db import (
    GAME_TIME_INDEX_SOURCE,
    create_game_time_index_table,
    get_game_time_index_insert_columns,
    load_game_time_index_for_season,
    upsert_game_time_index_df,
)
from psycopg import sql
from tqdm import tqdm

NBA_BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"


@dataclass(slots=True)
class GameTimeSyncSummary:
    season: str
    season_year: int
    source_games: int
    games_needing_fetch: int
    fetched: int
    failed: int
    upserted: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def parse_season_year(season: str | int) -> int:
    if isinstance(season, int):
        return season

    normalized = str(season).strip()
    if normalized.isdigit():
        return int(normalized)

    match = re.fullmatch(r"(\d{4})\s*-\s*(\d{2}|\d{4})", normalized)
    if not match:
        raise ValueError(
            "Season must look like 2024 or 2024-25 or 2024-2025."
        )

    season_year = int(match.group(1))
    end_raw = match.group(2)
    expected_end_year = season_year + 1

    if len(end_raw) == 2:
        if int(end_raw) != expected_end_year % 100:
            raise ValueError(f"Inconsistent season label: {season}")
    elif int(end_raw) != expected_end_year:
        raise ValueError(f"Inconsistent season label: {season}")

    return season_year


def format_season_label(season_year: int) -> str:
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def _get_game_payload(payload: dict[str, Any]) -> dict[str, Any]:
    game = payload.get("game")
    if not isinstance(game, dict):
        return {}
    return game


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, "", "null"):
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _parse_timestamp_text(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value).strip()


def _parse_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_game_index_record(
    payload: dict[str, Any],
    *,
    fallback_game_id: str,
    fallback_game_date: Any,
    season_year: int,
) -> dict[str, Any]:
    game = _get_game_payload(payload)
    arena = game.get("arena") if isinstance(game.get("arena"), dict) else {}
    home_team = (
        game.get("homeTeam") if isinstance(game.get("homeTeam"), dict) else {}
    )
    away_team = (
        game.get("awayTeam") if isinstance(game.get("awayTeam"), dict) else {}
    )
    home_team_stats = (
        home_team.get("statistics")
        if isinstance(home_team.get("statistics"), dict)
        else None
    )
    away_team_stats = (
        away_team.get("statistics")
        if isinstance(away_team.get("statistics"), dict)
        else None
    )

    return {
        "game_id": str(game.get("gameId") or fallback_game_id).strip(),
        "game_date": pd.to_datetime(fallback_game_date, errors="coerce").date(),
        "season_year": season_year,
        "game_time_local": _parse_timestamp_text(game.get("gameTimeLocal")),
        "game_time_utc": _parse_timestamp(game.get("gameTimeUTC")),
        "game_time_home": _parse_timestamp_text(game.get("gameTimeHome")),
        "game_time_away": _parse_timestamp_text(game.get("gameTimeAway")),
        "game_et": _parse_timestamp_text(game.get("gameEt")),
        "duration": _parse_int(game.get("duration")),
        "game_code": game.get("gameCode"),
        "game_status_text": game.get("gameStatusText"),
        "game_status": _parse_int(game.get("gameStatus")),
        "regulation_periods": _parse_int(game.get("regulationPeriods")),
        "period": _parse_int(game.get("period")),
        "game_clock": game.get("gameClock"),
        "attendance": _parse_int(game.get("attendance")),
        "sellout": game.get("sellout"),
        "arena_id": _parse_int(arena.get("arenaId")),
        "arena_name": arena.get("arenaName"),
        "arena_city": arena.get("arenaCity"),
        "arena_state": arena.get("arenaState"),
        "arena_country": arena.get("arenaCountry"),
        "arena_timezone": arena.get("arenaTimezone"),
        "home_team_id": _parse_int(home_team.get("teamId")),
        "home_team_name": home_team.get("teamName"),
        "home_team_city": home_team.get("teamCity"),
        "home_team_tricode": home_team.get("teamTricode"),
        "home_team_score": _parse_int(home_team.get("score")),
        "away_team_id": _parse_int(away_team.get("teamId")),
        "away_team_name": away_team.get("teamName"),
        "away_team_city": away_team.get("teamCity"),
        "away_team_tricode": away_team.get("teamTricode"),
        "away_team_score": _parse_int(away_team.get("score")),
        "home_team_statistics": home_team_stats,
        "away_team_statistics": away_team_stats,
        "game_payload": game if game else None,
        "source": GAME_TIME_INDEX_SOURCE,
    }


def load_distinct_games_for_season(
    season_year: int,
    conn: psycopg.Connection | None = None,
) -> pd.DataFrame:
    schema = get_schema_name_games()
    table = schema
    close_conn = False

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    try:
        query_obj = sql.SQL("""
            SELECT
                game_id,
                season_year,
                MAX(game_date) AS game_date
            FROM {}.{}
            WHERE season_year = %s
            GROUP BY game_id, season_year
            ORDER BY game_date, game_id
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table),
        )
        query = query_obj.as_string(conn)
        df = pd.read_sql_query(query, conn, params=(season_year,))
    finally:
        if close_conn:
            conn.close()

    if df.empty:
        return df

    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype(
        "Int64"
    )
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    return df


def _select_games_to_fetch(
    source_games_df: pd.DataFrame,
    existing_game_times_df: pd.DataFrame,
    *,
    refresh_all: bool,
) -> pd.DataFrame:
    if source_games_df.empty:
        return source_games_df.copy()

    if refresh_all or existing_game_times_df.empty:
        return source_games_df.copy()

    existing_cols = [
        "game_id",
        "game_time_utc",
        "home_team_statistics",
        "away_team_statistics",
        "game_status",
    ]
    existing = existing_game_times_df[existing_cols].copy()
    existing["game_id"] = existing["game_id"].astype(str).str.strip()
    existing["game_time_utc"] = pd.to_datetime(
        existing["game_time_utc"],
        errors="coerce",
        utc=True,
    )

    merged = source_games_df.merge(existing, on="game_id", how="left")
    missing_mask = (
        merged["game_time_utc"].isna()
        | merged["home_team_statistics"].isna()
        | merged["away_team_statistics"].isna()
        | merged["game_status"].isna()
    )
    return merged[missing_mask][source_games_df.columns].copy()


def _fetch_game_payload(
    game_id: str,
    *,
    session: requests.Session,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(NBA_BOXSCORE_URL.format(game_id), timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(0.5 * attempt, 2.0))

    if last_error is not None:
        raise last_error
    return None


def fetch_game_times_for_games(
    games_df: pd.DataFrame,
    *,
    timeout: int,
    sleep_seconds: float,
    max_retries: int,
    strict_fetch: bool,
) -> tuple[pd.DataFrame, int]:
    if games_df.empty:
        return pd.DataFrame(columns=get_game_time_index_insert_columns()), 0

    session = requests.Session()
    records: list[dict[str, object]] = []
    failed = 0

    for row in tqdm(
        games_df.itertuples(index=False),
        total=len(games_df),
        desc="Fetching NBA game times",
        unit="game",
    ):
        try:
            payload = _fetch_game_payload(
                str(row.game_id),
                session=session,
                timeout=timeout,
                max_retries=max_retries,
            )
            records.append(
                extract_game_index_record(
                    payload,
                    fallback_game_id=str(row.game_id),
                    fallback_game_date=row.game_date,
                    season_year=int(row.season_year),
                )
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:
            failed += 1
            message = f"Failed to fetch game payload for game_id={row.game_id}: {exc}"
            if strict_fetch:
                raise RuntimeError(message) from exc
            print(message)

    return pd.DataFrame.from_records(records), failed


def _build_season_snapshot_for_upsert(
    source_games_df: pd.DataFrame,
    existing_game_times_df: pd.DataFrame,
    fetched_game_times_df: pd.DataFrame,
) -> pd.DataFrame:
    out = source_games_df.copy()
    key_cols = ["game_id", "game_date", "season_year"]
    value_cols = [
        col for col in get_game_time_index_insert_columns() if col not in key_cols
    ]

    if existing_game_times_df.empty:
        for col in value_cols:
            out[col] = None
        out["source"] = GAME_TIME_INDEX_SOURCE
    else:
        existing = existing_game_times_df.copy()
        existing["game_id"] = existing["game_id"].astype(str).str.strip()
        out = out.merge(
            existing.drop(columns=["game_date", "season_year"], errors="ignore"),
            on="game_id",
            how="left",
        )
        for col in value_cols:
            if col not in out.columns:
                out[col] = None

    if not fetched_game_times_df.empty:
        fetched_rename = {
            col: f"fetched_{col}" for col in value_cols if col in fetched_game_times_df
        }
        fetched = fetched_game_times_df.rename(columns=fetched_rename)
        out = out.merge(
            fetched[["game_id", *fetched_rename.values()]],
            on="game_id",
            how="left",
        )
        for col in value_cols:
            fetched_col = f"fetched_{col}"
            if fetched_col in out.columns:
                out[col] = out[fetched_col].combine_first(out[col])
                out = out.drop(columns=[fetched_col])

    out["source"] = out["source"].fillna(GAME_TIME_INDEX_SOURCE)
    return out[get_game_time_index_insert_columns()]


def sync_game_time_index_for_season(
    season: str | int,
    *,
    refresh_all: bool = False,
    timeout: int = 20,
    sleep_seconds: float = 0.05,
    max_retries: int = 3,
    strict_fetch: bool = False,
    limit: int | None = None,
) -> GameTimeSyncSummary:
    season_year = parse_season_year(season)
    season_label = format_season_label(season_year)

    conn = connect_nba_db()
    try:
        if not create_game_time_index_table(conn=conn):
            raise RuntimeError("Unable to create the game-time index table.")

        source_games_df = load_distinct_games_for_season(season_year, conn=conn)
        if source_games_df.empty:
            raise ValueError(f"No games found in nba_games for season {season_label}.")

        existing_game_times_df = load_game_time_index_for_season(season_year, conn=conn)
        games_to_fetch_df = _select_games_to_fetch(
            source_games_df,
            existing_game_times_df,
            refresh_all=refresh_all,
        )
        if limit is not None:
            games_to_fetch_df = games_to_fetch_df.head(limit).copy()

        fetched_game_times_df, failed = fetch_game_times_for_games(
            games_to_fetch_df,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            strict_fetch=strict_fetch,
        )

        season_snapshot_df = _build_season_snapshot_for_upsert(
            source_games_df,
            existing_game_times_df,
            fetched_game_times_df,
        )
        upserted = upsert_game_time_index_df(season_snapshot_df, conn=conn)

        return GameTimeSyncSummary(
            season=season_label,
            season_year=season_year,
            source_games=len(source_games_df),
            games_needing_fetch=len(games_to_fetch_df),
            fetched=len(fetched_game_times_df),
            failed=failed,
            upserted=upserted,
        )
    finally:
        conn.close()
