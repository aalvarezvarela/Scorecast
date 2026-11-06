"""Step 1: expected pressers (two per final game) from the official schedule."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.manifest import (
    entry_key,
    load_manifest,
    merge_entries,
    new_manifest,
)
from lab.game_reports.season_reports import (
    SeasonConfig,
    atomic_write,
    json_bytes,
    season_games,
)

# Regular-season game counts the schedule must reproduce once a season is over.
REGULAR_SEASON_GAMES = {2019: 1059, 2020: 1080}
DEFAULT_REGULAR_SEASON_GAMES = 1230


def schedule_path(config: PresserConfig):
    return config.raw_dir / "schedule" / f"season={config.season}.json"


def season_is_over(payload: dict) -> bool:
    games = [
        g
        for d in payload.get("leagueSchedule", {}).get("gameDates", [])
        for g in d.get("games", [])
        if str(g.get("gameId", "")).startswith(("002", "004", "005", "006"))
    ]
    finals = [g for g in games if str(g["gameId"]).startswith("004")]
    return bool(games) and bool(finals) and all(g.get("gameStatus") == 3 for g in games)


def load_schedule(config: PresserConfig, *, refresh: bool = False) -> dict:
    """Cached ScheduleLeagueV2 payload; a finished season is never fetched again."""
    path = schedule_path(config)
    if path.is_file() and not refresh:
        cached = json.loads(path.read_bytes())
        if season_is_over(cached):
            return cached
    from nba_api.stats.endpoints import ScheduleLeagueV2

    payload = ScheduleLeagueV2(
        league_id="00", season=config.season, timeout=60
    ).get_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(payload))
    return payload


def expected_games(payload: dict, config: PresserConfig) -> dict[str, dict]:
    """One row per (game, team) in the configured phases."""
    games = season_games(
        payload,
        SeasonConfig(
            config.season_year,
            config.root,
            config.bucket or "local",
            phases=config.phases,
        ),
    )
    raw = {
        str(g["gameId"]): g
        for d in payload["leagueSchedule"]["gameDates"]
        for g in d.get("games", [])
    }
    rows = {}
    for game_id, game in games.items():
        source = raw[game_id]
        away, home = source.get("awayTeam") or {}, source.get("homeTeam") or {}
        codes = away.get("teamTricode"), home.get("teamTricode")
        if not all(codes):
            continue  # a playoff slot whose teams are not known yet
        playoff_game_no = int(game_id[-1]) if game["phase"] == "playoffs" else None
        for side, team, other in (("away", away, home), ("home", home, away)):
            row = {
                "game_id": game_id,
                "team": team["teamTricode"],
                "opponent": other["teamTricode"],
                "is_home": side == "home",
                "date": game["date"],
                "tip_utc": source.get("gameDateTimeUTC"),
                "phase": game["phase"],
                "playoff_game_no": playoff_game_no,
                "game_status": source.get("gameStatus"),
                "team_score": (
                    team.get("score") if source.get("gameStatus") == 3 else None
                ),
                "opponent_score": (
                    other.get("score") if source.get("gameStatus") == 3 else None
                ),
            }
            rows[entry_key(game_id, row["team"])] = row
    return dict(sorted(rows.items(), key=lambda kv: (kv[1]["date"], kv[0])))


def count_warnings(
    payload: dict, games: dict[str, dict], config: PresserConfig
) -> list:
    """Regular-season count check (two rows per game) for a finished season."""
    if not season_is_over(payload) or "regular_season" not in config.phases:
        return []
    regular = {g["game_id"] for g in games.values() if g["phase"] == "regular_season"}
    expected = REGULAR_SEASON_GAMES.get(
        config.season_year, DEFAULT_REGULAR_SEASON_GAMES
    )
    if len(regular) != expected:
        return [f"{len(regular)} regular-season games, expected {expected}"]
    return []


def prepare_manifest(config: PresserConfig, payload: dict | None = None) -> dict:
    """Refresh the inventory of expected pressers, keeping saved state."""
    payload = payload if payload is not None else load_schedule(config)
    games = expected_games(payload, config)
    old = load_manifest(config)
    manifest = new_manifest(config, merge_entries(old, games, config), old)
    manifest["warnings"] = count_warnings(payload, games, config)
    phases = Counter(g["phase"] for g in games.values() if g["game_status"] == 3)
    manifest["final_games_by_phase"] = {k: v // 2 for k, v in sorted(phases.items())}
    return manifest


def season_dates(manifest: dict) -> tuple[date, date]:
    dates = [e["date"] for e in manifest["entries"].values()]
    return date.fromisoformat(min(dates)), date.fromisoformat(max(dates))
