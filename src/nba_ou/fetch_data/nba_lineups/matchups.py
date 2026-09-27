"""Who guarded whom: BoxScoreMatchupsV3 as one Parquet table per season.

History comes from the ``shufinskiy/nba_data`` archive (``matchups_YYYY`` and
``matchups_po_YYYY``, 2017-18 onwards, when the NBA started tracking
matchups). It republishes the API's rows flattened to snake_case, so a season
costs one ~4 MB download. Games the archive lacks, and every game after its
last refresh, come from the API one call at a time; both sources are rendered
into the same columns here.

**Direction.** Each row is one offensive player against one defender. The
archive's ``person_id`` is the offensive player and ``matchups_person_id`` the
defender, although nba_api's endpoint labels the outer player's ``position``
and ``comment`` as ``positionDef``/``commentDef``. Measured on 2020-21 against
the box score (23,000 player-games): summed per outer player, ``player_points``
correlates 0.97 with that player's points and ``matchup_assists`` 0.96 with
his assists; summed per inner player, only ``matchup_blocks`` tracks the
player's own stats (0.86 with his blocks). The columns are therefore renamed
``off_*`` and ``def_*`` so nothing downstream has to remember which is which.

**Revisions.** The NBA reprocesses tracking after the fact. Checked against the
API on 2026-09-27: games from 2020-21 and 2024-25 match the archive exactly,
but 2025-26 games do not (median 1.9 s per pair, a few pairs added or dropped,
totals within ~1 %). The 2025-26 archive was evidently captured in-season, and
a game fetched the morning after it is played will be preliminary in the same
way. That is the version a model could have seen at the time, so ``source``
records where each game came from rather than papering over the difference.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from nba_ou.fetch_data.nba_lineups.client import (
    EmptyResponse,
    GameUnavailable,
    ServerTimeout,
)
from nba_ou.fetch_data.nba_lineups.pbp_archive import download_season_csv

#: The first season with matchup tracking.
FIRST_SEASON = 2017

PLAYER_FIELDS = ("player_id", "first_name", "family_name", "name_i")

#: Archive column -> our column, for the matchup statistics. The API's
#: camelCase names are the same words, so one table serves both sources.
STAT_COLUMNS = {
    "matchup_minutes_sort": "matchup_seconds",
    "partial_possessions": "partial_possessions",
    "percentage_defender_total_time": "pct_defender_total_time",
    "percentage_offensive_total_time": "pct_offensive_total_time",
    "percentage_total_time_both_on": "pct_total_time_both_on",
    "switches_on": "switches_on",
    "player_points": "player_points",
    "team_points": "team_points",
    "matchup_assists": "matchup_assists",
    "matchup_potential_assists": "matchup_potential_assists",
    "matchup_turnovers": "matchup_turnovers",
    "matchup_blocks": "matchup_blocks",
    "matchup_field_goals_made": "matchup_fgm",
    "matchup_field_goals_attempted": "matchup_fga",
    "matchup_three_pointers_made": "matchup_fg3m",
    "matchup_three_pointers_attempted": "matchup_fg3a",
    "matchup_free_throws_made": "matchup_ftm",
    "matchup_free_throws_attempted": "matchup_fta",
    "shooting_fouls": "shooting_fouls",
    "help_blocks": "help_blocks",
    "help_field_goals_made": "help_fgm",
    "help_field_goals_attempted": "help_fga",
}
FLOAT_STATS = frozenset(
    {
        "matchup_seconds",
        "partial_possessions",
        "pct_defender_total_time",
        "pct_offensive_total_time",
        "pct_total_time_both_on",
    }
)

COLUMNS = (
    "game_id",
    "season_year",
    "season_type",
    "home_team_id",
    "away_team_id",
    "off_team_id",
    *(f"off_{field}" for field in PLAYER_FIELDS),
    "off_position",
    "def_team_id",
    *(f"def_{field}" for field in PLAYER_FIELDS),
    *STAT_COLUMNS.values(),
    "source",
)

SEASON_TYPES = {"002": "Regular Season", "004": "Playoffs"}

STATUS_COLUMNS = ("game_id", "status", "source", "recorded_at")


def _snake(name: str) -> str:
    out = []
    for char in name:
        if char.isupper():
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def season_datasets(season_year: int, *, playoffs: bool = True) -> list[str]:
    names = [f"matchups_{season_year}"]
    if playoffs:
        names.append(f"matchups_po_{season_year}")
    return names


def _game_season(game_id: str) -> int:
    """``0022000001`` -> 2020: digits 4-5 are the season's start year."""
    return 2000 + int(game_id[3:5])


def _finish(frame: pd.DataFrame, season_year: int, source: str) -> pd.DataFrame:
    """Shared tail of both renderers: types, derived columns, order."""
    frame = frame.copy()
    frame["game_id"] = frame["game_id"].map(lambda value: str(int(value)).zfill(10))
    seasons = frame["game_id"].map(_game_season)
    if not seasons.eq(season_year).all():
        wrong = sorted(frame.loc[~seasons.eq(season_year), "game_id"].unique())[:3]
        raise ValueError(f"Games {wrong} do not belong to season {season_year}")
    frame["season_year"] = season_year
    frame["season_type"] = frame["game_id"].str[:3].map(SEASON_TYPES)
    if frame["season_type"].isna().any():
        odd = sorted(frame.loc[frame["season_type"].isna(), "game_id"].unique())[:3]
        raise ValueError(f"Unexpected game types: {odd}")
    for column in ("home_team_id", "away_team_id", "off_team_id",
                   "off_player_id", "def_player_id"):
        frame[column] = frame[column].astype("int64")
    # The defender plays for whichever side is not on offence.
    frame["def_team_id"] = frame["home_team_id"].where(
        frame["off_team_id"].ne(frame["home_team_id"]), frame["away_team_id"]
    )
    for column in STAT_COLUMNS.values():
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frame[column] = values.astype("float64" if column in FLOAT_STATS else "int64")
    for column in ("off_first_name", "off_family_name", "off_name_i", "off_position",
                   "def_first_name", "def_family_name", "def_name_i"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["source"] = source
    return frame[list(COLUMNS)].reset_index(drop=True)


def from_archive_csv(frame: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """Render one archive CSV (regular season or playoffs) into ``COLUMNS``."""
    renamed = {
        "game_id": "game_id",
        "home_team_id": "home_team_id",
        "away_team_id": "away_team_id",
        "team_id": "off_team_id",
        "person_id": "off_player_id",
        "first_name": "off_first_name",
        "family_name": "off_family_name",
        "name_i": "off_name_i",
        "position": "off_position",
        "matchups_person_id": "def_player_id",
        "matchups_first_name": "def_first_name",
        "matchups_family_name": "def_family_name",
        "matchups_name_i": "def_name_i",
        **STAT_COLUMNS,
    }
    missing = sorted(set(renamed) - set(frame.columns))
    if missing:
        raise ValueError(f"Archive CSV lacks columns {missing}")
    return _finish(frame[list(renamed)].rename(columns=renamed), season_year, "archive")


def from_api_payload(payload: dict | bytes | str, season_year: int) -> pd.DataFrame:
    """Render one BoxScoreMatchupsV3 response into ``COLUMNS``."""
    if not isinstance(payload, dict):
        payload = json.loads(payload)
    box = payload["boxScoreMatchups"]
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        for player in team.get("players", []):
            for matchup in player.get("matchups", []):
                stats = {_snake(key): value for key, value in matchup["statistics"].items()}
                rows.append(
                    {
                        "game_id": box["gameId"],
                        "home_team_id": box["homeTeamId"],
                        "away_team_id": box["awayTeamId"],
                        "off_team_id": team["teamId"],
                        "off_player_id": player["personId"],
                        "off_first_name": player.get("firstName"),
                        "off_family_name": player.get("familyName"),
                        "off_name_i": player.get("nameI"),
                        "off_position": player.get("position"),
                        "def_player_id": matchup["personId"],
                        "def_first_name": matchup.get("firstName"),
                        "def_family_name": matchup.get("familyName"),
                        "def_name_i": matchup.get("nameI"),
                        **{ours: stats.get(theirs) for theirs, ours in STAT_COLUMNS.items()},
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(COLUMNS))
    return _finish(pd.DataFrame(rows), season_year, "api")


class MatchupStore:
    """``<root>/matchups/season=YYYY/{matchups,games}.parquet``.

    ``matchups.parquet`` holds the rows; ``games.parquet`` records, per game,
    whether we hold it (``ok``) or the NBA answered with no tracking
    (``empty``), and from which source, so a fill knows what is still owed.
    """

    def __init__(self, root: Path = Path("data")) -> None:
        self.root = Path(root) / "matchups"

    def _dir(self, season_year: int) -> Path:
        return self.root / f"season={season_year}"

    def path(self, season_year: int) -> Path:
        return self._dir(season_year) / "matchups.parquet"

    def status_path(self, season_year: int) -> Path:
        return self._dir(season_year) / "games.parquet"

    def read(self, season_year: int) -> pd.DataFrame:
        path = self.path(season_year)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=list(COLUMNS))

    def statuses(self, season_year: int) -> pd.DataFrame:
        path = self.status_path(season_year)
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame(columns=list(STATUS_COLUMNS))

    def settled_games(self, season_year: int) -> set[str]:
        """Games that need no further request: held, or known untracked."""
        return set(self.statuses(season_year)["game_id"])

    def write(
        self,
        season_year: int,
        rows: pd.DataFrame,
        *,
        empty_games: Iterable[str] = (),
    ) -> None:
        """Replace whole games: every game in ``rows`` or ``empty_games``."""
        empty_games = set(empty_games)
        incoming = set(rows["game_id"]) | empty_games
        current = self.read(season_year)
        kept = current.loc[~current["game_id"].isin(incoming)]
        merged = rows if kept.empty else pd.concat([kept, rows], ignore_index=True)
        merged = merged.sort_values(
            ["game_id", "off_team_id", "off_player_id", "def_player_id"], kind="stable"
        ).reset_index(drop=True)

        now = pd.Timestamp.now(tz="UTC")
        sources = rows.groupby("game_id")["source"].first()
        fresh = pd.DataFrame(
            [(game_id, "ok", source, now) for game_id, source in sources.items()]
            + [(game_id, "empty", "api", now) for game_id in sorted(empty_games)],
            columns=list(STATUS_COLUMNS),
        )
        status = self.statuses(season_year)
        status = status.loc[~status["game_id"].isin(incoming)]
        status = fresh if status.empty else pd.concat([status, fresh], ignore_index=True)

        self._atomic(self.path(season_year), merged[list(COLUMNS)])
        self._atomic(
            self.status_path(season_year),
            status.sort_values("game_id").reset_index(drop=True),
        )

    @staticmethod
    def _atomic(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)

    def parquet_bytes(self, season_year: int) -> dict[str, bytes]:
        """Both files of a season, keyed by their path under the data root."""
        out = {}
        for path in (self.path(season_year), self.status_path(season_year)):
            if path.exists():
                out[str(path.relative_to(self.root.parent))] = path.read_bytes()
        return out


def import_season(
    season_year: int,
    *,
    store: MatchupStore,
    urls: dict[str, str],
    playoffs: bool = True,
    download=download_season_csv,
) -> dict[str, int]:
    """Load a season from the archive, replacing any games it covers."""
    counts = {"games": 0, "rows": 0, "missing_dataset": 0}
    frames = []
    for name in season_datasets(season_year, playoffs=playoffs):
        url = urls.get(name)
        if url is None:
            counts["missing_dataset"] += 1
            continue
        with tempfile.TemporaryDirectory() as scratch:
            csv_path = download(url, Path(scratch))
            frames.append(from_archive_csv(pd.read_csv(csv_path, low_memory=False), season_year))
    if not frames:
        return counts
    rows = pd.concat(frames, ignore_index=True)
    store.write(season_year, rows)
    counts["games"] = rows["game_id"].nunique()
    counts["rows"] = len(rows)
    return counts


def fill_from_api(
    season_year: int,
    game_ids: Iterable[str],
    *,
    store: MatchupStore,
    client,
) -> dict[str, int]:
    """Fetch games the store does not hold yet, one paced call each.

    Only a 200 with no players is recorded as ``empty``. Any 5xx, fast or slow,
    is left unrecorded so the next run asks again: on GameRotation a fast 500
    turned out to be a cached failure that expires, not proof of a hole
    (audited 2026-09-25), and nothing yet shows this endpoint differs.
    """
    counts = {"ok": 0, "empty": 0, "retry_later": 0}
    frames: list[pd.DataFrame] = []
    empty: list[str] = []
    try:
        for game_id in game_ids:
            try:
                raw = client.fetch("boxscorematchupsv3", game_id)
            except (GameUnavailable, ServerTimeout):
                counts["retry_later"] += 1
                continue
            except EmptyResponse:
                empty.append(game_id)
                counts["empty"] += 1
                continue
            frames.append(from_api_payload(raw, season_year))
            counts["ok"] += 1
    finally:
        # Also on CircuitOpen: keep what was fetched before the breaker opened.
        if frames or empty:
            rows = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=list(COLUMNS))
            )
            store.write(season_year, rows, empty_games=empty)
    return counts
