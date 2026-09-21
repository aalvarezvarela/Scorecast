"""Read validated stints from the local Parquet store.

The Parquet files under ``data/lineup_stints/`` are self-sufficient: they carry
the game date, so ratings can be fitted without a database. Loading the same
stints into Postgres (``postgre_db/lineups/``) stays optional and is only worth
it once the features earn their place in the training pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("data")
STINT_DIRNAME = "lineup_stints"


def season_dir(season_year: int, *, local_root: Path = DEFAULT_ROOT) -> Path:
    return Path(local_root) / STINT_DIRNAME / f"season={season_year}"


def available_seasons(*, local_root: Path = DEFAULT_ROOT) -> list[int]:
    root = Path(local_root) / STINT_DIRNAME
    return sorted(
        int(path.name.split("=", 1)[1])
        for path in root.glob("season=*")
        if path.is_dir() and path.name.split("=", 1)[1].isdigit()
    )


def read_game_statuses(
    season_years: Iterable[int] | None = None, *, local_root: Path = DEFAULT_ROOT
) -> pd.DataFrame:
    """Return every locally recorded build status, newest build layout."""
    seasons = (
        list(season_years)
        if season_years is not None
        else available_seasons(local_root=local_root)
    )
    frames = []
    for season_year in seasons:
        path = season_dir(season_year, local_root=local_root) / "game_status.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["season_year"] = season_year
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["game_id", "status", "reason", "season_year"])
    statuses = pd.concat(frames, ignore_index=True)
    statuses["game_id"] = statuses.game_id.astype(str).str.zfill(10)
    return statuses


def read_stints(
    season_years: Iterable[int] | None = None,
    *,
    local_root: Path = DEFAULT_ROOT,
    game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return validated stints, matching ``postgre_db.lineups.fetch_stints``."""
    seasons = (
        sorted(season_years)
        if season_years is not None
        else available_seasons(local_root=local_root)
    )
    wanted = {str(game_id).zfill(10) for game_id in game_ids} if game_ids else None
    frames = []
    for season_year in seasons:
        directory = season_dir(season_year, local_root=local_root)
        if not directory.exists():
            continue
        statuses = read_game_statuses([season_year], local_root=local_root)
        validated = set(statuses.loc[statuses.status.eq("ok"), "game_id"])
        for path in sorted(directory.glob("0*.parquet")):
            game_id = path.stem
            if game_id not in validated:
                continue
            if wanted is not None and game_id not in wanted:
                continue
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    stints = pd.concat(frames, ignore_index=True)
    if "game_date" not in stints:
        raise ValueError(
            "The local stint store predates game_date. "
            "Rebuild it with scripts/lineups/build_lineup_stints.py --force"
        )
    stints["game_date"] = pd.to_datetime(stints.game_date)
    return stints.sort_values(["game_date", "game_id", "seg_idx"], ignore_index=True)
