"""The local Parquet store must stand alone, with no database involved."""

import pandas as pd
import pytest
from nba_ou.data_processing.lineups.stint_store import (
    available_seasons,
    read_game_statuses,
    read_stints,
)


def _write_season(root, season_year, games):
    target = root / "lineup_stints" / f"season={season_year}"
    target.mkdir(parents=True, exist_ok=True)
    statuses = []
    for game_id, status, date in games:
        statuses.append({"game_id": game_id, "status": status, "reason": ""})
        if status != "ok":
            continue
        pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "season_year": season_year,
                    "game_date": pd.Timestamp(date),
                    "seg_idx": index,
                    "home_pts": 2,
                }
                for index in range(2)
            ]
        ).to_parquet(target / f"{game_id}.parquet", index=False)
    pd.DataFrame(statuses).to_parquet(target / "game_status.parquet", index=False)


def test_only_validated_games_are_returned_in_date_order(tmp_path):
    _write_season(
        tmp_path,
        2018,
        [
            ("0021800002", "ok", "2018-10-18"),
            ("0021800001", "ok", "2018-10-16"),
            ("0021800003", "failed", "2018-10-19"),
        ],
    )
    stints = read_stints(local_root=tmp_path)
    assert list(stints.game_id.unique()) == ["0021800001", "0021800002"]
    assert stints.game_date.is_monotonic_increasing
    assert available_seasons(local_root=tmp_path) == [2018]
    assert read_game_statuses(local_root=tmp_path).status.tolist().count("failed") == 1


def test_a_store_without_game_date_is_rejected(tmp_path):
    target = tmp_path / "lineup_stints" / "season=2018"
    target.mkdir(parents=True)
    pd.DataFrame([{"game_id": "0021800001", "status": "ok", "reason": ""}]).to_parquet(
        target / "game_status.parquet", index=False
    )
    pd.DataFrame([{"game_id": "0021800001", "seg_idx": 0}]).to_parquet(
        target / "0021800001.parquet", index=False
    )
    with pytest.raises(ValueError, match="--force"):
        read_stints(local_root=tmp_path)


def test_an_empty_store_returns_an_empty_frame(tmp_path):
    assert read_stints(local_root=tmp_path).empty
    assert available_seasons(local_root=tmp_path) == []
