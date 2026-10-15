"""The history memo reuses a season only while its inputs and the code are unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nba_ou.create_training_data import history_memo as memo


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    game_ids = [f"002{s % 100:02d}{g:05d}" for s in (2021, 2022, 2023) for g in range(40)]
    games = pd.DataFrame({"GAME_ID": game_ids, "PTS": rng.integers(80, 130, len(game_ids))})
    odds = pd.DataFrame({"game_id": game_ids, "line": rng.normal(220, 5, len(game_ids))})
    return {"games": (games, "GAME_ID"), "odds": (odds, "game_id")}


def test_digests_chain_over_earlier_seasons():
    inputs = _inputs()
    before = memo.season_input_digests(inputs, [2021, 2022, 2023])
    games, _ = inputs["games"]
    changed = games.copy()
    changed.loc[changed["GAME_ID"].str[3:5] == "22", "PTS"] += 1
    after = memo.season_input_digests(
        {**inputs, "games": (changed, "GAME_ID")}, [2021, 2022, 2023]
    )
    assert after[2021] == before[2021]
    assert after[2022] != before[2022]
    assert after[2023] != before[2023]  # a later season depends on 2022 too


def test_digests_do_not_see_later_seasons():
    inputs = _inputs()
    before = memo.season_input_digests(inputs, [2021, 2022])
    games, _ = inputs["games"]
    extra = pd.DataFrame({"GAME_ID": ["0022400001"], "PTS": [100]})
    grown = pd.concat([games, extra], ignore_index=True)
    after = memo.season_input_digests({**inputs, "games": (grown, "GAME_ID")}, [2021, 2022])
    assert after == before


def test_round_trip_and_refusals(tmp_path):
    backend = memo._LocalBackend(tmp_path)
    digests = memo.season_input_digests(_inputs(), [2021, 2022, 2023])
    writer = memo.HistoryMemo(backend, digests, max_season=2022)
    table = writer.table("rows", 2021)
    table[("0022100001", "1")] = ({"A": 1}, {"available": ["5"], "injured": []})
    assert writer.table("rows", 2023) is None  # past max_season
    writer.flush()

    reader = memo.HistoryMemo(backend, digests, max_season=2022)
    assert reader.table("rows", 2021) == table

    changed = {**digests, 2021: "different"}
    assert memo.HistoryMemo(backend, changed, 2022).table("rows", 2021) == {}

    payload = next(tmp_path.rglob("season=2021.pkl"))
    payload.write_bytes(payload.read_bytes()[:-3])
    corrupt = memo.HistoryMemo(backend, digests, 2022)
    assert corrupt.table("rows", 2021) == {}
    assert "corrupt" in corrupt.report[0]


def test_a_broken_backend_is_a_miss():
    class Broken:
        def get(self, key):
            raise OSError("unreachable")

        def put(self, key, data):
            raise OSError("unreachable")

    digests = memo.season_input_digests(_inputs(), [2021])
    broken = memo.HistoryMemo(Broken(), digests, 2021)
    table = broken.table("rows", 2021)
    assert table == {}
    table["k"] = 1
    broken.flush()  # reports, does not raise
    assert any("failed" in line for line in broken.report)


def test_season_of_game_ids():
    ids = pd.Series(["0022400001", "22300005", "0042000101", "bad"])
    assert memo.season_of_game_ids(ids).tolist() == [2024, 2023, 2020, -1]
    assert memo.season_of_game_id("0022400001") == 2024
    assert memo.season_of_game_id("bad") is None


def test_prefetched_tables_equal_loaded_ones(tmp_path):
    backend = memo._LocalBackend(tmp_path)
    digests = memo.season_input_digests(_inputs(), [2021, 2022, 2023])
    writer = memo.HistoryMemo(backend, digests, max_season=2023)
    for season in (2021, 2022):
        writer.table("rows", season)[season] = {"value": float(season) / 3}
    writer.flush()
    meta = next(tmp_path.rglob("season=2021.json")).read_text()
    assert '"compression": "zlib"' in meta

    reader = memo.HistoryMemo(backend, digests, max_season=2023)
    reader.prefetch(["rows", "other"])
    assert reader.table("rows", 2021) == {2021: {"value": 2021 / 3}}
    assert reader.table("rows", 2022) == {2022: {"value": 2022 / 3}}
    assert reader.table("rows", 2023) == {}
    assert reader.table("other", 2021) == {}
