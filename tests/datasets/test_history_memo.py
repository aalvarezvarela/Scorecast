"""The history memo reuses a season only while its inputs and its code are unchanged."""

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


def _with_tonight(games: pd.DataFrame) -> pd.DataFrame:
    """``games`` plus a 2023 game not played yet, as a prediction build holds it."""
    tonight = pd.DataFrame(
        {"GAME_ID": ["0022300999"], "PTS": [pd.NA], "GAME_TIME": ["19:30"]}
    )
    grown = pd.concat([games.assign(GAME_TIME=None), tonight], ignore_index=True)
    return grown.astype({"PTS": "Int64"})


def test_values_only_digests_ignore_the_schema_tonight_imposes():
    inputs = _inputs()
    games, _ = inputs["games"]
    live = {**inputs, "games": (_with_tonight(games), "GAME_ID")}
    seasons = [2021, 2022, 2023]
    stored = memo.season_input_digests(inputs, seasons, values_only=True)
    tonight = memo.season_input_digests(live, seasons, values_only=True)
    assert tonight[2021] == stored[2021]
    assert tonight[2022] == stored[2022]
    assert tonight[2023] != stored[2023]  # tonight's own season has a new row
    # The schema-strict digests differ in every season.
    strict = memo.season_input_digests(inputs, seasons)
    assert memo.season_input_digests(live, seasons)[2021] != strict[2021]


def test_values_only_digests_still_see_values():
    inputs = _inputs()
    games, _ = inputs["games"]
    before = memo.season_input_digests(inputs, [2021, 2022], values_only=True)
    changed = games.copy()
    changed.loc[changed["GAME_ID"].str[3:5] == "21", "PTS"] += 1
    after = memo.season_input_digests(
        {**inputs, "games": (changed, "GAME_ID")}, [2021, 2022], values_only=True
    )
    assert after[2021] != before[2021]
    assert after[2022] != before[2022]
    filled = games.assign(NOTE=None)
    filled.loc[filled["GAME_ID"].str[3:5] == "22", "NOTE"] = "x"
    noted = memo.season_input_digests(
        {**inputs, "games": (filled, "GAME_ID")}, [2021, 2022], values_only=True
    )
    assert noted[2021] == before[2021]  # empty in 2021: left out
    assert noted[2022] != before[2022]


def test_round_trip_and_refusals(tmp_path):
    backend = memo._LocalBackend(tmp_path)
    digests = memo.season_input_digests(_inputs(), [2021, 2022, 2023])
    writer = memo.HistoryMemo(backend, digests, max_season=2022, code="c")
    table = writer.table("rows", 2021)
    table[("0022100001", "1")] = ({"A": 1}, {"available": ["5"], "injured": []})
    assert writer.table("rows", 2023) is None  # past max_season
    writer.flush()

    reader = memo.HistoryMemo(backend, digests, max_season=2022, code="c")
    assert reader.table("rows", 2021) == table

    changed = {**digests, 2021: "different"}
    assert memo.HistoryMemo(backend, changed, 2022, code="c").table("rows", 2021) == {}

    payload = next(tmp_path.rglob("season=2021.pkl"))
    payload.write_bytes(payload.read_bytes()[:-3])
    corrupt = memo.HistoryMemo(backend, digests, 2022, code="c")
    assert corrupt.table("rows", 2021) == {}
    assert "corrupt" in corrupt.report[0]


def test_a_broken_backend_is_a_miss():
    class Broken:
        def get(self, key):
            raise OSError("unreachable")

        def put(self, key, data):
            raise OSError("unreachable")

    digests = memo.season_input_digests(_inputs(), [2021])
    broken = memo.HistoryMemo(Broken(), digests, 2021, code="c")
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
    writer = memo.HistoryMemo(backend, digests, max_season=2023, code="c")
    for season in (2021, 2022):
        writer.table("rows", season)[season] = {"value": float(season) / 3}
    writer.flush()
    meta = next(tmp_path.rglob("season=2021.json")).read_text()
    assert '"compression": "zlib"' in meta

    reader = memo.HistoryMemo(backend, digests, max_season=2023, code="c")
    reader.prefetch(["rows", "other"])
    assert reader.table("rows", 2021) == {2021: {"value": 2021 / 3}}
    assert reader.table("rows", 2022) == {2022: {"value": 2022 / 3}}
    assert reader.table("rows", 2023) == {}
    assert reader.table("other", 2021) == {}


def _package(tmp_path, files: dict[str, str]):
    """A throwaway package ``pkg`` with ``files`` (relative path -> source)."""
    root = tmp_path / "pkg"
    for name, source in {"__init__.py": "", **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _closure_names(root, *roots):
    files, external, dynamic = memo.module_closure(
        roots, package_root=root, package="pkg"
    )
    return {str(path.relative_to(root)) for path in files}, external, dynamic


def test_closure_follows_every_kind_of_import(tmp_path):
    root = _package(
        tmp_path,
        {
            "a/__init__.py": "",
            "a/entry.py": (
                "import numpy as np\n"
                "from pkg.b import helper\n"
                "from . import sibling\n"
                "def late():\n"
                "    from pkg.c.deep import thing\n"
                "    import pkg.d\n"
            ),
            "a/sibling.py": "from ..e import name\n",
            "b.py": "from collections import abc\n",
            "c/__init__.py": "",
            "c/deep.py": "",
            "d.py": "",
            "e.py": "",
            "unrelated.py": "import pkg.a.entry\n",
        },
    )
    names, external, dynamic = _closure_names(root, "pkg.a.entry")
    assert names == {
        "__init__.py",
        "a/__init__.py",
        "a/entry.py",
        "a/sibling.py",
        "b.py",
        "c/__init__.py",
        "c/deep.py",
        "d.py",
        "e.py",
    }
    assert {"numpy", "collections"} <= external
    assert not dynamic


def test_closure_flags_dynamic_imports(tmp_path):
    root = _package(
        tmp_path,
        {
            "entry.py": "from pkg import loader\n",
            "loader.py": "import importlib\nmod = importlib.import_module('pkg.x')\n",
            "x.py": "",
        },
    )
    names, _, dynamic = _closure_names(root, "pkg.entry")
    assert "x.py" not in names
    assert dynamic


def test_closure_rejects_an_unknown_root(tmp_path):
    root = _package(tmp_path, {})
    try:
        memo.module_closure(["pkg.missing"], package_root=root, package="pkg")
    except ValueError:
        return
    raise AssertionError("an unknown root must raise")


def _key(root, *roots, settings="s"):
    return memo._closure_key(
        roots, package_root=root, package="pkg", settings=settings
    )


def test_closure_key_sees_its_closure_only(tmp_path):
    root = _package(
        tmp_path,
        {"entry.py": "from pkg import used\n", "used.py": "X = 1\n", "other.py": ""},
    )
    before = _key(root, "pkg.entry")
    (root / "other.py").write_text("Y = 2\n")
    (root / "new_module.py").write_text("")
    assert _key(root, "pkg.entry") == before
    (root / "used.py").write_text("X = 2\n")
    assert _key(root, "pkg.entry") != before
    assert _key(root, "pkg.entry", settings="t") != _key(root, "pkg.entry")


def test_closure_key_with_a_dynamic_import_is_the_whole_package(tmp_path):
    root = _package(
        tmp_path,
        {"entry.py": "__import__('pkg.x')\n", "x.py": "", "other.py": ""},
    )
    before = _key(root, "pkg.entry")
    (root / "other.py").write_text("Y = 2\n")
    assert _key(root, "pkg.entry") != before


def test_the_package_imports_nothing_dynamically():
    # code_key would fall back to the whole package for any closure that does;
    # this keeps the per-root keys meaningful.
    dynamic = [
        str(path.relative_to(memo.PACKAGE_ROOT))
        for path in sorted(memo.PACKAGE_ROOT.rglob("*.py"))
        if memo._scan_imports(path, memo.PACKAGE_ROOT, memo.PACKAGE)[2]
    ]
    assert dynamic == []


def test_memo_roots_cover_their_stages():
    def closure(root):
        files, _, _ = memo.module_closure([root])
        return {memo._module_name(p, memo.PACKAGE_ROOT, memo.PACKAGE) for p in files}

    intermediate = closure(memo.INTERMEDIATE_BUILDER)
    assert {
        "nba_ou.create_training_data.intermediate_injuries",
        "nba_ou.create_training_data.intermediate_history",
        "nba_ou.create_training_data.history_memo",
    } <= intermediate
    closing = closure("nba_ou.create_training_data.create_df_to_predict")
    assert {
        "nba_ou.data_processing.players.attach_player_features",
        "nba_ou.data_processing.past_injuries.injury_effects",
        "nba_ou.postgre_db.odds.merge_odds_data",
    } <= closing
    # Serving code is outside both builders: changing it keeps their memos.
    assert "nba_ou.prediction.prediction" not in closing | intermediate
    assert memo.code_key("nba_ou.create_training_data.create_df_to_predict") != (
        memo.code_key(memo.INTERMEDIATE_BUILDER)
    )


def test_frozen_code_key_ignores_the_source(monkeypatch):
    root = "nba_ou.create_training_data.create_df_to_predict"
    frozen = memo.code_key(root)
    assert frozen.startswith("frozen-2_5-")
    assert frozen != memo.code_key(memo.INTERMEDIATE_BUILDER)  # roots kept apart
    # A change to the builders' source moves the source key, not the frozen one.
    monkeypatch.setattr(memo, "_closure_key", lambda roots, **kwargs: "changed")
    memo.source_code_key.cache_clear()
    try:
        assert memo.code_key(root) == frozen
        monkeypatch.setenv("NBA_OU_HISTORY_MEMO_FROZEN", "0")
        assert memo.code_key(root) == "changed"
    finally:
        memo.source_code_key.cache_clear()


def test_frozen_code_key_follows_the_main_book(monkeypatch):
    from nba_ou.config import odds_columns

    root = "nba_ou.create_training_data.create_df_to_predict"
    before = memo.code_key(root)
    monkeypatch.setattr(odds_columns, "get_main_book", lambda: "otherbook")
    memo.frozen_code_key.cache_clear()
    try:
        assert memo.code_key(root) != before
        assert "otherbook" in memo.code_key(root)
    finally:
        memo.frozen_code_key.cache_clear()


def test_rebuild_reads_nothing_and_rewrites(tmp_path, monkeypatch):
    backend = memo._LocalBackend(tmp_path)
    digests = memo.season_input_digests(_inputs(), [2021])
    writer = memo.HistoryMemo(backend, digests, 2021, code="c")
    writer.table("rows", 2021)["k"] = "old"
    writer.flush()

    monkeypatch.setenv("NBA_OU_HISTORY_MEMO_REBUILD", "1")
    rebuilt = memo.HistoryMemo(backend, digests, 2021, code="c")
    table = rebuilt.table("rows", 2021)
    assert table == {}
    table["k"] = "new"
    rebuilt.flush()

    monkeypatch.delenv("NBA_OU_HISTORY_MEMO_REBUILD")
    assert memo.HistoryMemo(backend, digests, 2021, code="c").table("rows", 2021) == {
        "k": "new"
    }


def test_parts_are_each_season_s_own_input_digests():
    parts = {}
    digests = memo.season_input_digests(_inputs(), [2021, 2022], parts=parts)
    assert digests == memo.season_input_digests(_inputs(), [2021, 2022])
    assert sorted(parts) == [2021, 2022]
    assert sorted(parts[2021]) == ["games", "odds"]
    # Not chained: a season's parts depend on its own rows only.
    assert parts[2021]["games"] != parts[2022]["games"]


def _written(backend, inputs, seasons):
    parts = {}
    digests = memo.season_input_digests(inputs, seasons, parts=parts)
    writer = memo.HistoryMemo(backend, digests, max(seasons), code="c", parts=parts)
    for season in seasons:
        writer.table("rows", season)["k"] = season
    writer.flush()


def _read(backend, inputs, seasons):
    parts = {}
    digests = memo.season_input_digests(inputs, seasons, parts=parts)
    reader = memo.HistoryMemo(backend, digests, max(seasons), code="c", parts=parts)
    tables = {season: reader.table("rows", season) for season in seasons}
    return tables, reader.report


def test_a_miss_names_the_inputs_that_changed(tmp_path):
    backend = memo._LocalBackend(tmp_path)
    _written(backend, _inputs(), [2021, 2022, 2023])
    games, _ = _inputs()["games"]
    changed = games.copy()
    changed.loc[changed["GAME_ID"].str[3:5] == "22", "PTS"] += 1

    tables, report = _read(
        backend, {**_inputs(), "games": (changed, "GAME_ID")}, [2021, 2022, 2023]
    )
    assert tables[2021] == {"k": 2021}
    assert tables[2022] == {} and tables[2023] == {}
    assert sorted(report) == [
        "rows 2022: inputs changed (games)",
        "rows 2023: inputs changed (an earlier season)",
    ]


def test_an_entry_stored_without_parts_misses_as_before(tmp_path):
    backend = memo._LocalBackend(tmp_path)
    digests = memo.season_input_digests(_inputs(), [2021])
    writer = memo.HistoryMemo(backend, digests, 2021, code="c")
    writer.table("rows", 2021)["k"] = 1
    writer.flush()

    games, _ = _inputs()["games"]
    _, report = _read(backend, {**_inputs(), "games": (games.iloc[1:], "GAME_ID")}, [2021])
    assert report == ["rows 2021: inputs changed"]
