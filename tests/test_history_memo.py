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
