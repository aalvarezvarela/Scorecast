"""S3 season pieces plus the database give exactly the single-query frame.

A fake database answers the queries ``season_tables`` sends from an in-memory
table of driver-style rows (``None`` for NULL, ``Decimal`` NUMERICs), through
pandas' own ``read_sql`` conversion. The S3 side is ``tests.fake_s3``.
"""

from __future__ import annotations

import datetime as dt
import random
from decimal import Decimal

import pandas as pd
import pandas.io.sql as pandas_sql
import pytest
from nba_ou.postgre_db import season_tables
from nba_ou.postgre_db.row_order import canonical_row_order
from nba_ou.postgre_db.season_store import (
    NotExactInParquet,
    SeasonStore,
    frame_differences,
    to_exact_parquet,
)
from nba_ou.postgre_db.season_tables import (
    SEASON_TABLES,
    SeasonTable,
    read_season_table,
    refresh_season_table,
)

from tests.support.fake_s3 import FakeS3Client

NOW = pd.Timestamp("2026-01-15 12:00", tz="UTC")  # season 2025 is current
COLUMNS = ["game_id", "season_year", "game_date", "team", "pts", "line", "flag"]


def _rows(seed: int = 0) -> list[tuple]:
    """Driver-style rows: a column NULL in whole seasons, NUMERIC NaN, dates."""
    rng = random.Random(seed)
    rows = []
    for season in (2021, 2022, 2023, 2024, 2025):
        start = dt.date(season, 10, 20)
        for i in range(60):
            day = start + dt.timedelta(days=i // 3)
            line = None if season in (2021, 2023) else Decimal(f"{220 + i % 7}.5")
            if season == 2022 and i == 5:
                line = Decimal("NaN")
            rows.append(
                (
                    f"00{season % 100}{i // 2:05d}",
                    season,
                    day,
                    rng.choice(["BOS", "LAL", None]),
                    rng.randint(80, 130),
                    line,
                    rng.choice([True, False]),
                )
            )
    rng.shuffle(rows)
    return rows


class _FakeDB:
    """Answers the three query shapes ``season_tables`` sends."""

    def __init__(self, rows, spec: SeasonTable) -> None:
        self.rows = rows
        self.spec = spec
        self.queries: list[tuple] = []

    def frame(self, rows):
        """The server's answer: rows in its ORDER BY (any order within ties)."""
        frame = pandas_sql._wrap_result(list(rows), COLUMNS, coerce_float=True)
        keys = self.spec.sort_keys or tuple((c, True) for c in self.spec.order_columns)
        if keys:
            frame = frame.sort_values(
                [k for k, _ in keys], ascending=[a for _, a in keys], kind="mergesort"
            ).reset_index(drop=True)
        return canonical_row_order(frame, self.spec.order_columns)

    def single_query(self, seasons, extras):
        return self.frame(r for r in self.rows if r[1] in seasons or r[0] in extras)

    def read_sql_canonical(self, query, conn, params=None):
        self.queries.append(params)
        if len(params) == 1 and isinstance(params[0], int):  # one season piece
            return self.frame(r for r in self.rows if r[1] == params[0])
        seasons, extras, requested = set(), set(), set()
        if (
            params
            and isinstance(params[0], list)
            and params[0]
            and isinstance(params[0][0], int)
        ):
            seasons = set(params[0])
            params = params[1:]
        if params:
            extras, requested = set(params[0]), set(params[1])
        return self.frame(
            r
            for r in self.rows
            if r[1] in seasons or (r[0] in extras and r[1] not in requested)
        )


@pytest.fixture
def spec():
    return SeasonTable(
        "t",
        "games",
        "game_date DESC, game_id DESC",
        (("game_date", False), ("game_id", False)),
    )


@pytest.fixture
def setup(monkeypatch, spec):
    db = _FakeDB(_rows(), spec)
    monkeypatch.setattr(season_tables, "read_sql_canonical", db.read_sql_canonical)
    version = {"value": "v1"}
    monkeypatch.setattr(
        season_tables,
        "season_fingerprint",
        lambda spec, conn, season: [season, version["value"]],
    )
    return db, SeasonStore(FakeS3Client(), "bucket"), version


def test_refresh_then_read_equals_the_single_query(setup, spec):
    db, store, _ = setup
    outcome = refresh_season_table(
        spec, [2021, 2022, 2023, 2024, 2025], store=store, conn=None, now=NOW
    )
    assert outcome[2025] == "current season, not stored"
    assert all(outcome[s].startswith("uploaded") for s in (2021, 2022, 2023, 2024))
    extras = ["00210000", "00220003"]  # games from seasons not requested
    frame, summary = read_season_table(
        spec, [2023, 2024, 2025], extras, store=store, conn=None, now=NOW
    )
    assert summary["s3"] == [2023, 2024] and summary["db"] == [2025]
    assert (
        frame_differences(frame, db.single_query({2023, 2024, 2025}, set(extras))) == []
    )


def test_stale_finished_seasons_are_served_and_missing_ones_queried(setup, spec):
    db, store, _ = setup
    refresh_season_table(
        spec, [2022], store=store, conn=None, now=NOW - pd.Timedelta(days=30)
    )
    refresh_season_table(spec, [2023], store=store, conn=None, now=NOW)
    frame, summary = read_season_table(
        spec, [2022, 2023, 2024], [], store=store, conn=None, now=NOW
    )
    assert summary["s3"] == [2022, 2023] and summary["db"] == [2024]
    assert frame_differences(frame, db.single_query({2022, 2023, 2024}, set())) == []


def test_without_trust_stale_seasons_come_from_the_database(setup, spec, monkeypatch):
    monkeypatch.setenv("NBA_OU_TRUST_FINISHED_SEASONS", "0")
    db, store, _ = setup
    refresh_season_table(
        spec, [2022], store=store, conn=None, now=NOW - pd.Timedelta(days=3)
    )
    refresh_season_table(spec, [2023], store=store, conn=None, now=NOW)
    frame, summary = read_season_table(
        spec, [2022, 2023, 2024], [], store=store, conn=None, now=NOW
    )
    assert summary["s3"] == [2023] and summary["db"] == [2022, 2024]
    assert frame_differences(frame, db.single_query({2022, 2023, 2024}, set())) == []


def test_changed_fingerprint_reuploads_only_that_season(setup, spec):
    _, store, version = setup
    refresh_season_table(spec, [2022, 2023], store=store, conn=None, now=NOW)
    puts = len(store.s3.put_order)
    again = refresh_season_table(spec, [2022, 2023], store=store, conn=None, now=NOW)
    assert again == {2022: "unchanged", 2023: "unchanged"}
    assert all(k.endswith(".json") for k in store.s3.put_order[puts:])  # touches only
    version["value"] = "v2"
    changed = refresh_season_table(spec, [2022], store=store, conn=None, now=NOW)
    assert changed[2022].startswith("uploaded")


def test_a_corrupted_piece_is_read_from_the_database(setup, spec):
    db, store, _ = setup
    refresh_season_table(spec, [2023], store=store, conn=None, now=NOW)
    key = next(k for k in store.s3.store if k.endswith("season=2023.parquet"))
    store.s3.store[key] = store.s3.store[key][:-8]
    frame, summary = read_season_table(
        spec, [2023], [], store=store, conn=None, now=NOW
    )
    assert summary["db"] == [2023]
    assert frame_differences(frame, db.single_query({2023}, set())) == []


@pytest.mark.parametrize("dataset", sorted(SEASON_TABLES))
def test_the_registered_loaders_assemble_exactly(monkeypatch, dataset):
    real = SEASON_TABLES[dataset]
    # Same ordering rules, on the fake table's columns.
    order_by = {
        "nba_games.nba_games": "game_date DESC, game_id DESC",
        "nba_all_star_voting.nba_all_star_voting": "season_year, team",
    }.get(dataset, real.order_by and "game_date DESC")
    sort_keys = (
        real.sort_keys
        if dataset == "nba_games.nba_games"
        else ((("game_date", False),) if real.sort_keys else ())
    )
    spec = SeasonTable("t", dataset.split(".")[1], order_by, sort_keys)
    db = _FakeDB(_rows(seed=3), spec)
    monkeypatch.setattr(season_tables, "read_sql_canonical", db.read_sql_canonical)
    monkeypatch.setattr(
        season_tables, "season_fingerprint", lambda s, c, season: [season]
    )
    store = SeasonStore(FakeS3Client(), "bucket")
    refresh_season_table(
        spec, [2021, 2022, 2023, 2024], store=store, conn=None, now=NOW
    )
    extras = [] if (spec.order_by and not spec.sort_keys) else ["00210001"]
    frame, _ = read_season_table(
        spec, [2022, 2023, 2024, 2025], extras, store=store, conn=None, now=NOW
    )
    assert (
        frame_differences(frame, db.single_query({2022, 2023, 2024, 2025}, set(extras)))
        == []
    )


def test_frames_parquet_would_change_are_refused():
    ints_as_objects = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype=object)})
    with pytest.raises(NotExactInParquet):
        to_exact_parquet(ints_as_objects)


class _Conn:
    rolled_back = False

    def rollback(self):
        self.rolled_back = True


def test_read_through_steps_aside_whenever_it_cannot_serve(monkeypatch):
    conn = _Conn()
    dataset = "nba_players.nba_players"
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "0")
    assert season_tables.read_through(dataset, [2024], [], conn) is None
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "1")
    assert season_tables.read_through(dataset, None, ["0022400001"], conn) is None
    monkeypatch.setattr(season_tables, "_shared_store", lambda: None)
    assert season_tables.read_through(dataset, [2024], [], conn) is None

    def broken(*args, **kwargs):
        raise TimeoutError("S3 unreachable")

    monkeypatch.setattr(season_tables, "_shared_store", lambda: object())
    monkeypatch.setattr(season_tables, "read_season_table", broken)
    assert season_tables.read_through(dataset, [2024], [], conn) is None
    assert conn.rolled_back  # the loader's own query can still run


def test_check_only_compares_without_writing(setup, spec):
    _, store, version = setup
    refresh_season_table(spec, [2022], store=store, conn=None, now=NOW)
    puts = len(store.s3.put_order)
    assert refresh_season_table(
        spec, [2022, 2023], store=store, conn=None, now=NOW, check_only=True
    ) == {2022: "unchanged", 2023: "not stored"}
    version["value"] = "v2"
    checked = refresh_season_table(
        spec, [2022], store=store, conn=None, now=NOW, check_only=True
    )
    assert checked[2022].startswith("CHANGED")
    assert len(store.s3.put_order) == puts


def test_servable(monkeypatch):
    from nba_ou.postgre_db.season_store import SeasonEntry

    old = SeasonEntry([1], NOW - pd.Timedelta(days=200), 10, "x")
    recent = SeasonEntry([1], NOW - pd.Timedelta(hours=1), 10, "x")
    assert not season_tables.servable(None, NOW)
    assert season_tables.servable(old, NOW) and season_tables.servable(recent, NOW)
    monkeypatch.setenv("NBA_OU_TRUST_FINISHED_SEASONS", "0")
    assert not season_tables.servable(old, NOW)
    assert season_tables.servable(recent, NOW)


def test_a_season_is_downloaded_once_per_process_and_copied(setup, spec, monkeypatch):
    db, store, _ = setup
    refresh_season_table(spec, [2023], store=store, conn=None, now=NOW)
    gets = []
    original = store._get

    def counting(key):
        gets.append(key)
        return original(key)

    monkeypatch.setattr(store, "_get", counting)
    first, _ = read_season_table(spec, [2023], [], store=store, conn=None, now=NOW)
    first.loc[:, "pts"] = -1  # a caller changing its frame changes no other's
    second, _ = read_season_table(spec, [2023], [], store=store, conn=None, now=NOW)
    assert sum(k.endswith(".parquet") for k in gets) == 1
    assert frame_differences(second, db.single_query({2023}, set())) == []


def test_without_the_memo_every_read_downloads(setup, spec, monkeypatch):
    monkeypatch.setenv("NBA_OU_SEASON_PIECE_MEMO", "0")
    _, store, _ = setup
    refresh_season_table(spec, [2023], store=store, conn=None, now=NOW)
    gets = []
    original = store._get
    monkeypatch.setattr(store, "_get", lambda key: gets.append(key) or original(key))
    for _ in range(2):
        read_season_table(spec, [2023], [], store=store, conn=None, now=NOW)
    assert sum(k.endswith(".parquet") for k in gets) == 2
