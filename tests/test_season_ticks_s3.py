"""Closing ticks from the S3 input cache are exactly the fetched ones."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
from nba_ou.postgre_db import season_tables
from nba_ou.postgre_db.line_history_aiven import fetch
from nba_ou.postgre_db.season_store import SeasonStore

from .fake_s3 import FakeS3Client
from .frame_identity import assert_identical


class _FakeConn:
    def cursor(self):
        return contextlib.nullcontext(self)

    def execute(self, *args, **kwargs):
        pass


def _ticks(season: int, version: int) -> pd.DataFrame:
    rng = np.random.default_rng(season * 10 + version)
    n = 40
    return pd.DataFrame(
        {
            "game_id": [f"002{season % 100:02d}{i:05d}" for i in range(n)],
            "season_year": season,
            "market": rng.choice(["totals", "point_spread"], n),
            "book": rng.choice(["bet365", "fanduel"], n),
            "line_ts": pd.date_range(f"{season}-11-01", periods=n, freq="h", tz="UTC"),
            "minutes_before_tip": rng.uniform(5, 300, n),
            "is_opener": rng.random(n) < 0.2,
            "left_line": np.where(rng.random(n) < 0.1, np.nan, rng.normal(220, 5, n)),
            "left_price": rng.integers(-130, -100, n),
            "right_line": rng.normal(220, 5, n),
            "right_price": rng.integers(-130, -100, n),
        }
    )[fetch.TICK_COLUMNS]


def _setup(monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_OU_INPUT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "1")
    server = {"version": {2022: 1, 2023: 1, 2024: 1}, "fetches": []}

    def fake_once(*, season_years, **kwargs):
        season = season_years[0]
        server["fetches"].append(season)
        return _ticks(season, server["version"][season])

    monkeypatch.setattr(fetch, "_fetch_closing_ticks_once", fake_once)
    monkeypatch.setattr(
        fetch, "connect_line_history_db", lambda: contextlib.nullcontext(_FakeConn())
    )
    monkeypatch.setattr(
        fetch,
        "line_history_season_fingerprint",
        lambda conn, season: [season, server["version"][season]],
    )
    monkeypatch.setattr(season_tables, "current_season_year", lambda now=None: 2024)
    store = SeasonStore(FakeS3Client(), "bucket")
    monkeypatch.setattr(season_tables, "_shared_store", lambda: store)
    return server, store


def _direct(seasons):
    return fetch._concat_ticks(
        [_ticks(s, 1) for s in seasons]
    )


def test_refreshed_seasons_are_served_from_s3(monkeypatch, tmp_path):
    server, store = _setup(monkeypatch, tmp_path)
    outcome = fetch.refresh_season_ticks(store, [2022, 2023, 2024])
    assert outcome[2024] == "current season, not stored"
    assert outcome[2022].startswith("uploaded")
    server["fetches"].clear()
    ticks = fetch.fetch_closing_ticks(season_years=[2022, 2023, 2024], use_cache=True)
    assert server["fetches"] == [2024]  # only the current season is fetched
    assert_identical(ticks, _direct([2022, 2023, 2024]))


def test_unchanged_seasons_are_only_touched_and_changed_ones_reuploaded(
    monkeypatch, tmp_path
):
    server, store = _setup(monkeypatch, tmp_path)
    fetch.refresh_season_ticks(store, [2022, 2023])
    puts = len(store.s3.put_order)
    assert fetch.refresh_season_ticks(store, [2022, 2023]) == {
        2022: "unchanged",
        2023: "unchanged",
    }
    assert all(k.endswith(".json") for k in store.s3.put_order[puts:])
    server["version"][2023] = 2
    assert fetch.refresh_season_ticks(store, [2023])[2023].startswith("uploaded")


def test_stale_finished_seasons_are_served_from_s3(monkeypatch, tmp_path):
    server, store = _setup(monkeypatch, tmp_path)
    fetch.refresh_season_ticks(
        store, [2022], now=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    )
    server["fetches"].clear()
    ticks = fetch.fetch_closing_ticks(season_years=[2022], use_cache=True)
    assert server["fetches"] == []
    assert_identical(ticks, _direct([2022]))


def test_without_trust_stale_seasons_are_fetched(monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_OU_TRUST_FINISHED_SEASONS", "0")
    server, store = _setup(monkeypatch, tmp_path)
    fetch.refresh_season_ticks(
        store, [2022], now=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
    )
    server["fetches"].clear()
    ticks = fetch.fetch_closing_ticks(season_years=[2022], use_cache=True)
    assert server["fetches"] == [2022]
    assert_identical(ticks, _direct([2022]))


def test_s3_failure_falls_back_to_fetching(monkeypatch, tmp_path):
    server, _ = _setup(monkeypatch, tmp_path)

    def broken():
        raise OSError("unreachable")

    monkeypatch.setattr(season_tables, "_shared_store", broken)
    ticks = fetch.fetch_closing_ticks(season_years=[2022, 2023], use_cache=True)
    assert server["fetches"] == [2022, 2023]
    assert_identical(ticks, _direct([2022, 2023]))


def test_pregame_ticks_are_served_from_s3_as_the_builder_fetches_them(
    monkeypatch, tmp_path
):
    server, store = _setup(monkeypatch, tmp_path)

    def fake_pregame(season_years, **kwargs):
        season = season_years[0]
        server["fetches"].append(season)
        return _ticks(season, server["version"][season])

    monkeypatch.setattr(fetch, "_fetch_pregame_ticks_once", fake_pregame)
    outcome = fetch.refresh_season_ticks(store, [2022, 2023], name="pregame_ticks")
    assert outcome[2022].startswith("uploaded")
    server["fetches"].clear()
    # The intermediate builder's call: its default excluded books.
    ticks = fetch.fetch_pregame_ticks(
        [2022, 2023, 2024], exclude_books=fetch.HISTORY_ONLY_BOOKS, use_cache=True
    )
    assert server["fetches"] == [2024]
    assert_identical(ticks, _direct([2022, 2023, 2024]))
    assert set(ticks.attrs["season_fingerprints"]) == {2022, 2023, 2024}


def test_check_only_compares_without_writing(monkeypatch, tmp_path):
    server, store = _setup(monkeypatch, tmp_path)
    fetch.refresh_season_ticks(store, [2022])
    puts = len(store.s3.put_order)
    server["version"][2022] = 2
    checked = fetch.refresh_season_ticks(store, [2022, 2023], check_only=True)
    assert checked[2022].startswith("CHANGED") and checked[2023] == "not stored"
    assert len(store.s3.put_order) == puts
