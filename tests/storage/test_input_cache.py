"""The input cache returns exactly what a fetch returns, and only while valid."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest
from nba_ou.postgre_db import input_cache
from nba_ou.postgre_db.input_cache import SeasonCache, code_key
from nba_ou.postgre_db.line_history_aiven import fetch

from tests.support.frame_identity import assert_identical


def _frame(season: int) -> pd.DataFrame:
    rng = np.random.default_rng(season)
    return pd.DataFrame(
        {
            "game_id": [f"00{season}{i:04d}" for i in range(50)],
            "line_ts": pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC"),
            "left_line": np.where(rng.random(50) < 0.1, np.nan, rng.normal(220, 5, 50)),
            "is_opener": rng.random(50) < 0.2,
            "book": pd.Series(rng.choice(["a", "b", None], 50), dtype=object),
        },
        index=pd.RangeIndex(50),
    )


class _Server:
    """A fake source: a fetch per season and a fingerprint that can change."""

    def __init__(self) -> None:
        self.version = {2024: 1, 2025: 1}
        self.fetches: list[int] = []

    def fingerprint(self, season: int) -> list:
        return [season, self.version[season], 12345678901234567890]

    def fetch(self, season: int) -> pd.DataFrame:
        self.fetches.append(season)
        return _frame(season + 1000 * self.version[season])


def test_hit_returns_the_fetched_frame_exactly(tmp_path):
    server = _Server()
    first = SeasonCache("t", "k", root=tmp_path).get(
        2024, server.fingerprint, server.fetch
    )
    cache = SeasonCache("t", "k", root=tmp_path)
    again = cache.get(2024, server.fingerprint, server.fetch)
    assert server.fetches == [2024]
    assert (cache.hits, cache.misses) == (1, 0)
    assert_identical(again, first)
    assert_identical(again, _frame(2024 + 1000))


def test_changed_fingerprint_refetches_only_that_season(tmp_path):
    server = _Server()
    for season in (2024, 2025):
        SeasonCache("t", "k", root=tmp_path).get(
            season, server.fingerprint, server.fetch
        )
    server.version[2025] = 2  # a repair rewrote 2025
    cache = SeasonCache("t", "k", root=tmp_path)
    frames = [cache.get(s, server.fingerprint, server.fetch) for s in (2024, 2025)]
    assert server.fetches == [2024, 2025, 2025]
    assert_identical(frames[1], _frame(2025 + 2000))
    # And the refetched season is cached under its new fingerprint.
    SeasonCache("t", "k", root=tmp_path).get(2025, server.fingerprint, server.fetch)
    assert server.fetches == [2024, 2025, 2025]


def test_corrupted_entry_is_refetched(tmp_path):
    server = _Server()
    cache = SeasonCache("t", "k", root=tmp_path)
    cache.get(2024, server.fingerprint, server.fetch)
    frame_path = cache.directory / "season=2024.pkl"
    frame_path.write_bytes(frame_path.read_bytes()[:-10])
    SeasonCache("t", "k", root=tmp_path).get(2024, server.fingerprint, server.fetch)
    assert server.fetches == [2024, 2024]


def test_different_keys_do_not_share_entries(tmp_path):
    server = _Server()
    SeasonCache("t", "k1", root=tmp_path).get(2024, server.fingerprint, server.fetch)
    SeasonCache("t", "k2", root=tmp_path).get(2024, server.fingerprint, server.fetch)
    assert server.fetches == [2024, 2024]


def test_code_key_changes_with_code_and_arguments():
    def loader_a():
        return 1

    def loader_b():
        return 2

    assert code_key(loader_a, last_n=5) == code_key(loader_a, last_n=5)
    assert code_key(loader_a, last_n=5) != code_key(loader_b, last_n=5)
    assert code_key(loader_a, last_n=5) != code_key(loader_a, last_n=None)


class _FakeConn:
    """Enough of a connection for the cache's session settings."""

    def cursor(self):
        return contextlib.nullcontext(self)

    def execute(self, *args, **kwargs):
        pass


def test_cached_closing_ticks_equal_the_fetched_ones(tmp_path, monkeypatch):
    """The cached path of fetch_closing_ticks against the plain season fetch."""
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "0")
    monkeypatch.setenv("NBA_OU_INPUT_CACHE_DIR", str(tmp_path))
    calls: list[list[int]] = []

    def fake_once(*, season_years, **kwargs):
        calls.append(list(season_years))
        season = season_years[0]
        if season == 2023:  # an empty season
            return pd.DataFrame(columns=fetch.TICK_COLUMNS)
        rows = _frame(season)
        ticks = pd.DataFrame(
            {
                "game_id": rows["game_id"],
                "season_year": season,
                "market": "totals",
                "book": rows["book"].fillna("x"),
                "line_ts": rows["line_ts"],
                "minutes_before_tip": 30.0,
                "is_opener": rows["is_opener"],
                "left_line": rows["left_line"],
                "left_price": -110.0,
                "right_line": rows["left_line"],
                "right_price": -110.0,
            }
        )
        return ticks[fetch.TICK_COLUMNS]

    monkeypatch.setattr(fetch, "_fetch_closing_ticks_once", fake_once)
    monkeypatch.setattr(
        fetch, "connect_line_history_db", lambda: contextlib.nullcontext(_FakeConn())
    )
    versions = {2023: "v1", 2024: "v1", 2025: "v1"}
    # One function throughout: the cache key covers the fingerprint's code.
    monkeypatch.setattr(
        fetch,
        "line_history_season_fingerprint",
        lambda conn, season: [season, versions[season]],
    )

    seasons = [2023, 2024, 2025]
    plain = fetch.fetch_closing_ticks(season_years=seasons)
    cold = fetch.fetch_closing_ticks(season_years=seasons, use_cache=True)
    warm = fetch.fetch_closing_ticks(season_years=seasons, use_cache=True)
    assert_identical(cold, plain)
    assert_identical(warm, plain)
    # plain: 3 fetches, cold: 3 fetches, warm: none.
    assert len(calls) == 6

    versions[2024] = "v2"  # the server's 2024 changed
    refreshed = fetch.fetch_closing_ticks(season_years=seasons, use_cache=True)
    assert calls[6:] == [[2024]]
    assert_identical(refreshed, plain)


def test_cache_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("NBA_OU_INPUT_CACHE", "0")
    assert not input_cache.input_cache_enabled()
    monkeypatch.delenv("NBA_OU_INPUT_CACHE")
    assert input_cache.input_cache_enabled()


@pytest.mark.parametrize("value", [12345678901234567890123, None, "x"])
def test_fingerprints_compare_after_a_json_round_trip(tmp_path, value):
    """Fingerprints hold big sums and NULLs; stored and fresh must still match."""
    fetches = []
    fingerprint = lambda season: [
        str(value) if value is not None else None,
        season,
    ]  # noqa: E731
    for _ in range(2):
        SeasonCache("t", "k", root=tmp_path).get(
            2024, fingerprint, lambda s: fetches.append(s) or _frame(s)
        )
    assert fetches == [2024]


def test_failed_fingerprint_fetches_and_stores_nothing(tmp_path):
    fetches = []

    def broken(season):
        raise TimeoutError("statement timeout")

    def fetch_season(season):
        fetches.append(season)
        return _frame(season)

    for _ in range(2):
        frame = SeasonCache("t", "k", root=tmp_path).get(2024, broken, fetch_season)
        assert_identical(frame, _frame(2024))
    assert fetches == [2024, 2024]
    assert not (tmp_path / "t" / "k").exists()


def test_cache_dir_expands_the_home_directory(monkeypatch, tmp_path):
    from nba_ou.postgre_db import input_cache

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NBA_OU_INPUT_CACHE_DIR", "~/.cache/nba_ou/inputs")
    assert input_cache.cache_dir() == tmp_path / ".cache" / "nba_ou" / "inputs"
