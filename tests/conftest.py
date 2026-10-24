"""Settings every test runs under."""

import pytest


@pytest.fixture(autouse=True)
def _no_local_model_cache(monkeypatch):
    """No test reads or writes the real ``~/.cache`` model copies.

    The fake registries reuse keys with different bytes from test to test; a
    test of the cache itself points it at its own directory.
    """
    monkeypatch.setenv("NBA_OU_MODEL_CACHE_DIR", "")


@pytest.fixture(autouse=True)
def _fresh_season_piece_memo():
    """Every test starts without the seasons an earlier one read."""
    from nba_ou.postgre_db import season_store

    season_store._PIECES.clear()
    yield
    season_store._PIECES.clear()
