"""Season queries: templates rebuild the loaders' queries, and S3 only ever helps."""

from __future__ import annotations

import pytest
from nba_ou.data_processing.injury_status import status_history
from nba_ou.postgre_db import season_queries, season_tables
from nba_ou.postgre_db.injury_report_aiven import fetch as report_fetch
from nba_ou.postgre_db.season_queries import SeasonQuery, registered_season_queries


def test_templates_rebuild_the_loader_queries():
    box = status_history._box_history_query()
    assert (
        box.query(status_history._BOX_HISTORY_SEASON_FILTER)
        == status_history._BOX_HISTORY_SQL
    )
    for name, (spec, _) in registered_season_queries().items():
        if name.startswith("injury_report."):
            original = getattr(report_fetch, "_" + name.split(".", 1)[1].upper())
            assert spec.query(report_fetch._SEASON_FILTER) == original


def test_ordered_queries_are_refused():
    with pytest.raises(ValueError):
        SeasonQuery("x", "SELECT 1 FROM t WHERE {season_filter} ORDER BY 1", "s", "db")


class _Conn:
    rolled_back = False

    def rollback(self):
        self.rolled_back = True


def test_read_through_steps_aside(monkeypatch):
    spec = status_history._box_history_query()
    conn = _Conn()
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "0")
    assert season_queries.read_through_query(spec, None, conn) is None
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "1")
    monkeypatch.setattr(season_tables, "_shared_store", lambda: None)
    assert season_queries.read_through_query(spec, None, conn) is None

    def broken(*args, **kwargs):
        raise TimeoutError("unreachable")

    monkeypatch.setattr(season_tables, "_shared_store", lambda: object())
    monkeypatch.setattr(season_queries, "read_season_query", broken)
    assert season_queries.read_through_query(spec, None, conn) is None
    assert conn.rolled_back
