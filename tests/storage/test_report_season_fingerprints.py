"""The injury-report season fingerprints the intermediate memos are keyed on."""

from __future__ import annotations

import pandas as pd
from nba_ou.postgre_db.injury_report_aiven import fetch


def _parts(dimensions: dict[int, str]) -> pd.DataFrame:
    rows = []
    for season in (2023, 2024, 2025, 2026):
        rows += [
            (season, "game", 1230, f"g{season}"),
            (season, "status_span", 9000, f"s{season}"),
            (season, "dimensions", 900, dimensions.get(season, f"d{season}")),
        ]
    return pd.DataFrame(rows, columns=["season", "part", "n", "h"])


def _fingerprints(monkeypatch, frame: pd.DataFrame) -> dict[int, str]:
    monkeypatch.setattr(fetch.pd, "read_sql_query", lambda *_, **__: frame)
    return fetch.report_season_fingerprints(conn=None)


def test_a_dimension_row_joined_by_one_season_keys_only_it_and_later(monkeypatch):
    """A new injury reason is joined by the current season's spans alone."""
    before = _fingerprints(monkeypatch, _parts({}))
    new_reason = _fingerprints(monkeypatch, _parts({2026: "d2026+reason"}))
    assert [s for s in before if before[s] != new_reason[s]] == [2026]

    changed_2024 = _fingerprints(monkeypatch, _parts({2024: "d2024-relabelled"}))
    assert [s for s in before if before[s] != changed_2024[s]] == [2024, 2025, 2026]


def test_the_query_keys_dimensions_by_season():
    """No season-less part is left: it would key every season again."""
    assert "SELECT NULL" not in fetch._SEASON_FINGERPRINTS
    assert "span_keys" in fetch._SEASON_FINGERPRINTS
