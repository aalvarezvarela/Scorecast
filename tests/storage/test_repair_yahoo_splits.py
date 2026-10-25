"""Yahoo split repair: which stored odds_yahoo rows it rewrites, and with what.

A market's splits are rewritten only from a fresh page that gives all four of
them; a missing value never overwrites a stored one. Rows whose re-scraped
lines differ, or that were not re-scraped at all (including an empty scrape),
are skipped, never repaired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clean_databases import repair_yahoo_splits as repair

SWAPPED = {
    "total_pct_bets_over": 44.12,
    "total_pct_bets_under": 55.88,
    "total_pct_money_over": 39.14,
    "total_pct_money_under": 60.86,
    "spread_pct_bets_away": 70.52,
    "spread_pct_bets_home": 29.48,
    "spread_pct_money_away": 77.48,
    "spread_pct_money_home": 22.52,
    "moneyline_pct_bets_away": 95.12,
    "moneyline_pct_bets_home": 4.88,
    "moneyline_pct_money_away": 96.92,
    "moneyline_pct_money_home": 3.08,
}
LINES = {
    "spread_home": -10.5,
    "spread_away": 10.5,
    "moneyline_home": -475.0,
    "moneyline_away": 360.0,
    "total_line": 228.5,
}


def _correct() -> dict:
    """The swapped splits put back: each pair exchanged."""
    out = {}
    for columns in repair.MARKET_SPLIT_COLUMNS.values():
        for a, b in (columns[0:2], columns[2:4]):
            out[a], out[b] = SWAPPED[b], SWAPPED[a]
    return out


def _stored(game_id: str = "0022600105") -> dict:
    return {
        "game_id": game_id,
        "game_date": "2026-10-24",
        "team_away": "Brooklyn Nets",
        "team_home": "Cleveland Cavaliers",
        **LINES,
        **SWAPPED,
    }


def _fresh(game_id: str = "0022600105", **changes) -> dict:
    return {"game_id": game_id, **LINES, **_correct(), **changes}


def _proposals(stored: list[dict], fresh: list[dict]) -> pd.DataFrame:
    columns = ["game_id", *repair.LINE_COLUMNS, *repair.SPLIT_COLUMNS]
    return repair.build_proposals(
        pd.DataFrame(stored), pd.DataFrame(fresh, columns=columns)
    ).set_index("game_id")


def _new_splits(row: pd.Series) -> dict:
    return {c: row[f"{c}_new"] for c in repair.SPLIT_COLUMNS}


def test_split_columns_are_the_twelve_table_columns():
    assert sorted(repair.SPLIT_COLUMNS) == sorted(SWAPPED)


def test_swapped_splits_are_repaired():
    row = _proposals([_stored()], [_fresh()]).iloc[0]
    assert row["status"] == "repair" and row["n_changed"] == 12
    assert _new_splits(row) == _correct()
    assert row["markets_kept"] == ""


def test_a_page_without_splits_never_blanks_stored_ones():
    no_splits = {c: np.nan for c in repair.SPLIT_COLUMNS}
    row = _proposals([_stored()], [_fresh(**no_splits)]).iloc[0]
    assert row["status"] == "unchanged" and row["n_changed"] == 0
    assert _new_splits(row) == SWAPPED
    assert row["markets_kept"] == "total spread moneyline"


def test_a_market_with_a_missing_split_is_kept_and_the_others_repaired():
    row = _proposals([_stored()], [_fresh(total_pct_money_under=np.nan)]).iloc[0]
    assert row["status"] == "repair" and row["n_changed"] == 8
    new = _new_splits(row)
    assert {c: new[c] for c in repair.MARKET_SPLIT_COLUMNS["total"]} == {
        c: SWAPPED[c] for c in repair.MARKET_SPLIT_COLUMNS["total"]
    }
    assert new["spread_pct_bets_away"] == 29.48
    assert row["markets_kept"] == "total"


def test_rows_whose_lines_differ_or_were_not_scraped_are_skipped():
    proposals = _proposals(
        [_stored("a"), _stored("b")], [_fresh("a", total_line=229.5)]
    )
    assert proposals.at["a", "status"] == "lines differ: skipped"
    assert proposals.at["b", "status"] == "not re-scraped: skipped"


def test_an_empty_scrape_skips_every_row(monkeypatch):
    monkeypatch.setattr(
        repair, "load_games_for_yahoo_update", lambda season_year: pd.DataFrame()
    )

    async def no_rows(days):
        return pd.DataFrame()

    monkeypatch.setattr(repair, "scrape_yahoo_days", no_rows)
    stored = pd.DataFrame([{**_stored(), "season_year": 2026}])
    proposals = repair.build_proposals(stored, repair.scrape_rows(stored))
    assert proposals["status"].tolist() == ["not re-scraped: skipped"]
