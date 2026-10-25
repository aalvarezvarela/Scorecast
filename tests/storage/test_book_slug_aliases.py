"""SBR's 2026-27 "Caesars Odds" keeps the ``caesars`` slug of earlier seasons."""

from __future__ import annotations

import pandas as pd
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import _slugify_book
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    slugify_bookmaker,
)
from nba_ou.postgre_db.line_history_aiven import fetch


def test_renamed_book_keeps_its_slug_when_scraped():
    assert slugify_bookmaker("Caesars Odds") == "caesars"
    assert _slugify_book("Caesars Odds") == "caesars"
    assert slugify_bookmaker("Fanatics Sportsbook") == "fanatics_sportsbook"


def _ticks(books):
    return pd.DataFrame(
        {
            "game_id": ["0022600001"] * len(books),
            "market": ["totals"] * len(books),
            "book": books,
            "line_ts": [pd.Timestamp("2026-10-20T20:00Z")] * len(books),
            "left_line": [220.5] * len(books),
        }
    )


def test_stored_rows_are_read_under_the_old_slug():
    ticks = _ticks(["caesars_odds", "bet365"])
    ticks.attrs["season_fingerprints"] = {2026: ["x"]}
    out = fetch._alias_books(ticks)
    assert sorted(out["book"]) == ["bet365", "caesars"]
    assert out.attrs["season_fingerprints"] == {2026: ["x"]}


def test_excluding_caesars_also_drops_the_renamed_rows():
    out = fetch._alias_books(_ticks(["caesars_odds", "bet365"]), ("caesars",))
    assert out["book"].tolist() == ["bet365"]


def test_a_game_under_both_slugs_keeps_one_copy():
    out = fetch._alias_books(_ticks(["caesars_odds", "caesars"]))
    assert out["book"].tolist() == ["caesars"]
