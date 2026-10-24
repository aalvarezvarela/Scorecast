"""Covers as SBR's fallback in the stored reads (``line_history_covers.fallback``)."""

from __future__ import annotations

import pandas as pd
from nba_ou.postgre_db.line_history_aiven.fetch import TICK_COLUMNS
from nba_ou.postgre_db.line_history_covers import fallback

TIP = pd.Timestamp("2024-03-29T23:00Z")


def _ticks(rows) -> pd.DataFrame:
    """rows: (game_id, book, minutes before tip)."""
    return pd.DataFrame(
        [
            {
                "game_id": game,
                "season_year": 2023,
                "market": "totals",
                "book": book,
                "line_ts": TIP - pd.Timedelta(minutes=minutes),
                "minutes_before_tip": float(minutes),
                "is_opener": False,
                "left_line": 220.5,
                "left_price": -110,
                "right_line": 220.5,
                "right_price": -110,
            }
            for game, book, minutes in rows
        ],
        columns=TICK_COLUMNS,
    )


class TestFallbackGames:
    def test_game_sbr_lacks_comes_from_covers(self):
        sbr = _ticks([("A", "bet365", 8)])
        covers = _ticks([("A", "bet365", 8), ("B", "bet365", 6)])
        assert fallback.fallback_game_ids(sbr, covers) == ["B"]

    def test_frozen_sbr_game_comes_from_covers(self):
        # SBR stops 11 h before tip, Covers 6 min before (2024-03-28).
        sbr = _ticks([("A", "bet365", 660)])
        covers = _ticks([("A", "bet365", 6)])
        assert fallback.fallback_game_ids(sbr, covers) == ["A"]

    def test_same_feed_within_margin_keeps_sbr(self):
        # The usual case: Covers a few minutes off SBR, or frozen the same.
        sbr = _ticks([("A", "bet365", 30), ("B", "bet365", 400)])
        covers = _ticks([("A", "bet365", 10), ("B", "bet365", 400)])
        assert fallback.fallback_game_ids(sbr, covers) == []

    def test_empty_sbr_takes_every_covers_game(self):
        covers = _ticks([("B", "bet365", 6), ("A", "fanduel", 9)])
        assert fallback.fallback_game_ids(_ticks([]), covers) == ["A", "B"]


class TestApply:
    def test_game_is_taken_whole_and_covers_only_books_dropped(self):
        sbr = _ticks([("A", "bet365", 660), ("A", "fanduel", 700), ("C", "bet365", 9)])
        sbr.attrs["season_fingerprints"] = {2023: ["x"]}
        covers = _ticks(
            [
                ("A", "bet365", 6),
                ("A", "thescore", 6),
                ("A", "consensus_opener", 900),
                ("C", "bet365", 7),
            ]
        )
        out = fallback.apply(sbr, covers)
        assert out.attrs["covers_game_ids"] == ["A"]
        assert out.attrs["season_fingerprints"] == {2023: ["x"]}
        a = out[out["game_id"] == "A"]
        # SBR's fanduel of a Covers game is not mixed in.
        assert a["book"].tolist() == ["bet365"]
        assert a["minutes_before_tip"].tolist() == [6.0]
        c = out[out["game_id"] == "C"]
        assert c["minutes_before_tip"].tolist() == [9.0]  # SBR kept
        assert out["left_price"].dtype == sbr["left_price"].dtype

    def test_nothing_to_take_returns_sbr_rows(self):
        sbr = _ticks([("A", "bet365", 9)])
        out = fallback.apply(sbr, _ticks([("A", "bet365", 9)]))
        pd.testing.assert_frame_equal(out, sbr)
        assert out.attrs["covers_game_ids"] == []


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.params = params

    def fetchone(self):
        return self.rows.pop(0)


class _Conn:
    def __init__(self, rows):
        self._cursor = _Cursor(rows)

    def cursor(self):
        return self._cursor


def test_fingerprints_change_only_for_seasons_covers_holds(monkeypatch):
    monkeypatch.setattr(fallback.schema_mod, "schema_exists", lambda conn: True)
    empty = (0, None, 0, None, 0, "m", "b")
    held = (10, 123, 2, 456, 2, "m", "b")
    conn = _Conn([empty, held])
    out = fallback.extend_fingerprints({2022: ["sbr22"], 2023: ["sbr23"]}, conn)
    assert out[2022] == ["sbr22"]
    assert out[2023] == ["sbr23", "covers", "10", "123", "2", "456", "2", "m", "b"]
