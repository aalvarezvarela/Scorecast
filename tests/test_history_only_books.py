"""Storing a new book (BetRivers) without silently changing any training frame.

Three guarantees:

* the closing scraper and the Supabase table carry the book's columns;
* backfilling it fills only its own NULL columns on stored games;
* every read path into a dataset drops it until it is admitted deliberately.
"""

import pandas as pd
import pytest
from nba_ou.config import odds_columns as oc
from nba_ou.fetch_data.odds_sportsbook.process_money_line_data import (
    ML_BOOKS,
    build_one_game_row_from_moneyline_group,
)
from nba_ou.fetch_data.odds_sportsbook.process_spread_data import SPREAD_BOOKS
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import TOTAL_BOOKS
from nba_ou.postgre_db.odds_sportsbook.create_db import (
    create_odds_sportsbook_db as db,
)
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db import (
    fetch_data_from_odds_sportsbook_db as loader,
)


class TestBookColumnMatching:
    @pytest.mark.parametrize(
        "column",
        [
            "total_betrivers_line_over",
            "spread_betrivers_price_home",
            "ml_betrivers_price_away",
            "ODDS_TOTAL_LINE_betrivers",
        ],
    )
    def test_matches_the_book_as_a_token(self, column):
        assert oc.is_book_column(column, "betrivers")

    def test_multi_token_slug(self):
        assert oc.is_book_column(
            "total_fanatics_sportsbook_line_over", "fanatics_sportsbook"
        )
        assert not oc.is_book_column("total_fanduel_line_over", "fanatics_sportsbook")

    def test_substring_is_not_a_match(self):
        # "bet365" must not capture a longer slug that merely starts with it.
        assert not oc.is_book_column("total_bet3650_line_over", "bet365")
        assert not oc.is_book_column("total_points", "betrivers")

    def test_drop_keeps_every_other_book(self):
        df = pd.DataFrame(
            columns=[
                "game_id",
                "total_bet365_line_over",
                "total_betrivers_line_over",
                "ml_betrivers_price_home",
                "spread_fanduel_line_home",
            ]
        )
        out = oc.drop_history_only_book_columns(df)
        assert list(out.columns) == [
            "game_id",
            "total_bet365_line_over",
            "spread_fanduel_line_home",
        ]

    def test_betrivers_is_history_only(self):
        assert "betrivers" in oc.HISTORY_ONLY_BOOKS


class TestScraperAndTable:
    @pytest.mark.parametrize("books", [TOTAL_BOOKS, SPREAD_BOOKS, ML_BOOKS])
    def test_scraper_book_lists_carry_betrivers(self, books):
        assert "betrivers" in books

    def test_table_has_every_betrivers_column(self):
        assert db.book_columns(("betrivers",)) == [
            "total_betrivers_line_over",
            "total_betrivers_price_over",
            "total_betrivers_line_under",
            "total_betrivers_price_under",
            "spread_betrivers_line_away",
            "spread_betrivers_price_away",
            "spread_betrivers_line_home",
            "spread_betrivers_price_home",
            "ml_betrivers_price_away",
            "ml_betrivers_price_home",
        ]

    def test_scraped_moneyline_reaches_the_betrivers_columns(self):
        group = pd.DataFrame(
            {
                "row_index": [0, 1],
                "team_name": ["Away", "Home"],
                "score": [100, 110],
                "date": ["2024-03-15", "2024-03-15"],
                "event_id": [1, 1],
                "season": [2023, 2023],
                "betrivers_price": [150, -175],
            }
        )
        row = build_one_game_row_from_moneyline_group(group)
        assert row["ml_betrivers_price_away"] == 150
        assert row["ml_betrivers_price_home"] == -175


class TestFillOnlyUpsert:
    COLS = ["game_id", "total_bet365_line_over", "total_betrivers_line_over"]

    def test_default_never_touches_a_stored_game(self):
        text = db.build_upsert_query("s", "t", self.COLS).as_string(None)
        assert "DO NOTHING" in text
        assert "UPDATE" not in text

    def test_backfill_updates_only_its_columns_and_only_nulls(self):
        text = db.build_upsert_query(
            "s", "t", self.COLS, ["total_betrivers_line_over"]
        ).as_string(None)
        assert "DO UPDATE SET" in text
        assert (
            '"total_betrivers_line_over" = COALESCE("t"."total_betrivers_line_over", '
            'EXCLUDED."total_betrivers_line_over")'
        ) in text
        # Another book's stored value is never in the SET clause.
        assert '"total_bet365_line_over" =' not in text

    def test_unknown_fill_column_is_rejected(self):
        with pytest.raises(ValueError):
            db.build_upsert_query("s", "t", self.COLS, ["total_nobook_line_over"])


class TestReadPathDropsHistoryOnlyBooks:
    @pytest.fixture
    def stored(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "game_id": ["1"],
                "total_bet365_line_over": [220.5],
                "total_betrivers_line_over": [221.5],
            }
        )

        class _Conn:
            def close(self):
                pass

        monkeypatch.setattr(loader, "connect_nba_db", lambda: _Conn())
        monkeypatch.setattr(
            loader.sql.Composed, "as_string", lambda self, conn=None: "SELECT"
        )
        monkeypatch.setattr(loader.pd, "read_sql_query", lambda *a, **k: frame.copy())
        return frame

    def test_closing_loader_drops_betrivers_by_default(self, stored):
        out = loader.load_odds_sportsbook_from_db()
        assert "total_betrivers_line_over" not in out.columns
        assert "total_bet365_line_over" in out.columns

    def test_closing_loader_can_opt_in(self, stored):
        out = loader.load_odds_sportsbook_from_db(include_history_only_books=True)
        assert "total_betrivers_line_over" in out.columns


class TestAddMissingColumns:
    class _Cur:
        def __init__(self, existing):
            self.existing = existing
            self.statements = []

        def execute(self, query, params=None):
            self.statements.append(
                query if isinstance(query, str) else query.as_string(None)
            )

        def fetchall(self):
            return [(name,) for name in self.existing]

    def test_nothing_is_altered_when_the_table_is_current(self):
        cur = self._Cur([name for name, _ in db._sportsbook_columns()])
        db._add_missing_columns(cur, "s", "t")
        # One information_schema read, and no lock-taking ALTER at all.
        assert len(cur.statements) == 1
        assert not any("ALTER" in s for s in cur.statements)

    def test_only_missing_columns_are_added(self):
        missing = set(db.book_columns(("betrivers",)))
        cur = self._Cur(
            [name for name, _ in db._sportsbook_columns() if name not in missing]
        )
        db._add_missing_columns(cur, "s", "t")
        altered = [s for s in cur.statements if s.startswith("ALTER")]
        assert len(altered) == len(missing)
        assert all("betrivers" in s for s in altered)
        assert any("lock_timeout" in s for s in cur.statements)
