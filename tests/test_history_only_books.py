"""Adding a book (BetRivers): stored first, admitted as a feature deliberately.

* the closing scraper and the Supabase table carry the book's columns;
* backfilling it fills only its own NULL columns on stored games;
* the history-only gate drops a listed book on every read path;
* BetRivers itself is admitted, so it reaches every feature book list.
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
        out = oc.drop_history_only_book_columns(df, books=("betrivers",))
        assert list(out.columns) == [
            "game_id",
            "total_bet365_line_over",
            "spread_fanduel_line_home",
        ]

    def test_betrivers_is_admitted(self):
        assert "betrivers" not in oc.HISTORY_ONLY_BOOKS

    def test_hard_rock_bet_is_history_only(self):
        # The slug line history stores for "Hard Rock Bet"; the exclusion SQL
        # matches it exactly, so a spelling drift would silently admit it.
        from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import _slugify_book

        assert _slugify_book("Hard Rock Bet") in oc.HISTORY_ONLY_BOOKS

    def test_hard_rock_bet_columns_are_dropped(self):
        df = pd.DataFrame(
            columns=[
                "game_id",
                "total_hard_rock_bet_line_over",
                "ml_hard_rock_bet_price_home",
                "total_bet365_line_over",
            ]
        )
        out = oc.drop_history_only_book_columns(df)
        assert list(out.columns) == ["game_id", "total_bet365_line_over"]

    def test_default_drop_is_a_no_op_with_nothing_listed(self, monkeypatch):
        monkeypatch.setattr(oc, "HISTORY_ONLY_BOOKS", ())
        df = pd.DataFrame(columns=["total_betrivers_line_over"])
        assert list(oc.drop_history_only_book_columns(df, books=())) == [
            "total_betrivers_line_over"
        ]


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

    def test_default_only_refreshes_odds_from_a_pre_tip_scrape(self):
        text = db.build_upsert_query("s", "t", self.COLS).as_string(None)
        assert '"total_bet365_line_over" = CASE WHEN' in text
        assert "EXCLUDED.scraped_at < EXCLUDED.sbr_start_time_utc" in text
        assert '"t".closes_repaired_at IS NULL' in text
        # The key is never rewritten.
        assert '"game_id" =' not in text

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

    def test_closing_loader_keeps_an_admitted_book(self, stored):
        out = loader.load_odds_sportsbook_from_db()
        assert "total_betrivers_line_over" in out.columns
        assert "total_bet365_line_over" in out.columns

    def test_closing_loader_drops_a_listed_book(self, stored, monkeypatch):
        listed = ("betrivers",)
        monkeypatch.setattr(
            loader,
            "drop_history_only_book_columns",
            lambda df: oc.drop_history_only_book_columns(df, books=listed),
        )
        out = loader.load_odds_sportsbook_from_db()
        assert "total_betrivers_line_over" not in out.columns
        assert "total_bet365_line_over" in out.columns
        opted_in = loader.load_odds_sportsbook_from_db(include_history_only_books=True)
        assert "total_betrivers_line_over" in opted_in.columns


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


class TestBetRiversReachesFeatures:
    def test_training_column_allow_list(self):
        from nba_ou.data_processing.merged_home_away_data import select_train_columns

        for col in (
            "total_betrivers_price_over",
            "spread_betrivers_line_home",
            "ml_betrivers_price_away",
        ):
            assert col in select_train_columns.ODDS_COLUMNS

    def test_total_line_is_merged_under_its_canonical_name(self):
        from nba_ou.data_processing.team import merge_game_df_with_odds_by_game_id as m

        df_team = pd.DataFrame({"GAME_ID": ["1", "1"], "HOME": [True, False]})
        df_odds = pd.DataFrame(
            {
                "game_id": ["1"],
                "spread_bet365_line_home": [-3.5],
                "spread_bet365_line_away": [3.5],
                "ml_bet365_price_home": [1.6],
                "ml_bet365_price_away": [2.4],
                "total_bet365_line_over": [220.5],
                "total_betrivers_line_over": [221.0],
            }
        )
        out = m.merge_total_spread_moneyline_by_game_id(
            df_odds=df_odds,
            df_team=df_team,
            book="bet365",
            total_line_book="bet365",
            total_lines_mode="all",
        )
        assert out["ODDS_TOTAL_LINE_betrivers"].tolist() == [221.0, 221.0]

    def test_joins_the_spread_consensus(self):
        from nba_ou.data_processing.odds import canonical_markets as cm

        df = pd.DataFrame(
            {"spread_bet365_line_home": [-3.0], "spread_betrivers_line_home": [-5.0]}
        )
        out = cm.add_canonical_spread_columns(df)
        assert out[cm.SPREAD_BOOK_COUNT_COL].iloc[0] == 2
        assert out[cm.SPREAD_CONSENSUS_MEDIAN_COL].iloc[0] == 4.0

    def test_missing_values_are_kept_not_imputed(self):
        from nba_ou.data_processing.missing_data import handle_missing_data as h

        assert h._is_market_keep_na("total_betrivers_price_over")
