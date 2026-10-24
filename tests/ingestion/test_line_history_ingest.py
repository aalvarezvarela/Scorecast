"""Turning scraped games into ``lh_line``/``lh_game`` rows.

Covers the parts that are easy to get silently wrong: resolving SBR events to
``nba_games`` ids, the storage encodings, and the staleness diff that decides
what gets scraped in the first place.
"""

from datetime import UTC, date, datetime

import pandas as pd
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven import ingest as ing

TIPOFF = datetime(2025, 4, 4, 23, 0, tzinfo=UTC)
MARKET_IDS = {"totals": 1, "point_spread": 2, "money_line": 3}
BOOK_IDS = {"betmgm": 2, "fanduel": 6}


def _tick(**overrides) -> LineTick:
    row = {
        "market": "totals",
        "book_slug": "betmgm",
        "book_name": "BetMGM",
        "line_ts": datetime(2025, 4, 4, 22, 40, tzinfo=UTC),
        "minutes_to_tip": -20,
        "is_opener": False,
        "left_line": 216.5,
        "left_price": -110,
        "right_line": 216.5,
        "right_price": -110,
    }
    row.update(overrides)
    return LineTick(**row)


def _game(ticks=None, **overrides) -> ScrapedGame:
    row = {
        "event_id": 316538,
        "game_date": date(2025, 4, 4),
        "season_year": 2024,
        "tipoff_utc": TIPOFF,
        "team_away": "Sacramento Kings",
        "team_home": "Charlotte Hornets",
        "status_text": "Final",
        "away_score": 125,
        "home_score": 102,
        "ticks": tuple(ticks if ticks is not None else [_tick()]),
    }
    row.update(overrides)
    return ScrapedGame(**row)


def _game_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022401118"],
            "game_date": [date(2025, 4, 4)],
            "game_season_year": [2024],
            "team_home": ["Charlotte Hornets"],
            "team_away": ["Sacramento Kings"],
        }
    )


def _build(games, **kwargs):
    return ing.build_frames(
        games,
        game_index=kwargs.pop("game_index", _game_index()),
        book_ids=kwargs.pop("book_ids", BOOK_IDS),
        market_ids=kwargs.pop("market_ids", MARKET_IDS),
        **kwargs,
    )


class TestGameResolution:
    def test_matched_game_gets_its_nba_game_id(self):
        rows, dim, stats = _build([_game()])
        assert rows["game_id"].tolist() == ["0022401118"]
        assert stats.matched_games == 1
        assert dim.loc[0, "event_id"] == 316538

    def test_unmatched_game_is_reported_not_loaded(self):
        # Preseason games are absent from nba_games; they must not be keyed on
        # a guess.
        rows, dim, stats = _build([_game(game_date=date(2025, 4, 5))])
        assert rows.empty and dim.empty
        assert stats.matched_games == 0
        assert stats.dropped["preseason_or_unmatched_game"] == 1
        assert len(stats.unmatched_games) == 1

    def test_season_year_comes_from_nba_games(self):
        rows, _, _ = _build([_game(season_year=1999)])
        assert rows["season_year"].tolist() == [2024]

    def test_tipoff_taken_from_the_page(self):
        _, dim, _ = _build([_game()])
        assert dim.loc[0, "tipoff_utc"] == TIPOFF


class TestEncoding:
    def test_lines_are_stored_as_doubled_half_points(self):
        rows, _, _ = _build([_game()])
        assert rows.loc[0, "left_line"] == 433  # 216.5 * 2
        assert rows.loc[0, "right_line"] == 433

    def test_moneyline_rows_carry_prices_only(self):
        tick = _tick(
            market="money_line",
            left_line=None,
            right_line=None,
            left_price=-550,
            right_price=400,
        )
        rows, _, _ = _build([_game([tick])])
        assert pd.isna(rows.loc[0, "left_line"])
        assert rows.loc[0, "left_price"] == -550

    def test_pregame_flag_follows_minutes_to_tip(self):
        ticks = [
            _tick(),
            _tick(minutes_to_tip=5, line_ts=datetime(2025, 4, 4, 23, 5, tzinfo=UTC)),
        ]
        rows, _, _ = _build([_game(ticks)])
        assert rows.sort_values("mins_to_tip")["is_pregame"].tolist() == [True, False]

    def test_unknown_book_is_dropped(self):
        rows, _, stats = _build([_game([_tick(book_slug="caesars")])])
        assert rows.empty
        assert stats.dropped["unknown_market_or_book"] == 1

    def test_implausible_pregame_total_is_nulled_but_row_kept(self):
        # A dropped decimal turns 228.5 into 2285; the prices are still good.
        rows, _, stats = _build([_game([_tick(left_line=2285.0, right_line=2285.0)])])
        assert len(rows) == 1
        assert pd.isna(rows.loc[0, "left_line"])
        assert rows.loc[0, "left_price"] == -110
        assert stats.repaired["implausible_pregame_line"] == 1

    def test_duplicate_timepoints_collapse_to_one_row(self):
        rows, _, stats = _build([_game([_tick(), _tick(left_price=-105)])])
        assert len(rows) == 1
        assert stats.dropped["duplicate_timepoint"] == 1


class TestTipoffCrossCheck:
    def _schedule(self, tipoff):
        return pd.DataFrame(
            {
                "game_id": ["0022401118"],
                "tipoff_utc": [pd.Timestamp(tipoff)],
                "team_home": ["CHA"],
                "team_away": ["SAC"],
            }
        )

    def test_agreement_is_silent(self):
        _, _, stats = _build([_game()], schedule=self._schedule(TIPOFF))
        assert stats.tipoff_disagreements == []

    def test_disagreement_is_reported_but_the_page_still_wins(self):
        feed = self._schedule("2025-04-04T21:00:00Z")
        _, dim, stats = _build([_game()], schedule=feed)
        assert len(stats.tipoff_disagreements) == 1
        assert dim.loc[0, "tipoff_utc"] == TIPOFF

    def test_team_codes_come_from_the_feed_when_present(self):
        _, dim, _ = _build([_game()], schedule=self._schedule(TIPOFF))
        assert dim.loc[0, "team_home"] == "CHA"
        assert dim.loc[0, "team_away"] == "SAC"

    def test_falls_back_to_scraped_names_without_a_feed(self):
        _, dim, _ = _build([_game()])
        assert dim.loc[0, "team_home"] == "Charlotte Hornets"


class TestFindMissingGames:
    class _Conn:
        def __init__(self, stored, first):
            self._stored, self._first = stored, first

        def cursor(self):
            outer = self

            class _Cur:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, query, *args):
                    self._is_range = "MIN(game_date)" in str(query)

                def fetchall(self):
                    return [(g,) for g in outer._stored]

                def fetchone(self):
                    return (outer._first, None)

            return _Cur()

    def _games(self):
        return pd.DataFrame(
            {
                "game_id": ["A", "B", "C"],
                "game_date": [date(2025, 1, 1), date(2025, 1, 2), date(2020, 1, 1)],
                "season_year": [2024, 2024, 2019],
            }
        )

    def test_returns_games_absent_from_the_store(self):
        conn = self._Conn({"A"}, date(2021, 10, 19))
        missing = ing.find_missing_games(conn, self._games())
        assert missing["game_id"].tolist() == ["B"]

    def test_ignores_games_older_than_the_store(self):
        # Before the store's first game there is no gap, only history that was
        # never collected.
        conn = self._Conn(set(), date(2021, 10, 19))
        missing = ing.find_missing_games(conn, self._games())
        assert "C" not in missing["game_id"].tolist()

    def test_explicit_window_overrides_the_default_start(self):
        conn = self._Conn(set(), date(2021, 10, 19))
        missing = ing.find_missing_games(
            conn, self._games(), start=date(2025, 1, 2), end=date(2025, 1, 2)
        )
        assert missing["game_id"].tolist() == ["B"]

    def test_missing_dates_are_unique_and_sorted(self):
        frame = pd.DataFrame(
            {"game_id": ["A", "B"], "game_date": [date(2025, 1, 2), date(2025, 1, 1)]}
        )
        assert ing.missing_dates(frame) == [date(2025, 1, 1), date(2025, 1, 2)]

    def test_no_games_means_no_dates(self):
        assert ing.missing_dates(pd.DataFrame()) == []


class TestEmptyInput:
    def test_no_games_yields_empty_frames(self):
        rows, dim, stats = _build([])
        assert rows.empty and dim.empty
        assert stats.scraped_games == 0


class TestResolveGameId:
    def _lookup(self):
        return ing.build_game_lookup(_game_index())

    def test_resolves_a_listed_game(self):
        hit = ing.resolve_game_id(
            self._lookup(),
            game_date=date(2025, 4, 4),
            team_away="Sacramento Kings",
            team_home="Charlotte Hornets",
        )
        assert hit == ("0022401118", 2024)

    def test_home_and_away_are_not_interchangeable(self):
        assert (
            ing.resolve_game_id(
                self._lookup(),
                game_date=date(2025, 4, 4),
                team_away="Charlotte Hornets",
                team_home="Sacramento Kings",
            )
            is None
        )

    def test_unknown_team_name_returns_none(self):
        # Used to skip fetching preseason games, so it must not raise.
        assert (
            ing.resolve_game_id(
                self._lookup(),
                game_date=date(2025, 4, 4),
                team_away="Not A Team",
                team_home="Charlotte Hornets",
            )
            is None
        )

    def test_empty_index_yields_empty_lookup(self):
        assert ing.build_game_lookup(pd.DataFrame()) == {}
