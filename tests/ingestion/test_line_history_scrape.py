"""Parsing of the SBR ``__NEXT_DATA__`` payload.

The fixtures mirror the real payload's shape exactly (verified against event
316538, whose rows are already in the store), so a change to SBR's schema shows
up here rather than as silently wrong odds.
"""

from datetime import UTC, date, datetime

import pytest
from nba_ou.fetch_data.odds_sportsbook import scrape_sportsbook_line_history as sh

# 2025-04-04 23:00 UTC == 19:00 EDT, so the game belongs to the 4 April slate.
TIPOFF = "2025-04-04T23:00:00+00:00"


def _payload(**overrides):
    odds_view = {
        "sportsbook": "betmgm",
        "totalHistory": [
            {
                "oddsDate": "2025-04-03T23:35:20+00:00",
                "overOdds": -110,
                "underOdds": -110,
                "total": 221.5,
            },
            {
                "oddsDate": "2025-04-04T22:40:20+00:00",
                "overOdds": -110,
                "underOdds": -110,
                "total": 216.5,
            },
        ],
        "spreadHistory": [
            {
                "oddsDate": "2025-04-03T23:35:20+00:00",
                "homeOdds": -105,
                "awayOdds": -115,
                "homeSpread": 10.5,
                "awaySpread": -10.5,
            }
        ],
        "moneyLineHistory": [
            {
                "oddsDate": "2025-04-03T23:35:20+00:00",
                "homeOdds": 400,
                "awayOdds": -550,
                "homeSpread": None,
                "awaySpread": None,
            }
        ],
    }
    odds_view.update(overrides.pop("odds_view", {}))
    model = {
        "lineHistory": {
            "gameView": {
                "gameId": 316538,
                "startDate": overrides.pop("startDate", TIPOFF),
                "awayTeam": {"fullName": "Sacramento Kings", "shortName": "SAC"},
                "homeTeam": {"fullName": "Charlotte Hornets", "shortName": "CHA"},
                "gameStatusText": "Final",
                "awayTeamScore": 125,
                "homeTeamScore": 102,
            },
            "oddsViews": [odds_view],
        },
        "sportsbooks": [{"machineName": "betmgm", "name": "BetMGM"}],
    }
    return {"props": {"pageProps": {"lineHistoryModel": model}}}


def _daily_payload():
    return {
        "props": {
            "pageProps": {
                "oddsTables": [
                    {
                        "oddsTableModel": {
                            "gameRows": [
                                {
                                    "gameView": {
                                        "gameId": 363235,
                                        "startDate": "2026-04-12T22:00:00+00:00",
                                        "awayTeam": {"fullName": "Charlotte Hornets"},
                                        "homeTeam": {"fullName": "New York Knicks"},
                                        "gameStatusText": "Final",
                                    }
                                },
                                {
                                    "gameView": {
                                        "gameId": 363247,
                                        # 00:30 UTC belongs to the previous ET day
                                        "startDate": "2026-04-13T00:30:00+00:00",
                                        "awayTeam": {"fullName": "Phoenix Suns"},
                                        "homeTeam": {
                                            "fullName": "Oklahoma City Thunder"
                                        },
                                        "gameStatusText": "Final",
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        }
    }


class TestParseUtc:
    def test_converts_offset_to_utc_and_drops_seconds(self):
        parsed = sh._parse_utc("2025-04-04T23:05:20+00:00")
        assert parsed == datetime(2025, 4, 4, 23, 5, tzinfo=UTC)

    def test_non_utc_offset_is_normalised(self):
        # Same instant, written in Madrid local time.
        assert sh._parse_utc("2025-04-05T01:05:20+02:00") == sh._parse_utc(
            "2025-04-04T23:05:20+00:00"
        )

    def test_naive_timestamp_is_refused(self):
        # A value without an offset is exactly the ambiguity this module exists
        # to avoid, so it must not be guessed at.
        assert sh._parse_utc("2025-04-04 23:05:20") is None

    def test_missing_and_malformed_are_none(self):
        assert sh._parse_utc(None) is None
        assert sh._parse_utc("") is None
        assert sh._parse_utc("not a date") is None


class TestSlugifyBookmaker:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("BetMGM", "betmgm"),
            ("FanDuel", "fanduel"),
            ("bet365", "bet365"),
            ("DraftKings", "draftkings"),
            ("Caesars", "caesars"),
            # Must keep matching the slug already in lh_book, or new rows would
            # land on a duplicate book.
            ("Fanatics Sportsbook", "fanatics_sportsbook"),
        ],
    )
    def test_matches_stored_slugs(self, name, expected):
        assert sh.slugify_bookmaker(name) == expected


class TestParseLineHistoryPayload:
    def test_game_metadata(self):
        game = sh.parse_line_history_payload(_payload())
        assert game.event_id == 316538
        assert game.tipoff_utc == datetime(2025, 4, 4, 23, 0, tzinfo=UTC)
        assert game.team_away == "Sacramento Kings"
        assert game.team_home == "Charlotte Hornets"
        assert game.away_score == 125
        assert game.home_score == 102

    def test_game_date_and_season_come_from_eastern_time(self):
        # 23:00 UTC is 19:00 EDT, so still 4 April in ET.
        game = sh.parse_line_history_payload(_payload())
        assert game.game_date == date(2025, 4, 4)
        assert game.season_year == 2024

    def test_late_tipoff_belongs_to_the_previous_eastern_day(self):
        game = sh.parse_line_history_payload(
            _payload(startDate="2025-04-05T02:30:00+00:00")
        )
        assert game.game_date == date(2025, 4, 4)

    def test_totals_put_over_left_and_under_right(self):
        tick = sh.parse_line_history_payload(_payload()).ticks_for(sh.MARKET_TOTALS)[0]
        assert (tick.left_line, tick.left_price) == (221.5, -110)
        assert (tick.right_line, tick.right_price) == (221.5, -110)

    def test_spread_puts_away_left_and_home_right(self):
        tick = sh.parse_line_history_payload(_payload()).ticks_for(sh.MARKET_SPREAD)[0]
        assert (tick.left_line, tick.left_price) == (-10.5, -115)
        assert (tick.right_line, tick.right_price) == (10.5, -105)

    def test_moneyline_carries_prices_only(self):
        tick = sh.parse_line_history_payload(_payload()).ticks_for(sh.MARKET_MONEYLINE)[
            0
        ]
        assert tick.left_line is None and tick.right_line is None
        assert (tick.left_price, tick.right_price) == (-550, 400)

    def test_minutes_to_tip_is_negative_before_tipoff(self):
        ticks = sh.parse_line_history_payload(_payload()).ticks_for(sh.MARKET_TOTALS)
        assert ticks[0].minutes_to_tip == -1405
        assert ticks[1].minutes_to_tip == -20
        assert all(tick.is_pregame for tick in ticks)

    def test_tick_at_or_after_tipoff_is_not_pregame(self):
        payload = _payload(
            odds_view={
                "totalHistory": [
                    {
                        "oddsDate": "2025-04-04T23:05:20+00:00",
                        "overOdds": -110,
                        "underOdds": -110,
                        "total": 215.5,
                    }
                ]
            }
        )
        tick = sh.parse_line_history_payload(payload).ticks_for(sh.MARKET_TOTALS)[0]
        assert tick.minutes_to_tip == 5
        assert not tick.is_pregame

    def test_only_the_earliest_tick_is_the_opener(self):
        ticks = sh.parse_line_history_payload(_payload()).ticks_for(sh.MARKET_TOTALS)
        assert [tick.is_opener for tick in ticks] == [True, False]

    def test_ticks_sharing_a_minute_collapse_to_the_last(self):
        # Truncation must never produce two rows on the same primary key.
        payload = _payload(
            odds_view={
                "totalHistory": [
                    {
                        "oddsDate": "2025-04-04T22:40:10+00:00",
                        "overOdds": -110,
                        "underOdds": -110,
                        "total": 216.5,
                    },
                    {
                        "oddsDate": "2025-04-04T22:40:50+00:00",
                        "overOdds": -105,
                        "underOdds": -115,
                        "total": 217.5,
                    },
                ]
            }
        )
        ticks = sh.parse_line_history_payload(payload).ticks_for(sh.MARKET_TOTALS)
        assert len(ticks) == 1
        assert ticks[0].left_line == 217.5

    def test_markets_can_be_restricted(self):
        game = sh.parse_line_history_payload(_payload(), markets=[sh.MARKET_TOTALS])
        assert {tick.market for tick in game.ticks} == {sh.MARKET_TOTALS}

    def test_book_slug_comes_from_the_display_name(self):
        game = sh.parse_line_history_payload(_payload())
        assert game.books == ("betmgm",)

    def test_payload_without_game_id_raises(self):
        payload = _payload()
        payload["props"]["pageProps"]["lineHistoryModel"]["lineHistory"]["gameView"][
            "gameId"
        ] = None
        with pytest.raises(sh.LineHistoryFetchError):
            sh.parse_line_history_payload(payload)

    def test_payload_without_start_date_raises(self):
        # Without a tipoff there is no mins_to_tip, and mins_to_tip is the
        # leakage filter -- loading such a game would be worse than skipping it.
        with pytest.raises(sh.LineHistoryFetchError):
            sh.parse_line_history_payload(_payload(startDate=None))


class TestParseDailyPayload:
    def test_lists_every_game_with_its_eastern_date(self):
        summaries = sh.parse_daily_payload(_daily_payload())
        assert [s.event_id for s in summaries] == [363235, 363247]
        # Both belong to the 12 April slate even though one tips after midnight UTC.
        assert {s.game_date for s in summaries} == {date(2026, 4, 12)}

    def test_empty_slate_returns_nothing(self):
        assert (
            sh.parse_daily_payload({"props": {"pageProps": {"oddsTables": []}}}) == []
        )


class TestExtractNextData:
    def test_reads_the_embedded_script(self):
        html = (
            '<html><body><script id="__NEXT_DATA__" type="application/json">'
            '{"props": {"a": 1}}</script></body></html>'
        )
        assert sh.extract_next_data(html) == {"props": {"a": 1}}

    def test_missing_payload_raises(self):
        with pytest.raises(sh.LineHistoryFetchError):
            sh.extract_next_data("<html><body>blocked</body></html>")

    def test_malformed_payload_raises(self):
        html = '<script id="__NEXT_DATA__" type="application/json">{oops</script>'
        with pytest.raises(sh.LineHistoryFetchError):
            sh.extract_next_data(html)
