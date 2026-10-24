"""OddsLogic (WagerTalk) line history: widget answers -> the SBR store's rows.

Pins the parsing rules the store relies on -- the stale-timestamp filter that
keeps in-play quotes out, the per-minute collapse, the side conventions, the
year of a year-less date -- plus the request plan, the empty-answer retries
and that writes stay in the ``line_history_oddslogic`` schema. Fixtures are
cut from real answers of 2026-10-20 (PHI @ NY, FanDuel).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from nba_ou.fetch_data.odds_oddslogic import client as client_mod
from nba_ou.fetch_data.odds_oddslogic import parse
from nba_ou.fetch_data.odds_oddslogic.books import ACTIVE_BOOKS, OL_BOOKS
from nba_ou.postgre_db.line_history_oddslogic import schema as schema_mod
from nba_ou.postgre_db.line_history_oddslogic import store
from nba_ou.postgre_db.line_history_oddslogic import update as update_mod

DAY = date(2026, 10, 20)
TIPOFF = datetime(2026, 10, 20, 23, 0, tzinfo=UTC)  # the schedule feed's
FANDUEL = 112

SCHEDULE = (
    "{0,All sports,2026-10-21 00:23:42 Pacific Time}"
    '{{header3,1,2,1,339037,"COLLEGE FOOTBALL - Tuesday, October 20th",,2026-10-20},'
    "{301,,7520607,,2026-10-20,2026-10-20,10-20,23:00,23:00,2026-10-20 23:00:00,"
    ",,,,,,{Middle Tennessee St.,MTU,,,,Florida International,FIU,,,}}}"
    '{{header6,2,3,1,331035,"NBA - Tuesday, October 20th",,2026-10-20},'
    "{501,,7398652,,2026-10-20,2026-10-20,10-20,19:08,19:08,2026-10-20 19:08:00,"
    ",,,,,,{Boston,BOS,,,,Detroit,DET,,,}}"
    "{503,,7398653,,2026-10-20,2026-10-20,10-20,23:10,23:10,2026-10-20 23:10:00,"
    ",,,,,,{Philadelphia,PHI,,,,New York,NY,,,}}"
    "{547,,7538280,,2026-10-23,2026-10-23,10-23,01:40,01:40,2026-10-24 01:40:00,"
    ',1,"Moody Center","Austin, TX",,,{Houston,HOU,,,,San Antonio,SA,,,}}}'
    '{{header12,2,8,1,338026,"WNBA FINALS - Tuesday, October 20th",,2026-10-20},'
    "{601,,7400001,,2026-10-20,2026-10-20,10-20,00:00,00:00,2026-10-21 00:00:00,"
    ",,,,,,{Las Vegas,LV,,,,New York,NY,,,}}}"
)

HEADER = (
    "[line_history{updated,number_of_teams,sportsbook,period,league,date,"
    "away_number,away_team,time,home_humber,home_team,types{line_type{line_date,"
    "line_time,away_line,home_line.public}}}]{2026-10-23 03:14:25 Pacific Time,2,"
    "Fanduel,Game,NBA,10/20,503,Philadelphia,11:10p,504,New York"
)

# FanDuel's spread: the first three rows repeat 15:35:21 over a -1.5 -132 that
# no other book showed (they were at -4.5); only the run's oldest row is real.
SPREAD_ANSWER = (
    HEADER + "{SPREADS{{"
    '{10/20,03:35:21pm,1&frac12;+100,-1&frac12;-132,"NY 57%, 40%"}'
    '{10/20,03:35:21pm,1&frac12;+100,-1&frac12;-132,"NY 57%, 40%"}'
    '{10/20,03:35:21pm,4&frac12;-108,-4&frac12;-112,"NY 54%, 35%"}'
    '{10/20,03:35:21pm,4&frac12;-105,-4&frac12;-115,"PHI 50%, 69%"}'
    '{10/20,12:21:24pm,4&frac12;-105,-4&frac12;-115,"PHI 52%, 71%"}'
    '{10/20,02:23:58am,5&frac12;-122,-5&frac12;+100,"PHI 66%, 68%"}'
    "}}"
    "TOTALS{{"
    '{10/20,03:35:21pm,231&frac12;o-110,231&frac12;u-110,"u61%, u54%"}'
    '{10/20,07:16:32am,232&frac12;o-102,232&frac12;u-120,"u66%, u57%"}'
    '{10/20,06:51:36am,,,"u66%, u57%"}'
    '{12/30,09:33:40pm,233&frac12;o-110,233&frac12;u-110," "}'
    "}}}"
)

ML_ANSWER = (
    HEADER + "{MONEYLINES{{"
    '{10/20,10:45:46pm,+145,-175,"PHI 73%, 58%"}'
    '{10/20,07:42:50pm,+150,-180,"PHI 62%, 57%"}'
    '{10/20,07:42:10pm,ev,-120,"PHI 62%, 57%"}'
    "}}"
    "TOTALS{{"
    '{10/20,03:35:21pm,231&frac12;o-110,231&frac12;u-110,"u61%, u54%"}'
    "}}}"
)


def _game(**overrides) -> parse.OddsLogicGame:
    fields = dict(
        game_date=DAY,
        rotation=503,
        event_id=7398653,
        tipoff_utc=datetime(2026, 10, 20, 23, 10, tzinfo=UTC),
        team_away="Philadelphia",
        away_abbr="PHI",
        team_home="New York",
        home_abbr="NY",
    )
    fields.update(overrides)
    return parse.OddsLogicGame(**fields)


def _ticks(answers=(SPREAD_ANSWER, ML_ANSWER)):
    return parse.book_ticks(
        list(answers), book_id=FANDUEL, game_date=DAY, tipoff_utc=TIPOFF
    )


def _market(ticks, market):
    return [t for t in ticks if t.market == market]


class TestSchedule:
    def test_only_nba_games(self):
        games = parse.parse_schedule(SCHEDULE, DAY)
        assert [g.rotation for g in games] == [501, 503, 547]

    def test_fields(self):
        game = parse.parse_schedule(SCHEDULE, DAY)[1]
        assert game == _game()

    def test_neutral_site_record(self):
        game = parse.parse_schedule(SCHEDULE, DAY)[2]
        assert (game.team_away, game.team_home) == ("Houston", "San Antonio")
        assert game.tipoff_utc == datetime(2026, 10, 24, 1, 40, tzinfo=UTC)

    @pytest.mark.parametrize(
        "label, nba",
        [
            ("NBA - Tuesday, October 20th", True),
            ("NBA PRESEASON - Friday, October 9th", False),
            ("NBA SUMMER LEAGUE - Las Vegas", False),
            ("NBA SERIES PRICES (BEST OF 7)", False),
            ("WNBA FINALS - Tuesday, October 20th", False),
        ],
    )
    def test_header_filter(self, label, nba):
        assert parse._is_nba_header(label) is nba


class TestQuotes:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("231.5o-110", (231.5, -110, "o")),
            ("231.5u+105", (231.5, 105, "u")),
            ("-4.5-108", (-4.5, -108, None)),
            ("4.5+100", (4.5, 100, None)),
            ("2-109", (2.0, -109, None)),
            ("-.5-20", (-0.5, -20, None)),
            ("pk-110", (0.0, -110, None)),
            ("-2.5 ev", (-2.5, 100, None)),
            ("", (None, None, None)),
            ("n/a", (None, None, None)),
        ],
    )
    def test_parse_quote(self, text, expected):
        assert parse.parse_quote(text) == expected

    def test_moneyline_prices_only(self):
        row = parse.HistoryRow(TIPOFF, "+145", "-175")
        assert parse._sides("money_line", row) == (None, 145, None, -175)
        even = parse.HistoryRow(TIPOFF, "ev", "-120")
        assert parse._sides("money_line", even) == (None, 100, None, -120)

    def test_totals_follow_the_side_letter(self):
        swapped = parse.HistoryRow(TIPOFF, "231.5u-105", "231.5o-115")
        assert parse._sides("totals", swapped) == (231.5, -115, 231.5, -105)


class TestTimestamps:
    def test_utc_twelve_hour_clock(self):
        assert parse._row_timestamp("10/20", "12:21:24pm", DAY) == datetime(
            2026, 10, 20, 12, 21, 24, tzinfo=UTC
        )
        assert parse._row_timestamp("10/19", "00:08:49am", DAY) == datetime(
            2026, 10, 19, 0, 8, 49, tzinfo=UTC
        )

    def test_year_closest_to_the_game(self):
        january = date(2027, 1, 15)
        assert parse._row_timestamp("12/30", "09:33:40pm", january).year == 2026
        october = date(2026, 10, 31)
        assert parse._row_timestamp("11/01", "01:00:00am", october).year == 2026


class TestStaleRepeats:
    def test_run_keeps_its_oldest_row(self):
        rows = parse.parse_history_sections(SPREAD_ANSWER, DAY)["SPREADS"]
        kept = parse.drop_stale_repeats(rows)
        assert [(r.line_ts.hour, r.left) for r in kept] == [
            (15, "4.5-105"),
            (12, "4.5-105"),
            (2, "5.5-122"),
        ]

    def test_in_play_quote_never_stored(self):
        spread = _market(_ticks(), "point_spread")
        assert all(t.right_line != -1.5 for t in spread)
        assert spread[-1].right_line == -4.5


class TestTicks:
    def test_three_markets(self):
        ticks = _ticks()
        assert {t.market for t in ticks} == {"totals", "point_spread", "money_line"}

    def test_sides_and_minutes(self):
        last = _market(_ticks(), "totals")[-1]
        assert (last.left_line, last.left_price) == (231.5, -110)  # over
        assert (last.right_line, last.right_price) == (231.5, -110)  # under
        assert last.line_ts == datetime(2026, 10, 20, 15, 35, tzinfo=UTC)
        assert last.minutes_to_tip == -445
        assert last.is_pregame

    def test_spread_is_away_home(self):
        first = _market(_ticks(), "point_spread")[0]
        assert (first.left_line, first.right_line) == (5.5, -5.5)

    def test_one_opener_per_market_and_it_is_the_earliest(self):
        for market in ("totals", "point_spread", "money_line"):
            ticks = _market(_ticks(), market)
            assert [t.is_opener for t in ticks] == [True] + [False] * (len(ticks) - 1)
            assert ticks == sorted(ticks, key=lambda t: t.line_ts)

    def test_empty_quote_rows_dropped(self):
        totals = _market(_ticks(), "totals")
        assert all(t.left_line is not None for t in totals)
        assert len(totals) == 3  # 12/30 (2025), 07:16, 15:35

    def test_latest_quote_in_a_minute_wins(self):
        answer = (
            HEADER + "{MONEYLINES{{"
            '{10/20,07:42:50pm,+150,-180," "}'
            '{10/20,07:42:10pm,+140,-170," "}'
            "}}}"
        )
        ticks = _ticks([answer])
        assert len(ticks) == 1
        assert (ticks[0].left_price, ticks[0].right_price) == (150, -180)

    def test_latest_quote_wins_when_the_listing_is_out_of_order(self):
        # Real BetRivers answer (2025-11-08): the 22:41:58 row is listed above
        # the 22:41:59 one. The later quote (-109) is the one that holds.
        answer = (
            HEADER + "{TOTALS{{"
            '{11/08,10:41:58pm,224&frac12;o-107,224&frac12;u-113," "}'
            '{11/08,10:41:59pm,224&frac12;o-111,224&frac12;u-109," "}'
            '{11/08,09:00:00pm,223&frac12;o-110,223&frac12;u-110," "}'
            "}}}"
        )
        ticks = parse.book_ticks(
            [answer], book_id=60, game_date=date(2025, 11, 8), tipoff_utc=TIPOFF
        )
        assert [(t.line_ts.minute, t.right_price) for t in ticks] == [
            (0, -110),
            (41, -109),
        ]
        assert ticks[0].is_opener

    def test_totals_in_both_answers_not_doubled(self):
        both = _market(_ticks(), "totals")
        alone = _market(_ticks([SPREAD_ANSWER]), "totals")
        assert both == alone

    def test_book_slug(self):
        assert {t.book_slug for t in _ticks()} == {"fanduel"}


class TestScrapedGame:
    def test_shape_for_the_sbr_ingest(self):
        game = parse.scraped_game(
            _game(), answers={FANDUEL: [SPREAD_ANSWER, ML_ANSWER]}, tipoff_utc=TIPOFF
        )
        assert game.status_text.lower().startswith("final")
        assert game.tipoff_utc == TIPOFF
        assert game.season_year == 2026
        assert (game.team_away, game.team_home) == ("Philadelphia", "New York")
        assert game.books == ("fanduel",)

    def test_settled(self):
        assert not parse.is_settled(TIPOFF, datetime(2026, 10, 21, 2, 59, tzinfo=UTC))
        assert parse.is_settled(TIPOFF, datetime(2026, 10, 21, 3, 0, tzinfo=UTC))


class TestBooks:
    def test_shared_books_use_sbr_slugs(self):
        slugs = {slug for slug, _ in OL_BOOKS.values()}
        assert {
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "fanatics_sportsbook",
            "hard_rock_bet",
            "betrivers",
            "consensus_opener",
        } <= slugs
        assert "bet365" not in slugs

    def test_ids_fit_smallint_and_are_unique(self):
        ids = schema_mod.book_ids()
        assert len(ids) == len(OL_BOOKS)
        assert all(0 < book_id < 32768 for book_id in ids.values())


class FakeClient(client_mod.OddsLogicClient):
    """Answers from a queue per URL; no network, no sleeping. A URL in
    ``failing`` raises as a request that failed after its retries."""

    def __init__(self, answers, failing=()):
        super().__init__(sleep=(0, 0), empty_pause=0)
        self.answers = answers
        self.failing = set(failing)
        self.urls: list[str] = []

    def get_text(self, url):
        self.urls.append(url)
        if url in self.failing:
            raise client_mod.OddsLogicError("GET failed after 3 attempts: 502")
        queue = self.answers.get(url) or [client_mod.NO_HISTORY]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class TestClient:
    def test_empty_answer_is_retried(self):
        url = client_mod.history_url(DAY, rotation=503, book_id=112, request_type="ML")
        fake = FakeClient({url: [client_mod.NO_HISTORY, ML_ANSWER]})
        assert fake.history(url) == ML_ANSWER
        assert fake.empty_answers == 1

    def test_still_empty_is_none(self):
        url = client_mod.history_url(DAY, rotation=503, book_id=112, request_type="ML")
        fake = FakeClient({})
        assert fake.history(url) is None
        assert len(fake.urls) == 1 + client_mod.DEFAULT_EMPTY_RETRIES

    def test_header_without_sections_is_empty(self):
        assert client_mod.is_empty_history(HEADER + "}")
        assert not client_mod.is_empty_history(ML_ANSWER)

    def test_one_book_per_request(self):
        url = client_mod.history_url(
            DAY, rotation=503, book_id=112, request_type="SPREAD"
        )
        assert "s=112" in url and "t=SPREAD" in url and "z=0000" in url
        with pytest.raises(ValueError):
            client_mod.history_url(DAY, rotation=503, book_id=112, request_type="TT")

    def test_second_pass_fills_an_empty_answer(self):
        game = _game()
        url = update_mod._url(game, FANDUEL, "ML")
        history = {FANDUEL: {"SPREAD": SPREAD_ANSWER, "ML": None}}
        fake = FakeClient({url: [ML_ANSWER]})
        update_mod.refetch_empty(fake, game, history, [])
        assert history[FANDUEL]["ML"] == ML_ANSWER
        assert fake.urls == [url]

    def test_a_failing_book_keeps_the_others(self):
        game = _game()
        draftkings = 109
        answers = {
            update_mod._url(game, FANDUEL, "SPREAD"): [SPREAD_ANSWER],
            update_mod._url(game, FANDUEL, "ML"): [ML_ANSWER],
        }
        failing = {
            update_mod._url(game, draftkings, "SPREAD"),
            update_mod._url(game, draftkings, "ML"),
        }
        failures: list[str] = []
        history = update_mod.fetch_game(
            FakeClient(answers, failing), game, [draftkings, FANDUEL], failures
        )
        assert history[FANDUEL] == {"SPREAD": SPREAD_ANSWER, "ML": ML_ANSWER}
        assert history[draftkings] == {"SPREAD": None, "ML": None}
        assert len(failures) == 2 and all("draftkings" in f for f in failures)

    def test_refetch_failure_is_recorded_not_raised(self):
        game = _game()
        url = update_mod._url(game, FANDUEL, "ML")
        history = {FANDUEL: {"SPREAD": SPREAD_ANSWER, "ML": None}}
        failures: list[str] = []
        update_mod.refetch_empty(FakeClient({}, {url}), game, history, failures)
        assert history[FANDUEL] == {"SPREAD": SPREAD_ANSWER, "ML": None}
        assert len(failures) == 1


class TestPlan:
    GAMES = pd.DataFrame(
        {
            "game_id": ["0022600001", "0022600002", "0022600090", "0022600120"],
            "game_date": [
                date(2026, 10, 20),
                date(2026, 10, 20),
                date(2026, 10, 21),
                date(2026, 10, 30),
            ],
        }
    )

    def test_new_games_get_every_book_old_complete_ones_nothing(self):
        stored = {"0022600001": set(ACTIVE_BOOKS)}
        plan = update_mod.plan_requests(
            self.GAMES, stored, today=date(2026, 10, 30), retry_days=3
        )
        assert plan == {
            date(2026, 10, 20): {"0022600002": set(ACTIVE_BOOKS)},
            date(2026, 10, 21): {"0022600090": set(ACTIVE_BOOKS)},
        }

    def test_recent_games_retry_only_missing_books(self):
        stored = {"0022600090": set(ACTIVE_BOOKS) - {112}}
        plan = update_mod.plan_requests(
            self.GAMES.iloc[2:3], stored, today=date(2026, 10, 23), retry_days=3
        )
        assert plan == {date(2026, 10, 21): {"0022600090": {112}}}

    def test_old_incomplete_games_are_left_alone(self):
        stored = {"0022600090": set(ACTIVE_BOOKS) - {112}}
        plan = update_mod.plan_requests(
            self.GAMES.iloc[2:3], stored, today=date(2026, 10, 28), retry_days=3
        )
        assert plan == {}

    def test_inactive_books_not_requested_by_default(self):
        plan = update_mod.plan_requests(
            self.GAMES.iloc[:1], {}, today=date(2026, 10, 30)
        )
        assert 60 not in plan[date(2026, 10, 20)]["0022600001"]
        named = update_mod.plan_requests(
            self.GAMES.iloc[:1], {}, today=date(2026, 10, 30), books=[60]
        )
        assert named[date(2026, 10, 20)]["0022600001"] == {60}

    def test_today_and_before_first_date_skipped(self):
        early = pd.DataFrame(
            {"game_id": ["0022500900"], "game_date": [date(2026, 4, 10)]}
        )
        assert update_mod.plan_requests(early, {}, today=date(2026, 10, 30)) == {}
        plan = update_mod.plan_requests(self.GAMES, {}, today=date(2026, 10, 30))
        assert date(2026, 10, 30) not in plan

    def test_season_games_one_row_per_game(self):
        team_games = pd.DataFrame(
            {
                "game_id": ["0022600001", "0022600001", "0022500001"],
                "game_date": ["2026-10-20", "2026-10-20", "2025-10-21"],
                "season_year": [2026, 2026, 2025],
            }
        )
        games = update_mod.season_games(team_games, 2026)
        assert games.to_dict("records") == [
            {"game_id": "0022600001", "game_date": date(2026, 10, 20)}
        ]


class _Cursor:
    def __init__(self, log, rows=()):
        self.log = log
        self.rows = list(rows)
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.log.append(query.as_string(None) if hasattr(query, "as_string") else query)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _Conn:
    def __init__(self, rows=()):
        self.log: list[str] = []
        self.rows = rows
        self.commits = 0

    def cursor(self):
        return _Cursor(self.log, self.rows)

    def commit(self):
        self.commits += 1


class TestSchemaWrites:
    def test_everything_lands_in_the_oddslogic_schema(self):
        conn = _Conn()
        schema_mod.create_schema(conn)
        schema_mod.create_season_partition(conn, 2026)
        statements = " ".join(conn.log)
        assert '"line_history_oddslogic".lh_line' in statements
        assert '"line_history_oddslogic"."lh_line_2026"' in statements
        assert "line_history." not in statements.replace('"', "")
        assert sum("INSERT INTO" in s and "lh_book" in s for s in conn.log) == len(
            OL_BOOKS
        )

    def test_same_tables_as_the_sbr_store(self):
        conn = _Conn()
        schema_mod.create_schema(conn)
        created = {
            word
            for statement in conn.log
            for word in statement.replace("(", " ").split()
            if word.startswith('"line_history_oddslogic".lh_')
        }
        assert created == {
            '"line_history_oddslogic".lh_book',
            '"line_history_oddslogic".lh_market',
            '"line_history_oddslogic".lh_game',
            '"line_history_oddslogic".lh_line',
            '"line_history_oddslogic".lh_load_meta',
        }

    def test_stored_books_counts_only_complete_books(self):
        conn = _Conn(rows=[("0022600001", 109), ("0022600002", None)])
        stored = store.stored_game_books(conn, 2026)
        assert stored == {"0022600001": {109}, "0022600002": set()}
        assert "COUNT(DISTINCT market_id)" in conn.log[0]
