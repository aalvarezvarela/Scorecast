"""Covers line history: matchups and line-history HTML -> the SBR store's rows.

Pins the parsing rules the store relies on -- visible spans only, the side
conventions, the year and time zone of the Eastern date labels, the per-minute
collapse and the skipped OPEN table -- plus the rebuilt ``consensus_opener``,
the request plan, the update loop's skips and that writes stay in the
``line_history_covers`` schema; then the pick counts (``lh_consensus``): which
games ask for them, that a failed one stops neither the lines nor the other
games, and that a dry run writes nothing. The markup is the site's, with the quotes of
MEM @ ORL on 2026-01-15 (BetMGM, bet365).
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from nba_ou.fetch_data.odds_covers import client as client_mod
from nba_ou.fetch_data.odds_covers import parse
from nba_ou.fetch_data.odds_covers.books import (
    ACTIVE_BOOKS,
    CONSENSUS_OPENER_ID,
    COVERS_BOOKS,
)
from nba_ou.postgre_db.line_history_covers import schema as schema_mod
from nba_ou.postgre_db.line_history_covers import store
from nba_ou.postgre_db.line_history_covers import update as update_mod

DAY = date(2026, 1, 15)
TIPOFF = datetime(2026, 1, 16, 0, 0, tzinfo=UTC)  # 19:00 ET
FETCHED = datetime(2026, 1, 16, 15, 0, tzinfo=UTC)
COVERS_ID = 362626
BETMGM, BET365 = 26, 1

MATCHUPS = (
    '<article id="nba-362626" class="gamebox postgamebox nba" '
    'data-url="/sports/game/362626" tabindex="0" '
    "data-away-team-shortname=mem data-away-team-fullname=memphis grizzlies "
    "data-away-team-name=memphis data-home-team-shortname=orl "
    "data-home-team-fullname=orlando magic data-home-team-name=orlando "
    "data-homeconference=eastern-conference>"
    '<a href="/sport/basketball/nba/matchup/362626">Boxscore</a></article>'
    '<article id="nba-362627" class="gamebox nba" '
    "data-away-team-fullname=philadelphia 76ers data-away-team-name=philadelphia "
    'data-home-team-fullname="la clippers" data-home-team-name=la>'
    "</article>"
    # The same game twice (a second widget on the page) and a card without teams.
    '<article id="nba-362626" class="gamebox nba" '
    "data-away-team-fullname=memphis grizzlies data-home-team-fullname=orlando magic>"
    '<article id="nba-1" class="gamebox nba" data-url="/sports/game/1">'
)


def _span(text: str, price: str | None, visible: bool) -> str:
    style = "inline-flex" if visible else "none"
    inner = f'<span class="text-d-grey u-ml-4px">{price}</span>' if price else ""
    return f'<span class="u-flex-row u-mb-4px" style="display: {style};">{text}{inner}</span>'


def _quote_cell(left: tuple, right: tuple) -> str:
    """Each side: (text, price) -- price None for moneyline. Hidden decimal and
    fractional copies follow each visible span, as on the site."""
    spans = []
    for text, price in (left, right):
        spans.append(_span(text, price, True))
        spans.append(_span(text if price else "9.99", "1.91" if price else None, False))
        spans.append(_span(text if price else "9/5", "10/11" if price else None, False))
    return '<td class="u-fs-14 u-flex">' + " ".join(spans) + " </td>"


def _row(label: str, clock: str, left: tuple, right: tuple) -> str:
    when = (
        f'<td class="u-flex"> <span class="u-fs-11 u-mb-4px"> {label}</span> '
        f'<span class="text-left u-fs-12"> {clock}</span> </td>'
    )
    teams = (
        '<td class="u-fs-14"><span><span class="u-md-contents">Memphis Grizzlies</span>'
        '<span class="u-sm-contents">MEM</span></span><span>'
        '<span class="u-md-contents">Orlando Magic</span>'
        '<span class="u-sm-contents">ORL</span></span></td>'
    )
    return (
        '<tr class="u-flex">'
        + when
        + teams
        + '<td class="text-center"> </td>'
        + _quote_cell(left, right)
        + "</tr>"
    )


HEADER = (
    '<tr class="u-flex"><th scope="col">Update (ET)</th><th scope="col">Game</th>'
    '<th scope="col"></th><th scope="col">Line</th></tr>'
)


def _tables(book_id: int, rows: list[str], open_row: str) -> str:
    """A book's OPEN table and its history table, as each tab lists them."""
    attrs = f'data-book-id="{book_id}" class="sportsbookLineHistory book-{book_id}"'
    return (
        f"<table {attrs}><tbody>{open_row}</tbody></table>"
        f"<table {attrs}><tbody>{HEADER}{''.join(rows)}</tbody></table>"
    )


def _open(clock: str, left: tuple, right: tuple) -> str:
    return _row("OPEN", clock, left, right)


def _answer(tabs: dict[str, str]) -> str:
    body = ""
    for tab in ("moneyline", "spread", "total"):
        body += (
            f'<div role="tabpanel" class="tab-pane" id="lh-tab-{tab}">'
            f"{tabs.get(tab, '')}</div>"
        )
    return f'<div id="matchLineHistory-content-tabs" class="tab-content">{body}</div>'


# Newest first, as the site lists them.
MGM_TOTALS = [
    _row("Jan 15", "19:20", ("o244.5", "-110"), ("u244.5", "-110")),  # in-play
    _row("Jan 15", "18:50", ("o227", "-110"), ("u227", "-110")),  # 10 min to tip
    _row("Jan 15", "13:57", ("o227.5", "-110"), ("u227.5", "-110")),
    # Two quotes in one minute: the first listed is the newer and wins.
    _row("Jan 15", "11:02", ("o228.5", "-112"), ("u228.5", "-108")),
    _row("Jan 15", "11:02", ("o228", "-110"), ("u228", "-110")),
    _row("Jan 14", "17:38", ("o231.5", "-105"), ("u231.5", "-115")),
]
MGM_SPREAD = [
    _row("Jan 15", "13:57", ("&#x2B;5.5", "-110"), ("-5.5", "-110")),
    _row("Jan 14", "17:38", ("&#x2B;6.5", "-105"), ("-6.5", "-115")),
]
MGM_ML = [
    _row("Jan 15", "14:00", ("&#x2B;180", None), ("-220", None)),
    _row("Jan 14", "17:38", ("&#x2B;200", None), ("-245", None)),
]
B365_TOTALS = [
    # Sides listed UNDER first: the o/u letters decide, not the order.
    _row("Jan 15", "12:00", ("u227", "-105"), ("o227", "-115")),
    _row("Jan 14", "18:10", ("o230.5", "-110"), ("u230.5", "-110")),
]

ANSWER = _answer(
    {
        "total": _tables(
            BETMGM, MGM_TOTALS, _open("17:38", ("o231.5", "-105"), ("u231.5", "-115"))
        )
        + _tables(
            BET365, B365_TOTALS, _open("18:10", ("o230.5", "-110"), ("u230.5", "-110"))
        ),
        "spread": _tables(
            BETMGM, MGM_SPREAD, _open("17:38", ("&#x2B;6.5", "-105"), ("-6.5", "-115"))
        ),
        "moneyline": _tables(
            BETMGM, MGM_ML, _open("17:38", ("&#x2B;200", None), ("-245", None))
        ),
    }
)

GAME = parse.CoversGame(
    game_date=DAY,
    covers_id=COVERS_ID,
    team_away="Memphis Grizzlies",
    team_home="Orlando Magic",
)


def _scraped(text: str = ANSWER) -> parse.ParsedGame:
    return parse.scraped_game(GAME, text=text, tipoff_utc=TIPOFF, fetched_at=FETCHED)


def _ticks(market: str, book: str, text: str = ANSWER):
    return [
        t
        for t in _scraped(text).game.ticks
        if t.market == market and t.book_slug == book
    ]


class TestMatchups:
    def test_games_with_full_team_names(self):
        games = parse.parse_matchups(MATCHUPS, DAY)
        assert [(g.covers_id, g.team_away, g.team_home) for g in games] == [
            (362626, "Memphis Grizzlies", "Orlando Magic"),
            # capwords, not title(): "76ers" stays as the name standardisation knows it.
            (362627, "Philadelphia 76ers", "La Clippers"),
        ]
        assert all(g.game_date == DAY for g in games)

    def test_names_resolve_against_the_sbr_standardisation(self):
        from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
            _normalize_sbr_team_name,
        )

        for game in parse.parse_matchups(MATCHUPS, DAY):
            _normalize_sbr_team_name(game.team_away)
            _normalize_sbr_team_name(game.team_home)


class TestQuotes:
    def test_line_and_price(self):
        span = '&#x2B;5.5<span class="text-d-grey u-ml-4px">-110</span>'
        assert parse.parse_quote(span) == (5.5, -110, None)

    def test_totals_side_letter(self):
        span = 'u227.5<span class="text-d-grey">-108</span>'
        assert parse.parse_quote(span) == (227.5, -108, "u")

    def test_moneyline_price_only(self):
        assert parse.parse_quote("&#x2B;180") == (None, 180, None)
        assert parse.parse_quote("-220") == (None, -220, None)

    def test_pick_and_even(self):
        assert parse.parse_quote('PK<span class="x">-110</span>') == (0.0, -110, None)
        assert parse.parse_quote('&#x2B;1.5<span class="x">EV</span>') == (
            1.5,
            100,
            None,
        )

    def test_only_visible_spans_count(self):
        ticks = _ticks("totals", "betmgm")
        # The hidden copies carry decimal odds (1.91); none may leak in.
        assert all(t.left_price in (-110, -112, -105) for t in ticks)


class TestTimestamps:
    def test_eastern_to_utc(self):
        stamp = parse._row_timestamp("Jan 15", "13:57", DAY, FETCHED)
        assert stamp == datetime(2026, 1, 15, 18, 57, tzinfo=UTC)

    def test_daylight_saving_time(self):
        october = parse._row_timestamp("Oct 23", "22:47", date(2026, 10, 24), FETCHED)
        assert october == datetime(2026, 10, 24, 2, 47, tzinfo=UTC)

    def test_december_row_of_a_january_game_is_last_year(self):
        stamp = parse._row_timestamp("Dec 30", "10:00", date(2026, 1, 2), FETCHED)
        assert stamp.year == 2025

    def test_february_29(self):
        stamp = parse._row_timestamp("Feb 29", "12:00", date(2028, 3, 1), FETCHED)
        assert stamp == datetime(2028, 2, 29, 17, 0, tzinfo=UTC)
        # In a game of a non-leap year it can only be an earlier leap year's.
        assert (
            parse._row_timestamp("Feb 29", "12:00", date(2025, 3, 1), FETCHED).year
            == 2024
        )

    def test_summer_opener_of_an_october_game(self):
        stamp = parse._row_timestamp("Aug 21", "15:58", date(2026, 10, 20), FETCHED)
        assert stamp == datetime(2026, 8, 21, 19, 58, tzinfo=UTC)

    def test_relative_labels_follow_the_eastern_fetch_date(self):
        # 02:00 UTC on Jan 16 is still Jan 15 in New York.
        late = datetime(2026, 1, 16, 2, 0, tzinfo=UTC)
        assert parse._row_timestamp("Today", "13:00", DAY, late) == datetime(
            2026, 1, 15, 18, 0, tzinfo=UTC
        )
        assert parse._row_timestamp("Yesterday", "13:00", DAY, late) == datetime(
            2026, 1, 14, 18, 0, tzinfo=UTC
        )


class TestTicks:
    def test_open_table_is_skipped_and_opener_is_the_oldest_row(self):
        ticks = _ticks("totals", "betmgm")
        assert [t.is_opener for t in ticks] == [True, False, False, False, False]
        assert ticks[0].line_ts == datetime(2026, 1, 14, 22, 38, tzinfo=UTC)
        assert (ticks[0].left_line, ticks[0].left_price) == (231.5, -105)

    def test_newest_quote_of_a_minute_wins(self):
        minute = datetime(2026, 1, 15, 16, 2, tzinfo=UTC)
        (tick,) = [t for t in _ticks("totals", "betmgm") if t.line_ts == minute]
        assert (tick.left_line, tick.left_price, tick.right_price) == (
            228.5,
            -112,
            -108,
        )

    def test_in_play_rows_are_kept_but_not_pregame(self):
        last = _ticks("totals", "betmgm")[-1]
        assert last.left_line == 244.5
        assert last.minutes_to_tip == 20 and not last.is_pregame
        assert _ticks("totals", "betmgm")[-2].minutes_to_tip == -10

    def test_spread_is_away_left_home_right(self):
        latest = _ticks("point_spread", "betmgm")[-1]
        assert (latest.left_line, latest.right_line) == (5.5, -5.5)

    def test_moneyline_has_no_lines(self):
        latest = _ticks("money_line", "betmgm")[-1]
        assert (latest.left_line, latest.left_price) == (None, 180)
        assert (latest.right_line, latest.right_price) == (None, -220)

    def test_totals_sides_follow_the_letters(self):
        latest = _ticks("totals", "bet365")[-1]
        assert (latest.left_line, latest.left_price) == (227.0, -115)  # OVER
        assert latest.right_price == -105  # UNDER


class TestConsensusOpener:
    def test_copies_betmgm_openers_in_every_market(self):
        consensus = {t.market: t for t in _ticks_all("consensus_opener")}
        openers = {t.market: t for t in _ticks_all("betmgm") if t.is_opener}
        assert set(consensus) == {"totals", "point_spread", "money_line"}
        for market, tick in consensus.items():
            mgm = openers[market]
            assert tick.is_opener and tick.line_ts == mgm.line_ts
            assert (
                tick.left_line,
                tick.left_price,
                tick.right_line,
                tick.right_price,
            ) == (
                mgm.left_line,
                mgm.left_price,
                mgm.right_line,
                mgm.right_price,
            )

    def test_raw_quote_not_centred(self):
        # SBR stores the opener as quoted; the pipeline centres it to -110 later.
        (totals,) = [t for t in _ticks_all("consensus_opener") if t.market == "totals"]
        assert (totals.left_line, totals.left_price, totals.right_price) == (
            231.5,
            -105,
            -115,
        )

    def test_nothing_without_betmgm(self):
        answer = _answer(
            {
                "total": _tables(
                    BET365,
                    B365_TOTALS,
                    _open("18:10", ("o230.5", "-110"), ("u230.5", "-110")),
                )
            }
        )
        assert not [
            t for t in _scraped(answer).game.ticks if t.book_slug == "consensus_opener"
        ]


def _ticks_all(book: str):
    return [t for t in _scraped().game.ticks if t.book_slug == book]


class TestScrapedGame:
    def test_game_fields(self):
        game = _scraped().game
        assert game.event_id == COVERS_ID and game.status_text == "Final"
        assert game.season_year == 2025 and game.tipoff_utc == TIPOFF
        assert game.books == ("bet365", "betmgm", "consensus_opener")

    def test_unknown_books_are_reported_not_stored(self):
        answer = _answer(
            {
                "total": _tables(
                    999,
                    B365_TOTALS,
                    _open("18:10", ("o230.5", "-110"), ("u230.5", "-110")),
                )
            }
        )
        parsed = _scraped(answer)
        assert parsed.unknown_books == {999}
        assert not parsed.game.ticks

    def test_settled_four_hours_after_tip(self):
        assert not parse.is_settled(TIPOFF, TIPOFF + pd.Timedelta(hours=3))
        assert parse.is_settled(TIPOFF, TIPOFF + pd.Timedelta(hours=4))


class TestBooks:
    def test_shared_books_use_sbr_slugs(self):
        slugs = {slug for slug, _ in COVERS_BOOKS.values()}
        assert {
            "bet365",
            "betmgm",
            "draftkings",
            "fanduel",
            "fanatics_sportsbook",
            "betrivers",
            "caesars",
            "hard_rock_bet",
            "consensus_opener",
        } <= slugs

    def test_ids_fit_smallint_and_are_unique(self):
        ids = schema_mod.book_ids()
        assert len(ids) == len(COVERS_BOOKS)
        assert all(0 < book_id < 32768 for book_id in ids.values())
        assert CONSENSUS_OPENER_ID in ids.values()


class FakeClient(client_mod.CoversClient):
    """Answers by URL; no network, no sleeping. A URL in ``failing`` raises as a
    request that failed after its retries."""

    def __init__(self, answers, failing=()):
        super().__init__(sleep=(0, 0))
        self.answers = answers
        self.failing = set(failing)
        self.urls: list[str] = []

    def get_text(self, url):
        self.urls.append(url)
        if url in self.failing:
            raise client_mod.CoversError("GET failed after 3 attempts: 502")
        return self.answers.get(url, "<html>error</html>")


class TestClient:
    def test_history_url_names_the_us_region(self):
        url = client_mod.history_url(COVERS_ID)
        assert url.startswith(
            f"{client_mod.BASE_URL}/sport/MatchupOdds/LineHistoryBrick?"
        )
        assert "gameId=362626" in url and "countryCode=us" in url
        assert "stateProvince=nj" in url and "oddsFormat=american" in url

    def test_answer_without_tabs_is_a_failure(self):
        fake = FakeClient({})
        with pytest.raises(client_mod.CoversError):
            fake.line_history(COVERS_ID)

    def test_answer_with_tabs_is_returned(self):
        fake = FakeClient({client_mod.history_url(COVERS_ID): ANSWER})
        assert fake.line_history(COVERS_ID) == ANSWER


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

    def test_new_games_planned_old_complete_ones_not(self):
        stored = {"0022600001": set(ACTIVE_BOOKS)}
        plan = update_mod.plan_requests(
            self.GAMES, stored, today=date(2026, 10, 30), retry_days=3
        )
        assert plan == {
            date(2026, 10, 20): {"0022600002": set(ACTIVE_BOOKS)},
            date(2026, 10, 21): {"0022600090": set(ACTIVE_BOOKS)},
        }

    def test_recent_games_missing_a_book_are_planned_again(self):
        stored = {"0022600090": set(ACTIVE_BOOKS) - {44}}
        plan = update_mod.plan_requests(
            self.GAMES.iloc[2:3], stored, today=date(2026, 10, 23), retry_days=3
        )
        assert plan == {date(2026, 10, 21): {"0022600090": {44}}}

    def test_old_incomplete_games_are_left_alone(self):
        stored = {"0022600090": set(ACTIVE_BOOKS) - {44}}
        plan = update_mod.plan_requests(
            self.GAMES.iloc[2:3], stored, today=date(2026, 10, 28), retry_days=3
        )
        assert plan == {}

    def test_today_and_before_first_date_skipped(self):
        early = pd.DataFrame(
            {"game_id": ["0022500900"], "game_date": [date(2026, 4, 10)]}
        )
        assert update_mod.plan_requests(early, {}, today=date(2026, 10, 30)) == {}
        plan = update_mod.plan_requests(self.GAMES, {}, today=date(2026, 10, 30))
        assert date(2026, 10, 30) not in plan

    def test_store_starts_with_the_2026_27_season(self):
        assert update_mod.FIRST_DATE == date(2026, 10, 20)
        assert update_mod.FIRST_SEASON == 2026


class TestBackfill:
    GAMES = pd.DataFrame(
        {
            "game_id": ["0041800111", "0022301062", "0022301063"],
            "game_date": [date(2019, 4, 13), date(2024, 3, 29), date(2024, 3, 29)],
        }
    )

    def test_past_games_planned_before_first_date(self):
        plan = update_mod.plan_backfill(self.GAMES, {}, today=date(2026, 10, 24))
        assert plan == {
            date(2019, 4, 13): {"0041800111": set(ACTIVE_BOOKS)},
            date(2024, 3, 29): {
                "0022301062": set(ACTIVE_BOOKS),
                "0022301063": set(ACTIVE_BOOKS),
            },
        }

    def test_stored_games_are_not_fetched_again(self):
        # Even with a book missing: a backfilled game is as Covers had it.
        stored = {"0022301062": {1}, "0041800111": set(ACTIVE_BOOKS)}
        plan = update_mod.plan_backfill(self.GAMES, stored, today=date(2026, 10, 24))
        assert plan == {date(2024, 3, 29): {"0022301063": set(ACTIVE_BOOKS)}}

    def test_today_is_not_planned(self):
        plan = update_mod.plan_backfill(self.GAMES, {}, today=date(2024, 3, 29))
        assert list(plan) == [date(2019, 4, 13)]

    def test_game_ids_filtered_by_reason(self):
        flagged = pd.DataFrame(
            {
                # Read from CSV without dtype: leading zeros lost.
                "game_id": [41800111, "0022301062", "0022100316", "0022000100"],
                "reasons": ["no_game", "no_ticks", "no_main_book;late_start", None],
            }
        )
        assert update_mod.backfill_game_ids(flagged) == {
            "0041800111",
            "0022301062",
            "0022100316",
        }
        assert update_mod.backfill_game_ids(flagged, ["late_start"]) == {"0022100316"}
        assert update_mod.backfill_game_ids(flagged, ["truncated"]) == set()


def _nba_games(game_id: str = "0022500578") -> pd.DataFrame:
    """Team-game rows as ``load_games_for_line_history_creation`` returns them."""
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "game_date": [DAY.isoformat()] * 2,
            "season_year": [2025, 2025],
            "team_name": ["Orlando Magic", "Memphis Grizzlies"],
            "home": [True, False],
        }
    )


def _schedule(game_id: str = "0022500578") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id],
            "tipoff_utc": [pd.Timestamp(TIPOFF)],
            "team_home": ["ORL"],
            "team_away": ["MEM"],
        }
    )


class _Archive:
    def __init__(self, fail=False):
        self.keys: list[str] = []
        self.fail = fail

    def put(self, key, text):
        if self.fail:
            raise OSError("archive unavailable")
        self.keys.append(key)


def _run(client, *, schedule=None, now=FETCHED, archive=None, dry_run=True, conn=None):
    return update_mod.update_covers_line_history(
        {DAY: {"0022500578": set(ACTIVE_BOOKS)}},
        games_df=_nba_games(),
        schedule=_schedule() if schedule is None else schedule,
        conn=conn,
        client=client,
        archive=archive,
        dry_run=dry_run,
        now=now,
        progress=lambda _: None,
    )


#: The Covers Consensus page: ATS (away, home) then O/U (over, under) "Total
#: Picks", then the experts' section, which is not read.
CONSENSUS_PAGE = """
<div id="consensus_analysis_content">
  <div class="covers-CoversConsensusDetailsTable-finalWagersleft">368</div>
  <div class="covers-CoversConsensusDetailsTable-picksFinal--center">Total Picks</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersRight">198</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersleft"> 167 </div>
  <div class="covers-CoversConsensusDetailsTable-picksFinal--center">Total Picks</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersRight"> 0 </div>
</div>
<div id="experts_analysis_content" style="display: none">
  <div class="covers-CoversConsensusDetailsTable-finalWagersleft">21</div>
  <div class="covers-CoversConsensusDetailsTable-finalWagersRight">7</div>
</div>
"""

#: The section is there, the counts are not (a game nobody picked yet).
CONSENSUS_EMPTY = '<div id="consensus_analysis_content"></div>'


def _client(answer: str = ANSWER):
    return FakeClient(
        {
            client_mod.matchups_url(DAY): MATCHUPS,
            client_mod.history_url(COVERS_ID): answer,
            client_mod.consensus_url(COVERS_ID): CONSENSUS_PAGE,
        }
    )


#: A history that stops seven hours before tip-off (bet365 only, last 12:00 ET).
TRUNCATED = _answer(
    {
        "total": _tables(
            BET365, B365_TOTALS, _open("18:10", ("o230.5", "-110"), ("u230.5", "-110"))
        )
    }
)

#: Tabs present, no book with a row.
EMPTY = _answer({})


class TestUpdate:
    def test_fetched_game_becomes_store_rows(self):
        result = _run(_client())
        assert result.fetched_games == 1 and not result.failed
        rows, games = result.frames[0]
        assert games.iloc[0].game_id == "0022500578"
        assert games.iloc[0].event_id == COVERS_ID
        ids = schema_mod.book_ids()
        assert set(rows["book_id"]) == {
            ids["betmgm"],
            ids["bet365"],
            ids["consensus_opener"],
        }
        # Lines are stored doubled, as in the SBR store.
        opener = rows[
            (rows["book_id"] == ids["consensus_opener"]) & (rows["market_id"] == 1)
        ]
        assert opener.iloc[0].left_line == 463
        assert rows["is_pregame"].sum() == len(rows) - 1  # the one in-play row

    def test_game_without_ticks_is_not_stored(self):
        conn = _Conn()
        result = _run(_client(EMPTY), dry_run=False, conn=conn)
        assert result.empty_games and not result.frames
        assert result.inserted_games == 0 and not conn.log

    def test_truncated_history_is_held_back_then_stored(self):
        held = _run(_client(TRUNCATED))
        assert len(held.held_back) == 1 and "420 min" in held.held_back[0]
        assert not held.frames
        late = _run(_client(TRUNCATED), now=TIPOFF + pd.Timedelta(days=4))
        assert not late.held_back
        assert late.frames[0][1].iloc[0].game_id == "0022500578"

    def test_game_and_lines_commit_together(self, monkeypatch):
        events: list[str] = []
        conn = _Conn()
        conn.commit = lambda: events.append("commit")

        def games(c, game_dim, *, commit=True):
            events.append(f"games commit={commit}")
            return len(game_dim)

        def lines(c, rows, season_year, *, commit=True):
            events.append(f"lines commit={commit}")
            return len(rows)

        monkeypatch.setattr(store, "insert_games", games)
        monkeypatch.setattr(store, "insert_lines", lines)
        result = _run(_client(), dry_run=False, conn=conn)
        # The partition DDL commits first, on its own.
        assert events[-3:] == ["games commit=False", "lines commit=False", "commit"]
        assert result.inserted_games == 1 and result.inserted_ticks > 0
        assert not conn.rollbacks

    def test_failed_line_write_rolls_the_game_back(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("COPY failed")

        monkeypatch.setattr(store, "insert_lines", broken)
        conn = _Conn()
        commits_before_write: list[int] = []
        original = store.insert_games

        def tracked(c, game_dim, *, commit=True):
            assert commit is False  # joins the caller's transaction
            commits_before_write.append(c.commits)
            return original(c, game_dim, commit=commit)

        monkeypatch.setattr(store, "insert_games", tracked)
        with pytest.raises(RuntimeError):
            _run(_client(), dry_run=False, conn=conn)
        assert conn.rollbacks == 1
        assert conn.commits == commits_before_write[0]  # nothing after the game

    def test_parse_error_is_recorded_not_raised(self, monkeypatch):
        def broken(*args, **kwargs):
            raise ValueError("unexpected markup")

        monkeypatch.setattr(update_mod, "scraped_game", broken)
        result = _run(_client())
        assert len(result.failed) == 1 and "parse error" in result.failed[0]

    def test_one_log_line_per_game_in_a_github_group(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        lines: list[str] = []
        update_mod.update_covers_line_history(
            {DAY: {"0022500578": set(ACTIVE_BOOKS)}},
            games_df=_nba_games(),
            schedule=_schedule(),
            conn=None,
            client=_client(),
            dry_run=True,
            now=FETCHED,
            progress=lines.append,
        )
        assert lines[0] == "::group::2026-01-15: 1 game(s) planned"
        assert lines[-1] == "::endgroup::"
        (game,) = [
            line for line in lines if "Memphis Grizzlies @ Orlando Magic" in line
        ]
        assert game.startswith("  + ") and "book(s)" in game and "tick(s)" in game

    def test_game_without_a_feed_tipoff_waits(self):
        client = _client()
        result = _run(
            client,
            schedule=pd.DataFrame(
                columns=["game_id", "tipoff_utc", "team_home", "team_away"]
            ),
        )
        assert result.feed_tipoff_missing and not result.frames
        assert client.urls == [client_mod.matchups_url(DAY)]

    def test_unsettled_game_waits(self):
        client = _client()
        result = _run(client, now=TIPOFF + pd.Timedelta(hours=1))
        assert result.unsettled_games and not result.frames
        assert client_mod.history_url(COVERS_ID) not in client.urls

    def test_game_missing_from_the_matchups_page(self):
        client = FakeClient({client_mod.matchups_url(DAY): "<html></html>"})
        result = _run(client)
        assert result.unlisted_games == [f"{DAY} 0022500578"]

    def test_failed_request_is_recorded_not_raised(self):
        client = FakeClient(
            {client_mod.matchups_url(DAY): MATCHUPS},
            failing={client_mod.history_url(COVERS_ID)},
        )
        result = _run(client)
        assert len(result.failed) == 1 and not result.frames

    def test_archive_failure_does_not_stop_the_store(self):
        result = _run(_client(), archive=_Archive(fail=True), dry_run=False)
        assert result.archive_failures == 2  # matchups and the game
        assert result.fetched_games == 1

    def test_raw_answers_archived_by_date_and_game(self):
        archive = _Archive()
        _run(_client(), archive=archive, dry_run=False)
        assert archive.keys[0].startswith(
            "train_data/odds_sources/covers/raw/v1/2026-01-15/matchups/"
        )
        assert archive.keys[1].startswith(
            "train_data/odds_sources/covers/raw/v1/2026-01-15/362626/"
        )

    def test_dry_run_archives_nothing(self):
        archive = _Archive()
        _run(_client(), archive=archive, dry_run=True)
        assert archive.keys == []


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
        self.rollbacks = 0

    def cursor(self):
        return _Cursor(self.log, self.rows)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class TestSchemaWrites:
    def test_everything_lands_in_the_covers_schema(self):
        conn = _Conn()
        schema_mod.create_schema(conn)
        schema_mod.create_season_partition(conn, 2026)
        statements = " ".join(conn.log)
        assert '"line_history_covers".lh_line' in statements
        assert '"line_history_covers"."lh_line_2026"' in statements
        assert "line_history." not in statements.replace('"', "")
        assert sum("INSERT INTO" in s and "lh_book" in s for s in conn.log) == len(
            COVERS_BOOKS
        )

    def test_same_tables_as_the_sbr_store(self):
        conn = _Conn()
        schema_mod.create_schema(conn)
        created = {
            word
            for statement in conn.log
            for word in statement.replace("(", " ").split()
            if word.startswith('"line_history_covers".lh_')
        }
        assert created == {
            '"line_history_covers".lh_book',
            '"line_history_covers".lh_market',
            '"line_history_covers".lh_game',
            '"line_history_covers".lh_line',
            '"line_history_covers".lh_load_meta',
            '"line_history_covers".lh_consensus',
        }

    def test_consensus_table_is_covers_only_and_keyed_by_game(self):
        conn = _Conn()
        schema_mod.create_schema(conn)
        (ddl,) = [s for s in conn.log if "lh_consensus" in s]
        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert "game_id           TEXT PRIMARY KEY" in ddl
        assert 'REFERENCES "line_history_covers".lh_game (game_id)' in ddl
        for column in store.CONSENSUS_COLUMNS[1:]:
            assert column in ddl
        assert "fetched_at        TIMESTAMPTZ" in ddl
        # Created after lh_game, which it references.
        game_ddl = next(i for i, s in enumerate(conn.log) if "lh_game (" in s)
        assert conn.log.index(ddl) > game_ddl

    def test_consensus_reads_and_writes_the_covers_schema(self):
        conn = _Conn(rows=[("0022600001",), ("0022600002",)])
        assert store.stored_consensus_game_ids(conn, 2026) == {
            "0022600001",
            "0022600002",
        }
        frame = pd.DataFrame(
            [("0022600003", 2026, 1, 2, 3, 0, FETCHED)],
            columns=store.CONSENSUS_COLUMNS,
        )
        assert store.insert_consensus(conn, frame) == 1
        assert '"line_history_covers".lh_consensus' in conn.log[0]
        assert "ON CONFLICT (game_id) DO NOTHING" in conn.log[1]
        assert conn.commits == 1

    def test_stored_books_read_the_covers_schema(self):
        conn = _Conn(rows=[("0022600001", 26), ("0022600002", None)])
        stored = store.stored_game_books(conn, 2026)
        assert stored == {"0022600001": {26}, "0022600002": set()}
        assert '"line_history_covers".lh_game' in conn.log[0]


def _load_cli():
    import importlib.util

    from tests.support.paths import REPO_ROOT

    path = REPO_ROOT / "scripts/update_databases/update_covers_line_history.py"
    spec = importlib.util.spec_from_file_location("covers_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class TestCli:
    def _main(
        self, monkeypatch, client: FakeClient, flags=(), stored_games=None
    ) -> int:
        from contextlib import contextmanager

        cli = _load_cli()
        if stored_games is None:
            stored_games = pd.DataFrame(columns=store.GAME_EVENT_COLUMNS)

        @contextmanager
        def connection():
            yield _Conn()

        class ContextClient:
            def __init__(self):
                self.inner = client
                self.requests_made = 0

            def __enter__(self):
                return self.inner

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(cli, "connect_line_history_db", connection)
        monkeypatch.setattr(cli, "load_games_for_line_history_creation", _nba_games)
        monkeypatch.setattr(cli, "fetch_schedules", lambda seasons: _schedule())
        monkeypatch.setattr(cli, "CoversClient", ContextClient)
        monkeypatch.setattr(cli.schema_mod, "schema_exists", lambda conn: True)
        monkeypatch.setattr(cli.store, "stored_game_books", lambda conn, season: {})
        monkeypatch.setattr(cli.schema_mod, "consensus_table_exists", lambda c: True)
        monkeypatch.setattr(
            cli.store, "stored_consensus_game_ids", lambda conn, season: set()
        )
        monkeypatch.setattr(
            cli.store, "fetch_game_events", lambda conn, season: stored_games
        )
        monkeypatch.setattr(
            "sys.argv",
            ["update_covers_line_history.py", "--dry-run", "--dates", str(DAY), *flags],
        )
        return cli.main()

    def test_failed_downloads_exit_1(self, monkeypatch):
        failing = FakeClient({}, failing={client_mod.matchups_url(DAY)})
        assert self._main(monkeypatch, failing) == 1

    def test_clean_run_exits_0(self, monkeypatch):
        client = _client()
        assert self._main(monkeypatch, client) == 0
        # The game the run prepared gets its pick counts too.
        assert client.urls[-1] == client_mod.consensus_url(COVERS_ID)

    def test_consensus_only_fetches_no_lines(self, monkeypatch):
        client = _client()
        code = self._main(
            monkeypatch, client, ["--consensus-only"], stored_games=_stored_games()
        )
        assert code == 0
        assert client.urls == [client_mod.consensus_url(COVERS_ID)]

    def test_failed_consensus_exits_1(self, monkeypatch):
        failing = FakeClient({}, failing={client_mod.consensus_url(COVERS_ID)})
        code = self._main(
            monkeypatch, failing, ["--consensus-only"], stored_games=_stored_games()
        )
        assert code == 1


# -- pick counts (lh_consensus) ------------------------------------------------

OTHER_ID, OTHER_COVERS = "0022500579", 362627


def _stored_games(*game_ids: str) -> pd.DataFrame:
    """``lh_game`` rows as ``store.fetch_game_events`` returns them."""
    game_ids = game_ids or ("0022500578",)
    covers = {"0022500578": COVERS_ID, OTHER_ID: OTHER_COVERS}
    return pd.DataFrame(
        {
            "game_id": list(game_ids),
            "game_date": [DAY] * len(game_ids),
            "season_year": [2025] * len(game_ids),
            "tipoff_utc": [pd.Timestamp(TIPOFF)] * len(game_ids),
            "event_id": [covers.get(g) for g in game_ids],
            "team_home": ["ORL"] * len(game_ids),
            "team_away": ["MEM"] * len(game_ids),
        }
    )


def _consensus_plan(*game_ids: str, stored=()) -> update_mod.ConsensusPlan:
    games = _stored_games(*game_ids)
    return update_mod.ConsensusPlan(
        targets=update_mod.plan_consensus(games, set(stored), now=FETCHED),
        stored=set(stored),
    )


def _run_consensus(client, consensus, *, plan=None, dry_run=True, conn=None, **kw):
    return update_mod.update_covers_line_history(
        plan or {},
        games_df=_nba_games(),
        schedule=_schedule(),
        conn=conn,
        client=client,
        dry_run=dry_run,
        now=FETCHED,
        progress=kw.pop("progress", lambda _: None),
        consensus=consensus,
        **kw,
    )


class TestConsensusPlan:
    def test_stored_games_without_counts_are_planned(self):
        games = _stored_games("0022500578", OTHER_ID)
        plan = update_mod.plan_consensus(games, {OTHER_ID}, now=FETCHED)
        assert plan == {
            DAY: [
                update_mod.ConsensusTarget(
                    "0022500578", COVERS_ID, 2025, "MEM @ ORL (0022500578)"
                )
            ]
        }

    def test_unsettled_games_wait(self):
        games = _stored_games()
        early = TIPOFF + pd.Timedelta(hours=3, minutes=59)
        assert update_mod.plan_consensus(games, set(), now=early) == {}
        late = TIPOFF + pd.Timedelta(hours=4)
        assert update_mod.plan_consensus(games, set(), now=late)

    def test_game_without_a_covers_id_is_skipped(self):
        games = _stored_games("0022500578", "0022500999")  # no Covers id
        plan = update_mod.plan_consensus(games, set(), now=FETCHED)
        assert [t.game_id for t in plan[DAY]] == ["0022500578"]

    def test_naive_tipoffs_read_as_utc(self):
        games = _stored_games()
        games["tipoff_utc"] = [pd.Timestamp("2026-01-16 00:00")]
        assert update_mod.plan_consensus(
            games, set(), now=TIPOFF + pd.Timedelta(hours=4)
        )


class TestConsensusUpdate:
    def test_counts_become_store_rows(self):
        client = _client()
        result = _run_consensus(client, _consensus_plan())
        assert client.urls == [client_mod.consensus_url(COVERS_ID)]
        (frame,) = result.consensus_frames
        row = frame.iloc[0]
        assert list(frame.columns) == store.CONSENSUS_COLUMNS
        assert row.game_id == "0022500578" and row.season_year == 2025
        assert (row.spread_away_picks, row.spread_home_picks) == (368, 198)
        # Zero is a count, not a missing value.
        assert (row.total_over_picks, row.total_under_picks) == (167, 0)
        assert result.consensus_planned == 1 and result.consensus_fetched == 1
        assert not result.failed and not result.frames

    def test_dry_run_writes_and_archives_nothing(self):
        conn, archive = _Conn(), _Archive()
        result = _run_consensus(
            _client(), _consensus_plan(), conn=conn, archive=archive, dry_run=True
        )
        assert result.consensus_frames and result.consensus_inserted == 0
        assert conn.log == [] and conn.commits == 0 and archive.keys == []

    def test_written_and_archived(self):
        conn, archive = _Conn(), _Archive()
        result = _run_consensus(
            _client(), _consensus_plan(), conn=conn, archive=archive, dry_run=False
        )
        assert result.consensus_inserted == 1 and conn.commits == 1
        assert any("lh_consensus" in s for s in conn.log)
        (key,) = archive.keys
        assert key.startswith(
            "train_data/odds_sources/covers/raw/v1/2026-01-15/362626/consensus/"
        )
        assert key.endswith(".html.gz")

    def test_game_with_counts_is_not_asked_again(self):
        client = _client()
        result = _run_consensus(
            client,
            _consensus_plan(stored={"0022500578"}),
            plan={DAY: {"0022500578": set(ACTIVE_BOOKS)}},
        )
        assert result.frames  # the lines were fetched again
        assert client_mod.consensus_url(COVERS_ID) not in client.urls

    def test_a_game_stored_by_the_run_gets_its_counts(self):
        client = _client()
        lines: list[str] = []
        result = _run_consensus(
            client,
            update_mod.ConsensusPlan(),
            plan={DAY: {"0022500578": set(ACTIVE_BOOKS)}},
            progress=lines.append,
        )
        assert result.frames and result.consensus_planned == 1
        # Lines first, then the counts, in the date's one group.
        assert client.urls[-2:] == [
            client_mod.history_url(COVERS_ID),
            client_mod.consensus_url(COVERS_ID),
        ]
        assert result.consensus_frames[0].iloc[0].game_id == "0022500578"
        (counts,) = [line for line in lines if "consensus ATS" in line]
        assert counts.startswith("  + MEM @ ORL (0022500578): consensus ATS 368 away")

    def test_failed_consensus_stops_neither_lines_nor_other_games(self, monkeypatch):
        monkeypatch.setattr(
            store, "insert_lines", lambda c, rows, season, *, commit=True: len(rows)
        )
        client = _client()
        client.failing.add(client_mod.consensus_url(OTHER_COVERS))
        conn = _Conn()
        result = _run_consensus(
            client,
            _consensus_plan(OTHER_ID),
            plan={DAY: {"0022500578": set(ACTIVE_BOOKS)}},
            conn=conn,
            dry_run=False,
        )
        assert result.inserted_games == 1 and result.inserted_ticks > 0
        (failure,) = result.failed
        assert OTHER_ID in failure and "consensus" in failure
        # The game the run stored still got its counts, after the failure.
        assert result.consensus_inserted == 1
        assert result.consensus_frames[0]["game_id"].tolist() == ["0022500578"]

    def test_unparseable_page_is_a_failure_not_an_exception(self, monkeypatch):
        def broken(text):
            raise ValueError("unexpected markup")

        monkeypatch.setattr(update_mod, "parse_consensus", broken)
        result = _run_consensus(_client(), _consensus_plan())
        (failure,) = result.failed
        assert "consensus parse error" in failure and not result.consensus_frames

    def test_page_without_counts_waits_for_the_next_run(self):
        client = FakeClient({client_mod.consensus_url(COVERS_ID): CONSENSUS_EMPTY})
        result = _run_consensus(client, _consensus_plan())
        assert result.consensus_empty and not result.failed
        assert not result.consensus_frames

    def test_consensus_only_dates_get_their_own_group(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        lines: list[str] = []
        _run_consensus(_client(), _consensus_plan(), progress=lines.append)
        assert lines[0] == "::group::2026-01-15: consensus of 1 stored game(s)"
        assert lines[-1] == "::endgroup::"
        assert any(line.startswith("  + MEM @ ORL") for line in lines)

    def test_without_a_consensus_plan_no_counts_are_asked(self):
        client = _client()
        result = _run(client)
        assert client_mod.consensus_url(COVERS_ID) not in client.urls
        assert result.consensus_planned == 0


# -- the store, against a real PostgreSQL --------------------------------------

DSN = os.getenv("NBA_OU_TEST_PG_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="NBA_OU_TEST_PG_DSN not set")


@pytest.fixture
def pg_conn():
    import psycopg

    def drop(connection):
        with connection.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema_mod.SCHEMA} CASCADE")
        connection.commit()

    with psycopg.connect(DSN) as connection:
        drop(connection)
        schema_mod.create_schema(connection)
        yield connection
        connection.rollback()
        drop(connection)


@pytest.mark.postgres
@needs_db
def test_consensus_insert_and_read(pg_conn):
    import psycopg

    game_dim = _stored_games("0022500578", OTHER_ID)
    store.insert_games(pg_conn, game_dim)
    assert schema_mod.consensus_table_exists(pg_conn)
    first = pd.DataFrame(
        [
            ("0022500578", 2025, 368, 198, 167, 0, FETCHED),
            (OTHER_ID, 2025, 0, 0, 0, 0, FETCHED),
        ],
        columns=store.CONSENSUS_COLUMNS,
    )
    assert store.insert_consensus(pg_conn, first) == 2
    # Insert-only: counts read again later do not replace the stored ones.
    again = first.assign(spread_away_picks=999)
    assert store.insert_consensus(pg_conn, again) == 0
    assert store.stored_consensus_game_ids(pg_conn, 2025) == {"0022500578", OTHER_ID}
    assert store.stored_consensus_game_ids(pg_conn, 2026) == set()

    read = store.fetch_consensus(pg_conn, game_ids=["0022500578"])
    assert read.to_dict("records") == [
        {
            "game_id": "0022500578",
            "season_year": 2025,
            "spread_away_picks": 368,
            "spread_home_picks": 198,
            "total_over_picks": 167,
            "total_under_picks": 0,
            "fetched_at": pd.Timestamp(FETCHED),
        }
    ]
    assert len(store.fetch_consensus(pg_conn)) == 2
    assert store.fetch_consensus(pg_conn, season_years=[2026]).empty

    events = store.fetch_game_events(pg_conn, 2025)
    assert events["event_id"].tolist() == [COVERS_ID, OTHER_COVERS]
    assert update_mod.plan_consensus(events, set(), now=FETCHED)

    # A game the store does not hold cannot get counts.
    orphan = first.iloc[:1].assign(game_id="0022500001")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.insert_consensus(pg_conn, orphan)
