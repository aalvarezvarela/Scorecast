"""Covers line history: matchups and line-history HTML -> the SBR store's rows.

Pins the parsing rules the store relies on -- visible spans only, the side
conventions, the year and time zone of the Eastern date labels, the per-minute
collapse and the skipped OPEN table -- plus the rebuilt ``consensus_opener``,
the request plan, the update loop's skips and that writes stay in the
``line_history_covers`` schema. The markup is the site's, with the quotes of
MEM @ ORL on 2026-01-15 (BetMGM, bet365).
"""

from __future__ import annotations

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


def _client(answer: str = ANSWER):
    return FakeClient(
        {
            client_mod.matchups_url(DAY): MATCHUPS,
            client_mod.history_url(COVERS_ID): answer,
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
        }

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
    def _main(self, monkeypatch, client: FakeClient) -> int:
        from contextlib import contextmanager

        cli = _load_cli()

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
        monkeypatch.setattr(
            "sys.argv",
            ["update_covers_line_history.py", "--dry-run", "--dates", str(DAY)],
        )
        return cli.main()

    def test_failed_downloads_exit_1(self, monkeypatch):
        failing = FakeClient({}, failing={client_mod.matchups_url(DAY)})
        assert self._main(monkeypatch, failing) == 1

    def test_clean_run_exits_0(self, monkeypatch):
        assert self._main(monkeypatch, _client()) == 0
