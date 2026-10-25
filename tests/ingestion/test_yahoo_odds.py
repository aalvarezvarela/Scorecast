"""Yahoo odds: server-rendered pages -> one ``odds_yahoo`` row per game.

Pins what the scraper relies on: the payload is read from the
``self.__next_f.push`` scripts, the game object is the copy that lists its
``bets``, and every value is taken from the option's own label. The two
options come in Yahoo's internal order, which since 2026-27 is not always
away/over first; the old browser scraper read them by position and swapped
the splits of those games. Also pinned: the "% of bets" / "% of money" fields,
the repeated-market rules, a failed server render being retried, and the
schedule's time-zone shift (a US date read from two schedule pages). The
numbers are the site's, BKN @ CLE on 2026-10-24, checked against the rendered
page.
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import date
from functools import partial

import httpx
import pandas as pd
import pytest
from nba_ou.fetch_data.odds_yahoo import client as client_mod
from nba_ou.fetch_data.odds_yahoo import parse, scrape_yahoo
from nba_ou.postgre_db.odds_yahoo.create_db.create_odds_yahoo_db import _yahoo_columns

#: Compact, as in the site's payload.
_dumps = partial(json.dumps, separators=(",", ":"))

GAME_ID = "2026102405"
AWAY, HOME = "nba.t.17", "nba.t.5"
PATH = f"/nba/brooklyn-nets-cleveland-cavaliers-{GAME_ID}/"


def _team(team_id: str, abbr: str, name: str) -> list[dict]:
    return [{"teamId": team_id, "displayName": name, "abbreviation": abbr}]


def _bets() -> list[dict]:
    """The page's markets, in the page's order: Under before Over."""
    return [
        {
            "provider": "ACTION_NETWORK",
            "type": "SPREAD",
            "status": "CLOSED",
            "options": [
                {
                    "id": 807666160,
                    "optionDetails": [{"value": "10.5", "key": "points"}],
                    "americanOdds": -106,
                    "shortName": "+10.5",
                    "teams": _team(AWAY, "BKN", "Brooklyn"),
                    "teamIds": [AWAY],
                    "stakePercentage": "22.52",
                    "wagerPercentage": "29.48",
                },
                {
                    "id": 807666161,
                    "optionDetails": [{"value": "-10.5", "key": "points"}],
                    "americanOdds": -114,
                    "shortName": "-10.5",
                    "teams": _team(HOME, "CLE", "Cleveland"),
                    "teamIds": [HOME],
                    "stakePercentage": "77.48",
                    "wagerPercentage": "70.52",
                },
            ],
        },
        {
            "provider": "ACTION_NETWORK",
            "type": "OVER_UNDER",
            "status": "CLOSED",
            "options": [
                {
                    "id": 807666162,
                    "optionDetails": [{"value": "228.5", "key": "under"}],
                    "americanOdds": -106,
                    "shortName": "U 228.5",
                    "teams": [],
                    "teamIds": [],
                    "stakePercentage": "39.14",
                    "wagerPercentage": "44.12",
                },
                {
                    "id": 807666163,
                    "optionDetails": [{"value": "228.5", "key": "over"}],
                    "americanOdds": -111,
                    "shortName": "O 228.5",
                    "teams": [],
                    "teamIds": [],
                    "stakePercentage": "60.86",
                    "wagerPercentage": "55.88",
                },
            ],
        },
        {
            "provider": "ACTION_NETWORK",
            "type": "MONEY_LINE",
            "status": "CLOSED",
            "options": [
                {
                    "id": 807672976,
                    "optionDetails": [],
                    "americanOdds": -475,
                    "shortName": "-475",
                    "teams": _team(HOME, "CLE", "Cleveland"),
                    "teamIds": [HOME],
                    "stakePercentage": "96.92",
                    "wagerPercentage": "95.12",
                },
                {
                    "id": 807672975,
                    "optionDetails": [],
                    "americanOdds": 360,
                    "shortName": "+360",
                    "teams": _team(AWAY, "BKN", "Brooklyn"),
                    "teamIds": [AWAY],
                    "stakePercentage": "3.08",
                    "wagerPercentage": "4.88",
                },
            ],
        },
    ]


def _game(bets: list[dict] | None = None, pregame: str = "-10.5, O/U 228.5") -> dict:
    return {
        "gameId": f"nba.g.{GAME_ID}",
        "displayName": "Brooklyn @ Cleveland",
        "awayTeamId": AWAY,
        "awayTeam": {
            "fullName": "Brooklyn Nets",
            "teamId": AWAY,
            "displayName": "Brooklyn",
            "abbreviation": "BKN",
        },
        "homeTeamId": HOME,
        "homeTeam": {
            "fullName": "Cleveland Cavaliers",
            "teamId": HOME,
            "displayName": "Cleveland",
            "abbreviation": "CLE",
        },
        "startTime": "2026-10-24T16:30:00-07:00",
        "status": "FINAL",
        "season": 2026,
        "gameOddsSummary": {"pregameOddsDisplay": pregame, "favoriteId": HOME},
        "bets": _bets() if bets is None else bets,
        "partialGameBets": [],
    }


#: Other copies of the game the payload carries: the scoreboard strip's stub
#: and a reference row whose ``bets`` points elsewhere.
_STUB = {"gameId": f"nba.g.{GAME_ID}", "status": "FINAL", "season": 2026}
_REFERENCE = {"gameId": f"nba.g.{GAME_ID}", "bets": "$ab:props:children:0:game:bets"}


def _page(*rows: str, links: tuple[str, ...] = ()) -> str:
    """A page whose payload is ``rows`` split over two pushes."""
    rows = rows or ('0:{"P":null,"q":"?season=2026"}',)
    payload = "".join(f"{row}\n" for row in rows)
    half = len(payload) // 2
    scripts = "".join(
        f"<script>self.__next_f.push([1,{json.dumps(part)}])</script>"
        for part in (payload[:half], payload[half:])
    )
    anchors = "".join(f'<a href="{link}">game</a>' for link in links)
    return (
        "<!DOCTYPE html><html><body>"
        "<script>self.__next_f.push([0])</script>"
        f"{anchors}{scripts}</body></html>"
    )


def _strip_copy() -> dict:
    """The scoreboard strip's copy of a game of today: lines, no splits."""
    bets = _bets()
    for bet in bets:
        bet.pop("provider")
        for option in bet["options"]:
            for field in ("optionDetails", "stakePercentage", "wagerPercentage"):
                option.pop(field)
    return {
        "gameId": f"nba.g.{GAME_ID}",
        "awayTeamId": AWAY,
        "homeTeamId": HOME,
        "status": "PREGAME",
        "bets": bets,
    }


def _game_page(game: dict | None = None) -> str:
    return _page(
        f"0:{_dumps({'scoreboard': [_strip_copy()]})}",
        f"1:{_dumps(_STUB)}",
        f"2:{_dumps({'regularSeasonSeries': {'games': [_STUB]}})}",
        f"3:{_dumps(_REFERENCE)}",
        f'4:["$","div",null,{_dumps({"game": game or _game()})}]',
    )


FAILED_RENDER = _page(f"1:{_dumps(_STUB)}", 'ab:E{"digest":"3358115448"}')

EXPECTED = {
    "game_date": date(2026, 10, 24),
    "season_year": 2026,
    "team_home": "Cleveland Cavaliers",
    "team_away": "Brooklyn Nets",
    "team_home_abbr": "CLE",
    "team_away_abbr": "BKN",
    "spread_home": -10.5,
    "spread_away": 10.5,
    "moneyline_home": -475,
    "moneyline_away": 360,
    "total_line": 228.5,
    "total_pct_bets_over": 55.88,
    "total_pct_bets_under": 44.12,
    "total_pct_money_over": 60.86,
    "total_pct_money_under": 39.14,
    "spread_pct_bets_away": 29.48,
    "spread_pct_bets_home": 70.52,
    "spread_pct_money_away": 22.52,
    "spread_pct_money_home": 77.48,
    "moneyline_pct_bets_away": 4.88,
    "moneyline_pct_bets_home": 95.12,
    "moneyline_pct_money_away": 3.08,
    "moneyline_pct_money_home": 96.92,
}


def _row(game: dict) -> dict | None:
    payload = parse.flight_payload(_game_page(game))
    return parse.game_odds_row(parse.find_game(payload, GAME_ID))


# --- parsing ----------------------------------------------------------------


def test_columns_are_the_table_columns_but_game_id():
    table = [name for name, _ in _yahoo_columns()]
    assert sorted(parse.YAHOO_COLUMNS) == sorted(c for c in table if c != "game_id")


def test_payload_joins_the_data_pushes_and_unescapes_them():
    payload = parse.flight_payload(_game_page())
    assert payload.startswith("0:{")
    assert '"wagerPercentage":"29.48"' in payload.replace(" ", "")
    assert not parse.render_failed(payload)


def test_find_game_takes_the_copy_that_lists_its_bets():
    game = parse.find_game(parse.flight_payload(_game_page()), GAME_ID)
    assert isinstance(game["bets"], list) and len(game["bets"]) == 3
    assert parse.find_game(parse.flight_payload(_game_page()), "2026102406") is None


def test_the_full_copy_wins_over_the_scoreboard_strip():
    # On the page of a game of today the strip's copy comes first.
    payload = parse.flight_payload(_game_page())
    assert payload.index('"status":"PREGAME"') < payload.index('"status":"FINAL"')
    assert parse.find_game(payload, GAME_ID)["season"] == 2026


def test_a_game_without_its_season_takes_it_from_the_date():
    game = _game()
    game.pop("season")
    assert _row(game)["season_year"] == 2026


def test_values_follow_the_labels_not_the_option_order():
    # Under, and the home moneyline, come first on this page; the old scraper
    # stored this game's over splits as 44.12 / 39.14.
    assert _row(_game()) == EXPECTED


def test_wager_is_share_of_bets_and_stake_share_of_money():
    row = _row(_game())
    assert (row["spread_pct_bets_away"], row["spread_pct_money_away"]) == (
        29.48,
        22.52,
    )


def test_old_pages_without_option_details_read_from_the_labels():
    bets = _bets()
    for bet in bets:
        for option in bet["options"]:
            option.pop("optionDetails")
    # A 2020-21 label trailing the team name.
    bets[0]["options"][0]["shortName"] = "+10.5BrooklynNets"
    assert _row(_game(bets)) == EXPECTED


def test_missing_splits_stay_missing_and_the_game_is_kept():
    bets = _bets()
    for option in bets[1]["options"]:
        option["wagerPercentage"] = option["stakePercentage"] = None
    row = _row(_game(bets))
    assert row["total_line"] == 228.5
    assert row["total_pct_bets_over"] is None and row["total_pct_money_under"] is None
    assert row["spread_pct_bets_home"] == 70.52


@pytest.mark.parametrize(
    "change",
    [
        lambda bets: bets.pop(1),  # no total market
        lambda bets: bets[2]["options"].pop(),  # one-sided moneyline
        lambda bets: bets[2]["options"][0].update(americanOdds=None),  # no price
        lambda bets: bets.clear(),  # odds-empty-state: "bets": []
    ],
)
def test_a_game_without_complete_lines_has_no_row(change):
    bets = _bets()
    change(bets)
    assert _row(_game(bets)) is None


def test_a_repeated_market_takes_the_pregame_copy():
    bets = _bets()
    in_play = copy.deepcopy(bets[1])
    for option in in_play["options"]:
        option["optionDetails"][0]["value"] = "207.5"
        option["shortName"] = option["shortName"][0] + " 207.5"
    bets.insert(1, in_play)
    assert _row(_game(bets))["total_line"] == 228.5
    # Identical copies are one market.
    assert _row(_game(_bets() + [copy.deepcopy(_bets()[0])])) == EXPECTED


def test_a_repeated_market_the_summary_cannot_settle_is_an_error():
    bets = _bets()
    other = copy.deepcopy(bets[2])
    other["options"][0]["americanOdds"] = -500
    bets.append(other)
    with pytest.raises(parse.YahooParseError, match="2 different MONEY_LINE"):
        _row(_game(bets))


def test_a_broken_copy_is_skipped_but_alone_it_is_an_error():
    broken = copy.deepcopy(_bets()[2])
    for option, label in zip(broken["options"], ("1", "2"), strict=True):
        option.update(teamIds=[], teams=[], shortName=label, americanOdds=None)
    assert _row(_game(_bets() + [broken])) == EXPECTED
    with pytest.raises(parse.YahooParseError, match="neither side"):
        _row(_game(_bets()[:2] + [broken]))


def test_spreads_that_do_not_mirror_are_an_error():
    bets = _bets()
    bets[0]["options"][1]["optionDetails"][0]["value"] = "-9.5"
    with pytest.raises(parse.YahooParseError, match="do not mirror"):
        _row(_game(bets))


def test_a_failed_server_render_is_detected():
    payload = parse.flight_payload(FAILED_RENDER)
    assert parse.render_failed(payload)
    assert parse.find_game(payload, GAME_ID) is None
    assert parse.render_failed("")


def test_matchup_links_carry_the_us_game_date():
    html = _page(links=(PATH, "/nba/teams/brooklyn/", PATH))
    assert parse.matchup_paths(html) == {GAME_ID: PATH}
    assert parse.game_date_from_id(GAME_ID) == date(2026, 10, 24)
    with pytest.raises(ValueError):
        parse.game_date_from_id("20261024")


def test_season_year_for_date():
    assert parse.season_year_for_date(date(2026, 10, 24)) == 2026
    assert parse.season_year_for_date(date(2026, 3, 1)) == 2025
    assert parse.season_year_for_date(date(2020, 10, 15)) == 2019
    assert scrape_yahoo.schedule_params(date(2026, 10, 25)) == {
        "season": "2026",
        "date": "2026-10-25",
    }


# --- HTTP -------------------------------------------------------------------


def _client(handler, **kwargs) -> client_mod.YahooClient:
    return client_mod.YahooClient(
        transport=httpx.MockTransport(handler), sleep=(0, 0), backoff=0, **kwargs
    )


def _find(html: str, payload: str) -> dict | None:
    return parse.find_game(payload, GAME_ID)


def test_failed_renders_and_404s_are_retried():
    answers = [
        httpx.Response(200, text=FAILED_RENDER),
        httpx.Response(404, text="not found"),
        httpx.Response(200, text=_page(f"1:{_dumps(_STUB)}")),  # no bets
        httpx.Response(200, text=_game_page()),
    ]

    async def run():
        async with _client(lambda request: answers.pop(0), retries=4) as client:
            game = await client.fetch(PATH, {"section": "odds"}, _find)
            return game, client.requests_made

    game, requests_made = asyncio.run(run())
    assert parse.game_odds_row(game) == EXPECTED
    assert requests_made == 4


def test_a_page_that_never_parses_raises_after_its_retries():
    async def run():
        handler = lambda request: httpx.Response(200, text=FAILED_RENDER)  # noqa: E731
        async with _client(handler, retries=2) as client:
            await client.fetch(PATH, None, _find)

    with pytest.raises(client_mod.YahooError, match="server render failed"):
        asyncio.run(run())


def test_scrape_reads_a_us_date_from_two_schedule_pages(monkeypatch):
    # Laid out in European time: the 24th's evening games are on the 25th's
    # page, next to an early game of the 25th; the 24th's page also lists a
    # game of the 23rd.
    early = "/nba/indiana-pacers-brooklyn-nets-2026102517/"
    before = "/nba/detroit-pistons-boston-celtics-2026102302/"
    broken = "/nba/miami-heat-charlotte-hornets-2026102430/"
    schedules = {
        "2026-10-24": _page(links=(before,)),
        "2026-10-25": _page(links=(PATH, early, broken)),
    }
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == scrape_yahoo.SCHEDULE_PATH:
            assert request.url.params["season"] == "2026"
            return httpx.Response(200, text=schedules[request.url.params["date"]])
        if request.url.path == PATH:
            assert request.url.params["section"] == "odds"
            return httpx.Response(200, text=_game_page())
        return httpx.Response(200, text=FAILED_RENDER)

    monkeypatch.setattr(
        scrape_yahoo,
        "YahooClient",
        partial(
            client_mod.YahooClient,
            transport=httpx.MockTransport(handler),
            sleep=(0, 0),
            backoff=0,
            retries=2,
        ),
    )
    out = asyncio.run(scrape_yahoo.scrape_yahoo_days([date(2026, 10, 24)]))

    assert list(out.columns) == parse.YAHOO_COLUMNS
    assert out.to_dict("records") == [EXPECTED]
    # Only the 24th's games are opened; the broken one is retried, then left.
    assert early not in seen and before not in seen
    assert seen.count(broken) == 2


def test_no_dates_no_requests():
    out = asyncio.run(scrape_yahoo.scrape_yahoo_days([]))
    assert out.empty and list(out.columns) == parse.YAHOO_COLUMNS
    assert isinstance(out, pd.DataFrame)
