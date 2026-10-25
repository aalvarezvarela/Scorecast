"""Scrape Yahoo Sports' NBA odds and public betting splits, one row per game.

For each US game date the schedule pages list the matchups, and each
matchup's odds page gives that game's full-game lines and "% of bets" /
"% of money" (``parse.game_odds_row``). Plain HTTP (``client.YahooClient``),
no browser.

Yahoo lays the schedule out in the time zone it guesses from the IP (Europe
for a European runner, where an evening US game falls on the next day), so a
US date D is read from the schedule pages of D and D + 1 and only the games
whose id carries D are kept. That holds in any time zone from US Pacific to
well east of Europe.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd
from nba_ou.fetch_data.odds_yahoo.client import (
    DEFAULT_CONCURRENCY,
    YahooClient,
    YahooError,
)
from nba_ou.fetch_data.odds_yahoo.parse import (
    YAHOO_COLUMNS,
    YahooParseError,
    find_game,
    game_date_from_id,
    game_odds_row,
    matchup_paths,
    season_year_for_date,
)

SCHEDULE_PATH = "/nba/schedule/"


def schedule_params(d: date) -> dict[str, str]:
    return {"season": str(season_year_for_date(d)), "date": d.isoformat()}


async def _schedule_matchups(
    client: YahooClient, day: date, failures: list[str]
) -> dict[str, str]:
    try:
        return await client.fetch(
            SCHEDULE_PATH,
            schedule_params(day),
            lambda html, payload: matchup_paths(html),
        )
    except YahooError as exc:
        failures.append(f"schedule {day.isoformat()}: {exc}")
        return {}


async def _game_row(
    client: YahooClient, yahoo_game_id: str, path: str, failures: list[str]
) -> dict | None:
    try:
        game = await client.fetch(
            path,
            {"section": "odds"},
            lambda html, payload: find_game(payload, yahoo_game_id),
        )
    except YahooError as exc:
        failures.append(f"game {yahoo_game_id}: {exc}")
        return None
    try:
        return game_odds_row(game)
    except (YahooParseError, KeyError, TypeError, ValueError) as exc:
        # One unreadable game must not cost the others.
        failures.append(f"game {yahoo_game_id} ({path}): {exc!r}")
        return None


async def scrape_yahoo_days(
    game_dates: Iterable[date], *, concurrency: int = DEFAULT_CONCURRENCY
) -> pd.DataFrame:
    """Yahoo's odds of every NBA game on ``game_dates`` (US dates).

    One row per game with odds, in the ``odds_yahoo`` columns but
    ``game_id``. A game without full-game lines is left out; one whose page
    cannot be read is left out and reported, so a later run can retry it.
    """
    dates = sorted(set(game_dates))
    if not dates:
        return pd.DataFrame(columns=YAHOO_COLUMNS)
    wanted = set(dates)
    schedule_days = sorted({d + timedelta(days=k) for d in dates for k in (0, 1)})
    failures: list[str] = []

    async with YahooClient(concurrency=concurrency) as client:
        pages = await asyncio.gather(
            *(_schedule_matchups(client, d, failures) for d in schedule_days)
        )
        matchups = {
            game_id: path
            for page in pages
            for game_id, path in page.items()
            if game_date_from_id(game_id) in wanted
        }
        rows = await asyncio.gather(
            *(
                _game_row(client, game_id, path, failures)
                for game_id, path in sorted(matchups.items())
            )
        )
        requests_made = client.requests_made

    out = pd.DataFrame([r for r in rows if r is not None], columns=YAHOO_COLUMNS)
    out = out.sort_values(["game_date", "team_home"], ignore_index=True)
    found = out.groupby("game_date").size()
    for d in dates:
        games_on_day = sum(game_date_from_id(g) == d for g in matchups)
        print(
            f"Yahoo {d.isoformat()}: {games_on_day} matchups, "
            f"{int(found.get(d, 0))} with odds"
        )
    print(f"Yahoo: {requests_made} requests")
    if failures:
        print(f"Yahoo: {len(failures)} page(s) could not be read:")
        for failure in failures:
            print(f"  - {failure}")
    return out


if __name__ == "__main__":
    yesterday = date.today() - timedelta(days=1)
    print(asyncio.run(scrape_yahoo_days([yesterday])).to_string())
