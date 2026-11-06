"""Builders and fakes for the coach presser archive tests (lab/coach_pressers/)."""

from __future__ import annotations

from lab.coach_pressers.config import PresserConfig

TEAM_NAMES = {"SAC": "Kings", "PHX": "Suns", "BOS": "Celtics", "DEN": "Nuggets"}


def game(
    game_id: str,
    day: str,
    away: str,
    home: str,
    *,
    status=3,
    tip=None,
    scores=(100, 110),
):
    """One ScheduleLeagueV2 game; ``day`` is the US date (YYYY-MM-DD)."""
    return {
        "gameId": game_id,
        "gameDateEst": f"{day}T00:00:00Z",
        "gameDateTimeUTC": tip or f"{day}T23:30:00Z",
        "gameStatus": status,
        "awayTeam": {
            "teamId": 1,
            "teamName": TEAM_NAMES[away],
            "teamTricode": away,
            "score": scores[0],
        },
        "homeTeam": {
            "teamId": 2,
            "teamName": TEAM_NAMES[home],
            "teamTricode": home,
            "score": scores[1],
        },
    }


def schedule(season: str, games: list[dict]) -> dict:
    return {"leagueSchedule": {"seasonYear": season, "gameDates": [{"games": games}]}}


def season_2021(*extra) -> dict:
    """SAC and PHX games in 2021-22 around the Walton -> Gentry change."""
    return schedule(
        "2021-22",
        [
            game("0022100010", "2021-11-19", "SAC", "PHX"),
            game("0022100020", "2021-11-21", "PHX", "SAC"),
            game("0022100030", "2021-12-01", "SAC", "PHX"),
            game("0012100001", "2021-10-05", "SAC", "PHX"),  # preseason: excluded
            *extra,
        ],
    )


LEAGUE_2021 = """
<div class="mw-heading mw-heading4"><h4 id="Off-season">Off-season</h4></div>
<ul><li>On June 1, 2021, the Boston Celtics hired Ime Udoka.</li></ul>
<div class="mw-heading mw-heading4"><h4 id="In-season">In-season</h4></div>
<ul>
<li>On November 21, 2021, the Sacramento Kings fired head coach Luke Walton and
named Alvin Gentry as their interim head coach.<sup>[62]</sup></li>
<li>On December 30, 2021, something unusual happened to a coach.</li>
</ul>
<div class="mw-heading mw-heading3"><h3 id="Coaches_of_the_Month">Coaches</h3></div>
"""

TEAM_PAGE = """
<table class="infobox"><tr><th>Head coach</th><td class="infobox-data">{coaches}</td></tr></table>
<table class="toccolours"><tr><td>
<div class="center"><b>Head coach</b></div>
<ul><li><a href="#">{end_coach}</a></li></ul>
<div class="center"><b>Assistant(s)</b></div>
<ul><li><a href="#">Doug Christie</a></li><li>Jonah Herscu</li></ul>
<hr><div class="center"><b>Legend</b></div>
</td></tr></table>
<table class="navbox"><tr><th>Head coach</th><td>Somebody Current</td></tr></table>
"""


def team_page(coaches: list[str], end_coach: str) -> str:
    return TEAM_PAGE.format(coaches="<br>".join(coaches), end_coach=end_coach)


class FakeWikipedia:
    def __init__(self, pages: dict[str, str]):
        self.pages, self.requests = pages, []

    def article(self, title: str):
        self.requests.append(title)
        return self.pages.get(title)


def wikipedia_2021() -> FakeWikipedia:
    return FakeWikipedia(
        {
            "2021–22 NBA season": LEAGUE_2021,
            "2021–22 Sacramento Kings season": team_page(
                [
                    '<a href="#">Luke Walton</a> (fired)',
                    "Alvin Gentry (interim)<sup>[1]</sup>",
                ],
                "Alvin Gentry",
            ),
            "2021–22 Phoenix Suns season": team_page(
                ["Monty Williams"], "Monty Williams"
            ),
            "2020–21 Sacramento Kings season": team_page(
                ["Luke Walton"], "Luke Walton"
            ),
        }
    )


def players_2021(season: str) -> dict[str, list[str]]:
    return {"SAC": ["De'Aaron Fox", "Harrison Barnes"], "PHX": ["Devin Booker"]}


def config_2021(tmp_path, **kwargs) -> PresserConfig:
    return PresserConfig(
        2021,
        tmp_path / "pressers",
        overrides_csv=tmp_path / "overrides.csv",
        channels_csv=tmp_path / "channels.csv",
        **kwargs,
    )
