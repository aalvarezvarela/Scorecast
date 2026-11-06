"""The 30 NBA teams: tricode -> Wikipedia name and nickname (2019-20 on)."""

from __future__ import annotations

TEAMS: dict[str, tuple[str, str]] = {
    "ATL": ("Atlanta Hawks", "Hawks"),
    "BOS": ("Boston Celtics", "Celtics"),
    "BKN": ("Brooklyn Nets", "Nets"),
    "CHA": ("Charlotte Hornets", "Hornets"),
    "CHI": ("Chicago Bulls", "Bulls"),
    "CLE": ("Cleveland Cavaliers", "Cavaliers"),
    "DAL": ("Dallas Mavericks", "Mavericks"),
    "DEN": ("Denver Nuggets", "Nuggets"),
    "DET": ("Detroit Pistons", "Pistons"),
    "GSW": ("Golden State Warriors", "Warriors"),
    "HOU": ("Houston Rockets", "Rockets"),
    "IND": ("Indiana Pacers", "Pacers"),
    "LAC": ("Los Angeles Clippers", "Clippers"),
    "LAL": ("Los Angeles Lakers", "Lakers"),
    "MEM": ("Memphis Grizzlies", "Grizzlies"),
    "MIA": ("Miami Heat", "Heat"),
    "MIL": ("Milwaukee Bucks", "Bucks"),
    "MIN": ("Minnesota Timberwolves", "Timberwolves"),
    "NOP": ("New Orleans Pelicans", "Pelicans"),
    "NYK": ("New York Knicks", "Knicks"),
    "OKC": ("Oklahoma City Thunder", "Thunder"),
    "ORL": ("Orlando Magic", "Magic"),
    "PHI": ("Philadelphia 76ers", "76ers"),
    "PHX": ("Phoenix Suns", "Suns"),
    "POR": ("Portland Trail Blazers", "Trail Blazers"),
    "SAC": ("Sacramento Kings", "Kings"),
    "SAS": ("San Antonio Spurs", "Spurs"),
    "TOR": ("Toronto Raptors", "Raptors"),
    "UTA": ("Utah Jazz", "Jazz"),
    "WAS": ("Washington Wizards", "Wizards"),
}

# Short forms that appear in video titles besides the nickname.
TITLE_ALIASES: dict[str, tuple[str, ...]] = {
    "CLE": ("Cavs",),
    "PHI": ("Sixers",),
    "POR": ("Blazers",),
    "MIN": ("Wolves", "T-Wolves"),
    "NOP": ("Pels",),
    "DAL": ("Mavs",),
    "LAC": ("LA Clippers",),
}


def team_names(team: str) -> tuple[str, ...]:
    """Every name of a team that may appear in a title: full, city-less, short."""
    full, nickname = TEAMS[team]
    return (full, nickname, *TITLE_ALIASES.get(team, ()))
