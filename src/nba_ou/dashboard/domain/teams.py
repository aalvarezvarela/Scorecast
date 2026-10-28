"""Team identity and logo URLs without importing the legacy application."""

from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION

# Preserve the existing ESPN team images. NBA IDs provide stable lookup even
# when Aiven's display names differ (e.g. LA / Los Angeles Clippers).
_ESPN_IDS = {
    "Atlanta Hawks": 1,
    "Boston Celtics": 2,
    "Brooklyn Nets": 17,
    "Charlotte Hornets": 30,
    "Chicago Bulls": 4,
    "Cleveland Cavaliers": 5,
    "Dallas Mavericks": 6,
    "Denver Nuggets": 7,
    "Detroit Pistons": 8,
    "Golden State Warriors": 9,
    "Houston Rockets": 10,
    "Indiana Pacers": 11,
    "Los Angeles Clippers": 12,
    "Los Angeles Lakers": 13,
    "Memphis Grizzlies": 29,
    "Miami Heat": 14,
    "Milwaukee Bucks": 15,
    "Minnesota Timberwolves": 16,
    "New Orleans Pelicans": 3,
    "New York Knicks": 18,
    "Oklahoma City Thunder": 25,
    "Orlando Magic": 19,
    "Philadelphia 76ers": 20,
    "Phoenix Suns": 21,
    "Portland Trail Blazers": 22,
    "Sacramento Kings": 23,
    "San Antonio Spurs": 24,
    "Toronto Raptors": 28,
    "Utah Jazz": 26,
    "Washington Wizards": 27,
}
_NAMES_BY_ID = {int(team_id): name for name, team_id in TEAM_ID_MAP.items()}


def full_team_name(name: str, nba_team_id=None) -> str:
    """Prefer stable NBA IDs to abbreviated or source-specific display names."""
    if nba_team_id is not None:
        try:
            name = _NAMES_BY_ID.get(int(nba_team_id), name)
        except (ValueError, TypeError):
            pass
    return TEAM_NAME_STANDARDIZATION.get(name, name)


def team_logo_url(name: str, nba_team_id=None) -> str | None:
    name = full_team_name(name, nba_team_id)
    espn_id = _ESPN_IDS.get(name)
    if espn_id is None:
        return None
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{espn_id}.png"
