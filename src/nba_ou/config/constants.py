"""
NBA Over/Under Predictor - Constants Module

This module contains all constant values used throughout the application,
including team mappings, season type definitions, and configuration values.
"""

OUT_STATUSES = {"Out", "Doubtful"}

# ==============================================================================
# TEAM MAPPINGS
# ==============================================================================

# Maps full team names to their NBA API team IDs
TEAM_ID_MAP = {
    "Atlanta Hawks": "1610612737",
    "Boston Celtics": "1610612738",
    "Cleveland Cavaliers": "1610612739",
    "New Orleans Pelicans": "1610612740",
    "Chicago Bulls": "1610612741",
    "Dallas Mavericks": "1610612742",
    "Denver Nuggets": "1610612743",
    "Golden State Warriors": "1610612744",
    "Houston Rockets": "1610612745",
    "Los Angeles Clippers": "1610612746",
    "Los Angeles Lakers": "1610612747",
    "Miami Heat": "1610612748",
    "Milwaukee Bucks": "1610612749",
    "Minnesota Timberwolves": "1610612750",
    "Brooklyn Nets": "1610612751",
    "New York Knicks": "1610612752",
    "Orlando Magic": "1610612753",
    "Indiana Pacers": "1610612754",
    "Philadelphia 76ers": "1610612755",
    "Phoenix Suns": "1610612756",
    "Portland Trail Blazers": "1610612757",
    "Sacramento Kings": "1610612758",
    "San Antonio Spurs": "1610612759",
    "Oklahoma City Thunder": "1610612760",
    "Toronto Raptors": "1610612761",
    "Utah Jazz": "1610612762",
    "Memphis Grizzlies": "1610612763",
    "Washington Wizards": "1610612764",
    "Detroit Pistons": "1610612765",
    "Charlotte Hornets": "1610612766",
}

# Maps various team name formats to standardized full names
# Handles abbreviations, historical names, and alternative spellings
TEAM_NAME_STANDARDIZATION = {
    # Short name mappings (city/state only)
    "Atlanta": "Atlanta Hawks",
    "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets",
    "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls",
    "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks",
    "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons",
    "Golden State": "Golden State Warriors",
    "Houston": "Houston Rockets",
    "Indiana": "Indiana Pacers",
    "Memphis": "Memphis Grizzlies",
    "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks",
    "Minnesota": "Minnesota Timberwolves",
    "New Jersey": "Brooklyn Nets",
    "New Orleans": "New Orleans Pelicans",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers",
    "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers",
    "Sacramento": "Sacramento Kings",
    "San Antonio": "San Antonio Spurs",
    "Seattle": "Oklahoma City Thunder",
    "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz",
    "Washington": "Washington Wizards",
    # LA Clippers special cases
    "LA Clippers": "Los Angeles Clippers",
    "L.A. Clippers": "Los Angeles Clippers",
    "LA": "Los Angeles Clippers",  # Common confusion
    # Historical team names
    "New Orleans/Oklahoma City Hornets": "New Orleans Hornets",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Jersey Nets": "Brooklyn Nets",
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "New Orleans Hornets": "New Orleans Pelicans",
    # Atlanta Hawks
    "ATL Hawks": "Atlanta Hawks",
    "Atlanta Hawks": "Atlanta Hawks",
    # Boston Celtics
    "BOS Celtics": "Boston Celtics",
    "Boston Celtics": "Boston Celtics",
    # Brooklyn Nets
    "BKN Nets": "Brooklyn Nets",
    "Brooklyn Nets": "Brooklyn Nets",
    # Charlotte Hornets / Bobcats
    "CHA Hornets": "Charlotte Hornets",
    "Charlotte Hornets": "Charlotte Hornets",
    # Chicago Bulls
    "CHI Bulls": "Chicago Bulls",
    "Chicago Bulls": "Chicago Bulls",
    # Cleveland Cavaliers
    "CLE Cavaliers": "Cleveland Cavaliers",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    # Dallas Mavericks
    "DAL Mavericks": "Dallas Mavericks",
    "Dallas Mavericks": "Dallas Mavericks",
    # Denver Nuggets
    "DEN Nuggets": "Denver Nuggets",
    "Denver Nuggets": "Denver Nuggets",
    # Detroit Pistons
    "DET Pistons": "Detroit Pistons",
    "Detroit Pistons": "Detroit Pistons",
    # Golden State Warriors
    "GS Warriors": "Golden State Warriors",
    "Golden State Warriors": "Golden State Warriors",
    # Houston Rockets
    "HOU Rockets": "Houston Rockets",
    "Houston Rockets": "Houston Rockets",
    # Indiana Pacers
    "IND Pacers": "Indiana Pacers",
    "Indiana Pacers": "Indiana Pacers",
    # Los Angeles Clippers
    "Los Angeles Clippers": "Los Angeles Clippers",
    # Los Angeles Lakers
    "LA Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Los Angeles": "Los Angeles Lakers",
    # Memphis Grizzlies
    "MEM Grizzlies": "Memphis Grizzlies",
    "Memphis Grizzlies": "Memphis Grizzlies",
    # Miami Heat
    "MIA Heat": "Miami Heat",
    "Miami Heat": "Miami Heat",
    # Milwaukee Bucks
    "MIL Bucks": "Milwaukee Bucks",
    "Milwaukee Bucks": "Milwaukee Bucks",
    # Minnesota Timberwolves
    "MIN Timberwolves": "Minnesota Timberwolves",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    # New Orleans Pelicans
    "NO Pelicans": "New Orleans Pelicans",
    "New Orleans Pelicans": "New Orleans Pelicans",
    # New York Knicks
    "NY Knicks": "New York Knicks",
    "New York Knicks": "New York Knicks",
    # Oklahoma City Thunder
    "OKC Thunder": "Oklahoma City Thunder",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    # Orlando Magic
    "ORL Magic": "Orlando Magic",
    "Orlando Magic": "Orlando Magic",
    # Philadelphia 76ers
    "PHI 76ers": "Philadelphia 76ers",
    "Philadelphia 76ers": "Philadelphia 76ers",
    # Phoenix Suns
    "PHO Suns": "Phoenix Suns",
    "Phoenix Suns": "Phoenix Suns",
    # Portland Trail Blazers
    "POR Trail Blazers": "Portland Trail Blazers",
    "Portland Trail Blazers": "Portland Trail Blazers",
    # Sacramento Kings
    "SAC Kings": "Sacramento Kings",
    "Sacramento Kings": "Sacramento Kings",
    # San Antonio Spurs
    "SA Spurs": "San Antonio Spurs",
    "San Antonio Spurs": "San Antonio Spurs",
    # Toronto Raptors
    "TOR Raptors": "Toronto Raptors",
    "Toronto Raptors": "Toronto Raptors",
    # Utah Jazz
    "UTA Jazz": "Utah Jazz",
    "Utah Jazz": "Utah Jazz",
    # Washington Wizards
    "WAS Wizards": "Washington Wizards",
    "Washington Wizards": "Washington Wizards",
    # Handle errors or irrelevant entries
    "Everton FC": None,  # Not an NBA team
}


TEAM_NAME_CONFERENCE_MAP = {
    # Eastern Conference
    "Atlanta Hawks": "East",
    "Boston Celtics": "East",
    "Brooklyn Nets": "East",
    "Charlotte Hornets": "East",
    "Chicago Bulls": "East",
    "Cleveland Cavaliers": "East",
    "Detroit Pistons": "East",
    "Indiana Pacers": "East",
    "Miami Heat": "East",
    "Milwaukee Bucks": "East",
    "New York Knicks": "East",
    "Orlando Magic": "East",
    "Philadelphia 76ers": "East",
    "Toronto Raptors": "East",
    "Washington Wizards": "East",
    # Western Conference
    "Dallas Mavericks": "West",
    "Denver Nuggets": "West",
    "Golden State Warriors": "West",
    "Houston Rockets": "West",
    "Los Angeles Clippers": "West",
    "Los Angeles Lakers": "West",
    "Memphis Grizzlies": "West",
    "Minnesota Timberwolves": "West",
    "New Orleans Pelicans": "West",
    "Oklahoma City Thunder": "West",
    "Phoenix Suns": "West",
    "Portland Trail Blazers": "West",
    "Sacramento Kings": "West",
    "San Antonio Spurs": "West",
    "Utah Jazz": "West",
}

TEAM_NAME_DIVISION_MAP = {
    # Atlantic
    "Boston Celtics": "Atlantic",
    "Brooklyn Nets": "Atlantic",
    "New York Knicks": "Atlantic",
    "Philadelphia 76ers": "Atlantic",
    "Toronto Raptors": "Atlantic",
    # Central
    "Chicago Bulls": "Central",
    "Cleveland Cavaliers": "Central",
    "Detroit Pistons": "Central",
    "Indiana Pacers": "Central",
    "Milwaukee Bucks": "Central",
    # Southeast
    "Atlanta Hawks": "Southeast",
    "Charlotte Hornets": "Southeast",
    "Miami Heat": "Southeast",
    "Orlando Magic": "Southeast",
    "Washington Wizards": "Southeast",
    # Northwest
    "Denver Nuggets": "Northwest",
    "Minnesota Timberwolves": "Northwest",
    "Oklahoma City Thunder": "Northwest",
    "Portland Trail Blazers": "Northwest",
    "Utah Jazz": "Northwest",
    # Pacific
    "Golden State Warriors": "Pacific",
    "Los Angeles Clippers": "Pacific",
    "Los Angeles Lakers": "Pacific",
    "Phoenix Suns": "Pacific",
    "Sacramento Kings": "Pacific",
    # Southwest
    "Dallas Mavericks": "Southwest",
    "Houston Rockets": "Southwest",
    "Memphis Grizzlies": "Southwest",
    "New Orleans Pelicans": "Southwest",
    "San Antonio Spurs": "Southwest",
}

# ==============================================================================
# SEASON TYPE MAPPINGS
# ==============================================================================

# Maps game ID prefixes to season types
SEASON_TYPE_MAP = {
    "001": "Preseason",
    "002": "Regular Season",
    "003": "All Star",
    "004": "Playoffs",
    "005": "Play-In Tournament",
    "006": "In-Season Final Game",
}

# Season types models train on: training_pipeline's default
# (data.exclude_playoffs=True) and what the daily refit keeps. Playoff
# basketball has shorter rotations and slower pace -- in the current training
# data playoff games average ~11.6 fewer total points than regular-season
# games -- so it is a materially different distribution from what the models
# are used to predict.
TRAINING_SEASON_TYPES: tuple[str, ...] = (
    "Regular Season",
    "Play-In Tournament",
)

# Season types no feature reads: exhibitions. Preseason rotations are
# experiments (starters play 15-25 minutes, camp invitees play a lot) and
# All-Star games are mini-games without defence. They are dropped when the
# builders' inputs are loaded (``nba_ou.utils.seasons.drop_exhibition_games``),
# so player averages, injury history and report history never count them.
EXHIBITION_SEASON_TYPES: tuple[str, ...] = (
    "Preseason",
    "All Star",
)

# ==============================================================================
# GAME CONFIGURATION
# ==============================================================================

# Aggregate team player-minutes in a 48-minute regulation game (5 * 48)
REGULATION_GAME_MINUTES = 240

# Aggregate team player-minutes corresponding to 40 game-clock minutes (5 * 40)
TEAM_MINUTES_PER_40 = 200

# Minimum minutes threshold for overtime detection
OVERTIME_THRESHOLD_MINUTES = 259

# ==============================================================================
# STATISTICAL WINDOWS
# ==============================================================================

# Default rolling window sizes for various statistics
DEFAULT_ROLLING_WINDOW = 5
WEIGHTED_ROLLING_WINDOW = 10

# Weights for weighted moving average (most recent to oldest)
WEIGHTED_MA_WEIGHTS = [20, 15, 10, 8, 6, 5, 4, 3, 2, 1]

# Minimum minutes played threshold for player statistics
MIN_MINUTES_THRESHOLD = 15

# Default number of top players to analyze
DEFAULT_TOP_PLAYERS = 3

# ==============================================================================
# EXTERNAL API CONFIGURATION
# ==============================================================================

# NBA Injury Report URL
NBA_INJURY_REPORTS_URL = "https://official.nba.com/nba-injury-report-2025-26-season/"

# Sports data API sport ID for NBA
NBA_SPORT_ID = 4

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

# Maintain old variable names for backward compatibility
TEAM_CONVERSION_DICT = TEAM_ID_MAP  # Legacy name
TEAM_NAME_EQUIVALENT_DICT = TEAM_NAME_STANDARDIZATION  # Legacy name
SEASON_TYPE_MAPPING = SEASON_TYPE_MAP  # Legacy name


CITY_TO_LATLON = {
    # NBA / US & Canada (team "cities" as in your dataframe)
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "Brooklyn": (40.6782, -73.9442),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Cleveland": (41.4993, -81.6944),
    "Dallas": (32.7767, -96.7970),
    "Denver": (39.7392, -104.9903),
    "Detroit": (42.3314, -83.0458),
    "Golden State": (37.7680, -122.3877),  # San Francisco area
    "Houston": (29.7604, -95.3698),
    "Indiana": (39.7684, -86.1581),  # Indianapolis
    "LA": (34.0522, -118.2437),
    "Los Angeles": (34.0522, -118.2437),
    "Memphis": (35.1495, -90.0490),
    "Miami": (25.7617, -80.1918),
    "Milwaukee": (43.0389, -87.9065),
    "Minnesota": (44.9778, -93.2650),  # Minneapolis
    "New Orleans": (29.9511, -90.0715),
    "New York": (40.7128, -74.0060),
    "Oklahoma City": (35.4676, -97.5164),
    "Orlando": (28.5383, -81.3792),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "Portland": (45.5152, -122.6784),
    "Sacramento": (38.5816, -121.4944),
    "San Antonio": (29.4241, -98.4936),
    "Toronto": (43.6532, -79.3832),
    "Utah": (40.7608, -111.8910),  # Salt Lake City
    "Washington": (38.9072, -77.0369),
    "Seattle": (47.6062, -122.3321),
    "New Jersey": (40.7357, -74.1724),  # Newark approximation
    "New Orleans/Oklahoma City": (35.4676, -97.5164),
    # International cities (preseason/exhibitions/etc.)
    "Adelaide": (-34.9285, 138.6007),
    "Athens": (37.9838, 23.7275),
    "Barcelona": (41.3851, 2.1734),
    "Beijing": (39.9042, 116.4074),
    "Berlin": (52.5200, 13.4050),
    "Buenos Aires": (-34.6037, -58.3816),
    "Brisbane": (-27.4698, 153.0251),
    "Cairns": (-16.9186, 145.7781),
    "Guangzhou": (23.1290, 113.2533),  # :contentReference[oaicite:0]{index=0}
    "Haifa": (32.7940, 34.9896),
    "Istanbul": (41.0082, 28.9784),
    "Madrid": (40.4168, -3.7038),
    "Melbourne": (-37.8136, 144.9631),
    "Milano": (45.4642, 9.1900),
    "Moscow": (55.7558, 37.6173),
    "Perth": (-31.9523, 115.8613),
    "Shanghai": (31.2304, 121.4737),
    "Sydney": (-33.8688, 151.2093),
    "Tel Aviv": (32.0853, 34.7818),
    "Vitoria": (
        42.8500,
        -2.6833,
    ),  # Vitoria-Gasteiz :contentReference[oaicite:1]{index=1}
    "Zalgiris Kaunas": (
        54.8985,
        23.9036,
    ),  # Kaunas :contentReference[oaicite:2]{index=2}
    # Team/club labels -> map to their home cities
    "Panathinaikos": (37.9838, 23.7275),  # Athens
    "Flamengo": (
        -22.9083,
        -43.1964,
    ),  # Rio de Janeiro :contentReference[oaicite:3]{index=3}
    "SESI/Franca": (
        -20.5393,
        -47.4013,
    ),  # Franca (SP), Brazil :contentReference[oaicite:4]{index=4}
    "Ratiopharm": (48.4000, 9.9833),  # Ulm :contentReference[oaicite:5]{index=5}
    # Ra'anana (Israel) as listed with punctuation variant too
    "Ra'anana": (32.1844, 34.8708),  # :contentReference[oaicite:6]{index=6}
    "Ra’anana": (32.1844, 34.8708),  # common apostrophe variant
}

CITY_TO_TIMEZONE = {
    # NBA / US & Canada
    "Atlanta": "America/New_York",
    "Boston": "America/New_York",
    "Brooklyn": "America/New_York",
    "Charlotte": "America/New_York",
    "Chicago": "America/Chicago",
    "Cleveland": "America/New_York",
    "Dallas": "America/Chicago",
    "Denver": "America/Denver",
    "Detroit": "America/New_York",
    "Golden State": "America/Los_Angeles",
    "Houston": "America/Chicago",
    "Indiana": "America/Indiana/Indianapolis",
    "LA": "America/Los_Angeles",
    "Los Angeles": "America/Los_Angeles",
    "Memphis": "America/Chicago",
    "Miami": "America/New_York",
    "Milwaukee": "America/Chicago",
    "Minnesota": "America/Chicago",
    "New Orleans": "America/Chicago",
    "New York": "America/New_York",
    "Oklahoma City": "America/Chicago",
    "Orlando": "America/New_York",
    "Philadelphia": "America/New_York",
    "Phoenix": "America/Phoenix",
    "Portland": "America/Los_Angeles",
    "Sacramento": "America/Los_Angeles",
    "San Antonio": "America/Chicago",
    "Toronto": "America/Toronto",
    "Utah": "America/Denver",
    "Washington": "America/New_York",
    "Seattle": "America/Los_Angeles",
    "New Jersey": "America/New_York",
    "New Orleans/Oklahoma City": "America/Chicago",
}
