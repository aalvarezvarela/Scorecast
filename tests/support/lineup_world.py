"""Synthetic lineups shared by closing and intermediate leakage tests."""

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.features import LINEUP_FEATURE_COLUMNS
from nba_ou.data_processing.lineups.style_matchup import STYLE_FEATURE_COLUMNS

PROJECTION_COLUMNS = [
    c for c in LINEUP_FEATURE_COLUMNS if c not in STYLE_FEATURE_COLUMNS
]


H, A, C, D = "1610612739", "1610612738", "1610612737", "1610612736"


ROSTERS = {team: [f"{team[-2:]}p{i}" for i in range(6)] for team in (H, A, C, D)}


DATES = [f"2025-01-{day:02d}" for day in range(1, 9)]


TARGET_DATE = DATES[-1]


def _game_id(index: int, slot: int) -> str:
    return f"00{index:06d}{slot:02d}"


TARGET = _game_id(len(DATES) - 1, 1)


SAME_DAY = _game_id(len(DATES) - 1, 2)


def _world(seed: int = 0):
    """Eight dates, two games a night, with every target-day row populated.

    The target date's box scores and results are filled in on purpose: in a
    historical build they exist, and the point is to prove they are not read.
    """
    rng = np.random.default_rng(seed)
    box, merged = [], []
    for index, date in enumerate(DATES):
        for slot, (home, away) in ((1, (H, A)), (2, (C, D))):
            game = _game_id(index, slot)
            for team in (home, away):
                minutes = rng.dirichlet(np.ones(6)) * 240.0
                for player, played in zip(ROSTERS[team], minutes, strict=True):
                    box.append(
                        {
                            "GAME_ID": game,
                            "TEAM_ID": team,
                            "GAME_DATE": pd.Timestamp(date),
                            "PLAYER_ID": player,
                            "MIN": float(played),
                        }
                    )
            merged.append(
                {
                    "GAME_ID": game,
                    "GAME_DATE": pd.Timestamp(date),
                    "TEAM_ID_TEAM_HOME": home,
                    "TEAM_ID_TEAM_AWAY": away,
                    "TOTAL_POINTS": float(rng.integers(190, 250)),
                }
            )
    ratings = []
    for as_of in pd.date_range(DATES[0], "2025-01-20"):
        for players in ROSTERS.values():
            for player in players:
                ratings.append(
                    {
                        "as_of_date": as_of,
                        "player_id": player,
                        "o_rating": float(rng.normal(0, 2)),
                        "d_rating": float(rng.normal(0, 2)),
                        "pace_rating": float(rng.normal(0, 1)),
                        "league_ortg": 112.0 + float(rng.normal(0, 1)),
                        "league_pace": 100.0 + float(rng.normal(0, 1)),
                        "fit_max_game_date": as_of - pd.Timedelta(days=1),
                    }
                )
    return pd.DataFrame(box), pd.DataFrame(merged), pd.DataFrame(ratings)
