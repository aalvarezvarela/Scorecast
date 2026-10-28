"""Small deterministic slates; no external services or credentials."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from nba_ou.dashboard.repositories.predictions import TodayData

NOW = pd.Timestamp("2026-10-28T18:00:00Z")


def make_prediction(*, game_id=1, horizon=240, target=2, kind=1, **changes):
    tipoff = NOW + pd.Timedelta(hours=4 if game_id == 1 else 2)
    row = {
        "game_id": game_id,
        "game_date": NOW.date(),
        "home_team": "Boston Celtics" if game_id == 1 else "Los Angeles Lakers",
        "away_team": "New York Knicks" if game_id == 1 else "Golden State Warriors",
        "tipoff_at": tipoff,
        "tipoff_at_prediction": tipoff,
        "predicted_at": NOW - pd.Timedelta(minutes=10),
        "run_finished_at": NOW - pd.Timedelta(minutes=5),
        "input_at": NOW - pd.Timedelta(minutes=15),
        "workflow": 3 if kind == 3 else 1,
        "kind": kind,
        "target": target,
        "status": 0,
        "dataset_type": 1 if horizon == 0 else 2,
        "input_mode": 0 if horizon == 0 else 2,
        "model_horizon_minutes": horizon,
        "snapshot_horizon_minutes": None
        if horizon == 0
        else 255
        if game_id == 1
        else 135,
        "time_to_tipoff_seconds": int((tipoff - NOW).total_seconds()) + 600,
        "schema_version": "2_5",
        "variant": "main",
        "fit_id": "fixture-fit",
        "model_build_id": target * 10000 + horizon + kind,
        "input_id": game_id * 10000 + target * 100 + horizon,
        "run_id": 1,
        "prediction_value": {1: 1.0, 2: 232.1, 3: -2.0}.get(target, 1.0),
        "model_line": 229.0 if target != 3 else 3.5,
        "bet_line_x2": 461 if target != 3 else 7,
        "anchor_book": "bet365",
        "books_mean_line": 230.0 if target != 3 else 3.0,
        "books_median_line": 229.5 if target != 3 else 3.5,
        "books_count": 6,
        "bet_line_age_minutes": 5.0,
        "line_source": 1,
    }
    row.update(changes)
    return row


@pytest.fixture
def today_data():
    games = pd.DataFrame(
        [
            {
                "game_id": game,
                "game_date": NOW.date(),
                "tipoff_at": NOW + pd.Timedelta(hours=4 if game == 1 else 2),
                "home_team": "Boston Celtics" if game == 1 else "Los Angeles Lakers",
                "away_team": "New York Knicks"
                if game == 1
                else "Golden State Warriors",
                "home_team_id": 1610612738 if game == 1 else 1610612747,
                "away_team_id": 1610612752 if game == 1 else 1610612744,
                "status": 0,
            }
            for game in (1, 2)
        ]
    )
    predictions = pd.DataFrame(
        [
            make_prediction(game_id=game, horizon=horizon, target=target)
            for game in (1, 2)
            for horizon in (0, 120, 240, 360)
            for target in (1, 2, 3)
        ]
    )
    return TodayData(games, predictions, NOW.to_pydatetime())


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = NOW.to_pydatetime()
        return value.astimezone(tz or UTC)
