"""The prediction job on a tonight-only feature frame.

``create_df_to_predict(target_game_ids=...)`` returns tonight's rows alone, so
nothing downstream may read history from that frame: the historical-average
baseline takes it from the build's layer inputs instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nba_ou.prediction import baseline_predictions
from nba_ou.prediction.baseline_predictions import (
    HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME,
    baseline_history_window,
)

import scripts.predict_nba_games as job


def _history() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-04-10",
            "2024-10-25",
            "2025-03-01",
            "2025-10-22",
            "2025-11-02",
            "2025-11-03",
        ]
    )
    return pd.DataFrame(
        {
            "GAME_ID": [f"g{i}" for i in range(len(dates))],
            "GAME_DATE": dates,
            "TOTAL_POINTS": [200.0, 210.0, 220.0, 230.0, 240.0, np.nan],
        }
    )


def test_window_is_this_season_and_the_one_before_until_tonight():
    window = baseline_history_window(_history(), "2025-11-03")
    assert window["GAME_ID"].tolist() == ["g1", "g2", "g3", "g4"]
    # On opening night only last season remains.
    opener = baseline_history_window(_history(), "2025-10-22")
    assert opener["GAME_ID"].tolist() == ["g1", "g2"]


def test_baselines_read_history_from_the_layer_inputs(monkeypatch):
    tonight = pd.DataFrame(
        {
            "GAME_ID": ["g5"],
            "GAME_DATE": [pd.Timestamp("2025-11-03")],
            "GAME_TIME": [pd.Timestamp("2025-11-03 19:00", tz="US/Eastern")],
            "ODDS_TOTAL_LINE_bet365": [225.5],
            "TOTAL_POINTS": [np.nan],
        }
    )
    history = _history()
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[],
            headless=True,
            nba_injury_reports_url="",
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
        ),
    )
    monkeypatch.setattr(job, "update_all_databases", lambda **_: None)
    monkeypatch.setattr(
        job,
        "get_all_info_for_scheduled_games",
        lambda **_: {"scheduled_games": tonight[["GAME_ID"]]},
    )
    built = {}

    def build(**kwargs):
        built.update(kwargs)
        return tonight.copy(), history

    monkeypatch.setattr(job, "create_df_to_predict", build)
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(job, "upload_input_features_snapshot", lambda **_: "k")
    uploaded = {}

    def upload(**kwargs):
        uploaded[kwargs["model_prefix"]] = kwargs["predictions_df"]
        return "a/b/c/d"

    monkeypatch.setattr(job, "upload_model_predictions_snapshot", upload)
    monkeypatch.setattr(
        baseline_predictions, "upload_predictions_to_postgre", lambda df: None
    )

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    job.predict_nba_games(run_tabpfn_client=False)

    assert built["target_game_ids"] == ["g5"]
    average = uploaded[HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME]
    assert average["PRED_TOTAL_POINTS"].tolist() == [
        np.mean([210.0, 220.0, 230.0, 240.0])
    ]


def test_tabpfn_is_refused_before_any_work(monkeypatch):
    monkeypatch.setattr(
        job, "update_all_databases", lambda **_: pytest.fail("work was started")
    )
    with pytest.raises(SystemExit, match="--no-tabpfn"):
        job.predict_nba_games(run_tabpfn_client=True)
