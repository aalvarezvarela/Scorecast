"""TabPFN on its own: the daily frame it fits on and the output it stores.

The hosted client is never called: a stand-in regressor takes its place, so
what is under test is everything around the fit -- which rows and columns it
is handed, and that its output goes through the store as a TabPFN build.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from nba_ou.create_training_data.daily_frame_store import (
    latest_daily_frame,
    load_daily_frame,
    upload_daily_frame,
)
from nba_ou.create_training_data.train_data_store import (
    daily_frame_key,
    parse_daily_frame_key,
)
from nba_ou.modeling.refit import with_residual_targets
from nba_ou.modeling.registry_models import (
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecTraining,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget
from nba_ou.postgre_db.predictions_aiven import schema as store_schema
from nba_ou.prediction import prediction as prediction_module
from nba_ou.prediction.prediction_store import PredictionRunRecorder
from nba_ou.prediction.prediction_tabpfn_client import (
    FittedTabPFN,
    feature_slot_for,
    model_variant,
    predict_with_tabpfn,
    tabpfn_training_set,
)

from tests.fake_s3 import FakeS3Client

BUCKET = "b"
FEATURES = ["FEAT_A", "FEAT_B"]
LINE = "ODDS_TOTAL_LINE_bet365"
PREDICTION_AT = datetime(2026, 10, 21, 18, 0, tzinfo=ZoneInfo("Europe/Madrid"))


# --- the daily frame in S3 ---------------------------------------------------


def test_the_daily_key_is_named_like_a_local_dataset():
    key = daily_frame_key(
        dataset_type="closing_line", schema_version="2_5", limit_date="20261021"
    )
    assert key == "train_data/2_5/daily/closing_line_data_2_5_20261021.parquet"
    assert parse_daily_frame_key(key) == ("closing_line", "2_5", "20261021")
    assert parse_daily_frame_key("train_data/2_5/other.parquet") is None


def test_the_newest_frame_is_found_and_older_ones_pruned():
    s3 = FakeS3Client()
    # Not a daily frame of this kind: never listed, never pruned.
    s3.put_object(Bucket=BUCKET, Key="train_data/2_5/daily/notes.txt", Body=b"x")
    frame = pd.DataFrame({"GAME_ID": ["1"], "VALUE": [1.5]})
    for day in range(1, 6):
        upload_daily_frame(
            frame.assign(VALUE=float(day)),
            s3_client=s3,
            bucket=BUCKET,
            dataset_type="closing_line",
            schema_version="2_5",
            limit_date=f"2026-10-0{day}",
            keep=3,
        )

    daily = sorted(k for k in s3.store if k.endswith(".parquet"))
    assert [parse_daily_frame_key(k)[2] for k in daily] == [
        "20261003",
        "20261004",
        "20261005",
    ]
    assert "train_data/2_5/daily/notes.txt" in s3.store

    key, limit_date = latest_daily_frame(
        s3_client=s3, bucket=BUCKET, dataset_type="closing_line", schema_version="2_5"
    )
    assert limit_date.isoformat() == "2026-10-05"
    loaded = load_daily_frame(s3_client=s3, bucket=BUCKET, key=key)
    assert loaded["VALUE"].tolist() == [5.0]
    assert (
        latest_daily_frame(
            s3_client=s3,
            bucket=BUCKET,
            dataset_type="closing_line",
            schema_version="2_6",
        )
        is None
    )


# --- which slot, which rows ------------------------------------------------


def _slot(target, horizon=0, variant="main"):
    return ModelSlot(
        schema_version="2_5", target=target, horizon_minutes=horizon, variant=variant
    )


def test_the_features_come_from_the_closing_slot_of_the_target():
    slots = [
        _slot("line_error", horizon=60),
        _slot("line_error", variant="alt"),
        _slot("line_error"),
        _slot("total_points"),
    ]
    assert feature_slot_for("line_error", slots) == _slot("line_error")
    assert feature_slot_for("total_points", slots) == _slot("total_points")
    with pytest.raises(ValueError, match="spread_error"):
        feature_slot_for("spread_error", slots)


def test_the_variant_names_the_model_version():
    assert model_variant("v3.5") == "tabpfn_v3_5"
    assert model_variant("v3.5-fast") == "tabpfn_v3_5_fast"


def _spec(*, train_games=4, max_na_per_row=0):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version="2_5",
            target=ModelTarget.LINE_ERROR,
            prediction_strategy="line_error_regressor",
            horizon_minutes=0,
            dataset_type="closing_line",
        ),
        features=SpecFeatures(feature_names=FEATURES, n_features=len(FEATURES)),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=max_na_per_row),
        training=SpecTraining(
            params={"max_depth": 4}, n_estimators=10, train_games=train_games
        ),
    )


def _history(n=8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "GAME_ID": [f"00225000{i:02d}" for i in range(n)],
            "GAME_DATE": pd.date_range("2026-03-01", periods=n).astype(str),
            "LINE_ERROR": rng.normal(size=n),
            # Object-typed, as a column of a mixed frame can be.
            "FEAT_A": pd.Series(rng.normal(size=n)).astype(object),
            "FEAT_B": rng.normal(size=n),
            "NOT_A_FEATURE": ["x"] * n,
        }
    )
    frame.loc[0, "FEAT_B"] = np.nan  # over the NaN budget: never trained on
    return frame


def test_the_training_set_is_the_specs_window_of_numeric_features():
    training_set = tabpfn_training_set(_history(), _spec(), frame_key="k")

    assert list(training_set.X.columns) == FEATURES
    assert (training_set.X.dtypes == "float64").all()
    assert training_set.n_games == 4
    assert str(training_set.train_date_min.date()) == "2026-03-05"
    assert str(training_set.train_date_max.date()) == "2026-03-08"
    assert len(training_set.y) == 4


def test_the_residual_targets_are_derived_from_the_closing_outcomes():
    """The closing builder ships TOTAL_POINTS and HOME_MARGIN, not the
    residuals the line_error and spread_error specs train on."""
    frame = pd.DataFrame(
        {
            "TOTAL_POINTS": [230.0, 210.0],
            LINE: [224.5, 215.0],
            "HOME_MARGIN": [7.0, -3.0],
            "ODDS_SPREAD_LINE_HOME_bet365": [4.5, 1.0],
        }
    )
    derived = with_residual_targets(frame)
    assert derived["LINE_ERROR"].tolist() == [5.5, -5.0]
    assert derived["SPREAD_ERROR"].tolist() == [2.5, -4.0]
    # Never overwritten: an intermediate frame carries its own per snapshot.
    carried = frame.assign(LINE_ERROR=[1.0, 2.0], SPREAD_ERROR=[3.0, 4.0])
    assert with_residual_targets(carried)["LINE_ERROR"].tolist() == [1.0, 2.0]
    assert with_residual_targets(carried)["SPREAD_ERROR"].tolist() == [3.0, 4.0]


def test_the_window_can_be_overridden():
    training_set = tabpfn_training_set(
        _history(), _spec(), frame_key="k", train_games=6
    )
    assert training_set.n_games == 6


# --- the output, through the store -----------------------------------------


class _MeanRegressor:
    """Stands in for the hosted regressor."""

    def fit(self, X, y):
        self.mean_ = float(y.mean())
        return self

    def predict(self, X):
        assert (X.dtypes == "float64").all()
        return np.full(len(X), self.mean_)


def _tonight() -> pd.DataFrame:
    tipoffs = pd.to_datetime(["2026-10-21 19:30", "2026-10-21 22:00"]).tz_localize(
        "America/New_York"
    )
    return pd.DataFrame(
        {
            "GAME_ID": ["0022600001", "0022600002"],
            "SEASON_TYPE": ["Regular Season"] * 2,
            "GAME_DATE": ["2026-10-21"] * 2,
            "GAME_TIME": tipoffs,
            "TEAM_NAME_TEAM_HOME": ["Boston Celtics", "Los Angeles Lakers"],
            "TEAM_NAME_TEAM_AWAY": ["New York Knicks", "Golden State Warriors"],
            "MATCHUP_TEAM_HOME": ["BOS vs. NYK", "LAL vs. GSW"],
            LINE: [224.5, 230.5],
            "FEAT_A": [0.1, -0.2],
            "FEAT_B": [1.0, 0.5],
        }
    )


def _schedule() -> pd.DataFrame:
    tonight = _tonight()
    return pd.DataFrame(
        {
            "GAME_ID": tonight["GAME_ID"],
            "GAME_DATE_EST": ["2026-10-21T00:00:00"] * 2,
            "GAME_TIME": tonight["GAME_TIME"],
            "HOME_TEAM_ID": [1610612738, 1610612747],
            "VISITOR_TEAM_ID": [1610612752, 1610612744],
        }
    )


def test_a_tabpfn_output_is_stored_as_a_tabpfn_build(monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "upload_predictions_to_postgre",
        lambda df: pytest.fail("TabPFN wrote to Supabase"),
    )
    spec = _spec()
    training_set = tabpfn_training_set(_history(), spec, frame_key="train_data/k")
    model = FittedTabPFN(
        _MeanRegressor().fit(training_set.X, training_set.y),
        model_version="v3.5",
        client_version="0.6.1",
    )

    tonight = _tonight()
    predictions = predict_with_tabpfn(
        df=tonight,
        spec=spec,
        model=model,
        training_set=training_set,
        prediction_datetime=PREDICTION_AT,
    )
    assert predictions["PRED_LINE_ERROR"].tolist() == pytest.approx(
        [training_set.y.mean()] * 2
    )
    assert predictions["FIT_ID"].unique().tolist() == [f"{spec.spec_id}-20260308-4g"]

    recorder = PredictionRunRecorder(
        workflow=store_schema.WORKFLOW_TABPFN, game_date="2026-10-21", book="bet365"
    )
    recorder.set_schedule(_schedule())
    recorder.add_frame("closing_line", tonight, version="2_5")
    rejected = recorder.add_tabpfn(
        predictions, label="2_5/line_error/t0000/tabpfn_v3_5"
    )
    assert rejected is None

    tables = recorder.tables(status=store_schema.RUN_STATUS_OK)
    assert tables.run["workflow"] == store_schema.WORKFLOW_TABPFN
    build = tables.model_builds.iloc[0]
    assert len(tables.model_builds) == 1
    assert build["kind"] == store_schema.KIND_TABPFN
    assert build["variant"] == "tabpfn_v3_5"
    assert build["spec_id"] == spec.spec_id
    assert build["s3_build_prefix"] == "train_data/k"
    assert len(tables.predictions) == 2
