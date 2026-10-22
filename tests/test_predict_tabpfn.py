"""TabPFN on its own: the rows it fits on and the output it stores.

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
    rows_of_horizon,
    tabpfn_training_set,
)

BUCKET = "b"
FEATURES = ["FEAT_A", "FEAT_B"]
LINE = "ODDS_TOTAL_LINE_bet365"
PREDICTION_AT = datetime(2026, 10, 21, 18, 0, tzinfo=ZoneInfo("Europe/Madrid"))


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


def test_an_intermediate_run_takes_the_slot_of_its_horizon():
    slots = [_slot("line_error"), _slot("line_error", horizon=360)]
    assert feature_slot_for("line_error", slots, horizon_minutes=360) == _slot(
        "line_error", horizon=360
    )
    with pytest.raises(ValueError, match="t0060"):
        feature_slot_for("line_error", slots, horizon_minutes=60)


def test_the_variant_names_the_model_version():
    assert model_variant("v3.5") == "tabpfn_v3_5"
    assert model_variant("v3.5-fast") == "tabpfn_v3_5_fast"


def _spec(*, train_games=4, max_na_per_row=0, horizon=0):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version="2_5",
            target=ModelTarget.LINE_ERROR,
            prediction_strategy="line_error_regressor",
            horizon_minutes=horizon,
            dataset_type="intermediate_line" if horizon else "closing_line",
        ),
        features=SpecFeatures(feature_names=FEATURES, n_features=len(FEATURES)),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=max_na_per_row),
        training=SpecTraining(
            params={"max_depth": 4}, n_estimators=10, train_games=train_games
        ),
    )


def _history(n=8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    line_error = rng.normal(size=n)
    line = np.full(n, 224.5)
    frame = pd.DataFrame(
        {
            "GAME_ID": [f"00225000{i:02d}" for i in range(n)],
            "GAME_DATE": pd.date_range("2026-03-01", periods=n).astype(str),
            # A closing frame carries the outcome and the main book's line.
            "TOTAL_POINTS": line + line_error,
            LINE: line,
            "LINE_ERROR": line_error,
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


def test_playoff_and_preseason_games_are_not_trained_on():
    """The XGBoost refit's rows: regular season and play-in only."""
    history = _history()
    history.loc[6, "GAME_ID"] = "0042500106"  # playoffs
    history.loc[7, "GAME_ID"] = "0012500107"  # preseason
    training_set = tabpfn_training_set(history, _spec(), frame_key="k")

    assert training_set.n_games == 4
    assert str(training_set.train_date_max.date()) == "2026-03-06"


def _snapshots(frame: pd.DataFrame, horizons=(360, 60)) -> pd.DataFrame:
    """``frame``'s games at each horizon, each with its own line and residual:
    an intermediate frame ships LINE_ERROR against the snapshot's line."""
    parts = []
    for step, horizon in enumerate(horizons):
        part = frame.assign(TIME_TO_MATCH_MIN=horizon)
        part[LINE] = part[LINE] + step
        part["LINE_ERROR"] = part["TOTAL_POINTS"] - part[LINE]
        part["FEAT_B"] = part["FEAT_B"] + 10 * step
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_an_intermediate_spec_trains_on_its_horizon_only():
    history = _history()
    training_set = tabpfn_training_set(
        _snapshots(history), _spec(horizon=360), frame_key="k"
    )

    assert training_set.n_games == 4
    assert len(training_set.X) == 4
    expected = history.iloc[4:]
    assert training_set.y.tolist() == pytest.approx(expected["LINE_ERROR"].tolist())
    assert training_set.X["FEAT_B"].tolist() == pytest.approx(
        expected["FEAT_B"].tolist()
    )


def test_an_intermediate_spec_refuses_a_frame_without_snapshots():
    with pytest.raises(ValueError, match="TIME_TO_MATCH_MIN"):
        rows_of_horizon(_history(), _spec(horizon=360))


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


def test_an_intermediate_output_is_stored_at_its_horizon(monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "upload_predictions_to_postgre",
        lambda df: pytest.fail("TabPFN wrote to Supabase"),
    )
    spec = _spec(horizon=360)
    training_set = tabpfn_training_set(
        _snapshots(_history()), spec, frame_key="train_data/2_5/k.parquet"
    )
    model = FittedTabPFN(
        _MeanRegressor().fit(training_set.X, training_set.y),
        model_version="v3.5",
        client_version="0.6.1",
    )

    tonight = _snapshots(_tonight().assign(TOTAL_POINTS=np.nan))
    tonight["SNAPSHOT_TS_UTC"] = pd.Timestamp("2026-10-21 17:00", tz="UTC")
    tonight["SNAPSHOT_PASSED"] = tonight["TIME_TO_MATCH_MIN"].eq(360)
    rows = rows_of_horizon(tonight, spec)
    predictions = predict_with_tabpfn(
        df=rows,
        spec=spec,
        model=model,
        training_set=training_set,
        prediction_datetime=PREDICTION_AT,
        passthrough_columns=("SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED", "TIME_TO_MATCH_MIN"),
    )
    assert len(predictions) == 2
    assert predictions["TIME_TO_MATCH_MIN"].tolist() == [360, 360]
    assert predictions["INPUT_MODE"].unique().tolist() == ["own"]
    assert predictions["MODEL_HORIZON_MINUTES"].unique().tolist() == [360]
    assert predictions["MODEL_DATASET_TYPE"].unique().tolist() == ["intermediate_line"]

    recorder = PredictionRunRecorder(
        workflow=store_schema.WORKFLOW_TABPFN, game_date="2026-10-21", book="bet365"
    )
    recorder.set_schedule(_schedule())
    recorder.add_frame("intermediate_line", rows, version="2_5")
    rejected = recorder.add_tabpfn(
        predictions, label="2_5/line_error/t0360/tabpfn_v3_5"
    )
    assert rejected is None

    tables = recorder.tables(status=store_schema.RUN_STATUS_OK)
    build = tables.model_builds.iloc[0]
    assert build["kind"] == store_schema.KIND_TABPFN
    assert build["horizon_minutes"] == 360
    assert build["s3_build_prefix"] == "train_data/2_5/k.parquet"
    assert tables.predictions["snapshot_horizon_minutes"].tolist() == [360, 360]
