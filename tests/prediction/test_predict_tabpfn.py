"""TabPFN on its own: the rows it fits on and the output it stores.

The hosted client is never called: a stand-in regressor takes its place, so
what is under test is everything around the fit -- which rows and columns it
is handed, and that its output goes through the store as a TabPFN build.
"""

from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock
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
from nba_ou.prediction import tabpfn_run
from nba_ou.prediction.prediction_store import PredictionRunRecorder
from nba_ou.prediction.prediction_tabpfn_client import (
    FittedTabPFN,
    feature_slot_for,
    fit_tabpfn,
    model_variant,
    parse_modes,
    predict_with_tabpfn,
    rows_of_horizon,
    tabpfn_training_set,
)

from scripts import predict_tabpfn as job

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


def test_a_thinking_mode_is_a_variant_of_its_own():
    """The standard model keeps the variant every earlier run was stored as."""
    assert model_variant("v3.5", "standard") == "tabpfn_v3_5"
    assert model_variant("v3.5", "thinking_medium") == "tabpfn_v3_5_thinking_medium"
    assert model_variant("v3.5", "thinking_high") == "tabpfn_v3_5_thinking_high"
    with pytest.raises(ValueError, match="thinking"):
        model_variant("v3.5", "thinking")


def test_modes_are_parsed_once_each_in_order():
    assert parse_modes(["standard"]) == ["standard"]
    assert parse_modes(["thinking_high, standard thinking_high"]) == [
        "thinking_high",
        "standard",
    ]
    with pytest.raises(ValueError, match="high"):
        parse_modes(["high"])
    with pytest.raises(ValueError, match="at least one"):
        parse_modes([" , "])


@pytest.mark.parametrize(
    ("mode", "overrides"),
    [
        ("standard", {}),
        ("thinking_medium", {"thinking_effort": "medium"}),
        ("thinking_high", {"thinking_effort": "high"}),
    ],
)
def test_a_thinking_fit_asks_the_client_for_its_effort(monkeypatch, mode, overrides):
    """Nothing reaches the hosted service: the client module is a stand-in."""
    created = []

    class _Regressor:
        @classmethod
        def create_default_for_version(cls, version, **options):
            created.append((version, options))
            return cls()

        def fit(self, X, y):
            return self

    client = SimpleNamespace(
        __version__="0.6.1",
        TabPFNRegressor=_Regressor,
        set_access_token=Mock(),
        init=Mock(),
    )
    monkeypatch.setitem(sys.modules, "tabpfn_client", client)
    monkeypatch.setenv("TABPFN_ACCESS_TOKEN", "token")
    training_set = tabpfn_training_set(_history(), _spec(), frame_key="k")

    model = fit_tabpfn(training_set, model_version="v3.5", mode=mode)
    assert created == [("v3.5", {"random_state": 0, **overrides})]
    assert model.mode == mode


def _spec(
    *,
    train_games=4,
    max_na_per_row=0,
    horizon=0,
    target="line_error",
    features=FEATURES,
    schema_version="2_5",
):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version=schema_version,
            target=ModelTarget(target),
            prediction_strategy=f"{target}_regressor",
            horizon_minutes=horizon,
            dataset_type="intermediate_line" if horizon else "closing_line",
        ),
        features=SpecFeatures(feature_names=features, n_features=len(features)),
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


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], ["line_error", "total_points"]),
        (["--targets", "line_error", "total_points"], ["line_error", "total_points"]),
        (["--targets", "line_error,total_points"], ["line_error", "total_points"]),
        (
            ["--target", "line_error total_points line_error"],
            ["line_error", "total_points"],
        ),
        (["--target", "spread_error"], ["spread_error"]),
    ],
)
def test_cli_accepts_target_lists_and_legacy_target(monkeypatch, args, expected):
    predict = Mock()
    monkeypatch.setattr(job, "predict_tabpfn", predict)
    job.main(args)
    assert predict.call_args.kwargs["targets"] == expected


@pytest.mark.parametrize(
    "args",
    [
        ["--targets", "totals"],
        ["--targets", "line_error", "typo"],
        ["--targets", " , "],
        ["--targets"],
        [
            "--targets",
            "line_error",
            "total_points",
            "--feature-slot",
            "2_5/line_error/t0000/main",
        ],
    ],
)
def test_invalid_target_lists_fail_before_starting(monkeypatch, args):
    predict = Mock()
    monkeypatch.setattr(job, "predict_tabpfn", predict)
    with pytest.raises(SystemExit) as exc:
        job.main(args)
    assert exc.value.code == 2
    predict.assert_not_called()


#: What the script itself calls; ``_run_dependencies`` patches the rest in
#: ``tabpfn_run``.
SCRIPT_DEPENDENCIES = {
    "make_s3_client",
    "update_all_databases",
    "check_line_feed",
    "get_all_info_for_scheduled_games",
}


def _run_dependencies(monkeypatch, *, horizon=0, total_schema="2_5"):
    """Fake I/O and paid inference; keep target preparation, serving and storage real."""
    specs = {
        "line_error": _spec(horizon=horizon),
        "total_points": _spec(
            target="total_points",
            train_games=6,
            features=["FEAT_B"],
            horizon=horizon,
            schema_version=total_schema,
        ),
    }
    clock = Mock(wraps=datetime)
    clock.now.side_effect = lambda tz: PREDICTION_AT.astimezone(tz)
    monkeypatch.setattr(job, "datetime", clock)
    monkeypatch.setattr(tabpfn_run, "datetime", clock)
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[spec.identity.slot() for spec in specs.values()],
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket=BUCKET,
            headless=True,
            nba_injury_reports_url="unused",
        ),
    )
    history = _history()
    tonight = _tonight()
    if horizon:
        history = _snapshots(history, horizons=(horizon, 60))
        tonight = _snapshots(
            tonight.assign(TOTAL_POINTS=np.nan), horizons=(horizon, 60)
        )
        tonight["SNAPSHOT_TS_UTC"] = pd.to_datetime(
            tonight["GAME_TIME"], utc=True
        ) - pd.to_timedelta(tonight["TIME_TO_MATCH_MIN"], unit="min")
        tonight["SNAPSHOT_PASSED"] = True

    def fit(training_set, *, model_version, mode):
        return FittedTabPFN(
            _MeanRegressor().fit(training_set.X, training_set.y),
            model_version=model_version,
            client_version="0.6.1",
            mode=mode,
        )

    dependencies = {
        "make_s3_client": Mock(return_value=object()),
        "resolve_channel": Mock(
            side_effect=lambda **kw: SimpleNamespace(
                spec=specs[kw["slot"].target.value]
            )
        ),
        "SeasonPartitions": Mock(
            return_value=SimpleNamespace(
                prefix="train_data/partitions",
                limit_date=lambda: PREDICTION_AT.date(),
            )
        ),
        "resolve_training_frame": Mock(
            return_value=SimpleNamespace(
                df=history,
                build_id="frame",
                checksum="sha256:test",
            )
        ),
        "update_all_databases": Mock(),
        "check_line_feed": Mock(return_value=object()),
        "get_all_info_for_scheduled_games": Mock(
            return_value={
                "scheduled_games": _schedule(),
                "games_not_updated": [],
            }
        ),
        "tonight_closing_rows": Mock(return_value=tonight),
        "tonight_intermediate_frame": Mock(return_value=tonight),
        "fit_tabpfn": Mock(side_effect=fit),
        "persist_prediction_run": Mock(return_value=[]),
    }
    for name, dependency in dependencies.items():
        # The job's own steps are the script's; the rest is the shared
        # TabPFN run (nba_ou.prediction.tabpfn_run).
        module = job if name in SCRIPT_DEPENDENCIES else tabpfn_run
        monkeypatch.setattr(module, name, dependency)
    return dependencies


@pytest.mark.parametrize("horizon", [0, 360])
@pytest.mark.parametrize("persist", [True, False])
def test_multiple_targets_share_preparation_and_keep_their_specs(
    monkeypatch, horizon, persist
):
    deps = _run_dependencies(monkeypatch, horizon=horizon)
    job.predict_tabpfn(
        targets=["line_error", "total_points", "line_error"],
        line="intermediate" if horizon else "closing",
        persist=persist,
    )
    for name in (
        "update_all_databases",
        "check_line_feed",
        "get_all_info_for_scheduled_games",
        "SeasonPartitions",
        "resolve_training_frame",
    ):
        deps[name].assert_called_once()
    builder = "tonight_intermediate_frame" if horizon else "tonight_closing_rows"
    deps[builder].assert_called_once()
    if horizon:
        # Only the slot's horizon is built, not the whole snapshot grid.
        assert deps[builder].call_args.kwargs["horizon"] == horizon
    fits = deps["fit_tabpfn"].call_args_list
    assert len(fits) == 2
    assert fits[0].args[0].X.columns.tolist() == FEATURES
    assert fits[1].args[0].X.columns.tolist() == ["FEAT_B"]
    assert fits[0].args[0].n_games == 4
    assert fits[1].args[0].n_games == 6
    assert fits[0].args[0].y.abs().max() < 10
    assert fits[1].args[0].y.min() > 200
    if not persist:
        deps["persist_prediction_run"].assert_not_called()
        return
    deps["persist_prediction_run"].assert_called_once()
    call = deps["persist_prediction_run"].call_args
    assert call.kwargs["partial"] is False
    recorder = call.args[0]
    tables = recorder.tables(status=store_schema.RUN_STATUS_OK)
    assert set(tables.model_builds["target"]) == {1, 2}
    assert set(tables.model_builds["kind"]) == {store_schema.KIND_TABPFN}
    assert len(tables.inputs) == 2
    assert len(tables.predictions) == 4
    assert [slot["status"] for slot in recorder.slots] == ["predicted", "predicted"]


@pytest.mark.parametrize("failed_index", [0, 1])
def test_target_failure_preserves_other_predictions_and_fails_run(
    monkeypatch, failed_index
):
    deps = _run_dependencies(monkeypatch)
    fit = deps["fit_tabpfn"].side_effect
    calls = 0

    def failing_fit(*args, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        if index == failed_index:
            raise RuntimeError("inference failed")
        return fit(*args, **kwargs)

    deps["fit_tabpfn"].side_effect = failing_fit
    with pytest.raises(SystemExit) as exc:
        job.predict_tabpfn(targets=["line_error", "total_points"])
    assert exc.value.code == 1
    assert calls == 2
    call = deps["persist_prediction_run"].call_args
    assert call.kwargs["partial"] is True
    recorder = call.args[0]
    tables = recorder.tables(status=store_schema.RUN_STATUS_PARTIAL)
    assert len(tables.predictions) == 2
    assert recorder.slots[failed_index]["status"] == "failed"
    assert recorder.slots[1 - failed_index]["status"] == "predicted"


def test_a_run_fits_the_standard_model_only_by_default(monkeypatch):
    deps = _run_dependencies(monkeypatch)
    job.predict_tabpfn(targets=["line_error", "total_points"])
    modes = [fit.kwargs["mode"] for fit in deps["fit_tabpfn"].call_args_list]
    assert modes == ["standard", "standard"]
    recorder = deps["persist_prediction_run"].call_args.args[0]
    tables = recorder.tables(status=store_schema.RUN_STATUS_OK)
    assert set(tables.model_builds["variant"]) == {"tabpfn_v3_5"}


def test_each_mode_is_its_own_fit_and_variant(monkeypatch):
    """Two modes over two targets: four fits on the targets' own rows, stored
    as four builds of one run."""
    deps = _run_dependencies(monkeypatch)
    job.predict_tabpfn(
        targets=["line_error", "total_points"], modes=["standard", "thinking_high"]
    )
    fits = deps["fit_tabpfn"].call_args_list
    assert [(fit.args[0].n_games, fit.kwargs["mode"]) for fit in fits] == [
        (4, "standard"),
        (4, "thinking_high"),
        (6, "standard"),
        (6, "thinking_high"),
    ]
    recorder = deps["persist_prediction_run"].call_args.args[0]
    tables = recorder.tables(status=store_schema.RUN_STATUS_OK)
    builds = tables.model_builds
    assert len(builds) == 4
    assert sorted(builds["variant"]) == [
        "tabpfn_v3_5",
        "tabpfn_v3_5",
        "tabpfn_v3_5_thinking_high",
        "tabpfn_v3_5_thinking_high",
    ]
    assert len(tables.predictions) == 8
    assert [slot["status"] for slot in recorder.slots] == ["predicted"] * 4


def test_the_cli_passes_modes_and_refuses_an_unknown_one(monkeypatch):
    predict = Mock()
    monkeypatch.setattr(job, "predict_tabpfn", predict)
    job.main([])
    assert predict.call_args.kwargs["modes"] == ["standard"]
    job.main(["--modes", "standard thinking_high"])
    assert predict.call_args.kwargs["modes"] == ["standard", "thinking_high"]
    predict.reset_mock()
    with pytest.raises(SystemExit) as exc:
        job.main(["--modes", "thinking"])
    assert exc.value.code == 2
    predict.assert_not_called()


def test_mixed_closing_schemas_use_one_frame_at_the_newest_version(monkeypatch):
    deps = _run_dependencies(monkeypatch, total_schema="2_6")
    job.predict_tabpfn(targets=["line_error", "total_points"])
    assert deps["resolve_training_frame"].call_count == 2
    first, second = deps["resolve_training_frame"].call_args_list
    assert first.kwargs["base_frames"] is second.kwargs["base_frames"]
    assert first.kwargs["layer_context"] is second.kwargs["layer_context"]
    deps["tonight_closing_rows"].assert_called_once()
    assert deps["tonight_closing_rows"].call_args.kwargs["schema_version"] == "2_6"
    recorder = deps["persist_prediction_run"].call_args.args[0]
    tables = recorder.tables(status=store_schema.RUN_STATUS_OK)
    assert set(tables.model_builds["schema_version"]) == {"2_5", "2_6"}


def _latest_quotes() -> pd.DataFrame:
    """Tonight's latest snapshot rows, with bet365's raw quotes."""
    return _tonight().assign(
        TIME_TO_MATCH_MIN=[47, 197],
        LATEST_INPUT=True,
        ODDS_SNAP_TOT_BET365_HAS_QUOTE=1.0,
        ODDS_SNAP_TOT_BET365_RAW_LINE=[225.0, 231.5],
        ODDS_SNAP_TOT_BET365_PRICE_LEFT=[-115.0, -110.0],
        ODDS_SNAP_TOT_BET365_PRICE_RIGHT=[-105.0, -110.0],
        ODDS_SNAP_TOT_BET365_LINE_AGE_MINUTES=[4.0, 12.0],
    )


def test_a_closing_run_stores_the_bet_line_of_the_latest_snapshot(monkeypatch):
    deps = _run_dependencies(monkeypatch)
    deps["tonight_intermediate_frame"].return_value = _latest_quotes()
    job.predict_tabpfn(targets=["line_error"])
    builder = deps["tonight_intermediate_frame"].call_args
    assert builder.kwargs["latest_input"] and builder.kwargs["latest_only"]
    call = deps["persist_prediction_run"].call_args
    assert call.kwargs["partial"] is False
    recorder = call.args[0]
    inputs = recorder.tables(status=store_schema.RUN_STATUS_OK).inputs
    assert inputs["input_mode"].tolist() == [store_schema.INPUT_MODE_CLOSING] * 2
    assert inputs["total_model_line"].tolist() == [224.5, 230.5]
    assert inputs["total_bet_line_x2"].tolist() == [450, 463]
    assert inputs["total_price_over"].tolist() == [-115, -110]
    assert inputs["total_line_age_minutes"].tolist() == [4.0, 12.0]
    # Only the closing rows were read: no intermediate input is stored.
    assert set(recorder.input_frames()) == {"closing_line"}


def test_a_failed_latest_snapshot_build_only_loses_the_bet_line(monkeypatch):
    deps = _run_dependencies(monkeypatch)
    deps["tonight_intermediate_frame"].side_effect = RuntimeError("no ticks")
    job.predict_tabpfn(targets=["line_error"])
    call = deps["persist_prediction_run"].call_args
    assert call.kwargs["partial"] is False
    tables = call.args[0].tables(status=store_schema.RUN_STATUS_OK)
    assert len(tables.predictions) == 2
    assert tables.inputs["total_bet_line_x2"].isna().all()


@pytest.mark.parametrize("horizon", [720, 480, 360, 240])
def test_scheduled_tabpfn_predicts_every_game_at_its_exact_snapshot_when_queued(
    monkeypatch, horizon
):
    deps = _run_dependencies(monkeypatch, horizon=horizon)
    # The runner is busy until 20:00 UTC. This must not change the selected
    # snapshot, including T-720 several hours earlier.
    monkeypatch.setattr(
        job.datetime,
        "now",
        lambda tz: pd.Timestamp("2026-10-21 20:00", tz="UTC")
        .to_pydatetime()
        .astimezone(tz),
    )
    cutoff = pd.to_datetime(_tonight()["GAME_TIME"], utc=True).iloc[0] - pd.Timedelta(
        minutes=horizon
    )
    job.predict_tabpfn(
        line="intermediate",
        horizon_minutes=horizon,
        game_date="2026-10-21",
        as_of=cutoff.isoformat(),
        update_db=False,
    )
    deps["update_all_databases"].assert_not_called()
    builder = deps["tonight_intermediate_frame"].call_args
    assert builder.kwargs["as_of"] == cutoff
    assert builder.kwargs["horizon"] == horizon
    # Not only the game whose snapshot triggered the run: the whole slate.
    assert builder.args[0]["scheduled_games"]["GAME_ID"].tolist() == [
        "0022600001",
        "0022600002",
    ]
    assert (
        builder.args[0]["roster_snapshots"] is None
    )  # load archived observations at the cutoff
    assert (
        deps["get_all_info_for_scheduled_games"].call_args.kwargs["dataset_type"]
        == "intermediate_line"
    )
    tables = (
        deps["persist_prediction_run"]
        .call_args.args[0]
        .tables(status=store_schema.RUN_STATUS_OK)
    )
    assert len(tables.predictions) == 4
    assert set(tables.model_builds["target"]) == {1, 2}
    assert set(tables.model_builds["horizon_minutes"]) == {horizon}
    assert set(tables.predictions["snapshot_horizon_minutes"]) == {horizon}
    assert set(tables.predictions["game_id"]) == {22600001, 22600002}


def test_explicit_game_date_survives_a_change_of_pacific_date(monkeypatch):
    deps = _run_dependencies(monkeypatch, horizon=720)
    monkeypatch.setattr(
        job.datetime,
        "now",
        lambda tz: pd.Timestamp("2026-10-21 06:30", tz="UTC")
        .to_pydatetime()
        .astimezone(tz),
    )
    job.predict_tabpfn(
        line="intermediate",
        horizon_minutes=720,
        game_date="2026-10-21",
        update_db=False,
    )
    assert (
        deps["get_all_info_for_scheduled_games"].call_args.kwargs["date_to_predict"]
        == "2026-10-21"
    )
    assert (
        deps["tonight_intermediate_frame"].call_args.kwargs["game_date"] == "2026-10-21"
    )


def test_unknown_requested_game_fails_instead_of_predicting_the_whole_slate(
    monkeypatch,
):
    deps = _run_dependencies(monkeypatch, horizon=360)
    with pytest.raises(ValueError, match="missing from the schedule"):
        job.predict_tabpfn(line="intermediate", game_ids=["0022609999"])
    deps["fit_tabpfn"].assert_not_called()


def test_snapshots_far_from_the_cutoff_are_predicted_as_built(monkeypatch):
    # Both games' T-360 (17:30 and 20:00 UTC) are over 15 min from --as-of:
    # neither is the triggering snapshot, so nothing is waited for or rebuilt.
    deps = _run_dependencies(monkeypatch, horizon=360)
    waited = []
    monkeypatch.setattr(tabpfn_run, "_wait_until", waited.append)
    job.predict_tabpfn(
        line="intermediate", as_of="2026-10-21T17:00:00Z", update_db=False
    )
    deps["tonight_intermediate_frame"].assert_called_once()
    assert waited == []
    assert deps["fit_tabpfn"].call_count == 2


@pytest.mark.parametrize("as_of", ["2026-10-21T17:25:00Z", "2026-10-21T17:35:00Z"])
def test_page_tipoff_a_few_minutes_off_rebuilds_at_the_rows_own_snapshot(
    monkeypatch, as_of
):
    # The game's T-360 from the line-history page is 17:30 UTC; the
    # scoreboard's tip-off puts --as-of 5 min before or after it.
    deps = _run_dependencies(monkeypatch, horizon=360)
    snapshot = pd.Timestamp("2026-10-21 17:30", tz="UTC")
    waited = []
    monkeypatch.setattr(tabpfn_run, "_wait_until", waited.append)
    job.predict_tabpfn(line="intermediate", as_of=as_of, update_db=False)
    cutoffs = [
        call.kwargs["as_of"]
        for call in deps["tonight_intermediate_frame"].call_args_list
    ]
    # Built at --as-of, then again at the snapshot the rows are labelled with.
    assert cutoffs == [pd.Timestamp(as_of), snapshot]
    assert waited == [snapshot]
    assert deps["fit_tabpfn"].call_count == 2


def _snapshot_rows(**snapshots: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": list(snapshots),
            "SNAPSHOT_TS_UTC": pd.to_datetime(list(snapshots.values()), utc=True),
        }
    )


@pytest.mark.parametrize("snapshot_games", [["0022600001"], None])
def test_another_games_snapshot_minutes_away_is_not_the_runs(
    monkeypatch, snapshot_games
):
    # Tip-offs 10 min apart: each snapshot has its own run, and neither
    # rebuilds the slate at the other's snapshot.
    rows = _snapshot_rows(
        **{"0022600001": "2026-10-21 18:00", "0022600002": "2026-10-21 18:10"}
    )
    build = Mock()
    monkeypatch.setattr(tabpfn_run, "_wait_until", Mock())
    cutoff = pd.Timestamp("2026-10-21 18:00", tz="UTC")
    got = tabpfn_run._rebuild_at_row_snapshot(rows, cutoff, build, snapshot_games)
    assert got is rows
    build.assert_not_called()


def test_the_snapshot_games_page_tipoff_off_rebuilds_at_their_snapshot(
    monkeypatch,
):
    # The run's game reads 18:05 on the line-history page; another game's
    # snapshot is exactly at the cutoff, and is not the run's.
    rows = _snapshot_rows(
        **{"0022600001": "2026-10-21 18:05", "0022600002": "2026-10-21 18:00"}
    )
    build = Mock(return_value="rebuilt")
    waited = []
    monkeypatch.setattr(tabpfn_run, "_wait_until", waited.append)
    cutoff = pd.Timestamp("2026-10-21 18:00", tz="UTC")
    got = tabpfn_run._rebuild_at_row_snapshot(rows, cutoff, build, ["0022600001"])
    snapshot = pd.Timestamp("2026-10-21 18:05", tz="UTC")
    assert got == "rebuilt"
    build.assert_called_once_with(snapshot)
    assert waited == [snapshot]


def test_wait_until_sleeps_only_for_a_snapshot_still_ahead(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tabpfn_run.time, "sleep", sleeps.append)
    now = pd.Timestamp.now(tz="UTC")
    tabpfn_run._wait_until(now - pd.Timedelta(minutes=1))
    assert sleeps == []
    tabpfn_run._wait_until(now + pd.Timedelta(minutes=5))
    assert len(sleeps) == 1 and 290 < sleeps[0] <= 300


def test_cli_passes_scheduled_game_and_snapshot_inputs(monkeypatch):
    predict = Mock()
    monkeypatch.setattr(job, "predict_tabpfn", predict)
    job.main(
        [
            "--line",
            "intermediate",
            "--horizon-minutes",
            "720",
            "--game-date",
            "2026-10-21",
            "--game-ids",
            "0022600001,0022600002",
            "--as-of",
            "2026-10-21T11:30:00Z",
            "--snapshot-game-ids",
            "0022600001",
            "--skip-db-update",
        ]
    )
    options = predict.call_args.kwargs
    assert options["game_date"] == "2026-10-21"
    assert options["game_ids"] == ["0022600001", "0022600002"]
    assert options["as_of"] == "2026-10-21T11:30:00Z"
    assert options["snapshot_game_ids"] == ["0022600001"]
    assert options["horizon_minutes"] == 720
    assert options["update_db"] is False


@pytest.mark.parametrize(
    "args",
    [
        ["--game-date", "2026-99-21"],
        ["--game-ids", "22600001"],
        ["--game-ids", " , "],
        ["--line", "intermediate", "--as-of", "2026-10-21 12:00"],
        ["--line", "intermediate", "--as-of", "NaT"],
        ["--as-of", "2026-10-21T12:00:00Z"],
        ["--line", "intermediate", "--snapshot-game-ids", "0022600001"],
        [
            "--line",
            "intermediate",
            "--as-of",
            "2026-10-21T12:00:00Z",
            "--snapshot-game-ids",
            "22600001",
        ],
    ],
)
def test_invalid_schedule_inputs_fail_before_starting(monkeypatch, args):
    predict = Mock()
    monkeypatch.setattr(job, "predict_tabpfn", predict)
    with pytest.raises(SystemExit) as exc:
        job.main(args)
    assert exc.value.code == 2
    predict.assert_not_called()


# --- jobs carried by the daily predictor (--tabpfn) --------------------------


def test_a_scheduled_job_reads_as_its_snapshot_and_games():
    from nba_ou.scheduling.tabpfn_jobs import format_job

    at = datetime(2026, 10, 28, 23, 0, tzinfo=ZoneInfo("UTC"))
    text = ";".join(
        [
            format_job(720, at, ["0022600132", "0022600131"]),
            format_job(480, at, ["0022600133"]),
        ]
    )
    jobs = tabpfn_run.parse_jobs(text, modes=["standard", "thinking_high"])
    assert [(job_.horizon, job_.snapshot_games) for job_ in jobs] == [
        (720, ("0022600131", "0022600132")),
        (480, ("0022600133",)),
    ]
    for parsed in jobs:
        assert parsed.dataset_type == "intermediate_line"
        assert parsed.as_of == pd.Timestamp(at)
        assert parsed.targets == ("line_error", "total_points")
        assert parsed.modes == ("standard", "thinking_high")
        assert parsed.name == f"TabPFN t{parsed.horizon:04d}"


def test_a_job_without_cutoff_is_live_and_horizon_zero_is_the_closing_line():
    live, closing = tabpfn_run.parse_jobs(" 360 ; 0 ;", targets=["line_error"])
    assert (live.horizon, live.as_of, live.snapshot_games) == (360, None, None)
    assert (closing.dataset_type, closing.horizon) == ("closing_line", 0)
    assert tabpfn_run.parse_jobs("") == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("t720", "not a number"),
        ("-60", "negative"),
        ("0@2026-10-28T23:00:00+00:00", "--line intermediate"),
        ("720@2026-10-28T23:00:00", "timezone"),
        ("720@2026-10-28T23:00:00+00:00@123", "10-digit"),
    ],
)
def test_a_malformed_job_is_refused(text, message):
    with pytest.raises(ValueError, match=message):
        tabpfn_run.parse_jobs(text)
