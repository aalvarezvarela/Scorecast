"""Intermediate slots predict tonight's snapshot row of their own horizon, passed or not."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.prediction import prediction
from nba_ou.prediction.intermediate_frame import mark_snapshots

TIP = pd.Timestamp("2026-04-10 23:00", tz="UTC")


def _rows(minutes=(30, 60, 120, 360)):
    games = ["g1", "g2"]
    frame = pd.DataFrame(
        [(g, m) for g in games for m in minutes],
        columns=["GAME_ID", "TIME_TO_MATCH_MIN"],
    )
    scoring = frame.assign(
        TIPOFF_UTC=TIP,
        SNAPSHOT_TS_UTC=TIP - pd.to_timedelta(frame["TIME_TO_MATCH_MIN"], unit="m"),
    )
    return frame, scoring


def test_every_snapshot_is_kept_and_marked_passed_or_not():
    frame, scoring = _rows()
    marked = mark_snapshots(frame, scoring, TIP - pd.Timedelta(minutes=60))
    assert len(marked) == len(frame)
    passed = marked.loc[marked["SNAPSHOT_PASSED"], "TIME_TO_MATCH_MIN"]
    assert sorted(passed.unique()) == [60, 120, 360]
    assert (
        marked["SNAPSHOT_TS_UTC"]
        == TIP - pd.to_timedelta(marked["TIME_TO_MATCH_MIN"], unit="m")
    ).all()
    # Matched on (game, snapshot), not by position.
    shuffled = mark_snapshots(frame, scoring.iloc[::-1], TIP - pd.Timedelta(minutes=60))
    pd.testing.assert_frame_equal(shuffled, marked)
    with pytest.raises(ValueError, match="lacks"):
        mark_snapshots(frame, scoring.iloc[1:], TIP)


def test_an_intermediate_slot_reads_the_rows_of_its_horizon(monkeypatch):
    import nba_ou.modeling.registry_store as store

    class Resolved:
        spec = SimpleNamespace(
            spec_id="s",
            identity=SimpleNamespace(
                dataset_type="intermediate_line",
                horizon_minutes=120,
                target="line_error",
                schema_version="2_5",
                variant="main",
            ),
            features=SimpleNamespace(feature_names=["x"]),
            cleaning=SimpleNamespace(max_na_per_row=800),
            provenance=SimpleNamespace(training_version="v"),
        )
        fit = SimpleNamespace(
            fit_id="f",
            model_name="m",
            train_date_min=pd.Timestamp("2025-01-01"),
            train_date_max=pd.Timestamp("2026-01-01"),
        )
        model_bytes = b""

    monkeypatch.setattr(store, "resolve_channel", lambda **_: Resolved())
    monkeypatch.setattr(prediction.XGBRegressor, "load_model", lambda self, raw: None)
    seen = {}

    def predict(*, df, **kwargs):
        seen["rows"] = df
        return df

    monkeypatch.setattr(prediction, "load_and_predict_model_for_nba_games", predict)
    frame, _ = _rows()
    slot = ModelSlot(
        schema_version="2_5", target="line_error", horizon_minutes=120, variant="main"
    )
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=slot,
        df=None,
        frames={"closing_line": pd.DataFrame(), "intermediate_line": frame},
    )
    assert seen["rows"]["TIME_TO_MATCH_MIN"].tolist() == [120, 120]
    assert seen["rows"]["GAME_ID"].tolist() == ["g1", "g2"]


def _serve(monkeypatch, *, dataset_type, horizon, schema="2_5"):
    """Point the registry at a stub spec; return what the predictor was given."""
    import nba_ou.modeling.registry_store as store

    class Resolved:
        spec = SimpleNamespace(
            spec_id="s",
            identity=SimpleNamespace(
                dataset_type=dataset_type,
                horizon_minutes=horizon,
                target="line_error",
                schema_version=schema,
                variant="main",
            ),
            features=SimpleNamespace(feature_names=["x"]),
            cleaning=SimpleNamespace(max_na_per_row=800),
            provenance=SimpleNamespace(training_version="v"),
        )
        fit = SimpleNamespace(
            fit_id="f",
            model_name="m",
            train_date_min=pd.Timestamp("2025-01-01"),
            train_date_max=pd.Timestamp("2026-01-01"),
        )
        model_bytes = b""

    monkeypatch.setattr(store, "resolve_channel", lambda **_: Resolved())
    monkeypatch.setattr(prediction.XGBRegressor, "load_model", lambda self, raw: None)
    seen = {}

    def predict(*, df, **kwargs):
        seen.update(kwargs, rows=df)
        return df

    monkeypatch.setattr(prediction, "load_and_predict_model_for_nba_games", predict)
    return seen


def _slot(horizon, schema="2_5"):
    return ModelSlot(
        schema_version=schema,
        target="line_error",
        horizon_minutes=horizon,
        variant="main",
    )


def test_a_snapshot_still_ahead_is_predicted_and_marked(monkeypatch):
    seen = _serve(monkeypatch, dataset_type="intermediate_line", horizon=30)
    frame, scoring = _rows()
    # Two hours before tip: T-30 has not happened yet.
    marked = mark_snapshots(frame, scoring, TIP - pd.Timedelta(minutes=120))
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=_slot(30),
        df=None,
        frames={"intermediate_line": marked},
        passthrough_columns=("SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED"),
    )
    assert seen["rows"]["TIME_TO_MATCH_MIN"].tolist() == [30, 30]
    assert not seen["rows"]["SNAPSHOT_PASSED"].any()
    assert seen["passthrough_columns"] == ("SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED")
    # The model's own training limit, from its spec.
    assert seen["max_na_per_row"] == 800


def test_a_closing_slot_predicts_every_game_whatever_the_time_to_tip(monkeypatch):
    seen = _serve(monkeypatch, dataset_type="closing_line", horizon=0)
    closing = pd.DataFrame(
        {"GAME_ID": ["early", "late"], "TIME_TO_MATCH_MINUTES": [600, 20]}
    )
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=_slot(0),
        df=None,
        frames={"closing_line": closing},
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["early", "late"]

    # The guard is still there for a caller that asks for it.
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=_slot(0),
        df=None,
        frames={"closing_line": closing},
        horizon_tolerance_minutes=45,
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["late"]


def test_a_slot_newer_than_its_frame_fails_by_name(monkeypatch):
    _serve(monkeypatch, dataset_type="intermediate_line", horizon=60, schema="2_6")
    frame, _ = _rows()
    with pytest.raises(ValueError, match="2_6 model.*built at 2_5"):
        prediction.load_registry_model_and_predict(
            s3_client=None,
            bucket="b",
            slot=_slot(60, schema="2_6"),
            df=None,
            frames={"closing_line": pd.DataFrame(), "intermediate_line": frame},
            frame_versions={"closing_line": "2_6", "intermediate_line": "2_5"},
        )


def test_clean_frames_rows_are_chosen_on_the_frame_and_read_from_the_clean_one(
    monkeypatch,
):
    """The slot picks its rows on the frame, then reads them from the frame
    cleaned once; a row cleaning dropped is not predicted."""
    seen = _serve(monkeypatch, dataset_type="intermediate_line", horizon=30)
    frame, scoring = _rows()
    marked = mark_snapshots(frame, scoring, TIP - pd.Timedelta(minutes=120))
    clean = marked.assign(CLEANED=True).drop(index=marked.index[4])
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=_slot(30),
        df=None,
        frames={"intermediate_line": marked},
        clean_frames={"intermediate_line": clean},
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["g1"]
    assert seen["rows"]["CLEANED"].all()
    assert seen["cleaned"] is True


def test_without_clean_frames_the_slot_cleans_its_own_rows(monkeypatch):
    seen = _serve(monkeypatch, dataset_type="intermediate_line", horizon=30)
    frame, scoring = _rows()
    marked = mark_snapshots(frame, scoring, TIP - pd.Timedelta(minutes=120))
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=_slot(30),
        df=None,
        frames={"intermediate_line": marked},
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["g1", "g2"]
    assert seen["cleaned"] is False


def test_builds_are_read_side_by_side_and_a_failed_read_is_kept_for_its_slot(
    monkeypatch,
):
    import nba_ou.modeling.registry_store as store

    def resolve(*, slot, **_):
        if slot.horizon_minutes == 60:
            raise ConnectionError("S3 timed out")
        return f"build of T-{slot.horizon_minutes}"

    monkeypatch.setattr(store, "resolve_channel", resolve)
    slots = [_slot(30), _slot(60), _slot(120)]
    builds = prediction.read_slot_builds(s3_client=None, bucket="b", slots=slots)
    assert list(builds) == slots
    assert builds[slots[0]] == "build of T-30"
    assert isinstance(builds[slots[1]], ConnectionError)
    assert builds[slots[2]] == "build of T-120"


def test_a_build_read_beforehand_is_not_read_again(monkeypatch):
    import nba_ou.modeling.registry_store as store

    seen = _serve(monkeypatch, dataset_type="intermediate_line", horizon=30)
    build = store.resolve_channel()
    monkeypatch.setattr(
        store,
        "resolve_channel",
        lambda **_: pytest.fail("the build was read again"),
    )
    frame, scoring = _rows()
    marked = mark_snapshots(frame, scoring, TIP - pd.Timedelta(minutes=120))
    prediction.load_registry_model_and_predict(
        s3_client=None,
        bucket="b",
        slot=_slot(30),
        df=None,
        frames={"intermediate_line": marked},
        resolved=build,
    )
    assert seen["rows"]["GAME_ID"].tolist() == ["g1", "g2"]
