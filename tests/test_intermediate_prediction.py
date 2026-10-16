"""Intermediate slots predict tonight's passed snapshot of their own horizon."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.prediction import prediction
from nba_ou.prediction.intermediate_frame import passed_snapshots

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


def test_only_snapshots_already_passed_are_kept():
    frame, scoring = _rows()
    kept = passed_snapshots(frame, scoring, TIP - pd.Timedelta(minutes=60))
    assert sorted(kept["TIME_TO_MATCH_MIN"].unique()) == [60, 120, 360]
    # Matched on (game, snapshot), not by position.
    shuffled = passed_snapshots(
        frame, scoring.iloc[::-1], TIP - pd.Timedelta(minutes=60)
    )
    pd.testing.assert_frame_equal(shuffled, kept)
    with pytest.raises(ValueError, match="lacks"):
        passed_snapshots(frame, scoring.iloc[1:], TIP)


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
