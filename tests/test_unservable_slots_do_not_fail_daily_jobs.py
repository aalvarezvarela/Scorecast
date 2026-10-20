"""An intermediate slot must not break the daily jobs for every other slot.

Until the daily intermediate dataset and live snapshot features exist, an
intermediate-line slot can neither be refitted nor served. Both daily jobs used
to treat that as a failure: the retrain job exited 1, so the workflow's
``if: success()`` smoke-test and promote steps skipped the closing models too,
and the prediction job re-raised the missing-feature error, killing every slot
after it. Both now skip such a slot and carry on.
"""

import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.modeling.refit import TrainingFrameUnavailable

import scripts.retrain_prediction_models as retrain

CLOSING = "2_5/line_error/t0000/main"
INTERMEDIATE = "2_5/line_error/t0060/main"


def _spec(dataset_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        spec_id=f"spec-{dataset_type}",
        identity=SimpleNamespace(dataset_type=dataset_type, horizon_minutes=0),
    )


@pytest.fixture
def daily_retrain(monkeypatch):
    """retrain.main() with S3 and fitting stubbed; returns what got staged."""
    staged: list[str] = []
    refit_fails: set[str] = set()

    monkeypatch.setattr(retrain, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(
        retrain,
        "resolve_config_spec",
        lambda *, slot, **_: _spec(
            "closing_line" if slot.horizon_minutes == 0 else "intermediate_line"
        ),
    )

    def frame_for(spec, **_):
        if spec.identity.dataset_type != "closing_line":
            raise TrainingFrameUnavailable("no daily intermediate build yet")
        return SimpleNamespace(df=pd.DataFrame(), build_id=None, checksum=None)

    def refit(spec, *, slot, **_):
        if slot.describe() in refit_fails:
            raise RuntimeError("refit exploded")
        return SimpleNamespace(
            model=object(),
            fit=SimpleNamespace(
                fit_id=f"fit-{slot.horizon_minutes}",
                model_name="m",
                n_train_games=6200,
                train_date_min=datetime(2021, 5, 1),
                train_date_max=datetime(2026, 4, 17),
            ),
        )

    monkeypatch.setattr(retrain, "resolve_training_frame", frame_for)
    monkeypatch.setattr(retrain, "refit_from_spec", refit)
    monkeypatch.setattr(retrain, "model_bytes_of", lambda model: b"")
    monkeypatch.setattr(retrain, "write_build", lambda **_: None)
    monkeypatch.setattr(
        retrain,
        "set_build_channel",
        lambda *, slot, **_: staged.append(slot.describe()),
    )

    def run(*slots: str) -> int:
        argv = ["retrain_prediction_models.py"]
        for slot in slots:
            argv += ["--slot", slot]
        monkeypatch.setattr(sys, "argv", argv)
        return retrain.main()

    run.staged = staged
    run.refit_fails = refit_fails
    return run


def test_an_unrefittable_intermediate_slot_leaves_the_job_green(daily_retrain, capsys):
    exit_code = daily_retrain(CLOSING, INTERMEDIATE)

    assert exit_code == 0  # so the workflow's smoke test and promote steps run
    assert daily_retrain.staged == [CLOSING]
    out = capsys.readouterr()
    assert "keeping the current build" in out.err
    assert "1 slot(s) kept on their current build" in out.out


def test_a_real_refit_failure_still_fails_the_job(daily_retrain):
    daily_retrain.refit_fails.add(CLOSING)

    assert daily_retrain(CLOSING, INTERMEDIATE) == 1


def test_an_intermediate_slot_is_skipped_by_the_prediction_job(monkeypatch):
    """Skipped before its booster is even read, and as an empty frame -- the
    signal predict_nba_games already treats as "not this model's turn"."""
    import nba_ou.modeling.registry_store as store
    from nba_ou.modeling.registry_paths import ModelSlot
    from nba_ou.prediction.prediction import load_registry_model_and_predict

    class Resolved:
        spec = _spec("intermediate_line")
        fit = SimpleNamespace(fit_id="f")

        @property
        def model_bytes(self):
            raise AssertionError("an unservable model must not be loaded")

    monkeypatch.setattr(store, "resolve_channel", lambda **_: Resolved())
    slot = ModelSlot(
        schema_version="2_5", target="line_error", horizon_minutes=60, variant="main"
    )

    result = load_registry_model_and_predict(
        s3_client=None, bucket="b", slot=slot, df=pd.DataFrame({"x": [1]})
    )

    assert result.empty
