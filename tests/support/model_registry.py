"""Specs and builds shared by registry and archival tests."""

from datetime import datetime, timedelta

from nba_ou.modeling.registry_models import (
    FitRecord,
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecTraining,
    build_model_name,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import ModelTarget, build_fit_id
from nba_ou.modeling.registry_store import set_build_channel, write_build, write_spec

BUCKET = "test-registry"


def make_spec(*, max_na_per_row: int = 500, features=("A", "B", "C")):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version="2_5",
            target=ModelTarget.LINE_ERROR,
            prediction_strategy="line_error_regressor",
            horizon_minutes=60,
            variant="main",
            dataset_type="intermediate_line",
        ),
        features=SpecFeatures(feature_names=list(features), n_features=len(features)),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=max_na_per_row),
        training=SpecTraining(params={"max_depth": 4}, n_estimators=82),
    )


def make_fit(spec, *, fitted_at: datetime, n_features: int | None = None):
    return FitRecord(
        fit_id=build_fit_id(spec.spec_id, fitted_at=fitted_at),
        spec_id=spec.spec_id,
        model_name=build_model_name(
            target=ModelTarget.LINE_ERROR,
            horizon_minutes=60,
            variant="main",
            schema_version="2_5",
            train_date_max=fitted_at,
        ),
        train_date_min=fitted_at - timedelta(days=900),
        train_date_max=fitted_at,
        n_train_games=3500,
        n_features=n_features or spec.features.n_features,
        fitted_at=fitted_at,
    )


def publish(s3, slot, spec, *, fitted_at, model_bytes=b"booster", channel=None):
    """Write a spec and a build, optionally pointing a channel at it."""
    write_spec(s3_client=s3, bucket=BUCKET, slot=slot, spec=spec)
    fit = make_fit(spec, fitted_at=fitted_at)
    write_build(
        s3_client=s3,
        bucket=BUCKET,
        slot=slot,
        fit=fit,
        model_bytes=model_bytes,
        spec=spec,
    )
    if channel is not None:
        set_build_channel(
            s3_client=s3, bucket=BUCKET, slot=slot, channel=channel, fit_id=fit.fit_id
        )
    return fit
