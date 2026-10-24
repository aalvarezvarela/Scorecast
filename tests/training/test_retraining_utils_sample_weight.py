from unittest.mock import patch

import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.modeling.modeling import (
    ModelBundleMetadata,
    ModelInfo,
    SchemaInfo,
    TrainingMetrics,
)
from nba_ou.modeling.retraining_utils import (
    ProductionArtifacts,
    build_retraining_settings_from_artifacts,
    prepare_retraining_dataframe_from_raw,
    retrain_model,
)


class _UnusedModel:
    pass


def _make_artifacts(
    *,
    sample_weight_lambda: float | None,
    sample_weight_lambda_bounds: tuple[float, float] | None,
    feature_names: list[str] | None = None,
) -> ProductionArtifacts:
    feature_names = feature_names or [total_line_col(), "PACE_LAST_10"]
    training_metrics = TrainingMetrics(
        best_params={
            "max_depth": 3,
            "learning_rate": 0.05,
            **(
                {}
                if sample_weight_lambda is None
                else {"sample_weight_lambda": sample_weight_lambda}
            ),
        },
        selected_trial_number=4,
        mean_best_iteration=125,
        median_best_iteration=120,
        train_games=4000,
        sample_weight_lambda_bounds=sample_weight_lambda_bounds,
        cv_mae=13.2,
        cv_rmse=17.5,
        cv_ou_acc=0.53,
        final_test_mae=13.1,
        final_test_rmse=17.2,
        final_test_ou_acc=0.54,
        nan_threshold=0.8,
        max_na_per_row=5,
        train_date_min=pd.Timestamp("2024-01-01").to_pydatetime(),
        train_date_max=pd.Timestamp("2025-01-01").to_pydatetime(),
    )
    metadata = ModelBundleMetadata(
        model_info=ModelInfo(
            name="five_seasons_xgb_line_error_01_01_25",
            model_version="01_01_25",
            model_type="five_seasons_error_line",
            prediction_source="five_seasons_xgb_line_error",
            training_code_tag="1.0",
        ),
        schema_info=SchemaInfo(
            feature_names=feature_names,
            n_features=len(feature_names),
        ),
        training_metrics=training_metrics,
    )
    raw_metadata = metadata.model_dump(by_alias=True, mode="json")

    return ProductionArtifacts(
        bucket="test-bucket",
        production_prefix="models/line_error_last_5_seasons/production/",
        model_key="models/line_error_last_5_seasons/production/model.json",
        meta_key="models/line_error_last_5_seasons/production/model.meta.json",
        model=_UnusedModel(),
        raw_metadata=raw_metadata,
        metadata=metadata,
    )


def test_build_retraining_settings_extracts_sample_weight_hyperparameters() -> None:
    artifacts = _make_artifacts(
        sample_weight_lambda=0.004,
        sample_weight_lambda_bounds=(1e-4, 0.01),
    )

    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column="GAME_DATE",
        minimum_line_value=100.0,
        xgb_static_params={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": 16,
        },
    )

    assert settings.target_column == "LINE_ERROR"
    assert settings.sample_weight_lambda == 0.004
    assert settings.sample_weight_lambda_bounds == (1e-4, 0.01)
    assert settings.xgb_params["n_estimators"] == 120
    assert "sample_weight_lambda" not in settings.xgb_params


def test_retrain_model_passes_recency_weights_only_at_fit_time() -> None:
    artifacts = _make_artifacts(
        sample_weight_lambda=0.01,
        sample_weight_lambda_bounds=(1e-4, 0.01),
    )
    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column="GAME_DATE",
        minimum_line_value=100.0,
        xgb_static_params={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": 16,
        },
    )
    training_window_df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]),
            total_line_col(): [220.5, 221.0, 219.5],
            "PACE_LAST_10": [99.1, 100.2, 98.7],
            "LINE_ERROR": [1.2, -0.8, 0.3],
        }
    )
    fit_calls: list[tuple[pd.DataFrame, pd.Series, dict]] = []

    class _FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, **kwargs):
            fit_calls.append((X.copy(), y.copy(), kwargs))
            return self

    with patch("nba_ou.modeling.retraining_utils.XGBRegressor", _FakeRegressor):
        model = retrain_model(training_window_df, settings=settings)

    assert isinstance(model, _FakeRegressor)
    assert "sample_weight_lambda" not in model.kwargs
    assert len(fit_calls) == 1
    _, _, fit_kwargs = fit_calls[0]
    assert fit_kwargs["verbose"] is False
    assert "sample_weight" in fit_kwargs
    assert len(fit_kwargs["sample_weight"]) == len(training_window_df)


def test_prepare_retraining_derives_line_error_from_total_points() -> None:
    """LINE_ERROR is computed as TOTAL_POINTS - required_line_col when missing."""
    artifacts = _make_artifacts(
        sample_weight_lambda=0.004,
        sample_weight_lambda_bounds=(1e-4, 0.01),
    )
    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column="GAME_DATE",
        minimum_line_value=None,
        xgb_static_params={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": 16,
        },
    )

    assert settings.target_column == "LINE_ERROR"

    raw_df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]),
            total_line_col(): [220.5, 221.0, 219.5],
            "PACE_LAST_10": [99.1, 100.2, 98.7],
            "TOTAL_POINTS": [222.0, 219.0, 220.0],
        }
    )

    prepared = prepare_retraining_dataframe_from_raw(raw_df, settings=settings)

    assert "LINE_ERROR" in prepared.columns
    expected = raw_df["TOTAL_POINTS"] - raw_df[total_line_col()]
    assert list(prepared["LINE_ERROR"]) == list(expected)


def test_prepare_retraining_uses_existing_line_error_column() -> None:
    """When LINE_ERROR already exists in raw_df it is used directly."""
    artifacts = _make_artifacts(
        sample_weight_lambda=0.004,
        sample_weight_lambda_bounds=(1e-4, 0.01),
    )
    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column="GAME_DATE",
        minimum_line_value=None,
        xgb_static_params={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": 16,
        },
    )

    raw_df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]),
            total_line_col(): [220.5, 221.0, 219.5],
            "PACE_LAST_10": [99.1, 100.2, 98.7],
            "TOTAL_POINTS": [222.0, 219.0, 220.0],
            "LINE_ERROR": [10.0, 20.0, 30.0],
        }
    )

    prepared = prepare_retraining_dataframe_from_raw(raw_df, settings=settings)

    assert list(prepared["LINE_ERROR"]) == [10.0, 20.0, 30.0]


def test_prepare_retraining_coerces_object_features_to_numeric() -> None:
    artifacts = _make_artifacts(
        sample_weight_lambda=0.004,
        sample_weight_lambda_bounds=(1e-4, 0.01),
        feature_names=[total_line_col(), "IS_US_HOLIDAY_BEFORE"],
    )
    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column="GAME_DATE",
        minimum_line_value=None,
        xgb_static_params={
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "random_state": 16,
        },
    )

    raw_df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]),
            total_line_col(): [220.5, 221.0, 219.5],
            "IS_US_HOLIDAY_BEFORE": ["0", "1", None],
            "LINE_ERROR": [10.0, 20.0, 30.0],
        }
    )

    prepared = prepare_retraining_dataframe_from_raw(raw_df, settings=settings)

    assert pd.api.types.is_numeric_dtype(prepared["IS_US_HOLIDAY_BEFORE"])
    assert prepared["IS_US_HOLIDAY_BEFORE"].iloc[0] == 0.0
    assert prepared["IS_US_HOLIDAY_BEFORE"].iloc[1] == 1.0
    assert pd.isna(prepared["IS_US_HOLIDAY_BEFORE"].iloc[2])
