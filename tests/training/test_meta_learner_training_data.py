import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.modeling.meta_learner_training_data import (
    build_feature_union_from_artifacts,
    build_meta_learner_base_frame,
    derive_prediction_column_name,
    select_latest_seasons,
)
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
)


class _UnusedModel:
    pass


class _ImportanceModel:
    def __init__(self, importances: list[float]) -> None:
        self.feature_importances_ = np.asarray(importances, dtype=float)


def _make_artifacts(
    *,
    production_prefix: str = "models/line_error_last_5_seasons/production/",
    feature_names: list[str] | None = None,
    importances: list[float] | None = None,
) -> ProductionArtifacts:
    resolved_feature_names = feature_names or [total_line_col(), "PACE_LAST_10"]
    training_metrics = TrainingMetrics(
        best_params={
            "max_depth": 3,
            "learning_rate": 0.05,
            "sample_weight_lambda": 0.004,
        },
        selected_trial_number=4,
        mean_best_iteration=125,
        median_best_iteration=120,
        train_games=4000,
        sample_weight_lambda_bounds=(1e-4, 0.01),
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
            feature_names=resolved_feature_names,
            n_features=len(resolved_feature_names),
        ),
        training_metrics=training_metrics,
    )
    raw_metadata = metadata.model_dump(by_alias=True, mode="json")

    return ProductionArtifacts(
        bucket="test-bucket",
        production_prefix=production_prefix,
        model_key=f"{production_prefix}model.json",
        meta_key=f"{production_prefix}model.meta.json",
        model=_UnusedModel()
        if importances is None
        else _ImportanceModel(importances=importances),
        raw_metadata=raw_metadata,
        metadata=metadata,
    )


def test_prepare_retraining_preserves_passthrough_columns() -> None:
    artifacts = _make_artifacts()
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
            "GAME_ID": ["1", "2", "3"],
            "SEASON_YEAR": [2023, 2023, 2024],
            "GAME_DATE": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"]),
            total_line_col(): [220.5, 221.0, 219.5],
            "PACE_LAST_10": [99.1, 100.2, 98.7],
            "TOTAL_POINTS": [222.0, 219.0, 220.0],
        }
    )

    prepared = prepare_retraining_dataframe_from_raw(
        raw_df,
        settings=settings,
        passthrough_columns=["GAME_ID", "SEASON_YEAR"],
    )

    assert list(prepared["GAME_ID"]) == ["1", "2", "3"]
    assert list(prepared["SEASON_YEAR"]) == [2023, 2023, 2024]


def test_select_latest_seasons_handles_int_and_string_labels() -> None:
    df = pd.DataFrame({"SEASON_YEAR": [2022, "2023-24", 2021, "2024-25", 2023]})

    selected = select_latest_seasons(df, n_seasons=2)

    assert selected == [2023, "2024-25"]


def test_build_feature_union_deduplicates_in_prefix_order() -> None:
    first_artifacts = _make_artifacts(
        production_prefix="models/total_points_full_dataset/production/",
        feature_names=["A", "B", "C"],
        importances=[0.9, 0.8, 0.1],
    )
    second_artifacts = _make_artifacts(
        production_prefix="models/line_error_last_3_seasons/production/",
        feature_names=["B", "D", "E"],
        importances=[0.95, 0.7, 0.6],
    )

    combined, per_model = build_feature_union_from_artifacts(
        [first_artifacts, second_artifacts],
        top_n_per_model=2,
    )

    assert combined == ["A", "B", "D"]
    assert per_model[first_artifacts.production_prefix] == ["A", "B"]
    assert per_model[second_artifacts.production_prefix] == ["B", "D"]
    assert (
        derive_prediction_column_name("models/line_error_last_3_seasons/production/")
        == "PRED_LINE_ERROR_LAST_3_SEASONS"
    )


def test_build_meta_learner_base_frame_adds_line_error_and_keeps_selected_columns() -> None:
    raw_df = pd.DataFrame(
        {
            "GAME_ID": ["10", "11", "12"],
            "GAME_DATE": pd.to_datetime(["2024-11-01", "2025-01-10", "2025-01-12"]),
            "SEASON_YEAR": [2023, 2024, 2024],
            "TEAM_ID_TEAM_HOME": ["100", "101", "102"],
            "TEAM_ID_TEAM_AWAY": ["200", "201", "202"],
            total_line_col(): [221.5, 225.5, 219.5],
            "TOTAL_POINTS": [223.0, 220.0, 218.0],
            "PACE_LAST_10": [99.0, 100.0, 101.0],
        }
    )

    base = build_meta_learner_base_frame(
        raw_df,
        prediction_seasons=[2024],
        selected_feature_names=["PACE_LAST_10"],
        merge_keys=["GAME_ID"],
        passthrough_columns=[
            "GAME_ID",
            "GAME_DATE",
            "SEASON_YEAR",
            "TOTAL_POINTS",
            total_line_col(),
            "LINE_ERROR",
        ],
    )

    assert list(base["GAME_ID"]) == ["11", "12"]
    assert "LINE_ERROR" in base.columns
    assert list(base["LINE_ERROR"]) == [-5.5, -1.5]
    assert "PACE_LAST_10" in base.columns
