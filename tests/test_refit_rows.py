"""The daily refit trains on the rows training would have used.

The parity tests run training_pipeline's own prepare_dataset next to
nba_ou.modeling.refit.prepare_refit_rows on one frame, for every target, so a
row rule added to one and not the other fails here rather than silently
changing what production is refitted on.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest
from nba_ou.modeling.refit import (
    DEFAULT_RANDOM_STATE,
    TARGET_COLUMNS,
    RefitError,
    prepare_refit_rows,
    select_training_window,
    settings_from_spec,
)
from nba_ou.modeling.registry_models import (
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecTraining,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget

from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    PredictionStrategy,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.data import (
    build_feature_matrix,
    load_raw_training_csv,
    prepare_dataset,
    rolling_window_index,
)
from training_pipeline.registry import (
    PromotionError,
    build_spec_from_run,
    check_refit_reproduces_rows,
)
from training_pipeline.reuse import RunHyperparameters

TOTAL_LINE = "ODDS_TOTAL_LINE_bet365"
SPREAD_LINE = "ODDS_SPREAD_LINE_HOME_bet365"
FEATURES = [f"FEATURE_{i}_BEFORE" for i in range(6)]
TRAIN_GAMES = 60


@pytest.fixture
def frame_path(tmp_path):
    """Two games a day, with one of every row training drops."""
    rng = np.random.default_rng(0)
    n = 140
    line = rng.uniform(200, 240, n).round(1)
    spread = rng.uniform(-10, 10, n).round(1)
    margin = (spread + rng.normal(0, 12, n)).round()
    df = pd.DataFrame(
        {
            "GAME_ID": [f"00225{i:05d}" for i in range(n)],
            "GAME_DATE": np.repeat(pd.date_range("2025-11-01", periods=n // 2), 2),
            "SEASON_YEAR": 2025,
            "TOTAL_POINTS": (line + rng.normal(0, 12, n)).round(),
            "HOME_MARGIN": margin,
            "PTS_TEAM_HOME": 110.0,
            "PTS_TEAM_AWAY": 110.0 - margin,
            TOTAL_LINE: line,
            SPREAD_LINE: spread,
            **{name: rng.normal(size=n) for name in FEATURES},
        }
    )
    df.loc[[5, 6], "GAME_ID"] = ["0042500101", "0042500102"]  # playoffs
    df.loc[7, "GAME_ID"] = "0012500001"  # preseason
    df.loc[8, "GAME_ID"] = "0052500001"  # play-in: kept
    df.loc[10, TOTAL_LINE] = np.nan  # no main total line, for every target
    df.loc[11, TOTAL_LINE] = 90.0  # implausible line
    df.loc[12, "TOTAL_POINTS"] = 120.0  # implausible total
    df.loc[13, SPREAD_LINE] = np.nan  # no spread: spread target only
    df.loc[14, FEATURES[:3]] = np.nan  # over the per-row NaN budget
    df.loc[n - 1, ["TOTAL_POINTS", "HOME_MARGIN"]] = np.nan  # not played yet
    path = tmp_path / "closing.csv"
    df.to_csv(path, index=False)
    return path


STRATEGIES = {
    TargetFamily.LINE_ERROR: PredictionStrategy.LINE_ERROR_REGRESSOR,
    TargetFamily.TOTAL_POINTS: PredictionStrategy.TOTAL_POINTS_REGRESSOR,
    TargetFamily.SPREAD_ERROR: PredictionStrategy.SPREAD_ERROR_REGRESSOR,
}


def _config(path, family: TargetFamily) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="parity",
        training_version="test",
        prediction_strategy=STRATEGIES[family],
        line_col=TOTAL_LINE if family is TargetFamily.TOTAL_POINTS else None,
        data=DataConfig(csv_path=path),
        cleaning=CleaningConfig(nan_threshold=50.0, max_na_per_row=2),
        walk_forward=WalkForwardConfig(train_games=TRAIN_GAMES),
    )


def _spec(config: ExperimentConfig, target: ModelTarget, features: list[str]):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version="2_5",
            target=target,
            prediction_strategy=config.strategy.value,
            horizon_minutes=0,
            variant="main",
            dataset_type="closing_line",
        ),
        features=SpecFeatures(feature_names=features, n_features=len(features)),
        cleaning=SpecCleaning(
            nan_threshold=config.cleaning.nan_threshold,
            max_na_per_row=config.cleaning.max_na_per_row,
        ),
        training=SpecTraining(
            params={},
            n_estimators=10,
            train_games=TRAIN_GAMES,
            required_line_col=config.line_col,
        ),
    )


@pytest.mark.parametrize(
    ("family", "target"),
    [
        (TargetFamily.LINE_ERROR, ModelTarget.LINE_ERROR),
        (TargetFamily.TOTAL_POINTS, ModelTarget.TOTAL_POINTS),
        (TargetFamily.SPREAD_ERROR, ModelTarget.SPREAD_ERROR),
    ],
)
def test_refit_window_equals_the_training_window(frame_path, family, target):
    config = _config(frame_path, family)
    with contextlib.redirect_stdout(io.StringIO()):
        prepared = prepare_dataset(config)
    X, y = build_feature_matrix(
        prepared.df_full,
        target_col=config.target_col,
        feature_names=prepared.feature_names,
    )
    window = rolling_window_index(prepared.df_full.index, TRAIN_GAMES, game_ids=None)

    spec = _spec(config, target, prepared.feature_names)
    rows = prepare_refit_rows(load_raw_training_csv(frame_path), spec)
    refit = select_training_window(rows, train_games=TRAIN_GAMES, date_col="GAME_DATE")

    assert len(rows) == len(prepared.df_full)
    pd.testing.assert_series_equal(
        prepared.df_full.loc[window, "GAME_DATE"].reset_index(drop=True),
        refit["GAME_DATE"].reset_index(drop=True),
        check_names=False,
    )
    np.testing.assert_allclose(
        y.loc[window].to_numpy(float), refit[TARGET_COLUMNS[target]].to_numpy(float)
    )
    np.testing.assert_allclose(
        X.loc[window].to_numpy(float),
        refit[prepared.feature_names].to_numpy(float),
    )


def _rows(frame_path, family, target):
    config = _config(frame_path, family)
    spec = _spec(config, target, FEATURES)
    return prepare_refit_rows(load_raw_training_csv(frame_path), spec)


def test_playoffs_and_preseason_are_left_out_and_play_in_is_kept(frame_path):
    rows = _rows(frame_path, TargetFamily.LINE_ERROR, ModelTarget.LINE_ERROR)
    prefixes = set(rows["GAME_ID"].str[:3])
    assert prefixes == {"002", "005"}


def test_residual_targets_are_derived_from_the_main_book(frame_path):
    raw = load_raw_training_csv(frame_path).set_index("GAME_ID")
    rows = _rows(frame_path, TargetFamily.LINE_ERROR, ModelTarget.LINE_ERROR)
    expected = (
        raw.loc[rows["GAME_ID"], "TOTAL_POINTS"] - raw.loc[rows["GAME_ID"], TOTAL_LINE]
    )
    np.testing.assert_allclose(rows["LINE_ERROR"], expected.to_numpy())

    rows = _rows(frame_path, TargetFamily.SPREAD_ERROR, ModelTarget.SPREAD_ERROR)
    expected = (
        raw.loc[rows["GAME_ID"], "HOME_MARGIN"] - raw.loc[rows["GAME_ID"], SPREAD_LINE]
    )
    np.testing.assert_allclose(rows["SPREAD_ERROR"], expected.to_numpy())


def test_a_spread_game_without_a_total_line_is_dropped_like_training(frame_path):
    rows = _rows(frame_path, TargetFamily.SPREAD_ERROR, ModelTarget.SPREAD_ERROR)
    kept = set(rows["GAME_ID"])
    for dropped in (
        "0022500010",
        "0022500011",
        "0022500012",
        "0022500013",
        "0022500014",
    ):
        assert dropped not in kept


def test_a_frame_without_the_spec_columns_is_an_error(frame_path):
    config = _config(frame_path, TargetFamily.LINE_ERROR)
    spec = _spec(config, ModelTarget.LINE_ERROR, [*FEATURES, "NOT_IN_FRAME"])
    with pytest.raises(RefitError, match="NOT_IN_FRAME"):
        prepare_refit_rows(load_raw_training_csv(frame_path), spec)


@pytest.mark.parametrize(
    "data_update",
    [
        {"exclude_playoffs": False},
        {"allowed_season_types": ("Regular Season",)},
        {"exclude_overtime_from_training": True},
    ],
)
def test_promotion_refuses_row_rules_the_refit_cannot_repeat(frame_path, data_update):
    config = _config(frame_path, TargetFamily.LINE_ERROR)
    config = config.model_copy(
        update={"data": config.data.model_copy(update=data_update)}
    )
    with pytest.raises(PromotionError, match="cannot reproduce"):
        check_refit_reproduces_rows(config)


def test_promotion_accepts_the_default_row_rules(frame_path):
    check_refit_reproduces_rows(_config(frame_path, TargetFamily.LINE_ERROR))


def test_a_spec_without_a_seed_keeps_the_id_it_was_stored_under(frame_path):
    config = _config(frame_path, TargetFamily.LINE_ERROR)
    unseeded = _spec(config, ModelTarget.LINE_ERROR, FEATURES)
    assert "random_state" not in unseeded.canonical_body()["training"]

    seeded = unseeded.model_copy(
        update={
            "training": unseeded.training.model_copy(update={"random_state": 17})
        }
    )
    assert seeded.compute_spec_id() != unseeded.spec_id


def test_the_refit_fits_on_cpu_with_the_seed_the_spec_records(frame_path):
    config = _config(frame_path, TargetFamily.LINE_ERROR)
    spec = _spec(config, ModelTarget.LINE_ERROR, FEATURES)
    params = settings_from_spec(spec).xgb_params
    assert params["device"] == "cpu"
    assert params["random_state"] == DEFAULT_RANDOM_STATE

    seeded = spec.model_copy(
        update={"training": spec.training.model_copy(update={"random_state": 17})}
    )
    assert settings_from_spec(seeded).xgb_params["random_state"] == 17


def test_promotion_records_the_run_seed(frame_path):
    config = _config(frame_path, TargetFamily.LINE_ERROR).model_copy(
        update={"random_state": 17}
    )
    hyperparameters = RunHyperparameters(
        run_dir=frame_path.parent,
        params={"max_depth": 2},
        n_estimators=10,
        sample_weight_lambda=None,
        source="selected",
        trial_number=1,
    )
    slot = ModelSlot(
        schema_version="2_5",
        target=ModelTarget.LINE_ERROR,
        horizon_minutes=0,
        variant="main",
    )
    spec = build_spec_from_run(
        config, hyperparameters=hyperparameters, feature_names=FEATURES, slot=slot
    )
    assert spec.training.random_state == 17


def test_promotion_refuses_an_objective_the_refit_does_not_fit(frame_path):
    config = _config(frame_path, TargetFamily.LINE_ERROR)
    config = config.model_copy(
        update={
            "optuna": config.optuna.model_copy(
                update={"objective_name": "reg:absoluteerror"}
            )
        }
    )
    with pytest.raises(PromotionError, match="objective"):
        check_refit_reproduces_rows(config)
