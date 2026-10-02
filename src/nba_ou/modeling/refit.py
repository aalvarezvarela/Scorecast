"""Refit a slot's model from its spec.

The difference from the path this replaces is the direction the settings come
from. The old ``build_retraining_settings_from_artifacts`` read *yesterday's
bundle* and derived today's model from it, so every refit inherited from the
one before and nothing was ever a fixed reference -- a value that was once
wrong stayed wrong indefinitely. Here the spec is the reference, and a refit
cannot change it.

What is deliberately NOT here: how the day's training frame is produced. That
is work item 10 in ``docs/model_registry_reorg_plan.md`` and it is a real
design problem (the intermediate dataset is 4.4 GB, and its snapshot grid is
frozen at build time). :func:`resolve_training_frame` is the seam; the
closing-line half works today and the intermediate half says what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from nba_ou.modeling.registry_models import (
    FitRecord,
    ModelSpec,
    build_model_name,
)
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget, build_fit_id
from nba_ou.modeling.retraining_utils import (
    RetrainingSettings,
    prepare_retraining_dataframe_from_raw,
    retrain_model,
)
from xgboost import XGBRegressor

#: Fixed across every model: these decide how XGBoost runs, not what it learns.
#: The tuned half lives in ``spec.training.params``.
XGB_STATIC_PARAMS: dict[str, object] = {
    "booster": "gbtree",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "random_state": 16,
    "n_jobs": -1,
    "verbosity": 0,
}

DATE_COLUMN = "GAME_DATE"
GAME_ID_COLUMN = "GAME_ID"
MINIMUM_LINE_VALUE = 100.0

#: Which column each target trains against.
TARGET_COLUMNS: dict[ModelTarget, str] = {
    ModelTarget.LINE_ERROR: "LINE_ERROR",
    ModelTarget.TOTAL_POINTS: "TOTAL_POINTS",
    ModelTarget.SPREAD_ERROR: "SPREAD_ERROR",
}


def model_bytes_of(model: XGBRegressor) -> bytes:
    """Serialise a booster the way the serving path expects to read it.

    Via a file rather than ``save_raw`` so that what is uploaded is
    byte-identical to what ``save_model`` has always produced, and
    ``XGBRegressor.load_model(bytearray(...))`` round-trips it.
    """
    from pathlib import Path as _Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        path = _Path(temp_dir) / "model.json"
        model.save_model(path)
        return path.read_bytes()


class RefitError(RuntimeError):
    """A slot could not be refitted."""


class TrainingFrameUnavailable(RefitError):
    """The dataset this spec needs is not available."""


@dataclass(frozen=True)
class TrainingFrame:
    """A day's training data, with the identity of the bytes it came from."""

    df: pd.DataFrame
    build_id: str | None
    checksum: str | None


@dataclass(frozen=True)
class RefitResult:
    model: XGBRegressor
    fit: FitRecord
    spec: ModelSpec


def settings_from_spec(spec: ModelSpec) -> RetrainingSettings:
    """Translate a spec into the shape the refit helpers expect.

    ``source_metadata`` stays None: there is no previous bundle in this path,
    which is the point.
    """
    target_column = TARGET_COLUMNS[ModelTarget(spec.identity.target)]
    xgb_params = {**XGB_STATIC_PARAMS, **spec.training.params}
    xgb_params["n_estimators"] = spec.training.n_estimators
    return RetrainingSettings(
        feature_names=list(spec.features.feature_names),
        target_column=target_column,
        date_column=DATE_COLUMN,
        required_line_col=spec.training.required_line_col,
        minimum_line_value=(
            spec.training.minimum_line_value
            if spec.training.minimum_line_value is not None
            else MINIMUM_LINE_VALUE
        ),
        nan_threshold=spec.cleaning.nan_threshold,
        max_na_per_row=spec.cleaning.max_na_per_row,
        train_games=spec.training.train_games,
        xgb_params=xgb_params,
        sample_weight_lambda=spec.training.sample_weight_lambda,
        sample_weight_lambda_bounds=None,
    )


def select_training_window(
    prepared_df: pd.DataFrame,
    *,
    date_col: str = DATE_COLUMN,
    train_games: int | None,
    game_id_col: str | None = GAME_ID_COLUMN,
) -> pd.DataFrame:
    """The last ``train_games`` GAMES, not the last ``train_games`` rows.

    On a pooled intermediate frame one game contributes several snapshot rows,
    so a row count silently divides the intended window by the snapshot count.
    The same mistake is called out in ``experiments/_base.yaml``'s row-counted
    windows and handled the same way in ``training_pipeline.promote``.
    """
    sorted_df = prepared_df.sort_values(date_col, kind="mergesort").reset_index(
        drop=True
    )
    if train_games is None:
        return sorted_df
    if train_games <= 0:
        raise ValueError("train_games must be greater than zero.")

    if game_id_col and game_id_col in sorted_df.columns:
        game_order = sorted_df[game_id_col].drop_duplicates()
        if len(game_order) <= train_games:
            return sorted_df
        keep = set(game_order.tail(train_games))
        return sorted_df[sorted_df[game_id_col].isin(keep)].reset_index(drop=True)

    if train_games >= len(sorted_df):
        return sorted_df
    return sorted_df.tail(train_games).reset_index(drop=True)


def count_games(df: pd.DataFrame, *, game_id_col: str | None = GAME_ID_COLUMN) -> int:
    if game_id_col and game_id_col in df.columns:
        return int(df[game_id_col].nunique())
    return int(len(df))


def resolve_training_frame(
    spec: ModelSpec, *, limit_date: str | None = None
) -> TrainingFrame:
    """Produce the frame this spec should be refitted on.

    Closing-line specs are built in-process, as the daily job already does.

    Intermediate-line specs are NOT yet supported, and the gap is a design
    decision rather than a missing function: the intermediate dataset is built
    by its own entry point, the current one is 4.4 GB, and its snapshot grid is
    frozen at build time -- so a daily refit needs a settled answer on rebuild
    strategy, storage, build selection and staleness before it can be wired up.
    See section 5 and work item 10 of the plan.
    """
    dataset_type = spec.identity.dataset_type
    if dataset_type == "closing_line":
        from nba_ou.create_training_data.create_df_to_predict import (
            create_df_to_predict,
        )

        df = create_df_to_predict(
            todays_prediction=False,
            recent_limit_to_include=limit_date,
            older_season_limit=None,
        )
        return TrainingFrame(df=df, build_id=None, checksum=None)

    raise TrainingFrameUnavailable(
        f"Spec {spec.spec_id} needs a {dataset_type!r} dataset, and the daily "
        "build for it is not implemented yet (plan section 5, work item 10). "
        "Until then, refit this slot by hand with "
        "`python -m training_pipeline.promote <run_dir> --csv <build> --to-s3`."
    )


def refit_from_spec(
    spec: ModelSpec,
    *,
    slot: ModelSlot,
    frame: TrainingFrame,
    fitted_by: str | None = None,
) -> RefitResult:
    """Fit one model exactly as its spec describes."""
    spec.check()
    spec.check_matches_slot(slot)

    settings = settings_from_spec(spec)
    prepared = prepare_retraining_dataframe_from_raw(
        frame.df,
        settings=settings,
        # Kept so the window can be counted in games. It is not a feature: the
        # spec's feature_names decide what the model is handed.
        passthrough_columns=[GAME_ID_COLUMN],
    )
    if prepared.empty:
        raise RefitError(
            f"{slot.describe()}: no rows survived preparation. The dataset may "
            "not carry this spec's columns."
        )

    window = select_training_window(
        prepared, train_games=settings.train_games, date_col=settings.date_column
    )
    n_train_games = count_games(window)
    if n_train_games == 0:
        raise RefitError(f"{slot.describe()}: training window is empty.")

    model = retrain_model(window, settings=settings)

    train_date_min = pd.Timestamp(window[settings.date_column].min()).to_pydatetime()
    train_date_max = pd.Timestamp(window[settings.date_column].max()).to_pydatetime()
    fitted_at = datetime.now(tz=UTC)

    fit = FitRecord(
        fit_id=build_fit_id(spec.spec_id, fitted_at=fitted_at),
        spec_id=spec.spec_id,
        model_name=build_model_name(
            target=slot.target,
            horizon_minutes=slot.horizon_minutes,
            variant=slot.variant,
            schema_version=slot.schema_version,
            train_date_max=train_date_max,
        ),
        train_date_min=train_date_min,
        train_date_max=train_date_max,
        n_train_games=n_train_games,
        n_features=spec.features.n_features,
        dataset_build_id=frame.build_id,
        dataset_checksum=frame.checksum,
        fitted_at=fitted_at,
        fitted_by=fitted_by,
    )
    return RefitResult(model=model, fit=fit, spec=spec)
