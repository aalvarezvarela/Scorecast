"""Refit a slot's model from its spec.

The difference from the path this replaces is the direction the settings come
from. The old ``build_retraining_settings_from_artifacts`` read *yesterday's
bundle* and derived today's model from it, so every refit inherited from the
one before and nothing was ever a fixed reference -- a value that was once
wrong stayed wrong indefinitely. Here the spec is the reference, and a refit
cannot change it.

The day's training frame comes from the season partitions
(``nba_ou.create_training_data.season_partitions``): each dataset kept in S3
one season per file, brought up to date by the daily job before it refits.
:func:`resolve_training_frame` reads them -- every season for a closing spec,
one horizon of every season for an intermediate one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from nba_ou.config.constants import SEASON_TYPE_MAP, TRAINING_SEASON_TYPES
from nba_ou.config.market_columns import HOME_MARGIN_COL, SPREAD_ERROR_COL, spread_error
from nba_ou.config.odds_columns import spread_line_home_col, total_line_col
from nba_ou.config.yahoo_features import compact_yahoo_columns
from nba_ou.create_training_data.season_partitions import (
    PartitionedFrame,
    SeasonPartitions,
)
from nba_ou.data_processing.missing_data.clean_df_for_training import basic_cleaning
from nba_ou.modeling.meta_learner_training_data import _ensure_line_error_column
from nba_ou.modeling.registry_models import FitRecord, ModelSpec, build_model_name
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget, build_fit_id
from nba_ou.modeling.retraining_utils import (
    RetrainingSettings,
    _coerce_feature_column_for_xgboost,
    retrain_model,
)
from xgboost import XGBRegressor

#: Fixed across every model: these decide how XGBoost runs, not what it learns.
#: The tuned half lives in ``spec.training.params``; the seed and the device
#: come from the spec too (refit_random_state, refit_device).
XGB_STATIC_PARAMS: dict[str, object] = {
    "booster": "gbtree",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_jobs": -1,
    "verbosity": 0,
}

#: The seed for a spec written before seeds were recorded: the training
#: default (ExperimentConfig.random_state), which 50 of the first 51 promoted
#: runs used. The exception, 2_5/line_error/t0000/main, ran with 17.
DEFAULT_RANDOM_STATE = 16

#: The device for a spec written before devices were recorded: all 51 of the
#: first promoted runs were tuned and fitted on CUDA.
DEFAULT_DEVICE = "cuda"


def refit_random_state(spec: ModelSpec) -> int:
    if spec.training.random_state is not None:
        return spec.training.random_state
    return DEFAULT_RANDOM_STATE


def refit_device(spec: ModelSpec) -> str:
    """The device to refit on: the one the run was evaluated on.

    CUDA and CPU build different trees from the same seed and data (on the
    closing line_error slot only ~79% of games get the same side), so a refit
    on another device is not the model that was evaluated.
    """
    return spec.training.device or DEFAULT_DEVICE


def fitted_device(model: XGBRegressor) -> str:
    """The device a fitted model actually trained on, e.g. "cuda:0" or "cpu"."""
    config = json.loads(model.get_booster().save_config())
    return str(config["learner"]["generic_param"]["device"])


def require_device(device: str) -> None:
    """Fail now if ``device`` cannot be used.

    Without a visible GPU, XGBoost does not raise: it logs a warning and fits
    on CPU, which here would silently ship a different model. A tiny fit shows
    which device it really used.
    """
    if not device.startswith("cuda"):
        return
    probe = XGBRegressor(n_estimators=1, device=device, verbosity=0)
    probe.fit(np.zeros((4, 1)), np.zeros(4))
    if not fitted_device(probe).startswith("cuda"):
        raise RefitError(
            f"The spec asks for device={device!r}, but XGBoost fell back to "
            f"{fitted_device(probe)!r}: no usable GPU or no CUDA build. Refit on "
            "the machine the configuration was evaluated on."
        )


DATE_COLUMN = "GAME_DATE"
GAME_ID_COLUMN = "GAME_ID"

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
    xgb_params = {
        **XGB_STATIC_PARAMS,
        "device": refit_device(spec),
        "random_state": refit_random_state(spec),
        **spec.training.params,
    }
    xgb_params["n_estimators"] = spec.training.n_estimators
    return RetrainingSettings(
        feature_names=list(spec.features.feature_names),
        target_column=target_column,
        date_column=DATE_COLUMN,
        required_line_col=spec.training.required_line_col,
        minimum_line_value=spec.training.minimum_line_value,
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


def with_residual_targets(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with ``LINE_ERROR`` and ``SPREAD_ERROR`` where it can carry them.

    The closing builder ships the outcomes (``TOTAL_POINTS``, ``HOME_MARGIN``)
    and the main book's lines, not the residuals: the training pipeline derives
    them at load time (``training_pipeline.data``), and a refit must train on
    the same target. Same derivations, never overwriting a column already
    there.
    """
    df = _ensure_line_error_column(df)
    line_col = spread_line_home_col()
    if SPREAD_ERROR_COL not in df.columns and {HOME_MARGIN_COL, line_col} <= set(
        df.columns
    ):
        df = df.copy()
        df[SPREAD_ERROR_COL] = spread_error(df[HOME_MARGIN_COL], df[line_col])
    return df


def count_games(df: pd.DataFrame, *, game_id_col: str | None = GAME_ID_COLUMN) -> int:
    if game_id_col and game_id_col in df.columns:
        return int(df[game_id_col].nunique())
    return int(len(df))


def resolve_training_frame(
    spec: ModelSpec,
    *,
    partitions: Mapping[str, SeasonPartitions],
    base_frames: dict[str, PartitionedFrame] | None = None,
    layer_context=None,
) -> TrainingFrame:
    """The frame this spec is refitted on, from its dataset's season partitions.

    ``partitions`` maps a dataset type to its partitions at the base schema
    version. A closing spec reads every season, and a spec of a newer version
    gets that frame plus its layers (``nba_ou.create_training_data.
    schema_layers``); pass the same ``base_frames`` dict (and
    ``layer_context``) for every spec of one run and the seasons are read, and
    each layer input loaded, only once. An intermediate spec reads its own
    horizon of every season.

    The frame is the full build's, row order included (``SeasonPartitions.
    load``), which the training window and the seeded subsampling depend on.
    """
    from nba_ou.create_training_data.schema_layers import (
        CLOSING_LINE,
        apply_layers,
        available_versions,
        is_layered,
    )

    dataset_type = str(spec.identity.dataset_type)
    schema_version = spec.identity.schema_version
    store = partitions.get(dataset_type)
    if store is None:
        raise TrainingFrameUnavailable(
            f"Spec {spec.spec_id} needs {dataset_type!r} season partitions, and "
            "none were given."
        )
    if dataset_type == CLOSING_LINE:
        if is_layered(schema_version) and schema_version not in available_versions():
            raise TrainingFrameUnavailable(
                f"Spec {spec.spec_id} needs a {schema_version} dataset, and this "
                f"checkout builds only {', '.join(available_versions())}."
            )
        cache = base_frames if base_frames is not None else {}
        if dataset_type not in cache:
            cache[dataset_type] = store.load()
        loaded = cache[dataset_type]
        df = loaded.df
        if is_layered(schema_version):
            df = apply_layers(
                df,
                to_version=schema_version,
                dataset_type=CLOSING_LINE,
                ctx=layer_context,
            )
        return TrainingFrame(
            df=with_residual_targets(df),
            build_id=loaded.build_id,
            checksum=loaded.checksum,
        )

    if is_layered(schema_version):
        raise TrainingFrameUnavailable(
            f"Spec {spec.spec_id} is a {schema_version} {dataset_type} spec, and "
            "the intermediate dataset has no schema-layer step yet."
        )
    horizon = spec.identity.horizon_minutes
    if horizon is None:
        raise TrainingFrameUnavailable(
            f"Spec {spec.spec_id} is pooled over every horizon; only one-horizon "
            "intermediate specs are refitted."
        )
    loaded = store.load(horizon=horizon)
    return TrainingFrame(
        df=with_residual_targets(loaded.df),
        build_id=loaded.build_id,
        checksum=loaded.checksum,
    )


def _with_target(
    df: pd.DataFrame, target: ModelTarget
) -> tuple[pd.DataFrame, list[str]]:
    """The frame with its target column, and the lines training requires present.

    The same derivations training uses (training_pipeline.data.prepare_dataset):
    LINE_ERROR is the main book's totals residual, SPREAD_ERROR the main book's
    spread residual, each derived only when the frame does not carry one.
    """
    if target is ModelTarget.LINE_ERROR:
        return _ensure_line_error_column(df), [total_line_col()]
    if target is ModelTarget.SPREAD_ERROR:
        line_col = spread_line_home_col()
        if SPREAD_ERROR_COL in df.columns:
            return df, [line_col]
        missing = [c for c in (HOME_MARGIN_COL, line_col) if c not in df.columns]
        if missing:
            raise RefitError(f"Cannot derive {SPREAD_ERROR_COL}: {missing} missing.")
        out = df.copy()
        out[SPREAD_ERROR_COL] = spread_error(out[HOME_MARGIN_COL], out[line_col])
        return out, [line_col]
    return df, [total_line_col()]


def prepare_refit_rows(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """The rows training would fit this spec on, oldest first.

    Follows training_pipeline.data.prepare_dataset step by step, so a refit
    trains on the games the promoted configuration was evaluated on:

    1. regular season and play-in only (TRAINING_SEASON_TYPES, by GAME_ID
       prefix -- the SEASON_TYPE text labels play-in games "Playoffs");
    2. the target, derived as training derives it;
    3. ``basic_cleaning``: implausible totals, and a missing or implausible
       main-book total line, for every target;
    4. ``max_na_per_row`` over the model's features, with the optional compact
       Yahoo block exempt (the rest of training's cleaned frame is outcome and
       date columns, which a finished game always has);
    5. a missing target or required line;
    6. sorted by date the way training sorts, so the window's boundary date
       keeps the same games.

    Column selection is not repeated: the spec's ``feature_names`` are what
    training's cleaning kept.
    """
    target = ModelTarget(spec.identity.target)
    target_col = TARGET_COLUMNS[target]
    features = list(spec.features.feature_names)

    if GAME_ID_COLUMN not in df.columns:
        raise RefitError(f"The frame has no {GAME_ID_COLUMN!r} to filter season types.")
    season_type = (
        df[GAME_ID_COLUMN].astype(str).str.strip().str[:3].map(SEASON_TYPE_MAP)
    )
    df = df.loc[season_type.isin(TRAINING_SEASON_TYPES)]

    df, line_cols = _with_target(df, target)
    if spec.training.required_line_col:
        line_cols.append(spec.training.required_line_col)
    required = [DATE_COLUMN, GAME_ID_COLUMN, target_col, *line_cols, *features]
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise RefitError(
            f"The training frame lacks {len(missing)} column(s) this spec needs, "
            f"e.g. {missing[:5]}."
        )

    df = basic_cleaning(df, verbose=0)

    if spec.cleaning.max_na_per_row >= 0:
        yahoo = set(compact_yahoo_columns(df.columns))
        counted = [c for c in features if c not in yahoo]
        df = df.loc[df[counted].isna().sum(axis=1) <= spec.cleaning.max_na_per_row]

    rows = df[
        list(
            dict.fromkeys(
                [DATE_COLUMN, GAME_ID_COLUMN, target_col, *line_cols, *features]
            )
        )
    ].copy()
    rows[target_col] = pd.to_numeric(rows[target_col], errors="coerce")
    rows = rows.dropna(subset=[target_col, *line_cols])
    if spec.training.minimum_line_value is not None and spec.training.required_line_col:
        line = pd.to_numeric(rows[spec.training.required_line_col], errors="coerce")
        rows = rows.loc[line > spec.training.minimum_line_value]

    rows[DATE_COLUMN] = pd.to_datetime(rows[DATE_COLUMN])
    # Default sort kind on purpose: it is the one prepare_dataset uses.
    rows = rows.sort_values(DATE_COLUMN).reset_index(drop=True)
    for feature in features:
        rows[feature] = _coerce_feature_column_for_xgboost(rows[feature])
    return rows


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
    prepared = prepare_refit_rows(frame.df, spec)
    if prepared.empty:
        raise RefitError(f"{slot.describe()}: no rows survived preparation.")

    window = select_training_window(
        prepared, train_games=settings.train_games, date_col=settings.date_column
    )
    n_train_games = count_games(window)
    if n_train_games == 0:
        raise RefitError(f"{slot.describe()}: training window is empty.")

    model = retrain_model(window, settings=settings)
    wanted = str(settings.xgb_params["device"])
    if wanted.startswith("cuda") and not fitted_device(model).startswith("cuda"):
        raise RefitError(
            f"{slot.describe()}: fitted on {fitted_device(model)!r} although the "
            f"spec asks for {wanted!r}; refusing to stage a different model."
        )

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
