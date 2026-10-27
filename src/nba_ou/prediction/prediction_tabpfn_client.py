"""TabPFN (hosted, ``tabpfn-client``) on a registry slot's features.

TabPFN keeps no model: it is fitted on every run, on the rows and columns of
one registry slot's production build -- by default the slot of the same
target at the closing line or at one intermediate horizon (T-360 by default):

* the features, cleaning and training window: that build's spec, so TabPFN is
  fitted on exactly the rows and columns its XGBoost counterpart was. The
  spec's recency weights are not applied: TabPFN takes no sample weights.
* the rows: the season partitions the daily refit
  (``scripts/retrain_prediction_models.py``) fits on and keeps current
  (``nba_ou.create_training_data.season_partitions``), read through
  ``nba_ou.modeling.refit.resolve_training_frame`` -- every season for a
  closing slot, the slot's horizon for an intermediate one.

Tonight's rows are built by the caller at the moment it predicts
(``scripts/predict_tabpfn.py``) -- for an intermediate slot, the row of its
horizon, whether that snapshot has passed or not (``SNAPSHOT_PASSED``) -- and
go through the registry slots' serving path
(``load_and_predict_model_for_nba_games``): the same NaN budget, picks and
output columns, for every target. The run is stored as ``model_builds.kind`` 3
(``KIND_TABPFN``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from nba_ou.modeling.refit import (
    count_games,
    prepare_refit_rows,
    select_training_window,
    settings_from_spec,
)
from nba_ou.modeling.registry_models import ModelSpec
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget, format_horizon
from nba_ou.modeling.retraining_utils import _coerce_feature_column_for_xgboost
from nba_ou.prediction.prediction import (
    load_and_predict_model_for_nba_games,
    resolve_prediction_target,
)

#: The newest model family ``tabpfn-client`` 0.6 serves
#: (``tabpfn_client.constants.ModelVersion``).
DEFAULT_MODEL_VERSION = "v3.5"
CLOSING_LINE = "closing_line"
INTERMEDIATE_LINE = "intermediate_line"
#: Minutes before tip-off of the intermediate slot TabPFN follows by default.
DEFAULT_INTERMEDIATE_HORIZON = 360
#: The intermediate frames' snapshot column: minutes before tip-off.
SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"


@dataclass(frozen=True)
class TabPFNTrainingSet:
    """The rows and columns TabPFN is fitted on, as the spec selects them."""

    X: pd.DataFrame
    y: pd.Series
    train_date_min: datetime
    train_date_max: datetime
    n_games: int
    #: The S3 key of the frame the rows came from.
    frame_key: str


def feature_slot_for(
    target: str, slots: Sequence[ModelSlot], *, horizon_minutes: int = 0
) -> ModelSlot:
    """The enabled slot of ``target`` at ``horizon_minutes`` (0: the closing
    line) whose features TabPFN reads.

    ``main`` when there are several variants; raises when there is none.
    """
    target = ModelTarget(target)
    found = [
        slot
        for slot in slots
        if slot.target == target and slot.horizon_minutes == horizon_minutes
    ]
    if not found:
        raise ValueError(
            f"No {target.value} {format_horizon(horizon_minutes)} slot in "
            "ENABLED_MODELS to take the features from; name one with "
            "--feature-slot."
        )
    return next((slot for slot in found if slot.variant == "main"), found[0])


def rows_of_horizon(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """``df``'s rows for ``spec``: all of a closing frame, the spec's horizon of
    an intermediate one (which holds one row per game and snapshot)."""
    dataset_type = str(spec.identity.dataset_type)
    if dataset_type == CLOSING_LINE:
        return df
    if dataset_type != INTERMEDIATE_LINE:
        raise ValueError(f"{spec.spec_id}: no TabPFN path for {dataset_type!r}.")
    if SNAPSHOT_COLUMN not in df.columns:
        raise ValueError(
            f"{spec.spec_id} is an intermediate spec, and the frame has no "
            f"{SNAPSHOT_COLUMN} column."
        )
    horizon = spec.identity.horizon_minutes
    rows = df.loc[pd.to_numeric(df[SNAPSHOT_COLUMN], errors="coerce").eq(horizon)]
    if rows["GAME_ID"].astype(str).duplicated().any():
        raise ValueError(f"More than one T-{horizon} row for a game.")
    return rows


def model_variant(model_version: str) -> str:
    """``tabpfn_v3_5``: the ``model_builds.variant`` of a TabPFN model."""
    return "tabpfn_" + model_version.lower().replace(".", "_").replace("-", "_")


def numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` as floats, coerced as the XGBoost refit coerces its features.

    TabPFN reads an object column as text, so a column that is numeric in the
    training frame and object in tonight's would change meaning between fit
    and predict. A column that cannot be made numeric raises.
    """
    return X.apply(_coerce_feature_column_for_xgboost).astype("float64")


def tabpfn_training_set(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    frame_key: str,
    train_games: int | None = None,
) -> TabPFNTrainingSet:
    """The spec's rows of ``frame``: its horizon, target, features, NaN budget
    and window.

    ``train_games`` overrides the spec's window (the last N games).
    """
    settings = settings_from_spec(spec)
    # The XGBoost refit's rows (nba_ou.modeling.refit.prepare_refit_rows), which
    # follow training: regular season and play-in only, the residual target
    # derived (an intermediate frame ships it, against the snapshot's line),
    # basic cleaning, and the NaN budget with Yahoo exempt.
    prepared = prepare_refit_rows(rows_of_horizon(frame, spec), spec)
    window = select_training_window(
        prepared,
        train_games=train_games or settings.train_games,
        date_col=settings.date_column,
    )
    if window.empty:
        raise ValueError(f"{spec.spec_id}: the training window is empty.")
    dates = window[settings.date_column]
    return TabPFNTrainingSet(
        X=numeric_features(window[settings.feature_names]),
        y=pd.to_numeric(window[settings.target_column], errors="coerce"),
        train_date_min=pd.Timestamp(dates.min()).to_pydatetime(),
        train_date_max=pd.Timestamp(dates.max()).to_pydatetime(),
        n_games=count_games(window),
        frame_key=frame_key,
    )


def _authenticate(tabpfn_client) -> None:
    """``TABPFN_ACCESS_TOKEN``, else whatever ``init`` finds (``TABPFN_TOKEN``)."""
    token = os.getenv("TABPFN_ACCESS_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
    else:
        tabpfn_client.init()


class FittedTabPFN:
    """A fitted TabPFN regressor that reads features as it was fitted on them."""

    def __init__(self, regressor, *, model_version: str, client_version: str):
        self.regressor = regressor
        self.model_version = model_version
        self.client_version = client_version

    def predict(self, X: pd.DataFrame):
        return self.regressor.predict(numeric_features(X))


def fit_tabpfn(
    training_set: TabPFNTrainingSet,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    random_state: int = 0,
) -> FittedTabPFN:
    """Fit the hosted regressor of ``model_version`` on ``training_set``."""
    try:
        import tabpfn_client
    except ImportError as exc:
        raise ImportError(
            "tabpfn-client is not installed: pip install -r "
            "requirements/predict-tabpfn-cuda.txt (or poetry install)"
        ) from exc

    _authenticate(tabpfn_client)
    regressor = tabpfn_client.TabPFNRegressor.create_default_for_version(
        model_version, random_state=random_state
    )
    regressor.fit(training_set.X, training_set.y)
    return FittedTabPFN(
        regressor,
        model_version=model_version,
        client_version=tabpfn_client.__version__,
    )


def predict_with_tabpfn(
    *,
    df: pd.DataFrame,
    spec: ModelSpec,
    model: FittedTabPFN,
    training_set: TabPFNTrainingSet,
    prediction_datetime: datetime | None = None,
    passthrough_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Tonight's rows of ``spec`` predicted by ``model``, shaped as a slot's
    output: every closing row, or the intermediate rows of the spec's horizon.

    The build is (spec, training cutoff, window size): the same rows and
    features give the same ``fit_id``, however often a day's runs repeat it.
    """
    dataset_type = str(spec.identity.dataset_type)
    rows = rows_of_horizon(df, spec)
    if rows.empty:
        return pd.DataFrame()
    target = str(spec.identity.target)
    horizon = spec.identity.horizon_minutes
    variant = model_variant(model.model_version)
    train_date_max = training_set.train_date_max
    # As load_registry_model_and_predict marks an intermediate slot's rows.
    input_mode = {} if dataset_type == CLOSING_LINE else {"INPUT_MODE": "own"}
    return load_and_predict_model_for_nba_games(
        df=rows,
        regressor=model,
        model_name=(
            f"{variant}_{target}_{format_horizon(horizon)}_{train_date_max:%Y%m%d}"
        ),
        model_type="TabPFNRegressor",
        model_version=model.model_version,
        required_features=list(spec.features.feature_names),
        prediction_source=variant,
        training_code_tag=f"tabpfn-client {model.client_version}",
        train_date_min=training_set.train_date_min.isoformat(),
        train_date_max=train_date_max.isoformat(),
        prediction_datetime=prediction_datetime,
        prediction_target=resolve_prediction_target(target),
        passthrough_columns=passthrough_columns,
        max_na_per_row=spec.cleaning.max_na_per_row,
        extra_summary_columns={
            "MODEL_TARGET": target,
            "MODEL_HORIZON_MINUTES": horizon,
            "MODEL_SCHEMA_VERSION": spec.identity.schema_version,
            "MODEL_VARIANT": variant,
            "SPEC_ID": spec.spec_id,
            "FIT_ID": (
                f"{spec.spec_id}-{train_date_max:%Y%m%d}-{training_set.n_games}g"
            ),
            "MODEL_DATASET_TYPE": dataset_type,
            # No build of its own: the frame it was fitted on.
            "MODEL_BUILD_PREFIX": training_set.frame_key,
            **input_mode,
        },
    )
