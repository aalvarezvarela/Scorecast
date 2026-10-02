"""Train a production model from an experiment you already trust.

This is the deployment step, deliberately separate from ``run_experiment``:
the experiment is what *justifies* a configuration, and this is what *ships*
it. It reuses the hyperparameters a past run's Optuna study selected, refits
them on whatever data you point it at (usually a fresher snapshot), and saves
the bundle. No tuning, no holdout, no evaluation -- one data prep and one fit.

    python -m training_pipeline.promote artifacts/experiments/temp_20260801_175710 \\
        --csv data/train_data/all_odds_training_data_until_20260704.csv

Because nothing here is measured, the saved bundle records which run vouched
for it, so a model on disk can always be traced back to the experiment whose
numbers justified deploying it.
"""

from __future__ import annotations

import argparse
import getpass
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from nba_ou.modeling.modeling import (
    ModelBundleMetadata,
    ModelInfo,
    TrainingMetrics,
    save_model_bundle,
)
from nba_ou.modeling.refit import model_bytes_of
from nba_ou.modeling.registry_models import (
    FitRecord,
    ModelSpec,
    build_model_name,
)
from nba_ou.modeling.registry_paths import Channel, ModelSlot, build_fit_id
from nba_ou.modeling.registry_store import (
    read_channel,
    set_build_channel,
    set_config_channel,
    write_build,
    write_spec,
)
from nba_ou.utils.s3_models import make_s3_client
from xgboost import XGBRegressor

from training_pipeline import naming
from training_pipeline.config import ExperimentConfig, RefitStrategy
from training_pipeline.data import prepare_dataset
from training_pipeline.registry import (
    PromotionError,
    build_spec_from_run,
    check_horizon_is_buildable,
    check_schema_version_against_checkout,
    dataset_build_id,
    enabled_models_row,
    parse_slot_argument,
    slot_from_config,
)
from training_pipeline.reuse import (
    RunHyperparameters,
    find_best_run_hyperparameters,
    load_run_hyperparameters,
)
from training_pipeline.tuning import fit_final_model


@dataclass
class ProductionModelResult:
    model: XGBRegressor
    model_path: Path | None
    meta_path: Path | None
    hyperparameters: RunHyperparameters
    source_run: Path
    csv_path: Path
    dataset_checksum: str | None
    n_train_games: int
    train_date_min: pd.Timestamp
    train_date_max: pd.Timestamp
    n_features: int
    #: The matrix the model was actually handed, in order. Correlation pruning
    #: is data-dependent, so this -- not the config -- is the feature contract.
    feature_names: list[str] = field(default_factory=list)
    #: The run's config after any --csv / --training-version overrides, so a
    #: registry spec can be built without re-preparing the dataset.
    config: ExperimentConfig | None = None


def load_run_config(run_dir: str | Path) -> ExperimentConfig:
    """Rebuild the config a run actually used, from its own snapshot."""
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found -- cannot tell how that run was configured. "
            "Was it run with save_experiment_artifacts enabled?"
        )
    # Fields removed from the schema since the run are ignored by pydantic,
    # so older snapshots still load.
    return ExperimentConfig.model_validate(json.loads(config_path.read_text()))


def _source_run_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "final_test_metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def train_production_model_from_run(
    run_dir: str | Path,
    *,
    csv_path: str | Path | None = None,
    training_version: str | None = None,
    expected_checksum: str | None = None,
    refit_strategy: RefitStrategy | None = None,
    overwrite: bool = False,
    save: bool = True,
) -> ProductionModelResult:
    """Refit a past run's chosen hyperparameters on (usually newer) data.

    ``csv_path`` defaults to whatever the source run used. When you point it
    at a different file, the run's pinned ``expected_checksum`` is dropped --
    it describes the old bytes -- unless you supply a new one.
    """
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    hyperparameters = load_run_hyperparameters(run_dir)

    if config.is_classifier:
        # Refused rather than half-supported. The serving path
        # (nba_ou.prediction.prediction) calls model.predict() and reads the
        # result as a points total; an XGBClassifier returns a 0/1 class there,
        # so a promoted classifier bundle would not crash -- the daily job would
        # quietly treat "1" as a one-point game. A loud stop beats that.
        #
        # Phase 2 territory: teach the serving path about predict_proba and give
        # TrainingMetrics real log-loss/Brier fields (it currently requires
        # cv_mae, which has no meaning for this strategy).
        raise ValueError(
            f"{run_dir.name} is an over_under_classifier run, which cannot be "
            "promoted yet: the prediction service reads model.predict() as a "
            "points total and would silently misinterpret a class label. "
            "Evaluate classifiers with --no-save-model until the serving path "
            "supports probabilities."
        )

    updates: dict[str, Any] = {}
    if csv_path is not None:
        data = config.data.model_copy(
            update={
                "csv_path": Path(csv_path),
                "expected_checksum": expected_checksum,
                "data_version": None if expected_checksum is None else config.data.data_version,
            }
        )
        updates["data"] = data
    elif expected_checksum is not None:
        updates["data"] = config.data.model_copy(
            update={"expected_checksum": expected_checksum}
        )
    if training_version is not None:
        updates["training_version"] = training_version
    if refit_strategy is not None:
        updates["refit"] = config.refit.model_copy(update={"strategy": refit_strategy})
    if updates:
        config = config.model_copy(update=updates)

    # The last door into production, and the one most likely to be walked
    # through months later by someone reading a leaderboard rather than a run's
    # metadata. A diagnostic run's CV numbers look excellent BECAUSE a feature
    # was derived from the target, so they are exactly the rows most likely to
    # be picked for promotion by accident.
    if config.is_diagnostic:
        raise ValueError(
            f"Refusing to promote {run_dir}: it is a diagnostic run carrying the "
            f"target-derived feature "
            f"{config.diagnostics.planted_signal.column!r}. Its metrics measure "
            "whether the pipeline can find a planted signal, not whether the "
            "model can predict anything."
        )

    prepared = prepare_dataset(config)

    # config.target_col, not a two-way branch: the classifier trains on
    # OVER_LABEL, and handing an XGBClassifier continuous point totals under a
    # binary objective is a silent nonsense at best.
    target_col = config.target_col
    from training_pipeline.data import build_feature_matrix, rolling_window_index

    X, y = build_feature_matrix(
        prepared.df_full, target_col=target_col, feature_names=prepared.feature_names
    )
    dates = prepared.df_full[config.data.date_col]

    # The selected trial may have tuned the window. The value saved in the run
    # config is only the fallback used when no trial-level value was selected.
    train_games = (
        hyperparameters.train_games
        if hyperparameters.train_games is not None
        else config.walk_forward.train_games
    )
    if config.refit.strategy == RefitStrategy.ROLLING_WINDOW and train_games:
        window = rolling_window_index(
            prepared.df_full.index,
            train_games,
            game_ids=(
                prepared.df_full[config.data.game_id_col]
                if prepared.is_pooled_snapshots
                and config.data.game_id_col in prepared.df_full.columns
                else None
            ),
        )
        X = X.loc[window]
        y = y.loc[window]
        dates = dates.loc[window]

    # On a pooled intermediate dataset one game contributes several snapshot
    # rows. Bundle metadata is consumed as a game window later, so recording
    # len(X) here would multiply the intended window by the snapshot count.
    if config.data.game_id_col in prepared.df_full.columns:
        n_train_games = int(
            prepared.df_full.loc[X.index, config.data.game_id_col].nunique()
        )
    else:
        n_train_games = len(X)

    model = fit_final_model(
        X_dev=X,
        y_dev=y,
        params=hyperparameters.params,
        n_estimators=hyperparameters.n_estimators,
        config=config,
        dates_dev=dates,
        sample_weight_lambda=hyperparameters.sample_weight_lambda,
    )

    model_path: Path | None = None
    meta_path: Path | None = None
    if save:
        model_name = naming.build_model_name(
            config, as_of=pd.Timestamp(dates.max()).date()
        )
        out_dir = naming.resolve_model_output_dir(config)
        naming.assert_model_bundle_is_writable(
            out_dir, model_name=model_name, overwrite_existing_model=overwrite
        )
        metadata = _build_metadata(
            config,
            model_name=model_name,
            hyperparameters=hyperparameters,
            source_run=run_dir,
            train_date_min=pd.Timestamp(dates.min()),
            train_date_max=pd.Timestamp(dates.max()),
            n_train_games=n_train_games,
        )
        model_path, meta_path = save_model_bundle(
            model=model,
            feature_names=list(X.columns),
            out_dir=out_dir,
            metadata=metadata,
        )

    return ProductionModelResult(
        model=model,
        model_path=model_path,
        meta_path=meta_path,
        hyperparameters=hyperparameters,
        source_run=run_dir,
        csv_path=Path(config.data.csv_path),
        dataset_checksum=prepared.dataset_checksum,
        n_train_games=n_train_games,
        train_date_min=pd.Timestamp(dates.min()),
        train_date_max=pd.Timestamp(dates.max()),
        n_features=len(X.columns),
        feature_names=list(X.columns),
        config=config,
    )


def publish_config_to_registry(
    result: ProductionModelResult,
    *,
    slot: ModelSlot,
    bucket: str | None = None,
    s3_client=None,
    replace_config: bool = False,
    promoted_by: str | None = None,
) -> RegistryPromotion:
    """Adopt a configuration: write its spec, stage its first build.

    Deliberately does NOT touch ``channels/production.json``. A new
    configuration takes effect when ``promote_build`` says so, and until then
    production keeps serving its existing build under the spec that build
    names. Changing the configuration can therefore never cause an outage.

    Write order is spec, build, config channel, staging channel -- nothing
    points at an object that is not there yet.
    """
    if result.config is None or not result.feature_names:
        raise PromotionError(
            "This result was produced with save=False; there is no feature "
            "matrix to promote."
        )

    from nba_ou.config.settings import SETTINGS

    if s3_client is None:
        s3_client = make_s3_client(
            profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        )
    bucket = bucket or SETTINGS.s3_bucket

    existing_config = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.CONFIG
    )
    if existing_config is not None and not replace_config:
        raise PromotionError(
            f"{slot.describe()} already has a configuration "
            f"(spec {existing_config.spec_id}). Re-promoting a slot is a real "
            "operation, so it has to be typed out: pass --replace-config."
        )

    source_metrics = _source_run_metrics(result.source_run)
    cv = source_metrics.get("cv") or {}
    holdout = source_metrics.get("holdout") or {}

    spec = build_spec_from_run(
        result.config,
        hyperparameters=result.hyperparameters,
        feature_names=result.feature_names,
        slot=slot,
        source_run=result.source_run,
        dataset_checksum=result.dataset_checksum,
        vouching_metrics={
            "cv_mae": cv.get("mae"),
            "cv_ou_acc": cv.get("ou_acc"),
            "holdout_mae": holdout.get("mae"),
            "holdout_ou_acc": holdout.get("ou_acc"),
        },
        promoted_by=promoted_by,
    )

    fitted_at = datetime.now(tz=UTC)
    fit = FitRecord(
        fit_id=build_fit_id(spec.spec_id, fitted_at=fitted_at),
        spec_id=spec.spec_id,
        model_name=build_model_name(
            target=slot.target,
            horizon_minutes=slot.horizon_minutes,
            variant=slot.variant,
            schema_version=slot.schema_version,
            train_date_max=result.train_date_max.to_pydatetime(),
        ),
        train_date_min=result.train_date_min.to_pydatetime(),
        train_date_max=result.train_date_max.to_pydatetime(),
        n_train_games=result.n_train_games,
        n_features=result.n_features,
        dataset_build_id=dataset_build_id(result.csv_path),
        dataset_checksum=result.dataset_checksum,
        fitted_at=fitted_at,
        fitted_by=promoted_by,
    )

    write_spec(s3_client=s3_client, bucket=bucket, slot=slot, spec=spec)
    write_build(
        s3_client=s3_client,
        bucket=bucket,
        slot=slot,
        fit=fit,
        model_bytes=model_bytes_of(result.model),
        spec=spec,
    )
    set_config_channel(
        s3_client=s3_client,
        bucket=bucket,
        slot=slot,
        spec_id=spec.spec_id,
        set_by=promoted_by,
    )
    set_build_channel(
        s3_client=s3_client,
        bucket=bucket,
        slot=slot,
        channel=Channel.STAGING,
        fit_id=fit.fit_id,
        set_by=promoted_by,
        reason=f"promote-config from {Path(result.source_run).name}",
    )
    return RegistryPromotion(
        slot=slot, spec=spec, fit=fit, bucket=bucket, replaced=existing_config is not None
    )


@dataclass
class RegistryPromotion:
    slot: ModelSlot
    spec: ModelSpec
    fit: FitRecord
    bucket: str
    replaced: bool


def _build_metadata(
    config: ExperimentConfig,
    *,
    model_name: str,
    hyperparameters: RunHyperparameters,
    source_run: Path,
    train_date_min: pd.Timestamp,
    train_date_max: pd.Timestamp,
    n_train_games: int,
) -> ModelBundleMetadata:
    """Bundle metadata that names the experiment vouching for this model.

    The metrics below were measured by that experiment on ITS data, not on the
    data this model was just fitted to -- this model is unevaluated by
    construction. ``training_code_tag`` carries the source run so the link is
    never lost.
    """
    source_metrics = _source_run_metrics(source_run)
    holdout = source_metrics.get("holdout") or {}
    cv = source_metrics.get("cv") or {}
    nan = float("nan")

    version = config.training_version or naming.DEFAULT_TRAINING_CODE_TAG
    return ModelBundleMetadata(
        model_info=ModelInfo(
            name=model_name,
            model_version=train_date_max.strftime("%d_%m_%y"),
            model_type=f"{config.resolved_window_name_label}_{config.family.value}",
            prediction_source=model_name,
            training_code_tag=f"{version}|from_run:{source_run.name}",
        ),
        training_metrics=TrainingMetrics(
            best_params=hyperparameters.params,
            selected_trial_number=hyperparameters.trial_number,
            mean_best_iteration=hyperparameters.n_estimators,
            median_best_iteration=hyperparameters.n_estimators,
            train_games=n_train_games,
            sample_weight_lambda_bounds=(
                config.sample_weight.lambda_bounds
                if hyperparameters.sample_weight_lambda is not None
                else None
            ),
            # Inherited from the source experiment; NOT measured on this data.
            cv_mae=float(cv.get("mae") or hyperparameters.cv_mae or nan),
            cv_rmse=cv.get("rmse"),
            cv_ou_acc=cv.get("ou_acc"),
            final_test_mae=float(holdout.get("mae") or nan),
            final_test_rmse=float(holdout.get("rmse") or nan),
            final_test_ou_acc=float(holdout.get("ou_acc") or nan),
            nan_threshold=config.cleaning.nan_threshold,
            max_na_per_row=config.cleaning.max_na_per_row,
            train_date_min=train_date_min.to_pydatetime(),
            train_date_max=train_date_max.to_pydatetime(),
        ),
    )


def _current_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - identity is a nicety, never a blocker
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a production model by reusing the hyperparameters an "
            "experiment's Optuna study selected, refitted on chosen data."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "run_dir", nargs="?", help="Experiment run directory to take hyperparameters from."
    )
    source.add_argument(
        "--from-best",
        metavar="ROOT",
        help="Use the best run under ROOT instead (ranked by ROI).",
    )
    parser.add_argument(
        "--csv", help="Train on this CSV instead of the one the source run used."
    )
    parser.add_argument(
        "--expected-checksum",
        help="Pin the new dataset's checksum (see data.compute_file_checksum).",
    )
    parser.add_argument("--training-version", help="Label recorded on the bundle.")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Fit on every game rather than the last walk_forward.train_games.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing bundle."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be trained without fitting or saving.",
    )

    registry = parser.add_argument_group(
        "registry",
        "Adopt this configuration as a slot's source of truth in S3. The daily "
        "refit then reads it; production is untouched until promote_build runs.",
    )
    registry.add_argument(
        "--to-s3",
        action="store_true",
        help="Write the spec and stage the first build in the S3 registry.",
    )
    registry.add_argument(
        "--slot",
        help=(
            "Target slot as schema_version/target/horizon/variant, e.g. "
            "'2_5/line_error/t0060/main'. Default: inferred from the run's own "
            "config, which is the usual case."
        ),
    )
    registry.add_argument(
        "--variant",
        default="main",
        help="Variant name when the slot is inferred (default: main).",
    )
    registry.add_argument(
        "--schema-version",
        help="Override the schema version parsed from the dataset filename.",
    )
    registry.add_argument(
        "--replace-config",
        action="store_true",
        help="Allow replacing a slot's existing configuration.",
    )
    return parser


def resolve_target_slot(
    config: ExperimentConfig,
    *,
    slot_argument: str | None,
    variant: str,
    schema_version: str | None,
    root: str | None = None,
) -> ModelSlot:
    """The slot to promote into: inferred from the run, checked against --slot.

    A disagreement is an error rather than a silent override. The run's config
    is what the experiment actually measured; if the two differ, one of them is
    wrong and guessing which would defeat the point of recording either.
    """
    inferred = slot_from_config(
        config, variant=variant, schema_version=schema_version, root=root
    )
    if slot_argument is None:
        return inferred
    given = parse_slot_argument(slot_argument, root=root)
    if given != inferred:
        raise PromotionError(
            f"--slot says {given.describe()} but the run's config implies "
            f"{inferred.describe()}. Fix the config or the flag; they must agree."
        )
    return given


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.from_best:
        hyperparameters = find_best_run_hyperparameters(args.from_best)
        run_dir = hyperparameters.run_dir
        print(f"Best run under {args.from_best}: {run_dir.name}")
    else:
        run_dir = Path(args.run_dir)
        hyperparameters = load_run_hyperparameters(run_dir)

    config = load_run_config(run_dir)
    csv_path = args.csv or config.data.csv_path

    print(f"Source run     : {run_dir}")
    print(f"  trial {hyperparameters.trial_number} ({hyperparameters.source})", end="")
    if hyperparameters.cv_mae is not None:
        print(f", CV MAE {hyperparameters.cv_mae:.4f}", end="")
    print()
    print(f"  params       : {hyperparameters.params}")
    print(f"  n_estimators : {hyperparameters.n_estimators}")
    print(f"  weight lambda: {hyperparameters.sample_weight_lambda}")
    print(f"Target         : {config.family.value}")
    print(f"Training data  : {csv_path}")
    selected_train_games = (
        hyperparameters.train_games
        if hyperparameters.train_games is not None
        else config.walk_forward.train_games
    )
    window_label = (
        "full dataset"
        if args.full_dataset or config.refit.strategy == RefitStrategy.FULL_DATASET
        else f"{selected_train_games} games"
    )
    print(f"Window         : {window_label} ({config.refit.strategy.value})")

    if args.dry_run:
        print("\n--dry-run: nothing trained or saved.")
        return

    result = train_production_model_from_run(
        run_dir,
        csv_path=args.csv,
        training_version=args.training_version,
        expected_checksum=args.expected_checksum,
        refit_strategy=RefitStrategy.FULL_DATASET if args.full_dataset else None,
        overwrite=args.overwrite,
    )

    print()
    print(f"Trained on {result.n_train_games} games "
          f"({result.train_date_min.date()} .. {result.train_date_max.date()}), "
          f"{result.n_features} features")
    print(f"Dataset checksum: {result.dataset_checksum}")
    print(f"Model    : {result.model_path}")
    print(f"Metadata : {result.meta_path}")
    print(
        "\nNote: this model is unevaluated by construction. The metrics in its "
        f"metadata were measured by {run_dir.name} on that run's own data."
    )

    if not args.to_s3:
        return

    assert result.config is not None
    slot = resolve_target_slot(
        result.config,
        slot_argument=args.slot,
        variant=args.variant,
        schema_version=args.schema_version,
    )
    check_schema_version_against_checkout(slot.schema_version)
    check_horizon_is_buildable(slot.horizon_minutes)

    promotion = publish_config_to_registry(
        result,
        slot=slot,
        replace_config=args.replace_config,
        promoted_by=_current_user(),
    )

    print()
    print(f"Registry slot  : {promotion.slot.describe()}")
    print(f"  bucket       : {promotion.bucket}")
    print(f"  spec_id      : {promotion.spec.spec_id}"
          f"{' (replaced previous configuration)' if promotion.replaced else ''}")
    print(f"  staged build : {promotion.fit.fit_id}")
    print(f"  model_name   : {promotion.fit.model_name}")
    print("\nProduction is unchanged. Promote the staged build with:")
    print(f"    python -m scripts.promote_build --slot {promotion.slot.describe()} --execute")
    print("\nEnable the slot by adding this row to [PredictionModels] ENABLED_MODELS:")
    print(enabled_models_row(promotion.slot))


if __name__ == "__main__":
    main()
