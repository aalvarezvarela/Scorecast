"""Manual prediction requests: the model choices and the dispatch each needs.

A request runs in a GitHub Action on the self-hosted runner, never in the
dashboard (``docs/serving/manual_predictions.md``). The dashboard stores it
as ``queued`` and dispatches the workflow with its id; the run marks it
``running`` and then ``done`` / ``partial`` / ``failed``.

Every request covers the whole slate (the games not yet started) and reads
each game's latest snapshot. An intermediate request chooses, by default, the
model (XGBoost) or training rows (TabPFN) of each game's nearest horizon, or
one fixed horizon for every game. A run predicts only the request's targets
(line error by default) and skips the database update.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from nba_ou.dashboard.config import PREDICTION_TARGET_LABELS
from nba_ou.dashboard.domain.dates import horizon_label
from nba_ou.postgre_db.app_aiven.schema import (
    REQUEST_QUEUED,
    REQUEST_RUNNING,
    REQUEST_STATUS_NAMES,
)
from nba_ou.postgre_db.predictions_aiven.schema import TARGET_CODES
from nba_ou.prediction.horizon_choice import nearest_horizon

PREDICTOR_WORKFLOW = "nba_predictor_daily.yml"
TABPFN_WORKFLOW = "nba_predictor_daily_tabpfn_client.yml"
KIND_XGBOOST = 1
KIND_TABPFN = 3
CLOSING = 1
INTERMEDIATE = 2
#: The intermediate horizons a request may fix (``ENABLED_MODELS``).
HORIZONS = (
    30,
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720,
    840,
    960,
    1080,
)
#: A request still queued after this is shown as expired: its job never ran
#: (predictions switched off, or the dispatch was lost).
QUEUE_EXPIRY = timedelta(hours=3)
#: Rough warm run times on the runner, for the request list.
TYPICAL_MINUTES = {
    (KIND_XGBOOST, CLOSING): "8-12",
    (KIND_XGBOOST, INTERMEDIATE): "5-8",
    (KIND_TABPFN, CLOSING): "6-10",
    (KIND_TABPFN, INTERMEDIATE): "8-20",
}


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    kind: int
    dataset_type: int


MODEL_CHOICES = (
    ModelChoice(
        "tabpfn_intermediate", "TabPFN intermediate", KIND_TABPFN, INTERMEDIATE
    ),
    ModelChoice("tabpfn_closing", "TabPFN closing", KIND_TABPFN, CLOSING),
    ModelChoice(
        "xgboost_intermediate", "XGBoost intermediate", KIND_XGBOOST, INTERMEDIATE
    ),
    ModelChoice("xgboost_closing", "XGBoost closing", KIND_XGBOOST, CLOSING),
)
CHOICES_BY_KEY = {choice.key: choice for choice in MODEL_CHOICES}
DEFAULT_CHOICES = ("tabpfn_intermediate",)
#: The targets a request may predict, as the workflows name them.
TARGETS = tuple(TARGET_CODES)
DEFAULT_TARGETS = ("line_error",)


def choice_of(kind: int, dataset_type: int) -> ModelChoice:
    for choice in MODEL_CHOICES:
        if (choice.kind, choice.dataset_type) == (kind, dataset_type):
            return choice
    raise ValueError(f"No model choice for kind {kind}, dataset type {dataset_type}")


def target_label(target: str) -> str:
    return PREDICTION_TARGET_LABELS[TARGET_CODES[target]]


def ordered_targets(targets: Sequence[str]) -> tuple[str, ...]:
    """``targets`` in the canonical order, however they were picked."""
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown target(s): {', '.join(sorted(unknown))}")
    if not targets:
        raise ValueError("A request needs at least one target.")
    return tuple(target for target in TARGETS if target in targets)


def horizon_choice_label(minutes: int | None) -> str:
    """``T−6h``; None is each game's nearest horizon."""
    return "Each game's own horizon" if minutes is None else horizon_label(minutes)


def request_label(kind: int, dataset_type: int, horizon: int | None) -> str:
    label = choice_of(kind, dataset_type).label
    if dataset_type == INTERMEDIATE:
        label += f" · {'per game' if horizon is None else horizon_label(horizon)}"
    return label


def dispatch_inputs(
    request_id: int,
    kind: int,
    dataset_type: int,
    horizon: int | None,
    targets: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """The workflow file and ``workflow_dispatch`` inputs of a request."""
    if dataset_type == CLOSING and horizon is not None:
        raise ValueError("A closing request has no horizon.")
    if horizon is not None and int(horizon) not in HORIZONS:
        raise ValueError(f"T-{horizon} is not an enabled horizon.")
    rid = str(int(request_id))
    names = " ".join(ordered_targets(targets))
    if kind == KIND_XGBOOST:
        if dataset_type == CLOSING:
            slots = "closing"
        else:
            slots = "nearest" if horizon is None else str(int(horizon))
        return PREDICTOR_WORKFLOW, {
            "request_id": rid,
            "slots": slots,
            "targets": names,
            "update_db": "false",
        }
    if kind == KIND_TABPFN:
        inputs = {"request_id": rid, "update_db": "false", "target": names}
        if dataset_type == CLOSING:
            inputs["line"] = "closing"
        else:
            inputs.update(
                line="intermediate",
                input="latest",
                horizon_minutes="nearest" if horizon is None else str(int(horizon)),
            )
        return TABPFN_WORKFLOW, inputs
    raise ValueError(f"Unknown model kind {kind!r}")


def status_label(status: int, requested_at: datetime, now: datetime) -> str:
    if status == REQUEST_QUEUED and now - requested_at > QUEUE_EXPIRY:
        return "expired"
    return REQUEST_STATUS_NAMES.get(int(status), "unknown")


def is_open(status: int, requested_at: datetime, now: datetime) -> bool:
    """Still waiting for its run (queued, not expired, or running)."""
    return status_label(status, requested_at, now) in (
        REQUEST_STATUS_NAMES[REQUEST_QUEUED],
        REQUEST_STATUS_NAMES[REQUEST_RUNNING],
    )


def preview_horizons(
    minutes_left: dict[str, float], horizons: Sequence[int] = HORIZONS
) -> dict[str, int]:
    """Per game, the horizon a per-game request would use now (indicative:
    the run decides when it starts)."""
    return {
        game: nearest_horizon(minutes, horizons)
        for game, minutes in minutes_left.items()
        if minutes > 0
    }
