"""Readable model families backed by explicit prediction-store metadata."""

from dataclasses import dataclass

import pandas as pd

from nba_ou.dashboard.config import (
    DISPLAY_SCHEMA_VERSION,
    MODEL_NAMES,
    PREDICTION_TARGET_LABELS,
)


@dataclass(frozen=True)
class ModelFamily:
    key: str
    label: str
    kind: int
    closing: bool
    intermediate: bool


def model_name(kind) -> str:
    return MODEL_NAMES.get(int(kind), f"Model kind {kind}")


def family_label(kind, schema_version, variant) -> str:
    # Keep the variant in family_key, but omit TabPFN's redundant v3_5 suffix.
    variant = str(variant).removeprefix("tabpfn_")
    show_variant = variant != "main" and not (int(kind) == 3 and variant == "v3_5")
    return f"{model_name(kind)} · {str(schema_version).replace('_', '.')}" + (
        f" · {variant}" if show_variant else ""
    )


def with_family_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["family_key"] = (
        work["kind"].astype(str)
        + ":"
        + work["schema_version"].astype(str)
        + ":"
        + work["variant"].astype(str)
    )
    work["model_name"] = work["kind"].map(model_name)
    work["target_label"] = work["target"].map(PREDICTION_TARGET_LABELS)
    work["model_label"] = [
        family_label(kind, version, variant)
        for kind, version, variant in zip(
            work["kind"], work["schema_version"], work["variant"], strict=True
        )
    ]
    return work


def displayed_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Always show the 2_5 registry models and any available TabPFN outputs."""
    if frame.empty:
        return frame.copy()
    return frame.loc[
        (frame["kind"].eq(1) & frame["schema_version"].eq(DISPLAY_SCHEMA_VERSION))
        | frame["kind"].eq(3)
    ].copy()


def model_families(
    frame: pd.DataFrame, target: int | tuple[int, ...]
) -> list[ModelFamily]:
    if frame.empty:
        return []
    targets = (target,) if isinstance(target, int) else target
    work = with_family_metadata(frame.loc[frame["target"].isin(targets)])
    return [
        ModelFamily(
            key=key,
            label=group["model_label"].iloc[0],
            kind=int(group["kind"].iloc[0]),
            closing=bool(group["dataset_type"].eq(1).any()),
            intermediate=bool(group["dataset_type"].eq(2).any()),
        )
        for key, group in work.groupby("family_key", sort=True)
    ]
