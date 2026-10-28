"""Readable model families backed by explicit prediction-store metadata."""

from dataclasses import dataclass

import pandas as pd

from nba_ou.dashboard.config import MODEL_NAMES


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
    # TabPFN variants repeat the model name (``tabpfn_v3_5``).
    variant = str(variant).removeprefix("tabpfn_")
    return f"{model_name(kind)} · {str(schema_version).replace('_', '.')}" + (
        f" · {variant}" if variant != "main" else ""
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
    work["model_label"] = [
        family_label(kind, version, variant)
        for kind, version, variant in zip(
            work["kind"], work["schema_version"], work["variant"], strict=True
        )
    ]
    return work


def model_families(frame: pd.DataFrame, target: int) -> list[ModelFamily]:
    if frame.empty:
        return []
    work = with_family_metadata(frame.loc[frame["target"].eq(target)])
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
