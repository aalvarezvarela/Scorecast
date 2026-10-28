"""Readable model families backed by explicit prediction-store metadata."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelFamily:
    key: str
    label: str
    closing: bool
    intermediate: bool


def with_family_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["family_key"] = (
        work["kind"].astype(str)
        + ":"
        + work["schema_version"].astype(str)
        + ":"
        + work["variant"].astype(str)
    )
    work["model_label"] = [
        f"{'TabPFN' if int(kind) == 3 else 'Registry model'} · "
        f"{str(version).replace('_', '.')}"
        + (f" · {variant}" if variant != "main" else "")
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
            closing=bool(group["dataset_type"].eq(1).any()),
            intermediate=bool(group["dataset_type"].eq(2).any()),
        )
        for key, group in work.groupby("family_key", sort=True)
    ]
