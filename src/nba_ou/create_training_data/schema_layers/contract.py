"""The append-only rule, checked every time a layer runs.

A layer may add columns and nothing else. Its output must line up row for row
with what it was given, carry exactly the columns it declared, collide with
nothing already in the dataset, and follow the leakage naming the rest of the
pipeline relies on (``docs/README_Training Data Processing.md``). Checked at
run time rather than only in tests because a layer's output depends on data --
a join that duplicates a key shows up on the build where it happens.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from nba_ou.config.leakage import rotation_leak_columns

from .base import SchemaLayer


class LayerContractError(ValueError):
    """A layer's output would do more than add its declared columns."""


def check_declared_columns(layer: SchemaLayer, dataset_type: str) -> None:
    """Names alone: unique, pre-game, and not a known leak family."""
    declared = layer.columns_for(dataset_type)
    duplicates = sorted({c for c in declared if declared.count(c) > 1})
    if duplicates:
        raise LayerContractError(f"Layer {layer.version} declares {duplicates} twice.")
    not_pregame = [c for c in declared if "_BEFORE" not in c]
    if not_pregame:
        raise LayerContractError(
            f"Layer {layer.version} declares columns without _BEFORE: {not_pregame}. "
            "Every new column must be a leakage-safe pre-game feature."
        )
    leaks = rotation_leak_columns(declared)
    if leaks:
        raise LayerContractError(
            f"Layer {layer.version} declares known leak families: {leaks}."
        )


def check_layer_output(
    layer: SchemaLayer,
    dataset_type: str,
    *,
    inputs: pd.DataFrame,
    output: pd.DataFrame,
    existing_columns: Iterable[str],
) -> pd.DataFrame:
    """Validate ``output`` and return it with columns in declared order."""
    declared = layer.columns_for(dataset_type)
    check_declared_columns(layer, dataset_type)

    if len(output) != len(inputs) or not output.index.equals(inputs.index):
        raise LayerContractError(
            f"Layer {layer.version} returned {len(output):,} rows for "
            f"{len(inputs):,} inputs, or a different index. A layer must return "
            "one row per input row, in the same order."
        )
    produced = list(output.columns)
    missing = [c for c in declared if c not in produced]
    undeclared = [c for c in produced if c not in declared]
    if missing or undeclared:
        raise LayerContractError(
            f"Layer {layer.version} ({dataset_type}): missing declared columns "
            f"{missing}, undeclared columns {undeclared}."
        )
    collisions = sorted(set(declared) & set(existing_columns))
    if collisions:
        raise LayerContractError(
            f"Layer {layer.version} would overwrite existing columns {collisions}. "
            "Layers only add columns; changing an existing one needs a new base."
        )
    return output[list(declared)]
