"""Which schema versions exist, how they chain, and how to apply them.

``BASE_SCHEMA_VERSION`` is what the monolithic builders (``create_df_to_predict``
and ``create_intermediate_line_df``) produce. Every later version is a
:class:`~.base.SchemaLayer` over its parent, so any version can be reached from
any earlier one by running the layers in between -- from a frame in memory
(:func:`apply_layers`) or from a dataset file
(``training_pipeline.layered_dataset``).

Adding a version: write ``vX_Y.py`` with one ``SchemaLayer`` whose ``parent`` is
the current latest, append it to ``_ORDERED_LAYERS``, bump
``TRAINING_DATA_SCHEMA_VERSION`` and add a history entry in
``nba_ou.config.dataset_versions``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION

from .base import (
    CLOSING_LINE,
    INTERMEDIATE_LINE,
    ROW_KEYS,
    LayerContext,
    SchemaLayer,
    check_dataset_type,
    parse_version,
)
from .contract import check_layer_output
from .v2_6 import LAYER_2_6

#: Oldest first; each layer's parent is the one before it (or the base).
_ORDERED_LAYERS: tuple[SchemaLayer, ...] = (LAYER_2_6,)

LAYERS: dict[str, SchemaLayer] = {layer.version: layer for layer in _ORDERED_LAYERS}


def _check_registry() -> None:
    expected_parent = BASE_SCHEMA_VERSION
    for layer in _ORDERED_LAYERS:
        if layer.parent != expected_parent:
            raise RuntimeError(
                f"Layer {layer.version} names parent {layer.parent}, but the "
                f"version before it is {expected_parent}."
            )
        expected_parent = layer.version


_check_registry()


def available_versions() -> tuple[str, ...]:
    """Every version this checkout can build, oldest first."""
    return (BASE_SCHEMA_VERSION, *LAYERS)


def latest_version() -> str:
    return available_versions()[-1]


def check_version(version: str) -> str:
    if version not in available_versions():
        raise ValueError(
            f"Schema version {version!r} is not buildable from this checkout. "
            f"Available: {', '.join(available_versions())}."
        )
    return version


def is_layered(version: str) -> bool:
    """True for a version above the base: reached by applying layers."""
    return parse_version(version) > parse_version(BASE_SCHEMA_VERSION)


def newest_version(versions: Iterable[str]) -> str:
    """The newest of ``versions``, or the base when none is above it.

    What one frame must be built at to serve every version listed: layers only
    add columns, so the newest version's frame holds every older one's.
    Versions older than the base are served by the base frame, as before
    layers existed.
    """
    layered = [v for v in versions if is_layered(v)]
    if not layered:
        return BASE_SCHEMA_VERSION
    return check_version(max(layered, key=parse_version))


def chain(from_version: str, to_version: str) -> list[SchemaLayer]:
    """The layers that turn ``from_version`` into ``to_version``, in order."""
    check_version(from_version)
    check_version(to_version)
    if parse_version(to_version) < parse_version(from_version):
        raise ValueError(
            f"Cannot go from {from_version} down to {to_version}: layers only add "
            "columns. Build the older version from its own base instead."
        )
    versions = available_versions()
    start = versions.index(from_version) + 1
    end = versions.index(to_version) + 1
    return [LAYERS[v] for v in versions[start:end]]


def required_input_columns(
    from_version: str, to_version: str, dataset_type: str
) -> list[str]:
    """Columns of the parent dataset a run of these layers reads (keys first).

    Includes each layer's optional columns; a caller reading a file should keep
    only the ones present.
    """
    columns = list(ROW_KEYS[check_dataset_type(dataset_type)])
    produced: set[str] = set()
    for layer in chain(from_version, to_version):
        for column in (*layer.requires, *layer.optional):
            if column not in columns and column not in produced:
                columns.append(column)
        produced.update(layer.columns_for(dataset_type))
    return columns


def compute_new_columns(
    inputs: pd.DataFrame,
    *,
    from_version: str,
    to_version: str,
    dataset_type: str,
    existing_columns: Iterable[str],
    ctx: LayerContext | None = None,
) -> pd.DataFrame:
    """Only the columns ``to_version`` adds over ``from_version``.

    ``inputs`` needs the row keys and the layers' ``requires`` columns (see
    :func:`required_input_columns`); ``existing_columns`` is every column of the
    parent dataset, for the collision check. A later layer may require a column
    an earlier one in the same chain produced.
    """
    check_dataset_type(dataset_type)
    ctx = ctx if ctx is not None else LayerContext()
    existing = set(existing_columns)
    available = inputs
    added: list[pd.DataFrame] = []
    for layer in chain(from_version, to_version):
        missing = [c for c in layer.requires if c not in available.columns]
        if missing:
            raise ValueError(
                f"Layer {layer.version} needs columns {missing} from its parent."
            )
        if not layer.columns_for(dataset_type):
            continue
        read = [
            c
            for c in (*ROW_KEYS[dataset_type], *layer.requires, *layer.optional)
            if c in available.columns
        ]
        print(f"Computing schema {layer.version} columns ({dataset_type}) ...")
        output = layer.build(available[read].copy(), ctx, dataset_type)
        output = check_layer_output(
            layer,
            dataset_type,
            inputs=available,
            output=output,
            existing_columns=existing,
        )
        existing.update(output.columns)
        available = pd.concat([available, output], axis=1)
        added.append(output)
    if not added:
        return pd.DataFrame(index=inputs.index)
    return pd.concat(added, axis=1)


def apply_layers(
    df: pd.DataFrame,
    *,
    from_version: str = BASE_SCHEMA_VERSION,
    to_version: str,
    dataset_type: str,
    ctx: LayerContext | None = None,
) -> pd.DataFrame:
    """``df`` with every column ``to_version`` adds appended on the right."""
    new_columns = compute_new_columns(
        df,
        from_version=from_version,
        to_version=to_version,
        dataset_type=dataset_type,
        existing_columns=df.columns,
        ctx=ctx,
    )
    if not len(new_columns.columns):
        return df
    return pd.concat([df, new_columns], axis=1)


def version_columns(version: str, dataset_type: str) -> tuple[str, ...]:
    """Columns ``version`` itself introduced. Empty for the base."""
    check_version(version)
    if version == BASE_SCHEMA_VERSION:
        return ()
    return LAYERS[version].columns_for(dataset_type)


def column_origin(columns: Iterable[str], dataset_type: str) -> dict[str, str]:
    """Version that introduced each column; the base for anything not layered."""
    introduced = {
        column: layer.version
        for layer in _ORDERED_LAYERS
        for column in layer.columns_for(dataset_type)
    }
    return {column: introduced.get(column, BASE_SCHEMA_VERSION) for column in columns}


#: Filename prefixes the builders write, newest naming first.
_FILENAME_PREFIXES = {
    "closing_line_data_": CLOSING_LINE,
    "training_data_": CLOSING_LINE,
    "historical_training_data_": CLOSING_LINE,
    "intermediate_line_data_": INTERMEDIATE_LINE,
}


def dataset_type_from_filename(path: str | Path) -> str:
    name = Path(path).name
    for prefix, dataset_type in sorted(
        _FILENAME_PREFIXES.items(), key=lambda item: -len(item[0])
    ):
        if name.startswith(prefix):
            return dataset_type
    raise ValueError(
        f"Cannot tell the dataset type of {name!r}; expected a name starting with "
        f"{', '.join(_FILENAME_PREFIXES)}."
    )
