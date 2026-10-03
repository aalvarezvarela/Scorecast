"""Training-data schema versions as layers of added columns.

The base version (``nba_ou.config.dataset_versions.BASE_SCHEMA_VERSION``) is
built end to end by ``create_df_to_predict`` / ``create_intermediate_line_df``.
Each later version only adds columns to its parent, so a 2_6 dataset is a 2_5
dataset plus the 2_6 columns, computable from a 2_5 frame or file without
rebuilding it. See ``registry`` for adding a version and ``contract`` for the
append-only rule.

    python -m nba_ou.create_training_data.schema_layers describe <dataset>
    python -m nba_ou.create_training_data.schema_layers diff 2_5 2_6
"""

from .base import (
    CLOSING_LINE,
    DATASET_TYPES,
    INTERMEDIATE_LINE,
    ROW_KEYS,
    LayerContext,
    SchemaLayer,
    parse_version,
)
from .contract import LayerContractError
from .registry import (
    LAYERS,
    apply_layers,
    available_versions,
    chain,
    check_version,
    column_origin,
    compute_new_columns,
    dataset_type_from_filename,
    is_layered,
    latest_version,
    newest_version,
    required_input_columns,
    version_columns,
)

__all__ = [
    "CLOSING_LINE",
    "DATASET_TYPES",
    "INTERMEDIATE_LINE",
    "LAYERS",
    "ROW_KEYS",
    "LayerContext",
    "LayerContractError",
    "SchemaLayer",
    "apply_layers",
    "available_versions",
    "chain",
    "check_version",
    "column_origin",
    "compute_new_columns",
    "dataset_type_from_filename",
    "is_layered",
    "latest_version",
    "newest_version",
    "parse_version",
    "required_input_columns",
    "version_columns",
]
