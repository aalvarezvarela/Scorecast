"""Where generated training datasets live in S3.

The bucket grew a flat ``train_data/`` prefix whose filenames say when a build
cut off but not what it contains. Its sibling
``scripts/create_train_data/create_train_data.py`` already writes the schema
version into every local filename, and both scripts call the same
``create_df_to_predict``: the same frame was being recorded two different ways
depending on where it landed.

So the prefix gains the version as its first segment, exactly as the model
registry does::

    train_data/2_5/historical_training_data_2_5_20260215.parquet

The six files that predate all of this keep their original names under
``1_0/``. That is a label, not a claim: they were built before
``TRAINING_DATA_SCHEMA_VERSION`` existed, so no real version can be recovered
for them, and ``1_0`` says "older than every schema we can name" without
pretending to know which one.
"""

from __future__ import annotations

import re

#: Root prefix for generated training datasets.
TRAIN_DATA_PREFIX = "train_data/"

#: Bucket for everything built before schema versions were recorded.
LEGACY_SCHEMA_VERSION = "1_0"

#: A version segment, e.g. "2_5". Matches the local CSV naming convention and
#: training_pipeline.registry.parse_schema_version.
SCHEMA_VERSION_RE = re.compile(r"^\d+_\d+$")


def train_data_key(*, schema_version: str, filename: str, root: str = TRAIN_DATA_PREFIX) -> str:
    """Key for a dataset of a known schema version."""
    if not SCHEMA_VERSION_RE.match(schema_version):
        raise ValueError(
            f"schema_version must look like '2_5'. Got {schema_version!r}."
        )
    return f"{root}{schema_version}/{filename}"


def historical_filename(*, schema_version: str, limit_date: str) -> str:
    """``historical_training_data_2_5_20260215.parquet``.

    The version sits immediately before an 8-digit date so the name matches
    ``training_pipeline.registry.parse_schema_version``, the same way the local
    CSVs do. ``limit_date`` is the training cutoff, not the build date.
    """
    return f"historical_training_data_{schema_version}_{limit_date}.parquet"


def is_unversioned_key(key: str, *, root: str = TRAIN_DATA_PREFIX) -> bool:
    """True for a dataset sitting loose at the root of the prefix.

    Deliberately narrower than "not a schema version": anything inside a
    subfolder is left alone, whatever it is called. The prefix is also an
    offered home for unrelated archives -- ``train_data/injury_reports`` is
    suggested by the injury-report storage helper -- and a sweep that moved
    those because their name is not ``2_5`` would be a bug that only shows up
    on whichever machine had run that import.
    """
    if not key.startswith(root):
        return False
    remainder = key[len(root) :]
    if not remainder:
        # The zero-byte placeholder standing for the prefix itself.
        return False
    return "/" not in remainder
