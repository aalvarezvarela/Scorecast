"""Build a newer schema version of a dataset file from an older one.

``nba_ou.create_training_data.schema_layers`` computes the columns a version
adds; this module does the file side. It reads only the parent's row keys and
the few columns the layers need, computes the new columns, and writes the
parent plus those columns as a new Parquet file:

* **Parquet** parent (every build since datasets went Parquet-only): row group
  by row group with pyarrow. The 4.6 GB intermediate dataset never sits in
  memory (``parquet_dataset`` measures a full load at ~14 GB), and the parent's
  arrays are copied, not re-encoded, so its columns are bit-identical.
* **CSV** parent (older builds still pinned by configs): loaded the way the
  training loader reads it and written once through
  ``parquet_dataset.write_training_dataset``.

The parent is never modified, the result is checked before it is kept, and a
manifest records what was built from what.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nba_ou.create_training_data.schema_layers import (
    INTERMEDIATE_LINE,
    LayerContext,
    check_version,
    compute_new_columns,
    dataset_type_from_filename,
    required_input_columns,
)
from nba_ou.create_training_data.schema_layers.manifest import (
    build_manifest,
    read_manifest,
    record_file,
    write_manifest,
)

from training_pipeline.data import (
    compute_file_checksum,
    finish_parquet_frame,
    is_parquet,
    load_raw_training_csv,
    read_dataset_columns,
)
from training_pipeline.parquet_dataset import (
    DEFAULT_COMPRESSION,
    write_training_dataset,
)

#: Parquet key-value metadata on a layered file.
SCHEMA_VERSION_KEY = b"nba_ou.schema_version"
PARENT_CHECKSUM_KEY = b"nba_ou.parent_checksum"

#: ``..._2_5_20260704...``: the version segment right before the 8-digit date,
#: as in ``training_pipeline.registry.parse_schema_version``.
_VERSION_SEGMENT = re.compile(r"_(\d+_\d+)_(\d{8})")


def schema_version_from_filename(path: str | Path) -> str:
    match = _VERSION_SEGMENT.search(Path(path).name)
    if match is None:
        raise ValueError(
            f"Cannot read a schema version from {Path(path).name!r}; expected a "
            "name like 'closing_line_data_2_5_20260704.parquet'."
        )
    return match.group(1)


def layered_filename(path: str | Path, to_version: str) -> str:
    """The parent's filename with its version swapped, always ``.parquet``."""
    name = Path(path).with_suffix(".parquet").name
    match = _VERSION_SEGMENT.search(name)
    if match is None:
        raise ValueError(f"No version segment in {name!r}.")
    return f"{name[: match.start(1)]}{to_version}{name[match.end(1) :]}"


def check_layerable_output(path: str | Path, *, base_version: str) -> None:
    """Fail before a long build if its output could not be layered on later.

    The upgrade reads the dataset type and version from the filename, so a
    custom ``--output`` must keep the builders' naming.
    """
    dataset_type_from_filename(path)
    version = schema_version_from_filename(path)
    if version != base_version:
        raise ValueError(
            f"{Path(path).name!r} names version {version}, but the builder "
            f"writes {base_version}."
        )


def dataset_header(path: str | Path) -> list[str]:
    path = Path(path)
    if is_parquet(path):
        return list(pq.read_schema(path).names)
    return list(pd.read_csv(path, nrows=0).columns)


def read_parent_inputs(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """The listed columns that the parent has, typed as the loader types them."""
    header = set(dataset_header(path))
    keep = [c for c in columns if c in header]
    return read_dataset_columns(path, columns=keep).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


def _merged_pandas_metadata(base: pa.Schema, added: pa.Schema) -> bytes | None:
    """The base file's pandas metadata with the new columns listed too.

    Keeps ``to_pandas`` reconstructing the parent's columns exactly as before;
    without it the reader would fall back to inferring them from Arrow types.
    """
    raw_base = (base.metadata or {}).get(b"pandas")
    raw_added = (added.metadata or {}).get(b"pandas")
    if raw_base is None or raw_added is None:
        return raw_base
    meta = json.loads(raw_base)
    added_meta = json.loads(raw_added)
    index_names = {
        entry for entry in added_meta.get("index_columns", []) if isinstance(entry, str)
    }
    meta["columns"] = meta.get("columns", []) + [
        column
        for column in added_meta.get("columns", [])
        if column.get("field_name") not in index_names
    ]
    return json.dumps(meta).encode()


def append_columns_to_parquet(
    parent: Path,
    new_columns: pd.DataFrame,
    out: Path,
    *,
    metadata: dict[bytes, bytes],
    compression: str = DEFAULT_COMPRESSION,
) -> None:
    """Write ``parent`` with ``new_columns`` added, one row group at a time."""
    source = pq.ParquetFile(parent)
    added_table = pa.Table.from_pandas(
        new_columns.reset_index(drop=True), preserve_index=False
    )
    if source.metadata.num_rows != added_table.num_rows:
        raise ValueError(
            f"{parent.name} has {source.metadata.num_rows:,} rows but "
            f"{added_table.num_rows:,} new values were computed."
        )
    base_schema = source.schema_arrow
    # The parent's own nba_ou keys (its version, its parent) describe the
    # parent, not this file.
    base_meta = {
        key: value
        for key, value in (base_schema.metadata or {}).items()
        if not key.startswith(b"nba_ou.")
    }
    pandas_meta = _merged_pandas_metadata(base_schema, added_table.schema)
    if pandas_meta is not None:
        base_meta[b"pandas"] = pandas_meta
    schema = pa.schema(
        list(base_schema) + list(added_table.schema),
        metadata={**base_meta, **metadata},
    )

    tmp = out.with_name(out.name + ".partial")
    offset = 0
    try:
        with pq.ParquetWriter(tmp, schema, compression=compression) as writer:
            for group in range(source.num_row_groups):
                table = source.read_row_group(group)
                piece = added_table.slice(offset, table.num_rows)
                for field, column in zip(piece.schema, piece.columns, strict=True):
                    table = table.append_column(field, column)
                writer.write_table(
                    table.replace_schema_metadata(schema.metadata),
                    row_group_size=table.num_rows,
                )
                offset += table.num_rows

        # Verify before keeping: parent columns unchanged, new ones as computed.
        written = pq.ParquetFile(tmp)
        for group in range(source.num_row_groups):
            original = source.read_row_group(group)
            copy = written.read_row_group(group, columns=original.column_names)
            if not copy.equals(original):
                raise AssertionError(f"Row group {group}: parent columns changed.")
        appended = finish_parquet_frame(
            written.read(columns=list(new_columns.columns)).to_pandas(),
            date_col="GAME_DATE",
        )
        expected = finish_parquet_frame(
            new_columns.reset_index(drop=True), date_col="GAME_DATE"
        )
        pd.testing.assert_frame_equal(appended, expected, check_exact=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)


def append_columns_from_csv(parent: Path, new_columns: pd.DataFrame, out: Path) -> None:
    """An older CSV parent: load it as training does, write parent + new columns."""
    frame = load_raw_training_csv(parent)
    if len(frame) != len(new_columns):
        raise ValueError(
            f"{parent.name} has {len(frame):,} rows but {len(new_columns):,} new "
            "values were computed."
        )
    added = new_columns.reset_index(drop=True)
    added.index = frame.index
    write_training_dataset(pd.concat([frame, added], axis=1), out)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def build_schema_version_file(
    parent: str | Path,
    *,
    to_version: str,
    out_dir: str | Path | None = None,
    ctx: LayerContext | None = None,
    build_args: dict[str, Any] | None = None,
    scoring_path: str | Path | None = None,
    allow_schedule_tipoffs: bool = False,
) -> Path:
    """Write ``parent`` upgraded to ``to_version`` beside it (or in ``out_dir``).

    Returns the Parquet path a config's ``data.csv_path`` should point at.

    An intermediate parent's snapshot times come from ``scoring_path`` or the
    ``<stem>_scoring`` sidecar beside it, which is copied beside the output so
    the next upgrade finds it too. Preloaded context timestamps must agree with
    that sidecar; they do not bypass its copy or provenance. Without either,
    a layer that needs those
    times fails unless ``allow_schedule_tipoffs`` lets it read the live
    line-history schedule. Where the times came from, and digests of the
    injury-report states read, are recorded in the manifest's ``build_args``.
    """
    started = time.time()
    parent = Path(parent)
    check_version(to_version)
    dataset_type = dataset_type_from_filename(parent)
    from_version = schema_version_from_filename(parent)
    parent_manifest = read_manifest(parent)
    if parent_manifest is not None:
        if parent_manifest["schema_version"] != from_version:
            raise ValueError(
                f"{parent.name} is named {from_version} but its manifest says "
                f"{parent_manifest['schema_version']}."
            )
        if parent_manifest["dataset_type"] != dataset_type:
            raise ValueError(
                f"{parent.name} looks like {dataset_type} but its manifest says "
                f"{parent_manifest['dataset_type']}."
            )
    check_version(from_version)

    out = Path(out_dir or parent.parent) / layered_filename(parent, to_version)
    if from_version == to_version or out.resolve() == parent.resolve():
        raise ValueError(f"{parent.name} is already {to_version}.")
    print(f"Upgrading {parent.name} ({from_version} -> {to_version}, {dataset_type})")

    header = dataset_header(parent)
    inputs = read_parent_inputs(
        parent, required_input_columns(from_version, to_version, dataset_type)
    )
    ctx = ctx if ctx is not None else LayerContext()
    ctx.allow_schedule_tipoffs = allow_schedule_tipoffs
    build_args = dict(build_args or {})
    sidecar = None
    if dataset_type == INTERMEDIATE_LINE:
        # The frozen base keeps timestamps in its scoring file, not in X.
        # Discover the file even when a caller has already seeded its times:
        # the output still needs the scoring rows and their provenance.
        if scoring_path is not None:
            sidecar = Path(scoring_path)
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"Snapshot scoring sidecar not found: {sidecar}"
                )
        else:
            sidecar = scoring_sidecar_path(parent)
        if sidecar is not None:
            sidecar_times = read_parent_inputs(
                sidecar,
                ["GAME_ID", "TIME_TO_MATCH_MIN", "TIPOFF_UTC", "SNAPSHOT_TS_UTC"],
            )
            if ctx.snapshot_times is None:
                ctx.snapshot_times = sidecar_times
            else:
                from nba_ou.create_training_data.schema_layers.inputs import (
                    resolve_snapshot_cutoffs,
                )

                seeded_cutoffs = resolve_snapshot_cutoffs(inputs, ctx.snapshot_times)
                sidecar_cutoffs = resolve_snapshot_cutoffs(inputs, sidecar_times)
                if not seeded_cutoffs["as_of"].eq(sidecar_cutoffs["as_of"]).all():
                    raise ValueError(
                        f"Seeded snapshot cutoffs disagree with scoring sidecar: {sidecar}"
                    )
            build_args["snapshot_scoring_path"] = str(sidecar.resolve())
            build_args["snapshot_scoring_checksum"] = compute_file_checksum(sidecar)
    new_columns = compute_new_columns(
        inputs,
        from_version=from_version,
        to_version=to_version,
        dataset_type=dataset_type,
        existing_columns=header,
        ctx=ctx,
    )
    print(f"  +{new_columns.shape[1]} columns over {len(inputs):,} rows")
    build_args.update(ctx.provenance)

    parent_checksum = compute_file_checksum(parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_parquet(parent):
        append_columns_to_parquet(
            parent,
            new_columns,
            out,
            metadata={
                SCHEMA_VERSION_KEY: to_version.encode(),
                PARENT_CHECKSUM_KEY: parent_checksum.encode(),
            },
        )
    else:
        append_columns_from_csv(parent, new_columns, out)

    manifest = build_manifest(
        schema_version=to_version,
        dataset_type=dataset_type,
        n_rows=len(inputs),
        n_columns=len(header) + new_columns.shape[1],
        parent={
            "filename": parent.name,
            "checksum": parent_checksum,
            "schema_version": from_version,
        },
        from_version=from_version,
        build_args=build_args,
    )
    write_manifest(out, manifest)
    record_file(out, compute_file_checksum(out))
    if sidecar is not None:
        # Layers never add or drop rows: the parent's scoring rows are the
        # output's, and an upgrade of the output must find them beside it.
        copied = out.with_name(f"{out.stem}_scoring{sidecar.suffix}")
        if copied.resolve() != sidecar.resolve():
            shutil.copyfile(sidecar, copied)
            print(f"Scoring sidecar: {copied}")
    print(f"Wrote {out} in {time.time() - started:.0f}s")
    return out


def scoring_sidecar_path(dataset: str | Path) -> Path | None:
    """The ``<stem>_scoring`` file beside ``dataset`` (Parquet or CSV), if any."""
    dataset = Path(dataset)
    for suffix in (".parquet", ".csv"):
        candidate = dataset.with_name(f"{dataset.stem}_scoring{suffix}")
        if candidate.is_file():
            return candidate
    return None


def write_base_manifest(
    dataset_path: str | Path,
    *,
    schema_version: str,
    dataset_type: str,
    n_rows: int,
    n_columns: int,
    build_args: dict[str, Any] | None = None,
) -> Path:
    """Manifest for a file the monolithic builder wrote (no parent file)."""
    dataset_path = Path(dataset_path)
    manifest = build_manifest(
        schema_version=schema_version,
        dataset_type=dataset_type,
        n_rows=n_rows,
        n_columns=n_columns,
        build_args=build_args,
    )
    path = write_manifest(dataset_path, manifest)
    record_file(dataset_path, compute_file_checksum(dataset_path))
    return path
