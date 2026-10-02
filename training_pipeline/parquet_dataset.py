"""Write training datasets as Parquet that loads exactly like the CSV.

The CSV stays the reference. ``convert_csv_to_parquet`` reads it through
``data.load_raw_training_csv`` -- the reader every experiment uses -- so the
Parquet file stores exactly the frame the pipeline sees (ID columns as text,
GAME_DATE as a date), then reads every row group back through the Parquet
loader's own post-read step and compares it to the CSV rows with
``assert_frame_equal`` (exact floats, dtypes and index). A file that fails is
deleted, never left half-trusted.

Why convert from the CSV rather than write the builder's frame directly: the
CSV round trip is what fixes today's dtypes -- ints that turn into text IDs,
dates, how a sparse column is inferred. Writing the in-memory frame would skip
it, and a dataset built as Parquet could load differently from the same build
as CSV with nothing to say so.

Row order is kept, not sorted by horizon: a pooled run reads every row in file
order, and sorting would hand it a different frame. The single-horizon read
stays small without sorting -- it filters one row group at a time.

**Memory:** the whole CSV is loaded once, ~14GB for the 4.6GB intermediate
file. Nothing else should be loading a dataset while it runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from training_pipeline.config import SNAPSHOT_COLUMN
from training_pipeline.data import (
    SOURCE_CSV_CHECKSUM_KEY,
    _read_parquet_rows,
    compute_file_checksum,
    finish_parquet_frame,
    load_raw_training_csv,
    parquet_source_checksum,
)

DEFAULT_ROW_GROUP_SIZE = 10_000
DEFAULT_COMPRESSION = "zstd"
#: What the dataset builders can write. ``both`` keeps the CSV beside the
#: Parquet file, e.g. for a tool that still reads CSV.
OUTPUT_FORMATS = ("csv", "parquet", "both")


class ParquetConversionError(ValueError):
    """The CSV has no faithful Parquet representation."""


def _schema_or_explain(df: pd.DataFrame) -> pa.Schema:
    """The Arrow schema, or an error naming every column that has none.

    A column holding both numbers and text (pandas' low_memory inference can
    produce one) has no Parquet type. Name them all rather than fail on the first.
    """
    try:
        return pa.Schema.from_pandas(df, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        bad = []
        for col in df.columns:
            try:
                pa.Schema.from_pandas(df[[col]], preserve_index=False)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                kinds = sorted({type(v).__name__ for v in df[col].dropna()})
                bad.append(f"{col} ({', '.join(kinds)})")
        raise ParquetConversionError(
            "These columns mix types and cannot be stored as Parquet:\n  "
            + "\n  ".join(bad)
        ) from None


def convert_csv_to_parquet(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    *,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    compression: str = DEFAULT_COMPRESSION,
    date_col: str = "GAME_DATE",
) -> Path:
    """Write ``csv_path`` as Parquet, verified row group by row group.

    ``out_path`` defaults to the CSV's path with a ``.parquet`` suffix. Returns
    the path written.
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path) if out_path is not None else csv_path.with_suffix(".parquet")
    started = time.time()
    # Recorded in the file: data.resolve_dataset_source substitutes this copy
    # for the CSV only when the checksum a config pins equals this one.
    csv_checksum = compute_file_checksum(csv_path)
    print(f"Reading {csv_path} ({csv_path.stat().st_size / 1e9:.2f} GB) ...", flush=True)
    df = load_raw_training_csv(csv_path, date_col=date_col)
    print(f"  {len(df):,} rows x {df.shape[1]:,} columns in {time.time() - started:.0f}s")

    schema = _schema_or_explain(df)
    schema = schema.with_metadata(
        {**(schema.metadata or {}), SOURCE_CSV_CHECKSUM_KEY: csv_checksum.encode()}
    )
    tmp_path = out_path.with_name(out_path.name + ".partial")
    print(f"Writing {tmp_path.name} ({compression}, {row_group_size:,}-row groups) ...")
    try:
        with pq.ParquetWriter(tmp_path, schema, compression=compression) as writer:
            for start in range(0, len(df), row_group_size):
                chunk = df.iloc[start : start + row_group_size]
                writer.write_table(
                    pa.Table.from_pandas(chunk, schema=schema, preserve_index=False),
                    row_group_size=row_group_size,
                )

        print("Verifying every row group against the CSV ...", flush=True)
        parquet = pq.ParquetFile(tmp_path)
        offset = 0
        for group in range(parquet.num_row_groups):
            frame = finish_parquet_frame(
                parquet.read_row_group(group).to_pandas(), date_col=date_col
            )
            frame.index = pd.RangeIndex(offset, offset + len(frame))
            pd.testing.assert_frame_equal(
                frame, df.iloc[offset : offset + len(frame)], check_exact=True
            )
            offset += len(frame)
        if offset != len(df):
            raise AssertionError(f"Parquet holds {offset:,} rows, the CSV {len(df):,}.")
        if parquet_source_checksum(tmp_path) != csv_checksum:
            raise AssertionError("The source CSV checksum was not recorded in the file.")

        # The single-horizon read is a different code path (row-group filter,
        # file-position index), so check it on one horizon too. Called directly
        # because the file is still named .partial, which the loader would
        # take for a CSV.
        if SNAPSHOT_COLUMN in df.columns:
            horizons = pd.to_numeric(df[SNAPSHOT_COLUMN], errors="coerce")
            probe = int(horizons.dropna().iloc[0])
            got = finish_parquet_frame(
                _read_parquet_rows(tmp_path, row_filter=(SNAPSHOT_COLUMN, probe)),
                date_col=date_col,
            )
            pd.testing.assert_frame_equal(
                got, df.loc[horizons == probe], check_exact=True
            )
            print(f"  single-horizon read at T-{probe}: {len(got):,} rows, identical")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(out_path)
    ratio = csv_path.stat().st_size / out_path.stat().st_size
    print(
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB, "
        f"{ratio:.1f}x smaller than the CSV) in {time.time() - started:.0f}s"
    )
    return out_path


def finish_dataset_output(
    csv_path: str | Path, *, output_format: str = "csv"
) -> Path:
    """The last step of a dataset build: convert if asked, print what to pin.

    ``csv_path`` must already be written. ``parquet`` converts it and deletes
    the CSV once the Parquet file has verified; ``both`` keeps it. Returns the
    path a config's ``data.csv_path`` should point at.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}")
    csv_path = Path(csv_path)
    if output_format == "csv":
        print(f'expected_checksum: "{compute_file_checksum(csv_path)}"')
        return csv_path

    parquet_path = convert_csv_to_parquet(csv_path)
    source_checksum = parquet_source_checksum(parquet_path)
    if output_format == "parquet":
        csv_path.unlink()
        print(f"Removed {csv_path.name}; the verified Parquet file replaces it.")
    # Either spelling works in a config. The CSV one resolves to the Parquet
    # copy through its recorded checksum (data.resolve_dataset_source), even
    # after the CSV is gone -- and matches how every existing config is written.
    print(f"data.csv_path: {csv_path}")
    print(f'expected_checksum: "{source_checksum}"')
    print(f"  (or data.csv_path: {parquet_path}")
    print(f'   expected_checksum: "{compute_file_checksum(parquet_path)}")')
    return parquet_path
