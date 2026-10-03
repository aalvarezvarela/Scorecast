"""Write training datasets as Parquet -- the one format datasets are stored in.

``write_training_dataset`` is what every builder calls. It stores the frame the
pipeline will see: identifier columns as text, missing text as NaN and the game
date as a date (``data.finish_parquet_frame``, the same step the loader applies
after reading). Every row group is then read back through that step and
compared to the frame with ``assert_frame_equal`` (exact floats, dtypes and
index). A file that fails is deleted, never left half-trusted.

There is no CSV beside it and no metadata tying it to one. A config names the
``.parquet`` file and pins its checksum, exactly as configs used to pin a CSV's.
CSV builds made before this are still readable (``data.load_raw_training_csv``)
and can be moved over with ``convert_csv_to_parquet``, which writes a new file
with its own checksum -- configs that should read it must be repointed.

Rows are written in frame order, not sorted by horizon: a pooled run reads every
row in file order, and sorting would hand it a different frame. The
single-horizon read stays small without sorting -- it filters one row group at a
time.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from training_pipeline.config import SNAPSHOT_COLUMN
from training_pipeline.data import (
    _read_parquet_rows,
    compute_file_checksum,
    finish_parquet_frame,
    is_parquet,
    load_raw_training_csv,
)

DEFAULT_ROW_GROUP_SIZE = 10_000
DEFAULT_COMPRESSION = "zstd"


class ParquetConversionError(ValueError):
    """The frame has no faithful Parquet representation."""


def _schema_or_explain(df: pd.DataFrame) -> pa.Schema:
    """The Arrow schema, or an error naming every column that has none.

    A column holding both numbers and text has no Parquet type. Name them all
    rather than fail on the first.
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


def write_training_dataset(
    df: pd.DataFrame,
    path: str | Path,
    *,
    date_col: str = "GAME_DATE",
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    compression: str = DEFAULT_COMPRESSION,
) -> str:
    """Write ``df`` to ``path`` as Parquet, verified; return the file's checksum.

    ``df`` is not modified, and is not copied either beyond the few columns the
    normalization retypes -- the intermediate frame is several GB.
    """
    path = Path(path)
    if not is_parquet(path):
        raise ValueError(f"Training datasets are written as Parquet; got {path}.")
    started = time.time()
    frame = finish_parquet_frame(df, date_col=date_col)
    # Positions, as a read hands them back. Set on the shallow copy that
    # finish_parquet_frame returned -- reset_index would copy every column.
    frame.index = pd.RangeIndex(len(frame))
    schema = _schema_or_explain(frame)
    tmp_path = path.with_name(path.name + ".partial")
    print(
        f"Writing {len(frame):,} rows x {frame.shape[1]:,} columns to {path} "
        f"({compression}, {row_group_size:,}-row groups) ...",
        flush=True,
    )
    try:
        with pq.ParquetWriter(tmp_path, schema, compression=compression) as writer:
            for start in range(0, len(frame), row_group_size):
                chunk = frame.iloc[start : start + row_group_size]
                writer.write_table(
                    pa.Table.from_pandas(chunk, schema=schema, preserve_index=False),
                    row_group_size=row_group_size,
                )
        _verify(tmp_path, frame, date_col=date_col)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(path)
    checksum = compute_file_checksum(path)
    print(
        f"Wrote {path} ({path.stat().st_size / 1e6:.0f} MB) in "
        f"{time.time() - started:.0f}s. To train on it:\n"
        f"  data.csv_path: {path}\n"
        f'  data.expected_checksum: "{checksum}"'
    )
    return checksum


def _verify(path: Path, frame: pd.DataFrame, *, date_col: str) -> None:
    """Every row group read back equals the frame that was written."""
    parquet = pq.ParquetFile(path)
    offset = 0
    for group in range(parquet.num_row_groups):
        got = finish_parquet_frame(
            parquet.read_row_group(group).to_pandas(), date_col=date_col
        )
        got.index = pd.RangeIndex(offset, offset + len(got))
        pd.testing.assert_frame_equal(
            got, frame.iloc[offset : offset + len(got)], check_exact=True
        )
        offset += len(got)
    if offset != len(frame):
        raise AssertionError(
            f"Parquet holds {offset:,} rows, the frame {len(frame):,}."
        )

    # The single-horizon read is a different code path (row-group filter,
    # file-position index), so check it on one horizon too. Called directly
    # because the file is still named .partial, which the loader would take
    # for a CSV.
    if SNAPSHOT_COLUMN in frame.columns:
        horizons = pd.to_numeric(frame[SNAPSHOT_COLUMN], errors="coerce")
        probe = int(horizons.dropna().iloc[0])
        got = finish_parquet_frame(
            _read_parquet_rows(path, row_filter=(SNAPSHOT_COLUMN, probe)),
            date_col=date_col,
        )
        pd.testing.assert_frame_equal(
            got, frame.loc[horizons == probe], check_exact=True
        )
        print(f"  single-horizon read at T-{probe}: {len(got):,} rows, identical")


def convert_csv_to_parquet(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    *,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    compression: str = DEFAULT_COMPRESSION,
    date_col: str = "GAME_DATE",
) -> Path:
    """Move a CSV build to Parquet: read it as the pipeline would, write it.

    ``out_path`` defaults to the CSV's path with a ``.parquet`` suffix. The
    result is a new file with its own checksum; nothing links it to the CSV.

    **Memory:** the whole CSV is loaded once, ~14GB for the 4.6GB intermediate
    file. Nothing else should be loading a dataset while it runs.
    """
    csv_path = Path(csv_path)
    out_path = (
        Path(out_path) if out_path is not None else csv_path.with_suffix(".parquet")
    )
    print(
        f"Reading {csv_path} ({csv_path.stat().st_size / 1e9:.2f} GB) ...", flush=True
    )
    df = load_raw_training_csv(csv_path, date_col=date_col)
    write_training_dataset(
        df,
        out_path,
        date_col=date_col,
        row_group_size=row_group_size,
        compression=compression,
    )
    return out_path
