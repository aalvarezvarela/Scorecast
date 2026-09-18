"""Write a large DataFrame to CSV on several processes, byte-identical to pandas.

``DataFrame.to_csv`` spends most of its time turning floats into text, one
column block at a time, on a single core. On the intermediate-line dataset
(~150k rows x ~3,200 columns, 4.6 GB) that alone is over ten minutes.

The formatting of a row does not depend on any other row except in one place:
pandas formats in chunks of ``_DEFAULT_CHUNKSIZE_CELLS // n_columns`` rows, and
a datetime column is written date-only when every value *in that chunk* is
midnight. Splitting the frame at multiples of that same chunk size therefore
reproduces pandas' own chunks exactly, so each worker formats every value the
way a single ``to_csv`` call would. The parent writes the pieces in order.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pandas as pd
from pandas.io.formats.csvs import _DEFAULT_CHUNKSIZE_CELLS

#: Rows below this are written with a plain ``to_csv``; forking costs more.
MIN_PARALLEL_ROWS = 20_000

#: Formatted cells per task. Large enough to amortise the inter-process copy,
#: small enough that the parent writes steadily and holds little in memory.
_CELLS_PER_TASK = 4_000_000

# Set in the parent right before the pool forks, so workers inherit the frame
# through copy-on-write memory instead of receiving it pickled.
_FRAME: pd.DataFrame | None = None
_CHUNKSIZE: int = 1


def _pandas_chunksize(n_columns: int) -> int:
    """The row chunk ``to_csv`` uses by default (``index=False``)."""
    return (_DEFAULT_CHUNKSIZE_CELLS // (n_columns or 1)) or 1


def _format_rows(bounds: tuple[int, int]) -> str:
    start, stop = bounds
    return _FRAME.iloc[start:stop].to_csv(
        index=False, header=False, chunksize=_CHUNKSIZE
    )


def write_csv(df: pd.DataFrame, path: str | Path, *, n_jobs: int | None = None) -> None:
    """``df.to_csv(path, index=False)``, formatted on ``n_jobs`` processes.

    The file is byte-for-byte what ``df.to_csv(path, index=False)`` writes;
    ``tests/test_parallel_csv.py`` pins that. Falls back to that call itself
    for small frames, a single job, or platforms without ``fork``.
    """
    global _FRAME, _CHUNKSIZE

    n_jobs = n_jobs or os.cpu_count() or 1
    fork_available = "fork" in multiprocessing.get_all_start_methods()
    if n_jobs <= 1 or len(df) < MIN_PARALLEL_ROWS or not fork_available:
        df.to_csv(path, index=False)
        return

    chunksize = _pandas_chunksize(len(df.columns))
    # A whole number of pandas chunks per task keeps every chunk boundary
    # exactly where a single to_csv call would put it.
    chunks_per_task = max(_CELLS_PER_TASK // (chunksize * max(len(df.columns), 1)), 1)
    rows_per_task = chunksize * chunks_per_task
    bounds = [
        (start, min(start + rows_per_task, len(df)))
        for start in range(0, len(df), rows_per_task)
    ]

    header = df.iloc[:0].to_csv(index=False)
    _FRAME, _CHUNKSIZE = df, chunksize
    try:
        context = multiprocessing.get_context("fork")
        with (
            context.Pool(n_jobs) as pool,
            open(path, "w", encoding="utf-8", newline="") as handle,
        ):
            handle.write(header)
            for text in pool.imap(_format_rows, bounds):
                handle.write(text)
    finally:
        _FRAME = None
