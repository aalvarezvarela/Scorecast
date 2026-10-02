"""``write_csv`` must write exactly the bytes ``DataFrame.to_csv`` writes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.utils import parallel_csv
from nba_ou.utils.parallel_csv import write_csv


def _frame(n_rows: int, n_float_columns: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    floats = rng.normal(0, 100, (n_rows, n_float_columns))
    floats[rng.random(floats.shape) < 0.1] = np.nan
    frame = pd.DataFrame(floats, columns=[f"F{i}" for i in range(n_float_columns)])
    # Values whose text form is easy to get wrong.
    frame.loc[::7, "F0"] = 1e-20
    frame.loc[::11, "F1"] = 0.1 + 0.2
    frame.loc[::13, "F2"] = -0.0
    frame.loc[::17, "F3"] = np.inf
    frame["INT"] = rng.integers(-5, 5, n_rows)
    frame["NULLABLE_INT"] = pd.array(
        np.where(rng.random(n_rows) < 0.2, None, rng.integers(0, 9, n_rows)),
        dtype="Int64",
    )
    frame["BOOL"] = rng.random(n_rows) < 0.5
    frame["TEXT"] = rng.choice(["plain", "with,comma", 'with "quote"', None], n_rows)
    frame["GAME_ID"] = [f"00{22100000 + i}" for i in range(n_rows)]
    # A datetime column is written date-only per pandas chunk when every value
    # in the chunk is midnight, so mix both kinds in runs of varying length.
    dates = pd.Timestamp("2021-10-19") + pd.to_timedelta(np.arange(n_rows), unit="D")
    with_time = (np.arange(n_rows) // 97) % 3 == 0
    frame["GAME_DATE"] = dates + pd.to_timedelta(np.where(with_time, 90, 0), unit="m")
    frame["TIPOFF_UTC"] = frame["GAME_DATE"].dt.tz_localize("UTC")
    frame.loc[::19, "TIPOFF_UTC"] = pd.NaT
    frame["TIME_TO_MATCH_MIN"] = rng.choice([0, 30, 360, 720], n_rows)
    return frame


@pytest.mark.parametrize("n_rows", [1, 999, 4321])
@pytest.mark.parametrize("n_jobs", [2, 3])
def test_parallel_write_is_byte_identical_to_to_csv(
    tmp_path, monkeypatch, n_rows, n_jobs
):
    monkeypatch.setattr(parallel_csv, "MIN_PARALLEL_ROWS", 0)
    # Small tasks, so the frame is split into many pieces.
    monkeypatch.setattr(parallel_csv, "_CELLS_PER_TASK", 10_000)
    frame = _frame(n_rows)

    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    frame.to_csv(expected, index=False)
    write_csv(frame, actual, n_jobs=n_jobs)

    assert actual.read_bytes() == expected.read_bytes()


def test_wide_frame_uses_small_pandas_chunks(tmp_path, monkeypatch):
    """Thousands of columns make pandas' own chunk a few rows long."""
    monkeypatch.setattr(parallel_csv, "MIN_PARALLEL_ROWS", 0)
    monkeypatch.setattr(parallel_csv, "_CELLS_PER_TASK", 50_000)
    frame = _frame(700, n_float_columns=3_000)
    assert parallel_csv._pandas_chunksize(frame.shape[1]) < 50

    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    frame.to_csv(expected, index=False)
    write_csv(frame, actual, n_jobs=3)

    assert actual.read_bytes() == expected.read_bytes()


def test_empty_and_serial_paths_match_to_csv(tmp_path):
    for frame in (_frame(0), _frame(50)):
        expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
        frame.to_csv(expected, index=False)
        write_csv(frame, actual, n_jobs=1)
        assert actual.read_bytes() == expected.read_bytes()
        write_csv(frame, actual, n_jobs=4)
        assert actual.read_bytes() == expected.read_bytes()
