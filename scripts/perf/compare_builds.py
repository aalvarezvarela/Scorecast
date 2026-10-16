#!/usr/bin/env python3
"""Compare two builds of a training dataset, value by value.

    python scripts/perf/compare_builds.py REFERENCE.parquet CANDIDATE.parquet [--exact]

Rows are matched on ``GAME_ID`` (plus ``TIME_TO_MATCH_MIN`` when both files
have it). The report covers:

* columns present in only one file, and column-order differences;
* row keys present in only one file, and row-order differences;
* dtype differences;
* per column, the share of rows whose value differs, where NaN equals NaN and
  any other NaN-pattern mismatch counts as a difference, plus the max |diff|.

``--exact`` (the speed-up acceptance test) fails on any difference at all:
values must be equal bit for bit (NaN equal to NaN), with the same dtypes,
the same column order and the same row order. Without it, numeric values
within ``--rtol`` count as equal, which is useful to diagnose a failure, never
to pass one.

Exit code 0 means equal under the chosen rule, 1 means they differ.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd


def frame_digest(df: pd.DataFrame) -> str:
    """sha256 over column names, dtypes and every value, in order."""
    digest = hashlib.sha256()
    for column in df.columns:
        digest.update(str(column).encode())
        digest.update(str(df[column].dtype).encode())
        values = pd.util.hash_pandas_object(df[column], index=False).to_numpy()
        digest.update(values.tobytes())
    return digest.hexdigest()


def _keys(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    keys = ["GAME_ID"]
    if "TIME_TO_MATCH_MIN" in a.columns and "TIME_TO_MATCH_MIN" in b.columns:
        keys.append("TIME_TO_MATCH_MIN")
    return keys


def _key_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df[keys].copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    return out


def _values_differ(
    x: pd.Series, y: pd.Series, *, exact: bool, rtol: float
) -> tuple[np.ndarray, float]:
    """Boolean mask of differing rows, and the max |diff| over numeric values."""
    x_na = x.isna().to_numpy()
    y_na = y.isna().to_numpy()
    numeric = pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
    if numeric and not pd.api.types.is_bool_dtype(x):
        xv = x.to_numpy(dtype="float64", na_value=np.nan)
        yv = y.to_numpy(dtype="float64", na_value=np.nan)
        both = ~x_na & ~y_na
        diff = np.zeros(len(xv))
        diff[both] = np.abs(xv[both] - yv[both])
        if exact:
            # Bit-for-bit, which also tells -0.0 from 0.0.
            unequal = np.zeros(len(xv), dtype=bool)
            unequal[both] = xv[both].view("int64") != yv[both].view("int64")
        else:
            unequal = diff > rtol * (1.0 + np.abs(np.where(both, yv, 0.0)))
        mask = (x_na != y_na) | unequal
        return mask, float(diff.max()) if len(diff) else 0.0
    xs = (
        x.astype("object")
        .where(~x_na, None)
        .map(lambda v: None if v is None else repr(v))
    )
    ys = (
        y.astype("object")
        .where(~y_na, None)
        .map(lambda v: None if v is None else repr(v))
    )
    mask = (x_na != y_na) | (~x_na & ~y_na & (xs.to_numpy() != ys.to_numpy()))
    return mask, float("nan")


def compare(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    exact: bool = True,
    rtol: float = 1e-9,
    show: int = 40,
) -> bool:
    equal = True
    ref_cols, cand_cols = list(reference.columns), list(candidate.columns)
    only_ref = [c for c in ref_cols if c not in set(cand_cols)]
    only_cand = [c for c in cand_cols if c not in set(ref_cols)]
    print(f"reference : {reference.shape[0]:,} rows x {reference.shape[1]:,} columns")
    print(f"candidate : {candidate.shape[0]:,} rows x {candidate.shape[1]:,} columns")
    if only_ref or only_cand:
        equal = False
        print(f"columns only in reference ({len(only_ref)}): {only_ref[:show]}")
        print(f"columns only in candidate ({len(only_cand)}): {only_cand[:show]}")
    shared = [c for c in ref_cols if c in set(cand_cols)]
    if [c for c in cand_cols if c in set(shared)] != shared:
        print("column ORDER differs")
        equal = equal and not exact

    keys = _keys(reference, candidate)
    ref_keys, cand_keys = _key_frame(reference, keys), _key_frame(candidate, keys)
    for name, frame in (("reference", ref_keys), ("candidate", cand_keys)):
        if frame.duplicated().any():
            print(f"{name} has {int(frame.duplicated().sum())} duplicated row key(s)")
            equal = False
    ref_index = pd.MultiIndex.from_frame(ref_keys)
    cand_index = pd.MultiIndex.from_frame(cand_keys)
    only_ref_rows = ref_index.difference(cand_index)
    only_cand_rows = cand_index.difference(ref_index)
    if len(only_ref_rows) or len(only_cand_rows):
        equal = False
        print(
            f"rows only in reference: {len(only_ref_rows):,}  e.g. {list(only_ref_rows[:5])}"
        )
        print(
            f"rows only in candidate: {len(only_cand_rows):,}  e.g. {list(only_cand_rows[:5])}"
        )
    common = ref_index.intersection(cand_index, sort=False)
    if not ref_index.equals(cand_index):
        print("row ORDER or row set differs")
        equal = equal and not exact

    ref_aligned = reference.set_axis(ref_index).loc[common].reset_index(drop=True)
    cand_aligned = candidate.set_axis(cand_index).loc[common].reset_index(drop=True)

    dtype_diffs = [
        (c, str(reference[c].dtype), str(candidate[c].dtype))
        for c in shared
        if reference[c].dtype != candidate[c].dtype
    ]
    if dtype_diffs:
        print(f"dtype differences ({len(dtype_diffs)}): {dtype_diffs[:show]}")
        equal = equal and not exact

    rows = []
    for column in shared:
        mask, max_abs = _values_differ(
            ref_aligned[column], cand_aligned[column], exact=exact, rtol=rtol
        )
        if mask.any():
            rows.append((column, float(mask.mean()), int(mask.sum()), max_abs))
    if rows:
        equal = False
        report = pd.DataFrame(
            rows, columns=["column", "frac_rows", "n_rows", "max_abs_diff"]
        ).sort_values(["frac_rows", "column"], ascending=[False, True])
        print(f"columns with differing values: {len(report)} of {len(shared)}")
        print(report.head(show).to_string(index=False))
    else:
        print(
            f"values: all {len(shared):,} shared columns equal on {len(common):,} rows"
        )

    if exact and equal:
        same_digest = frame_digest(reference) == frame_digest(candidate)
        print(f"frame digest equal: {same_digest}")
        equal = same_digest
    print("RESULT:", "EQUAL" if equal else "DIFFERENT")
    return equal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--show", type=int, default=40)
    parser.add_argument(
        "--ignore-column-order",
        action="store_true",
        help="put the candidate's columns in the reference's order first (the "
        "column SET and every value must still match)",
    )
    args = parser.parse_args()
    reference = pd.read_parquet(args.reference)
    candidate = pd.read_parquet(args.candidate)
    if args.ignore_column_order:
        order = [c for c in reference.columns if c in set(candidate.columns)]
        rest = [c for c in candidate.columns if c not in set(reference.columns)]
        candidate = candidate[order + rest]
    ok = compare(reference, candidate, exact=args.exact, rtol=args.rtol, show=args.show)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
