"""A fixed row order for query results, whatever order the server returns.

Postgres orders rows only as far as a query's ``ORDER BY`` says. Rows that
tie, and every row of a query without one, can come back in any order, and
several features are means over such rows: their last bits followed the
server's order (measured 2026-10-15, up to 1.4e-14 on ~44 injury-status and
news columns). The same result assembled from per-season pieces, as the S3
input cache does, would differ the same way.

``canonical_row_order`` sorts a loader's frame by the query's own ``ORDER BY``
keys and breaks every remaining tie on the other columns, so any order the
server may return, or any concatenation of season pieces, gives the same
frame. Text compares by its ``str`` in pandas, never by database collation,
so the order is the same on every machine.
"""

from __future__ import annotations

import re

import pandas as pd


def _sort_key(column: pd.Series) -> pd.Series:
    if column.dtype == object or isinstance(
        column.dtype, pd.StringDtype | pd.CategoricalDtype
    ):
        text = column.map(
            lambda value: (
                None
                if value is None
                or value is pd.NA
                or (isinstance(value, float) and value != value)
                else str(value)
            )
        )
        codes, _ = pd.factorize(text, sort=True)
        return pd.Series(codes, index=column.index)  # missing -> -1, first
    return column


def canonical_row_order(df: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """``df`` in a fixed order that keeps its SQL ``ORDER BY`` and fixes the rest.

    ``by`` names the query's ``ORDER BY`` columns; ``df`` must be in the order
    the server returned. Runs of rows that tie on ``by`` keep their place,
    exactly as the server ordered them (no collation or NULL rule is
    re-implemented); within a run, and for a query without ``ORDER BY`` (one
    run), rows are sorted by the other columns in frame order, as many as it
    takes to leave no tie. Stopping once the keys are unique keeps a wide frame
    cheap; the order is still determined by the data alone. Returns a fresh
    ``RangeIndex``, like ``read_sql``.
    """
    if df is None or len(df) < 2:
        return df if df is None else df.reset_index(drop=True)
    keys = pd.DataFrame(index=pd.RangeIndex(len(df)))
    if by:
        codes = pd.DataFrame({c: _sort_key(df[c]).to_numpy() for c in by})
        changed = codes.ne(codes.shift()).any(axis=1).to_numpy()
        changed[0] = True
        keys["run"] = changed.cumsum()
    rest = [c for c in df.columns if c not in set(by)]
    # Uniqueness is checked after 1, 2, 4, ... tie-break columns.
    next_check = 1
    for i, column in enumerate(rest):
        keys[f"k{i}"] = _sort_key(df[column]).to_numpy()
        if i + 1 == next_check:
            if not keys.duplicated().any():
                break
            next_check *= 2
    positions = keys.sort_values(
        list(keys.columns), kind="mergesort", na_position="first"
    ).index.to_numpy()
    return df.iloc[positions].reset_index(drop=True)


def _depths(text: str) -> list[int]:
    depth, out = 0, []
    for char in text:
        if char == ")":
            depth -= 1
        out.append(depth)
        if char == "(":
            depth += 1
    return out


def final_order_by(query: str) -> tuple[str, ...]:
    """Result-column names of the query's final top-level ``ORDER BY``.

    Window and subquery ``ORDER BY``s sit inside parentheses and are skipped;
    ``l.game_id DESC NULLS LAST`` becomes ``game_id``. Empty when there is none.
    """
    flat = "".join(c if d == 0 else " " for c, d in zip(query, _depths(query)))
    matches = list(re.finditer(r"\bORDER\s+BY\b", flat, flags=re.I))
    if not matches:
        return ()
    clause = flat[matches[-1].end() :]
    clause = re.split(r"\b(LIMIT|OFFSET|FETCH|FOR)\b", clause, flags=re.I)[0]
    keys = []
    for part in clause.split(","):
        expr = re.split(r"\s+(ASC|DESC|NULLS)\b", part.strip(), flags=re.I)[0]
        if expr.strip():
            keys.append(expr.strip().split(".")[-1].strip('"'))
    return tuple(keys)


def read_sql_canonical(query: str, conn, params=None) -> pd.DataFrame:
    """``pd.read_sql_query`` with the rows in ``canonical_row_order``.

    The ``ORDER BY`` is read from the query text, so a loader only swaps the
    function it calls.
    """
    if not isinstance(query, str):
        query = query.as_string(conn)
    df = pd.read_sql_query(query, conn, params=params)
    by = final_order_by(query)
    missing = [key for key in by if key not in df.columns]
    if missing:
        raise ValueError(f"ORDER BY keys {missing} are not result columns")
    return canonical_row_order(df, by)
