"""Reorder a query result only as far as its SQL allows.

Postgres guarantees row order only up to the query's final ``ORDER BY``: rows
that tie on every key, and every row of a query without one, may come back
in any order. A build that depends on that order would change if a query were
split by season (or just replanned). ``shuffle_within_order`` produces another
order the server could legally have returned. It keeps the recorded order of
the ordered keys, so no string collation is reimplemented, and shuffles each
run of rows that tie on all of them.
"""

from __future__ import annotations

import random
import re


def _top_level(text: str) -> list[tuple[int, str]]:
    """``(depth, char)`` for each character, depth counting open parentheses."""
    depth, out = 0, []
    for char in text:
        if char == ")":
            depth -= 1
        out.append((depth, char))
        if char == "(":
            depth += 1
    return out


def final_order_by(query: str) -> list[str] | None:
    """The column names of the query's final top-level ``ORDER BY``, or None.

    Window ``ORDER BY``s and subqueries sit inside parentheses and are skipped.
    ``l.game_id DESC NULLS LAST`` becomes ``game_id``.
    """
    chars = _top_level(query)
    flat = "".join(c if d == 0 else " " for d, c in chars)
    matches = list(re.finditer(r"\bORDER\s+BY\b", flat, flags=re.I))
    if not matches:
        return None
    clause = flat[matches[-1].end() :]
    clause = re.split(r"\b(LIMIT|OFFSET|FETCH|FOR)\b", clause, flags=re.I)[0]
    keys = []
    for part in clause.split(","):
        part = part.strip()
        if not part:
            continue
        expr = re.split(r"\s+(ASC|DESC|NULLS)\b", part, flags=re.I)[0].strip()
        keys.append(expr.split(".")[-1].strip('"'))
    return keys or None


def shuffle_within_order(
    query: str, description, rows: list, rng: random.Random
) -> list:
    """``rows`` in another order the query's ``ORDER BY`` permits."""
    if not rows or description is None:
        return rows
    keys = final_order_by(query)
    if keys is None:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled
    names = [column[0] for column in description]
    missing = [key for key in keys if key not in names]
    if missing:
        raise ValueError(
            f"ORDER BY keys {missing} are not result columns of:\n{query[:500]}"
        )
    positions = [names.index(key) for key in keys]
    out, run, run_key = [], [], object()
    for row in rows:
        key = tuple(row[p] for p in positions)
        if run and key != run_key:
            rng.shuffle(run)
            out.extend(run)
            run = []
        run_key = key
        run.append(row)
    rng.shuffle(run)
    out.extend(run)
    return out
