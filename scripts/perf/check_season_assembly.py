#!/usr/bin/env python3
"""Check that S3 season pieces reassemble into exactly what one query gives.

For every ``SELECT * FROM <table>`` result recorded on the given tapes:

* the reference is the loader's frame from the recorded multi-season rows
  (pandas' ``read_sql`` conversion, then ``canonical_row_order``);
* the same rows are split by season; past seasons go through the exact
  Parquet round trip, as the S3 store keeps them; the newest season and any
  extra games outside the requested seasons stand in for the database part;
* ``season_tables.assemble`` must give a frame identical to the reference
  (columns, dtypes, floats bit for bit, object values and their types).

    python scripts/perf/check_season_assembly.py artifacts/perf/ref_7bd96a9/*/tape
"""

from __future__ import annotations

import ast
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]

import pandas.io.sql as pandas_sql  # noqa: E402
from nba_ou.postgre_db.row_order import canonical_row_order  # noqa: E402
from nba_ou.postgre_db.season_store import (  # noqa: E402
    frame_differences,
    from_parquet,
    to_exact_parquet,
)
from nba_ou.postgre_db.season_tables import (  # noqa: E402
    SEASON_TABLES,
    _by_season_in_query_order,
    assemble,
)
from row_order import shuffle_within_order  # noqa: E402


def _frame(rows, columns, by):
    return canonical_row_order(pandas_sql._wrap_result(rows, columns, coerce_float=True), by)


def check(tape: Path) -> int:
    failures = 0
    seen = set()
    for entry in json.loads((tape / "index.json").read_text())["entries"]:
        query = " ".join(entry["query"].split())
        if not entry["file"] or not query.startswith("SELECT * FROM"):
            continue
        dataset = query.split()[3].replace('"', "")
        spec = SEASON_TABLES.get(dataset)
        if spec is None or (dataset, entry["params"]) in seen:
            continue
        seen.add((dataset, entry["params"]))
        description, rows = pickle.load(open(tape / entry["file"], "rb"))
        if not rows:
            continue
        columns = [c[0] for c in description]
        by = spec.order_columns
        reference = _frame(rows, columns, by)

        params = ast.literal_eval(entry["params"])
        requested = set(params[0]) if params else set()
        s_col = columns.index("season_year")
        seasons = sorted({r[s_col] for r in rows if r[s_col] in requested})
        newest = seasons[-1] if seasons else None
        rng = random.Random(0)
        stored, db_rows = {}, []
        for season in seasons:
            part = [r for r in rows if r[s_col] == season]
            if season == newest:
                db_rows += part
                continue
            # The season query's server order: any order its ORDER BY permits.
            part = shuffle_within_order(entry["query"], description, part, rng)
            piece = _frame(part, columns, by)
            stored[season] = from_parquet(to_exact_parquet(piece))
        db_rows += [r for r in rows if r[s_col] not in requested]  # extra games
        db_rows = shuffle_within_order(entry["query"], description, db_rows, rng)
        db_frame = _frame(db_rows, columns, by) if db_rows else None

        pieces = list(stored.values()) + ([db_frame] if db_frame is not None else [])
        if spec.order_by and not spec.sort_keys:
            pieces = _by_season_in_query_order(spec, pieces)
        assembled = assemble(spec, pieces)
        problems = frame_differences(assembled, reference)
        status = "EXACT" if not problems else "DIFFERENT"
        failures += bool(problems)
        print(
            f"{status:9s} {dataset:42s} {len(rows):>8,} rows, {len(stored)} stored "
            f"season(s), {len(db_rows):,} db rows {problems[:3] if problems else ''}"
        )
    return failures


def main() -> int:
    failures = sum(check(Path(t)) for t in sys.argv[1:])
    print("RESULT:", "EQUAL" if failures == 0 else f"DIFFERENT ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
