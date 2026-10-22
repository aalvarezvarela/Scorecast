"""Create the Aiven prediction store (schema ``prediction``).

Five tables, a BRIN index and the ``predictions_full`` view; see
``docs/serving/prediction_storage.md``. Idempotent (``IF NOT EXISTS`` / ``CREATE
OR REPLACE VIEW``). ``--print`` shows the SQL without connecting.

    python scripts/prediction_store/create_prediction_store.py --print
    python scripts/prediction_store/create_prediction_store.py
"""

from __future__ import annotations

import argparse

from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.predictions_aiven.schema import SCHEMA, create_schema, ddl_script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the SQL and exit without connecting",
    )
    args = parser.parse_args()
    if args.print_only:
        print(ddl_script())
        return
    with connect_nba_db(env="aiven") as conn:
        create_schema(conn)
    print(f"Schema {SCHEMA!r} is in place on Aiven.")


if __name__ == "__main__":
    main()
