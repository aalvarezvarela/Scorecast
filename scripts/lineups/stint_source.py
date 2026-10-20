"""Choose where the ratings scripts read validated stints from."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCES = ("parquet", "db")


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="parquet",
        help="Read stints from the local Parquet store (default) or Postgres",
    )
    parser.add_argument("--local-root", type=Path, default=Path("data"))


def load_stints(source: str, seasons: list[int], local_root: Path) -> pd.DataFrame:
    if source == "db":
        from nba_ou.postgre_db.lineups.fetch import fetch_stints

        return fetch_stints(seasons)
    from nba_ou.data_processing.lineups.stint_store import read_stints

    return read_stints(seasons, local_root=local_root)
