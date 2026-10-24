#!/usr/bin/env python3
"""Manual utility for inspecting a downloaded NBA injury-report PDF.

This requires a real report file and is therefore import-safe during pytest
collection. Run the module directly and pass the PDF path when needed.
"""

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    read_injury_report,
)


def inspect_injury_report(pdf_path: str | Path) -> pd.DataFrame:
    """Read an injury-report PDF and print its parsed table."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    print(f"Reading injury report from: {path}")
    df = read_injury_report(path)

    print("\nSuccessfully read injury report!")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\n" + "=" * 80)
    print("Preview of injury report data:")
    print("=" * 80)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        50,
    ):
        print(df.to_string(index=False))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", "-p", required=True, help="Path to the report PDF")
    args = parser.parse_args()
    inspect_injury_report(args.path)


if __name__ == "__main__":
    main()
