"""The phase 5 research dataset: the closing 2_6 file plus the ``PI_*`` block.

Milestone 1 is research only, so the block is **not** a registered schema
layer: a 2_7 ``SchemaLayer`` would become every dataset builder's default
version (``TRAINING_DATA_SCHEMA_VERSION`` must equal the latest layer), and
the builders would start running the phase 4A engine. Instead this joins the
stored features (``build_pi_features.py``, one parquet per season) onto the
pinned 2_6 closing file by ``GAME_ID``, unchanged otherwise: games without
features get NaN, as 2_6 does for ``LU_*``. The layer is registered only if
the block earns it.

Writes ``data/train_data/closing_line_research_2_6_pi_v1_<date>.parquet`` and
a ``.provenance.json`` next to it: parent file and its checksum, the feature
list, per-season coverage and each season's feature metadata (minutes
provider, matchup history, rating book, commit). Prints the checksum the
campaign configs pin (``training_pipeline.data.compute_file_checksum``).

    python scripts/player_graph/build_pi_research_file.py --seasons 2019-2025
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.pi_features import FEATURES

from training_pipeline.data import compute_file_checksum

PARENT = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
FEATURE_DIR = Path("data/player_graph/pi_features")


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=season_range, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/train_data"))
    args = parser.parse_args()
    parent = pd.read_parquet(PARENT)
    if any(column in parent.columns for column in FEATURES):
        raise SystemExit("The parent file already has PI columns")
    keys = parent["GAME_ID"].astype(str).str.zfill(10)
    features, metadata = [], {}
    for season in args.seasons:
        features.append(pd.read_parquet(FEATURE_DIR / f"season={season}.parquet"))
        metadata[season] = json.loads(
            (FEATURE_DIR / f"season={season}.json").read_text()
        )
    block = pd.concat(features, ignore_index=True).set_index("GAME_ID")
    if block.index.duplicated().any():
        raise SystemExit("Duplicated GAME_ID in the feature files")
    joined = parent.copy()
    for column in FEATURES:
        joined[column] = keys.map(block[column]).to_numpy()
    season_of = pd.to_numeric(parent["SEASON_YEAR"], errors="coerce")
    coverage = {
        int(season): {
            "games": int(part.shape[0]),
            "with_features": int(part[FEATURES[0]].notna().sum()),
        }
        for season, part in joined.groupby(season_of)
        if season in args.seasons
    }
    name = f"closing_line_research_2_6_pi_v1_{date.today():%Y%m%d}.parquet"
    path = args.out_dir / name
    joined.to_parquet(path, index=False)
    checksum = compute_file_checksum(path)
    provenance = {
        "parent": str(PARENT),
        "parent_checksum": compute_file_checksum(PARENT),
        "features": list(FEATURES),
        "seasons": args.seasons,
        "coverage": coverage,
        "feature_metadata": metadata,
        "checksum": checksum,
        "note": "research dataset: 2_6 + PI_* block v1; not a registered schema layer",
    }
    path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    print(f"{path}: {len(joined):,} rows, checksum {checksum}")
    for season, line in coverage.items():
        print(f"  {season}: {line['with_features']:,} of {line['games']:,} games")


if __name__ == "__main__":
    main()
