"""Inspect schema versions and dataset files.

    python -m nba_ou.create_training_data.schema_layers versions
    python -m nba_ou.create_training_data.schema_layers describe <dataset file>
    python -m nba_ou.create_training_data.schema_layers diff 2_5 2_6 [--dataset intermediate_line]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from .base import CLOSING_LINE, DATASET_TYPES
from .manifest import read_manifest
from .registry import (
    LAYERS,
    available_versions,
    chain,
    column_origin,
    dataset_type_from_filename,
)


def _header(path: Path) -> list[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def _versions() -> None:
    for version in available_versions():
        layer = LAYERS.get(version)
        if layer is None:
            print(f"{version}  base (built end to end)")
            continue
        counts = ", ".join(
            f"{dataset}: +{len(layer.columns_for(dataset))}"
            for dataset in DATASET_TYPES
        )
        print(f"{version}  over {layer.parent}  ({counts})  {layer.summary}")


def _describe(path: Path) -> None:
    manifest = read_manifest(path)
    dataset_type = (
        manifest["dataset_type"] if manifest else dataset_type_from_filename(path)
    )
    columns = _header(path)
    print(f"{path.name}")
    print(f"  dataset type   {dataset_type}")
    if manifest is None:
        print("  manifest       none (built before manifests, or by hand)")
    else:
        print(f"  schema         {manifest['schema_version']}")
        print(
            f"  built from     {manifest['built_from_version']}"
            f" via layers {manifest['layers_applied'] or '[]'}"
        )
        parent = manifest.get("parent")
        if parent:
            print(
                f"  parent file    {parent['filename']} ({parent['checksum']}, "
                f"schema {parent['schema_version']})"
            )
        print(
            f"  commit         {manifest['git_commit']}"
            f"{' (dirty)' if manifest.get('git_dirty') else ''}"
        )
        print(f"  built at       {manifest['built_at']}")
        for name, checksum in manifest.get("files", {}).items():
            print(f"  file           {name}  {checksum}")
    origins = Counter(column_origin(columns, dataset_type).values())
    print(f"  columns        {len(columns):,}")
    for version in available_versions():
        if not origins.get(version):
            continue
        # Columns no layer declared are the base's -- or, in a file older than
        # the base, whatever that build produced.
        label = "base columns  " if version not in LAYERS else f"from {version} layer"
        print(f"    {label}  {origins[version]:,}")


def _diff(older: str, newer: str, dataset_type: str) -> None:
    for layer in chain(older, newer):
        added = layer.columns_for(dataset_type)
        print(f"{layer.version} (+{len(added)} for {dataset_type}): {layer.summary}")
        for column in added:
            print(f"  {column}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("versions", help="List the buildable schema versions.")
    describe = commands.add_parser("describe", help="Version and lineage of a file.")
    describe.add_argument("path", type=Path)
    diff = commands.add_parser("diff", help="Columns added between two versions.")
    diff.add_argument("older")
    diff.add_argument("newer")
    diff.add_argument("--dataset", choices=DATASET_TYPES, default=CLOSING_LINE)
    args = parser.parse_args(argv)

    if args.command == "versions":
        _versions()
    elif args.command == "describe":
        _describe(args.path)
    else:
        _diff(args.older, args.newer, args.dataset)


if __name__ == "__main__":
    main()
