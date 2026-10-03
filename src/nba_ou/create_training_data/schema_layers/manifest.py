"""The sidecar that says what a dataset file is.

The filename carries the schema version and limit date
(``closing_line_data_2_6_20261003.parquet``), which is what
``training_pipeline.registry.parse_schema_version`` reads. The manifest beside it
(``closing_line_data_2_6_20261003.manifest.json``) carries the rest: which
layers were applied, which columns each version added, which parent file it was
built from (by checksum), and the commit that built it.

Checksums are computed by the caller -- ``training_pipeline.data`` owns the
checksum function, and ``nba_ou`` does not import ``training_pipeline``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION

from .registry import chain, version_columns

MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_FORMAT = 1

_REPO_ROOT = Path(__file__).resolve().parents[4]


def manifest_path(dataset_path: str | Path) -> Path:
    """``x.parquet`` (or an older ``x.csv``) maps to ``x.manifest.json``."""
    path = Path(dataset_path)
    return path.with_name(path.stem + MANIFEST_SUFFIX)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    schema_version: str,
    dataset_type: str,
    n_rows: int,
    n_columns: int,
    parent: dict[str, Any] | None = None,
    from_version: str = BASE_SCHEMA_VERSION,
    build_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a dataset of ``schema_version``.

    ``from_version`` is where this build started: the base for a full build, the
    parent file's version for a layered one. ``parent`` is
    ``{"filename", "checksum", "schema_version"}`` when built from a file.
    """
    layers = [layer.version for layer in chain(from_version, schema_version)]
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "manifest_format": MANIFEST_FORMAT,
        "schema_version": schema_version,
        "dataset_type": dataset_type,
        "base_schema_version": BASE_SCHEMA_VERSION,
        "built_from_version": from_version,
        "layers_applied": layers,
        "columns_added": {
            version: list(version_columns(version, dataset_type)) for version in layers
        },
        "parent": parent,
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "files": {},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "build_args": build_args or {},
    }


def write_manifest(dataset_path: str | Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(dataset_path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False, default=str) + "\n")
    return path


def read_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    path = manifest_path(dataset_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def record_file(dataset_path: str | Path, checksum: str) -> Path:
    """Record a written dataset file and its checksum."""
    manifest = read_manifest(dataset_path)
    if manifest is None:
        raise FileNotFoundError(f"No manifest beside {dataset_path}.")
    manifest.setdefault("files", {})[Path(dataset_path).name] = checksum
    return write_manifest(dataset_path, manifest)
