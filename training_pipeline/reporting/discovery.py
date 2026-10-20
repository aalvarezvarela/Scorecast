"""Turn whatever the user points at into a concrete list of run directories.

Four kinds of input all mean "these runs", and the notebooks should not care
which one they were given:

1. **An artifacts super-folder** -- ``artifacts/experiments``. Every run beneath
   it, at any depth, so it keeps working if runs are later filed into
   per-campaign subfolders.
2. **A campaign config folder** -- ``experiments/rolling_origin_2026_08`` or
   ``experiments/archived/strategy_window_2026_08``.
   These hold experiment *definitions* (YAML), not results, so each one is
   resolved to the runs it produced.
3. **A single config file** -- ``experiments/.../classifier_2500.yaml``.
4. **An individual run directory** --
   ``artifacts/experiments/archived/classifier_2500_20260803_005940``.

The config-to-run mapping is the part with a trap in it. Run directories are
named ``{experiment_name}_{YYYYmmdd}_{HHMMSS}``, so matching on a plain prefix
over-matches: ``total_points_2500`` is a prefix of both
``total_points_2500_20260803_002317`` (correct) and
``total_points_2500_old_games_20260704_20260802_115707`` (a different
experiment entirely). Matching therefore requires the timestamp suffix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml  # type: ignore[import-untyped]

from training_pipeline.leaderboard import add_derived_columns, load_run_summary

#: A directory is a run if it carries the metadata every run writes.
RUN_MARKER = "metadata.json"

#: ``{experiment_name}_{YYYYmmdd}_{HHMMSS}`` -- see the module docstring for why
#: the timestamp is part of the match rather than a plain prefix test.
_TIMESTAMP_SUFFIX = r"_\d{8}_\d{6}$"

DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "experiments"


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / RUN_MARKER).exists()


def find_run_dirs(root: Path) -> list[Path]:
    """Every run directory at or beneath ``root``, depth-independent."""
    if is_run_dir(root):
        return [root]
    return sorted(
        {marker.parent for marker in root.rglob(RUN_MARKER) if marker.parent.is_dir()}
    )


def experiment_names_in(path: Path) -> list[str]:
    """The ``experiment_name`` of a config file, or of every config in a folder.

    Reads the name out of the YAML rather than inferring it from the filename:
    the two are conventionally equal but nothing enforces it, and the run
    directory is named after the field, not the file.
    """
    files = [path] if path.is_file() else sorted(
        p for p in path.glob("*.y*ml") if not p.name.startswith("_")
    )
    names: list[str] = []
    for file in files:
        try:
            payload = yaml.safe_load(file.read_text()) or {}
        except yaml.YAMLError:
            continue
        name = payload.get("experiment_name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def runs_for_experiment_name(name: str, artifacts_root: Path) -> list[Path]:
    """Run directories produced by one experiment definition.

    Anchored on the timestamp suffix so a shorter experiment name cannot
    swallow a longer, unrelated one (see the module docstring).
    """
    pattern = re.compile(re.escape(name) + _TIMESTAMP_SUFFIX)
    return sorted(
        path for path in find_run_dirs(artifacts_root) if pattern.fullmatch(path.name)
    )


def _looks_like_config(path: Path) -> bool:
    if path.is_file():
        return path.suffix in {".yaml", ".yml"}
    return any(path.glob("*.y*ml")) and not find_run_dirs(path)


def resolve_run_dirs(
    sources: str | Path | list[str | Path],
    *,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    project_root: str | Path | None = None,
) -> list[Path]:
    """Resolve any mix of the four input kinds into run directories.

    Order is preserved and duplicates are dropped, so overlapping sources (a
    campaign folder plus one of its runs) cannot double-count a run.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]

    base = Path(project_root) if project_root else None

    def absolute(value: str | Path) -> Path:
        """Relative paths resolve against ``project_root`` when one is given.

        Deliberately does NOT consult the working directory first. A
        "try cwd, else project_root" rule would make the same SOURCES string
        point at different folders depending on where the kernel happened to
        start -- and the notebooks run from experiments/notebooks/, so that is
        not a hypothetical. Anchoring on the project root makes a relative path
        mean one thing everywhere.
        """
        path = Path(value)
        if path.is_absolute() or base is None:
            return path
        return base / path

    artifacts = absolute(artifacts_root)

    resolved: list[Path] = []
    seen: set[Path] = set()
    missing: list[str] = []

    for source in sources:
        path = absolute(source)
        if not path.exists():
            missing.append(str(source))
            continue

        if _looks_like_config(path):
            found: list[Path] = []
            for name in experiment_names_in(path):
                found.extend(runs_for_experiment_name(name, artifacts))
        else:
            found = find_run_dirs(path)

        for run_dir in found:
            key = run_dir.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(run_dir)

    if missing:
        raise FileNotFoundError(
            "These sources do not exist: " + ", ".join(missing)
        )
    return resolved


def load_runs(
    sources: str | Path | list[str | Path],
    *,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """One leaderboard-shaped row per resolved run.

    Adds the provenance columns the notebooks need on top of the standard
    summary: where the run came from and where its artifacts live, so a run
    pulled in from another folder is never silently mixed into the main cohort.
    """
    run_dirs = resolve_run_dirs(
        sources, artifacts_root=artifacts_root, project_root=project_root
    )
    if not run_dirs:
        raise ValueError(
            f"No runs found for {sources!r}. Point at an artifacts folder, a "
            "campaign config folder, a config file, or a single run directory."
        )

    rows = []
    for run_dir in run_dirs:
        row = load_run_summary(run_dir)
        row["run_dir"] = str(run_dir)
        row["source_root"] = run_dir.parent.name
        row["source_path"] = str(run_dir.parent)
        rows.append(row)

    frame = add_derived_columns(pd.DataFrame(rows))
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame
