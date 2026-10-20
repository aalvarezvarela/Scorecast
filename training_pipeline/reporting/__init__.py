"""Reporting helpers for the experiment notebooks.

Keeps the notebooks readable: they choose what to look at and write the prose,
while discovery, artifact loading, chart construction and verdict wording live
here where they can be tested and fixed once.

    from training_pipeline.reporting import charts, narrative, theme
    from training_pipeline.reporting.discovery import load_runs

    runs = theme.prepare_runs(load_runs("artifacts/experiments"))
    charts.plot_roi_with_seed_noise(runs)
"""

from training_pipeline.reporting import (
    alt_line,
    charts,
    coverage,
    factors,
    loaders,
    narrative,
    rescore,
    theme,
)
from training_pipeline.reporting.discovery import (
    find_run_dirs,
    is_run_dir,
    load_runs,
    resolve_run_dirs,
)
from training_pipeline.reporting.theme import apply_theme, prepare_runs

__all__ = [
    "alt_line",
    "charts",
    "coverage",
    "factors",
    "loaders",
    "narrative",
    "rescore",
    "theme",
    "apply_theme",
    "prepare_runs",
    "load_runs",
    "resolve_run_dirs",
    "find_run_dirs",
    "is_run_dir",
]
