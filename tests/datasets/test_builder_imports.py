"""The dataset builders import without the scraping dependencies.

The daily refit installs ``requirements/retrain-cuda.txt`` (Poetry groups main
and cuda), not ``scrape``. Only tonight's inputs parse injury-report PDFs, so a
builder that imports ``pymupdf`` at module level fails every refit of its
slots. Each import runs in a fresh interpreter: other tests may have imported
the PDF parser into this one already. It gets this one's ``sys.path``, which
holds pytest's ``pythonpath`` entries: CI does not install the package.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.support.paths import REPO_ROOT

BLOCK_AND_IMPORT = """
import importlib, sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"pymupdf", "fitz"}:
            raise ModuleNotFoundError(f"No module named {name!r}")

sys.meta_path.insert(0, Block())
importlib.import_module(sys.argv[1])
"""


@pytest.mark.parametrize(
    "module",
    [
        "nba_ou.create_training_data.create_df_to_predict",
        "nba_ou.create_training_data.create_intermediate_line_df",
        "nba_ou.create_training_data.season_partitions",
    ],
)
def test_builder_imports_without_pymupdf(module):
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_AND_IMPORT, module],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
