"""Pin the design of lineup_projection_2026_09: one deliberate difference per pair.

Each strategy has a control and a lineup cell that must differ only in which
CSV they read (plus labels). And every cell must start at 2021-22 or later:
the lineup ratings start there, so an earlier floor makes every ``LU_*`` column
season-gated, cleaning drops the family, and the treatment silently becomes a
copy of the control.
"""

from pathlib import Path

import pytest

from training_pipeline.cli import load_config

CAMPAIGN = (
    Path(__file__).resolve().parents[1] / "experiments" / "lineup_projection_2026_09"
)
PAIRS = {
    "line_error": ("a_line_error_control", "b_line_error_lineup"),
    "total_points": ("c_total_points_control", "d_total_points_lineup"),
}
LABELS = {"experiment_name", "hypothesis", "tags"}


def _dump(name: str) -> dict:
    return load_config(CAMPAIGN / f"{name}.yaml").model_dump(mode="json")


def _flatten(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= _flatten(value, f"{path}.")
        else:
            out[path] = value
    return out


@pytest.mark.parametrize("strategy", sorted(PAIRS))
def test_a_pair_differs_only_in_its_dataset(strategy):
    control, lineup = (_flatten(_dump(name)) for name in PAIRS[strategy])
    differing = {
        key
        for key in control.keys() | lineup.keys()
        if control.get(key) != lineup.get(key) and key not in LABELS
    }
    assert differing == {
        "data.csv_path",
        "data.data_version",
        "data.expected_checksum",
    } or (differing == {"data.csv_path", "data.data_version"})
    assert "with_lineup_features" in lineup["data.csv_path"]
    assert "with_lineup_features" not in control["data.csv_path"]


@pytest.mark.parametrize("name", [n for pair in PAIRS.values() for n in pair])
def test_every_cell_starts_where_the_ratings_do(name):
    config = _dump(name)
    assert config["data"]["season_year_floor"] >= 2021
    assert config["evaluation_seeds"], "no seeds means no error bar"
    assert config["optuna"]["timeout"] is None
