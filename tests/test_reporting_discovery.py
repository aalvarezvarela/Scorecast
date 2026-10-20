"""Resolving notebook inputs into run directories.

The interesting case is mapping a campaign config folder to the runs it
produced, because run directories are named ``{experiment_name}_{timestamp}``
and a plain prefix test over-matches: ``total_points_2500`` is a prefix of
``total_points_2500_old_games_20260704_20260802_115707``, a different
experiment entirely.
"""

import json

import pytest
import yaml

from training_pipeline.reporting.discovery import (
    experiment_names_in,
    find_run_dirs,
    is_run_dir,
    load_runs,
    resolve_run_dirs,
    runs_for_experiment_name,
)


def _run(root, name: str, **metadata):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "experiment_id": metadata.get("experiment_id", name[:12]),
        "target_family": metadata.get("target_family", "total_points"),
        "created_at": metadata.get("created_at", "2026-08-03T00:00:00+00:00"),
        **metadata,
    }))
    return run_dir


def _config(folder, filename: str, experiment_name: str):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(yaml.safe_dump({
        "experiment_name": experiment_name,
        "target_family": "total_points",
        "line_col": "ODDS_TOTAL_LINE_bet365",
    }))
    return path


@pytest.fixture
def workspace(tmp_path):
    artifacts = tmp_path / "artifacts" / "experiments"
    _run(artifacts, "total_points_2500_20260803_002317")
    _run(artifacts, "total_points_2500_old_games_20260704_20260802_115707")
    _run(artifacts, "classifier_2500_20260803_005940")

    campaign = tmp_path / "experiments" / "strategy_window_2026_08"
    _config(campaign, "total_points_2500.yaml", "total_points_2500")
    _config(campaign, "classifier_2500.yaml", "classifier_2500")

    return {"root": tmp_path, "artifacts": artifacts, "campaign": campaign}


def test_super_folder_yields_every_run(workspace):
    found = resolve_run_dirs(workspace["artifacts"], artifacts_root=workspace["artifacts"])
    assert len(found) == 3


def test_runs_nested_in_subfolders_are_still_found(tmp_path):
    """Runs may later be filed per campaign; discovery must not care."""
    artifacts = tmp_path / "artifacts" / "experiments"
    _run(artifacts / "campaign_a", "total_points_2500_20260803_002317")
    _run(artifacts / "campaign_b", "classifier_2500_20260803_005940")

    assert len(find_run_dirs(artifacts)) == 2


def test_campaign_config_folder_resolves_to_its_runs(workspace):
    found = resolve_run_dirs(workspace["campaign"], artifacts_root=workspace["artifacts"])
    assert {p.name for p in found} == {
        "total_points_2500_20260803_002317",
        "classifier_2500_20260803_005940",
    }


def test_a_shorter_experiment_name_does_not_swallow_a_longer_one(workspace):
    """The whole reason matching is anchored on the timestamp suffix."""
    found = runs_for_experiment_name("total_points_2500", workspace["artifacts"])

    assert [p.name for p in found] == ["total_points_2500_20260803_002317"]
    assert not any("old_games" in p.name for p in found)


def test_single_config_file_resolves_to_one_experiment(workspace):
    found = resolve_run_dirs(
        workspace["campaign"] / "classifier_2500.yaml", artifacts_root=workspace["artifacts"]
    )
    assert [p.name for p in found] == ["classifier_2500_20260803_005940"]


def test_individual_run_directory_is_accepted_directly(workspace):
    run_dir = workspace["artifacts"] / "classifier_2500_20260803_005940"
    assert is_run_dir(run_dir)
    assert resolve_run_dirs(run_dir, artifacts_root=workspace["artifacts"]) == [run_dir]


def test_mixed_sources_are_deduplicated(workspace):
    """A campaign folder plus one of its own runs must not double-count it."""
    found = resolve_run_dirs(
        [workspace["campaign"], workspace["artifacts"] / "classifier_2500_20260803_005940"],
        artifacts_root=workspace["artifacts"],
    )
    assert len(found) == len({p.resolve() for p in found}) == 2


def test_experiment_name_is_read_from_the_yaml_not_the_filename(tmp_path):
    """Nothing enforces that the two agree, and the run is named after the field."""
    campaign = tmp_path / "campaign"
    _config(campaign, "misleading_filename.yaml", "the_real_name")
    assert experiment_names_in(campaign) == ["the_real_name"]


def test_base_yaml_is_not_treated_as_an_experiment(tmp_path):
    """`_base.yaml` holds shared defaults and defines no experiment of its own."""
    campaign = tmp_path / "campaign"
    _config(campaign, "real.yaml", "real_experiment")
    (campaign / "_base.yaml").write_text(yaml.safe_dump({"experiment_name": "base"}))

    assert experiment_names_in(campaign) == ["real_experiment"]


def test_a_missing_source_names_itself(workspace):
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        resolve_run_dirs(
            workspace["root"] / "does_not_exist", artifacts_root=workspace["artifacts"]
        )


def test_relative_paths_resolve_against_the_project_root(workspace):
    """Notebooks run from experiments/notebooks/, so a path relative to the repo
    root would not exist relative to the working directory."""
    found = resolve_run_dirs(
        "artifacts/experiments",
        artifacts_root="artifacts/experiments",
        project_root=workspace["root"],
    )
    assert len(found) == 3
    assert all(str(path).startswith(str(workspace["root"])) for path in found)


def test_project_root_wins_over_a_same_named_folder_in_the_cwd(workspace, monkeypatch):
    """A "try cwd first, else project_root" rule would make the same SOURCES
    string resolve differently depending on where the kernel started. It must
    always mean the project-root-relative path.
    """
    decoy = workspace["root"] / "decoy"
    _run(decoy / "artifacts" / "experiments", "decoy_run_20260101_000000")
    monkeypatch.chdir(decoy)

    found = resolve_run_dirs(
        "artifacts/experiments",
        artifacts_root="artifacts/experiments",
        project_root=workspace["root"],
    )
    assert len(found) == 3
    assert not any("decoy_run" in path.name for path in found)


def test_load_runs_labels_each_row_with_its_provenance(workspace):
    frame = load_runs(workspace["artifacts"], artifacts_root=workspace["artifacts"])
    assert len(frame) == 3
    assert set(frame["source_root"]) == {"experiments"}
    assert frame["run_dir"].str.contains("artifacts/experiments").all()


def test_load_runs_refuses_an_empty_selection(tmp_path):
    empty = tmp_path / "artifacts" / "experiments"
    empty.mkdir(parents=True)
    with pytest.raises(ValueError, match="No runs found"):
        load_runs(empty, artifacts_root=empty)
