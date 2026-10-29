"""Self-hosted jobs install through the shared job-venv action and log unbuffered."""

import os
import subprocess
from unittest.mock import Mock

import pytest
import yaml

from tests.support.paths import REPO_ROOT

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ACTION = REPO_ROOT / ".github" / "actions" / "job-venv"


def _self_hosted_jobs():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for name, job in workflow["jobs"].items():
            if job.get("runs-on") == "self-hosted":
                yield path.name, name, job


SELF_HOSTED = list(_self_hosted_jobs())


def test_there_are_self_hosted_jobs():
    assert len(SELF_HOSTED) >= 7


@pytest.mark.parametrize(
    "workflow,name,job", SELF_HOSTED, ids=[f"{w}:{n}" for w, n, _ in SELF_HOSTED]
)
def test_a_self_hosted_job_reuses_its_venv_and_logs_unbuffered(workflow, name, job):
    steps = job["steps"]
    venvs = [s for s in steps if s.get("uses") == "./.github/actions/job-venv"]
    assert len(venvs) == 1
    requirements = venvs[0]["with"]["requirements"]
    assert (REPO_ROOT / requirements).is_file()
    # The checkout must come first: the action is read from it.
    assert steps.index(venvs[0]) > next(
        i
        for i, s in enumerate(steps)
        if s.get("uses", "").startswith("actions/checkout")
    )
    # No step builds a venv or installs packages by hand any more.
    scripts = "\n".join(s.get("run", "") for s in steps)
    assert "python -m venv" not in scripts
    assert "pip install" not in scripts
    assert job["env"]["PYTHONUNBUFFERED"] == "1"


def test_the_venv_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(ACTION / "job_venv.sh")], check=True)


@pytest.mark.parametrize("modes", ["standard", "standard thinking_high"])
def test_standalone_closing_step_passes_valid_closing_options(
    monkeypatch, tmp_path, modes
):
    from scripts import predict_tabpfn

    workflow = yaml.load(
        (WORKFLOWS / "nba_predictor_daily_tabpfn_client.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    step = next(
        s
        for s in workflow["jobs"]["run"]["steps"]
        if s.get("name") == "Run additional TabPFN closing line"
    )
    assert "inputs.include_closing" in step["if"]
    capture = tmp_path / "args"
    script = 'python() { printf "%s\\n" "$@" > "$CAPTURE_ARGS"; }\n' + step[
        "run"
    ].replace("${{ !inputs.persist && '--no-persist' || '' }}", "--no-persist")
    subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "CAPTURE_ARGS": str(capture),
            "TABPFN_TARGETS": "line_error total_points spread_error",
            "TABPFN_MODES": modes,
            "TABPFN_GAME_DATE": "2026-10-29",
            "TABPFN_GAME_IDS": "",
        },
        check=True,
        capture_output=True,
    )
    predict = Mock()
    monkeypatch.setattr(predict_tabpfn, "predict_tabpfn", predict)
    args = capture.read_text().splitlines()
    assert args[0] == "scripts/predict_tabpfn.py"
    predict_tabpfn.main(args[1:])
    options = predict.call_args.kwargs
    assert options["line"] == "closing"
    assert options["targets"] == ["line_error", "total_points", "spread_error"]
    assert options["modes"] == modes.split()
    assert options["horizon_minutes"] is None
    assert options["as_of"] is None
    assert options["snapshot_game_ids"] is None
    assert options["game_date"] == "2026-10-29"
    assert options["update_db"] is False and options["persist"] is False
