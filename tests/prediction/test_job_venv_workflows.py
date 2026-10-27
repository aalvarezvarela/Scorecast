"""Self-hosted jobs install through the shared job-venv action and log unbuffered."""

import subprocess

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
