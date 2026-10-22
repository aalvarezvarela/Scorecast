"""cron-job.org client shared by the scripts that book workflow dispatches.

cron-job.org calls GitHub's ``workflow_dispatch`` endpoint at a wall-clock
time, so each job carries the GitHub token in its request headers. The REST
API allows 100 requests a day and creates (``PUT /jobs``) at most once a
second and five times a minute; callers keep their request count small.

Secrets (never printed), each read from the environment first, then from any
section of ``config.secrets.ini`` (repo root, then ``src/nba_ou/``):
  - ``CRONJOB_API_KEY``: cron-job.org Console -> Settings -> API.
  - ``CRONJOB_GITHUB_DISPATCH_TOKEN``: fine-grained PAT on this repo only, with
    "Actions: Read and write". cron-job.org stores it in the job's headers.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
SECRETS_FILES = [
    REPO_ROOT / "config.secrets.ini",
    REPO_ROOT / "src/nba_ou/config.secrets.ini",
]

GITHUB_REPO = "aalvarezvarela/Scorecast"
GITHUB_REF = "main"
CRONJOB_API = "https://api.cron-job.org"
REQUEST_METHOD_POST = 1


def read_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    for path in SECRETS_FILES:
        if not path.exists():
            continue
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str  # keep key case
        parser.read(path)
        for section in parser.sections():
            for key, raw in parser.items(section):
                if key.upper() == name and raw.strip():
                    return raw.strip().strip('"').strip("'")
    sys.exit(f"Missing secret {name}: set it in the environment or config.secrets.ini")


def dispatch_url(workflow_file: str) -> str:
    return (
        f"https://api.github.com/repos/{GITHUB_REPO}"
        f"/actions/workflows/{workflow_file}/dispatches"
    )


def github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def dispatch_body() -> str:
    return json.dumps({"ref": GITHUB_REF})


def redacted(payload: dict) -> dict:
    shown = json.loads(json.dumps(payload))
    shown["extendedData"]["headers"]["Authorization"] = "Bearer <redacted>"
    return shown


def cronjob_request(method: str, path: str, api_key: str, body: dict | None = None):
    resp = requests.request(
        method,
        f"{CRONJOB_API}{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        sys.exit(
            f"cron-job.org {method} {path}: HTTP {resp.status_code} {resp.text[:300]}"
        )
    return resp.json()
