"""Start a manual request's workflow on GitHub (``workflow_dispatch``).

The same call cron-job.org makes for the scheduled runs
(``nba_ou.scheduling.cronjob_org``), with the dashboard's own token: a
fine-grained PAT on this repository with "Actions: Read and write", in the
Streamlit secret ``GITHUB_DISPATCH_TOKEN``. Never logged or shown.
"""

from __future__ import annotations

import requests

from nba_ou.scheduling.cronjob_org import dispatch_body, dispatch_url, github_headers

TIMEOUT_SECONDS = 15


class DispatchError(RuntimeError):
    """A safe message: the response status, never the token or headers."""


def dispatch_workflow(token: str, workflow_file: str, inputs: dict[str, str]) -> None:
    if not token:
        raise DispatchError("GITHUB_DISPATCH_TOKEN is not configured")
    try:
        response = requests.post(
            dispatch_url(workflow_file),
            headers=github_headers(token),
            data=dispatch_body(inputs),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise DispatchError(f"GitHub unreachable ({type(exc).__name__})") from None
    if response.status_code not in (200, 204):
        raise DispatchError(
            f"GitHub refused the dispatch of {workflow_file} "
            f"(HTTP {response.status_code})"
        )
