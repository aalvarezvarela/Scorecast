"""How a predictor dispatch names the TabPFN jobs it carries (``--tabpfn``).

``HORIZON[@AS_OF[@GAME_IDS]]`` items separated by ``;``: the scheduler writes
them (``scripts/cronjobs/schedule_snapshot_triggers.py``), and
``nba_ou.prediction.tabpfn_run.parse_jobs`` reads them. Kept apart from that
module so the scheduler, which installs only the core requirements, does not
import the feature builders.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

#: Separates the jobs, and the fields of each.
JOB_SEPARATOR = ";"
FIELD_SEPARATOR = "@"


def format_job(horizon: int, as_of: datetime, game_ids: Collection[str]) -> str:
    """The T-``horizon`` job of the snapshot ``as_of``, booked for ``game_ids``."""
    return FIELD_SEPARATOR.join(
        [str(horizon), as_of.isoformat(), " ".join(sorted(game_ids))]
    )
