"""How a predictor dispatch names the TabPFN jobs it carries (``--tabpfn``).

``HORIZON[@AS_OF[@GAME_IDS[@FITS]]]`` items separated by ``;``: the scheduler
writes them (``scripts/cronjobs/schedule_snapshot_triggers.py``), and
``nba_ou.prediction.tabpfn_run.parse_jobs`` reads them. Kept apart from that
module so the scheduler, which installs only the core requirements, does not
import the feature builders.

``FITS`` names a job's fits as ``MODE=TARGET[,TARGET]`` items separated by
spaces, e.g. ``standard=line_error,total_points thinking_high=line_error``:
each mode fits only its targets. Without it a job fits every target of the
run in every mode of the run. The same text is ``--fits`` of
``scripts/predict_tabpfn.py``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import datetime

#: Separates the jobs, and the fields of each.
JOB_SEPARATOR = ";"
FIELD_SEPARATOR = "@"
#: Within ``FITS``: the fits, each mode from its targets, and the targets.
FIT_SEPARATOR = " "
MODE_SEPARATOR = "="
TARGET_SEPARATOR = ","
DEFAULT_TARGETS = ("line_error", "total_points", "spread_error")

#: ``(mode, targets)`` per mode, in fitting order.
Fits = tuple[tuple[str, tuple[str, ...]], ...]


def format_fits(fits: Sequence[tuple[str, Sequence[str]]]) -> str:
    """``standard=line_error,total_points thinking_high=line_error``."""
    return FIT_SEPARATOR.join(
        f"{mode}{MODE_SEPARATOR}{TARGET_SEPARATOR.join(targets)}"
        for mode, targets in fits
    )


def format_job(
    horizon: int,
    as_of: datetime | None = None,
    game_ids: Collection[str] = (),
    fits: Sequence[tuple[str, Sequence[str]]] = (),
) -> str:
    """The T-``horizon`` job of the snapshot ``as_of``, booked for ``game_ids``,
    fitting ``fits`` (empty: the run's targets in the run's modes). Horizon 0
    is the closing line, without ``as_of`` or ``game_ids``."""
    fields = [
        str(horizon),
        "" if as_of is None else as_of.isoformat(),
        " ".join(sorted(game_ids)),
    ]
    if fits:
        fields.append(format_fits(fits))
    return FIELD_SEPARATOR.join(fields).rstrip(FIELD_SEPARATOR)
