"""The phase 4A minutes provider v1: ``q * (b + C)``, reconciled to the
physical constraints.

Per team and availability scenario, the raw minutes of each available player
are ``raw_i = q_i * (b_i + C_i)`` (participation A, baseline B, absence gains C,
``rotation`` and ``participation``); players out in the scenario play 0. The
reconciliation then imposes

    0 <= minutes_i <= 48        sum_i minutes_i = 240

with ``minutes_i = min(48, c * raw_i)`` and the scale ``c`` found by
bisection: the weighted least-squares projection (weights ``1 / raw_i``) onto
the capped simplex. It is proportional rescaling, as ``allocate_minutes``
does, until the 48-minute cap binds, and then the capped players' excess goes
to the others in proportion. ``raw_i >= 0`` keeps every minute non-negative.
Fewer than five players with positive raw minutes cannot fill 240 minutes:
such a team is flagged infeasible and left at its capped values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEAM_MINUTES = 240.0
PLAYER_CAP = 48.0


@dataclass(frozen=True)
class Reconciled:
    minutes: np.ndarray
    scale: float  # c
    feasible: bool
    capped: int  # players at the 48-minute cap


def reconcile(
    raw: np.ndarray,
    team_minutes: float = TEAM_MINUTES,
    cap: float = PLAYER_CAP,
    tolerance: float = 1e-9,
) -> Reconciled:
    """``min(cap, c * raw)`` summing to ``team_minutes``; see the module."""
    raw = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    positive = raw > 0
    if positive.sum() * cap < team_minutes - tolerance:
        minutes = np.where(positive, cap, 0.0)
        return Reconciled(minutes, np.inf, False, int(positive.sum()))
    low, high = 0.0, team_minutes / raw[positive].min()  # high fills every cap
    for _ in range(200):
        middle = (low + high) / 2
        if np.minimum(cap, middle * raw).sum() < team_minutes:
            low = middle
        else:
            high = middle
        if high - low < tolerance * max(1.0, high):
            break
    minutes = np.minimum(cap, high * raw)
    # Close the last rounding gap on the uncapped players.
    free = minutes < cap - 1e-12
    gap = team_minutes - minutes.sum()
    if free.any() and abs(gap) > 0:
        minutes[free] += gap * raw[free] / raw[free].sum()
    return Reconciled(minutes, float(high), True, int((minutes >= cap - 1e-9).sum()))
