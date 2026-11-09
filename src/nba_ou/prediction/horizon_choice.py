"""Which models a manual run predicts with (``docs/serving/manual_predictions.md``).

A dashboard request names a scope:

* ``all`` -- every enabled slot (the scheduled runs);
* ``closing`` -- the closing slots;
* ``nearest`` -- per game, the intermediate slot whose horizon is nearest the
  game's minutes to tip-off at the run's moment, so games at 4, 8 and 12 hours
  are predicted by the T-240, T-480 and T-720 models;
* a number ``H`` -- the T-H slots, for every game.

Rows are always each game's latest snapshot; the horizon only chooses the
model (XGBoost) or the training rows (TabPFN). "Nearest" is
``argmin |H - minutes left|``, ties to the longer horizon, as the dashboard's
``select_predictions`` picks among stored outputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

SCOPE_ALL = "all"
SCOPE_CLOSING = "closing"
SCOPE_NEAREST = "nearest"


def parse_scope(value: str | int | None) -> str | int:
    """``all``, ``closing``, ``nearest`` or a positive horizon in minutes."""
    if value is None or value == "":
        return SCOPE_ALL
    if isinstance(value, int):
        horizon = value
    else:
        text = str(value).strip().lower()
        if text in (SCOPE_ALL, SCOPE_CLOSING, SCOPE_NEAREST):
            return text
        try:
            horizon = int(text)
        except ValueError:
            raise ValueError(
                f"Unknown scope {value!r}: all, closing, nearest or a horizon"
            ) from None
    if horizon < 0:
        raise ValueError(f"A negative horizon: {value!r}")
    return SCOPE_CLOSING if horizon == 0 else horizon


def nearest_horizon(minutes_left: float, horizons: Iterable[int]) -> int:
    """The horizon nearest ``minutes_left``; on a tie, the longer one."""
    candidates = sorted({int(h) for h in horizons if int(h) > 0}, reverse=True)
    if not candidates:
        raise ValueError("No intermediate horizon to choose from")
    return min(candidates, key=lambda h: abs(h - float(minutes_left)))


def assign_horizons(
    minutes_left: Mapping[str, float], horizons: Iterable[int]
) -> dict[str, int]:
    """Per game id, its nearest horizon among ``horizons``."""
    horizons = tuple(horizons)
    return {
        str(game): nearest_horizon(minutes, horizons)
        for game, minutes in minutes_left.items()
    }


def group_by_horizon(assigned: Mapping[str, int]) -> dict[int, list[str]]:
    """The games of each horizon, longest horizon first."""
    groups: dict[int, list[str]] = {}
    for game, horizon in assigned.items():
        groups.setdefault(int(horizon), []).append(str(game))
    return {h: sorted(groups[h]) for h in sorted(groups, reverse=True)}


def scope_slots(slots: Sequence, scope: str | int) -> list:
    """The slots a run of ``scope`` may use (``nearest``: every intermediate
    slot; each game then reads one of them, ``slot_games``)."""
    if scope == SCOPE_ALL:
        return list(slots)
    if scope == SCOPE_CLOSING:
        return [slot for slot in slots if not slot.horizon_minutes]
    if scope == SCOPE_NEAREST:
        return [slot for slot in slots if slot.horizon_minutes]
    return [slot for slot in slots if slot.horizon_minutes == int(scope)]


def slot_games(
    slots: Sequence, minutes_left: Mapping[str, float]
) -> dict[object, list[str]]:
    """``nearest``: per intermediate slot, the games it predicts.

    Each game goes to the slot of its nearest horizon among the slots of the
    same target (and schema version and variant), so targets with different
    horizon grids each get their own nearest model. A slot with no game is
    left out.
    """
    families: dict[tuple, list] = {}
    for slot in slots:
        if slot.horizon_minutes:
            key = (slot.schema_version, slot.target, slot.variant)
            families.setdefault(key, []).append(slot)
    out: dict[object, list[str]] = {}
    for members in families.values():
        by_horizon = {int(slot.horizon_minutes): slot for slot in members}
        for game, horizon in assign_horizons(minutes_left, by_horizon).items():
            out.setdefault(by_horizon[horizon], []).append(game)
    return {slot: sorted(games) for slot, games in out.items()}
