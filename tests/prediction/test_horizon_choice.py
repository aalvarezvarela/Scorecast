"""Which slots a manual run predicts with (nba_ou.prediction.horizon_choice)."""

import pytest
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.prediction.horizon_choice import (
    SCOPE_ALL,
    SCOPE_CLOSING,
    SCOPE_NEAREST,
    assign_horizons,
    group_by_horizon,
    nearest_horizon,
    parse_scope,
    scope_slots,
    slot_games,
)

HORIZONS = (30, 60, 120, 240, 360, 480, 720)


def slots():
    out = []
    for target in ("line_error", "total_points"):
        out.append(ModelSlot("2_5", target, 0))
        out += [ModelSlot("2_5", target, h) for h in HORIZONS]
    return out


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(400, 360), (430, 480), (240, 240), (5, 30), (5000, 720)],
)
def test_nearest_horizon(minutes, expected):
    assert nearest_horizon(minutes, HORIZONS) == expected


def test_a_tie_goes_to_the_longer_horizon():
    # 300 is 60 from both 240 and 360.
    assert nearest_horizon(300, HORIZONS) == 360


def test_the_users_example_three_games_three_models():
    assigned = assign_horizons({"g4": 241, "g8": 475, "g12": 725}, HORIZONS)
    assert assigned == {"g4": 240, "g8": 480, "g12": 720}
    assert group_by_horizon(assigned) == {720: ["g12"], 480: ["g8"], 240: ["g4"]}


def test_scope_parsing():
    assert parse_scope("") == SCOPE_ALL
    assert parse_scope(None) == SCOPE_ALL
    assert parse_scope("Nearest") == SCOPE_NEAREST
    assert parse_scope("0") == SCOPE_CLOSING
    assert parse_scope("360") == 360
    with pytest.raises(ValueError):
        parse_scope("soon")
    with pytest.raises(ValueError):
        parse_scope("-30")


def test_scope_slots():
    every = slots()
    assert scope_slots(every, SCOPE_ALL) == every
    assert {s.horizon_minutes for s in scope_slots(every, SCOPE_CLOSING)} == {0}
    assert 0 not in {s.horizon_minutes for s in scope_slots(every, SCOPE_NEAREST)}
    fixed = scope_slots(every, 360)
    assert {(str(s.target), s.horizon_minutes) for s in fixed} == {
        ("line_error", 360),
        ("total_points", 360),
    }


def test_each_game_goes_to_one_slot_per_target():
    chosen = slot_games(scope_slots(slots(), SCOPE_NEAREST), {"a": 250, "b": 470})
    by_slot = {(str(s.target), s.horizon_minutes): g for s, g in chosen.items()}
    assert by_slot == {
        ("line_error", 240): ["a"],
        ("line_error", 480): ["b"],
        ("total_points", 240): ["a"],
        ("total_points", 480): ["b"],
    }


def test_targets_with_different_grids_pick_their_own_nearest():
    mixed = [ModelSlot("2_5", "line_error", 240), ModelSlot("2_5", "spread_error", 360)]
    chosen = slot_games(mixed, {"a": 250})
    assert {(str(s.target), s.horizon_minutes) for s in chosen} == {
        ("line_error", 240),
        ("spread_error", 360),
    }
