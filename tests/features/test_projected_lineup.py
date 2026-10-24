"""Projected starting five: replacement choice and the temporal contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.projected_lineup import (
    PROJECTED_LINEUP_COLUMNS,
    add_projected_lineup_features,
)

TEAM = "1610612739"


def _box(game_id, date, starters, bench=(), minutes=30.0, season=2024):
    """One finished team-game box score."""
    rows = []
    for player in starters:
        rows.append(
            {
                "GAME_ID": game_id,
                "TEAM_ID": TEAM,
                "GAME_DATE": date,
                "PLAYER_ID": player,
                "START_POSITION": "G",
                "MIN": minutes,
                "SEASON_YEAR": season,
            }
        )
    for player, played in bench:
        rows.append(
            {
                "GAME_ID": game_id,
                "TEAM_ID": TEAM,
                "GAME_DATE": date,
                "PLAYER_ID": player,
                "START_POSITION": "",
                "MIN": played,
                "SEASON_YEAR": season,
            }
        )
    return rows


FIVE = ["p1", "p2", "p3", "p4", "p5"]


def _history():
    """Three games with the usual five, then one p1 missed and p9 started."""
    rows = []
    rows += _box("001", "2025-01-01", FIVE, bench=[("p9", 8.0), ("p8", 12.0)])
    rows += _box("002", "2025-01-03", FIVE, bench=[("p9", 8.0), ("p8", 12.0)])
    # p1 out: p9 started in his place.
    rows += _box(
        "003",
        "2025-01-05",
        ["p9", "p2", "p3", "p4", "p5"],
        bench=[("p8", 12.0)],
    )
    rows += _box("004", "2025-01-07", FIVE, bench=[("p9", 8.0), ("p8", 12.0)])
    return pd.DataFrame(rows)


def _target(game_id="005", date="2025-01-09", season=2024):
    return pd.DataFrame(
        [
            {
                "GAME_ID": game_id,
                "TEAM_ID": TEAM,
                "GAME_DATE": date,
                "SEASON_YEAR": season,
            }
        ]
    )


def test_no_absence_keeps_the_latest_five():
    result = add_projected_lineup_features(_target(), _history(), out_sets={})
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    assert projection["projected_five"] == frozenset(FIVE)
    assert result["LU_PROJ_STARTERS_REPLACED_BEFORE"].iloc[0] == 0


def test_absent_starter_is_replaced_by_his_historical_replacement():
    result = add_projected_lineup_features(
        _target(), _history(), out_sets={("005", TEAM): ["p1"]}
    )
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    assert projection["projected_five"] == frozenset(["p9", "p2", "p3", "p4", "p5"])
    assert projection["replacements"] == ("p9",)
    assert result["LU_PROJ_STARTERS_REPLACED_BEFORE"].iloc[0] == 1
    # p9 started once in this season's history, so that is his start count.
    assert result["LU_REPL_MIN_STARTS_SEASON_BEFORE"].iloc[0] == 1
    assert result["LU_REPL_PRIOR_EVIDENCE_BEFORE"].iloc[0] == 1


def test_absence_without_precedent_falls_back_to_recent_minutes():
    """p2 has never missed a game, so the busiest available player wins instead."""
    result = add_projected_lineup_features(
        _target(), _history(), out_sets={("005", TEAM): ["p2"]}
    )
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    # Over the window p9 has 8 + 8 + 30 (his one start) + 8 = 54 minutes against
    # p8's 4 x 12 = 48, so p9 is the most-used available player, not p8.
    assert projection["replacements"] == ("p9",)
    # Zero evidence: this choice rests on minutes, not on a precedent for p2.
    assert result["LU_REPL_PRIOR_EVIDENCE_BEFORE"].iloc[0] == 0


def test_replacement_is_never_someone_also_projected_out():
    result = add_projected_lineup_features(
        _target(), _history(), out_sets={("005", TEAM): ["p1", "p9"]}
    )
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    assert "p9" not in projection["projected_five"]
    assert projection["replacements"] == ("p8",)


def test_target_game_box_score_never_reaches_its_own_features():
    """The mandatory leakage test: perturbing the target game changes nothing."""
    history = _history()
    target = _target()
    out_sets = {("005", TEAM): ["p1"]}
    before = add_projected_lineup_features(target, history, out_sets=out_sets)

    # The target game finishes with a completely different five.
    polluted = pd.concat(
        [
            history,
            pd.DataFrame(_box("005", "2025-01-09", ["p6", "p7", "p8", "p9", "p2"])),
        ],
        ignore_index=True,
    )
    after = add_projected_lineup_features(target, polluted, out_sets=out_sets)

    pd.testing.assert_frame_equal(
        before[list(PROJECTED_LINEUP_COLUMNS)], after[list(PROJECTED_LINEUP_COLUMNS)]
    )
    assert (
        before.attrs["projected_lineups"][("005", TEAM)]["projected_five"]
        == after.attrs["projected_lineups"][("005", TEAM)]["projected_five"]
    )


def test_same_day_games_are_held_back_together():
    """A game cannot read another game played the same calendar date."""
    history = pd.concat(
        [
            _history(),
            pd.DataFrame(_box("005", "2025-01-09", ["p6", "p7", "p8", "p9", "p2"])),
        ],
        ignore_index=True,
    )
    target = pd.DataFrame(
        [
            {
                "GAME_ID": "006",
                "TEAM_ID": TEAM,
                "GAME_DATE": "2025-01-09",
                "SEASON_YEAR": 2024,
            }
        ]
    )
    result = add_projected_lineup_features(target, history, out_sets={})
    projection = result.attrs["projected_lineups"][("006", TEAM)]
    # Game 005 is same-date, so the latest usable five is still game 004's.
    assert projection["latest_five"] == frozenset(FIVE)


def test_team_without_history_gets_nan_not_a_guess():
    target = pd.DataFrame(
        [
            {
                "GAME_ID": "001",
                "TEAM_ID": TEAM,
                "GAME_DATE": "2025-01-01",
                "SEASON_YEAR": 2024,
            }
        ]
    )
    result = add_projected_lineup_features(target, _history(), out_sets={})
    assert result["LU_PROJ_STARTERS_REPLACED_BEFORE"].isna().all()
    assert ("001", TEAM) not in result.attrs["projected_lineups"]


def test_uncovered_team_game_projects_nobody_out():
    """No report is not the same as an empty report, but neither invents absences."""
    result = add_projected_lineup_features(_target(), _history(), out_sets=None)
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    assert projection["projected_five"] == frozenset(FIVE)
    assert np.isnan(result["LU_REPL_MIN_STARTS_SEASON_BEFORE"].iloc[0])


def test_malformed_box_score_is_skipped_not_read_as_a_change():
    """Four starters means a broken row, not a four-man lineup."""
    broken = pd.concat(
        [
            _history(),
            pd.DataFrame(_box("004b", "2025-01-08", ["p6", "p7", "p8", "p9"])),
        ],
        ignore_index=True,
    )
    result = add_projected_lineup_features(_target(), broken, out_sets={})
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    assert projection["latest_five"] == frozenset(FIVE)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Team games are missing"):
        add_projected_lineup_features(pd.DataFrame({"GAME_ID": ["1"]}), _history())
    with pytest.raises(ValueError, match="Player history is missing"):
        add_projected_lineup_features(_target(), pd.DataFrame({"GAME_ID": ["1"]}))


def _traded_history():
    """p9 covers for p1 in January, then stops appearing: he was traded away."""
    rows = []
    rows += _box("001", "2025-01-01", FIVE, bench=[("p9", 8.0), ("p8", 12.0)])
    rows += _box(
        "002", "2025-01-03", ["p9", "p2", "p3", "p4", "p5"], bench=[("p8", 12.0)]
    )
    # p9 never appears again; twelve further games pass.
    for index in range(12):
        rows += _box(
            f"1{index:02d}",
            f"2025-01-{5 + index:02d}",
            FIVE,
            bench=[("p8", 12.0), ("p7", 10.0)],
        )
    return pd.DataFrame(rows)


def test_a_departed_player_is_never_projected_to_start():
    """The trade bug: p9 covered this exact absence, but he has left."""
    result = add_projected_lineup_features(
        _target(date="2025-01-20"), _traded_history(), out_sets={("005", TEAM): ["p1"]}
    )
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    assert "p9" not in projection["projected_five"]
    assert projection["replacements"] == ("p8",)


def test_a_player_on_tonights_report_counts_as_on_the_roster():
    """An acquired player has no box score yet; the report is how we know."""
    history = _traded_history()
    listed = {("005", TEAM): ["p9"]}
    result = add_projected_lineup_features(
        _target(date="2025-01-20"),
        history,
        out_sets={("005", TEAM): ["p1"]},
        listed_sets=listed,
    )
    projection = result.attrs["projected_lineups"][("005", TEAM)]
    # Back on the roster, and he is still the precedent for p1's absence.
    assert projection["replacements"] == ("p9",)


def test_season_opener_projects_from_the_previous_season():
    """Something at the start of a season beats nothing, but it is flagged."""
    previous = _history()
    target = pd.DataFrame(
        [
            {
                "GAME_ID": "900",
                "TEAM_ID": TEAM,
                "GAME_DATE": "2025-10-20",
                "SEASON_YEAR": 2025,
            }
        ]
    )
    result = add_projected_lineup_features(target, previous, out_sets={})
    projection = result.attrs["projected_lineups"][("900", TEAM)]
    assert projection["projected_five"] == frozenset(FIVE)
    # Zero games this season: the whole projection rests on last season.
    assert result["LU_PROJ_GAMES_THIS_SEASON_BEFORE"].iloc[0] == 0


def test_a_partial_report_at_an_opener_does_not_invent_departures():
    """The report lists ~5 players, not a squad; it must not shrink the five.

    Before the team has played a game this season there is no in-season
    appearance evidence, so a season-scoped roster would be the listing alone
    and four fifths of the returning five would look departed.
    """
    target = pd.DataFrame(
        [
            {
                "GAME_ID": "900",
                "TEAM_ID": TEAM,
                "GAME_DATE": "2025-10-20",
                "SEASON_YEAR": 2025,
            }
        ]
    )
    result = add_projected_lineup_features(
        target,
        _history(),
        out_sets={},
        listed_sets={("900", TEAM): ["p1", "p8"]},
    )
    projection = result.attrs["projected_lineups"][("900", TEAM)]
    assert projection["departed"] == frozenset()
    assert projection["projected_five"] == frozenset(FIVE)
    assert result["LU_PROJ_GAMES_THIS_SEASON_BEFORE"].iloc[0] == 0
