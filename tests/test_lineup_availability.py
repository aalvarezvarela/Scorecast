"""Building tonight's rosters: recency, trades, the report and leakage."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.lineups.availability import (
    build_player_nights,
    player_out_probabilities,
)

HOME = "1610612739"
AWAY = "1610612738"


def _games(rows):
    return pd.DataFrame(
        [{"GAME_ID": g, "TEAM_ID": t, "GAME_DATE": d} for g, t, d in rows]
    )


def _box(rows):
    return pd.DataFrame(
        [
            {"GAME_ID": g, "TEAM_ID": t, "GAME_DATE": d, "PLAYER_ID": p, "MIN": m}
            for g, t, d, p, m in rows
        ]
    )


def _history(n=6, team=HOME, minutes=((("a"), 30.0), (("b"), 20.0), (("c"), 10.0))):
    rows = []
    for index in range(n):
        for player, played in minutes:
            rows.append(
                (f"00{index:08d}", team, f"2025-01-{index + 1:02d}", player, played)
            )
    return _box(rows)


def test_minutes_are_the_recent_average_and_the_roster_is_sorted():
    target = _games([("0000000099", HOME, "2025-01-10")])
    nights = build_player_nights(target, _history())
    roster = nights[("0000000099", HOME)]
    assert [player.player_id for player in roster] == ["a", "b", "c"]
    assert roster[0].base_minutes == pytest.approx(30.0)
    assert roster[2].base_minutes == pytest.approx(10.0)
    assert all(player.p_out == 0.0 for player in roster)


def test_a_player_absent_all_window_leaves_the_roster():
    """Otherwise his stale average absorbs minutes from his own replacement."""
    rows = list(_history(n=2).itertuples(index=False, name=None))
    rows = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
    # Six further games without "a"; the window is five.
    for index in range(2, 8):
        for player, played in (("b", 25.0), ("c", 15.0)):
            rows.append(
                (f"00{index:08d}", HOME, f"2025-01-{index + 1:02d}", player, played)
            )
    target = _games([("0000000099", HOME, "2025-02-01")])
    nights = build_player_nights(
        _games([("0000000099", HOME, "2025-02-01")]), _box(rows)
    )
    assert {player.player_id for player in nights[("0000000099", HOME)]} == {"b", "c"}
    assert target is not None


def test_a_traded_player_leaves_his_old_roster():
    rows = list(_history(n=3).itertuples(index=False, name=None))
    rows = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
    rows.append(("0000000010", AWAY, "2025-01-05", "a", 28.0))
    nights = build_player_nights(
        _games(
            [("0000000099", HOME, "2025-01-10"), ("0000000099", AWAY, "2025-01-10")]
        ),
        _box(rows),
    )
    assert "a" not in {p.player_id for p in nights[("0000000099", HOME)]}
    assert "a" in {p.player_id for p in nights[("0000000099", AWAY)]}


def test_the_report_sets_p_out_and_silence_means_available():
    target = _games([("0000000099", HOME, "2025-01-10")])
    nights = build_player_nights(
        target, _history(), p_out={("0000000099", HOME, "a"): 0.42}
    )
    by_player = {p.player_id: p for p in nights[("0000000099", HOME)]}
    assert by_player["a"].p_out == pytest.approx(0.42)
    assert by_player["b"].p_out == 0.0


def test_tonights_box_score_never_reaches_tonights_roster():
    """The mandatory leakage check."""
    target = _games([("0000000099", HOME, "2025-01-10")])
    clean = build_player_nights(target, _history())
    polluted_rows = list(_history().itertuples(index=False, name=None))
    polluted_rows = [(r[0], r[1], r[2], r[3], r[4]) for r in polluted_rows]
    polluted_rows.append(("0000000099", HOME, "2025-01-10", "zz", 48.0))
    polluted = build_player_nights(target, _box(polluted_rows))
    assert clean[("0000000099", HOME)] == polluted[("0000000099", HOME)]


def test_same_day_games_are_held_back_together():
    rows = list(_history(n=3).itertuples(index=False, name=None))
    rows = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
    rows.append(("0000000050", HOME, "2025-01-10", "zz", 40.0))
    nights = build_player_nights(
        _games([("0000000099", HOME, "2025-01-10")]), _box(rows)
    )
    assert "zz" not in {p.player_id for p in nights[("0000000099", HOME)]}


def test_a_team_with_no_history_is_absent_rather_than_guessed():
    nights = build_player_nights(
        _games([("0000000001", HOME, "2025-01-01")]), _history()
    )
    assert ("0000000001", HOME) not in nights


def test_out_probabilities_come_from_chance_out():
    statuses = pd.DataFrame(
        {
            "game_id": ["0022400001"] * 4,
            "team_id": [HOME] * 4,
            "player_id": ["a", "b", "c", "d"],
            "status": ["out", "questionable", "probable", "available"],
            "reason_category": ["injury"] * 4,
        }
    )
    probabilities = player_out_probabilities(statuses)
    assert probabilities[("0022400001", HOME, "a")] == pytest.approx(1.0)
    # Questionable and probable fall back to the league-typical play rates.
    assert probabilities[("0022400001", HOME, "b")] == pytest.approx(0.42)
    assert probabilities[("0022400001", HOME, "c")] == pytest.approx(0.05)
    assert probabilities[("0022400001", HOME, "d")] == pytest.approx(0.02)


def test_empty_statuses_give_no_probabilities():
    assert player_out_probabilities(pd.DataFrame()) == {}


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Team games are missing"):
        build_player_nights(pd.DataFrame({"GAME_ID": ["1"]}), _history())
    with pytest.raises(ValueError, match="Player history is missing"):
        build_player_nights(
            _games([("1", HOME, "2025-01-10")]), pd.DataFrame({"GAME_ID": ["1"]})
        )
    with pytest.raises(ValueError, match="recent_games must be positive"):
        build_player_nights(
            _games([("1", HOME, "2025-01-10")]), _history(), recent_games=0
        )
