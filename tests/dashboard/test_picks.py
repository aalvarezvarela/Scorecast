import pandas as pd
import pytest
from nba_ou.dashboard.domain.picks import GamePick, bet_text, game_pick


def picks_of(*signals):
    return game_pick(pd.DataFrame({"signal": list(signals)}))


def test_unanimous_directions_are_a_bet_even_with_neutral_predictions():
    assert picks_of("OVER", "NO EDGE", "OVER") == GamePick(
        "BET", "OVER", 2, 3, {"OVER": 2}
    )


def test_any_opposing_direction_is_no_bet():
    pick = picks_of("HOME", "AWAY", "HOME")
    assert pick.status == "SPLIT"
    assert pick.counts == {"HOME": 2, "AWAY": 1}


def test_predictions_without_a_line_do_not_vote():
    assert picks_of("UNDER", "UNAVAILABLE").predictions == 1
    assert picks_of("UNAVAILABLE").status == "NO LINE"
    assert picks_of("NO EDGE").status == "NO BET"
    assert game_pick(pd.DataFrame({"signal": []})).status == "PENDING"


@pytest.mark.parametrize(
    "side,line,text",
    [
        ("OVER", 230.5, "OVER 230.5"),
        ("UNDER", None, "UNDER"),
        # Implied home margin 3.5: the home team gives 3.5, the away team gets it.
        ("HOME", 3.5, "Celtics -3.5"),
        ("AWAY", 3.5, "Knicks +3.5"),
        ("HOME", -2.0, "Celtics +2.0"),
        ("AWAY", 0.0, "Knicks PK"),
    ],
)
def test_bet_text_names_the_side_and_line_to_bet(side, line, text):
    assert bet_text(side, line, home="Celtics", away="Knicks") == text
