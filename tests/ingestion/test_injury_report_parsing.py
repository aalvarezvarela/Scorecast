"""Regression contract for the injury-report PDF table parser.

The parser rejoins reason text that wrapped onto a second line by merging the
trailing fragments of a block. The trigger is punctuation -- ``-`` or ``;`` --
because that is the shape of a reason (``Category - Detail; Detail``).

Player names carry the same punctuation, and when one was merged it swallowed
the player's name, status and reason into a single cell and the row vanished
from the report. Measured against 20 real reports spanning 2019-12 to 2026-04:
every single dropped player had a hyphen in their name, among them
Gilgeous-Alexander, Alexander-Walker, Carter-Williams and Finney-Smith.

These tests are pure -- no network, no PDF fixture.
"""

from __future__ import annotations

import pytest
from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    STATUS_VALUES,
    is_positional_row,
    is_wrapped_reason_fragment,
)


@pytest.mark.parametrize(
    "name",
    [
        "Gilgeous-Alexander, Shai",
        "Alexander-Walker, Nickeil",
        "Carter-Williams, Michael",
        "Finney-Smith, Dorian",
        "Aminu, Al-Farouq",  # hyphen in the given name, not the surname
        "Smith Jr., Dennis",
        "Mbah a Moute, Luc",
    ],
)
def test_player_names_are_never_merged_away(name: str) -> None:
    """A name must never be treated as a wrapped reason, or its row is lost."""
    assert is_wrapped_reason_fragment(name) is False


@pytest.mark.parametrize("status", sorted(STATUS_VALUES))
def test_statuses_are_never_merged_away(status: str) -> None:
    assert is_wrapped_reason_fragment(status) is False


@pytest.mark.parametrize(
    "fragment",
    [
        "Injury/Illness - Right Knee;",
        "Injury/Illness -",
        "G League - On Assignment",
        "Left Hamstring; Strain",
        "Return to Competition Recond-",
    ],
)
def test_wrapped_reasons_are_still_merged(fragment: str) -> None:
    """The original behaviour has to survive: reasons still rejoin."""
    assert is_wrapped_reason_fragment(fragment) is True


@pytest.mark.parametrize("token", ["Out", "Chicago Bulls", "CHI@CLE", "12/08/2021", ""])
def test_tokens_without_reason_punctuation_are_not_merged(token: str) -> None:
    assert is_wrapped_reason_fragment(token) is False


class TestPositionalRows:
    """A 7-element block is only a 7-column row if it actually looks like one.

    Found against the real corpus: a row that starts a new game carries Game
    Time, Matchup and Team inline, so a six-field row whose reason wrapped onto
    a second line also arrives with seven elements. Mapping that by position
    shifted every column left -- ``Game Date`` became ``"07:30 (ET)"`` and
    ``Current Status`` became a fragment of the reason.
    """

    def test_a_real_row_is_mapped_positionally(self):
        assert is_positional_row(
            [
                "12/08/2021",
                "07:00 (ET)",
                "CHI@CLE",
                "Chicago Bulls",
                "Caruso, Alex",
                "Out",
                "Injury/Illness - Right Hamstring; Strain",
            ]
        )

    def test_a_new_game_row_with_a_wrapped_reason_is_not(self):
        assert not is_positional_row(
            [
                "07:30 (ET)",
                "TOR@CLE",
                "Toronto Raptors",
                "Brown, Bruce",
                "Out",
                "Return to Competition",
                "Reconditioning",
            ]
        )

    @pytest.mark.parametrize("n", [2, 3, 5, 6, 8])
    def test_only_seven_elements_can_be_positional(self, n):
        assert not is_positional_row(["x"] * n)
