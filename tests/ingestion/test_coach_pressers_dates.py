"""Dates written in presser titles (lab/coach_pressers/title_dates.py)."""

from __future__ import annotations

from datetime import date

import pytest

from lab.coach_pressers.title_dates import title_dates


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Cavs at Hornets | Kenny Atkinson Post Game | 02.20.2026", date(2026, 2, 20)),
        (
            "Head Coach Jordan Ott Postgame Availability | Phoenix Suns | 2-26-26",
            date(2026, 2, 26),
        ),
        (
            "Mitch Johnson Postgame vs Golden State Warriors | 4.9.2025",
            date(2025, 4, 9),
        ),
        (
            "Steve Kerr Recaps Warriors Loss Against San Antonio Spurs | April 9, 2025",
            date(2025, 4, 9),
        ),
        ("Raptors Post Game: Darko Rajakovic - March 11, 2024", date(2024, 3, 11)),
        (
            "Chauncey Billups Press Conference | Portland Trail Blazers 100, Memphis "
            "Grizzlies 112 | Nov. 5, 2023",
            date(2023, 11, 5),
        ),
        ("Coach Taylor Jenkins press conference 11.05.23", date(2023, 11, 5)),
        (
            "CAVALIERS at HORNETS | FULL GAME HIGHLIGHTS | February 20, 2026",
            date(2026, 2, 20),
        ),
        ("Postgame | Sept. 30, 2020", date(2020, 9, 30)),
        ("Postgame 12/25/2023", date(2023, 12, 25)),
    ],
)
def test_title_dates(title, expected):
    assert title_dates(title) == [expected]


@pytest.mark.parametrize(
    "title",
    [
        "Doc Rivers Post-Game Press Conference | Clippers vs. Nuggets Game One",
        "Nuggets vs. Clippers Game 1: Michael Malone postgame interview",
        "Blazers 100, Grizzlies 112",
        "Version 1.2.3 notes",
        "Postgame 2-30-24",
    ],
)
def test_no_date(title):
    assert title_dates(title) == []
