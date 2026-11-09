"""Step 5 of the coach presser archive: the title classifier on real titles."""

from __future__ import annotations

import pytest

from lab.coach_pressers.classify import classify, prefilter
from lab.coach_pressers.coaches import staff_entry

STAFF = {
    "CHA": ["Charles Lee"],
    "CLE": ["Kenny Atkinson", "J.B. Bickerstaff"],
    "PHX": ["Jordan Ott", "Monty Williams"],
    "SAS": ["Mitch Johnson"],
    "GSW": ["Steve Kerr"],
    "DEN": ["Michael Malone", "David Adelman"],
    "TOR": ["Darko Rajaković"],
    "POR": ["Chauncey Billups"],
    "MEM": ["Taylor Jenkins"],
    "LAC": ["Doc Rivers"],
    "ORL": ["Steve Clifford"],
    "DET": ["Dwane Casey"],
}
PLAYERS = {
    "CHA": ["Brandon Miller", "LaMelo Ball", "Coby White"],
    "PHX": [
        "Royce O'Neale",
        "Oso Ighodaro",
        "Chris Paul",
        "Deandre Ayton",
        "Mikal Bridges",
        "Devin Booker",
    ],
    "SAS": ["Keldon Johnson", "Victor Wembanyama"],
    "DEN": ["Nikola Jokić", "Jamal Murray"],
    "TOR": ["Scottie Barnes"],
}

TITLES = [
    (
        "Hornets vs Cavaliers: Coach Charles Lee Postgame Media Availability",
        "CHA",
        "coach",
    ),
    ("Cavs at Hornets | Kenny Atkinson Post Game | 02.20.2026", "CLE", "coach"),
    (
        "Head Coach Jordan Ott Postgame Availability | Phoenix Suns | 2-26-26",
        "PHX",
        "coach",
    ),
    ("Mitch Johnson Postgame vs Golden State Warriors | 4.9.2025", "SAS", "coach"),
    (
        "Steve Kerr Recaps Warriors Loss Against San Antonio Spurs | April 9, 2025",
        "GSW",
        "coach",
    ),
    ("Coach Malone Full Post Game Press Conference vs. Raptors", "DEN", "coach"),
    ("Raptors Post Game: Darko Rajakovic - March 11, 2024", "TOR", "coach"),
    (
        "Chauncey Billups Press Conference | Portland Trail Blazers 100, Memphis "
        "Grizzlies 112 | Nov. 5, 2023",
        "POR",
        "coach",
    ),
    (
        "Grizzlies vs Trail Blazers: Coach Taylor Jenkins press conference 11.05.23",
        "MEM",
        "coach",
    ),
    (
        "Doc Rivers Post-Game Press Conference | Clippers vs. Nuggets Game One",
        "LAC",
        "coach",
    ),
    ("Nuggets vs. Clippers Game 1: Michael Malone postgame interview", "DEN", "coach"),
    (
        "Steve Clifford Postgame Sound vs. Golden State Warriors | Orlando Magic",
        "ORL",
        "coach",
    ),
    ("Cavs vs. Bulls Postgame: J.B. Bickerstaff", "CLE", "coach"),
    (
        "Suns vs Warriors Postgame Media Availability: Williams, Paul, Ayton & Bridges "
        "| Suns",
        "PHX",
        "combined",
    ),
    (
        "Detroit Pistons vs. Atlanta Hawks | Full Postgame Media Availability",
        "DET",
        "combined",
    ),
    (
        "Toronto Raptors Media Availability | Postgame at Denver Nuggets | March 11, 2024",
        "TOR",
        "combined",
    ),
    ("ALL Denver Nuggets Postgame Media vs. UTA", "DEN", "combined"),
    (
        "Hornets vs Cavaliers: Brandon Miller Postgame Media Availability",
        "CHA",
        "player",
    ),
    ("Royce O'Neale Postgame Availability | Phoenix Suns | 2-26-26", "PHX", "player"),
    ("Mitch Johnson Pregame vs Golden State Warriors | 4.9.2025", "SAS", "other"),
    (
        "Toronto Raptors Media Availability | Pregame at Denver Nuggets | March 11, 2024",
        "TOR",
        "other",
    ),
    (
        "Oso Ighodaro Shootaround Media Availability | Phoenix Suns | 2-26-26",
        "PHX",
        "other",
    ),
    ("Coby White Practice Media Availability", "CHA", "other"),
    ("CAVALIERS at HORNETS | FULL GAME HIGHLIGHTS | February 20, 2026", "CLE", "other"),
]


def staff(team):
    return [staff_entry(name, "head", "test") for name in STAFF.get(team, [])]


@pytest.mark.parametrize("title,team,expected", TITLES)
def test_fixture_titles(title, team, expected):
    verdict = classify(title, staff(team), PLAYERS.get(team, []))
    assert verdict.label == expected, verdict
    if expected == "coach":
        assert verdict.coach_guess in STAFF[team]
        threshold = 0.7 if "Recaps" in title else 0.8
        assert verdict.score >= threshold
    if "Recaps" in title:
        assert verdict.score == 0.7


def test_shared_last_name_needs_the_full_name():
    sas = staff("SAS")
    # Mitch or Keldon Johnson: never a confident coach match, left to the judge.
    ambiguous = classify("Johnson Postgame vs Warriors", sas, PLAYERS["SAS"])
    assert ambiguous.label != "coach" and ambiguous.score <= 0.6
    assert classify("Coach Johnson Postgame", sas, PLAYERS["SAS"]).label == "coach"


def test_unknown_name_with_a_cue_reaches_the_judge():
    verdict = classify(
        "Hornets vs Cavaliers: Jordan Surname Postgame Media Availability",
        staff("CHA"),
        PLAYERS["CHA"],
    )
    assert verdict.label == "player" and verdict.score <= 0.5
    assert "unknown_name" in verdict.notes
    assert prefilter(
        "Hornets vs Cavaliers: Jordan Surname Postgame Media Availability",
        staff("CHA"),
        PLAYERS["CHA"],
    )


def test_assistant_at_the_podium_is_a_coach_with_a_note():
    people = [
        staff_entry("Michael Malone", "head", "test"),
        staff_entry("David Adelman", "assistant", "test"),
    ]
    verdict = classify("David Adelman Postgame vs. Jazz", people, PLAYERS["DEN"])
    assert verdict.label == "coach" and "assistant_or_acting" in verdict.notes


def test_duration_guard():
    sac = [staff_entry("Mike Brown", "head", "test")]
    assert classify("Mike Brown Postgame", sac, [], duration=30).label == "other"
    long = classify("Mike Brown Postgame", sac, [], duration=40 * 60)
    assert long.label == "coach" and "long" in long.notes and long.score < 0.8


def test_prefilter_skips_players_and_other():
    assert not prefilter(
        "Royce O'Neale Postgame Availability", staff("PHX"), PLAYERS["PHX"]
    )
    assert not prefilter("Suns Top Plays of the Week", staff("PHX"), PLAYERS["PHX"])
    assert prefilter("Jordan Ott Postgame Availability", staff("PHX"), PLAYERS["PHX"])


def test_media_phrases_are_cues():
    uta = [staff_entry("Will Hardy", "head", "test")]
    verdict = classify(
        '"We\'ll get back to work tomorrow" | Coach Hardy speaks to the media after '
        "Memphis",
        uta,
        [],
    )
    assert verdict.label == "coach" and verdict.score >= 0.8


@pytest.mark.parametrize(
    "title,team",
    [
        ("Steve Kerr on Warriors Loss to Cavaliers | Apr. 2, 2026", "GSW"),
        ("Mitch Johnson On Spurs Game 1 Performance in the NBA Finals", "SAS"),
    ],
)
def test_coach_name_with_game_words_is_a_weak_candidate(title, team):
    verdict = classify(title, staff(team), PLAYERS.get(team, []))
    assert verdict.label == "coach" and verdict.score == 0.6
    assert "game_words_no_cue" in verdict.notes
    assert prefilter(title, staff(team), PLAYERS.get(team, []))


@pytest.mark.parametrize(
    "title,team",
    [
        ("Steve Kerr Previews Warriors 2026-27 Season | Media Day", "GSW"),
        ("Steve Kerr on Tragic Shooting at Brown University", "GSW"),
        (
            "Mitch Johnson on Spurs Playing to Their Strengths and Their Competitive "
            "Response Ahead of Game 2",
            "SAS",
        ),
        ("Chauncey Billups On Tonight's Playoff Game vs San Antonio Spurs", "POR"),
        ("Steve Kerr | Warriors 2026 End of Season Interview", "GSW"),
    ],
)
def test_pregame_and_offseason_titles_stay_out(title, team):
    assert classify(title, staff(team), PLAYERS.get(team, [])).label == "other"


def test_outlet_titles_must_name_the_team():
    dal = [staff_entry("Jason Kidd", "head", "test")]
    other = "Jamahl Mosley Head Coach of Magic | Press Conference | 12 12 2025"
    ours = "Jason Kidd Head Coach of Mavericks | Press Conference | 12 12 2025"
    nameless = "Postgame Interview | Mavericks vs Lakers"
    assert prefilter(other, dal, [])  # the team's own channel: the judge decides
    assert not prefilter(other, dal, [], team="DAL")
    assert prefilter(ours, dal, [], team="DAL")
    assert prefilter(nameless, dal, [], team="DAL")
