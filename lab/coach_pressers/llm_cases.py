"""Twenty-three hand-labelled judge inputs for tests and the live ``llm-check``.

Each case is the exact payload ``llm_judge.build_input`` produces, built from
titles seen on team channels during research (some rearranged into hard
situations: back-to-backs, acting coaches, wrong games, combined videos). The
label is the answer a person gives from the same information.
"""

from __future__ import annotations


def video(vid, title, published, duration=600, description="", chapters=(), flags=()):
    return {
        "video_id": vid,
        "channel_priority": 1,
        "title": title,
        "published_at_utc": published,
        "published_date": published[:10],
        "duration_s": duration,
        "requires_llm": bool(flags),
        "rule_flags": list(flags),
        "description": description,
        "chapters": [{"start_s": s, "title": t} for s, t in chapters],
    }


def payload(
    team,
    opponent,
    day,
    tip,
    coach,
    staff,
    candidates,
    *,
    home=True,
    score="110-104",
    phase="regular_season",
    game_no=None,
    nearby=(),
):
    return {
        "team": team,
        "game": {
            "date": day,
            "tip_utc": tip,
            "home": home,
            "opponent": opponent,
            "score": score,
            "phase": phase,
            "playoff_game_no": game_no,
        },
        "other_games_within_3_days": [
            {"date": d, "tip_utc": t, "opponent": o, "home": h} for d, t, o, h in nearby
        ],
        "coach_expected": coach,
        "staff": [{"name": n, "role": r} for n, r in staff],
        "candidates": candidates,
    }


def case(name, data, video_id, kind):
    return {
        "name": name,
        "input": data,
        "expected": {"video_id": video_id, "kind": kind},
    }


SUNS = [("Jordan Ott", "head"), ("Royal Ivey", "assistant")]
SPURS = [("Gregg Popovich", "head"), ("Mitch Johnson", "assistant")]
RAPTORS = [("Darko Rajaković", "head")]
NUGGETS = [("Michael Malone", "head"), ("David Adelman", "assistant")]
WARRIORS = [("Steve Kerr", "head")]
CAVS = [("Kenny Atkinson", "head")]
CELTICS = [("Joe Mazzulla", "head")]
NETS = [("Jordi Fernández", "head")]

CASES = [
    case(
        "coach_vs_player",
        payload(
            "Phoenix Suns",
            "Los Angeles Lakers",
            "2026-02-26",
            "2026-02-27T03:00:00Z",
            "Jordan Ott",
            SUNS,
            [
                video(
                    "p1",
                    "Royce O'Neale Postgame Availability | Phoenix Suns | 2-26-26",
                    "2026-02-27T06:10:00Z",
                    300,
                ),
                video(
                    "c1",
                    "Head Coach Jordan Ott Postgame Availability | Phoenix Suns | 2-26-26",
                    "2026-02-27T05:55:00Z",
                    540,
                ),
            ],
        ),
        "c1",
        "coach",
    ),
    case(
        "coach_only_preferred",
        payload(
            "Toronto Raptors",
            "Denver Nuggets",
            "2024-03-11",
            "2024-03-12T02:00:00Z",
            "Darko Rajaković",
            RAPTORS,
            [
                video(
                    "m1",
                    "Toronto Raptors Media Availability | Postgame at Denver Nuggets | March 11, 2024",
                    "2024-03-12T05:20:00Z",
                    1500,
                ),
                video(
                    "c2",
                    "Raptors Post Game: Darko Rajakovic - March 11, 2024",
                    "2024-03-12T05:00:00Z",
                    480,
                ),
            ],
            home=False,
        ),
        "c2",
        "coach",
    ),
    case(
        "combined_with_chapter",
        payload(
            "Toronto Raptors",
            "Denver Nuggets",
            "2024-03-11",
            "2024-03-12T02:00:00Z",
            "Darko Rajaković",
            RAPTORS,
            [
                video(
                    "m2",
                    "Toronto Raptors Media Availability | Postgame at Denver Nuggets | March 11, 2024",
                    "2024-03-12T05:20:00Z",
                    1500,
                    "0:00 Darko Rajakovic\n8:10 Scottie Barnes\n14:30 RJ Barrett",
                    [
                        (0, "Darko Rajakovic"),
                        (490, "Scottie Barnes"),
                        (870, "RJ Barrett"),
                    ],
                ),
            ],
            home=False,
        ),
        "m2",
        "combined",
    ),
    case(
        "back_to_back_title_date",
        payload(
            "San Antonio Spurs",
            "Golden State Warriors",
            "2025-04-09",
            "2025-04-10T02:30:00Z",
            "Mitch Johnson",
            SPURS,
            [
                video(
                    "b1",
                    "Mitch Johnson Postgame vs Portland Trail Blazers | 4.8.2025",
                    "2025-04-09T04:40:00Z",
                    420,
                ),
                video(
                    "b2",
                    "Mitch Johnson Postgame vs Golden State Warriors | 4.9.2025",
                    "2025-04-10T05:15:00Z",
                    450,
                ),
            ],
            nearby=[
                ("2025-04-08", "2025-04-09T01:00:00Z", "Portland Trail Blazers", False)
            ],
        ),
        "b2",
        "coach",
    ),
    case(
        "recap_verb_coach",
        payload(
            "Golden State Warriors",
            "San Antonio Spurs",
            "2025-04-09",
            "2025-04-10T02:30:00Z",
            "Steve Kerr",
            WARRIORS,
            [
                video(
                    "s1",
                    "Stephen Curry Postgame Press Conference | April 9, 2025",
                    "2025-04-10T05:30:00Z",
                    500,
                ),
                video(
                    "k1",
                    "Steve Kerr Recaps Warriors Loss Against San Antonio Spurs | April 9, 2025",
                    "2025-04-10T05:10:00Z",
                    420,
                ),
            ],
            home=False,
            score="111-114",
        ),
        "k1",
        "coach",
    ),
    case(
        "acting_coach",
        payload(
            "San Antonio Spurs",
            "Los Angeles Clippers",
            "2024-11-04",
            "2024-11-05T01:00:00Z",
            "Gregg Popovich",
            SPURS,
            [
                video(
                    "a1",
                    "Mitch Johnson Postgame vs LA Clippers | 11.4.2024",
                    "2024-11-05T03:50:00Z",
                    400,
                    "Acting head coach Mitch Johnson speaks after the win.",
                ),
            ],
        ),
        "a1",
        "coach",
    ),
    case(
        "only_players",
        payload(
            "Charlotte Hornets",
            "Cleveland Cavaliers",
            "2026-02-20",
            "2026-02-21T00:00:00Z",
            "Charles Lee",
            [("Charles Lee", "head")],
            [
                video(
                    "pl1",
                    "Hornets vs Cavaliers: Brandon Miller Postgame Media Availability",
                    "2026-02-21T03:00:00Z",
                    360,
                ),
                video(
                    "pl2",
                    "Hornets vs Cavaliers: LaMelo Ball Postgame Media Availability",
                    "2026-02-21T03:10:00Z",
                    300,
                ),
            ],
        ),
        None,
        "none",
    ),
    case(
        "wrong_game_title_date",
        payload(
            "Cleveland Cavaliers",
            "Charlotte Hornets",
            "2026-02-22",
            "2026-02-22T23:00:00Z",
            "Kenny Atkinson",
            CAVS,
            [
                video(
                    "w1",
                    "Cavs at Hornets | Kenny Atkinson Post Game | 02.20.2026",
                    "2026-02-23T02:30:00Z",
                    500,
                    "Kenny Atkinson after Friday's game in Charlotte.",
                ),
            ],
            nearby=[("2026-02-20", "2026-02-21T00:00:00Z", "Charlotte Hornets", False)],
        ),
        None,
        "none",
    ),
    case(
        "combined_no_chapters",
        payload(
            "Detroit Pistons",
            "Atlanta Hawks",
            "2025-01-15",
            "2025-01-16T00:00:00Z",
            "J.B. Bickerstaff",
            [("J.B. Bickerstaff", "head")],
            [
                video(
                    "d1",
                    "Detroit Pistons vs. Atlanta Hawks | Full Postgame Media Availability",
                    "2025-01-16T03:30:00Z",
                    1700,
                    "Hear from Coach Bickerstaff, Cade Cunningham and Jalen Duren.",
                ),
            ],
        ),
        "d1",
        "combined",
    ),
    case(
        "not_postgame_interview",
        payload(
            "Denver Nuggets",
            "Utah Jazz",
            "2024-02-08",
            "2024-02-09T02:00:00Z",
            "Michael Malone",
            NUGGETS,
            [
                video(
                    "n1",
                    "Michael Malone on the trade deadline | Practice Availability",
                    "2024-02-09T00:30:00Z",
                    400,
                ),
                video(
                    "n2",
                    "Coach Malone Full Post Game Press Conference vs. Jazz",
                    "2024-02-09T05:05:00Z",
                    560,
                ),
            ],
        ),
        "n2",
        "coach",
    ),
    case(
        "playoff_game_number",
        payload(
            "Denver Nuggets",
            "Los Angeles Clippers",
            "2020-09-05",
            "2020-09-06T01:00:00Z",
            "Michael Malone",
            NUGGETS,
            [
                video(
                    "g1",
                    "Nuggets vs. Clippers Game 1: Michael Malone postgame interview",
                    "2020-09-04T04:00:00Z",
                    600,
                ),
                video(
                    "g2",
                    "Nuggets vs. Clippers Game 2: Michael Malone postgame interview",
                    "2020-09-06T04:10:00Z",
                    610,
                ),
            ],
            phase="playoffs",
            game_no=2,
            nearby=[
                ("2020-09-03", "2020-09-04T01:00:00Z", "Los Angeles Clippers", True)
            ],
        ),
        "g2",
        "coach",
    ),
    case(
        "full_vs_clip",
        payload(
            "Memphis Grizzlies",
            "Portland Trail Blazers",
            "2023-11-05",
            "2023-11-05T23:00:00Z",
            "Taylor Jenkins",
            [("Taylor Jenkins", "head")],
            [
                video(
                    "j1",
                    "Taylor Jenkins: 'We have to be better' #shorts",
                    "2023-11-06T02:00:00Z",
                    40,
                ),
                video(
                    "j2",
                    "Grizzlies vs Trail Blazers: Coach Taylor Jenkins press conference 11.05.23",
                    "2023-11-06T02:20:00Z",
                    520,
                ),
            ],
        ),
        "j2",
        "coach",
    ),
    case(
        "ejected_head_coach",
        payload(
            "Phoenix Suns",
            "Sacramento Kings",
            "2026-01-10",
            "2026-01-11T03:00:00Z",
            "Jordan Ott",
            SUNS,
            [
                video(
                    "e1",
                    "Royal Ivey Postgame Availability | Phoenix Suns | 1-10-26",
                    "2026-01-11T05:50:00Z",
                    420,
                    "Assistant coach Royal Ivey speaks after head coach Jordan Ott was ejected in the third quarter.",
                ),
            ],
        ),
        "e1",
        "coach",
    ),
    case(
        "postgame_live_combined",
        payload(
            "Denver Nuggets",
            "Utah Jazz",
            "2024-03-01",
            "2024-03-02T02:00:00Z",
            "Michael Malone",
            NUGGETS,
            [
                video(
                    "l1",
                    "ALL Denver Nuggets Postgame Media vs. UTA",
                    "2024-03-02T04:20:00Z",
                    2100,
                    "Michael Malone, Nikola Jokic and Jamal Murray after the win over Utah.",
                ),
            ],
        ),
        "l1",
        "combined",
    ),
    case(
        "nameless_title_description",
        payload(
            "Orlando Magic",
            "Golden State Warriors",
            "2021-01-02",
            "2021-01-03T00:00:00Z",
            "Steve Clifford",
            [("Steve Clifford", "head")],
            [
                video(
                    "o1",
                    "Postgame Press Conference vs. Golden State Warriors | Orlando Magic",
                    "2021-01-03T02:40:00Z",
                    480,
                    "Magic head coach Steve Clifford meets the media after the game.",
                ),
            ],
        ),
        "o1",
        "coach",
    ),
    case(
        "recap_not_presser",
        payload(
            "Phoenix Suns",
            "Golden State Warriors",
            "2022-01-04",
            "2022-01-05T03:00:00Z",
            "Monty Williams",
            [("Monty Williams", "head")],
            [
                video(
                    "r1",
                    "Suns vs Warriors Game Recap | Phoenix Suns",
                    "2022-01-05T05:30:00Z",
                    120,
                ),
                video(
                    "r2",
                    "Suns vs Warriors Postgame Media Availability: Williams, Paul, Ayton & Bridges | Suns",
                    "2022-01-05T06:00:00Z",
                    1600,
                ),
            ],
        ),
        "r2",
        "combined",
    ),
    case(
        "interim_after_firing",
        payload(
            "Denver Nuggets",
            "Sacramento Kings",
            "2025-04-09",
            "2025-04-10T02:00:00Z",
            "David Adelman",
            NUGGETS,
            [
                video(
                    "i1",
                    "Michael Malone Postgame vs. Sacramento Kings | 3.24.2025",
                    "2025-04-10T04:30:00Z",
                    500,
                    "Reupload of the March 24 press conference.",
                ),
                video(
                    "i2",
                    "David Adelman Postgame vs. Sacramento Kings | 4.9.2025",
                    "2025-04-10T04:40:00Z",
                    520,
                ),
            ],
        ),
        "i2",
        "coach",
    ),
    case(
        "pregame_only",
        payload(
            "San Antonio Spurs",
            "Golden State Warriors",
            "2025-04-09",
            "2025-04-10T02:30:00Z",
            "Mitch Johnson",
            SPURS,
            [
                video(
                    "pg1",
                    "Mitch Johnson Pregame vs Golden State Warriors | 4.9.2025",
                    "2025-04-10T01:10:00Z",
                    300,
                ),
            ],
        ),
        None,
        "none",
    ),
    case(
        "billups_score_title",
        payload(
            "Portland Trail Blazers",
            "Memphis Grizzlies",
            "2023-11-05",
            "2023-11-05T23:00:00Z",
            "Chauncey Billups",
            [("Chauncey Billups", "head")],
            [
                video(
                    "t1",
                    "Chauncey Billups Press Conference | Portland Trail Blazers 100, Memphis Grizzlies 112 | Nov. 5, 2023",
                    "2023-11-06T02:00:00Z",
                    450,
                ),
                video(
                    "t2",
                    "Jerami Grant Press Conference | Portland Trail Blazers 100, Memphis Grizzlies 112 | Nov. 5, 2023",
                    "2023-11-06T02:15:00Z",
                    400,
                ),
            ],
            home=False,
            score="100-112",
        ),
        "t1",
        "coach",
    ),
    case(
        "late_upload_same_game",
        payload(
            "Los Angeles Clippers",
            "Denver Nuggets",
            "2020-09-03",
            "2020-09-04T01:00:00Z",
            "Doc Rivers",
            [("Doc Rivers", "head")],
            [
                video(
                    "x1",
                    "Doc Rivers Post-Game Press Conference | Clippers vs. Nuggets Game One",
                    "2020-09-05T18:00:00Z",
                    700,
                ),
            ],
            phase="playoffs",
            game_no=1,
            nearby=[("2020-09-05", "2020-09-06T01:00:00Z", "Denver Nuggets", False)],
        ),
        "x1",
        "coach",
    ),
    # Flagged by the rules (requires_llm): warnings, not verdicts.
    case(
        "flagged_name_without_cue",
        payload(
            "Brooklyn Nets",
            "Phoenix Suns",
            "2026-01-10",
            "2026-01-11T00:30:00Z",
            "Jordi Fernández",
            NETS,
            [
                video(
                    "f1",
                    "Fernández comments on loss to Suns",
                    "2026-01-11T03:05:00Z",
                    420,
                    flags=("coach_name_no_cue",),
                ),
            ],
            score="104-117",
        ),
        "f1",
        "coach",
    ),
    case(
        "flagged_pregame_of_next_game",
        payload(
            "Boston Celtics",
            "Atlanta Hawks",
            "2026-01-14",
            "2026-01-15T00:30:00Z",
            "Joe Mazzulla",
            CELTICS,
            [
                video(
                    "g1",
                    "Joe Mazzulla Pregame Interview | Celtics vs Pistons",
                    "2026-01-16T23:10:00Z",
                    240,
                    flags=("coach_name_no_cue", "keyword:pregame"),
                ),
            ],
            nearby=[("2026-01-16", "2026-01-17T00:30:00Z", "Detroit Pistons", True)],
        ),
        None,
        "none",
    ),
    case(
        "gap_fill_player_title_with_coach_chapter",
        payload(
            "Brooklyn Nets",
            "Denver Nuggets",
            "2026-02-03",
            "2026-02-04T00:30:00Z",
            "Jordi Fernández",
            NETS,
            [
                video(
                    "h1",
                    "Nets vs Nuggets Highlights | February 3, 2026",
                    "2026-02-04T02:50:00Z",
                    600,
                    flags=("gap_fill",),
                ),
                video(
                    "h2",
                    "Cam Thomas after the win over Denver",
                    "2026-02-04T03:20:00Z",
                    1080,
                    description="Postgame media availability after Brooklyn's "
                    "win over Denver.",
                    chapters=((0, "Jordi Fernández"), (520, "Cam Thomas")),
                    flags=("gap_fill",),
                ),
            ],
            score="121-115",
        ),
        "h2",
        "combined",
    ),
]
