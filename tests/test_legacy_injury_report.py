"""Contract for the legacy (2018-12-17 -> 2019-12-17) injury-report reader.

The fixtures are generated PDFs laid out like the real files: a portrait page
with ``/Rotate 90``, text drawn rotated, the table header repeated per page, and
a wrapped reason split over lines 4 pt apart with the rest of its row centred
between them. Measured on all 788 archived legacy reports, that geometry is what
the four legacy layouts share, and it is what the modern pattern reader got
wrong (a wrong status on ~48% of rows).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pymupdf
import pytest
from nba_ou.fetch_data.injury_reports.legacy_injury_report import (
    OUTPUT_COLUMNS,
    LegacyHeaderNotFound,
    read_legacy_injury_report,
)
from nba_ou.postgre_db.injury_report_aiven.parse import (
    LegacyLayoutError,
    is_legacy_layout,
    parse_report,
)

LAYOUT_A = [
    ("Game Date", 21),
    ("Game Time", 74),
    ("Matchup", 126),
    ("Team", 179),
    ("Player Name", 269),
    ("Category", 382),
    ("Reason", 495),
    ("Current Status", 607),
    ("Previous Status", 720),
]
LAYOUT_C = [
    ("Game Date", 21),
    ("Game Time", 82),
    ("Matchup", 143),
    ("Team", 204),
    ("Player Name", 309),
    ("Current Status", 440),
    ("Reason", 520),
    ("Previous Status", 702),
]


def _pdf(pages: list[tuple[list, list[tuple[float, dict]]]]) -> bytes:
    """``pages``: ``(layout, [(row_y, {column: text})...])`` per page."""
    doc = pymupdf.open()
    for layout, rows in pages:
        page = doc.new_page(width=595, height=842)
        page.set_rotation(90)
        to_page = page.derotation_matrix

        def put(x, y, text, page=page, to_page=to_page):
            page.insert_text(pymupdf.Point(x, y) * to_page, text, fontsize=7, rotate=90)

        put(300, 20, "Injury Report: 01/16/19 05:30 PM")
        for label, x in layout:
            put(x, 66, label)
        left = dict(layout)
        for y, cells in rows:
            for column, text in cells.items():
                put(left[column] - 1, y, text)
        put(400, 580, "Page 1 of 1")
    data = doc.tobytes()
    doc.close()
    return data


def _layout_a_report() -> bytes:
    first = {"Game Date": "01/16/2019", "Game Time": "07:00 (ET)", "Matchup": "ORL@DET"}
    return _pdf(
        [
            (
                LAYOUT_A,
                [
                    (
                        85,
                        {
                            **first,
                            "Team": "Detroit Pistons",
                            "Player Name": "Ellenson, Henry",
                            "Category": "Injury/Illness",
                            "Reason": "Left Ankle Sprain",
                            "Current Status": "Out",
                            "Previous Status": "Questionable",
                        },
                    ),
                    # wrapped reason: first line above, name/status centred, second line below
                    (99, {"Reason": "Right Lower Leg"}),
                    (
                        103,
                        {
                            "Player Name": "Pachulia, Zaza",
                            "Category": "Injury/Illness",
                            "Current Status": "Questionable",
                            "Previous Status": "-",
                        },
                    ),
                    (107, {"Reason": "Contusion"}),
                    (
                        121,
                        {
                            "Team": "Orlando Magic",
                            "Player Name": "Caupain, Troy",
                            "Category": "G League Team",
                            "Reason": "-",
                            "Current Status": "Out",
                            "Previous Status": "-",
                        },
                    ),
                    (
                        139,
                        {
                            "Game Time": "08:00 (ET)",
                            "Matchup": "BKN@HOU",
                            "Team": "Brooklyn Nets",
                            "Reason": "NOT YET SUBMITTED",
                        },
                    ),
                    (
                        157,
                        {
                            "Team": "Houston Rockets",
                            "Category": "ALL PLAYERS AVAILABLE",
                        },
                    ),
                ],
            ),
            # second page: the header repeats, merged cells carry over
            (
                LAYOUT_A,
                [
                    (
                        85,
                        {
                            "Player Name": "Capela, Clint",
                            "Category": "Injury/Illness",
                            "Reason": "Right Thumb",
                            "Current Status": "Doubtful",
                            "Previous Status": "-",
                        },
                    )
                ],
            ),
        ]
    )


def test_layout_a_rows_statuses_and_category_fold():
    frame = read_legacy_injury_report(_layout_a_report())
    assert list(frame.columns) == OUTPUT_COLUMNS
    players = frame[frame["Player Name"].notna()].set_index("Player Name")
    assert (
        players.loc["Ellenson, Henry", "Current Status"] == "Out"
    )  # not the previous status
    assert (
        players.loc["Ellenson, Henry", "Reason"] == "Injury/Illness - Left Ankle Sprain"
    )
    assert players.loc["Caupain, Troy", "Reason"] == "G League"


def test_a_wrapped_reason_stays_one_row():
    frame = read_legacy_injury_report(_layout_a_report())
    pachulia = frame[frame["Player Name"] == "Pachulia, Zaza"]
    assert len(pachulia) == 1
    row = pachulia.iloc[0]
    assert row["Current Status"] == "Questionable"
    assert row["Reason"] == "Injury/Illness - Right Lower Leg Contusion"
    assert row["Team"] == "Detroit Pistons"


def test_merged_cells_forward_fill_across_rows_and_pages():
    frame = read_legacy_injury_report(_layout_a_report())
    capela = frame[frame["Player Name"] == "Capela, Clint"].iloc[0]
    assert (capela["Game Date"], capela["Matchup"], capela["Team"]) == (
        "01/16/2019",
        "BKN@HOU",
        "Houston Rockets",
    )
    assert capela["Current Status"] == "Doubtful"


def test_unfiled_and_empty_team_rows_carry_no_player():
    frame = read_legacy_injury_report(_layout_a_report())
    nys = frame[frame["Reason"] == "NOT YET SUBMITTED"]
    assert list(nys["Team"]) == ["Brooklyn Nets"]
    assert nys["Player Name"].isna().all()
    empty = frame[frame["Team"] == "Houston Rockets"].iloc[0]
    assert empty["Player Name"] is None and empty["Current Status"] is None


def test_layout_with_status_before_reason():
    data = _pdf(
        [
            (
                LAYOUT_C,
                [
                    (
                        85,
                        {
                            "Game Date": "11/20/2019",
                            "Game Time": "07:00 (ET)",
                            "Matchup": "LAL@IND",
                            "Team": "Indiana Pacers",
                            "Player Name": "Oladipo, Victor",
                            "Current Status": "Out",
                            "Reason": "Injury/Illness - Right Knee; Rehab",
                            "Previous Status": "Available",
                        },
                    )
                ],
            )
        ]
    )
    row = read_legacy_injury_report(data).iloc[0]
    assert (row["Current Status"], row["Reason"]) == (
        "Out",
        "Injury/Illness - Right Knee; Rehab",
    )


def test_placeholder_names_are_not_players():
    data = _pdf(
        [
            (
                LAYOUT_C,
                [
                    (
                        85,
                        {
                            "Game Date": "02/15/2019",
                            "Game Time": "07:00 (ET)",
                            "Matchup": "UNK@UNK",
                            "Team": "Non-NBA Team",
                            "Player Name": ",",
                        },
                    )
                ],
            )
        ]
    )
    assert read_legacy_injury_report(data)["Player Name"].isna().all()


def test_parse_report_routes_legacy_layout_through_the_legacy_reader():
    data = _layout_a_report()
    assert is_legacy_layout(data)
    report = parse_report(data, datetime(2019, 1, 16, 22, 30, tzinfo=UTC))
    players = report.players.set_index("raw_name")
    assert players.loc["Ellenson, Henry", "status_id"] == 5
    assert players.loc["Pachulia, Zaza", "status_id"] == 3
    assert players.loc["Ellenson, Henry", "reason_category"] == "Injury/Illness"
    assert players.loc["Caupain, Troy", "reason_category"] == "G League"
    assert set(report.not_submitted["raw_team"]) == {"Brooklyn Nets"}
    assert {"DET", "HOU"} <= set(report.games["team_home"])


def test_legacy_file_without_a_header_is_reported_not_misparsed():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Previous Status appears but there is no table here")
    data = doc.tobytes()
    doc.close()
    with pytest.raises(LegacyHeaderNotFound):
        read_legacy_injury_report(data)
    with pytest.raises(LegacyLayoutError):
        parse_report(data, datetime(2019, 1, 16, tzinfo=UTC))
