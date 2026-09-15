"""Reader for the legacy NBA injury-report layouts (2018-12-17 -> 2019-12-17).

The modern reader (``get_latest_injury_report.read_injury_report``) classifies
tokens by pattern and assumes the 7-column layout. Before 2019-12-18 the NBA
published four other layouts, measured over all 788 archived reports:

====  ===========================================================  ======  =========================
code  columns after ``Player Name``                                reports  dates
====  ===========================================================  ======  =========================
A     Category | Reason | Current Status | Previous Status         685     2018-12-17 -> 2019-11-14
B     Reason | Current Status | Previous Status                    16      2019-11-13, 11-15 -> 11-19
C     Current Status | Reason | Previous Status                    80      2019-11-20 -> 12-16
D     Current Status | Reason | Previous Status | Previous Reason  4       2019-12-16 -> 12-17
====  ===========================================================  ======  =========================

Pattern classification cannot tell a ``Category`` from a ``Reason`` or a current
status from a previous one, which is why the modern reader mis-assigned the
status on ~48% of legacy rows. This reader uses **geometry** instead:

* Pages are rotated 90 degrees. Words are mapped through the page rotation
  matrix so x runs left to right and y top to bottom.
* Every page repeats the header. Its labels give each column's left edge, and a
  word belongs to the last column starting at or before it.
* A wrapped cell spreads one row over lines ~4 pt apart (its other cells are
  vertically centred between them), while rows are ~18 pt apart. Lines closer
  than :data:`ROW_GAP_PT` are therefore one row.

The output has exactly the modern reader's columns, so everything downstream is
shared. Layout A's separate ``Category`` is folded back into ``Reason`` as
``"Category - Reason"``, the shape the modern layout prints. ``Previous Status``
and ``Previous Reason`` describe the prior report and are dropped.
"""

from __future__ import annotations

import re

import pandas as pd
import pymupdf

#: Columns of the modern reader, which this reader reproduces.
OUTPUT_COLUMNS = [
    "Game Date",
    "Game Time",
    "Matchup",
    "Team",
    "Player Name",
    "Current Status",
    "Reason",
]

#: Every header label seen in the legacy layouts, longest first so "Previous
#: Status" is not read as "Status".
HEADER_LABELS: tuple[str, ...] = (
    "Previous Reason",
    "Previous Status",
    "Current Status",
    "Player Name",
    "Game Date",
    "Game Time",
    "Matchup",
    "Category",
    "Reason",
    "Team",
)

#: Lines closer than this (points) belong to one table row.
ROW_GAP_PT = 10.0
#: Data words start about 1 pt left of their header label.
COLUMN_SLACK_PT = 3.0

#: Legacy category labels that name the same thing as a modern category.
CATEGORY_ALIASES = {"G League Team": "G League"}

NOT_YET_SUBMITTED = "NOT YET SUBMITTED"
_PAGE_FOOTER = re.compile(r"^Page \d+ of \d+$")
_FORWARD_FILLED = ["Game Date", "Game Time", "Matchup", "Team"]


class LegacyHeaderNotFound(ValueError):
    """The report has no readable header (e.g. a scanned image)."""


def _page_words(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """``(x, y, text)`` per word in reading orientation."""
    matrix = page.rotation_matrix
    out = []
    for word in page.get_text("words"):
        rect = pymupdf.Rect(word[:4]) * matrix
        out.append((rect.x0, rect.y0, word[4]))
    return out


def _lines(words: list[tuple[float, float, str]]) -> list[tuple[float, list]]:
    """Group words sharing a baseline, top to bottom."""
    by_y: dict[float, list] = {}
    for x, y, text in words:
        key = next((k for k in by_y if abs(k - y) < 1.5), y)
        by_y.setdefault(key, []).append((x, text))
    return [(y, sorted(by_y[y])) for y in sorted(by_y)]


def _header_columns(line: list[tuple[float, str]]) -> list[tuple[float, str]] | None:
    """Column ``(left_edge, label)`` pairs if ``line`` is the table header."""
    tokens = [text for _, text in line]
    columns = []
    i = 0
    while i < len(tokens):
        for label in HEADER_LABELS:
            parts = label.split()
            if tokens[i : i + len(parts)] == parts:
                columns.append((line[i][0] - COLUMN_SLACK_PT, label))
                i += len(parts)
                break
        else:
            return None
    labels = [label for _, label in columns]
    if labels[:2] != ["Game Date", "Game Time"] or "Player Name" not in labels:
        return None
    return columns


def _assign(columns: list[tuple[float, str]], x: float) -> str | None:
    label = None
    for left, name in columns:
        if x >= left:
            label = name
        else:
            break
    return label


def read_legacy_injury_report(pdf_data: bytes | str) -> pd.DataFrame:
    """Parse a legacy-layout report into the modern reader's columns."""
    doc = (
        pymupdf.open(stream=pdf_data, filetype="pdf")
        if isinstance(pdf_data, bytes)
        else pymupdf.open(pdf_data)
    )
    rows: list[dict[str, str]] = []
    header_seen = False
    try:
        for page in doc:
            lines = _lines(_page_words(page))
            columns = None
            header_y = None
            for y, line in lines:
                columns = _header_columns(line)
                if columns:
                    header_y = y
                    break
            if columns is None or header_y is None:
                continue
            header_seen = True

            band: list[tuple[float, list]] = []
            bands: list[list[tuple[float, list]]] = []
            for y, line in lines:
                if y <= header_y:
                    continue
                if _PAGE_FOOTER.match(" ".join(t for _, t in line)):
                    continue
                if band and y - band[-1][0] >= ROW_GAP_PT:
                    bands.append(band)
                    band = []
                band.append((y, line))
            if band:
                bands.append(band)

            for band in bands:
                cells: dict[str, list[str]] = {}
                for _, line in band:
                    for x, text in line:
                        label = _assign(columns, x)
                        if label is not None:
                            cells.setdefault(label, []).append(text)
                rows.append({label: " ".join(parts) for label, parts in cells.items()})
    finally:
        doc.close()

    if not header_seen:
        raise LegacyHeaderNotFound("no legacy table header found in the report")
    return _to_modern_frame(rows)


def _clean(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text in ("", "-") else text


def _to_modern_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in [*OUTPUT_COLUMNS, "Category"]:
        if column not in frame.columns:
            frame[column] = None

    category = (
        frame["Category"]
        .map(_clean)
        .map(lambda c: CATEGORY_ALIASES.get(c, c) if c else c)
    )
    detail = frame["Reason"].map(_clean)
    reason = [
        f"{c} - {d}" if c and d else (c or d)
        for c, d in zip(category, detail, strict=True)
    ]
    frame["Reason"] = reason
    # NOT YET SUBMITTED can land in Category or Reason depending on layout.
    not_submitted = frame.apply(
        lambda r: any(NOT_YET_SUBMITTED in str(v) for v in r.values), axis=1
    )
    frame.loc[not_submitted, "Reason"] = NOT_YET_SUBMITTED

    frame[_FORWARD_FILLED] = frame[_FORWARD_FILLED].map(_clean).ffill()
    # All-Star weekend lists "Non-NBA Team" placeholders whose name cell is a
    # bare "," -- not a player.
    frame["Player Name"] = (
        frame["Player Name"]
        .map(_clean)
        .map(lambda n: n if n and re.search(r"[A-Za-z]", n) else None)
    )
    frame["Current Status"] = frame["Current Status"].map(_clean)
    return frame[OUTPUT_COLUMNS].reset_index(drop=True)
