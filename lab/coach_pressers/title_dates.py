"""Dates written in video titles, parsed with explicit patterns.

Numeric dates are US order (month first); two-digit years are 20xx. Fuzzy
parsers are avoided on purpose: they read "Game 1" or a score as a date.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
NUMERIC = re.compile(r"(?<![\d.])(\d{1,2})([./-])(\d{1,2})\2(\d{4}|\d{2})(?![\d.])")
TEXTUAL = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)


def _make(year: int, month: int, day: int) -> date | None:
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def title_dates(title: str) -> list[date]:
    """Every valid date written in a title, in order of appearance."""
    found: list[tuple[int, date]] = []
    for match in NUMERIC.finditer(title or ""):
        month, _, day, year = match.groups()
        value = _make(int(year), int(month), int(day))
        if value:
            found.append((match.start(), value))
    for match in TEXTUAL.finditer(title or ""):
        month, day, year = match.groups()
        value = _make(int(year), MONTHS[month.lower()], int(day))
        if value:
            found.append((match.start(), value))
    result = []
    for _, value in sorted(found, key=lambda pair: pair[0]):
        if value not in result:
            result.append(value)
    return result
