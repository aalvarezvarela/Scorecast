import re
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString
from nba_ou.config.request_headers import HEADERS_BROWSER_LIKE

ASSIGNMENT_COLUMNS = ["Game", "Crew Chief", "Referee", "Umpire", "Alternate"]


def fetch_nba_referee_assignments_today(
    nba_official_url: str, *, expected_assignment_date: str | None = None
) -> pd.DataFrame:
    """
    Returns the NBA referee assignments table for the current day as posted on official.nba.com.
    Note: the site posts daily assignments (typically same-day only).
    An empty NBA section/table means assignments are not published yet and
    returns an empty frame with the usual columns. Fetch/layout errors raise.
    """
    resp = requests.get(nba_official_url, headers=HEADERS_BROWSER_LIKE, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    headings = ["h1", "h2", "h3", "h4", "h5", "h6"]
    nba_heading = soup.find(
        lambda tag: (
            tag.name in headings
            and tag.get_text(" ", strip=True).casefold() == "nba referee assignments"
        )
    )
    assignment_date = None
    if nba_heading is not None:
        # WNBA uses identical table columns. Never use its crews when the NBA
        # section has no assignments yet.
        table_tags = []
        section_text = []
        for tag in nba_heading.next_elements:
            if isinstance(tag, NavigableString):
                section_text.append(str(tag))
                continue
            if (
                tag.name in headings
                and "referee assignments" in tag.get_text(" ", strip=True).casefold()
            ):
                break
            if tag.name == "table":
                table_tags.append(tag)
        date_match = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            " ".join(section_text),
        )
        if date_match:
            assignment_date = (
                pd.to_datetime(date_match.group(), format="%B %d, %Y")
                .date()
                .isoformat()
            )
        if expected_assignment_date and assignment_date not in (
            None,
            expected_assignment_date,
        ):
            raise ValueError(
                f"NBA assignments are dated {assignment_date}, expected {expected_assignment_date}"
            )
        if not table_tags:
            return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
    else:
        table_tags = soup.find_all("table")

    # Pick the NBA table: it has Crew Chief / Referee / Umpire columns.
    nba_tbl = None
    for table in table_tags:
        t = pd.read_html(StringIO(str(table)))[0]
        cols = set(map(str, t.columns))
        if {"Game", "Crew Chief", "Referee", "Umpire"}.issubset(cols):
            nba_tbl = t.copy()
            break

    if nba_tbl is None:
        raise RuntimeError(
            "Could not locate NBA referee assignments table on the page."
        )

    # Normalize columns (Alternate can be missing sometimes)
    if "Alternate" not in nba_tbl.columns:
        nba_tbl["Alternate"] = pd.NA

    if expected_assignment_date and not nba_tbl.empty and assignment_date is None:
        raise ValueError("Could not verify the NBA referee assignment date")
    nba_tbl.attrs["assignment_date"] = assignment_date
    return nba_tbl


if __name__ == "__main__":
    url = "https://official.nba.com/referee-assignments/"
    try:
        df_refs = fetch_nba_referee_assignments_today(url)
        print(df_refs)
    except Exception as e:
        print(f"Error fetching referee assignments: {e}")
