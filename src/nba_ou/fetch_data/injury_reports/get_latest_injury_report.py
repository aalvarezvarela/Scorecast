import os
import re
from datetime import datetime

import pandas as pd
import pymupdf  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from nba_ou.config.request_headers import HEADERS_BROWSER_LIKE
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

date_pattern = re.compile(r"^\d{2}/\d{2}/\d{4}$")  # e.g. 03/11/2025
time_pattern = re.compile(r"^\d{2}:\d{2} \(ET\)$")  # e.g. 07:00 (ET)
matchup_pattern = re.compile(r"^[A-Z]{3}@[A-Z]{3}$")  # e.g. BKN@CLE
team_pattern = re.compile(r"^[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)+$")

status_pattern = re.compile(r"^[A-Z][a-zA-Z]*$")
player_name_pattern = re.compile(r"^[A-Z][a-zA-Z'.\- ]+, [A-Z][a-zA-Z'.\-]+")

#: The values the "Current Status" column can take.
STATUS_VALUES = frozenset({"Out", "Questionable", "Probable", "Doubtful", "Available"})

#: Marker the NBA prints when a team has not filed yet. Not a player row.
NOT_YET_SUBMITTED = "NOT YET SUBMITTED"


def is_wrapped_reason_fragment(token):
    """True when ``token`` may be a reason that wrapped onto the next line.

    Reasons carry the ``Category - Detail; Detail`` shape, so a ``-`` or ``;``
    is what marks a fragment as mergeable. Player names share that punctuation
    ("Gilgeous-Alexander, Shai", "Aminu, Al-Farouq"), and merging one swallows
    the name, its status and its reason into a single cell, dropping the row.
    Statuses are excluded for the same reason.
    """
    token = token.strip()
    if player_name_pattern.match(token) or token in STATUS_VALUES:
        return False
    return ";" in token or "-" in token


def create_robust_session():
    """Create a requests session with retry strategy and browser-like headers."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Comprehensive headers to mimic a real browser
    session.headers.update(HEADERS_BROWSER_LIKE)

    return session


def get_latest_pdf(nba_injury_report_url) -> bytes:
    """Download the latest injury report PDF and return it as bytes."""
    session = create_robust_session()
    try:
        # Fetch the webpage content with improved headers and retry logic
        response = session.get(nba_injury_report_url, timeout=30, allow_redirects=True)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all PDF links (ignoring case)
        pdf_links = [
            link["href"]
            for link in soup.find_all("a", href=True)
            if link["href"].lower().endswith(".pdf")
        ]

        if not pdf_links:
            print("No PDF links found.")
            raise ValueError("No PDF links found.")

        # Filter only 'Injury-Report' links (case insensitive)
        injury_reports = [link for link in pdf_links if "injury-report" in link.lower()]

        latest_pdf_url = injury_reports[-1]
        print(f"The latest injury report link is: {latest_pdf_url}")

        # If the URL is relative, make it absolute
        if latest_pdf_url.startswith("/"):
            base_url = re.match(r"https?://[^/]+", nba_injury_report_url).group(0)
            latest_pdf_url = base_url + latest_pdf_url

        print(f"Downloading latest PDF: {latest_pdf_url}")

        # Download the PDF with improved headers and retry logic
        pdf_response = session.get(latest_pdf_url, timeout=30, allow_redirects=True)
        pdf_response.raise_for_status()

        print(f"PDF downloaded successfully ({len(pdf_response.content)} bytes)")
        return pdf_response.content

    except requests.exceptions.Timeout:
        print("Timeout error: The request took too long")
        raise
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Unable to connect to {nba_injury_report_url}")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.reason}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise
    finally:
        session.close()


def is_positional_row(elements):
    """True when a 7-element block really is one complete 7-column row.

    Length alone is not enough. A row that starts a new game carries its Game
    Time, Matchup and Team inline, so a *six*-field row whose reason wrapped onto
    a second line also arrives with seven elements -- and mapping that
    positionally shifts every column left, turning ``Game Date`` into
    ``"07:30 (ET)"`` and ``Current Status`` into a fragment of the reason.
    Verify the three leading fields before trusting the position.
    """
    if len(elements) != 7:
        return False
    return bool(
        date_pattern.match(elements[0])
        and time_pattern.match(elements[1])
        and matchup_pattern.match(elements[2])
    )


def classify_token(token, category):
    """Given a single token and the current partial row dict,
    decide which column it belongs to using regex patterns.
    The logic:
      1. If it matches date_pattern => 'Game Date'
      2. Else if time_pattern => 'Game Time'
      3. Else if matchup_pattern => 'Matchup'
      4. Else if team_pattern => 'Team'  (at least two words)
      5. Else if status_pattern => 'Current Status', IF not already filled
         (This might be too permissive, you can refine further)
      6. Else if 'Player Name' not yet assigned => 'Player Name'
      7. Else => append to 'Reason'
    """
    token = token.strip()
    if not token:
        return None

    if token == "NOT YET SUBMITTED":
        return category == "Reason"
    if category == "Player Name":
        if player_name_pattern.match(token):
            return True
    if category == "Game Date":
        if date_pattern.match(token):
            return True
    if category == "Game Time":
        if time_pattern.match(token):
            return True
    if category == "Matchup":
        if matchup_pattern.match(token):
            return True
    if category == "Team":
        if team_pattern.match(token) or token == "Philadelphia 76ers":
            if "," not in token:
                return True
    if category == "Current Status":
        if status_pattern.match(token):
            return True
    return False


def read_injury_report(pdf_data):
    """Read injury report from PDF bytes or file path."""
    # Open PDF from bytes if bytes-like, otherwise treat as file path
    if isinstance(pdf_data, bytes):
        doc = pymupdf.open(stream=pdf_data, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_data)

    col_names = [
        "Game Date",
        "Game Time",
        "Matchup",
        "Team",
        "Player Name",
        "Current Status",
        "Reason",
    ]

    dict_results_template = {col: None for col in col_names}

    data = []  # List to store extracted rows

    # ``joinwith_next`` deliberately lives outside the page loop: a reason can
    # wrap across a page break, and resetting it per page orphaned the tail.
    joinwith_next = None
    for page in doc:
        blocks = page.get_text("blocks")  # Extract text with bounding boxes
        page.get_text("html")
        # Only the first *table* block of a page can be a wrapped reason
        # continued from the previous page.
        at_page_start = True
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]  # Extract bounding box and text
            text = text.strip()
            if not text:
                continue
            text_elements = [t.strip() for t in text.split("\n")]

            # "Injury Report: ..." and "Page 3 of 10" banners carry no table
            # data. Skipped before anything else so they cannot break a join
            # that is pending across the page boundary.
            if len(text_elements) < 2:
                continue

            if text_elements[0] == "Game Date" and text_elements[1] == "Game Time":
                # skip header
                continue

            if joinwith_next:
                text_elements = joinwith_next + text_elements
                joinwith_next = None
                at_page_start = False

            # A row always ends with its Reason, so a block whose last element is
            # a bare status has had its reason pushed into the next block. This
            # happens whenever the reason is long enough to wrap: the player row
            # and the reason text end up as two separate blocks.
            if text_elements[-1] in STATUS_VALUES:
                joinwith_next = text_elements.copy()
                at_page_start = False
                continue

            # Every real row carries a status. A first-of-page block without one
            # is the tail of a reason that wrapped off the previous page -- give
            # it back to the row it belongs to rather than letting it parse as a
            # phantom player ("Knee Bone Bruise, Left Heel" matches the name
            # pattern, and "Contusion" then matches the status pattern).
            if (
                at_page_start
                and data
                and NOT_YET_SUBMITTED not in text
                and not any(element in STATUS_VALUES for element in text_elements)
            ):
                tail = " ".join(text_elements)
                previous = data[-1]
                previous["Reason"] = f"{previous['Reason'] or ''} {tail}".strip()
                at_page_start = False
                continue

            at_page_start = False

            if len(text_elements) >= 3 and is_wrapped_reason_fragment(
                text_elements[-2]
            ):
                text_elements[-2] = (
                    f"{text_elements[-2]} {text_elements[-1]}"  # Merge last two elements
                )
                text_elements.pop()

            if len(text_elements) >= 4 and is_wrapped_reason_fragment(
                text_elements[-3]
            ):
                text_elements[-3] = (
                    f"{text_elements[-3]} {text_elements[-2]} {text_elements[-1]}"  # Merge last three elements
                )
                text_elements.pop()
                text_elements.pop()

            current_line = text_elements.copy()
            results = dict_results_template.copy()
            if is_positional_row(current_line):
                results = {col: current_line[i] for i, col in enumerate(col_names)}
            else:
                last_matched_col = None
                consumed: set[int] = set()
                for index, element in enumerate(current_line):
                    for category in col_names:
                        # Skip already filled values
                        if element in results.values():
                            break

                        # Each column appears once per row, so a filled one must
                        # not be overwritten. Without this a trailing reason
                        # fragment that happens to be a single capitalised word
                        # ("Reconditioning") matches the status pattern and
                        # clobbers the real status.
                        if results[category] is not None:
                            continue

                        # Enforce logical sequence: after "Player Name", only allow "Current Status" or "Reason"
                        if last_matched_col == "Player Name" and category not in [
                            "Current Status",
                            "Reason",
                        ]:
                            continue

                        # ...and once the status is seen, everything after it is
                        # the reason. Without this a bare category ("Personal
                        # Reasons", "League Suspension") matches the two-capital-
                        # words team pattern and is filed as the Team, which
                        # ffill then propagates down the rest of the column.
                        if (
                            last_matched_col == "Current Status"
                            and category != "Reason"
                        ):
                            continue

                        if classify_token(element, category):
                            results[category] = element
                            last_matched_col = category
                            consumed.add(index)
                            break

                # The reason is whatever is left over, in order. Assigning it
                # inside the loop (as this once did) set it from the last element
                # before the status had been seen, which then made the status
                # look already-used and left ``Current Status`` empty.
                if not results["Reason"]:
                    leftover = [
                        element
                        for index, element in enumerate(current_line)
                        if index not in consumed
                    ]
                    if leftover:
                        results["Reason"] = " ".join(leftover)

            # append the current line to data
            data.append(results)

    df = pd.DataFrame(data)

    # ffil except in player name, reason, current Status
    columns_to_ffill = ["Matchup", "Team", "Game Date", "Game Time"]
    df[columns_to_ffill] = df[columns_to_ffill].ffill()
    return df


def retrieve_injury_report_as_df(nba_injury_reports_url, reports_path=None):
    """
    Retrieves the latest injury report PDF from the NBA website and extracts the data.

    Args:
        nba_injury_reports_url: URL to the NBA injury reports page
        reports_path: Optional path to save a copy of the PDF. If None, PDF is not saved.

    Returns:
        DataFrame with injury report data
    """
    # Download the PDF as bytes
    pdf_bytes = get_latest_pdf(nba_injury_reports_url)

    # Optionally save a copy to disk
    if reports_path:
        game_date = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(reports_path, exist_ok=True)
        pdf_name = f"latest_report_{game_date}.pdf"
        pdf_path = os.path.join(reports_path, pdf_name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"PDF saved to {pdf_path}")

    # Read the PDF from bytes and extract data
    df_report = read_injury_report(pdf_bytes)

    return df_report
