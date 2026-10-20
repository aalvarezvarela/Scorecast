"""Phase A: injury-report PDF bytes -> tidy rows.

Thin on purpose. The table extraction itself lives in
``fetch_data.injury_reports.get_latest_injury_report.read_injury_report``; this
module only splits the reason into ``(category, detail)``, maps status labels to
ids, and decides what to skip.

Two layouts, one output:

* **Legacy layouts** (2018-12-17 -> 2019-12-17) carry ``Previous Status`` and,
  depending on the month, a ``Category`` column or status/reason in the other
  order. The modern pattern reader puts a wrong status on ~48% of those rows, so
  they go through ``legacy_injury_report.read_legacy_injury_report``, which reads
  columns by position under each page's header. Routing keys on the page text
  containing ``Previous Status``, never on the cutover date, so a stray
  legacy-format file can never reach the modern reader. A legacy file with no
  readable header (a scanned image) raises :class:`LegacyLayoutError`.
* **``NOT YET SUBMITTED`` rows** carry no player and mean "this team has not
  filed yet", which is *unknown*, not "nobody is injured". They are routed to
  the filing table instead of the status table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pymupdf

from .schema import STATUS_BY_LABEL

NOT_YET_SUBMITTED = "NOT YET SUBMITTED"
LEGACY_MARKER = "Previous Status"

_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_MATCHUP_RE = re.compile(r"^([A-Z]{3})@([A-Z]{3})$")
_GAME_TIME_RE = re.compile(r"^(\d{2}):(\d{2}) \(ET\)$")
_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

GAME_COLUMNS = ["game_date", "team_away", "team_home", "game_time_raw"]


class LegacyLayoutError(RuntimeError):
    """Raised for a legacy-layout report that has no readable table header."""


@dataclass(frozen=True)
class ParsedReport:
    """One report's contents, before any id resolution."""

    observed_at: datetime
    #: One row per listed player.
    players: pd.DataFrame
    #: One row per (game, team) that has not filed yet.
    not_submitted: pd.DataFrame
    #: One row per game the report lists at all. A team of a listed game that
    #: is neither NOT YET SUBMITTED nor carrying players has filed an empty
    #: list, and a player who drops out of a filed team's list has been removed
    #: -- both are only knowable against the set of games actually listed.
    games: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=GAME_COLUMNS)
    )

    @property
    def n_rows(self) -> int:
        return len(self.players)


def split_reason(raw: object) -> tuple[str, str]:
    """``"Injury/Illness - Right Knee; Soreness"`` -> ``("Injury/Illness", ...)``.

    Splits on the **first** ``" - "`` only. Splitting on the last would turn a
    wrapped detail into a bogus category -- which is how
    ``"Not with Team - Return to Competition Reconditioning"`` comes to look like
    a category called ``Return to Competition Reconditioning``.
    """
    text = "" if raw is None else str(raw).strip()
    if text in ("", "-", "nan", "None"):
        return ("(none)", "")
    if " - " not in text:
        return (text, "")
    category, detail = text.split(" - ", 1)
    return (category.strip(), detail.strip())


def decode_game_time(game_date: object, raw: object) -> pd.Timestamp | None:
    """``(2026-05-21, "08:00 (ET)")`` -> ``2026-05-22 00:00 UTC``.

    The report prints a 12-hour clock with **no AM/PM**. It is decodable only
    because NBA games tip between 11:00 and 22:30 ET: hour 11 is morning, 12 is
    noon, and 1-10 are afternoon or evening.

    Used only as a fallback for games the schedule feed does not carry (the
    2025-26 postseason). Validated against the schedule on 2024-25: 1,317 of
    1,320 games agree exactly, including all 83 playoff and 6 play-in games; the
    three misses were rescheduled starts, which is why callers must take the
    time from the *latest* report listing the game.
    """
    match = _GAME_TIME_RE.match(str(raw or "").strip())
    if not match or game_date is None or pd.isna(game_date):
        return None
    hour12, minute = int(match.group(1)), int(match.group(2))
    hour = hour12 if hour12 in (11, 12) else hour12 + 12
    day = pd.Timestamp(game_date)
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=_ET, fold=0)
    return pd.Timestamp(local.astimezone(_UTC))


def category_code(label: str) -> str:
    """Case-fold a category into a stable key.

    The NBA prints both ``Not With Team`` and ``Not with Team``; they are the
    same category and must not become two dimension rows.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_") or "none"


def is_legacy_layout(data: bytes) -> bool:
    """True for the 9-column era, detected from the page text itself."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return any(LEGACY_MARKER in page.get_text() for page in doc)


def parse_report(data: bytes, observed_at: datetime) -> ParsedReport:
    """Parse one report's bytes into player rows and not-filed rows.

    ``observed_at`` is the report's UTC instant from the archive manifest, not
    anything read out of the PDF. It is the only ordering key the rest of the
    pipeline uses.
    """
    if is_legacy_layout(data):
        from nba_ou.fetch_data.injury_reports.legacy_injury_report import (
            LegacyHeaderNotFound,
            read_legacy_injury_report,
        )

        try:
            frame = read_legacy_injury_report(data)
        except LegacyHeaderNotFound as exc:
            raise LegacyLayoutError(
                f"legacy report at {observed_at:%Y-%m-%d %H:%M%z} has no readable "
                "table header"
            ) from exc
    else:
        # Local import: the parser module pulls in requests/bs4, which the DB
        # layer has no business requiring at import time.
        from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
            read_injury_report,
        )

        frame = read_injury_report(data)
    if frame.empty:
        empty = pd.DataFrame()
        return ParsedReport(observed_at, empty, empty)

    frame = frame.copy()
    frame["reason_raw"] = frame["Reason"].fillna("")
    matchups = frame["Matchup"].fillna("").astype(str).str.extract(_MATCHUP_RE)
    frame["team_away"] = matchups[0]
    frame["team_home"] = matchups[1]
    frame["game_date"] = pd.to_datetime(
        frame["Game Date"].where(frame["Game Date"].astype(str).str.match(_DATE_RE)),
        format="%m/%d/%Y",
        errors="coerce",
    ).dt.date

    not_filed = frame["reason_raw"].astype(str).str.strip() == NOT_YET_SUBMITTED
    listed = frame["Player Name"].notna() & ~not_filed

    players = frame.loc[listed].copy()
    reasons = players["reason_raw"].map(split_reason)
    players["reason_category"] = [c for c, _ in reasons]
    players["reason_detail"] = [d for _, d in reasons]
    players["status_id"] = players["Current Status"].map(STATUS_BY_LABEL)
    players = players.rename(columns={"Player Name": "raw_name", "Team": "raw_team"})[
        [
            "game_date",
            "team_home",
            "team_away",
            "raw_team",
            "raw_name",
            "status_id",
            "reason_category",
            "reason_detail",
        ]
    ]

    # A status the dimension does not know is a change in what the NBA
    # publishes, not a row to drop quietly.
    unknown = players.loc[players["status_id"].isna(), "raw_name"]
    if len(unknown):
        labels = sorted(
            set(
                frame.loc[players.index[players["status_id"].isna()], "Current Status"]
                .dropna()
                .astype(str)
            )
        )
        raise ValueError(
            f"unknown Current Status {labels} at {observed_at:%Y-%m-%d %H:%M%z}; "
            "the status vocabulary was verified closed over 34,505 rows, so this "
            "needs investigating before it is loaded"
        )

    not_submitted = frame.loc[not_filed].copy().rename(columns={"Team": "raw_team"})
    not_submitted = not_submitted[
        ["game_date", "team_home", "team_away", "raw_team"]
    ].drop_duplicates()

    games = frame.rename(columns={"Game Time": "game_time_raw"})[GAME_COLUMNS]
    games = games.dropna(subset=["game_date", "team_home", "team_away"])
    games = games.drop_duplicates(subset=["game_date", "team_away", "team_home"])

    players = players.dropna(subset=["game_date", "team_home", "raw_team"])
    not_submitted = not_submitted.dropna(subset=["game_date", "team_home", "raw_team"])
    return ParsedReport(
        observed_at, players, not_submitted, games.reset_index(drop=True)
    )
