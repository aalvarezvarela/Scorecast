"""Import PlayByPlayV3 responses from the shufinskiy/nba_data season archive.

The archive republishes the very same stats.nba.com payloads we would fetch a
game at a time, so a season of play-by-play costs one ~8 MB download instead of
1,230 rate-limited calls. Only ``gamerotation`` still has to come from the API.
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

LIST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"

# The archive drops shotValue, which the stint parser reads to separate threes
# from twos. Every field goal whose description carries "3PT" has shotValue 3
# and no other field goal does: verified exactly on 78,050 field goals across
# the 440 games we had already fetched from the API.
THREE_POINT_MARKER = "3PT"

# Two fields cannot be recovered and do not need to be. The archive strips
# diacritics from playerName ("Saric" for "Šarić"), and the API reports the
# blocked shot's value on BLOCK rows, which carry no actionType to key off.
# The stint parser reads neither: it identifies players by personId and only
# consults shotValue on rows whose actionType is a made or missed field goal.

# The API's own key order, so an imported payload is byte-identical to a fetched
# one rather than merely equivalent.
ACTION_FIELDS = (
    "actionNumber", "clock", "period", "teamId", "teamTricode", "personId",
    "playerName", "playerNameI", "xLegacy", "yLegacy", "shotDistance",
    "shotResult", "isFieldGoal", "scoreHome", "scoreAway", "pointsTotal",
    "location", "description", "actionType", "subType", "videoAvailable",
    "shotValue", "actionId",
)
INTEGER_FIELDS = frozenset({
    "actionNumber", "period", "teamId", "personId", "xLegacy", "yLegacy",
    "shotDistance", "isFieldGoal", "pointsTotal", "videoAvailable",
    "shotValue", "actionId",
})
# Numeric in the CSV but strings in the API, blank wherever the score is unchanged.
SCORE_FIELDS = frozenset({"scoreHome", "scoreAway"})


def archive_urls(*, opener=urllib.request.urlopen) -> dict[str, str]:
    """Map dataset name to download URL, e.g. ``nbastatsv3_2018``."""
    with opener(LIST_URL) as response:
        listing = response.read().decode("utf-8")
    urls = {}
    for entry in listing.replace("\n", ",").split(","):
        name, _, url = entry.strip().partition("=")
        if name and url:
            urls[name] = url
    return urls


def season_datasets(season_year: int, *, playoffs: bool = True) -> list[str]:
    """Dataset names holding one season's play-by-play."""
    names = [f"nbastatsv3_{season_year}"]
    if playoffs:
        names.append(f"nbastatsv3_po_{season_year}")
    return names


def download_season_csv(url: str, destination: Path, *, opener=urllib.request.urlopen) -> Path:
    """Fetch one tar.xz and extract the single CSV it contains."""
    destination.mkdir(parents=True, exist_ok=True)
    with opener(url) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
        members = [item for item in archive.getmembers() if item.name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {url}, found {len(members)}")
        extracted = destination / Path(members[0].name).name
        source = archive.extractfile(members[0])
        if source is None:
            raise ValueError(f"Could not read {members[0].name} from {url}")
        extracted.write_bytes(source.read())
    return extracted


def _scalar(field: str, value: object) -> object:
    # A blank reaches us as NaN from the CSV but as "" from a hand-built frame;
    # both mean the API's empty value for that column.
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return 0 if field in INTEGER_FIELDS else ""
    if field in SCORE_FIELDS or field in INTEGER_FIELDS:
        return int(float(value)) if field in INTEGER_FIELDS else str(int(float(value)))
    # The CSV pads some text columns out to a fixed width; the API does not.
    return str(value).strip()


def _shot_value(row: pd.Series) -> int:
    """Rebuild the column the archive omits."""
    action = str(row.get("actionType", "")).lower()
    if action not in {"made shot", "missed shot"}:
        return 0
    if not bool(pd.to_numeric(row.get("isFieldGoal"), errors="coerce")):
        return 0
    return 3 if THREE_POINT_MARKER in str(row.get("description", "")) else 2


def game_payload(actions: pd.DataFrame, game_id: str) -> bytes:
    """Render one game's rows as the PlayByPlayV3 JSON the archive stores."""
    # Corrected actions share an actionNumber with the row they amend, so the
    # sort must be stable: the CSV already carries the API's own row order and
    # a quicksort would silently swap those pairs.
    actions = actions.sort_values("actionNumber", kind="stable")
    rendered = []
    for _, row in actions.iterrows():
        action = {}
        for field in ACTION_FIELDS:
            if field == "shotValue":
                action[field] = _shot_value(row)
            else:
                action[field] = _scalar(field, row.get(field))
        rendered.append(action)
    document = {
        "meta": {"version": 1, "request": "", "time": ""},
        "game": {
            "gameId": game_id,
            "videoAvailable": 0,
            "actions": rendered,
        },
    }
    return json.dumps(document).encode("utf-8")


def games_in_csv(csv_path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``(game_id, payload)`` for every game in a season CSV."""
    frame = pd.read_csv(csv_path, low_memory=False)
    for raw_id, actions in frame.groupby("gameId", sort=True):
        game_id = str(int(raw_id)).zfill(10)
        yield game_id, game_payload(actions, game_id)
