"""Download written NBA.com game recaps for a selected NBA calendar date.

Edit DATE and run this file or its # %% cells in order; no CLI arguments needed.
Reports are saved under data/nba_game_reports/YYYY-MM-DD, keyed by the official
NBA game ID. JSON contains article text and metadata; TXT is for reading.
The manifest records unavailable recaps, unfinished games, and request errors.

Requires requests and beautifulsoup4. NBA.com's page structure may change.
"""

# %% Imports
import csv
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# %% Configuration
DATE = "2025-11-28"  # NBA calendar date in the US, YYYY-MM-DD.
PROJECT_ROOT = next(
    parent
    for parent in Path(globals().get("__file__", Path.cwd() / "get_game_reports.py"))
    .resolve()
    .parents
    if (parent / "pyproject.toml").is_file()
)
OUTPUT = PROJECT_ROOT / "data" / "nba_game_reports"
OVERWRITE = False  # Reuse previously saved reports unless set to True.
REQUEST_TIMEOUT = 30  # Seconds per request.
# NBA.com story dates are US Eastern wall-clock times without an offset: read
# that way, 2025-26 recaps land 18 min to hours after the final buzzer, and the
# delay does not jump at either daylight-saving change.
SOURCE_TIMEZONE = ZoneInfo("America/New_York")

MANIFEST_COLUMNS = [
    "date",
    "game_id",
    "away",
    "home",
    "status",
    "title",
    "author",
    "agency",
    "published_at",
    "published_at_utc",
    "word_count",
    "source_url",
    "json_path",
    "text_path",
    "error",
]


# %% Request and parsing helpers
def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nba.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
        ),
    )
    return session


def parse_page_props(html: str) -> dict:
    """Read the page's embedded data without requiring a browser."""
    script = BeautifulSoup(html, "html.parser").find("script", id="__NEXT_DATA__")
    if script is None:
        raise ValueError(
            "NBA.com page has no __NEXT_DATA__; it may be blocked or its layout changed"
        )
    payload = json.loads(script.get_text())
    props = payload.get("props", {}).get("pageProps")
    if not isinstance(props, dict):
        raise ValueError("NBA.com page has no pageProps object")
    return props


def fetch_page_props(session: requests.Session, url: str, timeout: int) -> dict:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_page_props(response.text)


def parse_games(props: dict, game_date: date) -> list[dict]:
    """Read game cards from the official NBA.com date selector."""
    if props.get("hasError") or props.get("selectedDate") != game_date.isoformat():
        raise ValueError("NBA.com did not return the requested calendar date")
    feed = props.get("gameCardFeed")
    if not isinstance(feed, dict) or not isinstance(feed.get("modules"), list):
        raise ValueError(
            "NBA.com gameCardFeed is missing or has an unexpected structure"
        )
    games = {}
    for module in feed["modules"]:
        for card in module.get("cards", []):
            game = card.get("cardData", {})
            game_id = game.get("gameId")
            if game_id is None:
                continue  # Non-game cards may also appear in the feed.
            game_id = str(game_id)
            if not re.fullmatch(r"\d{10}", game_id):
                raise ValueError(f"Invalid official NBA game ID: {game_id}")
            away, home = game["awayTeam"], game["homeTeam"]
            slug = f"{away['teamTricode']}-vs-{home['teamTricode']}-{game_id}".lower()
            games[game_id] = {
                "game_id": game_id,
                "away": away["teamName"],
                "home": home["teamName"],
                "game_status": game["gameStatus"],
                "source_url": f"https://www.nba.com/game/{slug}",
            }
    return list(games.values())


def published_at_utc(published_at: str | None) -> str:
    """Convert NBA.com's offset-less Eastern story date to UTC ("" if absent)."""
    if not published_at:
        return ""
    moment = datetime.fromisoformat(published_at)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=SOURCE_TIMEZONE)
    return moment.astimezone(UTC).isoformat()


def clean_text(value: str) -> str:
    return " ".join(
        BeautifulSoup(value, "html.parser").get_text(" ", strip=True).split()
    )


def parse_report(props: dict, game: dict, game_date: date) -> dict | None:
    """Extract the written story, never the separate gameRecap video excerpt."""
    details = props.get("game") or {}
    if details.get("gameId") != game["game_id"]:
        raise ValueError("The recap page belongs to a different game ID")
    if str(details.get("gameEt", ""))[:10] != game_date.isoformat():
        raise ValueError("The recap page belongs to a different NBA calendar date")
    if details.get("gameStatus") != 3:
        return None
    story = props.get("story")
    if not story:
        return None
    if not isinstance(story, dict):
        raise ValueError("Unexpected written-story structure")
    if story.get("storyType") != "recap":
        return None  # A pregame preview is not a postgame report.
    content = story.get("content") or []
    if not isinstance(content, list) or any(not isinstance(p, str) for p in content):
        raise ValueError(
            "Unexpected written-story content; expected a list of paragraphs"
        )
    paragraphs = [text for paragraph in content if (text := clean_text(paragraph))]
    if not paragraphs:
        return None
    header = story.get("header") or {}
    text = "\n\n".join(paragraphs)
    bytitle = clean_text(header.get("bytitle") or "")
    agency = bytitle
    if "associated press" in bytitle.lower() or "(AP)" in text:
        agency = "Associated Press"
    return {
        "game_id": game["game_id"],
        "date": game_date.isoformat(),
        "away": game["away"],
        "home": game["home"],
        "away_score": (details.get("awayTeam") or {}).get("score"),
        "home_score": (details.get("homeTeam") or {}).get("score"),
        "source_url": game["source_url"],
        "title": clean_text(header.get("headline") or ""),
        "author": clean_text(header.get("byline") or ""),
        "bytitle": bytitle,
        "agency": agency,
        # The source timestamp is kept verbatim (US Eastern, no offset).
        "published_at": story.get("date") or "",
        "published_at_utc": published_at_utc(story.get("date") or ""),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "paragraphs": paragraphs,
        "text": text,
        "word_count": len(text.split()),
    }


# %% Download helpers
def save_report(report: dict, json_path: Path, text_path: Path) -> None:
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    header = [
        report["title"],
        report["author"],
        report["agency"],
        f"NBA game ID: {report['game_id']}",
        f"Game date: {report['date']}",
        f"Published: {report['published_at']} US Eastern",
        f"Source: {report['source_url']}",
    ]
    text_path.write_text(
        "\n".join(line for line in header if line) + "\n\n" + report["text"] + "\n",
        encoding="utf-8",
    )


def download_reports(
    selected_date: str, output: Path, overwrite: bool = False, timeout: int = 30
) -> list[dict]:
    """Download available reports and return one manifest row per scheduled game."""
    game_date = date.fromisoformat(selected_date)
    destination = Path(output) / game_date.isoformat()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with create_session() as session:
        day_url = f"https://www.nba.com/games?date={game_date.isoformat()}"
        games = parse_games(fetch_page_props(session, day_url, timeout), game_date)
        print(f"NBA date: {game_date} | Games: {len(games)}")
        for index, game in enumerate(games, 1):
            row = dict.fromkeys(MANIFEST_COLUMNS, "")
            row.update(
                {key: game[key] for key in ("game_id", "away", "home", "source_url")}
            )
            row["date"] = game_date.isoformat()
            json_path = destination / f"{game['game_id']}.json"
            text_path = destination / f"{game['game_id']}.txt"
            print(
                f"[{index}/{len(games)}] {game['away']} @ {game['home']} | {game['game_id']}"
            )
            try:
                if game["game_status"] != 3:
                    row["status"] = "not_final"
                else:
                    cached = (
                        not overwrite and json_path.is_file() and text_path.is_file()
                    )
                    if cached:
                        report = json.loads(json_path.read_text(encoding="utf-8"))
                        if (
                            report.get("game_id") != game["game_id"]
                            or report.get("date") != game_date.isoformat()
                            or not report.get("text")
                        ):
                            raise ValueError(
                                "Invalid cached report; rerun with OVERWRITE = True"
                            )
                        # Reports saved before the UTC field existed.
                        report.setdefault(
                            "published_at_utc",
                            published_at_utc(report.get("published_at", "")),
                        )
                    else:
                        props = fetch_page_props(session, game["source_url"], timeout)
                        report = parse_report(props, game, game_date)
                    if report is None:
                        row["status"] = "not_available"
                    else:
                        if not cached:
                            save_report(report, json_path, text_path)
                        row.update(
                            {
                                key: report.get(key, "")
                                for key in (
                                    "title",
                                    "author",
                                    "agency",
                                    "published_at",
                                    "published_at_utc",
                                    "word_count",
                                )
                            }
                        )
                        row.update(
                            status="cached" if cached else "saved",
                            json_path=str(json_path.resolve()),
                            text_path=str(text_path.resolve()),
                        )
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:
                row.update(status="error", error=str(exc))
            print(
                f"    {row['status']}" + (f": {row['error']}" if row["error"] else "")
            )
            rows.append(row)

    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    available = sum(row["status"] in {"saved", "cached"} for row in rows)
    print(f"Reports available: {available}/{len(rows)}")
    print(f"Output directory: {destination}")
    print(f"Manifest: {manifest_path}")
    return rows


# %% Download the reports
if __name__ == "__main__":
    results = download_reports(
        DATE, OUTPUT, overwrite=OVERWRITE, timeout=REQUEST_TIMEOUT
    )
