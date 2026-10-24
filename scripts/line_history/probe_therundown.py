"""Read-only, bounded coverage probe for the existing RapidAPI subscription.

Example (run from any directory):
    python scripts/line_history/probe_therundown.py --dates 2026-10-24 \
        --check-open-close --history-line-id 19574377 --output /tmp/rundown.json

Only reference data and explicitly requested NBA dates/line IDs are queried.
No retries, database writes or production pipeline changes.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
HOST = "therundown-therundown-v1.p.rapidapi.com"
SBR_BOOKS = {
    "bet365": "bet365",
    "betmgm": "BetMGM",
    "fanduel": "Fanduel",
    "caesars": "Caesars Sportsbook",
    "draftkings": "Draftkings",
    "fanatics_sportsbook": "Fanatics",
    "betrivers": "BetRivers",
    "hard_rock_bet": "HardRock",
}
MARKET_FIELDS = {
    "total": ("total_over", "total_under", "total_over_money", "total_under_money"),
    "spread": (
        "point_spread_home",
        "point_spread_away",
        "point_spread_home_money",
        "point_spread_away_money",
    ),
    "moneyline": ("moneyline_home", "moneyline_away"),
}


def valid_market(market: dict, fields: tuple[str, ...]) -> bool:
    """Require both sides; V1's 0.0001 sentinel is unavailable, not a price."""
    for field in fields:
        value = market.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value == 0.0001
        ):
            return False
        if "money" in field and abs(value) < 100:
            return False
        if field in ("total_over", "total_under") and value <= 0:
            return False
    return True


def summarize_events(data: dict) -> dict:
    events = data.get("events", [])
    coverage = {}
    for slug, name in SBR_BOOKS.items():
        counts = Counter()
        examples = {}
        for event in events:
            for line in event.get("lines", {}).values():
                if (
                    line.get("affiliate", {}).get("affiliate_name", "").casefold()
                    != name.casefold()
                ):
                    continue
                counts["present"] += 1
                for market, fields in MARKET_FIELDS.items():
                    quote = line.get(market, {})
                    if not valid_market(quote, fields):
                        continue
                    counts[market] += 1
                    examples.setdefault(
                        market,
                        {
                            "event_id": event["event_id"],
                            "event_date": event["event_date"],
                            "line_id": line.get("line_id"),
                            "date_updated": quote.get("date_updated"),
                            **{field: quote[field] for field in fields},
                        },
                    )
        coverage[slug] = {
            "counts": {key: counts[key] for key in ("present", *MARKET_FIELDS)},
            "examples": examples,
        }
    return {
        "events": len(events),
        "statuses": dict(
            Counter(e.get("score", {}).get("event_status", "unknown") for e in events)
        ),
        "season_years": sorted(
            {e.get("schedule", {}).get("season_year") for e in events}, key=str
        ),
        "books": coverage,
    }


def season_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected YYYY-MM-DD") from exc
    if not date(2026, 10, 1) <= parsed < date(2027, 7, 1):
        raise argparse.ArgumentTypeError("Use dates within the 2026-27 NBA season")
    return parsed.isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", type=season_date, required=True)
    parser.add_argument("--check-open-close", action="store_true")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Date boundary offset in minutes; 240 = Eastern daylight time",
    )
    parser.add_argument(
        "--affiliate-ids", help="Comma-separated IDs returned by /affiliates"
    )
    parser.add_argument("--history-line-id", type=int)
    parser.add_argument(
        "--history-markets", nargs="+", choices=list(MARKET_FIELDS), default=["total"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.dates) > 3:
        parser.error("At most three explicit dates per probe")
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "src/nba_ou/config.ini")
    secrets = configparser.ConfigParser(interpolation=None)
    secrets.read([ROOT / "src/nba_ou/config.secrets.ini", ROOT / "config.secrets.ini"])
    key = os.getenv("ODDS_API_KEY") or secrets.get("Odds", "ODDS_API_KEY", fallback="")
    if not key:
        parser.error("Missing ODDS_API_KEY in environment or secrets config")
    base = config.get("Odds", "BASE_URL").rstrip("/")
    if base != f"https://{HOST}":
        parser.error("Expected the project's TheRundown RapidAPI host")
    paths = ["/affiliates"]
    for day in dict.fromkeys(args.dates):
        paths.append(f"/sports/4/events/{day}")
        if args.check_open_close:
            paths.extend(f"/sports/4/{kind}/{day}" for kind in ("openers", "closing"))
    if args.history_line_id is not None:
        paths.extend(
            f"/lines/{args.history_line_id}/{market}"
            for market in dict.fromkeys(args.history_markets)
        )
    report = {"checked_at": datetime.now(UTC).isoformat(), "requests": []}
    failed = False
    for path in paths:
        entry = {"path": path}
        params = {"offset": args.offset} if path.startswith("/sports/") else {}
        if args.affiliate_ids and path.startswith("/sports/"):
            params["affiliate_ids"] = args.affiliate_ids
        entry["params"] = params
        try:
            response = requests.get(
                base + path,
                headers={"X-RapidAPI-Key": key, "X-RapidAPI-Host": HOST},
                params=params,
                timeout=(10, 25),
                allow_redirects=False,
            )
            entry.update(
                status=response.status_code,
                quota={
                    k: v
                    for k, v in response.headers.items()
                    if "ratelimit-requests-" in k.lower()
                },
            )
            if response.status_code == 200:
                data = response.json()
                if path == "/affiliates":
                    entry["affiliates"] = [
                        {
                            k: a.get(k)
                            for k in ("affiliate_id", "affiliate_name", "status")
                        }
                        for a in data.get("affiliates", [])
                    ]
                elif "events" in data:
                    entry["summary"] = summarize_events(data)
                else:
                    market = path.rsplit("/", 1)[-1]
                    ticks = data.get(
                        {
                            "total": "totals",
                            "spread": "spreads",
                            "moneyline": "moneylines",
                        }[market],
                        [],
                    )
                    timestamps = sorted(
                        t["date_updated"] for t in ticks if t.get("date_updated")
                    )
                    entry[f"{market}_history"] = {
                        "rows": len(ticks),
                        "valid_rows": sum(
                            valid_market(t, MARKET_FIELDS[market]) for t in ticks
                        ),
                        "first_update": timestamps[0] if timestamps else None,
                        "last_update": timestamps[-1] if timestamps else None,
                        "possible_truncation": len(ticks) >= 1000,
                    }
            else:
                failed = True
                entry["error"] = response.text[:500].replace(key, "[REDACTED]")
        except (requests.RequestException, ValueError) as exc:
            failed = True
            entry["error"] = type(exc).__name__
        report["requests"].append(entry)
        print(path, entry.get("status", entry.get("error")), flush=True)
        if "error" in entry:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2).replace(key, "[REDACTED]") + "\n"
    )
    print(f"Report: {args.output}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
