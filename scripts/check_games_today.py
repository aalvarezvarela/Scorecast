#!/usr/bin/env python3
"""Whether the retrain workflow refits today: only on a day with NBA games.

    python scripts/check_games_today.py [--game-date YYYY-MM-DD] [--always]

The date defaults to today's Eastern date (the slate a 09:00 Madrid run
precedes). Prints the answer and, inside GitHub Actions, writes
``refit=true|false`` to ``$GITHUB_OUTPUT``. ``--always`` answers true whatever
the scoreboard says.

When the scoreboard is unavailable the answer is true: a refit on a day without
games costs one run, a refit skipped on a game day serves yesterday's builds.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_tipoffs_for_date

EASTERN = ZoneInfo("America/New_York")


def should_refit(game_date: str) -> bool:
    games = fetch_tipoffs_for_date(game_date)
    if games is None:
        print(f"! Scoreboard unavailable for {game_date}: refitting anyway")
        return True
    print(f"{len(games)} game(s) on {game_date}")
    return not games.empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--game-date", default=None, help="YYYY-MM-DD (default: today, Eastern)"
    )
    parser.add_argument("--always", action="store_true", help="refit regardless")
    args = parser.parse_args()

    game_date = args.game_date or datetime.now(EASTERN).date().isoformat()
    refit = args.always or should_refit(game_date)
    print(f"refit={'true' if refit else 'false'}")
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"refit={'true' if refit else 'false'}\n")


if __name__ == "__main__":
    main()
