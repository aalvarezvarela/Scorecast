"""Write final scores into the Aiven prediction store's ``games``.

Every scheduled game whose tip-off has passed is looked up, not only
yesterday's, so a postponed game or a failed fetch is retried on the next run.
Idempotent. ``--recheck-days`` also re-reads the games finalised in the last
N days, to pick up score corrections; ``--game-id`` re-reads only the games
named, whatever their status.

A game in progress is never written, even if the API already lists it with a
result: the live scoreboard's games are left out (or, if it cannot be read,
every game tipped in the last 4 hours), as the database update does.

A game still without a final score ``--stale-days`` after its stored tip-off
is reported: usually a postponement, which the next prediction run of its new
date retimes. Marking a game void (``status = 2``) is a manual update.

    python scripts/prediction_store/update_game_results.py [--recheck-days 3]
    python scripts/prediction_store/update_game_results.py --game-id 22600001
"""

from __future__ import annotations

import argparse

import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.predictions_aiven.results import (
    drop_unfinished,
    fetch_final_scores,
    live_game_ids,
)
from nba_ou.postgre_db.predictions_aiven.store import (
    games_by_id,
    pending_games,
    recently_final_games,
    stale_pending_games,
    update_game_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recheck-days",
        type=int,
        default=3,
        help="Also re-read games finalised in the last N days (default 3)",
    )
    parser.add_argument(
        "--game-id",
        type=int,
        action="append",
        help="Only these games (NBA id without leading zeros); repeatable",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=3,
        help="Report games still pending this many days after tip-off (default 3)",
    )
    args = parser.parse_args()
    with connect_nba_db(env="aiven") as conn:
        if args.game_id:
            games = games_by_id(conn, args.game_id)
            unknown = sorted(set(args.game_id) - set(games["game_id"]))
            if unknown:
                print(f"Not in the prediction store: {unknown}")
        else:
            games = pd.concat(
                [pending_games(conn), recently_final_games(conn, args.recheck_days)],
                ignore_index=True,
            ).drop_duplicates("game_id")
        if games.empty:
            print("No games awaiting a result.")
            return
        print(f"{len(games)} game(s) to look up")
        scores = fetch_final_scores(
            games["game_id"].tolist(), games["season_year"].tolist()
        )
        final = len(scores)
        scores = drop_unfinished(
            scores, games, live=live_game_ids(), now=pd.Timestamp.now(tz="UTC")
        )
        if len(scores) < final:
            print(f"{final - len(scores)} game(s) may still be in progress; skipped")
        changed = update_game_results(conn, scores)
        waiting = sorted(set(games["game_id"]) - set(scores["game_id"]))
        stale = stale_pending_games(conn, args.stale_days)
    print(f"{len(scores)} final, {changed} row(s) changed")
    if waiting:
        print(f"Still without a final result: {waiting}")
    if not stale.empty:
        print(
            f"Pending more than {args.stale_days} day(s) after tip-off "
            "(postponed? void?):"
        )
        print(stale.to_string(index=False))


if __name__ == "__main__":
    main()
