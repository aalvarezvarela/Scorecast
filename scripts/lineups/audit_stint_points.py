"""Audit the stored stints' points per season (diagnostic, read-only).

A game's segment points must sum to its final score, but opposite errors
cancel in that sum: the 2016-17 store once built from placeholder "0" scores
had stints of -131 and +262 in 79% of its games and still reconciled. This
reports, per season, the games with any negative stint (V3 score corrections,
normally 1-2 small ones) and the games outside
``lineups.stints.POINT_ANOMALY_LIMITS``, which ``validate_game_stints`` now
rejects as ``points_anomaly`` (a stored game built before that check can
still be one).

    python scripts/lineups/audit_stint_points.py --first-season 2016 --last-season 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.lineups.stints import (
    POINT_ANOMALY_LIMITS,
    point_anomalies,
    points_are_plausible,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--first-season", type=int, required=True)
    parser.add_argument("--last-season", type=int, required=True)
    args = parser.parse_args()
    print(f"limits: {POINT_ANOMALY_LIMITS}")
    for season in range(args.first_season, args.last_season + 1):
        stints = read_stints([season], local_root=args.local_root)
        if stints.empty:
            print(f"{season}: no stints")
            continue
        games = {
            game_id: point_anomalies(game)
            for game_id, game in stints.groupby("game_id")
        }
        corrected = [g for g, a in games.items() if a["negative_stints"]]
        anomalous = {g: a for g, a in games.items() if not points_are_plausible(a)}
        print(
            f"{season}: {len(games):,} games, {len(corrected)} with a negative stint, "
            f"{len(anomalous)} outside the limits"
            + "".join(f"\n    {g} {a}" for g, a in sorted(anomalous.items())[:10])
            + (f"\n    ... {len(anomalous) - 10} more" if len(anomalous) > 10 else "")
        )


if __name__ == "__main__":
    main()
