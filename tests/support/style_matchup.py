"""Synthetic stints shared by style and lineup feature tests."""

import numpy as np
import pandas as pd

TEAMS = {"H": [f"h{i}" for i in range(7)], "A": [f"a{i}" for i in range(7)]}


def _stints(dates, seed=0, shooter="h0", boost=0):
    """Random 5-v-5 stints between H and A; ``shooter`` raises his side's 3PA."""
    rng = np.random.default_rng(seed)
    rows = []
    for n, date in enumerate(dates):
        for _seg in range(12):
            home = sorted(rng.choice(TEAMS["H"], 5, replace=False))
            away = sorted(rng.choice(TEAMS["A"], 5, replace=False))
            row = {
                "game_id": f"00{n:08d}",
                "game_date": pd.Timestamp(date),
                "seconds": 240.0,
                "home_lineup": [int(p[1:]) + 100 for p in home],
                "away_lineup": [int(p[1:]) + 200 for p in away],
            }
            for side, lineup in (("home", home), ("away", away)):
                fga = int(rng.integers(6, 12))
                extra = boost if shooter in lineup else 0
                row |= {
                    f"{side}_pts": int(rng.integers(8, 16)),
                    f"{side}_fga": fga,
                    f"{side}_fg3a": min(fga, int(rng.integers(1, 5)) + extra),
                    f"{side}_fta": int(rng.integers(0, 4)),
                    f"{side}_oreb": int(rng.integers(0, 3)),
                    f"{side}_dreb": int(rng.integers(2, 6)),
                    f"{side}_tov": int(rng.integers(0, 3)),
                    f"{side}_poss": float(rng.integers(8, 12)),
                }
            rows.append(row)
    return pd.DataFrame(rows)


DATES = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-10-01", periods=60)]
