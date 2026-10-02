"""Strict DataFrame identity and synthetic ticks for the speed-up tests.

``assert_frame_equal(check_exact=True)`` still treats ``-0.0`` and ``0.0`` as
equal, and those are written differently to CSV. ``assert_identical`` adds a
bitwise comparison of every float column on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_identical(actual: pd.DataFrame | pd.Series, expected) -> None:
    if isinstance(expected, pd.Series):
        actual, expected = actual.to_frame(), expected.to_frame()
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=True, check_like=False, check_freq=True
    )
    for position in range(expected.shape[1]):
        left = actual.iloc[:, position]
        right = expected.iloc[:, position]
        if left.dtype.kind != "f":
            continue
        left, right = left.to_numpy(), right.to_numpy()
        nan = np.isnan(right)
        assert np.array_equal(np.isnan(left), nan), expected.columns[position]
        assert np.array_equal(left[~nan].view(np.int64), right[~nan].view(np.int64)), (
            f"bitwise mismatch in {expected.columns[position]!r}"
        )


def random_ticks(
    n_games: int = 12,
    books: tuple[str, ...] = ("bet365", "draftkings", "fanduel", "betmgm"),
    seed: int = 0,
) -> pd.DataFrame:
    """Ticks shaped like ``fetch_pregame_ticks``: all three markets, with ties
    in ``minutes_before_tip``, one-sided and missing quotes, openers, and
    moneyline prices whose devigged levels are not round numbers."""
    rng = np.random.default_rng(seed)
    records = []
    for game in range(n_games):
        game_id = f"00223{game:05d}"
        for market in ("totals", "spread", "moneyline"):
            for book in books:
                if rng.random() < 0.1:
                    continue
                n = int(rng.integers(1, 40))
                # Coarse minutes so equal timestamps occur within a series.
                minutes = np.sort(rng.choice(np.arange(0, 1600, 5.0), n))[::-1]
                if market == "totals":
                    line = 220 + np.cumsum(rng.choice([0, 0, 0.5, -0.5, 1.0], n))
                    left_line, right_line = line, line.copy()
                elif market == "spread":
                    line = -4 + np.cumsum(rng.choice([0, 0, 0.5, -0.5], n))
                    left_line, right_line = -line, line
                else:
                    left_line = right_line = np.full(n, np.nan)
                left_line = left_line.astype(float)
                right_line = right_line.astype(float)
                if market != "moneyline":
                    gone = rng.random(n) < 0.08
                    left_line[gone] = np.nan
                if market == "moneyline":
                    left_price = rng.choice([130.0, 145.0, 160.0, -105.0, 110.0], n)
                    right_price = rng.choice([-150.0, -170.0, -125.0, -115.0], n)
                else:
                    left_price = rng.choice([-110.0, -105.0, -115.0, -120.0, 100.0], n)
                    right_price = rng.choice([-110.0, -115.0, -105.0, -125.0], n)
                one_sided = rng.random(n) < 0.05
                right_price[one_sided] = np.nan
                for i in range(n):
                    records.append(
                        {
                            "game_id": game_id,
                            "season_year": 2023,
                            "market": market,
                            "book": book,
                            "line_ts": pd.Timestamp("2023-11-01", tz="UTC")
                            + pd.Timedelta(minutes=int(1600 - minutes[i])),
                            "minutes_before_tip": float(minutes[i]),
                            "left_line": left_line[i],
                            "right_line": right_line[i],
                            "left_price": left_price[i],
                            "right_price": right_price[i],
                            "is_opener": i == 0 and rng.random() < 0.7,
                        }
                    )
    ticks = pd.DataFrame(records)
    # The store does not hand ticks back in series order.
    return ticks.sample(frac=1.0, random_state=seed).reset_index(drop=True)
