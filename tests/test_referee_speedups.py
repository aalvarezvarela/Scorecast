"""``compute_referee_features`` must return exactly what it returned before.

It now reuses each date's past-game split and per-referee deltas across the
games of that date. Compared with the previous implementation
(``tests/legacy_referee_features.py``) bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.referees.add_referee_features import (
    compute_referee_features,
)

from .frame_identity import assert_identical
from .legacy_referee_features import compute_referee_features_before

REFEREES = [f"Referee {name}" for name in "ABCDEFGHIJKL"]


def crews_frame(seed: int = 0, n_games: int = 400) -> pd.DataFrame:
    """Several games per date across two seasons, crews of one to three,
    missing metrics, and rows out of date order."""
    rng = np.random.default_rng(seed)
    days = np.sort(rng.integers(0, 150, n_games))
    season = np.where(np.arange(n_games) < n_games // 2, 2022, 2023)
    dates = (
        np.where(
            season == 2022,
            pd.Timestamp("2021-10-19").value,
            pd.Timestamp("2022-10-18").value,
        )
        + days * 86_400_000_000_000
    )
    crews = []
    for _ in range(n_games):
        size = rng.choice([3, 3, 3, 2, 1])
        crew = sorted(rng.choice(REFEREES, size, replace=False))
        crews.append(crew + [pd.NA] * (3 - size))
    frame = pd.DataFrame(
        {
            "GAME_ID": [f"00{i:08d}" for i in range(n_games)],
            "GAME_DATE": pd.to_datetime(dates),
            "SEASON_YEAR": season,
            "REF_1": [c[0] for c in crews],
            "REF_2": [c[1] for c in crews],
            "REF_3": [c[2] for c in crews],
            "TOTAL_POINTS": rng.normal(225, 18, n_games),
            "DIFF_FROM_LINE": rng.normal(0, 15, n_games),
            "TOTAL_PF": rng.integers(30, 50, n_games).astype(float),
            "OTHER": rng.random(n_games),
        }
    )
    for column in ("TOTAL_POINTS", "DIFF_FROM_LINE", "TOTAL_PF"):
        frame.loc[rng.random(n_games) < 0.05, column] = np.nan
    return frame.sample(frac=1.0, random_state=seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("trio", [False, True])
def test_referee_features_match_the_previous_implementation(seed, trio):
    frame = crews_frame(seed)
    expected = compute_referee_features_before(frame, include_ref_trio_features=trio)
    actual = compute_referee_features(frame, include_ref_trio_features=trio)
    assert_identical(actual, expected)
    assert expected["REF_AVG_TOTAL_POINTS_DIFF_BEFORE"].notna().sum() > 100
