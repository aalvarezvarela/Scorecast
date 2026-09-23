"""The three leakage checks ``docs/lineup_projection_plan.md`` section 8.3 requires.

1. **Perturbation.** Changing the target game's own box score and result, a
   same-day game's, and any rating fitted on or after the game date leaves
   every ``LU_*`` value of the target unchanged.
2. **Fit windows.** Every rating a game reads was fitted on games strictly
   before that game's date.
3. **Injury cut-off.** A game's availability comes from its own pre-game
   listing only. Same-day and later listings cannot reach it; an earlier
   game's listing reaches it only through the level calibration, which reads
   that earlier game's projection -- known before tip, so not leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.features import (
    LINEUP_FEATURE_COLUMNS,
    RatingBook,
    add_lineup_features,
    attach_lineup_features,
)
from nba_ou.data_processing.lineups.style_matchup import STYLE_FEATURE_COLUMNS

#: The projection's columns. The three-point matchup columns come from stints,
#: which this world does not have; their temporal contract is tested in
#: ``tests/test_style_matchup.py``.
PROJECTION_COLUMNS = [
    c for c in LINEUP_FEATURE_COLUMNS if c not in STYLE_FEATURE_COLUMNS
]

H, A, C, D = "1610612739", "1610612738", "1610612737", "1610612736"
ROSTERS = {team: [f"{team[-2:]}p{i}" for i in range(6)] for team in (H, A, C, D)}
DATES = [f"2025-01-{day:02d}" for day in range(1, 9)]
TARGET_DATE = DATES[-1]


def _game_id(index: int, slot: int) -> str:
    return f"00{index:06d}{slot:02d}"


TARGET = _game_id(len(DATES) - 1, 1)
SAME_DAY = _game_id(len(DATES) - 1, 2)


def _world(seed: int = 0):
    """Eight dates, two games a night, with every target-day row populated.

    The target date's box scores and results are filled in on purpose: in a
    historical build they exist, and the point is to prove they are not read.
    """
    rng = np.random.default_rng(seed)
    box, merged = [], []
    for index, date in enumerate(DATES):
        for slot, (home, away) in ((1, (H, A)), (2, (C, D))):
            game = _game_id(index, slot)
            for team in (home, away):
                minutes = rng.dirichlet(np.ones(6)) * 240.0
                for player, played in zip(ROSTERS[team], minutes, strict=True):
                    box.append(
                        {
                            "GAME_ID": game,
                            "TEAM_ID": team,
                            "GAME_DATE": pd.Timestamp(date),
                            "PLAYER_ID": player,
                            "MIN": float(played),
                        }
                    )
            merged.append(
                {
                    "GAME_ID": game,
                    "GAME_DATE": pd.Timestamp(date),
                    "TEAM_ID_TEAM_HOME": home,
                    "TEAM_ID_TEAM_AWAY": away,
                    "TOTAL_POINTS": float(rng.integers(190, 250)),
                }
            )
    ratings = []
    for as_of in pd.date_range(DATES[0], "2025-01-20"):
        for players in ROSTERS.values():
            for player in players:
                ratings.append(
                    {
                        "as_of_date": as_of,
                        "player_id": player,
                        "o_rating": float(rng.normal(0, 2)),
                        "d_rating": float(rng.normal(0, 2)),
                        "pace_rating": float(rng.normal(0, 1)),
                        "league_ortg": 112.0 + float(rng.normal(0, 1)),
                        "league_pace": 100.0 + float(rng.normal(0, 1)),
                        "fit_max_game_date": as_of - pd.Timedelta(days=1),
                    }
                )
    return pd.DataFrame(box), pd.DataFrame(merged), pd.DataFrame(ratings)


def _target_values(box, merged, ratings, p_out=None) -> pd.Series:
    out = add_lineup_features(merged, box, RatingBook(ratings), p_out)
    return out.loc[out.GAME_ID.eq(TARGET), PROJECTION_COLUMNS].iloc[0]


P_OUT = {(TARGET, H, ROSTERS[H][0]): 1.0, (TARGET, A, ROSTERS[A][1]): 0.4}


def test_baseline_is_fully_populated():
    box, merged, ratings = _world()
    values = _target_values(box, merged, ratings, P_OUT)
    assert values.notna().all()
    assert values["LU_ABSENCE_IMPACT_PTS_BEFORE"] != 0


class TestPerturbation:
    def test_target_and_same_day_box_scores_are_not_read(self):
        box, merged, ratings = _world()
        baseline = _target_values(box, merged, ratings, P_OUT)
        today = box.GAME_DATE.eq(TARGET_DATE)
        assert set(box.loc[today, "GAME_ID"]) == {TARGET, SAME_DAY}
        box.loc[today, "MIN"] = box.loc[today, "MIN"][::-1].to_numpy() * 3.0
        box = pd.concat(
            [
                box,
                pd.DataFrame(
                    [
                        {
                            "GAME_ID": TARGET,
                            "TEAM_ID": H,
                            "GAME_DATE": pd.Timestamp(TARGET_DATE),
                            "PLAYER_ID": "newcomer",
                            "MIN": 40.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        pd.testing.assert_series_equal(
            _target_values(box, merged, ratings, P_OUT), baseline
        )

    def test_target_and_same_day_results_are_not_read(self):
        box, merged, ratings = _world()
        baseline = _target_values(box, merged, ratings, P_OUT)
        today = merged.GAME_DATE.eq(TARGET_DATE)
        merged.loc[today, "TOTAL_POINTS"] = [400.0, 50.0]
        pd.testing.assert_series_equal(
            _target_values(box, merged, ratings, P_OUT), baseline
        )

    def test_ratings_dated_after_the_game_are_not_read(self):
        box, merged, ratings = _world()
        baseline = _target_values(box, merged, ratings, P_OUT)
        later = ratings.as_of_date.gt(TARGET_DATE)
        ratings.loc[later, ["o_rating", "d_rating", "pace_rating"]] = 99.0
        ratings.loc[later, "league_ortg"] = 150.0
        pd.testing.assert_series_equal(
            _target_values(box, merged, ratings, P_OUT), baseline
        )

    def test_the_perturbation_would_be_seen_if_it_were_read(self):
        """Guards the tests above against passing vacuously."""
        box, merged, ratings = _world()
        baseline = _target_values(box, merged, ratings, P_OUT)
        on_the_day = ratings.as_of_date.eq(TARGET_DATE)
        ratings.loc[on_the_day, "league_ortg"] = 150.0
        changed = _target_values(box, merged, ratings, P_OUT)
        assert changed["LU_PROJ_TOTAL_BEFORE"] != pytest.approx(
            baseline["LU_PROJ_TOTAL_BEFORE"]
        )


class TestFitWindows:
    def test_every_rating_read_was_fitted_before_the_game(self):
        box, merged, ratings = _world()
        book = RatingBook(ratings)
        fits = ratings.groupby("as_of_date").fit_max_game_date.max()
        for date in merged.GAME_DATE.unique():
            chosen = book.as_of(pd.Timestamp(date))
            assert chosen is not None and chosen <= pd.Timestamp(date)
            assert fits.loc[chosen] < pd.Timestamp(date)

    def test_a_cache_row_fitted_on_its_own_date_is_refused(self):
        _, _, ratings = _world()
        ratings.loc[ratings.index[-1], "fit_max_game_date"] = ratings.loc[
            ratings.index[-1], "as_of_date"
        ]
        with pytest.raises(ValueError, match="strictly before"):
            RatingBook(ratings)


class TestInjuryCutoff:
    def _statuses(self, rows):
        return pd.DataFrame(
            rows,
            columns=["game_id", "team_id", "player_id", "status", "reason_category"],
        )

    def _values(self, statuses):
        box, merged, ratings = _world()
        out = attach_lineup_features(
            merged,
            box,
            enabled=True,
            injury_statuses=statuses,
            ratings=RatingBook(ratings),
            stints=pd.DataFrame(),
        )
        return out.loc[out.GAME_ID.eq(TARGET), PROJECTION_COLUMNS].iloc[0]

    def test_same_day_and_later_listings_do_not_reach_the_game(self):
        own = [(TARGET, H, ROSTERS[H][0], "out", "injury")]
        foreign = own + [
            (SAME_DAY, C, ROSTERS[C][0], "out", "injury"),
            # Same team, a later game.
            ("0099999901", H, ROSTERS[H][3], "out", "injury"),
        ]
        own_values = self._values(self._statuses(own))
        pd.testing.assert_series_equal(
            self._values(self._statuses(foreign)), own_values
        )
        assert own_values["LU_ABSENCE_IMPACT_PTS_BEFORE"] != 0

    def test_an_earlier_listing_moves_only_the_calibrated_level(self):
        """Yesterday's report is known before tonight's tip, and it changes
        yesterday's projection -- hence the level calibration tonight reads.
        It must not reach tonight's availability or counterfactual."""
        own = [(TARGET, H, ROSTERS[H][0], "out", "injury")]
        earlier = own + [
            (_game_id(len(DATES) - 2, 1), H, ROSTERS[H][2], "out", "injury")
        ]
        own_values = self._values(self._statuses(own))
        earlier_values = self._values(self._statuses(earlier))
        unaffected = [c for c in PROJECTION_COLUMNS if c != "LU_PROJ_TOTAL_BEFORE"]
        pd.testing.assert_series_equal(
            earlier_values[unaffected], own_values[unaffected]
        )
