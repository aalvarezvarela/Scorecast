"""The three leakage checks ``docs/lineups/lineup_projection.md`` section 8.3 requires.

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

import pandas as pd
import pytest
from nba_ou.data_processing.lineups import features
from nba_ou.data_processing.lineups.features import (
    RatingBook,
    add_lineup_features,
    attach_lineup_features,
)


from tests.support.lineup_world import (
    PROJECTION_COLUMNS,
    H,
    A,
    C,
    ROSTERS,
    DATES,
    TARGET_DATE,
    _game_id,
    TARGET,
    SAME_DAY,
    _world,
)

@pytest.fixture(autouse=True)
def calibrate_from_few_games(monkeypatch):
    """This world has a handful of games; production needs OFFSET_MIN_GAMES."""
    monkeypatch.setattr(features, "OFFSET_MIN_GAMES", 1)


#: The projection's columns. The three-point matchup columns come from stints,
#: which this world does not have; their temporal contract is tested in
#: ``tests/features/test_style_matchup.py``.


def _coverage(merged):
    return {
        (row.GAME_ID, team)
        for row in merged.itertuples(index=False)
        for team in (row.TEAM_ID_TEAM_HOME, row.TEAM_ID_TEAM_AWAY)
    }


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
            report_covered=_coverage(merged),
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
