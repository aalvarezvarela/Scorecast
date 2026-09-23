"""Phase D: residuals, decay, shrinkage and the shared-minutes weighting."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.lineups.synergy import (
    PAIRS_ON_COURT,
    SynergyAccumulator,
    expected_shared_minutes,
    projected_five_synergy,
    stint_residuals,
    team_synergy,
)

DAY = pd.Timestamp("2025-01-01")
FIVE = ("a", "b", "c", "d", "e")
OTHER = ("v", "w", "x", "y", "z")


def _ratings(date=DAY, players=FIVE + OTHER, o=0.0, d=0.0, league=112.0):
    return pd.DataFrame(
        {
            "as_of_date": [date] * len(players),
            "player_id": list(players),
            "o_rating": [o] * len(players),
            "d_rating": [d] * len(players),
            "league_ortg": [league] * len(players),
        }
    )


def _home_row(residuals):
    """Row 0 is the home side on offense; pandas cannot compare a tuple column."""
    return residuals.iloc[0]


def _stint(home_pts=10.0, home_poss=10.0, seconds=120.0, date=DAY):
    return pd.DataFrame(
        [
            {
                "game_date": date,
                "home_lineup": list(FIVE),
                "away_lineup": list(OTHER),
                "home_pts": home_pts,
                "away_pts": 10.0,
                "home_poss": home_poss,
                "away_poss": 10.0,
                "seconds": seconds,
            }
        ]
    )


class TestResiduals:
    def test_a_league_average_lineup_has_no_residual(self):
        residuals = stint_residuals(_stint(home_pts=11.2, home_poss=10.0), _ratings())
        assert _home_row(residuals).offense == FIVE
        assert _home_row(residuals).residual == pytest.approx(0.0)

    def test_outscoring_the_prediction_gives_a_positive_residual(self):
        residuals = stint_residuals(_stint(home_pts=15.0, home_poss=10.0), _ratings())
        assert _home_row(residuals).residual == pytest.approx(150.0 - 112.0)

    def test_the_ratings_prediction_is_subtracted(self):
        """Five players at +2 each should be expected to score 10 more per 100."""
        ratings = _ratings(o=2.0)
        residuals = stint_residuals(_stint(home_pts=11.2, home_poss=10.0), ratings)
        # Predicted 112 + 10 for the offence; the defence carries d_rating 0.
        assert _home_row(residuals).residual == pytest.approx(112.0 - (112.0 + 10.0))

    def test_a_segment_without_possessions_is_skipped(self):
        residuals = stint_residuals(_stint(home_poss=0.0), _ratings())
        # Only the away side survives, so the home lineup is absent.
        assert len(residuals) == 1
        assert residuals.iloc[0].offense == OTHER

    def test_empty_inputs_give_an_empty_frame(self):
        assert stint_residuals(pd.DataFrame(), _ratings()).empty
        assert stint_residuals(_stint(), pd.DataFrame()).empty


def _seed_league(accumulator, date=DAY, possessions=10_000_000.0):
    """Give the accumulator a neutral league to deviate from.

    Every value is reported as a deviation from the running league mean, so a
    lone observation is by definition the whole league and has zero synergy.
    A large neutral mass fixes that mean near zero without dominating any one
    pair's own possession count.
    """
    accumulator.observe(
        ("n1", "n2", "n3", "n4", "n5"),
        residual=0.0,
        possessions=possessions,
        seconds=possessions,
        date=date,
    )


class TestAccumulator:
    def test_an_unseen_pair_is_zero_not_missing(self):
        accumulator = SynergyAccumulator()
        assert accumulator.pair_value("a", "b", DAY) == 0.0
        assert accumulator.five_value(frozenset(FIVE), DAY) == 0.0

    def test_shrinkage_pulls_a_thin_sample_toward_zero(self):
        accumulator = SynergyAccumulator(shrinkage_possessions=200.0)
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=10.0, possessions=20.0, seconds=60.0, date=DAY
        )
        # 20 possessions against k=200, and a tenth again because the value is
        # a per-pair contribution to its lineup, not the lineup's own residual.
        assert accumulator.pair_value("a", "b", DAY) == pytest.approx(
            10.0 / PAIRS_ON_COURT * 20.0 / 220.0, rel=1e-3
        )

    def test_more_evidence_shrinks_less(self):
        thin = SynergyAccumulator(shrinkage_possessions=200.0)
        _seed_league(thin)
        thin.observe(FIVE, residual=10.0, possessions=20.0, seconds=60.0, date=DAY)
        thick = SynergyAccumulator(shrinkage_possessions=200.0)
        _seed_league(thick)
        thick.observe(FIVE, residual=10.0, possessions=2000.0, seconds=6000.0, date=DAY)
        assert thick.pair_value("a", "b", DAY) > thin.pair_value("a", "b", DAY)
        assert thick.pair_value("a", "b", DAY) == pytest.approx(
            10.0 / PAIRS_ON_COURT * 2000 / 2200, rel=1e-2
        )

    def test_evidence_decays_with_age(self):
        """A half-life old observation must count half as much, not twice."""
        accumulator = SynergyAccumulator(
            shrinkage_possessions=1.0, half_life_days=180.0
        )
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=10.0, possessions=100.0, seconds=60.0, date=DAY
        )
        fresh = accumulator.pair_value("a", "b", DAY)
        later = accumulator.pair_value("a", "b", DAY + pd.Timedelta(days=180))
        assert later < fresh
        # The mean is unchanged; the shrinkage sees half the possessions.
        assert later == pytest.approx(10.0 / PAIRS_ON_COURT * 50.0 / 51.0, rel=1e-2)

    def test_decay_does_not_change_the_mean_residual(self):
        accumulator = SynergyAccumulator(shrinkage_possessions=0.001)
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=7.0, possessions=500.0, seconds=60.0, date=DAY
        )
        old = accumulator.pair_value("a", "b", DAY + pd.Timedelta(days=360))
        assert old == pytest.approx(7.0 / PAIRS_ON_COURT, rel=1e-2)

    def test_recent_evidence_outweighs_old_evidence(self):
        accumulator = SynergyAccumulator(
            shrinkage_possessions=0.001, half_life_days=180.0
        )
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=-10.0, possessions=100.0, seconds=60.0, date=DAY
        )
        accumulator.observe(
            FIVE,
            residual=10.0,
            possessions=100.0,
            seconds=60.0,
            date=DAY + pd.Timedelta(days=360),
        )
        value = accumulator.pair_value("a", "b", DAY + pd.Timedelta(days=360))
        assert value > 0

    def test_the_league_mean_is_not_synergy(self):
        """A lone lineup is the whole league, so its deviation is zero."""
        accumulator = SynergyAccumulator(shrinkage_possessions=0.001)
        accumulator.observe(
            FIVE, residual=9.0, possessions=1000.0, seconds=600.0, date=DAY
        )
        assert accumulator.league_mean() == pytest.approx(9.0)
        assert accumulator.pair_value("a", "b", DAY) == pytest.approx(0.0)

    def test_a_pair_inherits_every_lineup_it_appeared_in(self):
        accumulator = SynergyAccumulator(shrinkage_possessions=0.001)
        accumulator.observe(
            FIVE, residual=10.0, possessions=100.0, seconds=60.0, date=DAY
        )
        accumulator.observe(
            ("a", "b", "x", "y", "z"),
            residual=10.0,
            possessions=100.0,
            seconds=60.0,
            date=DAY,
        )
        # a-b were in both; a-c in only one.
        assert accumulator.pair_seconds("a", "b") > accumulator.pair_seconds("a", "c")

    def test_a_negative_possession_segment_is_ignored(self):
        accumulator = SynergyAccumulator()
        accumulator.observe(
            FIVE, residual=10.0, possessions=-3.0, seconds=60.0, date=DAY
        )
        assert accumulator.pair_value("a", "b", DAY) == 0.0

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="positive"):
            SynergyAccumulator(shrinkage_possessions=0.0)
        with pytest.raises(ValueError, match="positive"):
            SynergyAccumulator(half_life_days=-1.0)


class TestSharedMinutes:
    def test_the_shares_add_up_to_ten_pairs_on_the_floor(self):
        accumulator = SynergyAccumulator()
        accumulator.observe(
            FIVE, residual=0.0, possessions=100.0, seconds=600.0, date=DAY
        )
        minutes = expected_shared_minutes(accumulator, list(FIVE))
        assert sum(minutes.values()) == pytest.approx(PAIRS_ON_COURT * 48.0)

    def test_an_absence_raises_everyone_elses_share(self):
        """The mechanism that makes the weighting react to news."""
        accumulator = SynergyAccumulator()
        accumulator.observe(
            FIVE, residual=0.0, possessions=100.0, seconds=600.0, date=DAY
        )
        full = expected_shared_minutes(accumulator, list(FIVE))
        without = expected_shared_minutes(accumulator, ["a", "b", "c", "d"])
        assert without[("a", "b")] > full[("a", "b")]
        assert ("a", "e") not in without

    def test_no_shared_history_spreads_the_floor_evenly(self):
        accumulator = SynergyAccumulator()
        minutes = expected_shared_minutes(accumulator, list(FIVE))
        assert len(set(minutes.values())) == 1
        assert sum(minutes.values()) == pytest.approx(PAIRS_ON_COURT * 48.0)

    def test_fewer_than_two_players_has_no_pairs(self):
        assert expected_shared_minutes(SynergyAccumulator(), ["a"]) == {}


class TestTeamSynergy:
    def test_one_five_playing_the_whole_game_sums_its_ten_pairs(self):
        """Section 6.1's definition of a five's value, in the ratings' scale."""
        accumulator = SynergyAccumulator(shrinkage_possessions=0.001)
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=2.0, possessions=1000.0, seconds=600.0, date=DAY
        )
        value = team_synergy(accumulator, list(FIVE), DAY)
        # Ten pairs, each contributing a tenth: the five's own deviation, once.
        assert value == pytest.approx(2.0, rel=1e-2)

    def test_a_team_with_no_history_has_no_synergy(self):
        assert team_synergy(SynergyAccumulator(), list(FIVE), DAY) == pytest.approx(0.0)

    def test_losing_the_good_pair_lowers_the_synergy(self):
        accumulator = SynergyAccumulator(shrinkage_possessions=0.001)
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=5.0, possessions=1000.0, seconds=600.0, date=DAY
        )
        accumulator.observe(
            ("a", "b", "x", "y", "z"),
            residual=-5.0,
            possessions=1000.0,
            seconds=600.0,
            date=DAY,
        )
        with_all = team_synergy(accumulator, list(FIVE), DAY)
        without_e = team_synergy(accumulator, ["a", "b", "c", "d"], DAY)
        assert with_all != pytest.approx(without_e)


class TestExactFive:
    def test_the_exact_five_value_comes_back_with_its_evidence(self):
        """Section 6.1 wants both values emitted, not one chosen for the model."""
        accumulator = SynergyAccumulator(shrinkage_possessions=0.001)
        _seed_league(accumulator)
        accumulator.observe(
            FIVE, residual=4.0, possessions=1000.0, seconds=600.0, date=DAY
        )
        value, possessions = projected_five_synergy(accumulator, frozenset(FIVE), DAY)
        # The five's own deviation, undivided: it is a lineup-level quantity.
        assert value == pytest.approx(4.0, rel=1e-2)
        assert possessions == pytest.approx(1000.0, rel=1e-2)

    def test_a_five_that_never_played_together_says_so(self):
        accumulator = SynergyAccumulator()
        _seed_league(accumulator)
        value, possessions = projected_five_synergy(accumulator, frozenset(FIVE), DAY)
        assert value == 0.0
        assert possessions == 0.0

    def test_the_evidence_decays_like_everything_else(self):
        accumulator = SynergyAccumulator(half_life_days=180.0)
        accumulator.observe(
            FIVE, residual=4.0, possessions=1000.0, seconds=600.0, date=DAY
        )
        _, fresh = projected_five_synergy(accumulator, frozenset(FIVE), DAY)
        _, later = projected_five_synergy(
            accumulator, frozenset(FIVE), DAY + pd.Timedelta(days=180)
        )
        assert later == pytest.approx(fresh / 2.0, rel=1e-2)
