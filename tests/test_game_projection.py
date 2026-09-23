"""Phase F v1: minutes allocation, scenarios, totals and the counterfactual."""

from __future__ import annotations

import numpy as np
import pytest
from nba_ou.data_processing.lineups.game_projection import (
    TEAM_MINUTES,
    PlayerNight,
    TeamRatings,
    allocate_minutes,
    enumerate_scenarios,
    project_game,
    project_totals,
    project_with_counterfactual,
    team_aggregate,
)


def _roster(n=10, minutes=24.0, p_out=0.0, prefix="h"):
    return [PlayerNight(f"{prefix}{i}", minutes, p_out) for i in range(n)]


def _flat_ratings(value=0.0, players=()):
    return TeamRatings(
        offense=dict.fromkeys(players, value),
        defense=dict.fromkeys(players, value),
        pace=dict.fromkeys(players, value),
    )


class TestAllocateMinutes:
    def test_minutes_always_sum_to_the_team_total(self):
        minutes = allocate_minutes(_roster(n=9, minutes=17.0))
        assert sum(minutes.values()) == pytest.approx(TEAM_MINUTES)

    def test_an_absence_is_redistributed_not_lost(self):
        players = [
            PlayerNight("a", 36.0),
            PlayerNight("b", 24.0),
            PlayerNight("c", 12.0),
        ]
        full = allocate_minutes(players)
        without_a = allocate_minutes(players, frozenset({"a"}))
        assert sum(without_a.values()) == pytest.approx(TEAM_MINUTES)
        assert "a" not in without_a
        # b and c keep their 2:1 ratio and both gain.
        assert without_a["b"] > full["b"]
        assert without_a["c"] > full["c"]
        assert without_a["b"] / without_a["c"] == pytest.approx(2.0)

    def test_everyone_sitting_returns_nothing_rather_than_dividing_by_zero(self):
        players = _roster(n=3)
        assert allocate_minutes(players, frozenset(p.player_id for p in players)) == {}


class TestTeamAggregate:
    def test_a_full_team_weighs_exactly_five_players(self):
        """Minutes sum to 240, so the weights sum to 5 -- an on-court five."""
        players = _roster(n=10, minutes=24.0)
        ratings = _flat_ratings(1.0, [p.player_id for p in players])
        off, dff, pace = team_aggregate(allocate_minutes(players), ratings)
        assert off == pytest.approx(5.0)
        assert dff == pytest.approx(5.0)
        assert pace == pytest.approx(5.0)

    def test_an_unrated_player_is_league_average_not_missing(self):
        players = _roster(n=10)
        ratings = _flat_ratings(1.0, ["h0"])
        off, _, _ = team_aggregate(allocate_minutes(players), ratings)
        # Only h0 is rated, holding a tenth of the minutes: 0.5 of the 5.
        assert off == pytest.approx(0.5)


class TestProjectTotals:
    def test_league_average_teams_reproduce_the_league(self):
        result = project_totals(
            (0, 0, 0), (0, 0, 0), league_ortg=112.0, league_pace=100.0
        )
        assert result["possessions"] == pytest.approx(100.0)
        assert result["total"] == pytest.approx(224.0)

    def test_home_court_splits_the_points_and_leaves_the_total_alone(self):
        """Venue cannot move a total in aggregate; it only decides the margin."""
        neutral = project_totals((0, 0, 0), (0, 0, 0), 112.0, 100.0, home_court=0.0)
        home = project_totals((0, 0, 0), (0, 0, 0), 112.0, 100.0, home_court=3.0)
        assert home["total"] == pytest.approx(neutral["total"])
        assert home["home_points"] - home["away_points"] == pytest.approx(3.0)

    def test_pace_ratings_move_possessions_and_therefore_the_total(self):
        slow = project_totals((0, 0, -2.0), (0, 0, -2.0), 112.0, 100.0)
        fast = project_totals((0, 0, 2.0), (0, 0, 2.0), 112.0, 100.0)
        assert slow["possessions"] == pytest.approx(96.0)
        assert fast["possessions"] == pytest.approx(104.0)
        assert fast["total"] > slow["total"]

    def test_defence_subtracts_from_the_opponent_not_from_itself(self):
        result = project_totals((0, 4.0, 0), (0, 0, 0), 112.0, 100.0)
        # The home defence is 4 better, so the AWAY team scores 4 fewer per 100.
        assert result["away_points"] == pytest.approx(108.0)
        assert result["home_points"] == pytest.approx(112.0)

    def test_overtime_lengthens_the_game_and_the_total(self):
        regulation = project_totals((0, 0, 0), (0, 0, 0), 112.0, 100.0)
        overtime = project_totals((0, 0, 0), (0, 0, 0), 112.0, 100.0, game_minutes=53.0)
        assert overtime["total"] > regulation["total"]
        assert overtime["possessions"] == pytest.approx(100.0 * 53.0 / 48.0)


class TestScenarios:
    def test_a_settled_roster_is_a_single_scenario(self):
        scenarios = enumerate_scenarios(_roster(n=8, p_out=0.0))
        assert len(scenarios) == 1
        assert scenarios[0] == (frozenset(), 1.0)

    def test_a_near_certain_absence_does_not_spawn_a_branch(self):
        players = [PlayerNight("a", 30.0, 0.97), PlayerNight("b", 30.0, 0.0)]
        scenarios = enumerate_scenarios(players)
        assert len(scenarios) == 1
        assert scenarios[0][0] == frozenset({"a"})

    def test_one_doubtful_player_gives_two_weighted_scenarios(self):
        players = [PlayerNight("a", 30.0, 0.4), PlayerNight("b", 30.0, 0.0)]
        scenarios = enumerate_scenarios(players)
        assert len(scenarios) == 2
        assert sum(weight for _, weight in scenarios) == pytest.approx(1.0)
        weights = {sitting: weight for sitting, weight in scenarios}
        assert weights[frozenset({"a"})] == pytest.approx(0.4)
        assert weights[frozenset()] == pytest.approx(0.6)

    def test_scenario_count_is_bounded_and_weights_still_sum_to_one(self):
        players = [PlayerNight(f"p{i}", 20.0, 0.5) for i in range(6)]
        scenarios = enumerate_scenarios(players)
        assert len(scenarios) == 2**3
        assert sum(weight for _, weight in scenarios) == pytest.approx(1.0)

    def test_the_most_uncertain_players_are_the_ones_enumerated(self):
        players = [
            PlayerNight("coinflip", 20.0, 0.5),
            PlayerNight("likely", 20.0, 0.15),
            PlayerNight("probable", 20.0, 0.85),
            PlayerNight("other", 20.0, 0.2),
        ]
        branched = {
            player
            for sitting, _ in enumerate_scenarios(players, max_enumerated=1)
            for player in sitting
        }
        assert "coinflip" in branched


class TestProjectGame:
    def test_a_settled_game_has_no_scenario_spread(self):
        home, away = _roster(prefix="h"), _roster(prefix="a")
        ratings = _flat_ratings(0.0, [p.player_id for p in home + away])
        result = project_game(home, away, ratings, ratings, 112.0, 100.0)
        assert result["total"] == pytest.approx(224.0)
        assert result["total_sd"] == pytest.approx(0.0)
        assert result["scenarios"] == 1

    def test_an_uncertain_star_widens_the_spread(self):
        home = [PlayerNight("star", 36.0, 0.5)] + _roster(n=8, minutes=25.5, prefix="h")
        away = _roster(prefix="a")
        ratings = TeamRatings(
            offense={"star": 6.0}, defense={"star": 0.0}, pace={"star": 0.0}
        )
        result = project_game(home, away, ratings, ratings, 112.0, 100.0)
        assert result["total_sd"] > 1.0
        assert result["scenarios"] == 2

    def test_the_mean_sits_between_the_two_scenarios(self):
        home = [PlayerNight("star", 36.0, 0.5)] + _roster(n=8, minutes=25.5, prefix="h")
        away = _roster(prefix="a")
        ratings = TeamRatings(
            offense={"star": 6.0}, defense={"star": 0.0}, pace={"star": 0.0}
        )
        playing = project_game(
            [PlayerNight(p.player_id, p.base_minutes, 0.0) for p in home],
            away,
            ratings,
            ratings,
            112.0,
            100.0,
        )
        sitting = project_game(
            [
                PlayerNight(
                    p.player_id, p.base_minutes, 1.0 if p.player_id == "star" else 0.0
                )
                for p in home
            ],
            away,
            ratings,
            ratings,
            112.0,
            100.0,
        )
        blended = project_game(home, away, ratings, ratings, 112.0, 100.0)
        assert sitting["total"] < blended["total"] < playing["total"]


class TestCounterfactual:
    def test_a_healthy_roster_has_no_absence_impact(self):
        home, away = _roster(prefix="h"), _roster(prefix="a")
        ratings = _flat_ratings(1.0, [p.player_id for p in home + away])
        result = project_with_counterfactual(home, away, ratings, ratings, 112.0, 100.0)
        assert result["absence_impact_points"] == pytest.approx(0.0)
        assert result["absence_impact_possessions"] == pytest.approx(0.0)

    def test_losing_a_scorer_lowers_the_projection_against_full_health(self):
        home = [PlayerNight("star", 36.0, 1.0)] + _roster(n=8, minutes=25.5, prefix="h")
        away = _roster(prefix="a")
        ratings = TeamRatings(
            offense={"star": 8.0}, defense={"star": 0.0}, pace={"star": 0.0}
        )
        result = project_with_counterfactual(home, away, ratings, ratings, 112.0, 100.0)
        assert result["absence_impact_points"] < -1.0
        assert result["healthy_total"] > result["total"]

    def test_losing_a_fast_player_shows_up_in_possessions(self):
        home = [PlayerNight("runner", 36.0, 1.0)] + _roster(
            n=8, minutes=25.5, prefix="h"
        )
        away = _roster(prefix="a")
        ratings = TeamRatings(
            offense={"runner": 0.0}, defense={"runner": 0.0}, pace={"runner": 3.0}
        )
        result = project_with_counterfactual(home, away, ratings, ratings, 112.0, 100.0)
        assert result["absence_impact_possessions"] < 0.0

    def test_the_impact_is_measured_against_the_same_roster_and_ratings(self):
        """Everything except availability must cancel, including bench depth."""
        home = [PlayerNight("star", 36.0, 1.0)] + _roster(n=8, minutes=25.5, prefix="h")
        away = _roster(prefix="a")
        strong = TeamRatings(offense={"star": 8.0, "h0": 4.0}, defense={}, pace={})
        weak = TeamRatings(offense={"star": 8.0, "h0": -4.0}, defense={}, pace={})
        deep = project_with_counterfactual(home, away, strong, strong, 112.0, 100.0)
        thin = project_with_counterfactual(home, away, weak, weak, 112.0, 100.0)
        # A better replacement absorbs more of the loss.
        assert deep["absence_impact_points"] > thin["absence_impact_points"]


def test_no_available_players_returns_nothing_rather_than_a_number():
    home = [PlayerNight("a", 30.0, 1.0)]
    away = _roster(prefix="a")
    ratings = _flat_ratings(0.0, ["a"])
    assert project_game(home, away, ratings, ratings, 112.0, 100.0) == {}


def test_the_projection_is_deterministic():
    home, away = _roster(n=9, p_out=0.5, prefix="h"), _roster(n=9, prefix="a")
    ratings = _flat_ratings(0.5, [p.player_id for p in home + away])
    first = project_game(home, away, ratings, ratings, 112.0, 100.0)
    second = project_game(home, away, ratings, ratings, 112.0, 100.0)
    assert first == second
    assert not np.isnan(first["total"])


def test_a_scenario_that_empties_a_team_is_dropped_not_averaged():
    """An empty team aggregates to (0,0,0), which reads as league average."""
    home = [PlayerNight("a", 30.0, 0.5), PlayerNight("b", 30.0, 0.5)]
    away = _roster(prefix="a")
    ratings = _flat_ratings(0.0, ["a", "b"])
    result = project_game(home, away, ratings, ratings, 112.0, 100.0)
    # Four scenarios, but the both-out one cannot be projected.
    assert result["scenarios"] == 3
    assert result["total"] == pytest.approx(224.0)
