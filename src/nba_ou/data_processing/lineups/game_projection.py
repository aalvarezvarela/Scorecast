"""Phase F v1: a game total projected from minutes and player lineup ratings.

Each team's available players contribute their phase-C ratings in proportion to
their projected minutes; the two teams' pace ratings set the possession count,
and possessions times efficiency give the points (``docs/lineup_projection_plan.md``
section 7.1)::

    off_T  = sum_i (min_i / 48) * o_rating_i
    def_T  = sum_i (min_i / 48) * d_rating_i
    pace_T = sum_i (min_i / 48) * pace_rating_i

    poss   = (league_pace + pace_H + pace_A) * game_minutes / 48
    ptsH   = poss / 100 * (league_ortg + off_H - def_A)
    ptsA   = poss / 100 * (league_ortg + off_A - def_H)

Each team's minutes sum to 240, so each ``sum_i (min_i / 48)`` is 5 -- the
average on-court five's summed rating, the same scale the ratings were fitted
on.

Players whose availability is genuinely uncertain generate **scenarios**: each
combination of them playing or sitting is projected separately and weighted by
its probability. The spread across scenarios is as useful as the mean, because
it says how much tonight's total depends on news that has not settled.

Two deliberate departures from the plan's section 7.1:

**Home court splits, it does not add.** The plan writes ``+ home_court`` on the
home team alone, which would raise every projected *total* by that amount.
Venue cannot move a total in aggregate -- every game has one home side and one
away side -- so the term is applied as ``+h/2`` and ``-h/2``. That keeps the
margin right and leaves the total, which is what this project predicts,
untouched.

**Synergy is absent.** It is phase D and does not exist yet. The projection is
additive in players for now, and section 7.1b's shared-minutes weighting plugs
in at ``team_aggregate`` when it does.

Minutes are an input, not a model: phase E will supply them. Until it does,
``allocate_minutes`` implements the fallback the plan specifies for a failed
go/no-go E -- each player's recent average, rescaled to the team's 240.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

#: Regulation minutes for one team (5 players x 48).
TEAM_MINUTES = 240.0

#: A player this likely to sit, or this likely to play, is treated as settled
#: rather than spawning a scenario. The plan's section 7.1 bounds.
SCENARIO_LOW, SCENARIO_HIGH = 0.1, 0.9

#: Uncertain players enumerated exhaustively per team; 2**3 = 8 scenarios.
MAX_ENUMERATED_UNCERTAIN = 3


@dataclass(frozen=True)
class PlayerNight:
    """One player's pre-game state: how much he plays and whether he plays."""

    player_id: str
    base_minutes: float
    p_out: float = 0.0


@dataclass(frozen=True)
class TeamRatings:
    """The three per-player ratings, as of the target date."""

    offense: dict[str, float]
    defense: dict[str, float]
    pace: dict[str, float]

    def get(self, player_id: str) -> tuple[float, float, float]:
        """A player with no fitted rating is league average, not missing."""
        return (
            self.offense.get(player_id, 0.0),
            self.defense.get(player_id, 0.0),
            self.pace.get(player_id, 0.0),
        )


def allocate_minutes(
    players: list[PlayerNight],
    sitting: frozenset[str] = frozenset(),
    team_minutes: float = TEAM_MINUTES,
) -> dict[str, float]:
    """Spread the team's minutes over whoever is playing tonight.

    Each available player keeps his share of the available players' recent
    minutes, rescaled so the team sums to ``team_minutes``. That is what makes
    an absence redistribute rather than vanish: when a 34-minute starter sits,
    his minutes are taken up by the others in proportion, which is the whole
    mechanism this projection needs to react to news.
    """
    available = {
        player.player_id: max(0.0, player.base_minutes)
        for player in players
        if player.player_id not in sitting
    }
    total = sum(available.values())
    if total <= 0:
        return {}
    scale = team_minutes / total
    return {player: minutes * scale for player, minutes in available.items()}


def team_aggregate(
    minutes: dict[str, float],
    ratings: TeamRatings,
) -> tuple[float, float, float]:
    """Minutes-weighted offense, defense and pace for one team."""
    off = dff = pace = 0.0
    for player, played in minutes.items():
        weight = played / 48.0
        player_off, player_def, player_pace = ratings.get(player)
        off += weight * player_off
        dff += weight * player_def
        pace += weight * player_pace
    return off, dff, pace


def project_totals(
    home: tuple[float, float, float],
    away: tuple[float, float, float],
    league_ortg: float,
    league_pace: float,
    home_court: float = 0.0,
    game_minutes: float = 48.0,
    total_offset: float = 0.0,
) -> dict[str, float]:
    """Turn two teams' aggregates into possessions and points.

    ``total_offset`` calibrates the league scoring level and must be estimated
    walk-forward by the caller, from games strictly earlier than this one.

    It is not a fudge factor. The ratings and their ``league_ortg`` intercept
    are fitted on stints, whose possessions are *estimated*
    (``FGA + 0.44*FTA - OREB + TOV``, with unattributed team rebounds left
    unclassified) rather than counted. That estimate runs high, so points per
    100 come out low: measured on 2024-25, the fitted intercept averages 111.37
    against a game-level implied 113.39, and projecting straight from it
    under-predicts every total by about 2.7 points. The offset is where that
    known measurement gap is corrected, in the open, instead of being absorbed
    into the ratings where it would distort the player values themselves.
    """
    home_off, home_def, home_pace = home
    away_off, away_def, away_pace = away
    possessions = (league_pace + home_pace + away_pace) * game_minutes / 48.0
    home_points = possessions / 100.0 * (league_ortg + home_off - away_def)
    away_points = possessions / 100.0 * (league_ortg + away_off - home_def)
    # Split, never add: see the module docstring.
    home_points += home_court / 2.0 + total_offset / 2.0
    away_points += -home_court / 2.0 + total_offset / 2.0
    return {
        "home_points": home_points,
        "away_points": away_points,
        "total": home_points + away_points,
        "possessions": possessions,
    }


def enumerate_scenarios(
    players: list[PlayerNight],
    max_enumerated: int = MAX_ENUMERATED_UNCERTAIN,
) -> list[tuple[frozenset[str], float]]:
    """``(who sits, how likely)`` for each combination worth projecting.

    Players who are all but certain either way are folded into every scenario
    rather than doubling the count for a branch worth 2% of the weight. When
    more than ``max_enumerated`` are genuinely uncertain, the most uncertain
    ones are enumerated and the rest are taken at their most likely state, so
    the scenario count stays bounded.
    """
    certain_out = frozenset(
        player.player_id for player in players if player.p_out >= SCENARIO_HIGH
    )
    uncertain = [
        player for player in players if SCENARIO_LOW < player.p_out < SCENARIO_HIGH
    ]
    # Most uncertain first: 0.5 carries more information than 0.15.
    uncertain.sort(key=lambda player: (abs(player.p_out - 0.5), player.player_id))
    enumerated, assumed = uncertain[:max_enumerated], uncertain[max_enumerated:]
    base = certain_out | frozenset(
        player.player_id for player in assumed if player.p_out >= 0.5
    )
    scenarios = []
    for states in product([False, True], repeat=len(enumerated)):
        sitting = set(base)
        weight = 1.0
        for player, sits in zip(enumerated, states, strict=True):
            weight *= player.p_out if sits else 1.0 - player.p_out
            if sits:
                sitting.add(player.player_id)
        scenarios.append((frozenset(sitting), weight))
    return scenarios


#: A team's starters are its top five by full-health minutes; everyone below
#: them is the bench whose absorbed minutes are measured on their own.
STARTERS = 5


def bench_replacement(
    players: list[PlayerNight],
    ratings: TeamRatings,
    max_enumerated: int = MAX_ENUMERATED_UNCERTAIN,
) -> tuple[float, float, float]:
    """Change in the healthy bench's (offense, defense, pace) from tonight's absences.

    The bench is fixed at full health -- everyone below the top ``STARTERS``
    by minutes -- so a reserve promoted into the starting five by an injury
    still counts as bench. Only the bench players who *play* in a scenario
    contribute, with ``(tonight's minutes - full-health minutes) / 48`` times
    their rating; a bench player who sits is an absentee, and his loss belongs
    to the absence itself, not to its replacement. Scenario-weighted like
    :func:`project_game`, with empty scenarios dropped the same way.
    """
    healthy = allocate_minutes(players)
    ranked = sorted(healthy, key=lambda player: (-healthy[player], player))
    bench = ranked[STARTERS:]
    change = np.zeros(3)
    weight_sum = 0.0
    for sitting, weight in enumerate_scenarios(players, max_enumerated):
        minutes = allocate_minutes(players, sitting)
        if not minutes:
            continue
        delta = {p: minutes[p] - healthy[p] for p in bench if p in minutes}
        change += weight * np.asarray(team_aggregate(delta, ratings))
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0, 0.0, 0.0
    off, dff, pace = change / weight_sum
    return float(off), float(dff), float(pace)


_AGGREGATE_KEYS = (
    "home_off",
    "home_def",
    "home_pace",
    "away_off",
    "away_def",
    "away_pace",
)


def project_game(
    home_players: list[PlayerNight],
    away_players: list[PlayerNight],
    home_ratings: TeamRatings,
    away_ratings: TeamRatings,
    league_ortg: float,
    league_pace: float,
    home_court: float = 0.0,
    game_minutes: float = 48.0,
    total_offset: float = 0.0,
    max_enumerated: int = MAX_ENUMERATED_UNCERTAIN,
) -> dict[str, float]:
    """Project one game over its availability scenarios.

    Returns the probability-weighted mean of each quantity, plus ``total_sd``,
    the weighted standard deviation of the total across scenarios: the
    projection's own statement of how unsettled tonight's news is. The mean
    team aggregates (``home_off``, ``home_def``, ``home_pace`` and the away
    three) come along so a caller can say *which* channel a change runs through.
    """
    home_scenarios = enumerate_scenarios(home_players, max_enumerated)
    away_scenarios = enumerate_scenarios(away_players, max_enumerated)
    totals, weights = [], []
    summed = dict.fromkeys(
        ("home_points", "away_points", "possessions", *_AGGREGATE_KEYS), 0.0
    )
    weight_sum = 0.0
    for home_sitting, home_weight in home_scenarios:
        home_minutes = allocate_minutes(home_players, home_sitting)
        # An empty allocation aggregates to (0, 0, 0), which is exactly what a
        # perfectly league-average team looks like. Projecting it would emit a
        # confident league-average total for a team we know nothing about, so
        # the scenario is dropped and its weight left out of the renormalisation.
        if not home_minutes:
            continue
        home_aggregate = team_aggregate(home_minutes, home_ratings)
        for away_sitting, away_weight in away_scenarios:
            away_minutes = allocate_minutes(away_players, away_sitting)
            if not away_minutes:
                continue
            away_aggregate = team_aggregate(away_minutes, away_ratings)
            projection = project_totals(
                home_aggregate,
                away_aggregate,
                league_ortg,
                league_pace,
                home_court,
                game_minutes,
                total_offset,
            )
            weight = home_weight * away_weight
            projection |= dict(
                zip(_AGGREGATE_KEYS, home_aggregate + away_aggregate, strict=True)
            )
            for key in summed:
                summed[key] += weight * projection[key]
            totals.append(projection["total"])
            weights.append(weight)
            weight_sum += weight
    if weight_sum <= 0:
        return {}
    mean = {key: value / weight_sum for key, value in summed.items()}
    mean["total"] = mean["home_points"] + mean["away_points"]
    values = np.asarray(totals, dtype=float)
    probabilities = np.asarray(weights, dtype=float) / weight_sum
    variance = float(probabilities @ (values - mean["total"]) ** 2)
    mean["total_sd"] = float(np.sqrt(max(0.0, variance)))
    mean["scenarios"] = float(len(totals))
    return mean


def project_with_counterfactual(
    home_players: list[PlayerNight],
    away_players: list[PlayerNight],
    home_ratings: TeamRatings,
    away_ratings: TeamRatings,
    league_ortg: float,
    league_pace: float,
    home_court: float = 0.0,
    game_minutes: float = 48.0,
    total_offset: float = 0.0,
) -> dict[str, float]:
    """Project tonight, and again with the same roster at full health.

    The difference is what the absences are worth in points and possessions,
    which is the quantity to compare against the market's reaction rather than
    against the outcome (plan section 8.1, group A). Projecting both from the
    same roster and the same ratings means everything except availability
    cancels.

    The difference is also split two ways, because the parts are priced
    differently (``docs/lineup_projection_plan.md`` section 8.6):

    - **by side**: each team's absences alone, the other at full health;
    - **by channel**: the points the absentees would have *scored* (offense),
      would have *prevented* (defense), and the rest, which is the change in
      possessions (pace). Offense and defense are valued at full-health
      possessions, so the three add up to the total impact exactly.

    And one slice of it on its own: ``absence_impact_bench_replacement``, the
    defense + pace points carried by the minutes the healthy **bench** absorbs
    (:func:`bench_replacement`). The market prices an absence by the player
    who is out; who plays his minutes is priced worse, and the bench's share
    worst (plan section 8.6, bench probe). Valued at full-health possessions
    and points per possession, the same linearisation as the channels above.
    """
    actual = project_game(
        home_players,
        away_players,
        home_ratings,
        away_ratings,
        league_ortg,
        league_pace,
        home_court,
        game_minutes,
        total_offset,
    )
    healthy_home = [
        PlayerNight(player.player_id, player.base_minutes, 0.0)
        for player in home_players
    ]
    healthy_away = [
        PlayerNight(player.player_id, player.base_minutes, 0.0)
        for player in away_players
    ]
    healthy = project_game(
        healthy_home,
        healthy_away,
        home_ratings,
        away_ratings,
        league_ortg,
        league_pace,
        home_court,
        game_minutes,
        total_offset,
    )
    if not actual or not healthy:
        return actual
    context = (
        home_ratings,
        away_ratings,
        league_ortg,
        league_pace,
        home_court,
        game_minutes,
        total_offset,
    )
    home_only = project_game(home_players, healthy_away, *context)
    away_only = project_game(healthy_home, away_players, *context)
    impact = actual["total"] - healthy["total"]
    per_100 = healthy["possessions"] / 100.0
    offense = per_100 * sum(
        actual[key] - healthy[key] for key in ("home_off", "away_off")
    )
    # A defensive rating is points prevented, so losing it adds points.
    defense = -per_100 * sum(
        actual[key] - healthy[key] for key in ("home_def", "away_def")
    )
    _, bench_def, bench_pace = np.add(
        bench_replacement(home_players, home_ratings),
        bench_replacement(away_players, away_ratings),
    )
    # d(total)/d(possessions) at full health: both teams' points per 100.
    points_per_possession = (
        2 * league_ortg
        + healthy["home_off"]
        + healthy["away_off"]
        - healthy["home_def"]
        - healthy["away_def"]
    ) / 100.0
    bench_points = -per_100 * bench_def + (
        bench_pace * game_minutes / 48.0 * points_per_possession
    )
    return actual | {
        "healthy_total": healthy["total"],
        "healthy_possessions": healthy["possessions"],
        "absence_impact_points": impact,
        "absence_impact_possessions": actual["possessions"] - healthy["possessions"],
        "absence_impact_offense": offense,
        "absence_impact_defense": defense,
        "absence_impact_pace": impact - offense - defense,
        "absence_impact_bench_replacement": float(bench_points),
        "absence_impact_home": home_only["total"] - healthy["total"],
        "absence_impact_away": away_only["total"] - healthy["total"],
        "margin": actual["home_points"] - actual["away_points"],
        "absence_impact_margin": (actual["home_points"] - actual["away_points"])
        - (healthy["home_points"] - healthy["away_points"]),
    }
