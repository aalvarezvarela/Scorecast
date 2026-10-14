import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.matchup_rating import (
    Penalties,
    defensive_ratings,
    player_teams,
    walk_forward_matchup_ratings,
)


def _pairs(defender_effect, team_effect=None, days=80, seed=0):
    """Attackers of team X face the defenders of team A or B each day; each
    attacker's points per partial possession follow his own quality, the
    defending team and his defenders' effects by guarding share."""
    rng = np.random.default_rng(seed)
    team_effect = team_effect or {}
    defenders = {team: [f"{team}{k}" for k in range(6)] for team in ("A", "B")}
    attackers = [f"x{k}" for k in range(10)]
    quality = dict(zip(attackers, rng.normal(0.2, 0.03, len(attackers)), strict=True))
    rows = []
    for day in range(days):
        team = "A" if day % 2 == 0 else "B"
        game = f"g{day}"
        for attacker in attackers:
            share = rng.dirichlet(np.ones(6))
            pp = 60.0 * share
            rate = quality[attacker] + team_effect.get(team, 0.0)
            rate += sum(
                s * defender_effect[d]
                for s, d in zip(share, defenders[team], strict=True)
            )
            points = rng.normal(rate * 60.0, 0.5)
            for s, d, p in zip(share, defenders[team], pp, strict=True):
                rows.append(
                    {
                        "game_id": game,
                        "game_date": pd.Timestamp("2018-01-01")
                        + pd.Timedelta(days=day),
                        "off_player_id": attacker,
                        "off_team_id": "X",
                        "def_player_id": d,
                        "def_team_id": team,
                        "attacker_is_home": day % 3 == 0,
                        "partial_possessions": p,
                        "player_points": points * s,
                    }
                )
    return pd.DataFrame(rows), defenders


def test_recovers_defender_effects_from_strictly_earlier_games():
    rng = np.random.default_rng(1)
    effects = {f"{t}{k}": rng.normal(0, 0.05) for t in "AB" for k in range(6)}
    pairs, _ = _pairs(effects)
    dates = sorted(pairs["game_date"].unique())
    (result,) = walk_forward_matchup_ratings(
        pairs,
        dates,
        [Penalties(attacker=10.0, team=1e9, defender=1.0)],
        half_life_days=1e9,
    )
    assert (result.fits["fit_max_game_date"] < result.fits["as_of_date"]).all()
    assert result.fits["as_of_date"].min() == dates[1]  # nothing to fit on day 0
    last = result.players.loc[result.players["as_of_date"].eq(dates[-1])]
    estimated = last.set_index("player_id")["defense"].dropna()
    true = pd.Series(effects).reindex(estimated.index)
    # Effects are identified up to each team's mean (the team term is pinned
    # at 0 here, the intercept is free): compare within team.
    for team in "AB":
        mine = estimated.index.str.startswith(team)
        e = estimated[mine] - estimated[mine].mean()
        t = true[mine] - true[mine].mean()
        assert np.corrcoef(e, t)[0, 1] > 0.95
    p = result.predictions
    assert (p["predicted"] - p["actual"]).pow(2).mean() < (
        p["predicted_without_defenders"] - p["actual"]
    ).pow(2).mean()


def test_team_term_takes_a_team_wide_shift():
    effects = {f"{t}{k}": 0.0 for t in "AB" for k in range(6)}
    pairs, defenders = _pairs(effects, team_effect={"A": 0.04, "B": -0.04})
    dates = sorted(pairs["game_date"].unique())
    (result,) = walk_forward_matchup_ratings(
        pairs,
        dates[-1:],
        [Penalties(attacker=10.0, team=10.0, defender=1000.0)],
        half_life_days=1e9,
    )
    teams = result.teams.set_index("team_id")["team"]
    assert teams["A"] - teams["B"] > 0.06
    assert result.players["defense"].abs().max() < 0.01
    on = player_teams(pairs, dates[-1:])
    ratings = defensive_ratings(result, on).set_index("player_id")
    # A's defenders allow more: negative d_rating (2_6 units, + = good).
    assert (ratings.loc[defenders["A"], "d_matchup"] < 0).all()
    assert (ratings.loc[defenders["B"], "d_matchup"] > 0).all()
    assert np.allclose(
        ratings["d_matchup"], ratings["d_team"] + ratings["d_individual"]
    )
