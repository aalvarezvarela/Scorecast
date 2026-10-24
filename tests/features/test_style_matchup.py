"""Three-point matchup features: traits, decay, the shift, and the temporal contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.game_projection import PlayerNight
from nba_ou.data_processing.lineups.style_matchup import (
    STYLE_FEATURE_COLUMNS,
    StyleTraits,
    _MonthlyModel,
    build_style_matchup_features,
    stint_directions,
)

from tests.support.style_matchup import (
    DATES,
    _stints,
)


def _ids(team):
    base = 100 if team == "H" else 200
    return [str(base + i) for i in range(7)]


def _nights(game_id, out=()):
    return {
        (game_id, "H"): [
            PlayerNight(p, 30.0, 1.0 if p in out else 0.0) for p in _ids("H")
        ],
        (game_id, "A"): [PlayerNight(p, 30.0) for p in _ids("A")],
    }


TARGET_DATE = "2024-11-25"
GAMES = pd.DataFrame(
    {
        "GAME_ID": ["0099999999"],
        "GAME_DATE": [TARGET_DATE],
        "HOME_TEAM_ID": ["H"],
        "AWAY_TEAM_ID": ["A"],
    }
)


def _build(stints, nights, **kw):
    return build_style_matchup_features(
        stints, GAMES, nights, neighbours=50, min_pool=200, **kw
    ).set_index("GAME_ID")


def test_neighbor_residual_can_be_a_noninteraction_nonlinearity():
    # A nonlinear offense-only rate leaves a residual after a linear additive
    # baseline, even though defense has no effect on the target.
    offense = np.linspace(-1, 1, 101)
    defense = np.tile([-1.0, 1.0], 51)[:101]
    vectors = np.column_stack([offense, defense])
    rates = 0.2 + 0.1 * offense**2
    model = _MonthlyModel(vectors, rates, np.ones(len(rates)), neighbours=5)
    assert model.neighbor_residual(np.array([1.0, 1.0])) > 0.02
    assert model.neighbor_residual(np.array([1.0, -1.0])) > 0.02


class TestStyleTraits:
    def test_no_evidence_is_league_average(self):
        traits = StyleTraits()
        traits.update(stint_directions(_stints(DATES[:3])), pd.Timestamp(DATES[2]))
        assert traits.trait("nobody", "o", "fg3a_rate") == 0.0

    def test_a_high_volume_shooter_gets_a_positive_trait(self):
        traits = StyleTraits()
        dirs = stint_directions(_stints(DATES[:30], shooter="h0", boost=4))
        for date, day in dirs.groupby("game_date"):
            traits.advance(date)
            traits.update(day, date)
        # Player "h0" is id 100 in the stints.
        assert traits.trait("100", "o", "fg3a_rate") > 0.02

    def test_evidence_decays_toward_the_league(self):
        traits = StyleTraits(half_life_days=30)
        dirs = stint_directions(_stints(DATES[:10], shooter="h0", boost=4))
        for date, day in dirs.groupby("game_date"):
            traits.advance(date)
            traits.update(day, date)
        fresh = traits.trait("100", "o", "fg3a_rate")
        traits.advance(pd.Timestamp(DATES[9]) + pd.Timedelta(days=300))
        assert 0 < traits.trait("100", "o", "fg3a_rate") < fresh


class TestFeatures:
    def test_columns_and_no_shift_without_absences(self):
        out = _build(_stints(DATES), _nights("0099999999"))
        assert list(out.columns) == list(STYLE_FEATURE_COLUMNS)
        assert out.loc["0099999999", "LU_ABSENCE_SHIFT_FG3A_RATE_BEFORE"] == 0.0

    def test_sitting_the_shooter_lowers_the_projected_rate(self):
        stints = _stints(DATES, shooter="h0", boost=4)
        healthy = _build(stints, _nights("0099999999"))
        without = _build(stints, _nights("0099999999", out=("100",)))
        assert without.loc["0099999999", "LU_ABSENCE_SHIFT_FG3A_RATE_BEFORE"] < 0
        assert (
            without.loc["0099999999", "LU_PROJ_FG3A_RATE_BEFORE_TEAM_HOME"]
            < healthy.loc["0099999999", "LU_PROJ_FG3A_RATE_BEFORE_TEAM_HOME"]
        )


class TestTemporalContract:
    def test_same_day_and_later_stints_are_not_read(self):
        stints = _stints(DATES)
        base = _build(stints, _nights("0099999999"))
        changed = stints.copy()
        late = changed.game_date >= pd.Timestamp(TARGET_DATE)
        assert late.any()
        changed.loc[late, ["home_fg3a", "away_fg3a"]] = changed.loc[
            late, ["home_fga", "away_fga"]
        ].to_numpy()
        pd.testing.assert_frame_equal(_build(changed, _nights("0099999999")), base)

    def test_the_perturbation_would_be_seen_if_it_were_read(self):
        """Guards the test above against passing vacuously."""
        stints = _stints(DATES)
        base = _build(stints, _nights("0099999999"))
        changed = stints.copy()
        early = changed.game_date < pd.Timestamp(TARGET_DATE)
        changed.loc[early, ["home_fg3a", "away_fg3a"]] = changed.loc[
            early, ["home_fga", "away_fga"]
        ].to_numpy()
        after = _build(changed, _nights("0099999999"))
        assert not np.allclose(after.to_numpy(), base.to_numpy())

    def test_no_model_until_the_pool_is_large_enough(self):
        out = build_style_matchup_features(
            _stints(DATES[:5]), GAMES, _nights("0099999999"), min_pool=10**6
        )
        assert out.empty
        assert list(out.columns) == ["GAME_ID", *STYLE_FEATURE_COLUMNS]


@pytest.mark.parametrize("column", STYLE_FEATURE_COLUMNS)
def test_every_column_is_leakage_gate_shaped(column):
    assert column.startswith("LU_") and "_BEFORE" in column
