"""Expected guarding rates: shrunk, decayed pair history strictly before D."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.expected_guard import (
    COLUMNS,
    ExpectedGuardParams,
    expected_guard,
    position_rates,
    shrink,
)

CUTOFF = pd.Timestamp("2024-01-01")
#: Hard positions: guards g*, centers c*.
POSITIONS = pd.DataFrame(
    {
        "p_G": [1.0, 1.0, 0.0, 0.0],
        "p_F": [0.0, 0.0, 0.0, 0.0],
        "p_C": [0.0, 0.0, 1.0, 1.0],
    },
    index=pd.Index(["g1", "g2", "c1", "c2"], name="player_id"),
)


def _row(day, off, dfn, matchup, cofloor):
    return {
        "game_id": f"0022300{day.dayofyear:03d}",
        "game_date": day,
        "off_player_id": off,
        "def_player_id": dfn,
        "matchup_seconds": matchup,
        "cofloor_seconds": cofloor,
    }


def _data(rows):
    return PointInTimeData.from_frames(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pair_game=pd.DataFrame(rows)
    )


def _history():
    day = CUTOFF - pd.Timedelta(days=10)
    return [
        # Guards guard guards, centers guard centers.
        _row(day, "g1", "g2", 300.0, 1000.0),
        _row(day, "g1", "c2", 20.0, 1000.0),
        _row(day, "c1", "c2", 400.0, 1000.0),
        _row(day, "c1", "g2", 10.0, 1000.0),
    ]


def _pairs(*pairs):
    return pd.DataFrame(pairs, columns=["off_player_id", "def_player_id"])


def _guard(rows, pairs, **params):
    return expected_guard(
        _data(rows).as_of(CUTOFF),
        pairs,
        params=ExpectedGuardParams(**params),
        positions=POSITIONS,
    ).set_index(["off_player_id", "def_player_id"])


def test_position_prior_learns_who_guards_whom():
    rates = position_rates(
        pd.DataFrame(_history()).rename(
            columns={"cofloor_seconds": "cofloor_w", "matchup_seconds": "matchup_w"}
        ),
        POSITIONS,
    )
    assert rates.loc["G", "G"] == pytest.approx(0.3)
    assert rates.loc["C", "C"] == pytest.approx(0.4)
    assert rates.loc["G", "C"] == pytest.approx(0.02)


def test_a_pair_without_history_gets_its_position_prior():
    out = _guard(_history(), _pairs(("g2", "g1"), ("c2", "g1")), half_life_days=1e9)
    unseen = out.loc[("g2", "g1")]
    assert unseen.r_hat == pytest.approx(0.3) and unseen.r_prior == pytest.approx(0.3)
    assert unseen.prior_weight == 1.0 and not unseen.has_pair_history
    assert unseen.n_games == 0
    assert out.loc[("c2", "g1")].r_hat == pytest.approx(0.01)


def test_history_is_shrunk_towards_the_prior_by_k_seconds():
    rows = [*_history(), _row(CUTOFF - pd.Timedelta(days=5), "g1", "c2", 900, 3000)]
    out = _guard(rows, _pairs(("g1", "c2")), k=1000.0, half_life_days=1e9)
    row = out.loc[("g1", "c2")]
    # Position prior G attacker x C defender, then 4000 s of pair evidence.
    prior = (20 + 900) / (1000 + 3000)
    assert row.r_prior == pytest.approx(prior)
    assert row.r_hat == pytest.approx((920 + 1000 * prior) / (4000 + 1000))
    assert row.prior_weight == pytest.approx(0.2)
    assert row.n_games == 2 and row.has_pair_history
    assert row.hist_cofloor_seconds_raw == 4000


def test_decay_weights_matchup_and_cofloor_seconds_alike():
    old = CUTOFF - pd.Timedelta(days=365)
    rows = [_row(old, "g1", "g2", 300.0, 1000.0)]
    row = _guard(rows, _pairs(("g1", "g2")), half_life_days=365.0).iloc[0]
    assert row.hist_cofloor_seconds_raw == 1000.0
    assert row.hist_cofloor_seconds_decayed == pytest.approx(500.0)
    assert row.hist_matchup_seconds_decayed == pytest.approx(150.0)


def test_rows_on_or_after_the_cutoff_and_outside_the_window_are_ignored():
    rows = [
        *_history(),
        _row(CUTOFF, "g1", "g2", 999.0, 1000.0),  # the game being encoded
        _row(CUTOFF - pd.Timedelta(days=2000), "g1", "g2", 999.0, 1000.0),
    ]
    row = _guard(rows, _pairs(("g1", "g2")), half_life_days=1e9).iloc[0]
    assert row.n_games == 1
    assert row.hist_matchup_seconds_decayed == pytest.approx(300.0)


def test_rows_without_cofloor_time_are_not_history():
    rows = [*_history(), _row(CUTOFF - pd.Timedelta(days=3), "g1", "g2", 50.0, 0.0)]
    assert _guard(rows, _pairs(("g1", "g2"))).iloc[0].n_games == 1


def test_shrink_recomputes_r_hat_for_another_k():
    rows = [*_history(), _row(CUTOFF - pd.Timedelta(days=5), "g1", "c2", 900, 3000)]
    at_600 = _guard(rows, _pairs(("g1", "c2")), k=600.0)
    at_50 = _guard(rows, _pairs(("g1", "c2")), k=50.0)
    r_hat, weight = shrink(
        at_600.hist_matchup_seconds_decayed,
        at_600.hist_cofloor_seconds_decayed,
        at_600.r_prior,
        50.0,
    )
    assert r_hat == pytest.approx(at_50.r_hat.to_numpy())
    assert weight == pytest.approx(at_50.prior_weight.to_numpy())


def test_extra_columns_are_carried_and_the_schema_is_fixed():
    pairs = _pairs(("g1", "g2")).assign(game_id="0022300999")
    out = expected_guard(_data(_history()).as_of(CUTOFF), pairs, positions=POSITIONS)
    assert list(out.columns) == ["game_id", *COLUMNS]
    assert np.isfinite(out.r_hat).all()


def test_no_history_at_all_leaves_the_rate_unknown():
    out = _guard([], _pairs(("g1", "g2")))
    assert np.isnan(out.iloc[0].r_hat) and np.isnan(out.iloc[0].r_prior)
    assert out.iloc[0].prior_weight == 1.0
