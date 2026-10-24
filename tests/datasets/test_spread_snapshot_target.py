"""The intermediate spread target must be per-snapshot, never per-game.

The failure this guards against has no symptom: joining the anchor spread on
GAME_ID alone gives every snapshot of a game the same (closing) line, the frame
still trains, and the reported numbers look ordinary. It also destroys the entire
point of the intermediate dataset -- a model told the closing line at the 720
minute horizon is being handed information it could not have had.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.config.market_columns import (
    SPREAD_ERROR_COL,
    home_margin,
    spread_error,
    spread_line_home_from_implied_margin,
)
from nba_ou.create_training_data.create_intermediate_line_df import (
    _resolve_target_spread_line,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)


def _panel(rows):
    return pd.DataFrame(
        rows, columns=["game_id", "snapshot_minutes", "market", "book", "norm_line"]
    )


def test_resolve_target_spread_line_is_per_snapshot():
    """One game, five horizons, five different lines -> five different rows."""
    panel = _panel(
        [
            ("G1", 720, MARKET_SPREAD, "bet365", 4.0),
            ("G1", 360, MARKET_SPREAD, "bet365", 4.5),
            ("G1", 120, MARKET_SPREAD, "bet365", 5.5),
            ("G1", 60, MARKET_SPREAD, "bet365", 6.0),
            ("G1", 30, MARKET_SPREAD, "bet365", 6.5),
        ]
    )
    resolved = _resolve_target_spread_line(panel, anchor="bet365")
    assert list(resolved.columns) == ["game_id", "snapshot_minutes", "target_spread_line"]
    assert resolved["target_spread_line"].tolist() == [4.0, 4.5, 5.5, 6.0, 6.5]
    # The whole point: the values are distinct per snapshot.
    assert resolved["target_spread_line"].nunique() == 5


def test_snapshot_spread_produces_a_different_target_per_snapshot():
    """The worked example from the spec, end to end.

    HOME_MARGIN = +8 with the line drifting from 4 to 6 must produce errors
    4, 3.5, 2.5, 2 -- not one repeated number.
    """
    panel = _panel(
        [
            ("G1", 720, MARKET_SPREAD, "bet365", 4.0),
            ("G1", 360, MARKET_SPREAD, "bet365", 4.5),
            ("G1", 120, MARKET_SPREAD, "bet365", 5.5),
            ("G1", 60, MARKET_SPREAD, "bet365", 6.0),
        ]
    )
    frame = _resolve_target_spread_line(panel, anchor="bet365").assign(
        PTS_TEAM_HOME=110, PTS_TEAM_AWAY=102
    )
    line = spread_line_home_from_implied_margin(frame["target_spread_line"])
    errors = spread_error(home_margin(frame), line)

    assert home_margin(frame).unique().tolist() == [8]
    assert errors.tolist() == [4.0, 3.5, 2.5, 2.0]
    assert errors.nunique() == 4


def test_snapshot_target_never_uses_the_closing_line():
    """A closing-only tick must not become any snapshot's reference line.

    ``snapshot_minutes == 0`` is the closest-to-tip row. If the resolver ever
    collapsed on game_id, every horizon would take that row's 9.0 line.
    """
    panel = _panel(
        [
            ("G1", 720, MARKET_SPREAD, "bet365", 4.0),
            ("G1", 120, MARKET_SPREAD, "bet365", 6.0),
            ("G1", 0, MARKET_SPREAD, "bet365", 9.0),
        ]
    )
    resolved = _resolve_target_spread_line(panel, anchor="bet365").set_index(
        "snapshot_minutes"
    )["target_spread_line"]
    assert resolved.loc[720] == 4.0
    assert resolved.loc[120] == 6.0
    assert resolved.loc[0] == 9.0
    assert (resolved != 9.0).sum() == 2


def test_resolver_ignores_other_markets_and_other_books():
    panel = _panel(
        [
            ("G1", 120, MARKET_SPREAD, "bet365", 5.5),
            ("G1", 120, MARKET_TOTALS, "bet365", 224.5),
            ("G1", 120, MARKET_MONEYLINE, "bet365", 0.62),
            ("G1", 120, MARKET_SPREAD, "fanduel", 6.5),
        ]
    )
    resolved = _resolve_target_spread_line(panel, anchor="bet365")
    assert len(resolved) == 1
    assert resolved["target_spread_line"].iloc[0] == 5.5


def test_missing_anchor_spread_raises_rather_than_falling_back():
    """No silent substitution of another book -- that would change the target."""
    panel = _panel([("G1", 120, MARKET_SPREAD, "fanduel", 6.5)])
    with pytest.raises(ValueError, match="no spread quotes"):
        _resolve_target_spread_line(panel, anchor="bet365")


def test_push_survives_as_a_snapshot_target():
    panel = _panel([("G1", 120, MARKET_SPREAD, "bet365", 8.0)])
    frame = _resolve_target_spread_line(panel, anchor="bet365").assign(
        PTS_TEAM_HOME=110, PTS_TEAM_AWAY=102
    )
    errors = spread_error(
        home_margin(frame),
        spread_line_home_from_implied_margin(frame["target_spread_line"]),
    )
    assert errors.tolist() == [0.0]
    assert errors.notna().all()
    assert SPREAD_ERROR_COL == "SPREAD_ERROR"
