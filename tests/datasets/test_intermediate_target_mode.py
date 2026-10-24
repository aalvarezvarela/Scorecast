"""Tonight-only intermediate builds give tonight's games the full build's rows."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.create_training_data import create_intermediate_line_df as builder
from nba_ou.data_processing.line_history.movement_features import DEFAULT_WINDOWS
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

from tests.support.frame_identity import assert_identical, random_ticks

GRID = (30, 60, 120, 360, 720)


def _ticks(seed: int) -> pd.DataFrame:
    ticks = random_ticks(
        n_games=14, books=("bet365", "draftkings", "fanduel", "betmgm"), seed=seed
    )
    return ticks.replace(
        {"market": {"spread": MARKET_SPREAD, "moneyline": MARKET_MONEYLINE}}
    )


def _side(ticks, schema=None):
    return builder._build_snapshot_side(
        ticks,
        snapshot_grid=GRID,
        windows=DEFAULT_WINDOWS,
        anchor="bet365",
        ridge_markets=[MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE],
        normalize_total_lines=True,
        normalize_spread_lines=True,
        null_extreme_spread_prices=True,
        stage=lambda message: None,
        progress=lambda label: None,
        verbose=False,
        schema=schema,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_few_games_get_the_full_builds_wide_rows(seed):
    ticks = _ticks(seed)
    games = sorted(ticks["game_id"].unique())[3:6]
    # Tonight's games come last, and may lack a book the others quote.
    tonight = ticks[ticks["game_id"].isin(games) & ticks["book"].ne("betmgm")]
    history = ticks[~ticks["game_id"].isin(games)]
    full = _side(pd.concat([history, tonight], ignore_index=True))
    earlier = full.combos["game_id"].isin(history["game_id"])
    earlier_markets = full.consensus_markets["game_id"].isin(history["game_id"])
    # Non-float columns: kept only where every build of the history keeps them.
    dtypes = builder._history_dtypes([_side(history).dtypes])
    part = _side(
        tonight,
        schema=(full.combos[earlier], full.consensus_markets[earlier_markets], dtypes),
    )
    expected = full.wide[full.wide["game_id"].isin(games)].reset_index(drop=True)
    assert_identical(part.wide, expected)
    for market, closing in part.ridge_closing_lines.items():
        full_closing = full.ridge_closing_lines[market]
        assert_identical(
            closing.reset_index(drop=True),
            full_closing[full_closing["GAME_ID"].isin(games)].reset_index(drop=True),
        )


def test_without_a_schema_the_columns_follow_the_ticks():
    ticks = _ticks(0)
    games = sorted(ticks["game_id"].unique())[:2]
    alone = _side(ticks[ticks["game_id"].isin(games) & ticks["book"].ne("betmgm")])
    assert not any("_BETMGM_" in column for column in alone.wide.columns)
