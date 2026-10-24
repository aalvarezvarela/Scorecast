"""Tests for the wide-pipeline Caesars/fanatics_sportsbook reconciliation.

``combine_caesars_and_fanatics`` is the choke point both create_df_to_predict
and create_intermediate_line_df's closing/opener side go through (via
load_and_merge_odds_yahoo_sportsbookreview). Mirrors
test_line_history_book_merge.py for the tidy Aiven tick-store side.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.odds.book_combination import (
    combine_caesars_and_fanatics,
    resolve_combine_books,
)


def test_noop_when_neither_option_is_set():
    df = pd.DataFrame(
        {
            "total_fanatics_sportsbook_line_over": [220.5, None],
            "total_caesars_line_over": [219.5, 221.0],
        }
    )
    out = combine_caesars_and_fanatics(df)
    pd.testing.assert_frame_equal(out, df)


def test_combine_fills_fanatics_nans_from_caesars_and_drops_caesars():
    df = pd.DataFrame(
        {
            "total_fanatics_sportsbook_line_over": [220.5, None, None],
            "total_caesars_line_over": [219.5, 221.0, None],
            "spread_fanatics_sportsbook_line_home": [-3.5, None, -1.0],
            "spread_caesars_line_home": [-3.0, -2.5, None],
            "ml_fanatics_sportsbook_price_home": [1.9, None, 1.5],
            "ml_caesars_price_home": [1.85, 1.6, None],
            "unrelated_column": [1, 2, 3],
        }
    )
    out = combine_caesars_and_fanatics(df, combine_with_fanatics=True)

    assert "total_caesars_line_over" not in out.columns
    assert "spread_caesars_line_home" not in out.columns
    assert "ml_caesars_price_home" not in out.columns
    assert out["total_fanatics_sportsbook_line_over"].iloc[:2].tolist() == [
        220.5,
        221.0,
    ]
    assert pd.isna(out["total_fanatics_sportsbook_line_over"].iloc[2])
    assert out["spread_fanatics_sportsbook_line_home"].tolist() == [-3.5, -2.5, -1.0]
    assert out["ml_fanatics_sportsbook_price_home"].tolist() == [1.9, 1.6, 1.5]
    assert out["unrelated_column"].tolist() == [1, 2, 3]


def test_combine_creates_fanatics_column_when_only_caesars_is_present():
    df = pd.DataFrame({"total_caesars_line_over": [219.5, 221.0]})
    out = combine_caesars_and_fanatics(df, combine_with_fanatics=True)
    assert "total_caesars_line_over" not in out.columns
    assert out["total_fanatics_sportsbook_line_over"].tolist() == [219.5, 221.0]


def test_exclude_caesars_drops_every_caesars_column():
    df = pd.DataFrame(
        {
            "total_caesars_line_over": [219.5],
            "spread_caesars_line_home": [-3.0],
            "ml_caesars_price_away": [2.1],
            "total_fanatics_sportsbook_line_over": [220.5],
        }
    )
    out = combine_caesars_and_fanatics(df, exclude_caesars=True)
    assert not [c for c in out.columns if "caesars" in c]
    assert out["total_fanatics_sportsbook_line_over"].tolist() == [220.5]


def test_exclude_caesars_and_combine_are_mutually_exclusive():
    df = pd.DataFrame({"total_caesars_line_over": [219.5]})
    with pytest.raises(ValueError, match="cannot both be True"):
        combine_caesars_and_fanatics(
            df, exclude_caesars=True, combine_with_fanatics=True
        )


# --- default resolution ---------------------------------------------------
#
# Combining is the default because it is the only option that fixes the
# season-leak without discarding a book. Making that default a plain `True`
# would turn every explicit `exclude_*` into a hard error -- which is exactly
# what happened to the intermediate CLI, whose default invocation started
# raising. The tri-state below is what keeps "merged by default" compatible
# with "exclude if you asked for it".


def test_combining_is_the_default_when_nothing_is_requested():
    assert resolve_combine_books(combine=None) is True


def test_an_explicit_exclusion_wins_over_the_default():
    assert resolve_combine_books(combine=None, exclude_caesars=True) is False
    assert resolve_combine_books(combine=None, exclude_fanatics=True) is False


def test_combining_can_still_be_turned_off_explicitly():
    assert resolve_combine_books(combine=False) is False
    assert resolve_combine_books(combine=False, exclude_caesars=True) is False


def test_explicitly_asking_for_both_is_the_one_real_contradiction():
    with pytest.raises(ValueError, match="no standalone book left to exclude"):
        resolve_combine_books(combine=True, exclude_caesars=True)
    with pytest.raises(ValueError, match="no standalone book left to exclude"):
        resolve_combine_books(combine=True, exclude_fanatics=True)
