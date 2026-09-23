"""Tests for redundant-column detection and the keep-preference ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    DEFAULT_CORR_THRESHOLD_OVERRIDES,
    advanced_column_cleaning,
)
from nba_ou.data_processing.missing_data.column_redundancy import (
    KeepPreference,
    RepeatedMeasuresRedundancy,
    find_identical_groups,
    pairwise_complete_corr,
    rank_columns,
    representative_row_index,
    resolve_column_thresholds,
    select_correlated_columns_to_drop,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(16)


# ---------------------------------------------------------------------------
# pairwise_complete_corr: must reproduce DataFrame.corr() exactly
# ---------------------------------------------------------------------------


def test_pairwise_complete_corr_matches_pandas_without_nans(rng):
    df = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"c{i}" for i in range(12)])

    result = pairwise_complete_corr(df)
    expected = df.corr().abs().to_numpy()

    off_diagonal = ~np.eye(len(df.columns), dtype=bool)
    np.testing.assert_allclose(result[off_diagonal], expected[off_diagonal], atol=1e-10)


def test_pairwise_complete_corr_matches_pandas_with_nans(rng):
    """The whole reason for not mean-imputing: pandas scores each pair only on
    the rows where both columns are present, and so must this."""
    values = rng.normal(size=(300, 10))
    df = pd.DataFrame(values, columns=[f"c{i}" for i in range(10)])
    # Punch holes at different rates per column, up to 30%.
    for index, column in enumerate(df.columns):
        holes = rng.random(len(df)) < (index * 0.03)
        df.loc[holes, column] = np.nan

    result = pairwise_complete_corr(df)
    expected = df.corr().abs().to_numpy()

    off_diagonal = ~np.eye(len(df.columns), dtype=bool)
    np.testing.assert_allclose(result[off_diagonal], expected[off_diagonal], atol=1e-10)


def test_pairwise_complete_corr_is_nan_for_constant_and_sparse_pairs():
    df = pd.DataFrame(
        {
            "varies": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "sparse": [1.0, np.nan, np.nan, np.nan],
        }
    )
    corr = pairwise_complete_corr(df)

    assert np.isnan(corr[0, 1]), "a constant column has no correlation"
    assert np.isnan(corr[0, 2]), "one shared observation is below min_periods"


def test_sign_flip_is_perfect_absolute_correlation():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [-1.0, -2.0, -3.0, -4.0]})
    assert pairwise_complete_corr(df)[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# find_identical_groups: replaces the O(p^2) Series.equals loop
# ---------------------------------------------------------------------------


def test_find_identical_groups_matches_pairwise_equals(rng):
    base = rng.normal(size=(50, 4))
    df = pd.DataFrame(
        {
            "a": base[:, 0],
            "a_copy": base[:, 0],
            "a_copy2": base[:, 0],
            "b": base[:, 1],
            "b_copy": base[:, 1],
            "c": base[:, 2],
        }
    )

    groups = {frozenset(group) for group in find_identical_groups(df)}
    assert groups == {frozenset({"a", "a_copy", "a_copy2"}), frozenset({"b", "b_copy"})}

    # The behaviour the old nested loop had, on the same frame.
    columns = list(df.columns)
    expected_pairs = {
        frozenset({x, y})
        for i, x in enumerate(columns)
        for y in columns[i + 1 :]
        if df[x].equals(df[y])
    }
    found_pairs = {
        frozenset({x, y})
        for group in find_identical_groups(df)
        for i, x in enumerate(group)
        for y in group[i + 1 :]
    }
    assert found_pairs == expected_pairs


def test_find_identical_groups_requires_matching_nan_positions():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "same_holes": [1.0, np.nan, 3.0],
            "other_holes": [1.0, 2.0, np.nan],
        }
    )
    groups = {frozenset(group) for group in find_identical_groups(df)}
    assert groups == {frozenset({"a", "same_holes"})}


def test_find_identical_groups_absolute_catches_mixed_signs():
    """Correlation already catches a pure sign flip as |r| = 1; the absolute
    pass exists for the mixed case, which it cannot see."""
    df = pd.DataFrame(
        {
            "a": [1.0, -2.0, 3.0, -4.0],
            "mixed": [-1.0, -2.0, 3.0, 4.0],
            "unrelated": [5.0, 1.0, 9.0, 2.0],
        }
    )
    groups = {frozenset(g) for g in find_identical_groups(df, absolute=True)}
    assert groups == {frozenset({"a", "mixed"})}
    assert find_identical_groups(df) == []


# ---------------------------------------------------------------------------
# rank_columns: the preference order that replaces "keep whichever came later"
# ---------------------------------------------------------------------------


def test_protected_columns_rank_first():
    df = pd.DataFrame({"plain": [1.0, 2.0], "protected": [1.0, 2.0]})
    preference = KeepPreference.build(protected=["protected"], main_book="bet365")
    assert rank_columns(df, ["plain", "protected"], preference)[0] == "protected"


def test_fewer_nans_wins():
    df = pd.DataFrame({"gappy": [1.0, np.nan, np.nan], "complete": [1.0, 2.0, 3.0]})
    preference = KeepPreference.build(main_book="bet365")
    assert rank_columns(df, ["gappy", "complete"], preference)[0] == "complete"


def test_main_book_beats_other_books():
    """The regression this whole ordering exists for. The old rule kept
    whichever column came later in the frame, which on the real dataset dropped
    ODDS_TOTAL_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME (r=0.9982) in favour of
    the betmgm column -- discarding the book that defines the target."""
    bet365 = "ODDS_TOTAL_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME"
    betmgm = "ODDS_TOTAL_LINE_betmgm_SEASON_BEFORE_AVG_TEAM_HOME"
    df = pd.DataFrame({betmgm: [1.0, 2.0, 3.0], bet365: [1.0, 2.0, 3.0]})

    preference = KeepPreference.build(main_book="bet365")
    assert rank_columns(df, [betmgm, bet365], preference)[0] == bet365


def test_main_book_match_respects_name_boundaries():
    preference = KeepPreference.build(main_book="bet365")
    assert preference.mentions_main_book("ODDS_TOTAL_LINE_bet365_TEAM_HOME")
    assert preference.mentions_main_book("ODDS_TOTAL_LINE_bet365")
    assert not preference.mentions_main_book("ODDS_TOTAL_LINE_bet365x_TEAM_HOME")
    assert not preference.mentions_main_book("ODDS_TOTAL_LINE_betmgm_TEAM_HOME")


def test_main_book_match_is_case_insensitive():
    """get_main_book() spells books lower case, the intermediate-line dataset
    pivots them into column names upper case. Matched case-sensitively this tier
    returned False for every ODDS_SNAP_* column, so on that whole dataset the
    preference silently did nothing."""
    preference = KeepPreference.build(main_book="bet365")
    assert preference.mentions_main_book("ODDS_SNAP_TOT_BET365_NORM_LINE")
    assert preference.mentions_main_book("ODDS_SNAP_SPR_Bet365_RAW_LINE")
    # Boundaries still hold under the looser matching -- BETMGM must not match
    # merely because it starts with BET.
    assert not preference.mentions_main_book("ODDS_SNAP_TOT_BETMGM_NORM_LINE")
    assert not preference.mentions_main_book("ODDS_SNAP_TOT_BET365X_NORM_LINE")


def test_main_book_beats_other_books_in_snapshot_spelling():
    """Same regression as test_main_book_beats_other_books, in the spelling the
    intermediate-line dataset actually uses. Measured there, the five books'
    snapshot totals lines correlate 0.998-0.9995, so one of them is kept and the
    rest go; it must be the book that settles the bet.

    The main book here is fanduel, not the configured bet365, and that is what
    makes the test bite. Every book in that dataset sorts after BET365
    ('BET3...' < 'BETM...' < 'CONSENSUS' < ...), so with bet365 as the main book
    the alphabetical fallback keeps it whether this tier fires or not -- a test
    written that way passes with the tier disabled and proves nothing. Naming a
    main book that would LOSE the alphabetical tiebreak isolates the tier.
    """
    bet365 = "ODDS_SNAP_TOT_BET365_NORM_LINE"
    fanduel = "ODDS_SNAP_TOT_FANDUEL_NORM_LINE"
    df = pd.DataFrame({bet365: [1.0, 2.0, 3.0], fanduel: [1.0, 2.0, 3.0]})

    preference = KeepPreference.build(main_book="fanduel")
    assert rank_columns(df, [bet365, fanduel], preference)[0] == fanduel


def test_main_book_loses_to_a_better_populated_column():
    """The tier order is unchanged: NaN count still outranks the book. Stated as
    a test because making the match case-insensitive widened how often this tier
    fires, and it must not have been promoted in the process."""
    bet365 = "ODDS_SNAP_TOT_BET365_NORM_LINE"
    fanduel = "ODDS_SNAP_TOT_FANDUEL_NORM_LINE"
    df = pd.DataFrame({bet365: [1.0, np.nan, 3.0], fanduel: [1.0, 2.0, 3.0]})

    preference = KeepPreference.build(main_book="bet365")
    assert rank_columns(df, [bet365, fanduel], preference)[0] == fanduel


def test_canonical_market_name_beats_raw_shape():
    """Measured at r=1.0000 on the real dataset: the same moneyline under the
    canonical post-merge name and the raw odds-database name."""
    canonical = "ODDS_MONEYLINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME"
    raw = "ODDS_ml_bet365_price_SEASON_BEFORE_AVG_TEAM_HOME"
    df = pd.DataFrame({raw: [1.0, 2.0], canonical: [1.0, 2.0]})

    preference = KeepPreference.build(main_book="bet365")
    assert rank_columns(df, [raw, canonical], preference)[0] == canonical


def test_ranking_is_independent_of_column_order():
    columns = ["z_col", "a_col", "m_col"]
    values = {name: [1.0, 2.0, 3.0] for name in columns}
    preference = KeepPreference.build(main_book="bet365")

    forward = rank_columns(pd.DataFrame(values), columns, preference)
    reversed_frame = pd.DataFrame({k: values[k] for k in reversed(columns)})
    backward = rank_columns(reversed_frame, list(reversed(columns)), preference)

    assert forward == backward == ["a_col", "m_col", "z_col"]


# ---------------------------------------------------------------------------
# resolve_column_thresholds
# ---------------------------------------------------------------------------


def test_thresholds_apply_by_case_insensitive_substring():
    columns = ["ODDS_TOTAL_LINE_bet365", "PACE_BEFORE_TEAM_HOME"]
    thresholds = resolve_column_thresholds(
        columns, default=0.95, overrides={"odds_": 0.995}
    )
    np.testing.assert_allclose(thresholds, [0.995, 0.95])


def test_diff_from_odds_columns_take_the_odds_threshold():
    """Substring, not prefix, matching. DIFF_FROM_ODDS_LINE_* contains "ODDS_"
    without starting with it, and there are 192 such columns in the real
    dataset -- the difference between pruning 70 non-odds columns and 216. They
    are rolling averages of total-minus-line, so the tolerance is intended, but
    it is surprising enough to pin."""
    columns = [
        "ODDS_TOTAL_LINE_bet365",
        "DIFF_FROM_ODDS_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME",
        "PACE_BEFORE_TEAM_HOME",
    ]
    thresholds = resolve_column_thresholds(
        columns, default=0.95, overrides=DEFAULT_CORR_THRESHOLD_OVERRIDES
    )
    np.testing.assert_allclose(thresholds, [0.99, 0.99, 0.95])


def test_most_tolerant_override_wins_when_several_match():
    thresholds = resolve_column_thresholds(
        ["ODDS_SNAP_TOT"], default=0.90, overrides={"ODDS_": 0.995, "SNAP": 0.97}
    )
    assert thresholds[0] == pytest.approx(0.995)


def test_empty_overrides_leaves_a_single_threshold():
    thresholds = resolve_column_thresholds(["ODDS_A", "B"], default=0.95, overrides={})
    np.testing.assert_allclose(thresholds, [0.95, 0.95])


# ---------------------------------------------------------------------------
# select_correlated_columns_to_drop: the greedy pass
# ---------------------------------------------------------------------------


def test_group_thresholds_prune_non_odds_harder(rng):
    """Two pairs correlated at ~0.97. Under odds 0.995 / other 0.95 the odds
    pair survives intact and the non-odds pair loses a member."""
    # Two INDEPENDENT pairs, so the only redundancy is within each pair.
    odds_base = rng.normal(size=200)
    pace_base = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": odds_base,
            "ODDS_TOTAL_LINE_betmgm": odds_base + 0.25 * rng.normal(size=200),
            "PACE_BEFORE_TEAM_HOME": pace_base,
            "POSS_BEFORE_TEAM_HOME": pace_base + 0.25 * rng.normal(size=200),
        }
    )
    corr = pairwise_complete_corr(df)
    assert 0.95 < corr[0, 1] < 0.995, "odds pair must sit between the thresholds"
    assert 0.95 < corr[2, 3] < 0.995, "non-odds pair must sit between them too"

    dropped, _ = select_correlated_columns_to_drop(
        df,
        default_threshold=0.95,
        overrides={"ODDS_": 0.995},
        preference=KeepPreference.build(main_book="bet365"),
    )

    assert not any(name.startswith("ODDS_") for name in dropped)
    assert len(dropped) == 1 and dropped[0].endswith("_BEFORE_TEAM_HOME")


def test_pair_is_judged_at_the_more_tolerant_threshold(rng):
    base = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": base,
            "PLAIN_FEATURE": base + 0.25 * rng.normal(size=200),
        }
    )
    dropped, _ = select_correlated_columns_to_drop(
        df,
        default_threshold=0.95,
        overrides={"ODDS_": 0.995},
        preference=KeepPreference.build(main_book="bet365"),
    )
    assert dropped == [], "the odds column's tolerance covers the mixed pair"


def test_greedy_keeps_the_ends_of_a_correlation_chain(rng):
    """A~B and B~C above threshold, A~C below it. The old mask-based rule
    dropped both A and B and kept only C; one column per redundant cluster
    means keeping A and C."""
    a = rng.normal(size=400)
    c = rng.normal(size=400)
    b = a + c  # correlates with both ends, which do not correlate with each other

    df = pd.DataFrame({"A": a, "B": b, "C": c})
    corr = pairwise_complete_corr(df)
    threshold = 0.60
    assert corr[0, 1] > threshold and corr[1, 2] > threshold
    assert corr[0, 2] < threshold

    dropped, _ = select_correlated_columns_to_drop(
        df,
        default_threshold=threshold,
        overrides={},
        preference=KeepPreference.build(main_book="bet365"),
    )
    assert dropped == ["B"]


def test_protected_column_survives_a_perfectly_correlated_partner():
    """Exactly the failure clean_for_training's reattach workaround guarded
    against: an opening line is near-perfectly correlated with the closing line,
    which is why it was pruned and why the comparison it supports matters."""
    df = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": [210.0, 215.0, 220.0, 225.0],
            "ODDS_TOTAL_LINE_consensus_opener": [210.5, 215.5, 220.5, 225.5],
        }
    )
    dropped, _ = select_correlated_columns_to_drop(
        df,
        default_threshold=0.95,
        overrides={},
        preference=KeepPreference.build(
            protected=["ODDS_TOTAL_LINE_consensus_opener"], main_book="bet365"
        ),
    )
    assert "ODDS_TOTAL_LINE_consensus_opener" not in dropped


def test_decisions_report_the_surviving_partner(rng):
    base = rng.normal(size=100)
    df = pd.DataFrame({"KEEP_ME": base, "DROP_ME": base * 2.0})
    _, decisions = select_correlated_columns_to_drop(
        df,
        default_threshold=0.95,
        overrides={},
        preference=KeepPreference.build(protected=["KEEP_ME"], main_book="bet365"),
    )
    assert len(decisions) == 1
    dropped_col, kept_col, correlation = decisions[0]
    assert (dropped_col, kept_col) == ("DROP_ME", "KEEP_ME")
    assert correlation == pytest.approx(1.0)


def test_uncorrelated_columns_all_survive(rng):
    df = pd.DataFrame(rng.normal(size=(500, 6)), columns=[f"c{i}" for i in range(6)])
    dropped, _ = select_correlated_columns_to_drop(
        df,
        default_threshold=0.95,
        overrides={},
        preference=KeepPreference.build(main_book="bet365"),
    )
    assert dropped == []


# ---------------------------------------------------------------------------
# integration through advanced_column_cleaning
# ---------------------------------------------------------------------------


def test_advanced_column_cleaning_applies_odds_tolerance_by_default(rng):
    odds_base = rng.normal(size=300)
    pace_base = rng.normal(size=300)
    df = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": odds_base,
            "ODDS_TOTAL_LINE_betmgm": odds_base + 0.25 * rng.normal(size=300),
            "PACE_BEFORE_TEAM_HOME": pace_base,
            "POSS_BEFORE_TEAM_HOME": pace_base + 0.25 * rng.normal(size=300),
        }
    )

    cleaned = advanced_column_cleaning(df, corr_threshold=0.95, verbose=0)

    assert "ODDS_TOTAL_LINE_bet365" in cleaned.columns
    assert "ODDS_TOTAL_LINE_betmgm" in cleaned.columns
    assert "PACE_BEFORE_TEAM_HOME" in cleaned.columns
    assert "POSS_BEFORE_TEAM_HOME" not in cleaned.columns


def test_advanced_column_cleaning_keeps_the_main_book_over_another_book():
    bet365 = "ODDS_TOTAL_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME"
    betmgm = "ODDS_TOTAL_LINE_betmgm_SEASON_BEFORE_AVG_TEAM_HOME"
    # betmgm listed first, so the old "keep whichever came later" rule would
    # have dropped it and kept bet365 -- order the column pair the other way to
    # reproduce the real failure.
    df = pd.DataFrame(
        {
            bet365: [210.0, 214.0, 219.0, 223.0, 228.0],
            betmgm: [210.5, 214.4, 219.6, 223.4, 228.5],
        }
    )
    cleaned = advanced_column_cleaning(df, corr_threshold=0.95, verbose=0)

    assert list(cleaned.columns) == [bet365]


def test_keep_columns_survives_correlation_pruning():
    df = pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": [210.0, 215.0, 220.0, 225.0],
            "ODDS_TOTAL_LINE_consensus_opener": [210.5, 215.5, 220.5, 225.5],
        }
    )
    cleaned = advanced_column_cleaning(
        df,
        corr_threshold=0.95,
        corr_threshold_overrides={},
        keep_columns=["ODDS_TOTAL_LINE_consensus_opener"],
        verbose=0,
    )
    assert "ODDS_TOTAL_LINE_consensus_opener" in cleaned.columns


def test_keep_all_cols_skips_every_redundancy_step():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "a_copy": [1.0, 2.0, 3.0]})
    cleaned = advanced_column_cleaning(df, keep_all_cols=True, verbose=0)
    assert list(cleaned.columns) == ["a", "a_copy"]


def test_default_overrides_are_applied_when_argument_omitted(rng):
    """None means "use the defaults"; {} means "no overrides". Those are
    different, and the difference is what makes the odds tolerance opt-out."""
    base = rng.normal(size=300)
    companion = base + 0.25 * rng.normal(size=300)
    df = pd.DataFrame(
        {"ODDS_A": base, "ODDS_B": companion},
    )

    with_defaults = advanced_column_cleaning(df, corr_threshold=0.95, verbose=0)
    assert set(with_defaults.columns) == {"ODDS_A", "ODDS_B"}

    without = advanced_column_cleaning(
        df, corr_threshold=0.95, corr_threshold_overrides={}, verbose=0
    )
    assert len(without.columns) == 1
    assert DEFAULT_CORR_THRESHOLD_OVERRIDES == {"ODDS_": 0.99}


# ---------------------------------------------------------------------------
# representative_row_index
# ---------------------------------------------------------------------------


def _rep(group, snapshot, target=60.0):
    return representative_row_index(
        pd.Series(group),
        None if snapshot is None else pd.Series(snapshot),
        target=target,
    )


def test_representative_picks_the_target_snapshot():
    index = _rep(["g1", "g1", "g1", "g2", "g2"], [30, 60, 720, 720, 60])
    assert sorted(index) == [1, 4]


def test_representative_returns_exactly_one_row_per_group():
    group = ["g1"] * 10 + ["g2"] * 6 + ["g3"] * 1
    snapshot = (
        [0, 30, 60, 120, 180, 240, 300, 360, 480, 720]
        + [0, 30, 120, 240, 480, 720]
        + [720]
    )
    index = _rep(group, snapshot)
    assert len(index) == 3


def test_representative_falls_back_to_the_nearest_horizon():
    """t=60 is present for every game on the current build but the builder does
    not guarantee it -- build_snapshot_panel emits a horizon only where a tick
    exists at or before it, and t=720 already misses 149 games. A game without
    the target must degrade to its nearest horizon, not disappear from the
    view: losing it would silently stop its historical features counting."""
    index = _rep(["g1", "g1", "g2", "g2"], [0, 720, 30, 480])
    assert sorted(index) == [0, 2]  # 0 is nearer 60 than 720; 30 nearer than 480


def test_representative_breaks_ties_toward_the_smaller_horizon():
    index = _rep(["g1", "g1"], [0, 120])  # both exactly 60 away
    assert list(index) == [0]


def test_representative_is_independent_of_row_order():
    group = ["g1", "g2", "g1", "g2", "g1"]
    snapshot = [720, 30, 60, 60, 30]
    forward = set(_rep(group, snapshot))
    order = [4, 2, 0, 3, 1]
    backward = representative_row_index(
        pd.Series([group[i] for i in order]),
        pd.Series([snapshot[i] for i in order]),
        target=60.0,
    )
    assert {group[i] for i in forward} == {"g1", "g2"}
    assert len(backward) == 2

    # the same (group, horizon) pairs are chosen either way
    def chosen(idx, groups, snapshots):
        return sorted((groups[i], snapshots[i]) for i in idx)

    assert chosen(forward, group, snapshot) == chosen(
        backward, [group[i] for i in order], [snapshot[i] for i in order]
    )


def test_representative_ignores_nan_horizons_when_a_real_one_exists():
    index = _rep(["g1", "g1"], [np.nan, 720])
    assert list(index) == [1]


def test_representative_without_a_snapshot_column_still_gives_one_row_each():
    index = _rep(["g1", "g1", "g2"], None)
    assert len(index) == 2


def test_exempt_prefixes_cover_the_snapshot_blocks():
    policy = RepeatedMeasuresRedundancy(group_col="GAME_ID")
    assert policy.is_exempt("ODDS_SNAP_TOT_BET365_NORM_LINE")
    assert policy.is_exempt("ODDS_LINE_HIST_ANYTHING")
    assert not policy.is_exempt("ODDS_TOTAL_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME")
    assert not policy.is_exempt("PACE_BEFORE_TEAM_HOME")
    assert not policy.is_exempt("ODDS_TOT_PRIOR_GAME_MOVE")


def test_zero_variance_pairs_are_nan_not_inf():
    """``pairwise_complete_corr`` promises to reproduce ``DataFrame.corr()``.

    For a column with no spread pandas returns NaN. Computing the variance as
    ``E[x^2] - E[x]^2`` lands it on either side of zero as roundoff, so the
    division returned inf -- and every caller here tests ``corr > threshold``,
    which inf passes. The whole frame then looks redundant with the constant.
    """
    rng = np.random.default_rng(1)
    for n in (6165, 8301):
        df = pd.DataFrame(
            {
                "CONST": np.full(n, 60.0),
                "y": rng.normal(220, 10, n),
                "z": rng.normal(0, 1, n),
            }
        )
        corr = pairwise_complete_corr(df)
        assert np.isnan(corr[0, 1]), f"n={n}: got {corr[0, 1]}"
        assert not (corr[0, 1] > 0.95)
        expected = np.abs(df.corr().to_numpy())
        np.fill_diagonal(expected, np.nan)
        assert np.allclose(corr, expected, equal_nan=True)


def test_constant_side_does_not_win_a_redundancy_decision():
    """A constant must not be able to evict a real feature."""
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame(
        {
            "A_CONST": np.full(n, 60.0),
            "B_REAL": rng.normal(100, 5, n),
            "C_REAL": rng.normal(50, 2, n),
        }
    )
    dropped, _ = select_correlated_columns_to_drop(
        df, default_threshold=0.95, overrides=None, preference=KeepPreference.build()
    )
    assert dropped == []
