"""Tests for the cleaning pipeline's hygiene fixes and the cleaning report.

Each of these pins a bug that was found by audit rather than by a failure --
none of them made anything crash, which is why they lasted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    MIN_PLAUSIBLE_TOTAL_LINE,
    MIN_PLAUSIBLE_TOTAL_POINTS,
    _normalize_nullable_dtypes,
    advanced_column_cleaning,
    basic_cleaning,
    clean_dataframe_for_training,
)
from nba_ou.data_processing.missing_data.cleaning_report import CleaningReport
from nba_ou.data_processing.missing_data.column_redundancy import (
    RepeatedMeasuresRedundancy,
)
from nba_ou.data_processing.missing_data.handle_missing_data import (
    apply_missing_policy,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(16)
    n = 60
    return pd.DataFrame(
        {
            "GAME_ID": [f"002240000{i:02d}" for i in range(n)],
            "TOTAL_POINTS": rng.normal(225, 20, n).round(),
            "ODDS_TOTAL_LINE_bet365": rng.normal(224, 15, n).round(),
            "ODDS_SPREAD_bet365": rng.normal(0, 6, n).round(),
            "ODDS_MONEYLINE_bet365_TEAM_HOME": rng.normal(-110, 40, n).round(),
            "ODDS_MONEYLINE_bet365_TEAM_AWAY": rng.normal(-110, 40, n).round(),
            "IS_US_HOLIDAY_BEFORE": [i % 7 == 0 for i in range(n)],
            "PACE_BEFORE_TEAM_HOME": rng.normal(100, 4, n),
        }
    )


# ---------------------------------------------------------------------------
# input mutation
# ---------------------------------------------------------------------------


def test_basic_cleaning_does_not_mutate_its_input(frame):
    """The dtype casts used to land on the caller's frame, so cleaning the same
    dataframe twice did not do the same thing twice."""
    before = frame.copy(deep=True)
    basic_cleaning(frame, verbose=0)

    pd.testing.assert_frame_equal(frame, before)


def test_clean_dataframe_for_training_does_not_mutate_its_input(frame):
    before = frame.copy(deep=True)
    clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)

    pd.testing.assert_frame_equal(frame, before)


def test_cleaning_is_idempotent_across_two_calls(frame):
    """The observable consequence of the mutation bug."""
    first = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)
    second = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)

    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# the implausible-total filter
# ---------------------------------------------------------------------------


def test_implausibly_low_totals_are_dropped(frame):
    frame.loc[0, "TOTAL_POINTS"] = 12.0
    frame.loc[1, "TOTAL_POINTS"] = float(MIN_PLAUSIBLE_TOTAL_POINTS)

    cleaned = basic_cleaning(frame, verbose=0)

    assert len(cleaned) == len(frame) - 2
    assert (cleaned["TOTAL_POINTS"] > MIN_PLAUSIBLE_TOTAL_POINTS).all()


def test_unknown_totals_survive(frame):
    """The prediction path cleans scheduled games, which have no final score.
    A plain `df[df.TOTAL_POINTS > 130]` compares against NaN, is False for every
    such row, and would silently return an empty frame -- deleting the daily
    prediction run rather than failing it."""
    frame["TOTAL_POINTS"] = np.nan

    cleaned = basic_cleaning(frame, verbose=0)

    assert len(cleaned) == len(frame)


def test_a_frame_with_no_target_column_still_cleans(frame):
    cleaned = basic_cleaning(frame.drop(columns=["TOTAL_POINTS"]), verbose=0)
    assert len(cleaned) == len(frame)


def test_implausible_lines_are_dropped(frame):
    frame.loc[0, "ODDS_TOTAL_LINE_bet365"] = float(MIN_PLAUSIBLE_TOTAL_LINE)
    cleaned = basic_cleaning(frame, verbose=0)
    assert len(cleaned) == len(frame) - 1


# ---------------------------------------------------------------------------
# nullable dtypes
# ---------------------------------------------------------------------------


def test_nullable_boolean_with_na_becomes_float_not_object():
    """A nullable boolean holding pd.NA converts via to_numpy() to dtype object,
    which XGBoost rejects. The previous normalisation could not reach it:
    select_dtypes(include=[np.number]) does not match `boolean`."""
    df = pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean")})

    assert df["flag"].to_numpy().dtype == object  # the trap

    out = _normalize_nullable_dtypes(df)

    assert out["flag"].dtype == np.float64
    assert out["flag"].to_numpy().dtype == np.float64
    assert np.isnan(out["flag"].to_numpy()[2])


def test_holiday_flag_leaves_cleaning_as_a_numpy_dtype(frame):
    """basic_cleaning deliberately casts this column to nullable boolean, so the
    end of the pipeline has to undo it."""
    frame["IS_US_HOLIDAY_BEFORE"] = frame["IS_US_HOLIDAY_BEFORE"].astype(object)
    frame.loc[0, "IS_US_HOLIDAY_BEFORE"] = None

    cleaned = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)

    assert not isinstance(
        cleaned["IS_US_HOLIDAY_BEFORE"].dtype, pd.api.extensions.ExtensionDtype
    )
    assert cleaned["IS_US_HOLIDAY_BEFORE"].to_numpy().dtype != object


def test_nullable_integer_is_normalised():
    df = pd.DataFrame({"n": pd.array([1, 2, None], dtype="Int64")})
    out = _normalize_nullable_dtypes(df)
    assert out["n"].dtype == np.float64


def test_categorical_columns_are_left_alone():
    """XGBoost consumes pandas Categorical natively under enable_categorical,
    which is what categorical_team_encoding relies on."""
    df = pd.DataFrame({"team": pd.Categorical(["BOS", "LAL", "BOS"])})
    out = _normalize_nullable_dtypes(df)
    assert isinstance(out["team"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------------------
# empty frame
# ---------------------------------------------------------------------------


def test_empty_frame_does_not_warn_or_drop_every_column(frame):
    """0/0 is not 100% NaN. Without the guard this emitted a RuntimeWarning and
    then dropped nothing anyway, because every comparison against NaN is False."""
    empty = frame.iloc[:0]

    with warnings_as_errors():
        cleaned = advanced_column_cleaning(empty, nan_threshold=50.0, verbose=0)

    assert len(cleaned.columns) > 0


def warnings_as_errors():
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            yield

    return ctx()


# ---------------------------------------------------------------------------
# the cleaning report
# ---------------------------------------------------------------------------


def test_report_is_only_returned_when_asked(frame):
    plain = clean_dataframe_for_training(frame, verbose=0, keep_all_cols=True)
    assert isinstance(plain, pd.DataFrame)

    pair = clean_dataframe_for_training(
        frame, verbose=0, keep_all_cols=True, return_report=True
    )
    assert isinstance(pair, tuple) and isinstance(pair[1], CleaningReport)


def test_report_names_the_step_that_dropped_a_column(frame):
    frame["ALWAYS_SAME"] = 1.0
    frame["A_STRING_COL"] = "text"

    _, report = clean_dataframe_for_training(
        frame, nan_threshold=50.0, verbose=0, return_report=True
    )

    assert report.why_dropped("ALWAYS_SAME")["step"] == "constant_columns"
    assert report.why_dropped("A_STRING_COL")["step"] == "string_columns"
    assert report.why_dropped("PACE_BEFORE_TEAM_HOME") is None


def test_report_records_the_correlation_partner(frame):
    frame["PACE_COPY_BEFORE"] = frame["PACE_BEFORE_TEAM_HOME"] * 2.0

    _, report = clean_dataframe_for_training(
        frame,
        corr_threshold=0.95,
        corr_threshold_overrides={},
        verbose=0,
        return_report=True,
    )

    dropped = [
        e
        for e in report.column_drops
        if e["step"] in ("correlated_columns", "duplicate_columns")
    ]
    assert dropped, "the duplicated column should have been recorded"
    assert "PACE" in dropped[0]["reason"] or "r=" in dropped[0]["reason"]


def test_report_records_row_drops_with_reasons(frame):
    frame.loc[0, "TOTAL_POINTS"] = 10.0
    frame.loc[1, "ODDS_TOTAL_LINE_bet365"] = np.nan

    _, report = clean_dataframe_for_training(frame, verbose=0, return_report=True)

    steps = {entry["step"] for entry in report.row_drops}
    assert "basic_cleaning.implausible_total" in steps
    assert "basic_cleaning.missing_total_line" in steps
    assert all(entry["rows_dropped"] > 0 for entry in report.row_drops)


def test_report_totals_match_the_returned_frame(frame):
    frame["A_STRING_COL"] = "text"
    cleaned, report = clean_dataframe_for_training(
        frame, nan_threshold=50.0, verbose=0, return_report=True
    )

    assert report.columns_in == len(frame.columns)
    assert report.columns_out == len(cleaned.columns)
    assert report.rows_in == len(frame)
    assert report.rows_out == len(cleaned)
    # Every dropped column is accounted for exactly once, so the per-step counts
    # sum to the columns that actually went. GAME_ID satisfies both the
    # pure-string and the _ID rule and must still be reported once.
    assert len(report.column_drops) == report.columns_in - report.columns_out
    recorded = [entry["column"] for entry in report.column_drops]
    assert len(recorded) == len(set(recorded))
    assert sum(report.columns_by_step().values()) == len(report.column_drops)


def test_report_serialises_to_json(frame, tmp_path):
    frame["A_STRING_COL"] = "text"
    _, report = clean_dataframe_for_training(frame, verbose=0, return_report=True)

    path = report.save(tmp_path / "cleaning_report.json")

    import json

    payload = json.loads(path.read_text())
    assert payload["columns_in"] == len(frame.columns)
    assert "column_drops" in payload and "columns_dropped_by_step" in payload


# ---------------------------------------------------------------------------
# per-group row retention
#
# Row cleaning is stated as one global threshold, but nothing makes it bite
# evenly. On the intermediate-line dataset a group is one pre-game snapshot
# horizon, and the long horizons carry more NaNs -- so the row filters
# re-weight the snapshot mix, which is the one axis that dataset exists to
# compare. These cover the record, not a policy: nothing here changes what is
# dropped.
# ---------------------------------------------------------------------------


@pytest.fixture
def grouped_frame(frame) -> pd.DataFrame:
    """``frame`` given a snapshot grain, with the rows a filter will delete
    concentrated in one group so retention has to come out uneven.

    The unfit rows are made unfit via TOTAL_POINTS rather than by planting NaNs:
    the missing-data policy runs before the row-NaN filter and fills these
    columns, so a planted NaN never reaches it and every group retains 100%.
    """
    df = pd.concat([frame.assign(TIME_TO_MATCH_MIN=t) for t in (30, 720)])
    df = df.reset_index(drop=True)
    far = df["TIME_TO_MATCH_MIN"] == 720
    df.loc[far & (df.index % 4 == 1), "TOTAL_POINTS"] = MIN_PLAUSIBLE_TOTAL_POINTS - 10
    return df


def test_group_survival_is_recorded_when_a_group_column_is_named(grouped_frame):
    _, report = clean_dataframe_for_training(
        grouped_frame,
        max_na_per_row=0,
        row_balance_group_col="TIME_TO_MATCH_MIN",
        verbose=0,
        return_report=True,
    )

    assert report.group_survival["group_col"] == "TIME_TO_MATCH_MIN"
    retention = {
        group["group"]: group["retention_pct"]
        for group in report.group_survival["groups"]
    }
    assert retention[30] == 100.0
    assert retention[720] < 100.0
    assert report.group_survival["retention_spread_pp"] == pytest.approx(
        retention[30] - retention[720]
    )


def test_group_survival_counts_reconcile_with_the_returned_frame(grouped_frame):
    cleaned, report = clean_dataframe_for_training(
        grouped_frame,
        max_na_per_row=0,
        row_balance_group_col="TIME_TO_MATCH_MIN",
        verbose=0,
        return_report=True,
    )

    for group in report.group_survival["groups"]:
        actual = int((cleaned["TIME_TO_MATCH_MIN"] == group["group"]).sum())
        assert group["rows_after"] == actual
    assert sum(g["rows_after"] for g in report.group_survival["groups"]) == len(cleaned)


def test_group_survival_is_absent_when_no_group_column_is_named(grouped_frame):
    _, report = clean_dataframe_for_training(
        grouped_frame, verbose=0, return_report=True
    )
    assert report.group_survival == {}


def test_group_survival_is_absent_when_the_column_is_not_in_the_frame(frame):
    """The closing-line dataset's ordinary case: one call site serves both
    datasets, so a missing group column is not an error."""
    _, report = clean_dataframe_for_training(
        frame,
        row_balance_group_col="TIME_TO_MATCH_MIN",
        verbose=0,
        return_report=True,
    )
    assert report.group_survival == {}


def test_group_survival_survives_the_group_column_being_dropped(grouped_frame):
    """The grouping column is an ordinary feature and can be pruned by column
    cleaning. The record is taken by index for exactly that reason, so it must
    still be produced -- a report that quietly stopped appearing whenever the
    column was dropped would be worse than none."""
    grouped_frame["TIME_TO_MATCH_MIN_COPY"] = grouped_frame["TIME_TO_MATCH_MIN"]

    cleaned, report = clean_dataframe_for_training(
        grouped_frame,
        max_na_per_row=0,
        keep_columns=["TIME_TO_MATCH_MIN_COPY"],
        row_balance_group_col="TIME_TO_MATCH_MIN",
        verbose=0,
        return_report=True,
    )

    assert "TIME_TO_MATCH_MIN" not in cleaned.columns
    assert report.why_dropped("TIME_TO_MATCH_MIN") is not None
    assert {g["group"] for g in report.group_survival["groups"]} == {30, 720}


def test_group_survival_is_serialised(grouped_frame, tmp_path):
    _, report = clean_dataframe_for_training(
        grouped_frame,
        max_na_per_row=0,
        row_balance_group_col="TIME_TO_MATCH_MIN",
        verbose=0,
        return_report=True,
    )

    import json

    payload = json.loads(report.save(tmp_path / "cleaning_report.json").read_text())
    assert payload["group_survival"]["group_col"] == "TIME_TO_MATCH_MIN"
    assert len(payload["group_survival"]["groups"]) == 2


def test_group_survival_orders_groups_naturally(grouped_frame):
    """Snapshot horizons must read 30, 720 rather than sorted as text, where a
    real grid comes out 0, 120, 180, 240, 30, 300 -- the table exists to be read
    down the horizon axis."""
    df = pd.concat(
        [grouped_frame.assign(TIME_TO_MATCH_MIN=t) for t in (0, 30, 120, 720)]
    ).reset_index(drop=True)

    _, report = clean_dataframe_for_training(
        df, row_balance_group_col="TIME_TO_MATCH_MIN", verbose=0, return_report=True
    )

    assert [g["group"] for g in report.group_survival["groups"]] == [0, 30, 120, 720]


def test_group_survival_tolerates_unorderable_group_keys():
    """A grouping column of mixed types must not crash the report; it is a
    record, and losing the whole run to it would be absurd."""
    report = CleaningReport()
    report.record_group_survival(
        group_col="MIXED", before_counts={1: 10, "a": 10}, after_counts={1: 5, "a": 10}
    )
    assert {g["group"] for g in report.group_survival["groups"]} == {1, "a"}


# ---------------------------------------------------------------------------
# repeated-measures redundancy
#
# Historical/base features are computed per GAME and copied onto every snapshot
# of it, so correlating them over the full frame counts each game up to ten
# times -- the same evidence repeated, not more of it. Snapshot/market columns
# are the opposite case: they are what the extra rows are for, and are exempt
# from correlation pruning altogether.
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_grain_frame() -> pd.DataFrame:
    """Six games x three snapshots. Historical columns are copied across a
    game's rows, exactly as the real builder copies them."""
    rng = np.random.default_rng(11)
    games = [f"002240000{i}" for i in range(40)]
    rows = []
    for game in games:
        pace = rng.normal(100, 4)
        level = rng.normal(224, 12)
        for snapshot in (30, 60, 720):
            rows.append(
                {
                    "GAME_ID": game,
                    "TIME_TO_MATCH_MIN": snapshot,
                    "TOTAL_POINTS": float(round(rng.normal(225, 20))),
                    "ODDS_TOTAL_LINE_bet365": level,
                    # historical, per game, copied across snapshots
                    "PACE_BEFORE_TEAM_HOME": pace,
                    "PACE_COPY_BEFORE_TEAM_HOME": pace,
                    # snapshot/market: varies across a game's rows
                    "ODDS_SNAP_TOT_BET365_NORM_LINE": level + rng.normal(0, 0.6),
                    "ODDS_SNAP_TOT_FANDUEL_NORM_LINE": level + rng.normal(0, 0.6),
                }
            )
    return pd.DataFrame(rows)


def _policy():
    return RepeatedMeasuresRedundancy(
        group_col="GAME_ID", snapshot_col="TIME_TO_MATCH_MIN"
    )


def test_snapshot_columns_are_never_correlation_pruned(snapshot_grain_frame):
    """The two book lines correlate above 0.99 over any view -- they share a
    level. They must both survive anyway."""
    both = ["ODDS_SNAP_TOT_BET365_NORM_LINE", "ODDS_SNAP_TOT_FANDUEL_NORM_LINE"]
    assert snapshot_grain_frame[both].corr().iloc[0, 1] > 0.99  # the trap

    cleaned, report = clean_dataframe_for_training(
        snapshot_grain_frame,
        repeated_measures=_policy(),
        verbose=0,
        return_report=True,
    )

    for column in both:
        assert column in cleaned.columns, report.why_dropped(column)


def test_snapshot_columns_still_lose_exact_duplicates(snapshot_grain_frame):
    """Exempt from correlation, not from duplicate detection."""
    snapshot_grain_frame["ODDS_SNAP_TOT_COPY_NORM_LINE"] = snapshot_grain_frame[
        "ODDS_SNAP_TOT_BET365_NORM_LINE"
    ]

    cleaned, report = clean_dataframe_for_training(
        snapshot_grain_frame,
        repeated_measures=_policy(),
        verbose=0,
        return_report=True,
    )

    survivors = [
        c
        for c in ("ODDS_SNAP_TOT_BET365_NORM_LINE", "ODDS_SNAP_TOT_COPY_NORM_LINE")
        if c in cleaned.columns
    ]
    assert len(survivors) == 1
    gone = report.why_dropped("ODDS_SNAP_TOT_COPY_NORM_LINE") or report.why_dropped(
        "ODDS_SNAP_TOT_BET365_NORM_LINE"
    )
    assert gone["step"] == "duplicate_columns"


def test_historical_columns_are_still_correlation_pruned(snapshot_grain_frame):
    """The other half of the policy: a redundant historical feature must still
    go, judged on the one-row-per-game view."""
    cleaned, report = clean_dataframe_for_training(
        snapshot_grain_frame,
        repeated_measures=_policy(),
        verbose=0,
        return_report=True,
    )

    survivors = [
        c
        for c in ("PACE_BEFORE_TEAM_HOME", "PACE_COPY_BEFORE_TEAM_HOME")
        if c in cleaned.columns
    ]
    assert len(survivors) == 1


def test_correlation_drops_name_the_one_row_per_group_step(snapshot_grain_frame):
    """The report must distinguish an exact duplicate from a correlation drop
    taken on the one-row-per-game view."""
    rng = np.random.default_rng(5)
    base = snapshot_grain_frame["PACE_BEFORE_TEAM_HOME"]
    snapshot_grain_frame["PACE_NEAR_BEFORE_TEAM_HOME"] = base + rng.normal(
        0, 0.01, len(base)
    )

    _, report = clean_dataframe_for_training(
        snapshot_grain_frame,
        repeated_measures=_policy(),
        verbose=0,
        return_report=True,
    )

    steps = report.columns_by_step()
    assert "correlated_columns_one_row_per_group" in steps
    assert "correlated_columns" not in steps


def test_repeated_rows_do_not_change_which_historical_columns_survive():
    """THE point of the policy. The same games, judged once each, must give the
    same verdict whether or not each game is repeated across snapshots."""
    rng = np.random.default_rng(19)
    n = 60
    one_row_per_game = pd.DataFrame(
        {
            "GAME_ID": [f"002240000{i:02d}" for i in range(n)],
            "TOTAL_POINTS": rng.normal(225, 20, n).round(),
            "ODDS_TOTAL_LINE_bet365": rng.normal(224, 15, n).round(),
            "PACE_BEFORE_TEAM_HOME": rng.normal(100, 4, n),
            "OFF_RATING_BEFORE_TEAM_HOME": rng.normal(112, 5, n),
            "DEF_RATING_BEFORE_TEAM_HOME": rng.normal(112, 5, n),
        }
    )
    repeated = pd.concat(
        [one_row_per_game.assign(TIME_TO_MATCH_MIN=t) for t in (30, 60, 720)]
    ).reset_index(drop=True)

    flat = clean_dataframe_for_training(one_row_per_game, verbose=0)
    grained = clean_dataframe_for_training(
        repeated, repeated_measures=_policy(), verbose=0
    )

    def historical(df):
        return sorted(c for c in df.columns if c.endswith("_BEFORE_TEAM_HOME"))

    assert historical(flat) == historical(grained)


def test_redundancy_view_is_recorded_and_serialised(snapshot_grain_frame, tmp_path):
    _, report = clean_dataframe_for_training(
        snapshot_grain_frame,
        repeated_measures=_policy(),
        verbose=0,
        return_report=True,
    )

    view = report.redundancy_view
    assert view["group_col"] == "GAME_ID"
    assert view["rows_in_view"] == snapshot_grain_frame["GAME_ID"].nunique()
    assert view["rows_total"] == len(snapshot_grain_frame)
    assert view["snapshots_used"] == {"60": snapshot_grain_frame["GAME_ID"].nunique()}
    assert any(c.startswith("ODDS_SNAP_") for c in view["exempt_columns"])

    import json

    payload = json.loads(report.save(tmp_path / "r.json").read_text())
    assert payload["redundancy_view"]["group_col"] == "GAME_ID"


def test_policy_without_its_group_column_raises(frame):
    """Silence here would mean quietly falling back to correlating over every
    row -- the behaviour the policy exists to prevent."""
    with pytest.raises(KeyError, match="MISSING_ID"):
        clean_dataframe_for_training(
            frame,
            repeated_measures=RepeatedMeasuresRedundancy(group_col="MISSING_ID"),
            verbose=0,
        )


def test_no_policy_leaves_cleaning_unchanged(snapshot_grain_frame):
    """One row per game is the normal case and must be untouched by any of
    this: without a policy, every row and every column is correlated as before."""
    _, report = clean_dataframe_for_training(
        snapshot_grain_frame, verbose=0, return_report=True
    )
    assert report.redundancy_view == {}
    assert "correlated_columns_one_row_per_group" not in report.columns_by_step()


def test_unequal_repetition_does_not_decide_a_historical_column(capsys):
    """The test that pins the whole policy, and it needs UNEQUAL repetition.

    Repeating every game the same number of times leaves a correlation
    coefficient unchanged, so an equally-repeated fixture cannot tell the two
    views apart -- an earlier version of these tests could not, and a mutant
    that correlated over every row passed all of them. Real data is not equal:
    measured on intermediate_line_data_10snap.csv, games carry between 6 and 10
    rows, because a horizon exists only where a tick reaches it.

    Here 16 games get 10 rows each and 44 get 2. Those 16 are games where the
    two historical columns happen to agree closely. Judged once per game they
    correlate 0.847 and both belong; judged over every row the over-represented
    16 pull it to 0.962 and one is discarded -- on nothing but how many
    snapshots those games happened to have.
    """
    rng = np.random.default_rng(191)
    heavy, light = 16, 44
    a_heavy = rng.normal(0, 4.0, heavy)
    b_heavy = a_heavy + rng.normal(0, 0.05, heavy)
    a_light = rng.normal(0, 1, light)
    b_light = a_light + rng.normal(0, 1.6, light)
    a = np.r_[a_heavy, a_light]
    b = np.r_[b_heavy, b_light]
    repeats = np.r_[np.full(heavy, 10), np.full(light, 2)]

    per_game = pd.DataFrame(
        {
            "GAME_ID": [f"002240000{i:02d}" for i in range(heavy + light)],
            "TOTAL_POINTS": rng.normal(225, 20, heavy + light).round(),
            "ODDS_TOTAL_LINE_bet365": rng.normal(224, 15, heavy + light).round(),
            "PACE_BEFORE_TEAM_HOME": a,
            "OFF_RATING_BEFORE_TEAM_HOME": b,
        }
    )
    rows = per_game.loc[per_game.index.repeat(repeats)].copy()
    rows["TIME_TO_MATCH_MIN"] = np.concatenate(
        [[30, 60, 120, 180, 240, 300, 360, 480, 720, 0][:r] for r in repeats]
    )
    rows = rows.reset_index(drop=True)

    pair = ["PACE_BEFORE_TEAM_HOME", "OFF_RATING_BEFORE_TEAM_HOME"]
    assert per_game[pair].corr().iloc[0, 1] < 0.95 < rows[pair].corr().iloc[0, 1]

    cleaned = clean_dataframe_for_training(
        rows, corr_threshold=0.95, repeated_measures=_policy(), verbose=0
    )

    assert all(column in cleaned.columns for column in pair)


# ---------------------------------------------------------------------------
# rotation-depth leakage
# ---------------------------------------------------------------------------


def _rotation_leak_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A frame carrying the leaked families, their innocent look-alikes, and a
    plain feature that must survive."""
    out = frame.copy()
    out["N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME"] = 10.0
    out["N_ACTIVE_PLAYERS_BEFORE_TEAM_AWAY"] = 11.0
    out["TOTAL_NON_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"] = 90.0
    out["TOTAL_NON_INJURED_PLAYER_TS_PCT_BEFORE_TEAM_AWAY"] = 0.55
    # Must NOT be dropped: built from each player's last game strictly before
    # this one, so it carries nothing from tonight's box score.
    out["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"] = 12.0
    out["N_INJURED_PLAYERS_BEFORE_TEAM_HOME"] = 3.0
    return out


def test_cleaning_drops_rotation_depth_leakage_columns(frame):
    """N_ACTIVE_PLAYERS and TOTAL_NON_INJURED_PLAYER_* aggregate over the players
    who logged minutes in the game being predicted, so they encode how the game
    went. They reached the feature matrix for months without erroring."""
    cleaned = clean_dataframe_for_training(
        _rotation_leak_frame(frame), verbose=0, keep_all_cols=True
    )

    assert "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME" not in cleaned.columns
    assert "N_ACTIVE_PLAYERS_BEFORE_TEAM_AWAY" not in cleaned.columns
    assert "TOTAL_NON_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME" not in cleaned.columns
    assert "TOTAL_NON_INJURED_PLAYER_TS_PCT_BEFORE_TEAM_AWAY" not in cleaned.columns


def test_cleaning_keeps_the_lagged_injured_columns(frame):
    """The prefix must not swallow TOTAL_INJURED_PLAYER_* or N_INJURED_PLAYERS:
    those resolve each player's last game BEFORE this one and are legitimate."""
    cleaned = clean_dataframe_for_training(
        _rotation_leak_frame(frame), verbose=0, keep_all_cols=True
    )

    assert "TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME" in cleaned.columns
    assert "N_INJURED_PLAYERS_BEFORE_TEAM_HOME" in cleaned.columns


def test_injured_availability_standard_error_keeps_unknown_precision_as_nan():
    """An unknown standard error is not a neutral zero-sized injury effect."""
    effect_col = "TOP3_INJURED_AVAILABILITY_EFFECT_HOME_MEAN_TOTAL_POINTS"
    uncertainty_col = "TOP3_INJURED_AVAILABILITY_EFFECT_HOME_MEAN_SE_TOTAL_POINTS"
    raw = pd.DataFrame(
        {
            "TOTAL_POINTS": [220.0, 221.0],
            "ODDS_TOTAL_LINE_bet365": [219.5, 220.5],
            effect_col: [np.nan, 2.0],
            uncertainty_col: [np.nan, 3.0],
        }
    )

    cleaned = apply_missing_policy(raw)

    assert cleaned[effect_col].tolist() == [0.0, 2.0]
    assert pd.isna(cleaned.loc[0, uncertainty_col])
    assert cleaned.loc[1, uncertainty_col] == 3.0


def test_keep_columns_cannot_rescue_a_rotation_leak(frame):
    """keep_columns is a caller-supplied field. A leakage guard it can switch
    off is the arrangement that let this ship in the first place."""
    cleaned = clean_dataframe_for_training(
        _rotation_leak_frame(frame),
        verbose=0,
        keep_all_cols=True,
        keep_columns=["N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME"],
    )

    assert "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME" not in cleaned.columns


def test_rotation_leak_drop_is_recorded_in_the_report(frame):
    _, report = clean_dataframe_for_training(
        _rotation_leak_frame(frame),
        verbose=0,
        keep_all_cols=True,
        return_report=True,
    )

    entry = report.why_dropped("N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME")
    assert entry is not None and entry["step"] == "rotation_leak"
    assert report.why_dropped("TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME") is None


def test_rotation_leak_runs_before_correlation_pruning(frame):
    """Ordering is load-bearing: a leaked column that wins a correlation pair
    evicts the legitimate feature it duplicates, so the surviving frame is
    shaped by a column that is about to be deleted anyway."""
    leaky = _rotation_leak_frame(frame)
    # Independent of every other column, so the only correlation pair in the
    # frame is (leak, twin) and nothing else can decide the twin's fate.
    signal = np.random.default_rng(3).normal(10, 2, len(frame))
    leaky["N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME"] = signal
    leaky["ROTATION_TWIN_BEFORE_TEAM_HOME"] = signal * 2.0

    cleaned = clean_dataframe_for_training(
        leaky,
        corr_threshold=0.95,
        corr_threshold_overrides={},
        verbose=0,
        return_report=False,
    )

    assert "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME" not in cleaned.columns
    assert "ROTATION_TWIN_BEFORE_TEAM_HOME" in cleaned.columns
