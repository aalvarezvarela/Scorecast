"""
Module for cleaning dataframes before training.

This module provides functions to:
- Perform basic data filtering and validation
- Remove low-quality columns (high NaN %, ID columns, string columns)
- Detect and remove duplicate or highly similar columns
- Remove constant columns
- Apply missing data policy
"""

import numpy as np
import pandas as pd
from nba_ou.config.leakage import rotation_leak_columns
from nba_ou.config.odds_columns import resolve_main_total_line_col
from nba_ou.data_processing.missing_data.cleaning_report import CleaningReport
from nba_ou.data_processing.missing_data.column_redundancy import (
    KeepPreference,
    RepeatedMeasuresRedundancy,
    find_identical_groups,
    rank_columns,
    representative_row_index,
    select_correlated_columns_to_drop,
)
from nba_ou.data_processing.missing_data.handle_missing_data import (
    TARGET_COL as TARGET_COLUMN,
)
from nba_ou.data_processing.missing_data.handle_missing_data import (
    apply_missing_policy,
)

#: Correlation thresholds applied to columns whose name CONTAINS the key,
#: overriding ``corr_threshold`` for those columns only.
#:
#: Odds features are the ones this problem is actually about, so near-duplicate
#: odds columns are worth keeping where near-duplicate everything-else is not.
#: The old single 0.995 threshold had exactly the opposite effect: of the 104
#: columns it dropped on ``training_data_2_0_20260819.csv``, 83 were odds-derived
#: and only 21 were not, because odds features are the most internally redundant
#: block in the frame (the same line quoted by seven books).
#:
#: Matching is by substring, not by prefix, and that is load-bearing here: the
#: 192 ``DIFF_FROM_ODDS_LINE_*`` columns contain ``ODDS_`` without starting with
#: it, so they take the tolerant threshold too. That is intended -- they are
#: rolling averages of total-minus-line, market features by any reading (see the
#: note on ``DIFF_FROM_`` in nba_ou.config.odds_columns) -- but it is worth
#: stating, since it is where the odds tolerance actually bites: 50 of the 212
#: columns dropped at the current settings are DIFF_FROM_ODDS_. To prune them at
#: the general threshold instead, anchor the pattern list yourself rather than
#: relying on this default.
#:
#: Measured column survival on that dataset, from 1,578 numeric columns:
#:
#:     thresholds                 dropped   ODDS_   DIFF_FROM_   other   survive
#:     0.995 everywhere (old)         104      77            6      21     1,474
#:     0.95 everywhere                425     209          146      70     1,153
#:     odds 0.995 / other 0.95        153      77            6      70     1,425
#:     odds 0.99  / other 0.95        212      92           50      70     1,366  <-
#:
#: 0.99 rather than 0.995 for odds: at r in [0.99, 0.995] two columns share
#: 98-99% of their variance, which is not a distinction worth 59 columns. Note
#: what moving this number does and does not do -- the "other" count is 70 in
#: both rows, because only odds-derived columns take this threshold.
DEFAULT_CORR_THRESHOLD_OVERRIDES: dict[str, float] = {"ODDS_": 0.99}


def _get_cols_matching_patterns(
    df: pd.DataFrame, patterns: list[str] | None
) -> set[str]:
    """
    Return dataframe columns whose names contain any of the provided patterns.

    Matching is case-insensitive. Empty patterns are ignored.
    """
    if not patterns:
        return set()

    normalized_patterns = [pattern.upper() for pattern in patterns if pattern]
    if not normalized_patterns:
        return set()

    return {
        col
        for col in df.columns
        if any(pattern in col.upper() for pattern in normalized_patterns)
    }


def find_season_gated_columns(
    df: pd.DataFrame, *, season_col: str, max_spread: float
) -> dict[str, str]:
    """Columns whose AVAILABILITY identifies the season.

    Returns ``{column: reason}`` for every column whose per-season NaN rate
    varies by more than ``max_spread`` percentage points between its best and
    worst season.

    This is the rule that actually captures "drop the columns that block the
    old seasons", and it is not the same thing as a NaN threshold. A plain
    threshold is computed over the whole window, so a column that is 100% NaN
    for two seasons and 0.8% for five averages to ~19% -- under any threshold
    worth setting, while still being the thing that gates those two seasons.
    Measured on training_data_2_0_20260819.csv at a 2019 floor, no value of
    ``nan_threshold`` separates the two groups: 50 and 40 drop nothing at all,
    and 15 catches only 200 of 272 offenders while taking 14 innocent columns
    with it.

    Spread catches it because the shape is a step, not a rate. It is also what
    the leakage concern actually is: a column present for one season and absent
    for another lets a model recover the season from availability alone, which
    is why dropping such columns from the OLD seasons only would be useless --
    the recent games have to lose them too.

    Measured, same dataset, floor 2019, ``max_spread=90``: 213 columns, being
    200 public-betting and 13 betmgm price columns (100% absent in 2020, ~0%
    from 2021). The betmgm ones are invisible to a hand-written list of
    public-betting substrings, which is the case for computing this rather than
    naming names.
    """
    if season_col not in df.columns:
        raise KeyError(
            f"Cannot find season-gated columns: {season_col!r} is not in the "
            "frame. Set cleaning.max_seasonal_nan_spread to None to skip this "
            "step, or point cleaning.season_col at the right column."
        )
    seasons = df[season_col]
    if seasons.nunique(dropna=True) < 2:
        # One season cannot gate anything, and this is the normal case on the
        # same-day prediction path.
        return {}

    per_season_nan = df.groupby(season_col).apply(
        lambda group: group.isna().mean() * 100, include_groups=False
    )
    spread = per_season_nan.max() - per_season_nan.min()

    gated = {}
    for column, value in spread.items():
        if pd.notna(value) and value > max_spread:
            worst = per_season_nan[column].idxmax()
            best = per_season_nan[column].idxmin()
            gated[str(column)] = (
                f"availability varies {value:.1f}pp across seasons "
                f"({per_season_nan.loc[worst, column]:.1f}% NaN in {worst}, "
                f"{per_season_nan.loc[best, column]:.1f}% in {best})"
            )
    return gated


#: Total below which a final score is not a basketball game but a broken box
#: score. The lowest total in the 2.0 build is 152 and the lowest ever recorded
#: in a modern NBA game is comfortably above this, so anything at or under it is
#: a data fault rather than a low-scoring night.
MIN_PLAUSIBLE_TOTAL_POINTS = 130

#: Likewise for the posted line. A book does not hang a total this low.
MIN_PLAUSIBLE_TOTAL_LINE = 100


def basic_cleaning(
    df: pd.DataFrame,
    verbose: int = 1,
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """
    Perform basic cleaning and filtering on the training dataframe.

    This function:
    - Filters out games with an implausibly low final total (<= 130)
    - Removes rows with missing main total line (ODDS_TOTAL_LINE_<main_book>)
    - Filters out games with unrealistic betting lines (<= 100)

    The totals filter is NaN-safe, and that is not incidental: this function is
    also what nba_ou.prediction.prediction cleans same-day games with, and those
    have no final score yet. A plain ``df[df[TARGET] > 130]`` would compare
    against NaN, evaluate False for every scheduled game, and silently return an
    empty frame -- deleting the daily prediction run rather than failing it.

    Args:
        df (pd.DataFrame): Training dataframe to clean
        verbose (int): Verbosity level (0=silent, 1=basic, 2=detailed). Default: 1
        report (CleaningReport | None): Collects what was removed and why.

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Copy before the dtype casts below: without this they land on the CALLER's
    # frame, so cleaning the same dataframe twice does not do the same thing
    # twice.
    df = df.copy()

    for holiday_col in ("IS_US_HOLIDAY_BEFORE", "IS_US_HOLIDAY"):
        if holiday_col in df.columns:
            df[holiday_col] = (
                df[holiday_col]
                .astype("Int64")  # ensures proper numeric handling
                .astype("boolean")  # pandas nullable boolean
            )

    initial_rows = len(df)
    if verbose >= 1:
        print(f"Starting basic cleaning with {initial_rows} rows")

    if TARGET_COLUMN in df.columns:
        rows_before = len(df)
        totals = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        # Keep unknown totals: a scheduled game has no score yet, and "unknown"
        # is not "implausible".
        df = df[~(totals.notna() & (totals <= MIN_PLAUSIBLE_TOTAL_POINTS))]
        if report is not None:
            report.record_rows(
                step="basic_cleaning.implausible_total",
                before=rows_before,
                after=len(df),
                reason=f"{TARGET_COLUMN} <= {MIN_PLAUSIBLE_TOTAL_POINTS}",
            )
        if verbose >= 2:
            print(
                f"Removed {rows_before - len(df)} rows with "
                f"{TARGET_COLUMN} <= {MIN_PLAUSIBLE_TOTAL_POINTS}"
            )

    main_total_line = resolve_main_total_line_col(df)
    if main_total_line is None:
        raise ValueError("No ODDS_TOTAL_LINE_<book> column found for basic cleaning.")

    # Count and report NaNs in the configured main total line column
    nans = df[main_total_line].isna().sum()
    if verbose >= 2:
        print(f"Number of NaNs in {main_total_line}: {nans}")

    # Drop rows with missing odds data
    rows_before = len(df)
    df = df.dropna(subset=[main_total_line])
    if report is not None:
        report.record_rows(
            step="basic_cleaning.missing_total_line",
            before=rows_before,
            after=len(df),
            reason=f"{main_total_line} is NaN",
        )
    if verbose >= 2:
        print(f"Removed {nans} rows with NaN in {main_total_line}")

    # Filter out unrealistic betting lines
    rows_before = len(df)
    df = df[df[main_total_line] > MIN_PLAUSIBLE_TOTAL_LINE].copy()
    if report is not None:
        report.record_rows(
            step="basic_cleaning.implausible_total_line",
            before=rows_before,
            after=len(df),
            reason=f"{main_total_line} <= {MIN_PLAUSIBLE_TOTAL_LINE}",
        )
    if verbose >= 2:
        print(
            f"Removed {rows_before - len(df)} rows with "
            f"{main_total_line} <= {MIN_PLAUSIBLE_TOTAL_LINE}"
        )

    if verbose >= 1:
        print(f"Basic cleaning complete: {len(df)} rows remaining\n")

    return df


def advanced_column_cleaning(
    df: pd.DataFrame,
    nan_threshold: float = 50.0,
    corr_threshold: float = 0.99,
    keep_columns: list[str] | None = None,
    exclude_cols_containing: list[str] | None = None,
    keep_all_cols: bool = False,
    corr_threshold_overrides: dict[str, float] | None = None,
    max_seasonal_nan_spread: float | None = None,
    season_col: str = "SEASON_YEAR",
    repeated_measures: RepeatedMeasuresRedundancy | None = None,
    representative_index: pd.Index | None = None,
    verbose: int = 1,
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """
    Perform advanced column cleaning on the training dataframe.

    This function:
    - Removes columns containing strings in every value
    - Removes columns with 'ID' in the name
    - Removes columns with high NaN percentage (configurable)
    - Removes duplicate columns and absolute-value matches
    - Removes columns highly correlated with another column (unless keep_all_cols=True)
    - Removes columns with constant values (unless keep_all_cols=True)

    The last three all resolve *which* member of a redundant set survives via
    nba_ou.data_processing.missing_data.column_redundancy, by explicit
    preference (protected > fewer NaNs > main book > canonical name) rather than
    by column order. ``keep_columns`` is honoured by every step, including these
    -- it previously protected columns only from the string/ID/NAME/high-NaN/
    constant steps, so a protected column could still be lost to correlation
    pruning.

    Args:
        df (pd.DataFrame): Training dataframe to clean
        nan_threshold (float): Percentage threshold for NaN values above which
            a column will be removed (e.g., 40.0 means 40%). Default: 50.0
        corr_threshold (float): Correlation threshold above which columns will be considered highly similar
            and one will be removed. Default: 0.99
        keep_columns (list[str] | None): List of column names to always keep regardless of type or quality.
            Useful for preserving date columns or other important non-numeric columns. Default: None
        exclude_cols_containing (list[str] | None): Substrings used to drop matching columns before
            the rest of the cleaning logic runs. Matching is case-insensitive. Default: None
        keep_all_cols (bool): If True, only drops ID, NAME, and string columns; keeps all others
            (high-NaN, constant, duplicate, correlated, absolute matches). Default: False
        corr_threshold_overrides (dict[str, float] | None): Per-substring correlation
            thresholds overriding ``corr_threshold`` for matching columns; a pair is
            judged against the more tolerant of its two columns. Pass ``{}`` for a
            single threshold everywhere. Default: DEFAULT_CORR_THRESHOLD_OVERRIDES
        max_seasonal_nan_spread (float | None): Drop columns whose per-season NaN
            rate varies by more than this many percentage points -- their
            availability identifies the season. None disables the step. See
            find_season_gated_columns.
        repeated_measures (RepeatedMeasuresRedundancy | None): Policy for a frame
            holding several rows per game. Exempts snapshot/market columns from
            correlation pruning entirely and judges the rest on one row per
            game. None (the default) leaves correlation pruning to run over
            every row and every column, which is right for one row per game.
        representative_index (pd.Index | None): The one-row-per-group labels the
            policy is applied over. Passed in rather than derived here because
            the columns identifying a group -- GAME_ID above all -- are dropped
            by step 2 of this very function, long before the correlation step
            needs them.
        season_col (str): Column holding the season, for the check above.
        verbose (int): Verbosity level (0=silent, 1=basic, 2=detailed). Default: 1

    Returns:
        pd.DataFrame: Dataframe with cleaned columns
    """
    if corr_threshold_overrides is None:
        corr_threshold_overrides = DEFAULT_CORR_THRESHOLD_OVERRIDES
    initial_cols = len(df.columns)
    if verbose >= 1:
        print(f"Starting advanced column cleaning with {initial_cols} columns")
    columns_to_drop = set()

    # Set of columns to always keep
    keep_columns_set = set(keep_columns) if keep_columns else set()
    if keep_columns_set and verbose >= 2:
        print(f"\nProtected columns (will not be removed): {sorted(keep_columns_set)}")

    cols_matching_patterns = _get_cols_matching_patterns(df, exclude_cols_containing)
    cols_to_exclude = sorted(cols_matching_patterns - keep_columns_set)
    if cols_to_exclude:
        if verbose >= 2:
            print(f"\nDropping columns matching exclude patterns: {cols_to_exclude}")
        df = df.drop(columns=cols_to_exclude)
        if report is not None:
            report.drop_columns(
                cols_to_exclude,
                step="exclude_patterns",
                reason=f"name matches exclude_cols_containing={exclude_cols_containing}",
            )
    elif cols_matching_patterns and verbose >= 2:
        print(
            "\nColumns matching exclude patterns were preserved because they are protected: "
            f"{sorted(cols_matching_patterns)}"
        )

    # 1. Remove columns that are purely string (object/string dtype and all non-null values are str)
    if verbose >= 2:
        print("\n1. Checking for pure string columns...")

    string_cols = []

    for col in df.columns:
        # Skip protected columns
        if col in keep_columns_set:
            continue
        dtype = df[col].dtype

        # Only object / string columns are candidates
        if dtype not in ("object", "string"):
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            # column is all NaN → treat as useless string-like column
            string_cols.append(col)
            continue

        # Drop if ALL non-null values are strings
        if non_null.map(type).eq(str).all():
            string_cols.append(col)

    if string_cols:
        if verbose >= 2:
            print(f"   Removing {len(string_cols)} pure string columns:")
            for c in string_cols:
                print(f"      - {c}")
        columns_to_drop.update(string_cols)
        if report is not None:
            report.drop_columns(
                string_cols, step="string_columns", reason="all non-null values are str"
            )
    elif verbose >= 2:
        print("   No pure string columns to remove")

    # 2. Remove columns containing 'ID' in the name
    if verbose >= 2:
        print("\n2. Checking for ID columns...")
    id_cols = [
        col
        for col in df.columns
        if "_ID" in col.upper() and col not in keep_columns_set
    ]
    if id_cols:
        if verbose >= 2:
            print(f"   Removing {len(id_cols)} _ID columns: {id_cols}")
        columns_to_drop.update(id_cols)
        if report is not None:
            report.drop_columns(
                id_cols, step="id_columns", reason="name contains '_ID'"
            )
    elif verbose >= 2:
        print("   No ID columns to remove")

    # 3. Remove columns containing '_NAME' in the name
    if verbose >= 2:
        print("\n3. Checking for _NAME columns...")
    name_cols = [
        col
        for col in df.columns
        if "_NAME" in col.upper() and col not in keep_columns_set
    ]
    if name_cols:
        if verbose >= 2:
            print(f"   Removing {len(name_cols)} _NAME columns: {name_cols}")
        columns_to_drop.update(name_cols)
        if report is not None:
            report.drop_columns(
                name_cols, step="name_columns", reason="name contains '_NAME'"
            )
    elif verbose >= 2:
        print("   No _NAME columns to remove")

    # 4. Remove columns with high NaN values (configurable)
    if verbose >= 2:
        print(f"\n4. Checking for high-NaN columns (>{nan_threshold}%)...")
    high_nan_cols = []
    # An empty frame has no NaN *proportion* -- 0/0 is not 100%. Guarding here
    # rather than dividing keeps the step from emitting a RuntimeWarning and
    # then quietly dropping nothing because every comparison against NaN is
    # False.
    if len(df):
        for col in df.columns:
            if col in columns_to_drop or col in keep_columns_set:
                continue
            nan_pct = df[col].isna().sum() / len(df) * 100
            if nan_pct > nan_threshold:
                high_nan_cols.append((col, nan_pct))

    if high_nan_cols and not keep_all_cols:
        if verbose >= 2:
            print(
                f"   Removing {len(high_nan_cols)} columns with >{nan_threshold}% NaN:"
            )
            for col, pct in high_nan_cols:
                print(f"      - {col}: {pct:.2f}% NaN")
        columns_to_drop.update(col for col, _ in high_nan_cols)
        if report is not None:
            report.drop_columns_with_reasons(
                {
                    col: f"{pct:.2f}% NaN, above nan_threshold={nan_threshold}"
                    for col, pct in high_nan_cols
                },
                step="high_nan_columns",
            )
    elif verbose >= 2:
        if keep_all_cols:
            print("   Skipping high-NaN column removal (keep_all_cols=True)")
        else:
            print("   No high-NaN columns to remove")

    # 4b. Remove columns whose availability identifies the season
    if max_seasonal_nan_spread is not None and not keep_all_cols:
        if verbose >= 2:
            print(
                f"\n4b. Checking for season-gated columns "
                f"(NaN rate varying >{max_seasonal_nan_spread}pp across seasons)..."
            )
        gated = find_season_gated_columns(
            df.drop(columns=list(columns_to_drop), errors="ignore"),
            season_col=season_col,
            max_spread=max_seasonal_nan_spread,
        )
        gated = {
            col: why
            for col, why in gated.items()
            if col not in keep_columns_set and col != season_col
        }
        if gated:
            if verbose >= 2:
                print(f"   Removing {len(gated)} season-gated columns:")
                for col, why in list(gated.items())[:20]:
                    print(f"      - {col}: {why}")
            columns_to_drop.update(gated)
            if report is not None:
                report.drop_columns_with_reasons(gated, step="season_gated_columns")
        elif verbose >= 2:
            print("   No season-gated columns to remove")

    # 5. Remove columns with constant values (same value in every row)
    if verbose >= 2:
        print("\n5. Checking for constant columns...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping constant column removal (keep_all_cols=True)")
    else:
        constant_cols = []
        for col in df.columns:
            if col in columns_to_drop or col in keep_columns_set:
                continue
            if df[col].nunique(dropna=False) == 1:
                constant_cols.append(col)

        if constant_cols:
            if verbose >= 2:
                print(
                    f"   Removing {len(constant_cols)} constant columns: {constant_cols}"
                )
            columns_to_drop.update(constant_cols)
            if report is not None:
                report.drop_columns(
                    constant_cols,
                    step="constant_columns",
                    reason="single distinct value (NaN counted as a value)",
                )
        elif verbose >= 2:
            print("   No constant columns to remove")

    # Drop the columns identified so far before checking for duplicates
    df = df.drop(columns=list(columns_to_drop))

    # Steps 6-7 both answer "several columns carry the same information, which
    # one survives?" and both answer it the same way: rank by preference, keep
    # the best. See column_redundancy.rank_columns.
    preference = KeepPreference.build(protected=sorted(keep_columns_set))

    # 6. Exact duplicates and absolute-value matches
    if verbose >= 2:
        print("\n6. Checking for duplicate and absolute-value-match columns...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping duplicate column removal (keep_all_cols=True)")
    else:
        for label, absolute in (("duplicate", False), ("absolute value match", True)):
            numeric = df.select_dtypes(include=[np.number])
            groups = find_identical_groups(numeric, absolute=absolute)
            cols_to_remove = []
            reasons: dict[str, str] = {}
            for group in groups:
                ranked = rank_columns(numeric, group, preference)
                keeper, losers = ranked[0], ranked[1:]
                cols_to_remove.extend(losers)
                marker = "abs" if absolute else ""
                for loser in losers:
                    reasons[loser] = f"{marker}({loser}) == {marker}({keeper})"
                    if verbose >= 2:
                        print(f"      - {reasons[loser]}")
            if cols_to_remove:
                if verbose >= 2:
                    print(f"   Removing {len(cols_to_remove)} {label} columns")
                df = df.drop(columns=cols_to_remove)
                if report is not None:
                    report.drop_columns_with_reasons(
                        reasons,
                        step=(
                            "absolute_value_match" if absolute else "duplicate_columns"
                        ),
                    )
            elif verbose >= 2:
                print(f"   No {label} columns found")

    # 7. Highly correlated columns
    if verbose >= 2:
        print("\n7. Checking for highly correlated columns...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping highly correlated column removal (keep_all_cols=True)")
    else:
        numeric = df.select_dtypes(include=[np.number])
        step_name = "correlated_columns"
        view_note = ""

        if repeated_measures is not None:
            # Snapshot/market columns are exempt outright -- see
            # RepeatedMeasuresRedundancy. Everything else is judged on one row
            # per game, so a feature copied onto ten snapshots weighs the same
            # as it would on the closing-line dataset, which holds one row per
            # game and is what these thresholds were calibrated against.
            exempt = [c for c in numeric.columns if repeated_measures.is_exempt(c)]
            candidates = [c for c in numeric.columns if c not in set(exempt)]
            numeric = numeric[candidates]
            if representative_index is not None:
                numeric = numeric.loc[representative_index]
            step_name = "correlated_columns_one_row_per_group"
            view_note = (
                f" [one row per {repeated_measures.group_col}: "
                f"{len(numeric)} of {len(df)} rows; "
                f"{len(exempt)} snapshot columns exempt]"
            )
            if verbose >= 2:
                print(f"  {view_note.strip()}")

        if numeric.shape[1] > 1:
            cols_to_remove, decisions = select_correlated_columns_to_drop(
                numeric,
                default_threshold=corr_threshold,
                overrides=corr_threshold_overrides,
                preference=preference,
            )
            if cols_to_remove:
                if verbose >= 2:
                    thresholds_note = (
                        f" (default {corr_threshold}"
                        + (
                            f", overrides {corr_threshold_overrides}"
                            if corr_threshold_overrides
                            else ""
                        )
                        + ")"
                    )
                    print(
                        f"   Found {len(decisions)} redundant columns{thresholds_note}:"
                    )
                    for dropped_col, kept_col, r in decisions:
                        print(
                            f"      - {dropped_col} ~ {kept_col} (correlation: {r:.4f})"
                        )
                    print(f"   Removing {len(cols_to_remove)} correlated columns")
                df = df.drop(columns=cols_to_remove)
                if report is not None:
                    report.drop_columns_with_reasons(
                        {
                            dropped_col: (
                                f"|r|={r:.4f} with kept column {kept_col}{view_note}"
                            )
                            for dropped_col, kept_col, r in decisions
                        },
                        step=step_name,
                    )
            elif verbose >= 2:
                print("   No highly correlated columns found")
        elif verbose >= 2:
            print("   No highly correlated columns found")

    final_cols = len(df.columns)
    if verbose >= 1:
        print(
            f"\nAdvanced column cleaning complete: {initial_cols} → {final_cols} columns "
            f"({initial_cols - final_cols} removed)\n"
        )

    return df


def _normalize_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable columns to numpy-backed float, pd.NA to np.nan.

    This replaces a loop that intended to do the same thing by calling
    ``fillna(np.nan)`` on every numeric column. That was a verified no-op --
    float64 NaN is already NaN and int64 cannot hold one -- and it could not
    have worked anyway, because ``select_dtypes(include=[np.number])`` does not
    match the nullable ``boolean`` dtype, which is the only nullable dtype the
    pipeline actually creates: ``basic_cleaning`` casts IS_US_HOLIDAY* to it.

    That combination is a live trap rather than a tidiness issue. A nullable
    boolean holding pd.NA converts via ``to_numpy()`` to dtype ``object``, which
    XGBoost rejects outright. It has not fired only because the holiday flag
    currently has no missing values; the first season it does, training breaks
    at fit time with an error pointing nowhere near here.

    pandas Categorical is deliberately left alone -- XGBoost consumes it
    natively under ``enable_categorical``, which is what
    ``categorical_team_encoding`` relies on.
    """
    nullable = [
        column
        for column in df.columns
        if isinstance(df[column].dtype, pd.api.extensions.ExtensionDtype)
        and getattr(df[column].dtype, "kind", "O") in "biufc"
    ]
    if not nullable:
        return df
    return df.assign(**{column: df[column].astype("float64") for column in nullable})


def clean_dataframe_for_training(
    df: pd.DataFrame,
    nan_threshold: float = 5.0,
    corr_threshold: float = 0.995,
    max_na_per_row: int = -1,
    create_missing_flags: bool = False,
    keep_columns: list[str] | None = None,
    exclude_cols_containing: list[str] | None = None,
    keep_all_cols: bool = False,
    corr_threshold_overrides: dict[str, float] | None = None,
    max_seasonal_nan_spread: float | None = None,
    season_col: str = "SEASON_YEAR",
    repeated_measures: RepeatedMeasuresRedundancy | None = None,
    row_balance_group_col: str | None = None,
    verbose: int = 1,
    strict_mode: int = -1,
    strict_mode_exclude_cols: list[str] | None = None,
    return_report: bool = False,
) -> tuple[pd.DataFrame, CleaningReport] | pd.DataFrame:
    """
    Complete cleaning pipeline for training dataframe.

    Applies:
    1. Basic row filtering
    2. Advanced column cleaning
    3. Missing data policy (drop critical rows, zero-fill, infer, fallback to medians)
    4. Optional row filtering based on remaining NaN counts

    Note on the two row-NaN mechanisms, which overlap and are easy to confuse:
    ``max_na_per_row`` counts NaNs across ALL columns and is the one in normal
    use; ``strict_mode`` counts them across all columns except
    ``strict_mode_exclude_cols`` and is off by default. Setting both applies
    both, in that order.

    Args:
        df (pd.DataFrame): Raw training dataframe
        nan_threshold (float): Percentage threshold for NaN values above which
            a column will be removed. Default: 5.0
        max_na_per_row (int): Maximum number of NaN values allowed per row. Rows exceeding this
            threshold will be dropped. Use -1 to disable, 0 to drop rows with any NaN. Default: -1
        exclude_cols_containing (list[str] | None): Substrings used to drop matching columns before
            the rest of the cleaning pipeline runs. Matching is case-insensitive. Default: None
        keep_all_cols (bool): If True, only drops ID, NAME, and string columns; keeps all others.
            Default: False
        corr_threshold_overrides (dict[str, float] | None): Per-substring correlation
            thresholds overriding ``corr_threshold`` for matching columns. Defaults to
            DEFAULT_CORR_THRESHOLD_OVERRIDES, which holds odds features to a more
            tolerant threshold than everything else. Pass ``{}`` to disable.
        verbose (int): Verbosity level (0=silent, 1=basic, 2=detailed). Default: 1
        strict_mode (int): Maximum number of columns allowed to have NaN values (excluding strict_mode_exclude_cols).
            Use 0 for no NaN columns allowed, -1 or any negative value to disable the check. Default: -1
        strict_mode_exclude_cols (list[str] | None): Columns to exclude from strict mode check.
            Defaults to ['MATCHUP_TEAM_HOME', 'TOTAL_POINTS'] if None.
        repeated_measures (RepeatedMeasuresRedundancy | None): Policy for a frame
            holding several rows per game -- currently the intermediate-line
            dataset, one row per (game, snapshot). Snapshot/market columns are
            exempt from correlation pruning altogether; every other column is
            judged on ONE ROW PER GAME, so a historical feature copied onto ten
            snapshots carries the weight of one observation rather than ten, and
            is judged exactly as the closing-line dataset would judge it. Exact
            duplicate and absolute-value-match detection stay global. None (the
            default) is correct for one row per game. Default: None
        row_balance_group_col (str | None): Column whose values the row filters
            should be reported against, e.g. ``TIME_TO_MATCH_MIN`` on the
            intermediate-line dataset. Purely a record -- see
            ``CleaningReport.record_group_survival``. Ignored when the column is
            absent, so one call site serves both datasets. Default: None
        return_report (bool): If True, return ``(df, CleaningReport)`` instead of
            just the frame. The report records which step dropped each column and
            why, so "where did this feature go?" is answerable without re-running.

    Returns:
        pd.DataFrame, or (pd.DataFrame, CleaningReport) when return_report=True
    """
    if verbose >= 1:
        print("=" * 80)
        print("STARTING DATAFRAME CLEANING PIPELINE")
        print("=" * 80)

    report = CleaningReport(columns_in=len(df.columns), rows_in=len(df))

    # Taken by INDEX, not by re-reading the column at the end: the grouping
    # column is an ordinary feature and can itself be dropped by column
    # cleaning, and a report that quietly stopped being produced whenever that
    # happened would be worse than no report.
    group_before: pd.Series | None = None
    if row_balance_group_col and row_balance_group_col in df.columns:
        group_before = df[row_balance_group_col]

    # Same reason, and more pressing: GAME_ID is dropped by the _ID step, which
    # runs well before the correlation step that needs it. Captured here while
    # it still exists, and resolved to row labels once basic_cleaning has
    # finished removing rows.
    redundancy_group: pd.Series | None = None
    redundancy_snapshot: pd.Series | None = None
    if repeated_measures is not None:
        if repeated_measures.group_col not in df.columns:
            raise KeyError(
                f"repeated_measures.group_col {repeated_measures.group_col!r} is "
                "not in the frame; redundancy cannot be judged one row per "
                "group without it."
            )
        redundancy_group = df[repeated_measures.group_col]
        if (
            repeated_measures.snapshot_col
            and repeated_measures.snapshot_col in df.columns
        ):
            redundancy_snapshot = df[repeated_measures.snapshot_col]

    # Rotation-depth leaks go first, before any other step measures the frame.
    # Not merely cosmetic ordering: the correlation step drops a column when it
    # duplicates another, so leaving these in until later lets a leaked column
    # win a pair and evict the legitimate feature it correlates with -- the
    # frame would then be shaped by a column that is about to be deleted.
    #
    # Unconditional, and deliberately NOT rescuable via keep_columns: these are
    # functions of the game being predicted (see ROTATION_LEAK_COLUMN_PREFIXES),
    # and a guard a caller can switch off is the arrangement that produced the
    # 66.7%-against-the-closing-spread run in the first place. Dropping rather
    # than raising, so the CSVs already built keep working.
    rotation_leaks = rotation_leak_columns(df.columns)
    if rotation_leaks:
        if verbose >= 1:
            print(
                f"Dropping {len(rotation_leaks)} rotation-depth leakage column(s) "
                "built from the predicted game's own box score: "
                f"{rotation_leaks}"
            )
        df = df.drop(columns=rotation_leaks)
        report.drop_columns(
            rotation_leaks,
            step="rotation_leak",
            reason=(
                "aggregates over the players who logged minutes in THIS game "
                "(nba_ou.config.leakage.ROTATION_LEAK_COLUMN_PREFIXES)"
            ),
        )

    # The exclude patterns used to be applied here AND again inside
    # advanced_column_cleaning. Doing it once here and passing None onward keeps
    # a single record of the drop in the report, and saves a second scan of
    # every column name for a step that could no longer match anything.
    cols_matching_patterns = _get_cols_matching_patterns(df, exclude_cols_containing)
    cols_to_exclude = sorted(
        cols_matching_patterns - (set(keep_columns) if keep_columns else set())
    )
    if cols_to_exclude and verbose >= 2:
        print(
            "Dropping columns matching exclude patterns before cleaning: "
            f"{cols_to_exclude}"
        )
    if cols_to_exclude:
        df_cleaned = df.drop(columns=cols_to_exclude)
        report.drop_columns(
            cols_to_exclude,
            step="exclude_patterns",
            reason=f"name matches exclude_cols_containing={exclude_cols_containing}",
        )
    else:
        df_cleaned = df

    # Basic cleaning (copies internally, so `df` is never mutated)
    df_cleaned = basic_cleaning(df_cleaned, verbose=verbose, report=report)

    # Resolved after basic_cleaning, so a game whose only representative row was
    # just deleted falls back to another of its rows rather than vanishing from
    # the view.
    representative_index: pd.Index | None = None
    if repeated_measures is not None and redundancy_group is not None:
        surviving = df_cleaned.index
        representative_index = representative_row_index(
            redundancy_group.loc[surviving],
            (
                None
                if redundancy_snapshot is None
                else redundancy_snapshot.loc[surviving]
            ),
            target=repeated_measures.target_snapshot,
        )
        if report is not None:
            report.record_redundancy_view(
                group_col=repeated_measures.group_col,
                snapshot_col=repeated_measures.snapshot_col,
                target_snapshot=repeated_measures.target_snapshot,
                rows_in_view=len(representative_index),
                rows_total=len(df_cleaned),
                exempt_columns=[
                    c for c in df_cleaned.columns if repeated_measures.is_exempt(c)
                ],
                snapshots_used=(
                    None
                    if redundancy_snapshot is None
                    else redundancy_snapshot.loc[representative_index]
                    .value_counts()
                    .to_dict()
                ),
            )
        if verbose >= 1:
            print(
                f"\nRedundancy judged on one row per {repeated_measures.group_col}: "
                f"{len(representative_index):,} of {len(df_cleaned):,} rows"
            )

    # Advanced column cleaning
    df_cleaned = advanced_column_cleaning(
        df_cleaned,
        nan_threshold=nan_threshold,
        corr_threshold=corr_threshold,
        keep_columns=keep_columns,
        exclude_cols_containing=None,  # already applied above
        keep_all_cols=keep_all_cols,
        corr_threshold_overrides=corr_threshold_overrides,
        max_seasonal_nan_spread=max_seasonal_nan_spread,
        season_col=season_col,
        repeated_measures=repeated_measures,
        representative_index=representative_index,
        verbose=verbose,
        report=report,
    )

    # Apply missing data policy
    if verbose >= 1:
        print("\nApplying missing data policy...")

    main_total_line = resolve_main_total_line_col(df_cleaned)

    rows_before_policy = len(df_cleaned)
    df_cleaned = apply_missing_policy(
        df_cleaned,
        current_total_line_col=main_total_line,
        create_missing_flags=create_missing_flags,
        keep_all_cols=keep_all_cols,
    )
    report.record_rows(
        step="missing_policy.required_columns",
        before=rows_before_policy,
        after=len(df_cleaned),
        reason="NaN in a column the missing-data policy requires",
    )
    if max_na_per_row >= 0:
        if verbose >= 1:
            if max_na_per_row == 0:
                print("\nDropping rows with any NaN values...")
            else:
                print(f"\nDropping rows with more than {max_na_per_row} NaN values...")

        initial_rows = len(df_cleaned)
        # Count NaN values per row
        na_per_row = df_cleaned.isna().sum(axis=1)
        # Keep rows with NaN count <= threshold
        df_cleaned = df_cleaned[na_per_row <= max_na_per_row]
        report.record_rows(
            step="max_na_per_row",
            before=initial_rows,
            after=len(df_cleaned),
            reason=f"more than {max_na_per_row} NaN values in the row",
        )

        if verbose >= 1:
            print(
                f"Removed {initial_rows - len(df_cleaned)} rows exceeding NaN threshold"
            )

    # Check for remaining NaN values in strict mode
    if strict_mode >= 0:
        # Default exclusions: columns kept for info but not used in model
        if strict_mode_exclude_cols is None:
            strict_mode_exclude_cols = ["MATCHUP_TEAM_HOME", "TOTAL_POINTS"]

        # Get all columns except excluded ones
        cols_to_check = [
            col for col in df_cleaned.columns if col not in strict_mode_exclude_cols
        ]

        # Count NaNs per row (only in non-excluded columns)
        nan_counts_per_row = df_cleaned[cols_to_check].isna().sum(axis=1)

        # Identify rows that exceed the strict_mode threshold
        rows_exceeding_threshold = nan_counts_per_row > strict_mode
        num_rows_exceeding = rows_exceeding_threshold.sum()

        if num_rows_exceeding > 0:
            total_rows = len(df_cleaned)

            # Check if ALL rows exceed the threshold (cannot be fixed)
            if num_rows_exceeding == total_rows:
                # Find which columns contribute to the issue
                nan_counts = df_cleaned[cols_to_check].isna().sum()
                columns_with_nan = nan_counts[nan_counts > 0].sort_values(
                    ascending=False
                )

                error_msg = f"Strict mode: ALL {total_rows} rows have NaNs in more than {strict_mode} columns (cannot be fixed by dropping rows).\n"
                error_msg += "Columns with NaN values:\n"
                for col, count in columns_with_nan.head(10).items():
                    pct = (count / total_rows) * 100
                    error_msg += f"  - {col}: {count} NaN values ({pct:.2f}%)\n"
                if len(columns_with_nan) > 10:
                    error_msg += (
                        f"  ... and {len(columns_with_nan) - 10} more columns\n"
                    )
                raise ValueError(error_msg)

            # Drop rows exceeding the threshold
            initial_rows = len(df_cleaned)

            if verbose >= 1:
                print(
                    f"\nStrict mode: Found {num_rows_exceeding} rows with NaNs in more than {strict_mode} columns"
                )
                if verbose >= 2:
                    # Show distribution of NaN counts per row
                    print("   Distribution of NaN counts per row:")
                    for nan_count in sorted(nan_counts_per_row.unique()):
                        if nan_count > strict_mode:
                            count = (nan_counts_per_row == nan_count).sum()
                            print(
                                f"      {count} rows with {int(nan_count)} NaN columns"
                            )
                print(f"Dropping rows with more than {strict_mode} NaN columns...")

            df_cleaned = df_cleaned[~rows_exceeding_threshold]
            rows_dropped = initial_rows - len(df_cleaned)
            report.record_rows(
                step="strict_mode",
                before=initial_rows,
                after=len(df_cleaned),
                reason=f"NaN in more than {strict_mode} columns",
            )

            if verbose >= 1:
                print(f"Dropped {rows_dropped} rows to meet strict mode requirements")
        else:
            if verbose >= 1:
                excluded_info = (
                    f" (excluding {strict_mode_exclude_cols})"
                    if strict_mode_exclude_cols
                    else ""
                )
                print(
                    f"\nStrict mode check passed: All rows have ≤{strict_mode} NaN columns{excluded_info}"
                )

    df_cleaned = _normalize_nullable_dtypes(df_cleaned)

    report.columns_out = len(df_cleaned.columns)
    report.rows_out = len(df_cleaned)

    if group_before is not None:
        assert row_balance_group_col is not None
        report.record_group_survival(
            group_col=row_balance_group_col,
            before_counts=group_before.value_counts().to_dict(),
            after_counts=group_before.loc[df_cleaned.index].value_counts().to_dict(),
        )
        if verbose >= 1:
            spread = report.group_survival.get("retention_spread_pp")
            print(
                f"\nRow retention across {row_balance_group_col}: "
                f"{spread}pp between the best- and worst-retained group"
            )
            if verbose >= 2:
                for group in report.group_survival["groups"]:
                    print(
                        f"   {group['group']!s:>8}  {group['rows_before']:>6} -> "
                        f"{group['rows_after']:>6}  ({group['retention_pct']}%)"
                    )

    if verbose >= 1:
        print("=" * 80)
        print("CLEANING COMPLETE")
        print(f"Final shape: {df_cleaned.shape}")
        print("=" * 80)

    if return_report:
        return df_cleaned, report
    return df_cleaned
