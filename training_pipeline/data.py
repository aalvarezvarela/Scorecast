"""Data loading, cleaning, and feature/target preparation for training_pipeline.

Wraps nba_ou.data_processing.missing_data.clean_df_for_training.clean_dataframe_for_training
and reuses nba_ou.modeling.meta_learner_training_data._ensure_line_error_column as the
single canonical LINE_ERROR derivation, rather than re-implementing it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.config.leakage import rotation_leak_columns
from nba_ou.config.market_columns import (
    HOME_MARGIN_COL,
    PTS_AWAY_COL,
    PTS_HOME_COL,
    SPREAD_ERROR_COL,
    spread_error,
)
from nba_ou.config.odds_columns import (
    resolve_main_spread_line_col,
    resolve_main_total_line_col,
    spread_line_home_col,
    total_line_col,
)
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from nba_ou.data_processing.missing_data.cleaning_report import CleaningReport
from nba_ou.data_processing.missing_data.column_redundancy import (
    RepeatedMeasuresRedundancy,
)
from nba_ou.modeling.meta_learner_training_data import (
    _ensure_line_error_column as ensure_line_error_column,
)

# Reused, not duplicated: this is documented as the one canonical implementation
# of LINE_ERROR = TOTAL_POINTS - ODDS_TOTAL_LINE_<book> in the repo. The leading
# underscore is a naming convention, not an enforced boundary; duplicating the
# formula here would risk the two definitions drifting apart over time.
from nba_ou.modeling.modeling import tail_n_games

from training_pipeline.config import (
    LEAKING_TARGET_COLUMNS,
    OUTCOME_ONLY_COLUMNS,
    OVER_LABEL_COL,
    SNAPSHOT_COLUMN,
    BaselineConfig,
    CleaningConfig,
    DatasetType,
    ExperimentConfig,
    Market,
    PredictionStrategy,
)
from training_pipeline.diagnostics import (
    PlantedSignalResult,
    build_planted_signal,
    measure_planted_signal,
)

# NOTE: "ODDS_total_line_books_median" (the engineered cross-book median total
# line, per src/nba_ou/data_processing/merged_home_away_data/odds_feature_engeneer.py)
# was the original default candidate for the baseline line column. It is
# deliberately NOT used as a silent default: verified empirically against a
# real archived training CSV (data/train_data/all_odds_training_data_until_20260318.csv)
# that this column's values (range roughly -0.5 to 17) do not correlate with
# TOTAL_POINTS or ODDS_TOTAL_LINE_bet365 (correlation ~0.02) for that snapshot --
# whatever generated it did not produce a points-scale median. Since historical
# CSV snapshots cannot be assumed trustworthy for this column, it is only used
# as the baseline when a caller explicitly opts in via BaselineConfig.line_col.
BOOKMAKER_MEDIAN_LINE_COL = "ODDS_total_line_books_median"

# SNAPSHOT_COLUMN is imported above rather than defined here. It moved to
# training_pipeline.config so the config layer can validate against it, and is
# still importable from this module because cleaning, snapshot_scoring and the
# pipeline all reach for it by this name -- one spelling is what stops the
# cleaner and the scorer grouping by different columns. Its absence from a frame
# is the ordinary case for the closing-line dataset and is not an error.


def compute_file_checksum(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streamed sha256 of a file's contents, as ``sha256:<first 16 hex>``.

    Content-based rather than path-based: training CSVs get regenerated in
    place under the same filename, and a path alone cannot tell you whether
    the bytes behind it changed. Streaming keeps memory flat on the ~400MB
    snapshots; the read costs well under a second.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


def verify_dataset_checksum(path: str | Path, *, expected_checksum: str | None) -> str:
    """Compute the dataset checksum, asserting it matches when one is pinned."""
    actual = compute_file_checksum(path)
    if expected_checksum and actual != expected_checksum:
        raise ValueError(
            f"Dataset checksum mismatch for {path}: expected {expected_checksum}, "
            f"got {actual}. The file's contents changed since this experiment "
            "was defined. Update data.expected_checksum if that was intentional."
        )
    return actual


def load_raw_training_csv(
    csv_path: str | Path, *, date_col: str = "GAME_DATE"
) -> pd.DataFrame:
    """Load a training CSV the same way the example notebooks do: ID-like columns
    forced to str (avoids mixed-type surprises from pandas' dtype inference on
    large sparse columns), GAME_DATE parsed to a plain date string then back to
    datetime for consistent downstream handling.
    """
    csv_path = Path(csv_path)
    header = pd.read_csv(csv_path, nrows=0)
    dtype_dict = {col: str for col in header.columns if "ID" in col.upper()}

    df = pd.read_csv(csv_path, dtype=dtype_dict)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def apply_season_year_floor(
    df: pd.DataFrame, *, season_col: str = "SEASON_YEAR", floor: int | None
) -> pd.DataFrame:
    if floor is None:
        return df
    return df[df[season_col] >= floor].copy()


def resolve_season_type(df: pd.DataFrame, *, game_id_col: str = "GAME_ID") -> pd.Series:
    """Season type derived from the GAME_ID prefix.

    This is deliberately NOT read from the ``SEASON_TYPE`` text column. In the
    training data that column labels Play-In Tournament games as "Playoffs"
    (verified: all 31 rows with GAME_ID prefix 005 carry SEASON_TYPE
    "Playoffs"), so filtering on it would silently discard the play-in games we
    want to keep. The GAME_ID prefix maps cleanly through
    nba_ou.config.constants.SEASON_TYPE_MAP: 002 regular season, 004 playoffs,
    005 play-in, 006 in-season final.

    Unmappable prefixes yield NaN.
    """
    if game_id_col not in df.columns:
        raise KeyError(
            f"Cannot determine season type: column {game_id_col!r} is missing. "
            "Set data.exclude_playoffs=False to skip season-type filtering, or "
            "point data.game_id_col at the right column."
        )
    return df[game_id_col].astype(str).str.strip().str[:3].map(SEASON_TYPE_MAP)


def filter_allowed_season_types(
    df: pd.DataFrame,
    *,
    allowed_season_types: tuple[str, ...],
    game_id_col: str = "GAME_ID",
) -> pd.DataFrame:
    """Keep only games whose season type is in ``allowed_season_types``.

    Rows whose GAME_ID prefix is unknown are dropped, since an unrecognised
    competition type is not something the model should silently train on.
    """
    season_types = resolve_season_type(df, game_id_col=game_id_col)
    keep = season_types.isin(allowed_season_types)
    return df.loc[keep].copy()


def filter_to_snapshot(
    df: pd.DataFrame, *, snapshot_col: str, minutes: int
) -> pd.DataFrame:
    """Keep one pre-game horizon, reducing the frame to one row per game.

    This is the "one model per timepoint" arm. It replaces the old approach of
    writing a second CSV with ``slice_intermediate_snapshot.py``: filtering here
    means the pooled arm and its control read the identical bytes, so one
    ``expected_checksum`` covers both and the two cannot drift apart.

    A horizon that is not in the frame is an error listing what IS there, not an
    empty frame. Asking for T=90 on a grid that stops at 60 and 120 is a typo,
    and the alternative -- training on zero rows -- fails much further downstream
    with nothing pointing back at the cause.
    """
    if snapshot_col not in df.columns:
        raise KeyError(
            f"data.snapshot_minutes={minutes} was requested but the frame has no "
            f"{snapshot_col!r} column. Only the intermediate-line dataset carries "
            "one; a closing-line CSV is already one row per game and needs no "
            "filtering."
        )
    horizons = pd.to_numeric(df[snapshot_col], errors="coerce")
    kept = df.loc[horizons == minutes].copy()
    if kept.empty:
        available = sorted(int(v) for v in horizons.dropna().unique())
        raise ValueError(
            f"data.snapshot_minutes={minutes} matched no rows. Horizons present "
            f"in {snapshot_col!r}: {available}."
        )
    return kept


def assert_one_row_per_game(
    df: pd.DataFrame, *, game_id_col: str, snapshot_col: str, context: str
) -> None:
    """Confirm the invariant every per-snapshot confidence interval rests on.

    Within one horizon there must be exactly one row per game. That is what
    makes ``n_bets`` a count of independent events, and it is checked rather
    than assumed because a duplicated (game, snapshot) pair would not error
    anywhere -- it would quietly narrow the interval and make a coin flip look
    significant.

    Silently skipped when either column is absent, which is the closing-line
    case: a frame with no snapshot column has nothing to check.
    """
    if game_id_col not in df.columns or snapshot_col not in df.columns:
        return
    duplicated = int(df.duplicated(subset=[game_id_col, snapshot_col]).sum())
    if duplicated:
        raise ValueError(
            f"{duplicated} ({game_id_col}, {snapshot_col}) pairs are duplicated "
            f"in {context}. A snapshot group would then hold the same game more "
            "than once, and every confidence interval computed on it would be "
            "too narrow. Refusing to continue."
        )


def load_scoring_sidecar(
    df: pd.DataFrame,
    *,
    csv_path: str | Path,
    game_id_col: str,
    snapshot_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach the closing-line sidecar to ``df``, returning it and its columns.

    Call this AFTER the feature matrix is built. The sidecar holds the closing
    line, which the intermediate-line builder deliberately keeps in a separate
    file so a model pricing a bet at T-720 cannot read the market's final
    number. Joining it before X was built would hand that number straight to
    the model, so the ordering is the safety property, not a detail.

    Left join on (game, snapshot): every training row keeps its place, and a row
    the sidecar has no entry for gets NaN rather than disappearing.
    """
    csv_path = Path(csv_path)
    keys = [game_id_col, snapshot_col]
    missing_keys = [key for key in keys if key not in df.columns]
    if missing_keys:
        raise KeyError(
            f"data.scoring_csv_path needs {missing_keys} in the training frame "
            "to join on, and they are not there. Both survive cleaning only on "
            "the intermediate-line path."
        )

    header = pd.read_csv(csv_path, nrows=0).columns
    missing_keys = [key for key in keys if key not in header]
    if missing_keys:
        raise KeyError(f"{csv_path} has no {missing_keys} column(s) to join on.")

    sidecar = pd.read_csv(csv_path, dtype={game_id_col: str})
    sidecar[snapshot_col] = pd.to_numeric(sidecar[snapshot_col], errors="coerce")

    attached = [c for c in sidecar.columns if c not in keys]
    collisions = sorted(set(attached) & set(df.columns))
    if collisions:
        raise ValueError(
            f"{csv_path} would overwrite {collisions}, which the training frame "
            "already carries. Rename them in the sidecar rather than letting a "
            "join decide which version wins."
        )

    merged = df.copy()
    merged[game_id_col] = merged[game_id_col].astype(str)
    merged[snapshot_col] = pd.to_numeric(merged[snapshot_col], errors="coerce")
    merged = merged.merge(sidecar, on=keys, how="left", validate="many_to_one")
    return merged, attached


def resolve_baseline_line_col(
    df: pd.DataFrame, baseline: BaselineConfig, *, market: Market = Market.TOTALS
) -> str:
    """Resolve which column stands in for "trust the bookmaker's line".

    Priority: explicit override (this is how a caller opts into
    BOOKMAKER_MEDIAN_LINE_COL or any other cross-book aggregate) -> the
    resolved main total-line column (the same single-book line already used
    everywhere else in the pipeline for cleaning and OU-accuracy scoring, and
    the only choice verified trustworthy across CSV snapshots; works for both
    the "wide" per-book and "reduced" single-column CSV schemas).
    """
    if baseline.line_col:
        if baseline.line_col not in df.columns:
            raise KeyError(
                f"baseline.line_col={baseline.line_col!r} not found in the loaded data."
            )
        return baseline.line_col

    if market is Market.SPREAD:
        # "Trust the bookmaker" means trusting the SPREAD for a spread run.
        # Falling through to the totals line here would have scored the naive
        # baseline against a number in the wrong units entirely -- and, because
        # both are numeric and present, it would have produced a plausible MAE
        # rather than an error.
        resolved_spread = resolve_main_spread_line_col(df, book=baseline.book)
        if resolved_spread is None:
            raise KeyError(
                "Could not resolve a spread baseline column: no "
                f"{spread_line_home_col(baseline.book)!r} in the loaded data. A "
                "2_0-schema CSV has no canonical spread column -- regenerate at "
                "schema 2_1."
            )
        return resolved_spread

    resolved = resolve_main_total_line_col(df, book=baseline.book)
    if resolved is None:
        raise KeyError(
            "Could not resolve a baseline line column: no "
            f"'{BOOKMAKER_MEDIAN_LINE_COL}' column and no ODDS_TOTAL_LINE_<book> "
            "column found. Set baseline.line_col explicitly."
        )
    return resolved


def _required_keep_columns(
    config: ExperimentConfig, baseline_line_col: str, target_line_col: str
) -> list[str]:
    keep = {
        config.data.date_col,
        config.data.season_col,
        "TOTAL_POINTS",
        baseline_line_col,
        target_line_col,
    }
    if config.line_col:
        keep.add(config.line_col)
    if config.strategy == PredictionStrategy.LINE_ERROR_REGRESSOR:
        keep.add("LINE_ERROR")
    if config.strategy == PredictionStrategy.SPREAD_ERROR_REGRESSOR:
        # The target, the outcome it is settled against, and the anchor line it
        # was built from. All three must survive cleaning: the target because it
        # is y, the outcome because bets settle against it, and the line because
        # verify_spread_error_column re-derives the identity between them.
        keep.update({SPREAD_ERROR_COL, HOME_MARGIN_COL, spread_line_home_col()})
        # The per-team finals too. Nothing computes with them here -- HOME_MARGIN
        # is what settles bets -- but keeping them makes the whole target chain
        # auditable on the frame the model actually saw: PTS_TEAM_HOME minus
        # PTS_TEAM_AWAY must equal HOME_MARGIN, and a diagnostic that cannot be
        # run on the real frame tends not to get run at all. They are blocked
        # from X by OUTCOME_ONLY_COLUMNS regardless.
        keep.update({PTS_HOME_COL, PTS_AWAY_COL})
        for price_col in (
            config.betting.home_price_col,
            config.betting.away_price_col,
        ):
            if price_col:
                keep.add(price_col)
    for price_col in (config.betting.over_price_col, config.betting.under_price_col):
        if price_col:
            keep.add(price_col)
    # Alternative lines to re-score against. These are especially exposed to
    # correlation pruning -- an opening line is near-perfectly correlated with
    # the closing line, which is exactly why it would be dropped, and exactly
    # why the comparison is interesting.
    keep.update(config.betting.comparison_line_cols)
    # Needed as a ROW attribute to filter training on, so it must survive
    # cleaning even though it is never a feature.
    if config.data.exclude_overtime_from_training:
        keep.add(config.data.overtime_col)

    # Carrier columns for the intermediate-line dataset, kept ONLY there.
    #
    # GAME_ID is what makes a row-count and a game-count different numbers, so
    # every *_games knob needs it to mean games rather than rows, and
    # snapshot_scoring needs it to report n_games. advanced_column_cleaning
    # drops it by default (its name contains "_ID") and on the closing-line
    # path that is right: rows are games there, nothing needs it, and keeping it
    # would change a cleaned frame that a dozen archived runs were produced
    # from. It is excluded from the feature matrix explicitly in
    # prepare_dataset -- as a string encoding date and sequence it is the last
    # thing that should reach a model.
    #
    # The snapshot column is a feature in POOLED mode. A single-horizon model
    # does not need the constant as a feature, but the scoring-sidecar join does
    # still need it as part of the (game, snapshot) key. In that case it is a
    # carrier column only and prepare_dataset excludes it from X explicitly.
    if config.data.dataset_type is DatasetType.INTERMEDIATE_LINE:
        keep.add(config.data.game_id_col)
        if (
            config.data.snapshot_minutes is None
            or config.data.scoring_csv_path is not None
        ):
            keep.add(config.data.snapshot_col)
    return sorted(keep)


def redundancy_policy_for(
    dataset_type: DatasetType,
    *,
    game_id_col: str = "GAME_ID",
    snapshot_col: str = SNAPSHOT_COLUMN,
) -> RepeatedMeasuresRedundancy | None:
    """How cleaning should judge redundancy for this kind of dataset.

    One branch per DatasetType, and the place a new type declares its grain.
    None means "one row per unit of interest", which needs no correction: the
    correlation over every row already IS the correlation over one row each.
    """
    if dataset_type is DatasetType.INTERMEDIATE_LINE:
        return RepeatedMeasuresRedundancy(
            group_col=game_id_col, snapshot_col=snapshot_col
        )
    return None


def assert_dataset_type_matches_frame(
    df: pd.DataFrame,
    dataset_type: DatasetType,
    *,
    game_id_col: str = "GAME_ID",
    snapshot_col: str = SNAPSHOT_COLUMN,
) -> None:
    """Raise when the declared type and the frame in hand contradict each other.

    Only the dangerous direction is an error. Declaring CLOSING_LINE for a frame
    that is plainly snapshot-grained means every historical feature is about to
    be correlated over up to ten copies of each game -- the exact silent
    mis-pruning the declaration exists to prevent, and nothing downstream would
    notice.

    The opposite direction is deliberately allowed: an INTERMEDIATE_LINE frame
    holding one row per game is what a single-horizon slice
    (scripts/create_train_data/slice_intermediate_snapshot.py) looks like, and
    the policy handles it correctly -- one row per game is simply every row.
    """
    if dataset_type is not DatasetType.CLOSING_LINE:
        return
    if game_id_col not in df.columns or snapshot_col not in df.columns:
        return
    if not df[game_id_col].duplicated().any():
        return
    if df[snapshot_col].nunique(dropna=True) < 2:
        return
    raise ValueError(
        f"data.dataset_type is {DatasetType.CLOSING_LINE.value!r}, but this "
        f"frame holds several rows per {game_id_col} across "
        f"{df[snapshot_col].nunique()} {snapshot_col} values -- it is the "
        "intermediate-line shape. Cleaning it as closing-line would correlate "
        "every historical feature over repeated copies of each game and prune "
        f"on that. Set data.dataset_type to "
        f"{DatasetType.INTERMEDIATE_LINE.value!r}."
    )


def clean_for_training(
    df: pd.DataFrame,
    cleaning: CleaningConfig,
    *,
    force_keep_columns: list[str],
    dataset_type: DatasetType = DatasetType.CLOSING_LINE,
    cleaning_season_col: str = "SEASON_YEAR",
    row_balance_group_col: str | None = SNAPSHOT_COLUMN,
    game_id_col: str = "GAME_ID",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Wrap clean_dataframe_for_training, guaranteeing force_keep_columns survive.

    keep_columns now ranks first in the redundancy preference order, so a
    protected column wins its correlation pairs outright rather than being
    dropped for looking like a per-book line column (see
    nba_ou.data_processing.missing_data.column_redundancy.rank_columns). That
    was not always true: the duplicate, correlation and absolute-value-match
    steps used to ignore keep_columns entirely, which could silently drop the
    baseline line column and break baseline comparability with no error.

    The snapshot-and-reattach below is kept as a safety net regardless. It costs
    one column copy, and the failure it guards against is invisible -- a run
    that completes and reports a baseline computed against the wrong column.
    """
    keep_columns = sorted(set(cleaning.keep_columns or []) | set(force_keep_columns))
    present_before = [c for c in force_keep_columns if c in df.columns]
    snapshot = df[present_before].copy()

    # WHICH DATASET IS THIS? Declared by the caller (data.dataset_type), not
    # guessed from the frame. The declaration decides how redundancy is judged;
    # the assertion below only refuses the one combination that would silently
    # prune against the wrong view.
    #
    # The other entry points into cleaning -- retraining, and the same-day
    # prediction path -- call clean_dataframe_for_training directly and pass no
    # policy at all, so they keep the closing-line behaviour by construction.
    assert_dataset_type_matches_frame(
        df,
        dataset_type,
        game_id_col=game_id_col,
        snapshot_col=row_balance_group_col or SNAPSHOT_COLUMN,
    )
    repeated_measures = redundancy_policy_for(
        dataset_type,
        game_id_col=game_id_col,
        snapshot_col=row_balance_group_col or SNAPSHOT_COLUMN,
    )

    cleaned, report = clean_dataframe_for_training(
        df,
        return_report=True,
        nan_threshold=cleaning.nan_threshold,
        corr_threshold=cleaning.corr_threshold,
        corr_threshold_overrides=cleaning.corr_threshold_overrides,
        max_seasonal_nan_spread=cleaning.max_seasonal_nan_spread,
        season_col=cleaning_season_col,
        repeated_measures=repeated_measures,
        row_balance_group_col=row_balance_group_col,
        max_na_per_row=cleaning.max_na_per_row,
        create_missing_flags=cleaning.create_missing_flags,
        keep_columns=keep_columns,
        exclude_cols_containing=cleaning.exclude_cols_containing,
        keep_all_cols=cleaning.keep_all_cols,
        verbose=cleaning.verbose,
        strict_mode=cleaning.strict_mode,
        strict_mode_exclude_cols=cleaning.strict_mode_exclude_cols,
    )

    missing_after = [c for c in present_before if c not in cleaned.columns]
    if missing_after:
        reattach = snapshot.loc[cleaned.index, missing_after]
        cleaned = pd.concat([cleaned, reattach], axis=1)
        # The report is a record of what the frame looks like, so a column put
        # back has to stop reading as dropped.
        report.column_drops = [
            entry
            for entry in report.column_drops
            if entry["column"] not in set(missing_after)
        ]
        report.columns_out = len(cleaned.columns)

    return cleaned, report


def training_eligible_mask(df: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    """Boolean per row: may this game be used to FIT a model?

    Evaluation ignores this entirely. Validation folds, the holdout and the
    walk-forward's prediction days always score every game, because ~5.2% of
    real games go to overtime and you are paid or not paid on those -- dropping
    them from scoring would measure a world that does not exist.

    Returns all-True when the filter is off, so callers need no branch.
    """
    if not config.data.exclude_overtime_from_training:
        return np.ones(len(df), dtype=bool)

    column = config.data.overtime_col
    if column not in df.columns:
        raise KeyError(
            f"data.exclude_overtime_from_training is on but {column!r} is not in "
            "the training data. It is dropped by training-data builds predating "
            "this option -- regenerate the CSV with "
            "scripts/create_train_data/create_train_data.py."
        )
    flag = pd.to_numeric(df[column], errors="coerce")
    # NaN means "unknown", which for a scheduled game means "not yet played".
    # Treat only a definite 1 as overtime so unknowns stay trainable.
    return (flag != 1).to_numpy(dtype=bool)


def ensure_spread_error_column(
    df: pd.DataFrame, *, spread_line_col: str, outcome_col: str
) -> pd.DataFrame:
    """Derive ``SPREAD_ERROR`` when the frame does not already carry one.

    Mirrors how the totals market already works: the closing dataset's residual
    target is derived here at training time (``ensure_line_error_column``), while
    the intermediate dataset derives its own upstream because only the builder
    knows which snapshot each row was quoted at.

    **Derives only if absent, never overwrites.** That is the whole safety
    property. The intermediate frame arrives with a per-snapshot SPREAD_ERROR
    already built against the line quoted at that horizon; recomputing it here
    would silently replace ten different, correct targets with whatever single
    anchor column happened to survive into the frame -- which is precisely the
    per-game-join failure this target exists to avoid. The caller verifies the
    identity afterwards either way.
    """
    if SPREAD_ERROR_COL in df.columns:
        return df

    for column in (spread_line_col, outcome_col):
        if column not in df.columns:
            raise KeyError(
                f"{column!r} is required to derive {SPREAD_ERROR_COL} but is not "
                "in the loaded data. A 2_0-schema CSV cannot train a spread "
                "model -- regenerate at schema 2_1 (see "
                "nba_ou.config.dataset_versions)."
            )

    out = df.copy()
    out[SPREAD_ERROR_COL] = spread_error(out[outcome_col], out[spread_line_col])
    return out


def verify_spread_error_column(
    df: pd.DataFrame,
    *,
    spread_line_col: str,
    outcome_col: str,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Check that SPREAD_ERROR really is ``HOME_MARGIN - <this line>``.

    The spread target is built upstream, per row, against the line quoted at that
    row's prediction point. Two ways that can go wrong produce no error at all:

    * the intermediate builder joins the anchor spread on ``game_id`` alone, so
      every snapshot of a game gets the SAME (closing) line -- the exact mistake
      the intermediate target exists to avoid;
    * a sign convention flips between datasets, and the target is wrong by twice
      the spread on every row.

    Both leave a frame that trains, scores, and reports ordinary-looking numbers.
    Recomputing the identity here is cheap and turns either into a hard failure.

    Rows where any input is missing are skipped rather than failed: coverage
    gaps are legitimate and are handled by the dropna on the target.
    """
    for column in (SPREAD_ERROR_COL, spread_line_col, outcome_col):
        if column not in df.columns:
            raise KeyError(
                f"{column!r} is required for the spread target but is not in the "
                "loaded data. A 2_0-schema CSV cannot train a spread model -- "
                "regenerate at schema 2_1 (see nba_ou.config.dataset_versions)."
            )

    expected = spread_error(df[outcome_col], df[spread_line_col])
    actual = pd.to_numeric(df[SPREAD_ERROR_COL], errors="coerce")
    comparable = expected.notna() & actual.notna()
    if not comparable.any():
        raise ValueError(
            f"No row has both {SPREAD_ERROR_COL!r} and {spread_line_col!r}, so the "
            "spread target could not be verified against its own line."
        )

    mismatch = comparable & ((expected - actual).abs() > tolerance)
    if mismatch.any():
        sample = df.loc[mismatch, [outcome_col, spread_line_col, SPREAD_ERROR_COL]]
        raise ValueError(
            f"{int(mismatch.sum())} of {int(comparable.sum())} rows have "
            f"{SPREAD_ERROR_COL} != {outcome_col} - {spread_line_col}. The target "
            "was built against a different line than bets would settle into "
            "(a sign convention, or a per-game join where a per-snapshot one was "
            f"needed). First rows:\n{sample.head().to_string()}"
        )
    return df


def assert_no_leaking_features(X: pd.DataFrame) -> None:
    """Fail loudly if an outcome-derived column reached the feature matrix.

    ``exclude_cols`` already lists these, but it is a config field a caller can
    overwrite, and the columns only appear in some CSV snapshots -- exactly the
    combination that produces a silent, spectacular-looking result rather than
    an error. This mirrors the ``_BEFORE`` leakage guard that
    ``select_training_columns`` already enforces upstream.

    Exact matches only: ``DIFF_FROM_LINE_*_BEFORE_*`` are legitimate pre-game
    rollups, and a substring check would discard hundreds of real features.
    """
    leaked = sorted(set(X.columns) & set(LEAKING_TARGET_COLUMNS))
    if leaked:
        raise ValueError(
            f"Outcome-derived column(s) {leaked} reached the feature matrix. "
            "These are functions of the final score, so a model trained on them "
            "would look excellent and predict nothing. Add them to "
            "config.exclude_cols. (Engineered *_BEFORE_* rollups are unaffected.)"
        )

    # Checked separately because they are enforced separately: these are dropped
    # centrally in prepare_dataset rather than through config.exclude_cols, so
    # that adding them could not change any archived config's fingerprint. The
    # message therefore must NOT tell the reader to edit exclude_cols.
    outcome_leaked = sorted(set(X.columns) & set(OUTCOME_ONLY_COLUMNS))
    if outcome_leaked:
        raise ValueError(
            f"Outcome column(s) {outcome_leaked} reached the feature matrix. The "
            "2_1 datasets carry these so a spread target can be built and settled; "
            "they are functions of the final score and are never features, for any "
            "strategy. They are dropped in prepare_dataset -- if one got through, "
            "a fit site is building X without that drop."
        )

    # Enforced separately again, and for a third reason: these are dropped
    # inside clean_dataframe_for_training, which is upstream of every fit site
    # (training, retrain and the same-day prediction path all route through it).
    # A survivor here therefore means a frame reached X without being cleaned,
    # which is a worse fault than the column itself.
    rotation_leaked = rotation_leak_columns(X.columns)
    if rotation_leaked:
        raise ValueError(
            f"Rotation-depth column(s) {rotation_leaked} reached the feature "
            "matrix. They aggregate over the players who logged minutes in the "
            "game being predicted, so they encode how the game went: the count "
            "correlates +0.55 with the final margin. They are dropped in "
            "clean_dataframe_for_training -- if one got through, X was built "
            "from a frame that never went through cleaning."
        )


def add_over_under_label(
    df: pd.DataFrame, *, line_col: str
) -> tuple[pd.DataFrame, int]:
    """Attach the binary OVER label, dropping pushes.

    ``1`` when the total beat the line, ``0`` when it fell short. Games landing
    exactly on the line are removed: they have no OVER/UNDER answer, so a label
    would have to be invented, and inventing one teaches the model a fiction on
    the very rows where the market was most precisely right.

    Dropping them costs almost nothing here -- measured on the 2.0 training
    data, pushes are 1.175% of games, because 53.3% of lines end in .5 and
    cannot push at all, and whole-number lines only push 2.52% of the time.
    That also settles the question of whether a three-outcome model is worth
    building: at this rate, it is not.

    Pushes are dropped from TRAINING only. Betting evaluation keeps them, where
    they are scored the way a sportsbook settles them -- stake returned,
    excluded from the win rate -- so profitability stays honest.

    Returns the frame and the number of pushes removed.
    """
    total = pd.to_numeric(df["TOTAL_POINTS"], errors="coerce")
    line = pd.to_numeric(df[line_col], errors="coerce")
    margin = total - line

    is_push = margin == 0
    n_pushes = int(is_push.sum())

    labelled = df.loc[~is_push].copy()
    labelled[OVER_LABEL_COL] = (
        (
            pd.to_numeric(labelled["TOTAL_POINTS"], errors="coerce")
            - pd.to_numeric(labelled[line_col], errors="coerce")
        )
        > 0
    ).astype(int)

    if labelled.empty:
        raise ValueError(f"No rows left after dropping pushes against {line_col!r}.")
    return labelled, n_pushes


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_col: str,
    exclude_cols: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """X and y for a frame, by allow-list when one is available.

    ``feature_names`` is the allow-list -- ``PreparedDataset.feature_names``,
    settled once by prepare_dataset. Prefer it: a deny-list only removes what it
    was told about, so every non-feature column added to the frame later (a game
    identifier kept for the window arithmetic, a closing line joined in for
    re-scoring) reaches the model unless someone remembers to extend
    ``exclude_cols`` at all five call sites. An allow-list cannot forget.

    ``exclude_cols`` remains for prepare_dataset itself, which is where the
    feature list is *derived* and so cannot consult it yet.
    """
    if feature_names is not None:
        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} feature(s) the run was prepared with are absent "
                f"from this frame, e.g. {missing[:5]}. The feature schema and the "
                "frame disagree, which would train on a different matrix than the "
                "one that was evaluated."
            )
        X = df[list(feature_names)]
    else:
        X = df.drop(columns=exclude_cols or [], errors="ignore")
    y = pd.to_numeric(df[target_col], errors="coerce")
    return X, y


def rolling_window_index(
    index: pd.Index, train_games: int, *, game_ids: pd.Series | None
) -> pd.Index:
    """Index labels of the last ``train_games`` GAMES of a chronological frame.

    The one place the rolling refit window becomes a set of rows, so the shipped
    model, the promoted model and the CV folds cannot disagree about what
    "3500" means.

    ``game_ids`` is a per-row game identifier sharing ``index``. None -- every
    one-row-per-game frame -- returns ``index[-train_games:]``, exactly the tail
    slice this replaced, so the closing-line path is untouched.
    """
    train_games = int(train_games)
    if game_ids is None:
        return index[-train_games:]
    frame = pd.DataFrame({"_game": game_ids.to_numpy()}, index=index)
    return tail_n_games(frame, train_games, group_col="_game").index


@dataclass(frozen=True)
class PreparedDataset:
    df_full: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    baseline_line_col: str
    #: The line the target is defined against and that bets are settled into.
    #: For TOTAL_POINTS this is the configured scoring line; for LINE_ERROR it
    #: is whatever line _ensure_line_error_column subtracted, i.e. the main
    #: book. Kept separate from ``baseline_line_col``, which may deliberately
    #: point at an alternative consensus line for the MAE baseline.
    target_line_col: str
    feature_names: list[str]
    #: sha256 of the source CSV actually read, recorded in run metadata so a
    #: saved run can be tied back to the exact bytes it trained on.
    dataset_checksum: str | None = None
    #: Games dropped because the total landed exactly on the line, which has no
    #: OVER/UNDER answer to learn. Classifier only; 0 for the regressors, which
    #: keep those rows (a push is a perfectly good regression target).
    n_pushes_excluded: int = 0
    #: What the planted diagnostic feature actually ended up carrying. None on
    #: every normal run -- its presence is the marker that this dataset is
    #: deliberately corrupted and cannot be read as evidence about the market.
    planted_signal: PlantedSignalResult | None = None
    #: Which cleaning step removed each dropped column and row, saved into the
    #: run directory as cleaning_report.json. Answers "where did this feature
    #: go?" without re-running the pipeline at verbose=2.
    cleaning_report: CleaningReport | None = None
    #: Distinct games behind ``df_full``. Equal to len(df_full) on every
    #: one-row-per-game dataset; smaller on a pooled intermediate-line frame,
    #: where it is the number the *_games knobs are supposed to mean.
    n_games: int = 0
    #: Distinct pre-game horizons present. 1 means one row per game, and is the
    #: readable flag that pooled betting metrics can be taken at face value.
    n_snapshots: int = 1
    #: The horizon this run was filtered to, or None for the pooled arm.
    snapshot_minutes: int | None = None
    #: Columns attached from data.scoring_csv_path. Present on df_full only --
    #: never in ``feature_names`` -- so they can be re-scored against without
    #: ever having been visible to the model.
    scoring_columns: list[str] = field(default_factory=list)

    @property
    def rows_per_game(self) -> float:
        """How many rows one game contributes. 1.0 unless the frame is pooled."""
        return len(self.df_full) / self.n_games if self.n_games else 1.0

    @property
    def is_pooled_snapshots(self) -> bool:
        """Several rows per game, so a row count is not a game count."""
        return self.n_snapshots > 1


def prepare_dataset(config: ExperimentConfig) -> PreparedDataset:
    dataset_checksum = verify_dataset_checksum(
        config.data.csv_path, expected_checksum=config.data.expected_checksum
    )
    df = load_raw_training_csv(config.data.csv_path, date_col=config.data.date_col)

    # Before anything else measures the frame. Filtering to one horizon changes
    # the row count, the cleaning statistics and the meaning of every *_games
    # knob, so it has to happen while the frame is still raw -- a horizon
    # dropped after cleaning would leave the cleaning decisions computed over
    # nine horizons the run then never sees.
    if config.data.snapshot_minutes is not None:
        df = filter_to_snapshot(
            df,
            snapshot_col=config.data.snapshot_col,
            minutes=config.data.snapshot_minutes,
        )
    assert_one_row_per_game(
        df,
        game_id_col=config.data.game_id_col,
        snapshot_col=config.data.snapshot_col,
        context=str(config.data.csv_path),
    )

    df = apply_season_year_floor(
        df, season_col=config.data.season_col, floor=config.data.season_year_floor
    )

    # Must run before cleaning: advanced_column_cleaning drops both GAME_ID
    # (name contains "_ID") and SEASON_TYPE (a pure-string column), so the
    # information needed to identify competition type is gone afterwards.
    if config.data.exclude_playoffs:
        df = filter_allowed_season_types(
            df,
            allowed_season_types=config.data.allowed_season_types,
            game_id_col=config.data.game_id_col,
        )
        if df.empty:
            raise ValueError(
                "No rows left after season-type filtering. Check "
                f"data.allowed_season_types={config.data.allowed_season_types}."
            )

    if config.strategy == PredictionStrategy.LINE_ERROR_REGRESSOR:
        df = ensure_line_error_column(df)
        # _ensure_line_error_column subtracts the configured main book's line,
        # so that is the line bets must be settled into for this target.
        target_line_col = total_line_col()
    elif config.strategy == PredictionStrategy.SPREAD_ERROR_REGRESSOR:
        # SPREAD_ERROR is derived UPSTREAM, per row, against the anchor spread
        # that row was quoted at -- the closing spread for the closing dataset,
        # the snapshot spread for each intermediate row. It is deliberately not
        # recomputed here: doing so would need one rule for both datasets, and
        # the only rule that fits both is "use the column already in the frame".
        #
        # What IS checked here is that the two agree, because a target computed
        # against one line and settled against another is the exact failure this
        # whole target is exposed to and it produces no error of its own.
        target_line_col = spread_line_home_col()
        # Derived here for the closing dataset, already present (per snapshot)
        # for the intermediate one. Verified in both cases.
        df = ensure_spread_error_column(
            df, spread_line_col=target_line_col, outcome_col=config.outcome_col
        )
        df = verify_spread_error_column(
            df, spread_line_col=target_line_col, outcome_col=config.outcome_col
        )
    else:
        # Both the total-points regressor and the classifier settle into the
        # explicitly configured line. For the classifier that line is part of
        # the label's definition, not merely a scoring choice.
        assert config.line_col is not None  # enforced by ExperimentConfig validation
        target_line_col = config.line_col
    target_col = config.target_col

    baseline_line_col = resolve_baseline_line_col(
        df, config.baseline, market=config.market
    )

    # Planted BEFORE cleaning, and deliberately NOT added to force_keep: the
    # point of the diagnostic is that the synthetic feature travels the same
    # path as a real one -- the NaN budget, the correlation prune, the constant
    # and duplicate-column steps, then build_feature_matrix. Force-keeping it
    # would exempt it from exactly what is being tested. It is checked for
    # survival below instead, so a drop is an error rather than a quiet null
    # result.
    planted = config.diagnostics.planted_signal
    if planted.enabled:
        if target_col not in df.columns:
            raise KeyError(
                f"Cannot plant a signal: target column {target_col!r} is not in "
                "the frame yet. The planted feature must be derived after the "
                "target exists and before cleaning."
            )
        df = df.copy()
        df[planted.column] = build_planted_signal(df[target_col], config=planted)

    force_keep = _required_keep_columns(config, baseline_line_col, target_line_col)
    df, cleaning_report = clean_for_training(
        df,
        config.cleaning,
        force_keep_columns=force_keep,
        dataset_type=config.data.dataset_type,
        cleaning_season_col=config.data.season_col,
        game_id_col=config.data.game_id_col,
    )

    if planted.enabled and planted.column not in df.columns:
        raise ValueError(
            f"The planted feature {planted.column!r} did not survive cleaning, so "
            "the diagnostic would measure nothing while appearing to run. Check "
            "cleaning.exclude_cols_containing and cleaning.corr_threshold."
        )

    if target_line_col not in df.columns:
        raise KeyError(
            f"Target line column {target_line_col!r} is missing after cleaning; "
            "bets cannot be settled without it."
        )

    # The label depends on TOTAL_POINTS and the line, so it must be derived
    # after cleaning has guaranteed both survive, and before the target column
    # is required to exist by the dropna below.
    n_pushes_excluded = 0
    if config.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
        df, n_pushes_excluded = add_over_under_label(df, line_col=target_line_col)

    dropna_subset = sorted({target_col, target_line_col, baseline_line_col})
    df = df.dropna(subset=dropna_subset).copy()

    df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])
    df = df.sort_values(config.data.date_col).reset_index(drop=True)

    # GAME_ID survives cleaning on the intermediate-line path (see
    # _required_keep_columns) because the window arithmetic needs it. It must
    # not survive into X: it is a string that monotonically encodes date and
    # sequence, which a tree will happily split on. Excluded here rather than
    # via config.exclude_cols so no existing fingerprint changes.
    # OUTCOME_ONLY_COLUMNS ride along here rather than in config.exclude_cols so
    # that no archived fingerprint changes (see the constant's docstring). The
    # target itself is among them for a spread run -- that is fine and mirrors
    # LINE_ERROR, which build_feature_matrix also drops from X while still
    # reading y from it.
    carrier_columns = [config.data.game_id_col, *OUTCOME_ONLY_COLUMNS]
    if (
        config.data.dataset_type is DatasetType.INTERMEDIATE_LINE
        and config.data.snapshot_minutes is not None
    ):
        # Force-kept only when a scoring sidecar needs the join key. It is
        # constant after the horizon filter and must not become a model input.
        carrier_columns.append(config.data.snapshot_col)

    X, y = build_feature_matrix(
        df,
        target_col=target_col,
        exclude_cols=[*config.exclude_cols, *carrier_columns],
    )
    assert_no_leaking_features(X)
    if config.data.game_id_col in X.columns:
        raise ValueError(
            f"{config.data.game_id_col!r} reached the feature matrix. It encodes "
            "date and sequence, so a model given it would split on when a game "
            "was played rather than on anything about the game."
        )

    n_games = (
        int(df[config.data.game_id_col].nunique())
        if config.data.game_id_col in df.columns
        else len(df)
    )
    n_snapshots = (
        int(pd.to_numeric(df[config.data.snapshot_col], errors="coerce").nunique())
        if config.data.snapshot_col in df.columns
        else 1
    )

    # AFTER X: these are closing lines the model must never see. See
    # load_scoring_sidecar -- the ordering is the safety property.
    scoring_columns: list[str] = []
    if config.data.scoring_csv_path is not None:
        df, scoring_columns = load_scoring_sidecar(
            df,
            csv_path=config.data.scoring_csv_path,
            game_id_col=config.data.game_id_col,
            snapshot_col=config.data.snapshot_col,
        )
        leaked = sorted(set(scoring_columns) & set(X.columns))
        if leaked:
            raise ValueError(
                f"Sidecar column(s) {leaked} are already in the feature matrix. "
                "The sidecar exists to hold columns the model must not see."
            )

    planted_result = None
    if planted.enabled:
        if planted.column not in X.columns:
            raise ValueError(
                f"The planted feature {planted.column!r} was cleaned through but "
                "never reached the feature matrix, so no model would ever see "
                "it. Check config.exclude_cols."
            )
        # Measured on the frame the model actually gets, not on the frame the
        # feature was generated against: rows are dropped in between, so the
        # realised correlation is the honest number.
        planted_result = measure_planted_signal(
            df, target_col=target_col, config=planted
        )

    return PreparedDataset(
        df_full=df,
        X=X,
        y=y,
        baseline_line_col=baseline_line_col,
        target_line_col=target_line_col,
        feature_names=list(X.columns),
        dataset_checksum=dataset_checksum,
        n_pushes_excluded=n_pushes_excluded,
        planted_signal=planted_result,
        cleaning_report=cleaning_report,
        n_games=n_games,
        n_snapshots=n_snapshots,
        snapshot_minutes=config.data.snapshot_minutes,
        scoring_columns=scoring_columns,
    )
