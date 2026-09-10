"""
Handle missing data in NBA prediction DataFrames using a deterministic, quality-first policy.

Core idea: make the NA policy explicit and auditable by categorizing columns upfront into:

1) DROP ROWS if any of these are missing (high-signal, non-imputable, or indicates pipeline failure)
2) KEEP NaN (structurally missing market features and explicit uncertainty estimates)
3) ZERO-FILL without flags (neutral constructs where 0 means "no effect")
4) INFER from season averages when possible (rolling-window features)
5) FINAL FALLBACK to training medians (optional, recommended, excludes market columns)

Important:
- This policy must be applied consistently at training and prediction time.
- For backtesting / CV, training medians must be computed within each fold only.

Usage:
    # Training time
    df_clean = apply_missing_policy(df_train, mode="train")

    # Prediction time
    df_clean = apply_missing_policy(df_predict, mode="predict")
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from nba_ou.config.odds_columns import moneyline_col, spread_col, total_line_col
from pandas.api.types import is_numeric_dtype

# ------------------------------------------------------------------------------
# 0) STRUCTURAL MISSING COLUMNS (ADD __is_missing FLAGS)
#    Paste your "NEW COLUMNS" list here for automatic flagging
# ------------------------------------------------------------------------------

STRUCTURAL_NA_COLS: list[str] = [
    # Market/odds columns that are structurally missing and should be flagged
    # Add any columns from your dataset that you want to flag when missing
]

# ------------------------------------------------------------------------------
# 1) REQUIRED FEATURES, DROP ROW IF MISSING
#    These should never be imputed because missing means data quality failure
# ------------------------------------------------------------------------------

DROP_IF_MISSING_EXACT = [
    spread_col(),
    f"{moneyline_col()}_TEAM_HOME",
    f"{moneyline_col()}_TEAM_AWAY",
]

# If you are training, also require the target
TARGET_COL = "TOTAL_POINTS"

# ------------------------------------------------------------------------------
# 2) MARKET / BOOK / PUBLIC BETTING FEATURES
#    Policy: keep NaN, add __is_missing flag, do NOT impute
# ------------------------------------------------------------------------------

MARKET_KEEP_NA_SUBSTRINGS = [
    "pct_bets",
    "pct_money",
    "consensus_pct",
    "consensus_opener",
    "_betmgm_",
    "_caesars_",
    "_draftkings_",
    "_fanduel_",
    "_pinnacle_",
    "_bovada_",
    "_betonline_",
    "_mybookie_",
]


def _is_market_keep_na(col: str) -> bool:
    c = col.lower()
    return any(s in c for s in MARKET_KEEP_NA_SUBSTRINGS)


def _is_availability_uncertainty_keep_na(col: str) -> bool:
    """Keep a missing availability standard error distinct from zero.

    The availability-effect builder emits ``MEAN_SE`` only when both the
    available and injured samples are large enough to estimate uncertainty.
    NaN therefore means "precision is unknown". These columns can also contain
    ``INJURED_`` in their prefix, so the broad injury zero-fill policy below
    must not turn that state into 0.0 (which means perfect precision).
    """
    upper = col.upper()
    return "AVAILABILITY_EFFECT" in upper and "_MEAN_SE_" in upper


# ------------------------------------------------------------------------------
# 3) ZERO FILL FEATURES (NO FLAGS)
#    Policy: fill NaN with 0.0, but do NOT add __is_missing flag
# ------------------------------------------------------------------------------

ZERO_FILL_SUBSTRINGS = [
    # injury and availability effects
    "injury_",
    "injured_",
    "absence_effect",
    "avg_injured",
    "_injured",
    "_absence",
]


def _is_zero_fill_no_flag(col: str) -> bool:
    """Check if column should be zero-filled without adding __is_missing flag."""
    lc = col.lower()
    return any(s in lc for s in ZERO_FILL_SUBSTRINGS)


# ------------------------------------------------------------------------------
# 4) INFER FROM SEASON AVERAGE WHEN POSSIBLE
# ------------------------------------------------------------------------------

INFER_ROLLING_PATTERNS_TO_REMOVE = (
    # any LAST_{HOME_AWAY|ALL}_{n}_MATCHES
    r"_LAST_(HOME_AWAY|ALL)_[0-9]+_MATCHES",
    # specific home_away patterns
    r"_LAST_HOME_AWAY_[0-9]+_MATCHES",
    r"_LAST_ALL_[0-9]+_MATCHES",
    # weighted moving averages
    r"_LAST_HOME_AWAY_[0-9]+_WMA",
    r"_LAST_[0-9]+_WMA",
)


def _build_season_avg_fallback(col: str) -> str | None:
    """
    Map rolling window feature to *_SEASON_BEFORE_AVG_TEAM_{HOME|AWAY} when feasible.

    Example:
      ODDS_TOTAL_LINE_consensus_opener_LAST_HOME_AWAY_10_MATCHES_BEFORE_TEAM_HOME
        -> ODDS_TOTAL_LINE_consensus_opener_SEASON_BEFORE_AVG_TEAM_HOME
    """
    m = re.search(r"_BEFORE_TEAM_(HOME|AWAY)$", col)
    if not m:
        return None
    side = m.group(1)

    col_core = re.sub(r"_BEFORE_TEAM_(HOME|AWAY)$", "", col)

    for pat in INFER_ROLLING_PATTERNS_TO_REMOVE:
        col_core = re.sub(pat, "", col_core)

    # If it still has LAST, mapping is ambiguous
    if "LAST" in col_core:
        return None

    return f"{col_core}_SEASON_BEFORE_AVG_TEAM_{side}"


# ------------------------------------------------------------------------------
# 5) POLICY OBJECT
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingPolicy:
    drop_cols: list[str]
    keep_na_cols: list[str]
    zero_fill_cols: list[str]  # zero-fill without flags (injury/absence)
    infer_pairs: list[tuple[str, str]]  # (col_to_fill, season_avg_fallback)
    flag_cols: list[str]  # cols that will get __is_missing (ONLY market cols)


def _existing(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def resolve_policy(
    df: pd.DataFrame,
    *,
    current_total_line_col: str | None,
    mode: Literal["train", "predict"] = "train",
) -> MissingPolicy:
    # 1) DROP columns
    drop_cols = []
    drop_cols += _existing(df, DROP_IF_MISSING_EXACT)
    if current_total_line_col:
        drop_cols += _existing(df, [current_total_line_col])
    if mode == "train":
        drop_cols += _existing(df, [TARGET_COL])

    # 2) Keep structural market NaNs and explicit uncertainty NaNs. The latter
    # must take precedence over the broad ``INJURED_`` zero-fill rule.
    market_keep_na_cols = [c for c in df.columns if _is_market_keep_na(c)]
    uncertainty_keep_na_cols = [
        c for c in df.columns if _is_availability_uncertainty_keep_na(c)
    ]
    keep_na_cols = market_keep_na_cols + uncertainty_keep_na_cols

    # 3) zero-fill cols (injury/absence - no flags)
    zero_fill_cols = [c for c in df.columns if _is_zero_fill_no_flag(c)]

    # 4) infer pairs (exclude keep_na, drop, and zero_fill)
    drop_set = set(drop_cols)
    keep_na_set = set(keep_na_cols)
    zero_fill_set = set(zero_fill_cols)

    infer_pairs: list[tuple[str, str]] = []
    for c in df.columns:
        if c in drop_set or c in keep_na_set or c in zero_fill_set:
            continue
        fb = _build_season_avg_fallback(c)
        if fb and fb in df.columns:
            infer_pairs.append((c, fb))

    # 5) Missing flags remain limited to market columns. The feature name
    # already identifies an uncertainty channel, and XGBoost handles its NaN.
    flag_cols = sorted(set(market_keep_na_cols))

    # Remove higher-priority categories from the broad zero-fill family. A
    # keep-NaN column may contain ``INJURED_`` in its name, but must still keep
    # its missing state.
    zero_fill_cols = [
        c for c in zero_fill_cols if c not in drop_set and c not in keep_na_set
    ]
    keep_na_cols = [c for c in keep_na_cols if c not in drop_set]
    infer_pairs = [(c, fb) for (c, fb) in infer_pairs if c not in drop_set]

    return MissingPolicy(
        drop_cols=sorted(set(drop_cols)),
        keep_na_cols=sorted(set(keep_na_cols)),
        zero_fill_cols=sorted(set(zero_fill_cols)),
        infer_pairs=infer_pairs,
        flag_cols=flag_cols,
    )


# ------------------------------------------------------------------------------
# 6) APPLY POLICY
# ------------------------------------------------------------------------------


def apply_missing_policy(
    df: pd.DataFrame,
    *,
    current_total_line_col: str | None = total_line_col(),
    mode: Literal["train", "predict"] = "train",
    create_missing_flags: bool = False,
    keep_all_cols: bool = False,
) -> pd.DataFrame:
    """
    Apply the deterministic missing data policy:
      1) Drop rows missing required cols (skipped if keep_all_cols=True)
      2) Add __is_missing flags for structural missing columns (if enabled)
      3) Infer rolling features from season averages
      4) Zero-fill injury/absence features (no flags)
      Remaining NAs are left as-is.

    Args:
        df: Input dataframe to clean
        current_total_line_col: Column name for current betting line
        mode: "train" or "predict" mode
        create_missing_flags: If True, create __is_missing flag columns for market features. Default: False
        keep_all_cols: If True, skip dropping rows with missing required columns. Default: False

    Returns:
        Cleaned dataframe
    """
    out = df.copy()
    before_rows = int(len(out))

    policy = resolve_policy(
        out, current_total_line_col=current_total_line_col, mode=mode
    )

    # A) DROP rows missing required cols (skip if keep_all_cols=True)
    dropped_reasons: dict[str, int] = {}
    if policy.drop_cols and not keep_all_cols:
        drop_mask = out[policy.drop_cols].isna().any(axis=1)
        dropped_reasons["missing_required_any"] = int(drop_mask.sum())
        out = out.loc[~drop_mask].copy()
    elif keep_all_cols and policy.drop_cols:
        dropped_reasons["missing_required_any"] = (
            0  # Would have dropped but keep_all_cols=True
        )

    # B) ADD __is_missing FLAGS (after dropping)
    # Use pd.concat to avoid DataFrame fragmentation
    flags_created = 0
    if create_missing_flags:
        missing_flags = {}
        for c in policy.flag_cols:
            if c in out.columns:
                missing_flags[f"{c}__is_missing"] = out[c].isna().astype("int8")

        if missing_flags:
            out = pd.concat([out, pd.DataFrame(missing_flags, index=out.index)], axis=1)

        flags_created = len(missing_flags)

    # C) INFER FROM SEASON AVERAGES (exclude market keep-na cols)
    infer_applied = 0
    keep_na_set = set(policy.keep_na_cols)
    for c, fb in policy.infer_pairs:
        if c in keep_na_set:
            continue
        if (
            c in out.columns
            and fb in out.columns
            and is_numeric_dtype(out[c])
            and is_numeric_dtype(out[fb])
        ):
            before_na = int(out[c].isna().sum())
            if before_na:
                out[c] = out[c].fillna(out[fb])
                after_na = int(out[c].isna().sum())
                if after_na < before_na:
                    infer_applied += 1

    # D) ZERO FILL FEATURES (injury/absence - numeric only, NO FLAGS)
    zero_cols_numeric = [
        c
        for c in policy.zero_fill_cols
        if c in out.columns and is_numeric_dtype(out[c])
    ]
    if zero_cols_numeric:
        out.loc[:, zero_cols_numeric] = out.loc[:, zero_cols_numeric].fillna(0.0)

    # Calculate summary statistics
    rows_dropped = int(before_rows - len(out))
    drop_rate_pct = (
        round(100 * rows_dropped / before_rows, 2) if before_rows > 0 else 0.0
    )

    report = {
        "mode": mode,
        "rows_in": before_rows,
        "rows_out": int(len(out)),
        "rows_dropped": rows_dropped,
        "drop_rate_pct": drop_rate_pct,
        "drop_cols_count": int(len(policy.drop_cols)),
        "keep_na_cols_count": int(len(policy.keep_na_cols)),
        "zero_cols_count": int(len(zero_cols_numeric)),
        "infer_pairs_count": int(len(policy.infer_pairs)),
        "infer_cols_applied_count": int(infer_applied),
        "missing_flags_created": int(flags_created),
        "remaining_na_count": int(out.isna().sum().sum()),
        "remaining_na_by_col": out.isna().sum()[out.isna().sum() > 0].to_dict(),
        "dropped_reasons": dropped_reasons,
        "drop_cols": policy.drop_cols,
        "keep_na_cols_sample": policy.keep_na_cols[:25],
        "zero_fill_cols_sample": zero_cols_numeric[:25],
        "infer_pairs_sample": policy.infer_pairs[:25],
    }

    # Print summary report
    print("\nMissing Data Policy Report:")
    print(f"  Rows dropped: {report['rows_dropped']} ({report['drop_rate_pct']}%)")
    print(f"  Critical columns requiring data: {report['drop_cols_count']}")
    print(f"  Columns zero-filled: {report['zero_cols_count']}")
    print(
        f"  Infer pairs applied: {report['infer_cols_applied_count']}/{report['infer_pairs_count']}"
    )
    print(f"  Remaining NaN cells: {report['remaining_na_count']}")

    return out
