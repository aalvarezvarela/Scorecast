"""Build an all-feature, Colab-ready CSV for the TabPFN-3 walk-forward notebook.

Every eligible numeric pre-game feature is retained unless it is constant (or
entirely missing) in pre-holdout history. Outcome/target carriers remain in the
CSV because they are y/scoring data, but are explicitly forbidden from X.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATE_COL = "GAME_DATE"
GAME_ID_COL = "GAME_ID"
TOTAL_COL = "TOTAL_POINTS"
TOTAL_LINE_COL = "ODDS_TOTAL_LINE_bet365"
HOME_MARGIN_COL = "HOME_MARGIN"
SPREAD_LINE_COL = "ODDS_SPREAD_LINE_HOME_bet365"
SPREAD_PRICE_HOME_COL = "ODDS_SPREAD_PRICE_HOME"
SPREAD_PRICE_AWAY_COL = "ODDS_SPREAD_PRICE_AWAY"
LINE_ERROR_COL = "LINE_ERROR"
SPREAD_ERROR_COL = "SPREAD_ERROR"

LEAKAGE_COLUMNS = {
    TOTAL_COL,
    LINE_ERROR_COL,
    SPREAD_ERROR_COL,
    HOME_MARGIN_COL,
    "PTS_TEAM_HOME",
    "PTS_TEAM_AWAY",
    "OVER_LABEL",
    "COVER_LABEL",
    "DIFF_FROM_LINE",
    "IS_OVERTIME",
}
NON_FEATURE_COLUMNS = {
    GAME_ID_COL,
    DATE_COL,
    "SEASON_YEAR",
    "SEASON_ID",
    "SEASON_TYPE",
}
# These exact columns settle ROI. Equivalent pre-game price features remain eligible,
# but the carriers themselves must not be reintroduced after compact-CSV reload.
SCORING_ONLY_COLUMNS = {SPREAD_PRICE_HOME_COL, SPREAD_PRICE_AWAY_COL}
ROTATION_LEAK_PREFIXES = ("N_ACTIVE_PLAYERS", "TOTAL_NON_INJURED_PLAYER_")


@dataclass(frozen=True)
class TargetSpec:
    target_col: str
    line_col: str


TARGET_SPECS = {
    "line_error": TargetSpec(LINE_ERROR_COL, TOTAL_LINE_COL),
    "total_points": TargetSpec(TOTAL_COL, TOTAL_LINE_COL),
    "spread_error": TargetSpec(SPREAD_ERROR_COL, SPREAD_LINE_COL),
}


def file_checksum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


def load_source(path: Path, *, include_playoffs: bool) -> pd.DataFrame:
    if "2_3" not in path.name:
        raise ValueError(f"Expected a 2_3 dataset, got {path.name!r}.")

    header = pd.read_csv(path, nrows=0)
    required = {
        DATE_COL,
        GAME_ID_COL,
        TOTAL_COL,
        TOTAL_LINE_COL,
        HOME_MARGIN_COL,
        SPREAD_LINE_COL,
    }
    missing = sorted(required - set(header.columns))
    if missing:
        raise KeyError(f"Required 2_3 columns are missing: {missing}")

    id_dtypes = {column: str for column in header.columns if "ID" in column.upper()}
    frame = pd.read_csv(path, dtype=id_dtypes, low_memory=False)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="raise").dt.normalize()
    for column in (TOTAL_COL, TOTAL_LINE_COL, HOME_MARGIN_COL, SPREAD_LINE_COL):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[LINE_ERROR_COL] = frame[TOTAL_COL] - frame[TOTAL_LINE_COL]
    frame[SPREAD_ERROR_COL] = frame[HOME_MARGIN_COL] - frame[SPREAD_LINE_COL]

    valid = (
        frame[TOTAL_COL].gt(130)
        & frame[TOTAL_LINE_COL].gt(100)
        & frame[[HOME_MARGIN_COL, SPREAD_LINE_COL]].notna().all(axis=1)
    )
    frame = frame.loc[valid].copy()
    if not include_playoffs:
        allowed_prefixes = {"002", "005", "006"}
        frame = frame.loc[
            frame[GAME_ID_COL].astype(str).str[:3].isin(allowed_prefixes)
        ].copy()
    return frame.sort_values([DATE_COL, GAME_ID_COL]).reset_index(drop=True)


def numeric_feature_candidates(frame: pd.DataFrame) -> list[str]:
    blocked = LEAKAGE_COLUMNS | NON_FEATURE_COLUMNS | SCORING_ONLY_COLUMNS
    candidates = [
        column
        for column in frame.columns
        if column not in blocked
        and not column.startswith(ROTATION_LEAK_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    leaked = sorted(set(candidates) & LEAKAGE_COLUMNS)
    if leaked:
        raise AssertionError(f"Outcome leakage survived: {leaked}")
    return candidates


def cleaning_feature_universe(
    frame: pd.DataFrame, *, cutoff: pd.Timestamp
) -> list[str]:
    history = frame.loc[frame[DATE_COL] <= cutoff]
    candidates = numeric_feature_candidates(history)
    varying = history[candidates].nunique(dropna=True) > 1
    return varying[varying].index.tolist()


def apply_row_na_budget(
    frame: pd.DataFrame, *, feature_columns: list[str], max_na_per_row: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    na_count = frame[feature_columns].isna().sum(axis=1)
    keep = na_count <= max_na_per_row
    report = {
        "max_na_per_row": max_na_per_row,
        "columns_counted": len(feature_columns),
        "rows_before": len(frame),
        "rows_after": int(keep.sum()),
        "rows_dropped": int((~keep).sum()),
        "max_na_seen_in_kept_rows": int(na_count.loc[keep].max()),
    }
    return frame.loc[keep].copy().reset_index(drop=True), report


def validate_preprocessed_output(
    output: Path,
    *,
    args: argparse.Namespace,
    cutoff: pd.Timestamp,
    expected_rows: int,
    expected_features: list[str],
) -> dict[str, object]:
    """Prove the compact CSV preserves the all-feature allow-list and rows."""
    compact = load_source(output, include_playoffs=args.include_playoffs)
    compact_universe = cleaning_feature_universe(compact, cutoff=cutoff)
    compact, compact_row_report = apply_row_na_budget(
        compact,
        feature_columns=compact_universe,
        max_na_per_row=args.max_na_per_row,
    )
    if len(compact) != expected_rows or compact_row_report["rows_dropped"]:
        raise AssertionError(
            "The compact CSV did not preserve the already-cleaned row population."
        )

    missing_features = sorted(set(expected_features) - set(compact_universe))
    unexpected_features = sorted(set(compact_universe) - set(expected_features))
    if missing_features or unexpected_features:
        raise AssertionError(
            "The compact CSV changed the all-feature allow-list: "
            f"missing={missing_features}, unexpected={unexpected_features}."
        )
    return {
        "reloaded_rows": len(compact),
        "row_budget_drops_on_reload": compact_row_report["rows_dropped"],
        "all_feature_allowlist_reproduced": True,
        "reloaded_feature_columns": len(compact_universe),
    }


def build_preprocessed_dataset(args: argparse.Namespace) -> dict:
    source = Path(args.input)
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")

    frame = load_source(source, include_playoffs=args.include_playoffs)
    cutoff = frame[DATE_COL].max() - pd.Timedelta(days=args.test_days)
    feature_universe = cleaning_feature_universe(frame, cutoff=cutoff)
    frame, row_report = apply_row_na_budget(
        frame,
        feature_columns=feature_universe,
        max_na_per_row=args.max_na_per_row,
    )
    clean_history_games = int((frame[DATE_COL] <= cutoff).sum())
    if clean_history_games < args.train_games:
        raise ValueError(
            f"Only {clean_history_games} clean pre-holdout games remain; "
            f"cannot build a {args.train_games}-game rolling window."
        )

    union_features = list(feature_universe)

    carriers = [
        GAME_ID_COL,
        DATE_COL,
        TOTAL_COL,
        LINE_ERROR_COL,
        HOME_MARGIN_COL,
        SPREAD_ERROR_COL,
        TOTAL_LINE_COL,
        SPREAD_LINE_COL,
        SPREAD_PRICE_HOME_COL,
        SPREAD_PRICE_AWAY_COL,
        "IS_OVERTIME",
    ]
    carriers = [column for column in carriers if column in frame.columns]
    output_columns = carriers + [c for c in union_features if c not in carriers]
    output_frame = frame[output_columns].copy()

    forbidden = LEAKAGE_COLUMNS | NON_FEATURE_COLUMNS | SCORING_ONLY_COLUMNS
    offenders = sorted(set(feature_universe) & forbidden)
    if offenders:
        raise AssertionError(f"All-feature allow-list leaks: {offenders}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    validation = validate_preprocessed_output(
        output,
        args=args,
        cutoff=cutoff,
        expected_rows=len(output_frame),
        expected_features=feature_universe,
    )
    manifest = {
        "source_file": str(source),
        "source_checksum": file_checksum(source),
        "output_file": str(output),
        "output_checksum": file_checksum(output),
        "test_days": args.test_days,
        "test_cutoff_exclusive": str(cutoff.date()),
        "train_games": args.train_games,
        "feature_selection": {
            "mode": "all_eligible_numeric_nonconstant",
            "constant_detection_history": "all_pre_holdout_rows",
            "target_correlation_filtering": False,
            "interfeature_correlation_pruning": False,
            "minimum_nonnull_rate": None,
        },
        "include_playoffs": args.include_playoffs,
        "included_game_id_prefixes": sorted(
            output_frame[GAME_ID_COL].astype(str).str[:3].unique().tolist()
        ),
        "playoff_games": int(
            output_frame[GAME_ID_COL].astype(str).str.startswith("004").sum()
        ),
        "overtime": {
            "treatment": "included_as_played_with_final_score",
            "is_overtime_blocked_from_features": True,
            "games": int(
                pd.to_numeric(
                    output_frame.get("IS_OVERTIME", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .fillna(0)
                .astype(bool)
                .sum()
            ),
            "pre_holdout_games": int(
                pd.to_numeric(
                    output_frame.loc[
                        output_frame[DATE_COL] <= cutoff,
                        "IS_OVERTIME",
                    ]
                    if "IS_OVERTIME" in output_frame
                    else pd.Series(dtype=float),
                    errors="coerce",
                )
                .fillna(0)
                .astype(bool)
                .sum()
            ),
            "holdout_games": int(
                pd.to_numeric(
                    output_frame.loc[
                        output_frame[DATE_COL] > cutoff,
                        "IS_OVERTIME",
                    ]
                    if "IS_OVERTIME" in output_frame
                    else pd.Series(dtype=float),
                    errors="coerce",
                )
                .fillna(0)
                .astype(bool)
                .sum()
            ),
        },
        "clean_pre_holdout_games": clean_history_games,
        "holdout_games": int((output_frame[DATE_COL] > cutoff).sum()),
        "row_cleaning": row_report,
        "feature_universe_columns": len(feature_universe),
        "union_feature_columns": len(union_features),
        "output_rows": len(output_frame),
        "output_columns": len(output_frame.columns),
        "carrier_columns": carriers,
        "forbidden_feature_columns": sorted(forbidden),
        "all_feature_allowlist": feature_universe,
        "features_per_target": len(feature_universe),
        "validation": validation,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/train_data/training_data_2_3_20260909.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/train_data/training_data_2_3_20260909_tabpfn_v3_all_features.csv"
        ),
    )
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--train-games", type=int, default=6500)
    parser.add_argument("--max-na-per-row", type=int, default=300)
    parser.add_argument(
        "--include-playoffs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    manifest = build_preprocessed_dataset(parse_args())
    summary_keys = (
        "output_file",
        "output_checksum",
        "output_rows",
        "output_columns",
        "clean_pre_holdout_games",
        "holdout_games",
        "union_feature_columns",
        "row_cleaning",
        "validation",
        "include_playoffs",
        "included_game_id_prefixes",
        "playoff_games",
        "overtime",
    )
    summary = {key: manifest[key] for key in summary_keys}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
