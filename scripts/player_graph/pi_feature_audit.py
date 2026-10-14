"""Redundancy audit of the phase 5 feature block v1 on 2018-19 (development).

Reads ``data/player_graph/pi_features/season=2018.parquet``
(``build_pi_features.py``) and the closing 2_6 file's rows of the same games,
and reports, without using any outcome:

1. coverage and distribution of each ``PI_*`` column;
2. correlations among the ``PI_*`` columns;
3. correlations with the 2_6 block (``LU_*``, ``STARTER_*``), which the
   primary arm excludes;
4. overlap with the 2_5 features the reference model sees (the file's numeric
   pre-game columns, minus identifiers, targets and the 2_6 block): the five
   strongest correlations per ``PI_*`` column, the in-sample R^2 of an OLS on
   its ten strongest 2_5 correlates, and a 5-fold cross-validated ridge R^2 on
   all of them.

Flags a ``PI_*`` column as redundant when some 2_5 or 2_6 column has
``|r| >= 0.9`` or the cross-validated R^2 from 2_5 is at least 0.8 (thresholds
fixed before running).

    python scripts/player_graph/pi_feature_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.pi_features import FEATURES

SEASON = 2018
CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
NOT_FEATURES = (
    "GAME_ID",
    "SEASON_YEAR",
    "TOTAL_POINTS",
    "GAME_DATE",
    "TEAM_ID_TEAM_HOME",
    "TEAM_ID_TEAM_AWAY",
)
TARGET_MARKERS = ("DIFF_FROM_LINE", "LINE_ERROR", "TOTAL_POINTS", "_PTS_TEAM_", "WIN")
BLOCK_2_6 = ("LU_", "STARTER_")
REDUNDANT_R = 0.9
REDUNDANT_R2 = 0.8
MIN_PAIRS = 200


def correlations(frame: pd.DataFrame, target: pd.Series) -> pd.Series:
    valid = frame.notna().sum() >= MIN_PAIRS
    return frame.loc[:, valid].corrwith(target).dropna()


def cv_ridge_r2(x: pd.DataFrame, y: pd.Series, folds: int = 5) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = x.fillna(x.median())
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    predicted = cross_val_predict(
        model, x, y, cv=KFold(folds, shuffle=True, random_state=0)
    )
    return float(1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    features = pd.read_parquet(
        args.local_root / "player_graph" / "pi_features" / f"season={SEASON}.parquet"
    )
    file = pd.read_parquet(CLOSING_2_6)
    file["GAME_ID"] = file["GAME_ID"].astype(str).str.zfill(10)
    season = file.loc[pd.to_numeric(file["SEASON_YEAR"], errors="coerce").eq(SEASON)]
    rows = season.merge(features, on="GAME_ID", how="inner")
    numeric = rows.select_dtypes("number")
    candidates = [
        c
        for c in numeric.columns
        if c not in NOT_FEATURES
        and c not in FEATURES
        and not any(m in c for m in TARGET_MARKERS)
    ]
    block_2_6 = [c for c in candidates if c.startswith(BLOCK_2_6)]
    block_2_5 = [c for c in candidates if not c.startswith(BLOCK_2_6)]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(
        f"2018-19: {len(season):,} games in the closing file, {len(features):,} with "
        f"PI features ({len(rows):,} joined); 2_5 candidate columns "
        f"{len(block_2_5):,}, 2_6 block {len(block_2_6)}"
    )

    print("\n1. COVERAGE AND DISTRIBUTION (games with features)")
    print(
        rows[list(FEATURES)]
        .describe(percentiles=[0.05, 0.5, 0.95])
        .T.round(3)
        .to_string()
    )

    print("\n2. CORRELATIONS AMONG PI COLUMNS")
    print(rows[list(FEATURES)].corr().round(2).to_string())

    print("\n3. CORRELATION WITH THE 2_6 BLOCK (|r| >= 0.3 shown)")
    table = {}
    for column in FEATURES:
        r = correlations(rows[block_2_6], rows[column])
        table[column] = r
    lu = pd.DataFrame(table)
    shown = lu.loc[(lu.abs() >= 0.3).any(axis=1)]
    print(shown.round(2).to_string() if len(shown) else "  none")

    print("\n4. OVERLAP WITH THE 2_5 FEATURES")
    summary = []
    for column in FEATURES:
        y = rows[column]
        r = correlations(rows[block_2_5], y)
        top = r.abs().sort_values(ascending=False)
        top10 = list(top.index[:10])
        x = rows[top10].fillna(rows[top10].median())
        design = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
        beta, *_ = np.linalg.lstsq(design, y.to_numpy(float), rcond=None)
        fitted = design @ beta
        ols_r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        usable = [c for c in block_2_5 if rows[c].notna().sum() >= MIN_PAIRS]
        ridge_r2 = cv_ridge_r2(rows[usable], y)
        max_lu = lu[column].abs().max()
        flag = (
            top.iloc[0] >= REDUNDANT_R
            or max_lu >= REDUNDANT_R
            or ridge_r2 >= REDUNDANT_R2
        )
        summary.append(
            {
                "feature": column,
                "top 2_5 correlates": "; ".join(
                    f"{c} {r[c]:+.2f}" for c in top.index[:5]
                ),
                "OLS R2 top-10": round(float(ols_r2), 3),
                "CV ridge R2 (all 2_5)": round(ridge_r2, 3),
                "max |r| 2_6": round(float(max_lu), 2),
                "redundant": "YES" if flag else "no",
            }
        )
    with pd.option_context("display.max_colwidth", 200):
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
