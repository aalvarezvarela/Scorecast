"""Score the checks pre-registered for 2019-20 and 2020-21 (plan section 8.6).

Those two seasons had no stints when the lineup family was selected on
2021-22 through 2025-26, so they are the one retrospective sample the
selection never saw. The checks, as written before the stints existed:

1. slope of LINE_ERROR on defense + pace absence impact > 0;
2. the offense channel's interval covers 0;
3. sign accuracy at |defense + pace| >= 3 above 52.38%;
4. the absence-driven 3PA/FGA shift has a positive LINE_ERROR slope while the
   line does not move with it; the pace shift has a positive slope;
5. the bench-replacement term has a positive slope alongside the rest of the
   defense + pace impact in one regression, per point at least the rest's.

Interpretation fixed before the run: "slope > 0" is judged on the point
estimate, with whether the 95% date-clustered interval excludes 0 reported
beside it; "the pace shift" is the absence's change in projected possessions
(``impact_possessions``); "the line does not move with it" means the closing
line's own slope on the 3PA shift has an interval covering 0.

Input is ``projection_games.parquet`` from::

    python scripts/lineups/evaluate_game_projection.py --first-season 2018 \\
        --last-season 2020 --style --lines <training csv> --output-dir <dir>

2018-19 is loaded only to warm the rosters and the level calibration; it is
not scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.data_processing.lineups.projection_eval import (
    BREAK_EVEN_110,
    directional_accuracy,
    line_error_slope,
)

TEST_SEASONS = (2019, 2020)
SHIFT_3PA = "LU_ABSENCE_SHIFT_FG3A_RATE_BEFORE"


def joint_bench_slopes(
    frame: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """LINE_ERROR ~ bench + rest of defense+pace, with date-clustered intervals."""
    data = frame[
        ["LINE_ERROR", "impact_def_pace", "impact_bench_replacement", "GAME_DATE"]
    ].dropna()
    bench = data["impact_bench_replacement"].to_numpy(float)
    rest = data["impact_def_pace"].to_numpy(float) - bench
    ys = data["LINE_ERROR"].to_numpy(float)

    def fit(index: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(index)), bench[index], rest[index]])
        return np.linalg.lstsq(design, ys[index], rcond=None)[0][1:]

    point = fit(np.arange(len(data)))
    groups = list(data.groupby("GAME_DATE", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    boots = np.array(
        [
            fit(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(n_boot)
        ]
    )
    return {
        "n": len(data),
        "bench": float(point[0]),
        "bench_ci": [float(v) for v in np.percentile(boots[:, 0], [2.5, 97.5])],
        "rest": float(point[1]),
        "rest_ci": [float(v) for v in np.percentile(boots[:, 1], [2.5, 97.5])],
        "share_boot_bench_ge_rest": float(np.mean(boots[:, 0] >= boots[:, 1])),
    }


def _slope(frame: pd.DataFrame, column: str, **kwargs) -> dict:
    row = line_error_slope(frame, column, **kwargs)
    row["ci_excludes_0"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)
    return row


def score(frame: pd.DataFrame) -> dict:
    line = total_line_col()
    frame = frame.copy()
    frame["CLOSING_LINE"] = frame[line]
    def_pace = _slope(frame, "impact_def_pace")
    offense = _slope(frame, "impact_offense")
    accuracy = directional_accuracy(frame, "impact_def_pace", 3.0)
    shift_error = _slope(frame, SHIFT_3PA)
    shift_line = _slope(frame, SHIFT_3PA, target="CLOSING_LINE")
    pace = _slope(frame, "impact_possessions")
    bench = joint_bench_slopes(frame)
    return {
        "games": int(frame["LINE_ERROR"].notna().sum()),
        "1_def_pace_slope_positive": {
            "pass": def_pace["slope"] > 0, **def_pace,
        },
        "2_offense_interval_covers_0": {
            "pass": not offense["ci_excludes_0"], **offense,
        },
        "3_def_pace_accuracy_at_3": {
            "pass": accuracy.get("accuracy", np.nan) > BREAK_EVEN_110, **accuracy,
        },
        "4a_3pa_shift_positive_line_flat": {
            "pass": shift_error["slope"] > 0 and not shift_line["ci_excludes_0"],
            "line_error": shift_error,
            "closing_line": shift_line,
        },
        "4b_pace_shift_positive": {"pass": pace["slope"] > 0, **pace},
        "5_bench_positive_and_at_least_rest": {
            "pass": bench["bench"] > 0 and bench["bench"] >= bench["rest"],
            **bench,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=Path, required=True,
                        help="projection_games.parquet from evaluate_game_projection.py --style")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    games = pd.read_parquet(args.games)
    if SHIFT_3PA not in games.columns:
        parser.error("Run evaluate_game_projection.py with --style first")
    scored = games.dropna(subset=["LINE_ERROR", "proj_total"])
    windows = {f"season_{s}": scored.loc[scored["season"].eq(s)] for s in TEST_SEASONS}
    windows["pooled"] = scored.loc[scored["season"].isin(TEST_SEASONS)]
    result = {name: score(frame) for name, frame in windows.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    for name, checks in result.items():
        print(f"\n{name} ({checks['games']} games)")
        for check, row in checks.items():
            if check != "games":
                print(f"  {'PASS' if row['pass'] else 'FAIL'}  {check}")


if __name__ == "__main__":
    main()
