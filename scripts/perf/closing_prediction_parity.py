#!/usr/bin/env python3
"""A tonight-only closing prediction build equals one that skips nothing.

    python scripts/perf/closing_prediction_parity.py --date 2026-04-10 \
        --as-of 2026-04-10T22:30:00Z --out artifacts/perf/closing_prediction

Replays ``date``'s scheduled data from storage (``live_replay.
replay_scheduled_data``) and builds it in prediction mode twice: with the
date's games as targets (what predict_nba_games runs), and with every built
game a target -- every row computed in full, no history memo -- then compares
the date's rows exactly. Read-only (the history memo aside).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: The checkout whose code is built, e.g. a worktree pinned to one commit.
CODE_ROOT = Path(os.environ.get("NBA_OU_CODE_ROOT", ROOT)).resolve()
sys.path[:0] = [str(CODE_ROOT / "src"), str(Path(__file__).parent)]

import pandas as pd  # noqa: E402
from compare_builds import compare  # noqa: E402
from nba_ou.create_training_data.create_df_to_predict import (  # noqa: E402
    create_df_to_predict,
)
from nba_ou.create_training_data.live_replay import (  # noqa: E402
    replay_scheduled_data,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = replay_scheduled_data(args.date, pd.Timestamp(args.as_of))
    tonight = data["scheduled_games"]["GAME_ID"].astype(str).tolist()
    print(f"{len(tonight)} games")

    def build(targets):
        started = time.perf_counter()
        frame, layer_inputs = create_df_to_predict(
            todays_prediction=True,
            scheduled_data=data,
            target_game_ids=targets,
            return_layer_inputs=True,
        )
        print(f"{len(targets)} targets: {time.perf_counter() - started:.0f} s")
        return frame, layer_inputs

    target, layer_inputs = build(tonight)
    if target.empty:
        print("RESULT: DIFFERENT (the tonight-only build has no rows for the date)")
        return 1
    everything = sorted(set(layer_inputs["GAME_ID"].astype(str)) | set(tonight))
    full, _ = build(everything)
    full = full[full["GAME_ID"].astype(str).isin(set(tonight))]
    target.reset_index(drop=True).to_parquet(args.out / "target.parquet")
    full.reset_index(drop=True).to_parquet(args.out / "full.parquet")
    ok = compare(full.reset_index(drop=True), target.reset_index(drop=True), exact=True)
    print("RESULT:", "EQUAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
