#!/usr/bin/env python3
"""A tonight-only live build equals a full live build, for a replayed date.

    python scripts/perf/live_replay_parity.py --date 2026-04-10 \
        --as-of 2026-04-10T23:30:00Z --seasons 2025 --out artifacts/perf/live_replay

Replays ``date`` as of ``as_of`` (``live_replay.replay_live_game_day``: stored
ticks quoted by then, archived injury reports read live, stored crews) and
builds the intermediate dataset twice from it -- in full, and for the date's
games only (``target_game_ids``) -- then compares the date's rows of both,
dataset and scoring, exactly. Read-only (the history memo aside).
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
from nba_ou.create_training_data.create_intermediate_line_df import (  # noqa: E402
    create_intermediate_line_df,
)
from nba_ou.create_training_data.live_replay import replay_live_game_day  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--seasons", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else None
    args.out.mkdir(parents=True, exist_ok=True)

    live = replay_live_game_day(args.date, pd.Timestamp(args.as_of))
    print(
        f"{len(live.game_ids)} games; {len(live.line_history.ticks):,} ticks; "
        f"{len(live.injury_reports.spans)} report spans"
    )
    built = {}
    for name, targets in (("target", live.game_ids), ("full", None)):
        started = time.perf_counter()
        frame, scoring = create_intermediate_line_df(
            live=live,
            season_years=seasons,
            return_scoring=True,
            target_game_ids=targets,
            verbose=True,
        )
        tonight = frame["GAME_ID"].astype(str).isin(set(live.game_ids))
        built[name] = (
            frame[tonight].reset_index(drop=True),
            scoring[
                scoring["GAME_ID"].astype(str).isin(set(live.game_ids))
            ].reset_index(drop=True),
        )
        built[name][0].to_parquet(args.out / f"{name}_dataset.parquet")
        built[name][1].to_parquet(args.out / f"{name}_scoring.parquet")
        print(f"{name}: {time.perf_counter() - started:.0f} s, {tonight.sum()} rows")
    ok = True
    for i, part in enumerate(("dataset", "scoring")):
        print(f"--- {part}")
        ok &= compare(built["full"][i], built["target"][i], exact=True)
    print("RESULT:", "EQUAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
