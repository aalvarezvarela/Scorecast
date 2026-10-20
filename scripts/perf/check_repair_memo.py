#!/usr/bin/env python3
"""Check the per-season closing-line repair memo against the whole-frame repair.

    python scripts/perf/check_repair_memo.py artifacts/perf/ref_7bd96a9/closing_full/tape

Replays the all-season odds load from a tape. When the repair runs, the
pre-repair frame is repaired three ways:

* whole: ``repair(frame)``, what the build does without a memo;
* cold: ``_repair_by_season`` with an empty local memo (repairs, stores);
* warm: the same again (every finished season from the memo).

Each must equal the whole repair exactly (frame and audit). Taped builds have
no server fingerprints, so stand-ins are used; the memo keys on them as it
would on real ones.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]
os.environ["NBA_OU_INPUT_CACHE"] = "0"
os.environ["NBA_OU_S3_SEASONS"] = "0"
os.environ["NBA_OU_HISTORY_MEMO_DIR"] = tempfile.mkdtemp()

import db_tape  # noqa: E402
from nba_ou.postgre_db.odds import merge_odds_data  # noqa: E402
from nba_ou.postgre_db.season_store import frame_differences  # noqa: E402
from nba_ou.utils.seasons import get_seasons_between_dates  # noqa: E402

import pandas as pd  # noqa: E402

original = merge_odds_data._repair_by_season
failures = []


def hook(frame, seasons, fingerprints, repair):
    stand_ins = {s: ["stand-in", s] for s in seasons}
    started = time.perf_counter()
    whole = repair(frame)
    print(f"whole repair {time.perf_counter() - started:.1f} s")
    for label in ("cold", "warm"):
        started = time.perf_counter()
        got = original(frame, seasons, stand_ins, repair)
        print(f"{label} {time.perf_counter() - started:.1f} s")
        problems = frame_differences(got[0], whole[0])
        audit = frame_differences(
            got[1].sort_values(["game_id", "market", "book"], kind="mergesort")
            .reset_index(drop=True),
            whole[1].sort_values(["game_id", "market", "book"], kind="mergesort")
            .reset_index(drop=True),
        )
        print(f"  {label}: frame {problems[:3] or 'EXACT'}, audit {audit[:3] or 'EXACT'}")
        failures.extend(problems + audit)
    return whole


merge_odds_data._repair_by_season = hook
seasons = get_seasons_between_dates(pd.Timestamp("2017-10-01"), pd.Timestamp("2026-10-03"))
with db_tape.replay(sys.argv[1]):
    merge_odds_data.load_and_merge_odds_yahoo_sportsbookreview(
        season_years=seasons, extra_game_ids=[]
    )
print("RESULT:", "EQUAL" if not failures else f"DIFFERENT ({len(failures)})")
sys.exit(1 if failures else 0)
