"""Old build (20261105) vs new build (20261111): production recipe with daily
refit (stage 3) and the move-to-close models (stage 4), same harness."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(sys.argv[1]); V = Path(sys.argv[2])
rng = np.random.default_rng(0)


def wr(d, col, lo=None, hi=None, q=None):
    b = d[(d[col].abs() >= 0.1) & (d.y != 0)] if q is None else d[(d[col] != 0) & (d.y != 0)]
    if q is not None:
        b = b[b[col].abs() >= b[col].abs().quantile(q)]
    if lo is not None:
        b = b[(b.GAME_DATE >= lo) & (b.GAME_DATE <= hi)]
    w = np.sign(b[col]) == np.sign(b.y)
    return f"{100 * w.mean():.1f} (n={len(b)})" if len(b) else "-"


print("== production recipe, daily refit, bet when |edge| >= 0.1")
for h in (480, 720, 360):
    for tag, S in (("old 20261105", O), ("new 20261111", V)):
        f = S / f"stage3_t{h}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f); c = "D0_daily"
        seas = {int(s): wr(g, c) for s, g in d.groupby("SEASON_YEAR")}
        print(f"T-{h} {tag}: all {wr(d, c)} | holdout window {wr(d, c, '2026-01-18', '2026-04-17')} | since 2026-10-28 {wr(d, c, '2026-10-28', '2026-12-31')}")
        print("     by season:", seas)
    fo, fn = O / f"stage3_t{h}.parquet", V / f"stage3_t{h}.parquet"
    if fo.exists() and fn.exists():
        a = pd.read_parquet(fo).merge(pd.read_parquet(fn), on=["GAME_ID", "GAME_DATE", "y"], suffixes=("_o", "_n"))
        a = a[(a.y != 0) & (a.D0_daily_o.abs() >= 0.1) & (a.D0_daily_n.abs() >= 0.1)]
        wo = (np.sign(a.D0_daily_o) == np.sign(a.y)).astype(float); wn = (np.sign(a.D0_daily_n) == np.sign(a.y)).astype(float)
        days = a.GAME_DATE.values; u, inv = np.unique(days, return_inverse=True)
        diff = np.bincount(inv, wn - wo); cnt = np.bincount(inv)
        bs = [diff[i].sum() / cnt[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(4000))]
        print(f"     paired new-old: {100 * (wn - wo).mean():+.2f}pp CI [{100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}] | same side {100 * (np.sign(a.D0_daily_o) == np.sign(a.D0_daily_n)).mean():.1f}% | corr {np.corrcoef(a.D0_daily_o, a.D0_daily_n)[0, 1]:.3f}")

print("\n== move-to-close models: win% all bets / top 25% by |predicted move|")
for h in (60, 720, 240):
    for tag, S in (("old", O), ("new", V)):
        f = S / f"stage4_t{h}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        print(f"T-{h} {tag}: " + " | ".join(f"{m} {wr(d, m, q=0)} / {wr(d, m, q=0.75)}" for m in ("M1_move", "M2_move_shallow")))
