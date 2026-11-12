import sys, numpy as np, pandas as pd
from pathlib import Path
S = Path(sys.argv[1])
pd.set_option("display.width", 200)
for h in (360, 720, 60):
    meta = pd.read_parquet(S / f"meta_t{h}.parquet").reset_index(drop=True); N = pd.read_parquet(S / f"new_t{h}.parquet")
    d = pd.to_datetime(meta.GAME_DATE); t = (d >= "2022-10-01").values
    y = meta.y.values; mtc = (meta.ODDS_CLOSING_TOTAL_LINE_bet365 - meta.ODDS_TOTAL_LINE_bet365).values
    rows = []
    for c in N.columns:
        if "KTH_DIST" in c or "OVR10" in c: continue
        f = N[c].values; ok = t & ~np.isnan(f) & (y != 0)
        bet = np.sign(f[ok]); win = (bet == np.sign(y[ok]))[bet != 0]
        okm = t & ~np.isnan(f) & ~np.isnan(mtc)
        rows.append(dict(feature=c.replace("_BEFORE", ""), n=int(win.size), r_err=np.corrcoef(f[ok], y[ok])[0, 1],
                         win=win.mean(), z=(win.mean() - .5) / np.sqrt(.25 / win.size),
                         r_move=np.corrcoef(f[okm], mtc[okm])[0, 1]))
    print(f"\n== T-{h}  (test 2022-10..2026-11; over rate {np.mean(y[t & (y != 0)] > 0):.3f})")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
