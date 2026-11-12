"""Ridge move feature alone vs the XGBoost move-to-close models, same games."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(sys.argv[1])
R = "ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE"
rows = []
for h in [int(x) for x in sys.argv[2:]]:
    f = S / (f"stage4_t{h}.parquet" if (S / f"stage4_t{h}.parquet").exists() else f"stage2_t{h}.parquet")
    d = pd.read_parquet(f)
    meta = pd.read_parquet(S / f"meta_t{h}.parquet").reset_index(drop=True)
    x = pd.read_parquet(S / f"X_t{h}.parquet", columns=[R]).reset_index(drop=True)
    d = d.merge(pd.DataFrame({"GAME_ID": meta.GAME_ID, "ridge": x[R]}), on="GAME_ID")
    d["avg"] = (d.M1_move + d.M2_move_shallow) / 2
    ok = d.move.notna()
    for name in ("ridge", "M1_move", "M2_move_shallow"):
        p = d[name]; v = ok & p.notna()
        r2 = 1 - ((d.move - p)[v] ** 2).sum() / (d.move[v] ** 2).sum()
        b = d[p.notna() & (p != 0) & (d.y != 0)]
        win = (np.sign(b[name]) == np.sign(b.y))
        top = b[name].abs() >= b[name].abs().quantile(0.75)
        mvd = (np.sign(b[name]) * b.move)
        rows.append(dict(h=h, model=name, corr_move=np.corrcoef(p[v], d.move[v])[0, 1], r2_vs_zero=r2,
                         n=len(b), win_all=100 * win.mean(), win_top25=100 * win[top].mean(),
                         clv_all=mvd.mean(), clv_top25=mvd[top].mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
