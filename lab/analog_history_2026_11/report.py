import sys, numpy as np, pandas as pd
from pathlib import Path
S = Path(sys.argv[1]); rng = np.random.default_rng(0)
def wr(p, y, thr=0.1):
    ok = (np.abs(p) >= thr) & (y != 0); return (np.sign(p[ok]) == np.sign(y[ok])).mean(), ok.sum()
allw = {}
for h in [int(a) for a in sys.argv[2:]]:
    f = S / f"wf_t{h}.parquet"
    if not f.exists(): continue
    d = pd.read_parquet(f); y = d.y.values; arms = [c for c in d.columns if c[0] == "A" and c[1].isdigit() and "__" not in c]
    days = d.GAME_DATE.values; ud, inv = np.unique(days, return_inverse=True)
    print(f"\n== T-{h}: {len(d)} games, {len(ud)} days, {d.GAME_DATE.min().date()}..{d.GAME_DATE.max().date()}  MAE(line)={np.abs(y).mean():.3f}")
    rows = []
    for a in arms:
        p = d[a].values; w, n = wr(p, y)
        seeds = [wr(d[c].values, y)[0] for c in d.columns if c.startswith(a + "__")]
        se = (d[a].values - y) ** 2 - (d["A0_base"].values - y) ** 2           # paired vs baseline
        hit = ((np.sign(p) == np.sign(y)) & (y != 0)).astype(float) - ((np.sign(d["A0_base"].values) == np.sign(y)) & (y != 0))
        bs = []; bw = []
        for _ in range(2000):                                               # bootstrap clustered by game day
            k = rng.integers(0, len(ud), len(ud)); cnt = np.bincount(k, minlength=len(ud))[inv]
            bs.append((se * cnt).sum() / cnt.sum()); bw.append((hit * cnt).sum() / cnt.sum())
        rows.append(dict(arm=a, n_bets=n, win=w, win_lo=w - 1.96 * np.sqrt(w * (1 - w) / n), win_hi=w + 1.96 * np.sqrt(w * (1 - w) / n),
                         seed_min=min(seeds), seed_max=max(seeds), mae=np.abs(p - y).mean(), r=np.corrcoef(p, y)[0, 1],
                         d_win_pp=100 * hit.mean(), d_win_lo=100 * np.percentile(bw, 2.5), d_win_hi=100 * np.percentile(bw, 97.5),
                         d_mse=se.mean(), d_mse_lo=np.percentile(bs, 2.5), d_mse_hi=np.percentile(bs, 97.5)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("by season (win rate, ensemble):")
    print(pd.DataFrame({a: d.groupby("SEASON_YEAR").apply(lambda g: wr(g[a].values, g.y.values)[0]) for a in arms}).round(3).assign(n=d.groupby("SEASON_YEAR").size()).to_string())
