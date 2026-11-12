import sys, numpy as np, pandas as pd
from pathlib import Path
S = Path(sys.argv[1]); pd.set_option("display.width", 220)
def wr(p, y, mask=None):
    ok = (p != 0) & (y != 0) & ~np.isnan(p)
    if mask is not None: ok &= mask
    n = ok.sum(); w = (np.sign(p[ok]) == np.sign(y[ok])).mean() if n else np.nan
    return f"{w:.3f} (n={n}, ±{1.96*np.sqrt(.25/max(n,1)):.3f})"
for h in (720, 360, 60):
    f = S / f"stage2_t{h}.parquet"
    if not f.exists(): continue
    d = pd.read_parquet(f); y = d.y.values; mv = d.move.values; ok = ~np.isnan(mv)
    print(f"\n== T-{h}: {len(d)} games")
    print("line_error variants (bet |edge|>=0.1):")
    for a in ["B0_base", "R1_no_ot_train", "R2_clip25", "R3_huber18", "R4_sign_x10"]:
        p = d[a].values; print(f"  {a:16s} win {wr(p, y, np.abs(p) >= 0.1)}  mae {np.abs(p - y).mean():.3f}  r {np.corrcoef(p, y)[0, 1]:+.3f}  sign-agree-with-base {(np.sign(p) == np.sign(d.B0_base.values)).mean():.2f}")
    for a in ["M1_move", "M2_move_shallow"]:
        p = d[a].values
        r2 = 1 - ((mv[ok] - p[ok]) ** 2).mean() / ((mv[ok] - 0) ** 2).mean()
        mvd = ok & (mv != 0)
        print(f"{a}: r(pred,move) {np.corrcoef(p[ok], mv[ok])[0, 1]:.3f}  R2 vs zero {r2:.3f}  direction acc when line moved {(np.sign(p[mvd]) == np.sign(mv[mvd])).mean():.3f}  r(pred,line_error) {np.corrcoef(p, y)[0, 1]:+.3f}")
        print(f"   bet snapshot line toward predicted move, all: {wr(p, y)}")
        ap = np.abs(p)
        for q in (0.5, 0.75, 0.9):
            t = np.quantile(ap, q); m = ap >= t
            print(f"   |pred move| top {int(100 - 100 * q)}% (>= {t:.2f}): win {wr(p, y, m)}   dir acc {(np.sign(p[m & mvd]) == np.sign(mv[m & mvd])).mean():.3f}  mean realised move in bet direction {np.nanmean(np.sign(p[m]) * mv[m]):+.2f}")
        print("   by season, top 25%:", {int(s): wr(g[a].values, g.y.values, np.abs(g[a].values) >= np.quantile(ap, .75)) for s, g in d.groupby("SEASON_YEAR")})
    b = d.B0_base.values; p = d.M1_move.values
    print("combos: base&move agree:", wr(b, y, np.sign(b) == np.sign(p)), "| disagree (base side):", wr(b, y, np.sign(b) != np.sign(p)), "| base+move sum:", wr(b + p, y))
