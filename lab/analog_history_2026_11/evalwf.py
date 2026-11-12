"""Walk-forward 2022-10 .. 2026-11: refit every month on the last 6225 earlier
games (production window and hyperparameters), predict the month.
Resumable: one checkpoint per (horizon, block); rerun the same command to continue."""
import sys, json, glob, time, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
S = Path(sys.argv[1]); h = int(sys.argv[2]); SEEDS = [1, 2, 3, 4, 5]
CK = S / f"ckpt_t{h}"; CK.mkdir(exist_ok=True)
meta = pd.read_parquet(S / f"meta_t{h}.parquet").reset_index(drop=True)
X = pd.read_parquet(S / f"X_t{h}.parquet").reset_index(drop=True)
N = pd.read_parquet(S / f"new_t{h}.parquet")
prm = json.load(open(glob.glob(f"artifacts/experiments/retrain_line_error_2_5_2026_10/retrain25_line_error_t{h}_*/backtest_summary.json")[0]))
P = dict(prm["xgb_params"], device="cuda", tree_method="hist", objective="reg:squarederror"); NT = prm["n_estimators"]
date = pd.to_datetime(meta.GAME_DATE).values; y = meta.y.values.astype(np.float32)
an = [c for c in N if c.startswith("ANALOG")]; tmc = [c for c in N if c.startswith("TEAMMKT")]
noise = np.random.default_rng(0).normal(size=N.shape).astype(np.float32)
Xv, Nv = X.values.astype(np.float32), N.values.astype(np.float32)
arms = {"A0_base": Xv, "A1_+analog": np.c_[Xv, N[an].values.astype(np.float32)],
        "A2_+teammkt": np.c_[Xv, N[tmc].values.astype(np.float32)], "A3_+both": np.c_[Xv, Nv],
        "A4_+noise38": np.c_[Xv, noise], "A5_only_new": Nv}
di = pd.DatetimeIndex(date); blk = di.year * 100 + di.month
blocks = [b for b in np.unique(blk) if b >= 202210 and b % 100 not in (5, 6)]
for i, b in enumerate(blocks):
    f = CK / f"{b}.parquet"
    if f.exists(): continue
    t0 = time.time(); te = np.where(blk == b)[0]; tr = np.arange(max(0, te[0] - 6225), te[0])
    out = meta.loc[te, ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "y"]].copy(); out["blk"] = b
    for a, XA in arms.items():
        dtr = xgb.DMatrix(XA[tr], y[tr]); dte = xgb.DMatrix(XA[te])
        ps = [xgb.train(dict(P, seed=s), dtr, NT).predict(dte) for s in SEEDS]
        for j, p in enumerate(ps): out[f"{a}__s{j}"] = p
        out[a] = np.mean(ps, 0)
    tmp = f.with_suffix(".tmp"); out.to_parquet(tmp); tmp.rename(f)
    print(f"T-{h} block {b} ({i + 1}/{len(blocks)}) n_test={len(te)} n_train={len(tr)} {time.time() - t0:.0f}s", flush=True)
d = pd.concat([pd.read_parquet(CK / f"{b}.parquet") for b in blocks])
d.to_parquet(S / f"wf_t{h}.parquet"); print(f"T-{h} DONE {len(d)} games", flush=True)
