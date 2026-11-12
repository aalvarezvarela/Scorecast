"""Stage 4. Replicate the line-move model (stage 2, M1/M2) on other horizons.
Also saves feature gain of the last block's M2 model. Resumable per block."""
import sys, time, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
S = Path(sys.argv[1]); h = int(sys.argv[2]); SEEDS = [1, 2, 3]
CK = S / f"ckpt4_t{h}"; CK.mkdir(exist_ok=True)
meta = pd.read_parquet(S / f"meta_t{h}.parquet").reset_index(drop=True)
Xd = pd.read_parquet(S / f"X_t{h}.parquet"); cols = list(Xd.columns); Xv = Xd.values.astype(np.float32); del Xd
PM = dict(max_depth=4, learning_rate=0.05, min_child_weight=20, subsample=0.7, colsample_bytree=0.3,
          reg_lambda=1.0, device="cuda", tree_method="hist")
y = meta.y.values.astype(np.float32)
mv = (meta.ODDS_CLOSING_TOTAL_LINE_bet365 - meta.ODDS_TOTAL_LINE_bet365).values.astype(np.float32)
di = pd.DatetimeIndex(pd.to_datetime(meta.GAME_DATE)); blk = (di.year * 100 + di.month).values
blocks = [b for b in np.unique(blk) if b >= 202210 and b % 100 not in (5, 6)]
for i, b in enumerate(blocks):
    f = CK / f"{b}.parquet"
    if f.exists(): continue
    t0 = time.time(); te = np.where(blk == b)[0]; tr = np.arange(max(0, te[0] - 6225), te[0]); tr = tr[~np.isnan(mv[tr])]
    out = meta.loc[te, ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "y"]].copy(); out["move"] = mv[te]
    dtr = xgb.DMatrix(Xv[tr], mv[tr], feature_names=cols); dte = xgb.DMatrix(Xv[te], feature_names=cols)
    for name, p in (("M1_move", PM), ("M2_move_shallow", dict(PM, max_depth=2))):
        ms = [xgb.train(dict(p, seed=s), dtr, 300) for s in SEEDS]
        out[name] = np.mean([m.predict(dte) for m in ms], 0)
    if b == blocks[-1]:
        pd.Series(ms[0].get_score(importance_type="total_gain")).sort_values(ascending=False).to_csv(S / f"m2_gain_t{h}.csv")
    tmp = f.with_suffix(".tmp"); out.to_parquet(tmp); tmp.rename(f)
    print(f"T-{h} block {b} ({i + 1}/{len(blocks)}) {time.time() - t0:.0f}s", flush=True)
pd.concat([pd.read_parquet(CK / f"{b}.parquet") for b in blocks]).to_parquet(S / f"stage4_t{h}.parquet")
print(f"T-{h} DONE", flush=True)
