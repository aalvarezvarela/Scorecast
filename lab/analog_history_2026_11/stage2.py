"""Stage 2. (a) a model trained on the MARKET MOVE (closing line - snapshot line)
instead of the game outcome; (b) noise-robust variants of the line_error model.
Same monthly walk-forward as evalwf.py. Resumable per (horizon, block)."""
import sys, json, glob, time, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
S = Path(sys.argv[1]); h = int(sys.argv[2]); SEEDS = [1, 2, 3]
CK = S / f"ckpt2_t{h}"; CK.mkdir(exist_ok=True)
meta = pd.read_parquet(S / f"meta_t{h}.parquet").reset_index(drop=True)
Xv = pd.read_parquet(S / f"X_t{h}.parquet").values.astype(np.float32)
prm = json.load(open(glob.glob(f"artifacts/experiments/retrain_line_error_2_5_2026_10/retrain25_line_error_t{h}_*/backtest_summary.json")[0]))
PB = dict(prm["xgb_params"], device="cuda", tree_method="hist"); NT = prm["n_estimators"]
PM = dict(max_depth=4, learning_rate=0.05, min_child_weight=20, subsample=0.7, colsample_bytree=0.3,
          reg_lambda=1.0, device="cuda", tree_method="hist")       # untuned; move target has ~100x less variance
y = meta.y.values.astype(np.float32)
mv = (meta.ODDS_CLOSING_TOTAL_LINE_bet365 - meta.ODDS_TOTAL_LINE_bet365).values.astype(np.float32)
ot = meta.IS_OVERTIME.fillna(0).values.astype(bool)
di = pd.DatetimeIndex(pd.to_datetime(meta.GAME_DATE)); blk = (di.year * 100 + di.month).values
blocks = [b for b in np.unique(blk) if b >= 202210 and b % 100 not in (5, 6)]
def fit(tr, te, target, params, nt, mask=None, **kw):
    if mask is not None: tr = tr[mask[tr]]
    dtr = xgb.DMatrix(Xv[tr], target[tr]); dte = xgb.DMatrix(Xv[te])
    return np.mean([xgb.train(dict(params, seed=s, **kw), dtr, nt).predict(dte) for s in SEEDS], 0)
for i, b in enumerate(blocks):
    f = CK / f"{b}.parquet"
    if f.exists(): continue
    t0 = time.time(); te = np.where(blk == b)[0]; tr = np.arange(max(0, te[0] - 6225), te[0])
    out = meta.loc[te, ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "y"]].copy(); out["move"] = mv[te]
    out["B0_base"] = fit(tr, te, y, PB, NT)
    out["M1_move"] = fit(tr, te, mv, PM, 300, mask=~np.isnan(mv))
    out["M2_move_shallow"] = fit(tr, te, mv, dict(PM, max_depth=2), 300, mask=~np.isnan(mv))
    out["R1_no_ot_train"] = fit(tr, te, y, PB, NT, mask=~ot)
    out["R2_clip25"] = fit(tr, te, np.clip(y, -25, 25), PB, NT)
    out["R3_huber18"] = fit(tr, te, y, PB, NT, objective="reg:pseudohubererror", huber_slope=18.0, base_score=0.0)
    out["R4_sign_x10"] = fit(tr, te, 10 * np.sign(y), PB, NT)
    tmp = f.with_suffix(".tmp"); out.to_parquet(tmp); tmp.rename(f)
    print(f"T-{h} block {b} ({i + 1}/{len(blocks)}) {time.time() - t0:.0f}s", flush=True)
pd.concat([pd.read_parquet(CK / f"{b}.parquet") for b in blocks]).to_parquet(S / f"stage2_t{h}.parquet")
print(f"T-{h} DONE", flush=True)
