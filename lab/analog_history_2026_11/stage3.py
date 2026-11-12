"""Stage 3. The production recipe with its DAILY refit (train on the last 6225
games strictly before each game day), over every game day 2022-10 .. 2026-11.
Checks whether the monthly-refit harness understates the production model.
Resumable: one checkpoint per month."""
import sys, json, glob, time, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
S = Path(sys.argv[1]); h = int(sys.argv[2]); SEEDS = [1, 2, 3]
CK = S / f"ckpt3_t{h}"; CK.mkdir(exist_ok=True)
meta = pd.read_parquet(S / f"meta_t{h}.parquet").reset_index(drop=True)
Xv = pd.read_parquet(S / f"X_t{h}.parquet").values.astype(np.float32)
prm = json.load(open(glob.glob(f"artifacts/experiments/retrain_line_error_2_5_2026_10/retrain25_line_error_t{h}_*/backtest_summary.json")[0]))
PB = dict(prm["xgb_params"], device="cuda", tree_method="hist"); NT = prm["n_estimators"]
y = meta.y.values.astype(np.float32); date = pd.to_datetime(meta.GAME_DATE).values
di = pd.DatetimeIndex(date); blk = (di.year * 100 + di.month).values
blocks = [b for b in np.unique(blk) if b >= 202210 and b % 100 not in (5, 6)]
for i, b in enumerate(blocks):
    f = CK / f"{b}.parquet"
    if f.exists(): continue
    t0 = time.time(); rows = np.where(blk == b)[0]
    out = meta.loc[rows, ["GAME_ID", "GAME_DATE", "SEASON_YEAR", "y"]].copy()
    P = np.zeros((len(rows), len(SEEDS)), np.float32)
    for d in np.unique(date[rows]):
        te = np.where(date == d)[0]; tr = np.arange(max(0, te[0] - 6225), te[0])
        dtr = xgb.DMatrix(Xv[tr], y[tr]); dte = xgb.DMatrix(Xv[te])
        for j, s in enumerate(SEEDS):
            P[np.searchsorted(rows, te), j] = xgb.train(dict(PB, seed=s), dtr, NT).predict(dte)
    for j in range(len(SEEDS)): out[f"D0_daily__s{j}"] = P[:, j]
    out["D0_daily"] = P.mean(1)
    tmp = f.with_suffix(".tmp"); out.to_parquet(tmp); tmp.rename(f)
    print(f"T-{h} month {b} ({i + 1}/{len(blocks)}) {time.time() - t0:.0f}s", flush=True)
pd.concat([pd.read_parquet(CK / f"{b}.parquet") for b in blocks]).to_parquet(S / f"stage3_t{h}.parquet")
print(f"T-{h} DONE", flush=True)
