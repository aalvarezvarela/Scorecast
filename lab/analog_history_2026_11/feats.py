"""As-of analogue features: for each game, the K most similar EARLIER games
(same horizon) and how the market moved / erred on them. Plus team-level
market-error memory. Everything uses only games dated strictly before."""
import sys, json, glob, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
S = Path(sys.argv[1]); h = int(sys.argv[2])
F = "data/train_data/intermediate_line_data_2_5_20261105.parquet"
meta = pd.read_parquet(S / f"meta_t{h}.parquet"); X = pd.read_parquet(S / f"X_t{h}.parquet")
MKT = ["ODDS_SNAP_TOT_CONSENSUS_MOVE_FROM_OPEN", "ODDS_SNAP_TOT_CONSENSUS_MOVE_RECENT",
       "ODDS_SNAP_TOT_BET365_DEVIATION_FROM_CONSENSUS", "ODDS_SNAP_TOT_CONSENSUS_CROSSBOOK_STD",
       "ODDS_SNAP_TOT_CONSENSUS_STEAM_NET", "ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE",
       "ODDS_SNAP_SPR_CONSENSUS_LINE", "ODDS_SNAP_SPR_CONSENSUS_MOVE_FROM_OPEN",
       "ODDS_SNAP_TOT_BET365_MOVE_FROM_OPEN", "ODDS_SNAP_TOT_CONSENSUS_LINE"]
FUND = ["PACE_PER40_SEASON_BEFORE_AVG_TEAM_HOME", "PACE_PER40_SEASON_BEFORE_AVG_TEAM_AWAY",
        "OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME", "OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
        "DEF_RATING_SEASON_BEFORE_AVG_TEAM_HOME", "DEF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
        "OFF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME", "OFF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY",
        "DEF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME", "DEF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY",
        "REST_DAYS_BEFORE_MATCH_TEAM_HOME", "REST_DAYS_BEFORE_MATCH_TEAM_AWAY",
        "PTS_SEASON_BEFORE_AVG_TEAM_HOME", "PTS_SEASON_BEFORE_AVG_TEAM_AWAY",
        "ODDS_SNAP_TOT_CONSENSUS_LINE", "ODDS_SNAP_SPR_CONSENSUS_LINE"]
ids = ["GAME_ID", "TIME_TO_MATCH_MIN", "TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY"]
raw = pd.read_parquet(F, columns=sorted(set(ids + MKT + FUND)), filters=[("TIME_TO_MATCH_MIN", "==", h)])
raw = raw.drop_duplicates("GAME_ID").set_index("GAME_ID").reindex(meta.GAME_ID.values)
assert raw.TEAM_ID_TEAM_HOME.notna().all()
date = pd.to_datetime(meta.GAME_DATE).values; y = meta.y.values.astype(float)
line = meta["ODDS_TOTAL_LINE_bet365"].values.astype(float)
close = meta["ODDS_CLOSING_TOTAL_LINE_bet365"].values.astype(float)
mtc = close - line                               # market move snapshot -> close
close_err = meta.TOTAL_POINTS.values - close     # closing-line error
order = np.argsort(date, kind="stable"); assert (order == np.arange(len(date))).all(), "not date sorted"

def derived(df):
    d = df.copy()
    # line level relative to the league's own recent level (totals drift by season)
    lvl = pd.Series(d["ODDS_SNAP_TOT_CONSENSUS_LINE"].values, index=pd.to_datetime(date))
    daily = lvl.groupby(level=0).mean()
    roll = daily.shift(1).rolling(30, min_periods=5).mean()
    d["LINE_VS_LEAGUE30"] = lvl.values - roll.reindex(lvl.index).values
    d["ABS_SPREAD"] = d["ODDS_SNAP_SPR_CONSENSUS_LINE"].abs()
    if "PTS_SEASON_BEFORE_AVG_TEAM_HOME" in d:
        d["LINE_MINUS_PTS_AVG"] = d["ODDS_SNAP_TOT_CONSENSUS_LINE"] - d["PTS_SEASON_BEFORE_AVG_TEAM_HOME"] - d["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    return d.drop(columns=["ODDS_SNAP_TOT_CONSENSUS_LINE"])

spaces = {"MKT": derived(raw[MKT]).drop(columns=["ODDS_SNAP_SPR_CONSENSUS_LINE"]),
          "FUND": derived(raw[FUND])}
# supervised space: top features of a model fit on pre-2022-10 games only (static, no test leakage)
pre = date < np.datetime64("2022-10-01")
prm = json.load(open(glob.glob(f"artifacts/experiments/retrain_line_error_2_5_2026_10/retrain25_line_error_t{h}_*/backtest_summary.json")[0]))
m = xgb.XGBRegressor(**prm["xgb_params"], n_estimators=150, device="cuda", tree_method="hist", random_state=0)
m.fit(X[pre], y[pre])
imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
spaces["TOP"] = X[imp.index[:30]].astype(float).reset_index(drop=True)
print("TOP space:", list(imp.index[:12]))

udates = np.unique(date); out = {}
for name, sp in spaces.items():
    A = sp.values.astype(float); n = len(A)
    res = {k: np.full((n, 4), np.nan) for k in (50, 200)}
    for d in udates:
        te = np.where(date == d)[0]; ntr = te[0]            # rows strictly before this date
        if ntr < 600: continue
        tr = A[:ntr]
        med = np.nanmedian(tr, 0); sd = np.nanstd(tr, 0); sd[sd == 0] = 1
        Z = np.nan_to_num(np.clip((tr - med) / sd, -4, 4)); Q = np.nan_to_num(np.clip((A[te] - med) / sd, -4, 4))
        D = ((Q[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        ok = ~np.isnan(mtc[:ntr])
        for k in res:
            idx = np.argpartition(D, k, axis=1)[:, :k]
            yy = y[:ntr][idx]; mm = np.where(ok[idx], mtc[:ntr][idx], np.nan)
            res[k][te] = np.c_[yy.mean(1), (yy > 0).mean(1) - (yy < 0).mean(1), np.nanmean(mm, 1),
                               np.sqrt(np.take_along_axis(D, idx, 1).max(1))]
    for k, r in res.items():
        for j, s in enumerate(["LINE_ERR_MEAN", "OVER_MINUS_UNDER", "MOVE_TO_CLOSE_MEAN", "KTH_DIST"]):
            out[f"ANALOG_{name}_K{k}_{s}_BEFORE"] = r[:, j]

# team market-error memory: closing-line error of the team's previous games
tm = pd.concat([pd.DataFrame({"row": np.arange(len(meta)), "team": raw[c].values, "date": date, "e": close_err, "side": s})
                for c, s in (("TEAM_ID_TEAM_HOME", "HOME"), ("TEAM_ID_TEAM_AWAY", "AWAY"))]).sort_values(["team", "date"])
g = tm.groupby("team").e
for w in (5, 10, 20, 40):
    tm[f"L{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=max(3, w // 2)).mean())
tm["OVR10"] = tm.groupby("team").e.transform(lambda s: (s.shift(1) > 0).rolling(10, min_periods=5).mean())
for s in ("HOME", "AWAY"):
    t = tm[tm.side == s].set_index("row").sort_index()
    for c in ("L5", "L10", "L20", "L40", "OVR10"):
        out[f"TEAMMKT_CLOSE_ERR_{c}_BEFORE_TEAM_{s}"] = t[c].reindex(np.arange(len(meta))).values
for c in ("L5", "L10", "L20", "L40"):
    out[f"TEAMMKT_CLOSE_ERR_{c}_SUM_BEFORE"] = out[f"TEAMMKT_CLOSE_ERR_{c}_BEFORE_TEAM_HOME"] + out[f"TEAMMKT_CLOSE_ERR_{c}_BEFORE_TEAM_AWAY"]
out = pd.DataFrame(out); out.to_parquet(S / f"new_t{h}.parquet")
print(out.shape, out.notna().mean().round(2).min())
