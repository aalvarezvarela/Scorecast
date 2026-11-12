"""Column-level differences between two intermediate dataset files (shared rows),
and a scoring file for the new one (old scoring rows; closing total of new games
from their T-0 snapshot row)."""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

old, new, out = sys.argv[1:4]
K = ["GAME_ID", "TIME_TO_MATCH_MIN"]
co, cn = pq.ParquetFile(old).schema_arrow.names, pq.ParquetFile(new).schema_arrow.names
print("columns only old:", sorted(set(co) - set(cn))[:20], "only new:", sorted(set(cn) - set(co))[:20])
ko, kn = pd.read_parquet(old, columns=K), pd.read_parquet(new, columns=K)
ko["io"] = np.arange(len(ko)); kn["in"] = np.arange(len(kn))
m = ko.merge(kn, on=K)
print("rows old", len(ko), "new", len(kn), "shared", len(m), "games new-only", kn[~kn.GAME_ID.isin(ko.GAME_ID)].GAME_ID.nunique(),
      "old-only", ko[~ko.GAME_ID.isin(kn.GAME_ID)].GAME_ID.nunique())
shared = [c for c in cn if c in set(co) and c not in K]
res = []
for i in range(0, len(shared), 400):
    cols = shared[i:i + 400]
    a = pd.read_parquet(old, columns=cols).iloc[m.io.values].reset_index(drop=True)
    b = pd.read_parquet(new, columns=cols).iloc[m["in"].values].reset_index(drop=True)
    for c in cols:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x = x.astype(float).values; y = y.astype(float).values
            d = ~((np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=False))
            if d.any():
                both = d & ~np.isnan(x) & ~np.isnan(y)
                res.append(dict(col=c, n_diff=int(d.sum()), nan_flip=int((np.isnan(x) != np.isnan(y)).sum()),
                                mean_abs=float(np.abs(x - y)[both].mean()) if both.any() else np.nan,
                                sd_col=float(np.nanstd(y))))
        else:
            d = ~((x.isna() & y.isna()) | (x.astype(str) == y.astype(str)))
            if d.any():
                res.append(dict(col=c, n_diff=int(d.sum()), nan_flip=int((x.isna() != y.isna()).sum()), mean_abs=np.nan, sd_col=np.nan))
r = pd.DataFrame(res).sort_values("n_diff", ascending=False)
r["pct_rows"] = 100 * r.n_diff / len(m)
r.to_csv(out, index=False)
print("differing columns:", len(r), "of", len(shared))
print(r.head(40).to_string(index=False))
r["fam"] = r.col.str.replace(r"_TEAM_(HOME|AWAY)", "", regex=True).str.replace(r"\d+", "N", regex=True).str[:38]
print(r.groupby("fam").agg(cols=("col", "size"), max_pct=("pct_rows", "max")).sort_values("max_pct", ascending=False).head(40).to_string())
# scoring file for the new build
so = pd.read_parquet(old.replace(".parquet", "_scoring.parquet"))
line = pd.read_parquet(new, columns=K + ["ODDS_TOTAL_LINE_bet365"])
s = line[K].merge(so, on=K, how="left")
t0 = line[line.TIME_TO_MATCH_MIN == 0].set_index("GAME_ID").ODDS_TOTAL_LINE_bet365
chk = s.dropna(subset=["ODDS_CLOSING_TOTAL_LINE_bet365"]).drop_duplicates("GAME_ID")
print("T-0 row equals stored closing total:", float((chk.GAME_ID.map(t0) == chk.ODDS_CLOSING_TOTAL_LINE_bet365).mean()))
miss = s.ODDS_CLOSING_TOTAL_LINE_bet365.isna()
s.loc[miss, "ODDS_CLOSING_TOTAL_LINE_bet365"] = s.loc[miss, "GAME_ID"].map(t0)
print("scoring rows", len(s), "closing filled from T-0:", int(miss.sum()), "still missing:", int(s.ODDS_CLOSING_TOTAL_LINE_bet365.isna().sum()))
s.to_parquet(new.replace(".parquet", "_scoring.parquet"))
