"""Reproduce the dashboard Historical page numbers with its own code."""
import sys
from datetime import date

import pandas as pd

from nba_ou.dashboard.domain.analysis import aggregate_cells, prepare_analysis
from nba_ou.dashboard.repositories.analysis import read_history_range

pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
start = date.fromisoformat(sys.argv[1]); end = date.fromisoformat(sys.argv[2])
target = int(sys.argv[3]) if len(sys.argv) > 3 else 1
data = read_history_range(start, end, target, 2)
rows = prepare_analysis(data.predictions, min_edge={1: 0.1, 3: 0.01})
rows.to_parquet(sys.argv[4]) if len(sys.argv) > 4 else None
g = data.games
print("games", len(g), "final", int((g.status.eq(1) & g.home_points.notna()).sum()))
print(rows.groupby(["source", "kind", "dataset_type"]).size())
cols = ["games", "wins", "losses", "win_pct", "ci_low", "ci_high"]
x = rows[rows.kind.eq(1)]
print("\n== own snapshot (diagonal), per source x slot")
d = aggregate_cells(x[x.on_horizon], by=["source", "model_horizon_minutes"])
print(d.sort_values(["source", "model_horizon_minutes"])[["source", "model_horizon_minutes", *cols]].round(1).to_string(index=False))
print("\n== pooled diagonal per source")
print(aggregate_cells(x[x.on_horizon], by=["source"])[["source", *cols, "rows"]].round(1).to_string(index=False))
print("\n== every slot x time bucket, live, win% (bets)")
c = aggregate_cells(x[x.source.eq("Live predictions")], by=["slot_label", "model_horizon_minutes", "bucket"])
c["cell"] = c.win_pct.round(0).astype("Int64").astype(str) + " (" + (c.wins + c.losses).astype(str) + ")"
print(c.pivot(index="model_horizon_minutes", columns="bucket", values="cell").sort_index(axis=1, ascending=False).to_string())
print("\n== per slot pooled over all time buckets, per source")
s = aggregate_cells(x, by=["source", "slot_label", "model_horizon_minutes"])
print(s.sort_values(["source", "model_horizon_minutes"])[["source", "slot_label", *cols, "rows"]].round(1).to_string(index=False))
