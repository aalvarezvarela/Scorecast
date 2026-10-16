# Retraining the corrected 2_5 data: total_points

17 cells: closing plus the 16 intermediate production horizons, 150 trials per
cell, CUDA, evaluation only. Source recipes are
`experiments/totals_spread_2_5_2026_10` (including closing NaN budget 800).

The shared protocol, dataset pinning, preflight, parallel-launch commands and
review/promotion procedure are in the
[line_error campaign README](../retrain_line_error_2_5_2026_10/README.md).

```bash
bash experiments/runners/run_retrain_total_points_2_5_2026_10.sh --preflight-only
bash experiments/runners/run_retrain_total_points_2_5_2026_10.sh --skip-existing
```
