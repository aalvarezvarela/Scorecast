# Legacy CV replication on current closing datasets

This campaign tests the useful part of the historical recipes with the current
schema 2.0 and 2.2 closing datasets:

- `line_error`: fixed 4,500-game training window.
- `total_points`: fixed 3,750-game training window.

The comparison changes the CV geometry and fixed training window while keeping
the recent experiment stack. It uses the historical `test_anchored` layout:
12 latest folds, 50 validation games per anchor and a 60-game step. It keeps
history from 2019, `max_na_per_row: 300`, pooled MAE, tuned tree count, the
top-15%-and-0.04 lexicographic rule, seed 16, and a 90-day daily holdout.

Overtime games are explicitly included in training and scoring. Playoffs are
explicitly excluded; regular-season and Play-In games remain eligible.

The primary betting threshold is 0.1 for broad-coverage comparison. The saved
betting sweep also reports edge 2.0, which was the filter behind the historical
59.57% total-points headline.
