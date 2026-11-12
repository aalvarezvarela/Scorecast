# Analogue-history and line-move study (2026-11)

Exploration scripts behind `docs/intermediate/history_analogues_and_line_move_study.md`.
Every script takes the output folder as its first argument and a horizon in
minutes; the walk-forward scripts checkpoint per month and resume when rerun.

```bash
O=artifacts/research/analog_history_2026_11
python lab/analog_history_2026_11/prep.py   $O 360 720 60   # cleaned matrices, production recipe
python lab/analog_history_2026_11/feats.py  $O 360          # as-of analogue + team-memory features
python lab/analog_history_2026_11/direct.py $O              # model-free value of those features
python lab/analog_history_2026_11/evalwf.py $O 360          # walk-forward, base vs added features
python lab/analog_history_2026_11/report.py $O 360 720 60
python lab/analog_history_2026_11/stage2.py $O 360          # line-move target, robust targets
python lab/analog_history_2026_11/report2.py $O
python lab/analog_history_2026_11/stage3.py $O 360          # production recipe, daily refit
python lab/analog_history_2026_11/stage4.py $O 30           # line-move target, other horizons


# 2026-11-12 rerun on the rebuilt partitions, and the dashboard check
python lab/analog_history_2026_11/export_partitions.py data/train_data
python lab/analog_history_2026_11/diff_builds.py <old.parquet> <new.parquet> $O/v20261111/diff.csv
STUDY_FILE=data/train_data/intermediate_line_data_2_5_20261111.parquet python lab/analog_history_2026_11/prep.py $O/v20261111 480 720
python lab/analog_history_2026_11/report_rerun.py $O $O/v20261111
python lab/analog_history_2026_11/ridge_vs_model.py $O 60 720
python lab/analog_history_2026_11/dashboard_check.py 2026-10-28 2026-11-11 1
```

Run long jobs detached (`setsid nohup ... > $O/run.log 2>&1 &`).
