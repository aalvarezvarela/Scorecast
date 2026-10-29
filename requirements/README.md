# Per-job requirements

Generated files: do not edit. `pyproject.toml` is the single source of
dependencies. Each file here is `poetry export` of `poetry.lock` restricted to
the Poetry groups one job uses, with every transitive package pinned and
hashed. Workflows install with pip and never need Poetry:

    pip install -r requirements/<file>.txt
    pip install --no-deps -e .    # the repo itself, without its Poetry deps

Keep `--no-deps`: the requirements file already pins everything, and a plain
`pip install -e .` would resolve the main dependencies again, outside the lock.

The standalone web dashboard is separate: `streamlit/requirements.txt` is a
small, directly maintained deployment manifest, outside these generated job
exports. Community Cloud selects it next to `streamlit/streamlit_app.py`, so
it installs only the dashboard runtime. It pins Streamlit independently of
the full-project Poetry lock; shared numpy/pandas/pyarrow/psycopg versions
match the project pins. See `docs/serving/streamlit_dashboard.md`.

## Files

| File | Poetry groups | Used by |
|---|---|---|
| `core.txt` | main | `update_finished_matches_daily`, `db_backup`, `schedule_snapshot_triggers` |
| `db-update.txt` | main, scrape | `update_database_daily` |
| `predict-cuda.txt` | main, cuda, scrape | `nba_predictor_daily`, `nba_predict_past_snapshots` |
| `predict-cpu.txt` | main, cpu, scrape | not used yet |
| `predict-tabpfn-cuda.txt` | main, cuda, scrape, tabpfn | `nba_predictor_daily_tabpfn_client` |
| `predict-tabpfn-cpu.txt` | main, cpu, scrape, tabpfn | not used yet |
| `retrain-cuda.txt` | main, cuda | `retrain_prediction_models` |
| `retrain-cpu.txt` | main, cpu | manual model checks on a CPU-only machine |
| `tests-cpu.txt` | main, cpu, scrape, train, dashboard, dev | `tests` |

The groups are defined and commented in `pyproject.toml`. `retrain-*` leaves
out `scrape` on purpose: the refit builds the closing frame for history, so it
never reaches the odds scrapers or the injury-report PDFs, although those
modules are in its static import closure.

`tests-cpu.txt` includes `dashboard` because the suite imports the chart modules
and uses Streamlit's `AppTest`. Both Streamlit and its Altair dependency must be
installed for test collection.

No file includes the `browser` group (Playwright): every job scrapes over
plain HTTP, and Chromium is only needed by the manual scrapers (SBR's
`--engine browser` fallback, the All-Star voting scrape). Playwright is
imported only where those run, so the jobs import without it.

## CUDA and CPU

XGBoost ships as two distributions with the same `xgboost` import:

- `cuda`: the default PyPI wheel. It is built with CUDA, needs NCCL
  (`nvidia-nccl-cu12`, a large download) and also runs on a machine without a
  GPU. The specs are fitted with it, so jobs that fit models use it.
- `cpu`: `xgboost-cpu`, the CPU-only build of the same version, with Linux and
  Windows wheels only. It is a much smaller install, for hosted runners. It
  loads GPU-fitted boosters and predicts the same values; manual S3 model
  checks can use it. A spec that asks for `device=cuda` cannot be refitted on it
  (`refit.require_device` fails).

The `cpu` group is optional, so `poetry install` installs `cuda` only. Never
install both in one environment: they overwrite each other's files.

## Changing dependencies

    # edit pyproject.toml (pick the group the import belongs to), then
    poetry lock
    python scripts/export_requirements.py
    # commit pyproject.toml, poetry.lock and requirements/ together

The export needs `poetry self add poetry-plugin-export`. CI (`tests.yml`, job
`requirements`) runs `poetry check --lock` and
`scripts/export_requirements.py --check`, which fails on any stale file. To add
a job or change a job's groups, edit `JOBS` in the export script.

Core pins are exact on purpose: the history memo and the input cache key on
installed package versions, so every job that builds or reads frames must have
the same pandas, pyarrow and builder dependencies.
