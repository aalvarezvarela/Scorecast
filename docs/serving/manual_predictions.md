# Manual predictions and bets from the dashboard

Status: **Implemented** (2026-11-06). Needs the `app` schema on Aiven and the
dashboard secrets before it can be used (see "Setup").

Signed-in, approved users of the [dashboard](streamlit_dashboard.md) can
launch a prediction run themselves and record the bets they place on it. The
run never happens in Streamlit: Community Cloud has neither the S3 models,
the runner's caches nor the memory, and a run takes 5–20 minutes. The
dashboard stores a request and starts a GitHub Actions workflow on the
self-hosted runner, as cron-job.org does for the scheduled runs
(`nba_ou.scheduling.cronjob_org`).

## Flow

1. **Request.** The Bets page inserts a row in `app.prediction_requests`
   (status `queued`) per chosen model and dispatches its workflow with
   `request_id`. A dispatch GitHub refuses marks the request `failed`.
2. **Run.** The workflow queues behind any scheduled run (same concurrency
   group, `queue: max`) and respects `PREDICTIONS_ENABLED`. The script marks
   the request `running` when it starts and `done`, `partial` (some model
   failed) or `failed` (nothing predicted) when it ends
   (`nba_ou.postgre_db.app_aiven.requests`). A job that dies before the
   script can report is marked `failed` by its last workflow step. A request
   still queued after 3 hours shows as expired.
3. **Store.** The predictions go to the prediction store like any run
   (`runs.workflow` 1 or 3) with `runs.request_id` set. Today shows them, the
   card details say "Manual #id"; Yesterday and Historical analysis leave
   them out (`r.request_id IS NULL`), so the evaluated series stays the
   scheduled runs.
4. **Bet.** The Bets page lists the user's manual predictions (optionally the
   scheduled ones too) of games not yet started. A bet records side, line,
   decimal odds, stake and book in `app.bets`, with the prediction it
   followed (`model_build_id`, `input_id`, `request_id`). `app.bets_full`
   grades it against `prediction.games`.

## Model choices

Every request covers the whole slate (the games not yet started) and reads
each game's **latest snapshot**, the data of the moment the run starts. The
horizon only chooses the model (XGBoost) or the training rows (TabPFN), and is
computed when the job starts, not when it was requested.

| Choice | Workflow | Inputs | What runs |
|---|---|---|---|
| XGBoost closing | `nba_predictor_daily.yml` | `slots=closing` | The closing slots on tonight's closing rows (latest lines). |
| XGBoost intermediate, each game's horizon | `nba_predictor_daily.yml` | `slots=nearest` | Per game and target, the slot of the nearest horizon: games at 4, 8 and 12 hours use the T-240, T-480 and T-720 models. |
| XGBoost intermediate, fixed T-H | `nba_predictor_daily.yml` | `slots=H` | The T-H slots for every game. |
| TabPFN closing | `nba_predictor_daily_tabpfn_client.yml` | `line=closing` | The closing fit, as the scheduled closing job. |
| **TabPFN intermediate, each game's horizon** (default) | `nba_predictor_daily_tabpfn_client.yml` | `line=intermediate input=latest horizon_minutes=nearest` | Games grouped by nearest horizon; one fit per group on that horizon's training rows, each predicting its games. One stored run per group. |
| TabPFN intermediate, fixed T-H | `nba_predictor_daily_tabpfn_client.yml` | `line=intermediate input=latest horizon_minutes=H` | One fit on the T-H rows for every game. |

TabPFN requests pass `update_db=false` and fit all three targets in standard
mode. "Nearest" is `argmin |H − minutes to tip|` over the enabled horizons,
ties to the longer one (`nba_ou.prediction.horizon_choice`), the rule the
dashboard already uses to pick among stored outputs. A latest row predicted
by a T-H model is limited to what its training rows carried
(`limit_to_training_support`), as in the predictor's `latest` mode. Games
already under way are left out.

The scheduled runs are unchanged: `predict_nba_games.py` without `--slots`
runs every enabled slot, `predict_tabpfn.py` without `--input` reads the
horizon's own row.

## Schema `app`

DDL in `nba_ou.postgre_db.app_aiven.schema`, created by
`scripts/prediction_store/create_app_store.py` after the prediction store.

- `users`: one row per Google account that signed in. `status` 0 pending,
  1 active, 2 disabled; `is_admin`. Emails in the `ADMIN_EMAILS` secret
  start active and admin.
- `prediction_requests`: `kind` (1 XGBoost, 3 TabPFN), `dataset_type`
  (1 closing, 2 intermediate), `horizon_minutes` (NULL = each game's own),
  `status` 0 queued, 1 running, 2 done, 3 partial, 4 failed, timestamps and
  `error`.
- `bets`: `market` (1 totals, 2 spread), `side` (+1 OVER/HOME, −1
  UNDER/AWAY), `line` (totals: the total; spread: the **home handicap** as
  books print it, home −3.5 is `-3.5`), `odds_decimal`, `stake`, `book`,
  the optional prediction references, `placed_at`, `voided_at`. A bet can be
  voided until tip-off; nothing is deleted.
- `bets_full`: `result` (`pending`, `win`, `loss`, `push`, `void`) and
  `profit` (stake × (odds − 1) won, −stake lost, 0 push/void). The same rules
  are in `nba_ou.dashboard.domain.bets`, and a test checks both agree.
- `prediction.runs.request_id`: the column belongs to the prediction store,
  the foreign key to this schema.

The dashboard connects as its own role `dashboard_app`: read on `prediction`,
read and write (no DELETE) on `app`.

## The bet form

Defaults, all editable: the model's pick as side, the stored bet365 line of
the prediction (spread shown as the home handicap), the stored bet365 price of
that side as decimal odds (−110 when missing), the user's last stake or 10 €,
and bet365 as book.

## Setup

1. Google Cloud: an OAuth client of type *Web application* with the redirect
   URIs `http://localhost:8501/oauth2callback` and
   `https://<app>.streamlit.app/oauth2callback`.
2. Aiven, once (prompts for the role's password):

   ```bash
   python scripts/prediction_store/create_app_store.py --print   # review
   python scripts/prediction_store/create_app_store.py --role
   ```

3. Streamlit secrets (Cloud *Secrets* field, or `.streamlit/secrets.toml`
   locally, never committed):

   ```toml
   AIVEN_DB_URL = "postgresql://...?sslmode=require"          # read-only pages
   AIVEN_APP_DB_URL = "postgresql://dashboard_app:...@...?sslmode=require"
   GITHUB_DISPATCH_TOKEN = "github_pat_..."  # fine-grained, this repo, Actions: read and write
   ADMIN_EMAILS = ["you@example.com"]

   [auth]
   redirect_uri = "https://<app>.streamlit.app/oauth2callback"
   cookie_secret = "<long random string>"
   client_id = "<Google client id>"
   client_secret = "<Google client secret>"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```

4. Merge to `main`: the workflows the dashboard dispatches run from `main`.

## Code

- `nba_ou.prediction.horizon_choice`: scopes and the nearest horizon.
- `scripts/predict_nba_games.py` `--slots`, `--request-id`;
  `scripts/predict_tabpfn.py` `--input latest`, `--horizon-minutes nearest`,
  `--request-id` (`fit_nearest`, `tabpfn_run.latest_rows`).
- `nba_ou.postgre_db.app_aiven`: schema and request status writes.
- `nba_ou.dashboard.domain.{access,manual,bets}`,
  `repositories.{app_store,dispatch}`, `ui.{auth,bets,admin,app_loaders}`,
  pages `welcome`, `pending`, `bets`, `admin`.
