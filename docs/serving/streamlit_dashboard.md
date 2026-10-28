# NBA prediction dashboard

The unified app reads **Aiven only**. It never calls Supabase, S3, a scraper,
the predictor, or a results updater. Logos retain the ESPN images used by the
original apps. Every displayed timestamp, including tip-off, prediction times
and market snapshots, uses `Europe/Madrid` with automatic DST changes.

## Run

From the repository root, with the main and dashboard Poetry dependency groups
installed (Streamlit 1.54+):

```bash
poetry run streamlit run streamlit/streamlit_app.py
```

Run from the root so Streamlit reads `.streamlit/config.toml`. Configure
`AIVEN_DB_URL` in the environment, the existing local Aiven configuration, or
Streamlit secrets (top-level `AIVEN_DB_URL` or `[Aiven].AIVEN_DB_URL`). The
dashboard explicitly selects Aiven regardless of `DB_ENV`; its connections
use read-only transactions with connection and statement timeouts.

`streamlit_app_old.py` and `streamlit_ou_app_old.py` preserve both original
applications as references. The new application does not import either.

## Pages and architecture

The entrypoint registers three pages using `st.navigation`. Today is the
default; Yesterday and Historical contain only their English titles and do
not query the database. Page scripts compose modules in `nba_ou.dashboard`:

- `repositories/`: bounded, parameterized reads from `prediction.games` and
  `prediction.predictions_full`, joining model builds for `anchor_book` and
  runs for completion timestamps, in one repeatable-read transaction.
- `domain/`: pure selection, conversion, model identity, team logos and dates.
- `ui/`: source caching, filters, native cards, tables, and Altair comparisons.

The design uses the Streamlit skill's native layouts, financial-dashboard
theme patterns, bounded caching, and fragments. No injected CSS or legacy
HTML components are used. APIs are compatible with the installed 1.54.

## Today's rules

1. The default NBA slate date follows the predictor's Pacific date. This is
   separate from displayed dates: tip-off always shows the full Madrid date
   and 24-hour time, so midnight never makes a game appear on the wrong night.
2. Only scheduled games with future tip-offs are shown. `prediction.games`
   contains games accepted by at least one prediction run, not an independent
   complete NBA schedule. Empty data therefore means no games recorded yet,
   not a verified off day. Games without a prediction for the selected target
   remain visible in Cards.
3. Only completed live daily/TabPFN runs and pre-tip outputs available at the
   read time qualify. Baselines and the retrospective past-snapshot workflow
   are excluded. A partial run's accepted outputs remain usable.
4. For each game, target and family (`kind`, `schema_version`, `variant`), show
   the latest closing prediction and one intermediate model: minimize the
   distance between its trained horizon and the game's actual time remaining.
   Equal distances prefer the longer horizon. For each slot, the newest
   output wins across fits. Different families may have different available
   horizons; labels disclose each one (particularly TabPFN).
5. Model horizon, input horizon, and prediction timestamp remain separate.
   Choosing a trained horizon does not imply its nominal snapshot has passed.
   The details expose both horizon fields and the original input mode.
6. Total points is the direct output. Line error reconstructs the total as
   `model_line + prediction_value`. Error vs model line is retained; the
   displayed betting edge subtracts the chosen market reference from that
   total. Changing the comparison reference never changes a predicted total.
7. Market values come from one coherent latest stored input snapshot per
   game/market, independently of model filters, with `anchor_book = bet365`.
   `bet_line_x2` is divided by two; cross-book mean, median and count are used
   as stored (main book included, `CONSENSUS` excluded). No missing quote or
   mean/median is filled from a different snapshot or assumed value.
8. Totals use OVER / UNDER against the selected reference (Bet365 by default).
   Spread reconstructs the home margin as `model_line + prediction_value`,
   then subtracts the selected current stored spread line to produce its
   HOME / AWAY market edge. Cards/tables retain the original spread error and
   show this separate market edge; the spread chart plots that same edge.
   Changing reference works for all targets. Missing lines produce UNAVAILABLE,
   never a pick against an old model line. Stored spread lines are implied
   home margins, not a displayed handicap for the home team.
9. Original inputs and prediction times are visible in details. For closing
   inputs, the snapshot time is approximated by run start; intermediate input
   times come from the stored tip-off minus input horizon. Market values are
   stored observations, not live quotes. Old predictions/quotes are marked.
10. Source reads are cached for 60 seconds, with a maximum of eight entries.
    Target/model filters and nearest-horizon selection run outside the cache.
    Today's fragment refreshes every minute while the session is active;
    Refresh data clears that slate's cached read without launching inference.
11. Directional badges require `abs(market_edge) > minimum_edge`; equality and
    smaller values show NO EDGE while the numeric values remain visible.
    The minimum is per model kind: 0.1 point for registry (XGBoost) models and
    0.01 for TabPFN, whose outputs sit much closer to the line. Both are
    adjustable in Settings over `(0, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 3, 4, 6)`.
    This is a display filter, not a calibrated confidence score.
12. Each game reduces to one instruction (`domain/picks.py`): a bet when every
    directional prediction points the same way (predictions below their
    minimum count in the `n of m`, those without a line do not vote), "models
    disagree" when any two oppose, otherwise no bet. The bet names the line at
    the selected reference: `OVER 230.5`, or for spread the team and its
    handicap (implied home margin 3.5 → home `-3.5`, away `+3.5`). Bets are
    listed above the cards. Line error is the default target; charts are off
    by default.
13. A model kind with no live output for the target (typically TabPFN before
    its first run of the day) is still listed in every card.
14. Line freshness: the line history stores changes only, so a quote's stored
    age is how long Bet365 had gone without moving line or price at the
    snapshot. Cards show that as "Last Bet365 move HH:MM" (snapshot time minus
    age) next to the snapshot time, and flag the snapshot itself once it is
    60 minutes old, since the line may have moved after it.
15. Cards and Table share the game order: tip-off by default, or the largest
    absolute market edge per game (missing edges last, tip-off breaks ties).
    Within a game, Largest edge also orders the prediction rows by magnitude.

## Validation

```bash
poetry run pytest tests/dashboard -q
poetry run ruff check src/nba_ou/dashboard streamlit/streamlit_app.py streamlit/app_pages tests/dashboard
```

Tests use synthetic fixtures and never require external services. They cover
per-game horizons, fresh output selection, input mode separation, exclusions,
market normalization and movements, missing data, signal thresholds, shared
card/table ordering, Madrid DST and UI navigation/targets.

Deferred: yesterday/history analytics, SHAP, inference controls, and an
independent schedule ingestion into Aiven. No compatibility adapter for the
legacy storage is part of this application.
