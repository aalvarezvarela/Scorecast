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
default. Yesterday shows stored predictions and results; Historical remains
a title-only placeholder without database reads. Page scripts compose
modules in `nba_ou.dashboard`:

- `repositories/`: bounded, parameterized reads from `prediction.games` and
  `prediction.predictions_full`, joining model builds for `anchor_book` and
  runs for completion timestamps, in one repeatable-read transaction.
- `domain/`: pure selection, conversion, model identity, team logos and dates.
- `ui/`: source caching, filters, native cards, tables, and Altair comparisons.

The design uses the Streamlit skill's native layouts, an NBA-inspired navy,
red and white theme, bounded caching, and fragments. The NBA logo's blue
`#17408B` and red `#C9082A` anchor the palette over a mid-navy `#182A4E`
background; lighter tints keep signal text at 4.5:1 contrast or better.
OVER/HOME use red, UNDER/AWAY use blue. A scoped stylesheet handles the
background glow, card surfaces with a red/blue top edge, growing card widths
and larger menus. Totals comparisons use `st.html` for
semantic spanning table headers, which native `st.table` does not render.
No JavaScript or legacy HTML components are used. APIs are compatible with
the installed 1.54.

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
   The market selector offers Totals (both `line_error` and `total_points`
   together) and Spread. Each underlying target keeps its own closing and
   nearest intermediate prediction; neither is averaged away. Rows and chart
   labels identify the target explicitly.
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
12. Each game reduces to one instruction (`domain/picks.py`): the side with
    the most directional votes wins, with one equal vote per prediction.
    Predictions below their minimum abstain but count in the `n of m`; those
    without a line do not vote. Tied directions show "No bet: tied vote";
    no directional votes also means no bet. The bet names the line at
    the selected reference: `OVER 230.5`, or for spread the team and its
    handicap (implied home margin 3.5 → home `-3.5`, away `+3.5`). Bets are
    listed after all games, with the summary metrics, in both Cards and Table.
    Totals is the default market; charts are off by default. In Totals, the
    headline considers both targets and both closing/intermediate outputs
    from every available displayed model, including TabPFN, with no model
    weights. Three UNDER votes and one OVER vote therefore give UNDER; the
    caption discloses both counts instead of requiring unanimity.
13. A model kind with no live output for the target (typically TabPFN before
    its first run of the day) is still listed in every card.
    Model selection is currently fixed: registry/XGBoost schema `2_5`, plus
    available TabPFN builds. Other registry schemas are excluded. There is no
    model-selection widget; old session preferences cannot hide a model.
14. Line freshness: the line history stores changes only, so a quote's stored
    age is how long Bet365 had gone without moving line or price at the
    snapshot. Cards show that as "Last Bet365 move HH:MM" (snapshot time minus
    age) next to the snapshot time, and flag the snapshot itself once it is
    60 minutes old, since the line may have moved after it.
15. Cards and Table share the game order: tip-off by default, or the largest
    absolute market edge per game (missing edges last, tip-off breaks ties).
    Within a game, Largest edge also orders the prediction rows by magnitude.
16. Cards wrap by their content width and grow to share any unused space in
    each row. There are no configured pixel widths or caps on the number of
    games per row; browser layout reflows them as the available width changes.
    Body text uses a 16px base, so content-sized cards and their spacing scale
    with the slightly larger text while retaining responsive wrapping.
    Logos are 48px and team names stay written in full, resolved from the NBA
    team IDs in `prediction.games`. Both teams, logos and Madrid tip-off text share a white header with
    dark NBA-blue text, on both Today and Yesterday cards. UNDER/AWAY use a
    brighter sky-blue tint (`#7AD7FF`) consistently across headings, badges,
    tables and charts. The same team names appear in the table,
    details and tonight's bets. Each totals card uses one table with Model
    and Snapshot row labels, grouping closing and intermediate rows under
    each available model. Total points and Line error appear in adjacent
    column groups. Each target has separate Prediction and Margin columns
    (four data columns in total); margins retain the colored pick. Different
    target horizons remain labeled individually; missing
    targets are shown as a dash, never a synthetic prediction. Pivoting the
    display does not combine or remove any vote.
    Column-group headers span the Prediction and Margin columns, numbers
    align right, and the table scrolls horizontally on narrow screens.
    TabPFN labels omit the `v3_5` suffix; family keys and votes retain it.
    Navigation and menu options use 50% larger text and taller controls;
    card text retains the 16px base.
    The pick headline includes its reference, market mean, median and book
    count in small text alongside it. Directional vote counts appear below
    in bold text, coloring only the winning direction and keeping the other
    direction gray (both gray for a tie), without the previous majority
    sentence or a separate market-line row. No-bet cards still show the
    reference line.
    Today's title and the native top-bar logo use a local silhouette-only
    SVG derived from `https://cdn.nba.com/logos/leagues/logo-nba.svg`.
    The lettering is removed, retaining the original red and blue paths.
    The same mark is used as the browser-tab icon.

## Yesterday's predictions

The previous **NBA slate** is `slate_date() - 1 day`, using the same Pacific
date boundary as Today, with every displayed time converted to Madrid.
The target selector separates Line error, Total points and Spread. Model
selection remains registry schema `2_5` plus available TabPFN families.

The default source is **Live predictions**. **Snapshot replay** switches
intermediate rows to stored exact own-horizon outputs; it does not rerun any
model. Both choices share the same recorded live closing predictions.

Selection is independent for every game, model family and target:

| Model type | Checkpoints | Selection |
| --- | --- | --- |
| Closing | T−6h, T−4h | Latest live output fully available by the checkpoint, at most 30 minutes old. |
| Closing | Final pre-game | Latest live output fully available before tip-off; actual time remains visible. |
| Intermediate, live | T−8h, T−6h, T−4h | The model trained at that horizon, on an input from the preceding 30 minutes; publication must be between 30 minutes before and 30 minutes after the checkpoint, and before tip-off. |
| Intermediate, replay | T−8h, T−6h, T−4h | Exact match of model and input horizon, using the latest own-snapshot reconstruction (workflow 2 for XGBoost; own-snapshot workflow 3 for TabPFN). |

Availability is the later of prediction time and run completion.
Closing T−6h/T−4h also require the output itself to have been produced in the
preceding 30 minutes; finishing a long run does not make an old output fresh.
Future input snapshots are rejected even if TabPFN labeled a pre-horizon row
as `own`.
Live never reads workflow 2; replay never silently fills an absent XGBoost
reconstruction with a live row. Intermediate live publication delays are
explained on the page; exact input, prediction and availability times are in
View details. Each slot is deduplicated across refits/reruns. Missing slots
are explicit and never replaced with another trained horizon.

Cards reuse the team header, NBA theme and wrapping/growing layout. Their
white header includes the final away/home score, total points and home margin
alongside the teams and Madrid tip-off time. Below it is one table:
Model, Checkpoint, Predicted total (Spread error for spreads), Pick / Line,
Result. Line-error outputs use their original anchor to recover total points.
Original outputs, model lines, horizons and source metadata remain inspectable.
Games with no predictions remain visible, including their scores in Table view.

Every pick is recalculated with the selected per-kind minimum edge against
**that row's stored bookmaker quote**, never a shared latest/final line.
For spreads, the implied home margin is reconstructed before comparison.
The view's `pick_result` grades its stored emitted side, which is a different
policy; this page grades its displayed thresholded signal instead.

Results are WIN, LOSS, PUSH, NO BET, NO LINE, MISSING, PENDING or VOID.
WIN badges use green, LOSS badges use red, and other results remain neutral.
Only games marked final with both scores are settled. There is no majority across
hours or combined-target bet. Before the individual games, the daily summary
separates both XGBoost and TabPFN into Closing at 6h, Closing at 4h, Closing at closing (the
last available pre-game prediction), and Intermediate at 8h/6h/4h. Its compact
table shows evaluated games / all games, wins / settled picks, win rate and
MAE per checkpoint. TabPFN shows only intermediate checkpoints when no closing
model outputs were stored; absent games within an existing checkpoint retain
their missing-data status.
Win rate is `wins / (wins + losses)`; pooled metrics use pooled counts and
errors rather than averaging checkpoint percentages or means. MAE includes
available predictions for final games regardless of whether they cleared the
betting threshold. Game counts are distinct matches, not prediction rows.
Metric help shows exclusions; repeated checkpoint rows are not independent
games. Selection rules remain in Settings, and individual cards retain the
checkpoint details.

Yesterday reads the same two Aiven sources in one read-only transaction;
the games query includes final scores and the output query includes workflows
1/2/3 and just closing/8h/6h/4h model slots. Post-tip replay outputs are allowed
in the read, then separated in the domain layer. Both sources are fetched in
one cached read, so changing source, target or thresholds needs no extra SQL.
Results refresh every minute; the dashboard never calls the NBA API or the
result updater, and no database migration is needed.

## Read path and freshness timestamps

The freshness text makes **no additional query** and does not read any
`line_history` schema. `market_snapshots` uses values already in the cached
prediction frame. "Last Bet365 move" is `market_at - bet_line_age_minutes`;
"line read" is `market_at`. The former describes the last recorded quote
change (line or price), not proof that the line stayed unchanged after the
snapshot. Closing `market_at` is approximated by run start, as noted above.

One cache miss makes two SELECTs in one read-only transaction:

1. The date's `prediction.games` rows.
2. `prediction.predictions_full`, joined with `prediction.model_builds` for
   `anchor_book` and `prediction.runs` for `finished_at`.

The view itself joins five distinct physical tables, all in `prediction`:
`predictions`, `inputs`, `model_builds`, `games`, and `runs`. The two outer
joins repeat tables already present in that view; they are not calls to
another service or extra SQL round trips. Cache hits and market/model
presentation changes do not trigger these reads.

No repository query, prediction schema, or freshness calculation was changed
for the combined Totals layout. Possible follow-up improvements, **not
implemented here**: expose `anchor_book` and `run_finished_at` directly in the
view to simplify the outer query, select only dashboard columns instead of
`v.*`, and store a precise market-observation timestamp for closing inputs.
Measure actual query plans before adding indexes or materializing another
table; the timestamp display itself requires neither.

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
