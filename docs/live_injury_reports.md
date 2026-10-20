# Injury reports before today's prediction

Predictions read injury information from Aiven. Before building features,
`get_all_info_for_scheduled_games` calls
`fetch_data.injury_reports.sync.synchronize_injury_reports`, then reads Aiven
at the captured prediction cutoff. It no longer parses a separate latest PDF
into a second, potentially inconsistent player dictionary.

## Reports and durable receipts

The first refresh creates `injury_report.ir_document` alongside the existing
injury schema. It records each source URL, expected publication time, validated
PDF publication time, exact S3 key, archive time, ingestion completion time, and
last error. An `ir_report` row is a report observation; a completed document
receipt proves the live writer finished updating the associated timelines.
Historical report rows without a live receipt are not assumed fully ingested.

Reports are assumed immutable. Completed documents are not requested from NBA,
read from S3, or parsed again. Identical statuses across two publication times
still produce two `ir_report` records; only the status intervals collapse.

The synchronizer enumerates the previous and requested Eastern dates using the
existing era-aware CDN URL schedule, excluding future publication slots. This
also catches PDFs no longer linked on the season landing page. It compares
individual URLs against Aiven receipts, rather than skipping everything older
than the newest loaded timestamp. It is a game-day catch-up window, not an
all-season backfill; older historical gaps still use the archive tools.

An unregistered or unfinished slot first makes a direct GET for its deterministic
S3 key. If absent, it downloads from NBA, validates the PDF and header time,
uploads the bytes, records the archive receipt, then parses in memory. No local
PDF or local manifest is written. A 403/404 from NBA remains retryable because
that CDN uses 403 for both missing publications and throttling. Successful
documents are never re-fetched to check for corrections. Requests are sequential
and spaced by 0.6 seconds. The first catch-up can process many PDFs; later runs
skip completed reports.

S3 paths use the existing archive naming convention under the writable prefix:

```text
train_data/injury_reports/raw/season=2026-27/date=2026-10-16/
  injury-report_2026-10-16T1415-0400.pdf
```

The original PDF stays in S3; player and team state lives in Aiven. A successful
S3 upload followed by failed database bookkeeping is recovered by the same exact
key on retry, without another NBA download. S3 access errors are not treated as
missing objects. A failed S3 write prevents that document from being ingested.

## Incremental database update

`postgre_db.injury_report_aiven.live_ingest` resolves upcoming games from the
schedule, supplemented by the fresh scoreboard. Player matching uses historical
team rosters followed by unique exact names from the NBA player catalog for
new signings and opening-night rosters. A name neither route resolves is
dropped, logged and recorded in `ir_unresolved` in the same transaction, as the
historical ingest does: nba_api's bundled catalog lags each draft class, and one
unknown rookie must not block every other player's status. Unknown games still
fail the batch and leave receipts pending.

Live predictions additionally pass official `CommonTeamRoster` names and IDs to
the resolver before historical matching. These matches are scoped to the
scheduled team-game, so an opening-night rookie can resolve without a prior box
score or a bundled catalog entry. See [live rosters](live_rosters.md).

For affected games, existing compressed intervals are expanded at their original
report instants and combined with the new parsed observations. Completed PDFs
do not need to be parsed again. Timelines are rebuilt per game, including
player removals, filing changes, and games disappearing with empty filed lists.
A late report can fill a gap without losing later known observations.

One transaction holds a writer advisory lock, replaces only affected game
intervals, writes every new report record, and marks its document receipt
complete. A failed write rolls back all of these changes. Existing backfill
loader functions retain their default commit behavior; the live writer opts out
of internal commits to own the transaction.

## Feature reads and failures

Today’s closing feature state uses `load_prediction_report_state`: historical
games retain their training state, while today’s games are replaced with an
explicit Aiven snapshot strictly before `min(prediction cutoff, tipoff)`.
The same state feeds the scheduled absence dictionary and richer report features.
Intermediate snapshot readers already use their own strict publication cutoffs,
so newly stored reports remain invisible to earlier horizons. Intermediate model
serving uses the tonight-only path from `perf`: `fetch_live_game_day` synchronizes
current runs, then the builders read Aiven at each horizon. The scheduled-data
path shares its official roster snapshots with that bundle. Explicit `as_of`
reads do not refresh PDFs; archived-PDF overlays remain available to historical
replay diagnostics, rather than current predictions.

Refresh failures log an error and leave Aiven at its previous known state.
Existing stored reports can still be used with their report-age features.
The closing job's existing filing requirement remains: games with an uncovered
team are excluded, and if none have complete filing coverage the job stops.
Unknown coverage does not imply a healthy roster.

The shared fetcher accepts `dataset_type="intermediate_line"` to retain games
even when neither team has filed. Its exclusion list (`games_not_updated`)
is empty in this mode; `games_missing_injury_reports` preserves the coverage
diagnostics. Intermediate features already set report coverage to 0 and mask
the uncovered side's injury/availability features to NaN, including derived
availability effects. No settled inactive list or later report is substituted.
The default `dataset_type="closing_line"` retains the closing filing policy.

Verified against both `intermediate_line_data_2_5_20261003.parquet` and
`intermediate_line_data_2_6_20261003.parquet` (146,815 rows each): home report
coverage is absent in 8.59% of rows. Home report age, Out counters, injured-player
points and Questionable-player points are NaN for all those uncovered rows.
Missing home coverage is 0.18% at T-60, 9.88% at T-360, and 15.10% at T-720.
This missingness occurs in every season in the files (2019–2025), so accepting
NaNs at prediction time matches the intermediate training data.

The prediction workflows supply `AIVEN_DB_URL` from the corresponding GitHub
secret. Runtime credentials need DDL permission to create the receipt table and
missing season partitions, plus read/write access to the injury-report schema
and read/write access to the S3 prefix. The old `save_reports_path` and injury
landing-page arguments remain accepted for compatibility; the prediction path
does not use them to write local files or select a single latest PDF.

Tests cover retries and S3 recovery, skipped completed reports, unchanged and
empty reports, late reports, removals, not-submitted teams, transaction rollback,
opening-night player resolution, and prediction-time state replacement. These
are isolated tests; running the actual predictor also updates other databases
and is a separate operational action.
