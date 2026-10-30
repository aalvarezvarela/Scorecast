# Intermediate opening-total parity

The intermediate prediction builder now reads the main SBR page's opening
total before the game's historical odds row is stored. It applies the same
American-price conversion and total normalization as training.

Previously, `create_base_game_features` read this field only from historical
odds. The past-snapshot workflow excludes the evaluated date from its database
update, so the feature was missing in prediction and present in next-day
training. The timestamped per-book opener was a separate feature and did not
populate `ODDS_TOTAL_LINE_consensus_opener`.

## Data flow

- Live prediction reuses `df_odds_sportsbook_scheduled` when it comes from SBR.
  When Covers supplies current prices, a separate read of SBR's main totals
  page prefers its opening quote. A valid SBR opener does not require a fresh
  current-price feed. If that read fails or lacks a complete quote, available
  fallback openers are retained and read failures are logged.
- `predict_past_snapshots.past_scheduled_data` reads the main totals page into
  `df_odds_sportsbook_opener_scheduled`, independently of the stored odds row.
  A read failure is reported and leaves the opener missing rather than
  discarding all predictions or substituting a subsequently stored value.
- `tonight_intermediate_frame` passes these raw opening quotes through
  `create_intermediate_line_df` to `create_base_game_features`.
- `merge_scheduled_opener` takes only the opening total's over/under lines and
  prices, restricted to the scheduled NBA game IDs. It uses
  `merge_yahoo_sportsbook_odds` for the existing conversion and normalization.
  Historical rows are unchanged; absent scheduled quotes remain missing.

The opening quote must include both prices. For example, a 229.5 opening total
at over -115 / under -105 becomes 230.0 under the existing training settings.
Copying 229.5 without its prices would leave another training/prediction mismatch.
The main-page opener is also not interchangeable with a median across books
or the first BetMGM history tick: NYK–DET on 2026-10-27 has a main-page 217.5
(-115/-105), normalized to 218.0, versus a first history quote of 218.5.

## Validation on 2026-10-30

The four main SBR totals pages for October 26–29 were read and passed through
the corrected opener merge. All 27 games present in the comparison training
partition matched its opening-total feature. Their archived T-240 prediction
rows had the feature missing in all 27 cases. The 28th replay game, ATL–CLE on
October 29, was absent from that training partition and was not counted.

The training reference is immutable partition
`season=2026.ef817072d034273c.parquet`, written October 30 at 07:30 UTC,
checksum `sha256:ef817072d034273c`. For NOP–SAC and POR–MEM on October 29,
the prior investigation matched its aggregate checksum to the exact next-day
fit (`sha256:d7706ab1d999b0d5`). October 26–28 comparisons use the October 30
version; they do not establish equality to those dates' next-day versions.

The main page for October 30 also supplied opening totals and both prices for
all ten scheduled games at 16:56 UTC, before every game's T-240 snapshot.

## WAGERS / public betting percentages

Intermediate models exclude **current-game** Yahoo bet/money percentages and
SBR consensus percentages. They lack timestamped history for past snapshots.
They retain percentage statistics computed from earlier games.

The 136 percentage features used by the three inspected T-240 intermediate
specs were compared between archived replay inputs and that same training
partition: **27 games × 136 features = 3,672 comparisons, zero differences**.
Thus this sample shows no matching WAGERS discrepancy in intermediates.
The new opener merge explicitly excludes current percentages, closing quotes
and scores, so fetching a main page after the game cannot import those fields.

**Reconstructed closing remains a separate limitation.** Its model uses raw
current-game percentages, but `closing_replay.historical_odds` reconstructs
book prices without timestamped Yahoo percentages or historical consensus
counts. In the archived NOP–SAC T-240 case, all 14 such features used by the
closing models are missing. Examples:

| Feature | Archived closing replay | Training row |
| --- | --- | --- |
| Yahoo total % bets over | NaN | 70.0 |
| Yahoo total % money over | NaN | 69.16 |
| SBR total consensus % over | NaN | 51.0 |

These final percentages cannot safely fill a four-hours-before-tip input.
Resolving this requires archived percentage observations at the desired
cutoff, or a separately trained closing model that excludes unavailable
current-game percentages. This change does neither and does not claim to
restore closing replay parity.

Opening totals are read as published by the main page at the time of the
reconstruction. The change does not recover historical revisions to that
page or establish exactly when a particular opening quote first appeared.
Strict reproduction of live observations requires an archive of those reads.
