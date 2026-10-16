# Live prediction rosters

Closing and intermediate predictions use NBA Stats `CommonTeamRoster` as the
official total player pool for a scheduled team-game. This is membership, not
confirmation that a player will dress or play. `POSITION` is archived for
reference; it does not establish a starting lineup.

The shared scheduled-game fetch loads each participating team/season once before
synchronizing injuries. The response's named `common_team_roster` dataset must
contain valid, unique player IDs, names, the requested team and season. Its IDs
also resolve new injury-report players missing from historical box scores and
nba_api's bundled player catalog. These current roster matches apply only to the
scheduled game; past report/game matching keeps its existing behavior.

For each covered team-game, the total pool is the official roster union every
player named on the eligible official injury report, including Available and
Probable designations. A report can add a player missing from the roster. Existing
Out/Doubtful and Questionable classification rules then determine availability.
Old inferred members cannot expand a successful official roster. If the endpoint
and its cache are unavailable, historical membership remains the fallback and
report-listed players can still supplement it.

Membership is separate from statistical evidence. A rookie without appearances
remains in the pool with NaN historical statistics. A transfer can use their
previous team's strictly earlier statistics. No current game box-score values
are used. Existing counts of players with *recent minutes* keep that definition;
they are not relabeled as total roster size.

Player history comes from the same prepared rows as training, including DNP and
missing-minute rows. Out and Questionable players use the last row strictly
before the game, with its existing shifted EWMA and minutes threshold. Only
available players can read the synthetic same-day state. A roster member with
no earlier state keeps unknown statistics; membership alone supplies no value.

## Storage and failure handling

No live roster is written to PostgreSQL or a local file. S3 uses:

```text
train_data/live_inputs/v1/rosters/
  game_date=2026-10-16/season=2026-27/team_id=1610612744/
    index.json
    fetched_at=2026-10-16T18-00-00.000000Z/players.parquet
```

Parquet uses Zstandard compression. Only changed assignments produce another
snapshot. The small manifest lists timestamped versions and checksums, so a
fallback gets a known key directly without listing the bucket. Conditional
manifest writes protect against concurrent runs replacing each other's index.
Failed cache writes do not discard a successful live roster.

The in-memory `roster_snapshots` frame includes team, season, player ID/name,
position, UTC observation time and source (`nba` or `s3`). Failures are isolated
per team and logged; absence of a snapshot requests historical inference and
never asserts an empty or healthy roster.

## Intermediate cutoffs and historical builds

`fetch_live_game_day` accepts the same `roster_snapshots` returned by
`get_all_info_for_scheduled_games`, avoiding duplicate live fetches. Without
them, a current run loads rosters before synchronizing injuries. An explicit
`as_of` uses archived observations, without making a current roster request or
refreshing PDFs. A manually constructed `LiveGameDay` with no roster frame can
also load the archive at its `as_of`.

Every intermediate horizon uses the same pool as the closing prediction: the
latest same-day roster observed by the live run's cutoff, even when it was
fetched after that horizon's snapshot timestamp. Rosters cannot be recovered
afterwards the way report PDFs can, so a per-horizon limit would leave the row
being predicted (whose snapshot precedes the run) on historical inference. The
accepted cost: a transaction between a horizon and the run shows up early. If no
same-day observation exists, every horizon keeps historical inference. Player,
All-Star and continuity features use the selected pool; injury report coverage
and NaN masking retain their existing policy.

Historical training builds without live inputs keep their existing membership
logic, which recognises a transaction one game late. Live official rosters
deliberately differ from it for new arrivals, debuts and departures; this
train/serve mismatch is accepted until training adopts a game-day definition
(see [TODO fixes](TODO_fixes.md));
timestamped observations support reproducible live replay from their first
observation onward, not retroactive reconstruction of older roster changes.
