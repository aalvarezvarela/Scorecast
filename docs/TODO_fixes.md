# Fixes to schedule

Known defects that are not fixed yet, usually because the fix changes training
features and so needs a rebuild and retraining rather than a hot patch. Each
entry says what is wrong, why it was left, and how to fix it.

# For the 2_5 rebuild

Everything in this section changes 2_5 training features. Do it in one go: one
code change set, one dataset rebuild (closing and intermediate), one
history-memo invalidation, one retraining and comparison against the current
models. Until then, live prediction reproduces the current behaviour, defects
included, with one deliberate exception: roster membership (below).

## Injured and questionable players lose their last game

**Found:** 2026-10-16, while checking live-roster parity on
`fix/todays_prediction_data`.

**What is wrong.** `precompute_cumulative_avg_stat` gives every row the
EWMA of the player's *earlier* games (`shift(1)`), which is right for the row
of the game being predicted. A player who is Out/Doubtful (or Questionable) for
the target game has no row in it, so `latest_player_states(..., injured=True)`
and `_latest_states_on_roster(..., injured=True)` take his last row strictly
before the game. That row's value was computed for predicting *that* game, so
it excludes it. His most recent game is therefore always left out, although it
was played and including it leaks nothing.

Example: 10 points, then 30 points, then Out tonight. Training reads 10.0; an
average that includes the 30-point game reads 11.34 (EWMA with the
previous-season fill).

**Affected columns:** `TOP*_INJURED_*`, `AVG_INJURED_*`,
`TOTAL_INJURED_*`, the `QUESTIONABLE` twins, and the minutes threshold
(`MEETS_MIN_THRESHOLD`) that orders those rankings. Closing and intermediate
builds both read them.

**Why it was left.** Every trained model learned the current behaviour, and
live prediction must match training, not be more correct on one side only.
The live-roster path was aligned with the current rule on purpose.

**How to fix.**
1. For the injured and questionable groups, use the average updated with the
   player's last appearance: the value a row after that game would carry. The
   live roster path's synthetic same-day state already computes this; reuse it
   for both training and live.
2. Rebuild the closing and intermediate datasets with the rest of this section.
3. Invalidate the history memo for the player step. Its digests cover the
   inputs, not the code, so it would otherwise serve the old rows.
4. Retrain and compare with the current models as a normal experiment. The
   expected effect is small (a one-game lag, and injury information is mostly
   priced at close), so judge it against the noise floor.
5. Extend the live-roster parity tests with varying statistics. With constant
   statistics both rules give the same value and hide the difference.

## Roster membership lags transactions by one game

**Found:** 2026-10-16, checking whether live official rosters
(`CommonTeamRoster`) match training on `fix/todays_prediction_data`.

**What is wrong.** Training decides who belongs to a team-game in
`PlayerLookupIndex.lookup_for` (`past_injuries.py`): the player's last
box-score team strictly before the game, plus anyone the game's report lists
Out, Doubtful or Questionable. A season opener falls back to last season's
assignments. Box scores are the only transaction signal, so every roster move
is recognised one game late. Verified with the production functions on a
constructed trade (X traded from A to B after A's game 3):

| Game | Training | Official roster |
|---|---|---|
| A's next game, X gone | X still counted for A | not counted |
| B's first game with X, X unlisted or Available | X not counted for B | counted |
| Same, X listed Out/Doubtful/Questionable for B | counted (report) | counted |
| B's second game with X | counted | counted |

The same lag hits debuting rookies (not members until their first box score)
and released players (members of their old team until they play elsewhere).
The worst case is a season opener, where training uses last season's rosters.

**Affected columns:** everything built on membership: the available, injured
and questionable groups (`TOP*_PLAYER_*`, `TOP*_INJURED_*`,
`*_QUESTIONABLE_*`, `AVG_*`, `TOTAL_*`, `N_*` counts), availability effects,
All-Star vote shares and roster continuity.

**Why it was left, and the accepted mismatch.** Models learned the lagged
membership. Live prediction nevertheless uses the official roster (decided
2026-10-16), so tonight's predictions reflect trades, signings and releases.
Until training changes, live rows deliberately differ from training rows in
exactly the cases in the table above.

**How to fix.**
1. Pick a game-day membership definition that training can reproduce for
   every past game and live can reproduce before tip. Candidate: the game's
   box-score players plus its inactive list. That is close to the official
   roster on game day and is roster information, not outcome information; it
   must not depend on minutes played (see `nba_ou.config.leakage`).
2. Live uses `CommonTeamRoster` (archived per fetch under
   `train_data/live_inputs/v1/rosters/`) plus report-listed players. Measure
   how often it differs from the training definition on archived days before
   trusting the match.
3. Season openers then need no prior-season fallback.
4. Rebuild, invalidate the memo and retrain together with the other entries in
   this section.
