# NBA findings that inform starting choices

These observations come from this repository's NBA totals/spread experiments.
They are evidence for what to test first in another sport, not universal rules.
Preserve the distinction between a controlled comparison and a historical
association.

## Temporal CV geometry

Historical fixed-game runs used approximately 12 folds of 50--60 games, with a
60-game step, covering roughly 650 OOF games. More recent rolling-origin runs
used about 23--28 smaller folds over approximately 850 OOF games.

A controlled panel held eight model candidates and 850 OOF games constant while
changing retraining cadence:

| Target | 5 date / 23-fold holdout | 10 date / 12-fold holdout | Observation |
|---|---:|---:|---|
| line error MAE | 14.3532 | 14.2553 | 10-date selected better |
| line error win rate | 53.11% | 56.46% | 10-date selected better |
| total points MAE | 14.4127 | 14.3852 | small 10-date gain |
| total points win rate | 52.71% | 53.48% | small 10-date gain |

Twenty-date/seven-fold CV did not improve consistently. This supports an
intermediate region rather than “fewer folds are always better.” The cadence
change also changes model age and training histories, so the effect cannot be
attributed to fold count alone.

Practical prior for a new fixed-game NBA campaign: 12 folds, 50 validation
games, 60-game step. A different sport must translate event count by schedule
density and retraining cadence.

## Boosting rounds and early stopping

Restoring fold-local early stopping was one of the strongest useful historical
changes:

| Target, modern 2.2 | Tuned global rounds CV MAE | Fold-local early-stop CV MAE |
|---|---:|---:|
| line error | 13.6363 | 13.4547 |
| total points | 13.9111 | 13.6852 |

For total points, holdout MAE improved from 14.5216 to 14.4575. For modern line
error, global rounds had slightly better holdout MAE (14.3333 vs 14.3590), while
early stopping had slightly better broad directional accuracy. In a replay on
the historical CSV, early stopping improved line-error broad win rate and
total-points high-edge win rate.

Interpretation: prefer early stopping as a candidate/default to test here, not
as a law. The current implementation stops on the outer validation fold and
scores that same fold, so Optuna-facing CV metrics are optimistic. A new sport
pipeline should use a chronological inner tail of training for stopping if
possible.

## Missing rows and correlation pruning

Fixed-hyperparameter screens used the same 643 validation games across every
cleaning policy.

### Line error

The current permissive row policy was clearly better:

- `max_na_per_row=300`, global correlation `0.95`, odds correlation `0.99`:
  pooled CV MAE 13.6918;
- `max_na_per_row=80`, same correlation policy: 13.8116;
- `max_na_per_row=300`, correlation `0.995` everywhere: 13.8060.

Paired intervals indicated approximately +0.11 to +0.12 MAE degradation for
the stricter/historical alternatives. For this target, discarding incomplete
rows removed useful history.

### Total points

The same screen was inconclusive:

- `NA<=300 / corr=0.995`: MAE 13.8717, win rate 55.07% at edge 0.1;
- `NA<=80 / corr=0.95, odds=0.99`: MAE 13.8756, 54.76%;
- current `NA<=300 / corr=0.95, odds=0.99`: MAE 13.8838, 54.02%.

The apparent MAE gains were only 0.008--0.012 and paired intervals crossed
zero. Combining strict rows with relaxed correlation performed worse (52.41%
win rate), so effects were not additive.

Interpretation: a permissive row limit and market-aware correlation threshold
are good starting points, particularly for residual targets. Screen by target;
do not impose one cleaning winner on every model family.

## Dataset versions and availability safety

Modern closing schema 2.0 vs 2.2 was target-dependent under early stopping:

- line error holdout MAE: 2.2 slightly better (14.3590 vs 14.3942);
- total points holdout MAE: 2.0 slightly better (14.4216 vs 14.4575).

These small differences do not justify a universal schema winner. Retaining an
old-schema control for total points is reasonable.

The 2026-09-07 availability-safe 2.2 CSV contains the same 11,543 game keys as
the earlier 2.2 CSV. It removes 14 rotation/availability columns and recalculates
injury/player aggregates: 194 shared numeric columns and 497,141 cells differed
at tolerance `1e-12`. This is a material data-version change even though event
coverage is identical. Pair old/new line-error and total-points runs on the same
protocol; do not infer equivalence from matching rows.

Intermediate T-360 data was not affected because those injury inputs were not
present.

## Target behavior

Line-error residual regression generally showed more useful signal than
absolute total-points regression. This is plausible: total-points loss spends
capacity reproducing the market line, while residual regression focuses on the
component relevant to the bet. It remains an empirical target choice.

In temporal tests, the bookmaker total line was hard to beat on absolute MAE.
Total-points models could show positive directional accuracy while still having
worse absolute MAE than the market. Report both; neither substitutes for the
other.

## Why some historical win rates looked larger

Historical headlines often used an edge threshold of 2 points:

- line error reached about 61.6% on only 87 bets in one run;
- total points reached about 59.6% on 191 bets.

Broader edge-0.1 evaluations produced many more bets and typically lower win
rates. Threshold, volume, period, and holdout length must accompany every rate.
The older holdout covered about 416 games/60 days; the newer one covers about
609 games/90 days, which is more informative but less likely to show extreme
rates.

## Selection tie-break

A local selection rule admits roughly the top 15% of completed trials by MAE,
with an absolute MAE cap of 0.04, then uses pooled directional accuracy as a
tie-break. It prevents win rate from rescuing a candidate far behind on loss.

The evidence was mixed. In the 10-date temporal panel, pure MAE and tie-break
selected the same candidate for both targets. In the 5-date panel, the
tie-broken candidate transferred worse to holdout than the pure-MAE candidate.
Keep both choices in artifacts and evaluate the rule as part of the procedure.

## Pruning is not early stopping

Recent 12-fold campaigns used pruning warmup 13, which means no Optuna trial is
pruned before all folds are evaluated. This was deliberate: temporal fold order
can make median pruning favor candidates that happen to perform well early.

Early stopping still operates within each XGBoost fit. When reproducing a
campaign, describe these two mechanisms separately.

## Spread leakage warning

Archived spread results around 62--67% used 14 rotation-depth columns based on
players who logged minutes in the event being predicted. Those columns carry
post-event participation information and are not a valid benchmark. After the
cleaning gate removed them, a comparable closing spread result was about 52.4%.

For another sport, audit equivalent fields: final starting lineup, players who
actually appeared, innings/minutes played, substitutions, and settled inactive
status. Spectacular improvement from an availability feature is a leakage alarm
before it is a modelling result.

## Current local starting configuration

The latest registered NBA campaign uses:

- availability-safe closing 2.2; unchanged intermediate 2.2 at T-360;
- 12 fixed-game anchored folds;
- 50 validation games and 60-game anchor step;
- 150 Optuna trials, seed 16;
- pooled MAE;
- fold-local early stopping, 1,000 maximum rounds, patience 70;
- pruning effectively disabled;
- approximate top-15% tie band capped at +0.04 MAE;
- 90-day daily walk-forward holdout;
- overtime included, playoffs excluded, regular season and Play-In retained;
- row NaN limit 300, global correlation 0.95, odds correlation 0.99.

Registered training windows differ by target because a fixed-game anchored
study cannot safely tune X when larger windows make early folds infeasible:

| Setting | X events |
|---|---:|
| closing spread | 4,500 |
| closing total points | 4,000 |
| closing line error | 6,250 |
| T-360 spread | 4,000 |
| T-360 total points | 4,000 |
| T-360 line error | 3,000 |

These values are campaign choices informed by prior runs. They should be
screened again after porting to another sport or materially changing features.

## Known limitations to carry into interpretation

- The current pipeline learns correlation pruning before splitting holdout;
  this should move inside the temporal fit for a stricter implementation.
- The same recent holdout has been inspected repeatedly and is now exploratory.
- Most searches use one primary seed; extra seeds are still needed for model
  variance.
- Flat −110 ROI is diagnostic and may not represent executable prices.
- Fixed-model preprocessing screens do not prove the same ranking after a full
  hyperparameter search.

The safe conclusion is usually “promising enough to test forward,” not “best.”
