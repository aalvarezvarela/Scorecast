---
name: sports-model-pipeline
description: Design, reproduce, and audit cleaning, training, temporal cross-validation, hyperparameter search, and forward evaluation for sports prediction models. Use when porting this modelling workflow to another sport or deciding whether an experimental result is trustworthy; not for raw data ingestion or feature-family implementation.
---

# Sports model training and evaluation

Build a reproducible path from an availability-safe event table to a model
that is evaluated as it would operate in time. Treat measured tendencies as
starting hypotheses, not universal laws. The temporal invariants are strict;
the numerical settings are sport-, market-, dataset-, and target-dependent.

For any task that designs or interprets CV/holdout, read
[references/temporal-evaluation.md](references/temporal-evaluation.md). When
choosing initial settings from the NBA work, or explaining why this repository
uses them, also read [references/nba-findings.md](references/nba-findings.md).

## Classify every recommendation

Keep these categories separate in plans and reports:

1. **Validity invariant** — breaking it invalidates the experiment: no future
   information, train strictly before validation, identical evaluation cohorts
   for candidates, and no holdout feedback into selection.
2. **Measured local observation** — evidence from a particular dataset and
   period, such as early stopping outperforming tuned tree counts here.
3. **Starting default** — a reasonable value to test first, not a conclusion.

Never promote category 2 or 3 into category 1. State the evidence and the
conditions under which it was measured.

## Workflow

### 1. Define the prediction event before cleaning

Write down:

- prediction timestamp or horizon;
- target and market line available at that timestamp;
- event grain and unique key;
- competitions and phases included;
- which prices would actually settle a bet;
- retraining cadence and training-history policy.

For every candidate feature, answer:

> At the prediction timestamp, which archived source supplied this value, and
> had that exact value already been published?

“It could have been known” is weaker than “it was archived as-of that time.”
Post-event injury status, final participants, settled lineups, corrected
rosters, backdated transactions, and closing prices are recurring sources of
optimism in sports data.

### 2. Validate the raw dataset

Before modelling:

- pin a checksum and semantic data-version label;
- validate row count, unique event keys, chronological range, and grain;
- reject duplicate columns and duplicated event/snapshot keys;
- inventory columns added, removed, and recalculated between dataset versions;
- compare shared values, not only headers;
- verify target, line, outcome, competition phase, and event-time columns;
- record feature availability by season or era.

A new CSV with the same games can still be a materially different dataset.
Pair old/new experiments on the same folds when a data correction might change
results.

### 3. Apply a semantic missing-data policy

Do not use one blanket imputation rule. Assign policies by meaning:

| Missingness type | Useful starting treatment |
|---|---|
| Required target/anchor line or identity | Drop or fail loudly |
| Book, feed, or optional source unavailable | Preserve NaN; add a missingness flag if absence can be informative |
| A neutral effect with no evidence | Fill with the estimator's defined neutral value, often zero |
| Insufficient rolling history | Use a documented fallback: current history → prior completed season → league/neutral prior |
| Ordinary numeric gap | Training-only median or another fitted imputer, if genuinely needed |

Tree models can use NaN directly. Imputing everything may erase useful source
availability information; dropping every incomplete row may erase older
seasons and season openings.

Treat row- and column-level thresholds as separate levers. Measure which one
actually removes data, by season and feature family. A more permissive row NaN
limit is a useful prior when it preserves valid historical games, but inspect
whether missingness itself reveals season, provider, or competition phase.

Apply deterministic semantic cleanup before row thresholds: required-column
checks, known neutral fills, safe inference pairs, and explicit forbidden
columns. Report counts after every step. A threshold such as “300 NaNs” refers
to the post-policy frame, not the raw CSV.

### 4. Remove leakage and target proxies explicitly

Maintain a denylist of outcomes and columns computed from post-event facts,
drop them at the feature-matrix boundary, then assert they are absent. Prefer a
machine-enforced temporal naming convention such as `_BEFORE` or `_ASOF`.

Availability deserves special scrutiny. “Players who actually participated”
or “players finally marked inactive” may be known only after the event even if
the feature name says “before.” Build training values from the same kind of
as-of source available in production.

Fit learned preprocessing on past data only. In a strict evaluation:

- learn imputers, encoders, scalers, correlation pruning, and feature selection
  on each fold's training rows;
- apply that fitted transformation to validation rows;
- fit the final preprocessing state on development data before touching the
  holdout.

Computing correlation pruning on the entire dataset before the holdout split is
a known convenience but not a fully blind simulation. If inherited code does
this, disclose it and avoid claiming independent confirmation.

### 5. Prune redundancy without discarding the market

Start by removing exact duplicates. For correlation pruning, consider separate
thresholds for market-derived and non-market features: market columns are often
highly correlated by construction, but their small differences may be the
signal of interest.

A reasonable initial experiment is approximately `0.95` globally and `0.99`
for market/odds features. Do not assume it transfers. Screen nearby policies
with fixed model hyperparameters on identical temporal validation games before
spending a full hyperparameter-search budget.

### 6. Choose the target deliberately

For a totals market, useful alternatives include:

- **absolute outcome regression**: predict final total, then compute
  `predicted_total - available_line`;
- **residual regression**: predict `final_total - available_line` directly;
- **classification/probability**: predict the probability of either side.

For a spread market, predict outcome margin relative to the available spread.

Do not compare unlike metrics. Absolute-outcome and residual regressors have
point MAE; classifiers have probability metrics. All can be compared on
direction, bet volume, executable-price ROI, and calibration appropriate to
their outputs. Always include the market line as a baseline: zero is the
equivalent baseline in residual space.

### 7. Tune with a temporally valid objective

Use temporal outer folds and pool per-event losses unless equal fold weight is
an intentional choice. Pooled MAE answers “average error per event”; mean fold
MAE lets a small fold weigh as much as a large one.

For tree count, compare these alternatives rather than assuming one wins:

- a single `n_estimators` sampled by the hyperparameter search and held fixed
  across folds;
- fold-local early stopping with a generous maximum and patience, followed by
  a documented rule for the final tree count.

The NBA evidence makes fold-local early stopping a useful default to try, but
it was not universally better on holdout. Also, stopping on the same outer
validation rows later used to report performance makes that outer score
optimistic. Prefer an inner chronological tail of each training window for
stopping when implementing a new pipeline. If reproducing the current method,
retain it exactly and label the limitation.

Keep the training-window length fixed inside a fixed-game anchored study unless
the splitter guarantees every proposed length uses the same validation folds.
Otherwise different trials are evaluated on different games. Compare training
window sizes in a separate fixed-hyperparameter screen or construct all folds
against the largest required history first.

For reproducible searches:

- use a fixed sampler/model seed for the primary study;
- use a fixed trial budget rather than a wall-clock timeout;
- use enough trials for the dimensionality of the search space;
- store every trial, fold metric, prediction, and chosen tree count;
- test additional seeds after candidate selection to measure fit variance;
- distinguish Optuna trial pruning from model early stopping.

Pruning stops unpromising hyperparameter trials across folds; early stopping
stops boosting rounds within a fit. With few temporal folds, aggressive pruning
can reward candidates that happen to see easy early folds. Disable it for a
clean comparison or use a warmup that has been justified against fold order.

### 8. Select conservatively

Rank primarily by the registered predictive loss. A secondary directional
metric can break near-ties, but it should not rescue a materially worse model.
A useful local rule has been:

- admit roughly the best 15% of completed trials;
- cap the MAE band at a small absolute tolerance;
- among admitted candidates, consider pooled directional accuracy, then RMSE
  and MAE.

This is a measured heuristic, not a universal selector. Record both the pure-
loss winner and the tie-broken winner. The NBA results were mixed: the
directional tie-break sometimes improved holdout transfer and sometimes chose
a worse model.

Do not tune the betting threshold on the final holdout. Report a predeclared
threshold sweep with bet count. High win rates on small, high-edge subsets are
not comparable to broad-coverage rates.

### 9. Evaluate like production

Use three levels:

1. **Temporal CV on development data** to choose hyperparameters.
2. **Daily/event-batch walk-forward holdout** to estimate transfer: at each
   date, train only on completed earlier events, optionally including earlier
   holdout dates, then predict that date without updating within it.
3. **Untouched future forward test** for confirmation after the research
   choices are frozen.

Repeatedly consulting the same holdout turns it into development data. Label
all later analyses of that period exploratory, even if each individual script
technically selects from CV before scoring it.

### 10. Preserve an auditable run

Save at least:

- resolved configuration and dataset checksum;
- cleaning report and feature schema;
- exact fold dates, event keys, train/validation sizes, and exclusions;
- environment/library versions and effective compute device;
- all search trials, their fold metrics, stopping rounds, and selection band;
- out-of-fold and holdout predictions with line and event identifiers;
- market baselines, threshold sweeps, bet counts, and uncertainty summaries;
- a manifest stating whether the holdout was scored;
- whether a production model was deliberately not saved.

Before a costly campaign, run a real preflight that performs cleaning and fold
construction. Verify every requested training window is available in every
fold; `tail(N)` may silently return fewer than N rows.

## Reporting language

Prefer:

- “improved on this cohort under this protocol”;
- “a useful candidate for confirmation”;
- “the interval includes zero”;
- “the result is conditional on a previously inspected holdout.”

Avoid:

- “the optimal setting” from one seed or one period;
- “profitable” from flat diagnostic odds rather than executable prices;
- “independent test” after the same holdout guided multiple decisions;
- “more folds are better” or “fewer folds reduce noise” without a controlled
  comparison on common events.

The goal is not to make the pipeline maximally cautious. It is to make each
claim proportional to the evidence while keeping the experiment reproducible.
