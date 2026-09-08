# Temporal evaluation for sports models

Read this reference whenever designing, modifying, or interpreting temporal
cross-validation or holdout evaluation.

## Why random CV is usually the wrong question

A sports model is deployed forward in time. Randomly distributing events among
folds lets training contain seasons, rosters, market regimes, and possibly
source availability that occur after validation. Even if no target column
leaks directly, it estimates interpolation across time rather than future
performance.

Sort by the event timestamp and preserve same-time batches. All training rows
must be strictly earlier than the validation batch. Never split simultaneous
events from one slate or round between training and validation if their shared
information can create dependence.

## Fixed-game anchored CV

This repository's portable fixed-game design is:

1. Reserve the final holdout first.
2. Work only on development data.
3. Locate the latest eligible validation anchor.
4. Form a validation block containing at least `test_games`, extending to the
   end of the event date rather than splitting a date.
5. Step backward or forward by `step_games_between_tests` to locate the next
   anchor.
6. At each anchor, train on the last X eligible events strictly before the
   validation start.
7. Keep only folds for which the registered X is fully available.

An NBA starting point supported by the local evidence is:

```yaml
strategy: test_anchored
test_games: 50
step_games_between_tests: 60
max_folds: 12
fold_selection: latest
```

This produces roughly 50--60 events per fold because dates are indivisible.
The 60-game step leaves some separation between validation blocks and spreads
evaluation across time. It does not create an embargo between training and
validation: the training window still ends immediately before each validation
start. Add a real time embargo when source publication delays require one.

### What is portable

The algorithm and invariants transfer; the number 50 does not necessarily.
Fifty events span only a few days in a high-volume league and many weeks in a
low-volume league. Choose event count so a fold:

- contains enough events for stable loss and directional metrics;
- is short enough not to average over a large regime change;
- resembles how long a deployed model stays unchanged;
- covers complete dates/rounds;
- yields enough separated folds across relevant seasons.

If uncertain, compare a small set of fixed sizes using the same frozen model
candidates and the same total OOF cohort. Do not launch independent searches
and then attribute every difference to fold size.

## Rolling-origin CV by dates

An alternative groups a fixed number of observed event dates per origin. It
better represents a retraining cadence, but fold event counts vary with the
schedule. “Ten days” should mean ten distinct event dates present after
cleaning, not ten civil days, unless explicitly implemented otherwise.

Important details:

- a minimum event count may extend a nominal date window;
- a short remainder may be merged into the preceding fold;
- excluded competition phases or months change which dates exist;
- no OOF event should be predicted twice;
- a batched `predict` call does not imply future features were known at the
  fold start, provided every row's features are independently as-of correct;
- the model is simply older for later dates inside the fold.

Fixed-game folds are easier to compare across schedule density. Date-based
folds more directly model retraining age. They answer related but not identical
questions.

## Holdout: daily or event-batch walk-forward

A production-like holdout repeats this for each event date D:

```text
train = eligible events with timestamp < D
train = last X events (or expanding history)
fit preprocessing on train
fit model with frozen hyperparameters
predict all events on D
after D is complete, its outcomes may enter future training
```

Do not update between games on the same date unless production truly receives
results and retrains in that interval. Features and market lines may differ by
game; the model state remains fixed for that batch.

A calendar-day holdout has variable event count. Record start/end dates, number
of event dates, event count, and retrain count. A 90-day holdout is often more
informative than a 60-day one, but only a new future period is independent after
the earlier period has guided model decisions.

## Common-cohort comparison

When comparing candidates, verify equality of:

- validation and holdout event keys and order;
- target/outcome and target line;
- competition and overtime/extra-time policy;
- betting threshold and its units;
- training eligibility and requested X;
- prediction horizon and available feature schema.

If cleaning policies retain different events, evaluate their predictions on
the intersection of common validation keys while allowing each policy's
training history to reflect its actual retained rows. Report both the common
evaluation count and each policy's cleaned/training counts.

Never let `tail(X)` silently become “up to X.” Assert that every fold contains
exactly X training rows when the experiment claims a fixed window.

## Objective aggregation

For folds with unequal size:

```text
pooled MAE = sum over all OOF events of abs(error) / total OOF events
mean-fold MAE = mean of each fold's MAE
```

Pooled MAE gives every event equal weight and is the usual choice. Mean-fold
MAE intentionally gives every temporal origin equal weight. State which
estimand matters rather than treating them as interchangeable.

Keep per-fold metrics even when selecting on pooled loss. Large fold variance,
one dominant period, or a ranking that reverses when one month is removed is
evidence of instability.

## Metrics and baselines

### Predictive metrics

- MAE in target units for regressors.
- RMSE as a tail-error diagnostic, not a substitute chosen after results.
- Log loss, Brier score, and calibration for probabilistic classifiers.
- Always compute the bookmaker/market baseline on exactly the same events.

For absolute total prediction:

```text
model error = actual_total - predicted_total
market error = actual_total - available_line
predicted edge = predicted_total - available_line
```

For residual prediction:

```text
target = actual_total - available_line
market baseline prediction = 0
predicted edge = model prediction
```

### Betting metrics

Report together:

- threshold and its units;
- candidates, bets, pushes, wins, and losses;
- win rate and interval;
- bet rate;
- ROI and profit units;
- price source and whether prices were executable.

At decimal odds 1.90909 (American −110), break-even is 52.38%. Flat-odds ROI is
a diagnostic, not proof of executable profit. A 60% win rate over 80 high-edge
bets is not directly better than 54% over 600 broad-coverage bets.

## Uncertainty and temporal dependence

Ordinary iid intervals understate dependence among games in the same week,
slate, team cycle, or market regime. Useful summaries include:

- Wilson interval for a single win rate;
- paired differences on common event predictions;
- block bootstrap over calendar weeks or sport-appropriate temporal blocks;
- results by predeclared calendar blocks, season phase, and prediction horizon;
- leave-one-period-out ranking stability;
- repeated model seeds, reported separately from temporal uncertainty.

A paired block bootstrap compares per-event loss differences, resamples whole
time blocks, then pools their events. It is descriptive when candidates were
already chosen using related data and does not correct for repeated searches.

## Model selection and holdout discipline

The outer CV selects hyperparameters. The holdout estimates the selected
procedure. Once the holdout affects fold geometry, preprocessing, threshold,
target choice, or search-space decisions, it has become part of research and
must not be described as an untouched test.

Freeze before final confirmation:

- dataset and availability rules;
- feature/cleaning policy;
- target and market line;
- CV geometry and objective;
- hyperparameter space and trial budget;
- selection and tie-break rule;
- threshold reporting plan;
- training window and retraining cadence.

Then evaluate once on genuinely later data. This forward test is the most
important measurement for a weak-signal sports market.

## Frequent temporal failure modes

| Failure | Why it can look valid |
|---|---|
| Random CV | Aggregate metrics look stable because regimes are mixed across folds |
| Correlation/feature selection before split | It uses no label explicitly, yet future availability shapes the schema |
| Fold-local stopping on the scored outer fold | It is called “early stopping,” but the evaluation rows influenced tree count |
| Different X silently shortens old folds | The model still fits and produces metrics |
| Same-day events split across train/validation | Timestamps differ while shared slate information leaks |
| Post-event availability used as “pregame” | The semantic concept existed before play, but the recorded truth did not |
| Threshold selected on holdout | The model is unchanged, disguising that the decision rule was fitted |
| Reusing a familiar holdout | Every individual run is technically clean, but the research process is not |

Audit behavior and event keys, not just configuration names.
