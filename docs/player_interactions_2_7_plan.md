# Schema 2_7: player interactions plan

Oct 6, 2026 · Adrián Álvarez Varela · **Status: plan, not implemented** (corrected 2026-10-07)

## Goal and principles

The goal is to give XGBoost features that capture who plays tonight, for how many minutes, **who guards whom**, and how those players interact. The project is as much about learning graph and representation methods as about beating the baseline.

**Scope.** 2_7 is a `SchemaLayer` on top of 2_6 (`src/nba_ou/create_training_data/schema_layers/`). Like every layer it **only adds columns**: it cannot change any 2_5 or 2_6 column, including the `LU_*` family. Improving an existing `LU_*` column in place would be a new base (`3_0`), so everything this plan produces lands as new `PI_*` columns (see the catalog). The stage 1 models are feature-pipeline components, like `minutes_model.py` today: they are trained and checkpointed to build the layer, not promoted through the XGBoost model registry. Both datasets get the layer: closing, and intermediate at each snapshot cutoff.

**Feature base: 2_5, not 2_6.** Two things are kept apart here:

- **The file.** Layers form a straight chain (`schema_layers/registry.py` requires 2_7's parent to be 2_6), so a 2_7 file always contains the 25 columns 2_6 added (8 `STARTER_*`, 17 `LU_*`).
- **The model.** The 2_7 models start from the **2_5 features** plus the new `PI_*` columns. The 2_6 columns are dropped in the training config with `cleaning.exclude_cols_containing: ["LU_", "STARTER_"]`; on the 2_6 file these two patterns match exactly those 25 columns and nothing from 2_5. The 2_6 columns only come back in an explicit variant (phase 8).

2_7 reuses 2_6's **code** (availability, scenarios, ratings) to build its graph, not 2_6's **columns**.

**Two-stage architecture**

- **Stage 1, player and game representation.** Learns from stints and from attacker-defender matchups how players, their combinations and their direct opponents produce efficiency, pace and style. It uses **no betting information** in v1: no total, spread or moneyline, neither as input nor as target. Market inputs (for example the spread for blowout risk in phase 4A) are tested only as explicit variants, so a stage 1 output never partly restates the market that XGBoost already sees.
- **Stage 2, XGBoost.** Takes the stage 1 outputs for each game, plus the existing 2_5 features, and predicts the repo's targets: `TOTAL_POINTS`, `LINE_ERROR` and `SPREAD_ERROR`.

**Main path and controls.** The central artifact is a **player graph** (phase 2): players as nodes, teammate, opponent and **guards** edges, built from stints and matchup data. Every later phase either reads it or improves its weights. The main model path is a learned game embedding over that graph: a factorization machine (FM) first, then a GNN. The 2_6 RAPM projection is the control. Archetypes are an optional diagnostic.

**Principles**

1. **Everything as of T.** No data, profile or model uses information after the prediction time T. Injury snapshot and line come from the same T.
2. **Temporal out-of-fold everywhere.** Every model that feeds XGBoost is trained only on data before the games it encodes, including XGBoost's own training rows. The same holds for models that feed other stage 1 models: the profile prior, the minutes model and the matchup model are refit as of each date or checkpoint, never once on all seasons.
3. **Validate each piece on its own.** First at stint, minutes or matchup level, then in the XGBoost walk-forward.
4. **Each phase is compared with the previous one.** Improvement is not a gate: noise is high and the project is also for learning. Comparisons still need enough seeds to be readable (phase 1).
5. **Prefer stable scalars; test raw embeddings as an experiment.** Head predictions and dot products have a fixed meaning. Raw embedding coordinates are tested with warm starts and low dimensionality.
6. **One global model to learn, the game graph to encode.** Pretraining uses all stints and matchups; each game is encoded from its as-of-T graph.
7. **Expected edges, never observed ones, as encoder input.** Tonight's minutes and tonight's guarding assignments are both unknown at T. Every graph the encoder reads, in training and in prediction, carries **expected** guarding shares as of the game's date (phase 4), exactly as nodes use projected minutes. The guarding shares observed in the game being encoded are never an input: they are labels (phase 6) and diagnostics (the phase 2 oracle) only. Shares observed in earlier games are allowed: they are the history the expected share is built from.
8. **Build the graph first, improve its weights later.** Projected minutes and guarding shares are node and edge weights of the graph. Simple v0 weights make the graph usable from phase 2; the phase 4 models are upgrades, not blockers.

## What 2_7 adds over 2_6

| | 2_6 | 2_7 |
| --- | --- | --- |
| Player representation | Three fitted scalars (offense, defense, pace) plus hand-built style traits | Learned embeddings (FM vectors, GNN node embeddings) with a profile prior |
| Data structure | Tables and per-module projections | One player graph shared by features and models |
| Teammates | Additive: minutes-weighted sum of ratings. Pair synergy measured but not emitted | Chemistry terms (FM) and teammate edges (GNN) |
| Opponents | Team level only: my offense minus your defense, summed over players | **Player against player**: who guards whom, from tracking data, as weighted directed edges |
| Matchup data | Imported, unused | Expected guarding shares, matchup-adjusted defensive ratings, edge-level training targets |
| Game output | 17 `LU_*` scalars | New scalars, encoder head outputs, absence counterfactuals, optional game embedding z |

## Starting point: what 2_6 already has

Much of the groundwork was built and measured for 2_6 ([lineup_projection_plan.md](lineup_projection_plan.md), code in `src/nba_ou/data_processing/lineups/`). 2_7 builds on it instead of rebuilding it.

| This plan | Already built | Module | Status |
| --- | --- | --- | --- |
| Phase 0: as-of-T availability | Roster from strictly earlier games plus the last injury report before T; closing and per intermediate snapshot | `availability.py`, `intermediate_features.py` | In 2_6 |
| Phase 0: doubtful scenarios | Every play/sit combination weighted by `chance_out`; spread across scenarios emitted as an SD | `game_projection.py`, `injury_status/news.py` | In 2_6 |
| Phase 1: harness | Campaigns under `experiments/`, runs under `artifacts/experiments/` | `training_pipeline` | Exists; the 2_5 vs 2_6 campaign is still pending |
| Phase 2: graph inputs | Validated stints 2016-17 → 2025-26; who guarded whom 2017-18 → 2025-26 ([player_lineup_and_matchup_data.md](player_lineup_and_matchup_data.md)) | `stint_store.py`, `fetch_data/nba_lineups/matchups.py` | Stints used by 2_6; **matchups imported, nothing reads them** |
| Phase 2: v0 node weights | Recent-minutes average redistributed to 240, pair shared-minutes weighting | `availability.py`, `game_projection.py` | In 2_6 |
| Phase 3: RAPM | Walk-forward ridge for offense, defense, pace; daily rating dates; λ_offdef = 1000, λ_pace = 30000, half-life 180 days | `player_ratings.py`, `rating_cache.py` | In 2_6. No profile prior |
| Phase 4A: minutes model | Gradient-boosted `E[min | play]` | `minutes_model.py` | Built, not connected |
| Phase 5: game features | 17 `LU_*` columns, including the absence counterfactual split into offense, defense and pace | `features.py`, `style_matchup.py` | In 2_6 |
| Pair and five synergy | Shrunk residual per pair and per exact five | `synergy.py` | Built and measured, not emitted |

**What was measured, and what it means for expectations.** Treat these as the priors this plan starts from, not as reasons to stop.

- **The signal in 2_6 is the absence counterfactual**, not the projected level. `LU_ABSENCE_IMPACT_*` regresses on `LINE_ERROR` with a positive slope in every season tested, and it lives in the **defense and pace** channels (offense is already priced): when a good defender is out, games go over more than the line expects.
- **Team-level style interactions are null.** Synergy was null. A bilinear offense-trait × defense-trait model on 320k stints gained ≤ +0.003% out of sample. kNN over past lineup matchups found a small non-additive part in 3PA/FGA only. All of these treat the opposing five as a block.
- **Player-against-player was never tested.** The lineup plan (§8.6) left it open for lack of data and named its most likely payoff: a sharper **defensive** rating, the channel the market under-prices. Its caution also applies: for a total, a suppressed star's shots partly move to his teammates, so a matchup effect on one player is diluted at team level.
- **Minutes accuracy barely moves team totals.** The minutes model beat the baseline by 23.8% (MAE 5.23 vs 6.86) and moved game-projection MAE by 0.004. Each team's minutes sum to 240, so redistributing them barely changes the sum. Minutes matter for the counterfactual, much less for the level.
- **The fixed-parameter 2_5 vs 2_6 trial was null**, and the difference between the two arms was no larger than seed noise (predictions correlated 0.54-0.71 between arms).

So the most promising things a learned, matchup-aware encoder can add are **a better defensive counterfactual** (whom did the absent defender guard, and who takes those assignments tonight) and a better replacement estimate for low-sample players. Phase 8 tests that explicitly.

## Graphs in this plan

The plan has **two graph families**, both built in phase 2:

- **Training graphs** describe what happened: the players actually on the floor in each stint, with the real outcome as label. They need no projection models.
- **Prediction graphs** (game graphs) describe what is expected at T: the available players of a game, with projected minutes and expected guarding shares as weights. They improve as phase 4 improves the weights.

The other graphs in the plan are views of the same data:

| Graph | Nodes | Edges | Built | Used in |
| --- | --- | --- | --- | --- |
| Stint graph (training) | The ten players of a stint | Three types: teammate, opponent, **guards** (weighted by guarding share) | 2 | 3, 6 |
| Stint hypergraph | Players | One hyperedge per stint joining ten players, signed by side: the incidence matrix of the stint graphs | 2 | 3 |
| **Matchup graph** (training) | Attackers and defenders of one game (bipartite) | Directed defender → attacker; observed guarding share and the attacker's production as edge labels | 2 | 4B, 6 |
| Game graph (prediction, as of T) | Available players of both teams, weighted by projected minutes | Teammate and opponent edges weighted by expected shared minutes; guard edges weighted by expected guarding share | 2 (v0), upgraded in 4 | 5, 7 |
| Substitution graph | Players of one team | Directed X → Y, weight = extra minutes Y gets when X is out | 4A | 4A |
| Teammate co-play graph | Players of one team | Undirected, weight = historical shared minutes | 2 | 4A, 5, 6 |

The matchup graph is the only one built from direct player-against-player observation; every other opponent relationship in the plan is inferred from who shared the floor.

## Phase map

| Phase | What | Needs | Output |
| --- | --- | --- | --- |
| 0 | Point-in-time data: remaining gaps, matchup store completion | 2_6 availability, matchup store | Play probabilities by status and time; complete, daily-updated matchups |
| 1 | Walk-forward harness and baseline | `experiments/` | Reference metrics for 2_5 |
| 2 | **Player graph dataset** | 0 | Training graphs, matchup graphs, game-graph builder with v0 weights, oracle ceiling |
| 3 | RAPM with a profile prior | 2 | Prior-adjusted ratings |
| 4A | Minutes model | 2 | Better node weights (projected minutes, shared minutes) |
| 4B | Matchup model | 2 | Better guard-edge weights; matchup-adjusted defensive ratings |
| 5 | Scalar game features (milestone 1) | 2, 3, 4 | New scalar `PI_*` columns as readouts of the game graph |
| 6 | Multitask pretraining: FM, then GNN | 2 (v0 weights), rerun after 4B | Encoder trained on stint and matchup graphs |
| 7 | Game embeddings, out-of-fold | 4, 6 | Head outputs and z per game and T |
| 8 | Embeddings into XGBoost (ablations) | 5, 7 | Which outputs go into 2_7 |
| 9 | Production | All | To be defined |

```
0 ─ 1 ─ 2 ─┬─ 3 ──────────┐
           ├─ 4A ─────────┼─ 5 ─────────┐
           ├─ 4B ─────────┤             ├─ 8 ─ 9
           └─ 6 (v0) ── 6 (v1 weights) ─ 7 ─┘
```

Phases 0 and 1 set up data and measurement. Phase 2 builds the graph every later phase reads, and ends with the oracle ceiling, which sets how much effort phases 4A and 4B deserve. After it, three tracks run in parallel: phase 3 improves the ratings, phase 4 improves the weights, and phase 6 starts learning on the training graphs with v0 weights. Phase 5 is the first end-to-end system. Phase 7 needs phase 4, because prediction graphs are only as good as their projected weights. Phase 6 is rerun once phase 4B's guarding shares replace v0.

## Phase 0. Point-in-time data (as of T)

This phase answers "what was known at time T?". Most of it exists in 2_6 (`availability.py`); what remains is play probabilities that depend on the time of the report, and making the matchup store complete and current.

**Already in place (reuse it)**

- **Team roster.** Players with appearances on strictly earlier dates, a player belonging to only one team at a time, dropped while absent for the whole recent window, plus the injury report at T. The roster is **not** taken from the box score of the game being predicted. Membership read from that game is a known leak class in this repo: the `N_ACTIVE_PLAYERS` family was exactly that, and removing it took a spread model from 66.7% to 52.4% (`nba_ou.config.leakage`).
- **Zero-minute box-score rows.** Box scores do contain DND/NWT/DNP rows (700-1,400 a season). `availability.py` treats a coach's decision as a zero-minute appearance, any other reason as an absence, and G-League assignments as neither.
- **Availability from the injury report at T.** The last report filed before tip (closing) or before each snapshot cutoff (intermediate). Both teams must have filed, or the game gets NaN.
- **Prediction times T.** Closing, plus the intermediate dataset's existing snapshots (six per game), each with its own injury cutoff and the line at that snapshot.
- **Doubtful scenarios.** `game_projection.py` enumerates every play/sit combination of uncertain players and weights each by `chance_out`.

**Deliverables (new)**

- **Play probabilities by status and time to tip.** `chance_out` is status-based. A doubt at noon is not a doubt an hour before tip; estimate `P(play | status, hours to tip, status history)` from report histories and calibrate it.
- **Complete matchup store.** 2019-20 is at 91.9%, mostly the 88 bubble seeding games: `import_matchups.py --fill-missing`.
- **Daily matchup update.** None exists today; without it no matchup feature can be served. `BoxScoreMatchupsV3` answers in ≤1.6 s, so one call per finished game fits next to the daily stint job.
- **An `as_of(T)` access layer** for stints, matchups, box scores and injuries, so later phases cannot read later data by accident. The 2_6 modules each implement the cutoff themselves; centralize it before the graph builder becomes its main consumer.

**Doubtful players: scenarios**

- As in 2_6 (`enumerate_scenarios`): players with 0.1 < `p_out` < 0.9 are uncertain; up to 3 per team are enumerated (8 scenarios per team, 64 per game). The rest are folded into every scenario: they sit if `p_out` ≥ 0.5, play otherwise.
- **Inside a scenario nobody is uncertain.** Each player either plays or sits. A playing player gets `E[min | play]`, redistributed with his teammates to 240. The probability lives only in the scenario weight; using `P(play) · E[min | play]` inside a scenario counts it twice.
- Each scenario is its own game graph (phase 2): nodes, minutes, guarding shares and every feature or embedding are recomputed.
- Pass XGBoost the probability-weighted mean plus a spread measure of the key outputs. 2_6 uses the **SD** across scenarios. Min and max are dominated by unlikely corner scenarios (for example everyone sitting), so prefer the SD, or probability-weighted quantiles, for consistency with 2_6.

**Decisions**

- Raise the limit of 3 enumerated players per team, or sample above it.
- Whether to extend stints back to 2014-15 and 2015-16 via the PlayByPlayV2 rebuild. Not required; stints start in 2016-17 and matchups in 2017-18.

**Things to check**

- Calibration of play probabilities by status and time to tip.
- Games stints do not cover: Play-In (`005`) and the NBA Cup final (`006`). Their features come from earlier games, but they add no stints to training. Check whether the matchup store covers them.
- Matchup revisions: the NBA reprocesses tracking. Compare a morning-after fetch with a later refetch for a sample of games.

**Done when:** for any game and T (closing or intermediate), we get each team's roster, the doubtful scenarios with time-aware weights, the line at T, and the stint and matchup history up to T, through `as_of(T)`.

## Phase 1. Walk-forward harness and baseline

Before building anything, fix how things are measured. The harness exists (`training_pipeline`, campaigns under `experiments/`); this phase fixes the protocol 2_7 will use.

**Deliverables**

- Walk-forward from 2019-20 with the same splits for every experiment. 2019-20 is also where `LU_*` coverage starts (it needs injury reports for both teams).
- Reference: the promoted 2_5 specs, trained on the 2_5 features. Every 2_7 arm is 2_5 plus new columns, so the comparison is always against 2_5. The pending 2_5 vs 2_6 campaign (lineup plan §10) is useful context, not a prerequisite.
- Every 2_7 training config sets `cleaning.exclude_cols_containing: ["LU_", "STARTER_"]` unless the arm is meant to include 2_6 columns (phase 8, variant L).
- Metrics per target: MAE for `TOTAL_POINTS` and spread; directional accuracy at the repo's min-edge thresholds and simulated return against the line at T for `LINE_ERROR` (`scorers.py`).
- The experiment log is the campaign config plus the run artifacts: which features, which stage 1 checkpoint, which graph weight versions, which T.
- Stage 1 checkpoints stored by training date, so any past feature value can be reproduced.

**Temporal out-of-fold algorithm (shared by every stage 1 model)**

Checkpoints are written C_k to keep them apart from the prediction time T.

1. Pick checkpoint dates C_1, C_2, … (monthly to start).
2. At each C_k, train the stage 1 model on training graphs with game date strictly before C_k, warm-starting from the C_(k−1) model.
3. Freeze it and encode the game graph of every game dated in [C_k, C_(k+1)).
4. XGBoost only ever sees these out-of-fold outputs, for training and for evaluation.

Note on cadence: 2_6 RAPM is refit for every game date, so a monthly checkpoint is up to a month older than the control. Rosters and availability are still current, because the game graph is built at T; only the weights are older. When comparing FM or GNN with RAPM, run RAPM at the same cadence as well, or the comparison mixes model and freshness.

**Decisions**

- Checkpoint frequency.
- Main reference metric. Suggestion: `LINE_ERROR` accuracy and return, since that is the betting decision; `TOTAL_POINTS` MAE as secondary.
- Fixed analysis slices: games with key absences (for example |`LU_ABSENCE_IMPACT_PTS_BEFORE`| > 3), games with a key **defender** absent, early season, after the trade deadline.

**Approach:** XGBoost comparisons use **several seeds per arm** (a seed ensemble), or tuning per arm. The 2_6 trial showed why: with parameters fixed and one seed, adding columns changes `colsample` draws enough that 25-32% of predicted sides flip between arms, so a single-seed difference is noise. Stage 1 models can use one fixed seed. With this much noise, not beating the baseline at one step does not mean the step is not worth pursuing.

**Done when:** a reproducible baseline with metrics per season and per slice, and a measured seed-noise band.

## Phase 2. Player graph dataset

Builds the graph every later phase reads: one schema for nodes, edges and weights, two graph families (training and prediction), and a builder that produces both through `as_of(T)`. The weights start simple (v0) and are upgraded by phase 4 without changing the schema.

**Nodes**

- A node is **a player on a date**: player ID (for the embedding lookup in phase 6), team, side (home or away), position.
- Node features: the as-of-date statistical profile (lagged box-score rates, minutes history, games played, sample size). All computed from games strictly before the node's date.
- Node weight: on a training graph every player has the stint's duration; on a game graph, projected minutes.

**Edges**

| Type | Direction | Between | Weight | Attributes |
| --- | --- | --- | --- | --- |
| Teammate | Undirected | Same side | Shared minutes (observed on training graphs, expected on game graphs) | Historical shared minutes (familiarity) |
| Opponent | Undirected | Opposite sides | Shared minutes | — |
| Guards | Directed, defender → attacker | Opposite sides | Guarding share m_ij (sums to 1 per attacker over the defenders on the floor) | Pair history: games, matchup seconds |

**Guarding shares.** For a pair and a game, the guarding rate is matchup seconds divided by the seconds both players were on the floor together (`pct_total_time_both_on`, cross-checked against the stints). Dividing by co-floor time separates "X guards Y" from "X and Y played a lot". For each attacker, rates over the defenders on the floor are normalized:

```math
m_{ij} = \frac{r_{ij}}{\sum_{l \in D} r_{il}}, \qquad j \in D
```

**Training graphs (what happened)**

- **Stint graphs.** One per stint: the ten players on the floor, with the stint-level labels of phase 6 per side (points per possession, pace, turnover, 3PA, FTA and offensive rebound rates). ~389k stints, 2016-17 → 2025-26.
- **Matchup graphs.** One per game: bipartite, defender → attacker, with the **observed** guarding share and the attacker's production against that defender (FGA, 3PA, shot efficiency, turnovers, shooting fouls) as edge labels. ~2.1M edges, 2017-18 → 2025-26.
- **Guard weights on stint graphs use expected shares, not observed ones** (principle 7). Assignments react to the game (a hot scorer draws the best defender), so a game's observed shares carry outcome information. Expected shares as of the game's date, built only from earlier games, are what the game graph will have at T, so the model trains on the same kind of weight it is served. The same-game observed shares are stored in a separate label table that the encoder's loader cannot read as input; they serve as edge labels and for the oracle below.

**Prediction graphs (what is expected at T)**

- One per game, T and doubtful scenario: nodes are the players who play in that scenario, each with `E[min | play]` redistributed to 240 per team (phase 0: the scenario's probability is its weight, not part of the minutes); edges carry expected shared minutes and expected guarding shares among players expected on the floor together.
- The full-health version of the same game graph (no absences) is built alongside: it is the counterfactual for every absence feature.

**Weight providers, versioned.** The builder takes its weights from pluggable providers, and every graph records which versions built it:

| Weight | v0 (phase 2) | v1 |
| --- | --- | --- |
| Projected minutes (per scenario, given play) | 2_6 rule: recent average redistributed to 240 (`availability.py`) | Phase 4A: `E[min | play]` |
| Expected shared minutes | Pair history rescaled to projected minutes; with no history, independence: (min_i / 48) × (min_k / 48) × 48 | Phase 4A |
| Expected guarding share | Shrunk pair history from earlier games; with no history, same position guards same position | Phase 4B |

No provider reads betting data in v1 (see stage 1 in the goal section).

**Storage.** Store tables, not graph objects: a node table, a game-level pair table (shared minutes, guarding rates, matchup labels) and the stint table. Complete teammate and opponent edges are generated in the data loader. Parquet stays the source of truth (testable with pandas, consistent with the rest of the repo); conversion to PyTorch Geometric or plain tensors happens at load time.

**Checks (the graph's own tests)**

- Five players per side on every stint graph; guarding shares sum to 1 per attacker; projected minutes sum to 240 per team; expected shared minutes never exceed either player's minutes.
- Temporal tests in the style of lineup plan §8.3: nothing on a graph dated D reads data from D or later.
- Input contract: the encoder's input schema contains no same-game observed share and no betting column; a test fails if either appears.
- Within every scenario, each team's minutes sum to 240 and no node's minutes are scaled by a probability.
- **Reproduce 2_6 as a readout.** The 2_6 projection is a linear readout of the v0 game graph: minutes-weighted RAPM ratings over the nodes. Computing it from the graph must give `LU_PROJ_TOTAL_BEFORE` and `LU_ABSENCE_IMPACT_PTS_BEFORE`. This proves the builder and the 2_6 pipeline agree before anything new is learned.
- **Smoke-test GNN.** A tiny GNN on a few months of v0 training graphs, end to end, to test loaders, batching and labels. No claim about signal.

**Oracle ceiling (run before phase 4)**

The most informative experiment before building the minutes and matchup models: how much would perfect weights be worth? Build **oracle game graphs** with the minutes each player actually played and the guarding shares actually observed in that game, and compare the readouts with the v0 graphs.

| Graph | Minutes | Guarding shares |
| --- | --- | --- |
| v0 | Projected (2_6 rule) | v0 expected |
| Oracle minutes | Actual | v0 expected |
| Oracle matchups | Projected | Actual |
| Oracle both | Actual | Actual |

- Readouts: the 2_6 projection (minutes-weighted RAPM ratings, as in the reproduction check) and the matchup-weighted defense of phase 5, using RAPM `def_j` weighted by guarding share (the matchup-adjusted rating comes later, in 4B).
- Measures: total MAE and the slope on `LINE_ERROR`, overall and in the absence slices.
- Reading it: the gap between v0 and an oracle is the most that 4A (minutes) or 4B (matchups) could buy, and the split between the two oracles says which deserves the effort. A small gap is a strong signal to keep that model simple. A large gap is only an upper bound: actual minutes carry outcome information (garbage time, foul trouble, overtime) and observed shares react to the game, so part of any gap can never be projected.
- Oracle graphs are diagnostics only. They never produce a feature and never train the encoder (principle 7).

**Decisions**

- Where the builder lives (suggestion: a new `src/nba_ou/data_processing/player_graph/` next to `lineups/`).
- Graph library: PyTorch Geometric (`HeteroData`) or plain tensors.
- Which pairs get guard edges on game graphs: all pairs expected on the floor together, or only pairs above a minimum expected share.

**Done when:** for any stint, a training graph with labels; for any game, its matchup graph; for any game and T, a game graph per doubtful scenario plus its full-health counterfactual, with v0 weights; the 2_6 readout reproduced; temporal and input-contract tests passing; the oracle ceiling measured.

## Phase 3. Additive player model (RAPM with a profile prior)

Each player gets three numbers learned from stints: how much he adds on offense, how much he takes away on defense, and how much he speeds up the game. 2_6 already has these (`player_ratings.py`). What is new here is the **profile prior**. It matters most for the replacement players behind the absence counterfactual, whose ratings are the noisiest.

**The graph in this phase.** The stint hypergraph, which is the incidence matrix of the phase 2 stint graphs: one row per stint and side, one column per player. In the efficiency model an offensive player gets +1 and a defender −1; in the pace model all ten get +1. RAPM is the simplest model on the training graphs: no edges, only nodes summed per side.

**Models** (one row per stint and side; O = offensive five, D = defensive five)

```math
\text{pts}/100_{O} = \mu + h \cdot \text{HCA} + \sum_{i \in O} \text{off}_i - \sum_{j \in D} \text{def}_j, \qquad h = \begin{cases} +1 & O \text{ is home} \\ -1 & O \text{ is away} \end{cases}
```

```math
\text{poss}/48 = \pi + \sum_{k \in O \cup D} \text{pace}_k
```

HCA is split between the two sides (+h), as in 2_6, so it moves the margin and not the total. Fit with weighted ridge (possessions × recency) and a free intercept.

**Profile prior (offset trick)**

1. Fit `f`: predicts a player's ratings from his node features (as-of-date profile and position), using players with large samples.
2. Compute each stint's expected outcome from those prior ratings.
3. Ridge on the residual: gives each player's deviation from his prior.
4. Final rating = prior + deviation.

Rookies with no games: prior = average rookie at his position, computed from our own data. The profile fills in over the first games.

**The prior must be temporal too.** It feeds every rating, so a leak here reaches everything downstream, and quietly:

- `f` is refit at every rating date (or checkpoint) on (profile, rating) pairs available **before** that date. Fitting it once on all seasons would teach the prior what each player turned out to be.
- Its training targets are the **as-of ratings** from the walk-forward rating cache on each date, not end-of-season or final ratings.
- Profiles are lagged: rates from games strictly before the date, as for the node features.
- The rookie prior is computed only from rookies seen before the date.
- Hyperparameters (λ, half-life, prior strength) tuned on seasons that are later used for evaluation are a milder form of the same leak. Record which seasons tuned them, and keep at least one evaluation season outside.

**Decisions**

- Keep the tuned 2_6 values as the starting point (half-life 180 days, λ_offdef = 1000, λ_pace = 30000). Both λ hit the edge of the grid they were tuned on (`scripts/lineups/README.md`), so widen the grid before retuning.
- Data window (2-3 seasons with decay is enough).

**Things to try**

- Ridge toward zero (2_6) vs ridge toward the profile prior.
- Gain concentrated in low-minute players and replacement players.
- Whether pace ratings are stable across seasons.
- Effect of estimated possessions: team rebounds without a player are not classified, and the league mean residual of +2.75/100 is a possession-estimator gap, not skill.

**Own validation:** error on the following weeks' stints vs the 2_6 ratings (not only vs team averages, which 2_6 already beats).

**Done when:** it predicts future stints at least as well as the 2_6 ratings, and better for low-sample players.

## Phase 4. Weight models: minutes and matchups

Phase 2's game graphs use v0 weights. This phase replaces them with models: 4A for the nodes (minutes), 4B for the guard edges (guarding shares). The two run in parallel with each other, with phase 3 and with phase 6. Each ships as a new weight-provider version; the graph schema does not change.

### Phase 4A. Minutes model (node weights)

Projects how many minutes each available player will play, and how many each pair of teammates and opponents will share.

**Why it matters, and where.** Every game-level output weights players by projected minutes, from the additive team rating to the GNN pooling:

```math
\text{off}_{\text{team}} = \sum_i \frac{\text{min}_i}{48} \cdot \text{off}_i
```

But each team's minutes sum to 240, so the level of this sum barely depends on how minutes are split between similar players; the 2_6 measurement was 0.004 of game MAE. Minutes matter where they decide **who replaces an absent player**, which is exactly the counterfactual. Measure phase 4A there.

**Model.** `minutes_model.py` (built, not connected): `E[min] = P(play) · E[min | play]`, with a gradient-boosted regressor refit monthly. Its output today, `walk_forward_minutes`, is the **unconditional** expectation `(1 − p_out) · E[min | play]`. That is right for a single collapsed projection and wrong inside a scenario graph, where availability is already decided and its probability is the scenario weight (phase 0). Provider v1 therefore exposes the `E[min | play]` component, and the builder redistributes it over the scenario's playing players to 240. Connect it, and add what it lacks:

1. Absence redistribution via the substitution graph: edge X → Y weighted by the extra minutes Y gets when X is out, learned from past games with X absent. No history for that absence: split by position and role.
2. Doubtful players: one minutes split per phase 0 scenario, from `E[min | play]` of the players who play in it.
3. Shared minutes per pair: the pair's historical share, rescaled to the new projected minutes.

**Variant, not v1: blowout risk from the market spread.** Starters play less in expected blowouts, and the spread is the best predictor of one. But it puts betting information into stage 1 (see the goal section), so v1 stays line-free and the spread is tested as a variant. If it is kept, it must be the spread at the same T as the injury snapshot, per intermediate snapshot.

**Decisions**

- How many games before trusting a specific pair's substitution edge.

**Things to try**

- Minutes error separately for games with and without key absences.
- Effect on the **absence counterfactual** (slope on `LINE_ERROR`), not only on minutes MAE.
- The spread-based blowout variant vs line-free v1, judged by what it adds in XGBoost, not by minutes MAE alone.

**Done when:** minutes error below v0 in games with absences, and the counterfactual built on it is at least as strong as 2_6's.

### Phase 4B. Matchup model (guard-edge weights)

The defensive counterpart of the minutes model. Minutes say who is on the floor; this says **who guards whom while they are**, and how much a defender changes what his assignments produce. It learns from the phase 2 matchup graphs.

**The data** ([player_lineup_and_matchup_data.md](player_lineup_and_matchup_data.md) §4). One row per attacker × defender × game: `matchup_seconds`, `partial_possessions`, `pct_total_time_both_on`, and the attacker's points, FGA, 3PA, FTA, assists and turnovers against that defender. ~245k rows a season, 2017-18 onward. The primary defender covers a median 36% of an attacker's matched time across ~9 defenders, so this is a distribution of assignments, not one assignment.

**Guarding shares (provider v1)**

- Target per pair and game: the guarding rate r_ij defined in phase 2, observed in that game. Inputs come only from earlier games, and the model is refit walk-forward like the minutes model.
- Model: the pair's own shrunk history blended with a small model that predicts the rate from both players' positions and profiles (size, usage, role) and the defending team's tendencies.
- Team scheme as a proxy: the switch and help columns are zero at the source, but a team whose defenders spread their time evenly across attackers behaves like a switching team. The dispersion of its guarding rates is a usable scheme feature.
- **Absences reassign edges.** When a defender is out, the model redistributes the attackers he usually guarded to tonight's available defenders. This is the matchup version of the substitution graph in 4A.

**Matchup-adjusted defensive rating**

A ridge model on the matchup edges: the attacker's points per partial possession (or shot-based efficiency) against defender j, minus the attacker's own norm, explained by a defender effect and weighted by partial possessions. This gives a defensive rating per player that knows **whom he guarded**, unlike the stint `def_i`, which credits all five defenders equally. Compare the two and blend if useful.

**Data limitations**

- Only from 2017-18; the first walk-forward seasons have little pair history.
- No timestamps: rates are game-level aggregates, assumed constant across a game's stints.
- `player_points` sums above the real score (median +4 per team-game): use relative measures or shot-based ones, not a points ledger.
- Recent data are preliminary: the NBA reprocesses tracking, and a game fetched the morning after can change.
- No switch or help data at all: help defense stays a team-level effect.

**Things to try**

- Weighting by `partial_possessions` vs `matchup_seconds`.
- Edges for each team's top-usage attackers only vs all pairs.
- Shot-based vs points-based efficiency in the defensive rating.

**Own validation:** predict next games' guarding rates better than the v0 rule (same position guards same position); matchup-adjusted defensive ratings predict future stints' points allowed at least as well as the stint `def_i`.

**Done when:** for any game and T, each attacker has an expected distribution over tonight's available defenders, including after absences, and each defender has a matchup-adjusted rating.

## Phase 5. Scalar game features into XGBoost (milestone 1)

Reads scalar features off the game graph, using phase 3 ratings and the phase 4 weights. Every feature here is an aggregation over the game graph: weighted sums over nodes, or over guard edges. The main arm is 2_5 plus these columns; the 2_6 `LU_*` columns are in the file but excluded from training (see the goal section).

**Per-game calculation** (poss = projected possessions per team; spread in the repo's implied-home-margin convention)

```math
\text{total}_{\text{lineup}} = \text{poss} \cdot \frac{\text{ORtg}_{\text{home}} + \text{ORtg}_{\text{away}}}{100} \qquad \text{margin}_{\text{lineup}} = \text{poss} \cdot \frac{\text{ORtg}_{\text{home}} - \text{ORtg}_{\text{away}}}{100}
```

HCA is already inside each side's ORtg (phase 3), so it is not added again.

**Matchup-weighted defense.** Instead of the opposing defense entering as a plain sum, each attacker faces the defenders expected to guard him, weighted by his usage and his projected minutes (a weighted sum over guard edges):

```math
\text{def\_faced}_{\text{side}} = \sum_{i \in \text{side}} \text{usage}_i \cdot \frac{\text{min}_i}{48} \sum_{j} m_{ij} \cdot \text{def}^{\text{mu}}_j
```

with `def^mu` the matchup-adjusted rating from phase 4B.

**New features** (see the catalog): ORtg per side, share of tonight's minutes for low-sample players, the prior-adjusted absence counterfactual, matchup-weighted defense faced per side, defense faced by each side's top scorers, and the **matchup absence counterfactual**: how much defense each side loses when an absent defender's assignments pass to tonight's replacements (tonight's graph vs its full-health counterfactual). The 2_6 level columns (projected pace, total, margin, scenario SD) are not recomputed: the level was null in 2_6, and the encoder heads (phase 8C) cover it. The absence idea is carried by the new `PI_*` counterfactuals; whether 2_6's own `LU_ABSENCE_IMPACT_*` still adds anything is phase 8's variant L.

**Doubtful scenarios.** Features are computed on every scenario's game graph and passed as weighted mean and SD.

**Diagnostic: rerun the oracle on v1.** Repeat the phase 2 oracle ceiling with the v1 weights from phase 4. How much of the v0-to-oracle gap v1 closes is what phase 4 bought; what remains locates the bottleneck:

- Oracle still well above v1 → the minutes or matchup model is the problem.
- No gain even with oracle weights → the player model or the features are the problem.

**Things to try**

- Lineup features alone vs combined with current features.
- v0 vs v1 weights, to see what phase 4 bought.
- Performance by slice: games with absences, games with a key defender absent, early season, after trades.
- Different prediction times T: where the edge against the line sits.

**What to look at:** comparison with 2_5 per season and per slice; whether the matchup absence counterfactual adds slope beyond `LU_ABSENCE_IMPACT_DEF_PTS_BEFORE` (an analysis on the file, not a training arm); whether new features get sensible weight and sign in XGBoost.

## Phase 6. Multitask pretraining (FM, then GNN with guard edges)

Learns player representations from the phase 2 training graphs: ~389k stint graphs (~778k stint sides) and ~2.1M matchup edges, far more material than the ~8k games XGBoost sees. Three models of increasing flexibility share the same targets and validation, so each step can be compared with the previous one.

**When it runs.** It can start right after phase 2, on v0 guard weights, in parallel with phases 3 and 4. Once phase 4B ships, the stint graphs are rebuilt with v1 guarding shares and phase 6 is rerun. Compare both runs: a weak v0 must not be read as "matchups do not help".

**Stint-level targets, per stint and side** (each weighted by the stint's possessions)

- Points per possession.
- Pace: possessions per 48 minutes (same unit as phase 3).
- Turnovers per possession, 3PA/FGA, FTA/FGA, offensive rebound rate OREB / (OREB + opponent DREB).

All are computable from the stint store, whose counts reconcile exactly with the box score. Several targets force the model to learn what kind of basketball those ten players produce, not only how much they score, and regularize noisy short stints (median ~73 s). The 3PA/FGA head deserves attention: it is the only target where a non-additive lineup effect has been measured so far.

**Edge-level targets, per attacker-defender pair and game** (from the matchup graphs, each weighted by partial possessions)

- Attacker's FGA and 3PA per partial possession against that defender, his shot efficiency, turnovers, and shooting fouls drawn.
- These train the same player embeddings to predict what happens **when this player guards that one**. This is direct supervision on opponent relationships, which the stint targets only give through the five-man sum.
- The same-game observed guarding share may also be a target (predicting it teaches the embeddings who tends to guard whom). It is never an input; the guard edges the encoder reads always carry expected shares (principle 7).

No betting information is a target or an input.

**Node inputs.** Each player enters with a learned ID embedding plus his node features (as-of-C_k profile and position). The profile gives rookies and low-minute players a representation from day one (the "similar players" idea, built into the model). 2016-17 stints have no matchup data: either start pretraining in 2017-18, or use v0 position-based guarding shares for that season.

**Step 6a. Factorization machine (linear interactions)**

```math
\hat{y}_O = \mu + h \cdot \text{HCA} + \sum_{i \in O} \text{off}_i - \sum_{j \in D} \text{def}_j
+ \frac{1}{10} \sum_{\substack{i<k \\ i,k \in O}} \langle w_i, w_k \rangle
+ \frac{1}{10} \sum_{\substack{j<l \\ j,l \in D}} \langle w'_j, w'_l \rangle
+ \frac{1}{25} \Big\langle \sum_{i \in O} u_i, \sum_{j \in D} v_j \Big\rangle
+ \frac{1}{5} \sum_{i \in O} \sum_{j \in D} m_{ij} \, \langle a_i, b_j \rangle
```

- `w`: offensive chemistry (teammate edges on offense); `w'`: defensive chemistry (teammate edges on defense); `u`, `v`: team-level offensive and defensive style (opponent edges, uniform); `a`, `b`: individual attacking and guarding vectors, interacting only through guard edges.
- The last term is the matchup term: each attacker is matched with the defenders who guard him, weighted by m_ij (which sums to 1 per attacker). The uniform `u·v` term is the closest thing to the null 2_6 bilinear test; the matchup term is new. Keeping both lets the model separate "this defense as a whole" from "this defender on this attacker".
- The pair terms are **means**, so an interaction has the scale of one pair. The 2_6 synergy work hit exactly this trap: summing ten pair terms inflated it tenfold.
- The edge-level targets use the same `a_i`, `b_j`: the predicted production of attacker i against defender j is a head on `⟨a_i, b_j⟩` plus both players' base terms.
- Prior inside the model: each vector = `A·x_i + δ_i`, with `x_i` the node features, `A` shared and `δ_i` penalized so it only grows with data.
- One output per target; the embeddings are shared across targets.
- Fast to train: also used to debug the pipeline before the GNN.

**Step 6b. GNN (contextual, non-linear)**

- Message passing over the three edge types of the stint graph (teammate, opponent, guards). Each player's representation is updated from each type separately, so the effect of a defender on an attacker can depend on the rest of the floor.
- Without guard edges the stint graph is complete and the GNN is close to attention over ten players with a side marker. The guard edges are what give it real graph structure. A relational GNN or a graph attention network with edge types and edge weights fits; a set transformer can take m_ij as an attention bias.
- Edge-level targets are predicted from the two endpoint embeddings after message passing.

**Optional branch: archetypes.** Soft clustering of node features (8-12 groups) and archetype × archetype tables, including attacker-archetype × defender-archetype matchup tables. Interpretable and cheap; useful to understand what the FM and GNN learn, not a prerequisite.

**Decisions**

- Embedding dimension (start small: 8-16).
- GNN architecture and depth.
- Task weights between stint-level and edge-level losses.
- Pretraining start: 2016-17 (with v0 shares) or 2017-18.
- Deep-learning dependency (PyTorch, possibly PyTorch Geometric). Neither is in `pyproject.toml`; add it as an optional group so production installs stay light until phase 9 decides.

**Own validation:** error on future stints and future matchup edges, per target, for RAPM → FM without the matchup term → FM with it → GNN (and archetypes if built), with RAPM at the same checkpoint cadence, on v0 and on v1 weights.

**What to look at:** FM with the matchup term beats FM without → who-guards-whom carries signal beyond the five-man block. FM beats RAPM → interactions carry signal. GNN beats FM → non-linear context carries signal. Given the 2_6 results, expect small gains on points per possession; per-target results (especially defense, 3PA/FGA and pace) are more informative than the pooled loss.

## Phase 7. Game embeddings and temporal out-of-fold generation

Runs the pretrained encoder on the phase 2 game graphs, with phase 4 weights, to get one vector per game built only from what was known at T. Two levels of embedding are involved.

**Node embeddings (h_i).** One per available player, after message passing over the game graph: "player i in the context of this game, and of whom he is expected to guard and be guarded by". They do not go to XGBoost directly.

**Game embedding (z_game).** Pooling of the node embeddings, weighted by projected minutes:

```math
z_{\text{game}} = \text{POOL}_{\text{min-weighted}}(h_1, \dots, h_n) \in \mathbb{R}^{16\text{–}32}
```

**The game graph is the as-of-T graph.** Roster, availability at T, projected minutes, expected guarding shares and doubtful scenarios from phases 0, 2 and 4, never the players who actually played, their real minutes or their real matchups. Otherwise the model learns from information it will not have at prediction time.

**Counterfactual encoding.** Encode each game graph and its full-health counterfactual (built in phase 2, with nodes **and guard edges** recomputed). The difference of the head outputs is the encoder's version of `LU_ABSENCE_IMPACT_*`, the one column family where 2_6 found signal. Because guard edges are reassigned when a defender is out, this counterfactual knows whom the absent defender would have guarded. Keep it split by head (points, possessions), mirroring the 2_6 split by channel.

**Optional game-level fine-tuning.** Same encoder, light fine-tuning on as-of-T game graphs with heads for home points, away points and possessions (never the line). The heads' outputs are kept as stable scalar features; the 16-32 dimensional z is kept as well.

**Out-of-fold generation.** Follows the phase 1 algorithm: train (pretraining and fine-tuning) on data before C_k, warm-start from the previous checkpoint, freeze, encode games in [C_k, C_(k+1)). Fine-tuning targets are almost XGBoost's targets, so this rule matters even more here.

**Doubtful scenarios.** Encode each scenario's game graph; pass the weighted mean of head outputs and z, plus the SD of head outputs.

**Stability of the latent space**

- Warm-start every checkpoint from the previous one instead of training from scratch.
- Fixed seed; optional penalty on changes between checkpoints.
- Procrustes alignment between checkpoints to inspect drift, and to apply if drift proves harmful. Alignment uses only the previous checkpoint, so applying it keeps the out-of-fold rule.

**Decisions**

- Size of z (start with 16).
- Pooling: minute-weighted mean per team, then combine both teams. For totals a symmetric combination (sum) fits; for spread a difference; concatenation keeps both but doubles the columns.
- Whether fine-tuning is worth its cost versus pretraining only.

## Phase 8. Embeddings into XGBoost (ablations)

Compares what each kind of stage 1 output adds on top of 2_5. Every variant uses the same walk-forward, the same out-of-fold features and the seed protocol from phase 1. The winning variant's columns are what the 2_7 layer ships.

| Variant | Features | What it tells us |
| --- | --- | --- |
| R | 2_5 only | Reference: the promoted 2_5 specs |
| A | R + phase 5 scalars | Whether the graph's scalar readouts help |
| B | A + FM scalars: chemistry, style interaction, matchup interaction, injured-replacement similarity | Whether named, stable interaction features help |
| C | A + GNN head outputs (home points, away points, possessions) and their absence counterfactual | Whether the learned encoder beats the additive model |
| C0 | As C, from a GNN trained **without** guard edges or edge-level targets | Whether who-guards-whom is what makes the encoder useful |
| D | C + small z (8-16 dims) | Whether there is signal beyond what the heads capture |
| E | C + full z (16-32 dims) | Whether more dimensions help or just add noise |
| F | Best of the above with FM embeddings instead of GNN | Whether the GNN's extra complexity pays off end to end |
| L | Best of the above + the 9 `LU_ABSENCE_IMPACT_*` columns from 2_6 | Whether 2_6's one family with measured signal still adds anything on top of 2_7 |

Variant L excludes `["STARTER_", "LU_PROJ_", "LU_FG3A_", "LU_ABSENCE_SHIFT_"]` instead of `["LU_", "STARTER_"]`; on the 2_6 file that keeps exactly the 9 `LU_ABSENCE_IMPACT_*` columns and drops nothing from 2_5.

A vs R is the first end-to-end test of the graph. C vs C0 is the end-to-end test of the matchup graph; phase 5's matchup scalars inside A are the simple version of the same test.

**Things to watch**

- With ~8k games and a noisy target, many unnamed columns invite spurious splits: check stronger XGBoost regularization for D and E.
- The promoted specs use low `colsample` (0.20-0.31) over ~2-3k columns, so a handful of new columns is rarely drawn. Tune per arm, or report how often new columns are used.
- Stability of D and E across seasons, since each season's z comes from different checkpoints.
- Performance on games with absences, and especially with a key defender absent, where matchups should matter most. Pre-register these slices before running.

**What to look at:** which variant does best per target, and whether the gains concentrate in the slices where we expect them.

## Phase 9. Production

To be defined. Constraints already known:

- Daily prediction needs the latest stage 1 checkpoint, the graph builder with the same weight-provider versions as training, and the stints and matchups up to yesterday. Stints are updated daily; matchups are not yet (phase 0); RAPM ratings still lack an automatic daily refresh (lineup plan §10).
- Checkpoints belong in S3 under their own prefix, not in git-tracked `models/`.
- Inference must run in the GitHub Actions environment, likely on CPU.
- Serving and training must compute the same columns at the same cutoff (feature parity check, as for 2_6).

## Candidate feature catalog

All features are computed per game and T from the game graph (roster, availability at T, projected minutes and expected guarding shares), as the weighted mean over doubtful scenarios unless stated. They are added in blocks, phase by phase. New columns use the `PI_` prefix and the `_BEFORE` convention that `select_training_columns()` filters on; per-side columns end in `_TEAM_HOME` / `_TEAM_AWAY`.

**In the file from 2_6, excluded from the main arm**

These ideas from the earlier draft already exist as 2_6 columns. They are not recomputed as `PI_*` columns, and the main arm excludes them (goal section); only variant L re-admits the absence-impact family.

| Idea in the earlier draft | 2_6 column |
| --- | --- |
| Projected pace | `LU_PROJ_POSS_BEFORE` |
| Lineup total and spread | `LU_PROJ_TOTAL_BEFORE`, `LU_PROJ_MARGIN_BEFORE` |
| Range of the total across doubtful scenarios | `LU_PROJ_AVAILABILITY_TOTAL_SD_BEFORE` |
| Absence impact per side, and by channel | `LU_ABSENCE_IMPACT_PTS_BEFORE_TEAM_{HOME,AWAY}`, `LU_ABSENCE_IMPACT_{OFF,DEF,PACE}_PTS_BEFORE`, `LU_ABSENCE_IMPACT_BENCH_DP_PTS_BEFORE` |
| Lineup total minus line | Not emitted on purpose: measured null, and XGBoost already has both inputs |

**New in 2_7**

| Feature | Phase | Unit | What it measures |
| --- | --- | --- | --- |
| `PI_ORTG_PROJ_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 5 | pts/100 | Expected offensive efficiency of each side vs the opposing defense |
| `PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 5 | 0-1 | Share of tonight's minutes for players with little history |
| `PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE` | 5 | points | Absence counterfactual with prior-adjusted ratings and the phase 4A minutes |
| `PI_MU_DEF_FACED_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 5 | pts/100 | Matchup-weighted defensive quality each side's attackers face |
| `PI_MU_DEF_VS_TOP_SCORERS_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 5 | pts/100 | Defensive quality expected on each side's top two scorers |
| `PI_MU_ABSENCE_DEF_IMPACT_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 5 | pts/100 | Defense lost when absent defenders' assignments pass to tonight's replacements |
| `PI_MU_SCHEME_DISPERSION_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 5 | unitless | How evenly each defense spreads its guarding (a switching proxy) |
| `PI_CHEMISTRY_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 8B | pts/100 | FM interaction of teammate pairs sharing the floor tonight |
| `PI_FAMILIARITY_MIN_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 8B | minutes | Historical minutes those pairs played together |
| `PI_STYLE_INTERACTION_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 8B | pts/100 | Team-level fit of one side's offensive style vs the other's defense |
| `PI_MATCHUP_INTERACTION_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 8B | pts/100 | FM matchup term: attackers vs their expected defenders |
| `PI_REPLACEMENT_SIMILARITY_BEFORE_TEAM_HOME`, `_TEAM_AWAY` | 8B | −1 to 1 | How similar the players inheriting the minutes are to the absent ones |
| `PI_HEAD_PTS_BEFORE_TEAM_HOME`, `_TEAM_AWAY`, `PI_HEAD_POSS_BEFORE` | 8C | points, possessions | Game-level head outputs of the encoder |
| `PI_HEAD_ABSENCE_IMPACT_PTS_BEFORE`, `PI_HEAD_ABSENCE_IMPACT_POSS_BEFORE` | 8C | points, possessions | Full-health minus tonight, from the encoder, with guard edges reassigned |
| `PI_HEAD_TOTAL_SD_BEFORE` | 8C | points | SD of the head total across doubtful scenarios |
| `PI_Z01_BEFORE` … `PI_Z16_BEFORE` (or `Z32`) | 8D-E | unitless | Game embedding coordinates |

Before any of these is added, follow the "Adding Or Modifying Features" checklist in [README_Training Data Processing.md](README_Training%20Data%20Processing.md) and record the columns in `nba_ou.config.dataset_versions`.

## Risks and open questions

**Risks**

- **Leakage between stages.** If a stage 1 model is trained on stints or games that XGBoost then uses, features look excellent and fail live. Mitigation: the temporal out-of-fold algorithm, for training rows too.
- **Train/serve mismatch in game graphs.** Fine-tuning on actual players, minutes or matchups teaches the model information it will not have at T. Mitigation: always as-of-T graphs, with projected minutes and expected guarding shares.
- **Observed matchups react to the game.** Coaches change assignments when someone gets hot, so a game's observed shares partly encode its outcome. Mitigation: expected shares as the encoder's edge weights everywhere, same-game observed shares only as labels and in the oracle, and an input-contract test (phase 2).
- **Probability counted twice in scenarios.** `minutes_model.walk_forward_minutes` returns `(1 − p_out) · E[min | play]`; used inside a scenario graph it applies the play probability a second time. Mitigation: scenario graphs use `E[min | play]`, and a phase 2 check forbids probability-scaled minutes.
- **Betting information leaking into stage 1.** A market input (for example the spread for blowout risk) makes stage 1 outputs partly restate the line. Mitigation: line-free v1, market inputs only as explicit variants.
- **Leakage through the profile prior.** A prior fitted once on all seasons, or on final ratings, encodes what players turned out to be. Mitigation: refit as of each date on as-of ratings and lagged profiles (phase 3).
- **Weak v0 weights mistaken for a null.** Early phase 6 runs use crude guarding shares. Mitigation: rerun on v1 weights before concluding anything about matchups.
- **Weight versions drifting between training and serving.** A model trained on graphs from one provider version and served with another sees a different input. Mitigation: graphs record their provider versions, and checkpoints record what they were trained on.
- **Matchup dilution for totals.** Suppressing one attacker moves shots to his teammates, so a strong player-level matchup effect can vanish at team level. Expect matchups to matter more for defense and spread than for the total level.
- **Matchup data quality.** No timestamps, no switch or help data, points that over-count, and revisions after the fact. Mitigation: game-level rates, shot-based measures, and a revision check (phase 0).
- **Roster membership from the target game.** Reading who appears in tonight's box score leaks the coach's rotation (the `N_ACTIVE_PLAYERS` case). Mitigation: roster from strictly earlier games plus the injury report, as in 2_6.
- **Line and injuries out of sync.** Using a 5 pm injury snapshot with a noon line invents an edge. Mitigation: the same T for both.
- **Latent drift across checkpoints.** z from different checkpoints may not mean the same thing. Mitigation: warm starts, small z, alignment checks, and head outputs as a stable fallback.
- **Weak interaction signal.** Stints are short and pairwise effects noisy; 2_6 measured synergy and team-level style interactions as null. Expect small gains, most likely in the defensive counterfactual and in 3PA/pace rather than in points per possession.
- **Comparisons lost in seed noise.** One seed per arm cannot resolve the effects expected here. Mitigation: seed ensembles or per-arm tuning (phase 1).
- **Few rows for XGBoost.** ~8k games in the initial walk-forward. Mitigation: few features, added in blocks, with strong regularization for embedding variants.
- **`GameRotation` dependency for new games.** PlayByPlayV2 is dead from 2024-25; the V3-only rebuild self-validates 89% of 2025-26 games, and the rest still depend on `GameRotation`, which can return corrupt rotations. Mitigation: improve the V3 rebuild.
- **Schema-layer rule.** 2_7 can only add columns. Anything that would change an `LU_*` value needs a `3_0` base instead.
- **2_6 columns slipping into the main arm.** The 2_5 start lives in each training config, not in the file, so a config that forgets the exclusion silently trains on 2_5 + 2_6 + 2_7. Mitigation: every 2_7 campaign config sets the exclusion, and the campaign README lists the feature count per arm so a stray 25 columns is visible.

**Open questions**

- [x] Which prediction times T to simulate? Closing plus the intermediate dataset's snapshots.
- [x] Keep the FM as an intermediate step before the GNN? Yes.
- [x] Box score DNP rows stored? Yes; `availability.py` classifies them.
- [x] Feature base for the 2_7 models? 2_5. The file is built on 2_6 (layer chain), and the training configs exclude the 2_6 columns.
- [x] Use who-guards-whom data? Yes, as a central part of the graph (phases 2, 4B, 6, 7).
- [x] When is the graph built? Phase 2, before any model, with v0 weights that phase 4 upgrades.
- [x] Can stage 1 use betting information? Not in v1; the spread-based blowout adjustment is tested as a variant.
- [x] When does the oracle (actual minutes and matchups) run? At the end of phase 2, before phase 4 is built.
- [ ] Use all six intermediate snapshots, or a subset, for stage 2 evaluation?
- [ ] Main reference metric: line error or total error? (Suggestion in phase 1.)
- [ ] Doubtful scenarios: keep 2_6's limit of 3 enumerated players per team, or raise it / sample?
- [ ] Checkpoint frequency: monthly or weekly?
- [ ] Graph library: PyTorch Geometric or plain tensors?
- [ ] GNN architecture: relational GNN, graph attention network with edge types, or set transformer with a matchup bias?
- [ ] Pretraining start: 2016-17 with v0 shares, or 2017-18?
- [ ] Task weights between stint-level and edge-level targets?
- [ ] Is game-level fine-tuning worth it versus pretraining only?
- [ ] Build the archetypes branch, or skip it?
- [ ] Where stage 1 checkpoints live and how production loads them (phase 9).
