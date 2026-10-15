# Pregame graph campaign v1: do learned representations add to XGBoost? (prepared, not run)

Research only, nothing is promoted. Decided 2026-10-15; the research log
(`docs/player_graph/frozen_decisions_and_future_checks.md`, "Phase 7") is the
record. Phase 5's campaign (`experiments/pi_block_v1_closing_2026_10/`) is
unchanged and separate. **Nothing here may change once the campaign stage
starts.**

## Pre-registration

| | |
| --- | --- |
| Encoders | `gnn_node` (node-only, capacity-matched) and `gnn_full` (teammate, opponent and expected-guard message passing), width 32, two layers, the frozen step 6b definitions; seed 0; exact training-mean centering; trained on the 88 monthly folds 2016-12 → 2026-04, each fold on stints before its month (`fm_oof.py`) |
| Adapter | Each pregame scenario (phase 4A v1 minutes, report scenarios incl. questionable / doubtful, corrected expected-guard history) as 48 one-minute lineups per team (systematic sampling on expected minutes, slot k vs slot k); outputs averaged over slots, probability-weighted over scenarios; the full-health graph is the counterfactual (`pregame_encoder.py`) |
| Out of fold | Every game encoded by the checkpoint of its own month (trained on earlier stints only), training rows and predicted rows alike; checked when the research file is built |
| Features | `GE_STYLE_*` (16): predicted turnovers per possession, 3PA/FGA, FTA/FGA and offensive rebound rate per offensive side, tonight and the absence shift, from `gnn_node` · `GE_NODE_EMB_*` (16): `gnn_node`'s game embedding · `GE_GRAPH_EMB_*` (16): `gnn_full`'s · embedding = each side's pooled embedding on 8 fold principal directions (last 20,000 training stints), Procrustes-aligned fold to fold |
| Dataset | Phase 5's research file (`sha256:32fcdc51ef979b53`) + the `GE_` columns by `GAME_ID` (`prepare_pregame_campaign.py`; checksum pinned in the generated configs) |
| Arms | A 2_5 · B 2_5 + PI primary · C B + `GE_STYLE_` · D C + `GE_NODE_EMB_` · E C + `GE_GRAPH_EMB_` (all: `LU_`, `STARTER_` excluded as phase 5) |
| Model and walk-forward | Phase 5's exactly: promoted 2_5 closing `LINE_ERROR` spec `d055949f99db`, fixed hyperparameters, daily refits, first day 2021-01-21, to 2026-04-17, seeds 101 … 1010 paired across arms, one device for every run |
| Comparisons | B vs A · **C vs B** (learned style) · **D vs C** (neural embedding) · **E vs D** (graph relations beyond neural capacity) · E vs C |
| Statistic and rule | Phase 5's: mean of `|e_ref| − |e_arm|` (ten-seed ensemble), SE clustered by date; improves if the 95% interval is above 0 and ≥ 8 of 10 seeds positive, hurts if the mirror, else null |

**Not an untouched holdout**, as phase 5. Five comparisons on one window are
read as a family; none is a confirmation alone. The market line stays
downstream in XGBoost; the market-conditioned encoder is a future ablation.

## Running it

`bash experiments/runners/run_pregame_graph_v1.sh <stage> [cpu|cuda]`, stages
`encoders`, `store` (needs the database), `outputs`, `prepare`, `campaign`,
`report`; each resumes. Stages up to `prepare` take a few hours on this
machine's CPU; `campaign` is about 2.5× phase 5's run (five arms instead of
two).
