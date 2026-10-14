# Phase 5: 2_5 vs 2_5 + PI block v1, closing `LINE_ERROR` (paired walk-forward)

The first downstream test of the player-graph work (schema 2_7 plan, phase 5):
do three game-graph features improve the promoted 2_5 closing `LINE_ERROR`
model? Research only, nothing is promoted. Decided 2026-10-14; the research
log (`docs/player_graph/frozen_decisions_and_future_checks.md`, "Phase 5,
step 4") is the record. **Nothing here may be changed after a run starts**,
and the protocol or hyperparameters do not change for hardware reasons
without a decision first.

## Pre-registration (fixed before any evaluation run)

| | |
|---|---|
| Dataset | `data/train_data/closing_line_research_2_6_pi_v1_20261014.parquet` (checksum `sha256:32fcdc51ef979b53`): the 2_6 closing file `closing_line_data_2_6_20261003.parquet` + the `PI_*` columns, joined by `GAME_ID`; provenance in the `.provenance.json` next to it |
| Arm A (`a_baseline_2_5.yaml`) | 2_5 features: `LU_`, `STARTER_`, `PI_` excluded (1,239 features after cleaning) |
| Arm B (`b_pi_primary.yaml`) | 2_5 + `PI_LOW_SAMPLE_MIN_SHARE_BEFORE_TEAM_HOME`, `_TEAM_AWAY`, `PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE`; the rating levels (`PI_OFF_RATING_*`, `PI_DEF_RATING_*`) excluded, kept for future ablations |
| Model | Promoted 2_5 closing `line_error` spec `d055949f99db`, hyperparameters fixed (92 trees, depth 4, colsample 0.204, ...), its cleaning, floor 2019-20, regular season + play-in |
| Walk-forward | Daily: each game day is predicted by a model refitted on every earlier game (window capped at 6,200 games). No prediction before 1,250 earlier games: first day **2021-01-21**; window to **2026-04-17**, 945 game days, 7,042 games |
| Seeds | 101, 202, ..., 1010, paired (the same seed in both arms) |
| Primary loss | Absolute error of the `LINE_ERROR` prediction |
| Primary statistic | Mean over games of `|error_A| − |error_B|` (positive = the block helps), each arm's prediction being the mean of its 10 seeds; SE clustered by game date, 95% interval |
| Training randomness | Arm A's seed band (MAE per seed, its 45 pairwise differences), measured before arm B runs; the 10 per-seed paired differences |
| Decision rule | **Improves** if the 95% interval is above 0 **and** at least 8 of the 10 per-seed differences are positive; **hurts** if below 0 and at least 8 negative; otherwise **null**. One primary test |
| Secondary (no decisions) | Squared error; calibration slope; per season; slices: \|`LU_ABSENCE_IMPACT_PTS_BEFORE`\| > 3, no PI coverage (placebo, should be ~0), the first 30 days of each season, after each trade deadline; betting at the spec's 0.1-point edge against the stored prices (win rate with a Wilson interval, ROI) as an economic diagnostic |

**Not an untouched holdout.** 2019-25 already served the phase 3 and 4A
out-of-sample evaluations and the 2_6 trial, and the 2_5 hyperparameters were
selected on folds inside it (a bias both arms share). This is the first
downstream comparison under a frozen specification.

## Running it on the GPU machine

Everything the run needs is the repository at the committed revision and one
data file; no database.

1. **Code.** `git fetch && git checkout feat/player-graph-2-7` at the commit
   that adds this folder, then `poetry install` (Python 3.11-3.13; the
   optional `graph` group is not needed). Check XGBoost and CUDA:

   ```bash
   poetry run python -c "import xgboost; print(xgboost.__version__, xgboost.build_info()['USE_CUDA'])"
   nvidia-smi
   ```

   Expected: `2.1.4 True` (the PyPI wheel for Linux includes CUDA).
2. **Data.** Copy `data/train_data/closing_line_research_2_6_pi_v1_20261014.parquet` and its
   `.provenance.json` to the same path, then verify the checksum:

   ```bash
   poetry run python -c "from training_pipeline.data import compute_file_checksum as c; print(c('data/train_data/closing_line_research_2_6_pi_v1_20261014.parquet'))"
   ```

   It must print `sha256:32fcdc51ef979b53` (the configs pin it; a mismatch stops
   every run).
3. **Benchmark** (minutes) and choose the device:

   ```bash
   poetry run python scripts/player_graph/benchmark_pi_fits.py \
       --config experiments/pi_block_v1_closing_2026_10/b_pi_primary.yaml \
       --out artifacts/experiments/pi_block_v1_benchmark.json
   ```

   It times the walk-forward's own fit on CPU and GPU at 1,250 / 3,000 / 6,200
   training games, reports the peak GPU memory and the CPU-vs-GPU prediction
   difference, and projects the full run. GPU XGBoost need not be faster for
   92 shallow trees. **Whichever device is chosen, every run of the
   comparison uses it** (CPU and GPU predictions differ slightly).
4. **Smoke test** (a few minutes; writes to `*_smoke`, never mixed with the
   real run):

   ```bash
   poetry run python scripts/player_graph/run_pi_walk_forward.py \
       --config experiments/pi_block_v1_closing_2026_10/a_baseline_2_5.yaml \
       --arm a_baseline --expect-pi none --seeds 101 --max-days 2 --device DEVICE
   poetry run python scripts/player_graph/run_pi_walk_forward.py \
       --config experiments/pi_block_v1_closing_2026_10/b_pi_primary.yaml \
       --arm b_pi_primary --expect-pi primary --seeds 101 --max-days 2 --device DEVICE
   ```

5. **The run**: `bash experiments/runners/run_pi_block_v1.sh DEVICE` (in
   `tmux` / `nohup`). It runs arm A's ten seeds, prints arm A's seed band,
   runs arm B's ten seeds and writes the report. Each month of each seed is
   a checkpoint: after an interruption, run the same command again and
   finished months are skipped. Do not change the device mid-run (the report
   refuses mixed devices).
6. **Bring back** `artifacts/experiments/pi_block_v1/` (per-month predictions,
   `run.json` per arm and seed, `report/`) and the log under
   `artifacts/logs/`.

## Output

- `artifacts/experiments/pi_block_v1/<arm>/seed=<s>/month=<YYYY-MM>.parquet`:
  one row per game (`GAME_ID`, date, `y_true`, `y_pred`, target and baseline
  lines, actual total, predicted edge, over / under decimal odds).
- `.../<arm>/seed=<s>/run.json`: config fingerprint, dataset checksum, device,
  feature list, hyperparameters, package versions, machine.
- `.../report/seed_band.json`: arm A's band (before arm B runs).
- `.../report/{report.md,metrics.json,games.parquet}`: the pre-registered
  reading above; the report refuses arms that do not cover the same games,
  arms whose 2_5 features differ, or runs on different devices.

## Runtime

18,900 fits (945 days x 10 seeds x 2 arms). On the development machine
(12 CPU threads, `benchmark_pi_fits.py`, 2026-10-14) one fit takes 2.8 / 5.0 /
12.0 s at 1,250 / 3,000 / 6,200 training games: about 44 h for the whole run.
The benchmark gives the GPU machine's figure.

## Known limits

- **No real prices.** The closing dataset carries the lines but no over /
  under prices, and the promoted spec configures none, so the betting
  diagnostic settles at -110 (the spec's `flat_decimal_odds`). It is a
  secondary diagnostic only.
- **Game ids.** On the closing path the pipeline drops `GAME_ID` during
  cleaning; the runner matches each row back to the dataset file on columns
  cleaning leaves unchanged and stops unless the match is one to one.
