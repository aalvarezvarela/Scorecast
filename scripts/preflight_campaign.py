"""Check a campaign is runnable BEFORE committing hours of GPU time to it.

    poetry run python scripts/preflight_campaign.py experiments/window_overtime_2026_08

Pass individual YAML files when a comparison group shares a directory with
other campaigns:

    poetry run python scripts/preflight_campaign.py \
        experiments/intermediate_line_2026_08/pooled_7snapshot_line_error_no_time_decay.yaml \
        experiments/intermediate_line_2026_08/t360_control_line_error_no_time_decay.yaml

Exits non-zero if anything would block the campaign, so a runner can gate on it.

The check that justifies this script's existence is the training-window one.
``make_test_anchored_walk_forward_splits`` selects a fold's training rows with
``tail(train_games)`` and only skips the fold when fewer than
``min_train_games`` remain. So a window larger than the data supports does NOT
raise -- the early folds quietly train on fewer games than requested and the
run completes looking healthy, having silently stopped being the comparison it
was designed to be. That ceiling depends on the row count AFTER cleaning, which
is not knowable from the CSV without doing the cleaning, which is why guessing
it from raw row counts got it wrong twice.

Everything here is measured, never assumed: it builds the real splits with the
real code and reports the actual per-fold training sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_pipeline.cli import load_config  # noqa: E402
from training_pipeline.config import CVStrategy, ExperimentConfig  # noqa: E402
from training_pipeline.data import (  # noqa: E402
    compute_file_checksum,
    prepare_dataset,
    training_eligible_mask,
)
from training_pipeline.splits import (  # noqa: E402
    build_holdout_split,
    build_rolling_origin_plan,
    build_walk_forward_splits,
)

OK, WARN, FAIL = "ok", "warn", "FAIL"


def _check_dataset_key_integrity(
    csv: Path, config: ExperimentConfig
) -> tuple[str, list[str]]:
    """Cheaply validate the row identity fields even under ``--skip-data``.

    A checksum proves only that the bytes did not change; it can faithfully pin
    a malformed file. Reading the date/game/snapshot columns catches truncated
    rows and bad sentinels before a long campaign spends time on an earlier,
    healthy experiment and reaches the broken dataset hours later.
    """
    data = config.data
    columns = [data.game_id_col, data.date_col]
    if data.dataset_type.value == "intermediate_line":
        columns.append(data.snapshot_col)

    try:
        frame = pd.read_csv(
            csv,
            usecols=columns,
            dtype={data.game_id_col: "string", data.date_col: "string"},
        )
    except Exception as exc:  # noqa: BLE001 - turn parser failures into preflight output
        return "", [f"{csv.name}: cannot read key columns -- {type(exc).__name__}: {exc}"]

    problems: list[str] = []
    invalid_game_ids = int(frame[data.game_id_col].isna().sum())
    parsed_dates = pd.to_datetime(frame[data.date_col], errors="coerce", format="mixed")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_game_ids:
        problems.append(f"{csv.name}: {invalid_game_ids} row(s) have no game id")
    if invalid_dates:
        examples = frame.loc[parsed_dates.isna(), data.date_col].astype(str).unique()[:3]
        problems.append(
            f"{csv.name}: {invalid_dates} row(s) have invalid {data.date_col}, "
            f"e.g. {examples.tolist()}"
        )

    key_columns = [data.game_id_col]
    snapshot_summary = ""
    if data.dataset_type.value == "intermediate_line":
        snapshots = pd.to_numeric(frame[data.snapshot_col], errors="coerce")
        invalid_snapshots = int(snapshots.isna().sum())
        if invalid_snapshots:
            problems.append(
                f"{csv.name}: {invalid_snapshots} row(s) have invalid "
                f"{data.snapshot_col}"
            )
        key_columns.append(data.snapshot_col)
        snapshot_summary = f", snapshots={snapshots.nunique(dropna=True)}"

    duplicate_keys = int(frame.duplicated(key_columns).sum())
    if duplicate_keys:
        problems.append(
            f"{csv.name}: {duplicate_keys} duplicate row key(s) on {key_columns}"
        )

    summary = f"rows={len(frame):,}, unique_keys={frame[key_columns].drop_duplicates().shape[0]:,}{snapshot_summary}"
    return summary, problems


def _data_fingerprint(config: ExperimentConfig) -> str:
    """Configs sharing this share a cleaned dataset, so it is prepared once.

    Cleaning a ~250MB CSV is the expensive step; a campaign typically has far
    fewer distinct data settings than configs.
    """
    payload = {
        "data": config.data.model_dump(mode="json"),
        "cleaning": config.cleaning.model_dump(mode="json"),
        "family": config.family.value,
        "line_col": config.line_col,
    }
    return json.dumps(payload, sort_keys=True)


def _prepare_quietly(config: ExperimentConfig) -> tuple[Any, Any, Any]:
    """prepare_dataset + holdout split with the cleaning report suppressed."""
    payload = config.model_dump()
    payload["cleaning"]["verbose"] = 0
    quiet = ExperimentConfig.model_validate(payload)
    with redirect_stdout(StringIO()):
        prepared = prepare_dataset(quiet)
        df_dev, df_test = build_holdout_split(prepared.df_full, quiet)
    return prepared, df_dev, df_test


def check_configs(
    configs_paths: list[Path],
    *,
    label: str,
    skip_data: bool,
    allow_short_training_windows: bool = False,
) -> int:
    """Check an explicit, already-resolved set of experiment definitions."""
    if not configs_paths:
        print(f"No experiment configs found in {label}")
        return 1

    print(f"Pre-flight: {label}  ({len(configs_paths)} configs)\n")
    problems: list[str] = []

    # --- 1. every config parses ---------------------------------------------
    configs: dict[str, ExperimentConfig] = {}
    for path in configs_paths:
        try:
            configs[path.stem] = load_config(path)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the check
            problems.append(f"{path.name}: does not load -- {type(exc).__name__}: {exc}")
            print(f"  {FAIL}  {path.name}: {type(exc).__name__}: {exc}")
    if not configs:
        return 1
    print(f"  {OK}    all {len(configs)} configs parse\n")

    # --- 2. datasets exist, and their bytes are the pinned ones -------------
    print("Datasets")
    seen: dict[Path, str] = {}
    for name, config in configs.items():
        csv = Path(config.data.csv_path)
        csv = csv if csv.is_absolute() else REPO_ROOT / csv
        if not csv.exists():
            problems.append(f"{name}: dataset missing -- {csv}")
            print(f"  {FAIL}  {name}: missing {csv}")
            continue
        if csv not in seen:
            seen[csv] = compute_file_checksum(csv)
        actual, pinned = seen[csv], config.data.expected_checksum
        if pinned is None:
            print(f"  {WARN}  {name}: no expected_checksum pinned (actual {actual})")
        elif pinned != actual:
            problems.append(f"{name}: checksum mismatch, pinned {pinned} got {actual}")
            print(f"  {FAIL}  {name}: checksum {pinned} != {actual}")
        else:
            print(f"  {OK}    {name}: {csv.name} matches {actual}")
    print()

    # A checksum can pin corrupt bytes just as faithfully as healthy ones. This
    # deliberately runs even with --skip-data; it reads only identity columns,
    # so it is cheap compared with cleaning thousands of features.
    print("Dataset key integrity")
    checked_integrity: set[tuple[Path, str, str, str]] = set()
    for name, config in configs.items():
        csv = Path(config.data.csv_path)
        csv = csv if csv.is_absolute() else REPO_ROOT / csv
        identity = (
            csv,
            config.data.game_id_col,
            config.data.date_col,
            config.data.snapshot_col,
        )
        if identity in checked_integrity or not csv.exists():
            continue
        checked_integrity.add(identity)
        summary, integrity_problems = _check_dataset_key_integrity(csv, config)
        if integrity_problems:
            for problem in integrity_problems:
                problems.append(f"{name}: {problem}")
                print(f"  {FAIL}  {problem}")
        else:
            print(f"  {OK}    {csv.name}: {summary}")
    print()

    # --- 3. the design matrix -----------------------------------------------
    print("Design matrix (fields that differ across the campaign)")
    flat = {
        name: {
            "strategy": c.strategy.value,
            "train_games": c.walk_forward.train_games,
            "csv": Path(c.data.csv_path).name,
            "dataset_type": c.data.dataset_type.value,
            "snapshot_minutes": c.data.snapshot_minutes,
            "no_overtime": c.data.exclude_overtime_from_training,
            "drop_playoffs": c.data.exclude_playoffs,
            "max_na_per_row": c.cleaning.max_na_per_row,
            "nan_threshold": c.cleaning.nan_threshold,
            "exclude_cols": str(c.cleaning.exclude_cols_containing),
            "n_trials": c.optuna.n_trials,
            "seeds": str(list(c.evaluation_seeds)),
            # Without this a planted-signal campaign -- whose four cells differ
            # in nothing else -- prints "every config is identical, nothing is
            # being compared", which is the exact opposite of the truth.
            "cv_strategy": c.walk_forward.strategy.value,
            "retrain_every_days": c.walk_forward.retrain_every_days,
            # Paired with retrain_every_days: the cadence sets where a fold
            # STARTS and this floor sets how far it then grows, so a campaign
            # moving one without the other is not the campaign it looks like.
            "min_validation_games": c.walk_forward.min_validation_games,
            "eval_span_games": c.walk_forward.eval_span_games,
            "train_games_choices": str(c.walk_forward.train_games_choices),
            "time_decay": c.sample_weight.enabled,
            "tune_decay": c.sample_weight.tune_lambda,
            "allow_unweighted": c.sample_weight.allow_unweighted,
            "tune_n_estimators": c.optuna.tune_n_estimators,
            "objective_agg": c.optuna.objective_aggregation.value,
            "planted_variance": (
                c.diagnostics.planted_signal.variance_explained
                if c.diagnostics.planted_signal.enabled
                else None
            ),
        }
        for name, c in configs.items()
    }
    varying = [
        key for key in next(iter(flat.values()))
        if len({str(v[key]) for v in flat.values()}) > 1
    ]
    if not varying:
        print("  (every config is identical -- nothing is being compared)")
    else:
        width = max(len(n) for n in flat)
        widths = {
            k: max(len(k), max(len(str(v[k])) for v in flat.values())) for k in varying
        }
        header = "  ".join(f"{k:<{widths[k]}}" for k in varying)
        print(f"  {'config':<{width}}  {header}")
        for name, row in flat.items():
            cells = "  ".join(f"{str(row[k]):<{widths[k]}}" for k in varying)
            print(f"  {name:<{width}}  {cells}")
    print()

    if not configs.values() or all(not c.evaluation_seeds for c in configs.values()):
        print(f"  {WARN}  no config sets evaluation_seeds: results will have no error bar\n")

    if skip_data:
        print("Skipping the data-dependent window check (--skip-data).")
        return _verdict(problems, checked_windows=False)

    # --- 4. does each training window actually fit? -------------------------
    print("Training window feasibility (cleaning each distinct dataset once)")
    groups: dict[str, list[str]] = {}
    for name, config in configs.items():
        groups.setdefault(_data_fingerprint(config), []).append(name)

    for members in groups.values():
        config = configs[members[0]]
        try:
            prepared, df_dev, df_test = _prepare_quietly(config)
        except Exception as exc:  # noqa: BLE001
            for name in members:
                problems.append(f"{name}: prepare_dataset failed -- {exc}")
                print(f"  {FAIL}  {name}: prepare_dataset raised {type(exc).__name__}: {exc}")
            continue

        eligible = int(training_eligible_mask(df_dev, config).sum())
        print(f"  {Path(config.data.csv_path).name}"
              f"  (playoffs {'kept' if not config.data.exclude_playoffs else 'dropped'},"
              f" max_na={config.cleaning.max_na_per_row})")
        print(f"     cleaned={len(prepared.df_full)}  dev={len(df_dev)}  "
              f"holdout={len(df_test)}  train-eligible={eligible}")

        for name in members:
            member = configs[name]
            if member.walk_forward.strategy == CVStrategy.ROLLING_ORIGIN:
                problems.extend(
                    _check_rolling_origin(name, member, df_dev)
                )
                continue
            choices = member.walk_forward.train_games_choices
            requested = max(choices) if choices else member.walk_forward.train_games
            # Feasibility must be measured with training-row filters OFF.
            # build_walk_forward_splits applies them (overtime, etc.) AFTER
            # tail(train_games), so a filtered fold is legitimately smaller than
            # the window -- that is the filter working, not the window
            # overflowing. Only an unfiltered shortfall means the data does not
            # reach back far enough.
            unfiltered = member.model_dump()
            unfiltered["data"]["exclude_overtime_from_training"] = False
            unfiltered["cleaning"]["verbose"] = 0
            probe = ExperimentConfig.model_validate(unfiltered)
            try:
                with redirect_stdout(StringIO()):
                    splits, _ = build_walk_forward_splits(df_dev, probe)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: split building failed -- {exc}")
                print(f"     {FAIL}  {name}: {type(exc).__name__}: {exc}")
                continue

            sizes = [len(train) for train, _ in splits]
            n_folds = len(splits)
            short = [s for s in sizes if requested and s < requested]
            expected_folds = member.walk_forward.max_folds

            if requested and short:
                largest_ok = min(short)
                message = (
                    f"{name}: {len(short)} of {n_folds} folds cannot reach "
                    f"train_games={requested} (smallest pool {largest_ok}). The window "
                    "does not fit, and this does NOT raise at runtime -- those folds "
                    f"would quietly train on less. Largest window all folds support: "
                    f"~{largest_ok}."
                )
                if not allow_short_training_windows:
                    problems.append(message)
                print(f"     {WARN if allow_short_training_windows else FAIL}  "
                      f"{name}: {len(short)}/{n_folds} folds short "
                      f"(min {largest_ok} vs requested {requested}) "
                      f"-> max feasible ~{largest_ok}"
                      f"{' (explicitly allowed)' if allow_short_training_windows else ''}")
            elif n_folds < expected_folds:
                problems.append(
                    f"{name}: only {n_folds} folds built, {expected_folds} configured -- "
                    "fold counts must match for runs to be comparable."
                )
                print(f"     {FAIL}  {name}: {n_folds} folds, expected {expected_folds}")
            else:
                note = ""
                if member.data.exclude_overtime_from_training:
                    with redirect_stdout(StringIO()):
                        filtered, _ = build_walk_forward_splits(df_dev, member)
                    actual = min(len(t) for t, _ in filtered)
                    note = (f"; training filter then removes rows, leaving "
                            f"{actual}/fold (expected, not a misfit)")
                print(f"     {OK}    {name}: {n_folds} folds, all reach "
                      f"{requested}{note}")
        print()

    return _verdict(problems, checked_windows=True)


def _check_rolling_origin(
    name: str, member: ExperimentConfig, df_dev: Any
) -> list[str]:
    """Fold count, realised volume and window feasibility for a rolling-origin cell.

    The two things worth catching here, both of which look healthy at runtime:

    * ``eval_span_games`` is a target, not a promise -- the region grows in whole
      game-days, so the realised volume must be printed rather than assumed.
    * every ``train_games_choices`` value has to fit the EARLIEST fold's history.
      One that does not would make that fold train short for some trials and not
      others, so the window comparison would be measuring the shortfall.
    """
    problems: list[str] = []
    unfiltered = member.model_dump()
    unfiltered["data"]["exclude_overtime_from_training"] = False
    unfiltered["cleaning"]["verbose"] = 0
    probe = ExperimentConfig.model_validate(unfiltered)

    try:
        with redirect_stdout(StringIO()):
            plan = build_rolling_origin_plan(df_dev, probe)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{name}: rolling-origin plan failed -- {exc}")
        print(f"     {FAIL}  {name}: {type(exc).__name__}: {exc}")
        return problems

    wf = member.walk_forward
    choices = wf.train_games_choices or ((wf.train_games,) if wf.train_games else ())
    ceiling = plan.min_history_games
    too_big = [choice for choice in choices if choice and choice > ceiling]

    print(
        f"     {FAIL if too_big else OK}    {name}: {plan.n_folds} folds, "
        f"{plan.n_validation_days} game-days, {plan.n_validation_games} validation "
        f"games (asked ~{wf.eval_span_games}); every-{wf.retrain_every_days}-day "
        f"retrain; window ceiling {ceiling}"
    )
    if too_big:
        problems.append(
            f"{name}: train_games choices {too_big} exceed the {ceiling} games the "
            "earliest fold has available. Those trials would train short on that "
            f"fold. Largest window every fold supports: {ceiling}."
        )
    if member.data.exclude_overtime_from_training:
        with redirect_stdout(StringIO()):
            filtered = build_rolling_origin_plan(df_dev, member)
        print(
            f"           training filter leaves a {filtered.min_history_games}-game "
            "history on the earliest fold (expected, not a misfit)"
        )
    return problems


def check_campaign(
    campaign_dir: Path,
    *,
    skip_data: bool,
    allow_short_training_windows: bool = False,
) -> int:
    """Backward-compatible directory entry point used by existing callers."""
    configs_paths = sorted(
        path
        for path in campaign_dir.glob("*.y*ml")
        if not path.name.startswith("_")
    )
    return check_configs(
        configs_paths,
        label=str(campaign_dir),
        skip_data=skip_data,
        allow_short_training_windows=allow_short_training_windows,
    )


def _resolve_config_paths(inputs: list[Path]) -> tuple[list[Path], list[str]]:
    """Expand directories and retain explicitly selected YAML files.

    A campaign directory remains the convenient default. Explicit files make
    it possible to preflight one matched comparison group when several groups
    intentionally live beside one another.
    """
    configs: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = raw if raw.is_absolute() else REPO_ROOT / raw
        if path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.glob("*.y*ml")
                if not candidate.name.startswith("_")
            )
            if not candidates:
                errors.append(f"No experiment configs found in directory: {path}")
        elif path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            candidates = [path]
        elif path.exists():
            errors.append(f"Not a YAML config or directory: {path}")
            candidates = []
        else:
            errors.append(f"Path does not exist: {path}")
            candidates = []

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                configs.append(candidate)

    duplicate_stems = {
        path.stem for path in configs if sum(p.stem == path.stem for p in configs) > 1
    }
    if duplicate_stems:
        errors.append(
            "Config filenames must be unique across one preflight; duplicate stems: "
            + ", ".join(sorted(duplicate_stems))
        )

    return configs, errors


def _verdict(problems: list[str], *, checked_windows: bool) -> int:
    if problems:
        print(f"{len(problems)} blocking problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if checked_windows:
        print("Pre-flight passed: configs, checksums and training windows all check out.")
    else:
        # Saying "passed" here would imply the one check that actually needs
        # doing, and the one that silently misbehaves when wrong.
        print("Configs, checksums and dataset keys check out. The training-window "
              "check was SKIPPED, so a window too large for the data would still go "
              "unnoticed -- rerun without --skip-data before committing the campaign.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help=(
            "One or more experiment YAML files or folders. Folders contribute "
            "all top-level YAML files except underscore-prefixed base files."
        ),
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Config, checksum and row-key checks only; skips cleaning, "
                             "which is the slow part but also the only way to verify "
                             "the training window.")
    parser.add_argument(
        "--allow-short-training-windows",
        action="store_true",
        help=(
            "Report but do not fail when early folds contain fewer rows than "
            "train_games. Intended only for faithful replay of historical runs "
            "that already had this behavior."
        ),
    )
    args = parser.parse_args()

    configs, errors = _resolve_config_paths(args.paths)
    if errors:
        for error in errors:
            print(error)
        return 1
    label = ", ".join(str(path) for path in args.paths)
    return check_configs(
        configs,
        label=label,
        skip_data=args.skip_data,
        allow_short_training_windows=args.allow_short_training_windows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
