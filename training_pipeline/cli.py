"""argparse CLI entry point: run one training_pipeline experiment from a YAML
config file. Matches the repo's existing convention -- every script under
scripts/ uses argparse, never click.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from training_pipeline.config import ExperimentConfig
from training_pipeline.pipeline import run_experiment
from training_pipeline.snapshot_scoring import format_snapshot_table

#: Shared defaults every experiment file is merged on top of. Looked up by
#: walking upwards from the experiment file, so nested directories such as
#: experiments/rolling_origin_2026_08/ inherit experiments/_base.yaml.
#:
#: Nearest wins, which is what keeps experiments/archived/ reproducible: it
#: holds its own frozen _base.yaml, so the campaigns beneath it still resolve
#: to the protocol they actually ran under rather than to today's defaults.
BASE_CONFIG_FILENAME = "_base.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Nested dicts merge key-by-key so an experiment can change one field of a
    section (e.g. only ``optuna.mae_tolerance_abs``) without restating the rest
    of it. Lists are replaced wholesale, not concatenated -- appending would
    make it impossible to shrink a list such as ``edge_thresholds``.

    An EMPTY mapping in the override replaces rather than merges, which is the
    only way to say "none of these" about an inherited mapping. Merging it
    key-by-key is a no-op by construction -- it has no keys -- so without this
    rule ``corr_threshold_overrides: {}`` resolves to whatever ``_base.yaml``
    set, and a cell written to have no overrides silently inherits them. That
    exact case turned a single-factor control into a two-factor one. ``null``
    is not an escape either: several fields read None as "use my built-in
    default", so it means the opposite of empty. Nothing writes ``{}`` to mean
    "leave this alone", which is why the shorter rule loses nothing.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict) and value:
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def find_base_config(config_path: Path) -> Path | None:
    """Nearest ``_base.yaml`` at or above the experiment file's directory."""
    for directory in config_path.resolve().parents:
        candidate = directory / BASE_CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return None


def campaign_scoped_root(config_path: Path, root: Path) -> Path:
    """Group a campaign's runs under a folder named after the campaign.

    ``experiments/<campaign>/cell.yaml`` writes to
    ``artifacts/experiments/<campaign>/<run>``, so the artifacts tree mirrors
    the way the definitions are organised and one campaign's runs stay together
    instead of interleaving with every other campaign's by timestamp.

    A config sitting directly beside ``_base.yaml`` is not part of a campaign
    and is left at the root. Nesting is safe for every reader: run discovery
    finds runs by recursive search for the run marker and resolves them by
    ``experiment_name``, never by depth (see reporting.discovery).
    """
    parent = config_path.resolve().parent
    if (parent / BASE_CONFIG_FILENAME).exists():
        return root
    return root / parent.name


def load_config(config_path: str | Path, *, use_base: bool = True) -> ExperimentConfig:
    """Load an experiment definition, layered on the shared defaults.

    The experiment file states only what it changes; everything else falls back
    to ``_base.yaml``, and anything absent there falls back to the pydantic
    defaults. Pass ``use_base=False`` to load a file in isolation.

    ``experiment_root_dir`` additionally gets scoped to the campaign folder,
    unless this file sets it itself. That has to be decided from the file's OWN
    contents rather than the merged result: _base.yaml always supplies a value,
    so after merging, "inherited the default" and "asked for this path" look
    identical.
    """
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping.")

    states_own_root = "experiment_root_dir" in raw

    if use_base and config_path.name != BASE_CONFIG_FILENAME:
        base_path = find_base_config(config_path)
        if base_path is not None:
            base_raw = yaml.safe_load(base_path.read_text()) or {}
            raw = deep_merge(base_raw, raw)

        if not states_own_root:
            inherited = raw.get("experiment_root_dir", "artifacts/experiments")
            raw["experiment_root_dir"] = str(
                campaign_scoped_root(config_path, Path(inherited))
            )

    return ExperimentConfig.model_validate(raw)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a training_pipeline experiment from a YAML config file."
    )
    parser.add_argument("config_path", help="Path to a YAML experiment config file.")
    # Tri-state on purpose: default None means "let refit.train_production_model
    # decide". It used to be a plain store_true, so the absent flag passed
    # save_model=True and OVERRODE a config that had turned the refit off --
    # which is why every campaign runner in experiments/runners/ passes
    # --no-save-model with a comment explaining that its cells would otherwise
    # collide on the bundle name. The config asked for no model and got one.
    parser.add_argument(
        "--save-model",
        dest="save_model",
        action="store_true",
        default=None,
        help=(
            "Force a production refit and save the bundle, overriding "
            "refit.train_production_model."
        ),
    )
    parser.add_argument(
        "--no-save-model",
        dest="save_model",
        action="store_false",
        help="Force-skip the production refit even if the config asks for one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved config; do not train.",
    )
    return parser


def _print_snapshot_report(result: Any) -> None:
    """Print the per-horizon tables, or say why the pooled row cannot be read.

    This runs on the ordinary CLI path rather than in a wrapper script, because
    a wrapper is something you have to remember. Before this, running a pooled
    intermediate-line config through ``python -m training_pipeline.cli``
    trained and evaluated it correctly and then printed a headline ROI whose
    ``n_bets`` counted one game once per snapshot, with a Wilson interval built
    on that count and nothing anywhere saying so. The only warning lived in a
    comment in an archived YAML.
    """
    if not result.prepared.is_pooled_snapshots:
        return

    print(
        f"\n{'=' * 78}\n"
        f"POOLED ROW IS NOT A BET COUNT: this dataset holds "
        f"{result.prepared.rows_per_game:.1f} rows per game "
        f"({result.prepared.n_snapshots} pre-game horizons over "
        f"{result.prepared.n_games:,} games), so the ROI above counts one game "
        f"once per horizon.\nRead the per-horizon tables below: within one "
        f"horizon there is exactly one row per\ngame, so n_bets counts "
        f"independent events and the interval is honest."
    )

    report = getattr(result, "snapshot_report", None)
    if not report:
        print(
            "\nNo per-horizon table was produced -- the snapshot column was not "
            "found on the\nprediction frames. The pooled numbers above should "
            "not be reported."
        )
        return

    for name, table in report.items():
        title = {
            "cv": "CROSS-VALIDATION FOLDS (pooled validation rows)",
            "holdout": "HELD-OUT TEST PERIOD (daily walk-forward)",
        }.get(name, name.upper())
        print(f"\n{'=' * 78}\n{title}\nby virtual bet time, minutes before tip\n")
        print(format_snapshot_table(table))

    print(
        "\nThe ALL row pools every horizon, so its interval and significance "
        "verdict are\nleft blank rather than reported: correlated repeats break "
        "the binomial\nassumption behind them in the anti-conservative direction."
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.dry_run:
        print(config.model_dump_json(indent=2))
        return

    result = run_experiment(config, save_model=args.save_model)

    print(f"Experiment: {config.experiment_name}")
    print(f"Target: {config.family.value}")
    evaluation = result.holdout_result or result.walk_forward_result
    if evaluation is not None:
        print(f"Evaluation mode: {config.holdout_evaluation.value}")
        print(f"Test MAE: {evaluation.mae:.4f}")
        baseline = (
            result.holdout_result.baseline_holdout
            if result.holdout_result is not None
            else result.walk_forward_result.baseline  # type: ignore[union-attr]
        )
        print(f"Bookmaker-line baseline MAE: {baseline.mae:.4f}")
        primary = evaluation.betting_primary
        if primary.roi is not None:
            pooled = result.prepared.is_pooled_snapshots
            unit = "rows" if pooled else "bets"
            significance = (
                "see per-horizon table"
                if pooled
                else f"significant: {primary.is_significant}"
            )
            print(
                f"ROI @ edge>{primary.min_edge}: {primary.roi:+.2%} "
                f"on {primary.n_bets} {unit} ({significance})"
            )

    _print_snapshot_report(result)

    if result.run_dir is not None:
        print(f"Run directory: {result.run_dir}")
    if result.model_path is not None:
        print(f"Model saved to: {result.model_path}")


if __name__ == "__main__":
    main()
