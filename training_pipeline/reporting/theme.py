"""Shared visual language for the reporting notebooks.

One fixed colour per prediction strategy, assigned in slot order and never
cycled, so a strategy keeps its identity no matter which runs a filter leaves
on screen. The three slots are validated to stay distinguishable under
colour-vision deficiency across every pair, which is what makes them safe for
scatter and small-multiple charts as well as bars.

The bookmaker line is deliberately grey rather than taking a fourth hue: it is
a reference the models are measured against, not a competing series.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

STRATEGY_COLOR: dict[str, str] = {
    "total_points_regressor": "#2a78d6",   # blue
    "line_error_regressor": "#eb6834",     # orange
    "over_under_classifier": "#1baf7a",    # aqua
}
STRATEGY_SHORT: dict[str, str] = {
    "total_points_regressor": "total_points",
    "line_error_regressor": "line_error",
    "over_under_classifier": "classifier",
}
#: Runs saved before ``prediction_strategy`` existed carry only a target family.
FAMILY_TO_STRATEGY: dict[str, str] = {
    "total_points": "total_points_regressor",
    "line_error": "line_error_regressor",
    "over_under": "over_under_classifier",
}

LINE_REF = "#898781"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

#: -110 American odds, the standard price on NBA totals.
DECIMAL_ODDS = 1.0 + 100.0 / 110.0
BREAK_EVEN = 1.0 / DECIMAL_ODDS


def apply_theme() -> None:
    """Recessive grid, muted axes, ink-coloured text. Call once per notebook."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 7,
    })


def colour_of(row: Any) -> str:
    return STRATEGY_COLOR.get(row["prediction_strategy"], MUTED)


def short_name(strategy: str) -> str:
    return STRATEGY_SHORT.get(strategy, strategy)


def strategy_legend(ax: plt.Axes, strategies: Any, extra: list[Line2D] | None = None) -> None:
    """A legend is always present for two or more series, so identity never
    rests on colour alone."""
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=9,
               color=STRATEGY_COLOR.get(s, MUTED), label=short_name(s))
        for s in strategies
    ]
    handles += extra or []
    if len(handles) >= 2:
        ax.legend(handles=handles, fontsize=8, loc="best")


def label_bars(ax: plt.Axes, bars: Any, values: Any, fmt: str = "{:.1%}") -> None:
    """Direct value labels on bars.

    Required relief for the lower-contrast palette slot, and it lets every bar
    be read without tracking back to the axis.
    """
    for bar, value in zip(bars, values, strict=True):
        if pd.isna(value):
            continue
        height = bar.get_height()
        ax.annotate(
            fmt.format(value), (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points", xytext=(0, 4 if height >= 0 else -11),
            ha="center", fontsize=8, color=INK_2,
        )


def rotate_xticks(ax: plt.Axes, rotation: int = 20) -> None:
    ax.tick_params(axis="x", rotation=rotation)
    for tick in ax.get_xticklabels():
        # set_horizontalalignment, not the set_ha alias: the alias is not on
        # the typed Text stub, so mypy rejects it.
        tick.set_horizontalalignment("right")


def percent_frame(df: pd.DataFrame, percent: Any = (), decimals: Any = ()) -> Any:
    """Style a table: percentages as %, chosen columns to 3 decimals."""
    formatters = {c: "{:.2%}" for c in percent if c in df.columns}
    formatters.update({c: "{:.3f}" for c in decimals if c in df.columns})
    return df.style.format(formatters, na_rep="—")


#: Config fields that can distinguish one run from another, with how each
#: renders as a label token. Order here is the order tokens appear.
#:
#: A token is only added when the field actually VARIES across the runs being
#: compared, and only for runs that differ from the most common value. So a
#: label says what is unusual about a run rather than restating the campaign's
#: shared settings -- "line_error · 4500 · na200" reads as "the 4500 cell with
#: the relaxed row cap", which is the question that cell exists to answer.
LABEL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("data.csv_path", lambda v: "old-data" if "old_" in str(v) else "2.0-data"),
    #: Which dataset the run trained on. Without this a closing-line run and an
    #: intermediate-line run of the same strategy and window get IDENTICAL
    #: labels, which is how the snapshot campaign's three cells all read as
    #: "line_error - 3500" and became impossible to tell apart on a chart.
    #: factors.py has carried this since the campaign was designed; the label
    #: did not, so the notebooks matched on it and then hid it.
    ("data.dataset_type", lambda v: str(v).replace("_line", "")),
    #: Pooled over every pre-game snapshot (null) versus one model dedicated to
    #: a single horizon. Different training grain, different question.
    ("data.snapshot_minutes", lambda v: "pooled" if v is None else f"T-{v:g}"),
    ("data.season_year_floor", lambda v: f"from{v:.0f}"),
    ("data.exclude_overtime_from_training", lambda v: "no-OT" if v else "with-OT"),
    ("data.exclude_playoffs", lambda v: "no-playoffs" if v else "+playoffs"),
    ("cleaning.max_na_per_row", lambda v: f"na{v}"),
    ("cleaning.exclude_cols_containing",
     lambda v: "no-consensus" if v and "consensus_pct" in str(v)
     else "no-global" if v and "GLOBAL_" in str(v).upper()
     else "std-cols"),
    ("cleaning.nan_threshold", lambda v: f"nanthr{v:g}"),
    ("optuna.n_trials", lambda v: f"{v}trials"),
)


#: Fields where ``null`` is a CHOICE rather than a missing value, so it has to
#: count towards whether the field discriminates between runs.
#:
#: Everywhere else ``None`` means "this run's config does not mention the
#: field", and treating that as a value would put a token on runs that never
#: made the choice. ``data.snapshot_minutes`` is the opposite: null IS the
#: pooled-over-every-horizon setting. Because closing-line runs also leave it
#: null, the old rule saw a single non-null value (360), decided the field was
#: not a discriminator, and dropped it -- so the pooled and T-360 cells of the
#: snapshot campaign carried the same label while being different experiments.
NONE_IS_A_VALUE: frozenset[str] = frozenset({"data.snapshot_minutes"})


def _label_tokens(configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Per run, the tokens describing how it departs from the common setup."""
    tokens: dict[str, list[str]] = {name: [] for name in configs}
    for field, render in LABEL_FIELDS:
        meaningful_null = field in NONE_IS_A_VALUE
        values = {name: cfg.get(field) for name, cfg in configs.items()}
        if meaningful_null:
            # Only for runs that actually carry the key: a run whose config
            # predates the field is still genuinely missing it.
            values = {
                name: value
                for name, value in values.items()
                if field in configs[name]
            }
        present = {v for v in values.values() if meaningful_null or v is not None}
        if len(present) < 2:
            continue  # not a discriminator among these runs
        counts = pd.Series([str(v) for v in values.values()]).value_counts()
        baseline = counts.index[0]
        for name, value in values.items():
            if (meaningful_null or value is not None) and str(value) != baseline:
                tokens[name].append(str(render(value)))
    return tokens


def prepare_runs(runs: pd.DataFrame, *, describe: bool = True) -> pd.DataFrame:
    """Add the strategy/label columns every chart keys off.

    Labels start from strategy and training window, then gain a token for each
    config field that differs from the rest of the comparison (see
    LABEL_FIELDS) -- so runs from a campaign that varies overtime, playoffs or
    the missing-data cap are told apart by name instead of by a hash.

    ``describe=False`` falls back to strategy + window only, which is enough
    when every run differs on those alone.
    """
    runs = runs.copy()
    runs["prediction_strategy"] = runs["prediction_strategy"].fillna(
        runs["target_family"].map(FAMILY_TO_STRATEGY)
    )
    runs["strategy_short"] = runs["prediction_strategy"].map(STRATEGY_SHORT).fillna(
        runs["prediction_strategy"]
    )
    runs["is_classifier"] = runs["prediction_strategy"] == "over_under_classifier"

    runs["label"] = (
        runs["strategy_short"] + " · " + runs["train_games"].astype("Int64").astype(str)
    )

    if describe:
        from training_pipeline.reporting import loaders

        configs = {
            str(row["run_name"]): loaders.load_config_flat(row)
            for _, row in runs.iterrows()
        }
        configs = {k: v for k, v in configs.items() if v}
        if configs:
            tokens = _label_tokens(configs)
            runs["label"] += runs["run_name"].map(
                lambda name: "".join(f" · {t}" for t in tokens.get(str(name), []))
            ).fillna("")

    if runs["source_root"].nunique() > 1:
        runs["label"] += " · " + runs["source_root"]

    # Last resort. Reaching here means two runs are genuinely indistinguishable
    # by configuration -- a repeat, or a difference in a field not listed above.
    duplicated = runs["label"].duplicated(keep=False)
    if duplicated.any():
        runs.loc[duplicated, "label"] = (
            runs.loc[duplicated, "label"]
            + " [" + runs.loc[duplicated, "experiment_id"].astype(str).str[:4] + "]"
        )

    return runs.sort_values(
        ["prediction_strategy", "train_games", "created_at"]
    ).reset_index(drop=True)


#: What each strategy is actually being asked to predict. Used to replace the
#: terse artifact name with the question the run answers, so a reader who has
#: not memorised the strategy names can still read the chart.
STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "line_error_regressor": "Predicts error vs bookmaker line",
    "total_points_regressor": "Predicts final total points",
    "over_under_classifier": "Predicts over/under probability",
}


def describe_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Add the identity columns a report needs on top of ``prepare_runs``.

    Three of them, and they are different lengths on purpose:

    ``panel_label``   the compact form from :func:`prepare_runs`, minus the
                      campaign-folder suffix. Short enough for a chart panel.
    ``label``         the same information written out as prose, for tables and
                      headings where there is room for it.
    ``dataset_type`` / ``snapshot_minutes`` / ``rows_per_game``
                      read from each run's ``metadata.json``. ``rows_per_game``
                      in particular decides whether a binomial interval may be
                      reported at all, so it is a column rather than a lookup
                      every caller has to remember to do.
    """
    from training_pipeline.reporting import loaders

    runs = runs.copy()
    runs["panel_label"] = runs["label"]
    if runs["source_root"].nunique() > 1:
        runs["panel_label"] = [
            str(label).removesuffix(f" · {root}")
            for label, root in zip(runs["label"], runs["source_root"], strict=True)
        ]

    metadata = {
        str(row["label"]): loaders.load_metadata(row) for _, row in runs.iterrows()
    }
    runs["dataset_type"] = runs["label"].map(
        lambda label: metadata.get(label, {}).get("dataset_type")
    )
    runs["snapshot_minutes"] = runs["label"].map(
        lambda label: metadata.get(label, {}).get("snapshot_minutes")
    )
    runs["rows_per_game"] = runs["label"].map(
        lambda label: float(metadata.get(label, {}).get("rows_per_game", 1.0) or 1.0)
    )

    runs["label"] = [_describe(row) for _, row in runs.iterrows()]
    return runs


def _describe(run: Any) -> str:
    """One run as prose: what it predicts, on how much history, plus its
    distinguishing tokens."""
    window_raw = (
        str(int(run["train_games"])) if pd.notna(run["train_games"]) else "<NA>"
    )
    prefix = f"{run['strategy_short']} · {window_raw}"
    existing = str(run["label"])
    suffix = existing[len(prefix):] if existing.startswith(prefix) else ""
    description = STRATEGY_DESCRIPTIONS.get(
        run["prediction_strategy"],
        str(run["prediction_strategy"]).replace("_", " "),
    )
    return f"{description} · {window_text(run)}{suffix}"


def window_text(run: Any) -> str:
    if pd.isna(run["train_games"]):
        return "full-history window"
    return f"{int(run['train_games']):,}-game window"


def horizon_text(run: Any) -> str:
    """Closing line, pooled over every snapshot, or one named horizon.

    Read from the run's recorded ``dataset_type`` and ``snapshot_minutes``
    rather than from its name: a null ``snapshot_minutes`` means "pooled" on
    the intermediate dataset and "not applicable" on the closing one, and only
    the pair distinguishes them.
    """
    if run.get("dataset_type") == "closing_line":
        return "closing"
    snapshot = run.get("snapshot_minutes")
    return "pooled snapshots" if pd.isna(snapshot) else f"T-{snapshot:g}"


def run_spec(run: Any) -> str:
    """The few facts that tell one run from another, on one line."""
    rows_per_game = float(run.get("rows_per_game", 1.0) or 1.0)
    repeats = f" · {rows_per_game:.0f} rows/game" if rows_per_game > 1.05 else ""
    return (
        f"{run['strategy_short']} · {window_text(run)} · "
        f"{horizon_text(run)}{repeats}"
    )
