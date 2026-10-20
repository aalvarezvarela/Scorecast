"""Per-snapshot betting metrics for a dataset with several rows per game.

The intermediate-line dataset holds one row per (game, pre-game snapshot), so a
single game appears once at every configured horizon. One model is trained on
all of those rows, with ``TIME_TO_MATCH_MIN`` as an ordinary feature, so it can
learn how the mapping
changes with time to tip. This module is about how such a model is *scored*.

Why the scoring has to be split
-------------------------------
``evaluate_betting`` counts every row as one independent bet, and
``wilson_interval`` builds its confidence interval from that count. Handed the
pooled predictions, one game contributes several rows, so the evaluator
reports several bets for what is largely one game bet one way and counted
repeatedly. Within a game the snapshot lines differ by much less than the
typical outcome noise, so those rows are heavily correlated -- they share one
outcome.

The inflation is real but its exact size is NOT ``sqrt(N)`` for N snapshots.
That figure would hold only if every snapshot were present and produced an
identical decision. In practice horizons are missing (T=720 covers 97.6% of
games, not 100%), the
min-edge filter selects different subsets at different snapshots, and two
snapshots can even take opposite sides once the line has moved across the
model's estimate. The direction is certain and the magnitude is not, which is
why this module suppresses the pooled row's interval and significance verdict
entirely rather than printing a corrected one: an honest interval there needs
game-clustered inference, which is not implemented here.

Grouping by snapshot fixes it arithmetically: within one ``TIME_TO_MATCH_MIN``
there is exactly one row per game, so the rows really are independent events and
the binomial maths is correct again.

It also matches how the bet is actually placed. Nobody bets the same game once
per configured snapshot; they bet it once, at some hour before tip. "What is
my win rate if I
always bet 12 hours out?" is a strategy someone can run, and it is exactly the
``720`` group. A single pooled number measures a strategy nobody runs.

What this module does NOT do
----------------------------
It fits nothing. Every number here is a regrouping of predictions the pipeline
has already produced under its own walk-forward discipline -- CV folds train
strictly before their validation dates, and the daily holdout retrains once per
game-day on games played strictly before it. Re-scoring cannot introduce
look-ahead because it never trains. The decisions themselves (which side, and
whether the bet clears the edge threshold) are taken verbatim from the pooled
run, so slicing changes the grouping and nothing else.

Nothing in ``training_pipeline`` is modified by this module. It reads the
existing result object, reuses ``evaluate_betting`` unchanged, and returns
tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training_pipeline.betting import (
    OUTCOME_COLUMN,
    BettingMetrics,
    evaluate_betting,
    outcome_from_predictions,
)
from training_pipeline.config import ExperimentConfig
from training_pipeline.data import SNAPSHOT_COLUMN as _SNAPSHOT_COLUMN
from training_pipeline.decisions import collect_prices, primary_threshold

#: Column carrying minutes-before-tip in the intermediate-line dataset.
#: Re-exported from training_pipeline.data, which is also where cleaning reads
#: it to report per-horizon row retention -- one spelling, so the scorer and the
#: cleaner cannot end up grouping by different columns.
SNAPSHOT_COLUMN = _SNAPSHOT_COLUMN

#: Label for the pooled row. Kept for reference only: its ``n_bets`` counts one
#: game once per snapshot, so its interval and significance verdict are
#: suppressed rather than reported (see ``_metrics_to_row``). The column
#: ``n_snapshots`` next to it says how many snapshots went into each row.
POOLED_LABEL = "ALL"


class SnapshotAlignmentError(RuntimeError):
    """Raised when predictions cannot be mapped back to their source rows.

    Always a bug rather than a data condition: it means a positional index no
    longer indexes what it used to. Loud, because a silent misalignment would
    attribute each prediction to the wrong snapshot and every number below it
    would be quietly wrong.
    """


def _metrics_to_row(
    label: object,
    metrics: BettingMetrics,
    *,
    n_rows: int,
    n_snapshots: int,
    n_games: int | None,
) -> dict[str, object]:
    """One output row.

    ``n_snapshots`` is the count of distinct ``TIME_TO_MATCH_MIN`` values feeding
    this row, and it is the readable trustworthiness flag: 1 means every row is
    a different game, so ``n_bets`` really is a count of independent events.
    Anything above 1 means the same game is counted once per snapshot.

    It replaces an earlier ``rows_per_game`` column that could never be
    populated. That column needed GAME_ID, which ``advanced_column_cleaning``
    drops (its name contains "_ID") and which ``_required_keep_columns`` does
    not protect -- so it was blank in every real report while the docs told the
    reader to filter on it. Worse, "just keep GAME_ID" is not the fix: the
    default ``exclude_cols`` is only ["TOTAL_POINTS", "SEASON_YEAR",
    "GAME_DATE"], so a surviving GAME_ID would land in the feature matrix as a
    string that monotonically encodes date and sequence. It is dropped for good
    reason.

    ``n_games`` is therefore populated only when a caller genuinely has game
    identifiers (tests, or a caller working from the raw CSV). It is None in the
    normal pipeline path, and nothing in the output depends on it.

    ``is_significant`` is suppressed on any row with more than one snapshot:
    the binomial interval behind it assumes independent trials, and correlated
    per-game repeats break that assumption in the anti-conservative direction.
    Reporting a significance verdict there would be asserting something this
    module cannot support without game-clustered inference.
    """
    pooled_rows = n_snapshots > 1
    return {
        "snapshot": label,
        "n_rows": n_rows,
        "n_snapshots": n_snapshots,
        "n_games": n_games,
        "n_candidates": metrics.n_candidates,
        "n_bets": metrics.n_bets,
        "bet_rate": metrics.bet_rate,
        "n_wins": metrics.n_wins,
        "n_losses": metrics.n_losses,
        "n_pushes": metrics.n_pushes,
        "win_rate": metrics.win_rate,
        # Interval and significance are only meaningful on independent trials.
        "win_rate_ci_low": None if pooled_rows else metrics.win_rate_ci_low,
        "win_rate_ci_high": None if pooled_rows else metrics.win_rate_ci_high,
        "roi": metrics.roi,
        "profit_units": metrics.profit_units,
        "break_even_rate": metrics.break_even_rate,
        "edge_vs_break_even": metrics.edge_vs_break_even,
        "beats_break_even": metrics.beats_break_even,
        "is_significant": None if pooled_rows else metrics.is_significant,
    }


def _unique_game_count(
    game_id: pd.Series | np.ndarray | None, mask: np.ndarray
) -> int | None:
    if game_id is None:
        return None
    return int(pd.Series(np.asarray(game_id)[mask]).nunique())


def score_by_snapshot(
    *,
    snapshot: pd.Series | np.ndarray,
    predicted_edge: np.ndarray,
    actual_total: np.ndarray,
    line: np.ndarray,
    selection_score: np.ndarray | None = None,
    decimal_odds_over: np.ndarray | None = None,
    decimal_odds_under: np.ndarray | None = None,
    min_edge: float = 0.0,
    flat_decimal_odds: float,
    game_id: pd.Series | np.ndarray | None = None,
    include_pooled: bool = True,
) -> pd.DataFrame:
    """One row of betting metrics per snapshot, plus an optional pooled row.

    Every argument is an already-computed array from a finished run. The
    per-snapshot call is the *same* ``evaluate_betting`` the pipeline uses, with
    the same edge threshold and the same prices -- only the rows handed to it
    differ.

    ``game_id`` only fills the informational ``n_games`` column. It is None on
    the normal pipeline path, because GAME_ID does not survive cleaning and
    must not: nothing in the output depends on it. The trustworthiness flag is
    ``n_snapshots``, which is derived from the snapshot values themselves and is
    therefore always available.
    """
    snapshot_values = pd.Series(np.asarray(snapshot)).reset_index(drop=True)
    if len(snapshot_values) != len(predicted_edge):
        raise SnapshotAlignmentError(
            f"snapshot has {len(snapshot_values)} values but predicted_edge has "
            f"{len(predicted_edge)}; they must be row-aligned."
        )

    def _slice(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
        return None if values is None else np.asarray(values)[mask]

    rows: list[dict[str, object]] = []
    for label in sorted(snapshot_values.dropna().unique()):
        mask = (snapshot_values == label).to_numpy()
        metrics = evaluate_betting(
            predicted_edge=np.asarray(predicted_edge)[mask],
            actual_total=np.asarray(actual_total)[mask],
            line=np.asarray(line)[mask],
            min_edge=min_edge,
            flat_decimal_odds=flat_decimal_odds,
            decimal_odds_over=_slice(decimal_odds_over, mask),
            decimal_odds_under=_slice(decimal_odds_under, mask),
            selection_score=_slice(selection_score, mask),
        )
        rows.append(
            _metrics_to_row(
                label,
                metrics,
                n_rows=int(mask.sum()),
                n_snapshots=1,
                n_games=_unique_game_count(game_id, mask),
            )
        )

    if include_pooled:
        everything = np.ones(len(snapshot_values), dtype=bool)
        pooled = evaluate_betting(
            predicted_edge=np.asarray(predicted_edge),
            actual_total=np.asarray(actual_total),
            line=np.asarray(line),
            min_edge=min_edge,
            flat_decimal_odds=flat_decimal_odds,
            decimal_odds_over=decimal_odds_over,
            decimal_odds_under=decimal_odds_under,
            selection_score=selection_score,
        )
        rows.append(
            _metrics_to_row(
                POOLED_LABEL,
                pooled,
                n_rows=len(snapshot_values),
                n_snapshots=int(snapshot_values.dropna().nunique()),
                n_games=_unique_game_count(game_id, everything),
            )
        )

    return pd.DataFrame(rows)


def _lookup_by_position(
    source: pd.DataFrame, positions: np.ndarray, column: str
) -> np.ndarray:
    """Values of ``column`` for positional indices into ``source``.

    The saved prediction frames carry a positional index back into the frame
    they were produced from (``row_in_dev`` for CV folds, ``row_in_test_final``
    for the daily walk-forward). That index is how a prediction is reunited with
    its snapshot without changing any existing code to carry the snapshot along.
    """
    if column not in source.columns:
        raise SnapshotAlignmentError(
            f"Column {column!r} is missing from the source frame, so predictions "
            "cannot be grouped by snapshot. The intermediate-line dataset must "
            "be built with the snapshot column present."
        )
    if positions.size and (positions.max() >= len(source) or positions.min() < 0):
        raise SnapshotAlignmentError(
            f"Positional index out of range for a frame of {len(source)} rows "
            f"(min {positions.min()}, max {positions.max()})."
        )
    return source[column].to_numpy()[positions]


#: Date column names the prediction frames actually use. The CV frame is built
#: with ``config.data.date_col`` ("GAME_DATE"), but the daily walk-forward frame
#: is built inside nba_ou.modeling and names its column "date". An earlier
#: version of this check looked only for the configured name and RETURNED
#: SILENTLY when it was absent, which meant the holdout -- the more fragile of
#: the two joins -- was never verified at all. Never return quietly from a
#: safety check: if it cannot run, that is a failure, not a pass.
_DATE_COLUMN_CANDIDATES = ("GAME_DATE", "date")


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """First candidate present in ``frame``.

    Resolved per frame, not shared: the daily walk-forward's prediction frame
    calls its date column "date" while ``df_test`` calls it "GAME_DATE", so the
    two NEVER agree on a name. Requiring a shared name made the check
    unsatisfiable on exactly the join it most needed to verify.
    """
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _verify_alignment(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    positions: np.ndarray,
    *,
    date_col: str,
    #: Defaults to the totals outcome, which is what this function assumed
    #: before the spread market existed. Both pipeline call sites pass it
    #: explicitly; the default exists so a direct caller checking only the date
    #: alignment (as the tests do) need not know about markets.
    outcome_col: str = "TOTAL_POINTS",
) -> None:
    """Confirm the positional index still points where it is assumed to.

    Compares every value the two frames hold in common -- the date, and where
    available the realised total and the settlement line. A mismatch means the
    frame was reordered or re-indexed somewhere, and every snapshot attribution
    built on it would be silently wrong, so this raises rather than returning
    suspect numbers.

    Why more than the date: a date-only check passes any permutation WITHIN a
    day, and a day here holds several games at several snapshots each. Adding
    the total and the line makes an undetected swap require two rows that agree
    on date, final score and line at once.
    """
    candidates = (date_col, *_DATE_COLUMN_CANDIDATES)
    predictions_date = _resolve_column(predictions, candidates)
    source_date = _resolve_column(source, candidates)
    if predictions_date is None or source_date is None:
        raise SnapshotAlignmentError(
            "Cannot verify the prediction-to-source join: no date column from "
            f"{sorted(set(candidates))} found in "
            + ("predictions" if predictions_date is None else "the source frame")
            + f" (predictions have {sorted(predictions.columns)[:8]}...). "
            "Refusing to report per-snapshot metrics from an unverified join."
        )

    expected_dates = pd.to_datetime(
        pd.Series(_lookup_by_position(source, positions, source_date))
    ).reset_index(drop=True)
    actual_dates = pd.to_datetime(
        predictions[predictions_date].reset_index(drop=True)
    ).reset_index(drop=True)
    if not expected_dates.equals(actual_dates):
        n_bad = int((expected_dates != actual_dates).sum())
        raise SnapshotAlignmentError(
            f"{n_bad} of {len(actual_dates)} predictions do not line up on date "
            f"({predictions_date!r} vs {source_date!r}) with the rows their "
            "positional index points at."
        )

    # Value columns that pin down the individual game, not merely its day.
    #
    # The outcome is spelled differently on each side: predictions carry it as
    # ``actual_outcome`` (or ``TOTAL_POINTS`` on an archived run), while the
    # source frame carries it under the market's own name -- TOTAL_POINTS or
    # HOME_MARGIN. Pairing them explicitly is what keeps this check from
    # degrading into the date-only check it exists to strengthen.
    pairs: tuple[tuple[str, str], ...] = (
        (OUTCOME_COLUMN, outcome_col),
        ("TOTAL_POINTS", outcome_col),
        ("target_line", "target_line"),
    )
    checked_outcome = False
    for column, source_column in pairs:
        if column not in predictions.columns or source_column not in source.columns:
            continue
        if source_column == outcome_col:
            if checked_outcome:
                continue
            checked_outcome = True
        expected = pd.to_numeric(
            pd.Series(_lookup_by_position(source, positions, source_column)),
            errors="coerce",
        ).reset_index(drop=True)
        actual = pd.to_numeric(
            predictions[column].reset_index(drop=True), errors="coerce"
        ).reset_index(drop=True)
        mismatch = ~np.isclose(
            expected.to_numpy(dtype=float),
            actual.to_numpy(dtype=float),
            equal_nan=True,
        )
        if mismatch.any():
            raise SnapshotAlignmentError(
                f"{int(mismatch.sum())} of {len(actual)} predictions disagree "
                f"with the source frame on {column!r}, so the positional join "
                "is attributing predictions to the wrong rows."
            )


def holdout_snapshot_metrics(
    result: object, *, snapshot_col: str = SNAPSHOT_COLUMN
) -> pd.DataFrame | None:
    """Per-snapshot metrics over the held-out test period.

    Handles the default ``daily_walk_forward`` holdout, whose predictions carry
    ``row_in_test_final``. Returns None for a ``single_shot`` run, whose saved
    frame has no positional index to join on.
    """
    walk_forward = getattr(result, "walk_forward_result", None)
    if walk_forward is None:
        return None

    config: ExperimentConfig = result.config  # type: ignore[attr-defined]
    df_test: pd.DataFrame = result.df_test  # type: ignore[attr-defined]
    predictions = walk_forward.predictions
    positions = predictions["row_in_test_final"].to_numpy()

    _verify_alignment(
        predictions,
        df_test,
        positions,
        date_col=config.data.date_col,
        outcome_col=config.outcome_col,
    )

    over_prices, under_prices = collect_prices(df_test, config, positions=positions)
    return score_by_snapshot(
        snapshot=_lookup_by_position(df_test, positions, snapshot_col),
        predicted_edge=predictions["predicted_edge"].to_numpy(dtype=float),
        actual_total=outcome_from_predictions(predictions),
        line=predictions["target_line"].to_numpy(dtype=float),
        selection_score=predictions["selection_score"].to_numpy(dtype=float),
        decimal_odds_over=over_prices,
        decimal_odds_under=under_prices,
        min_edge=primary_threshold(config),
        flat_decimal_odds=config.betting.flat_decimal_odds,
        game_id=(
            _lookup_by_position(df_test, positions, config.data.game_id_col)
            if config.data.game_id_col in df_test.columns
            else None
        ),
    )


def cv_snapshot_metrics(
    result: object, *, snapshot_col: str = SNAPSHOT_COLUMN
) -> pd.DataFrame | None:
    """Per-snapshot metrics pooled over the CV validation folds.

    Several times the bet volume of the holdout, which is the binding
    constraint on telling a real edge from a lucky one at these sample sizes.
    Read it to compare configurations, not as a live ROI estimate: it is
    biased by the hyperparameter selection that produced it, exactly as the
    pooled CV betting metrics already are.
    """
    cv_betting = getattr(result, "cv_betting", None)
    if cv_betting is None:
        return None

    config: ExperimentConfig = result.config  # type: ignore[attr-defined]
    df_dev: pd.DataFrame = result.df_dev  # type: ignore[attr-defined]
    predictions = cv_betting.predictions
    positions = predictions["row_in_dev"].to_numpy()

    _verify_alignment(
        predictions,
        df_dev,
        positions,
        date_col=config.data.date_col,
        outcome_col=config.outcome_col,
    )

    over_prices, under_prices = collect_prices(df_dev, config, positions=positions)
    return score_by_snapshot(
        snapshot=_lookup_by_position(df_dev, positions, snapshot_col),
        predicted_edge=predictions["predicted_edge"].to_numpy(dtype=float),
        actual_total=outcome_from_predictions(predictions),
        line=predictions["target_line"].to_numpy(dtype=float),
        selection_score=predictions["selection_score"].to_numpy(dtype=float),
        decimal_odds_over=over_prices,
        decimal_odds_under=under_prices,
        min_edge=primary_threshold(config),
        flat_decimal_odds=config.betting.flat_decimal_odds,
        game_id=(
            _lookup_by_position(df_dev, positions, config.data.game_id_col)
            if config.data.game_id_col in df_dev.columns
            else None
        ),
    )


def build_snapshot_report(
    result: object, *, snapshot_col: str = SNAPSHOT_COLUMN
) -> dict[str, pd.DataFrame]:
    """Both tables, keyed ``"cv"`` and ``"holdout"``. Missing ones are omitted."""
    report: dict[str, pd.DataFrame] = {}
    cv_table = cv_snapshot_metrics(result, snapshot_col=snapshot_col)
    if cv_table is not None and not cv_table.empty:
        report["cv"] = cv_table
    holdout_table = holdout_snapshot_metrics(result, snapshot_col=snapshot_col)
    if holdout_table is not None and not holdout_table.empty:
        report["holdout"] = holdout_table
    return report


def save_snapshot_report(
    report: dict[str, pd.DataFrame], run_dir: str | Path
) -> dict[str, Path]:
    """Write each table into ``run_dir`` as ``snapshot_<name>_metrics.csv``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in report.items():
        path = run_dir / f"snapshot_{name}_metrics.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def format_snapshot_table(table: pd.DataFrame) -> str:
    """Compact console rendering, one line per snapshot."""
    columns = [
        "snapshot",
        "n_rows",
        "n_snapshots",
        "n_bets",
        "win_rate",
        "win_rate_ci_low",
        "win_rate_ci_high",
        "roi",
        "is_significant",
    ]
    present = [column for column in columns if column in table.columns]
    display = table[present].copy()
    for column in ("win_rate", "win_rate_ci_low", "win_rate_ci_high", "roi"):
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "-" if pd.isna(value) else f"{value:.4f}"
            )
    return display.to_string(index=False)
