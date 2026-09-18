"""Line-movement features, each computable from ticks at or before the snapshot.

Every quantity here is derived only from ticks with
``minutes_before_tip >= snapshot_minutes``, so nothing can encode a price the
bettor could not have seen. The windowed moves reuse the snapshot machinery
itself: "how far has the line moved in the last hour, as of T" is exactly
``line(T) - line(T + 60)``, which is a second as-of read rather than a separate
code path with its own leakage surface.

Two distinct movement counts are produced, because they answer different
questions:

* ``n_moves_so_far`` -- changes between the opener and T. It grows as T
  approaches tip, so it is partly a proxy for elapsed time; ``n_moves_per_hour``
  is emitted beside it so a model can separate "busy market" from "more hours
  elapsed".
* the open-to-close count for *previous* games, which is a different feature
  entirely and lives in the historical block (see ``history_features.py``).

Missing windows are flagged rather than left as bare NaN. Long horizons are
systematically the ones whose history does not reach back far enough, so bare
NaNs would let ``cleaning.max_na_per_row`` delete precisely the 8h and 12h rows
this dataset exists to compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_ou.postgre_db.line_history_aiven.fetch import MARKET_MONEYLINE

from .snapshots import (
    DEFAULT_SNAPSHOT_GRID,
    GROUP_KEYS,
    build_snapshot_panel,
    market_level,
    market_up_probability,
    resolve_line,
)

#: Look-back windows, in minutes before the snapshot.
#:
#: The short end (15, 30) is where late sharp money lands and is the reason the
#: 30-minute snapshot previously had no window shorter than itself. It is also
#: sparse: measured on the store a given book moves in the trailing 60 minutes
#: on only 32% of (row, book), and 37% of rows have no book moving at all -- so
#: expect ``move_last_15`` to be zero far more often than not. That is a real
#: reading of a quiet market, not missing data, and ``has_window_15`` separates
#: the two.
#:
#: Cost scales with ``len(grid) * len(windows)``, not with either alone: every
#: pair needs its own as-of read at ``T + w`` (see ``extended_grid``).
DEFAULT_WINDOWS: tuple[int, ...] = (15, 30, 60, 120, 180, 360, 720)

#: Value used where a window's history does not reach back far enough. Paired
#: with a ``HAS_`` flag so the model can tell "no movement" from "no history",
#: and so the row does not accumulate NaNs (see module docstring).
MISSING_SENTINEL = 0.0

_PANEL_KEYS = [*GROUP_KEYS, "snapshot_minutes"]


def extended_grid(
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> tuple[int, ...]:
    """Horizons needed to answer every windowed question on ``grid``."""
    horizons = set(grid) | {t + w for t in grid for w in windows}
    return tuple(sorted(horizons))


def prepare_tick_history(ticks: pd.DataFrame) -> pd.DataFrame:
    """Annotate each tick with its change relative to the previous one.

    Computed once over the full history. Filtering to a snapshot later keeps a
    chronological *prefix*, and a prefix never changes the predecessor of any
    row it retains -- so these columns stay valid under every horizon.
    """
    working = ticks.copy()
    working["raw_line"] = resolve_line(working)

    from .normalization import devig_two_way

    fair = devig_two_way(working["left_price"], working["right_price"])
    working["fair_left"] = fair["fair_left"]
    working["fair_right"] = fair["fair_right"]
    working["overround"] = fair["overround"]
    # Aligned with ``level``: the side that wins when the level goes up. Using
    # fair_left for every market reversed the sign against the level on spread
    # and moneyline.
    working["fair_up"] = market_up_probability(working)

    # One canonical level per market -- see ``snapshots.market_level``. Using
    # the resolved line here instead would leave the moneyline with no level at
    # all and silently zero every count below.
    working["line"] = market_level(working)

    working = working.sort_values(
        [*GROUP_KEYS, "minutes_before_tip"], ascending=[True, True, True, False]
    ).reset_index(drop=True)

    grouped = working.groupby(GROUP_KEYS, sort=False)
    previous_line = grouped["line"].shift()
    previous_fair = grouped["fair_up"].shift()

    working["line_delta"] = working["line"] - previous_line
    working["fair_delta"] = working["fair_up"] - previous_fair

    # A "move" is a change in LEVEL. A tick that only re-prices the same number
    # is counted separately -- books routinely price a half-tick before taking
    # it, and that pressure is invisible in the line itself.
    working["is_move"] = working["line_delta"].notna() & working["line_delta"].ne(0.0)
    working["is_price_only"] = (
        working["line_delta"].notna()
        & working["line_delta"].eq(0.0)
        & working["fair_delta"].notna()
        & working["fair_delta"].ne(0.0)
    )
    # On the moneyline the level *is* the price, so the two categories collapse:
    # every re-price is a move and "price-only" cannot occur by construction.
    # Left as a structural zero rather than dropped, so the column stays
    # comparable across markets.
    working.loc[working["market"].eq(MARKET_MONEYLINE), "is_price_only"] = False
    working["move_sign"] = np.sign(working["line_delta"].fillna(0.0))
    return working


def _first_nonzero_sign(signs: pd.Series) -> float:
    """Direction of the first actual move, ignoring the opener's NaN delta."""
    nonzero = signs[signs.ne(0.0) & signs.notna()]
    return float(nonzero.iloc[0]) if len(nonzero) else 0.0


def _first_nonzero_signs(eligible: pd.DataFrame) -> pd.Series:
    """``_first_nonzero_sign`` for every series at once."""
    signs = eligible["move_sign"]
    nonzero = signs.where(signs.ne(0.0) & signs.notna())
    # ``first`` skips NaN, so this is each series' first non-zero sign.
    first = nonzero.groupby([eligible[key] for key in GROUP_KEYS], sort=False).first()
    return first.fillna(0.0)


def _abs_sums(eligible: pd.DataFrame, grouped, column: str) -> pd.Series | None:
    """``lambda s: s.abs().sum()`` for every series, summed exactly as it was.

    That lambda reaches ``np.sum`` over the series' values with NaN set to 0,
    in row order. ``ndarray.sum`` over the same contiguous slice runs the
    identical reduction, so every total matches to the bit, without a pandas
    call per series. Returns None when the series are not contiguous runs of
    rows in group order; the caller then falls back to the lambda.
    """
    codes = grouped.ngroup().to_numpy()
    if len(codes) == 0:
        return None
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    if len(starts) != grouped.ngroups or not np.array_equal(
        codes[starts], np.arange(len(starts))
    ):
        return None
    values = np.abs(eligible[column].to_numpy(dtype="float64", copy=True))
    values[np.isnan(values)] = 0.0
    stops = np.r_[starts[1:], len(values)]
    totals = np.fromiter(
        (values[start:stop].sum() for start, stop in zip(starts, stops, strict=True)),
        dtype="float64",
        count=len(starts),
    )
    return pd.Series(totals, index=grouped.size().index)


def _reversal_counts(eligible: pd.DataFrame) -> pd.Series:
    """Number of times the direction of travel flipped."""
    moves = eligible[eligible["is_move"]]
    if moves.empty:
        return pd.Series(dtype="float64")
    previous_sign = moves.groupby(GROUP_KEYS, sort=False)["move_sign"].shift()
    flipped = previous_sign.notna() & moves["move_sign"].ne(previous_sign)
    return flipped.groupby([moves[key] for key in GROUP_KEYS], sort=False).sum()


def _history_aggregates(
    working: pd.DataFrame, snapshot_minutes: int
) -> pd.DataFrame:
    """Everything derivable from the tick history up to one snapshot."""
    eligible = working[working["minutes_before_tip"] >= snapshot_minutes]
    if eligible.empty:
        return pd.DataFrame()

    grouped = eligible.groupby(GROUP_KEYS, sort=False)

    aggregates = grouped.agg(
        n_ticks_total=("line_ts", "size"),
        n_moves_so_far=("is_move", "sum"),
        n_price_only_ticks=("is_price_only", "sum"),
        n_distinct_levels=("line", "nunique"),
        line_max_so_far=("line", "max"),
        line_min_so_far=("line", "min"),
        line_std_so_far=("line", "std"),
        first_minutes_before_tip=("minutes_before_tip", "max"),
    )
    # Both used to be per-series Python callables inside ``agg`` -- most of
    # this stage's run time. Same values, same column order.
    aggregates["first_move_direction"] = _first_nonzero_signs(eligible).reindex(
        aggregates.index
    )
    abs_move_total = _abs_sums(eligible, grouped, "line_delta")
    if abs_move_total is None:
        abs_move_total = grouped["line_delta"].agg(lambda s: s.abs().sum())
    aggregates["abs_move_total"] = abs_move_total.reindex(aggregates.index)

    # The opener proper when the scrape labelled one, else the earliest tick we
    # hold. In practice openers sit ~25h before tip, well outside every horizon
    # on the grid, so the two agree for all but a handful of series.
    labelled_opener = (
        eligible["line"]
        .where(eligible["is_opener"])
        .groupby([eligible[key] for key in GROUP_KEYS], sort=False)
        .first()
    )
    aggregates["opener_line"] = labelled_opener.reindex(aggregates.index).fillna(
        grouped["line"].first()
    )
    aggregates["opener_fair_left"] = grouped["fair_up"].first()

    reversals = _reversal_counts(eligible)
    aggregates["n_reversals"] = (
        reversals.reindex(aggregates.index).fillna(0.0) if not reversals.empty else 0.0
    )

    aggregates["snapshot_minutes"] = int(snapshot_minutes)
    return aggregates.reset_index()


def add_movement_features(
    panel: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    null_extreme_spread_prices: bool = True,
    progress: str | None = None,
) -> pd.DataFrame:
    """Attach movement features to a snapshot ``panel``.

    ``progress`` labels tqdm bars over the snapshot horizons; ``None`` runs
    silently.

    ``panel`` is the base grid; the windowed look-backs are answered from a
    second panel built on the extended grid, so both come from the same as-of
    implementation.
    """
    if not windows:
        raise ValueError("windows must not be empty.")
    if any(window <= 0 for window in windows):
        # A negative window would read the panel at (snapshot - |w|), i.e. a
        # LATER moment than the snapshot itself: a direct look-ahead that would
        # pass every column-name check downstream.
        raise ValueError(
            f"look-back windows must be positive minutes; got {windows}."
        )
    if panel.empty:
        return panel

    working = prepare_tick_history(ticks)

    aggregate_frames = [
        _history_aggregates(working, snapshot_minutes)
        for snapshot_minutes in tqdm(
            sorted(set(grid)),
            desc=progress and f"{progress}: history",
            unit="snapshot",
            disable=progress is None,
        )
    ]
    aggregate_frames = [frame for frame in aggregate_frames if not frame.empty]
    out = panel.copy()
    if aggregate_frames:
        out = out.merge(
            pd.concat(aggregate_frames, ignore_index=True),
            on=_PANEL_KEYS,
            how="left",
        )

    out = _add_open_to_now_features(out)
    out = _add_windowed_features(
        out,
        ticks,
        grid=grid,
        windows=windows,
        null_extreme_spread_prices=null_extreme_spread_prices,
        progress=progress and f"{progress}: windows",
    )
    out = _add_shape_features(out)
    return out


def _add_open_to_now_features(out: pd.DataFrame) -> pd.DataFrame:
    """Movement since the opener -- the longest look-back always available.

    Both ends come from ``level``. Measuring a centered "now" against a raw
    "then" is what put 946 spread rows outside their own realised range.
    """
    out["move_from_open"] = out["level"] - out["opener_line"]
    out["abs_move_from_open"] = out["move_from_open"].abs()

    # Percentage move is only meaningful against a non-zero base. A pick'em
    # spread sits at zero, so guard rather than emit an infinity.
    base = out["opener_line"].replace(0.0, np.nan)
    out["pct_move_from_open"] = out["move_from_open"] / base.abs()

    out["move_direction"] = np.sign(out["move_from_open"].fillna(0.0))
    out["minutes_since_open"] = (
        out["first_minutes_before_tip"] - out["snapshot_minutes"]
    )
    out["prob_move_from_open"] = out["fair_up"] - out["opener_fair_left"]
    return out


def _add_windowed_features(
    out: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    grid: tuple[int, ...],
    windows: tuple[int, ...],
    null_extreme_spread_prices: bool = True,
    progress: str | None = None,
) -> pd.DataFrame:
    """Moves over trailing windows, via a second as-of read."""
    lookup = build_snapshot_panel(
        ticks,
        grid=extended_grid(grid, windows),
        null_extreme_spread_prices=null_extreme_spread_prices,
        progress=progress,
    )
    if lookup.empty:
        return out

    lookup = lookup[[*_PANEL_KEYS, "level", "fair_up"]].rename(
        columns={"level": "line_then", "fair_up": "fair_then"}
    )

    for window in windows:
        shifted = lookup.copy()
        # Line as of (snapshot + window) is the state one window earlier.
        shifted["snapshot_minutes"] = shifted["snapshot_minutes"] - window
        merged = out[_PANEL_KEYS].merge(shifted, on=_PANEL_KEYS, how="left")

        has_history = merged["line_then"].notna().to_numpy()
        move = out["level"].to_numpy() - merged["line_then"].to_numpy()
        prob_move = out["fair_up"].to_numpy() - merged["fair_then"].to_numpy()

        out[f"has_window_{window}"] = has_history.astype(int)
        out[f"move_last_{window}"] = np.where(has_history, move, MISSING_SENTINEL)
        out[f"abs_move_last_{window}"] = np.abs(out[f"move_last_{window}"])
        out[f"velocity_last_{window}"] = out[f"move_last_{window}"] / (window / 60.0)
        out[f"prob_move_last_{window}"] = np.where(
            has_history, prob_move, MISSING_SENTINEL
        )

    # Acceleration: recent pace against the longer trend it sits inside.
    if 60 in windows and 180 in windows:
        out["move_acceleration"] = (
            out["move_last_60"] - out["move_last_180"] / 3.0
        )
    return out


def _add_shape_features(out: pd.DataFrame) -> pd.DataFrame:
    """Where the current line sits within the path it has travelled."""
    # A series with a single observation has undefined std but zero realised
    # dispersion, which is the honest reading and keeps the NaN count down.
    out["line_std_so_far"] = out["line_std_so_far"].fillna(0.0)
    out["line_range_so_far"] = out["line_max_so_far"] - out["line_min_so_far"]

    span = out["line_range_so_far"].replace(0.0, np.nan)
    out["position_in_range"] = (out["level"] - out["line_min_so_far"]) / span
    # A line that has never moved is at neither extreme; the midpoint is the
    # honest encoding, and the range itself already says it did not move.
    out["position_in_range"] = out["position_in_range"].fillna(0.5)

    elapsed_hours = (out["minutes_since_open"] / 60.0).replace(0.0, np.nan)
    out["n_moves_per_hour"] = out["n_moves_so_far"] / elapsed_hours

    # Direction of travel now versus the direction the market FIRST moved -- a
    # reversal after a strong initial move is a different state from a steady
    # drift of the same size. Previously this compared the last hour against the
    # NET opener-to-now direction, which is not the opening direction at all:
    # a line that moved up then further up scored the same as one that moved
    # down then back up past its open.
    recent = out.get("move_last_60")
    recent_direction = np.sign(recent) if recent is not None else out["move_direction"]
    out["opposes_opening_direction"] = (
        (recent_direction * out["first_move_direction"]) < 0
    ).astype(int)
    out["net_opposes_opening_direction"] = (
        (out["move_direction"] * out["first_move_direction"]) < 0
    ).astype(int)

    # A pick'em opens at zero, so the percentage move has no base. Structural,
    # not missing data -- and left as NaN it would inflate the row's NaN count.
    out["pct_move_from_open"] = out["pct_move_from_open"].fillna(MISSING_SENTINEL)

    # The moneyline no longer needs special-casing here: ``market_level`` gives
    # it a real level (devigged home probability), so every column above is
    # populated for it rather than structurally absent.
    return out
