"""Build the intermediate-line training dataset.

Grain: **one row per (game, pre-game snapshot)**, so a model can be trained to
bet at whatever time before tip-off it is actually betting, rather than only at
close.

This is a separate pipeline that *reuses* existing feature functions. It does
not modify, wrap or subclass ``create_df_to_predict``; both datasets can be
built independently and neither can change the other's output.

Scope: **historical training data only.** Serving same-day predictions is
deliberately not implemented here.

One naming decision deserves stating plainly, because it will otherwise be
misread. In this dataset ``ODDS_TOTAL_LINE_<book>`` holds that book's line **as
of the snapshot**, not at close. That is not a fudge: it is genuinely that
book's line, quoted at the moment this model bets. It is done this way because
the training pipeline derives ``line_error`` against the main book column and
settles bets against the same one, so target and settlement must agree. The
actual closing line is carried separately as ``ODDS_CLOSING_TOTAL_LINE_<book>``
and is excluded from features -- it exists only to measure closing-line value.

Every odds-derived column in this dataset -- snapshot (``ODDS_SNAP_*``),
closing/opener (``ODDS_TOTAL_LINE_*`` / ``ODDS_CLOSING_*``), and history-dynamics
(``ODDS_TOT_*`` / ``ODDS_SPR_*`` / ``ODDS_ML_*``) -- carries the same leading
``ODDS_`` marker, so odds features can be selected as a group just as
leakage-safe columns are selected via ``_BEFORE``.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nba_ou.config.market_columns import (
    HOME_MARGIN_COL,
    PTS_AWAY_COL,
    PTS_HOME_COL,
    SPREAD_ERROR_COL,
    home_margin,
    spread_error,
    spread_line_home_from_implied_margin,
)
from nba_ou.config.odds_columns import (
    HISTORY_ONLY_BOOKS,
    get_main_book,
    spread_line_home_col,
    total_line_col,
)
from nba_ou.create_training_data import intermediate_history
from nba_ou.create_training_data.create_base_game_features import (
    create_base_game_features,
)
from nba_ou.create_training_data.historical_ridge_movement import (
    EXPECTED_MOVE_COLUMNS,
    add_historical_ridge_movement,
    current_line_column,
    ridge_input_columns,
)
from nba_ou.create_training_data.history_memo import season_of_game_id
from nba_ou.create_training_data.intermediate_injuries import (
    add_snapshot_injury_features,
)
from nba_ou.create_training_data.intermediate_market_dynamics import (
    attach_market_dynamics,
    intermediate_market_dynamics_columns,
)
from nba_ou.create_training_data.intermediate_referees import (
    add_intermediate_referee_features,
    add_snapshot_referee_interactions,
    mask_referees_before_release,
)
from nba_ou.create_training_data.predict_data_utils import normalize_game_ids
from nba_ou.create_training_data.select_intermediate_columns import (
    assert_no_bare_closing_odds,
    audit_closing_line_reconstruction,
    select_intermediate_training_columns,
)
from nba_ou.data_processing.line_history.anchor_total_path import (
    add_anchor_total_path_features,
)
from nba_ou.data_processing.line_history.book_merge import (
    CAESARS_BOOK,
    merge_caesars_into_fanatics_ticks,
)
from nba_ou.data_processing.line_history.cross_book import (
    add_book_deviation,
    aggregate_across_books,
)
from nba_ou.data_processing.line_history.history_features import (
    add_prior_game_line_dynamics,
    build_game_line_dynamics,
)
from nba_ou.data_processing.line_history.market_dynamics import ORDER_COLUMN
from nba_ou.data_processing.line_history.movement_features import (
    DEFAULT_WINDOWS,
    add_movement_features,
)
from nba_ou.data_processing.line_history.snapshots import (
    DEFAULT_SNAPSHOT_GRID,
    build_snapshot_panel,
)
from nba_ou.data_processing.odds.book_combination import resolve_combine_books
from nba_ou.data_processing.referees.referee_tendencies import (
    DEFAULT_REFEREE_HISTORY_SEASONS,
)
from nba_ou.fetch_data.live_rosters import load_live_rosters
from nba_ou.postgre_db.input_cache import input_cache_enabled
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
    PARTIAL_COVERAGE_BOOKS,
    available_seasons,
    fetch_games,
    fetch_pregame_ticks,
    pregame_tick_fingerprints,
)
from nba_ou.postgre_db.line_history_aiven.live import LiveLineHistory

if TYPE_CHECKING:
    # Tonight's inputs only: its import closure reaches the injury-report PDF
    # parser (pymupdf), which the retrain environment does not install.
    from nba_ou.create_training_data.live_game_day import LiveGameDay

MARKET_SHORT = {MARKET_TOTALS: "TOT", MARKET_SPREAD: "SPR", MARKET_MONEYLINE: "ML"}

#: Seasons of team history loaded *before* the first line-history season, purely
#: so the rolling features of that first season are not computed from nothing.
#:
#: The line-history store starts at 2021-22; the NBA database goes back further.
#: Without this the two were pinned together -- ``season_start_date`` was derived
#: from ``min(season_years)`` -- so the earliest line-history season began with
#: every rolling window empty, every ``_SEASON_BEFORE_AVG`` unfilled and every
#: trend slope on its no-history value. Passing older years in ``season_years``
#: cannot fix that: they are intersected against the store and dropped.
#:
#: These seasons never become rows. The snapshot join is an inner merge on
#: ``GAME_ID``, so a game with no ticks cannot survive it -- exactly the pattern
#: ``create_base_game_features`` already uses for ``player_context_seasons``.
#:
#: Two rather than one: one satisfies the previous-season fallbacks
#: (``_SEASON_BEFORE_AVG``, the trend-slope chain, roster continuity's window
#: opening the preceding March), two leaves margin for the 20-game rollups to be
#: warm rather than merely defined.
#:
#: Two is also the most that buys anything, which is a coincidence worth
#: recording rather than rediscovering: the line-history store starts at 2021-22,
#: and 2021 - 2 = 2019 is exactly ``FIRST_SEASON_WITH_CLOSING_ODDS``.
DEFAULT_BASE_LOOKBACK_SEASONS = 2

#: First season the closing-odds tables actually cover, measured rather than
#: assumed.
#:
#: ``create_base_game_features`` defaults its floor to 2017-10-01, which the odds
#: data does not honour. Measured on ``odds_sportsbook``: season 2018 holds 32
#: rows, all dated 2019-05-03 to 2019-06-13 -- a fragment of one postseason, not
#: a season -- and nothing exists before it. 2019-20 (1,137 games) and 2020-21
#: (1,176) are complete despite the low counts; both were COVID-shortened.
#: ``odds_yahoo`` starts a season later still, at 2020.
#:
#: This is the cliff behind the earlier finding that seasons 2017-2020 were
#: discarded wholesale by ``cleaning.max_na_per_row``: those rows carry no odds
#: columns at all, not merely cold rolling windows. Per-book coverage from 2019
#: is uneven too -- betmgm is at 0% in 2020 and 37.5% in 2019 against ~100% from
#: 2022 -- so a book's mere presence remains partly a season indicator this far
#: back, which is what ``PARTIAL_COVERAGE_BOOKS`` and the Caesars/fanatics
#: combination exist to handle for the recent end.
FIRST_SEASON_WITH_CLOSING_ODDS = 2019

#: The consensus OPENING line. Safe at every snapshot -- openers land a median
#: ~25h before tip, well outside the 12h grid -- and it is the baseline
#: ``betting.comparison_line_cols`` in experiments/_base.yaml is configured
#: against, so it must survive under exactly this name.
OPENER_LINE_COLUMN = total_line_col("consensus_opener")

#: Emitted for EVERY book, not just the anchor. The anchor is special only in
#: that its line defines the target; there is no reason its odds should be the
#: only ones exported in full. Previously the other four books carried a reduced
#: nine-field summary with no raw prices and no normalised line, so "all five
#: books' odds are in the dataset" was not literally true.
FULL_BOOK_FEATURES: tuple[str, ...] = (
    # --- raw quote, exactly as the book published it -------------------
    "raw_line",
    "price_left",
    "price_right",
    "has_quote",
    "line_age_minutes",
    # --- derived pricing ------------------------------------------------
    "norm_line",
    "norm_minus_raw",
    "level",
    "fair_left",
    "fair_right",
    "fair_up",
    "overround",
    # --- path so far ----------------------------------------------------
    "n_ticks_total",
    "n_moves_so_far",
    "n_price_only_ticks",
    "n_distinct_levels",
    "line_std_so_far",
    "opener_line",
    "n_reversals",
    "move_from_open",
    "abs_move_from_open",
    "pct_move_from_open",
    "move_direction",
    "first_move_direction",
    "minutes_since_open",
    "prob_move_from_open",
    "line_range_so_far",
    "position_in_range",
    "n_moves_per_hour",
    "opposes_opening_direction",
    "net_opposes_opening_direction",
    "move_acceleration",
    # --- position against the rest of the market ------------------------
    "deviation_from_consensus",
    "abs_deviation_from_consensus",
    "deviation_z",
    "is_outlier_book",
)

CONSENSUS_FEATURES: tuple[str, ...] = (
    "consensus_line",
    "consensus_norm_line",
    "consensus_raw_line",
    "consensus_fair_left",
    "consensus_overround",
    "crossbook_std",
    "crossbook_range",
    "n_books_quoting",
    "median_line_age",
    "max_line_age",
    "consensus_move_from_open",
    "consensus_move_recent",
    "consensus_has_quote",
    "steam_movers",
    "steam_agreement",
    "consensus_n_moves",
    "consensus_opener_line",
    "steam_books_up",
    "steam_books_down",
    "steam_net",
    "steam_fraction",
)


def _stage_logger(verbose: bool) -> Callable[[str], None]:
    """Print ``[mm:ss] message`` stage markers, timed from the logger's creation."""
    started = time.perf_counter()

    def stage(message: str) -> None:
        if verbose:
            minutes, seconds = divmod(int(time.perf_counter() - started), 60)
            print(f"[{minutes:02d}:{seconds:02d}] {message}", flush=True)

    return stage


def _windowed_feature_names(windows: tuple[int, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for window in windows:
        names.extend(
            [
                f"has_window_{window}",
                f"move_last_{window}",
                f"abs_move_last_{window}",
                f"velocity_last_{window}",
                f"prob_move_last_{window}",
            ]
        )
    return tuple(names)


def _pivot_book_features(
    panel: pd.DataFrame, features: tuple[str, ...], books: list[str]
) -> pd.DataFrame:
    """(game, snapshot) x (market, book, feature) -> wide columns."""
    subset = panel[panel["book"].isin(books)]
    present = [feature for feature in features if feature in subset.columns]
    if subset.empty or not present:
        return pd.DataFrame()

    indexed = subset.set_index(["game_id", "snapshot_minutes", "market", "book"])[
        present
    ]
    wide = indexed.unstack(["market", "book"])
    wide.columns = [
        f"ODDS_SNAP_{MARKET_SHORT.get(market, market.upper())}_{book.upper()}_"
        f"{feature.upper()}"
        for feature, market, book in wide.columns
    ]
    return wide.reset_index()


def _pivot_consensus(consensus: pd.DataFrame) -> pd.DataFrame:
    present = [f for f in CONSENSUS_FEATURES if f in consensus.columns]
    indexed = consensus.set_index(["game_id", "snapshot_minutes", "market"])[present]
    wide = indexed.unstack("market")
    wide.columns = [
        f"ODDS_SNAP_{MARKET_SHORT.get(market, market.upper())}_CONSENSUS_"
        f"{feature.upper().replace('CONSENSUS_', '')}"
        for feature, market in wide.columns
    ]
    return wide.reset_index()


SNAPSHOT_WIDE_KEYS = ["game_id", "snapshot_minutes"]


def _merge_wide_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Join the wide blocks on (game, snapshot), refusing to duplicate a column.

    A plain chain of merges is silent about overlap: it renames the collision to
    ``<name>_x`` / ``<name>_y`` and keeps both. That is exactly how the consensus
    block came to be pivoted and merged twice, putting 63 pairs of bit-identical
    columns into the dataset -- invisible in every column-name check, but they
    inflate the feature count, split permutation importance between twins and
    count double against ``cleaning.max_na_per_row``.

    Overlap between these blocks is always a construction error (each block owns
    a disjoint name space by design), so it raises rather than resolving.
    """
    populated = [part for part in parts if not part.empty]
    if not populated:
        return pd.DataFrame(columns=SNAPSHOT_WIDE_KEYS)

    merged = populated[0]
    for part in populated[1:]:
        overlap = (set(merged.columns) & set(part.columns)) - set(SNAPSHOT_WIDE_KEYS)
        if overlap:
            shown = ", ".join(sorted(overlap)[:10])
            more = f" (+{len(overlap) - 10} more)" if len(overlap) > 10 else ""
            raise ValueError(
                f"{len(overlap)} column(s) would be duplicated when joining the "
                f"snapshot blocks: {shown}{more}. Each block must own a disjoint "
                "set of column names; a repeated block is the usual cause."
            )
        merged = merged.merge(part, on=SNAPSHOT_WIDE_KEYS, how="outer")
    return merged


def _resolve_target_line(panel: pd.DataFrame, *, anchor: str) -> pd.DataFrame:
    """The line the target is defined against, per (game, snapshot).

    The anchor must be a real book. A cross-book consensus was considered and
    rejected: it is steadier and better sampled, but it is not a price anyone
    can take, so profit measured against it is hypothetical.

    The **normalised** line is used, which is a deliberate trade. It is the
    right modelling target -- comparable across books and across snapshots
    rather than confounded with how each book happened to be pricing -- but it
    is not literally the number on the board, so **ROI computed against it is
    not executable**. On the default bet365 anchor the two differ on only a
    handful of rows, because its totals are almost always -110/-110; on other
    books the gap is material. The raw line and both side prices are exported
    as features so the difference stays visible.
    """
    rows = panel[panel["market"].eq(MARKET_TOTALS) & panel["book"].eq(anchor)]
    if rows.empty:
        raise ValueError(
            f"Anchor book {anchor!r} has no totals quotes in the line-history store."
        )
    return rows[["game_id", "snapshot_minutes", "norm_line"]].rename(
        columns={"norm_line": "target_line"}
    )


def _resolve_target_spread_line(panel: pd.DataFrame, *, anchor: str) -> pd.DataFrame:
    """The Bet365 spread AS OF EACH SNAPSHOT, per (game, snapshot).

    This is the whole point of the intermediate spread target: the row for the
    720-minute snapshot must be scored against the line that was on the board 720
    minutes before tip, not against the closing line. Because the panel is keyed
    by ``(game_id, snapshot_minutes)`` and ``build_snapshot_panel`` already
    resolves each snapshot with a strictly backward-looking ``>=`` filter, taking
    the panel value here is contemporaneous by construction -- there is no way to
    reach a later tick from this function.

    Measured on the real data this genuinely matters: 92.6% of games take more
    than one distinct spread value across their snapshots, and the 720-minute
    line differs from the 0-minute line in 79.8% of games.

    Orientation: ``norm_line`` for the spread market is ALREADY the market-implied
    home margin (``snapshots.resolve_line`` stores the away side, which for a
    mirrored spread equals the implied home margin). Verified against realised
    margins at correlation +0.460. It is therefore passed through rather than
    negated -- the opposite of the closing dataset's handicap column, which is
    exactly why both go through ``nba_ou.config.market_columns``.

    ``norm_line`` rather than ``raw_line`` matches the totals anchor above, and
    carries the same caveat: it is the price-centred line, so ROI against it is
    comparable rather than literally executable.
    """
    rows = panel[panel["market"].eq(MARKET_SPREAD) & panel["book"].eq(anchor)]
    if rows.empty:
        raise ValueError(
            f"Anchor book {anchor!r} has no spread quotes in the line-history store."
        )
    return rows[["game_id", "snapshot_minutes", "norm_line"]].rename(
        columns={"norm_line": "target_spread_line"}
    )


def _ridge_closing_lines(
    panel: pd.DataFrame, *, market: str, anchor: str
) -> pd.DataFrame:
    """The anchor's 0-minute tick-series value per game, in Ridge label units.

    Totals and spread use ``norm_line`` (the same centred line as the targets),
    the moneyline its ``level`` (de-vigged home probability).
    """
    value = "level" if market == MARKET_MONEYLINE else "norm_line"
    rows = panel.loc[
        panel["market"].eq(market)
        & panel["book"].eq(anchor)
        & panel["snapshot_minutes"].eq(0),
        ["game_id", value],
    ]
    closes = rows.rename(columns={"game_id": "GAME_ID", value: "CLOSING_LINE"})
    if not closes["GAME_ID"].is_unique:
        raise ValueError(f"Ridge close must be unique for each game ({market})")
    return closes


#: Columns that belong beside the predictions, never in the feature matrix.
#: Keyed by (GAME_ID, TIME_TO_MATCH_MIN) so they can be joined back after
#: scoring.
SCORING_KEYS = ["GAME_ID", "TIME_TO_MATCH_MIN"]


@dataclass
class _SnapshotSide:
    """The snapshot side of the build, one row per (game, output snapshot).

    ``combos`` lists the (game, market, book) quotes and ``consensus_markets``
    the (game, market) consensus rows behind the wide columns: which columns
    exist depends on them, which is how a build of a few games is given the
    full build's columns (see ``_build_snapshot_side``'s ``schema``).
    """

    wide: pd.DataFrame
    ridge_closing_lines: dict[str, pd.DataFrame]
    combos: pd.DataFrame
    consensus_markets: pd.DataFrame
    #: Every column's dtype. A gap anywhere in a build turns a column float
    #: (the pivots and left joins fill it with NaN), so it depends on all rows.
    dtypes: dict[str, str]


#: Game id of the rows that carry a column into the pivots without a value;
#: "~" sorts after every real id, and the rows are dropped after the pivots.
_SCHEMA_GAME_ID = "~schema"


def _schema_rows(
    present: pd.DataFrame, wanted: pd.DataFrame, keys: list[str], snapshot: int
) -> pd.DataFrame:
    """One all-NaN row per distinct ``keys`` combination of ``wanted``, in order.

    The pivots order their columns by first appearance, so these rows go
    first: the columns then come out in the order the full build met them,
    and a combination only ``present`` has comes last, as tonight's do there.
    """
    combos = wanted[keys].drop_duplicates(ignore_index=True)
    rows = pd.DataFrame(np.nan, index=range(len(combos)), columns=present.columns)
    for key in keys:
        rows[key] = combos[key].to_numpy()
    rows["game_id"] = _SCHEMA_GAME_ID
    rows["snapshot_minutes"] = snapshot
    return rows.astype(present.dtypes.to_dict(), errors="ignore")


def _full_build_dtypes(wide: pd.DataFrame, history: dict[str, str]) -> pd.DataFrame:
    """The dtypes a build over these rows and the history together would give.

    A column stays non-float there only if the history keeps it so and these
    rows have no gap in it; anything else is float64, the dtype a gap gives.
    """
    casts = {}
    for column, dtype in wide.dtypes.items():
        if column in SNAPSHOT_WIDE_KEYS:
            continue
        kept = history.get(column)
        if kept is not None and wide[column].notna().all():
            target = kept
        else:
            target = "float64"
        if str(dtype) != target:
            if not (
                pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
            ):
                raise TypeError(f"{column}: cannot align a {dtype} column")
            casts[column] = target
    return wide.astype(casts) if casts else wide


def _history_dtypes(dtypes_by_build: list[dict[str, str]]) -> dict[str, str]:
    """``_full_build_dtypes``' history: columns every build keeps non-float,
    with that dtype. A build without the column had a gap there."""
    if not dtypes_by_build:
        return {}
    first, *rest = dtypes_by_build
    return {
        column: dtype
        for column, dtype in first.items()
        if not pd.api.types.is_float_dtype(dtype)
        and all(other.get(column) == dtype for other in rest)
    }


def _build_snapshot_side(
    ticks: pd.DataFrame,
    *,
    snapshot_grid: tuple[int, ...],
    windows: tuple[int, ...],
    anchor: str,
    ridge_markets: list[str],
    normalize_total_lines: bool,
    normalize_spread_lines: bool,
    null_extreme_spread_prices: bool,
    stage: Callable[[str], None],
    progress: Callable[[str], str | None],
    verbose: bool,
    schema: tuple[pd.DataFrame, pd.DataFrame, dict[str, str]] | None = None,
) -> _SnapshotSide:
    """Panel, movement, consensus and pivot: every value is a function of one
    game's own ticks.

    ``schema`` -- ``(combos, consensus_markets, dtypes)`` of the games a full
    build would also hold, in its panel's order -- puts an all-NaN row for each
    (market, book) and consensus market those games quote ahead of these
    ticks' rows, just before the pivots, and drops them after: the wide frame
    then has the full build's columns, in its order. ``dtypes`` maps a column
    to the dtype every build of those games gives it, where they agree on one
    that is not float; the frame then takes the full build's dtypes too (see
    ``_full_build_dtypes``).
    """
    stage("Building the snapshot panel ...")
    panel_grid = (
        tuple(sorted(set(snapshot_grid) | {0})) if ridge_markets else snapshot_grid
    )
    panel = build_snapshot_panel(
        ticks,
        grid=panel_grid,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        progress=progress("As-of snapshots"),
    )
    stage("Adding movement features ...")
    panel = add_movement_features(
        panel,
        ticks,
        grid=panel_grid,
        windows=windows,
        null_extreme_spread_prices=null_extreme_spread_prices,
        progress=progress("Movement"),
    )
    stage("Cross-book consensus and anchor path ...")
    consensus = aggregate_across_books(panel)
    panel = add_book_deviation(panel)
    anchor_path = add_anchor_total_path_features(panel, ticks, anchor=anchor)
    if verbose:
        print(f"✓ Snapshot panel: {len(panel):,} (game, market, book, snapshot) rows")
    combos = panel[["game_id", "market", "book"]].drop_duplicates(ignore_index=True)
    consensus_markets = consensus[["game_id", "market"]].drop_duplicates(
        ignore_index=True
    )

    pivot_panel, pivot_consensus = panel, consensus
    if schema is not None:
        wanted_combos, wanted_markets, wanted_dtypes = schema
        pivot_panel = pd.concat(
            [
                _schema_rows(panel, wanted_combos, ["market", "book"], panel_grid[0]),
                panel,
            ],
            ignore_index=True,
        )
        pivot_consensus = pd.concat(
            [
                _schema_rows(consensus, wanted_markets, ["market"], panel_grid[0]),
                consensus,
            ],
            ignore_index=True,
        )

    books = sorted(pivot_panel["book"].unique())
    other_books = [book for book in books if book != anchor]

    # Same fields for every book, including all configured movement windows.
    book_features = FULL_BOOK_FEATURES + _windowed_feature_names(windows)
    stage(f"Pivoting {len(books)} book(s) to wide snapshot rows ...")
    wide_parts = [
        _pivot_book_features(pivot_panel, book_features, [anchor]),
        _pivot_book_features(pivot_panel, book_features, other_books),
        _pivot_consensus(pivot_consensus),
    ]
    snapshot_wide = _merge_wide_parts(wide_parts)
    del wide_parts, consensus, pivot_consensus, pivot_panel
    if schema is not None:
        snapshot_wide = snapshot_wide[
            snapshot_wide["game_id"].ne(_SCHEMA_GAME_ID)
        ].reset_index(drop=True)
    snapshot_wide = snapshot_wide.merge(
        anchor_path,
        on=["game_id", "snapshot_minutes"],
        how="left",
        validate="one_to_one",
    )

    target_line = _resolve_target_line(panel, anchor=anchor)
    ridge_closing_lines = {
        market: _ridge_closing_lines(panel, market=market, anchor=anchor)
        for market in ridge_markets
    }
    # Internal closing snapshots must not appear in the requested CSV.
    if 0 not in snapshot_grid:
        snapshot_wide = snapshot_wide[
            snapshot_wide["snapshot_minutes"].isin(snapshot_grid)
        ].copy()
    snapshot_wide = snapshot_wide.merge(
        target_line, on=["game_id", "snapshot_minutes"], how="left"
    )

    # The spread anchor, resolved on the SAME (game, snapshot) key as the totals
    # anchor. Joining on snapshot_minutes is what keeps each row's reference line
    # contemporaneous; a join on game_id alone would collapse every snapshot onto
    # one line and silently produce ten copies of a single closing-spread target.
    target_spread = _resolve_target_spread_line(panel, anchor=anchor)
    snapshot_wide = snapshot_wide.merge(
        target_spread, on=["game_id", "snapshot_minutes"], how="left"
    )
    # The long panel is millions of (game, market, book, snapshot) rows and
    # nothing reads it past this point: every frame derived from it is built
    # above. Holding it through the join below is what made the wide merge --
    # the single largest allocation in the build -- run out of memory.
    del panel, anchor_path, target_line, target_spread
    gc.collect()
    if schema is not None:
        snapshot_wide = _full_build_dtypes(snapshot_wide, wanted_dtypes)
    return _SnapshotSide(
        wide=snapshot_wide,
        ridge_closing_lines=ridge_closing_lines,
        combos=combos,
        consensus_markets=consensus_markets,
        dtypes={column: str(dtype) for column, dtype in snapshot_wide.dtypes.items()},
    )


def _build_scoring_frame(merged: pd.DataFrame, gated: pd.DataFrame) -> pd.DataFrame:
    """Closing lines, weights and timestamps -- everything X must never see.

    These used to ride along in the training frame behind a ``ODDS_CLOSING_`` prefix,
    on the assumption that ``feature_columns()`` would filter them. It does not:
    ``training_pipeline.data.build_feature_matrix`` drops only the *configured*
    exclusions, so every closing line would have entered X and contaminated the
    result. Physical separation is the only version of this that cannot be
    forgotten at the call site.

    Deliberately no per-row weight column. Pooling several snapshots of one game
    does overstate the evidence -- adjacent 30m/60m rows resolve to the
    identical tick 65.7% of the time -- but a weight nothing reads is worse than
    no weight at all, and that is what ``SNAPSHOT_WEIGHT`` was: emitted here,
    consumed nowhere. The correction now lives where it can actually be seen, in
    ``training_pipeline.snapshot_scoring``, which reports one row per horizon so
    each group really is one row per game, and blanks the pooled row's
    confidence interval rather than printing an inflated one.
    """
    keys = merged[SCORING_KEYS].copy()
    closing_columns = [c for c in merged.columns if c.startswith("ODDS_CLOSING_")]
    timestamp_columns = [
        c for c in ["TIPOFF_UTC", "SNAPSHOT_TS_UTC"] if c in merged.columns
    ]

    scoring = pd.concat([keys, merged[closing_columns + timestamp_columns]], axis=1)

    # Restrict to rows that actually survived into the training frame, so the
    # two files line up row for row.
    kept = gated[SCORING_KEYS].drop_duplicates()
    scoring = scoring.merge(kept, on=SCORING_KEYS, how="inner")
    return scoring.sort_values(SCORING_KEYS).reset_index(drop=True)


def append_live_line_history(
    ticks: pd.DataFrame,
    lh_games: pd.DataFrame,
    live: LiveLineHistory,
    *,
    exclude_books: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stored ticks and games plus the live ones, live winning per GAME_ID.

    Live rows replace stored rows of the same game rather than being added to
    them: live they are tonight's games, which the store never holds; replayed,
    they are a past game truncated "as of" an earlier moment, and mixing in the
    stored full history would defeat the replay. The same book exclusions as
    the store read apply.
    """
    live_ids = set(live.games["game_id"].astype(str))
    live_ticks = live.ticks[~live.ticks["book"].isin(exclude_books)]
    stored_ticks = ticks[~ticks["game_id"].astype(str).isin(live_ids)]
    stored_games = lh_games[~lh_games["game_id"].astype(str).isin(live_ids)]
    frames = [frame for frame in (stored_ticks, live_ticks) if not frame.empty]
    merged_ticks = (
        pd.concat(frames, ignore_index=True) if frames else stored_ticks
    ).sort_values(["game_id", "market", "book", "line_ts"], kind="stable")
    game_frames = [frame for frame in (stored_games, live.games) if not frame.empty]
    merged_games = (
        pd.concat(game_frames, ignore_index=True) if game_frames else stored_games
    ).sort_values(["game_date", "game_id"], kind="stable")
    return merged_ticks.reset_index(drop=True), merged_games.reset_index(drop=True)


def _target_snapshot_inputs(
    targets: set[str],
    *,
    season_years: list[int],
    lh_games: pd.DataFrame,
    live: LiveGameDay | None,
    excluded_books: tuple[str, ...],
    combine_books: bool,
    side_settings: dict,
    verbose: bool,
    history_game_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, intermediate_history.SnapshotHistory]:
    """Tonight's ticks and games, and every other game's snapshot history.

    The history comes from the memo, season by season (``intermediate_history``),
    for every game of ``season_years`` but the targets. The targets' ticks are
    the live ones with ``live`` -- which replace stored rows of the same game,
    as in a full build -- and otherwise the stored ones.
    """

    def fetch_season_ticks(season: int) -> pd.DataFrame:
        season_ticks = fetch_pregame_ticks(
            [season], exclude_books=excluded_books, use_cache=input_cache_enabled()
        )
        if combine_books:
            season_ticks = merge_caesars_into_fanatics_ticks(season_ticks)
        return season_ticks

    def build_side(season_ticks: pd.DataFrame) -> _SnapshotSide:
        return _build_snapshot_side(
            season_ticks,
            **side_settings,
            stage=_stage_logger(False),
            progress=lambda label: None,
            verbose=False,
        )

    histories = intermediate_history.load_season_histories(
        season_years,
        fingerprints=pregame_tick_fingerprints(
            season_years, exclude_books=excluded_books
        ),
        settings=intermediate_history.settings_key(
            **side_settings,
            excluded_books=excluded_books,
            combine_books=combine_books,
        ),
        fetch_season_ticks=fetch_season_ticks,
        build_side=build_side,
        anchor=side_settings["anchor"],
        verbose=verbose,
    )
    snapshot_history = intermediate_history.assemble_history(
        histories,
        keep_game=lambda ids: ~ids.isin(targets)
        & (True if history_game_ids is None else ids.isin(history_game_ids)),
        history_dtypes=_history_dtypes,
        # A live build sorts its ticks by game across seasons
        # (append_live_line_history); a stored one reads them season by season.
        by_game_id=live is not None,
    )
    if live is not None:
        ticks, lh_games = append_live_line_history(
            live.line_history.ticks.iloc[0:0],
            lh_games,
            live.line_history,
            exclude_books=excluded_books,
        )
    else:
        target_seasons = sorted(
            {season_of_game_id(game) for game in targets} & set(season_years)
        )
        ticks = fetch_pregame_ticks(
            target_seasons,
            exclude_books=excluded_books,
            use_cache=input_cache_enabled(),
        )
    ticks = ticks[ticks["game_id"].astype(str).isin(targets)].reset_index(drop=True)
    return ticks, lh_games, snapshot_history


def _base_order(base: pd.DataFrame) -> dict[str, int]:
    """Each game's position in the base frame: the merged frame's game order."""
    return {game: i for i, game in enumerate(base["GAME_ID"].astype(str))}


def _history_rows(
    snapshot_history: intermediate_history.SnapshotHistory,
    *,
    base: pd.DataFrame,
    lh_games: pd.DataFrame,
) -> pd.DataFrame:
    """The other rows a full build's merged frame holds, as the cross-row
    stages read them: the stored snapshot values of every game the base holds
    (the full build's inner join), with the base and tip-off columns those
    stages need and the full build's row order."""
    order = _base_order(base)
    rows = snapshot_history.rows
    rows = rows[rows["game_id"].astype(str).isin(order)].rename(
        columns={"game_id": "GAME_ID", "snapshot_minutes": "TIME_TO_MATCH_MIN"}
    )
    rows["GAME_ID"] = rows["GAME_ID"].astype(str)
    game_columns = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "TEAM_ID_TEAM_HOME",
        "TEAM_ID_TEAM_AWAY",
    ]
    rows = rows.merge(base[game_columns], on="GAME_ID", how="left")
    rows = rows.merge(
        lh_games[["game_id", "tipoff_utc"]].rename(
            columns={"game_id": "GAME_ID", "tipoff_utc": "TIPOFF_UTC"}
        ),
        on="GAME_ID",
        how="left",
    )
    rows["SNAPSHOT_TS_UTC"] = rows["TIPOFF_UTC"] - pd.to_timedelta(
        rows["TIME_TO_MATCH_MIN"], unit="m"
    )
    rows[ORDER_COLUMN] = intermediate_history.order_key(
        rows["GAME_ID"], rows["TIME_TO_MATCH_MIN"], order
    )
    return rows.sort_values(ORDER_COLUMN, kind="stable").reset_index(drop=True)


def _ridge_with_history(
    target_input: pd.DataFrame,
    history_rows: pd.DataFrame,
    closing_lines: pd.DataFrame,
    *,
    anchor: str,
    market: str,
    line_column: str,
) -> pd.Series:
    """The Ridge feature of ``target_input``'s rows, fitted as a full build
    fits it: on the earlier games' gated rows of the seasons it reads, merged
    with these in the full build's order."""
    seasons = pd.to_numeric(target_input["SEASON_YEAR"]).astype(int).unique()
    fit_seasons = {int(s) for s in seasons} | {int(s) - 1 for s in seasons}
    history = history_rows[
        history_rows["target_line"].notna()
        & pd.to_numeric(history_rows["SEASON_YEAR"]).astype(int).isin(fit_seasons)
    ]
    # The cross-market gap is stored under its own name; the spread design
    # reads it as the G3 feature it equals.
    history = history.rename(columns=intermediate_history.gap_columns(anchor))
    if market == MARKET_TOTALS:
        history = history.rename(columns={"target_line": line_column})
    columns = [c for c in target_input.columns if c in history.columns]
    frame = pd.concat(
        [history[columns], target_input], ignore_index=False, keys=["h", "t"]
    ).sort_values(ORDER_COLUMN, kind="stable")
    fitted = add_historical_ridge_movement(
        frame.drop(columns=ORDER_COLUMN).reset_index(drop=True),
        closing_lines,
        anchor=anchor,
        market=market,
    )[EXPECTED_MOVE_COLUMNS[market]]
    fitted.index = frame.index
    return fitted.loc["t"].reindex(target_input.index)


def nearest_grid_horizon(minutes: int, grid: tuple[int, ...]) -> int:
    """The grid horizon closest to ``minutes``; a tie goes to the earlier one."""
    return min(grid, key=lambda horizon: (abs(horizon - minutes), -horizon))


def _keep_extra_snapshots(
    wide: pd.DataFrame, grid: tuple[int, ...], extra: Mapping[str, int]
) -> pd.DataFrame:
    """``wide``'s grid rows, and of the extra horizons only each game's own."""
    minutes = wide["snapshot_minutes"].astype(int)
    own_extra = wide["game_id"].astype(str).map(extra)
    keep = minutes.isin(grid) | minutes.eq(own_extra)
    return wide.loc[keep].reset_index(drop=True)


def create_intermediate_line_df(
    *,
    recent_limit_to_include: str | pd.Timestamp | None = None,
    season_years: list[int] | None = None,
    base_lookback_seasons: int = DEFAULT_BASE_LOOKBACK_SEASONS,
    referee_history_seasons: int = DEFAULT_REFEREE_HISTORY_SEASONS,
    include_same_season_referee_variants: bool = False,
    snapshot_grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    anchor_book: str | None = None,
    exclude_fanatics: bool = False,
    exclude_caesars: bool = False,
    combine_fanatics_and_caesars: bool | None = None,
    categorical_team_encoding: bool = False,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    include_ridge_movement: bool = True,
    include_market_dynamics: bool = True,
    return_scoring: bool = False,
    live: LiveGameDay | None = None,
    target_game_ids=None,
    extra_target_snapshots: Mapping[str, int] | None = None,
    target_snapshots: Mapping[str, int] | None = None,
    strict_history_cutoff: bool = False,
    verbose: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (game, snapshot) dataset.

    Returns the training frame. With ``return_scoring=True`` returns
    ``(training, scoring)``, where ``scoring`` holds the closing lines, the
    per-row snapshot weight and the raw timestamps -- everything that must stay
    out of the feature matrix. The default is the safe one: closing lines are
    absent unless explicitly asked for.

    ``strict_history_cutoff`` is for historical replay with ``live`` and
    explicit ``recent_limit_to_include``: prior-game market history and
    injury-status box-score form respect that date too, including when it
    deliberately leaves out the preceding overnight slate.

    ``base_lookback_seasons`` loads team history from before the first
    line-history season so that season's rolling features are warm. Those extra
    seasons are context only: the snapshot join is an inner merge, so a game the
    store holds no ticks for cannot become a row. Adding older years to
    ``season_years`` does **not** achieve this -- they are intersected against
    the store's own seasons and dropped.

    ``include_ridge_movement`` adds a walk-forward estimate of the anchor
    total's remaining move to its tick-series closing line. It uses only
    completed prior games; set it to False to omit the column and its fitting
    work entirely.

    ``include_market_dynamics`` adds how injury news and the markets evolved up
    to each snapshot: injury news (``INJ_SNAP_*``), the market's reaction to it
    (``ODDS_SNAP_NEWS_*``), cross-market coherence (``ODDS_SNAP_XMKT_*``) and the
    walk-forward spread and moneyline move-to-close estimates. See
    ``docs/intermediate/market_dynamics.md``.

    ``fanatics_sportsbook`` only exists from season 2025 and the odds-fetching
    pipeline no longer scrapes the now-discontinued Caesars, so left alone
    fanatics_sportsbook is a de facto season indicator: a book present exactly
    when ``season_year == 2025`` lets a model recover the season from column
    availability alone. Three mutually-aware options control this:

    * ``combine_fanatics_and_caesars`` (the default) -- fold Caesars into
      fanatics_sportsbook (fanatics values kept where present, Caesars fills
      the gap) across both the tick-level snapshot pipeline and the closing/
      opener wide pipeline, giving one continuously-covered book instead of
      two disjoint, season-correlated ones. This is the recommended default:
      it fixes the season-leak problem below without discarding either book's
      data.
    * ``exclude_fanatics`` -- drop fanatics_sportsbook outright instead.
    * ``exclude_caesars`` -- drop Caesars outright instead.

    Combining is tri-state so that "default on" cannot turn an explicit
    exclusion into an error: left unset it yields to either ``exclude_*``, and
    only an explicit ``combine_fanatics_and_caesars=True`` alongside an
    exclusion raises. See ``resolve_combine_books``. Excluding here rather than
    relying on the training config's ``cleaning.exclude_cols_containing`` keeps
    unwanted columns out of the CSV entirely, so they also cannot inflate
    ``max_na_per_row`` for earlier rows.
    ``live`` adds tonight's schedule and line history in memory through
    ``live_game_day.fetch_live_game_day``. Injury reports are synchronized to
    Aiven first, and official roster snapshots are archived in S3.
    Its line history is added to the ticks and games read from the
    store (replacing any stored rows of the same GAME_ID, which is what lets a
    past date be replayed "as of" an earlier moment to test train/serve
    parity), its schedule adds tonight's base rows, built from earlier games
    only, and its referee crews, where released, feed the referee features --
    NaN otherwise, as before the 09:00 ET release in training. History then
    stops the day before (``recent_limit_to_include`` defaults to it), and the
    live season is admitted even before the store holds any of it.

    ``extra_target_snapshots`` (tonight-only mode) gives a target game one more
    snapshot off the grid, in minutes before its tip -- the prediction run's own
    moment, so a model can read the latest market. Every value of that row is
    computed as of that moment, like a grid row's; only the injury stage's
    calibration on earlier games, which a full build has at grid horizons only,
    reads them at the nearest grid horizon (``nearest_grid_horizon``). The other
    games' history is the grid's, so a training build cannot change.

    ``target_snapshots`` (tonight-only mode) builds each target game at one
    snapshot only, in minutes before its tip; a target it leaves out gets no
    row. An off-grid value is an extra snapshot as above. Each row kept is the
    row the same build at every snapshot gives it: the other rows are dropped
    once the snapshot side is built, and the injury stage then computes only
    the horizons the kept rows read. The prediction job uses it when every
    intermediate slot reads the latest snapshot.
    """
    targets = (
        None
        if target_game_ids is None
        else set(normalize_game_ids([str(g) for g in target_game_ids]))
    )
    if strict_history_cutoff and (
        live is None or targets is None or recent_limit_to_include is None
    ):
        raise ValueError(
            "strict_history_cutoff requires live replay, target games and an explicit history limit"
        )
    extra_snapshots: dict[str, int] = {}
    if extra_target_snapshots:
        if targets is None:
            raise ValueError("extra_target_snapshots needs target_game_ids")
        extra_snapshots = {
            str(game).strip(): int(minutes)
            for game, minutes in extra_target_snapshots.items()
        }
        unknown = sorted(set(extra_snapshots) - targets)
        if unknown:
            raise ValueError(f"extra_target_snapshots of non-target games: {unknown}")
        if any(minutes < 0 for minutes in extra_snapshots.values()):
            raise ValueError("extra_target_snapshots must be minutes before tip")
    only_snapshots: dict[str, int] | None = None
    if target_snapshots is not None:
        if targets is None:
            raise ValueError("target_snapshots needs target_game_ids")
        only_snapshots = {
            str(game).strip(): int(minutes)
            for game, minutes in target_snapshots.items()
        }
        unknown = sorted(set(only_snapshots) - targets)
        if unknown:
            raise ValueError(f"target_snapshots of non-target games: {unknown}")
        if any(minutes < 0 for minutes in only_snapshots.values()):
            raise ValueError("target_snapshots must be minutes before tip")
        for game, minutes in only_snapshots.items():
            if minutes in snapshot_grid:
                continue
            if extra_snapshots.get(game, minutes) != minutes:
                raise ValueError(
                    f"{game}: target snapshot T-{minutes} and extra snapshot "
                    f"T-{extra_snapshots[game]} disagree"
                )
            extra_snapshots[game] = minutes
    # Argument validation before any I/O: a bad call should fail immediately,
    # not after opening a database connection.
    if base_lookback_seasons < 0:
        raise ValueError(
            "base_lookback_seasons must be >= 0; it only ever loads history "
            "EARLIER than the first line-history season."
        )
    if referee_history_seasons < 1:
        raise ValueError("referee_history_seasons must be >= 1")
    combine_books = resolve_combine_books(
        combine=combine_fanatics_and_caesars,
        exclude_caesars=exclude_caesars,
        exclude_fanatics=exclude_fanatics,
    )

    anchor = anchor_book or get_main_book()
    # Books stored in line history but not yet admitted as features are
    # excluded unconditionally, see HISTORY_ONLY_BOOKS.
    excluded_books: tuple[str, ...] = HISTORY_ONLY_BOOKS
    if exclude_fanatics:
        excluded_books += PARTIAL_COVERAGE_BOOKS
    if exclude_caesars:
        excluded_books += (CAESARS_BOOK,)

    # The default scope is SBR's seasons: it defines the dataset (the base
    # lookback and every expanding mean start from its first season), so a
    # season only the Covers fallback holds (2018-19's backfilled games) must
    # not widen it. Asked for explicitly, such a season is readable.
    store_seasons = available_seasons(covers_fallback=False)
    readable_seasons = set(available_seasons()) | set(store_seasons)
    if live is not None and not live.line_history.games.empty:
        live_seasons = set(
            int(y) for y in live.line_history.games["season_year"].unique()
        )
        store_seasons = sorted(set(store_seasons) | live_seasons)
        readable_seasons |= live_seasons
    if live is not None and recent_limit_to_include is None:
        recent_limit_to_include = live.history_limit
    if season_years is None:
        season_years = store_seasons
    else:
        season_years = sorted(set(season_years) & readable_seasons)
    if not season_years:
        raise ValueError(
            f"No requested season is present in the line-history store "
            f"(available: {store_seasons})."
        )

    stage = _stage_logger(verbose)

    def progress(label: str) -> str | None:
        return label if verbose else None

    if verbose:
        print(f"Line-history seasons in scope: {season_years}")
        print(f"Snapshot grid ({len(snapshot_grid)}): {sorted(snapshot_grid)}")

    # Ridge labels are the 0-minute snapshot from this very tick series.
    # Generate it internally even if the caller omits 0 from the output grid.
    ridge_markets = [
        market
        for market, wanted in (
            (MARKET_TOTALS, include_ridge_movement),
            (MARKET_SPREAD, include_market_dynamics),
            (MARKET_MONEYLINE, include_market_dynamics),
        )
        if wanted
    ]
    side_settings = {
        "snapshot_grid": snapshot_grid,
        "windows": windows,
        "anchor": anchor,
        "ridge_markets": ridge_markets,
        "normalize_total_lines": normalize_total_lines,
        "normalize_spread_lines": normalize_spread_lines,
        "null_extreme_spread_prices": null_extreme_spread_prices,
    }

    # ---- snapshot side -------------------------------------------------
    snapshot_history = None
    if targets is None:
        stage(f"Fetching pre-game ticks for {len(season_years)} season(s) ...")
        ticks = fetch_pregame_ticks(
            season_years, exclude_books=excluded_books, use_cache=input_cache_enabled()
        )
        lh_games = fetch_games(season_years)
        if live is not None:
            ticks, lh_games = append_live_line_history(
                ticks, lh_games, live.line_history, exclude_books=excluded_books
            )
    else:
        stage("Snapshot history of the finished games ...")
        lh_games = fetch_games(season_years)
        history_game_ids = None
        if strict_history_cutoff:
            permitted = pd.to_datetime(lh_games["game_date"]).le(
                pd.Timestamp(recent_limit_to_include)
            )
            permitted |= lh_games["game_id"].astype(str).isin(targets)
            lh_games = lh_games.loc[permitted].copy()
            history_game_ids = set(lh_games["game_id"].astype(str)) - targets
        ticks, lh_games, snapshot_history = _target_snapshot_inputs(
            targets,
            season_years=season_years,
            lh_games=lh_games,
            live=live,
            excluded_books=excluded_books,
            combine_books=combine_books,
            side_settings=side_settings,
            verbose=verbose,
            history_game_ids=history_game_ids,
        )
    if ticks.empty:
        raise ValueError("No pre-game ticks returned for the requested seasons.")
    if combine_books:
        ticks = merge_caesars_into_fanatics_ticks(ticks)
    if verbose:
        print(f"✓ {len(ticks):,} pre-game ticks over {ticks.game_id.nunique()} games")
    # The extra horizons widen the targets' grid only: the history above was
    # built on the grid itself, as a full build's is.
    target_settings = side_settings
    if extra_snapshots:
        target_settings = {
            **side_settings,
            "snapshot_grid": tuple(
                sorted(set(snapshot_grid) | set(extra_snapshots.values()))
            ),
        }
        if verbose:
            print(f"Extra target snapshots: {dict(sorted(extra_snapshots.items()))}")
    side = _build_snapshot_side(
        ticks,
        **target_settings,
        stage=stage,
        progress=progress,
        verbose=verbose,
        schema=None if snapshot_history is None else snapshot_history.schema,
    )
    snapshot_wide = side.wide
    if extra_snapshots:
        snapshot_wide = _keep_extra_snapshots(
            snapshot_wide, snapshot_grid, extra_snapshots
        )
    if only_snapshots is not None:
        own = snapshot_wide["game_id"].astype(str).map(only_snapshots)
        snapshot_wide = snapshot_wide.loc[
            snapshot_wide["snapshot_minutes"].astype(int).eq(own)
        ].reset_index(drop=True)
        if verbose:
            print(f"Target snapshots only: {dict(sorted(only_snapshots.items()))}")
    ridge_closing_lines = side.ridge_closing_lines
    if snapshot_history is not None:
        ridge_closing_lines = {
            market: pd.concat(
                [snapshot_history.ridge_closing_lines[market], closing],
                ignore_index=True,
            )
            for market, closing in ridge_closing_lines.items()
        }
    del side
    gc.collect()

    # ---- historical line dynamics (prior games only) -------------------
    # Per market: how a team's spread and moneyline get re-priced is different
    # information from how its totals do, and defaulting to totals alone threw
    # two thirds of it away.
    history = lh_games[["game_id"]].copy()
    for market in (MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE):
        stage(f"Prior-game line dynamics ({MARKET_SHORT[market]}) ...")
        if snapshot_history is None:
            market_history = add_prior_game_line_dynamics(
                lh_games, ticks, market=market
            )
        else:
            market_history = add_prior_game_line_dynamics(
                lh_games,
                None,
                market=market,
                dynamics=pd.concat(
                    [
                        snapshot_history.dynamics[market],
                        build_game_line_dynamics(ticks, market=market),
                    ],
                    ignore_index=True,
                ),
            )
        suffix = MARKET_SHORT[market]
        market_history = market_history.rename(
            columns={
                column: f"ODDS_{suffix}_{column}"
                for column in market_history.columns
                if column != "game_id"
            }
        )
        history = history.merge(market_history, on="game_id", how="left")

    # ---- base per-game features ---------------------------------------
    # Older seasons are context for the rolling features only; the inner join
    # below drops every game the line-history store has no ticks for.
    base_start_year = min(season_years) - base_lookback_seasons
    if verbose and base_lookback_seasons:
        print(
            f"Base features from {base_start_year}-10-01 "
            f"({base_lookback_seasons} season(s) before the first line-history "
            "season, as rolling context only)"
        )
    if base_start_year < FIRST_SEASON_WITH_CLOSING_ODDS:
        print(
            f"WARNING: base features start at {base_start_year}, but the "
            f"closing-odds tables begin at {FIRST_SEASON_WITH_CLOSING_ODDS}. "
            "The extra season(s) carry team and player history but no odds, so "
            "they lengthen the build without warming any odds rollup."
        )
    stage("Creating base game features (closing pipeline) ...")
    roster_snapshots = live.roster_snapshots if live is not None else None
    if live is not None and roster_snapshots is None:
        roster_snapshots = load_live_rosters(live.scheduled_games, as_of=live.as_of)
    base, injury_context = create_base_game_features(
        recent_limit_to_include=recent_limit_to_include,
        season_start_date=pd.Timestamp(year=base_start_year, month=10, day=1),
        categorical_team_encoding=categorical_team_encoding,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        exclude_caesars=exclude_caesars,
        combine_fanatics_and_caesars=combine_books,
        return_injury_context=True,
        scheduled_games=live.scheduled_games if live is not None else None,
        roster_snapshots=roster_snapshots,
        live_as_of=live.as_of if live is not None else None,
        target_game_ids=targets,
        verbose=verbose,
    )
    base["GAME_ID"] = base["GAME_ID"].astype(str)
    stage(f"✓ Base features: {base.shape[0]:,} games x {base.shape[1]:,} columns")
    stage("Referee features ...")
    base = add_intermediate_referee_features(
        base,
        history_seasons=referee_history_seasons,
        include_same_season_variants=include_same_season_referee_variants,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        exclude_caesars=exclude_caesars,
        combine_fanatics_and_caesars=combine_books,
        df_referees_scheduled=live.df_referees_scheduled if live is not None else None,
        target_game_ids=targets,
    )

    # Availability-effect history reads completed games' closing lines. Keep
    # this per-game frame before the current game's closing columns are renamed
    # into the scoring-only namespace below.
    injury_base = base

    # Closing lines are renamed, not kept: they leave in the scoring sidecar so
    # they cannot reach the feature matrix. The opener is deliberately NOT swept
    # up -- it is known ~25h before tip, so it is safe at every snapshot, and
    # renaming it silently broke the `comparison_line_cols` baseline configured
    # in experiments/_base.yaml.
    # Every market's CURRENT-GAME closing value, not just totals. The spread and
    # moneyline families are swept for the same reason totals always were: in
    # this dataset those quantities must come from the SNAPSHOT panel, so a
    # closing-valued column of the same shape sitting in the frame is either a
    # leak or a silent overwrite of the snapshot value. Renaming them to
    # ODDS_CLOSING_ routes them to the scoring sidecar, where they can measure
    # closing-line value without ever being reachable from X.
    closing_source_prefixes = (
        "ODDS_TOTAL_LINE_",
        "ODDS_SPREAD_LINE_HOME_",
        "ODDS_SPREAD_CONSENSUS_",
        "ODDS_SPREAD_BET365_MINUS_CONSENSUS",
        "ODDS_SPREAD_CROSS_BOOK_",
        "ODDS_SPREAD_BOOK_COUNT",
        "ODDS_SPREAD_PRICE_",
        "ODDS_ML_PRICE_",
        "ODDS_ML_PROB_",
    )
    closing_source_columns = [
        column
        for column in base.columns
        if column.startswith(closing_source_prefixes)
        and "_BEFORE" not in column
        and column != OPENER_LINE_COLUMN
    ]
    if exclude_fanatics:
        closing_source_columns = [
            column
            for column in closing_source_columns
            if not any(book in column for book in PARTIAL_COVERAGE_BOOKS)
        ]
        # Every column mentioning the book, including the ``_BEFORE`` rolling
        # history features. Those are leakage-safe in the usual sense, but the
        # book only exists from 2025: a column that is NaN for four seasons and
        # populated for one IS a season indicator, whatever its suffix says.
        base = base.drop(
            columns=[
                column
                for column in base.columns
                if any(book in column for book in PARTIAL_COVERAGE_BOOKS)
            ],
            errors="ignore",
        )
    # ODDS_ stays the leading marker even under CLOSING_: strip it off the
    # source name before rebuilding as ODDS_CLOSING_<...> rather than
    # prefixing twice.
    base = base.rename(
        columns={
            column: f"ODDS_CLOSING_{column.removeprefix('ODDS_')}"
            for column in closing_source_columns
        }
    )

    # ---- join ----------------------------------------------------------
    stage("Joining base features onto snapshot rows ...")
    # Merged in two steps rather than one chained expression so each right-hand
    # side can be released immediately; chaining keeps both alive alongside the
    # intermediate result.
    merged = base.merge(
        snapshot_wide, left_on="GAME_ID", right_on="game_id", how="inner"
    )
    del snapshot_wide
    gc.collect()
    merged = merged.merge(history, left_on="GAME_ID", right_on="game_id", how="left")
    del history
    gc.collect()
    merged = merged.drop(
        columns=[
            c for c in ["game_id_x", "game_id_y", "game_id"] if c in merged.columns
        ]
    )

    merged = merged.rename(columns={"snapshot_minutes": "TIME_TO_MATCH_MIN"})
    merged = merged.merge(
        lh_games[["game_id", "tipoff_utc"]].rename(
            columns={"game_id": "GAME_ID", "tipoff_utc": "TIPOFF_UTC"}
        ),
        on="GAME_ID",
        how="left",
    )
    merged["SNAPSHOT_TS_UTC"] = merged["TIPOFF_UTC"] - pd.to_timedelta(
        merged["TIME_TO_MATCH_MIN"], unit="m"
    )
    stage(f"✓ Joined: {merged.shape[0]:,} rows x {merged.shape[1]:,} columns")
    history_rows = None
    if snapshot_history is not None:
        history_rows = _history_rows(snapshot_history, base=base, lh_games=lh_games)
        merged[ORDER_COLUMN] = intermediate_history.order_key(
            merged["GAME_ID"], merged["TIME_TO_MATCH_MIN"], _base_order(base)
        )
    # Market dynamics read no injury feature: they are computed on these rows
    # while the injury horizons run, and attached after the injury columns.
    market: dict[str, pd.DataFrame] = {}

    def market_dynamics(snapshots=merged) -> None:
        stage("Market dynamics (injury news, cross-market, news betas) ...")
        market["columns"] = intermediate_market_dynamics_columns(
            snapshots,
            ticks=ticks,
            anchor=anchor,
            verbose=verbose,
            history=history_rows,
            live_reports=live.injury_reports if live is not None else None,
            visible_until=live.as_of if live is not None else None,
        )

    stage("Snapshot injury features (per horizon) ...")
    keys_before = merged[["GAME_ID", "TIME_TO_MATCH_MIN"]]
    merged = add_snapshot_injury_features(
        merged,
        injury_base,
        injury_context,
        progress=progress("Injuries"),
        target_game_ids=targets,
        history_cutoffs=history_rows,
        live_reports=live.injury_reports if live is not None else None,
        history_limit=(
            pd.Timestamp(recent_limit_to_include) if strict_history_cutoff else None
        ),
        while_waiting=market_dynamics if include_market_dynamics else None,
        horizon_aliases={
            minutes: nearest_grid_horizon(minutes, snapshot_grid)
            for minutes in set(extra_snapshots.values())
            if minutes not in snapshot_grid
        },
    )
    if include_market_dynamics:
        if (
            not merged[["GAME_ID", "TIME_TO_MATCH_MIN"]]
            .reset_index(drop=True)
            .equals(keys_before.reset_index(drop=True))
        ):
            raise ValueError("Injury features reordered the snapshot rows")
        merged = attach_market_dynamics(merged, market["columns"])
        stage("✓ Injury features and market dynamics attached")
    row_order = merged.pop(ORDER_COLUMN) if ORDER_COLUMN in merged else None
    # ---- targets -------------------------------------------------------
    stage("Targets, referee masking and the leakage gate ...")
    main_line_column = total_line_col(anchor)
    # The main-book column now holds the SNAPSHOT line, so the derived target
    # and the settlement price are the same number. See the module docstring.
    merged[main_line_column] = merged["target_line"]
    merged["LINE_ERROR"] = pd.to_numeric(
        merged["TOTAL_POINTS"], errors="coerce"
    ) - pd.to_numeric(merged[main_line_column], errors="coerce")

    # ---- spread target, per snapshot -----------------------------------
    # The canonical anchor spread column holds the SNAPSHOT line, so -- exactly as
    # for totals above -- the derived target and the line a bet settles into are
    # the same number.
    #
    # `target_spread_line` is already in implied-home-margin orientation (see
    # _resolve_target_spread_line), so it is passed through
    # spread_line_home_from_implied_margin rather than negated. The closing
    # dataset's column needs the opposite conversion; routing both through named
    # helpers is what stops that difference from becoming a silent sign error.
    main_spread_column = spread_line_home_col(anchor)
    merged[main_spread_column] = spread_line_home_from_implied_margin(
        merged["target_spread_line"]
    )
    merged = add_snapshot_referee_interactions(merged, book=anchor)
    # This masks the legacy, crew tendency and interaction families together.
    # Earlier snapshots must not inherit that game's eventual referee crew.
    merged = mask_referees_before_release(
        merged, visible_until=live.as_of if live is not None else None
    )
    missing_scores = [
        column
        for column in (PTS_HOME_COL, PTS_AWAY_COL)
        if column not in merged.columns
    ]
    if missing_scores:
        raise KeyError(
            f"{missing_scores} are absent, so HOME_MARGIN cannot be derived. They "
            "are carried through select_training_columns for exactly this reason; "
            "if they were dropped upstream the spread target would otherwise be "
            "built from whatever HOME_MARGIN happened to be lying around."
        )

    # Re-derived from the per-team finals rather than trusting the column carried
    # down from merge_home_away. The two must agree; deriving it here from the
    # scores that are right there makes the target reproducible from the CSV
    # alone, which is what the verification in training_pipeline re-checks.
    merged[HOME_MARGIN_COL] = home_margin(merged)
    merged[SPREAD_ERROR_COL] = spread_error(
        merged[HOME_MARGIN_COL], merged[main_spread_column]
    )

    before = len(merged)
    merged = merged[merged[main_line_column].notna()].copy()
    if verbose and before != len(merged):
        print(f"Dropped {before - len(merged)} rows with no anchor line at snapshot")

    # Rows without a Bet365 spread at this snapshot keep NaN in SPREAD_ERROR and
    # are dropped later by the training pipeline's dropna on its own target. They
    # are NOT dropped here: a missing spread must not cost the totals model a row
    # it could have trained on. Measured coverage is ~100% at every horizon
    # (worst case 99.95% at 720 minutes), so this costs almost nothing either way.
    merged = merged.drop(columns=["target_line", "target_spread_line"])

    # ---- leakage gate --------------------------------------------------
    gated = select_intermediate_training_columns(merged)
    gated = gated.assign(
        **{
            main_line_column: merged[main_line_column],
            main_spread_column: merged[main_spread_column],
        }
    )
    for market in ridge_markets:
        stage(f"Walk-forward ridge move-to-close ({MARKET_SHORT[market]}) ...")
        # The training frame is thousands of columns wide. Keep this fit on a
        # narrow view so adding one feature does not copy the whole dataset.
        line_column = current_line_column(market, anchor)
        ridge_columns = list(
            dict.fromkeys(
                [
                    "GAME_ID",
                    "SEASON_YEAR",
                    "TIME_TO_MATCH_MIN",
                    line_column,
                    *ridge_input_columns(gated.columns, market, anchor),
                ]
            )
        )
        ridge_input = gated[ridge_columns].join(
            merged[["TIPOFF_UTC", "SNAPSHOT_TS_UTC"]]
        )
        output = EXPECTED_MOVE_COLUMNS[market]
        if history_rows is None:
            ridge_feature = add_historical_ridge_movement(
                ridge_input,
                ridge_closing_lines[market],
                anchor=anchor,
                market=market,
            )
            gated[output] = ridge_feature[output]
        else:
            gated[output] = _ridge_with_history(
                ridge_input.assign(**{ORDER_COLUMN: row_order.loc[ridge_input.index]}),
                history_rows,
                ridge_closing_lines[market],
                anchor=anchor,
                market=market,
                line_column=line_column,
            )
    assert_no_bare_closing_odds(gated, allowed=(main_line_column, main_spread_column))

    # Audit BOTH markets. The spread audit is not optional politeness: the
    # snapshot spread family (ODDS_SNAP_SPR_*) contains a raw line, a normalised
    # line, an opener and a move-from-open, and an opener plus its total movement
    # reconstructs the current line exactly -- the same additive-pair shape the
    # totals audit was built to catch.
    stage("Auditing closing-line reconstruction ...")
    # A guard over the dataset's columns, not a value: a tonight-only build's
    # rows carry the same columns as the full build that audited them (and
    # live, no closing line yet), so it is not repeated here.
    audited = (
        ()
        if targets is not None
        else (
            ("total", main_line_column),
            ("spread", main_spread_column),
        )
    )
    for market_name, anchor_column in audited:
        closing_reference = merged.get(
            f"ODDS_CLOSING_{anchor_column.removeprefix('ODDS_')}"
        )
        if closing_reference is None:
            continue
        findings = audit_closing_line_reconstruction(gated, closing_reference)
        if not findings.empty:
            raise ValueError(
                f"Closing {market_name} line is reconstructable from kept "
                f"features:\n{findings.to_string(index=False)}"
            )

    gated = gated.sort_values(["GAME_DATE", "GAME_ID", "TIME_TO_MATCH_MIN"])
    gated = gated.reset_index(drop=True)
    scoring = _build_scoring_frame(merged, gated)

    stage("Done.")
    if verbose:
        print()
        print("--" * 20)
        print(f"Intermediate-line dataset: {len(gated):,} rows")
        print(f"Games: {gated['GAME_ID'].nunique():,}")
        print(f"Snapshots per game: {sorted(gated['TIME_TO_MATCH_MIN'].unique())}")
        print(f"Columns: {gated.shape[1]:,}")
        print(f"Scoring sidecar: {scoring.shape[1]:,} columns (never features)")
        print("--" * 20)

    if return_scoring:
        return gated, scoring
    return gated
