"""Redundant-column detection: which of two equivalent columns survives.

Three kinds of redundancy are resolved here -- exact duplicates, absolute-value
matches, and high correlation -- and all three go through the same two-part
answer:

1. **Which columns are redundant with which**, computed vectorised. The previous
   implementation asked this with nested Python loops over column pairs:
   ``df[a].equals(df[b])`` and ``df[a].abs().equals(df[b].abs())``, each
   1.24M calls on the current 1,578-column dataset. Measured on
   ``training_data_2_0_20260819.csv``: 11s for the duplicate pass, 34s for the
   absolute-value pass, 19.6s for ``DataFrame.corr()`` -- 65s total, growing as
   O(p^2) with the feature count. The vectorised equivalents here run in ~1s.

2. **Which one of a redundant set to keep**, by explicit preference rather than
   by column order. This is the part that was quietly wrong. The old rule kept
   whichever column appeared *later* in the frame, which on real data meant:

       r=0.9982  DROP ODDS_TOTAL_LINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME
                 KEEP ODDS_TOTAL_LINE_betmgm_SEASON_BEFORE_AVG_TEAM_HOME

   bet365 is ``get_main_book()`` -- the book whose line defines the target and
   settles every bet -- discarded in favour of betmgm on nothing but column
   position. See :func:`rank_columns` for the ordering that replaces it.

The correlation pass is also greedy rather than mask-based, which fixes a
second-order bug: the old code computed one boolean mask over the full matrix
and dropped every column having *any* later partner above threshold. Given a
chain A~B~C where A and C are not themselves correlated, that drops both A and
B and keeps only C. Walking best-first and keeping a column unless it is
redundant against something **already kept** keeps A and C, which is what
"remove redundancy" is supposed to mean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import get_main_book, strip_odds_prefix

#: Placeholder written into the byte image of a column before hashing. Any value
#: works as long as it cannot occur in the data, so that two columns agree on a
#: row only when both are NaN there or both hold the same number.
_NAN_SENTINEL = -9.876543210987654e17

#: Column prefixes that are exempt from correlation pruning under a
#: :class:`RepeatedMeasuresRedundancy` policy. These are the snapshot/market
#: features: the whole reason the intermediate-line dataset exists is the small
#: cross-book differences and line movements among them, and a correlation is
#: the wrong instrument for deciding that any of them is surplus. Exact
#: duplicates are still removed, by the hash steps, which do not consult a
#: threshold.
SNAPSHOT_FEATURE_PREFIXES: tuple[str, ...] = ("ODDS_SNAP_", "ODDS_LINE_HIST_")

#: Minutes before tip of the snapshot preferred as a game's representative row.
#: Measured on ``intermediate_line_data_10snap.csv``, t=60 is present for
#: 6,585 of 6,585 games -- but it is NOT guaranteed: ``build_snapshot_panel``
#: emits a horizon only where a tick exists at or before it, and t=720 already
#: misses 149 games. Hence the nearest-horizon fallback in
#: :func:`representative_row_index` rather than an assumption.
REPRESENTATIVE_SNAPSHOT_MINUTES = 60.0

#: Canonical post-merge market names, as built by ``total_line_col`` and friends.
#: Their lowercase counterparts (``total_``, ``ml_``, ...) are the raw shapes the
#: odds databases deliver, and the two coexist for the same quantity -- measured
#: at r=1.0000 between ``ODDS_MONEYLINE_bet365_SEASON_BEFORE_AVG_TEAM_HOME`` and
#: ``ODDS_ml_bet365_price_SEASON_BEFORE_AVG_TEAM_HOME``. When a pair like that
#: has to lose one member, the canonical spelling is the one to keep.
_CANONICAL_MARKET_PREFIXES: tuple[str, ...] = (
    "TOTAL_LINE_",
    "SPREAD_",
    "MONEYLINE_",
)


#: Variance at or below this fraction of a column's mean square counts as no
#: spread at all. A column of 60.0 has mean square 3600, so the cutoff there is
#: 3.6e-9 -- a standard deviation of 6e-5 against a mean of 60, which is a
#: constant to every digit float64 carries.
_CONSTANT_VARIANCE_RTOL = 1e-12


def pairwise_complete_corr(df: pd.DataFrame, *, min_periods: int = 2) -> np.ndarray:
    """Absolute Pearson correlation over pairwise-complete observations.

    Reproduces ``DataFrame.corr()`` exactly -- same pairwise-complete semantics,
    where each pair is scored only on the rows where *both* columns are present
    -- but as four BLAS matrix products instead of pandas' pairwise Cython loop.

    Mean-imputing first and taking a single standardised matmul would be simpler
    and slightly faster, but it is not the same quantity: measured against
    ``DataFrame.corr()`` on the current dataset it deviates by up to 0.293 on the
    327 columns carrying NaNs, because the imputed constant block is not the
    subset pandas scores. Float precision is not the constraint (float64 agrees
    to 7e-13 on NaN-free columns), so there is no reason to accept that bias.

    Pairs with fewer than ``min_periods`` shared observations, and pairs where
    either side is constant on the shared rows, come back as NaN -- matching
    pandas, and read as "no evidence of redundancy" by the callers here.
    """
    values = df.to_numpy(dtype=np.float64, na_value=np.nan)
    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)
    mask = present.astype(np.float64)

    # Sy and Syy are the transposes of Sx and Sxx, so four products suffice.
    n = mask.T @ mask
    sum_x = filled.T @ mask
    sum_xx = (filled * filled).T @ mask
    sum_xy = filled.T @ filled

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_n = np.where(n > 0, n, np.nan)
        mean_x = sum_x / safe_n
        mean_y = mean_x.T
        cov = sum_xy / safe_n - mean_x * mean_y
        var_x = sum_xx / safe_n - mean_x * mean_x
        # A side with no spread on the shared rows carries no correlation
        # information, and pandas reports NaN for it. Computed this way its
        # variance lands on either side of zero as roundoff, so dividing by it
        # gives inf rather than NaN -- which every caller here reads as
        # "perfectly redundant" and acts on by dropping the other column.
        # Compare against the column's own mean square so the test does not
        # depend on the units the column happens to be measured in.
        var_x = np.where(var_x > _CONSTANT_VARIANCE_RTOL * (sum_xx / safe_n), var_x, np.nan)
        var_y = var_x.T
        denominator = np.sqrt(var_x * var_y)
        corr = np.abs(cov / denominator)

    corr[n < min_periods] = np.nan
    np.fill_diagonal(corr, np.nan)
    return corr


def _column_byte_keys(values: np.ndarray) -> list[int]:
    """One hash per column, equal exactly when the columns are bit-identical."""
    normalized = np.where(np.isfinite(values), values, _NAN_SENTINEL)
    return [hash(column.tobytes()) for column in np.ascontiguousarray(normalized).T]


def find_identical_groups(
    df: pd.DataFrame, *, absolute: bool = False
) -> list[list[str]]:
    """Group columns that hold identical values, as name lists of size >= 2.

    Hashing the byte image replaces the O(p^2) pass of ``Series.equals`` calls
    with a single O(p) bucketing. Two deliberate differences from that loop:
    values are compared as float64, so an int column and a float column holding
    the same numbers now group together (they are the same feature); and NaN
    positions must match, since they are normalised to a sentinel rather than
    being unequal to themselves.

    With ``absolute=True`` the magnitudes are compared instead, which catches the
    mixed-sign case that correlation cannot -- a pure sign flip already shows up
    as ``|r| = 1``.
    """
    values = df.to_numpy(dtype=np.float64, na_value=np.nan)
    if absolute:
        values = np.abs(values)

    buckets: dict[int, list[str]] = {}
    for name, key in zip(df.columns, _column_byte_keys(values), strict=True):
        buckets.setdefault(key, []).append(name)
    return [group for group in buckets.values() if len(group) > 1]


@dataclass(frozen=True)
class KeepPreference:
    """Which column to keep when several carry the same information.

    ``protected`` are the caller's ``keep_columns``. They rank first because the
    correlation step is precisely where they used to be lost: an opening line is
    near-perfectly correlated with the closing line, which is why it was pruned
    and exactly why the comparison it supports is interesting.
    """

    protected: frozenset[str] = frozenset()
    main_book: str | None = None

    @classmethod
    def build(
        cls, protected: list[str] | None = None, main_book: str | None = None
    ) -> KeepPreference:
        return cls(
            protected=frozenset(protected or []),
            main_book=main_book if main_book is not None else get_main_book(),
        )

    def mentions_main_book(self, column: str) -> bool:
        """Whether ``column`` is about the book that defines and settles the bet.

        Case-insensitive, and that is load-bearing rather than defensive.
        ``get_main_book()`` returns ``bet365`` in lower case because that is how
        the closing-line pipeline spells books, but the intermediate-line
        dataset pivots them into the column name upper-cased
        (``ODDS_SNAP_TOT_BET365_NORM_LINE``). Matched case-sensitively this
        returned False for every snapshot column, so on that dataset the whole
        tier silently did nothing and the winner of a cross-book tie fell
        through to NaN count and then to alphabetical order -- which happened to
        keep bet365, by luck of the spelling rather than by preference.
        """
        if not self.main_book:
            return False
        return (
            re.search(
                rf"(?:^|_){re.escape(self.main_book)}(?:_|$)", column, re.IGNORECASE
            )
            is not None
        )

    def is_canonical_market_name(self, column: str) -> bool:
        return strip_odds_prefix(column).startswith(_CANONICAL_MARKET_PREFIXES)


@dataclass(frozen=True)
class RepeatedMeasuresRedundancy:
    """How to judge redundancy when a frame holds several rows per unit.

    The intermediate-line dataset is one row per (game, pre-game snapshot). Its
    historical and base features are computed per GAME and then copied onto
    every snapshot of that game, so a frame of 65,666 rows carries only 6,585
    independent observations of them. Correlating them over the full frame
    counts each game up to ten times, which is not a measurement of anything --
    it is the same evidence repeated. This policy says: judge those columns on
    one row per game, exactly as the closing-line dataset would.

    ``exempt_prefixes`` names the columns this does NOT apply to, which is the
    other half of the policy. Snapshot/market features genuinely vary across a
    game's rows -- they are what the extra rows are FOR -- and they are exempt
    from correlation pruning altogether rather than merely judged differently.
    The small cross-book differences and line movements between them are the
    signal this dataset was built to preserve, and a correlation coefficient
    cannot tell those apart from redundancy: measured on this data, 99.06% of a
    book's line variance is simply which game it is, so every pair of books
    reads as near-identical (r 0.9986-0.9995) however differently they move.
    Exact duplicates among them are still removed, by the hash steps.
    """

    group_col: str
    snapshot_col: str | None = None
    target_snapshot: float = REPRESENTATIVE_SNAPSHOT_MINUTES
    exempt_prefixes: tuple[str, ...] = SNAPSHOT_FEATURE_PREFIXES

    def is_exempt(self, column: str) -> bool:
        return column.upper().startswith(
            tuple(prefix.upper() for prefix in self.exempt_prefixes)
        )


def representative_row_index(
    group: pd.Series, snapshot: pd.Series | None, *, target: float
) -> pd.Index:
    """One row label per group: the snapshot nearest ``target``, deterministically.

    ``target`` is preferred rather than required. It is present for every game
    on the current build, but the dataset builder does not guarantee it -- a
    game whose ticks do not reach a horizon simply has no row there -- so a
    missing t=60 must degrade to the nearest available horizon rather than drop
    the game from the view. Dropping it would be the damaging failure: that
    game's historical features would stop counting toward redundancy at all,
    silently and without error.

    Ties break toward the SMALLER horizon (t=30 over t=120 for a target of 60),
    then toward the first row label, so the result never depends on row order.
    Which snapshot is chosen barely matters for correctness -- the columns being
    judged are constant within a game by construction -- but it must be the same
    one on every run, or two runs of the same config could prune differently.
    """
    frame = pd.DataFrame({"group": group.to_numpy()}, index=group.index)
    if snapshot is None:
        frame["distance"] = 0.0
        frame["horizon"] = 0.0
    else:
        horizon = pd.to_numeric(snapshot, errors="coerce")
        frame["distance"] = (horizon - target).abs()
        frame["horizon"] = horizon
    # A NaN horizon must not win a group by sorting first.
    frame["distance"] = frame["distance"].fillna(np.inf)
    frame["order"] = np.arange(len(frame))
    ordered = frame.sort_values(
        ["group", "distance", "horizon", "order"], kind="stable"
    )
    return ordered.groupby("group", sort=False).head(1).index


def rank_columns(
    df: pd.DataFrame, columns: list[str], preference: KeepPreference
) -> list[str]:
    """Order ``columns`` best-first, so the winner of any tie comes first.

    The criteria, in order:

    1. **Protected** -- the caller asked for it explicitly.
    2. **Fewer NaNs** -- between two columns carrying the same information, the
       one populated on more rows is strictly more useful.
    3. **Names the main book** -- the book whose line defines the target and
       settles bets. Keeping betmgm's rolling line while dropping bet365's is
       the specific failure this fixes.
    4. **Canonical market spelling** -- ``MONEYLINE_<book>`` over the raw
       ``ml_<book>_price`` shape for the identical quantity.
    5. **Name** -- a deterministic tiebreak, so the surviving column set does not
       depend on the order the frame happened to be built in.
    """
    nan_counts = df[columns].isna().sum()

    def sort_key(column: str) -> tuple:
        return (
            0 if column in preference.protected else 1,
            int(nan_counts[column]),
            0 if preference.mentions_main_book(column) else 1,
            0 if preference.is_canonical_market_name(column) else 1,
            column,
        )

    return sorted(columns, key=sort_key)


def resolve_column_thresholds(
    columns: list[str],
    *,
    default: float,
    overrides: dict[str, float] | None = None,
) -> np.ndarray:
    """Per-column correlation threshold, by case-insensitive substring match.

    A column matching several patterns takes the **highest** threshold, i.e. the
    most tolerant, so adding a pattern can only ever protect columns.
    """
    thresholds = np.full(len(columns), float(default), dtype=np.float64)
    if not overrides:
        return thresholds

    for pattern, value in overrides.items():
        if not pattern:
            continue
        needle = pattern.upper()
        for index, column in enumerate(columns):
            if needle in column.upper():
                thresholds[index] = max(thresholds[index], float(value))
    return thresholds


def select_correlated_columns_to_drop(
    df: pd.DataFrame,
    *,
    default_threshold: float,
    overrides: dict[str, float] | None = None,
    preference: KeepPreference,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedily keep one column per redundant cluster, best-first.

    A pair is judged against the **more tolerant** of its two columns'
    thresholds, so an odds column pairing with a non-odds one is held to the
    odds tolerance. In practice this decides almost nothing: on the current
    dataset only 6 of the 915 pairs above 0.95 are odds/non-odds, because the
    two groups form near-disjoint correlation clusters.

    Returns the column names to drop and the ``(dropped, kept, r)`` triples
    behind each decision, for reporting.
    """
    columns = list(df.columns)
    if len(columns) < 2:
        return [], []

    corr = pairwise_complete_corr(df)
    thresholds = resolve_column_thresholds(
        columns, default=default_threshold, overrides=overrides
    )
    position = {column: index for index, column in enumerate(columns)}

    kept_positions: list[int] = []
    dropped: list[tuple[str, str, float]] = []

    for column in rank_columns(df, columns, preference):
        index = position[column]
        if kept_positions:
            against = np.asarray(kept_positions)
            correlations = corr[index, against]
            limits = np.maximum(thresholds[index], thresholds[against])
            # NaN (too few shared rows, or a constant side) is not evidence of
            # redundancy, and NaN > limit is already False -- no masking needed.
            redundant = correlations > limits
            if redundant.any():
                winner = int(against[np.argmax(np.where(redundant, correlations, -1))])
                dropped.append((column, columns[winner], float(corr[index, winner])))
                continue
        kept_positions.append(index)

    return [name for name, _, _ in dropped], dropped
