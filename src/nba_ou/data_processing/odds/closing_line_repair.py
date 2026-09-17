"""Repair SBR closing lines from the time-stamped line history.

The SBR closing table (Supabase ``odds_sportsbook``) is contaminated with
in-play lines: the day page shows each book's *last* number, which for many
games is a live line. Audited 2026-09-17, 2,453 of 8,936 games carry at least
one in-play closing value, and on those rows the gap to the real close
correlates ~0.36 with the result -- a direct leak into per-book features,
consensus columns and the ``LINE_ERROR`` / ``SPREAD_ERROR`` targets.

The Aiven line history is time-stamped, so its last priced tick before the
scheduled tip is a close that could actually have been bet. Two steps:

**Step A -- history close.** For every (game, market, book) cell whose line
history has a usable pre-tip quote, the whole quote (line and both prices) is
taken from history, regardless of what SBR stored. Replacing the line alone
would pair a pre-game number with SBR's live prices and normalize into a line
that never existed. Where history has no quote the SBR value is kept, as
``unverified``.

**Step B -- cross-book outliers.** Deliberately conservative: it only acts
where a quote is very likely a data mistake, not merely a book that disagreed.
Comparison happens on normalized levels (centered to -110/-110 for totals and
spreads, the no-vig home probability for moneylines), because only those are
comparable across books. A cell is an outlier only when all hold:

* at least ``MIN_OTHER_BOOKS`` *history-verified* other books quote the game,
* those books agree tightly (range within ``OTHERS_MAX_RANGE``),
* the cell is at least ``OUTLIER_THRESHOLDS`` away from their median,
* and, for history cells, it was already that far out of line against the other
  books *as of its own tick time*. This last clause is what keeps a book that
  closed before a late market-wide jump from being "corrected" to the jump.

Thresholds were measured on history-verified closes, 2019-20 to 2025-26. Among
cells with 3+ agreeing other books, the result residual carries no information
about the deviation beyond 5 points (slope -0.13 +/- 0.44 on totals, -0.12 +/-
0.27 on spreads), i.e. such quotes behave like errors. The flagged cells are
almost all stale (a BetMGM 2019 quote 27 hours old) or sign-flipped (Caesars
+10 against a -10 market). Deviations of 4-6 points were left alone because
their slope is too noisy to call.

Outlier cells are emptied, except for ``TARGET_ANCHOR_BOOK``, which anchors the
residual targets and is filled with the verified books' median instead.

Provenance is returned as a separate audit frame and never written into the
odds frame: whether SBR held an in-play value is itself correlated with the
game (large live moves), and unverified cells cluster by book and season.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from nba_ou.config.market_columns import TARGET_ANCHOR_BOOK
from nba_ou.data_processing.line_history.normalization import (
    MARGIN_SIGMA,
    TOTAL_SIGMA,
    center_two_way_line,
    devig_two_way,
    round_to_increment_signed,
)
from nba_ou.data_processing.odds.normalize_spread_lines import (
    DEFAULT_MAX_REASONABLE_SPREAD_ABS,
    spread_price_extreme_mask,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

CELL_KEYS = ["game_id", "market", "book"]

#: Wide SBR column names per market, oriented like the line history: ``left`` is
#: OVER / AWAY, ``right`` is UNDER / HOME, and the spread line is the AWAY
#: handicap (which equals the expected home margin).
_WIDE_COLUMNS: dict[str, dict[str, str | None]] = {
    MARKET_TOTALS: {
        "line": "total_{book}_line_over",
        "line_mirror": "total_{book}_line_under",
        "left_price": "total_{book}_price_over",
        "right_price": "total_{book}_price_under",
    },
    MARKET_SPREAD: {
        "line": "spread_{book}_line_away",
        "line_mirror": "spread_{book}_line_home",
        "left_price": "spread_{book}_price_away",
        "right_price": "spread_{book}_price_home",
    },
    MARKET_MONEYLINE: {
        "line": None,
        "line_mirror": None,
        "left_price": "ml_{book}_price_away",
        "right_price": "ml_{book}_price_home",
    },
}

#: Books whose columns are not a closing quote of a single book.
NON_BOOK_SLUGS: tuple[str, ...] = ("consensus_opener",)

#: Two-way overround outside this band is not an ordinary quote.
MIN_OVERROUND = 0.0
MAX_OVERROUND = 0.25

#: Plausible total range; anything outside is a scrape artefact.
TOTAL_LINE_BOUNDS = (150.0, 300.0)

MIN_OTHER_BOOKS = 3
OUTLIER_THRESHOLDS: dict[str, float] = {
    MARKET_TOTALS: 6.0,
    MARKET_SPREAD: 5.0,
    MARKET_MONEYLINE: 0.15,
}
OTHERS_MAX_RANGE: dict[str, float] = {
    MARKET_TOTALS: 1.5,
    MARKET_SPREAD: 1.5,
    MARKET_MONEYLINE: 0.05,
}

#: Minimum other books quoting at the outlier's own tick time for the as-of
#: comparison to count; with fewer, only the closing comparison is used.
MIN_OTHER_BOOKS_AS_OF = 2

ACTION_HISTORY_SAME = "history_same"
ACTION_HISTORY_REPLACED = "history_replaced"
ACTION_HISTORY_FILLED = "history_filled"
ACTION_UNVERIFIED = "unverified"
ACTION_OUTLIER_NULLED = "outlier_nulled"
ACTION_OUTLIER_MEDIAN = "outlier_median_filled"

AUDIT_COLUMNS = [
    "game_id",
    "season_year",
    "market",
    "book",
    "action",
    "sbr_level",
    "history_level",
    "final_level",
    "history_minutes_before_tip",
    "others_median",
    "others_range",
    "n_others",
    "deviation",
]

TickLoader = Callable[[list[str]], pd.DataFrame]


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #


def quote_levels(quotes: pd.DataFrame) -> pd.Series:
    """Normalized level of each long-format American-odds quote.

    Totals and spreads are centered to their -110/-110 line (the spread in
    home-margin space); moneylines become the no-vig HOME win probability.
    """
    level = pd.Series(np.nan, index=quotes.index, dtype="float64")
    for market, sigma, left_wins_above in (
        (MARKET_TOTALS, TOTAL_SIGMA, True),
        (MARKET_SPREAD, MARGIN_SIGMA, False),
    ):
        rows = quotes["market"].eq(market)
        if rows.any():
            level[rows] = center_two_way_line(
                quotes.loc[rows, "line"],
                quotes.loc[rows, "left_price"],
                quotes.loc[rows, "right_price"],
                sigma=sigma,
                left_wins_above=left_wins_above,
            )
    rows = quotes["market"].eq(MARKET_MONEYLINE)
    if rows.any():
        level[rows] = devig_two_way(
            quotes.loc[rows, "left_price"], quotes.loc[rows, "right_price"]
        )["fair_right"]
    return level


def valid_quote_mask(quotes: pd.DataFrame) -> pd.Series:
    """Quotes plausible enough to stand as a book's close."""
    left = pd.to_numeric(quotes["left_price"], errors="coerce")
    right = pd.to_numeric(quotes["right_price"], errors="coerce")
    line = pd.to_numeric(quotes["line"], errors="coerce")
    overround = devig_two_way(left, right)["overround"]

    valid = overround.between(MIN_OVERROUND, MAX_OVERROUND)

    totals = quotes["market"].eq(MARKET_TOTALS)
    spread = quotes["market"].eq(MARKET_SPREAD)
    lined = totals | spread
    extreme = spread_price_extreme_mask(
        left, odds_format="american"
    ) | spread_price_extreme_mask(right, odds_format="american")
    valid &= ~lined | (line.notna() & ~extreme)
    valid &= ~totals | line.between(*TOTAL_LINE_BOUNDS)
    valid &= ~spread | line.abs().le(DEFAULT_MAX_REASONABLE_SPREAD_ABS)
    return valid.fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Long <-> wide
# --------------------------------------------------------------------------- #


def closing_books(df: pd.DataFrame) -> list[str]:
    """Books with a full column family for at least one market in ``df``."""
    books: set[str] = set()
    for market, templates in _WIDE_COLUMNS.items():
        head = templates["left_price"]
        prefix, suffix = head.split("{book}")
        for column in df.columns:
            if column.startswith(prefix) and column.endswith(suffix):
                book = column[len(prefix) : len(column) - len(suffix)]
                if book and book not in NON_BOOK_SLUGS:
                    books.add(book)
    return sorted(books)


def closing_quote_columns(df: pd.DataFrame) -> list[str]:
    """Every per-book closing column in ``df`` the repair may rewrite."""
    columns = []
    for book in closing_books(df):
        for market in _WIDE_COLUMNS:
            for column in _columns_for(market, book).values():
                if column is not None and column in df.columns:
                    columns.append(column)
    return columns


def _columns_for(market: str, book: str) -> dict[str, str | None]:
    return {
        key: (template.format(book=book) if template else None)
        for key, template in _WIDE_COLUMNS[market].items()
    }


def wide_to_long_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (row index, market, book) with American prices."""
    frames = []
    for book in closing_books(df):
        for market in _WIDE_COLUMNS:
            cols = _columns_for(market, book)
            if cols["left_price"] not in df.columns or cols["right_price"] not in df.columns:
                continue
            if cols["line"] is not None and cols["line"] not in df.columns:
                continue
            line = np.nan
            if cols["line"] is not None:
                line = pd.to_numeric(df[cols["line"]], errors="coerce")
                if cols["line_mirror"] in df.columns:
                    mirror = pd.to_numeric(df[cols["line_mirror"]], errors="coerce")
                    line = line.fillna(-mirror if market == MARKET_SPREAD else mirror)
                line = line.to_numpy()
            frame = pd.DataFrame(
                {
                    "row": df.index,
                    "game_id": df["game_id"].astype(str).to_numpy(),
                    "market": market,
                    "book": book,
                    "line": line,
                    "left_price": pd.to_numeric(
                        df[cols["left_price"]], errors="coerce"
                    ).to_numpy(),
                    "right_price": pd.to_numeric(
                        df[cols["right_price"]], errors="coerce"
                    ).to_numpy(),
                }
            )
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["row", *CELL_KEYS, "line", "left_price", "right_price"]
        )
    return pd.concat(frames, ignore_index=True)


def _write_cells(df: pd.DataFrame, cells: pd.DataFrame) -> None:
    """Write long ``cells`` (row, market, book, line, prices) back into ``df``."""
    for (market, book), group in cells.groupby(["market", "book"], sort=False):
        cols = _columns_for(market, book)
        rows = group["row"].to_numpy()
        if cols["line"] is not None:
            line = group["line"].to_numpy(dtype="float64")
            mirror = -line if market == MARKET_SPREAD else line
            df.loc[rows, cols["line"]] = line
            if cols["line_mirror"] in df.columns:
                df.loc[rows, cols["line_mirror"]] = mirror
        df.loc[rows, cols["left_price"]] = group["left_price"].to_numpy(dtype="float64")
        df.loc[rows, cols["right_price"]] = group["right_price"].to_numpy(
            dtype="float64"
        )


# --------------------------------------------------------------------------- #
# Step A: history closes
# --------------------------------------------------------------------------- #


def _resolve_tick_line(ticks: pd.DataFrame) -> pd.Series:
    left = pd.to_numeric(ticks["left_line"], errors="coerce")
    right = pd.to_numeric(ticks["right_line"], errors="coerce")
    right_as_left = right.where(ticks["market"].ne(MARKET_SPREAD), -right)
    return left.fillna(right_as_left).mask(ticks["market"].eq(MARKET_MONEYLINE))


def history_quotes(ticks: pd.DataFrame) -> pd.DataFrame:
    """Valid, leveled quotes from ticks shaped like ``fetch_closing_ticks``."""
    columns = [*CELL_KEYS, "line_ts", "minutes_before_tip", "line", "left_price", "right_price", "level"]
    if ticks.empty:
        return pd.DataFrame(columns=columns)
    quotes = ticks.assign(
        game_id=ticks["game_id"].astype(str),
        line=_resolve_tick_line(ticks),
        left_price=pd.to_numeric(ticks["left_price"], errors="coerce"),
        right_price=pd.to_numeric(ticks["right_price"], errors="coerce"),
    )
    quotes = quotes[valid_quote_mask(quotes)].copy()
    quotes["level"] = quote_levels(quotes)
    quotes = quotes[quotes["level"].notna()]
    return quotes[columns].reset_index(drop=True)


def select_history_closes(ticks: pd.DataFrame) -> pd.DataFrame:
    """The last valid pre-tip quote per (game, market, book)."""
    quotes = history_quotes(ticks)
    if quotes.empty:
        return quotes
    return (
        quotes.sort_values("line_ts")
        .groupby(CELL_KEYS, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Step B: cross-book outliers
# --------------------------------------------------------------------------- #


def _leave_one_out_stats(cells: pd.DataFrame) -> pd.DataFrame:
    """Median, range and count of the *verified* other books for each cell."""
    verified = cells["verified"].to_numpy()
    level = cells["final_level"].to_numpy(dtype="float64")
    out = np.full((len(cells), 3), np.nan)
    for index in cells.groupby(["game_id", "market"], sort=False).indices.values():
        pool = index[verified[index] & ~np.isnan(level[index])]
        for position in index:
            others = level[pool[pool != position]]
            out[position, 2] = len(others)
            if len(others):
                out[position, 0] = np.median(others)
                out[position, 1] = others.max() - others.min()
    return pd.DataFrame(
        out, columns=["others_median", "others_range", "n_others"], index=cells.index
    )


def _as_of_deviation(
    candidates: pd.DataFrame, game_ticks: pd.DataFrame
) -> pd.Series:
    """Deviation of each candidate from other books as of its own tick time.

    NaN where fewer than ``MIN_OTHER_BOOKS_AS_OF`` other books had a valid
    quote at that moment.
    """
    deviation = pd.Series(np.nan, index=candidates.index, dtype="float64")
    if game_ticks.empty:
        return deviation
    quotes = history_quotes(game_ticks).sort_values("line_ts")
    for index, cell in candidates.iterrows():
        if pd.isna(cell["history_line_ts"]):
            continue
        others = quotes[
            quotes["game_id"].eq(cell["game_id"])
            & quotes["market"].eq(cell["market"])
            & quotes["book"].ne(cell["book"])
            & quotes["line_ts"].le(cell["history_line_ts"])
        ]
        latest = others.groupby("book").tail(1)
        if len(latest) >= MIN_OTHER_BOOKS_AS_OF:
            deviation[index] = cell["final_level"] - latest["level"].median()
    return deviation


def _american_from_probability(probability: pd.Series) -> pd.Series:
    """Zero-vig American price for a win probability."""
    p = pd.to_numeric(probability, errors="coerce")
    return pd.Series(
        np.where(p >= 0.5, -100.0 * p / (1.0 - p), 100.0 * (1.0 - p) / p),
        index=p.index,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def repair_closing_lines(
    df: pd.DataFrame,
    history_ticks: pd.DataFrame,
    *,
    game_tick_loader: TickLoader | None = None,
    anchor_book: str = TARGET_ANCHOR_BOOK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace SBR closes with line-history closes, then null clear outliers.

    Args:
        df: Wide odds frame with raw **American** prices and ``game_id`` /
            ``season_year`` columns (as loaded from ``odds_sportsbook``).
        history_ticks: Ticks shaped like ``fetch_closing_ticks`` output; only
            the tail per (game, market, book) is needed.
        game_tick_loader: Returns every closing-eligible tick for a list of
            game ids. Used for the as-of check on outlier candidates only. When
            ``None``, the as-of check falls back to ``history_ticks``.
        anchor_book: Book filled with the verified median instead of emptied.

    Returns:
        The repaired frame (a copy) and the per-cell audit frame.
    """
    out = df.copy()
    if out.empty or "game_id" not in out.columns:
        return out, pd.DataFrame(columns=AUDIT_COLUMNS)
    if not out.index.is_unique:
        raise ValueError("repair_closing_lines needs a unique index.")

    sbr = wide_to_long_quotes(out)
    if sbr.empty:
        return out, pd.DataFrame(columns=AUDIT_COLUMNS)
    sbr["has_sbr"] = sbr[["line", "left_price", "right_price"]].notna().any(axis=1)
    sbr["sbr_level"] = quote_levels(sbr)

    closes = select_history_closes(history_ticks).rename(
        columns={
            "line": "history_line",
            "left_price": "history_left_price",
            "right_price": "history_right_price",
            "level": "history_level",
            "line_ts": "history_line_ts",
            "minutes_before_tip": "history_minutes_before_tip",
        }
    )
    cells = sbr.merge(closes, on=CELL_KEYS, how="left")
    cells = cells[cells["has_sbr"] | cells["history_level"].notna()].reset_index(
        drop=True
    )

    # ---- Step A ---------------------------------------------------------- #
    verified = cells["history_level"].notna()
    same_line = cells["line"].eq(cells["history_line"]) | (
        cells["line"].isna() & cells["history_line"].isna()
    )
    same = (
        verified
        & same_line
        & cells["left_price"].eq(cells["history_left_price"])
        & cells["right_price"].eq(cells["history_right_price"])
    )
    cells["action"] = np.select(
        [same, verified & cells["has_sbr"], verified],
        [ACTION_HISTORY_SAME, ACTION_HISTORY_REPLACED, ACTION_HISTORY_FILLED],
        default=ACTION_UNVERIFIED,
    )
    cells["verified"] = verified
    for column in ("line", "left_price", "right_price"):
        cells[f"final_{column}"] = cells[f"history_{column}"].where(
            verified, cells[column]
        )
    cells["final_level"] = cells["history_level"].where(verified, cells["sbr_level"])

    # ---- Step B ---------------------------------------------------------- #
    cells = cells.join(_leave_one_out_stats(cells))
    cells["deviation"] = cells["final_level"] - cells["others_median"]
    threshold = cells["market"].map(OUTLIER_THRESHOLDS)
    max_range = cells["market"].map(OTHERS_MAX_RANGE)
    candidate = (
        cells["n_others"].ge(MIN_OTHER_BOOKS)
        & cells["others_range"].le(max_range)
        & cells["deviation"].abs().ge(threshold)
    )

    if candidate.any():
        verified_candidates = cells[candidate & cells["verified"]]
        if not verified_candidates.empty:
            game_ids = sorted(verified_candidates["game_id"].unique())
            game_ticks = (
                game_tick_loader(game_ids)
                if game_tick_loader is not None
                else history_ticks[history_ticks["game_id"].astype(str).isin(game_ids)]
            )
            as_of = _as_of_deviation(verified_candidates, game_ticks)
            in_line_when_quoted = as_of.abs().lt(threshold[as_of.index]).fillna(False)
            candidate[in_line_when_quoted[in_line_when_quoted].index] = False

    outlier = candidate
    anchor = cells["book"].eq(anchor_book)
    cells.loc[outlier & ~anchor, "action"] = ACTION_OUTLIER_NULLED
    cells.loc[outlier & anchor, "action"] = ACTION_OUTLIER_MEDIAN

    nulled = outlier & ~anchor
    cells.loc[nulled, ["final_line", "final_left_price", "final_right_price", "final_level"]] = np.nan

    median_fill = outlier & anchor
    if median_fill.any():
        lined = median_fill & cells["market"].ne(MARKET_MONEYLINE)
        cells.loc[lined, "final_line"] = round_to_increment_signed(
            cells.loc[lined, "others_median"]
        )
        cells.loc[lined, ["final_left_price", "final_right_price"]] = -110.0
        moneyline = median_fill & cells["market"].eq(MARKET_MONEYLINE)
        home = cells.loc[moneyline, "others_median"]
        cells.loc[moneyline, "final_right_price"] = _american_from_probability(home)
        cells.loc[moneyline, "final_left_price"] = _american_from_probability(1.0 - home)
        cells.loc[lined, "final_level"] = cells.loc[lined, "final_line"]
        cells.loc[moneyline, "final_level"] = home

    # ---- Write back ------------------------------------------------------ #
    changed = cells["action"].ne(ACTION_HISTORY_SAME) & cells["action"].ne(
        ACTION_UNVERIFIED
    )
    if changed.any():
        _write_cells(
            out,
            cells.loc[changed, ["row", "market", "book"]].assign(
                line=cells.loc[changed, "final_line"],
                left_price=cells.loc[changed, "final_left_price"],
                right_price=cells.loc[changed, "final_right_price"],
            ),
        )

    season = out["season_year"] if "season_year" in out.columns else pd.Series(np.nan, index=out.index)
    cells["season_year"] = season.loc[cells["row"]].to_numpy()
    audit = cells[AUDIT_COLUMNS].reset_index(drop=True)
    return out, audit


def summarize_repair(audit: pd.DataFrame) -> pd.DataFrame:
    """Cell counts per season, market and action."""
    if audit.empty:
        return pd.DataFrame()
    return (
        audit.groupby(["season_year", "market", "action"])
        .size()
        .unstack("action", fill_value=0)
    )


def print_repair_summary(audit: pd.DataFrame) -> None:
    """Log the repair: overall action counts plus a per-book breakdown."""
    if audit.empty:
        print("Closing-line repair: no cells.")
        return
    counts = audit["action"].value_counts()
    print(
        "Closing-line repair (cells): "
        + ", ".join(f"{action}={int(count)}" for action, count in counts.items())
    )
    replaced = audit[audit["action"].eq(ACTION_HISTORY_REPLACED)]
    if not replaced.empty:
        gap = (replaced["sbr_level"] - replaced["history_level"]).abs()
        big = replaced.assign(gap=gap)
        lined = big["market"].ne(MARKET_MONEYLINE)
        print(
            "  SBR vs history on replaced cells: "
            f"{int((lined & big['gap'].ge(2.0)).sum())} line cells off by 2+ points, "
            f"{int((~lined & big['gap'].ge(0.05)).sum())} moneyline cells off by 0.05+ "
            "win probability."
        )
    outliers = audit[
        audit["action"].isin([ACTION_OUTLIER_NULLED, ACTION_OUTLIER_MEDIAN])
    ]
    if not outliers.empty:
        print(
            "  Outliers by book: "
            + ", ".join(
                f"{book}={int(count)}"
                for book, count in outliers["book"].value_counts().items()
            )
        )


def audit_repaired_closes(repaired: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Fail if a history-verified cell did not end up holding its intended quote.

    Re-reads the repaired wide frame, recomputes each cell's level and compares
    it with the audit's ``final_level``. A mismatch means the write-back went to
    the wrong columns or orientation -- exactly the silent failure that would
    put a live line back into training.
    """
    if audit.empty:
        return
    check = wide_to_long_quotes(repaired)
    check["level"] = quote_levels(check)
    merged = audit.merge(
        check[[*CELL_KEYS, "level"]], on=CELL_KEYS, how="left", validate="one_to_one"
    )
    expected = merged["action"].ne(ACTION_UNVERIFIED) & merged["final_level"].notna()
    wrong = expected & ~np.isclose(
        merged["level"].fillna(np.inf), merged["final_level"], rtol=0.0, atol=1e-6
    )
    nulled = merged["action"].eq(ACTION_OUTLIER_NULLED) & merged["level"].notna()
    if wrong.any() or nulled.any():
        sample = merged.loc[wrong | nulled, [*CELL_KEYS, "action", "final_level", "level"]]
        raise AssertionError(
            f"Closing-line repair write-back mismatch on {int((wrong | nulled).sum())} "
            f"cells:\n{sample.head(10)}"
        )
