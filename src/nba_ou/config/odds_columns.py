from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from nba_ou.config.settings import SETTINGS

DEFAULT_MAIN_BOOK = "consensus_opener"
BOOK_ALIASES = {
    "bet_365": "bet365",
}


#: Books collected into the odds stores (Supabase ``odds_sportsbook`` and the
#: Aiven line history) but not yet admitted as model features. Both dataset
#: builders drop them at read time, so storing a new book never changes a
#: training frame silently.
#:
#: BetRivers is here because its SBR history only starts with the 2021-22
#: season: admitting it unflagged would put a book whose mere presence encodes
#: the season into every per-book and consensus column (the same problem
#: ``PARTIAL_COVERAGE_BOOKS`` handles for Fanatics). Remove a slug from this
#: tuple only as a deliberate, ablated feature change.
HISTORY_ONLY_BOOKS: tuple[str, ...] = ("betrivers",)


def is_book_column(column: str, book: str) -> bool:
    """True if ``column`` belongs to ``book`` (``total_betrivers_line_over``, ...).

    Matches the slug as a whole ``_``-delimited token, so ``bet365`` never
    matches a hypothetical ``bet3650`` and a slug that is a prefix of another
    cannot capture the longer book's columns.
    """
    tokens = column.lower().split("_")
    slug = book.lower().split("_")
    width = len(slug)
    return any(tokens[i : i + width] == slug for i in range(len(tokens) - width + 1))


def drop_history_only_book_columns(
    df: pd.DataFrame, books: Iterable[str] = HISTORY_ONLY_BOOKS
) -> pd.DataFrame:
    """Drop the columns of books that are stored but not yet model features."""
    books = tuple(books)
    drop = [c for c in df.columns if any(is_book_column(c, b) for b in books)]
    return df.drop(columns=drop) if drop else df


def get_main_book() -> str:
    configured = getattr(SETTINGS, "main_sportsbook", None)
    if configured is None:
        return DEFAULT_MAIN_BOOK
    configured = str(configured).strip()
    if not configured:
        return DEFAULT_MAIN_BOOK
    return BOOK_ALIASES.get(configured, configured)


#: Marker prefix for every column derived from betting-odds data, unified across both training
#: pipelines so odds features can be selected as easily as leakage-safe ``_BEFORE`` columns.
ODDS_COLUMN_PREFIXES: tuple[str, ...] = ("ODDS_",)


def is_odds_column(column: str) -> bool:
    """True if ``column`` carries the unified odds-derived marker prefix."""
    return column.startswith(ODDS_COLUMN_PREFIXES)


#: Column-name shapes that are odds-derived *by construction*: every column
#: whose name starts with one of these comes from a bookmaker market, whichever
#: route it took through the pipeline. Uppercase entries are the canonical
#: post-merge names built by ``total_line_col`` and friends; lowercase entries
#: are the raw per-book market columns as they arrive from the odds databases
#: (``total_<book>_price_over``, ``spread_<book>_line_home``, ``ml_<book>_price_home``,
#: ``moneyline_pct_bets_home``, ...) plus every rolling ``_BEFORE`` variant built
#: on top of them.
#:
#: This list is what makes the ``ODDS_`` marker an enforceable invariant rather
#: than a convention that holds only for the paths someone remembered to update:
#: a new odds feature named in any of these shapes is caught even if it never
#: passes through the rename helpers.
#:
#: Deliberately NOT included: ``DIFF_FROM_`` and ``IS_OVER_``. Both are
#: odds-derived by any reasonable definition, but they are named after the
#: *target* relationship rather than the market and are not currently prefixed
#: anywhere in the pipeline. Bringing them in is a separate, deliberate rename --
#: not something this guard should force silently.
ODDS_SHAPED_PREFIXES: tuple[str, ...] = (
    "TOTAL_LINE_",
    "SPREAD_",
    "MONEYLINE_",
    "total_",
    "spread_",
    "ml_",
    "moneyline_",
)


#: Columns named after a TARGET relationship rather than after a market, which
#: therefore must NOT be swept up by the ``ODDS_`` prefix invariant even though
#: their names collide with an odds-shaped prefix.
#:
#: ``SPREAD_ERROR`` is the spread market's residual target (HOME_MARGIN minus the
#: anchor spread). It begins with ``SPREAD_`` and so matches ODDS_SHAPED_PREFIXES
#: by accident of naming, but prefixing it would make the training target look
#: like a market feature to every consumer that selects on ``is_odds_column`` --
#: exactly backwards. This is the same reasoning already applied to ``DIFF_FROM_``
#: and ``IS_OVER_`` above; those simply do not happen to match a prefix.
TARGET_NAMED_COLUMNS: frozenset[str] = frozenset({"SPREAD_ERROR"})


def is_odds_shaped_column(column: str) -> bool:
    """True if ``column`` is odds-derived by its name shape, prefix or not."""
    if column in TARGET_NAMED_COLUMNS:
        return False
    return column.startswith(ODDS_SHAPED_PREFIXES)


def strip_odds_prefix(column: str) -> str:
    """Return ``column`` without its leading ``ODDS_`` marker, if it has one."""
    return column.removeprefix("ODDS_") if is_odds_column(column) else column


def find_unprefixed_odds_columns(columns: Iterable[str]) -> list[str]:
    """Odds-derived columns that are missing the unified ``ODDS_`` marker."""
    return [
        column
        for column in columns
        if is_odds_shaped_column(column) and not is_odds_column(column)
    ]


def apply_odds_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix every odds-shaped column with ``ODDS_``.

    Idempotent: columns that already carry the marker are left alone, so this
    can be applied at the end of any pipeline without tracking whether an
    earlier stage already ran it.

    Must run **after** the last stage that reads raw market columns by their
    unprefixed names -- notably ``engineer_odds_features``, which resolves its
    inputs as ``total_<book>_price_over`` and would silently emit fewer features
    if the rename had already happened.
    """
    rename = {
        column: f"ODDS_{column}" for column in find_unprefixed_odds_columns(df.columns)
    }
    return df.rename(columns=rename) if rename else df


def assert_odds_columns_prefixed(columns: Iterable[str], *, context: str) -> None:
    """Fail if any odds-derived column reached the output without ``ODDS_``.

    The point of the marker is that ``[c for c in df.columns if is_odds_column(c)]``
    selects *every* odds feature. That only holds if nothing can slip through, so
    this raises rather than warns: an odds column arriving without the marker is
    invisible to every consumer that selects on it, and nothing else would ever
    surface the omission.
    """
    offenders = find_unprefixed_odds_columns(columns)
    if offenders:
        shown = ", ".join(offenders[:20])
        more = f" (+{len(offenders) - 20} more)" if len(offenders) > 20 else ""
        raise ValueError(
            f"{len(offenders)} odds-derived column(s) reached {context} without the "
            f"unified 'ODDS_' prefix: {shown}{more}. Every column named like a "
            "bookmaker market must carry the marker so it can be selected via "
            "nba_ou.config.odds_columns.is_odds_column(). Route the frame through "
            "apply_odds_prefix() or name the feature with the prefix at source."
        )


def total_line_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"ODDS_TOTAL_LINE_{b}"


def spread_col(book: str | None = None) -> str:
    """The RAW per-book spread column: a HOME HANDICAP, negative when home is favoured.

    Verified on real games: this column correlates -0.457 with the realised home
    margin. Do not use it as a line without converting -- see
    ``spread_line_home_col`` and ``nba_ou.config.market_columns``.
    """
    b = book or get_main_book()
    return f"ODDS_SPREAD_{b}"


def spread_line_home_col(book: str | None = None) -> str:
    """The CANONICAL spread column: market-implied final HOME MARGIN.

    Deliberately shaped like ``total_line_col`` (``ODDS_TOTAL_LINE_<book>``) so
    the spread residual target resolves through exactly the same machinery the
    totals residual target already uses.

    It carries the ``ODDS_`` marker because it IS a market quote and every
    odds-derived column in this repo must be selectable via ``is_odds_column``.
    The bare name ``SPREAD_LINE_HOME`` is the concept; this is its column.
    """
    b = book or get_main_book()
    return f"ODDS_SPREAD_LINE_HOME_{b}"


def extract_spread_line_home_books(df: pd.DataFrame) -> list[str]:
    """Books with a canonical ``ODDS_SPREAD_LINE_HOME_<book>`` column, sorted."""
    books = set()
    for col in df.columns:
        m = re.match(r"^ODDS_SPREAD_LINE_HOME_(.+)$", col)
        if m:
            books.add(m.group(1))
    return sorted(books)


def resolve_main_spread_line_col(
    df: pd.DataFrame, book: str | None = None
) -> str | None:
    """Resolve the canonical spread line column for ``book``.

    Unlike ``resolve_main_total_line_col`` this does NOT fall back to another
    book. The spread target is defined as "home margin minus the Bet365 line";
    silently answering with a different book's line would change what the target
    MEANS from row to row, which is a worse outcome than a missing column.
    """
    preferred = spread_line_home_col(book)
    return preferred if preferred in df.columns else None


def moneyline_col(book: str | None = None) -> str:
    b = book or get_main_book()
    return f"ODDS_MONEYLINE_{b}"


def extract_total_line_books(df: pd.DataFrame) -> list[str]:
    """
    Infer sportsbook names from columns shaped as ODDS_TOTAL_LINE_<book>.
    Returns books in deterministic sorted order.
    """
    books = set()
    for col in df.columns:
        m = re.match(r"^ODDS_TOTAL_LINE_(.+)$", col)
        if m:
            books.add(m.group(1))
    return sorted(books)


def total_line_over_col_raw(book: str | None = None) -> str:
    """
    Get the raw odds data column name for total line over.
    Format: total_{book}_line_over (used in odds data before merge).
    """
    b = book or get_main_book()
    return f"ODDS_TOTAL_LINE_{b}"


def resolve_main_total_line_col(
    df: pd.DataFrame, book: str | None = None
) -> str | None:
    """
    Resolve the active total-line column.
    Prefer configured book; fallback to first available ODDS_TOTAL_LINE_* column.
    """
    preferred = total_line_col(book)
    if preferred in df.columns:
        return preferred

    candidates = extract_total_line_books(df)
    if not candidates:
        return None
    return f"ODDS_TOTAL_LINE_{candidates[0]}"
