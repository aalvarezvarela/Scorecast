"""Normalize raw per-book market columns into canonical, market-neutral form.

Runs once, at the end of ``select_training_columns``, so BOTH datasets that pass
through that gate (the closing dataset via ``create_df_to_predict`` and the
intermediate dataset's base features via ``create_base_game_features``) get the
same canonical columns from the same code. Two call sites deriving the same
quantity separately is how the two halves of a pipeline end up disagreeing about
a sign.

What it produces:

* ``ODDS_SPREAD_LINE_HOME_<book>`` -- every book's spread as an implied HOME
  margin (see ``nba_ou.config.market_columns`` for the verified conventions).
* Cross-book spread consensus features, median-based.
* ``ODDS_ML_PRICE_HOME`` / ``_AWAY`` and their de-vigged probabilities.

It derives NO moneyline target. Moneyline support here is data readiness only.
"""

from __future__ import annotations

import pandas as pd
from nba_ou.config.market_columns import (
    ML_PRICE_AWAY_COL,
    ML_PRICE_HOME_COL,
    ML_PROB_AWAY_NOVIG_COL,
    ML_PROB_HOME_NOVIG_COL,
    PLAUSIBLE_SPREAD_ABS_MAX,
    SPREAD_BET365_MINUS_CONSENSUS_COL,
    SPREAD_BOOK_COUNT_COL,
    SPREAD_CONSENSUS_MEDIAN_COL,
    SPREAD_CROSS_BOOK_RANGE_COL,
    SPREAD_CROSS_BOOK_STD_COL,
    SPREAD_PRICE_AWAY_COL,
    SPREAD_PRICE_HOME_COL,
    TARGET_ANCHOR_BOOK,
    spread_line_home_from_handicap,
)
from nba_ou.config.odds_columns import spread_line_home_col

#: Raw per-book spread sources, in HOME-HANDICAP orientation (negative when the
#: home team is favoured). Each becomes ``ODDS_SPREAD_LINE_HOME_<book>`` by
#: negation. Both spellings are listed because the closing CSV carries the
#: ``_TEAM_HOME`` suffixed form for the main book and the ``_line_home`` form for
#: the rest.
_HANDICAP_SOURCE_TEMPLATES: tuple[str, ...] = (
    "ODDS_SPREAD_{book}_TEAM_HOME",
    "ODDS_spread_{book}_line_home",
    "spread_{book}_line_home",
    "ODDS_spread_home_line_{book}",
)

#: Books to look for. Ordering only affects column order in the output.
_KNOWN_BOOKS: tuple[str, ...] = (
    "bet365",
    "betmgm",
    "fanduel",
    "draftkings",
    "fanatics_sportsbook",
    "betrivers",
    "consensus_opener",
)

_ML_PRICE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("ODDS_MONEYLINE_{book}_TEAM_HOME", "ODDS_MONEYLINE_{book}_TEAM_AWAY"),
    ("ODDS_ml_{book}_price_home", "ODDS_ml_{book}_price_away"),
    ("ml_{book}_price_home", "ml_{book}_price_away"),
)

_SPREAD_PRICE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("ODDS_spread_{book}_price_home", "ODDS_spread_{book}_price_away"),
    ("spread_{book}_price_home", "spread_{book}_price_away"),
)


def _first_present(
    df: pd.DataFrame, templates: tuple[str, ...], book: str
) -> str | None:
    for template in templates:
        name = template.format(book=book)
        if name in df.columns:
            return name
    return None


def _first_present_pair(
    df: pd.DataFrame, templates: tuple[tuple[str, str], ...], book: str
) -> tuple[str, str] | None:
    for home_template, away_template in templates:
        home = home_template.format(book=book)
        away = away_template.format(book=book)
        if home in df.columns and away in df.columns:
            return home, away
    return None


def add_canonical_spread_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ODDS_SPREAD_LINE_HOME_<book>`` plus cross-book consensus features."""
    out = df.copy()
    produced: list[str] = []

    for book in _KNOWN_BOOKS:
        source = _first_present(out, _HANDICAP_SOURCE_TEMPLATES, book)
        if source is None:
            continue
        target = spread_line_home_col(book)
        # Every known source is a handicap; the ONE column in this repo that is
        # already an implied margin lives in the intermediate snapshot panel and
        # never reaches this function.
        out[target] = spread_line_home_from_handicap(out[source])
        produced.append(target)

    if not produced:
        return out

    # Consensus is built from PLAUSIBLE quotes only. Per-book spread columns in
    # the real closing data contain values like -120 and +77.5 where a price bled
    # into the line field; a median is robust to a few of those, but excluding
    # them outright is cheap and makes the count column honest.
    frame = out[produced].apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame.abs() <= PLAUSIBLE_SPREAD_ABS_MAX)

    out[SPREAD_BOOK_COUNT_COL] = frame.notna().sum(axis=1).astype("int64")
    out[SPREAD_CONSENSUS_MEDIAN_COL] = frame.median(axis=1, skipna=True)
    out[SPREAD_CROSS_BOOK_STD_COL] = frame.std(axis=1, skipna=True)
    out[SPREAD_CROSS_BOOK_RANGE_COL] = frame.max(axis=1, skipna=True) - frame.min(
        axis=1, skipna=True
    )

    anchor = spread_line_home_col(TARGET_ANCHOR_BOOK)
    if anchor in out.columns:
        out[SPREAD_BET365_MINUS_CONSENSUS_COL] = pd.to_numeric(
            out[anchor], errors="coerce"
        ) - out[SPREAD_CONSENSUS_MEDIAN_COL]

    prices = _first_present_pair(out, _SPREAD_PRICE_TEMPLATES, TARGET_ANCHOR_BOOK)
    if prices is not None:
        home_price, away_price = prices
        out[SPREAD_PRICE_HOME_COL] = pd.to_numeric(out[home_price], errors="coerce")
        out[SPREAD_PRICE_AWAY_COL] = pd.to_numeric(out[away_price], errors="coerce")

    return out


def devig_two_way_prices(
    home_price: pd.Series, away_price: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Proportional de-vig of a two-way DECIMAL price pair into probabilities.

    Returns ``(p_home, p_away)`` summing to 1 where both prices are valid, NaN
    elsewhere. Proportional (rather than shin or logarithmic) because it is what
    the snapshot panel already uses, and readiness work should not introduce a
    second, differently-biased devig into the same repo.
    """
    home = pd.to_numeric(home_price, errors="coerce")
    away = pd.to_numeric(away_price, errors="coerce")
    valid = (home > 1.0) & (away > 1.0)

    raw_home = 1.0 / home.where(valid)
    raw_away = 1.0 / away.where(valid)
    overround = raw_home + raw_away
    return raw_home / overround, raw_away / overround


def add_canonical_moneyline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical anchor-book moneyline prices and de-vigged probabilities.

    Verified on real games: ``ODDS_MONEYLINE_bet365_TEAM_HOME`` is the HOME
    price in DECIMAL odds (when it is the cheaper side the home team went on to
    win 68.4% of the time; when the away price is cheaper, 33.7%).

    No label and no strategy: this exists so a moneyline model can be added later
    without another upstream redesign.
    """
    out = df.copy()
    prices = _first_present_pair(out, _ML_PRICE_TEMPLATES, TARGET_ANCHOR_BOOK)
    if prices is None:
        return out

    home_price, away_price = prices
    out[ML_PRICE_HOME_COL] = pd.to_numeric(out[home_price], errors="coerce")
    out[ML_PRICE_AWAY_COL] = pd.to_numeric(out[away_price], errors="coerce")

    prob_home, prob_away = devig_two_way_prices(
        out[ML_PRICE_HOME_COL], out[ML_PRICE_AWAY_COL]
    )
    out[ML_PROB_HOME_NOVIG_COL] = prob_home
    out[ML_PROB_AWAY_NOVIG_COL] = prob_away
    return out


def add_canonical_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Both markets, in one call. Safe to run on a frame missing either."""
    return add_canonical_moneyline_columns(add_canonical_spread_columns(df))
