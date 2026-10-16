import re

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import (
    BOOK_SLUG_PATTERN,
    extract_total_line_books,
    resolve_main_total_line_col,
)

# Optional legacy default list (only used when explicitly provided by caller)
BET_BOOKS: list[str] = []
# Sources used as reference baselines should not be treated as regular books.
NON_PER_BOOK_SOURCES = {"consensus_opener"}


def as_float(s: pd.Series) -> pd.Series:
    """Centralized numeric coercion to float."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def decimal_to_prob(dec_odds: pd.Series) -> pd.Series:
    """
    Decimal odds (includes stake) -> implied probability.
    p = 1 / dec
    """
    d = as_float(dec_odds)
    out = np.where(d > 1.0, 1.0 / d, np.nan)  # decimal odds must be > 1
    return pd.Series(out, index=dec_odds.index)


def no_vig_two_way_prob(p_a: pd.Series, p_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Remove vig for a two-way market by normalization.
    """
    a = as_float(p_a)
    b = as_float(p_b)
    denom = a + b
    a_nv = np.where(denom > 0, a / denom, np.nan)
    b_nv = np.where(denom > 0, b / denom, np.nan)
    return pd.Series(a_nv, index=p_a.index), pd.Series(b_nv, index=p_b.index)


def safe_mean(x: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return x.mean(axis=1, skipna=True)


def safe_std(x: pd.DataFrame | pd.Series, ddof: int = 0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(np.nan, index=x.index)
    return x.std(axis=1, skipna=True, ddof=ddof)


def safe_range(x: pd.DataFrame) -> pd.Series:
    return x.max(axis=1, skipna=True) - x.min(axis=1, skipna=True)


def safe_iqr(x: pd.DataFrame) -> pd.Series:
    """Interquartile range (Q3 - Q1) for robust dispersion measure."""
    if isinstance(x, pd.Series):
        return pd.Series(np.nan, index=x.index)
    q25 = x.quantile(0.25, axis=1)
    q75 = x.quantile(0.75, axis=1)
    return q75 - q25


def safe_mad(x: pd.DataFrame) -> pd.Series:
    """Median absolute deviation across columns (row-wise)."""
    if isinstance(x, pd.Series):
        return pd.Series(np.nan, index=x.index)
    med = x.median(axis=1, skipna=True)
    abs_dev = x.sub(med, axis=0).abs()
    return abs_dev.median(axis=1, skipna=True)


def engineer_odds_features(
    df: pd.DataFrame,
    prefix: str = "ODDS_",
    books: list[str] | None = None,
) -> pd.DataFrame:
    """
    Odds feature engineering focused on close-time TOTAL/SPREAD/ML data.
    Prices are expected to already be decimal odds.

    Main outputs:
      - Robust close total consensus: line median/mean + dispersion (std/iqr/mad/range)
      - Per-book total skew features: log(price_over/price_under), prob_diff_novig, vig
      - Coverage signals: number/ratio of books with total line and total prices present
      - Minimal spread+moneyline summaries (favorite strength and uncertainty proxies)
      - Reference baseline helper:
          close_total_consensus

    Note:
      - Consensus opener inputs are still used when available, but exposed as "ref" fields.
      - Move-from-opener features are intentionally excluded to avoid train/inference mismatch.

    Expected (if available):
      - Total current lines per book (UPPERCASE, from merge_total_spread_moneyline):
          ODDS_TOTAL_LINE_{book}   (e.g., ODDS_TOTAL_LINE_betmgm)
      - Total current prices per book (lowercase, from merge_remaining):
          total_{book}_price_over, total_{book}_price_under   (DECIMAL)
      - Consensus total opener line (used as reference baseline):
          ODDS_TOTAL_LINE_consensus_opener
      - Consensus total opener prices (used as reference baseline):
          total_consensus_opener_price_over, total_consensus_opener_price_under   (DECIMAL)
      - Spread per book (from merge_remaining):
          spread_{book}_line_home, spread_{book}_line_away
          spread_consensus_opener_line_home, spread_consensus_opener_line_away   (optional)
      - Moneyline per book (from merge_remaining, DECIMAL):
          ml_{book}_price_home, ml_{book}_price_away
    """
    out = df.copy()
    # Dictionary to collect all new columns to avoid fragmentation
    new_cols = {}

    if books is None:
        # Raw per-book columns only. By the time this runs the frame also holds
        # each book's rolling means, trends and season averages
        # (``ODDS_TOTAL_LINE_bet365_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME``);
        # matching any suffix once read ~205 of them as books, so the books'
        # consensus, dispersion and coverage ran over ~211 "books" instead of 6.
        inferred_books = set(extract_total_line_books(out))
        raw_book_columns = (
            rf"total_({BOOK_SLUG_PATTERN})_price_(?:over|under)",
            rf"spread_({BOOK_SLUG_PATTERN})_line_(?:home|away)",
            rf"ml_({BOOK_SLUG_PATTERN})_price_(?:home|away)",
        )
        for col in out.columns:
            for pattern in raw_book_columns:
                m = re.fullmatch(pattern, col)
                if m:
                    inferred_books.add(m.group(1))
                    break

        books = sorted(inferred_books)
    else:
        books = list(dict.fromkeys(books))

    # Avoid redundant/self-referential features by excluding baseline sources
    # from per-book loops (they are handled explicitly in section 0/2).
    books = [b for b in books if b not in NON_PER_BOOK_SOURCES]

    # -----------------------------
    # 0) Consensus total reference line/probability
    # -----------------------------
    # Source is still consensus opener data, but this script treats it as a reference baseline.
    cons_ref_line_col = "ODDS_TOTAL_LINE_consensus_opener"
    cons_ref_price_over = "total_consensus_opener_price_over"
    cons_ref_price_under = "total_consensus_opener_price_under"

    if cons_ref_line_col in out.columns:
        new_cols[f"{prefix}consensus_total_ref_line_mid"] = as_float(
            out[cons_ref_line_col]
        )
    else:
        # Fallback only if reference line was not merged with canonical name.
        main_total_line = resolve_main_total_line_col(out)
        if main_total_line is not None:
            new_cols[f"{prefix}consensus_total_ref_line_mid"] = as_float(
                out[main_total_line]
            )

    if cons_ref_price_over in out.columns and cons_ref_price_under in out.columns:
        p_over = decimal_to_prob(out[cons_ref_price_over])
        p_under = decimal_to_prob(out[cons_ref_price_under])
        p_over_nv, p_under_nv = no_vig_two_way_prob(p_over, p_under)
        new_cols[f"{prefix}consensus_total_ref_prob_over_novig"] = p_over_nv
        new_cols[f"{prefix}consensus_total_ref_vig"] = (p_over + p_under) - 1.0
        new_cols[f"{prefix}consensus_total_ref_prob_diff_novig"] = (
            p_over_nv - p_under_nv
        )

    # -----------------------------
    # 1) TOTAL lines and TOTAL prices (per book, close-time robust summary)
    # -----------------------------
    total_line_mids = []
    total_prob_diff_novig = []
    total_vigs = []
    total_price_log_ratios = []
    total_line_presence_series = []
    total_price_presence_series = []

    for book in books:
        total_line_col = f"ODDS_TOTAL_LINE_{book}"
        price_over = f"total_{book}_price_over"
        price_under = f"total_{book}_price_under"

        has_lines = total_line_col in out.columns
        has_prices = price_over in out.columns and price_under in out.columns

        if has_lines:
            col_mid = f"{prefix}book_total_line_mid_{book}"
            new_cols[col_mid] = as_float(out[total_line_col])
            total_line_mids.append(col_mid)
            total_line_presence_series.append(new_cols[col_mid].notna().astype(float))

        if has_prices:
            dec_over = as_float(out[price_over])
            dec_under = as_float(out[price_under])
            total_price_presence_series.append(
                (dec_over.notna() & dec_under.notna()).astype(float)
            )

            # Keep log ratio for price skew (drop decimal diff to reduce redundancy).
            col_log_ratio = f"{prefix}book_total_price_log_ratio_over_div_under_{book}"
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log(dec_over / dec_under)
            new_cols[col_log_ratio] = pd.Series(log_ratio, index=out.index).replace(
                [np.inf, -np.inf], np.nan
            )
            total_price_log_ratios.append(col_log_ratio)

            # Keep only one no-vig probability representation per book.
            p_over = decimal_to_prob(out[price_over])
            p_under = decimal_to_prob(out[price_under])
            p_over_nv, p_under_nv = no_vig_two_way_prob(p_over, p_under)
            col_p_diff = f"{prefix}book_total_prob_diff_novig_{book}"
            col_vig = f"{prefix}book_total_vig_{book}"
            new_cols[col_p_diff] = p_over_nv - p_under_nv
            new_cols[col_vig] = (p_over + p_under) - 1.0
            total_prob_diff_novig.append(col_p_diff)
            total_vigs.append(col_vig)

    # Coverage features (missingness is signal)
    if total_line_presence_series:
        tmp_presence = pd.concat(total_line_presence_series, axis=1)
        new_cols[f"{prefix}n_books_total_line_present"] = tmp_presence.sum(axis=1)
        new_cols[f"{prefix}n_books_total_line_present_ratio"] = new_cols[
            f"{prefix}n_books_total_line_present"
        ] / len(total_line_presence_series)
    if total_price_presence_series:
        tmp_presence = pd.concat(total_price_presence_series, axis=1)
        new_cols[f"{prefix}n_books_total_price_present"] = tmp_presence.sum(axis=1)
        new_cols[f"{prefix}n_books_total_price_present_ratio"] = new_cols[
            f"{prefix}n_books_total_price_present"
        ] / len(total_price_presence_series)

    # Aggregates across books: current total line levels
    if total_line_mids:
        tmp_lines = (
            pd.DataFrame({col: new_cols[col] for col in total_line_mids})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}total_line_books_mean"] = safe_mean(tmp_lines)
        new_cols[f"{prefix}total_line_books_median"] = tmp_lines.median(
            axis=1, skipna=True
        )
        new_cols[f"{prefix}total_line_books_std"] = safe_std(tmp_lines)
        new_cols[f"{prefix}total_line_books_range"] = safe_range(tmp_lines)
        new_cols[f"{prefix}total_line_books_iqr"] = safe_iqr(tmp_lines)
        new_cols[f"{prefix}total_line_books_mad"] = safe_mad(tmp_lines)

    # Aggregates across books: probability skew (prob_diff only)
    if total_prob_diff_novig:
        tmp_d = (
            pd.DataFrame({col: new_cols[col] for col in total_prob_diff_novig})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}total_prob_diff_books_mean"] = safe_mean(tmp_d)
        new_cols[f"{prefix}total_prob_diff_books_std"] = safe_std(tmp_d)
        new_cols[f"{prefix}total_prob_diff_books_range"] = safe_range(tmp_d)
        new_cols[f"{prefix}total_prob_diff_books_iqr"] = safe_iqr(tmp_d)
        new_cols[f"{prefix}total_prob_diff_books_mad"] = safe_mad(tmp_d)

    # Aggregates across books: vig level and dispersion
    if total_vigs:
        tmp_v = (
            pd.DataFrame({col: new_cols[col] for col in total_vigs})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}total_vig_books_mean"] = safe_mean(tmp_v)
        new_cols[f"{prefix}total_vig_books_median"] = tmp_v.median(axis=1, skipna=True)
        new_cols[f"{prefix}total_vig_books_std"] = safe_std(tmp_v)
        new_cols[f"{prefix}total_vig_books_iqr"] = safe_iqr(tmp_v)
        new_cols[f"{prefix}total_vig_books_mad"] = safe_mad(tmp_v)

    # Aggregates across books: log-ratio (market skew)
    if total_price_log_ratios:
        tmp_lr = (
            pd.DataFrame({col: new_cols[col] for col in total_price_log_ratios})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}total_price_log_ratio_books_mean"] = safe_mean(tmp_lr)
        new_cols[f"{prefix}total_price_log_ratio_books_std"] = safe_std(tmp_lr)
        new_cols[f"{prefix}total_price_log_ratio_books_range"] = safe_range(tmp_lr)
        new_cols[f"{prefix}total_price_log_ratio_books_iqr"] = safe_iqr(tmp_lr)
        new_cols[f"{prefix}total_price_log_ratio_books_mad"] = safe_mad(tmp_lr)

    # -----------------------------
    # 2) SPREAD features using FAVORITE logic (fixed calculation)
    # -----------------------------
    spread_home_lines = []
    spread_fav_abs = []
    spread_home_is_fav = []
    # Consensus spread reference (optional) - use favorite logic
    if (
        "spread_consensus_opener_line_home" in out.columns
        and "spread_consensus_opener_line_away" in out.columns
    ):
        h_open = as_float(out["spread_consensus_opener_line_home"])
        a_open = as_float(out["spread_consensus_opener_line_away"])
        # Favorite is the more negative line (keep as Series for safety)
        fav_open = pd.Series(np.minimum(h_open, a_open), index=out.index)
        new_cols[f"{prefix}spread_consensus_ref_fav_line"] = fav_open
        new_cols[f"{prefix}spread_consensus_ref_fav_abs"] = np.abs(fav_open)
        new_cols[f"{prefix}spread_consensus_ref_home_is_fav"] = (
            h_open < a_open
        ).astype(float)

    for book in books:
        l_home = f"spread_{book}_line_home"
        l_away = f"spread_{book}_line_away"
        if l_home in out.columns and l_away in out.columns:
            h_line = as_float(out[l_home])
            a_line = as_float(out[l_away])

            # Keep home line (encodes direction and magnitude)
            col_home = f"{prefix}spread_home_line_{book}"
            new_cols[col_home] = h_line
            spread_home_lines.append(col_home)

            # Favorite line (more negative, keep as Series)
            fav_line = pd.Series(np.minimum(h_line, a_line), index=out.index)
            col_fav_abs = f"{prefix}spread_fav_abs_{book}"
            new_cols[col_fav_abs] = np.abs(fav_line)
            spread_fav_abs.append(col_fav_abs)

            # Home is favorite indicator
            col_home_fav = f"{prefix}spread_home_is_fav_{book}"
            new_cols[col_home_fav] = (h_line < a_line).astype(float)
            spread_home_is_fav.append(col_home_fav)

    # Aggregates across books
    if spread_home_lines:
        tmp = (
            pd.DataFrame({col: new_cols[col] for col in spread_home_lines})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}spread_home_books_mean"] = safe_mean(tmp)
        new_cols[f"{prefix}spread_home_books_std"] = safe_std(tmp)

    if spread_fav_abs:
        tmp = (
            pd.DataFrame({col: new_cols[col] for col in spread_fav_abs})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}spread_fav_abs_books_mean"] = safe_mean(tmp)
        new_cols[f"{prefix}spread_fav_abs_books_std"] = safe_std(tmp)
        new_cols[f"{prefix}spread_fav_abs_books_range"] = safe_range(tmp)

    if spread_home_is_fav:
        tmp = (
            pd.DataFrame({col: new_cols[col] for col in spread_home_is_fav})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        new_cols[f"{prefix}spread_home_is_fav_books_mean"] = safe_mean(tmp)

    # -----------------------------
    # 3) Keep a minimal set of MONEYLINE features (implied win prob, no vig)
    # -----------------------------
    ml_home_probs = []
    ml_vigs = []

    for book in books:
        p_home = f"ml_{book}_price_home"
        p_away = f"ml_{book}_price_away"

        if p_home in out.columns and p_away in out.columns:
            ph = decimal_to_prob(out[p_home])
            pa = decimal_to_prob(out[p_away])
            ph_nv, pa_nv = no_vig_two_way_prob(ph, pa)

            col_home = f"{prefix}ml_home_prob_novig_{book}"
            col_vig = f"{prefix}ml_vig_{book}"

            new_cols[col_home] = ph_nv
            new_cols[col_vig] = (ph + pa) - 1.0

            ml_home_probs.append(col_home)
            ml_vigs.append(col_vig)

    if ml_home_probs:
        ml_home_mean = safe_mean(
            pd.DataFrame({col: new_cols[col] for col in ml_home_probs})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        ml_away_mean = 1.0 - ml_home_mean
        new_cols[f"{prefix}ml_favorite_prob"] = pd.concat(
            [ml_home_mean, ml_away_mean], axis=1
        ).max(axis=1, skipna=True)
        new_cols[f"{prefix}ml_win_prob_gap"] = (ml_home_mean - ml_away_mean).abs()

    if ml_vigs:
        new_cols[f"{prefix}ml_vig_books_mean"] = safe_mean(
            pd.DataFrame({col: new_cols[col] for col in ml_vigs})
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )

    # -----------------------------
    # 4) Interactions and derived features for predicting residual error
    # -----------------------------

    # Basic interactions
    if (
        f"{prefix}total_line_books_mean" in new_cols
        and f"{prefix}total_prob_diff_books_mean" in new_cols
    ):
        new_cols[f"{prefix}total_line_x_prob_diff"] = (
            new_cols[f"{prefix}total_line_books_mean"]
            * new_cols[f"{prefix}total_prob_diff_books_mean"]
        )

    # Use corrected spread_fav_abs instead of spread_mid
    if (
        f"{prefix}total_line_books_mean" in new_cols
        and f"{prefix}spread_fav_abs_books_mean" in new_cols
    ):
        new_cols[f"{prefix}total_line_x_spread_fav_abs"] = (
            new_cols[f"{prefix}total_line_books_mean"]
            * new_cols[f"{prefix}spread_fav_abs_books_mean"]
        )

    # Disagreement + vig interaction (uncertainty proxy)
    if (
        f"{prefix}total_line_books_std" in new_cols
        and f"{prefix}total_vig_books_mean" in new_cols
    ):
        new_cols[f"{prefix}line_std_x_vig"] = (
            new_cols[f"{prefix}total_line_books_std"]
            * new_cols[f"{prefix}total_vig_books_mean"]
        )

    # -----------------------------
    # 5) Implied team totals from total and spread
    # -----------------------------
    # From consensus total T and representative spread S (home line, signed):
    # implied_home_points = T/2 - S/2
    # implied_away_points = T - implied_home
    if (
        f"{prefix}total_line_books_mean" in new_cols
        and f"{prefix}spread_home_books_mean" in new_cols
    ):
        total_line = new_cols[f"{prefix}total_line_books_mean"]
        spread_home = new_cols[f"{prefix}spread_home_books_mean"]

        # Home spread is negative when home is favored
        # implied_home = T/2 - S/2 (when S is negative, home gets more points)
        new_cols[f"{prefix}implied_home_points"] = total_line / 2.0 - spread_home / 2.0
        new_cols[f"{prefix}implied_away_points"] = (
            total_line - new_cols[f"{prefix}implied_home_points"]
        )

        # Ratio of implied totals (useful for lopsided matchups)
        with np.errstate(divide="ignore", invalid="ignore"):
            implied_ratio = (
                new_cols[f"{prefix}implied_home_points"]
                / new_cols[f"{prefix}implied_away_points"]
            )
        new_cols[f"{prefix}implied_points_ratio"] = pd.Series(
            implied_ratio, index=out.index
        ).replace([np.inf, -np.inf], np.nan)
        # Max/min implied points
        new_cols[f"{prefix}implied_points_max"] = pd.DataFrame(
            {
                "home": new_cols[f"{prefix}implied_home_points"],
                "away": new_cols[f"{prefix}implied_away_points"],
            }
        ).max(axis=1)
        new_cols[f"{prefix}implied_points_min"] = pd.DataFrame(
            {
                "home": new_cols[f"{prefix}implied_home_points"],
                "away": new_cols[f"{prefix}implied_away_points"],
            }
        ).min(axis=1)
        new_cols[f"{prefix}implied_points_gap"] = (
            new_cols[f"{prefix}implied_points_max"]
            - new_cols[f"{prefix}implied_points_min"]
        )

    # -----------------------------
    # 6) Reference baseline helper (no target-derived calculations here)
    # -----------------------------
    # Prefer robust close-consensus proxy (median across books), fallback to mean.
    if f"{prefix}total_line_books_median" in new_cols:
        new_cols[f"{prefix}close_total_consensus"] = as_float(
            new_cols[f"{prefix}total_line_books_median"]
        )
    elif f"{prefix}total_line_books_mean" in new_cols:
        new_cols[f"{prefix}close_total_consensus"] = as_float(
            new_cols[f"{prefix}total_line_books_mean"]
        )

    # Concatenate all new columns at once to avoid fragmentation
    if new_cols:
        new_cols_df = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, new_cols_df], axis=1)

    return out
