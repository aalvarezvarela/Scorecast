"""News-anchored market reaction (G2) and cross-market coherence (G3).

Both are built per (game, snapshot) from the raw ticks, read with the same
as-of rule as the snapshot panel: the last tick at least ``m`` minutes before
tip. Levels are ``snapshots.market_level`` -- raw total, raw expected home
margin, de-vigged home win probability -- so every move here is comparable
with the existing movement family. Reading ticks directly, rather than the
panel's ``move_last_<w>`` columns, keeps these features independent of the
configured ``windows`` and lets the pre-news anchor sit at any instant.

Measurements behind the definitions: ``docs/intermediate_market_dynamics_plan.md``
sections 1.4-1.5 and 3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_ou.data_processing.injury_status.news import (
    MATERIAL_NEWS_POINTS,
    InjuryNewsTimeline,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

from .normalization import MARGIN_SIGMA, devig_two_way
from .snapshots import market_level, resolve_line
from .walk_forward import walk_forward_least_squares

MARKET_SHORT = {MARKET_TOTALS: "TOT", MARKET_SPREAD: "SPR", MARKET_MONEYLINE: "ML"}
#: How news is signed per market: spread and moneyline levels rise when the home
#: side gets relatively stronger (the away team loses more), totals fall when
#: either team loses scoring.
NEWS_SIGN = {MARKET_TOTALS: "total", MARKET_SPREAD: "side", MARKET_MONEYLINE: "side"}

#: Material news older than this is not "recent" for the reaction features.
NEWS_LOOKBACK_MINUTES = 480
#: The pre-news anchor sits this long before the news, since much of the market
#: move precedes the report (plan section 1.1/1.4: correlation plateaus by 60).
PRE_NEWS_MINUTES = 60
BETA_WINDOW_MINUTES = 60
BETA_MIN_TRAIN_ROWS = 300
MOVE_WINDOW_MINUTES = 60
UNEXPLAINED_WINDOW_MINUTES = 180
SIDE_MOVE_LIMITS = {MARKET_SPREAD: 0.25, MARKET_MONEYLINE: 0.01}
LEAGUE_WINDOW_GAME_DAYS = 30
ML_PROBABILITY_CLIP = (0.01, 0.99)
_MINUTE_NS = 60 * 1_000_000_000
_EPS = 1e-9


# --------------------------------------------------------------------------------
# As-of level reads
# --------------------------------------------------------------------------------


def tick_levels(ticks: pd.DataFrame) -> pd.DataFrame:
    """One ``level`` per tick: ``game_id, market, book, minutes_before_tip, level``."""
    working = ticks[["game_id", "market", "book", "minutes_before_tip"]].copy()
    working["game_id"] = working["game_id"].astype(str)
    working["raw_line"] = resolve_line(ticks)
    working["fair_right"] = devig_two_way(ticks["left_price"], ticks["right_price"])[
        "fair_right"
    ].to_numpy()
    working["level"] = market_level(working)
    return working[["game_id", "market", "book", "minutes_before_tip", "level"]]


class LevelReader:
    """Vectorised "level of (game, market, book) as of m minutes before tip"."""

    def __init__(self, levels: pd.DataFrame) -> None:
        levels = levels.copy()
        levels["game_id"] = levels["game_id"].astype(str)
        self._index = pd.MultiIndex.from_frame(
            levels[["game_id", "market", "book"]].drop_duplicates()
        )
        levels["_series"] = self._index.get_indexer(
            pd.MultiIndex.from_frame(levels[["game_id", "market", "book"]])
        )
        levels["_t"] = -pd.to_numeric(levels["minutes_before_tip"]).astype(float)
        self._ticks = levels[["_series", "_t", "level"]].sort_values(
            "_t", kind="mergesort"
        )
        self.books = {
            market: sorted(levels.loc[levels["market"].eq(market), "book"].unique())
            for market in levels["market"].unique()
        }
        opener = levels.sort_values("_t", kind="mergesort").groupby("_series").head(1)
        self._openers = opener.set_index("_series")["level"]

    def _series(self, game_ids, market: str, book: str) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays(
            [
                np.asarray(game_ids, dtype=object),
                np.full(len(game_ids), market, dtype=object),
                np.full(len(game_ids), book, dtype=object),
            ]
        )
        return self._index.get_indexer(keys)

    def at(self, game_ids, market: str, book: str, minutes) -> np.ndarray:
        """Level as of ``minutes`` before tip; NaN without a tick by then."""
        minutes = np.asarray(minutes, dtype=float)
        out = np.full(len(minutes), np.nan)
        series = self._series(game_ids, market, book)
        valid = (series >= 0) & np.isfinite(minutes)
        if not valid.any():
            return out
        queries = pd.DataFrame(
            {
                "_series": series[valid],
                "_t": -minutes[valid],
                "_i": np.flatnonzero(valid),
            }
        ).sort_values("_t", kind="mergesort")
        found = pd.merge_asof(
            queries,
            self._ticks,
            on="_t",
            by="_series",
            direction="backward",
            allow_exact_matches=True,
        )
        out[found["_i"].to_numpy()] = found["level"].to_numpy(float)
        return out

    def opener(self, game_ids, market: str, book: str) -> np.ndarray:
        series = self._series(game_ids, market, book)
        values = self._openers.reindex(series).to_numpy(float)
        values[series < 0] = np.nan
        return values

    def book_matrix(self, game_ids, market: str, minutes) -> np.ndarray:
        """Levels for every book of ``market``: shape (rows, books)."""
        books = self.books.get(market, [])
        if not books:
            return np.full((len(game_ids), 0), np.nan)
        return np.column_stack(
            [self.at(game_ids, market, book, minutes) for book in books]
        )


def _median_move(now: np.ndarray, then: np.ndarray) -> np.ndarray:
    """Median across books quoted at both instants; NaN where none was."""
    moves = now - then
    if moves.shape[1] == 0:
        return np.full(len(moves), np.nan)
    with warnings.catch_warnings():
        # All-NaN rows (no book quoted at both instants) are expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(moves, axis=1)


def _moneyline_margin(probability: np.ndarray) -> np.ndarray:
    return MARGIN_SIGMA * norm.ppf(np.clip(probability, *ML_PROBABILITY_CLIP))


# --------------------------------------------------------------------------------
# G3 -- cross-market
# --------------------------------------------------------------------------------


def _league_mean_before(
    dates: pd.Series, values: np.ndarray, game_ids: np.ndarray
) -> np.ndarray:
    """Mean of ``values`` over the previous game days, strictly earlier dates.

    Counted in game days rather than calendar days so the first days of a season
    read the end of the previous one instead of an empty off-season window.

    Each game carries weight one: its snapshots are averaged first, so a game
    quoted at more horizons does not pull the league reference towards itself.
    Which horizons enter that per-game average still follows the snapshot grid.
    """
    frame = pd.DataFrame(
        {
            "game": np.asarray(game_ids, dtype=object),
            "date": pd.to_datetime(dates).dt.normalize().to_numpy(),
            "v": values,
        }
    )
    frame = frame.loc[np.isfinite(frame["v"])]
    per_game = frame.groupby("game").agg(date=("date", "first"), v=("v", "mean"))
    daily = per_game.groupby("date")["v"].agg(["sum", "count"]).sort_index()
    sums = daily["sum"].rolling(LEAGUE_WINDOW_GAME_DAYS, min_periods=1).sum().shift(1)
    counts = (
        daily["count"].rolling(LEAGUE_WINDOW_GAME_DAYS, min_periods=1).sum().shift(1)
    )
    means = (sums / counts).rename("mean")
    looked_up = pd.to_datetime(dates).dt.normalize().map(means)
    return looked_up.to_numpy(float)


def cross_market_features(
    reader: LevelReader, rows: pd.DataFrame, *, anchor: str
) -> pd.DataFrame:
    """G3 per row. ``rows``: ``GAME_ID``, ``TIME_TO_MATCH_MIN``, ``GAME_DATE``.

    Positionally aligned with ``rows``.
    """
    game_ids = rows["GAME_ID"].astype(str).to_numpy()
    minutes = pd.to_numeric(rows["TIME_TO_MATCH_MIN"]).to_numpy(float)
    earlier = minutes + MOVE_WINDOW_MINUTES
    prefix = f"ODDS_SNAP_XMKT_{anchor.upper()}_ML_MARGIN_MINUS_SPREAD"

    def gap(market_minutes=None, opener=False) -> np.ndarray:
        if opener:
            spread = reader.opener(game_ids, MARKET_SPREAD, anchor)
            probability = reader.opener(game_ids, MARKET_MONEYLINE, anchor)
        else:
            spread = reader.at(game_ids, MARKET_SPREAD, anchor, market_minutes)
            probability = reader.at(game_ids, MARKET_MONEYLINE, anchor, market_minutes)
        return _moneyline_margin(probability) - spread

    now = gap(minutes)
    out = pd.DataFrame(index=np.arange(len(rows)))
    out[prefix] = now
    out[f"{prefix}_MOVE_60"] = now - gap(earlier)
    out[f"{prefix}_FROM_OPEN"] = now - gap(opener=True)
    out[f"{prefix}_VS_LEAGUE"] = now - _league_mean_before(
        rows["GAME_DATE"], now, game_ids
    )

    consensus = {
        market: _median_move(
            reader.book_matrix(game_ids, market, minutes),
            reader.book_matrix(game_ids, market, earlier),
        )
        for market in (MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE)
    }
    # Three states: a side market seen moving decides the row (0), all seen
    # quiet keeps the total move, and an unobserved side with no observed mover
    # leaves it unknown. Treating an unobserved move as zero would claim a
    # quiet side market nobody could see.
    side_moved = np.zeros(len(rows), dtype=bool)
    side_unknown = np.zeros(len(rows), dtype=bool)
    for market, limit in SIDE_MOVE_LIMITS.items():
        move = consensus[market]
        observed = np.isfinite(move)
        side_moved |= observed & (np.abs(np.nan_to_num(move)) >= limit)
        side_unknown |= ~observed
    total_move = consensus[MARKET_TOTALS]
    out["ODDS_SNAP_XMKT_TOTAL_MOVE_WITHOUT_SIDE_MOVE_60"] = np.where(
        ~np.isfinite(total_move) | (side_unknown & ~side_moved),
        np.nan,
        np.where(side_moved, 0.0, total_move),
    )
    out["ODDS_SNAP_XMKT_ABS_SPREAD_MOVE_60"] = np.abs(consensus[MARKET_SPREAD])
    return out


# --------------------------------------------------------------------------------
# G2 -- reaction to injury news
# --------------------------------------------------------------------------------


def _signed_news(home: np.ndarray, away: np.ndarray, market: str) -> np.ndarray:
    if NEWS_SIGN[market] == "side":
        return away - home
    return -(home + away)


def news_reaction_features(
    reader: LevelReader,
    rows: pd.DataFrame,
    timeline: InjuryNewsTimeline,
    *,
    anchor: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """G2 per row, positionally aligned with ``rows``.

    ``rows``: ``GAME_ID``, ``SEASON_YEAR``, ``TIME_TO_MATCH_MIN``,
    ``TIPOFF_UTC``, ``SNAPSHOT_TS_UTC``, ``HOME_TEAM_ID``, ``AWAY_TEAM_ID``.

    beta, the market move per signed expected missing point, is fitted
    walk-forward (games tipped before T, current and previous season) on
    material rows: the consensus move over the last 60 minutes against the news
    over the same 60 minutes. One beta per market, applied to every window.

    Two notions of news live here, deliberately:

    * ``EXPECTED_MOVE_W240`` and ``UNEXPLAINED_MOVE_W180`` use ALL news in their
      window, however small, so they can be non-zero while ``HAS_RECENT`` is 0.
    * ``HAS_RECENT`` and the ``*_SINCE_PRE_NEWS`` / ``REACTION_RESIDUAL`` family
      are anchored on the latest MATERIAL change (one report step of at least
      ``MATERIAL_NEWS_POINTS``) within ``NEWS_LOOKBACK_MINUTES``.

    Without recent material news the anchored family has no anchor. Those rows
    get 0.0 with ``HAS_RECENT`` = 0 -- the ``has_window_<w>`` / ``move_last_<w>``
    convention -- rather than NaN: no news is the common case (58-77% of rows
    by season), so NaN would have the cleaning ``nan_threshold`` drop the whole
    family and make its NaN rate a season indicator. NaN is kept for rows with
    news whose move could not be observed (no quote at one of the instants).
    """
    n = len(rows)
    game_ids = rows["GAME_ID"].astype(str).to_numpy()
    minutes = pd.to_numeric(rows["TIME_TO_MATCH_MIN"]).to_numpy(float)
    seasons = pd.to_numeric(rows["SEASON_YEAR"]).to_numpy(int)
    tip_ns = pd.to_datetime(rows["TIPOFF_UTC"], utc=True).astype("int64").to_numpy()
    snapshot_ns = (
        pd.to_datetime(rows["SNAPSHOT_TS_UTC"], utc=True).astype("int64").to_numpy()
    )
    teams = {
        "home": rows["HOME_TEAM_ID"].astype(str).to_numpy(),
        "away": rows["AWAY_TEAM_ID"].astype(str).to_numpy(),
    }

    covered = np.ones(n, dtype=bool)
    expected_now = {}
    for side, team_ids in teams.items():
        covered &= timeline.covered_at(game_ids, team_ids, snapshot_ns)
        expected_now[side] = timeline.expected_missing_at(
            game_ids, team_ids, snapshot_ns
        )

    def news_over(start_ns: np.ndarray, market: str) -> np.ndarray:
        change = {
            side: expected_now[side]
            - timeline.expected_missing_at(game_ids, team_ids, start_ns)
            for side, team_ids in teams.items()
        }
        signed = _signed_news(change["home"], change["away"], market)
        return np.where(covered, signed, np.nan)

    news_start = {
        window: snapshot_ns - window * _MINUTE_NS
        for window in (BETA_WINDOW_MINUTES, UNEXPLAINED_WINDOW_MINUTES, 240)
    }

    news_ns = timeline.latest_material_news(
        game_ids, snapshot_ns, NEWS_LOOKBACK_MINUTES
    )
    has_news = np.isfinite(news_ns)
    pre_news_ns = np.where(has_news, news_ns - PRE_NEWS_MINUTES * _MINUTE_NS, np.nan)
    pre_news_minutes = (tip_ns - pre_news_ns) / _MINUTE_NS
    pre_news_ns_int = np.where(has_news, pre_news_ns, snapshot_ns).astype("int64")

    out = pd.DataFrame(index=np.arange(n))
    for market, short in MARKET_SHORT.items():
        books_now = reader.book_matrix(game_ids, market, minutes)
        books_beta = reader.book_matrix(game_ids, market, minutes + BETA_WINDOW_MINUTES)
        news_beta_window = news_over(news_start[BETA_WINDOW_MINUTES], market)
        consensus_move = _median_move(books_now, books_beta)

        label = (
            np.isfinite(news_beta_window)
            & (np.abs(np.nan_to_num(news_beta_window)) >= MATERIAL_NEWS_POINTS)
            & np.isfinite(consensus_move)
        )
        fit = walk_forward_least_squares(
            np.nan_to_num(news_beta_window).reshape(-1, 1),
            np.nan_to_num(consensus_move),
            game_ids=pd.Series(game_ids),
            seasons=seasons,
            tip_ns=tip_ns,
            snapshot_ns=snapshot_ns,
            label_mask=label,
            predict_mask=np.ones(n, dtype=bool),
            alpha=0.0,
            min_train_rows=BETA_MIN_TRAIN_ROWS,
        )
        beta = np.nan_to_num(fit.coefficients[:, 0], nan=0.0)
        if verbose:
            order = np.argsort(snapshot_ns, kind="stable")
            last = pd.Series(beta[order], index=seasons[order]).groupby(level=0).last()
            shown = ", ".join(f"{s}: {b:.4f}" for s, b in last.items())
            print(f"  news beta {short} (per expected point, end of season): {shown}")

        news_prefix = f"ODDS_SNAP_NEWS_{short}_"
        out[news_prefix + "EXPECTED_MOVE_W240"] = beta * news_over(
            news_start[240], market
        )

        anchor_now = reader.at(game_ids, market, anchor, minutes)
        anchor_pre = reader.at(game_ids, market, anchor, pre_news_minutes)
        books_pre = reader.book_matrix(game_ids, market, pre_news_minutes)
        anchor_move = np.where(has_news, anchor_now - anchor_pre, 0.0)
        consensus_since = np.where(has_news, _median_move(books_now, books_pre), 0.0)
        moved = np.abs(books_now - books_pre) > _EPS
        n_moved = np.where(has_news, moved.sum(axis=1).astype(float), 0.0)
        news_since = np.where(has_news, news_over(pre_news_ns_int, market), 0.0)

        out[news_prefix + f"{anchor.upper()}_MOVE_SINCE_PRE_NEWS"] = anchor_move
        out[news_prefix + "CONSENSUS_MOVE_SINCE_PRE_NEWS"] = consensus_since
        out[news_prefix + f"{anchor.upper()}_REACTION_RESIDUAL"] = (
            anchor_move - beta * news_since
        )
        out[news_prefix + "N_BOOKS_MOVED_SINCE_PRE_NEWS"] = n_moved
        if market in (MARKET_TOTALS, MARKET_SPREAD):
            anchor_then = reader.at(
                game_ids, market, anchor, minutes + UNEXPLAINED_WINDOW_MINUTES
            )
            out[news_prefix + f"{anchor.upper()}_UNEXPLAINED_MOVE_W180"] = (
                anchor_now - anchor_then
            ) - beta * news_over(news_start[UNEXPLAINED_WINDOW_MINUTES], market)
    out["ODDS_SNAP_NEWS_HAS_RECENT"] = has_news.astype(float)
    return out
