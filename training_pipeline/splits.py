"""CV/holdout split construction.

Mostly a thin dispatch over the builders in ``nba_ou.modeling.modeling``. The
exception is the rolling-origin plan below, which is new machinery: it exists
because the two older splitters describe a fold as "a block of N games", and the
protocol worth selecting hyperparameters under is "train on everything up to a
date, predict the next few game-days" -- which is a statement about dates, and
produces folds whose size the schedule decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.modeling import (
    assert_valid_time_splits,
    make_test_anchored_walk_forward_splits,
    make_walk_forward_last_n_seasons_splits,
    split_latest_dates_holdout,
)

from training_pipeline.config import CVStrategy, ExperimentConfig
from training_pipeline.data import training_eligible_mask

#: Optuna parameter name for the tuned training-window size. One constant so the
#: sampler, the artifacts, the reporting factors and the final refit cannot drift
#: onto different spellings of the same knob.
TRAIN_GAMES_PARAM = "train_games"

Split = tuple[np.ndarray, np.ndarray]


def resolve_game_codes(df: pd.DataFrame, *, game_id_col: str) -> np.ndarray | None:
    """Integer code per row saying which GAME it belongs to, or None.

    None means "every row is its own game", which is true of every
    one-row-per-game dataset and is exactly the arithmetic this module did
    before game-awareness existed. Returning None rather than ``arange`` is
    deliberate: it lets the fast path stay literally the old code, so the
    closing-line behaviour is unchanged by construction rather than by
    numerical coincidence.

    On a pooled intermediate-line frame one game owns up to ten rows, and
    without this every ``*_games`` knob silently means rows -- a
    ``train_games`` of 3500 becoming 350 games, a ``min_validation_games`` of
    25 becoming 2.5. That was previously corrected by hand-multiplying the
    numbers in a YAML comment, which is a correction that cannot be verified
    and stops being right the moment the snapshot grid changes.
    """
    if game_id_col not in df.columns:
        return None
    codes = pd.factorize(df[game_id_col].astype(str))[0]
    if len(np.unique(codes)) == len(df):
        # One row per game: identical to no key at all, and worth detecting so
        # the closing-line path never pays for the general case.
        return None
    return np.asarray(codes)


def _distinct(codes: np.ndarray | None, idx: np.ndarray) -> int:
    """Games covered by ``idx``. Row count when there is no game key."""
    if codes is None:
        return int(len(idx))
    return int(len(np.unique(codes[idx])))


def _tail_games(
    codes: np.ndarray | None, idx: np.ndarray, train_games: int
) -> np.ndarray:
    """The rows of the last ``train_games`` distinct games in ``idx``.

    ``idx`` is chronological, so "last" is by first appearance in it. Every row
    of a selected game is kept, including snapshots that appear out of order --
    a window that held six of a game's ten snapshots would be neither a row
    window nor a game window.
    """
    if codes is None:
        return idx[-int(train_games) :]
    fold_codes = codes[idx]
    _, first_position = np.unique(fold_codes, return_index=True)
    ordered = fold_codes[np.sort(first_position)]
    if int(train_games) >= len(ordered):
        return idx
    return idx[np.isin(fold_codes, ordered[-int(train_games) :])]


def split_latest_days_holdout(
    df: pd.DataFrame, *, date_col: str, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fixed calendar window from the end of the data.

    The cut is on the date, not on a row count, so every game on the boundary
    day lands on the same side of the split -- a count-based cut can slice a
    game-day in half, putting some of a day's games in training and the rest in
    test, which is exactly the leak the daily walk-forward exists to avoid.

    Counted back from the last game present, so the window is defined by the
    data rather than by today's date; re-running an old config gives the same
    split.
    """
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.max() - pd.Timedelta(days=test_days)

    df_dev = df.loc[dates <= cutoff].copy().reset_index(drop=True)
    df_test = df.loc[dates > cutoff].copy().reset_index(drop=True)

    if df_test.empty:
        raise ValueError(
            f"holdout.test_days={test_days} selected no games. The data ends "
            f"{dates.max().date()}."
        )
    if df_dev.empty:
        raise ValueError(
            f"holdout.test_days={test_days} consumed the entire dataset "
            f"({dates.min().date()} to {dates.max().date()}); nothing left to "
            "train on."
        )
    return df_dev, df_test


def build_holdout_split(
    df_full: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.holdout.test_days is not None:
        return split_latest_days_holdout(
            df_full,
            date_col=config.data.date_col,
            test_days=config.holdout.test_days,
        )
    return split_latest_dates_holdout(
        df=df_full,
        date_col=config.data.date_col,
        test_size=config.holdout.test_size,
        test_games=config.holdout.test_games,
    )


@dataclass(frozen=True)
class RollingOriginFold:
    """One retrain point: what was knowable, and what it then predicted.

    ``history_idx`` is every training-eligible row dated strictly before
    ``origin_date``, in chronological order and NOT yet windowed. The window is
    applied by :meth:`RollingOriginPlan.splits`, which is what lets one plan
    serve every ``train_games`` an Optuna trial might choose without rebuilding
    the fold layout -- and therefore lets different trials be compared on
    identical validation games.
    """

    fold: int
    #: Training uses games strictly before this date. It is the fold's first
    #: validation day, so "strictly before" is exactly what production knows.
    origin_date: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    #: The game-days this fold predicts. Length <= retrain_every_days: a fold is
    #: closed early at a season boundary.
    valid_dates: tuple[pd.Timestamp, ...]
    #: Positional indices into df_dev. Every game on the validation days,
    #: including training-ineligible ones -- the filter changes what is learned
    #: from, never what is scored.
    valid_idx: np.ndarray
    history_idx: np.ndarray
    season: object = None
    #: Game code per row of the WHOLE dev frame, or None when every row is its
    #: own game. Shared by reference across folds; see resolve_game_codes.
    game_codes: np.ndarray | None = None

    def train_idx(self, train_games: int | None) -> np.ndarray:
        if train_games is None:
            return self.history_idx
        return _tail_games(self.game_codes, self.history_idx, train_games)

    @property
    def n_history_games(self) -> int:
        return _distinct(self.game_codes, self.history_idx)

    @property
    def n_valid_games(self) -> int:
        return _distinct(self.game_codes, self.valid_idx)


@dataclass
class RollingOriginPlan:
    """The rolling-origin fold layout, independent of the training window."""

    folds: list[RollingOriginFold]
    fold_info: pd.DataFrame
    #: Games in the pooled validation region, i.e. what the objective is
    #: actually measured on. Reported because ``eval_span_games`` is a target
    #: and whole game-days are indivisible, so the two rarely match exactly.
    n_validation_games: int
    n_validation_days: int
    #: Validation ROWS. Equal to n_validation_games except on a pooled
    #: intermediate-line frame, where one game contributes several rows and the
    #: objective is therefore averaged over this many predictions, not that
    #: many independent games.
    n_validation_rows: int = 0
    #: Set when max_folds trimmed the layout, so callers can say so out loud.
    n_folds_before_max: int = 0
    #: The configured floor, carried so readers of a saved plan can tell a fold
    #: that is small because the schedule was thin from one that is small
    #: because no floor was ever asked for.
    min_validation_games: int | None = None

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def fold_game_counts(self) -> list[int]:
        return [fold.n_valid_games for fold in self.folds]

    @property
    def fold_row_counts(self) -> list[int]:
        """Rows per fold. Differs from fold_game_counts only when pooled."""
        return [int(len(fold.valid_idx)) for fold in self.folds]

    @property
    def n_folds_below_min(self) -> int:
        """Folds that still fall short of ``min_validation_games``.

        Not necessarily a bug: a season boundary closes a fold early by design.
        Reported rather than raised so the count is visible, because "the floor
        was set and 9 folds ignored it" is a thing to know before reading the
        objective.
        """
        if self.min_validation_games is None:
            return 0
        return sum(1 for n in self.fold_game_counts if n < self.min_validation_games)

    @property
    def min_history_games(self) -> int:
        """Smallest training history any fold has available.

        The ceiling on ``train_games``: ask for more than this and the earliest
        folds quietly train on less than requested, which is the silent shrink
        that has corrupted a window comparison before.
        """
        return min(fold.n_history_games for fold in self.folds)

    def splits(self, train_games: int | None) -> list[Split]:
        return [(fold.train_idx(train_games), fold.valid_idx) for fold in self.folds]

    def assert_window_fits(self, train_games: int | None) -> None:
        if train_games is None:
            return
        available = self.min_history_games
        if int(train_games) > available:
            raise ValueError(
                f"train_games={train_games} exceeds the {available} GAMES the "
                "earliest rolling-origin fold has available, so that fold would "
                "silently train on fewer games than every other fold and the "
                "comparison would no longer be the one you designed. Lower "
                "train_games, shorten walk_forward.eval_span_games (fewer folds "
                "reach as far back), or lower data.season_year_floor to add "
                "history."
            )


def build_rolling_origin_plan(
    df_dev: pd.DataFrame, config: ExperimentConfig
) -> RollingOriginPlan:
    """Lay out rolling origins over the tail of ``df_dev``.

    The protocol, matching ``holdout_evaluation: daily_walk_forward``:

        train on everything strictly before T -> predict the next
        ``retrain_every_days`` game-days -> advance the origin past them ->
        repeat, with the games just predicted now part of history.

    Definitions that matter, and why:

    * A step is counted in **game-days**, not calendar dates. "The next 4 days
      with games" is what a bettor experiences; counting dates would let an
      All-Star break silently consume a whole window.
    * Months in ``exclude_test_months`` never appear in a validation window, the
      same rule the older splitters apply. Those games stay available as
      training history -- excluding them from what a model may LEARN from is a
      different decision, and not this one.
    * A fold never spans two seasons when ``require_same_season_test`` is on. It
      is closed early instead. Without this, skipping May and June would glue
      late April onto late October into one "4-day" window whose origin is five
      months before the games it predicts.
    * Validation windows are consumed sequentially and can never overlap, so no
      game is scored twice.
    """
    wf = config.walk_forward
    date_col = config.data.date_col
    season_col = config.data.season_col

    if len(df_dev) == 0:
        raise ValueError("Cannot build rolling-origin folds from an empty frame.")

    dates = pd.to_datetime(df_dev[date_col], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"Invalid dates in column {date_col!r}.")

    eligible = training_eligible_mask(df_dev, config)
    positions = np.arange(len(df_dev))
    # None on every one-row-per-game dataset, which keeps the whole of what
    # follows on the pre-existing row arithmetic.
    game_codes = resolve_game_codes(df_dev, game_id_col=config.data.game_id_col)

    # Position lists per game-day, chronological. Sorting explicitly rather than
    # trusting the frame's order: a subtly unsorted dev frame would otherwise
    # produce a "history" containing future games, which is the one bug this
    # module must not have.
    order = np.argsort(dates.to_numpy(), kind="stable")
    ordered_dates = dates.to_numpy()[order]
    ordered_positions = positions[order]

    unique_dates, day_starts = np.unique(ordered_dates, return_index=True)
    day_positions = np.split(ordered_positions, day_starts[1:])
    unique_dates = [pd.Timestamp(value) for value in unique_dates]

    season_of_day = {
        day: df_dev[season_col].to_numpy()[members[0]] if season_col in df_dev else None
        for day, members in zip(unique_dates, day_positions, strict=True)
    }
    # GAMES per day, not rows. On a pooled frame a ten-snapshot day of three
    # games is 30 rows and 3 games; eval_span_games and min_validation_games
    # are both written in games, so this is the number they must consume.
    games_on_day = {
        day: _distinct(game_codes, members)
        for day, members in zip(unique_dates, day_positions, strict=True)
    }
    idx_on_day = dict(zip(unique_dates, day_positions, strict=True))

    excluded_months = set(wf.exclude_test_months)
    candidate_days = [day for day in unique_dates if day.month not in excluded_months]
    if not candidate_days:
        raise ValueError(
            "No game-day is eligible to validate on: every date in dev falls in "
            f"walk_forward.exclude_test_months={tuple(sorted(excluded_months))}."
        )

    # --- pick the chronological evaluation region ---------------------------
    # Walk back from the newest candidate day, accumulating whole days until the
    # requested span is covered. Whole days only, so a day is never split
    # between training and validation.
    if wf.eval_span_games is None:
        first_day_position = 0
    else:
        accumulated = 0
        first_day_position = len(candidate_days) - 1
        for position in range(len(candidate_days) - 1, -1, -1):
            accumulated += games_on_day[candidate_days[position]]
            first_day_position = position
            if accumulated >= wf.eval_span_games:
                break

    region_days = candidate_days[first_day_position:]

    # --- consume the region in retrain_every_days chunks ---------------------
    folds: list[RollingOriginFold] = []
    cursor = 0
    while cursor < len(region_days):
        window: list[pd.Timestamp] = [region_days[cursor]]
        season = season_of_day[region_days[cursor]]
        next_cursor = cursor + 1
        while (
            len(window) < wf.retrain_every_days and next_cursor < len(region_days)
        ):
            candidate = region_days[next_cursor]
            if wf.require_same_season_test and season_of_day[candidate] != season:
                break
            window.append(candidate)
            next_cursor += 1

        # Then, optionally, keep absorbing WHOLE game-days until the fold holds
        # min_validation_games. Days, never part-days: a date split between
        # train and validation is the leak the whole protocol is built to
        # avoid. Same-season and end-of-region limits still bind, so a fold can
        # legitimately finish short -- the trailing case is merged below.
        if wf.min_validation_games is not None:
            games_in_window = sum(games_on_day[day] for day in window)
            while (
                games_in_window < wf.min_validation_games
                and next_cursor < len(region_days)
            ):
                candidate = region_days[next_cursor]
                if wf.require_same_season_test and season_of_day[candidate] != season:
                    break
                window.append(candidate)
                games_in_window += games_on_day[candidate]
                next_cursor += 1

        origin_date = window[0]
        history_mask = (dates.to_numpy() < np.datetime64(origin_date)) & eligible
        history_idx = positions[history_mask]
        # Chronological, because the window is taken as a tail.
        history_idx = history_idx[np.argsort(dates.to_numpy()[history_idx], kind="stable")]

        if _distinct(game_codes, history_idx) < wf.min_train_games:
            # Not enough history yet. Step ONE day, not past the whole window:
            # otherwise where the accepted region starts depends on how wide
            # the window happens to be, and two cells differing only in
            # retrain_every_days or min_validation_games would be scored on
            # different games. Only ever runs at the very start of the data --
            # with eval_span_games set, the region is a tail with ample history
            # behind it and no fold is rejected at all.
            cursor += 1
            continue

        valid_idx = np.concatenate([idx_on_day[day] for day in window])
        folds.append(
            RollingOriginFold(
                fold=len(folds) + 1,
                origin_date=origin_date,
                valid_start=window[0],
                valid_end=window[-1],
                valid_dates=tuple(window),
                valid_idx=np.sort(valid_idx),
                history_idx=history_idx,
                season=season,
                game_codes=game_codes,
            )
        )
        cursor = next_cursor

    # A fold can still finish short in two ways the growth loop cannot fix: the
    # region runs out (the trailing remainder), or require_same_season_test
    # closes it at a season boundary. Cell A has both -- a 2-game fold on
    # 2025-04-18, the last day of season 2024, whose next available day is the
    # following October.
    #
    # Absorb a short fold into its PREDECESSOR rather than dropping it. Dropping
    # would silently shrink the OOF cohort, and a cell that changed this knob
    # would then be scored on different games than the one it is compared with.
    # Merging only extends the predecessor's validation window forward: its
    # origin, and therefore its training history, is untouched, so no leak is
    # introduced -- those games are simply predicted by a slightly staler model.
    # The season check still binds, so a short fold that OPENS a season is left
    # alone rather than glued onto the previous April.
    #
    # One left-to-right pass, against the running merged list, so a run of short
    # folds collapses into one rather than pairing up arbitrarily. A short FIRST
    # fold has no predecessor and stays short; n_folds_below_min reports it.
    if wf.min_validation_games is not None and len(folds) > 1:
        merged: list[RollingOriginFold] = []
        for fold in folds:
            previous = merged[-1] if merged else None
            if (
                previous is not None
                and fold.n_valid_games < wf.min_validation_games
                and not (
                    wf.require_same_season_test and fold.season != previous.season
                )
            ):
                merged[-1] = RollingOriginFold(
                    fold=previous.fold,
                    origin_date=previous.origin_date,
                    valid_start=previous.valid_start,
                    valid_end=fold.valid_end,
                    valid_dates=previous.valid_dates + fold.valid_dates,
                    valid_idx=np.sort(
                        np.concatenate([previous.valid_idx, fold.valid_idx])
                    ),
                    history_idx=previous.history_idx,
                    season=previous.season,
                    game_codes=game_codes,
                )
            else:
                merged.append(fold)
        folds = [
            RollingOriginFold(
                fold=number,
                origin_date=fold.origin_date,
                valid_start=fold.valid_start,
                valid_end=fold.valid_end,
                valid_dates=fold.valid_dates,
                valid_idx=fold.valid_idx,
                history_idx=fold.history_idx,
                season=fold.season,
                game_codes=fold.game_codes,
            )
            for number, fold in enumerate(merged, start=1)
        ]

    if not folds:
        raise ValueError(
            "No valid rolling-origin folds were created. The most likely cause "
            f"is walk_forward.min_train_games={wf.min_train_games} exceeding the "
            "history available before the evaluation region; shorten "
            "walk_forward.eval_span_games or lower min_train_games."
        )

    n_folds_before_max = len(folds)
    if wf.max_folds is not None and wf.max_folds < len(folds):
        if wf.eval_span_games is not None:
            raise ValueError(
                f"walk_forward.max_folds={wf.max_folds} would trim the "
                f"{len(folds)} folds that walk_forward.eval_span_games="
                f"{wf.eval_span_games} asked for, so the evaluation volume you "
                "configured is not the volume you would get. Set max_folds: null "
                "under rolling_origin, or drop eval_span_games and cap by folds "
                "alone."
            )
        # Keep the LATEST folds: the most recent history is the most relevant,
        # and it matches fold_selection's existing default.
        folds = folds[-wf.max_folds :]
        folds = [
            RollingOriginFold(
                fold=number,
                origin_date=fold.origin_date,
                valid_start=fold.valid_start,
                valid_end=fold.valid_end,
                valid_dates=fold.valid_dates,
                valid_idx=fold.valid_idx,
                history_idx=fold.history_idx,
                season=fold.season,
                game_codes=fold.game_codes,
            )
            for number, fold in enumerate(folds, start=1)
        ]

    choices = wf.train_games_choices
    nominal_window = max(choices) if choices else wf.train_games
    fold_info = pd.DataFrame(
        [
            {
                "fold": fold.fold,
                # At the nominal window (the largest choice when tuning), so the
                # column answers "does every fold get the window I asked for?".
                #
                # *_n_games are GAMES and *_n_rows are training/scoring rows.
                # They are equal on every one-row-per-game dataset and differ by
                # the snapshot count on a pooled one, which is the difference
                # that used to be invisible: "3500" meant 350 games and the
                # fold table said 3500 either way.
                "train_n_games": _distinct(
                    fold.game_codes, fold.train_idx(nominal_window)
                ),
                "train_n_rows": int(len(fold.train_idx(nominal_window))),
                "history_n_games": fold.n_history_games,
                "test_n_games": fold.n_valid_games,
                "test_n_rows": int(len(fold.valid_idx)),
                "n_valid_days": len(fold.valid_dates),
                "train_start_date": pd.Timestamp(
                    dates.to_numpy()[fold.train_idx(nominal_window)].min()
                ),
                "train_end_date": pd.Timestamp(
                    dates.to_numpy()[fold.train_idx(nominal_window)].max()
                ),
                "origin_date": fold.origin_date,
                "test_start_date": fold.valid_start,
                "test_end_date": fold.valid_end,
                "test_season": fold.season,
            }
            for fold in folds
        ]
    )

    plan = RollingOriginPlan(
        folds=folds,
        fold_info=fold_info,
        n_validation_games=int(sum(fold.n_valid_games for fold in folds)),
        n_validation_rows=int(sum(len(fold.valid_idx) for fold in folds)),
        n_validation_days=int(sum(len(fold.valid_dates) for fold in folds)),
        n_folds_before_max=n_folds_before_max,
        min_validation_games=wf.min_validation_games,
    )

    # An impossible floor is a config error, not a layout to run: if not one
    # fold in the whole region can reach it, the knob is silently doing nothing
    # useful and every fold is "short".
    if wf.min_validation_games is not None and plan.n_folds_below_min == plan.n_folds:
        raise ValueError(
            f"walk_forward.min_validation_games={wf.min_validation_games} was "
            f"met by none of the {plan.n_folds} folds (largest is "
            f"{max(plan.fold_game_counts)} games). Lower it, raise "
            "walk_forward.retrain_every_days, or widen "
            "walk_forward.eval_span_games."
        )

    if wf.verbose >= 1:
        counts = plan.fold_game_counts
        pooled_note = (
            ""
            if plan.n_validation_rows == plan.n_validation_games
            else f" / {plan.n_validation_rows} rows"
        )
        print(
            f"Created {plan.n_folds} rolling-origin folds "
            f"({plan.n_validation_days} game-days, {plan.n_validation_games} "
            f"validation games{pooled_note}, min history "
            f"{plan.min_history_games} games)"
        )
        print(
            f"  games/fold: min {min(counts)} median "
            f"{int(np.median(counts))} max {max(counts)}"
            + (
                ""
                if wf.min_validation_games is None
                else f" | floor {wf.min_validation_games}, "
                f"{plan.n_folds_below_min} fold(s) below it"
            )
        )
        print(fold_info.to_string(index=False))

    return plan


@dataclass
class SplitProvider:
    """Hands the tuner a fold set for whichever training window a trial picks.

    Two shapes behind one interface:

    * a fixed list of splits (``test_anchored`` / ``last_n_seasons``); for tuned
      test-anchored runs, these store the full training histories and are trimmed
      at read time so every trial scores the same validation games;
    * a :class:`RollingOriginPlan`, which also applies the window at read time.

    Fixed validation games make tuning the window legitimate. If the validation
    set moved with the window, a trial preferring 4500 games would be scored on
    different games than one preferring 2500, measuring the cohort as well.
    """

    fold_info: pd.DataFrame
    default_train_games: int | None
    plan: RollingOriginPlan | None = None
    fixed_splits: list[Split] | None = None
    train_games_choices: tuple[int, ...] | None = None
    fixed_training_mask: np.ndarray | None = None
    _cache: dict[int | None, list[Split]] = field(default_factory=dict, repr=False)

    @property
    def tunes_train_games(self) -> bool:
        return bool(self.train_games_choices)

    @property
    def n_folds(self) -> int:
        if self.plan is not None:
            return self.plan.n_folds
        assert self.fixed_splits is not None
        return len(self.fixed_splits)

    @property
    def n_validation_games(self) -> int:
        if self.plan is not None:
            return self.plan.n_validation_games
        assert self.fixed_splits is not None
        return int(sum(len(valid) for _, valid in self.fixed_splits))

    @property
    def fold_game_counts(self) -> list[int]:
        """Validation games per fold, in fold order.

        Recorded per run because under rolling_origin the schedule decides it:
        "30 folds, 855 games" hides the difference between thirty 28-game folds
        and a layout with a 2-game one in it.
        """
        if self.plan is not None:
            return self.plan.fold_game_counts
        assert self.fixed_splits is not None
        return [int(len(valid)) for _, valid in self.fixed_splits]

    def suggest_train_games(self, trial: optuna.Trial) -> int | None:
        """Sample the window for one trial, or return the fixed value.

        Sampled FIRST in the objective, before any XGBoost parameter, so the
        draw order a seeded TPE sampler sees is stable and documented. When the
        window is not tuned no ``suggest_*`` call is issued at all, which is what
        keeps existing studies reproducible.
        """
        if not self.train_games_choices:
            return self.default_train_games
        return int(
            trial.suggest_categorical(
                TRAIN_GAMES_PARAM, list(self.train_games_choices)
            )
        )

    def splits_for(self, train_games: int | None) -> list[Split]:
        if self.plan is None:
            assert self.fixed_splits is not None
            if self.train_games_choices:
                if train_games is None or int(train_games) <= 0:
                    raise ValueError("A tuned test-anchored window must be positive.")
                available = min(len(train_idx) for train_idx, _ in self.fixed_splits)
                if int(train_games) > available:
                    raise ValueError(
                        f"train_games={train_games} exceeds the {available} GAMES "
                        "available before the earliest test-anchored fold."
                    )
                key = int(train_games)
                if key not in self._cache:
                    mask = self.fixed_training_mask
                    splits = []
                    for history_idx, valid_idx in self.fixed_splits:
                        tail = history_idx[-key:]
                        if mask is not None:
                            tail = tail[mask[tail]]
                        splits.append((tail, valid_idx))
                    self._cache[key] = splits
                return self._cache[key]
            if train_games != self.default_train_games:
                raise ValueError(
                    f"This split set was built for train_games="
                    f"{self.default_train_games} and cannot be re-materialised "
                    f"at {train_games}: under "
                    f"{CVStrategy.TEST_ANCHORED.value}/"
                    f"{CVStrategy.LAST_N_SEASONS.value} the window is part of "
                    "the fold layout."
                )
            return self.fixed_splits

        key = None if train_games is None else int(train_games)
        if key not in self._cache:
            self.plan.assert_window_fits(key)
            self._cache[key] = self.plan.splits(key)
        return self._cache[key]


def build_split_provider(
    df_dev: pd.DataFrame, config: ExperimentConfig
) -> SplitProvider:
    """Build the CV fold set, in whichever shape the configured strategy needs."""
    wf = config.walk_forward

    if wf.strategy == CVStrategy.ROLLING_ORIGIN:
        plan = build_rolling_origin_plan(df_dev, config)
        provider = SplitProvider(
            fold_info=plan.fold_info,
            default_train_games=wf.train_games,
            plan=plan,
            train_games_choices=wf.train_games_choices,
        )
        # Validate every window a trial could pick, not just the nominal one: a
        # leak or a silently-shrunk fold that only appears at one choice would
        # otherwise surface as an unexplained result rather than an error.
        for candidate in wf.train_games_choices or (wf.train_games,):
            validate_splits(
                df_dev,
                provider.splits_for(candidate),
                date_col=config.data.date_col,
            )
        return provider

    if wf.strategy == CVStrategy.TEST_ANCHORED and wf.train_games_choices:
        if min(wf.train_games_choices) < wf.min_train_games:
            raise ValueError(
                "Every test-anchored train_games choice must be at least "
                f"min_train_games={wf.min_train_games}."
            )
        # Build validation windows ONCE with unbounded histories. Applying a
        # candidate's tail afterwards cannot add/drop a fold or move its test
        # dates, unlike rebuilding the upstream splitter for each trial.
        histories, fold_info = make_test_anchored_walk_forward_splits(
            df=df_dev,
            date_col=config.data.date_col,
            season_col=config.data.season_col,
            test_games=wf.test_games,
            step_games_between_tests=wf.step_games_between_tests,
            train_games=None,
            min_train_games=wf.min_train_games,
            exclude_test_months=wf.exclude_test_months,
            require_same_season_test=wf.require_same_season_test,
            max_folds=wf.max_folds,
            fold_selection=wf.fold_selection,
            verbose=wf.verbose,
        )
        largest = max(wf.train_games_choices)
        provider = SplitProvider(
            fold_info=fold_info.copy(),
            default_train_games=wf.train_games,
            fixed_splits=histories,
            train_games_choices=wf.train_games_choices,
            fixed_training_mask=training_eligible_mask(df_dev, config),
        )
        provider.fold_info["history_n_games"] = [len(train) for train, _ in histories]
        for candidate in wf.train_games_choices:
            validate_splits(
                df_dev,
                provider.splits_for(candidate),
                date_col=config.data.date_col,
            )
        nominal = provider.splits_for(largest)
        provider.fold_info["train_n_games"] = [len(train) for train, _ in nominal]
        provider.fold_info["train_n_rows"] = provider.fold_info["train_n_games"]
        dates = pd.to_datetime(df_dev[config.data.date_col])
        provider.fold_info["train_start_date"] = [
            dates.iloc[train].min() for train, _ in nominal
        ]
        return provider

    splits, fold_info = build_walk_forward_splits(df_dev, config)
    return SplitProvider(
        fold_info=fold_info,
        default_train_games=wf.train_games,
        fixed_splits=splits,
    )


def build_walk_forward_splits(
    df_dev: pd.DataFrame, config: ExperimentConfig
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Return the configured fixed splits or the nominal tuned window.

    Existing fixed-window configs keep their original splitter. A tuned
    test-anchored config gets the same fixed validation folds via a provider,
    which this entry point also exposes to ``scripts/preflight_campaign.py``.
    """
    wf = config.walk_forward

    if wf.strategy == CVStrategy.TEST_ANCHORED and wf.train_games_choices:
        provider = build_split_provider(df_dev, config)
        return provider.splits_for(max(wf.train_games_choices)), provider.fold_info

    if wf.strategy == CVStrategy.ROLLING_ORIGIN:
        plan = build_rolling_origin_plan(df_dev, config)
        choices = wf.train_games_choices
        window = max(choices) if choices else wf.train_games
        plan.assert_window_fits(window)
        splits = plan.splits(window)
        validate_splits(df_dev, splits, date_col=config.data.date_col)
        return splits, plan.fold_info

    if wf.strategy == CVStrategy.TEST_ANCHORED:
        splits, fold_info = make_test_anchored_walk_forward_splits(
            df=df_dev,
            date_col=config.data.date_col,
            season_col=config.data.season_col,
            test_games=wf.test_games,
            step_games_between_tests=wf.step_games_between_tests,
            train_games=wf.train_games,
            min_train_games=wf.min_train_games,
            exclude_test_months=wf.exclude_test_months,
            require_same_season_test=wf.require_same_season_test,
            max_folds=wf.max_folds,
            fold_selection=wf.fold_selection,
            verbose=wf.verbose,
        )
    else:  # LAST_N_SEASONS
        splits, fold_info = make_walk_forward_last_n_seasons_splits(
            df=df_dev,
            date_col=config.data.date_col,
            season_col=config.data.season_col,
            train_seasons=wf.train_seasons,
            test_games=wf.test_games,
            step_games=wf.step_games_between_tests,
            min_train_games=wf.min_train_games,
            max_folds=wf.max_folds,
            fold_selection=wf.fold_selection,
            verbose=wf.verbose,
        )

    # Applied to the TRAIN half of each fold only. This single hook covers both
    # the Optuna objective and cv_betting, which share these splits -- so the
    # hyperparameters are selected under exactly the training regime the final
    # model will use, while every fold is still SCORED on all its games.
    splits = apply_training_filter(df_dev, splits, config)

    validate_splits(df_dev, splits, date_col=config.data.date_col)
    return splits, fold_info


def apply_training_filter(
    df_dev: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop training-ineligible rows from each fold's TRAIN indices.

    Validation indices are returned untouched, which is the whole point: the
    filter changes what the model learns from, never what it is judged on.
    """
    mask = training_eligible_mask(df_dev, config)
    if mask.all():
        return splits

    filtered = []
    for train_idx, valid_idx in splits:
        kept = train_idx[mask[train_idx]]
        if len(kept) == 0:
            raise ValueError(
                "A CV fold has no training rows left after filtering overtime "
                "games. Widen walk_forward.train_games or turn "
                "data.exclude_overtime_from_training off."
            )
        filtered.append((kept, valid_idx))
    return filtered


def validate_splits(
    df_dev: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    date_col: str,
) -> None:
    assert_valid_time_splits(df=df_dev, splits=splits, date_col=date_col)
