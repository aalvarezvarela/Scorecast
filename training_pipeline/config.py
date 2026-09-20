"""Experiment configuration schema for training_pipeline.

`ExperimentConfig` is the single object that fully describes one training run:
which CSV to load, how to clean it, how to split it for CV/holdout, which
target to train, how to tune it with Optuna, how to refit the final model,
and where to save results. It is designed so that every axis observed to vary
across the repo's example notebooks (lab/total_points/regressor/*.ipynb,
lab/diff_from_line/regressor/*.ipynb) is a config field, not a hardcoded
notebook constant.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.config.market_columns import Market
from pydantic import BaseModel, ConfigDict, Field, model_validator

from training_pipeline.betting import DECIMAL_ODDS_MINUS_110, DEFAULT_EDGE_THRESHOLDS

#: Re-exported from nba_ou.config.market_columns rather than redefined.
#:
#: Two identical StrEnums in two packages is not a duplication that stays
#: harmless: ``nba_ou...Market.SPREAD is training_pipeline...Market.SPREAD``
#: would be False, so an identity check would silently take the wrong branch
#: depending on which module the caller imported from. One class, one identity.
#:
#: MONEYLINE is declared there because the datasets now carry normalised
#: moneyline data. It has no strategy and no target -- that is deliberate scope.
__all_market_reexport__ = Market  # keeps linters aware the import is used

class TargetFamily(StrEnum):
    TOTAL_POINTS = "total_points"
    LINE_ERROR = "line_error"
    #: The binary "did the game go OVER this line" label. Not a regression
    #: target -- kept in this enum only because artifact paths, the model
    #: registry and the leaderboard all key off target_family.
    OVER_UNDER = "over_under"
    #: HOME_MARGIN minus the anchor book's spread. The spread market's exact
    #: analogue of LINE_ERROR, and modelled by the same kind of regressor.
    SPREAD_ERROR = "spread_error"


class PredictionStrategy(StrEnum):
    """What is predicted AND with which kind of model.

    ``target_family`` alone conflates two things, which lets invalid
    combinations be expressed (a "total_points classifier" is not a thing).
    This enum names the whole approach in one field, so every value is a real,
    supported configuration.

    Every strategy ultimately answers the same betting question -- OVER or
    UNDER this line -- but they differ in how much of the model's capacity goes
    to the decision versus to reproducing the line:

    - ``total_points_regressor``: predicts the total. Most of the loss is spent
      reproducing the line, which the model can already read off its features.
    - ``line_error_regressor``: predicts total minus line. The nuisance part is
      subtracted out, so all of the target is the part you bet on.
    - ``over_under_classifier``: predicts P(OVER) directly. Takes that logic to
      its limit, at the cost of discarding magnitude -- a game 30 points over
      the line and one 1 point over become the same label.
    """

    TOTAL_POINTS_REGRESSOR = "total_points_regressor"
    LINE_ERROR_REGRESSOR = "line_error_regressor"
    OVER_UNDER_CLASSIFIER = "over_under_classifier"
    #: The spread market's residual regressor: predicts HOME_MARGIN minus the
    #: Bet365 spread available at the prediction point. Structurally identical to
    #: LINE_ERROR_REGRESSOR -- same model kind, same "the target IS the edge"
    #: property, same absence of a line_col -- against a different market.
    #:
    #: There is deliberately ONE spread strategy, used with two dataset configs.
    #: A closing run and an intermediate run predict the same conceptual
    #: quantity; only the dataset decides which line "available at the prediction
    #: point" refers to. Separate enum values would have encoded a dataset choice
    #: as a modelling choice and doubled every branch below for no gain.
    SPREAD_ERROR_REGRESSOR = "spread_error_regressor"

    @property
    def target_family(self) -> TargetFamily:
        return _STRATEGY_TARGET_FAMILY[self]

    @property
    def market(self) -> Market:
        return _STRATEGY_MARKET[self]

    @property
    def is_classifier(self) -> bool:
        return self is PredictionStrategy.OVER_UNDER_CLASSIFIER


_STRATEGY_TARGET_FAMILY: dict[PredictionStrategy, TargetFamily] = {
    PredictionStrategy.TOTAL_POINTS_REGRESSOR: TargetFamily.TOTAL_POINTS,
    PredictionStrategy.LINE_ERROR_REGRESSOR: TargetFamily.LINE_ERROR,
    PredictionStrategy.OVER_UNDER_CLASSIFIER: TargetFamily.OVER_UNDER,
    PredictionStrategy.SPREAD_ERROR_REGRESSOR: TargetFamily.SPREAD_ERROR,
}

_STRATEGY_MARKET: dict[PredictionStrategy, Market] = {
    PredictionStrategy.TOTAL_POINTS_REGRESSOR: Market.TOTALS,
    PredictionStrategy.LINE_ERROR_REGRESSOR: Market.TOTALS,
    PredictionStrategy.OVER_UNDER_CLASSIFIER: Market.TOTALS,
    PredictionStrategy.SPREAD_ERROR_REGRESSOR: Market.SPREAD,
}

#: The reverse map, used only to infer a strategy from a legacy config that
#: sets target_family alone. OVER_UNDER is intentionally absent: it postdates
#: target_family, so nothing can have been written with it and no legacy config
#: needs to resolve to the classifier.
_TARGET_FAMILY_STRATEGY: dict[TargetFamily, PredictionStrategy] = {
    TargetFamily.TOTAL_POINTS: PredictionStrategy.TOTAL_POINTS_REGRESSOR,
    TargetFamily.LINE_ERROR: PredictionStrategy.LINE_ERROR_REGRESSOR,
}

#: Column holding the binary label. Derived in training_pipeline.data, never
#: present in a raw training CSV.
OVER_LABEL_COL = "OVER_LABEL"

#: Columns that are functions of the final score. Excluded from the feature
#: matrix for every strategy, since any of them reaching X gives away the
#: target regardless of which target is being trained.
#:
#: EXACT names only. The engineered ``DIFF_FROM_LINE_*_BEFORE_*`` rollups are
#: leakage-safe pre-game features and must survive -- a substring match here
#: would silently delete hundreds of legitimate columns.
LEAKING_TARGET_COLUMNS: tuple[str, ...] = (
    "TOTAL_POINTS",
    "LINE_ERROR",
    "OVER_LABEL",
    "DIFF_FROM_LINE",
    # A post-game fact, not a forecast: a game that reached overtime played at
    # least five extra minutes and therefore scored more. It is retained in the
    # CSV purely so training rows can be FILTERED on it (see
    # DataConfig.exclude_overtime_from_training) -- never as a feature.
    "IS_OVERTIME",
)

#: Outcome-derived columns that the 2_1 datasets carry so a spread target can be
#: built and settled, and which must NEVER reach a feature matrix -- for EVERY
#: strategy, totals included. PTS_TEAM_HOME alone gives away TOTAL_POINTS,
#: HOME_MARGIN and SPREAD_ERROR at once.
#:
#: Deliberately NOT merged into LEAKING_TARGET_COLUMNS, and deliberately not
#: appended to ``exclude_cols``. ``exclude_cols`` is inside
#: ``ExperimentConfig.fingerprint()``, so adding names to it would change the
#: fingerprint of every archived totals config and fork every persistent Optuna
#: study -- silently, since a forked study simply starts from zero trials and the
#: run still completes.
#:
#: These are enforced the way GAME_ID already is (see
#: ``training_pipeline.data.prepare_dataset``): dropped centrally just before the
#: feature matrix is built, and asserted absent from X afterwards. That is
#: strictly stronger than a config field a caller can overwrite.
OUTCOME_ONLY_COLUMNS: tuple[str, ...] = (
    "PTS_TEAM_HOME",
    "PTS_TEAM_AWAY",
    "HOME_MARGIN",
    "SPREAD_ERROR",
    # Not produced by any current strategy -- listed so that if a spread
    # classifier is added later, the label cannot reach X during the window
    # between deriving it and remembering to exclude it.
    "COVER_LABEL",
)

#: Target columns whose value IS the edge, so "trust the line" is predicting 0.
#: Both markets' residual regressors. Named once here because at least three
#: places have to agree about it (baseline_pred's space, the decision layer's
#: edge, and putting a prediction back into outcome-level space), and a fourth
#: that forgets produces numbers in the wrong space rather than an error.
RESIDUAL_TARGET_COLUMNS: frozenset[str] = frozenset({"LINE_ERROR", "SPREAD_ERROR"})

#: XGBoost objective for the classifier. Logistic loss is what makes the raw
#: output a probability rather than an arbitrary score, which is the whole
#: point -- the betting rule compares it against a break-even rate.
DEFAULT_CLASSIFIER_OBJECTIVE = "binary:logistic"


class CVStrategy(StrEnum):
    TEST_ANCHORED = "test_anchored"
    LAST_N_SEASONS = "last_n_seasons"
    #: Rolling-origin CV: repeatedly train on everything strictly before an
    #: origin date and predict only the next few game-days, then advance. This
    #: is the protocol ``holdout_evaluation: daily_walk_forward`` already uses
    #: to score the test period, so hyperparameters get selected under the
    #: regime they will actually run in. See splits.build_rolling_origin_plan.
    ROLLING_ORIGIN = "rolling_origin"


class DatasetType(StrEnum):
    """What a training CSV *is*, which decides how cleaning treats it.

    Declared rather than sniffed from the frame's shape. Shape detection worked
    for the two datasets that exist today, but it answers "does this look
    repeated?" when the question is "which dataset is this?" -- and the two come
    apart at exactly the awkward moments: a closing-line CSV that picked up a
    duplicated row, or a single-horizon slice of the intermediate one. A
    declared type also gives a third dataset somewhere to attach itself.

    ADDING A TYPE: add a member here, give it a branch in
    ``training_pipeline.data.redundancy_policy_for``, and say in that branch
    what its grain is. Everything else follows -- the config field, the YAML
    key, the fingerprint and the validation are all driven off this enum.
    """

    #: One row per game: the closing-line dataset, and the shape every other
    #: entry point into cleaning (retraining, same-day prediction) produces.
    #: Correlation pruning runs over every row and every column.
    CLOSING_LINE = "closing_line"

    #: One row per (game, pre-game snapshot). Historical/base features are
    #: computed per game and copied onto each snapshot, so they are judged on
    #: one row per game; snapshot/market features are exempt from correlation
    #: pruning entirely. See column_redundancy.RepeatedMeasuresRedundancy.
    INTERMEDIATE_LINE = "intermediate_line"


#: Minutes before tip-off. Present only on the intermediate-line dataset, where
#: it is what makes the grain one row per (game, snapshot) rather than one per
#: game. Defined here rather than in ``training_pipeline.data`` because the
#: config layer now validates against it and ``data`` imports from here, never
#: the other way round; ``data.SNAPSHOT_COLUMN`` re-exports it so the cleaner
#: and the scorer cannot end up grouping by different spellings.
SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"


#: MedianPruner's warmup before this became fold-count aware. Kept as the floor
#: so no configuration ever becomes *less* patient than the runs already in
#: artifacts/experiments.
_LEGACY_PRUNER_WARMUP_STEPS = 5


class TieTolerancePolicy(StrEnum):
    """How wide the band of "indistinguishable on the primary metric" is.

    Selection is lexicographic: best primary metric within a tolerance, then the
    best betting outcome. The tolerance decides which of the two metrics is
    really choosing the model, and a fixed constant cannot know that -- on cell
    A of public_betting_tradeoff_2026_08 the historical 0.10 admitted 58 of 60
    completed trials, whose entire MAE spread was 0.0987. The primary metric
    ranked nothing and the whole decision fell to pooled OU accuracy, a
    statistic with ~1.7pp of binomial noise on 855 games.

    ``fixed`` uses ``mae_tolerance_abs`` / ``mae_tolerance_pct`` exactly as
    before. Reproducibility, and the only mode whose cutoff does not depend on
    which trials happened to finish.

    ``quantile`` (default) derives the band from the observed trial
    distribution: it is the MAE gap spanning the best ``tie_max_fraction`` of
    completed trials, clamped between ``tie_tolerance_floor`` and
    ``tie_tolerance_cap``. That is a rank rule, deliberately, and it is worth
    being clear about why rather than dressing it up.

    The statistically ideal band would be the standard error of the DIFFERENCE
    in pooled MAE between two trials. It cannot be computed here: it needs each
    trial's per-game predictions, and only aggregates are stored. The available
    substitutes are worse than useless -- the absolute SE of a pooled MAE is
    ~0.37 on this data (larger than the historical constant it would replace,
    so more permissive, not less), and a dispersion estimate expands the band
    exactly when the metric is discriminating well, which is backwards.

    A rank rule sidesteps both. It cannot admit "most trials" by construction,
    it needs no distributional assumption, and it adapts in the right direction:
    a tightly packed frontier yields a narrow MAE band, while a spread-out one
    still admits only the same small share. The cap keeps it from ever being
    more permissive than the old constant.
    """

    FIXED = "fixed"
    QUANTILE = "quantile"


class ObjectiveAggregation(StrEnum):
    """How per-fold metrics become the single number Optuna minimises.

    ``mean`` averages the folds' own metrics, weighting a 4-game fold the same
    as a 12-game one. That was harmless at 12 folds of ~50 games each; it is
    not harmless under ``rolling_origin``, where a fold is a handful of
    game-days and its size swings with the schedule.

    ``pooled`` concatenates every validation prediction and computes one metric
    over all of them, so each GAME contributes equally. Per-fold metrics are
    still recorded for diagnostics either way.
    """

    MEAN = "mean"
    POOLED = "pooled"


class RefitStrategy(StrEnum):
    """How the production model is fitted once evaluation is done."""

    #: Last ``walk_forward.train_games`` games of the FULL dataset (dev+test).
    ROLLING_WINDOW = "rolling_window"
    #: Every available game, ignoring the window.
    FULL_DATASET = "full_dataset"


class HoldoutEvaluation(StrEnum):
    """How the held-out test period is scored.

    ``daily_walk_forward`` retrains once per test day on the games available
    strictly before it, so the test period is scored the way production would
    actually operate. It costs one model fit per game-day.

    ``single_shot`` fits one model on the dev window and predicts the whole
    test period at once. Much cheaper, and useful for smoke runs, but it
    understates how a daily-retrained model behaves because the model never
    absorbs completed test days.
    """

    DAILY_WALK_FORWARD = "daily_walk_forward"
    SINGLE_SHOT = "single_shot"


#: Season types kept when ``DataConfig.exclude_playoffs`` is True. Values must
#: match nba_ou.config.constants.SEASON_TYPE_MAP. Playoff basketball has
#: shorter rotations and slower pace -- in the current training data playoff
#: games average ~11.6 fewer total points than regular-season games -- so it is
#: a materially different distribution from what the models are used to predict.
DEFAULT_ALLOWED_SEASON_TYPES: tuple[str, ...] = (
    "Regular Season",
    "Play-In Tournament",
)

#: The season floor every experiment uses unless it says otherwise. Not a round
#: number picked for tidiness -- it is where the public-betting columns start
#: being populated, and therefore where rows stop failing cleaning.max_na_per_row.
#: See DataConfig.extend_history_dropping_season_gated_columns.
DEFAULT_SEASON_YEAR_FLOOR = 2021

#: How far back the data reaches once the public-betting columns are gone.
#: Measured on training_data_2_0_20260819.csv with max_na_per_row=80: seasons
#: 2019 and 2020 go from 0 surviving rows to 1,012 and 1,086.
#:
#: Not lower, because 2017-2018 are unreachable by any of this -- they carry
#: almost no odds data at all (per-book odds ~87% NaN, consensus opener ~84%)
#: and are dropped earlier, by the missing-data policy's required-column rule,
#: which no row budget can override.
EXTENDED_SEASON_YEAR_FLOOR = 2019

#: The public betting percentage family: what share of tickets and money is on
#: each side. Substrings, matched case-insensitively against column names.
#:
#: This family is most of why the season floor sits at 2021. It is ~45% NaN in
#: season 2020 and ~0.8% from 2021, and there are 272 such columns -- enough
#: missing cells per row on its own to blow past cleaning.max_na_per_row and
#: delete two entire seasons.
#:
#: Kept as a named list because "do public betting percentages earn their place?"
#: is a question worth asking on its own, at an unchanged season floor. It is NOT
#: how extend_history_dropping_season_gated_columns works -- that computes the
#: offenders rather than naming them, and measurably does better (see
#: EXTENDED_SEASON_NAN_SPREAD).
#:
#: "consensus_pct" and not "consensus": ODDS_TOTAL_LINE_consensus_opener is the
#: opening line, a different thing entirely, and it is the configured
#: betting.comparison_line_cols baseline. Dropping it would silently remove the
#: closing-vs-opening comparison.
PUBLIC_BETTING_SUBSTRINGS: tuple[str, ...] = (
    "pct_bets",
    "pct_money",
    "consensus_pct",
)

#: Seasonal availability spread, in percentage points, above which a column is
#: treated as a season indicator and dropped -- from every season, not just the
#: old ones. Used when extend_history_dropping_season_gated_columns is on.
#:
#: 90 means "essentially absent in at least one season and present in another",
#: which is unambiguous. Measured on training_data_2_0_20260819.csv at a 2019
#: floor it selects exactly 213 columns -- 200 public-betting and 13 betmgm
#: price columns -- with no false positives, and beats naming the public-betting
#: family on BOTH axes:
#:
#:     approach                        rows    columns
#:     named public-betting family    8,262      1,171
#:     seasonal spread > 90pp         8,279      1,231
#:     seasonal spread > 50pp         8,283      1,214
#:
#: It keeps the 72 public-betting columns whose availability does NOT vary by
#: season, and catches betmgm price columns (100% absent in 2020, ~0% from 2021)
#: that a public-betting substring list cannot see. Lower it to ~50 to also
#: remove partially-gated columns, at a cost of ~17 columns for ~4 rows.
EXTENDED_SEASON_NAN_SPREAD = 90.0


class DataConfig(BaseModel):
    csv_path: Path

    #: Which dataset this CSV is, and therefore how cleaning judges redundancy.
    #: Defaults to the closing-line shape, which is what every dataset except
    #: the intermediate-line one has. Set it to "intermediate_line" for a
    #: (game, snapshot) CSV; cleaning raises if the declaration and the frame
    #: contradict each other, so a forgotten setting fails loudly rather than
    #: pruning against the wrong view. See DatasetType.
    dataset_type: DatasetType = DatasetType.CLOSING_LINE

    date_col: str = "GAME_DATE"
    season_col: str = "SEASON_YEAR"
    #: Season type is resolved from this column's 3-character prefix rather
    #: than from the SEASON_TYPE text column, which mislabels Play-In
    #: Tournament games as "Playoffs" (see training_pipeline.data).
    game_id_col: str = "GAME_ID"

    #: Drop overtime games from the TRAINING rows only -- CV validation folds,
    #: the holdout and the daily walk-forward's prediction days all keep them.
    #:
    #: The idea being tested: an overtime game's total is inflated by at least
    #: five minutes of extra basketball that no pre-game feature could have
    #: predicted, so those rows may be teaching the model noise. Removing them
    #: from training while still being scored on them measures whether the model
    #: learns the predictable part better without them.
    #:
    #: Deliberately asymmetric. Excluding them from evaluation too would be
    #: scoring on a world that does not exist -- roughly 5.2% of real games go
    #: to overtime and you are paid or not paid on those.
    #:
    #: Default False: every game is used, matching all runs to date. Costs about
    #: 130 of a 2500-game window when enabled.
    exclude_overtime_from_training: bool = False
    overtime_col: str = "IS_OVERTIME"
    season_year_floor: int | None = None
    #: Trade season-gated columns for two extra seasons of games.
    #:
    #: THE choice between the two shapes this dataset can take:
    #:
    #:     False -> 6,148 rows, 1,362 features   seasons 2021+, every column
    #:     True  -> 8,279 rows, ~1,231 features  seasons 2019+, minus 213
    #:
    #: These two effects are one decision, so they are one switch. Admitting
    #: seasons 2019-2020 *without* dropping the columns would be the worst of
    #: both: those columns are absent for whole seasons, and with
    #: cleaning.create_missing_flags off that NaN pattern IS a season indicator
    #: the model can split on directly. Dropping them without admitting the
    #: seasons discards features for nothing.
    #:
    #: When True:
    #:   - cleaning.max_seasonal_nan_spread is set to EXTENDED_SEASON_NAN_SPREAD,
    #:     so any column whose availability identifies the season is dropped from
    #:     EVERY season -- the recent games lose it too, or the column set itself
    #:     would say which season a row came from.
    #:   - season_year_floor drops to EXTENDED_SEASON_YEAR_FLOOR. A floor already
    #:     set below the standard DEFAULT_SEASON_YEAR_FLOOR is left alone, so
    #:     `floor: 2020` + this flag gives the COVID season without the bubble.
    #:
    #: Note what this is NOT: cleaning.nan_threshold cannot express it. That
    #: threshold is computed over the whole window, so a column absent for two
    #: seasons and present for five averages to ~19% NaN -- below any useful
    #: threshold, while still being exactly the thing that gates those seasons.
    #: Measured at a 2019 floor, nan_threshold 50 and 40 drop nothing at all. See
    #: nba_ou...clean_df_for_training.find_season_gated_columns.
    #:
    #: Which arm is better is an open question, not a settled one: the two extra
    #: seasons are the 2019-20 bubble and the 72-game 2020-21, so this trades
    #: columns for 2,131 games of different-distribution history. Run it as an
    #: A/B rather than assuming.
    extend_history_dropping_season_gated_columns: bool = False
    #: Drop games whose season type is not in ``allowed_season_types``.
    #: Defaults to True: regular season + play-in only.
    exclude_playoffs: bool = True
    allowed_season_types: tuple[str, ...] = DEFAULT_ALLOWED_SEASON_TYPES
    #: Human label for the dataset snapshot, e.g. "20260318-all-odds". Purely
    #: descriptive; the checksum below is what actually identifies the bytes.
    data_version: str | None = None
    #: Optional integrity assertion. When set, prepare_dataset verifies the
    #: CSV's sha256 matches and raises if it does not -- so regenerating a CSV
    #: in place is caught loudly instead of silently changing your results.
    #: The actual checksum is always recorded in run metadata regardless.
    expected_checksum: str | None = None

    #: Column carrying minutes-before-tip on the intermediate-line dataset.
    snapshot_col: str = SNAPSHOT_COLUMN

    #: ONE MODEL PER TIMEPOINT. Keep only this horizon, in minutes before tip,
    #: reducing the intermediate-line frame to one row per game.
    #:
    #: The two modes this dataset supports, and the whole difference between
    #: them:
    #:
    #:   null  POOLED. Every snapshot of every game is a training row and
    #:         ``snapshot_col`` is an ordinary feature, so one model learns how
    #:         the mapping changes with time to tip and can price a bet placed
    #:         at any hour. Its betting numbers must be read per horizon --
    #:         see training_pipeline.snapshot_scoring.
    #:   720   ONE HORIZON. One row per game, structurally identical to the
    #:         closing-line dataset. This is the control that answers "is
    #:         pooling earning its complexity?", and the only mode a single
    #:         fixed bet time needs.
    #:
    #: This supersedes scripts/create_train_data/slice_intermediate_snapshot.py,
    #: which did the same thing by writing a second ~187MB CSV. Filtering here
    #: means both arms read the identical bytes, so the checksum pin covers the
    #: control too and the two cannot silently diverge.
    #:
    #: In fingerprint(): it changes which rows exist, so a pooled study must
    #: never resume a single-horizon one.
    snapshot_minutes: int | None = None

    #: Sidecar CSV of closing lines and timestamps, joined on
    #: (``game_id_col``, ``snapshot_col``) AFTER the feature matrix is built.
    #:
    #: The intermediate-line builder holds these in a separate file on purpose:
    #: a model betting the line on the board at T-720 must not be able to read
    #: the closing line, and physical separation is the only version of that
    #: which cannot be undone by a forgotten config entry. Naming the file here
    #: attaches those columns to ``df_full`` only -- never to X -- so
    #: ``betting.comparison_line_cols`` can measure closing-line value.
    scoring_csv_path: Path | None = None

    @model_validator(mode="after")
    def _validate_snapshot_options(self) -> DataConfig:
        if self.dataset_type is DatasetType.INTERMEDIATE_LINE:
            if self.snapshot_minutes is not None and self.snapshot_minutes < 0:
                raise ValueError(
                    "data.snapshot_minutes is minutes BEFORE tip-off and cannot "
                    "be negative."
                )
            return self
        # Both knobs describe a (game, snapshot) frame. On a closing-line CSV
        # there is no snapshot column to filter or join on, so they would be
        # silent no-ops -- the exact failure dataset_type exists to prevent.
        for field, value in (
            ("snapshot_minutes", self.snapshot_minutes),
            ("scoring_csv_path", self.scoring_csv_path),
        ):
            if value is not None:
                raise ValueError(
                    f"data.{field} requires data.dataset_type="
                    f"{DatasetType.INTERMEDIATE_LINE.value!r}; on a "
                    f"{self.dataset_type.value!r} frame it would do nothing."
                )
        return self

    @model_validator(mode="after")
    def _validate_allowed_season_types(self) -> DataConfig:
        if self.exclude_playoffs and not self.allowed_season_types:
            raise ValueError(
                "data.allowed_season_types must not be empty when "
                "data.exclude_playoffs is True (it would drop every row)."
            )
        unknown = set(self.allowed_season_types) - set(SEASON_TYPE_MAP.values())
        if unknown:
            raise ValueError(
                f"Unknown season types {sorted(unknown)}. Valid values: "
                f"{sorted(set(SEASON_TYPE_MAP.values()))}"
            )
        return self


class CleaningConfig(BaseModel):
    """Mirrors nba_ou.data_processing.missing_data.clean_df_for_training.clean_dataframe_for_training."""

    nan_threshold: float = 5.0
    #: The general correlation threshold, applied to every column not matched by
    #: ``corr_threshold_overrides``.
    corr_threshold: float = 0.95
    #: Per-substring thresholds overriding ``corr_threshold``. A pair is judged
    #: against the more tolerant of its two columns, which decides very little in
    #: practice: on training_data_2_0_20260819.csv only 6 of the 915 pairs above
    #: 0.95 are odds/non-odds, because the two groups form near-disjoint
    #: correlation clusters.
    #:
    #: None means "use DEFAULT_CORR_THRESHOLD_OVERRIDES" -- odds features held to
    #: 0.99 while everything else is pruned at 0.95, taking that dataset from
    #: 1,474 surviving numeric columns to 1,366. ``{}`` means "no overrides", one
    #: threshold everywhere. The two are different, deliberately.
    #:
    #: The previous single 0.995 threshold was biased the wrong way: 83 of the
    #: 104 columns it dropped were odds-derived, since the same line quoted by
    #: seven books is the most internally redundant block in the frame.
    corr_threshold_overrides: dict[str, float] | None = None
    #: Drop columns whose per-season NaN rate varies by more than this many
    #: percentage points -- their availability identifies the season, which a
    #: model can split on directly. None disables the step; it is also a no-op
    #: on a single-season frame, which is the same-day prediction case.
    #: data.extend_history_dropping_season_gated_columns sets it.
    max_seasonal_nan_spread: float | None = None
    max_na_per_row: int = -1
    create_missing_flags: bool = False
    keep_columns: list[str] | None = None
    exclude_cols_containing: list[str] | None = None
    keep_all_cols: bool = False
    verbose: int = 1
    strict_mode: int = -1
    strict_mode_exclude_cols: list[str] | None = None


class HoldoutConfig(BaseModel):
    """How much of the tail of the data is held back for final scoring.

    Exactly one of the three must be set.

    ``test_days`` is the one to prefer, for two reasons. The daily walk-forward
    retrains once per game-day, so a fixed number of days fixes how much
    production operation is actually being simulated. And two runs on different
    datasets get the *identical calendar window*, which a fraction does not: a
    5% holdout produced Mar 8 / 287 games on one dataset and Mar 7 / 293 on
    another in the same A/B, quietly making the two numbers non-comparable.
    Fixing games would equalise the count but could still shift the window;
    fixing days compares the same games and surfaces coverage differences
    instead of hiding them.

    One precondition: the window is counted back from each dataset's own last
    game, so two datasets align only if they END on the same date. A CSV
    rebuilt to a later date shifts its whole window -- which is why the
    comparison notebook's cohort check still earns its place.

    On sizing: no holdout here can establish profitability. At -110 a true 55%
    win rate needs ~1400 bets before its interval clears break-even, and even a
    full season (~1236 games, ~494 bets) lands at [50.4%, 59.2%]. The holdout's
    job is to be an honest out-of-sample check on the CV number and to run
    enough retrains to be meaningful -- statistical power comes from the CV
    folds and from accumulating results over time.
    """

    test_size: float | None = 0.15
    test_games: int | None = None
    #: Calendar days, counted back from the last game in the data. 60 days is
    #: ~55 game-days (retrains), ~417 games and ~166 bets on this dataset, for
    #: 6.8% of the rows.
    test_days: int | None = None

    @model_validator(mode="after")
    def _exactly_one_sizing_rule(self) -> HoldoutConfig:
        provided = [
            name
            for name, value in (
                ("test_size", self.test_size),
                ("test_games", self.test_games),
                ("test_days", self.test_days),
            )
            if value is not None
        ]
        if len(provided) != 1:
            raise ValueError(
                "Provide exactly one of holdout.test_size, holdout.test_games or "
                f"holdout.test_days (got: {provided or 'none'}). Setting a "
                "fraction in _base.yaml and days in an experiment counts as two "
                "-- null out the one you are replacing."
            )
        if self.test_days is not None and self.test_days <= 0:
            raise ValueError("holdout.test_days must be > 0.")
        return self


class WalkForwardConfig(BaseModel):
    strategy: CVStrategy = CVStrategy.TEST_ANCHORED
    test_games: int = 30
    step_games_between_tests: int | None = None
    train_games: int | None = 5000
    train_seasons: int | None = None
    min_train_games: int = 300
    max_folds: int | None = 12
    fold_selection: str = "latest"
    exclude_test_months: tuple[int, ...] = (5, 6)
    require_same_season_test: bool = True
    verbose: int = 0

    # --- rolling_origin only -------------------------------------------------
    #: How many GAME-DAYS each origin predicts before retraining. Counted in
    #: days that actually contain games, so a dark day on the NBA calendar is
    #: skipped rather than consuming part of the window -- "the next 4 days with
    #: games", not "the next 4 dates". 1 reproduces the daily walk-forward
    #: exactly; 4 costs a quarter of the fits and, measured across 36 runs'
    #: existing folds, loses nothing (win rate is flat over days 0-6 of a fold).
    retrain_every_days: int = 4
    #: Approximate size of the chronological validation region, in games,
    #: measured back from the end of dev. Approximate by construction: the
    #: region is grown whole game-days at a time so no day is ever split
    #: between training and validation. None uses every fold the data allows,
    #: subject to max_folds.
    eval_span_games: int | None = None
    #: Floor on how many GAMES a validation fold must contain. None (default)
    #: keeps the historical behaviour exactly: a fold is ``retrain_every_days``
    #: game-days and whatever games those days happened to hold.
    #:
    #: When set, a fold starts at ``retrain_every_days`` days and then absorbs
    #: further WHOLE game-days until it holds at least this many games. Whole
    #: days only -- splitting a date between train and validation is the leak
    #: the daily walk-forward exists to prevent -- and a fold still stops at a
    #: season boundary when ``require_same_season_test`` is on, so it can end up
    #: short there.
    #:
    #: Why it exists: on cell A's real layout the fold sizes ran
    #: 2, 15, 17, 26, ... 36. Under ``objective_aggregation: pooled`` a 2-game
    #: fold is harmless (it carries 2/855 of the weight, which is correct), but
    #: under ``mean`` it carries 1/30 -- and the pruner reads a running metric
    #: whose early steps are those same folds. This makes the floor explicit
    #: rather than leaving it to the NBA calendar.
    min_validation_games: int | None = None
    #: Discrete training-window sizes for Optuna to choose between. When set,
    #: ``train_games`` becomes a tuned hyperparameter sampled once per trial and
    #: held fixed across that trial's folds; ``train_games`` below is then only
    #: the fallback used when tuning is skipped. Under ``test_anchored``, folds
    #: are constructed with their full histories first, then every candidate
    #: trims those SAME folds to its requested history size.
    train_games_choices: tuple[int, ...] | None = None

    @property
    def tunes_train_games(self) -> bool:
        return bool(self.train_games_choices)

    @model_validator(mode="after")
    def _validate_rolling_origin(self) -> WalkForwardConfig:
        if self.retrain_every_days <= 0:
            raise ValueError("walk_forward.retrain_every_days must be > 0.")
        if self.min_validation_games is not None:
            if self.min_validation_games <= 0:
                raise ValueError(
                    "walk_forward.min_validation_games must be > 0 when set."
                )
            if self.strategy != CVStrategy.ROLLING_ORIGIN:
                raise ValueError(
                    "walk_forward.min_validation_games requires "
                    "walk_forward.strategy='rolling_origin'. The other "
                    "splitters already size their folds in games "
                    "(walk_forward.test_games), so the knob would silently do "
                    "nothing there."
                )
            if (
                self.eval_span_games is not None
                and self.min_validation_games > self.eval_span_games
            ):
                raise ValueError(
                    f"walk_forward.min_validation_games="
                    f"{self.min_validation_games} exceeds "
                    f"walk_forward.eval_span_games={self.eval_span_games}, so "
                    "the whole evaluation region could not fill a single fold."
                )
        if self.eval_span_games is not None and self.eval_span_games <= 0:
            raise ValueError("walk_forward.eval_span_games must be > 0 when set.")
        if self.train_games_choices is not None:
            if len(set(self.train_games_choices)) < 2:
                raise ValueError(
                    "walk_forward.train_games_choices needs at least two "
                    "distinct values -- a one-element categorical is a fixed "
                    "value that costs a search dimension. Set "
                    "walk_forward.train_games instead."
                )
            if any(choice <= 0 for choice in self.train_games_choices):
                raise ValueError(
                    "walk_forward.train_games_choices values must all be > 0."
                )
            if self.strategy not in (CVStrategy.ROLLING_ORIGIN, CVStrategy.TEST_ANCHORED):
                raise ValueError(
                    "walk_forward.train_games_choices requires "
                    "walk_forward.strategy='rolling_origin' or 'test_anchored'. "
                    f"Got {self.strategy.value!r}."
                )
        return self

    @model_validator(mode="after")
    def _train_seasons_required_for_last_n_seasons(self) -> WalkForwardConfig:
        if self.strategy == CVStrategy.LAST_N_SEASONS:
            if self.train_seasons is None or self.train_seasons <= 0:
                raise ValueError(
                    "walk_forward.train_seasons must be a positive int when "
                    "walk_forward.strategy == 'last_n_seasons'."
                )
        return self


#: Weights are exp(-lambda * age_in_days), so lambda maps to a half-life:
#:
#:   lambda   half-life        weight of the oldest game in a 2500-game window
#:   0.0005   1386 d (3.8 yr)  68%    gentle -- barely tilts toward recency
#:   0.001     693 d (1.9 yr)  46%
#:   0.002     347 d (0.9 yr)  21%
#:   0.005     139 d (0.4 yr)   2%    moderate -- the practical ceiling
#:   0.010      69 d (0.2 yr)   0.05% window effectively discarded
#:   0.050      14 d           ~0%    absurd
#:
#: A 2500-game window spans roughly 770 calendar days, so anything above
#: ~0.005 shrinks the effective training set to a few weeks and makes the
#: configured window meaningless. The bounds stop there deliberately.
_DEFAULT_SAMPLE_WEIGHT_LAMBDA_BOUNDS = (0.0005, 0.005)


class SampleWeightConfig(BaseModel):
    """Exponential recency weighting: recent games count for more.

    Available to BOTH target families. Upstream's optuna_total_points.py has no
    sample-weight parameters, so training_pipeline supplies its own objective
    and final-fit path (see tuning.run_objective / tuning.fit_final_model) that
    apply weights identically for TOTAL_POINTS and LINE_ERROR.

    A weight of exp(-lambda * age_in_days) is applied per training row, so
    lambda=0.005 gives a game one year old roughly 16% of today's weight.
    """

    enabled: bool = False
    lambda_: float | None = None
    tune_lambda: bool = False
    lambda_bounds: tuple[float, float] = _DEFAULT_SAMPLE_WEIGHT_LAMBDA_BOUNDS
    #: When tuning, let Optuna also decide *whether* to weight at all, via a
    #: categorical use_sample_weight parameter. Without this the search is
    #: forced to weight, and "no weighting" can only be approximated by
    #: driving lambda toward zero -- which a log-uniform range can never
    #: actually reach, and which wastes sampler resolution near the floor.
    #: With it, "off" is one clean binary decision the sampler can evaluate,
    #: and you can read off how many trials preferred it.
    allow_unweighted: bool = True
    date_col: str = "GAME_DATE"

    @model_validator(mode="after")
    def _validate_bounds(self) -> SampleWeightConfig:
        low, high = self.lambda_bounds
        if low <= 0:
            raise ValueError(
                "sample_weight.lambda_bounds lower bound must be > 0 (the range "
                "is log-uniform). Use allow_unweighted=True to let Optuna turn "
                "weighting off instead of trying to sample zero."
            )
        if low >= high:
            raise ValueError(
                "sample_weight.lambda_bounds must be (low, high) with low < high."
            )
        if self.lambda_ is not None and self.lambda_ < 0:
            raise ValueError("sample_weight.lambda_ must be >= 0.")
        return self

    def is_default(self) -> bool:
        return self == SampleWeightConfig()


class IntRange(BaseModel):
    low: int
    high: int
    log: bool = False

    @model_validator(mode="after")
    def _validate(self) -> IntRange:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        return self


class FloatRange(BaseModel):
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _validate(self) -> FloatRange:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        if self.log and self.low <= 0:
            raise ValueError("log-scaled ranges require low > 0.")
        return self


class SearchSpaceConfig(BaseModel):
    """XGBoost hyperparameter search space for Optuna.

    Defaults are tuned for this problem's shape: ~2500 training games against
    1400-2100 features (p/n roughly 0.6-0.8), with a signal worth only ~3% of
    baseline MAE. Two ranges deviate deliberately from the legacy space in
    ``nba_ou.modeling.optuna_*.py`` (preserved as UPSTREAM_SEARCH_SPACE):

    * ``colsample_bytree`` reaches down to 0.05. At the old 0.35 floor a tree
      chose ~15 splits from ~720 candidate features on a noisy bootstrap,
      which mines noise rather than signal.
    * ``gamma`` spans 1-500 (log). For ``reg:squarederror`` with residual
      sigma ~19.5, a 500-row node's gradient sum fluctuates by ~436 from
      chance alone, so chance split gains are O(100s) -- the old 0.1-3.0 range
      pruned nothing at all. Gamma is the direct "only split on a real gain"
      control, which is exactly what a noise-dominated target needs.

    ``min_child_weight`` also starts at 20 rather than 5: with squared error
    the hessian is 1, so it is effectively minimum samples per leaf, and a
    leaf holding 5 of 2500 games is noise. ``max_depth`` reaches down to 1
    because additive stumps cannot manufacture spurious interactions.

    These are reasoned from the problem's structure, not measured -- run the
    A/B against UPSTREAM_SEARCH_SPACE on a fixed holdout before trusting them.
    Note a wider space needs more trials to search, so under a tight budget it
    can underperform a narrow one.

    Changing any range changes the config fingerprint, so a persistent study
    will start fresh rather than mixing incomparable trials.
    """

    max_depth: IntRange = IntRange(low=1, high=4)
    min_child_weight: FloatRange = FloatRange(low=20.0, high=250.0, log=True)
    gamma: FloatRange = FloatRange(low=1.0, high=500.0, log=True)
    subsample: FloatRange = FloatRange(low=0.55, high=0.95)
    colsample_bytree: FloatRange = FloatRange(low=0.05, high=0.8, log=True)
    learning_rate: FloatRange = FloatRange(low=0.0075, high=0.06, log=True)
    reg_alpha: FloatRange = FloatRange(low=1e-2, high=20.0, log=True)
    reg_lambda: FloatRange = FloatRange(low=1.0, high=500.0, log=True)
    #: Upper bound on boosting rounds in the LEGACY early-stopping mode, where
    #: each fold's own validation set picks the real number. Ignored once
    #: ``n_estimators_range`` is set.
    n_estimators: int = 1000
    early_stopping_rounds: int = 70
    #: Search range for the boosting rounds. When set, ``n_estimators`` becomes
    #: an ordinary tuned hyperparameter: sampled once per trial, held fixed
    #: across every fold, and used verbatim by the holdout walk-forward and the
    #: production refit. Fold-level early stopping is disabled, because the two
    #: are alternatives -- see OptunaConfig.tune_n_estimators for why the old
    #: arrangement was a problem. Left None the legacy behaviour is unchanged.
    n_estimators_range: IntRange | None = None


#: The search space hardcoded in nba_ou.modeling.optuna_*.py, kept verbatim so
#: pre-existing results stay reproducible and so the A/B against the new
#: defaults is one config change. A test asserts this still produces draws
#: identical to upstream's builders under a seeded sampler.
UPSTREAM_SEARCH_SPACE = SearchSpaceConfig(
    max_depth=IntRange(low=2, high=4),
    min_child_weight=FloatRange(low=5.0, high=60.0, log=True),
    gamma=FloatRange(low=0.1, high=3.0),
    subsample=FloatRange(low=0.55, high=0.95),
    colsample_bytree=FloatRange(low=0.35, high=0.8),
    learning_rate=FloatRange(low=0.0075, high=0.06, log=True),
    reg_alpha=FloatRange(low=1e-2, high=20.0, log=True),
    reg_lambda=FloatRange(low=1.0, high=50.0, log=True),
)


#: The same space rescaled for ``binary:logistic``. Applied automatically to
#: classifier runs -- see ExperimentConfig._scale_search_space_to_the_objective.
#:
#: The regression defaults above were reasoned entirely in squared-error terms,
#: where the hessian is 1 per sample. Logistic loss has hessian ``p(1-p)``,
#: which at this problem's ~50% base rate is 0.25. Reusing the regression space
#: therefore does not merely "regularise a bit differently" -- it changes what
#: the numbers MEAN, by roughly a factor of 4 on leaf size and 387 on gain:
#:
#:   min_child_weight is a sum of hessians, so 20-250 becomes 80-1000 SAMPLES
#:   per leaf instead of 20-250. Against a 2500-game window the upper end
#:   permits at most two leaves.
#:
#:   Chance-level split gain is ~0.49 under logistic versus ~189 under squared
#:   error (500-row node, lambda=1). A gamma FLOOR of 1.0 therefore prunes every
#:   split, real or not -- and since gamma is sampled log-uniformly over 1-500,
#:   its geometric midpoint of ~22 puts half the space beyond that cliff.
#:
#: Measured on synthetic data carrying a planted, known signal (true P(OVER)
#: spanning 0.18-0.81):
#:
#:   space / region                  predicted p-range   log-loss improvement
#:   regression floor  (mcw 20,  g 1)    0.118-0.866            +0.321
#:   regression middle (mcw 125, g 22)   0.468-0.530            +0.006
#:   regression ceiling(mcw 250, g 500)  0.504-0.504            -0.000  (constant!)
#:   classifier floor  (mcw 5,   g .003) 0.083-0.923            +0.455
#:   classifier middle (mcw 30,  g .06)  0.178-0.833            +0.241
#:   classifier ceiling(mcw 60,  g 1.3)  0.329-0.707            +0.071
#:
#: The regression middle reproduces 0.468-0.530 -- almost exactly the 0.47-0.52
#: seen in the first real classifier smoke run. That run's "no signal" reading
#: was the search space strangling the model, not an absence of signal.
#:
#: Only the four scale-dependent parameters move. max_depth, subsample,
#: colsample_bytree and learning_rate are dimensionless here and stay put.
CLASSIFIER_SEARCH_SPACE = SearchSpaceConfig(
    max_depth=IntRange(low=1, high=4),
    # /4: keeps 20-240 actual samples per leaf, matching the regression intent.
    min_child_weight=FloatRange(low=5.0, high=60.0, log=True),
    # /387, the measured gain-scale ratio between the two objectives.
    gamma=FloatRange(low=0.002, high=2.0, log=True),
    subsample=FloatRange(low=0.55, high=0.95),
    colsample_bytree=FloatRange(low=0.05, high=0.8, log=True),
    learning_rate=FloatRange(low=0.0075, high=0.06, log=True),
    # Leaf weights are log-odds, roughly 10x smaller than points-space weights.
    reg_alpha=FloatRange(low=1e-3, high=2.0, log=True),
    # Competes with H = 0.25n rather than n.
    reg_lambda=FloatRange(low=0.5, high=150.0, log=True),
)


#: Default ``n_estimators`` search range per strategy, applied automatically
#: when ``optuna.tune_n_estimators`` is on and no range was stated -- the same
#: "fires only when the value was inherited rather than chosen" rule that
#: CLASSIFIER_SEARCH_SPACE uses.
#:
#: The bounds are read off where fold-level early stopping actually landed
#: across the 38 runs in artifacts/experiments, per strategy (median over each
#: selected trial's folds, and the p10-p90 of that median across all trials):
#:
#:   strategy                p10   median   p90   trials under 50 rounds
#:   line_error               21      41     75            64%
#:   over_under_classifier    13      31     53            85%
#:   total_points             75     112    269             2%
#:
#: So the two line-relative strategies live at a few dozen rounds and
#: total_points at a few hundred -- it has to reproduce the line itself before
#: it reaches the residual, which is a large, genuinely learnable component.
#: A single shared 100-1500 range would sit entirely above where two of the
#: three strategies operate. Log scale because the useful resolution is
#: multiplicative: 20 vs 40 rounds matters, 420 vs 440 does not.
#:
#: Ranges extend past the observed p90 on purpose. Early stopping never had a
#: reason to explore beyond where its noisy 50-game eval set happened to dip,
#: and low ``colsample_bytree`` needs many rounds before every feature has been
#: offered even once (at colsample 0.07, 50 rounds offers each feature ~3.5
#: times out of ~1458 features) -- a region the old setup could not reach.
N_ESTIMATORS_RANGES: dict[PredictionStrategy, IntRange] = {
    PredictionStrategy.LINE_ERROR_REGRESSOR: IntRange(low=10, high=500, log=True),
    # Mirrors LINE_ERROR: the same residual-regression shape on a target with a
    # comparable scale (spread errors have s.d. ~13 points against totals'
    # ~15.7), so the range that suits one suits the other.
    PredictionStrategy.SPREAD_ERROR_REGRESSOR: IntRange(low=10, high=500, log=True),
    PredictionStrategy.OVER_UNDER_CLASSIFIER: IntRange(low=10, high=500, log=True),
    PredictionStrategy.TOTAL_POINTS_REGRESSOR: IntRange(low=30, high=1000, log=True),
}


class OptunaConfig(BaseModel):
    """Optuna tuning knobs.

    Note on ``persistent_storage``: when enabled, the study is stored in a
    SQLite file keyed by the experiment name *and a fingerprint of the
    data/CV-affecting config* (see ExperimentConfig.fingerprint), so changing
    the CSV, cleaning thresholds, or fold layout starts a fresh study instead
    of silently resuming trials computed on different data.

    Note on ``n_trials`` when resuming: Optuna's ``study.optimize(n_trials=N)``
    runs N *additional* trials on a resumed study, it is not a target total.
    Re-running the same persistent config with n_trials=80 twice yields 160
    trials.
    """

    n_trials: int = 80
    timeout: int | None = None
    objective_name: str = "reg:squarederror"
    study_name: str | None = None
    persistent_storage: bool = False
    storage_filename: str = "optuna_study.db"
    load_if_exists: bool = True
    mae_tolerance_abs: float | None = 0.10
    mae_tolerance_pct: float | None = None
    #: Which rule turns the trial MAEs into a tie band. See TieTolerancePolicy.
    #: 'fixed' reproduces every run made before this existed.
    tie_tolerance: TieTolerancePolicy = TieTolerancePolicy.QUANTILE
    #: Under 'quantile': the share of COMPLETED trials the band may span. 0.10
    #: means "OU accuracy breaks ties among roughly the best tenth", which is
    #: what a tie-break is for. The realised share can exceed it only through
    #: exact ties at the boundary or through tie_tolerance_floor, both of which
    #: are reported.
    tie_max_fraction: float = 0.10
    #: Lower bound on the band. Exists so trials separated by floating-point
    #: dust are not ranked as if the difference were real. Raising it can admit
    #: more than tie_max_fraction -- that is the point of a floor, and the
    #: realised fraction is recorded either way.
    tie_tolerance_floor: float = 0.001
    #: Hard maximum, in primary-metric units. The band can never be wider than
    #: this however the data falls. 0.10 is the historical constant, so 'quantile'
    #: is guaranteed to be at least as strict as the old behaviour.
    tie_tolerance_cap: float = 0.10
    #: Diagnostic threshold. If the tie set ends up larger than this share of
    #: completed trials, the secondary metric is effectively selecting and the
    #: run says so out loud instead of letting it pass.
    tie_warn_fraction: float = 0.25
    #: Lexicographic tolerance for the CLASSIFIER, in log-loss units. Two
    #: orders of magnitude tighter than mae_tolerance_abs because log loss has
    #: almost no dynamic range on a ~50/50 outcome: a perfectly calibrated 55%
    #: model scores 0.68814 against 0.69315 for a coin flip, a total spread of
    #: 0.005. 0.002 therefore admits trials within roughly half the distance
    #: between "worthless" and "genuinely good", which is the same spirit as
    #: 0.10 on a ~13.3 MAE.
    #:
    #: The tolerance matters more here than for the regressors: simulated at
    #: 600 validation games, log loss ranks a truly-53% trial above a truly-52%
    #: one only 64% of the time. The ordering is close to noise, so selection
    #: leans on the secondary criterion (betting outcome) by design.
    logloss_tolerance_abs: float | None = 0.002
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)

    #: Sample ``n_estimators`` per trial instead of letting each fold early-stop
    #: on its own validation set.
    #:
    #: What was wrong with the old arrangement, in three parts:
    #:
    #: 1. The fold's ``eval_set`` WAS the fold that then scored the trial, so
    #:    the reported metric is the minimum of a noisy curve over ~1000
    #:    candidate stopping points. That bias is not constant across trials --
    #:    a higher-capacity trial harvests more of it -- so it tilted the
    #:    ranking, not just the level.
    #: 2. The number chosen in CV was never the number used. Every downstream
    #:    fit took ``median_best_iteration`` across folds and no early stopping.
    #:    Measured on the selected trials in artifacts/experiments, folds within
    #:    ONE trial stopped anywhere from 2 to 922 rounds; the coefficient of
    #:    variation across folds is 1.04 for line_error and 1.01 for the
    #:    classifier -- a CV of 1.0 is the signature of a memoryless process,
    #:    i.e. the stopping point was not measuring anything.
    #: 3. Optuna could not control capacity. Capacity is roughly
    #:    ``learning_rate x rounds``; the sampler set the rate and early
    #:    stopping then set the rounds in reaction. Measured Spearman
    #:    correlation between them is -0.75 to -0.90 for total_points, so the
    #:    sampler was modelling an axis it did not own.
    #:
    #: With this on, the round count is part of the configuration being scored,
    #: evaluated exactly as it will be used, on games that had no say in
    #: choosing it. Off by default so every existing config reproduces byte for
    #: byte.
    tune_n_estimators: bool = False
    #: See ObjectiveAggregation. ``pooled`` is required in spirit by
    #: rolling_origin, whose folds vary in size; the default stays ``mean`` so
    #: existing configs are unchanged.
    objective_aggregation: ObjectiveAggregation = ObjectiveAggregation.MEAN
    #: Folds to complete before the pruner may act. None derives it from the
    #: fold count via ``pruner_warmup_fraction``, which is what you want once a
    #: fold is a handful of game-days: the old fixed 5 was 5 of 12 folds (~250
    #: games) and becomes 5 of ~50 (~35 games), i.e. pruning on noise.
    pruner_warmup_steps: int | None = None
    #: Fraction of the fold count to use as warmup when pruner_warmup_steps is
    #: None. 0.25 of 28 folds is 7.
    pruner_warmup_fraction: float = 0.25

    #: Skip tuning entirely and use these hyperparameters. Populate from a
    #: previous run with training_pipeline.reuse.load_run_hyperparameters(),
    #: which prints a ready-to-paste YAML block -- that is how you avoid
    #: re-running a long study just to reuse what it already found.
    fixed_params: dict[str, Any] | None = None
    fixed_n_estimators: int | None = None
    fixed_sample_weight_lambda: float | None = None

    @property
    def skip_tuning(self) -> bool:
        return self.fixed_params is not None

    @model_validator(mode="after")
    def _validate_fixed_params(self) -> OptunaConfig:
        if self.fixed_params is not None and not self.fixed_n_estimators:
            raise ValueError(
                "optuna.fixed_n_estimators is required alongside "
                "optuna.fixed_params (there is no study to infer it from)."
            )
        return self

    @model_validator(mode="after")
    def _validate_pruner_warmup(self) -> OptunaConfig:
        if self.pruner_warmup_steps is not None and self.pruner_warmup_steps < 1:
            raise ValueError("optuna.pruner_warmup_steps must be >= 1 when set.")
        if not 0.0 < self.pruner_warmup_fraction <= 1.0:
            raise ValueError(
                "optuna.pruner_warmup_fraction must be in (0, 1]. It is a "
                "fraction of the fold count, not a number of folds."
            )
        return self

    @model_validator(mode="after")
    def _at_most_one_mae_tolerance(self) -> OptunaConfig:
        if self.tie_tolerance_floor > self.tie_tolerance_cap:
            raise ValueError(
                f"optuna.tie_tolerance_floor={self.tie_tolerance_floor} exceeds "
                f"optuna.tie_tolerance_cap={self.tie_tolerance_cap}; the band "
                "would have no valid width."
            )
        if not 0.0 < self.tie_max_fraction <= 1.0:
            raise ValueError("optuna.tie_max_fraction must be in (0, 1].")
        if not 0.0 < self.tie_warn_fraction <= 1.0:
            raise ValueError("optuna.tie_warn_fraction must be in (0, 1].")
        if self.mae_tolerance_abs is not None and self.mae_tolerance_pct is not None:
            raise ValueError(
                "Provide at most one of optuna.mae_tolerance_abs or "
                "optuna.mae_tolerance_pct."
            )
        return self


class RefitConfig(BaseModel):
    """How the final model is fitted once tuning has chosen hyperparameters.

    There is deliberately no ``train_games`` here: the rolling-window size is
    ``walk_forward.train_games``, so the final model is trained on the same
    amount of history each CV fold used. Two separate knobs could silently
    diverge, which would mean selecting hyperparameters for one training-set
    size and then fitting on another.
    """

    strategy: RefitStrategy = RefitStrategy.ROLLING_WINDOW
    use_lexicographic_selection: bool = True
    #: Train (and save) the production model. Off by default: most experiments
    #: only want the evaluation. When on, the model is fitted on the freshest
    #: data available -- the last walk_forward.train_games games of dev AND
    #: test, with no split -- because in production nothing is held back.
    train_production_model: bool = False


class BettingConfig(BaseModel):
    """How model predictions get turned into bets and scored for profit.

    Post-hoc evaluation only -- these settings never change what a model or an
    Optuna trial *is*, so they are excluded from ExperimentConfig.fingerprint()
    and can be changed without forking a persistent study.
    """

    edge_thresholds: tuple[float, ...] = DEFAULT_EDGE_THRESHOLDS
    # The threshold whose metrics get promoted to the leaderboard. 2.0 points
    # is a reasonable default: it filters out coin-flip calls where the model
    # essentially agrees with the line.
    primary_edge_threshold: float = 2.0
    flat_decimal_odds: float = DECIMAL_ODDS_MINUS_110
    # Opt in to real per-book prices (decimal) instead of the flat price, e.g.
    # "total_bet365_price_over" / "total_bet365_price_under". Rows with missing
    # or invalid decimal prices fall back to flat_decimal_odds; American-format
    # columns are rejected before evaluation.
    over_price_col: str | None = None
    under_price_col: str | None = None

    #: Spread-market prices, in the same role as over/under above: the price for
    #: backing HOME and the price for backing AWAY.
    #:
    #: Separate fields rather than reusing over/under because the mapping is not
    #: obvious and getting it backwards is silent. Verified on the real data:
    #: HOME is the "over"-equivalent side, since predicted_edge > 0 means the home
    #: team beats the spread. On the intermediate dataset the snapshot panel
    #: stores the two sides as LEFT/RIGHT, where RIGHT is HOME -- confirmed by
    #: the moneyline, whose cheaper RIGHT side won 68.8% of the time.
    #:
    #: Closing dataset: ODDS_spread_bet365_price_home / _price_away.
    #: The intermediate ODDS_SNAP_SPR_*_PRICE_LEFT/RIGHT fields are American
    #: odds and must not be configured here without first converting them to
    #: decimal prices that belong to the exact line being settled.
    home_price_col: str | None = None
    away_price_col: str | None = None

    #: Also score the CV folds for profit, not just the holdout. Costs one extra
    #: fit per fold (~= one Optuna trial) and buys roughly 5x the bet volume:
    #: 12 folds x ~50 validation games ~= 600, versus ~290 in a 5% holdout. At
    #: these sample sizes volume is the binding constraint on being able to tell
    #: a real edge from a lucky one, so this is close to free power.
    #:
    #: Read it as a COMPARISON between configurations, not as an unbiased
    #: estimate of live ROI: the hyperparameters were selected on these same
    #: folds, so the number is optimistically biased. The holdout stays the
    #: out-of-sample estimate.
    evaluate_cv_folds: bool = True

    #: Expected-value thresholds for the classifier: the probability-side
    #: counterpart to edge_thresholds. At decimal odds d a win probability p
    #: has EV = p*d - 1 per unit staked, positive exactly when p beats the
    #: break-even rate 1/d (52.38% at -110). EV is the right common currency --
    #: comparable across sides, across prices, and interpretable next to a
    #: regressor's points edge, which a raw probability is not.
    ev_thresholds: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    #: Headline EV threshold: "bet whenever the model thinks the price is wrong
    #: at all". Starting at 0.0 rather than imposing a safety margin is
    #: deliberate. The regressor's 2.0-point primary_edge_threshold was
    #: measured to buy nothing -- the [0.0,0.5) edge bucket won 55.4% while
    #: [3.5,4.0) won 47.8% -- yet it discarded more than half the bets.
    #: Choosing a margin before seeing this model's own sweep repeats that
    #: mistake. Read ev_thresholds first, then impose one if the data earns it.
    primary_ev_threshold: float = 0.0
    #: Buckets for the reliability table (predicted probability vs observed
    #: frequency). 10 gives ~60 games per bucket across the CV folds.
    calibration_buckets: int = 10

    #: Extra total-line columns to re-score the very same predictions against,
    #: for information only -- nothing about training or selection changes.
    #:
    #: Why it matters: bets are settled here against the CLOSING line, which by
    #: definition is the last price quoted and therefore one you cannot actually
    #: take. Beating it is the strict test of "do I have information", but it is
    #: not the number you would have got. In this dataset the line moves 2.54
    #: points on average and moves at all in 93% of games -- larger than the
    #: default 2.0-point bet trigger -- so an edge measured against the close can
    #: correspond to a materially different bet against the open.
    #:
    #: "ODDS_TOTAL_LINE_consensus_opener" is the natural counterpart. Columns absent
    #: from the data are skipped rather than raising, since line availability
    #: varies across CSV snapshots.
    comparison_line_cols: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> BettingConfig:
        if not self.edge_thresholds:
            raise ValueError("betting.edge_thresholds must not be empty.")
        if self.flat_decimal_odds <= 1.0:
            raise ValueError("betting.flat_decimal_odds must be > 1.0.")
        if (self.over_price_col is None) != (self.under_price_col is None):
            raise ValueError(
                "Provide both betting.over_price_col and betting.under_price_col, "
                "or neither."
            )
        if (self.home_price_col is None) != (self.away_price_col is None):
            raise ValueError(
                "Provide both betting.home_price_col and betting.away_price_col, "
                "or neither."
            )
        if self.primary_edge_threshold not in self.edge_thresholds:
            raise ValueError(
                f"betting.primary_edge_threshold ({self.primary_edge_threshold}) must "
                f"be one of betting.edge_thresholds ({self.edge_thresholds})."
            )
        if not self.ev_thresholds:
            raise ValueError("betting.ev_thresholds must not be empty.")
        if self.primary_ev_threshold not in self.ev_thresholds:
            raise ValueError(
                f"betting.primary_ev_threshold ({self.primary_ev_threshold}) must be "
                f"one of betting.ev_thresholds ({self.ev_thresholds})."
            )
        if self.calibration_buckets < 2:
            raise ValueError("betting.calibration_buckets must be >= 2.")
        return self


#: Hyperparameters for the daily backtest when none are supplied. These are the
#: hand-tuned values from the repo's 3-season total_points notebook -- a known
#: working configuration rather than an invented one.
DEFAULT_BACKTEST_XGB_PARAMS: dict[str, float | int] = {
    "max_depth": 4,
    "learning_rate": 0.057,
    "subsample": 0.8,
    "colsample_bytree": 0.86,
    "reg_alpha": 0.57,
    "reg_lambda": 1.78,
    "min_child_weight": 5.48,
    "gamma": 1.77,
}
DEFAULT_BACKTEST_N_ESTIMATORS = 75


class BacktestConfig(BaseModel):
    """Daily walk-forward backtest: retrain every day, predict that day only.

    This simulates production operation. For each game-day in the backtest
    window the model is refitted on every game that finished strictly before
    that day (including earlier backtest days, which become training data once
    played), then predicts that day's games. No result is ever visible to the
    model before it would have been in real life.

    Hyperparameters are held fixed across all days -- re-tuning daily would be
    both prohibitively slow and unrealistic. Note that if those hyperparameters
    came from an Optuna study fitted on data overlapping the backtest window,
    the backtest inherits that selection bias; tune on data preceding the
    window to keep it clean.
    """

    #: ~200 games is one month of NBA regular season; 300 is roughly 6.5 weeks
    #: and ~40 retrains, which buys meaningfully more bets for significance.
    test_games: int = 300
    #: Rolling training window in games. None keeps every prior game
    #: (expanding window), which is closest to how retraining actually runs.
    train_games: int | None = None
    xgb_params: dict[str, Any] | None = None
    n_estimators: int | None = None
    show_progress: bool = True

    @model_validator(mode="after")
    def _validate(self) -> BacktestConfig:
        if self.test_games <= 0:
            raise ValueError("backtest.test_games must be > 0.")
        if self.train_games is not None and self.train_games <= 0:
            raise ValueError("backtest.train_games must be > 0 when provided.")
        if self.n_estimators is not None and self.n_estimators <= 0:
            raise ValueError("backtest.n_estimators must be > 0 when provided.")
        return self

    def resolved_xgb_params(self) -> dict[str, Any]:
        return dict(self.xgb_params or DEFAULT_BACKTEST_XGB_PARAMS)

    def resolved_n_estimators(self) -> int:
        return self.n_estimators or DEFAULT_BACKTEST_N_ESTIMATORS


class BaselineConfig(BaseModel):
    """Controls which column stands in for "the bookmaker's line" baseline.

    Resolution order (see training_pipeline.data.resolve_baseline_line_col):
    1. line_col, if set explicitly here -- e.g. set this to
       "ODDS_total_line_books_median" to use the engineered cross-book
       median instead of a single book's line, once you've verified that
       column is trustworthy in the specific CSV you're using (it was found
       empirically NOT to be points-scale in at least one archived CSV
       snapshot, so it is never used as a silent default).
    2. nba_ou.config.odds_columns.resolve_main_total_line_col(df, book=book)
       -- the same single-book line already used everywhere else in the
       pipeline for cleaning and OU-accuracy scoring.
    """

    line_col: str | None = None
    book: str | None = None


class PlantedSignalConfig(BaseModel):
    """A synthetic feature carrying a KNOWN, small amount of target information.

    Diagnostic only. It exists to answer "can this pipeline recover a weak
    signal it is handed?", which is the question every negative line_error
    result silently depends on. See training_pipeline.diagnostics for the
    construction and the reasoning.

    ``variance_explained`` is a fraction of the RUN'S OWN TARGET variance, so
    0.01 means the feature explains 1% of LINE_ERROR's variance on a line_error
    run and 1% of TOTAL_POINTS' variance on a total_points run. Those are very
    different absolute quantities -- total points is dominated by the line,
    which the model can already read off its features -- so cells are only
    comparable within one strategy.

    Off by default, and every path that could turn a run carrying it into a
    shipped model refuses: refit.train_production_model, run_experiment's
    save_model, and training_pipeline.promote.
    """

    enabled: bool = False
    #: Fraction of target variance the planted feature should explain. 0.0 is a
    #: real control, not a no-op: the feature is still added, as pure
    #: independent noise, so the 0% cell measures the cost of one extra random
    #: column rather than being a different experiment.
    variance_explained: float = 0.0
    #: Seeds ONLY the planted feature, never the model. Separate from
    #: random_state on purpose: the planted column must be identical across
    #: cells that differ only in strength, and must not move when the model seed
    #: is varied to measure fit noise.
    seed: int = 12345
    column: str = "PLANTED_SIGNAL"

    @model_validator(mode="after")
    def _validate(self) -> PlantedSignalConfig:
        if not 0.0 <= self.variance_explained < 1.0:
            raise ValueError(
                "diagnostics.planted_signal.variance_explained must be in "
                "[0, 1). It is a fraction of target variance; 1.0 would mean a "
                "noiseless copy of the target, which tests nothing."
            )
        if not self.column.strip():
            raise ValueError("diagnostics.planted_signal.column must be named.")
        return self


class DiagnosticsConfig(BaseModel):
    """Switches that deliberately corrupt a run to measure the pipeline itself.

    Anything here makes a run diagnostic: informative about the protocol, and
    invalid as evidence about live performance.
    """

    planted_signal: PlantedSignalConfig = Field(default_factory=PlantedSignalConfig)

    @property
    def any_enabled(self) -> bool:
        return self.planted_signal.enabled


class ExperimentConfig(BaseModel):
    experiment_name: str
    #: Manually curated label for the training approach behind this run, e.g.
    #: "2.1" or "v3-style-matchup-features". Set it by hand whenever you change
    #: something you want to be able to group runs by later; it is recorded in
    #: the run metadata, surfaced on the leaderboard, and written into the saved
    #: model bundle's ``training_code_tag``.
    #:
    #: Deliberately excluded from fingerprint(): this is a human label, so
    #: renaming it must not fork a persistent Optuna study. Changes that
    #: genuinely alter what a trial means (data, cleaning, folds, objective)
    #: are already captured by the fingerprint on their own.
    training_version: str | None = None
    #: How the held-out test period is scored. See HoldoutEvaluation.
    holdout_evaluation: HoldoutEvaluation = HoldoutEvaluation.DAILY_WALK_FORWARD

    # --- research log: why this run exists and what it should be read next to.
    # All are labels, so all are excluded from fingerprint().
    #: Runs sharing a comparison_group are meant to be read against each other
    #: (same data, same holdout, one deliberate difference). This is the honest
    #: answer to "are these numbers comparable?" -- the leaderboard can group by
    #: it instead of ranking unrelated cohorts together.
    comparison_group: str | None = None
    #: What this run is supposed to tell you, e.g. "does a 5000-game window beat
    #: 2500 on ROI?". Costs one line now and saves guessing in six months.
    hypothesis: str | None = None
    tags: tuple[str, ...] = ()
    #: What to predict and with which kind of model. This is the field to set;
    #: ``target_family`` is derived from it and kept only so that artifact
    #: paths, the model registry and every run saved before this existed keep
    #: working. Supplying ``target_family`` alone still works and infers the
    #: matching regressor strategy; supplying both requires them to agree.
    prediction_strategy: PredictionStrategy | None = None
    target_family: TargetFamily | None = None
    line_col: str | None = None

    data: DataConfig
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    holdout: HoldoutConfig = Field(default_factory=HoldoutConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    sample_weight: SampleWeightConfig = Field(default_factory=SampleWeightConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    refit: RefitConfig = Field(default_factory=RefitConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    betting: BettingConfig = Field(default_factory=BettingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    #: Deliberate corruptions used to measure the pipeline rather than the
    #: market. Included in fingerprint(): a planted signal changes what every
    #: trial means, so a diagnostic study must never resume a real one.
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    exclude_cols: list[str] = Field(
        default_factory=lambda: ["TOTAL_POINTS", "SEASON_YEAR", "GAME_DATE"]
    )

    window_dir_label: str | None = None
    window_name_label: str | None = None

    save_experiment_artifacts: bool = True
    experiment_root_dir: Path = Path("artifacts/experiments")
    model_output_root: Path = Path("models")
    overwrite_existing_model: bool = False
    #: XGBoost's `device` param: "cpu" or "cuda". A hardware setting, not part
    #: of what a trial means (tree_method stays "hist" either way), so it is
    #: excluded from fingerprint() like the other output/runtime fields below.
    device: str = "cpu"

    #: The seed for the Optuna sampler AND every model fit. Threaded everywhere
    #: rather than hardcoded, which is what makes evaluation_seeds possible.
    random_state: int = 16

    #: Additional seeds to repeat the holdout evaluation under, holding the data,
    #: the split and the tuned hyperparameters fixed so the only thing that
    #: changes is XGBoost's own randomness (subsample / colsample draws).
    #:
    #: This measures the error bar you have otherwise been comparing experiments
    #: without. If two configurations differ by less than the spread across
    #: seeds, the difference between them is not evidence of anything. Empty by
    #: default because each extra seed re-runs the whole evaluation (~35 fits
    #: under daily_walk_forward); two or three extras is the useful range.
    #:
    #: random_state itself is always evaluated and is the reported headline; it
    #: is removed from this list if repeated, and duplicates are dropped.
    evaluation_seeds: tuple[int, ...] = ()

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _resolve_extended_history(self) -> ExperimentConfig:
        """Apply data.extend_history_dropping_season_gated_columns to both halves.

        Resolved here rather than at read time in prepare_dataset, and resolved
        *into the config object*, so that the values which actually ran are what
        --dry-run prints, what config.json records, and what fingerprint()
        hashes. A flag interpreted later would leave the run record saying
        `season_year_floor: 2021` for a run that trained on 2019 onward.

        It also means two configs that reach the same place by different routes
        -- the flag, or spelling out the floor and the exclusions -- share an
        Optuna study, which is correct: they describe the same trials.
        """
        if not self.data.extend_history_dropping_season_gated_columns:
            return self

        floor = self.data.season_year_floor
        # A floor already below the standard one is a deliberate narrowing
        # (e.g. 2020 to take the COVID season without the bubble); leave it.
        if floor is None or floor >= DEFAULT_SEASON_YEAR_FLOOR:
            self.data.season_year_floor = EXTENDED_SEASON_YEAR_FLOOR

        # An explicit spread wins: a caller who set one has already answered
        # this question more precisely than the flag can.
        if self.cleaning.max_seasonal_nan_spread is None:
            self.cleaning.max_seasonal_nan_spread = EXTENDED_SEASON_NAN_SPREAD
        return self

    @model_validator(mode="after")
    def _normalize_evaluation_seeds(self) -> ExperimentConfig:
        """Drop duplicates and the primary seed, preserving order.

        The primary seed is always run as the headline evaluation, so leaving it
        in the extras list would double the work and report the same fit twice
        as if it were independent evidence of stability.
        """
        seen: set[int] = {self.random_state}
        deduped: list[int] = []
        for seed in self.evaluation_seeds:
            if seed not in seen:
                seen.add(seed)
                deduped.append(seed)
        self.evaluation_seeds = tuple(deduped)
        return self

    @model_validator(mode="after")
    def _normalize_training_version(self) -> ExperimentConfig:
        """Treat a blank/whitespace label as unset rather than a real version."""
        if self.training_version is not None:
            stripped = self.training_version.strip()
            self.training_version = stripped or None
        return self

    @model_validator(mode="after")
    def _reconcile_prediction_strategy(self) -> ExperimentConfig:
        """Make ``prediction_strategy`` and ``target_family`` agree.

        Both are accepted so that configs and saved runs written before
        prediction_strategy existed keep loading unchanged, while new work can
        name the strategy explicitly.
        """
        if self.prediction_strategy is None and self.target_family is None:
            raise ValueError(
                "Set prediction_strategy (one of: "
                f"{', '.join(s.value for s in PredictionStrategy)})."
            )

        if self.prediction_strategy is None:
            assert self.target_family is not None
            inferred = _TARGET_FAMILY_STRATEGY.get(self.target_family)
            if inferred is None:
                raise ValueError(
                    f"target_family={self.target_family.value!r} does not identify a "
                    "prediction strategy on its own. Set prediction_strategy instead."
                )
            self.prediction_strategy = inferred
        elif self.target_family is None:
            self.target_family = self.prediction_strategy.target_family
        elif self.target_family != self.prediction_strategy.target_family:
            raise ValueError(
                f"prediction_strategy={self.prediction_strategy.value!r} implies "
                f"target_family={self.prediction_strategy.target_family.value!r}, but "
                f"target_family={self.target_family.value!r} was given. Set one or "
                "make them agree."
            )
        return self

    @model_validator(mode="after")
    def _validate_target_family_constraints(self) -> ExperimentConfig:
        strategy = self.prediction_strategy
        assert strategy is not None  # guaranteed by _reconcile_prediction_strategy

        if strategy == PredictionStrategy.TOTAL_POINTS_REGRESSOR:
            if not self.line_col:
                raise ValueError(
                    "line_col is required for 'total_points_regressor' "
                    "(optuna_total_points.py scores against the betting line)."
                )
        elif strategy in (
            PredictionStrategy.LINE_ERROR_REGRESSOR,
            PredictionStrategy.SPREAD_ERROR_REGRESSOR,
        ):
            if self.line_col:
                raise ValueError(
                    f"line_col must be omitted for {strategy.value!r}: a residual "
                    "regressor predicts the edge directly, so no line is needed "
                    "to convert its output into one."
                )
        else:  # OVER_UNDER_CLASSIFIER
            if not self.line_col:
                raise ValueError(
                    "line_col is required for 'over_under_classifier'. The label "
                    "IS 'did the total beat this line', so which line it refers to "
                    "is part of the target's definition, not a scoring detail."
                )

        # Excluded for EVERY strategy, not just the one that trains on each.
        # All of these are functions of the final score, so any of them reaching
        # X hands over the answer:
        #   TOTAL_POINTS  the outcome itself
        #   LINE_ERROR    TOTAL_POINTS - line; add back the line (which IS a
        #                 feature) and you have the total, and its sign alone is
        #                 the classifier's label exactly
        #   OVER_LABEL    the classifier's label
        # Three CSVs under data/train_data ship a raw LINE_ERROR column, so this
        # is a live hazard rather than a theoretical one. Note that the
        # engineered DIFF_FROM_LINE_*_BEFORE_* rollups are legitimate pre-game
        # features and are deliberately NOT swept up here.
        for leaking in LEAKING_TARGET_COLUMNS:
            if leaking not in self.exclude_cols:
                self.exclude_cols = [*self.exclude_cols, leaking]

        return self

    @model_validator(mode="after")
    def _scale_search_space_to_the_objective(self) -> ExperimentConfig:
        """Give a classifier a search space measured in ITS loss's units.

        The default ranges are reasoned in squared-error terms, where the
        hessian is 1 per sample. Under logistic loss the hessian is ~0.25, so
        the same numbers mean something else entirely: min_child_weight becomes
        80-1000 samples per leaf instead of 20-250, and a gamma floor of 1.0
        exceeds the ~0.49 chance-level split gain, pruning every split. Measured
        on a planted signal, the upper half of the regression space drives a
        classifier to a literally constant prediction.

        Fires ONLY when the space is exactly the regression default -- i.e. the
        space was inherited rather than chosen. Any deliberate customisation,
        including deliberately reusing the regression ranges, is left alone.
        """
        if not self.strategy.is_classifier:
            return self
        if self.optuna.search_space != SearchSpaceConfig():
            return self
        self.optuna.search_space = CLASSIFIER_SEARCH_SPACE.model_copy(deep=True)
        return self

    @model_validator(mode="after")
    def _resolve_n_estimators_range(self) -> ExperimentConfig:
        """Reconcile ``tune_n_estimators`` with ``search_space.n_estimators_range``.

        One switch, one range, and they can never disagree: asking for tuning
        without a range fills in the strategy's default from
        N_ESTIMATORS_RANGES, and stating a range implies tuning. Two fields
        that could contradict each other is how a knob ends up silently
        applied at some fit sites and not others.

        Must run AFTER _scale_search_space_to_the_objective, which only fires
        while the space is byte-identical to the regression default -- writing a
        range into the space first would suppress the classifier rescale
        entirely.
        """
        space = self.optuna.search_space
        if self.optuna.tune_n_estimators and space.n_estimators_range is None:
            assert self.prediction_strategy is not None
            default = N_ESTIMATORS_RANGES.get(self.prediction_strategy)
            if default is None:
                raise ValueError(
                    "optuna.tune_n_estimators is on but no default n_estimators "
                    f"range exists for {self.prediction_strategy.value!r}. State "
                    "optuna.search_space.n_estimators_range explicitly."
                )
            space.n_estimators_range = default.model_copy(deep=True)
        elif space.n_estimators_range is not None:
            self.optuna.tune_n_estimators = True
        return self

    @model_validator(mode="after")
    def _default_objective_to_the_strategy(self) -> ExperimentConfig:
        """Point XGBoost at a loss that matches the model class.

        ``optuna.objective_name`` defaults to a regression loss. Silently
        training a classifier under 'reg:squarederror' would fit and produce
        numbers, just meaningless ones, so switch the default and reject an
        explicit mismatch.
        """
        assert self.prediction_strategy is not None
        is_classifier = self.prediction_strategy.is_classifier
        objective = self.optuna.objective_name

        if is_classifier and objective.startswith("reg:"):
            if objective == OptunaConfig.model_fields["objective_name"].default:
                self.optuna.objective_name = DEFAULT_CLASSIFIER_OBJECTIVE
            else:
                raise ValueError(
                    f"optuna.objective_name={objective!r} is a regression objective "
                    "but prediction_strategy is 'over_under_classifier'. Use "
                    f"{DEFAULT_CLASSIFIER_OBJECTIVE!r}."
                )
        elif not is_classifier and objective.startswith("binary:"):
            raise ValueError(
                f"optuna.objective_name={objective!r} is a classification objective "
                f"but prediction_strategy is {self.prediction_strategy.value!r}."
            )
        return self

    @property
    def strategy(self) -> PredictionStrategy:
        """The resolved strategy, non-optional.

        ``prediction_strategy`` is declared optional only so a config may set
        ``target_family`` instead; validation always fills it in.
        """
        assert self.prediction_strategy is not None
        return self.prediction_strategy

    @property
    def family(self) -> TargetFamily:
        """The resolved target family, non-optional (see :attr:`strategy`)."""
        assert self.target_family is not None
        return self.target_family

    @property
    def is_classifier(self) -> bool:
        return self.strategy.is_classifier

    @property
    def target_col(self) -> str:
        """Column the model is trained against."""
        if self.strategy == PredictionStrategy.LINE_ERROR_REGRESSOR:
            return "LINE_ERROR"
        if self.strategy == PredictionStrategy.SPREAD_ERROR_REGRESSOR:
            return "SPREAD_ERROR"
        if self.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
            return OVER_LABEL_COL
        return "TOTAL_POINTS"

    @property
    def market(self) -> Market:
        """Which betting market this run is about."""
        return self.strategy.market

    @property
    def outcome_col(self) -> str:
        """The realised outcome a bet settles against.

        ``TOTAL_POINTS`` for the totals market, ``HOME_MARGIN`` for the spread.
        Everything that scores profit compares this against a line, so naming it
        once here is what lets one betting layer serve both markets instead of
        two copies of the same arithmetic drifting apart.
        """
        return "HOME_MARGIN" if self.market is Market.SPREAD else "TOTAL_POINTS"

    @model_validator(mode="after")
    def _pooled_snapshots_need_a_game_aware_splitter(self) -> ExperimentConfig:
        """A pooled (game, snapshot) frame may only be split by rolling_origin.

        Under ``rolling_origin`` the fold layout is built here, in
        training_pipeline.splits, which counts distinct games -- so
        ``train_games: 3500`` is 3,500 games however many snapshots each one
        contributes.

        ``test_anchored`` and ``last_n_seasons`` come from nba_ou.modeling and
        describe a fold as a block of N ROWS. On a ten-snapshot frame that
        makes every window a tenth of what it says: ``test_games: 50`` scores 5
        games, ``train_games: 3500`` trains on 350. Nothing errors -- the run
        completes and reports numbers that look ordinary. The archived
        intermediate-line configs handled it by hand-multiplying every knob by
        the snapshot count in a YAML comment, which is a correction no test can
        check and which stops being right the moment the grid changes.

        Refused rather than auto-scaled: multiplying silently would leave the
        config saying one thing and the run doing another, and the whole reason
        this dataset needs care is that rows and games are not the same number.
        Use rolling_origin, or set data.snapshot_minutes to train one model per
        timepoint, where one row IS one game and every splitter is correct
        again.
        """
        if self.data.dataset_type is not DatasetType.INTERMEDIATE_LINE:
            return self
        if self.data.snapshot_minutes is not None:
            return self
        if self.walk_forward.strategy is CVStrategy.ROLLING_ORIGIN:
            return self
        raise ValueError(
            f"walk_forward.strategy={self.walk_forward.strategy.value!r} counts "
            "a fold in ROWS, and a pooled intermediate-line frame holds several "
            "rows per game, so every *_games knob would silently mean a "
            f"fraction of what it says. Use "
            f"{CVStrategy.ROLLING_ORIGIN.value!r}, which counts games, or set "
            "data.snapshot_minutes to train a single-timepoint model (one row "
            "per game, where the two are the same number)."
        )

    @model_validator(mode="after")
    def _diagnostic_runs_must_announce_themselves(self) -> ExperimentConfig:
        """A planted-signal run must be unmistakable, and must not ship a model.

        Two guards, both refusing rather than correcting:

        * the experiment name must start with ``diag_planted``, so the run
          directory, the model-bundle name, the study name, the leaderboard row
          and every log line carry the marker. Auto-prefixing would be friendlier
          and worse -- the name in the YAML would stop matching the name in the
          artifacts, and the one thing this run must never do is look like
          something it is not.
        * ``refit.train_production_model`` must be off. The feature is derived
          from the target, so a model fitted with it would score spectacularly
          and predict nothing. run_experiment repeats this check for its
          ``save_model`` override, and promote.py repeats it again at the point
          a run becomes a shipped bundle.
        """
        if not self.diagnostics.any_enabled:
            return self

        from training_pipeline.diagnostics import DIAGNOSTIC_NAME_PREFIX

        if not self.experiment_name.startswith(DIAGNOSTIC_NAME_PREFIX):
            raise ValueError(
                f"experiment_name={self.experiment_name!r} must start with "
                f"{DIAGNOSTIC_NAME_PREFIX!r} when a diagnostic is enabled. This "
                "run carries a target-derived feature; its artifacts must say so "
                "in their own name."
            )
        if self.refit.train_production_model:
            raise ValueError(
                "refit.train_production_model must be false when a diagnostic is "
                "enabled. The planted feature is derived from the target, so the "
                "resulting model would look excellent and predict nothing."
            )
        return self

    @property
    def is_diagnostic(self) -> bool:
        return self.diagnostics.any_enabled

    @property
    def tunes_n_estimators(self) -> bool:
        """Single authority: is the round count a sampled hyperparameter?

        Everything that needs to know -- the param builder, the objective's
        early-stopping decision, the final-params resolver -- reads this one
        property rather than re-deriving the condition, which is how a filter
        ends up applied at N-1 of N fit sites.
        """
        return self.optuna.search_space.n_estimators_range is not None

    @property
    def uses_fold_early_stopping(self) -> bool:
        """Legacy mode: each CV fold early-stops on its own validation set."""
        return not self.tunes_n_estimators

    @property
    def tunes_train_games(self) -> bool:
        return self.walk_forward.tunes_train_games

    @property
    def pools_objective(self) -> bool:
        return self.optuna.objective_aggregation == ObjectiveAggregation.POOLED

    def resolve_pruner_warmup_steps(self, n_folds: int | None) -> int:
        """Folds a trial must complete before the pruner may kill it.

        Proportional by default because the fixed 5 was chosen against 12 folds
        of ~50 games (~250 games seen). Under rolling_origin a fold is a few
        game-days, so the same 5 would prune on ~35 games -- well inside the
        noise of any metric here.

        Floored at the historical 5 rather than replacing it, so the derived
        value can only ever be MORE patient than before: 12 folds still gives 5
        (every existing config is unchanged), 28 folds gives 7, 50 gives 13.
        """
        if self.optuna.pruner_warmup_steps is not None:
            return max(1, int(self.optuna.pruner_warmup_steps))
        if not n_folds:
            return _LEGACY_PRUNER_WARMUP_STEPS
        proportional = int(round(n_folds * self.optuna.pruner_warmup_fraction))
        return max(_LEGACY_PRUNER_WARMUP_STEPS, proportional)

    @property
    def resolved_window_dir_label(self) -> str:
        if self.window_dir_label:
            return self.window_dir_label
        if self.walk_forward.tunes_train_games:
            # The window is a tuned hyperparameter, so no single value names the
            # run. Saying "tuned_window" is honest; naming one of the choices
            # would label the run after a value it may not have selected.
            return "tuned_window"
        train_games = self.walk_forward.train_games
        return f"{train_games}_games" if train_games else "full_dataset"

    @property
    def resolved_window_name_label(self) -> str:
        if self.window_name_label:
            return self.window_name_label
        return self.resolved_window_dir_label

    @property
    def resolved_study_name(self) -> str:
        if self.optuna.study_name:
            return self.optuna.study_name
        return f"xgb_{self.family.value}_{self.resolved_window_name_label}_mae"

    def _fingerprint_excluded_betting_fields(self) -> bool:
        """Betting settings are post-hoc for every prediction strategy.

        Classifier tuning used to record ROI and use it as a lexicographic
        tie-breaker. It now selects on probability loss and directional accuracy
        only, so changing prices or reporting thresholds must not fork a study.
        """
        return True

    def fingerprint(self) -> str:
        """Short stable hash of everything that changes what a trial *means*.

        Used to key persistent Optuna storage so a study can only ever be
        resumed by a config that would produce comparable trials. Purely
        cosmetic/output-side fields (experiment name, labels, where artifacts
        and models are written, whether artifacts are saved at all) are
        excluded, so renaming an experiment does not throw away its study.
        """
        payload = self.model_dump(
            mode="json",
            exclude={
                "experiment_name": True,
                "training_version": True,
                # Research-log labels: they describe intent, not computation.
                "comparison_group": True,
                "hypothesis": True,
                "tags": True,
                # data_version is a label; expected_checksum IS included via
                # DataConfig, since different bytes mean incomparable trials.
                #
                # extend_history_dropping_season_gated_columns is excluded
                # because it is pure shorthand: _resolve_extended_history has
                # already written its entire effect into season_year_floor and
                # cleaning.max_seasonal_nan_spread, both of which ARE hashed.
                # Including it too would fork the study between the flag and a
                # hand-written config that produces identical data.
                "data": {
                    "data_version",
                    "extend_history_dropping_season_gated_columns",
                },
                "window_dir_label": True,
                "window_name_label": True,
                "save_experiment_artifacts": True,
                "experiment_root_dir": True,
                "model_output_root": True,
                "overwrite_existing_model": True,
                # Hardware choice, not part of what a trial means.
                "device": True,
                # Repeating the evaluation under more seeds measures the result,
                # it does not change what a trial means. random_state is NOT
                # excluded: it does change every fit.
                "evaluation_seeds": True,
                # Post-hoc scoring settings; changing a bet threshold or a
                # backtest window must not invalidate an existing study's trials.
                #
                # Betting outcomes are diagnostics, never model-selection inputs.
                "betting": self._fingerprint_excluded_betting_fields(),
                "backtest": True,
                # n_trials/timeout change how *long* tuning runs, not what a
                # trial means, so resuming across them is legitimate.
                "optuna": {"n_trials", "timeout", "study_name"},
            },
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
