"""The leakage gate for the intermediate-line dataset.

Fail-closed by construction: a column is dropped unless it is positively
recognised. The closing-line dataset's ``select_training_columns`` cannot be
reused for this, because its central rule -- "``_BEFORE`` means safe" -- is not
true here. A closing line is known when the existing model bets and unknown when
this one does, so several columns that are legitimately ``_BEFORE`` over there
are leakage over here.

Two such traps exist in the shared feature code, both found by measurement
rather than by reading names:

* ``THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE`` / ``_RANGE_BEFORE``
  (``global_market_features.py``) -- dispersion of *this* game's closing lines
  across books.
* ``IMPLIED_PTS_HOME_BEFORE`` / ``IMPLIED_PTS_AWAY_BEFORE``
  (``add_features_after_merging.py``) -- built from this game's closing total
  and spread. These two **sum to the closing line exactly**: measured over 2,626
  games the reconstruction error is 0.0. A model given both can simply add them
  and read off the number it is supposed to be predicting.

The rolled-up cousins are fine and are deliberately kept:
``GLOBAL_CROSSBOOK_TOTAL_STD_AVG_*G_BEFORE`` goes through ``_rolling_game_agg``,
which reads ``values[start - window : start]`` -- strictly earlier dates only.

``audit_closing_line_reconstruction`` exists so this list does not have to stay
correct by vigilance alone; it re-derives the check on real data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from nba_ou.config.market_columns import (
    HOME_MARGIN_COL,
    PTS_AWAY_COL,
    PTS_HOME_COL,
    SPREAD_ERROR_COL,
    TARGET_ANCHOR_BOOK,
)
from nba_ou.config.odds_columns import (
    assert_odds_columns_prefixed,
    spread_line_home_col,
    strip_odds_prefix,
    total_line_col,
)

#: Columns that name themselves ``_BEFORE`` but are computed from the current
#: game's CLOSING prices. Safe in the closing-line dataset, leakage here.
LEAKY_BEFORE_COLUMNS: tuple[str, ...] = (
    "THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE",
    "THIS_GAME_CROSSBOOK_TOTAL_RANGE_BEFORE",
    "IMPLIED_PTS_HOME_BEFORE",
    "IMPLIED_PTS_AWAY_BEFORE",
)

#: Identity, scheduling and grain columns. No market content.
METADATA_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "GAME_DATE",
    "SEASON_ID",
    "SEASON_TYPE",
    "SEASON_YEAR",
    "IS_OVERTIME",
    "TIME_TO_MATCH_MIN",
    # TIPOFF_UTC / SNAPSHOT_TS_UTC deliberately absent: raw timestamps would let
    # a model pin individual games, and nothing downstream needs them here. They
    # live in the scoring sidecar.
    "TEAM_ID_TEAM_HOME",
    "TEAM_ID_TEAM_AWAY",
    "TEAM_ABBREVIATION_TEAM_HOME",
    "TEAM_ABBREVIATION_TEAM_AWAY",
    "TEAM_NAME_TEAM_HOME",
    "TEAM_NAME_TEAM_AWAY",
    "TEAM_CITY_TEAM_HOME",
    "TEAM_CITY_TEAM_AWAY",
    "MATCHUP_TEAM_HOME",
    "MATCHUP_TEAM_AWAY",
    "GAME_NUMBER_TEAM_HOME",
    "GAME_NUMBER_TEAM_AWAY",
)

#: Travel and schedule columns that carry no ``_BEFORE`` marker but are known
#: from the fixture list alone, long before any line exists.
SCHEDULE_COLUMN_PREFIXES: tuple[str, ...] = (
    "TOTAL_KM_IN_LAST_",
    "JETLAG_HOURS_FROM_LAST_GAME_",
    "PLAYOFF_GAMES_LAST_SEASON",
    "TEAM_IS_",
)

#: Snapshot-derived features. Everything under these prefixes is built only from
#: ticks at or before the snapshot horizon.
#: ``INJ_SNAP_`` is injury news up to the snapshot (reports strictly before T);
#: its columns also carry ``_BEFORE``, but the family is recognised explicitly.
SNAPSHOT_COLUMN_PREFIXES: tuple[str, ...] = ("ODDS_SNAP_", "ODDS_LINE_HIST_", "INJ_SNAP_")

# The closing pipeline's availability-effect estimator names these columns
# without a _BEFORE suffix. In the intermediate dataset they are built from
# reports selected strictly before each snapshot and from earlier games only.
SNAPSHOT_INJURY_EFFECT_PREFIXES: tuple[str, ...] = (
    "TOP3_AVAILABILITY_EFFECT_",
    "TOP3_INJURED_AVAILABILITY_EFFECT_",
    "TOP2_QUESTIONABLE_AVAILABILITY_EFFECT_",
)

#: Outcome columns and derived targets that survive the intermediate gate.
#:
#: HOME_MARGIN / PTS_TEAM_HOME / PTS_TEAM_AWAY are outcome facts carried so the
#: spread target can be built and settled; SPREAD_ERROR is the spread residual
#: target, derived per snapshot in create_intermediate_line_df against the
#: Bet365 line as of THAT snapshot.
#:
#: Being here means "survives into the CSV", NOT "may be a feature". Every one of
#: these is blocked from the feature matrix unconditionally by
#: training_pipeline.data.assert_no_leaking_features.
TARGET_COLUMNS: tuple[str, ...] = (
    "TOTAL_POINTS",
    "LINE_ERROR",
    HOME_MARGIN_COL,
    SPREAD_ERROR_COL,
    PTS_HOME_COL,
    PTS_AWAY_COL,
)

#: Never enter the training frame at all. They are split into a separate scoring
#: file by ``create_intermediate_line_df``.
#:
#: They used to be *kept* here behind this prefix, on the assumption that
#: ``feature_columns()`` would filter them downstream. It does not:
#: ``training_pipeline.data.build_feature_matrix`` drops only the configured
#: exclusions, so a ``ODDS_CLOSING_`` column left in the CSV lands in X.
#: Presence in the file is what matters, not presence in a helper's output.
SCORING_ONLY_PREFIXES: tuple[str, ...] = ("ODDS_CLOSING_",)

#: The consensus OPENING line. Known ~25h before tip, so safe at every snapshot
#: on the grid, and the configured ``betting.comparison_line_cols`` baseline.
#: Must survive under its own name despite matching the closing-odds shape.
SAFE_ODDS_COLUMNS: tuple[str, ...] = (
    total_line_col("consensus_opener"),
    # The canonical Bet365 spread AS OF THE SNAPSHOT. It is the spread target's
    # reference line and the price the bet settles into, exactly as
    # ODDS_TOTAL_LINE_bet365 is for totals, so it must survive the gate under its
    # own name. It is a snapshot quote, never a closing one -- the closing spread
    # is carried separately under ODDS_CLOSING_ and excluded from features.
    spread_line_home_col(TARGET_ANCHOR_BOOK),
)


def _is_snapshot_column(column: str) -> bool:
    return column.startswith(SNAPSHOT_COLUMN_PREFIXES)


def _is_schedule_column(column: str) -> bool:
    return column.startswith(SCHEDULE_COLUMN_PREFIXES)


def _is_scoring_only_column(column: str) -> bool:
    return column.startswith(SCORING_ONLY_PREFIXES)


def is_kept_column(column: str) -> bool:
    """Whether ``column`` survives the gate."""
    if column in LEAKY_BEFORE_COLUMNS:
        return False
    if _is_scoring_only_column(column):
        return False
    if column in METADATA_COLUMNS or column in TARGET_COLUMNS:
        return True
    if column in SAFE_ODDS_COLUMNS:
        return True
    if (
        _is_snapshot_column(column)
        or _is_schedule_column(column)
        or column.startswith(SNAPSHOT_INJURY_EFFECT_PREFIXES)
    ):
        return True
    return "_BEFORE" in column


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns a model may train on: kept, minus metadata and targets.

    A reporting convenience only. It is **not** what the training pipeline uses
    to build X, so it must never be the sole thing standing between a leaky
    column and the model -- keep such columns out of the frame instead.
    """
    return [
        column
        for column in df.columns
        if is_kept_column(column)
        and column not in METADATA_COLUMNS
        and column not in TARGET_COLUMNS
    ]


def select_intermediate_training_columns(
    df: pd.DataFrame, *, debug: bool = False
) -> pd.DataFrame:
    """Apply the gate, then verify it actually closed.

    Raises rather than warns: a silently-passed leak produces a model that looks
    excellent and is worthless, which is far more expensive than a failed build.
    """
    kept = [column for column in df.columns if is_kept_column(column)]
    dropped = [column for column in df.columns if column not in set(kept)]

    if debug:
        print(f"Intermediate gate: keeping {len(kept)}, dropping {len(dropped)}")
        for column in dropped[:40]:
            print(f"  - {column}")

    out = df[kept].copy()

    survivors = [column for column in out.columns if column in LEAKY_BEFORE_COLUMNS]
    if survivors:
        raise ValueError(
            "Closing-line-derived columns survived the intermediate gate: "
            f"{survivors}. These carry a _BEFORE name but are computed from "
            "this game's closing prices."
        )

    # Same invariant the closing-line pipeline enforces, checked independently
    # here: this dataset is built from a different set of sources (tick history,
    # snapshot panel) and must not be able to drift out of the convention.
    assert_odds_columns_prefixed(
        out.columns, context="select_intermediate_training_columns"
    )

    return out


def assert_no_bare_closing_odds(df: pd.DataFrame, *, allowed: tuple[str, ...]) -> None:
    """Fail if a current-game closing odds column is present as a feature.

    ``allowed`` names the columns that legitimately hold a *snapshot* value
    despite a closing-style name -- ``ODDS_TOTAL_LINE_<book>`` is the snapshot
    line in this dataset, by design, so the target and the settlement price
    agree.

    Deliberately scans ``df.columns`` rather than ``feature_columns(df)``. Those
    two are not interchangeable here: ``feature_columns`` already excludes
    everything this function looks for, so screening it would make the check a
    no-op that passes no matter what. Independent defence has to look at the
    raw frame.
    """
    offenders = []
    for column in df.columns:
        if "_BEFORE" in column or _is_snapshot_column(column):
            continue
        if column in allowed or column in METADATA_COLUMNS:
            continue
        if column in SAFE_ODDS_COLUMNS:
            continue
        if column in TARGET_COLUMNS:
            continue
        if _is_scoring_only_column(column):
            offenders.append(column)
            continue
        # Shape-match on the name with the unified marker removed. Every
        # odds-derived column now carries ``ODDS_``, so matching the prefixed
        # spellings directly would quietly stop recognising the lowercase raw
        # market columns (``ODDS_total_<book>_price_over``) this check exists
        # to catch.
        bare = strip_odds_prefix(column)
        if bare.startswith(("TOTAL_LINE_", "SPREAD_", "MONEYLINE_")):
            offenders.append(column)
        if bare.startswith(("total_", "spread_", "ml_", "moneyline_")):
            offenders.append(column)
    if offenders:
        raise ValueError(
            f"Bare closing-odds columns present as features: {sorted(set(offenders))}"
        )


def audit_closing_line_reconstruction(
    df: pd.DataFrame,
    closing_line: pd.Series,
    *,
    candidate_threshold: float = 0.5,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Look for feature columns that rebuild the closing line.

    Single columns are checked by correlation; pairs are then checked for an
    exact additive reconstruction, but only among columns that already correlate
    above ``candidate_threshold``. That keeps the pair search to a few hundred
    combinations instead of a million, and it is what catches the
    ``IMPLIED_PTS_*`` case -- neither column alone looks alarming at r ~ 0.8,
    while the two together are the line exactly.

    Returns the findings; an empty frame means nothing was detected.
    """
    line = pd.to_numeric(closing_line, errors="coerce")
    findings: list[dict] = []

    candidates: dict[str, pd.Series] = {}
    for column in feature_columns(df):
        values = pd.to_numeric(df[column], errors="coerce")
        usable = line.notna() & values.notna()
        if usable.sum() < 100 or values[usable].std() == 0:
            continue
        correlation = abs(np.corrcoef(line[usable], values[usable])[0, 1])
        if np.isnan(correlation):
            continue
        if correlation > candidate_threshold:
            candidates[column] = values
        if correlation > 0.999:
            findings.append(
                {"kind": "single", "columns": column, "detail": round(correlation, 6)}
            )

    for left, right in itertools.combinations(sorted(candidates), 2):
        total = candidates[left] + candidates[right]
        usable = line.notna() & total.notna()
        if usable.sum() < 100:
            continue
        if float((total[usable] - line[usable]).abs().max()) <= tolerance:
            findings.append(
                {
                    "kind": "pair_sum",
                    "columns": f"{left} + {right}",
                    "detail": int(usable.sum()),
                }
            )

    return pd.DataFrame(findings, columns=["kind", "columns", "detail"])
