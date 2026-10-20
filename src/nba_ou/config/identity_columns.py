"""Columns that identify or describe a game rather than measure it.

Exact names, not name patterns. Readers used to treat every column whose name
merely *contained* "ID" as an identifier and read it as text, which also caught
``ODDS_..._line_mid_...``, ``..._REACTION_RESIDUAL``, ``..._RIDGE_...``,
``..._SIDE_MOVE_...`` and ``IS_US_HOLIDAY_BEFORE``. Cleaning then dropped them
as text columns: seven unique market-dynamics features never reached an
intermediate-line model. A list can only be wrong about the columns it names.

Two kinds of non-feature column, kept apart because they are read differently:

* identifiers are read as text, so ``GAME_ID`` keeps its leading zeros -- the
  pipeline resolves the season type from its 3-character prefix;
* descriptive text is text already and needs no dtype.

Cleaning drops both from every frame (see
nba_ou.data_processing.missing_data.clean_df_for_training), except where a
caller protects one through ``keep_columns``. A text column that is in neither
list is not dropped quietly: it is reported, because it is either a new
identifier to add here or a feature that arrived with the wrong dtype.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

#: Read as text by every loader. ``GAME_ID`` is the one that needs it -- read
#: as a number it loses its leading zeros. The others are text for consistency
#: with every dataset built so far.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "SEASON_ID",
    "TEAM_ID_TEAM_HOME",
    "TEAM_ID_TEAM_AWAY",
)

#: Descriptive text carried by the training datasets. ``SEASON_TYPE`` is here
#: rather than filtered on directly because it labels Play-In games as
#: "Playoffs"; the season type is resolved from the GAME_ID prefix instead.
TEXT_METADATA_COLUMNS: tuple[str, ...] = (
    "TEAM_CITY_TEAM_HOME",
    "TEAM_ABBREVIATION_TEAM_HOME",
    "TEAM_NAME_TEAM_HOME",
    "MATCHUP_TEAM_HOME",
    "TEAM_CITY_TEAM_AWAY",
    "TEAM_ABBREVIATION_TEAM_AWAY",
    "TEAM_NAME_TEAM_AWAY",
    "MATCHUP_TEAM_AWAY",
    "SEASON_TYPE",
)

#: Never model inputs. Cleaning drops whichever of these are present.
NON_FEATURE_COLUMNS: tuple[str, ...] = IDENTIFIER_COLUMNS + TEXT_METADATA_COLUMNS


def identifier_dtypes(columns: Iterable[str]) -> dict[str, type]:
    """``read_csv`` dtype mapping for the identifier columns among ``columns``."""
    present = set(columns)
    return {column: str for column in IDENTIFIER_COLUMNS if column in present}


def apply_identifier_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Identifier columns as text, missing values as NaN -- as read_csv gives them.

    For frames that did not come through ``read_csv(dtype=...)``: a Parquet file
    written straight from a builder's frame stores ``TEAM_ID`` as an integer.
    Converting here keeps a frame's dtypes independent of the file format it
    was read from. Already-text columns are left as they are.
    """
    out = df
    for column in IDENTIFIER_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column]
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            continue
        if out is df:
            # Shallow: assigning a column replaces it in the copy only, so a
            # multi-GB frame is not duplicated to retype four columns.
            out = df.copy(deep=False)
        # Through Int64 so a float column holding NaN reads 1610612737, not
        # 1610612737.0.
        as_int = (
            values.astype("Int64") if pd.api.types.is_float_dtype(values) else values
        )
        out[column] = as_int.astype(str).where(values.notna(), np.nan)
    return out
