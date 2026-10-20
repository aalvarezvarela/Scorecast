"""Select the shared Yahoo feature schema after the final ODDS_ prefix pass."""

import numpy as np
import pandas as pd
from nba_ou.config.yahoo_features import (
    YAHOO_FEATURE_COLUMNS,
    YAHOO_HISTORY_COLUMNS,
    is_yahoo_percentage_column,
)


def select_yahoo_features(
    df: pd.DataFrame, *, include_raw: bool = True
) -> pd.DataFrame:
    """Keep 36 closing / 24 intermediate columns, preserving observed values.

    Missing inputs stay NaN, including a completely absent Yahoo feed. Never
    replace raw observations with team history or opposite-side percentages.
    """
    keep = YAHOO_FEATURE_COLUMNS if include_raw else YAHOO_HISTORY_COLUMNS
    drop = [c for c in df.columns if is_yahoo_percentage_column(c) and c not in keep]
    result = df.drop(columns=drop)
    missing = {c: np.nan for c in keep if c not in result.columns}
    if missing:
        result = result.assign(**missing)
    return result
