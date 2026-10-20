"""What a model fed a later snapshot than its own may read of it.

An intermediate model trained at T-H can be served the latest snapshot of
tonight instead of its own (``LATEST_INPUT_SLOTS``). Two families then carry
information its training rows never had, and the model must not see it:

* **Referees.** Training masks every ``REF_`` column before 09:00 ET on game
  day (``mask_referees_before_release``). The same mask is applied here at the
  model's nominal snapshot, ``tip - H``: a T-720 model of a 19:00 ET game reads
  no crew, of a 22:00 ET game it does -- as its training rows did.
* **Injury news.** The news windows (``INJ_SNAP_*``, ``ODDS_SNAP_NEWS_*``) are
  nearly empty overnight, when no report is published. Where fewer than
  ``SUPPORT_THRESHOLD`` of a column's training rows at H carry a value
  (``input_support_table``), a value is set to 0, the no-news reading; NaN
  stays NaN, as training also has it. The injury status itself (the last
  report) was present at every horizon and is not limited.

These limits apply to the latest input only. A model's own snapshot is already
what its training rows were.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from nba_ou.create_training_data.intermediate_referees import (
    mask_referees_before_release,
)
from nba_ou.prediction.input_support_table import UNSUPPORTED_NEWS_COLUMNS

#: Whether an intermediate row is its game's latest snapshot.
LATEST_INPUT_COLUMN = "LATEST_INPUT"
REFEREE_PREFIX = "REF_"
#: The news families whose support is measured per horizon.
NEWS_PREFIXES = ("INJ_SNAP_", "ODDS_SNAP_NEWS_")
#: A column is unsupported at a horizon where fewer of its rows carry a value.
SUPPORT_THRESHOLD = 0.05
#: Per row, how many feature cells the limits changed.
LIMITED_COUNT_COLUMN = "INPUT_LIMITED_COUNT"


def _carries_value(values: pd.DataFrame) -> pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    return numeric.notna() & numeric.ne(0)


def unsupported_news_columns(
    frame: pd.DataFrame,
    *,
    horizon_col: str = "TIME_TO_MATCH_MIN",
    threshold: float = SUPPORT_THRESHOLD,
    prefixes: tuple[str, ...] = NEWS_PREFIXES,
) -> dict[str, tuple[int, ...]]:
    """Per news column, the horizons where under ``threshold`` of rows carry a value.

    ``frame`` is an intermediate training dataset; a value is anything but NaN
    and 0. Columns supported at every horizon are left out.
    """
    columns = [c for c in frame.columns if c.startswith(prefixes)]
    if not columns:
        return {}
    share = _carries_value(frame[columns]).groupby(frame[horizon_col]).mean()
    table = {}
    for column in columns:
        horizons = share.index[share[column] < threshold]
        if len(horizons):
            table[column] = tuple(int(h) for h in sorted(horizons))
    return table


def limit_to_training_support(
    rows: pd.DataFrame,
    *,
    horizon_minutes: int,
    unsupported: Mapping[str, tuple[int, ...]] = UNSUPPORTED_NEWS_COLUMNS,
) -> pd.DataFrame:
    """``rows`` as a model trained at ``horizon_minutes`` could have seen them.

    ``rows`` are intermediate rows with ``TIME_TO_MATCH_MIN`` and
    ``SNAPSHOT_TS_UTC``. Adds ``LIMITED_COUNT_COLUMN``.
    """
    limited = rows.copy()
    changed = pd.Series(0, index=rows.index, dtype="int64")

    referee_columns = [c for c in rows.columns if c.startswith(REFEREE_PREFIX)]
    if referee_columns:
        snapshot = pd.to_datetime(rows["SNAPSHOT_TS_UTC"], utc=True)
        tipoff = snapshot + pd.to_timedelta(
            pd.to_numeric(rows["TIME_TO_MATCH_MIN"]), unit="m"
        )
        nominal = tipoff - pd.Timedelta(minutes=int(horizon_minutes))
        masked = mask_referees_before_release(
            rows[referee_columns].assign(TIPOFF_UTC=tipoff, SNAPSHOT_TS_UTC=nominal)
        )[referee_columns]
        changed += (rows[referee_columns].notna() & masked.isna()).sum(axis=1)
        limited[referee_columns] = masked

    news_columns = [
        column
        for column, horizons in unsupported.items()
        if column in rows.columns and int(horizon_minutes) in horizons
    ]
    if news_columns:
        carries = _carries_value(rows[news_columns])
        changed += carries.sum(axis=1)
        limited[news_columns] = rows[news_columns].mask(carries, 0.0)

    limited[LIMITED_COUNT_COLUMN] = changed.astype(np.int64)
    return limited
