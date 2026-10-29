"""Small saved partitions for historical prediction tests; no source builders."""

from types import SimpleNamespace
from unittest.mock import Mock

from nba_ou.config.market_columns import TARGET_ANCHOR_BOOK


def saved_store(frame):
    return SimpleNamespace(
        prefix="saved/intermediate_line/",
        manifests=Mock(
            return_value={
                2026: SimpleNamespace(
                    checksum="sha256:saved", file="season=2026.saved.parquet"
                )
            }
        ),
        read=Mock(return_value=frame),
    )


def quoted_frame(frame):
    frame = frame.copy()
    for prefix, line in [("TOT", 225.0), ("SPR", 3.5), ("ML", float("nan"))]:
        stem = f"ODDS_SNAP_{prefix}_{TARGET_ANCHOR_BOOK.upper()}"
        frame[f"{stem}_HAS_QUOTE"] = 1
        frame[f"{stem}_RAW_LINE"] = line
        frame[f"{stem}_PRICE_LEFT"] = -110
        frame[f"{stem}_PRICE_RIGHT"] = -110
        frame[f"{stem}_LINE_AGE_MINUTES"] = 7
    return frame
