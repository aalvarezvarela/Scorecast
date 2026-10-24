"""Synthetic line-history ticks shared by snapshot and movement tests."""

import pandas as pd


def make_ticks(rows: list[dict]) -> pd.DataFrame:
    """Build a tick frame shaped like ``fetch_pregame_ticks`` output."""
    base = {
        "game_id": "0022300001",
        "season_year": 2023,
        "market": "totals",
        "book": "bet365",
        "is_opener": False,
        "left_price": -110.0,
        "right_price": -110.0,
        "right_line": None,
    }
    records = []
    for index, row in enumerate(rows):
        record = {**base, **row}
        record.setdefault("right_line", record["left_line"])
        record["line_ts"] = pd.Timestamp("2023-11-01", tz="UTC") + pd.Timedelta(
            minutes=index
        )
        records.append(record)
    return pd.DataFrame(records)
