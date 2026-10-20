"""Phase C (transform): resolved observations -> validity spans.

One row per (game, player) *state*, not per (report, player). A player listed
``Out`` across forty consecutive reports is one span, not forty rows.

The clamp at tipoff is the part that matters. Without it the final span before a
game carries an open end, and an as-of read at any later instant reports a status
that was never resolved -- the "still Doubtful three hours after the final
buzzer" failure. Clamping makes the honest answer (nothing) structural.
"""

from __future__ import annotations

import pandas as pd

#: Columns a span carries into the loader.
SPAN_COLUMNS = [
    "game_id",
    "player_id",
    "team_id",
    "season_year",
    "valid_from",
    "valid_to",
    "status_id",
    "reason_id",
    "first_report_id",
    "mins_to_tip",
    "is_pregame",
]

FILING_COLUMNS = [
    "game_id",
    "team_id",
    "season_year",
    "valid_from",
    "valid_to",
    "submitted",
]


def build_spans(observations: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-report observations into validity spans.

    ``observations`` needs one row per (game, player, observed_at) with
    ``status_id``, ``reason_id``, ``report_id``, ``tipoff_utc`` and the keys in
    :data:`SPAN_COLUMNS`. It must cover a *contiguous* run of reports for each
    game -- a partial reload would otherwise merge across a gap and invent a span
    that spans reports nobody looked at.
    """
    if observations.empty:
        return pd.DataFrame(columns=SPAN_COLUMNS)

    frame = observations.sort_values(["game_id", "player_id", "observed_at"]).copy()
    keys = ["game_id", "player_id"]

    # A new span starts wherever the (status, reason) pair differs from the
    # previous observation of the same player for the same game. Compared as one
    # string key because a removed player carries a NULL status, and NaN != NaN
    # would otherwise open a fresh span at every report he stays off the list.
    state = (
        frame["status_id"].astype("Int64").astype(str)
        + "|"
        + frame["reason_id"].astype(str)
    )
    changed = state.groupby([frame["game_id"], frame["player_id"]]).shift() != state
    first = ~frame.duplicated(subset=keys)
    frame["span_no"] = (
        (changed | first).groupby([frame["game_id"], frame["player_id"]]).cumsum()
    )

    grouped = frame.groupby(keys + ["span_no"], sort=False)
    spans = grouped.agg(
        team_id=("team_id", "first"),
        season_year=("season_year", "first"),
        status_id=("status_id", "first"),
        reason_id=("reason_id", "first"),
        first_report_id=("report_id", "first"),
        valid_from=("observed_at", "first"),
        tipoff_utc=("tipoff_utc", "first"),
        last_seen=("observed_at", "last"),
    ).reset_index()

    # The span ends where the next one begins; the final span of a game ends at
    # tipoff. Never at last_seen -- the status held until something replaced it.
    spans = spans.sort_values(keys + ["span_no"])
    next_start = spans.groupby(keys, sort=False)["valid_from"].shift(-1)
    spans["valid_to"] = next_start.fillna(spans["tipoff_utc"])

    # Reports published after tip cannot open a span, and nothing may outlive
    # its game.
    spans = spans.loc[spans["valid_from"] < spans["tipoff_utc"]].copy()
    spans["valid_to"] = spans[["valid_to", "tipoff_utc"]].min(axis=1)

    delta = spans["tipoff_utc"] - spans["valid_from"]
    spans["mins_to_tip"] = (delta.dt.total_seconds() // 60).astype("int64")
    spans["is_pregame"] = spans["mins_to_tip"] > 0
    # NULL means "removed from the report"; keep it NULL rather than NaN-float.
    spans["status_id"] = spans["status_id"].astype("Int64")
    return spans[SPAN_COLUMNS].reset_index(drop=True)


def build_filing_spans(observations: pd.DataFrame) -> pd.DataFrame:
    """Same treatment for "has this team filed yet".

    ``NOT YET SUBMITTED`` means *unknown*, not "nobody is injured", so it needs
    its own timeline rather than being inferred from the absence of player rows.
    """
    if observations.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)

    frame = observations.sort_values(["game_id", "team_id", "observed_at"]).copy()
    keys = ["game_id", "team_id"]
    changed = frame.groupby(keys, sort=False)["submitted"].shift() != frame["submitted"]
    first = ~frame.duplicated(subset=keys)
    frame["span_no"] = (
        (changed | first).groupby([frame["game_id"], frame["team_id"]]).cumsum()
    )

    spans = (
        frame.groupby(keys + ["span_no"], sort=False)
        .agg(
            season_year=("season_year", "first"),
            submitted=("submitted", "first"),
            valid_from=("observed_at", "first"),
            tipoff_utc=("tipoff_utc", "first"),
        )
        .reset_index()
    )

    spans = spans.sort_values(keys + ["span_no"])
    next_start = spans.groupby(keys, sort=False)["valid_from"].shift(-1)
    spans["valid_to"] = next_start.fillna(spans["tipoff_utc"])
    spans = spans.loc[spans["valid_from"] < spans["tipoff_utc"]].copy()
    spans["valid_to"] = spans[["valid_to", "tipoff_utc"]].min(axis=1)
    return spans[FILING_COLUMNS].reset_index(drop=True)
