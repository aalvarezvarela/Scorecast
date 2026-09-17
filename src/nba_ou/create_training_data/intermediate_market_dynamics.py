"""Snapshot features on how injury news and the markets evolved up to T.

Three groups (``docs/intermediate_market_dynamics_plan.md`` section 3):

* **G1 injury news** -- ``INJ_SNAP_*_BEFORE_TEAM_{HOME,AWAY}``: recent changes
  in expected missing points, the biggest single-player change, and how long
  ago the last material change was.
* **G2 reaction** -- ``ODDS_SNAP_NEWS_*``: how far each market has moved since
  just before the latest material news, against what that news usually moves.
* **G3 cross-market** -- ``ODDS_SNAP_XMKT_*``: whether total, spread and
  moneyline tell a consistent story.

The fourth group, the spread and moneyline continuation estimates, is fitted
after the leakage gate beside the total one
(``historical_ridge_movement``). None of this re-emits a level of expected
missing minutes or points or a play probability; 2_5 already has those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.data_processing.injury_status.news import (
    NEWS_FEATURE_NAMES,
    InjuryNewsTimeline,
    build_news_timeline,
    estimate_listing_inputs,
    estimation_targets,
    news_column,
)
from nba_ou.data_processing.injury_status.status_history import (
    load_player_box_history,
)
from nba_ou.data_processing.line_history.market_dynamics import (
    LevelReader,
    cross_market_features,
    news_reaction_features,
    tick_levels,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.injury_report_aiven import fetch

REQUIRED_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "GAME_DATE",
    "SEASON_YEAR",
    "TIME_TO_MATCH_MIN",
    "TIPOFF_UTC",
    "SNAPSHOT_TS_UTC",
    "TEAM_ID_TEAM_HOME",
    "TEAM_ID_TEAM_AWAY",
)


def load_injury_news_timeline(
    season_years: list[int], *, verbose: bool = True
) -> InjuryNewsTimeline:
    """Read the report spans and 2_5 estimators needed for the news timeline.

    One season before the first requested season is loaded too, so the opening
    games have a previous game to carry statuses from.
    """
    requested = {int(s) for s in season_years}
    seasons = sorted(requested | {min(requested) - 1})
    with connect_nba_db("aiven") as conn:
        spans = fetch.status_spans(conn, seasons)
        filings = fetch.filing_spans(conn, seasons)
        schedule = fetch.team_game_schedule(conn, seasons)
        # History for P_PLAY: every season, dates strictly before each target.
        status_events = fetch.listed_status_events(conn)
        listed_pairs = fetch.listed_pairs(conn)
        filings_at_tip = fetch.filing_at_tip(conn)
    targets = estimation_targets(spans, schedule)
    estimates = estimate_listing_inputs(
        targets,
        box=load_player_box_history(),
        status_events=status_events,
        listed_pairs=listed_pairs,
        filings_at_tip=filings_at_tip,
    )
    timeline = build_news_timeline(spans, filings, schedule, estimates)
    if verbose:
        print(
            f"✓ Injury news timeline: {len(spans):,} spans, "
            f"{len(timeline.events):,} team changes over "
            f"{timeline.team_games['game_id'].nunique():,} games"
        )
    return timeline


def _team_id(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
    return numeric.astype(str).to_numpy()


def add_intermediate_market_dynamics(
    snapshots: pd.DataFrame,
    *,
    ticks: pd.DataFrame,
    anchor: str,
    timeline: InjuryNewsTimeline | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return ``snapshots`` with the G1, G2 and G3 columns appended.

    ``snapshots`` is the merged (game, snapshot) frame; ``ticks`` the same
    pre-game ticks its snapshot panel was built from.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in snapshots]
    if missing:
        raise KeyError(f"Market dynamics need snapshot columns: {missing}")
    if snapshots.duplicated(["GAME_ID", "TIME_TO_MATCH_MIN"]).any():
        raise ValueError("Market dynamics need one row per game and snapshot")
    if timeline is None:
        seasons = sorted(pd.to_numeric(snapshots["SEASON_YEAR"]).astype(int).unique())
        timeline = load_injury_news_timeline(seasons, verbose=verbose)

    rows = pd.DataFrame(
        {
            "GAME_ID": snapshots["GAME_ID"].astype(str).to_numpy(),
            "GAME_DATE": pd.to_datetime(snapshots["GAME_DATE"]).to_numpy(),
            "SEASON_YEAR": pd.to_numeric(snapshots["SEASON_YEAR"]).to_numpy(),
            "TIME_TO_MATCH_MIN": pd.to_numeric(
                snapshots["TIME_TO_MATCH_MIN"]
            ).to_numpy(),
            "TIPOFF_UTC": pd.to_datetime(snapshots["TIPOFF_UTC"], utc=True).to_numpy(),
            "SNAPSHOT_TS_UTC": pd.to_datetime(
                snapshots["SNAPSHOT_TS_UTC"], utc=True
            ).to_numpy(),
            "HOME_TEAM_ID": _team_id(snapshots["TEAM_ID_TEAM_HOME"]),
            "AWAY_TEAM_ID": _team_id(snapshots["TEAM_ID_TEAM_AWAY"]),
        }
    )
    snapshot_ns = (
        pd.to_datetime(rows["SNAPSHOT_TS_UTC"], utc=True).astype("int64").to_numpy()
    )

    parts = []
    for side, team_column in (("HOME", "HOME_TEAM_ID"), ("AWAY", "AWAY_TEAM_ID")):
        side_features = timeline.features_at(
            rows["GAME_ID"], rows[team_column], snapshot_ns
        )
        parts.append(
            side_features[list(NEWS_FEATURE_NAMES)].rename(
                columns={name: news_column(name, side) for name in NEWS_FEATURE_NAMES}
            )
        )

    reader = LevelReader(tick_levels(ticks))
    parts.append(cross_market_features(reader, rows, anchor=anchor))
    parts.append(
        news_reaction_features(reader, rows, timeline, anchor=anchor, verbose=verbose)
    )
    added = pd.concat(parts, axis=1)
    added.index = snapshots.index

    overlap = sorted(set(added.columns) & set(snapshots.columns))
    if overlap:
        raise ValueError(f"Market dynamics columns overlap existing columns: {overlap}")
    if verbose:
        print(f"✓ Market dynamics: {added.shape[1]} columns (G1 injury news, G2, G3)")
    return pd.concat([snapshots, added], axis=1)
