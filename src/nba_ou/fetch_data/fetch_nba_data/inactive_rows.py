"""Inactive players in the box score, moved to the injuries table's layout.

Up to 2025-26 the NBA stats API published each game's inactive list in
``BoxScoreSummaryV3`` (``inactives``), stored in ``nba_injuries``, and the box
score held only dressed players. From 2026-27 that list comes back empty and
the same players are rows of ``BoxScoreTraditionalV3`` / ``BoxScoreAdvancedV3``
with no minutes and a ``COMMENT`` of ``Inactive - <reason>`` (Injury/Illness,
G League - Two-Way, Coach's Decision, Not With Team).

Every reader of ``nba_players`` and ``nba_injuries`` was written for the old
layout, so ingestion splits those rows back out: they go to ``nba_injuries``
and never reach ``nba_players``.
"""

import pandas as pd
from nba_ou.postgre_db.injuries_refs.mapping_v3_v2_injuries_refs import (
    V2_REQUIRED_INJURIES_COLUMNS,
)

INACTIVE_COMMENT_PREFIX = "inactive"

#: First season whose inactive list is in the box score (and empty in the
#: summary). Earlier seasons' inactive list comes only from the summary.
FIRST_BOX_SCORE_INACTIVES_SEASON = 2026

# Box-score columns (after the V3 -> V2 rename) -> injuries columns. The name
# and jersey fields keep their V3 names in the player frame.
_BOX_TO_INJURIES = {
    "PLAYER_ID": "PLAYER_ID",
    "firstName": "FIRST_NAME",
    "familyName": "LAST_NAME",
    "jerseyNum": "JERSEY_NUM",
    "TEAM_ID": "TEAM_ID",
    "TEAM_CITY": "TEAM_CITY",
    "TEAM_NAME": "TEAM_NAME",
    "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
    "GAME_ID": "GAME_ID",
}


def is_inactive_comment(comment: pd.Series) -> pd.Series:
    """True where a box-score comment marks a player on the inactive list."""
    return (
        comment.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.startswith(INACTIVE_COMMENT_PREFIX)
    )


def inactive_rows_to_injuries(
    inactive: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Inactive box-score rows in ``nba_injuries``' upload layout.

    ``games`` supplies ``SEASON_ID`` and ``GAME_DATE`` per ``GAME_ID`` (one row
    per team is fine). A game missing from it is dropped: the upload needs both.
    """
    if inactive.empty:
        return pd.DataFrame(columns=V2_REQUIRED_INJURIES_COLUMNS)

    missing = [col for col in _BOX_TO_INJURIES if col not in inactive.columns]
    if missing:
        raise ValueError(f"Inactive box-score rows lack columns {missing}")

    injuries = inactive[list(_BOX_TO_INJURIES)].rename(columns=_BOX_TO_INJURIES)
    for col in ("PLAYER_ID", "TEAM_ID", "GAME_ID"):
        injuries[col] = injuries[col].astype(str)

    game_info = games[["GAME_ID", "SEASON_ID", "GAME_DATE"]].copy()
    game_info["GAME_ID"] = game_info["GAME_ID"].astype(str)
    game_info = game_info.drop_duplicates(subset=["GAME_ID"])
    injuries = injuries.merge(game_info, on="GAME_ID", how="inner")

    injuries = injuries.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
    return injuries[V2_REQUIRED_INJURIES_COLUMNS].reset_index(drop=True)


def split_inactive_rows(
    players: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a box-score player frame into dressed players and inactives.

    Returns ``(players, injuries)``: the player frame without its inactive rows,
    and those rows in ``nba_injuries``' layout (empty for games in the old
    format, whose box score has none).
    """
    if players is None or players.empty or "COMMENT" not in players.columns:
        return players, pd.DataFrame(columns=V2_REQUIRED_INJURIES_COLUMNS)

    mask = is_inactive_comment(players["COMMENT"])
    injuries = inactive_rows_to_injuries(players.loc[mask], games)
    return players.loc[~mask].reset_index(drop=True), injuries
