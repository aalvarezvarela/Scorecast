"""Closing-pipeline injury features evaluated at each intermediate snapshot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.config.odds_columns import get_main_book
from nba_ou.create_training_data.create_base_game_features import BaseInjuryContext
from nba_ou.create_training_data.select_intermediate_columns import (
    SNAPSHOT_INJURY_EFFECT_PREFIXES,
)
from nba_ou.data_processing.all_star_voting.attach_all_star_voting_features import (
    QUESTIONABLE_FEATURE_COLUMNS,
    add_all_star_voting_features,
    all_star_season_year_for_game_date,
)
from nba_ou.data_processing.injury_status.features import (
    add_injury_report_features,
    mask_uncovered_group_columns,
)
from nba_ou.data_processing.injury_status.report_state import (
    COVERED_COL,
    InjuryReportState,
    apply_report_out_overrides,
    nested_status_dict,
    report_out_overrides,
    report_questionable_sets,
)
from nba_ou.data_processing.injury_status.status_history import load_player_box_history
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_fresh_absence_sums,
)
from nba_ou.data_processing.past_injuries.injury_effects import (
    add_top3_availability_effect_features_for_columns,
)
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
    is_player_identifier_column,
)
from nba_ou.postgre_db.all_star_voting.fetch_data_from_db.fetch_all_star_voting_from_db import (
    load_all_star_voting_from_db,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (
    get_injury_data_from_db,
)
from nba_ou.postgre_db.injury_report_aiven import fetch

KEYS = ["GAME_ID", "TIME_TO_MATCH_MIN"]
PLAYER_STATS = ["PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT", "MIN"]
INJURED_ALL_STAR_COLUMNS = (
    "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE",
    "ALL_STAR_MIN_INJURED_SCORE_BEFORE",
    *QUESTIONABLE_FEATURE_COLUMNS,
)


def _states_at_snapshots(cutoffs: pd.DataFrame) -> dict[int, InjuryReportState]:
    """Bulk-read report spans at the exact timestamps used by market snapshots."""
    with connect_nba_db("aiven") as conn:
        statuses, filings, ages = fetch.report_state_at_snapshots(conn, cutoffs)
        # These two frames serve history estimators. They use earlier game dates
        # only, so the current game's later report cannot enter its own features.
        events = fetch.listed_status_events(conn)
        listed = fetch.listed_pairs(conn)

    states = {}
    for horizon in sorted(cutoffs["snapshot_minutes"].unique()):

        def at_horizon(
            frame: pd.DataFrame, selected_horizon: int = int(horizon)
        ) -> pd.DataFrame:
            return (
                frame.loc[frame["snapshot_minutes"].eq(selected_horizon)]
                .drop(columns="snapshot_minutes")
                .reset_index(drop=True)
            )

        states[int(horizon)] = InjuryReportState(
            statuses=at_horizon(statuses),
            filings=at_horizon(filings),
            report_age=at_horizon(ages),
            status_events=events,
            listed_pairs=listed,
        )
    return states


def _wide_team_injury_features(
    team: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Pair the home and away team rows without rerunning the full game merge."""
    parts = []
    for home, side in ((True, "HOME"), (False, "AWAY")):
        part = team.loc[team["HOME"].eq(home), ["GAME_ID", *feature_columns]].copy()
        part = part.rename(columns={c: f"{c}_TEAM_{side}" for c in feature_columns})
        parts.append(part)
    return parts[0].merge(parts[1], on="GAME_ID", validate="one_to_one")


def _add_availability_effects(
    games: pd.DataFrame, injured_dict: dict, availability_dict: dict, *, book: str
) -> pd.DataFrame:
    families = (
        (
            "TOP3_AVAILABILITY_EFFECT",
            (
                "TOP1_PLAYER_ID_PTS",
                "TOP2_PLAYER_ID_PTS",
                "TOP3_PLAYER_ID_PTS",
                "TOP1_PLAYER_ID_MIN",
            ),
        ),
        (
            "TOP3_INJURED_AVAILABILITY_EFFECT",
            (
                "TOP1_INJURED_PLAYER_ID_PTS",
                "TOP2_INJURED_PLAYER_ID_PTS",
                "TOP3_INJURED_PLAYER_ID_PTS",
                "TOP1_INJURED_PLAYER_ID_MIN",
            ),
        ),
        (
            "TOP2_QUESTIONABLE_AVAILABILITY_EFFECT",
            (
                "TOP1_QUESTIONABLE_PLAYER_ID_PTS",
                "TOP2_QUESTIONABLE_PLAYER_ID_PTS",
                "TOP1_QUESTIONABLE_PLAYER_ID_MIN",
            ),
        ),
    )
    for prefix, names in families:
        games = add_top3_availability_effect_features_for_columns(
            games,
            injured_dict,
            availability_dict=availability_dict,
            total_line_book=book,
            spread_line_book=book,
            home_player_cols=tuple(f"{name}_BEFORE_TEAM_HOME" for name in names),
            away_player_cols=tuple(f"{name}_BEFORE_TEAM_AWAY" for name in names),
            out_prefix=prefix,
            shrinkage_k=10.0,
            include_per_player_columns=False,
            include_detailed_sample_size_features=False,
        )
        games = mask_uncovered_group_columns(games, prefix)
    return games


def _one_horizon(
    base: pd.DataFrame,
    context: BaseInjuryContext,
    state: InjuryReportState,
    injuries: pd.DataFrame,
    box: pd.DataFrame,
    voting: pd.DataFrame,
    snapshot_game_ids: set[str],
) -> pd.DataFrame:
    overrides = report_out_overrides(state)
    questionable = report_questionable_sets(state)
    report_listed: dict[str, dict[str, list[str]]] = {}
    for game_id, team_id, player_id in state.statuses[
        ["game_id", "team_id", "player_id"]
    ].itertuples(index=False, name=None):
        if (game_id, team_id) in state.covered:
            report_listed.setdefault(game_id, {}).setdefault(team_id, []).append(player_id)
    original_columns = set(context.team_games.columns)
    team, injured_dict, availability = add_player_history_features(
        context.team_games.copy(),
        context.players.copy(),
        injuries,
        PLAYER_STATS,
        return_availability_dict=True,
        report_out_overrides=overrides,
        report_questionable_sets=questionable,
        include_available_roster_count=True,
        snapshot_game_ids=snapshot_game_ids,
        snapshot_report_listed_players=report_listed,
    )
    team = add_injury_report_features(team, state, box)
    team = add_all_star_voting_features(
        team,
        context.players,
        voting,
        injured_dict=apply_report_out_overrides(injured_dict, overrides),
        questionable_dict=nested_status_dict(questionable),
    )
    # When no team has filed at this horizon, the shared All-Star builder has
    # no questionable group to emit. Keep the snapshot schema stable anyway.
    for column in INJURED_ALL_STAR_COLUMNS:
        if column not in team:
            team[column] = np.nan
    feature_columns = [
        c
        for c in team
        if c not in original_columns
        and "_BEFORE" in c
        and not c.startswith("ALL_STAR_")
    ]
    feature_columns.extend(
        c for c in INJURED_ALL_STAR_COLUMNS if c not in feature_columns
    )
    wide = _wide_team_injury_features(team, feature_columns)
    games = base.merge(wide, on="GAME_ID", how="left", validate="one_to_one")

    # Every feature depending on the target game's roster is unknown for an
    # unfiled team, even though the closing builder can fall back to a settled
    # inactive list. The coverage flag itself remains 0.
    for side in ("HOME", "AWAY"):
        flag = f"{COVERED_COL}_TEAM_{side}"
        unavailable = games[flag].fillna(0).eq(0)
        columns = [
            f"{name}_TEAM_{side}"
            for name in feature_columns
            if name != COVERED_COL and f"{name}_TEAM_{side}" in games
        ]
        games.loc[unavailable, columns] = np.nan

    games = add_fresh_absence_sums(games)
    games = _add_availability_effects(
        games, injured_dict, availability, book=get_main_book()
    )

    # The closing merge derives these from player columns after the team merge.
    for side in ("HOME", "AWAY"):
        top = f"TOP1_PLAYER_PTS_BEFORE_TEAM_{side}"
        season = f"PTS_SEASON_BEFORE_AVG_TEAM_{side}"
        if top in games and season in games:
            games[f"STAR_PTS_PERCENTAGE_BEFORE_TEAM_{side}"] = pd.to_numeric(
                games[top], errors="coerce"
            ) / pd.to_numeric(games[season], errors="coerce").replace(0, np.nan)
        injured_pts = f"TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_{side}"
        if injured_pts in games and season in games:
            games[f"INJURY_PTS_SHARE_{side}_BEFORE"] = pd.to_numeric(
                games[injured_pts], errors="coerce"
            ) / pd.to_numeric(games[season], errors="coerce").replace(0, np.nan)

    generated = [
        c
        for c in games
        if c not in base.columns
        and ("_BEFORE" in c or c.startswith(SNAPSHOT_INJURY_EFFECT_PREFIXES))
    ]
    generated = [c for c in generated if not is_player_identifier_column(c)]
    return games[["GAME_ID", *generated]]


def add_snapshot_injury_features(
    snapshots: pd.DataFrame,
    base: pd.DataFrame,
    context: BaseInjuryContext,
) -> pd.DataFrame:
    """Attach 2_5 injury features using the latest report before each snapshot.

    ``snapshots`` needs GAME_ID, TIME_TO_MATCH_MIN and SNAPSHOT_TS_UTC. The
    report source uses strict publication time: a report stamped exactly at the
    cutoff is visible only to a later snapshot.
    """
    existing_effects = [
        c for c in base if c.startswith(SNAPSHOT_INJURY_EFFECT_PREFIXES)
    ]
    if existing_effects:
        raise ValueError(
            "Base game features already contain availability effects; cannot "
            f"distinguish closing-derived values from snapshot values: {existing_effects[:5]}"
        )
    cutoffs = (
        snapshots[KEYS + ["SNAPSHOT_TS_UTC"]]
        .drop_duplicates(KEYS)
        .rename(
            columns={
                "GAME_ID": "game_id",
                "TIME_TO_MATCH_MIN": "snapshot_minutes",
                "SNAPSHOT_TS_UTC": "as_of",
            }
        )
    )
    states = _states_at_snapshots(cutoffs)
    injuries = get_injury_data_from_db(context.player_context_seasons)
    box = load_player_box_history()
    voting_years = sorted(
        {all_star_season_year_for_game_date(d) for d in context.team_games["GAME_DATE"]}
    )
    voting = load_all_star_voting_from_db(season_years=voting_years)
    if voting is None:
        raise RuntimeError(
            "Could not load All-Star voting for snapshot injury features"
        )

    parts = []
    for horizon, state in states.items():
        ids = set(cutoffs.loc[cutoffs.snapshot_minutes.eq(horizon), "game_id"])
        features = _one_horizon(base, context, state, injuries, box, voting, ids)
        features = features.loc[features.GAME_ID.isin(ids)].copy()
        features["TIME_TO_MATCH_MIN"] = horizon
        parts.append(features)
    injury = pd.concat(parts, ignore_index=True)
    overlap = (set(injury.columns) & set(snapshots.columns)) - set(KEYS)
    if overlap:
        raise ValueError(
            f"Snapshot injury columns overlap existing columns: {sorted(overlap)}"
        )
    return snapshots.merge(injury, on=KEYS, how="left", validate="one_to_one")
