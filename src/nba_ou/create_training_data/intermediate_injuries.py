"""Closing-pipeline injury features evaluated at each intermediate snapshot."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_ou.config.market_columns import HOME_MARGIN_COL
from nba_ou.config.odds_columns import (
    get_main_book,
    spread_line_home_col,
    total_line_col,
)
from nba_ou.create_training_data import history_memo
from nba_ou.create_training_data.create_base_game_features import BaseInjuryContext
from nba_ou.create_training_data.select_intermediate_columns import (
    SNAPSHOT_INJURY_EFFECT_PREFIXES,
)
from nba_ou.data_processing.all_star_voting.attach_all_star_voting_features import (
    QUESTIONABLE_FEATURE_COLUMNS,
    PreparedAllStarInputs,
    add_all_star_voting_features,
    all_star_season_year_for_game_date,
    prepare_all_star_inputs,
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
    report_counter_features,
    report_out_overrides,
    report_questionable_sets,
    snapshot_report_states,
)
from nba_ou.data_processing.injury_status.status_history import (
    PreparedBoxHistory,
    add_status_features,
    load_player_box_history,
    prepare_box_and_form,
)
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_fresh_absence_sums,
)
from nba_ou.data_processing.past_injuries.injury_effects import (
    add_top3_availability_effect_features_for_columns,
)
from nba_ou.data_processing.players.attach_player_features import (
    PreparedPlayerHistory,
    add_player_history_features,
    is_player_identifier_column,
    prepare_player_history,
)
from nba_ou.postgre_db.all_star_voting.fetch_data_from_db.fetch_all_star_voting_from_db import (
    load_all_star_voting_from_db,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (
    get_injury_data_from_db,
)
from nba_ou.postgre_db.injury_report_aiven import fetch
from nba_ou.utils.row_cache import RowCache

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

    return snapshot_report_states(
        statuses,
        filings,
        ages,
        cutoffs["snapshot_minutes"],
        events=events,
        listed=listed,
    )


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
    games: pd.DataFrame,
    injured_dict: dict,
    availability_dict: dict,
    *,
    book: str,
    effect_cache: RowCache | None = None,
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
            effect_cache=effect_cache,
        )
        games = mask_uncovered_group_columns(games, prefix)
    return games


def _one_horizon(
    base: pd.DataFrame,
    context: BaseInjuryContext,
    state: InjuryReportState,
    injuries: pd.DataFrame,
    box: PreparedBoxHistory,
    voting: pd.DataFrame,
    snapshot_game_ids: set[str],
    player_rows: RowCache | None = None,
    all_star_rows: RowCache | None = None,
    prepared_players: PreparedPlayerHistory | None = None,
    prepared_all_star: PreparedAllStarInputs | None = None,
    effect_cache: RowCache | None = None,
    target_game_ids: set[str] | None = None,
) -> pd.DataFrame:
    overrides = report_out_overrides(state)
    questionable = report_questionable_sets(state)
    report_listed: dict[str, dict[str, list[str]]] = {}
    for game_id, team_id, player_id in state.statuses[
        ["game_id", "team_id", "player_id"]
    ].itertuples(index=False, name=None):
        if (game_id, team_id) in state.covered:
            report_listed.setdefault(game_id, {}).setdefault(team_id, []).append(
                player_id
            )
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
        row_cache=player_rows,
        prepared_players=prepared_players,
    )
    if target_game_ids is None:
        team = add_injury_report_features(team, state, box)
    else:
        # add_injury_report_features, with its per-player statuses for the
        # targets only. Every row keeps its coverage flag: an uncovered
        # earlier game's players are blanked below, which keeps it out of the
        # availability effects' history, exactly as in a full build.
        positional = team.reset_index(drop=True)
        is_target = positional["GAME_ID"].astype(str).isin(target_game_ids)
        counters = report_counter_features(positional, state)
        target_rows = positional[is_target]
        statuses = add_status_features(
            target_rows.reset_index(drop=True), state, box
        ).set_axis(target_rows.index)
        added = pd.concat([counters, statuses.reindex(positional.index)], axis=1)
        added.index = team.index
        overlap = [c for c in added.columns if c in team.columns]
        team = pd.concat([team.drop(columns=overlap), added], axis=1)
    team = add_all_star_voting_features(
        team,
        context.players,
        voting,
        injured_dict=apply_report_out_overrides(injured_dict, overrides),
        questionable_dict=nested_status_dict(questionable),
        row_cache=all_star_rows,
        prepared=prepared_all_star,
        target_game_ids=target_game_ids,
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
        games,
        injured_dict,
        availability,
        book=get_main_book(),
        effect_cache=effect_cache,
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


#: Memo tables of the tonight-only mode (``add_snapshot_injury_features``).
PLAYER_ROWS_MEMO = "intermediate_injury_player_rows/" + "-".join(PLAYER_STATS)
EFFECTS_MEMO = "intermediate_injury_effects"


def _effects_history_columns(book: str) -> list[str]:
    """The base columns ``_add_availability_effects``' game history reads."""
    return [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "TEAM_ID_TEAM_HOME",
        "TEAM_ID_TEAM_AWAY",
        "TOTAL_POINTS",
        total_line_col(book),
        spread_line_home_col(book),
        HOME_MARGIN_COL,
    ]


def _injury_history_memo(
    targets: set[str],
    *,
    base: pd.DataFrame,
    context: BaseInjuryContext,
    injuries: pd.DataFrame,
    voting: pd.DataFrame,
) -> history_memo.HistoryMemo | None:
    """The memo the per-row caches persist earlier games' rows in.

    Each season's digest chains every input those rows read up to it: the
    team games and players the player step reads, the injuries, the game
    history the effects read, and the All-Star votes (season-less, so they
    enter every season). What varies by horizon is in each row's own key.
    """
    backend = history_memo.default_backend()
    seasons = [history_memo.season_of_game_id(g) for g in targets]
    seasons = [s for s in seasons if s is not None]
    if backend is None or not seasons:
        return None

    def game_id_column(frame):
        return "GAME_ID" if "GAME_ID" in frame.columns else "game_id"

    book = get_main_book()
    history_columns = [c for c in _effects_history_columns(book) if c in base]
    inputs = {
        "team_games": context.team_games,
        "players": context.players,
        "injuries": injuries,
        "effects_history": base[history_columns],
    }
    digests = history_memo.season_input_digests(
        {name: (frame, game_id_column(frame)) for name, frame in inputs.items()},
        range(2000, max(seasons) + 1),
    )
    votes = hashlib.sha256(history_memo._frame_digest(voting)).hexdigest()
    digests = {
        season: hashlib.sha256(f"{digest}|{votes}".encode()).hexdigest()
        for season, digest in digests.items()
    }
    memo = history_memo.HistoryMemo(backend, digests, max(seasons))
    memo.prefetch([PLAYER_ROWS_MEMO, EFFECTS_MEMO])
    return memo


def add_snapshot_injury_features(
    snapshots: pd.DataFrame,
    base: pd.DataFrame,
    context: BaseInjuryContext,
    *,
    progress: str | None = None,
    target_game_ids: set[str] | None = None,
    history_cutoffs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach 2_5 injury features using the latest report before each snapshot.

    ``progress`` labels a tqdm bar over the snapshot horizons; ``None`` runs
    silently.

    ``snapshots`` needs GAME_ID, TIME_TO_MATCH_MIN and SNAPSHOT_TS_UTC. The
    report source uses strict publication time: a report stamped exactly at the
    cutoff is visible only to a later snapshot.

    ``target_game_ids`` is the tonight-only mode: ``snapshots`` holds those
    games' rows, and ``history_cutoffs`` the (GAME_ID, TIME_TO_MATCH_MIN,
    SNAPSHOT_TS_UTC) of every other row a full build would hold. Each horizon
    still runs every team-game, as the availability effects read earlier
    games' players and availability at that horizon, under their own report;
    but their rows come from the memo (``history_memo.PersistentRowCache``), and
    the report and All-Star features are built for the targets only.
    """
    existing_effects = [
        c for c in base if c.startswith(SNAPSHOT_INJURY_EFFECT_PREFIXES)
    ]
    if existing_effects:
        raise ValueError(
            "Base game features already contain availability effects; cannot "
            f"distinguish closing-derived values from snapshot values: {existing_effects[:5]}"
        )
    cutoff_rows = snapshots[KEYS + ["SNAPSHOT_TS_UTC"]]
    if history_cutoffs is not None:
        cutoff_rows = pd.concat(
            [history_cutoffs[KEYS + ["SNAPSHOT_TS_UTC"]], cutoff_rows],
            ignore_index=True,
        )
    cutoffs = cutoff_rows.drop_duplicates(KEYS).rename(
        columns={
            "GAME_ID": "game_id",
            "TIME_TO_MATCH_MIN": "snapshot_minutes",
            "SNAPSHOT_TS_UTC": "as_of",
        }
    )
    states = _states_at_snapshots(cutoffs)
    injuries = get_injury_data_from_db(context.player_context_seasons)
    # Normalised and turned into player form once: the same for every horizon.
    box = prepare_box_and_form(load_player_box_history())
    voting_years = sorted(
        {all_star_season_year_for_game_date(d) for d in context.team_games["GAME_DATE"]}
    )
    voting = load_all_star_voting_from_db(season_years=voting_years)
    if voting is None:
        raise RuntimeError(
            "Could not load All-Star voting for snapshot injury features"
        )

    # Every horizon rebuilds the same team games; only rows whose game sees a
    # different report than at an earlier horizon need rebuilding.
    player_rows, all_star_rows = RowCache(), RowCache()
    # Raw availability effects, shared by the three families and every horizon:
    # the game history they read is the same, only the availability changes.
    effect_cache = RowCache()
    memo = None
    if target_game_ids is not None:
        targets = {str(g) for g in target_game_ids}
        all_star_rows = None  # built for the targets alone
        memo = _injury_history_memo(
            targets, base=base, context=context, injuries=injuries, voting=voting
        )
        if memo is not None:
            # RowCache signatures end in the build's sizes, which differ
            # between builds; the memo's digests cover what they stood for.
            player_rows = history_memo.PersistentRowCache(
                memo,
                PLAYER_ROWS_MEMO,
                season_of=lambda key: history_memo.season_of_game_id(key[0]),
                persist=lambda key: key[0] not in targets,
                size_positions=(3, 4),
            )
            # An effect reads the games before its date: never tonight's.
            effect_cache = history_memo.PersistentRowCache(
                memo,
                EFFECTS_MEMO,
                season_of=lambda key: max(key[1]),
                size_positions=(1,),
            )
    # The same for every horizon: the players' cumulative statistics, sort
    # and lookup indexes, and the All-Star votes, do not depend on the report.
    prepared_players = prepare_player_history(context.players.copy(), PLAYER_STATS)
    prepared_all_star = prepare_all_star_inputs(context.players, voting)
    parts = []
    for horizon, state in tqdm(
        states.items(),
        total=len(states),
        desc=progress,
        unit="snapshot",
        disable=progress is None,
    ):
        ids = set(cutoffs.loc[cutoffs.snapshot_minutes.eq(horizon), "game_id"])
        features = _one_horizon(
            base,
            context,
            state,
            injuries,
            box,
            voting,
            ids,
            player_rows,
            all_star_rows,
            prepared_players,
            prepared_all_star,
            effect_cache,
            target_game_ids=targets if target_game_ids is not None else None,
        )
        features = features.loc[features.GAME_ID.isin(ids)].copy()
        if target_game_ids is not None:
            features = features.loc[features.GAME_ID.astype(str).isin(targets)]
        features["TIME_TO_MATCH_MIN"] = horizon
        parts.append(features)
    if memo is not None:
        memo.flush()
    injury = pd.concat(parts, ignore_index=True)
    overlap = (set(injury.columns) & set(snapshots.columns)) - set(KEYS)
    if overlap:
        raise ValueError(
            f"Snapshot injury columns overlap existing columns: {sorted(overlap)}"
        )
    return snapshots.merge(injury, on=KEYS, how="left", validate="one_to_one")
