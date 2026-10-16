"""Closing-pipeline injury features evaluated at each intermediate snapshot."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

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
    _build_availability_games_index,
    add_top3_availability_effect_features_for_columns,
)
from nba_ou.data_processing.past_injuries.past_injuries import (
    get_injured_players_dict,
)
from nba_ou.data_processing.players.attach_player_features import (
    PreparedPlayerHistory,
    add_player_history_features,
    is_player_identifier_column,
    prepare_player_history,
)
from nba_ou.data_processing.players.live_roster import (
    game_day_rosters,
    previous_game_rosters,
    report_membership,
    supplement_rosters,
)
from nba_ou.data_processing.players.roster_continuity import (
    add_roster_continuity_feature,
)
from nba_ou.fetch_data.live_rosters import roster_overrides
from nba_ou.postgre_db.all_star_voting.fetch_data_from_db.fetch_all_star_voting_from_db import (
    load_all_star_voting_from_db,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (
    get_injury_data_from_db,
)
from nba_ou.postgre_db.injury_report_aiven import fetch, live
from nba_ou.postgre_db.injury_report_aiven.live import LiveInjuryReports
from nba_ou.postgre_db.row_order import canonical_row_order
from nba_ou.utils.row_cache import RowCache

KEYS = ["GAME_ID", "TIME_TO_MATCH_MIN"]
PLAYER_STATS = ["PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT", "MIN"]
INJURED_ALL_STAR_COLUMNS = (
    "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE",
    "ALL_STAR_MIN_INJURED_SCORE_BEFORE",
    *QUESTIONABLE_FEATURE_COLUMNS,
)


#: Memo table of the earlier games' report states at their snapshots.
REPORT_STATES_MEMO = "intermediate_report_states"


def _stored_report_states(
    conn, cutoffs: pd.DataFrame, targets: set[str]
) -> tuple[list[tuple], pd.DataFrame]:
    """The earlier games' report reads, season by season from the memo.

    A season's entry is keyed on its cutoffs and on the chained fingerprint of
    every stored report row up to it (``fetch.report_season_fingerprints``);
    a season whose key does not match is read and stored. Returns the parts
    read so far and the cutoffs still to read (tonight's).
    """
    backend = history_memo.default_backend()
    games = cutoffs["game_id"].astype(str)
    later = cutoffs[games.isin(targets)]
    earlier = cutoffs[~games.isin(targets)]
    if backend is None or earlier.empty:
        return [], cutoffs
    seasons = pd.Series(
        history_memo.season_of_game_ids(earlier["game_id"]), index=earlier.index
    )
    fingerprints = fetch.report_season_fingerprints(conn)
    digests = {}
    for season in sorted(seasons.unique()):
        known = [s for s in fingerprints if s <= season]
        chain = fingerprints[max(known)] if known else "none"
        rows = earlier[seasons.eq(season)].sort_values(["game_id", "snapshot_minutes"])
        digest = hashlib.sha256(chain.encode())
        digest.update(history_memo._frame_digest(rows.reset_index(drop=True)))
        digests[int(season)] = digest.hexdigest()
    memo = history_memo.HistoryMemo(backend, digests, max(digests))
    memo.prefetch([REPORT_STATES_MEMO])
    parts = []
    for season in sorted(digests):
        table = memo.table(REPORT_STATES_MEMO, season)
        stored = table.get("value") if table is not None else None
        if stored is None:
            stored = fetch.report_state_at_snapshots(conn, earlier[seasons.eq(season)])
            if table is not None:
                table["value"] = stored
        parts.append(stored)
    memo.flush()
    return parts, later


def _states_at_snapshots(
    cutoffs: pd.DataFrame,
    live_reports: LiveInjuryReports | None = None,
    target_game_ids: set[str] | None = None,
) -> dict[int, InjuryReportState]:
    """Bulk-read report spans at the exact timestamps used by market snapshots.

    Current predictions read synchronized Aiven timelines. ``live_reports``
    optionally overlays archived reports for historical replay diagnostics.
    With ``target_game_ids`` every other game's reads come from the memo
    (``_stored_report_states``); the parts are put in the order one read over
    every cutoff gives -- the canonical order is a function of the rows.
    """
    with connect_nba_db("aiven") as conn:
        parts, remaining = (
            _stored_report_states(conn, cutoffs, target_game_ids)
            if target_game_ids is not None
            else ([], cutoffs)
        )
        if not remaining.empty or not parts:
            parts.append(live.report_state_at_snapshots(conn, remaining, live_reports))
        statuses, filings, ages = (
            parts[0][i]
            if len(parts) == 1
            else canonical_row_order(
                pd.concat(
                    [part[i] for part in parts if not part[i].empty] or [parts[0][i]],
                    ignore_index=True,
                )
            )
            for i in range(3)
        )
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


#: The availability-effect families: output prefix and the player columns read.
EFFECT_FAMILIES = (
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


def _fit_continues_from_priors(
    games: pd.DataFrame,
    priors: dict | None,
    horizon: int | None,
    first_current_season: int | None,
) -> bool:
    """Whether ``_add_availability_effects`` computes the current seasons only.

    True when every family's shrinkage sums over the earlier seasons are stored
    for this horizon, and there are rows both before and from
    ``first_current_season``: the conditions under which it reads no earlier
    row's effect.
    """
    if priors is None or first_current_season is None:
        return False
    seasons = pd.to_numeric(games["SEASON_YEAR"], errors="coerce")
    current = (seasons >= first_current_season).to_numpy()
    if not current.any() or current.all():
        return False
    return all((horizon, prefix) in priors for prefix, _ in EFFECT_FAMILIES)


def _add_availability_effects(
    games: pd.DataFrame,
    injured_dict: dict,
    availability_dict: dict,
    *,
    book: str,
    effect_cache: RowCache | None = None,
    priors: dict | None = None,
    horizon: int | None = None,
    first_current_season: int | None = None,
) -> pd.DataFrame:
    """The three availability-effect families of one horizon.

    ``priors`` (tonight-only builds) holds, per (horizon, family), the
    shrinkage fit's running sums over the games of the seasons before
    ``first_current_season``: with them only the later games' effects are
    computed, and the fit continues from the sums exactly; without them every
    game's are, and the sums are stored for the next build.
    """
    availability_index = (
        _build_availability_games_index(availability_dict)
        if availability_dict is not None
        else None
    )
    computed_rows = before = None
    if priors is not None:
        seasons = pd.to_numeric(games["SEASON_YEAR"], errors="coerce")
        computed_rows = (seasons >= first_current_season).to_numpy()
        current_dates = pd.to_datetime(games.loc[computed_rows, "GAME_DATE"])
        before = current_dates.min().toordinal() if len(current_dates) else None
        if before is None or not (~computed_rows).any():
            priors = None
    for prefix, names in EFFECT_FAMILIES:
        stored = priors.get((horizon, prefix)) if priors is not None else None
        totals = {} if priors is not None and stored is None else None
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
            availability_index=availability_index,
            computed_rows=computed_rows if stored is not None else None,
            shrinkage_prior=stored,
            shrinkage_totals_before=before if totals is not None else None,
            shrinkage_totals=totals,
        )
        if totals is not None:
            priors[(horizon, prefix)] = totals
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
    current_rosters: dict | None = None,
    target_game_ids: set[str] | None = None,
    priors: dict | None = None,
    horizon: int | None = None,
    first_current_season: int | None = None,
    injured_players: dict | None = None,
    stored_rosters: dict | None = None,
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
    current_rosters = supplement_rosters(current_rosters, report_membership(state))
    original_columns = set(context.team_games.columns)
    team_games = context.team_games
    if target_game_ids is not None and _fit_continues_from_priors(
        base, priors, horizon, first_current_season
    ):
        # Only the current seasons' effects are computed (the earlier ones'
        # shrinkage sums are stored), and an effect reads its own season and
        # the one before (``_infer_last_two_season_years_for_row``): earlier
        # team-games' availability is never read. Tonight's rows read their
        # season and the one before too (the streak lookups); every other read
        # is of the row's own game, or of the players and injuries, which stay
        # whole. So those two seasons' rows are what a build of every row gives.
        seasons = history_memo.season_of_game_ids(team_games["GAME_ID"])
        team_games = team_games[seasons >= first_current_season - 1]
    team, injured_dict, availability = add_player_history_features(
        team_games.copy(),
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
        current_rosters=current_rosters,
        current_game_ids=set(context.scheduled_games.GAME_ID.astype(str))
        if context.scheduled_games is not None
        else None,
        injured_players=injured_players,
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
        # Stored games' game-day rosters, as the player features read them.
        current_rosters={**(stored_rosters or {}), **current_rosters},
        target_game_ids=target_game_ids,
    )
    roster_columns = []
    if current_rosters:
        team = add_roster_continuity_feature(
            team,
            context.players,
            injured_dict=injured_dict,
            current_rosters=current_rosters,
            target_game_ids={game for game, _ in current_rosters},
        )
        roster_columns = [
            c for c in team if c.startswith("ROSTER_") and c not in original_columns
        ]
        roster_columns.extend(
            ["ALL_STAR_FAN_VOTE_SHARE_BEFORE", "ALL_STAR_MIN_SCORE_BEFORE"]
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
    feature_columns = list(dict.fromkeys([*feature_columns, *roster_columns]))
    wide = _wide_team_injury_features(team, feature_columns)
    overlap = (set(wide) & set(base)) - {"GAME_ID"}
    games = base.drop(columns=list(overlap)).merge(
        wide, on="GAME_ID", how="left", validate="one_to_one"
    )
    if team_games is not context.team_games:
        # The games left out above have no rows here, and none is read again:
        # their effects are not computed and their rows not returned. A build
        # of every row has one per game, so its integer columns (the coverage
        # flags) stay integer; they do here too, at 0 on the left-out games.
        left_out = ~games["GAME_ID"].isin(wide["GAME_ID"])
        for column in wide.columns:
            if column != "GAME_ID" and pd.api.types.is_integer_dtype(wide[column]):
                games[column] = (
                    games[column].mask(left_out, 0).astype(wide[column].dtype)
                )

    # Availability-dependent features remain unknown for an unfiled team.
    # Official total-roster features do not require an injury filing.
    for side in ("HOME", "AWAY"):
        flag = f"{COVERED_COL}_TEAM_{side}"
        unavailable = games[flag].fillna(0).eq(0)
        columns = [
            f"{name}_TEAM_{side}"
            for name in feature_columns
            if name != COVERED_COL and f"{name}_TEAM_{side}" in games
        ]
        columns = [
            c
            for c in columns
            if not any(c == f"{name}_TEAM_{side}" for name in roster_columns)
        ]
        games.loc[unavailable, columns] = np.nan

    games = add_fresh_absence_sums(games)
    games = _add_availability_effects(
        games,
        injured_dict,
        availability,
        book=get_main_book(),
        effect_cache=effect_cache,
        priors=priors,
        horizon=horizon,
        first_current_season=first_current_season,
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
    generated = list(dict.fromkeys([*generated, *sorted(overlap)]))
    result = games[["GAME_ID", *generated]].copy()
    for home, side in ((True, "HOME"), (False, "AWAY")):
        eligible = {
            (str(game), str(team))
            for game, team in context.team_games.loc[
                context.team_games.HOME.eq(home), ["GAME_ID", "TEAM_ID"]
            ].itertuples(index=False, name=None)
        } & set(current_rosters)
        result[f"_LIVE_ROSTER_{side}"] = result.GAME_ID.isin(
            {game for game, _ in eligible}
        )
    return result


@dataclass
class _HorizonInputs:
    """What every horizon of ``add_snapshot_injury_features`` reads."""

    base: pd.DataFrame
    context: BaseInjuryContext
    states: dict[int, InjuryReportState]
    cutoffs: pd.DataFrame
    injuries: pd.DataFrame
    box: PreparedBoxHistory
    voting: pd.DataFrame
    player_rows: RowCache | None
    all_star_rows: RowCache | None
    prepared_players: PreparedPlayerHistory
    prepared_all_star: PreparedAllStarInputs
    effect_cache: RowCache | None
    targets: set[str] | None
    priors: dict | None
    first_current_season: int | None
    injured_players: dict | None = None
    current_rosters: dict | None = None
    stored_rosters: dict | None = None


def _horizon_features(inputs: _HorizonInputs, horizon: int) -> pd.DataFrame:
    """One horizon's injury rows, for the games with a snapshot there."""
    cutoffs = inputs.cutoffs
    ids = set(cutoffs.loc[cutoffs.snapshot_minutes.eq(horizon), "game_id"])
    features = _one_horizon(
        inputs.base,
        inputs.context,
        inputs.states[horizon],
        inputs.injuries,
        inputs.box,
        inputs.voting,
        ids,
        inputs.player_rows,
        inputs.all_star_rows,
        inputs.prepared_players,
        inputs.prepared_all_star,
        inputs.effect_cache,
        current_rosters=inputs.current_rosters,
        target_game_ids=inputs.targets,
        priors=inputs.priors,
        horizon=horizon,
        first_current_season=inputs.first_current_season,
        injured_players=inputs.injured_players,
        stored_rosters=inputs.stored_rosters,
    )
    features = features.loc[features.GAME_ID.isin(ids)].copy()
    if inputs.targets is not None:
        features = features.loc[features.GAME_ID.astype(str).isin(inputs.targets)]
    features["TIME_TO_MATCH_MIN"] = horizon
    return features


def _horizon_workers(n_horizons: int) -> int:
    """Processes for the horizons: ``NBA_OU_INJURY_WORKERS``, else up to 8.

    One core is left free; 1 (or fork being unavailable) runs them in turn.
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        return 1
    configured = os.environ.get("NBA_OU_INJURY_WORKERS")
    if configured:
        return max(1, min(int(configured), n_horizons))
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    return max(1, min(8, cores - 1, n_horizons))


#: Past this, the forked horizons are abandoned and run in turn.
FORKED_TIMEOUT_SECONDS = 600

#: The forked workers' inputs: set in the parent just before the fork, so
#: every worker reads them from the copied memory instead of a pickle.
_FORKED_INPUTS: _HorizonInputs | None = None


def _forked_horizon(horizon: int):
    inputs = _FORKED_INPUTS
    priors_before = set(inputs.priors) if inputs.priors is not None else set()
    features = _horizon_features(inputs, horizon)
    added = {
        name: cache.added_entries()
        for name, cache in (
            ("players", inputs.player_rows),
            ("effects", inputs.effect_cache),
        )
        if isinstance(cache, history_memo.PersistentRowCache)
    }
    priors = (
        {k: v for k, v in inputs.priors.items() if k not in priors_before}
        if inputs.priors is not None
        else {}
    )
    return features, added, priors


def _stop(pool: ProcessPoolExecutor) -> None:
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)


def _horizons_in_processes(
    inputs: _HorizonInputs,
    memo: history_memo.HistoryMemo,
    workers: int,
    progress: str | None,
    while_waiting: Callable[[], None] | None = None,
) -> list[pd.DataFrame]:
    """The horizons in forked processes, each the same computation as in turn.

    Each worker starts from the parent's caches and returns the season
    entries and shrinkage sums it added; the parent merges them, so the memo
    gains what a sequential build would have stored. Horizons are
    independent: a cached row is the row its key computes, whichever
    process computed it first. ``while_waiting`` runs here meanwhile, once.
    """
    global _FORKED_INPUTS
    memo.wait()
    _FORKED_INPUTS = inputs
    horizons = list(inputs.states)
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    )
    failure = None
    try:
        try:
            pending = pool.map(
                _forked_horizon, horizons, timeout=FORKED_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 -- the fork itself failed
            pending, failure = None, exc
        if while_waiting is not None:
            while_waiting()
        if pending is not None:
            try:
                results = list(
                    tqdm(
                        pending,
                        total=len(horizons),
                        desc=progress,
                        unit="snapshot",
                        disable=progress is None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- a stuck or broken worker
                failure = exc
    except BaseException:
        _stop(pool)
        raise
    finally:
        _FORKED_INPUTS = None
    if failure is not None:
        # A fork of a threaded process can deadlock; the horizons then run in
        # turn, which costs time, never a value.
        print(f"Forked injury horizons failed ({failure!r}); running them in turn")
        _stop(pool)
        return [_horizon_features(inputs, horizon) for horizon in horizons]
    pool.shutdown()
    parts = []
    for features, added, priors in results:
        for name, cache in (
            ("players", inputs.player_rows),
            ("effects", inputs.effect_cache),
        ):
            if name in added:
                cache.merge_entries(added[name])
        if inputs.priors is not None:
            inputs.priors.update(priors)
        parts.append(features)
    return parts


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


#: Memo table of the shrinkage fits' running sums over finished seasons.
EFFECT_PRIORS_MEMO = "intermediate_injury_effect_priors"


def _effect_prior_table(
    memo: history_memo.HistoryMemo, cutoffs: pd.DataFrame, targets: set[str]
) -> tuple[history_memo.HistoryMemo | None, dict | None, int | None]:
    """The stored shrinkage sums over every season before the targets'.

    Per (horizon, family) and metric, the sums the availability-effect fit has
    run up over the finished seasons' games. They depend on those games'
    players and history (the injury memo's digest of the last finished
    season), their reports (the chained report fingerprint) and their
    snapshots (the cutoffs), and the entry is keyed on all three.
    """
    seasons = [history_memo.season_of_game_id(g) for g in targets]
    seasons = [s for s in seasons if s is not None]
    if not seasons:
        return None, None, None
    first_current = min(seasons)
    last_finished = first_current - 1
    if last_finished not in memo.digests:
        return None, None, None
    with connect_nba_db("aiven") as conn:
        reports = fetch.report_season_fingerprints(conn)
    known = [s for s in reports if s <= last_finished]
    finished = cutoffs[
        history_memo.season_of_game_ids(cutoffs["game_id"]) <= last_finished
    ].sort_values(["game_id", "snapshot_minutes"])
    digest = hashlib.sha256(
        f"{memo.digests[last_finished]}|{reports[max(known)] if known else 'none'}".encode()
    )
    digest.update(history_memo._frame_digest(finished.reset_index(drop=True)))
    prior_memo = history_memo.HistoryMemo(
        memo.backend, {last_finished: digest.hexdigest()}, last_finished
    )
    table = prior_memo.table(EFFECT_PRIORS_MEMO, last_finished)
    if table is None:
        return None, None, None
    return prior_memo, table, first_current


def add_snapshot_injury_features(
    snapshots: pd.DataFrame,
    base: pd.DataFrame,
    context: BaseInjuryContext,
    *,
    progress: str | None = None,
    target_game_ids: set[str] | None = None,
    history_cutoffs: pd.DataFrame | None = None,
    live_reports: LiveInjuryReports | None = None,
    while_waiting: Callable[[], None] | None = None,
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

    ``while_waiting`` is other work of the caller's, run once in this process
    while the horizons run in others (or before them, when they run in turn).
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
    states = _states_at_snapshots(
        cutoffs,
        live_reports,
        target_game_ids=(
            None if target_game_ids is None else {str(g) for g in target_game_ids}
        ),
    )
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
    memo = prior_memo = priors = first_current_season = None
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
            prior_memo, priors, first_current_season = _effect_prior_table(
                memo, cutoffs, targets
            )
    # The same for every horizon: the players' cumulative statistics, sort
    # and lookup indexes, and the All-Star votes, do not depend on the report.
    prepared_players = prepare_player_history(context.players.copy(), PLAYER_STATS)
    prepared_all_star = prepare_all_star_inputs(context.players, voting)
    # One pool for every horizon: the latest same-day official roster known to
    # this run, as the closing prediction uses. A roster fetched after an
    # earlier horizon's cutoff still applies to it; only an intraday transaction
    # between that cutoff and the run would differ, which is accepted.
    pools = roster_overrides(
        context.scheduled_games,
        context.roster_snapshots,
        as_of=context.live_as_of,
    )
    realized = get_injured_players_dict(injuries, df_players=context.players)
    if context.scheduled_games is not None and not context.scheduled_games.empty:
        # No official roster: the previous game's, as the closing prediction.
        pools.update(
            previous_game_rosters(
                context.team_games,
                context.players,
                realized,
                context.scheduled_games["GAME_ID"],
                rosters=pools,
            )
        )
    scheduled_ids = (
        set(context.scheduled_games.GAME_ID.astype(str))
        if context.scheduled_games is not None
        else set()
    )
    inputs = _HorizonInputs(
        current_rosters=pools,
        stored_rosters=game_day_rosters(
            context.players, realized, exclude_games=scheduled_ids
        ),
        base=base,
        context=context,
        states=states,
        cutoffs=cutoffs,
        injuries=injuries,
        box=box,
        voting=voting,
        player_rows=player_rows,
        all_star_rows=all_star_rows,
        prepared_players=prepared_players,
        prepared_all_star=prepared_all_star,
        effect_cache=effect_cache,
        targets=targets if target_game_ids is not None else None,
        priors=priors,
        first_current_season=first_current_season,
        # Realized absences: the same at every horizon.
        injured_players=realized,
    )
    workers = _horizon_workers(len(states)) if memo is not None else 1
    if workers > 1:
        parts = _horizons_in_processes(
            inputs, memo, workers, progress, while_waiting=while_waiting
        )
    else:
        if while_waiting is not None:
            while_waiting()
        parts = [
            _horizon_features(inputs, horizon)
            for horizon in tqdm(
                states,
                total=len(states),
                desc=progress,
                unit="snapshot",
                disable=progress is None,
            )
        ]
    if memo is not None:
        memo.flush()
    if prior_memo is not None:
        prior_memo.flush()
    injury = pd.concat(parts, ignore_index=True)
    overlap = (set(injury.columns) & set(snapshots.columns)) - set(KEYS)
    replacements = {c for c in overlap if c.startswith(("ALL_STAR_", "ROSTER_"))}
    if overlap - replacements:
        raise ValueError(
            f"Snapshot injury columns overlap existing columns: {sorted(overlap)}"
        )
    result = snapshots.merge(
        injury, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_ROSTER")
    )
    for column in replacements:
        side = "HOME" if column.endswith("_TEAM_HOME") else "AWAY"
        replace = result[f"_LIVE_ROSTER_{side}"].fillna(False).astype(bool)
        result[column] = result[column].where(~replace, result[f"{column}_ROSTER"])
    return result.drop(
        columns=[
            "_LIVE_ROSTER_HOME",
            "_LIVE_ROSTER_AWAY",
            *[f"{c}_ROSTER" for c in replacements],
        ]
    )
