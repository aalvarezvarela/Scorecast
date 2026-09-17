"""Per-game pre-game features for the intermediate-line dataset.

A *different composition* of the same feature functions ``create_df_to_predict``
uses -- not a copy of them, and not a modification. Every function called here is
imported from its existing home, so the two datasets cannot drift apart in how a
shared feature is computed.

Three stages are deliberately absent from this base stage:

* **Player-level statistics and injury status** -- attached after the base game
  build using the latest report strictly before each snapshot. The base stage
  only computes team history and supplies the player context for that step.
* **Referees** -- attached after the base game build, then masked per snapshot
  until 09:00 Eastern on game day.
* **``engineer_odds_features``** -- close-time by its own docstring. Its useful
  parts (consensus, dispersion, vig) are rebuilt from snapshot prices in
  ``data_processing/line_history/`` instead.

Two stages that *are* included, and were not in the first version: **team-level
all-star vote share** and **roster continuity**. Both need the roster, which is
not the same thing as injury status -- see ``create_base_game_features`` for how
the two are separated.

What *is* kept is the rolling team history, including rolling statistics over
**prior** games' closing odds. Those games had closed long before any snapshot
of the game being predicted, so they are leakage-safe and are some of the
strongest features in the existing model.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.config.odds_columns import (
    apply_odds_prefix,
    assert_odds_columns_prefixed,
    get_main_book,
)

# Imported rather than reimplemented: this is the same team pipeline the
# closing-line dataset runs, so rolling statistics are identical between them.
from nba_ou.create_training_data.create_df_to_predict import (
    process_team_statistics_for_training,
)
from nba_ou.create_training_data.predict_data_utils import (
    filter_by_seasons_with_extra_game_ids,
)
from nba_ou.data_processing.all_star_voting.attach_all_star_voting_features import (
    add_all_star_voting_features,
    all_star_season_year_for_game_date,
)
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_betting_stats_differences,
    add_derived_features_after_computed_stats,
    add_fresh_absence_sums,
    add_game_date_features,
    add_high_value_features_for_team_points,
)
from nba_ou.data_processing.merged_home_away_data.global_market_features import (
    add_global_market_features,
)
from nba_ou.data_processing.merged_home_away_data.merge_home_away import (
    merge_home_away_data,
)
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    select_training_columns,
)
from nba_ou.data_processing.merged_home_away_data.team_one_hot_features import (
    add_team_one_hot_features,
)
from nba_ou.data_processing.past_injuries.past_injuries import get_injured_players_dict
from nba_ou.data_processing.players.attach_player_features import (
    clear_player_statistics,
)
from nba_ou.data_processing.players.roster_continuity import (
    add_roster_continuity_feature,
)
from nba_ou.data_processing.players.starter_history import (
    add_starter_history_features,
)
from nba_ou.data_processing.team.merge_game_df_with_odds_by_game_id import (
    merge_remaining_odds_by_game_id,
)
from nba_ou.data_processing.team.style_matchups import add_style_matchup_features
from nba_ou.data_processing.travel.travel_processing import compute_travel_features
from nba_ou.postgre_db import load_all_nba_data_from_db
from nba_ou.postgre_db.all_star_voting.fetch_data_from_db.fetch_all_star_voting_from_db import (
    load_all_star_voting_from_db,
)
from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (
    get_injury_data_from_db,
)
from nba_ou.postgre_db.odds.merge_odds_data import (
    load_and_merge_odds_yahoo_sportsbookreview,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.seasons import get_seasons_between_dates

warnings.simplefilter(action="ignore", category=FutureWarning)

DEFAULT_BOOK = get_main_book()


@dataclass
class BaseInjuryContext:
    """Inputs already loaded for snapshot-specific player and report features."""

    team_games: pd.DataFrame
    players: pd.DataFrame
    seasons: list[str]
    player_context_seasons: list[str]


#: ``merge_home_away_data`` derives two "star player" ratios inline and indexes
#: these columns directly, so it raises if the player stage never ran. They are
#: injected as NaN rather than patching the shared function, which must keep
#: behaving exactly as it does for ``create_df_to_predict``.
_PLAYER_PLACEHOLDER_COLUMNS = (
    "TOP1_PLAYER_OFF_RATING_BEFORE",
    "TOP1_PLAYER_PTS_BEFORE",
)

#: What those two ratios become without players: a constant 0 and a column of
#: NaN. Neither carries information, so both are dropped after the merge.
_STAR_DERIVED_COLUMNS = (
    "STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE",
    "STAR_PTS_PERCENTAGE_BEFORE",
)


#: All-star column families kept. The ``*_INJURED_*`` families are dropped: they
#: are the only part of this stage that depends on injury data, which this
#: dataset has no trustworthy timestamped history for.
_ALL_STAR_KEEP_PREFIXES = (
    "ALL_STAR_FAN_VOTE_SHARE_BEFORE",
    "ALL_STAR_MIN_SCORE_BEFORE",
)


def _add_all_star_team_features(
    df: pd.DataFrame, df_players: pd.DataFrame, *, verbose: bool
) -> pd.DataFrame:
    """Team-level all-star fan-vote share, without the injured-player variants.

    Leakage-safe by construction: ``all_star_season_year_for_game_date`` always
    resolves to the most recently *completed* vote. A November 2024 game maps to
    the January 2024 ballot, never to the one published the following January.
    """
    required_years = sorted(
        {all_star_season_year_for_game_date(date) for date in df["GAME_DATE"]}
    )
    voting = load_all_star_voting_from_db(season_years=required_years)
    if voting is None:
        raise RuntimeError(
            "load_all_star_voting_from_db returned None for season_years="
            f"{required_years}. Cannot build all-star features."
        )

    # An empty injured_dict leaves the *_INJURED_* columns unpopulated rather
    # than wrong; they are dropped below regardless.
    out = add_all_star_voting_features(
        df_team=df,
        df_players=df_players,
        all_star_voting_df=voting,
        injured_dict={},
    )

    dropped = [
        column
        for column in out.columns
        if column.startswith("ALL_STAR_")
        and not column.startswith(_ALL_STAR_KEEP_PREFIXES)
    ]
    if verbose and dropped:
        print(f"  dropped {len(dropped)} injury-dependent/auxiliary all-star columns")
    return out.drop(columns=dropped)


def _lag_injured_dict_by_one_team_game(
    injured_dict: dict, game_context: pd.DataFrame
) -> dict:
    """Re-key each game's injury entry onto that team's NEXT game.

    Why this exists. ``add_roster_continuity_feature`` uses the injury report
    purely as a *roster-assignment* source -- it records "player X was on team Y
    on date D" and never reads the status (out/questionable/available). That is
    roster membership, not injury news, and it is worth a lot: dropping it moves
    27-54% of team-games, by up to 0.58 absolute.

    But the function stamps injury events as known on the game date itself
    (box scores only become known the following day), so the *current* game's
    report counts as pre-tip information. At a 12h snapshot that report may not
    have existed yet, which is exactly the train/inference mismatch this dataset
    exists to avoid.

    Lagging by one team-game removes the exposure by construction: a game only
    ever sees reports from games its team has already played. Measured over
    2023-24 and 2024-25 this still captures **74-89%** of the full report's
    effect, so almost nothing is given up. That works because the underlying
    fact is stable -- a player on the roster tonight was on it last game too --
    while only genuinely new assignments (trades, 10-day deals, call-ups) need
    the same-day report.

    Purely a caller-side transform; ``roster_continuity.py`` is untouched.
    """
    metadata = (
        game_context[["GAME_ID", "TEAM_ID", "GAME_DATE"]].dropna().drop_duplicates()
    )
    metadata = metadata.astype({"GAME_ID": str, "TEAM_ID": str})
    metadata = metadata.sort_values(["TEAM_ID", "GAME_DATE"])
    metadata["NEXT_GAME_ID"] = metadata.groupby("TEAM_ID")["GAME_ID"].shift(-1)
    next_game = {
        (row.TEAM_ID, row.GAME_ID): row.NEXT_GAME_ID
        for row in metadata.itertuples(index=False)
        if isinstance(row.NEXT_GAME_ID, str)
    }

    lagged: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for game_id, team_map in injured_dict.items():
        for team_id, player_ids in team_map.items():
            # A team's final game has no successor, so its report is dropped
            # rather than carried into a season it does not belong to.
            successor = next_game.get((str(team_id), str(game_id)))
            if successor is None:
                continue
            lagged[successor][str(team_id)].extend(player_ids)
    return {game: dict(teams) for game, teams in lagged.items()}


def _build_roster_injury_dict(
    mode: str, seasons: list[str], df_players: pd.DataFrame, game_context: pd.DataFrame
) -> dict | None:
    """Injury reports as a roster source, under the requested timing policy."""
    if mode == "none":
        return None
    if mode not in {"lagged", "full"}:
        raise ValueError(
            f"roster_injury_reports must be 'none', 'lagged' or 'full'; got {mode!r}."
        )

    injuries = get_injury_data_from_db(seasons)
    injured_dict = get_injured_players_dict(injuries, df_players=df_players)
    # Sets are fine for the consumer but awkward to concatenate when lagging.
    injured_dict = {
        game: {team: list(players) for team, players in team_map.items()}
        for game, team_map in injured_dict.items()
    }
    if mode == "full":
        return injured_dict
    return _lag_injured_dict_by_one_team_game(injured_dict, game_context)


def _inject_player_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns ``merge_home_away_data`` indexes unconditionally."""
    out = df.copy()
    for column in _PLAYER_PLACEHOLDER_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def create_base_game_features(
    *,
    recent_limit_to_include: str | pd.Timestamp | None = None,
    older_season_limit: int | None = None,
    season_start_date: str | pd.Timestamp | None = None,
    categorical_team_encoding: bool = False,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    include_all_star: bool = True,
    include_roster_continuity: bool = True,
    roster_injury_reports: str = "lagged",
    exclude_caesars: bool = False,
    combine_fanatics_and_caesars: bool | None = None,
    return_injury_context: bool = False,
    verbose: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, BaseInjuryContext]:
    """One row per game, carrying only leakage-safe pre-game team features.

    The returned frame still contains this game's closing-odds columns -- they
    are needed by ``add_global_market_features`` and by the historical rollups.
    Stripping them is the job of ``select_intermediate_columns``, which runs
    after the snapshot join so the exclusion is enforced in exactly one place.

    ``include_all_star`` and ``include_roster_continuity`` both need the player
    roster, which is *not* the same thing as injury data. Rosters are known days
    ahead; who is ruled out tonight is not, and only the latter is excluded from
    this dataset. The all-star stage is given an empty ``injured_dict``, so its
    ``*_INJURED_*`` columns are never built.

    ``roster_injury_reports`` controls how continuity uses the injury feed as a
    *roster-assignment* source (it never reads the status):

    * ``"lagged"`` (default) -- each game's report is re-keyed onto that team's
      next game, so a game only sees reports from games already played. No
      same-day timestamp assumption, and it still captures 74-89% of the full
      report's effect.
    * ``"none"`` -- box scores only. Safe but throws real signal away: dropping
      the report moves 27-54% of team-games.
    * ``"full"`` -- as the closing-line pipeline does. Uses the current game's
      report, which may not have existed at a 12h snapshot.

    ``exclude_caesars`` / ``combine_fanatics_and_caesars`` reconcile the
    discontinued Caesars book with fanatics_sportsbook. Combining is the
    default; passing ``exclude_caesars=True`` switches to dropping it instead.
    See ``nba_ou.data_processing.odds.book_combination``.
    """
    if recent_limit_to_include is None:
        recent_limit_to_include = pd.Timestamp.now(
            tz=ZoneInfo("US/Pacific")
        ) - pd.Timedelta(days=1)
    recent_limit_to_include = pd.to_datetime(recent_limit_to_include)

    if season_start_date is not None:
        season_start = pd.to_datetime(season_start_date)
    elif older_season_limit is not None:
        current_season_year = get_season_year_from_date(recent_limit_to_include)
        season_start = pd.Timestamp(
            year=current_season_year - (older_season_limit - 1), month=10, day=1
        )
    else:
        season_start = pd.to_datetime("2017-10-01")

    seasons = get_seasons_between_dates(season_start, recent_limit_to_include)

    # Roster continuity measures a window that opens on March 15 of the
    # PRECEDING season, so one extra season of player history is loaded purely
    # as context. It never becomes an output row.
    needs_players = include_all_star or include_roster_continuity
    player_context_seasons = (
        get_seasons_between_dates(
            season_start - pd.DateOffset(years=1), recent_limit_to_include
        )
        if needs_players
        else seasons
    )

    if verbose:
        print(f"Loading games for seasons: {seasons}")
    df, df_players = load_all_nba_data_from_db(seasons=player_context_seasons)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df_team_player_context = filter_by_seasons_with_extra_game_ids(
        df,
        seasons=player_context_seasons,
        recent_limit_to_include=recent_limit_to_include,
    )
    df = filter_by_seasons_with_extra_game_ids(
        df, seasons=seasons, recent_limit_to_include=recent_limit_to_include
    )
    original_columns = df.columns.tolist()
    if verbose:
        print(f"✓ Loaded {len(df)} team-game rows")

    if needs_players:
        df_players = clear_player_statistics(df_players, df_team_player_context)
        df_players = filter_by_seasons_with_extra_game_ids(
            df_players,
            seasons=player_context_seasons,
            recent_limit_to_include=recent_limit_to_include,
        )
        if verbose:
            print(f"✓ Loaded {len(df_players)} player rows for roster context")

    df_odds = load_and_merge_odds_yahoo_sportsbookreview(
        season_years=seasons,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        exclude_caesars=exclude_caesars,
        combine_fanatics_and_caesars=combine_fanatics_and_caesars,
    )

    if verbose:
        print("Processing team statistics...")
    df = process_team_statistics_for_training(
        df,
        df_odds,
        scheduled_games=None,
        spread_ml_book=DEFAULT_BOOK,
        total_line_book=DEFAULT_BOOK,
    )
    injury_context = (
        BaseInjuryContext(
            team_games=df.copy(),
            players=df_players.copy(),
            seasons=seasons,
            player_context_seasons=player_context_seasons,
        )
        if return_injury_context
        else None
    )

    if include_all_star:
        if verbose:
            print("Adding all-star fan-vote share features...")
        df = _add_all_star_team_features(df, df_players, verbose=verbose)

    if include_roster_continuity:
        if verbose:
            print(
                "Adding roster continuity features "
                f"(injury reports: {roster_injury_reports})..."
            )
        df = add_roster_continuity_feature(
            df,
            df_players,
            injured_dict=_build_roster_injury_dict(
                roster_injury_reports,
                player_context_seasons,
                df_players,
                df_team_player_context,
            ),
            df_game_context=df_team_player_context,
        )

    if needs_players:
        df = add_starter_history_features(df, df_players)

    if verbose:
        print("Merging home/away data...")
    df_merged = merge_home_away_data(
        _inject_player_placeholders(df), todays_prediction=False
    )
    df_merged = df_merged.drop(
        columns=[
            column
            for column in df_merged.columns
            if column.startswith(_STAR_DERIVED_COLUMNS)
            or column.startswith(_PLAYER_PLACEHOLDER_COLUMNS)
        ],
        errors="ignore",
    )
    df_merged = add_team_one_hot_features(
        df_merged, categorical_team_encoding=categorical_team_encoding
    )
    df_merged = merge_remaining_odds_by_game_id(
        df_odds=df_odds,
        df_merged=df_merged,
        exclude_books=[DEFAULT_BOOK],
        exclude_yahoo=False,
    )
    df_merged = add_betting_stats_differences(df_merged)
    df_merged = add_global_market_features(df_merged)
    df_merged = add_fresh_absence_sums(df_merged)

    # The existing leakage gate runs first and unchanged; the intermediate
    # dataset's stricter gate runs later, on top of it.
    df_training = select_training_columns(df_merged, original_columns)

    df_training = add_derived_features_after_computed_stats(df_training)
    df_training = compute_travel_features(df_training, log_scale=True)
    df_training = add_high_value_features_for_team_points(df_training)
    df_training = add_style_matchup_features(df_training)
    df_training = add_game_date_features(df_training)

    # Applied last, for the same reason as in create_df_to_predict: the feature
    # adders above still resolve market columns by their raw names, so the ODDS_
    # marker goes on only once nothing reads them unprefixed.
    df_training = apply_odds_prefix(df_training)
    assert_odds_columns_prefixed(
        df_training.columns, context="create_base_game_features"
    )

    if verbose:
        print(
            f"✓ Base game features: {df_training.shape[0]} games, "
            f"{df_training.shape[1]} columns"
        )
    if injury_context is not None:
        return df_training, injury_context
    return df_training
