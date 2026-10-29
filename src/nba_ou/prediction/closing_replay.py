"""Closing-model inputs reconstructed at four explicit pregame checkpoints.

Run the canonical closing feature builder on historical quotes and reports,
never on the day page served after the game. Outcomes remain outside inputs.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.odds_columns import (
    drop_history_only_book_columns,
    drop_sportsbook_metadata_columns,
)
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.create_training_data.intermediate_referees import (
    mask_referees_before_release,
)
from nba_ou.create_training_data.live_replay import stored_crews
from nba_ou.data_processing.injury_status.report_state import (
    load_prediction_report_state_at_cutoffs,
    nested_status_dict,
    report_out_overrides,
)
from nba_ou.fetch_data.live_rosters import load_live_rosters
from nba_ou.fetch_data.odds_covers.day_odds import covers_day_odds
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_pregame_ticks
from nba_ou.utils.general_utils import get_season_year_from_date

CLOSING_REPLAY_HORIZONS = (480, 360, 240, 0)


def historical_odds(
    scheduled: pd.DataFrame,
    ticks: pd.DataFrame,
    horizon: int,
    game_date: str,
    *,
    cutoffs: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Stored pregame ticks -> the live closing builder's SBR day-page schema."""
    names = {str(team_id): name for name, team_id in TEAM_ID_MAP.items()}
    season = int(get_season_year_from_date(game_date))
    frames = []
    for game in scheduled.itertuples(index=False):
        tipoff = pd.Timestamp(game.GAME_TIME).tz_convert("UTC")
        cutoff = (cutoffs or {}).get(
            str(game.GAME_ID), tipoff - pd.Timedelta(minutes=horizon)
        )
        known = ticks.loc[
            ticks.game_id.astype(str).eq(str(game.GAME_ID))
            & pd.to_datetime(ticks.line_ts, utc=True).le(cutoff)
            & pd.to_numeric(ticks.minutes_before_tip).gt(0)
        ]
        quotes = tuple(
            LineTick(
                market=t.market,
                book_slug=t.book,
                book_name=t.book,
                line_ts=pd.Timestamp(t.line_ts).to_pydatetime(),
                minutes_to_tip=-int(t.minutes_before_tip),
                is_opener=bool(t.is_opener),
                left_line=None if pd.isna(t.left_line) else float(t.left_line),
                right_line=None if pd.isna(t.right_line) else float(t.right_line),
                left_price=None if pd.isna(t.left_price) else int(t.left_price),
                right_price=None if pd.isna(t.right_price) else int(t.right_price),
            )
            for t in known.itertuples(index=False)
        )
        scraped = ScrapedGame(
            event_id=int(game.GAME_ID),
            game_date=pd.Timestamp(game_date).date(),
            season_year=season,
            tipoff_utc=tipoff.to_pydatetime(),
            team_home=names[str(game.HOME_TEAM_ID)],
            team_away=names[str(game.VISITOR_TEAM_ID)],
            status_text="Scheduled",
            home_score=None,
            away_score=None,
            ticks=quotes,
        )
        # SBR's opener is the first known BetMGM quote, while Covers stores
        # an explicit consensus_opener. Never read an opener after the cutoff.
        opener = []
        for market, part in known.loc[known.book.eq("betmgm")].groupby("market"):
            if not any(
                t.book_slug == "consensus_opener" and t.market == market for t in quotes
            ):
                first = part.sort_values("line_ts").iloc[0]
                quote = next(
                    t
                    for t in quotes
                    if t.book_slug == "betmgm"
                    and t.market == market
                    and pd.Timestamp(t.line_ts) == pd.Timestamp(first.line_ts)
                )
                opener.append(
                    replace(quote, book_slug="consensus_opener", is_opener=True)
                )
        scraped = replace(scraped, ticks=(*quotes, *opener))
        odds = covers_day_odds(
            [scraped], game_date=pd.Timestamp(game_date).date(), as_of=cutoff
        )
        odds["game_id"] = str(game.GAME_ID)
        frames.append(odds)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return drop_sportsbook_metadata_columns(drop_history_only_book_columns(raw))


def closing_replay_frame(
    scheduled: pd.DataFrame,
    snapshots: pd.DataFrame,
    game_date: str,
    *,
    frame_version: str = BASE_SCHEMA_VERSION,
    ticks: pd.DataFrame | None = None,
    crews: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Four closing inputs per game, retaining raw snapshot quotes for grading."""
    from nba_ou.create_training_data.schema_layers import (
        compute_new_columns,
        is_layered,
    )

    snapshots = snapshots.loc[
        snapshots.TIME_TO_MATCH_MIN.isin(CLOSING_REPLAY_HORIZONS)
        & snapshots.SNAPSHOT_PASSED
    ].copy()
    if snapshots.empty:
        return pd.DataFrame()
    season = int(get_season_year_from_date(game_date))
    if ticks is None:
        ticks = fetch_pregame_ticks([season], use_cache=True)
    ids = set(scheduled.GAME_ID.astype(str))
    ticks = ticks.loc[ticks.game_id.astype(str).isin(ids)]
    if crews is None:
        crews = stored_crews(sorted(ids), season)
    # One archived roster read; roster_overrides applies each game's cutoff
    # below, so an earlier snapshot cannot inherit a later observation.
    rosters = load_live_rosters(
        scheduled, as_of=pd.to_datetime(snapshots.SNAPSHOT_TS_UTC, utc=True).max()
    )
    frames = []
    for horizon in CLOSING_REPLAY_HORIZONS:
        snapshot = snapshots.loc[snapshots.TIME_TO_MATCH_MIN.eq(horizon)].copy()
        if snapshot.empty:
            continue
        games = scheduled.loc[
            scheduled.GAME_ID.astype(str).isin(snapshot.GAME_ID.astype(str))
        ]
        cutoffs = pd.DataFrame(
            {
                "game_id": games.GAME_ID.astype(str),
                "snapshot_minutes": horizon,
                "as_of": games.GAME_ID.map(
                    snapshot.set_index("GAME_ID").SNAPSHOT_TS_UTC
                ),
            }
        ).reset_index(drop=True)
        state = load_prediction_report_state_at_cutoffs(cutoffs)
        game_ids = set(games.GAME_ID.astype(str))
        # Match live closing eligibility and roster inputs at this checkpoint.
        # The state also carries historical games, whose realized absences
        # must not be replaced by these scheduled-game overrides.
        injury_dict = nested_status_dict(
            {
                key: players
                for key, players in report_out_overrides(state).items()
                if key[0] in game_ids
            }
        )
        missing_reports = [
            str(game.GAME_ID)
            for game in games.itertuples()
            if any(
                (str(game.GAME_ID), str(team)) not in state.covered
                for team in (game.HOME_TEAM_ID, game.VISITOR_TEAM_ID)
            )
        ]
        data = {
            "scheduled_games": games,
            "df_referees_scheduled": crews,
            "injury_dict_scheduled": injury_dict,
            "games_not_updated": missing_reports,
            "injury_report_state": state,
            "injury_as_of_utc": cutoffs.as_of.max(),
            "roster_as_of_utc": cutoffs.set_index("game_id").as_of.to_dict(),
            "roster_snapshots": rosters,
            "df_odds_yahoo_scheduled": pd.DataFrame(),
            "df_odds_sportsbook_scheduled": historical_odds(
                games,
                ticks,
                horizon,
                game_date,
                cutoffs=cutoffs.set_index("game_id").as_of.to_dict(),
            ),
        }
        frame, layer_inputs = create_df_to_predict(
            todays_prediction=True,
            scheduled_data=data,
            recent_limit_to_include=(
                pd.Timestamp(game_date) - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d"),
            strict_mode=-1,
            target_game_ids=list(games.GAME_ID),
            return_layer_inputs=True,
        )
        if is_layered(frame_version):
            added = compute_new_columns(
                layer_inputs,
                from_version=BASE_SCHEMA_VERSION,
                to_version=frame_version,
                dataset_type="closing_line",
                existing_columns=frame.columns,
            )
            frame = pd.concat([frame, added.loc[frame.index]], axis=1)
        frame = frame.loc[frame.GAME_ID.astype(str).isin(ids)].copy()
        frame["GAME_ID"] = frame.GAME_ID.astype(str)
        # The raw quote/age columns are bookkeeping only; all model features
        # above came through create_df_to_predict with the historical cutoff.
        raw_columns = [
            c
            for c in snapshot
            if c.startswith("ODDS_SNAP_")
            or c in ("GAME_ID", "SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED", "LINE_SOURCE")
        ]
        frame = frame.merge(
            snapshot[raw_columns], on="GAME_ID", how="inner", validate="one_to_one"
        )
        frame["TIME_TO_MATCH_MIN"] = horizon
        frame["INPUT_MODE"] = "replay"
        frame["TIPOFF_UTC"] = pd.to_datetime(
            frame.SNAPSHOT_TS_UTC, utc=True
        ) + pd.Timedelta(minutes=horizon)
        frame = mask_referees_before_release(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
