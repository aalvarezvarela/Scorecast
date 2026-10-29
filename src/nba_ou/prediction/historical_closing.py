"""Closing-model inputs rebuilt at an earlier pregame timestamp.

The training dataset is closing_line; today's input is the closing feature
pipeline fed only quotes/rosters/reports available at the chosen timepoint.
Stored final quotes and untimestamped Yahoo values are never used as inputs.
"""

from __future__ import annotations

import pandas as pd
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.create_training_data.intermediate_referees import (
    mask_referees_before_release,
)
from nba_ou.data_processing.injury_status.report_state import (
    InjuryReportState,
    load_prediction_report_state,
    nested_status_dict,
    report_out_overrides,
)
from nba_ou.data_processing.odds.closing_line_repair import select_history_closes
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db.fetch_data_from_odds_sportsbook_db import (
    load_odds_sportsbook_from_db,
)


def sportsbook_odds_at_snapshots(template, ticks, games, cutoffs):
    """Use the stored table's schema, clearing every value except game identity."""
    identity = {
        "game_id",
        "game_date",
        "season_year",
        "team_home",
        "team_away",
        "team_home_abbr",
        "team_away_abbr",
    }
    rows = []
    for game in games.itertuples():
        game_id = str(game.game_id)
        original = (
            template.loc[template.game_id.astype(str).eq(game_id)]
            if "game_id" in template
            else pd.DataFrame()
        )
        values = {column: float("nan") for column in template.columns}
        if not original.empty:
            values.update(
                {
                    column: original.iloc[0][column]
                    for column in identity
                    if column in original
                }
            )
        values.update(
            game_id=game_id,
            game_date=pd.Timestamp(game.game_date),
            season_year=int(game.season_year),
            team_home=game.team_home,
            team_away=game.team_away,
        )
        eligible = ticks.loc[
            ticks.game_id.astype(str).eq(game_id)
            & pd.to_datetime(ticks.line_ts, utc=True).le(cutoffs[game_id])
        ]
        for quote in select_history_closes(eligible).itertuples():
            book = quote.book
            if quote.market == MARKET_TOTALS:
                values.update(
                    {
                        f"total_{book}_line_over": quote.line,
                        f"total_{book}_line_under": quote.line,
                        f"total_{book}_price_over": quote.left_price,
                        f"total_{book}_price_under": quote.right_price,
                    }
                )
            elif quote.market == MARKET_SPREAD:
                values.update(
                    {
                        f"spread_{book}_line_away": quote.line,
                        f"spread_{book}_line_home": -quote.line,
                        f"spread_{book}_price_away": quote.left_price,
                        f"spread_{book}_price_home": quote.right_price,
                    }
                )
            elif quote.market == MARKET_MONEYLINE:
                values.update(
                    {
                        f"ml_{book}_price_away": quote.left_price,
                        f"ml_{book}_price_home": quote.right_price,
                    }
                )
        rows.append(values)
    return pd.DataFrame(rows)


def reports_at_own_snapshots(cutoffs):
    states = {
        game: load_prediction_report_state([game], cutoff)
        for game, cutoff in cutoffs.items()
    }
    first = next(iter(states.values()))
    ids = set(cutoffs)

    def combine(name):
        earlier = getattr(first, name)
        frames = [earlier.loc[~earlier.game_id.astype(str).isin(ids)]]
        for game, state in states.items():
            current = getattr(state, name)
            frames.append(current.loc[current.game_id.astype(str).eq(game)])
        return pd.concat(frames, ignore_index=True)

    return InjuryReportState(
        statuses=combine("statuses"),
        filings=combine("filings"),
        report_age=combine("report_age"),
        status_events=first.status_events.loc[
            ~first.status_events.game_id.astype(str).isin(ids)
        ].copy(),
        listed_pairs=combine("listed_pairs"),
    )


def build_closing_frame(live, horizon, ids):
    # Imported lazily to share the backfill's time and roster policy without
    # a module import cycle.
    from nba_ou.prediction.historical_backfill import history_limit, snapshot_rosters

    scheduled = live.scheduled_games.loc[
        live.scheduled_games.GAME_ID.astype(str).isin(ids)
    ].copy()
    games = live.line_history.games.loc[
        live.line_history.games.game_id.astype(str).isin(ids)
    ].copy()
    cutoffs = {
        str(game.game_id): pd.Timestamp(game.tipoff_utc) - pd.Timedelta(minutes=horizon)
        for game in games.itertuples()
    }
    limit = history_limit(live.game_date.date().isoformat(), pd.Series(cutoffs))
    state = reports_at_own_snapshots(cutoffs)
    not_filed = [
        str(game.GAME_ID)
        for game in scheduled.itertuples()
        if any(
            (str(game.GAME_ID), str(team)) not in state.covered
            for team in (game.HOME_TEAM_ID, game.VISITOR_TEAM_ID)
        )
    ]
    season = int(games.season_year.iloc[0])
    template = load_odds_sportsbook_from_db([f"{season}-{str(season + 1)[-2:]}"])
    if template is None:
        raise ValueError("Could not read sportsbook odds schema for closing replay")
    sportsbook = sportsbook_odds_at_snapshots(
        template, live.line_history.ticks, games, cutoffs
    )
    current_out = {
        key: players
        for key, players in report_out_overrides(state).items()
        if str(key[0]) in ids
    }
    data = {
        "scheduled_games": scheduled,
        "df_referees_scheduled": live.df_referees_scheduled,
        "injury_dict_scheduled": nested_status_dict(current_out),
        "injury_report_state": state,
        "injury_as_of_utc": max(cutoffs.values()),
        "roster_snapshots": snapshot_rosters(scheduled, live.roster_snapshots, cutoffs),
        "games_not_updated": not_filed,
        "df_odds_yahoo_scheduled": pd.DataFrame(),
        "df_odds_sportsbook_scheduled": sportsbook,
    }
    frame = create_df_to_predict(
        todays_prediction=True,
        scheduled_data=data,
        recent_limit_to_include=limit,
        strict_mode=-1,
        target_game_ids=sorted(ids),
        strict_history_cutoff=True,
    )
    frame = frame.loc[frame.GAME_ID.astype(str).isin(ids)].copy()
    if frame.GAME_ID.duplicated().any():
        raise ValueError("Closing replay returned duplicate games")
    frame["TIME_TO_MATCH_MIN"] = horizon
    frame["SNAPSHOT_TS_UTC"] = frame.GAME_ID.astype(str).map(cutoffs)
    frame["SNAPSHOT_PASSED"] = True
    tipoffs = games.set_index("game_id").tipoff_utc
    frame["TIPOFF_UTC"] = frame.GAME_ID.astype(str).map(tipoffs)
    frame = mask_referees_before_release(frame, visible_until=max(cutoffs.values()))
    return frame, limit
