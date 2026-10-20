"""Phase E: how many minutes each player is expected to play tonight.

The projection in ``game_projection`` currently splits a team's 240 minutes in
proportion to recent averages, which is the fallback the plan allows for a
failed go/no-go E. That rule cannot know that a bench forward is the one who
absorbs a starting forward's minutes, only that everyone absorbs a share, so it
is the obvious place a real model should help.

The structure is the plan's (section 6.2)::

    E[min] = P(play) * E[min | play]

``P(play)`` comes from the injury report through ``chance_out``, the same
probability every other injury feature in this repo uses.  ``E[min | play]`` is
a gradient-boosted regressor trained only on games the player actually played,
so the two factors stay separable and an Out listing cannot be learned as
"plays few minutes" instead of "does not play".

**Temporal contract.** Features read games on strictly earlier dates, so
same-date games are held back together. The model is refit monthly on an
expanding window, and a prediction for month M comes only from a model fitted
on games before month M. Nothing is persisted to the model registry: this is a
feature-pipeline component, refit on the fly.

**Inputs the plan asks for and this does not have.** Closing spread size, which
would need an odds join this module deliberately avoids, and days since a
trade, for which there is no transactions feed (see the plan's open questions).
Both are noted rather than approximated.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

#: Windows behind the recent-minutes features, in team games.
MINUTES_WINDOWS = (3, 5, 10)

#: A player at least this likely to sit is treated as out when measuring how
#: many minutes his teammates have to absorb.
TEAMMATE_OUT_THRESHOLD = 0.5

#: Team minutes in regulation.
TEAM_MINUTES = 240.0

FEATURE_COLUMNS = (
    "min_last_3",
    "min_last_5",
    "min_last_10",
    "min_share_last_5",
    "started_last",
    "starts_season",
    "games_since_absence",
    "p_out",
    "team_minutes_missing",
    "position_minutes_missing",
    "rest_days",
    "team_game_number",
)

_REQUIRED = {"GAME_ID", "TEAM_ID", "GAME_DATE", "PLAYER_ID", "MIN", "START_POSITION"}


def _mean(values: deque[float], window: int) -> float:
    recent = list(values)[-window:]
    return float(np.mean(recent)) if recent else np.nan


def build_minutes_frame(
    df_players: pd.DataFrame,
    p_out: dict[tuple[str, str, str], float] | None = None,
    recent_games: int = 5,
) -> pd.DataFrame:
    """One row per player-game, with pre-game features and the minutes target.

    The population is every player on the team's roster as of the game, which
    here means anyone who has played for it within the recent window. A player
    who is listed Out still gets a row: predicting his zero correctly is part of
    the job, and dropping him would train the model on survivors only.
    """
    if missing := _REQUIRED - set(df_players.columns):
        raise ValueError(f"Player history is missing {sorted(missing)}")
    p_out = p_out or {}

    frame = df_players[list(_REQUIRED)].copy()
    for column in ("GAME_ID", "TEAM_ID", "PLAYER_ID"):
        frame[column] = frame[column].astype(str)
    frame["GAME_ID"] = frame["GAME_ID"].str.zfill(10)
    frame["GAME_DATE"] = pd.to_datetime(
        frame["GAME_DATE"], errors="coerce"
    ).dt.normalize()
    frame["MIN"] = (
        pd.to_numeric(frame["MIN"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    frame["START_POSITION"] = frame["START_POSITION"].fillna("").astype(str).str.strip()
    frame = frame.dropna(subset=["GAME_DATE"]).drop_duplicates(
        ["GAME_ID", "TEAM_ID", "PLAYER_ID"]
    )

    minutes: dict[tuple[str, str], deque[float]] = defaultdict(
        lambda: deque(maxlen=max(MINUTES_WINDOWS))
    )
    last_five: dict[str, frozenset[str]] = {}
    starts: dict[tuple[str, str, object], int] = defaultdict(int)
    last_absence: dict[tuple[str, str], int] = {}
    games_played: dict[str, int] = defaultdict(int)
    last_seen: dict[tuple[str, str], int] = {}
    last_date: dict[str, pd.Timestamp] = {}
    position: dict[tuple[str, str], str] = {}
    rosters: dict[str, set[str]] = defaultdict(set)
    assignments: dict[str, str] = {}

    rows = []
    for date, day in frame.groupby("GAME_DATE", sort=True):
        for (game_id, team), game in day.groupby(["GAME_ID", "TEAM_ID"], sort=False):
            played = games_played[team]
            cutoff = played - recent_games
            roster = [
                player
                for player in rosters[team]
                if minutes[(team, player)]
                and last_seen.get((team, player), -1) > cutoff
            ]
            if not roster:
                continue
            season = game["GAME_DATE"].iloc[0].year - (
                game["GAME_DATE"].iloc[0].month < 8
            )
            base = {
                player: _mean(minutes[(team, player)], recent_games)
                for player in roster
            }
            probabilities = {
                player: p_out.get((game_id, team, player), 0.0) for player in roster
            }
            missing_total = sum(
                base[player]
                for player in roster
                if probabilities[player] >= TEAMMATE_OUT_THRESHOLD
            )
            missing_by_position: dict[str, float] = defaultdict(float)
            for player in roster:
                if probabilities[player] >= TEAMMATE_OUT_THRESHOLD:
                    missing_by_position[position.get((team, player), "")] += base[
                        player
                    ]
            team_recent = sum(base.values()) or np.nan
            actual = dict(zip(game["PLAYER_ID"], game["MIN"], strict=True))
            rest = (date - last_date[team]).days if team in last_date else np.nan
            for player in roster:
                own_position = position.get((team, player), "")
                rows.append(
                    {
                        "GAME_ID": game_id,
                        "TEAM_ID": team,
                        "PLAYER_ID": player,
                        "GAME_DATE": date,
                        "season": season,
                        "min_last_3": _mean(minutes[(team, player)], 3),
                        "min_last_5": _mean(minutes[(team, player)], 5),
                        "min_last_10": _mean(minutes[(team, player)], 10),
                        "min_share_last_5": (
                            base[player] / team_recent
                            if team_recent and not np.isnan(team_recent)
                            else np.nan
                        ),
                        "started_last": float(
                            player in last_five.get(team, frozenset())
                        ),
                        "starts_season": float(starts[(team, player, season)]),
                        # 0 means he missed the immediately preceding game;
                        # a player who has never missed one keeps growing away
                        # from it, which is the ordering the model wants.
                        "games_since_absence": float(
                            played - last_absence.get((team, player), -1)
                        ),
                        "p_out": probabilities[player],
                        # His own expected absence is not minutes for him to
                        # absorb, so it is excluded from both totals.
                        "team_minutes_missing": missing_total
                        - (
                            base[player]
                            if probabilities[player] >= TEAMMATE_OUT_THRESHOLD
                            else 0.0
                        ),
                        "position_minutes_missing": missing_by_position[own_position]
                        - (
                            base[player]
                            if probabilities[player] >= TEAMMATE_OUT_THRESHOLD
                            else 0.0
                        ),
                        "rest_days": rest,
                        "team_game_number": float(played),
                        "baseline_minutes": base[player],
                        "MIN": float(actual.get(player, 0.0)),
                    }
                )
        # Tonight's box scores become history only after every game on this
        # date has produced its rows.
        for (_game_id, team), game in day.groupby(["GAME_ID", "TEAM_ID"], sort=False):
            games_played[team] += 1
            last_date[team] = date
            season = game["GAME_DATE"].iloc[0].year - (
                game["GAME_DATE"].iloc[0].month < 8
            )
            starters = frozenset(game.loc[game["START_POSITION"].ne(""), "PLAYER_ID"])
            if len(starters) == 5:
                last_five[team] = starters
            for row in game.itertuples(index=False):
                player = row.PLAYER_ID
                previous = assignments.get(player)
                if previous is not None and previous != team:
                    rosters[previous].discard(player)
                assignments[player] = team
                rosters[team].add(player)
                minutes[(team, player)].append(float(row.MIN))
                last_seen[(team, player)] = games_played[team]
                if row.START_POSITION:
                    position[(team, player)] = row.START_POSITION
                    starts[(team, player, season)] += 1
                elif (team, player) not in position:
                    position[(team, player)] = ""
                if row.MIN <= 0:
                    last_absence[(team, player)] = games_played[team]
    return pd.DataFrame(rows)


def apply_team_constraint(
    frame: pd.DataFrame,
    column: str = "predicted_minutes",
    team_minutes: float = TEAM_MINUTES,
) -> pd.Series:
    """Rescale each team-game's predictions so the team sums to its minutes.

    A regressor fitted per player has no idea that a basketball team plays
    exactly 240 minutes, so its raw predictions do not add up. Rescaling is what
    makes one player's absence actually arrive somewhere else.
    """
    values = frame[column].clip(lower=0.0)
    totals = values.groupby([frame["GAME_ID"], frame["TEAM_ID"]]).transform("sum")
    scaled = np.where(totals > 0, values * team_minutes / totals, np.nan)
    return pd.Series(scaled, index=frame.index)


def walk_forward_minutes(
    frame: pd.DataFrame,
    *,
    min_train_rows: int = 5000,
    seed: int = 0,
    params: dict | None = None,
) -> pd.DataFrame:
    """Predict minutes month by month, each month fitted only on earlier games.

    Returns ``frame`` with two columns added. ``predicted_minutes`` is
    ``(1 - p_out)`` times the regressor's estimate, rescaled so each team sums
    to 240; that is the expected-minutes number. ``minutes_given_play`` is the
    regressor's estimate on its own, for a consumer that applies availability
    itself -- ``game_projection``'s scenarios do, and feeding them the first
    column would apply ``p_out`` twice.
    """
    from xgboost import XGBRegressor

    if frame.empty:
        return frame.assign(predicted_minutes=pd.Series(dtype=float))
    work = frame.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).copy()
    work["_month"] = work["GAME_DATE"].dt.to_period("M")
    settings = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": 4,
        "tree_method": "hist",
    } | (params or {})
    features = list(FEATURE_COLUMNS)
    predictions = pd.Series(np.nan, index=work.index, dtype=float)
    given_play_out = pd.Series(np.nan, index=work.index, dtype=float)
    for month in sorted(work["_month"].unique()):
        history = work.loc[work["_month"] < month]
        # Only games he actually played teach "minutes given that he plays";
        # the zeros are the job of P(play), not of this regressor.
        trainable = history.loc[history["MIN"] > 0]
        if len(trainable) < min_train_rows:
            continue
        model = XGBRegressor(**settings)
        model.fit(trainable[features], trainable["MIN"])
        target = work.loc[work["_month"].eq(month)]
        given_play = model.predict(target[features])
        given_play_out.loc[target.index] = given_play
        predictions.loc[target.index] = (1.0 - target["p_out"].to_numpy()) * given_play
    work["predicted_minutes"] = predictions
    # The unconditional factor, before availability is applied. A consumer that
    # models availability itself -- game_projection's scenarios do -- must use
    # this one, or p_out is counted twice.
    work["minutes_given_play"] = given_play_out
    scaled = apply_team_constraint(work.dropna(subset=["predicted_minutes"]))
    work["predicted_minutes"] = scaled.reindex(work.index)
    return work.drop(columns="_month")
