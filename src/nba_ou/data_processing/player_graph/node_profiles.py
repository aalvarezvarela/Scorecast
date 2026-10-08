"""As-of node profiles: what a player looked like before date D (v0).

These are the node features of every graph and the initial inputs of the
player embeddings (plan phases 2 and 6). Everything is computed from box scores
of games **strictly before** the cutoff, through the ``as_of`` view; preseason
and All-Star games are left out.

Per player:

* **Rates per 36 minutes** (``RATE_STATS``), decayed and shrunk toward the
  league: ``(sum w * stat + m0 * league_rate) / (sum w * minutes + m0) * 36``
  with ``w = 0.5 ** (age_days / half_life_days)`` and ``m0`` prior minutes.
* **Efficiency and usage**: true shooting ``PTS / (2 * (FGA + 0.44 FTA))``,
  three-point percentage and ``USG_PCT``, shrunk the same way on their own
  denominators (shooting attempts, three-point attempts, minutes).
* **Role**: mean minutes over the last ``RECENT_GAMES`` games played (2_6's
  window length; unlike 2_6's node weights, coach's-decision zeros are not
  counted here), share of starts in the window, soft G/F/C position
  (``positions.positions_as_of``).
* **Sample size and confidence**: raw minutes and games in the window, decayed
  minutes (effective sample), ``prior_weight = m0 / (decayed minutes + m0)``,
  days since the last appearance, games in the data so far (experience),
  ``has_box_history``.

A player with no box scores before the date (a debut, or anyone in the first
games of 2016-17, where 2_7's data starts) gets league rates with
``prior_weight`` 1, so a graph can always be built and the encoder can tell the
profile is only a prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nba_ou.data_processing.lineups.availability import RECENT_GAMES

from .as_of import AsOfView
from .positions import PROBABILITY_COLUMNS, positions_as_of

RATE_STATS = (
    "PTS",
    "FGA",
    "FG3A",
    "FTA",
    "OREB",
    "DREB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PF",
)
_COUNTED_GAME_PREFIXES = ("002", "004", "005", "006")

PROFILE_COLUMNS = (
    *(f"{stat.lower()}_per36" for stat in RATE_STATS),
    "ts_pct",
    "fg3_pct",
    "usage",
    "recent_minutes",
    "start_share",
    *PROBABILITY_COLUMNS,
    "minutes_window",
    "games_window",
    "minutes_decayed",
    "prior_weight",
    "days_since_last_game",
    "games_in_data",
    "has_box_history",
)


@dataclass(frozen=True)
class ProfileParams:
    window_days: int = 2 * 365
    half_life_days: float = 180.0
    prior_minutes: float = 200.0
    prior_attempts: float = 100.0
    prior_three_attempts: float = 50.0


DEFAULT_PARAMS = ProfileParams()


def counted_box_scores(box: pd.DataFrame) -> pd.DataFrame:
    """Box-score rows of games a profile counts (no preseason or All-Star),
    with ids as text and stat columns numeric."""
    prefix = box["GAME_ID"].astype(str).str.zfill(10).str[:3]
    box = box.loc[prefix.isin(_COUNTED_GAME_PREFIXES)].copy()
    box["PLAYER_ID"] = box["PLAYER_ID"].astype(str)
    numeric = ["MIN", *RATE_STATS, "FG3M", "USG_PCT"]
    for column in numeric:
        if column in box.columns:
            box[column] = pd.to_numeric(box[column], errors="coerce").fillna(0.0)
        else:
            box[column] = 0.0
    return box


def node_profiles(
    view: AsOfView,
    players: pd.Index | list[str] | None = None,
    params: ProfileParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Profiles of ``players`` (default: everyone in the window) as of the cutoff.

    Indexed by ``player_id`` with ``PROFILE_COLUMNS``.
    """
    since = view.cutoff - pd.Timedelta(days=params.window_days)
    box = counted_box_scores(view.box_scores(since=since))
    played = box.loc[box["MIN"] > 0]
    wanted = (
        pd.Index(players, dtype=str)
        if players is not None
        else pd.Index(played["PLAYER_ID"].unique(), dtype=str)
    )

    age = (view.cutoff - played["GAME_DATE"]).dt.days.to_numpy(float)
    weight = 0.5 ** (age / params.half_life_days)
    weighted = played[["MIN", *RATE_STATS, "FG3M"]].mul(weight, axis=0)
    weighted["USG_MIN"] = (
        played["USG_PCT"].to_numpy() * played["MIN"].to_numpy() * weight
    )
    sums = weighted.groupby(played["PLAYER_ID"]).sum().reindex(wanted).fillna(0.0)

    if len(played):
        league_rates = weighted[list(RATE_STATS)].sum() / weighted["MIN"].sum()
        attempts = weighted["FGA"] + 0.44 * weighted["FTA"]
        league_ts = weighted["PTS"].sum() / (2 * attempts.sum())
        league_fg3 = weighted["FG3M"].sum() / max(weighted["FG3A"].sum(), 1e-9)
        league_usage = weighted["USG_MIN"].sum() / weighted["MIN"].sum()
    else:  # nothing before the cutoff: generic league values
        league_rates = pd.Series(0.0, index=list(RATE_STATS))
        league_ts, league_fg3, league_usage = 0.56, 0.35, 0.20

    m0 = params.prior_minutes
    out = pd.DataFrame(index=wanted)
    out.index.name = "player_id"
    for stat in RATE_STATS:
        out[f"{stat.lower()}_per36"] = (
            (sums[stat] + m0 * league_rates[stat]) / (sums["MIN"] + m0) * 36.0
        )
    shooting = sums["FGA"] + 0.44 * sums["FTA"]
    out["ts_pct"] = (sums["PTS"] + 2 * params.prior_attempts * league_ts) / (
        2 * (shooting + params.prior_attempts)
    )
    out["fg3_pct"] = (sums["FG3M"] + params.prior_three_attempts * league_fg3) / (
        sums["FG3A"] + params.prior_three_attempts
    )
    out["usage"] = (sums["USG_MIN"] + m0 * league_usage) / (sums["MIN"] + m0)

    # Role: recent minutes per appearance (2_6 roster rule), share of starts.
    ordered = played.sort_values(["GAME_DATE", "GAME_ID"], kind="mergesort")
    recent = ordered.groupby("PLAYER_ID")["MIN"].apply(
        lambda minutes: minutes.tail(RECENT_GAMES).mean()
    )
    out["recent_minutes"] = recent.reindex(wanted).fillna(0.0)
    starts = (
        played["START_POSITION"].fillna("").astype(str).str.strip().ne("")
        if "START_POSITION" in played.columns
        else pd.Series(False, index=played.index)
    )
    out["start_share"] = (
        starts.groupby(played["PLAYER_ID"]).mean().reindex(wanted).fillna(0.0)
    )
    positions = positions_as_of(view)
    league_position = (
        positions[list(PROBABILITY_COLUMNS)].mean()
        if len(positions)
        else pd.Series(1 / 3, index=list(PROBABILITY_COLUMNS))
    )
    for column in PROBABILITY_COLUMNS:
        out[column] = positions[column].reindex(wanted).fillna(league_position[column])

    # Sample size and confidence.
    out["minutes_window"] = (
        played.groupby("PLAYER_ID")["MIN"].sum().reindex(wanted).fillna(0.0)
    )
    out["games_window"] = played.groupby("PLAYER_ID").size().reindex(wanted).fillna(0)
    out["minutes_decayed"] = sums["MIN"]
    out["prior_weight"] = m0 / (sums["MIN"] + m0)
    last = played.groupby("PLAYER_ID")["GAME_DATE"].max().reindex(wanted)
    out["days_since_last_game"] = (view.cutoff - last).dt.days.astype(float)
    everything = counted_box_scores(view.box_scores())
    out["games_in_data"] = (
        everything.loc[everything["MIN"] > 0]
        .groupby("PLAYER_ID")
        .size()
        .reindex(wanted)
        .fillna(0)
        .astype(int)
    )
    out["has_box_history"] = out["games_window"] > 0
    out["games_window"] = out["games_window"].astype(int)
    return out[list(PROFILE_COLUMNS)]
