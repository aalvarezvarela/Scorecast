"""Per-player features for players listed Questionable, Probable or Doubtful.

Every listed player gets ``FORM_{MIN,PTS,PACE_PER40}``: their own pre-game form,
so the model knows how big the player at risk is, the way
``TOP1_INJURED_PLAYER_PTS`` does for the out set. One ranking per status
(:data:`RANK_STAT`) picks who fills each slot, and all of that slot's columns
then describe that one player -- unlike ``TOP{i}_INJURED_PLAYER_<stat>``, which
re-ranks per statistic and can name a different player in each.

For Questionable and Probable, two more things are estimated from earlier games
only:

* ``P_PLAY`` -- how often this player has played when given that status, shrunk
  toward the league rate. The player's history counts every game they held the
  status at ANY pre-tip moment (for Questionable: 3-8x the samples of
  last-report listings, and a better predictor, log-loss 0.608 vs 0.623 on
  2023-25). The prior is the league rate for players still holding the status on
  the last report, over the current and previous season.
* ``EFFECT_{MIN,PTS,PACE_PER40}`` -- the change versus the player's own form
  when they play under that status, net of the drift unlisted players show.
  Shrunk toward the league effect for the same status and minutes role; a
  status with too little history borrows the effect pooled over both.

These are status features, not availability-group features, and the two families
are deliberately separate. ``report_state`` rule 2 puts Questionable in its own
availability group -- not subtracted from the team, not counted as available --
and gives that group the same top-N treatment the injured and available groups
get. What is here answers a different question about the same player: given
that the report says Questionable, how often does he play, and what does he do
when he does. A Questionable player therefore carries both.

Doubtful gets the form columns but no history columns. It is in the out set and
plays ~2% of the time at the last report -- 3 players since 2021 -- so a play
probability or an effect estimated for it would be noise, while the form of the
best Doubtful player is simply a fact about who is missing.

Every quantity for a game on date D uses only events dated strictly before D.
Same-day games are excluded because their outcomes are unknown at an earlier tip
that day. Measurements behind the constants: ``docs/injury_status_tiers_plan.md``
section 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .report_state import UNCOUNTED_REASON_CATEGORIES, InjuryReportState

#: Statuses whose listings get a play probability, an effect and a league rate.
HISTORY_STATUSES: tuple[str, ...] = ("questionable", "probable")

#: Every status with per-player slots. Doubtful is here for its form only.
LISTED_STATUSES: tuple[str, ...] = (*HISTORY_STATUSES, "doubtful")

#: Per-player columns for the N best players of each status. Questionable is
#: common enough for two; a second Probable or Doubtful rotation player on one
#: team is rare.
DEFAULT_TOP_N: dict[str, int] = {"questionable": 2, "probable": 1, "doubtful": 1}

#: Which form statistic ranks the listed players of each status. Minutes for
#: the two out-set statuses: the slot is there to say how much of the rotation
#: is at risk, and minutes measure that directly, where points confound role
#: with scoring rate. It only decides anything when a team lists more players
#: of one status than that status has slots -- 17.8% of Doubtful team-games and
#: 8.5% of Questionable ones -- and there the two keys pick a different player
#: about a fifth of the time, the minutes pick averaging +4 form minutes.
RANK_STAT: dict[str, str] = {
    "questionable": "MIN",
    "doubtful": "MIN",
    "probable": "PTS",
}

EFFECT_STATS: tuple[str, ...] = ("MIN", "PTS", "PACE_PER40")

#: Form is the same EWMA the TOP-N player features use.
FORM_HALFLIFE_GAMES = 10

#: Pace over a handful of minutes is noise.
MIN_MINUTES_FOR_PACE = 8.0

#: Unlisted players below this form are left out of the drift baseline, which
#: would otherwise be dominated by garbage-time minutes.
MIN_FORM_MINUTES_FOR_BASELINE = 10.0

ROLE_BINS = (-np.inf, 10.0, 20.0, 28.0, 34.0, np.inf)
ROLE_LABELS = ("lt10", "10-20", "20-28", "28-34", "34+")

#: Prior weight of the league rate in a player's play probability. k = 2-5 were
#: indistinguishable on 2023-25 Questionable log-loss (0.6076-0.6080).
PLAYER_PLAY_PRIOR_K = 3.0

#: League window below this many last-report listings falls back to all history.
MIN_LEAGUE_LISTINGS = 200

#: A prior level (status-role, status, pooled-role) needs this many events,
#: otherwise the next, broader level is used.
MIN_PRIOR_EVENTS = 30

#: Shrinkage constants when too little earlier history exists to fit them.
#: Measured on Questionable 2019-2025: minutes ~32, points ~62, pace ~123.
DEFAULT_EFFECT_K: dict[str, float] = {"MIN": 30.0, "PTS": 60.0, "PACE_PER40": 120.0}
EFFECT_K_BOUNDS = (5.0, 500.0)
MIN_PLAYERS_FOR_K_FIT = 30
MIN_EVENTS_PER_PLAYER_FOR_K_FIT = 5

#: The player's own pre-game form, for every status with slots.
FORM_NAMES: tuple[str, ...] = tuple(f"FORM_{s}" for s in EFFECT_STATS)

#: History estimates, only for :data:`HISTORY_STATUSES`.
HISTORY_NAMES: tuple[str, ...] = (
    "P_PLAY",
    "N_HISTORY",
    *(f"EFFECT_{s}" for s in EFFECT_STATS),
)

PER_PLAYER_NAMES: tuple[str, ...] = (*HISTORY_NAMES, *FORM_NAMES)

#: What an empty slot on a covered team-game means. A slot with no player
#: contributes exactly what a player certain to play unchanged would, and no
#: form at all: the status counter is what tells "no such player" apart from
#: "a player whose form is zero".
NEUTRAL_SLOT_VALUES: dict[str, float] = {
    "P_PLAY": 1.0,
    "N_HISTORY": 0.0,
    **{f"EFFECT_{s}": 0.0 for s in EFFECT_STATS},
    **{name: 0.0 for name in FORM_NAMES},
}


def per_player_names(status: str) -> tuple[str, ...]:
    """Slot columns for ``status``: form always, history estimates where they
    are not noise."""
    if status in HISTORY_STATUSES:
        return PER_PLAYER_NAMES
    return FORM_NAMES


def top_col(status: str, i: int, name: str) -> str:
    return f"TOP{i}_{status.upper()}_{name}_BEFORE"


def sum_exp_players_col(status: str) -> str:
    return f"SUM_{status.upper()}_EXP_PLAYERS_BEFORE"


def sum_exp_col(status: str, stat: str) -> str:
    return f"SUM_{status.upper()}_EXP_{stat}_BEFORE"


def mean_p_play_col(status: str) -> str:
    return f"MEAN_{status.upper()}_P_PLAY_BEFORE"


def league_p_play_col(status: str) -> str:
    return f"LEAGUE_{status.upper()}_P_PLAY_BEFORE"


def status_feature_columns(top_n: dict[str, int] | None = None) -> list[str]:
    top_n = DEFAULT_TOP_N if top_n is None else top_n
    columns: list[str] = []
    for status in LISTED_STATUSES:
        columns += [
            top_col(status, i, name)
            for i in range(1, top_n.get(status, 0) + 1)
            for name in per_player_names(status)
        ]
        if status in HISTORY_STATUSES:
            columns += [
                sum_exp_players_col(status),
                sum_exp_col(status, "MIN"),
                sum_exp_col(status, "PTS"),
                mean_p_play_col(status),
                league_p_play_col(status),
            ]
    return columns


# --------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------


def load_player_box_history(season_years: list[int] | None = None) -> pd.DataFrame:
    """Box-score rows for regular season, play-in and playoffs, from Supabase.

    ``None`` loads 2017-18 onward: one season before the first report, so the
    previous-season form fallback exists from the first covered game.
    """
    from nba_ou.data_processing.players.attach_player_features import (
        _parse_minutes_series,
    )
    from nba_ou.postgre_db.config.db_config import connect_nba_db

    query = """
        SELECT p.game_id, p.team_id::text AS team_id, p.player_id::text AS player_id,
               p.season_year, g.game_date, p.min, p.pts, p.pace_per40
        FROM nba_players.nba_players p
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games.nba_games) g
          ON g.game_id = p.game_id
        WHERE left(p.game_id, 3) IN ('002', '004', '005')
          AND (%(season_years)s::int[] IS NULL OR p.season_year = ANY(%(season_years)s))
          AND p.season_year >= 2017
    """
    with connect_nba_db("supabase") as conn:
        box = pd.read_sql_query(
            query,
            conn,
            params={
                "season_years": None if season_years is None else list(season_years)
            },
        )
    box["MIN"] = _parse_minutes_series(box.pop("min"))
    box["PTS"] = pd.to_numeric(box.pop("pts"), errors="coerce")
    box["PACE_PER40"] = pd.to_numeric(box.pop("pace_per40"), errors="coerce")
    return prepare_box_history(box)


def prepare_box_history(box: pd.DataFrame) -> pd.DataFrame:
    """Normalise a box frame: string ids, datetime dates, one row per game-player."""
    out = box.copy()
    for col in ("game_id", "team_id", "player_id"):
        out[col] = out[col].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["season_year"] = out["season_year"].astype(int)
    for col in ("MIN", *EFFECT_STATS):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(["game_id", "player_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------------


def build_player_form(box: pd.DataFrame) -> pd.DataFrame:
    """EWMA of each stat through each played game, per player-season.

    One row per played game: ``form_<STAT>`` includes that game. Read it for a
    later game with :func:`form_before`, which takes the last row strictly
    before the date.
    """
    played = box.loc[box["MIN"].fillna(0) > 0].sort_values(
        ["player_id", "season_year", "game_date", "game_id"], kind="mergesort"
    )
    played = played.copy()
    groups = [played["player_id"], played["season_year"]]
    for stat in EFFECT_STATS:
        values = played[stat]
        if stat == "PACE_PER40":
            values = values.where(played["MIN"] >= MIN_MINUTES_FOR_PACE)
        played[f"form_{stat}"] = values.groupby(groups).transform(
            lambda s: s.ewm(
                halflife=FORM_HALFLIFE_GAMES, adjust=False, ignore_na=True
            ).mean()
        )
    return played[
        ["player_id", "season_year", "game_date", *(f"form_{s}" for s in EFFECT_STATS)]
    ].reset_index(drop=True)


def _previous_regular_season_means(box: pd.DataFrame) -> pd.DataFrame:
    regular = box.loc[
        (box["MIN"].fillna(0) > 0) & box["game_id"].str[:3].isin(["002", "005"])
    ].copy()
    regular["PACE_PER40"] = regular["PACE_PER40"].where(
        regular["MIN"] >= MIN_MINUTES_FOR_PACE
    )
    means = regular.groupby(["player_id", "season_year"])[list(EFFECT_STATS)].mean()
    means.columns = [f"prev_{s}" for s in EFFECT_STATS]
    means = means.reset_index()
    means["season_year"] = means["season_year"] + 1
    return means


def form_before(
    targets: pd.DataFrame, form: pd.DataFrame, box: pd.DataFrame
) -> pd.DataFrame:
    """``form_<STAT>`` for each target (player_id, season_year, game_date).

    This season to date, strictly before the date; otherwise the player's
    previous regular-season mean; otherwise NaN. Aligned to ``targets``' index.
    """
    cols = [f"form_{s}" for s in EFFECT_STATS]
    if targets.empty:
        return pd.DataFrame(columns=cols, index=targets.index, dtype=float)
    left = targets[["player_id", "season_year", "game_date"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("game_date", kind="mergesort")
    merged = pd.merge_asof(
        left,
        form.sort_values("game_date", kind="mergesort"),
        on="game_date",
        by=["player_id", "season_year"],
        allow_exact_matches=False,
    )
    merged = merged.merge(
        _previous_regular_season_means(box),
        on=["player_id", "season_year"],
        how="left",
    )
    for stat in EFFECT_STATS:
        merged[f"form_{stat}"] = merged[f"form_{stat}"].fillna(merged[f"prev_{stat}"])
    merged = merged.sort_values("_row")
    return pd.DataFrame(merged[cols].to_numpy(), columns=cols, index=targets.index)


def player_form_slots(
    targets: pd.DataFrame, form: pd.DataFrame, box: pd.DataFrame
) -> pd.DataFrame:
    """:func:`form_before` under the published ``FORM_<STAT>`` names.

    A player with no form at all -- no game this season and none in the previous
    regular season -- reads 0, the same as an empty slot: both mean "no minutes
    to lose here".
    """
    values = form_before(targets, form, box).fillna(0.0)
    values.columns = list(FORM_NAMES)
    return values


# --------------------------------------------------------------------------------
# Expanding aggregates
# --------------------------------------------------------------------------------


def _prior_sums(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    by: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """Sum and non-null count of ``value_cols`` over history strictly before each
    target's ``game_date``, per ``by`` group. Aligned to ``targets``' index;
    columns ``<col>_sum`` and ``<col>_n`` (0 where no history).
    """
    out_cols = [f"{c}_{kind}" for c in value_cols for kind in ("sum", "n")]
    if targets.empty:
        return pd.DataFrame(columns=out_cols, index=targets.index, dtype=float)
    if history.empty:
        return pd.DataFrame(0.0, columns=out_cols, index=targets.index)
    hist = history[by + ["game_date"] + value_cols].copy()
    for c in value_cols:
        hist[f"{c}_n"] = hist[c].notna().astype(float)
        hist[f"{c}_sum"] = hist[c].fillna(0.0)
    daily = hist.groupby(by + ["game_date"])[out_cols].sum().reset_index()
    daily = daily.sort_values(by + ["game_date"], kind="mergesort")
    if by:
        daily[out_cols] = daily.groupby(by)[out_cols].cumsum()
    else:
        daily[out_cols] = daily[out_cols].cumsum()
    left = targets[by + ["game_date"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("game_date", kind="mergesort")
    daily["game_date"] = pd.to_datetime(daily["game_date"])
    merged = pd.merge_asof(
        left,
        daily.sort_values("game_date", kind="mergesort"),
        on="game_date",
        by=by or None,
        allow_exact_matches=False,
    ).sort_values("_row")
    return pd.DataFrame(
        merged[out_cols].fillna(0.0).to_numpy(dtype=float),
        columns=out_cols,
        index=targets.index,
    )


def _role(form_minutes: pd.Series) -> pd.Series:
    return pd.cut(
        form_minutes.fillna(0.0),
        bins=list(ROLE_BINS),
        labels=list(ROLE_LABELS),
        right=False,
    ).astype(str)


def _mean(sums: pd.DataFrame, col: str) -> pd.Series:
    n = sums[f"{col}_n"]
    return sums[f"{col}_sum"] / n.where(n > 0)


# --------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------


def build_status_events(
    state: InjuryReportState, box: pd.DataFrame, form: pd.DataFrame
) -> pd.DataFrame:
    """Every Questionable/Probable listing with its outcome and adjusted changes.

    Columns: ids, ``status``, ``game_date``, ``season_year``,
    ``at_last_report``, ``played``, ``role``, ``form_<STAT>`` and, for played
    games, ``d_<STAT>`` = stat − form − unlisted drift before that date.
    """
    events = state.status_events.copy()
    if events.empty:
        return events.assign(
            played=pd.Series(dtype=float),
            role=pd.Series(dtype=str),
            **{f"d_{s}": pd.Series(dtype=float) for s in EFFECT_STATS},
        )
    events["season_year"] = events["season_year"].astype(int)
    events["at_last_report"] = events["at_last_report"].astype(bool)
    outcome = box[["game_id", "player_id", *EFFECT_STATS]]
    events = events.merge(outcome, on=["game_id", "player_id"], how="left")
    events["played"] = (events["MIN"].fillna(0) > 0).astype(float)
    events[[f"form_{s}" for s in EFFECT_STATS]] = form_before(events, form, box)
    events["role"] = _role(events["form_MIN"])

    drift = unlisted_drift(state, box, form, events)
    for stat in EFFECT_STATS:
        value = events[stat]
        if stat == "PACE_PER40":
            value = value.where(events["MIN"] >= MIN_MINUTES_FOR_PACE)
        raw = (value - events[f"form_{stat}"]).where(events["played"] > 0)
        events[f"d_{stat}"] = raw - drift[f"drift_{stat}"]
    return events.drop(columns=list(EFFECT_STATS))


def unlisted_drift(
    state: InjuryReportState,
    box: pd.DataFrame,
    form: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Mean (stat − form) of unlisted players who played, before each target date.

    Regression to the mean exists for everyone; a status effect is what remains
    after it. Only games a report covered count, since elsewhere "unlisted"
    cannot be told apart from "no report". 0 where no history.
    """
    cols = [f"drift_{s}" for s in EFFECT_STATS]
    covered_games = set(state.filings["game_id"])
    listed = set(
        zip(state.listed_pairs["game_id"], state.listed_pairs["player_id"], strict=True)
    )
    rows = box.loc[
        (box["MIN"].fillna(0) > 0) & box["game_id"].isin(covered_games)
    ].copy()
    if rows.empty or targets.empty:
        return pd.DataFrame(0.0, columns=cols, index=targets.index)
    keep = [
        (g, p) not in listed
        for g, p in zip(rows["game_id"], rows["player_id"], strict=True)
    ]
    rows = rows.loc[keep]
    rows[[f"form_{s}" for s in EFFECT_STATS]] = form_before(rows, form, box)
    rows = rows.loc[rows["form_MIN"] >= MIN_FORM_MINUTES_FOR_BASELINE]
    for stat in EFFECT_STATS:
        value = rows[stat]
        if stat == "PACE_PER40":
            value = value.where(rows["MIN"] >= MIN_MINUTES_FOR_PACE)
        rows[f"raw_{stat}"] = value - rows[f"form_{stat}"]
    sums = _prior_sums(rows, targets, [], [f"raw_{s}" for s in EFFECT_STATS])
    return pd.DataFrame(
        {f"drift_{s}": _mean(sums, f"raw_{s}").fillna(0.0) for s in EFFECT_STATS},
        index=targets.index,
    )


# --------------------------------------------------------------------------------
# Play probability and effects for target listings
# --------------------------------------------------------------------------------


def league_play_rate(events: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """Played share of last-report listings in ``events``, current + previous
    season, before each target date. Falls back to all earlier seasons when the
    window has fewer than ``MIN_LEAGUE_LISTINGS``; NaN with no history at all.

    ``events`` must already be restricted to one status.
    """
    rate = pd.Series(np.nan, index=targets.index, dtype=float)
    if targets.empty or events.empty:
        return rate
    last = events.loc[events["at_last_report"]]
    for season in targets["season_year"].astype(int).unique():
        mask = targets["season_year"].astype(int) == season
        window = last.loc[last["season_year"].isin([season - 1, season])]
        sums = _prior_sums(window, targets.loc[mask], [], ["played"])
        rate.loc[mask] = _mean(sums, "played").where(
            sums["played_n"] >= MIN_LEAGUE_LISTINGS
        )
    return rate.fillna(_mean(_prior_sums(last, targets, [], ["played"]), "played"))


def fit_effect_k(events: pd.DataFrame, stat: str, before_season: int) -> float | None:
    """Shrinkage constant sigma²/tau² for one stat, from seasons < ``before_season``.

    tau² is the between-player variance of the effect net of sampling noise;
    sigma² is the pooled within-player variance. A tau² of zero means players
    are indistinguishable, and the upper bound applies. ``None`` when too few
    players have enough events to fit anything.
    """
    col = f"d_{stat}"
    hist = events.loc[(events["season_year"] < before_season) & events[col].notna()]
    stats = hist.groupby("player_id")[col].agg(["mean", "var", "size"])
    stats = stats.loc[stats["size"] >= MIN_EVENTS_PER_PLAYER_FOR_K_FIT]
    if len(stats) < MIN_PLAYERS_FOR_K_FIT:
        return None
    dof = stats["size"] - 1
    sigma2 = float((stats["var"] * dof).sum() / dof.sum())
    tau2 = float(stats["mean"].var(ddof=1) - (stats["var"] / stats["size"]).mean())
    low, high = EFFECT_K_BOUNDS
    if not np.isfinite(sigma2) or tau2 <= 0:
        return high
    return float(np.clip(sigma2 / tau2, low, high))


def _effect_k(
    status_events: pd.DataFrame, pooled_events: pd.DataFrame, stat: str, season: int
) -> float:
    for source in (status_events, pooled_events):
        k = fit_effect_k(source, stat, season)
        if k is not None:
            return k
    return DEFAULT_EFFECT_K[stat]


def status_player_estimates(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    form: pd.DataFrame,
    box: pd.DataFrame,
    status: str,
) -> pd.DataFrame:
    """``FORM_<STAT>``, ``P_PLAY``, ``N_HISTORY`` and ``EFFECT_<STAT>`` per target.

    ``targets``: player_id, season_year, game_date (any index). ``events`` holds
    every listed status; the estimates use the ``status`` rows, and the effect
    prior borrows from all of them when the status alone is too thin.
    """
    out = pd.DataFrame(index=targets.index)
    if targets.empty:
        for col in PER_PLAYER_NAMES:
            out[col] = pd.Series(dtype=float)
        return out

    targets = targets.copy()
    targets["season_year"] = targets["season_year"].astype(int)
    out[list(FORM_NAMES)] = player_form_slots(targets, form, box)
    status_events = events.loc[events["status"] == status]

    # --- play probability ---------------------------------------------------
    league = league_play_rate(status_events, targets)
    player = _prior_sums(status_events, targets, ["player_id"], ["played"])
    prior = league.fillna(0.5)
    out["N_HISTORY"] = player["played_n"]
    out["P_PLAY"] = (player["played_sum"] + PLAYER_PLAY_PRIOR_K * prior) / (
        player["played_n"] + PLAYER_PLAY_PRIOR_K
    )

    # --- effect when playing --------------------------------------------------
    d_cols = [f"d_{s}" for s in EFFECT_STATS]
    status_played = status_events.loc[status_events["played"] > 0]
    pooled_played = events.loc[events["played"] > 0]
    targets["role"] = _role(out["FORM_MIN"])
    #: Narrowest first: this status and role, this status, all listed statuses
    #: and role, all listed statuses.
    levels = (
        _prior_sums(status_played, targets, ["role"], d_cols),
        _prior_sums(status_played, targets, [], d_cols),
        _prior_sums(pooled_played, targets, ["role"], d_cols),
        _prior_sums(pooled_played, targets, [], d_cols),
    )
    player_sums = _prior_sums(status_played, targets, ["player_id"], d_cols)
    seasons = targets["season_year"].unique()
    for stat in EFFECT_STATS:
        col = f"d_{stat}"
        prior_effect = pd.Series(np.nan, index=targets.index)
        for depth, sums in enumerate(levels):
            enough = sums[f"{col}_n"] >= (
                MIN_PRIOR_EVENTS if depth < len(levels) - 1 else 1
            )
            prior_effect = prior_effect.fillna(_mean(sums, col).where(enough))
        prior_effect = prior_effect.fillna(0.0)
        k_by_season = {
            s: _effect_k(status_events, events, stat, int(s)) for s in seasons
        }
        k = targets["season_year"].map(k_by_season)
        out[f"EFFECT_{stat}"] = (player_sums[f"{col}_sum"] + k * prior_effect) / (
            player_sums[f"{col}_n"] + k
        )
    return out


def add_status_features(
    team_games: pd.DataFrame,
    state: InjuryReportState,
    box: pd.DataFrame,
    *,
    top_n: dict[str, int] | None = None,
    game_id_col: str = "GAME_ID",
    team_id_col: str = "TEAM_ID",
    game_date_col: str = "GAME_DATE",
    season_year_col: str = "SEASON_YEAR",
) -> pd.DataFrame:
    """Per-status columns for each team-game row, aligned to its index.

    * Uncovered team-game: every column NaN except the league rates.
    * Covered, with an empty slot (fewer players of that status than its top
      N): the slot takes :data:`NEUTRAL_SLOT_VALUES` -- ``P_PLAY = 1``,
      ``N_HISTORY = 0``, every effect and every form 0 -- and ``MEAN_P_PLAY``
      reads 1 when nobody holds the status. An empty slot contributes exactly
      what a player certain to play unchanged would, which is also what the sums
      say.

    Leaving empty slots NaN is not an option: ~90% of team-games have no
    Questionable player (Probable and Doubtful are rarer still), and the training
    cleaner drops any column above its NaN threshold (5% by default).
    """
    top_n = DEFAULT_TOP_N if top_n is None else top_n
    box = prepare_box_history(box)
    form = build_player_form(box)
    events = build_status_events(state, box, form)

    rows = pd.DataFrame(
        {
            "game_id": team_games[game_id_col].astype(str).to_numpy(),
            "team_id": team_games[team_id_col].astype(str).to_numpy(),
            "game_date": pd.to_datetime(team_games[game_date_col]).to_numpy(),
            "season_year": pd.to_numeric(team_games[season_year_col])
            .astype(int)
            .to_numpy(),
        },
        index=team_games.index,
    )
    covered = pd.Series(
        [
            (g, t) in state.covered
            for g, t in zip(rows["game_id"], rows["team_id"], strict=True)
        ],
        index=rows.index,
    )
    out = pd.DataFrame(
        np.nan, index=team_games.index, columns=status_feature_columns(top_n)
    )

    for status in LISTED_STATUSES:
        n_top = top_n.get(status, 0)
        names = per_player_names(status)
        has_history = status in HISTORY_STATUSES
        if has_history:
            out[league_p_play_col(status)] = league_play_rate(
                events.loc[events["status"] == status], rows
            ).to_numpy()
            for col in (
                sum_exp_players_col(status),
                sum_exp_col(status, "MIN"),
                sum_exp_col(status, "PTS"),
            ):
                out.loc[covered, col] = 0.0
            out.loc[covered, mean_p_play_col(status)] = 1.0
        for i in range(1, n_top + 1):
            for name in names:
                out.loc[covered, top_col(status, i, name)] = NEUTRAL_SLOT_VALUES[name]

        #: Same filter as the counters: a two-way or assignment listing is a
        #: roster mechanic, not a player at risk, so the slots and
        #: ``N_REPORT_<STATUS>_PLAYERS`` always agree on whether there is one.
        listed = state.statuses.loc[
            state.statuses["status"].eq(status)
            & ~state.statuses["reason_category"].isin(UNCOUNTED_REASON_CATEGORIES)
        ]
        targets = rows.reset_index(names="_row").merge(
            listed[["game_id", "team_id", "player_id"]], on=["game_id", "team_id"]
        )
        targets = targets.loc[
            [
                (g, t) in state.covered
                for g, t in zip(targets["game_id"], targets["team_id"], strict=True)
            ]
        ].reset_index(drop=True)
        if targets.empty:
            continue

        if has_history:
            est = status_player_estimates(targets, events, form, box, status)
        else:
            est = player_form_slots(targets, form, box)
        targets = pd.concat([targets, est], axis=1)

        if has_history:
            for stat in ("MIN", "PTS"):
                targets[f"exp_{stat}"] = targets["P_PLAY"] * (
                    targets[f"FORM_{stat}"] + targets[f"EFFECT_{stat}"].fillna(0.0)
                ).clip(lower=0.0)
            grouped = targets.groupby("_row")
            index = grouped.size().index
            out.loc[index, sum_exp_players_col(status)] = grouped["P_PLAY"].sum()
            out.loc[index, sum_exp_col(status, "MIN")] = grouped["exp_MIN"].sum()
            out.loc[index, sum_exp_col(status, "PTS")] = grouped["exp_PTS"].sum()
            out.loc[index, mean_p_play_col(status)] = grouped["P_PLAY"].mean()

        ranked = targets.sort_values(
            ["_row", f"FORM_{RANK_STAT.get(status, 'PTS')}", "player_id"],
            ascending=[True, False, True],
            na_position="last",
        )
        ranked["_rank"] = ranked.groupby("_row").cumcount() + 1
        for i in range(1, n_top + 1):
            slot = ranked.loc[ranked["_rank"] == i].set_index("_row")
            for name in names:
                out.loc[slot.index, top_col(status, i, name)] = slot[name]
    return out
