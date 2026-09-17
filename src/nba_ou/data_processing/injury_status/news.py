"""Injury news up to each intermediate snapshot.

2_5 already says what the report states at a snapshot. This module says what
*changed* on the way there, in one unit: **expected missing points**,

    E(t) = sum over a team's players of p_out(t) x FORM_PTS

where ``p_out`` is the chance the player sits given their status at ``t`` and
``FORM_PTS`` is 2_5's pre-game scoring form (``status_history.player_form_slots``).
News over a window W is E(T) - E(T - W). Measurements behind every choice below:
``docs/intermediate_market_dynamics_plan.md`` sections 1.2-1.3.

Chance of sitting, by status:

=============  ==========================================
Out            1.0
Doubtful       0.98
Questionable   1 - 2_5 ``P_PLAY`` for Questionable
Probable       1 - 2_5 ``P_PLAY`` for Probable
Available      0.02
dropped off    0.0
G League       0.0 (a roster mechanic, as in the 2_5 counters)
=============  ==========================================

Rules that stop a long-term absence from reading as fresh news every game:

* **Baseline carry.** Until a player's first listing in this game, they keep
  their status on the last report of the team's previous game. A player this
  game never lists is left out entirely. Counting such players as returning at
  the team's first filing was measured and tracked the market worse (spread
  move from open, correlation 0.30 vs 0.34 at 120 minutes on 2023-24), so it
  is not done.
* A status change with the same chance of sitting (a new reason only) is not
  news.
* Everything is read **strictly before** the snapshot, like every other report
  read: a report stamped exactly at T is not visible at T.
* Fixed windows (60 and 240 minutes) rather than "since the previous report",
  which is a multi-hour gap in 2019-20, an hour in 2021-24 and 15 minutes from
  2025.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .report_state import UNCOUNTED_REASON_CATEGORIES

P_OUT_BY_STATUS: dict[str, float] = {"out": 1.0, "doubtful": 0.98, "available": 0.02}
PLAY_PROBABILITY_COLUMNS: dict[str, str] = {
    "questionable": "P_PLAY_QUESTIONABLE",
    "probable": "P_PLAY_PROBABLE",
}
#: A team-report change at least this large, in expected points, is "material".
MATERIAL_NEWS_POINTS = 3.0
NEWS_WINDOWS_MINUTES: tuple[int, ...] = (60, 240)
MAX_PLAYER_WINDOW_MINUTES = 240
MATERIAL_FLAG_WINDOW_MINUTES = 240
MAX_MINUTES_SINCE_NEWS = 1440.0
_MINUTE_NS = 60 * 1_000_000_000
_EPS = 1e-9

NEWS_FEATURE_NAMES: tuple[str, ...] = (
    "NEWS_EXP_PTS_W60",
    "NEWS_EXP_PTS_W240",
    "NEWS_EXP_PTS_SINCE_PREV_GAME",
    "MAX_PLAYER_NEWS_EXP_PTS_W240",
    "MIN_SINCE_MATERIAL_NEWS",
    "HAS_MATERIAL_NEWS",
)


def news_column(name: str, side: str) -> str:
    """``INJ_SNAP_<name>_BEFORE_TEAM_<HOME|AWAY>``."""
    return f"INJ_SNAP_{name}_BEFORE_TEAM_{side}"


def _ns(values) -> np.ndarray:
    return pd.to_datetime(pd.Series(values), utc=True).astype("int64").to_numpy()


def chance_out(frame: pd.DataFrame) -> np.ndarray:
    """``p_out`` per row from ``status``, ``reason_category`` and the play rates.

    A missing play probability falls back to the league-typical value (0.58 for
    Questionable, 0.95 for Probable) rather than NaN, so one unestimated player
    cannot blank a team's timeline.
    """
    status = frame["status"]
    p = status.map(P_OUT_BY_STATUS).astype(float)
    for code, column, typical in (
        ("questionable", PLAY_PROBABILITY_COLUMNS["questionable"], 0.58),
        ("probable", PLAY_PROBABILITY_COLUMNS["probable"], 0.95),
    ):
        rows = status.eq(code)
        if column in frame:
            play = pd.to_numeric(frame[column], errors="coerce")
        else:
            play = pd.Series(np.nan, index=frame.index)
        p = p.mask(rows, 1.0 - play.fillna(typical).clip(0.0, 1.0))
    p = p.fillna(0.0)
    if "reason_category" in frame:
        p = p.mask(frame["reason_category"].isin(UNCOUNTED_REASON_CATEGORIES), 0.0)
    return p.to_numpy(float)


def baseline_statuses(spans: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Each player's status on the last pre-tip report of the team's previous game.

    Returns ``game_id`` (the CURRENT game), ``team_id``, ``player_id``,
    ``status`` and ``reason_category``, only for players the current game lists
    at some point. A player who had dropped off that report is not carried.
    """
    ordered = spans.sort_values("valid_from", kind="mergesort")
    last = ordered.groupby(["game_id", "team_id", "player_id"], sort=False).tail(1)
    last = last.loc[last["status"].notna()]
    last = last[["game_id", "team_id", "player_id", "status", "reason_category"]]
    links = schedule.loc[
        schedule["prev_game_id"].notna(), ["game_id", "team_id", "prev_game_id"]
    ]
    carried = links.merge(
        last.rename(columns={"game_id": "prev_game_id"}),
        on=["prev_game_id", "team_id"],
        how="inner",
    )
    listed_now = spans[["game_id", "team_id", "player_id"]].drop_duplicates()
    carried = carried.merge(listed_now, on=["game_id", "team_id", "player_id"])
    return carried.drop(columns="prev_game_id").reset_index(drop=True)


def estimation_targets(spans: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Every (game, player) needing ``FORM_PTS`` and play probabilities.

    The players listed in the game plus those carried from the previous game.
    Columns: ``game_id``, ``player_id``, ``season_year``, ``game_date``.
    """
    carried = baseline_statuses(spans, schedule)[["game_id", "player_id"]]
    pairs = pd.concat(
        [spans[["game_id", "player_id"]], carried], ignore_index=True
    ).drop_duplicates()
    dates = schedule[["game_id", "season_year", "game_date"]].drop_duplicates("game_id")
    pairs = pairs.merge(dates, on="game_id", how="inner")
    pairs["game_date"] = pd.to_datetime(pairs["game_date"])
    pairs["season_year"] = pairs["season_year"].astype(int)
    return pairs.reset_index(drop=True)


def estimate_listing_inputs(
    targets: pd.DataFrame,
    *,
    box: pd.DataFrame,
    status_events: pd.DataFrame,
    listed_pairs: pd.DataFrame,
    filings_at_tip: pd.DataFrame,
) -> pd.DataFrame:
    """2_5's ``FORM_PTS`` and ``P_PLAY`` per target, from the 2_5 estimators.

    No new estimator: the same ``player_form_slots`` / ``status_player_estimates``
    the snapshot status features use, so a Questionable player's chance of
    sitting here is exactly one minus the ``P_PLAY`` 2_5 publishes for that player.
    """
    from .report_state import InjuryReportState
    from .status_history import (
        build_player_form,
        build_status_events,
        player_form_slots,
        prepare_box_history,
        status_player_estimates,
    )

    empty = InjuryReportState.empty()
    state = InjuryReportState(
        statuses=empty.statuses,
        filings=filings_at_tip,
        report_age=empty.report_age,
        status_events=status_events,
        listed_pairs=listed_pairs,
    )
    box = prepare_box_history(box)
    form = build_player_form(box)
    events = build_status_events(state, box, form)

    out = targets[["game_id", "player_id"]].copy()
    out["FORM_PTS"] = player_form_slots(targets, form, box)["FORM_PTS"].to_numpy()
    for status, column in PLAY_PROBABILITY_COLUMNS.items():
        estimates = status_player_estimates(targets, events, form, box, status)
        out[column] = estimates["P_PLAY"].to_numpy()
    return out


@dataclass
class InjuryNewsTimeline:
    """Per team-game step changes in expected missing points.

    ``events``: ``game_id``, ``team_id``, ``time_ns``, ``d_points`` (team change
    at that report), ``max_player_d_points`` (the signed largest single-player
    change in it) and ``expected_points`` (E just after it). ``team_games``:
    ``game_id``, ``team_id``, ``base_points`` (E before any report of this
    game) and ``prev_covered`` (whether the baseline report existed).
    ``filings``: the filing spans with ``valid_from_ns`` / ``valid_to_ns``.
    """

    events: pd.DataFrame
    team_games: pd.DataFrame
    filings: pd.DataFrame

    # ---- point reads -----------------------------------------------------

    def _queries(self, game_ids, team_ids, times_ns) -> pd.DataFrame:
        n = len(times_ns)
        teams = (
            np.full(n, "", dtype=object)
            if team_ids is None
            else pd.Series(team_ids).astype(str).to_numpy()
        )
        return pd.DataFrame(
            {
                "game_id": pd.Series(game_ids).astype(str).to_numpy(),
                "team_id": teams,
                "time_ns": np.asarray(times_ns, dtype="int64"),
                "_i": np.arange(n),
            }
        )

    def expected_missing_at(self, game_ids, team_ids, times_ns) -> np.ndarray:
        """E strictly before each time; the baseline before any report."""
        queries = self._queries(game_ids, team_ids, times_ns)
        found = pd.merge_asof(
            queries.sort_values("time_ns", kind="mergesort"),
            self.events[["game_id", "team_id", "time_ns", "expected_points"]]
            .rename(columns={"time_ns": "event_ns"})
            .sort_values("event_ns", kind="mergesort"),
            left_on="time_ns",
            right_on="event_ns",
            by=["game_id", "team_id"],
            direction="backward",
            allow_exact_matches=False,
        )
        found = found.merge(
            self.team_games[["game_id", "team_id", "base_points"]],
            on=["game_id", "team_id"],
            how="left",
        ).sort_values("_i")
        value = found["expected_points"].fillna(found["base_points"]).fillna(0.0)
        return value.to_numpy(float)

    def covered_at(self, game_ids, team_ids, times_ns) -> np.ndarray:
        """Whether the team had a submitted filing strictly before each time."""
        queries = self._queries(game_ids, team_ids, times_ns)
        found = pd.merge_asof(
            queries.sort_values("time_ns", kind="mergesort"),
            self.filings[
                ["game_id", "team_id", "valid_from_ns", "valid_to_ns", "submitted"]
            ].sort_values("valid_from_ns", kind="mergesort"),
            left_on="time_ns",
            right_on="valid_from_ns",
            by=["game_id", "team_id"],
            direction="backward",
            allow_exact_matches=False,
        ).sort_values("_i")
        submitted = found["submitted"].astype("boolean").fillna(False).to_numpy(bool)
        still_open = (found["valid_to_ns"] >= found["time_ns"]).fillna(False)
        return submitted & still_open.to_numpy(bool)

    def events_in_window(
        self, game_ids, team_ids, times_ns, window_minutes: float | None
    ) -> pd.DataFrame:
        """Events strictly before each time, and within the window when given.

        Returns the matching events with the query position ``_i``.
        """
        queries = self._queries(game_ids, team_ids, times_ns)
        keys = ["game_id"] if team_ids is None else ["game_id", "team_id"]
        if team_ids is None:
            queries = queries.drop(columns="team_id")
        joined = queries.merge(
            self.events.rename(columns={"time_ns": "event_ns"}), on=keys, how="inner"
        )
        keep = joined["event_ns"] < joined["time_ns"]
        if window_minutes is not None:
            keep &= joined["event_ns"] >= joined["time_ns"] - int(
                window_minutes * _MINUTE_NS
            )
        return joined.loc[keep]

    # ---- features ---------------------------------------------------------

    def features_at(self, game_ids, team_ids, times_ns) -> pd.DataFrame:
        """The G1 feature values for one side, positionally aligned.

        An uncovered team at T gets NaN values and a 0 flag, as the 2_5 report
        features do.
        """
        times_ns = np.asarray(times_ns, dtype="int64")
        n = len(times_ns)
        now = self.expected_missing_at(game_ids, team_ids, times_ns)
        out = pd.DataFrame(index=np.arange(n))
        for window in NEWS_WINDOWS_MINUTES:
            then = self.expected_missing_at(
                game_ids, team_ids, times_ns - window * _MINUTE_NS
            )
            out[f"NEWS_EXP_PTS_W{window}"] = now - then

        meta = pd.DataFrame(
            {
                "game_id": pd.Series(game_ids).astype(str).to_numpy(),
                "team_id": pd.Series(team_ids).astype(str).to_numpy(),
            }
        ).merge(
            self.team_games[["game_id", "team_id", "base_points", "prev_covered"]],
            on=["game_id", "team_id"],
            how="left",
        )
        prev_covered = (
            meta["prev_covered"].astype("boolean").fillna(False).to_numpy(bool)
        )
        out["NEWS_EXP_PTS_SINCE_PREV_GAME"] = np.where(
            prev_covered, now - meta["base_points"].fillna(0.0).to_numpy(), np.nan
        )

        recent = self.events_in_window(
            game_ids, team_ids, times_ns, MAX_PLAYER_WINDOW_MINUTES
        )
        biggest = np.zeros(n)
        if not recent.empty:
            pick = recent.loc[
                recent["max_player_d_points"].abs().groupby(recent["_i"]).idxmax()
            ]
            biggest[pick["_i"].to_numpy()] = pick["max_player_d_points"].to_numpy()
        out["MAX_PLAYER_NEWS_EXP_PTS_W240"] = biggest

        material = self.events_in_window(game_ids, team_ids, times_ns, None)
        material = material.loc[material["d_points"].abs() >= MATERIAL_NEWS_POINTS]
        minutes_since = np.full(n, MAX_MINUTES_SINCE_NEWS)
        if not material.empty:
            latest = material.groupby("_i")["event_ns"].max()
            elapsed = (
                times_ns[latest.index.to_numpy()] - latest.to_numpy()
            ) / _MINUTE_NS
            minutes_since[latest.index.to_numpy()] = np.minimum(
                elapsed, MAX_MINUTES_SINCE_NEWS
            )
        out["MIN_SINCE_MATERIAL_NEWS"] = minutes_since
        out["HAS_MATERIAL_NEWS"] = (
            minutes_since < MATERIAL_FLAG_WINDOW_MINUTES
        ).astype(float)

        covered = self.covered_at(game_ids, team_ids, times_ns)
        values = [c for c in NEWS_FEATURE_NAMES if c != "HAS_MATERIAL_NEWS"]
        out.loc[~covered, values] = np.nan
        out.loc[~covered, "HAS_MATERIAL_NEWS"] = 0.0
        out["COVERED"] = covered.astype(float)
        return out

    def latest_material_news(
        self, game_ids, times_ns, lookback_minutes: float
    ) -> np.ndarray:
        """Latest material team change of EITHER team strictly before each time.

        Only within ``lookback_minutes``; NaN where there is none.
        """
        times_ns = np.asarray(times_ns, dtype="int64")
        found = self.events_in_window(game_ids, None, times_ns, lookback_minutes)
        found = found.loc[found["d_points"].abs() >= MATERIAL_NEWS_POINTS]
        latest = np.full(len(times_ns), np.nan)
        if not found.empty:
            best = found.groupby("_i")["event_ns"].max()
            latest[best.index.to_numpy()] = best.to_numpy(float)
        return latest


def build_news_timeline(
    spans: pd.DataFrame,
    filings: pd.DataFrame,
    schedule: pd.DataFrame,
    estimates: pd.DataFrame,
) -> InjuryNewsTimeline:
    """Turn report spans into per team-game changes in expected missing points.

    ``spans``: ``fetch.status_spans`` output. ``filings``: ``fetch.filing_spans``.
    ``schedule``: ``fetch.team_game_schedule``. ``estimates``: one row per
    (game, player) with ``FORM_PTS`` and the ``P_PLAY_*`` columns
    (:func:`estimate_listing_inputs`).
    """
    spans = spans.copy()
    for column in ("game_id", "team_id", "player_id"):
        spans[column] = spans[column].astype(str)
    schedule = schedule.copy()
    for column in ("game_id", "team_id"):
        schedule[column] = schedule[column].astype(str)
    schedule["prev_game_id"] = schedule["prev_game_id"].where(
        schedule["prev_game_id"].notna(), None
    )
    estimates = estimates.copy()
    for column in ("game_id", "player_id"):
        estimates[column] = estimates[column].astype(str)
    estimates = estimates.drop_duplicates(["game_id", "player_id"])

    filings = filings.copy()
    for column in ("game_id", "team_id"):
        filings[column] = filings[column].astype(str)
    filings["valid_from_ns"] = _ns(filings["valid_from"])
    filings["valid_to_ns"] = _ns(filings["valid_to"])
    filings["submitted"] = filings["submitted"].astype(bool)

    form = estimates[["game_id", "player_id", "FORM_PTS"]]

    # ---- baseline: carried from the previous game -----------------------
    carried = baseline_statuses(spans, schedule)
    carried = carried.merge(estimates, on=["game_id", "player_id"], how="left")
    carried["p_base"] = chance_out(carried)
    carried = carried[["game_id", "team_id", "player_id", "p_base"]]

    submitted = filings.loc[filings["submitted"]]
    filed_team_games = set(zip(submitted["game_id"], submitted["team_id"], strict=True))
    team_games = schedule[["game_id", "team_id", "prev_game_id"]].copy()
    team_games["prev_covered"] = [
        prev is not None and (prev, team) in filed_team_games
        for prev, team in zip(
            team_games["prev_game_id"], team_games["team_id"], strict=True
        )
    ]
    base = (
        carried.merge(form, on=["game_id", "player_id"], how="left")
        .assign(points=lambda f: f["p_base"] * f["FORM_PTS"].fillna(0.0).clip(lower=0))
        .groupby(["game_id", "team_id"])["points"]
        .sum()
        .rename("base_points")
    )
    team_games = team_games.merge(
        base.reset_index(), on=["game_id", "team_id"], how="left"
    )
    team_games["base_points"] = team_games["base_points"].fillna(0.0)

    # ---- per-player step changes ----------------------------------------
    listed = spans.merge(estimates, on=["game_id", "player_id"], how="left")
    steps = pd.DataFrame(
        {
            "game_id": listed["game_id"],
            "team_id": listed["team_id"],
            "player_id": listed["player_id"],
            "time_ns": _ns(listed["valid_from"]),
            "p": chance_out(listed),
        }
    )
    steps = steps.merge(
        carried, on=["game_id", "team_id", "player_id"], how="left"
    ).merge(form, on=["game_id", "player_id"], how="left")
    steps["p_base"] = steps["p_base"].fillna(0.0)
    steps["FORM_PTS"] = steps["FORM_PTS"].fillna(0.0).clip(lower=0.0)
    steps = steps.sort_values(
        ["game_id", "team_id", "player_id", "time_ns"], kind="mergesort"
    ).reset_index(drop=True)
    previous = steps.groupby(["game_id", "team_id", "player_id"], sort=False)[
        "p"
    ].shift()
    steps["d_points"] = (steps["p"] - previous.fillna(steps["p_base"])) * steps[
        "FORM_PTS"
    ]
    steps = steps.loc[steps["d_points"].abs() > _EPS]

    if steps.empty:
        events = pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "time_ns",
                "d_points",
                "max_player_d_points",
                "expected_points",
            ]
        ).astype({"time_ns": "int64", "d_points": float, "expected_points": float})
    else:
        keys = ["game_id", "team_id", "time_ns"]
        biggest = steps.loc[
            steps["d_points"].abs().groupby([steps[k] for k in keys]).idxmax()
        ].set_index(keys)["d_points"]
        events = (
            steps.groupby(keys)["d_points"]
            .sum()
            .to_frame()
            .join(biggest.rename("max_player_d_points"))
            .reset_index()
            .sort_values(keys, kind="mergesort")
        )
        events = events.merge(
            team_games[["game_id", "team_id", "base_points"]],
            on=["game_id", "team_id"],
            how="left",
        )
        events["expected_points"] = events.groupby(["game_id", "team_id"])[
            "d_points"
        ].cumsum() + events["base_points"].fillna(0.0)
        events = events.drop(columns="base_points").reset_index(drop=True)
    events["time_ns"] = events["time_ns"].astype("int64")

    return InjuryNewsTimeline(
        events=events,
        team_games=team_games[["game_id", "team_id", "base_points", "prev_covered"]],
        filings=filings[
            ["game_id", "team_id", "valid_from_ns", "valid_to_ns", "submitted"]
        ],
    )
