"""The last injury report before tipoff, shaped for the closing-line dataset.

Rules (``docs/injury_status_tiers_plan.md`` section 1):

1. One snapshot per game: the last report published strictly before tipoff,
   whatever its age.
2. The report splits a covered roster into three availability groups:
   **injured** (Out ∪ Doubtful), **questionable** (Questionable) and
   **available** (Probable, Available, anyone unlisted). Questionable used to be
   folded into the out set; it is now its own group, because at the last report
   those players play 62% of the time, and the downstream availability families
   are computed for all three groups independently.
3. Per-category counters.
4. A team-game the report does not cover -- no report listed the game before
   tip, or the team had not submitted -- keeps exactly the legacy behaviour:
   its out set stays the inactive-list ∪ comment-regex set, and every
   report-derived column is NaN.

Coverage is decided per team-game, not per game: a team can file while its
opponent has not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Report statuses that put a player in the pre-game out set, i.e. the
#: "injured" availability group. Doubtful belongs here: at the last report
#: before tip it plays 3% of the time.
OUT_SET_STATUSES: frozenset[str] = frozenset({"out", "doubtful"})

#: The third availability group, between injured and available. Questionable
#: plays 62% of the time at the last report, so subtracting it from the team
#: like an absence -- what schema 2_4 did -- is wrong in both directions.
QUESTIONABLE_SET_STATUSES: frozenset[str] = frozenset({"questionable"})

#: Categories counted by ``N_REPORT_<STATUS>_PLAYERS``.
COUNTED_STATUSES: tuple[str, ...] = ("out", "doubtful", "questionable", "probable")

#: Two-way and assignment listings are roster mechanics, not absences from the
#: rotation; 30% of all Out rows are these. They stay in the out set (the
#: legacy inactive list carries them too) but are not counted.
UNCOUNTED_REASON_CATEGORIES: frozenset[str] = frozenset({"g_league"})

COVERED_COL = "INJURY_REPORT_COVERED_BEFORE"
#: Deliberately free of ``injury_``/``injured_``: those substrings are
#: zero-filled by the missing-data policy, and a missing age must not read as a
#: report published at tipoff.
REPORT_AGE_COL = "LAST_STATUS_REPORT_AGE_MIN_BEFORE"


def counter_col(status: str) -> str:
    return f"N_REPORT_{status.upper()}_PLAYERS_BEFORE"


REPORT_COUNTER_COLUMNS: tuple[str, ...] = (
    COVERED_COL,
    REPORT_AGE_COL,
    *(counter_col(s) for s in COUNTED_STATUSES),
)


@dataclass
class InjuryReportState:
    """Everything the closing-line features read from ``injury_report``.

    All ids are strings. ``statuses`` is the last report before tip;
    ``filings`` holds the ``submitted`` flag at tip per team-game;
    ``status_events`` is every game a player was Questionable or Probable at
    any pre-tip moment (one row per status held); ``listed_pairs``
    is every (game, player) with any designation before tip.
    """

    statuses: pd.DataFrame
    filings: pd.DataFrame
    report_age: pd.DataFrame
    status_events: pd.DataFrame
    listed_pairs: pd.DataFrame
    _covered: set[tuple[str, str]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for frame in (
            self.statuses,
            self.filings,
            self.report_age,
            self.status_events,
            self.listed_pairs,
        ):
            for col in ("game_id", "team_id", "player_id"):
                if col in frame.columns:
                    frame[col] = frame[col].astype(str)
        for frame in (self.statuses, self.status_events):
            if "game_date" in frame.columns:
                frame["game_date"] = pd.to_datetime(frame["game_date"])

    @property
    def covered(self) -> set[tuple[str, str]]:
        """(game_id, team_id) pairs whose team had submitted by tipoff."""
        if self._covered is None:
            filed = self.filings.loc[self.filings["submitted"].astype(bool)]
            self._covered = set(
                zip(filed["game_id"], filed["team_id"], strict=True)
            )
        return self._covered

    @classmethod
    def empty(cls) -> InjuryReportState:
        return cls(
            statuses=pd.DataFrame(
                columns=[
                    "game_id",
                    "team_id",
                    "player_id",
                    "status",
                    "reason_category",
                    "reason_detail",
                    "game_date",
                    "season_year",
                ]
            ),
            filings=pd.DataFrame(columns=["game_id", "team_id", "submitted"]),
            report_age=pd.DataFrame(columns=["game_id", "report_age_minutes"]),
            status_events=pd.DataFrame(
                columns=[
                    "game_id",
                    "team_id",
                    "player_id",
                    "status",
                    "game_date",
                    "season_year",
                    "at_last_report",
                ]
            ),
            listed_pairs=pd.DataFrame(columns=["game_id", "player_id"]),
        )


def load_injury_report_state(
    season_years: list[int] | None = None,
) -> InjuryReportState:
    """Read the state from Aiven. ``None`` reads every loaded season.

    History features need every earlier season regardless of the dataset's own
    window, so callers normally pass ``None``.
    """
    from nba_ou.postgre_db.config.db_config import connect_nba_db
    from nba_ou.postgre_db.injury_report_aiven import fetch

    with connect_nba_db("aiven") as conn:
        return InjuryReportState(
            statuses=fetch.last_status_before_tip(conn, season_years),
            filings=fetch.filing_at_tip(conn, season_years),
            report_age=fetch.report_age_at_tip(conn, season_years),
            status_events=fetch.listed_status_events(conn, season_years),
            listed_pairs=fetch.listed_pairs(conn, season_years),
        )


def report_status_sets(
    state: InjuryReportState,
    statuses: frozenset[str],
) -> dict[tuple[str, str], list[str]]:
    """``(game_id, team_id) -> players`` holding one of ``statuses``, for every
    covered team-game.

    Every covered team-game is a key, mapping to an empty list when nobody
    holds any of those statuses: that is a real "nobody", not missing data.
    Uncovered team-games are absent entirely (rule 4).
    """
    sets: dict[tuple[str, str], list[str]] = {key: [] for key in state.covered}
    listed = state.statuses.loc[state.statuses["status"].isin(statuses)]
    for game_id, team_id, player_id in listed[
        ["game_id", "team_id", "player_id"]
    ].itertuples(index=False):
        key = (game_id, team_id)
        if key in sets:
            sets[key].append(player_id)
    return {key: sorted(set(players)) for key, players in sets.items()}


def report_out_overrides(
    state: InjuryReportState,
) -> dict[tuple[str, str], list[str]]:
    """The pre-game out set -- the injured group, Out ∪ Doubtful -- per covered
    team-game.

    An empty list must replace, not fall back to, the legacy inactive-list set,
    which is what :func:`apply_report_out_overrides` does with it.
    """
    return report_status_sets(state, OUT_SET_STATUSES)


def report_questionable_sets(
    state: InjuryReportState,
) -> dict[tuple[str, str], list[str]]:
    """The questionable group per covered team-game.

    Unlike the out set this overrides nothing: before the report there was no
    such group, so an uncovered team-game simply has none and its
    questionable-group columns stay NaN.
    """
    return report_status_sets(state, QUESTIONABLE_SET_STATUSES)


def nested_status_dict(
    sets: dict[tuple[str, str], list[str]] | None,
) -> dict[str, dict[str, list[str]]]:
    """``{(game, team): players}`` in the nested ``{game: {team: players}}``
    shape the older availability helpers take.

    Covered team-games with nobody listed are kept as empty lists, so a
    consumer can tell "nobody at risk" from "no report".
    """
    nested: dict[str, dict[str, list[str]]] = {}
    for (game_id, team_id), players in (sets or {}).items():
        nested.setdefault(str(game_id), {})[str(team_id)] = [
            str(p) for p in players if pd.notna(p)
        ]
    return nested


def apply_report_out_overrides(
    injured_dict: dict,
    overrides: dict[tuple[str, str], list[str]] | None,
) -> dict:
    """``injured_dict`` with every covered team-game replaced by the report.

    Keys of the result are strings. Team-games without an override keep their
    legacy entry unchanged, so ``overrides=None`` (or ``{}``) returns the legacy
    dict itself, re-keyed.
    """
    result: dict[str, dict[str, list]] = {}
    for game_id, team_map in (injured_dict or {}).items():
        game_bucket = result.setdefault(str(game_id), {})
        for team_id, players in team_map.items():
            game_bucket[str(team_id)] = list(players)
    for (game_id, team_id), players in (overrides or {}).items():
        game_bucket = result.setdefault(str(game_id), {})
        if players:
            game_bucket[str(team_id)] = list(players)
        else:
            game_bucket.pop(str(team_id), None)
    return {game: teams for game, teams in result.items() if teams}


def union_membership_dicts(*dicts: dict) -> dict:
    """Per team-game union of player lists, used only for roster membership."""
    result: dict[str, dict[str, set]] = {}
    for mapping in dicts:
        for game_id, team_map in (mapping or {}).items():
            game_bucket = result.setdefault(str(game_id), {})
            for team_id, players in team_map.items():
                game_bucket.setdefault(str(team_id), set()).update(
                    str(p) for p in players if pd.notna(p)
                )
    return {
        game: {team: sorted(players) for team, players in teams.items()}
        for game, teams in result.items()
    }


def report_counter_features(
    team_games: pd.DataFrame,
    state: InjuryReportState,
    *,
    game_id_col: str = "GAME_ID",
    team_id_col: str = "TEAM_ID",
) -> pd.DataFrame:
    """Coverage flag, report age and per-status counters, one row per team-game.

    Returned frame is aligned to ``team_games``' index. Uncovered team-games get
    ``INJURY_REPORT_COVERED_BEFORE = 0`` and NaN everywhere else.
    """
    keys = pd.DataFrame(
        {
            "game_id": team_games[game_id_col].astype(str).to_numpy(),
            "team_id": team_games[team_id_col].astype(str).to_numpy(),
        },
        index=team_games.index,
    )
    covered = np.array(
        [(g, t) in state.covered for g, t in zip(keys["game_id"], keys["team_id"], strict=True)],
        dtype=bool,
    )
    out = pd.DataFrame(index=team_games.index)
    out[COVERED_COL] = covered.astype(int)

    age = state.report_age.drop_duplicates("game_id").set_index("game_id")[
        "report_age_minutes"
    ]
    out[REPORT_AGE_COL] = keys["game_id"].map(age).astype(float).where(covered)

    counted = state.statuses.loc[
        ~state.statuses["reason_category"].isin(UNCOUNTED_REASON_CATEGORIES)
    ]
    counts = (
        counted.groupby(["game_id", "team_id", "status"]).size().unstack(fill_value=0)
    )
    joined = keys.join(counts, on=["game_id", "team_id"])
    for status in COUNTED_STATUSES:
        values = (
            joined[status].fillna(0).astype(float)
            if status in joined.columns
            else pd.Series(0.0, index=keys.index)
        )
        out[counter_col(status)] = values.where(covered)
    return out
