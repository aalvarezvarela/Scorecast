"""Rotation structure for the phase 4A minutes provider: rosters, baseline
minutes and who absorbs an absent player's minutes.

Only pieces B and C of the provider and the expanded roster live here;
participation (A) and the final reconciliation to 0-48 / 240 come later.
Minutes are **regulation** minutes from the validated stints.

**Expanded roster.** A player belongs to the team he last appeared for (or
was last listed for on an injury report) until he appears for another team.
Within a season he stays however many games he misses. Across a season
boundary he stays for the team's first ``carryover_games`` games, but he
creates no absence event until he appears or is listed this season: a player
who left in the summer and never played elsewhere would otherwise haunt his
old roster as a phantom absence.

**B, baseline minutes.** ``b_Y = E[minutes_Y | Y plays, full-health
context]``: a decayed mean (half-life in team games) of the minutes Y played,
each game's minutes first reduced by what the absences of that game gave him
(the attributed gains below).

**Absence events.** X is absent from a team game when he is on the roster and
does not play; the minutes he vacates are ``V_X = q_X * b_X``. Until the
participation model (A) exists, ``q_X`` is 1 for a rotation player
(``b_X >= rotation_minutes``: his missed games are unavailability, not
coach's decisions) and his recent participation rate otherwise (deep-bench
DNPs are coach's decisions). Events with ``V_X`` below ``min_vacated`` carry
no redistribution.

**C, redistribution shares.** ``s(X -> Y) >= 0`` with ``sum_Y s(X -> Y) = 1``
over the teammates eligible tonight: the share of X's vacated minutes that Y
absorbs when he plays. Evidence for a pair is what Y gained over his baseline
in the team's earlier games without X; when several players were out at
once, each player's gain is split between them in proportion to
``V_X * s(X -> Y)`` (an EM-style attribution), so simultaneous absences are
not counted twice. The pair estimate is shrunk toward a structural prior
``s0`` (``structural_share``), clipped at 0 and renormalized over the
teammates available tonight.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RotationParams:
    baseline_half_life: float = 15.0  # team games
    participation_window: int = 10  # team games before the current streak
    share_half_life: float = 41.0  # team games (frozen on 2018-19)
    history_games: int = 164  # team games kept per team
    min_vacated: float = 10.0  # minutes
    kappa: float = 30.0  # prior strength, in vacated minutes
    iterations: int = 3
    carryover_games: int = 10  # team games of a new season
    feature_set: str = "v1"  # structural prior features: "v1" or "v2"
    rotation_minutes: float = 15.0  # baseline above which a DNP is unavailability


DEFAULT_PARAMS = RotationParams()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def team_game_minutes(stints: pd.DataFrame) -> pd.DataFrame:
    """Regulation minutes per ``(game_id, team_id, player_id)`` with the game
    date, season and whether the player started (in the first stint)."""
    regulation = stints.loc[stints["period"] <= 4].sort_values(
        ["game_id", "seg_idx"], kind="mergesort"
    )
    parts = []
    first = regulation.drop_duplicates("game_id")
    starters = {
        (row.game_id, str(side_team)): {str(p) for p in lineup}
        for row in first.itertuples(index=False)
        for side_team, lineup in (
            (row.home_team_id, row.home_lineup),
            (row.away_team_id, row.away_lineup),
        )
    }
    for side in ("home", "away"):
        lineups = np.stack(regulation[f"{side}_lineup"].map(np.asarray).to_numpy())
        parts.append(
            pd.DataFrame(
                {
                    "game_id": np.repeat(
                        regulation["game_id"].astype(str).to_numpy(), 5
                    ),
                    "game_date": np.repeat(regulation["game_date"].to_numpy(), 5),
                    "season": np.repeat(regulation["season_year"].to_numpy(), 5),
                    "team_id": np.repeat(
                        regulation[f"{side}_team_id"].astype(str).to_numpy(), 5
                    ),
                    "player_id": lineups.astype(str).ravel(),
                    "minutes": np.repeat(regulation["seconds"].to_numpy(float), 5) / 60,
                }
            )
        )
    out = (
        pd.concat(parts, ignore_index=True)
        .groupby(
            ["game_id", "game_date", "season", "team_id", "player_id"], as_index=False
        )["minutes"]
        .sum()
    )
    out["started"] = [
        p in starters.get((g, t), set())
        for g, t, p in zip(
            out["game_id"], out["team_id"], out["player_id"], strict=True
        )
    ]
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    return out.sort_values(["game_date", "game_id", "team_id"], ignore_index=True)


# --------------------------------------------------------------------------
# Structural prior
# --------------------------------------------------------------------------

#: Features of a (absent X, teammate Y) pair for the structural prior.
PAIR_FEATURES = (
    "log_b_y",
    "rank_y",
    "rank_gap",
    "below_x",
    "position_similarity",
    "start_share_y",
    "participation_y",
)
#: v2 adds "next man up" features: who replaced X last time, Y's minutes and
#: start in the team's last game, his recent trend, and whether he sits just
#: outside tonight's usual rotation.
PAIR_FEATURES_V2 = (
    *PAIR_FEATURES,
    "replaced_x_last_time",
    "last_minutes_y",
    "started_last_y",
    "trend_y",
    "fringe_y",
)


def pair_features(
    absent: str,
    eligible: list[str],
    baseline: Mapping[str, float],
    participation: Mapping[str, float],
    start_share: Mapping[str, float],
    positions: Mapping[str, np.ndarray],
    extra: Mapping[str, Mapping[str, float]] | None = None,
) -> np.ndarray:
    """``(len(eligible), len(PAIR_FEATURES))`` features of each eligible
    teammate Y for absent X. Ranks are by baseline among X and the eligible.
    With ``extra`` (feature name -> Y -> value) the v2 columns follow."""
    ranked = sorted([absent, *eligible], key=lambda p: (-baseline.get(p, 0.0), p))
    rank = {p: i + 1 for i, p in enumerate(ranked)}
    px = positions.get(absent, np.full(3, 1 / 3))
    rows = []
    for y in eligible:
        rows.append(
            [
                np.log1p(max(0.0, baseline.get(y, 0.0))),
                rank[y],
                abs(rank[y] - rank[absent]),
                float(rank[y] > rank[absent]),
                float(px @ positions.get(y, np.full(3, 1 / 3))),
                start_share.get(y, 0.0),
                participation.get(y, 0.0),
                *(
                    [extra[name].get(y, 0.0) for name in PAIR_FEATURES_V2[7:]]
                    if extra is not None
                    else []
                ),
            ]
        )
    return np.asarray(rows, dtype=float)


@dataclass
class StructuralShare:
    """``s0(X -> Y) = softmax_Y(theta . f(X, Y))`` over the eligible
    teammates, fitted by least squares on minute gains of single-absence
    events. Unfitted, it is proportional to the baseline minutes (what
    ``allocate_minutes`` does)."""

    theta: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def __call__(self, features: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        if self.theta is None or not len(features):
            total = baseline.sum()
            return (
                baseline / total
                if total > 0
                else np.full(len(baseline), 1 / max(len(baseline), 1))
            )
        score = ((features - self.mean) / self.scale) @ self.theta
        score = np.exp(score - score.max())
        return score / score.sum()

    @classmethod
    def fit(cls, events: list[dict], ridge: float = 1.0) -> StructuralShare:
        """``events``: dicts with ``features`` (n, k), ``vacated`` V_X,
        ``gain`` (n,) minutes over baseline, NaN for teammates who did not
        play (left out of the loss but kept in the softmax)."""
        from scipy.optimize import minimize

        if not events:
            return cls()
        stacked = np.vstack([e["features"] for e in events])
        mean = stacked.mean(axis=0)
        scale = np.where(stacked.std(axis=0) > 1e-9, stacked.std(axis=0), 1.0)
        prepared = [
            ((e["features"] - mean) / scale, e["vacated"], e["gain"]) for e in events
        ]

        def loss(theta: np.ndarray) -> float:
            total = 0.0
            for x, vacated, gain in prepared:
                score = x @ theta
                share = np.exp(score - score.max())
                share /= share.sum()
                played = ~np.isnan(gain)
                total += np.sum((gain[played] - vacated * share[played]) ** 2)
            return total / len(prepared) + ridge * float(theta @ theta)

        result = minimize(loss, np.zeros(stacked.shape[1]), method="L-BFGS-B")
        return cls(result.x, mean, scale)


# --------------------------------------------------------------------------
# Walk-forward engine
# --------------------------------------------------------------------------


@dataclass
class _TeamGame:
    game_id: str
    played: dict[str, float]  # player -> minutes
    started: set[str]
    roster: list[str]
    absences: dict[str, float]  # X -> vacated minutes
    s0: dict[tuple[str, str], float]  # (X, Y) -> structural share at the time
    date: pd.Timestamp | None = None


@dataclass
class RotationState:
    """Per-team history and league-wide roster membership, as of a date."""

    params: RotationParams = DEFAULT_PARAMS
    team_of: dict[str, str] = field(default_factory=dict)
    season_of: dict[str, int] = field(default_factory=dict)
    history: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    season_games: dict[tuple[str, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    #: (team, absent X) -> the teammate who gained most the last time X was out.
    last_replacement: dict[tuple[str, str], str] = field(default_factory=dict)

    def roster(
        self, team: str, season: int, listed: Iterable[str] = ()
    ) -> tuple[list[str], set[str]]:
        """``(roster, current)`` before tonight's game. ``current``: last
        team is ``team`` and seen or listed this season (or listed tonight);
        the roster adds last season's players during the team's first
        ``carryover_games`` games."""
        listed = set(listed)
        current = {
            p
            for p, t in self.team_of.items()
            if t == team and self.season_of.get(p) == season
        } | {p for p in listed if self.team_of.get(p, team) == team}
        carried = set()
        if self.season_games[(team, season)] < self.params.carryover_games:
            carried = {
                p
                for p, t in self.team_of.items()
                if t == team and self.season_of.get(p) == season - 1
            }
        return sorted(current | carried), current


def _decayed(n: int, half_life: float) -> np.ndarray:
    """Weights of the last ``n`` games, oldest first."""
    return 0.5 ** (np.arange(n)[::-1] / half_life)


def estimate(
    history: list[_TeamGame],
    roster: list[str],
    params: RotationParams,
) -> tuple[
    dict[str, float], dict[str, float], dict[tuple[str, str], tuple[float, float]]
]:
    """``(baseline b, participation q, pair evidence)`` from a team's earlier
    games, oldest first. Pair evidence is ``(X, Y) -> (attributed gain,
    vacated minutes)``, both decayed."""
    n = len(history)
    w_b = _decayed(n, params.baseline_half_life)
    w_s = _decayed(n, params.share_half_life)
    players = set(roster)
    share: dict[tuple[str, str], float] = {}

    def attributed(game: _TeamGame, y: str) -> dict[str, float]:
        """Share of Y's gain in ``game`` owed to each absentee."""
        weights = {
            x: v * share.get((x, y), game.s0.get((x, y), 0.0))
            for x, v in game.absences.items()
        }
        total = sum(weights.values())
        return {x: w / total for x, w in weights.items()} if total > 0 else {}

    baseline: dict[str, float] = {}
    for _ in range(params.iterations):
        # B: baseline from played minutes net of absence gains.
        baseline = {}
        for y in players:
            num = den = 0.0
            for i, game in enumerate(history):
                if y not in game.played:
                    continue
                gain = sum(
                    game.absences[x] * share.get((x, y), game.s0.get((x, y), 0.0))
                    for x in game.absences
                )
                # Never below 0: a noisy share cannot attribute more than
                # he played.
                num += w_b[i] * max(0.0, game.played[y] - gain)
                den += w_b[i]
            if den > 0:
                baseline[y] = num / den
        # C: pair evidence from attributed gains over the new baseline.
        evidence: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for i, game in enumerate(history):
            for x, v in game.absences.items():
                for y in game.played:
                    # Shares are conditional on Y playing: only games he
                    # played count, in the gain and in the vacated minutes.
                    if y not in baseline:
                        continue
                    part = attributed(game, y).get(x, 0.0)
                    evidence[(x, y)][0] += (
                        w_s[i] * part * (game.played[y] - baseline[y])
                    )
                    evidence[(x, y)][1] += w_s[i] * v
        share = {
            key: max(0.0, gain / vacated) if vacated > 0 else 0.0
            for key, (gain, vacated) in evidence.items()
        }
    participation = {}
    for y in roster:
        streak = 0
        for game in reversed(history):
            if y in game.played or y not in game.roster:
                break
            streak += 1
        window = [g for g in history[: n - streak] if y in g.roster][
            -params.participation_window :
        ]
        participation[y] = (
            sum(y in g.played for g in window) / len(window) if window else 1.0
        )
    return baseline, participation, {k: tuple(v) for k, v in evidence.items()}


def shares(
    absent: str,
    eligible: list[str],
    evidence: Mapping[tuple[str, str], tuple[float, float]],
    s0: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Final ``s(absent -> Y)`` over ``eligible``: pair evidence shrunk toward
    ``s0``, clipped at 0, normalized."""
    values = []
    for y, prior in zip(eligible, s0, strict=True):
        gain, vacated = evidence.get((absent, y), (0.0, 0.0))
        values.append(max(0.0, (gain + kappa * prior) / (vacated + kappa)))
    values = np.asarray(values)
    total = values.sum()
    return values / total if total > 0 else s0


def walk_forward(
    minutes: pd.DataFrame,
    positions: Callable[[pd.Timestamp, str], np.ndarray],
    listed: Mapping[tuple[str, str], set[str]] | None = None,
    params: RotationParams = DEFAULT_PARAMS,
    refit_monthly: bool = True,
    score_v2: bool = False,
    on_team_game: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process every team-game in date order; return ``(players, absorption)``.

    ``players``: per roster player and team-game, ``b``, ``q`` and his actual
    minutes (0 if he did not play). ``absorption``: per absence event and
    eligible teammate, ``V_X``, the structural ``s0``, the final ``s``, the
    proportional share (``b_Y / sum b``) and Y's actual gain over ``b_Y``.
    Everything for a game is computed from earlier games only.

    ``on_team_game`` receives, before each team-game is looked at, the
    engine's state as of that game (roster, baselines, participation, pair
    evidence, structural prior, history): what a pre-game provider may use.
    Nothing about the game itself is passed.

    ``score_v2`` also scores the v2 structural prior on the same events
    (columns ``s0_v2``, ``s_v2``) without letting it feed back into the
    baselines, so both feature sets are compared on identical rows.
    """
    listed = listed or {}
    state = RotationState(params)
    structural = StructuralShare()
    structural_v2 = StructuralShare()
    single_events: list[dict] = []
    single_events_v2: list[dict] = []
    month = None
    player_rows, absorption_rows = [], []
    absorption_rows_v2: list[tuple[float, float]] = []
    for (date, game_id, team), game in minutes.groupby(
        ["game_date", "game_id", "team_id"], sort=True
    ):
        if refit_monthly and date.to_period("M") != month:
            month = date.to_period("M")
            if len(single_events) >= 200:
                structural = StructuralShare.fit(single_events)
            if score_v2 and len(single_events_v2) >= 200:
                structural_v2 = StructuralShare.fit(single_events_v2)
        season = int(game["season"].iloc[0])
        roster, current = state.roster(team, season, listed.get((game_id, team), ()))
        played = dict(zip(game["player_id"], game["minutes"], strict=True))
        history = list(state.history[team])
        baseline, participation, evidence = estimate(history, roster, params)
        start_share = {
            y: np.mean(
                [y in g.started for g in history[-10:] if y in g.played] or [0.0]
            )
            for y in roster
        }
        pos = {p: positions(date, p) for p in roster}
        if on_team_game is not None:
            on_team_game(
                {
                    "game_id": game_id,
                    "team": team,
                    "date": date,
                    "season": season,
                    "roster": roster,
                    "current": current,
                    "baseline": baseline,
                    "participation": participation,
                    "evidence": evidence,
                    "structural": structural,
                    "start_share": start_share,
                    "positions": pos,
                    "history": history,
                    "season_game": state.season_games[(team, season)],
                    "params": params,
                }
            )
        absences = {
            x: baseline[x]
            * (
                1.0
                if baseline[x] >= params.rotation_minutes
                else participation.get(x, 0.0)
            )
            for x in current
            if x not in played and x in baseline
        }
        absences = {x: v for x, v in absences.items() if v >= params.min_vacated}
        eligible = [y for y in roster if y not in absences]
        gain_c: dict[str, float] = defaultdict(float)
        context = None
        if params.feature_set == "v2" or score_v2:
            last = history[-1] if history else None
            ranked = sorted(
                (y for y in eligible if y in baseline), key=lambda y: -baseline[y]
            )
            rotation = sum(baseline[y] >= params.rotation_minutes for y in ranked)
            recent_played = {
                y: [g.played[y] for g in history if y in g.played][-3:] for y in roster
            }
            context = {
                "last_minutes_y": {
                    y: last.played.get(y, 0.0) if last else 0.0 for y in roster
                },
                "started_last_y": {
                    y: float(last is not None and y in last.started) for y in roster
                },
                "trend_y": {
                    y: (
                        (np.mean(recent_played[y]) - baseline[y])
                        if recent_played[y] and y in baseline
                        else 0.0
                    )
                    for y in roster
                },
                "fringe_y": {
                    y: float(1 <= i + 1 - rotation <= 2) for i, y in enumerate(ranked)
                },
            }
        s0_record = {}
        for x, vacated in absences.items():
            candidates = [y for y in eligible if y in baseline]
            if not candidates:
                continue
            extra = None
            if context is not None and params.feature_set == "v2":
                extra = {
                    **context,
                    "replaced_x_last_time": {
                        y: float(state.last_replacement.get((team, x)) == y)
                        for y in candidates
                    },
                }
            features = pair_features(
                x, candidates, baseline, participation, start_share, pos, extra
            )
            b = np.array([baseline[y] for y in candidates])
            s0 = structural(features, b)
            final = shares(x, candidates, evidence, s0, params.kappa)
            if score_v2:
                extra_v2 = {
                    **context,
                    "replaced_x_last_time": {
                        y: float(state.last_replacement.get((team, x)) == y)
                        for y in candidates
                    },
                }
                features_v2 = pair_features(
                    x, candidates, baseline, participation, start_share, pos, extra_v2
                )
                s0_v2 = structural_v2(features_v2, b)
                final_v2 = shares(x, candidates, evidence, s0_v2, params.kappa)
            proportional = b / b.sum()
            gain = np.array(
                [played[y] - baseline[y] if y in played else np.nan for y in candidates]
            )
            for k, (y, a, f_, p_, g_) in enumerate(
                zip(candidates, s0, final, proportional, gain, strict=True)
            ):
                s0_record[(x, y)] = a
                gain_c[y] += vacated * f_
                if score_v2:
                    absorption_rows_v2.append((s0_v2[k], final_v2[k]))
                absorption_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": team,
                        "game_date": date,
                        "absent": x,
                        "vacated": vacated,
                        "n_absent": len(absences),
                        "player_id": y,
                        "s0": a,
                        "s": f_,
                        "proportional": p_,
                        "pair_evidence": evidence.get((x, y), (0.0, 0.0))[1],
                        "gain": g_,
                        "played": y in played,
                    }
                )
            if len(absences) == 1:
                single_events.append(
                    {"features": features, "vacated": vacated, "gain": gain}
                )
                if score_v2:
                    single_events_v2.append(
                        {"features": features_v2, "vacated": vacated, "gain": gain}
                    )
            if not np.isnan(gain).all():
                state.last_replacement[(team, x)] = candidates[int(np.nanargmax(gain))]
        # Participation features: tonight's scenario is the other players'
        # absences; a player who is himself an absence event is described as
        # if he were available (ranked among the eligible, no gain).
        vacated_total = sum(absences.values())
        available = [y for y in eligible if y in baseline]
        previous = history[-1] if history else None
        rest = (date - previous.date).days if previous and previous.date else np.nan
        game_number = state.season_games[(team, season)]
        for y in sorted(set(roster) | set(played)):
            pool = available if y in available else [*available, y]
            ranked = sorted(pool, key=lambda p: (-baseline.get(p, 0.0), p))
            streak = 0
            for g in reversed(history):
                if y in g.played or y not in g.roster:
                    break
                streak += 1
            player_rows.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "game_date": date,
                    "player_id": y,
                    "on_roster": y in roster,
                    "b": baseline.get(y, np.nan),
                    "q": participation.get(y, np.nan),
                    "absent_event": y in absences,
                    "n_absent": len(absences),
                    "minutes": played.get(y, 0.0),
                    "gain_c": gain_c.get(y, 0.0),
                    "vacated_others": vacated_total - absences.get(y, 0.0),
                    "n_available": len(available),
                    "rank": ranked.index(y) + 1 if y in baseline else np.nan,
                    "streak": streak,
                    "last_minutes": previous.played.get(y, 0.0) if previous else 0.0,
                    "start_share": start_share.get(y, 0.0),
                    "rest_days": rest,
                    "season_game": game_number,
                }
            )
        # Tonight becomes history.
        state.history[team].append(
            _TeamGame(
                game_id,
                played,
                set(game.loc[game["started"], "player_id"]),
                roster,
                absences,
                s0_record,
                date,
            )
        )
        while len(state.history[team]) > params.history_games:
            state.history[team].popleft()
        state.season_games[(team, season)] += 1
        for p in played:
            state.team_of[p] = team
            state.season_of[p] = season
        for p in listed.get((game_id, team), ()):
            if state.team_of.get(p) in (None, team):
                state.team_of[p] = team
                state.season_of[p] = season
    absorption = pd.DataFrame(absorption_rows)
    if score_v2:
        absorption[["s0_v2", "s_v2"]] = np.asarray(absorption_rows_v2)
    return pd.DataFrame(player_rows), absorption
