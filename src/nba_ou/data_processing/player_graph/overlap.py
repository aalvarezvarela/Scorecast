"""Expected shared floor time of two players: the v0 overlap provider.

Teammate and opponent edges of a game graph are weighted by how long the two
players are expected to share the floor, and the same overlap enters every
guard edge (``m_ij`` is proportional to ``r_hat_ij * E[overlap_ij]``), so it
decides who is expected to face whom.

**Independence** (the baseline, provider ``independence``): two players with
``min_i`` and ``min_j`` minutes in a game of length ``L`` share
``min_i * min_j / L`` if their stints are unrelated. That under-counts starters
(they open every half together) and over-counts players who replace each other.

**Pair lift** (provider ``pair_lift``, v0): the ratio of the time a pair
actually shared to what independence predicted, over their earlier games,
shrunk toward the lift of its relation::

    lift = (sum w * shared_seconds + k_rel * relation_lift)
           / (sum w * independent_seconds + k_rel)
    E[overlap] = min(lift * min_i * min_j / 48, min_i, min_j)

``relation_lift`` is the decayed league-wide ratio for teammates or opponents
in the same window. It is not 1 for teammates: with a player on the floor only
four teammate slots are left, so independence over-counts teammate overlap
(actual / independent = 0.906 over 2023-24). For opponents it is 1 by
construction (each player shares exactly 5x his seconds with opponents).

``w = 0.5 ** (age_days / half_life_days)`` decays both sums alike; ``k_rel``
is in independent-overlap seconds and differs by relation (opponents meet a few
times a season, so they need far more shrinkage than teammates). Teammate and opponent pairs keep separate
histories (a traded pair's games as teammates do not describe them as
opponents). Everything is read strictly before the cutoff, through ``as_of``.

The per-game table (:func:`build_overlap_game`) lists **every** pair of players
who both played a game, including pairs that never shared the floor: that
zero is exactly what a lift below 1 has to learn.

Known v0 limitation: pairwise overlaps need not add up to a realizable
five-man rotation. They are used only as expected edge weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .as_of import AsOfView

PROVIDERS = ("pair_lift", "independence")
VERSION = "v0"
RELATIONS = ("teammate", "opponent")

GAME_COLUMNS = (
    "game_id",
    "game_date",
    "season_year",
    "player_a",
    "player_b",
    "relation",
    "seconds_a",
    "seconds_b",
    "shared_seconds",
    "independent_seconds",
)

LIFT_COLUMNS = (
    "player_a",
    "player_b",
    "relation",
    "lift",
    "relation_lift",
    "lift_prior_weight",
    "n_games",
    "hist_shared_seconds_decayed",
    "hist_independent_seconds_decayed",
    "has_pair_history",
)


@dataclass(frozen=True)
class OverlapParams:
    #: Pseudo-counts in independent-overlap seconds, per relation, chosen on
    #: 2018-19 only (``scripts/player_graph/evaluate_overlap.py --seasons 2018``).
    #: Teammates have long pair histories and want little shrinkage (TVD flat
    #: at 0.172 for k = 0-300, 0.175 at 1200); opponents meet a few times a
    #: season and want much more (TVD 0.176 pair-only, 0.1560 at 1200, 0.1563
    #: at 2400).
    k_teammate: float = 150.0
    k_opponent: float = 1200.0
    half_life_days: float = 365.0
    window_days: int = 3 * 365
    provider: str = "pair_lift"

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")

    def k_for(self, relation: pd.Series) -> np.ndarray:
        """The pseudo-count of each row's relation."""
        return np.where(
            relation.eq("teammate").to_numpy(), self.k_teammate, self.k_opponent
        )


DEFAULT_PARAMS = OverlapParams()


def pair_key(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Undirected key: the smaller id first."""
    a, b = a.astype(str), b.astype(str)
    swap = a > b
    return a.where(~swap, b), b.where(~swap, a)


def build_overlap_game(stints: pd.DataFrame) -> pd.DataFrame:
    """One row per game and pair of players who both played it (``GAME_COLUMNS``)."""
    if stints.empty:
        return pd.DataFrame(columns=list(GAME_COLUMNS))
    seconds = stints["seconds"].to_numpy(float)
    game = stints["game_id"].astype(str).to_numpy()
    home = np.stack(stints["home_lineup"].map(np.asarray).to_numpy()).astype(str)
    away = np.stack(stints["away_lineup"].map(np.asarray).to_numpy()).astype(str)

    # Seconds per player and game, and the game's length (overtime included).
    on_floor = pd.DataFrame(
        {
            "game_id": np.concatenate([np.repeat(game, 5)] * 2),
            "side": np.repeat(["home", "away"], len(game) * 5),
            "player": np.concatenate([home.ravel(), away.ravel()]),
            "seconds": np.concatenate([np.repeat(seconds, 5)] * 2),
        }
    )
    played = on_floor.groupby(["game_id", "side", "player"], as_index=False)[
        "seconds"
    ].sum()
    length = pd.Series(seconds, index=game).groupby(level=0).sum().rename("length")

    # Actual shared seconds of the pairs that met in a stint.
    shared = []
    for i, j in combinations(range(5), 2):
        for lineup in (home, away):
            shared.append(
                pd.DataFrame(
                    {
                        "game_id": game,
                        "a": lineup[:, i],
                        "b": lineup[:, j],
                        "relation": "teammate",
                        "shared": seconds,
                    }
                )
            )
    for i in range(5):
        shared.append(
            pd.DataFrame(
                {
                    "game_id": np.repeat(game, 5),
                    "a": np.repeat(home[:, i], 5),
                    "b": away.ravel(),
                    "relation": "opponent",
                    "shared": np.repeat(seconds, 5),
                }
            )
        )
    shared = pd.concat(shared, ignore_index=True)
    shared["a"], shared["b"] = pair_key(shared["a"], shared["b"])
    shared = shared.groupby(["game_id", "a", "b", "relation"], as_index=False)[
        "shared"
    ].sum()

    # Every pair who both played, met or not.
    left = played.rename(columns={"player": "a", "seconds": "seconds_a"})
    right = played.rename(columns={"player": "b", "seconds": "seconds_b"})
    pairs = left.merge(right, on="game_id", suffixes=("_a", "_b"))
    pairs = pairs.loc[
        (pairs["side_a"].eq(pairs["side_b"]) & (pairs["a"] < pairs["b"]))
        | (pairs["side_a"].eq("home") & pairs["side_b"].eq("away"))
    ]
    pairs["relation"] = np.where(
        pairs["side_a"].eq(pairs["side_b"]), "teammate", "opponent"
    )
    swap = pairs["a"].astype(str) > pairs["b"].astype(str)
    pairs["a"], pairs["b"] = pair_key(pairs["a"], pairs["b"])
    pairs[["seconds_a", "seconds_b"]] = np.where(
        swap.to_numpy()[:, None],
        pairs[["seconds_b", "seconds_a"]].to_numpy(),
        pairs[["seconds_a", "seconds_b"]].to_numpy(),
    )
    pairs = pairs.merge(shared, on=["game_id", "a", "b", "relation"], how="left")
    pairs["shared"] = pairs["shared"].fillna(0.0)
    pairs["independent"] = (
        pairs["seconds_a"] * pairs["seconds_b"] / pairs["game_id"].map(length)
    )

    games = stints.drop_duplicates("game_id").set_index(
        stints.drop_duplicates("game_id")["game_id"].astype(str)
    )
    out = pd.DataFrame(
        {
            "game_id": pairs["game_id"].to_numpy(),
            "game_date": pd.to_datetime(
                pairs["game_id"].map(games["game_date"])
            ).to_numpy(),
            "season_year": pairs["game_id"].map(games["season_year"]).to_numpy(),
            "player_a": pairs["a"].to_numpy(),
            "player_b": pairs["b"].to_numpy(),
            "relation": pairs["relation"].to_numpy(),
            "seconds_a": pairs["seconds_a"].to_numpy(float),
            "seconds_b": pairs["seconds_b"].to_numpy(float),
            "shared_seconds": pairs["shared"].to_numpy(float),
            "independent_seconds": pairs["independent"].to_numpy(float),
        }
    )
    return out.sort_values(
        ["game_date", "game_id", "relation", "player_a", "player_b"],
        kind="mergesort",
        ignore_index=True,
    )


def shrink_lift(
    shared_decayed, independent_decayed, prior, k
) -> tuple[np.ndarray, np.ndarray]:
    """``(lift, prior_weight)`` for shrinkage strength ``k`` toward ``prior``."""
    shared = np.asarray(shared_decayed, float)
    independent = np.asarray(independent_decayed, float)
    prior = np.asarray(prior, float)
    k = np.asarray(k, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (shared + k * prior) / (independent + k), k / (independent + k)


def expected_lift(
    view: AsOfView, pairs: pd.DataFrame, *, params: OverlapParams = DEFAULT_PARAMS
) -> pd.DataFrame:
    """Shrunk lift of each requested pair as of ``view.cutoff``.

    ``pairs`` has ``player_a``, ``player_b`` and ``relation`` (any order of the
    two ids); other columns are carried through. With the ``independence``
    provider every lift is 1 and the evidence columns are still filled.
    """
    out = pairs.copy()
    out["player_a"], out["player_b"] = pair_key(out["player_a"], out["player_b"])
    keys = ["player_a", "player_b", "relation"]

    since = view.cutoff - pd.Timedelta(days=params.window_days)
    history = view.overlap_game(since=since)
    relation_lift: dict[str, float] = {}
    if history.empty:
        agg = pd.DataFrame(
            columns=[
                "n_games",
                "hist_shared_seconds_decayed",
                "hist_independent_seconds_decayed",
            ],
            index=pd.MultiIndex.from_arrays([[], [], []], names=keys),
        )
    else:
        age = (view.cutoff - history["game_date"]).dt.days.to_numpy(float)
        weight = 0.5 ** (age / params.half_life_days)
        history = history.assign(
            shared_w=weight * history["shared_seconds"].to_numpy(float),
            independent_w=weight * history["independent_seconds"].to_numpy(float),
        )
        agg = history.groupby(keys).agg(
            n_games=("shared_seconds", "size"),
            hist_shared_seconds_decayed=("shared_w", "sum"),
            hist_independent_seconds_decayed=("independent_w", "sum"),
        )
        totals = history.groupby("relation")[["shared_w", "independent_w"]].sum()
        relation_lift = (totals["shared_w"] / totals["independent_w"]).to_dict()
    out = out.merge(agg, left_on=keys, right_index=True, how="left")
    evidence = [
        "n_games",
        "hist_shared_seconds_decayed",
        "hist_independent_seconds_decayed",
    ]
    out[evidence] = out[evidence].apply(pd.to_numeric).fillna(0.0)
    out["n_games"] = out["n_games"].astype(int)
    # No history in the window at all: nothing to correct independence with.
    out["relation_lift"] = out["relation"].map(relation_lift).fillna(1.0)
    lift, prior_weight = shrink_lift(
        out["hist_shared_seconds_decayed"],
        out["hist_independent_seconds_decayed"],
        out["relation_lift"],
        params.k_for(out["relation"]),
    )
    if params.provider == "independence":
        lift = np.ones(len(out))
    out["lift"] = lift
    out["lift_prior_weight"] = prior_weight
    out["has_pair_history"] = out["n_games"] > 0
    extra = [c for c in pairs.columns if c not in keys]
    return out[[*extra, *LIFT_COLUMNS]]


def expected_overlap(
    minutes_a: np.ndarray | pd.Series,
    minutes_b: np.ndarray | pd.Series,
    lift: np.ndarray | pd.Series,
    game_minutes: float = 48.0,
) -> np.ndarray:
    """``min(lift * min_a * min_b / game_minutes, min_a, min_b)`` in minutes."""
    a = np.asarray(minutes_a, float)
    b = np.asarray(minutes_b, float)
    overlap = np.asarray(lift, float) * a * b / game_minutes
    return np.clip(overlap, 0.0, np.minimum(a, b))


def provider_metadata(params: OverlapParams) -> dict:
    return {"provider": params.provider, "version": VERSION, **asdict(params)}
