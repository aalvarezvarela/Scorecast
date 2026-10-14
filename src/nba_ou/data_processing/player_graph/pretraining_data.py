"""Stint-graph pretraining dataset, versioned (plan phase 6).

The phase 6 encoders (FM, then GNN) learn from one training graph per stint:
the ten players on the floor, typed edges (``stint_graph``: teammate,
opponent, guards) and six stint-level targets. This module builds, stores,
loads and checks that dataset. It is a set of tables (plan phase 2,
"Storage"); arrays are built at load time (:func:`load_stints`)::

    data/player_graph/pretraining/<VERSION>/
        stints/season=YYYY.parquet  one row per stint: keys, the ten players in
                                    node order, seconds, per-side counts, the
                                    targets and their exposures
        nodes/season=YYYY.parquet   (as_of_date, player_id) -> node features
        guard/season=YYYY.parquet   (game_id, off_player_id, def_player_id) ->
                                    expected guarding rate and its provenance
        manifest.json               inputs, parameters, exclusions, checks

**Seasons.** 2016-17 onward (decided 2026-10-14). 2016-17 has stints and box
scores but no matchup tracking: its guard edges are position-based estimates,
never observations, and it has no edge-level labels (those start in 2017-18,
``edge_targets``).

**As of the stint's date.** Everything an encoder reads is built from games
strictly before the stint's game date: node features, positions, expected
guarding rates. A game's own observed matchups are never an input (plan
principle 7): they are labels, in a separate table, and every input table
passes ``stint_graph.check_encoder_inputs``.

**Node features** (``NODE_COLUMNS``): the stored v0 node profiles
(``node_profiles/season=YYYY``) with the positions recomputed from box-score
starts as well (``positions_as_of(..., box_starts=True)``), plus
``position_starts``, the starts they rest on. v0 positions read starts only
from the matchups, so every 2016-17 player is 1/3 G / F / C there.

**Guard edges.** For attacker i and defender j on the floor together, the
rate comes from, in order (``guard_source``):

* ``pair_history`` / ``position_prior``: ``expected_guard`` v0 rebuilt on the
  pair history without the 69 corrupt matchup games
  (``v0_without_corrupt_matchups``; frozen k, half-life and window), with or
  without history of that pair (without it, ``r_hat`` is the learned
  position prior);
* ``structural``: no matchup history at all before the date (2016-17 and the
  first date of 2017-18), so ``expected_guard`` has no rate: the
  same-position rule ``r = sum_x p_i(x) p_j(x)`` on the as-of positions
  above. It has no parameter and reads no later data;
  ``structural_guard_check.py`` scores it against observed shares in
  2017-18 and 2018-19;
* ``uniform``: an attacker whose five rates are not all positive gets 1/5
  each (``stint_graph``'s fallback).

``m_ij = r_ij / sum_l r_il`` over the five defenders on the floor. The source
travels with every edge, so a model can discount position-based edges.

**Targets** (``LABELS``, per stint and offensive side; :func:`stint_targets`),
each weighted by its exposure, the denominator of the rate:

=============================  =======================
points per possession          possessions
turnovers per possession       possessions
possessions per 48             the stint's seconds
3PA / FGA                      FGA
FTA / FGA                      FGA
OREB / (OREB + opponent DREB)  rebound chances
=============================  =======================

Pace is the stint's: the mean of both sides' possessions per 48 (phase 3's
pace rows, so RAPM is scored on the same rows), stored in both sides' slots
with half the seconds each, so a loss over sides counts each stint once. A
target whose exposure is not positive is NaN with weight 0.

**Exclusions.** The two games the stint validator rejects
(``STINT_REJECTED_GAMES``) have no stints, hence no stint graph; the checks
fail if they appear. The 69 corrupt matchup games keep their stint graphs
(their stints are sound) but are out of the guard history and of every
observed-guarding label.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .as_of import CORRECTED_MATCHUP_HISTORY, AsOfView
from .expected_guard import DEFAULT_PARAMS as GUARD_PARAMS
from .expected_guard import _position_matrix
from .node_profiles import PROFILE_COLUMNS, profile_positions
from .pair_game import cofloor_seconds
from .positions import positions_as_of
from .stint_graph import (
    LABELS,
    N_NODES,
    _lineups,
    check_encoder_inputs,
    guard_pairs,
)

VERSION = "v1"
ROOT = Path("player_graph") / "pretraining"
FIRST_SEASON = 2016
#: First season with observed matchups: edge labels start here.
FIRST_MATCHUP_SEASON = 2017

#: Rejected by the stint points validator (d2d9c04): no stints, no graphs, and
#: never reintroduced into stint-based training data.
STINT_REJECTED_GAMES = frozenset({"0021800927", "0022101075"})

NODE_COLUMNS = (*PROFILE_COLUMNS, "position_starts")
SIDES = ("home", "away")
COUNTS = ("pts", "poss", "tov", "fga", "fg3a", "fta", "oreb", "dreb")

GUARD_SOURCES = ("pair_history", "position_prior", "structural", "uniform")
GUARD_COLUMNS = (
    "r_hat",
    "r_prior",
    "prior_weight",
    "n_games",
    "hist_cofloor_seconds_decayed",
    "has_pair_history",
    "r_structural",
    "guard_source",
)


def season_path(root: Path, table: str, season: int) -> Path:
    return root / table / f"season={season}.parquet"


def _dates(values) -> pd.Series:
    """Normalized ``datetime64[ns]`` dates (the stores mix resolutions)."""
    return pd.to_datetime(pd.Series(values)).dt.normalize().astype("datetime64[ns]")


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def stint_targets(stints: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, exposures)``, both ``(n, 2, len(LABELS))``; side 0 = home on
    offense. Exposures are 0 where the label is NaN (module docstring)."""
    seconds = stints["seconds"].to_numpy(float)
    poss = {side: stints[f"{side}_poss"].to_numpy(float) for side in SIDES}
    stint_poss = (poss["home"] + poss["away"]) / 2
    pace = np.where(seconds > 0, _ratio(stint_poss * 2880.0, seconds), np.nan)
    pace_weight = np.where(np.isfinite(pace), seconds / 2, 0.0)
    labels, exposures = [], []
    for offense, defense in (("home", "away"), ("away", "home")):
        col = {s: stints[f"{offense}_{s}"].to_numpy(float) for s in COUNTS}
        chances = col["oreb"] + stints[f"{defense}_dreb"].to_numpy(float)
        rates = {
            "pts_per_poss": (_ratio(col["pts"], col["poss"]), col["poss"]),
            "tov_per_poss": (_ratio(col["tov"], col["poss"]), col["poss"]),
            "poss_per_48": (pace, pace_weight),
            "fg3a_per_fga": (_ratio(col["fg3a"], col["fga"]), col["fga"]),
            "fta_per_fga": (_ratio(col["fta"], col["fga"]), col["fga"]),
            "oreb_rate": (_ratio(col["oreb"], chances), chances),
        }
        labels.append(np.stack([rates[name][0] for name in LABELS], axis=-1))
        exposures.append(np.stack([rates[name][1] for name in LABELS], axis=-1))
    label_array = np.stack(labels, axis=1)
    exposure = np.stack(exposures, axis=1)
    exposure = np.where(np.isfinite(label_array) & (exposure > 0), exposure, 0.0)
    label_array = np.where(exposure > 0, label_array, np.nan)
    return label_array, exposure


def target_columns() -> list[str]:
    """Stint-table columns of the targets and exposures, side-major."""
    return [
        f"{side}_{name}{suffix}"
        for side in SIDES
        for name in LABELS
        for suffix in ("", "_exposure")
    ]


# --------------------------------------------------------------------------
# Building one date
# --------------------------------------------------------------------------


def same_position_rate(
    off_positions: np.ndarray, def_positions: np.ndarray
) -> np.ndarray:
    """The structural guarding rate: the chance both play the same position."""
    return np.einsum("ix,ix->i", off_positions, def_positions)


def directed_pairs(stints: pd.DataFrame) -> pd.DataFrame:
    """Attacker-defender pairs that shared the floor, per game, both ways."""
    cofloor = cofloor_seconds(stints)
    keys = ["game_id", "off_player_id", "def_player_id"]
    return pd.concat(
        [
            cofloor.rename(
                columns={
                    "home_player_id": "off_player_id",
                    "away_player_id": "def_player_id",
                }
            )[keys],
            cofloor.rename(
                columns={
                    "away_player_id": "off_player_id",
                    "home_player_id": "def_player_id",
                }
            )[keys],
        ],
        ignore_index=True,
    )


def stint_table(stints: pd.DataFrame) -> pd.DataFrame:
    """The stored stint rows: keys, players in node order, counts, targets."""
    stints = stints.reset_index(drop=True)
    home = _lineups(stints, "home_lineup")
    away = _lineups(stints, "away_lineup")
    out = stints[
        [
            "game_id",
            "season_year",
            "game_date",
            "seg_idx",
            "period",
            "seconds",
            "home_team_id",
            "away_team_id",
        ]
    ].copy()
    out["game_id"] = out["game_id"].astype(str)
    out["game_date"] = _dates(out["game_date"])
    for k in range(5):
        out[f"home_{k}"] = home[:, k]
    for k in range(5):
        out[f"away_{k}"] = away[:, k]
    for side in SIDES:
        for stat in COUNTS:
            out[f"{side}_{stat}"] = stints[f"{side}_{stat}"].to_numpy()
    labels, exposures = stint_targets(stints)
    for s, side in enumerate(SIDES):
        for t, name in enumerate(LABELS):
            out[f"{side}_{name}"] = labels[:, s, t]
            out[f"{side}_{name}_exposure"] = exposures[:, s, t]
    return out


def build_day(
    view: AsOfView,
    stints: pd.DataFrame,
    expected: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(nodes, guard)`` rows of one date's stints, as of that date.

    ``expected``: ``expected_guard`` rows of the date's games (corrected
    history); ``profiles``: stored v0 node-profile rows of the date. Raises if
    a node has no profile.
    """
    positions = positions_as_of(
        view, window_days=GUARD_PARAMS.window_days, box_starts=True
    )
    players = pd.Index(
        sorted(
            {
                str(p)
                for column in ("home_lineup", "away_lineup")
                for lineup in stints[column]
                for p in lineup
            }
        ),
        name="player_id",
    )
    stored = profiles.assign(player_id=profiles["player_id"].astype(str)).set_index(
        "player_id"
    )
    missing = players.difference(stored.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} nodes on {view.cutoff.date()} have no node profile, "
            f"e.g. {list(missing[:3])}"
        )
    nodes = stored.loc[players, list(PROFILE_COLUMNS)].copy()
    as_of_positions = profile_positions(positions, players)
    nodes[list(as_of_positions.columns)] = as_of_positions
    nodes["position_starts"] = (
        positions["n_starts"].reindex(players).fillna(0).astype(int)
    )
    nodes = nodes.reset_index().assign(as_of_date=_dates([view.cutoff])[0])

    pairs = directed_pairs(stints)
    keys = ["game_id", "off_player_id", "def_player_id"]
    table = expected.assign(**{key: expected[key].astype(str) for key in keys})
    guard = pairs.merge(
        table[[*keys, *GUARD_COLUMNS[:6]]], on=keys, how="left", validate="one_to_one"
    )
    guard["r_structural"] = same_position_rate(
        _position_matrix(guard["off_player_id"], positions),
        _position_matrix(guard["def_player_id"], positions),
    )
    known = np.isfinite(guard["r_hat"].to_numpy(float))
    history = guard["has_pair_history"].eq(True).to_numpy()
    guard["guard_source"] = np.where(
        known,
        np.where(history, "pair_history", "position_prior"),
        "structural",
    )
    guard["game_date"] = _dates([view.cutoff])[0]
    columns = ["game_id", "game_date", *keys[1:], *GUARD_COLUMNS]
    return nodes[["as_of_date", "player_id", *NODE_COLUMNS]], guard[columns]


def build_season(
    data,
    season: int,
    expected: pd.DataFrame,
    profiles: pd.DataFrame,
    progress=None,
) -> dict[str, pd.DataFrame]:
    """The three tables of one season (``data``: a ``PointInTimeData`` with the
    corrected matchup history)."""
    if data.matchup_history != CORRECTED_MATCHUP_HISTORY:
        raise ValueError("The pretraining dataset reads the corrected matchup history")
    stints = data.stints.loc[data.stints["season_year"].eq(season)]
    expected = expected.assign(game_id=expected["game_id"].astype(str))
    by_game = dict(tuple(expected.groupby("game_id", sort=False)))
    profiles = profiles.assign(as_of_date=_dates(profiles["as_of_date"]).to_numpy())
    by_date = dict(tuple(profiles.groupby("as_of_date", sort=False)))
    nodes, guard = [], []
    for date, day in stints.groupby("game_date", sort=True):
        games = day["game_id"].astype(str).unique()
        day_expected = pd.concat(
            [by_game[g] for g in games if g in by_game] or [expected.iloc[:0]],
            ignore_index=True,
        )
        day_nodes, day_guard = build_day(
            data.as_of(date), day, day_expected, by_date.get(date, profiles.iloc[:0])
        )
        nodes.append(day_nodes)
        guard.append(day_guard)
        if progress is not None:
            progress(date)
    return {
        "stints": stint_table(stints),
        "nodes": pd.concat(nodes, ignore_index=True),
        "guard": pd.concat(guard, ignore_index=True),
    }


def write_season(root: Path, season: int, tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        path = season_path(root, name, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)


def read_tables(root: Path, seasons: Iterable[int]) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("stints", "nodes", "guard"):
        out[name] = pd.concat(
            [pd.read_parquet(season_path(root, name, s)) for s in sorted(seasons)],
            ignore_index=True,
        )
    return out


def write_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


# --------------------------------------------------------------------------
# Loading: tables -> arrays
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PretrainingStints:
    """Stint graphs as arrays sharing the leading ``n_stints`` axis.

    Nodes 0-4 are the home five, 5-9 the away five; guard arrays are in
    ``stint_graph.GUARD_EDGES`` order; side 0 of the targets is home on
    offense.
    """

    game_id: np.ndarray  # (n,)
    seg_idx: np.ndarray  # (n,)
    game_date: np.ndarray  # (n,) datetime64
    season: np.ndarray  # (n,)
    players: np.ndarray  # (n, 10) player ids
    seconds: np.ndarray  # (n,)
    nodes: np.ndarray  # (n, 10, len(NODE_COLUMNS)) raw node features
    guard_share: np.ndarray  # (n, 50) m_ij
    guard_source: np.ndarray  # (n, 50) int8 codes into GUARD_SOURCES
    guard_r_hat: np.ndarray  # (n, 50) expected_guard rate, 0 where unknown
    guard_prior_weight: np.ndarray  # (n, 50) 1 where unknown
    guard_log_exposure: np.ndarray  # (n, 50) log1p decayed co-floor history
    labels: np.ndarray  # (n, 2, len(LABELS))
    label_weight: np.ndarray  # (n, 2, len(LABELS)) exposures

    def __len__(self) -> int:
        return len(self.game_id)

    def subset(self, index) -> PretrainingStints:
        return PretrainingStints(
            **{f.name: getattr(self, f.name)[index] for f in fields(self)}
        )

    @classmethod
    def concat(cls, parts: Sequence[PretrainingStints]) -> PretrainingStints:
        return cls(
            **{
                f.name: np.concatenate([getattr(p, f.name) for p in parts])
                for f in fields(cls)
            }
        )


def guard_arrays(
    stints: pd.DataFrame, guard: pd.DataFrame, structural: bool = True
) -> dict[str, np.ndarray]:
    """Shares and provenance of every stint's 50 guard edges.

    ``structural=False`` gives uniform shares where ``expected_guard`` has no
    rate (``stint_graph``'s v0 behaviour), for an ablation.
    """
    n = len(stints)
    flat = guard_pairs(stints, guard, GUARD_COLUMNS)

    def grid(column: str) -> np.ndarray:
        return flat[column].to_numpy(float).reshape(n, 2, 5, 5)

    r_hat = grid("r_hat")
    r_struct = grid("r_structural")
    source = flat["guard_source"].map(
        {name: code for code, name in enumerate(GUARD_SOURCES)}
    )
    if source.isna().any():
        raise ValueError(f"{int(source.isna().sum())} stint pairs have no guard row")
    source = source.to_numpy(np.int8).reshape(n, 2, 5, 5)

    def usable(rates: np.ndarray) -> np.ndarray:
        total = rates.sum(axis=3, keepdims=True)
        return (
            np.isfinite(total) & (total > 0) & (rates >= 0).all(axis=3, keepdims=True)
        )

    from_hat = usable(r_hat)
    from_struct = (
        ~from_hat & usable(r_struct) if structural else np.zeros_like(from_hat)
    )
    rates = np.where(from_hat, r_hat, np.where(from_struct, r_struct, 1.0))
    share = rates / rates.sum(axis=3, keepdims=True)
    uniform = ~(from_hat | from_struct)
    codes = np.where(
        np.broadcast_to(uniform, source.shape),
        GUARD_SOURCES.index("uniform"),
        np.where(
            np.broadcast_to(from_struct, source.shape),
            GUARD_SOURCES.index("structural"),
            source,
        ),
    ).astype(np.int8)
    return {
        "guard_share": share.reshape(n, 50),
        "guard_source": codes.reshape(n, 50),
        "guard_r_hat": np.nan_to_num(r_hat, nan=0.0).reshape(n, 50),
        "guard_prior_weight": np.nan_to_num(grid("prior_weight"), nan=1.0).reshape(
            n, 50
        ),
        "guard_log_exposure": np.log1p(
            np.nan_to_num(grid("hist_cofloor_seconds_decayed"))
        ).reshape(n, 50),
    }


def node_array(stints: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    """``(n, 10, len(NODE_COLUMNS))``: each node's row as of its stint's date."""
    n = len(stints)
    players = np.concatenate([_player_columns(stints, side) for side in SIDES], axis=1)
    slots = pd.DataFrame(
        {
            "as_of_date": np.repeat(_dates(stints["game_date"]).to_numpy(), N_NODES),
            "player_id": players.reshape(-1),
        }
    )
    table = nodes.assign(
        as_of_date=_dates(nodes["as_of_date"]).to_numpy(),
        player_id=nodes["player_id"].astype(str),
    )
    joined = slots.merge(
        table, on=["as_of_date", "player_id"], how="left", validate="many_to_one"
    )
    values = joined[list(NODE_COLUMNS)].astype(float)
    if values["prior_weight"].isna().any():
        raise ValueError(
            f"{int(values['prior_weight'].isna().sum())} nodes lack features"
        )
    return values.to_numpy(np.float64).reshape(n, N_NODES, len(NODE_COLUMNS))


def _player_columns(stints: pd.DataFrame, side: str) -> np.ndarray:
    return stints[[f"{side}_{k}" for k in range(5)]].astype(str).to_numpy()


def to_arrays(
    tables: dict[str, pd.DataFrame], structural: bool = True
) -> PretrainingStints:
    """The loader: stored tables -> :class:`PretrainingStints`."""
    for name in ("nodes", "guard"):
        check_encoder_inputs(tables[name])
    stints = tables["stints"].sort_values(
        ["game_date", "game_id", "seg_idx"], kind="mergesort", ignore_index=True
    )
    players = np.concatenate([_player_columns(stints, s) for s in SIDES], axis=1)
    lineups = stints.assign(
        home_lineup=list(players[:, :5]), away_lineup=list(players[:, 5:])
    )
    guard = guard_arrays(lineups, tables["guard"], structural=structural)
    labels = np.stack(
        [
            stints[[f"{side}_{name}" for name in LABELS]].to_numpy(float)
            for side in SIDES
        ],
        axis=1,
    )
    weights = np.stack(
        [
            stints[[f"{side}_{name}_exposure" for name in LABELS]].to_numpy(float)
            for side in SIDES
        ],
        axis=1,
    )
    return PretrainingStints(
        game_id=stints["game_id"].astype(str).to_numpy(),
        seg_idx=stints["seg_idx"].to_numpy(),
        game_date=_dates(stints["game_date"]).to_numpy(),
        season=stints["season_year"].to_numpy(),
        players=players,
        seconds=stints["seconds"].to_numpy(float),
        nodes=node_array(stints, tables["nodes"]),
        labels=labels,
        label_weight=weights,
        **guard,
    )


def load_stints(
    root: Path,
    seasons: Iterable[int],
    *,
    since=None,
    until=None,
    structural: bool = True,
) -> PretrainingStints:
    """Stints of ``seasons`` dated in ``[since, until)``, as arrays."""
    parts = []
    for season in sorted(seasons):
        tables = read_tables(root, [season])
        dates = pd.to_datetime(tables["stints"]["game_date"])
        keep = pd.Series(True, index=dates.index)
        if since is not None:
            keep &= dates >= pd.Timestamp(since)
        if until is not None:
            keep &= dates < pd.Timestamp(until)
        if not keep.any():
            continue
        tables["stints"] = tables["stints"].loc[keep]
        parts.append(to_arrays(tables, structural=structural))
    if not parts:
        raise ValueError("No stints in the requested seasons and dates")
    return PretrainingStints.concat(parts)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_tables(tables: dict[str, pd.DataFrame], excluded_matchups: set[str]) -> dict:
    """Structural and temporal checks of stored tables; raises on a failure,
    returns the counts it checked."""
    stints, nodes, guard = tables["stints"], tables["nodes"], tables["guard"]
    failures = []
    players = np.concatenate([_player_columns(stints, s) for s in SIDES], axis=1)
    distinct = np.array([len(set(row)) for row in players])
    if (distinct != N_NODES).any():
        failures.append(
            f"{int((distinct != N_NODES).sum())} stints without ten distinct players"
        )
    rejected = STINT_REJECTED_GAMES.intersection(stints["game_id"].astype(str))
    if rejected:
        failures.append(f"stint-rejected games present: {sorted(rejected)}")
    for name in ("nodes", "guard"):
        try:
            check_encoder_inputs(tables[name])
        except ValueError as error:
            failures.append(f"{name}: {error}")
    # Every node read as of its stint's date, every pair's guard row of that date.
    slots = pd.DataFrame(
        {
            "as_of_date": np.repeat(_dates(stints["game_date"]).to_numpy(), N_NODES),
            "player_id": players.reshape(-1),
        }
    ).drop_duplicates()
    found = slots.merge(
        nodes[["player_id"]].assign(
            as_of_date=_dates(nodes["as_of_date"]).to_numpy(), found=True
        ),
        on=["as_of_date", "player_id"],
        how="left",
    )["found"]
    if found.isna().any():
        failures.append(
            f"{int(found.isna().sum())} nodes without features as of their date"
        )
    lineups = stints.assign(
        home_lineup=list(players[:, :5]), away_lineup=list(players[:, 5:])
    )
    pairs = directed_pairs(lineups).merge(
        guard[["game_id", "off_player_id", "def_player_id", "game_date"]],
        on=["game_id", "off_player_id", "def_player_id"],
        how="left",
    )
    if pairs["game_date"].isna().any():
        failures.append(
            f"{int(pairs['game_date'].isna().sum())} stint pairs without a guard row"
        )
    game_dates = stints.drop_duplicates("game_id").set_index("game_id")["game_date"]
    guard_dates = _dates(guard["game_date"])
    if not guard_dates.eq(_dates(guard["game_id"].map(game_dates)).to_numpy()).all():
        failures.append("guard rows dated differently from their game")
    # No rate at all is a property of the date (no matchup history before it):
    # a structural rate never stands in for a missing pair on a covered date.
    structural = guard["guard_source"].eq("structural")
    mixed = structural.groupby(guard_dates.to_numpy()).agg(["any", "all"])
    if (mixed["any"] & ~mixed["all"]).any():
        failures.append("structural rates on dates with matchup history")
    labels = stints[target_columns()]
    exposure = labels[[c for c in labels.columns if c.endswith("_exposure")]]
    if (exposure.to_numpy() < 0).any():
        failures.append("negative exposures")
    for column in exposure.columns:
        label = labels[column.removesuffix("_exposure")]
        if (label.notna() != exposure[column].gt(0)).any():
            failures.append(f"{column}: labels and exposures disagree")
    recomputed, _ = stint_targets(stints)
    stored = np.stack(
        [
            stints[[f"{side}_{name}" for name in LABELS]].to_numpy(float)
            for side in SIDES
        ],
        axis=1,
    )
    if not np.allclose(recomputed, stored, equal_nan=True):
        failures.append("stored targets differ from the counts")
    if failures:
        raise ValueError("Pretraining tables failed checks: " + "; ".join(failures))
    return {
        "stints": len(stints),
        "games": int(stints["game_id"].nunique()),
        "node_rows": len(nodes),
        "guard_rows": len(guard),
        "corrupt_matchup_games_with_stints": len(
            excluded_matchups.intersection(stints["game_id"].astype(str))
        ),
        "guard_source_share": guard["guard_source"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
    }
