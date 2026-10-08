"""Can we learn who absorbs an absent player's minutes? (phase 4A, B and C)

Runs ``player_graph.rotation.walk_forward`` over 2016-17 -> 2018-19 (each team
game from earlier games only) and checks, on the development season 2018-19:

1. the expanded roster's coverage of the players who actually play, against
   2_6's 10-game roster window;
2. B: baseline minutes ``b`` (given the player plays, absences netted out) in
   team-games without any absence event, against the mean of the last 10
   games played;
3. C: the minutes each available teammate gains over ``b`` when someone is
   out, predicted as ``sum_X V_X * s(X -> Y)`` with three share rules -- the
   proportional split ``allocate_minutes`` makes, the structural prior
   ``s0`` alone, and ``s0`` plus each pair's own history -- and whether the
   rule picks the teammate who absorbs the most.

Absences are the realized ones (on the roster, did not play, ``V_X >= 10``):
this measures the structure, not the injury report's accuracy.

    python scripts/player_graph/rotation_diagnostic.py

Development season only. Reads the stint store and node profiles; the
database for the injury reports used as roster evidence.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.positions import PROBABILITY_COLUMNS
from nba_ou.data_processing.player_graph.rotation import (
    DEFAULT_PARAMS,
    team_game_minutes,
    walk_forward,
)

SEASON = 2018
FIRST = 2016
RULES = ("proportional", "s0", "s")


def listed_players() -> dict[tuple[str, str], set[str]]:
    from nba_ou.data_processing.injury_status.report_state import (
        load_injury_report_state,
    )

    statuses = load_injury_report_state().statuses
    out: dict[tuple[str, str], set[str]] = {}
    for g, t, p in statuses[["game_id", "team_id", "player_id"]].itertuples(
        index=False
    ):
        out.setdefault((str(g).zfill(10), str(t)), set()).add(str(p))
    return out


def predicted_gains(absorption: pd.DataFrame) -> pd.DataFrame:
    """Per team-game and teammate: actual gain and the gain each rule predicts
    (summed over that game's absentees)."""
    work = absorption.copy()
    for rule in RULES:
        work[f"pred_{rule}"] = work["vacated"] * work[rule]
    return work.groupby(["game_id", "team_id", "player_id"], as_index=False).agg(
        gain=("gain", "first"),
        played=("played", "first"),
        n_absent=("n_absent", "first"),
        evidence=("pair_evidence", "max"),
        **{f"pred_{rule}": (f"pred_{rule}", "sum") for rule in RULES},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    started = time.time()
    stints = read_stints(list(range(FIRST, SEASON + 1)), local_root=args.local_root)
    minutes = team_game_minutes(stints)
    profiles = pd.concat(
        [
            pd.read_parquet(
                args.local_root
                / "player_graph"
                / "node_profiles"
                / f"season={s}.parquet",
                columns=["as_of_date", "player_id", *PROBABILITY_COLUMNS],
            )
            for s in range(FIRST, SEASON + 1)
        ],
        ignore_index=True,
    )
    lookup = {
        (d, p): np.array(v)
        for d, p, *v in profiles[
            ["as_of_date", "player_id", *PROBABILITY_COLUMNS]
        ].itertuples(index=False)
    }

    def positions(date, player):
        return lookup.get((date, player), np.full(3, 1 / 3))

    listed = listed_players()
    print(f"inputs in {time.time() - started:.0f}s; params {DEFAULT_PARAMS}")
    started = time.time()
    players, absorption = walk_forward(minutes, positions, listed)
    print(f"walk-forward in {time.time() - started:.0f}s")
    season_games = set(minutes.loc[minutes["season"].eq(SEASON), "game_id"])
    players = players.loc[players["game_id"].isin(season_games)]
    absorption = absorption.loc[absorption["game_id"].isin(season_games)]
    pd.set_option("display.width", 250, "display.max_columns", 30)

    # 1. Roster coverage.
    team_games = players.groupby(["game_id", "team_id"]).ngroups
    played = players.loc[players["minutes"] > 0]
    off = played.loc[~played["on_roster"]]
    history = minutes.sort_values(["game_date", "game_id"])
    order = history.drop_duplicates(["game_id", "team_id"]).copy()
    order["team_game"] = order.groupby("team_id").cumcount()
    index = order.set_index(["game_id", "team_id"])["team_game"]
    last = {}
    window_off = []
    for row in history.itertuples(index=False):
        game_no = index[(row.game_id, row.team_id)]
        previous = last.get((row.team_id, row.player_id))
        if row.season == SEASON and (previous is None or game_no - previous > 10):
            window_off.append(row.minutes)
        last[(row.team_id, row.player_id)] = game_no
    print("\n1. ROSTER COVERAGE of players who played (2018-19)")
    print(
        f"  expanded roster: {1 - len(off) / len(played):.2%} of player-games, "
        f"{off['minutes'].sum() / team_games:.2f} min per team-game off the roster "
        f"({len(off)} player-games); 10-game window: "
        f"{sum(window_off) / team_games:.2f} min per team-game off "
        f"({len(window_off)} player-games)"
    )
    first_seen = minutes.groupby("player_id")["game_date"].min()
    off = off.assign(
        kind=np.where(
            off["game_date"].eq(off["player_id"].map(first_seen)),
            "first game in the data",
            "new to this team",
        )
    )
    print(
        off.groupby("kind")["minutes"].agg(["size", "mean", "sum"]).round(1).to_string()
    )

    # 2. Baseline minutes in games without absence events.
    clean = played.loc[played["n_absent"].eq(0) & played["b"].notna()].copy()
    recent = {}
    last10 = []
    for row in history.itertuples(index=False):
        key = (row.team_id, row.player_id)
        values = recent.setdefault(key, [])
        last10.append(np.mean(values[-10:]) if values else np.nan)
        values.append(row.minutes)
    history["mean_last10_played"] = last10
    clean = clean.merge(
        history[["game_id", "team_id", "player_id", "mean_last10_played"]],
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    print("\n2. BASELINE b in team-games without absence events (players who played)")
    for name, column in (("b", "b"), ("mean of last 10 played", "mean_last10_played")):
        error = (clean[column] - clean["minutes"]).dropna()
        print(
            f"  {name:24s} MAE {error.abs().mean():.3f}, bias {error.mean():+.3f}, n {len(error):,}"
        )

    # 3. Redistribution.
    gains = predicted_gains(absorption)
    on = gains.loc[gains["played"]]
    print(
        f"\n3. REDISTRIBUTION: {absorption.groupby(['game_id', 'team_id']).ngroups} "
        f"team-games with an absence event, {absorption[['game_id', 'team_id', 'absent']].drop_duplicates().shape[0]} "
        f"absences, {len(on):,} teammate-games played"
    )
    rows = {}
    for label, part in (
        ("all", on),
        ("one absence", on.loc[on["n_absent"].eq(1)]),
        ("2+ absences", on.loc[on["n_absent"].ge(2)]),
        ("pair history >= 50 vacated min", on.loc[on["evidence"].ge(50)]),
        ("no pair history", on.loc[on["evidence"].eq(0)]),
    ):
        line = {"teammate-games": len(part)}
        for rule in RULES:
            error = part[f"pred_{rule}"] - part["gain"]
            line[f"MSE {rule}"] = (error**2).mean()
            line[f"corr {rule}"] = np.corrcoef(part[f"pred_{rule}"], part["gain"])[0, 1]
        line["var(gain)"] = part["gain"].var()
        rows[label] = line
    print(pd.DataFrame(rows).T.round(3).to_string())
    noise = played.loc[played["n_absent"].eq(0) & played["b"].notna()]
    print(
        f"  ordinary variation: var(minutes - b) without absences {((noise['minutes'] - noise['b']) ** 2).mean():.2f}"
    )

    single = absorption.loc[absorption["n_absent"].eq(1) & absorption["vacated"].ge(20)]
    hits = {rule: [] for rule in RULES}
    captured = {rule: [] for rule in RULES}
    taken = []
    for _, event in single.groupby(["game_id", "team_id"]):
        actual = event.loc[event["played"]]
        if actual.empty:
            continue
        top = actual.loc[actual["gain"].idxmax()]
        taken.append(top["gain"] / top["vacated"])
        for rule in RULES:
            if event[rule].isna().all():
                continue
            pick = event.loc[event[rule].idxmax()]
            hits[rule].append(pick["player_id"] == top["player_id"])
            captured[rule].append(top[rule])
    print(
        f"  top absorber, {len(taken)} single-absence events (V >= 20): he takes "
        f"{np.median(taken):.0%} of V (median)"
    )
    for rule in RULES:
        print(
            f"    {rule:12s} picks him {np.mean(hits[rule]):.1%}; share it gives him "
            f"{np.median(captured[rule]):.0%} (median)"
        )
    out = args.local_root / "player_graph" / "rotation"
    out.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out / f"players_season={SEASON}.parquet", index=False)
    absorption.to_parquet(out / f"absorption_season={SEASON}.parquet", index=False)


if __name__ == "__main__":
    main()
