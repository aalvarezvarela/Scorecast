"""Minutes provider v1 vs v0 on 2018-19 (phase 4A, controlled environment).

Every version sees the same scenario, the rotation engine's realized
absences (a rostered player who did not play and vacates >= 10 min), so the
comparison isolates how minutes are projected, not the injury report:

1. v0: 2_6's ``game_nights`` (10-game roster, recent averages) and
   ``allocate_minutes`` with those absentees sitting;
2. v1 raw: ``q * (b + C)`` over tonight's available players;
3. v1 reconciled: ``minutes_provider.reconcile`` (0-48, sum 240);
4. actual regulation minutes.

Reports the reconciliation (raw sums, scale ``c``, minutes moved, largest
single change, 48-minute caps, extreme cases), the player MAE and the
misallocated minutes per team-game by slice, and whether the reconciliation
keeps C's redistribution (correlation of projected and actual change over
the baseline for teammates of the absentees).

    python scripts/player_graph/minutes_provider_diagnostic.py

Needs the database (2_6 rosters, injury reports for the q labels).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.availability import player_out_probabilities
from nba_ou.data_processing.lineups.features import game_nights
from nba_ou.data_processing.lineups.game_projection import allocate_minutes
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.minutes_provider import reconcile
from nba_ou.data_processing.player_graph.participation import (
    labelled_rows,
    walk_forward_q,
)
from nba_ou.data_processing.player_graph.positions import PROBABILITY_COLUMNS
from nba_ou.data_processing.player_graph.rotation import team_game_minutes, walk_forward

SEASON = 2018
FIRST = 2016
VERSIONS = ("v0", "v1_raw", "v1")


def misallocated(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["game_id", "team_id"]
    per_team = pd.DataFrame(
        {
            v: (frame[v] - frame["actual"])
            .abs()
            .groupby([frame[k] for k in keys])
            .sum()
            / 2
            for v in VERSIONS
        }
    )
    if by is None:
        return per_team.mean().to_frame("all").T
    labels = frame.groupby(keys)[by].first()
    return per_team.join(labels).groupby(by, observed=True)[list(VERSIONS)].mean()


def player_mae(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    errors = pd.DataFrame({v: (frame[v] - frame["actual"]).abs() for v in VERSIONS})
    if by is None:
        out = errors.mean().to_frame("all").T
        out["rows"] = len(frame)
        return out
    out = errors.groupby(frame[by], observed=True).mean()
    out["rows"] = frame.groupby(by, observed=True).size()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    started = time.time()
    data = PointInTimeData.load(range(FIRST, SEASON + 1), closing_injuries=True)
    state = data.injury_report("closing")
    p_out = player_out_probabilities(state.statuses)
    covered = {(str(g).zfill(10), str(t)) for g, t in state.covered}
    listed: dict[tuple[str, str], set[str]] = {}
    for g, t, p in state.statuses[["game_id", "team_id", "player_id"]].itertuples(
        index=False
    ):
        listed.setdefault((str(g).zfill(10), str(t)), set()).add(str(p))
    minutes = team_game_minutes(data.stints)
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
    lookup = {(d, p): np.array(v) for d, p, *v in profiles.itertuples(index=False)}
    players, _ = walk_forward(
        minutes, lambda d, p: lookup.get((d, p), np.full(3, 1 / 3)), listed
    )
    rows = labelled_rows(players, p_out, covered)
    season_games = set(minutes.loc[minutes["season"].eq(SEASON), "game_id"])
    players = players.loc[players["game_id"].isin(season_games)]
    available = players.loc[
        players["on_roster"] & ~players["absent_event"] & players["b"].notna()
    ].copy()
    available["q_hat"] = walk_forward_q(rows, available)
    print(f"engine and q in {time.time() - started:.0f}s")

    # v1 raw and reconciled.
    available["v1_raw"] = available["q_hat"] * (available["b"] + available["gain_c"])
    recon = []
    for (game_id, team), part in available.groupby(["game_id", "team_id"]):
        result = reconcile(part["v1_raw"].to_numpy())
        recon.append(
            pd.DataFrame(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "player_id": part["player_id"].to_numpy(),
                    "v1": result.minutes,
                    "scale": result.scale,
                    "feasible": result.feasible,
                }
            )
        )
    recon = pd.concat(recon, ignore_index=True)

    # v0 under the same realized absences.
    games = (
        minutes.loc[minutes["season"].eq(SEASON)]
        .groupby("game_id")
        .agg(
            GAME_DATE=("game_date", "first"),
            teams=("team_id", lambda t: sorted(set(t))),
        )
    )
    stints = data.stints.loc[data.stints["season_year"].eq(SEASON)].drop_duplicates(
        "game_id"
    )
    home = stints.set_index("game_id")[["home_team_id", "away_team_id"]].astype(str)
    games = games.join(home).reset_index()
    games = games.rename(
        columns={
            "home_team_id": "HOME_TEAM_ID",
            "away_team_id": "AWAY_TEAM_ID",
            "game_id": "GAME_ID",
        }
    )
    nights = game_nights(
        games[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"]],
        data.box_scores_2_6,
    )
    absent = (
        players.loc[players["absent_event"]]
        .groupby(["game_id", "team_id"])["player_id"]
        .apply(set)
        .to_dict()
    )
    v0_rows = []
    for (game_id, team), roster in nights.items():
        allocation = allocate_minutes(
            roster, frozenset(absent.get((game_id, team), set()))
        )
        v0_rows.extend((game_id, team, p, m) for p, m in allocation.items())
    v0 = pd.DataFrame(v0_rows, columns=["game_id", "team_id", "player_id", "v0"])

    actual = minutes.loc[
        minutes["season"].eq(SEASON),
        ["game_id", "team_id", "player_id", "minutes", "started"],
    ]
    keys = ["game_id", "team_id", "player_id"]
    table = (
        actual.rename(columns={"minutes": "actual"})
        .merge(v0, on=keys, how="outer")
        .merge(available[[*keys, "v1_raw", "q_hat"]], on=keys, how="outer")
        .merge(recon[[*keys, "v1"]], on=keys, how="outer")
        .merge(
            players[
                [*keys, "b", "rank", "streak", "absent_event", "n_absent", "on_roster"]
            ],
            on=keys,
            how="left",
        )
    )
    team_games = set(
        zip(available["game_id"], available["team_id"], strict=True)
    ) & set(zip(v0["game_id"], v0["team_id"], strict=True))
    table = table.loc[
        [k in team_games for k in zip(table["game_id"], table["team_id"], strict=True)]
    ]
    for column in ("actual", *VERSIONS):
        table[column] = table[column].fillna(0.0)
    table["started"] = table["started"].eq(True)
    importance = (
        players.loc[players["absent_event"]].groupby(["game_id", "team_id"])["b"].max()
    )
    table["importance"] = pd.cut(
        [
            importance.get((g, t), 0.0)
            for g, t in zip(table["game_id"], table["team_id"], strict=True)
        ],
        [-1, 0, 15, 25, 32, 99],
        labels=["nobody out", "< 15", "15-25", "25-32", "> 32"],
    )
    table["n_absent"] = table["n_absent"].fillna(0)
    table["absences"] = pd.cut(
        table.groupby(["game_id", "team_id"])["n_absent"].transform("max"),
        [-1, 0, 1, 99],
        labels=["0", "1", "2+"],
    )
    table["role"] = np.where(table["started"], "starter", "bench")
    table["returning"] = np.where(
        table["streak"].fillna(0) >= 5, "returning (>= 5 out)", "other"
    )
    table["depth"] = np.where(
        table["rank"].fillna(99) >= 11, "deep bench (rank 11+)", "rank <= 10"
    )
    print(
        f"{table.groupby(['game_id', 'team_id']).ngroups} team-games, {len(table):,} player rows"
    )

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n1. RECONCILIATION")
    per_team = (
        available.merge(recon, on=keys)
        .groupby(["game_id", "team_id"])
        .agg(
            raw=("v1_raw", "sum"),
            scale=("scale", "first"),
            feasible=("feasible", "first"),
            moved=("v1_raw", lambda s: 0.0),
        )
    )
    merged = available.merge(recon, on=keys)
    merged["change"] = merged["v1"] - merged["v1_raw"]
    per_team["moved"] = (
        merged["change"].abs().groupby([merged["game_id"], merged["team_id"]]).sum()
    )
    per_team["largest"] = (
        merged["change"].abs().groupby([merged["game_id"], merged["team_id"]]).max()
    )
    per_team["capped"] = (
        (merged["v1"] >= 48 - 1e-9)
        .groupby([merged["game_id"], merged["team_id"]])
        .sum()
    )
    print(
        per_team[["raw", "scale", "moved", "largest", "capped"]]
        .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
        .round(3)
        .to_string()
    )
    print(
        f"  infeasible team-games: {(~per_team['feasible']).sum()}; with a player at the "
        f"48 cap: {(per_team['capped'] > 0).mean():.2%}; scale outside 0.85-1.15: "
        f"{((per_team['scale'] < 0.85) | (per_team['scale'] > 1.15)).mean():.2%}"
    )
    extreme = per_team.loc[
        (per_team["scale"] < 0.8) | (per_team["scale"] > 1.25)
    ].sort_values("scale")
    print(f"  extreme cases (scale < 0.8 or > 1.25): {len(extreme)}")
    print(extreme.head(8).round(2).to_string())

    print("\n2. PLAYER MAE (rows: anyone projected or playing)")
    print(player_mae(table).round(3).to_string())
    for by in ("role", "absences", "importance", "returning", "depth"):
        print(player_mae(table, by).round(3).to_string())
    print("\n3. MISALLOCATED MINUTES PER TEAM-GAME (sum |projected - actual| / 2)")
    print(misallocated(table).round(2).to_string())
    for by in ["absences", "importance"]:
        print(misallocated(table, [by]).round(2).to_string())

    print(
        "\n4. C's REDISTRIBUTION AFTER RECONCILIATION (teammates of absentees who played)"
    )
    mates = table.loc[
        table["n_absent"].gt(0)
        & ~table["absent_event"].eq(True)
        & table["actual"].gt(0)
        & table["b"].notna()
    ]
    for version in VERSIONS:
        change = mates[version] - mates["b"]
        real = mates["actual"] - mates["b"]
        print(
            f"  {version:7s} corr(projected - b, actual - b) {np.corrcoef(change, real)[0, 1]:.3f}; "
            f"minutes MSE {((mates[version] - mates['actual']) ** 2).mean():.2f}"
        )


if __name__ == "__main__":
    main()
