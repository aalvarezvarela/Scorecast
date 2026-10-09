"""True pre-game evaluation of the minutes providers on 2018-19 (phase 4A).

For every game whose two teams had filed a closing injury report, both
providers project minutes for the report's scenarios (``p_out`` per player,
``enumerate_scenarios``) and the expected minutes are the scenario-weighted
mean:

* v0: 2_6's ``game_nights`` and ``allocate_minutes``;
* v1: ``minutes_provider.ScenarioProvider`` on the rotation engine's pre-game
  state (expanded roster, B, C, the month's q model fitted on earlier months,
  reconciliation).

Reports:

0. integration check: v0's scenario graphs reproduce 2_6's closing file
   (absence impact and possessions) with 2_6's rating cache;
1. expected minutes vs actual regulation minutes, player MAE and misallocated
   minutes per team-game, by slice, including whether the report's most
   likely scenario matched who sat;
2. how much each provider loses from the realized absences (the controlled
   evaluation) to the report's scenarios: the error owed to availability;
3. the total: R1 from each provider's scenario graphs and its absence impact
   against the same provider's full-health graph, calibrated with
   ``walk_forward_offset``; ratings are the clean refit of 2_6's penalties,
   the same for both providers.

    python scripts/player_graph/pregame_minutes_evaluation.py
    python scripts/player_graph/pregame_minutes_evaluation.py --evaluation --seasons 2019-2025

Development season 2018-19 by default (R1 with the clean refit of 2_6's
penalties, which covers only 2018-19). ``--evaluation`` is required for any
later season: the one-pass check of the frozen provider, with R1 on 2_6's
rating cache for both providers. Needs the database.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import (
    game_nights,
    game_phase,
    load_rating_book,
    walk_forward_offset,
)
from nba_ou.data_processing.lineups.game_projection import allocate_minutes
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.game_graph import readout_2_6
from nba_ou.data_processing.player_graph.minutes_provider import (
    FULL_HEALTH,
    ScenarioProvider,
    expected_minutes,
    joint_graphs,
    reconcile,
    team_scenarios_v0,
)
from nba_ou.data_processing.player_graph.participation import (
    labelled_rows,
    monthly_models,
    walk_forward_q,
)
from nba_ou.data_processing.player_graph.positions import PROBABILITY_COLUMNS
from nba_ou.data_processing.player_graph.rotation import (
    DEFAULT_PARAMS,
    team_game_minutes,
    walk_forward,
)

SEASON = 2018
FIRST = 2016
CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
CLEAN_RATINGS = Path("data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet")
KEYS = ["game_id", "team_id", "player_id"]


def mae(frame: pd.DataFrame, columns, by=None) -> pd.DataFrame:
    errors = pd.DataFrame({c: (frame[c] - frame["actual"]).abs() for c in columns})
    if by is None:
        out = errors.mean().to_frame("all").T
        out["rows"] = len(frame)
        return out
    out = errors.groupby(frame[by], observed=True).mean()
    out["rows"] = frame.groupby(by, observed=True).size()
    return out


def misallocated(frame: pd.DataFrame, columns, by=None) -> pd.DataFrame:
    per = pd.DataFrame(
        {
            c: (frame[c] - frame["actual"])
            .abs()
            .groupby([frame["game_id"], frame["team_id"]])
            .sum()
            / 2
            for c in columns
        }
    )
    if by is None:
        return per.mean().to_frame("all").T
    labels = frame.groupby(["game_id", "team_id"])[by].first()
    return per.join(labels).groupby(by, observed=True)[list(columns)].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--roster-absence-games",
        type=int,
        default=DEFAULT_PARAMS.roster_absence_games,
        help="Team games out before a player needs a recent report listing; 0 = no limit",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--seasons", default=str(SEASON), help="2018 or 2019-2025")
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Required past 2018-19: the frozen provider's one-pass evaluation",
    )
    args = parser.parse_args()
    first_season, _, last_season = args.seasons.partition("-")
    seasons = list(range(int(first_season), int(last_season or first_season) + 1))
    if max(seasons) > SEASON and not args.evaluation:
        raise SystemExit("Seasons after 2018-19 need --evaluation")
    ratings_path = CLEAN_RATINGS if seasons == [SEASON] else None
    label = f"{seasons[0]}" if len(seasons) == 1 else f"{seasons[0]}_{seasons[-1]}"
    if args.evaluation:
        print(
            "EVALUATION: frozen 4A v1 provider, one pass, nothing is selected on these seasons"
        )
    params = replace(
        DEFAULT_PARAMS, roster_absence_games=args.roster_absence_games or None
    )
    print(
        f"roster rule: {params.roster_absence_games} games, report recency "
        f"{params.report_recency_games}"
    )
    started = time.time()
    data = PointInTimeData.load(range(FIRST, max(seasons) + 1), closing_injuries=True)
    state = data.injury_report("closing")
    p_out = player_out_probabilities(state.statuses)
    excluded = roster_exclusions(state.statuses)
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
            for s in range(FIRST, max(seasons) + 1)
        ],
        ignore_index=True,
    )
    lookup = {(d, p): np.array(v) for d, p, *v in profiles.itertuples(index=False)}

    def positions(d, p):
        return lookup.get((d, p), np.full(3, 1 / 3))

    # Pass 1: the engine's rows train q month by month.
    players, _ = walk_forward(minutes, positions, listed, params)
    rows = labelled_rows(players, p_out, covered)
    season_minutes = minutes.loc[minutes["season"].isin(seasons)]
    months = sorted(season_minutes["game_date"].dt.to_period("M").unique())
    models = monthly_models(rows, months)
    # Pass 2: the same engine feeds the scenario provider.
    provider = ScenarioProvider(p_out, excluded, covered, models)
    walk_forward(minutes, positions, listed, params, on_team_game=provider)
    print(f"engine passes in {time.time() - started:.0f}s")

    first = data.stints.loc[data.stints["season_year"].isin(seasons)].drop_duplicates(
        "game_id"
    )
    games = pd.DataFrame(
        {
            "GAME_ID": first["game_id"].astype(str),
            "GAME_DATE": pd.to_datetime(first["game_date"]).dt.normalize(),
            "HOME_TEAM_ID": first["home_team_id"].astype(str),
            "AWAY_TEAM_ID": first["away_team_id"].astype(str),
        }
    )
    games = games.loc[
        [
            (g, h) in covered and (g, a) in covered
            for g, h, a in zip(
                games["GAME_ID"],
                games["HOME_TEAM_ID"],
                games["AWAY_TEAM_ID"],
                strict=True,
            )
        ]
    ]
    nights = game_nights(games, data.box_scores_2_6, p_out, excluded=excluded)
    v0_scen, v0_nodes = team_scenarios_v0(nights)
    v0_graphs = joint_graphs(v0_scen, v0_nodes, games)
    v1_graphs = joint_graphs(provider.team_scenarios, provider.team_nodes, games)
    both = set(v0_graphs.scenarios["game_id"]) & set(v1_graphs.scenarios["game_id"])
    print(
        f"{len(games)} covered games; graphs: v0 {v0_graphs.scenarios['game_id'].nunique()}, v1 {v1_graphs.scenarios['game_id'].nunique()}, both {len(both)}"
    )

    print("\n0. INTEGRATION: v0 scenario graphs vs the 2_6 closing file (2_6 ratings)")
    stored = readout_2_6(v0_graphs, load_rating_book()).set_index("GAME_ID")
    file = pd.read_parquet(
        CLOSING_2_6,
        columns=["GAME_ID", "LU_ABSENCE_IMPACT_PTS_BEFORE", "LU_PROJ_POSS_BEFORE"],
    )
    file["GAME_ID"] = file["GAME_ID"].astype(str).str.zfill(10)
    file = file.set_index("GAME_ID").reindex(stored.index)
    print(
        f"  {len(stored)} games; max |impact diff| "
        f"{(stored['impact_points'] - file['LU_ABSENCE_IMPACT_PTS_BEFORE']).abs().max():.2e}, "
        f"max |possessions diff| {(stored['possessions'] - file['LU_PROJ_POSS_BEFORE']).abs().max():.2e}"
    )

    # Expected minutes vs actual.
    v0_exp = expected_minutes(v0_scen, v0_nodes).rename(columns={"expected": "v0"})
    v1_exp = expected_minutes(provider.team_scenarios, provider.team_nodes).rename(
        columns={"expected": "v1"}
    )
    actual = season_minutes[
        ["game_id", "team_id", "player_id", "minutes", "started"]
    ].rename(columns={"minutes": "actual"})
    table = actual.merge(v0_exp, on=KEYS, how="outer").merge(
        v1_exp, on=KEYS, how="outer"
    )
    table = table.loc[table["game_id"].isin(both)]
    season_players = players.loc[players["game_id"].isin(both)]
    table = table.merge(
        season_players[[*KEYS, "b", "rank", "streak", "absent_event"]],
        on=KEYS,
        how="left",
    )

    # Controlled (realized absences) for the same team-games.
    available = season_players.loc[
        season_players["on_roster"]
        & ~season_players["absent_event"]
        & season_players["b"].notna()
    ].copy()
    available["q_hat"] = walk_forward_q(rows, available)
    available["raw"] = available["q_hat"] * (available["b"] + available["gain_c"])
    realized_v1 = []
    for (g, t), part in available.dropna(subset=["raw"]).groupby(
        ["game_id", "team_id"]
    ):
        realized_v1.extend(
            zip(
                [g] * len(part),
                [t] * len(part),
                part["player_id"],
                reconcile(part["raw"].to_numpy()).minutes,
                strict=True,
            )
        )
    realized_v1 = pd.DataFrame(realized_v1, columns=[*KEYS, "v1_realized"])
    absent = (
        season_players.loc[season_players["absent_event"]]
        .groupby(["game_id", "team_id"])["player_id"]
        .apply(set)
        .to_dict()
    )
    realized_v0 = []
    plain = game_nights(games, data.box_scores_2_6)
    for (g, t), roster in plain.items():
        for p, m in allocate_minutes(
            roster, frozenset(absent.get((g, t), set()))
        ).items():
            realized_v0.append((g, t, p, m))
    realized_v0 = pd.DataFrame(realized_v0, columns=[*KEYS, "v0_realized"])
    table = table.merge(realized_v1, on=KEYS, how="left").merge(
        realized_v0, on=KEYS, how="left"
    )
    columns = ("v0_realized", "v1_realized", "v0", "v1")
    for column in ("actual", *columns):
        table[column] = table[column].fillna(0.0)
    table["started"] = table["started"].eq(True)

    # Slices.
    team_p = {}
    for (g, t, p), value in p_out.items():
        team_p.setdefault((g, t), []).append((p, value))

    def team_label(g, t):
        values = [v for _, v in team_p.get((g, t), [])]
        return (
            "with questionable / doubtful"
            if any(0.1 < v < 0.9 for v in values)
            else "no uncertainty"
        )

    def n_out(g, t):
        return sum(v >= 0.9 for _, v in team_p.get((g, t), []))

    keys = list(zip(table["game_id"], table["team_id"], strict=True))
    table["uncertainty"] = [team_label(g, t) for g, t in keys]
    table["report_out"] = pd.cut(
        [n_out(g, t) for g, t in keys],
        [-1, 0, 1, 99],
        labels=["0 out", "1 out", "2+ out"],
    )
    baseline = season_players.set_index(KEYS)["b"].to_dict()

    def key_out(g, t):
        outs = [
            baseline.get((g, t, p), 0.0) for p, v in team_p.get((g, t), []) if v >= 0.9
        ]
        return (
            "key player out (b >= 25)"
            if outs and max(outs) >= 25
            else "no key player out"
        )

    table["key_out"] = [key_out(g, t) for g, t in keys]
    table["role"] = np.where(table["started"], "starter", "bench")
    table["returning"] = np.where(
        table["streak"].fillna(0) >= 5, "returning (>= 5 out)", "other"
    )
    table["depth"] = np.where(
        table["rank"].fillna(99) >= 11, "deep bench (rank 11+)", "rank <= 10"
    )
    # Did the report's most likely scenario match who sat (among listed players)?
    played = set(
        zip(actual["game_id"], actual["team_id"], actual["player_id"], strict=True)
    )
    v1_scen = pd.DataFrame(provider.team_scenarios)
    likely = (
        v1_scen.loc[v1_scen["scenario_id"].ne(FULL_HEALTH)]
        .sort_values("weight")
        .groupby(["game_id", "team_id"])
        .tail(1)
    )
    matched = {}
    for row in likely.itertuples(index=False):
        listed_here = {
            p for p, v in team_p.get((row.game_id, row.team_id), []) if v > 0.1
        }
        sat = {p for p in listed_here if (row.game_id, row.team_id, p) not in played}
        matched[(row.game_id, row.team_id)] = set(row.sitting) & listed_here == sat
    table["scenario"] = [
        (
            "most likely scenario matched"
            if matched.get(k, True)
            else "most likely scenario missed"
        )
        for k in keys
    ]

    # Games out since the last appearance for the team, from the stints alone
    # (independent of either roster rule).
    order = minutes.drop_duplicates(["game_id", "team_id"]).sort_values(
        ["game_date", "game_id"]
    )
    order["index"] = order.groupby("team_id").cumcount()
    index = order.set_index(["game_id", "team_id"])["index"].to_dict()
    appearances = minutes.assign(
        index=[
            index[k] for k in zip(minutes["game_id"], minutes["team_id"], strict=True)
        ]
    )
    seen = (
        appearances.groupby(["team_id", "player_id"])["index"].apply(sorted).to_dict()
    )

    def gap(g, t, p):
        idx = index.get((g, t))
        past = [i for i in seen.get((t, p), []) if i < idx] if idx is not None else []
        return idx - past[-1] - 1 if past else np.nan

    table["gap"] = [
        gap(g, t, p)
        for g, t, p in zip(
            table["game_id"], table["team_id"], table["player_id"], strict=True
        )
    ]
    table["gap_band"] = pd.cut(
        table["gap"], [-1, 10, 30, 10_000], labels=["0-10 out", "11-30 out", "31+ out"]
    )
    on_roster = season_players.set_index(KEYS)["on_roster"].to_dict()
    table["on_roster"] = [
        on_roster.get(k, False)
        for k in zip(
            table["game_id"], table["team_id"], table["player_id"], strict=True
        )
    ]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nGAP SINCE LAST APPEARANCE (stints): rows not playing / playing")
    for played_flag, part in table.groupby(table["actual"] > 0):
        print(f"  played = {played_flag}")
        summary = part.groupby("gap_band", observed=True).agg(
            rows=("actual", "size"),
            actual=("actual", "mean"),
            v0=("v0", "mean"),
            v1=("v1", "mean"),
            v1_realized=("v1_realized", "mean"),
        )
        print(summary.round(3).to_string())
    returns = table.loc[(table["actual"] > 0) & (table["gap"] > 10)]
    print(
        f"  true returns (play after > 10 games out): {len(returns)}; on the v1 "
        f"roster {returns['on_roster'].mean():.1%}; MAE v0 "
        f"{(returns['v0'] - returns['actual']).abs().mean():.2f}, v1 "
        f"{(returns['v1'] - returns['actual']).abs().mean():.2f}"
    )
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(
        f"\n1. EXPECTED MINUTES (report scenarios) vs ACTUAL: {table.groupby(['game_id', 'team_id']).ngroups} team-games, {len(table):,} rows"
    )
    print(mae(table, columns).round(3).to_string())
    for by in (
        "role",
        "uncertainty",
        "report_out",
        "key_out",
        "scenario",
        "returning",
        "depth",
    ):
        print(mae(table, columns, by).round(3).to_string())
    print("\n  misallocated minutes per team-game:")
    print(misallocated(table, columns).round(2).to_string())
    for by in ("uncertainty", "report_out", "scenario"):
        print(misallocated(table, columns, by).round(2).to_string())

    print("\n2. FROM REALIZED ABSENCES TO REPORT SCENARIOS (player MAE)")
    overall = mae(table, columns).iloc[0]
    for v in ("v0", "v1"):
        print(
            f"  {v}: realized {overall[f'{v}_realized']:.3f} -> report {overall[v]:.3f} (+{overall[v] - overall[f'{v}_realized']:.3f} from availability)"
        )

    print(
        "\n3. TOTAL: R1 and absence impact, each provider with its own full-health graph"
    )
    book = load_rating_book(ratings_path)
    readouts = {}
    for name, graphs in (("v0", v0_graphs), ("v1", v1_graphs)):
        frame = readout_2_6(graphs, book).set_index("GAME_ID")
        readouts[name] = frame.loc[frame.index.isin(both)]
    totals = pd.read_parquet(
        CLOSING_2_6, columns=["GAME_ID", "GAME_DATE", "TOTAL_POINTS"]
    )
    totals["GAME_ID"] = totals["GAME_ID"].astype(str).str.zfill(10)
    totals = totals.set_index("GAME_ID")
    frame = (
        pd.DataFrame({f"{n}_raw": r["raw_total"] for n, r in readouts.items()})
        .join(
            pd.DataFrame(
                {f"{n}_impact": r["impact_points"] for n, r in readouts.items()}
            )
        )
        .join(totals)
    )
    frame = frame.sort_values("GAME_DATE")
    phases = game_phase(pd.Series(frame.index, index=frame.index))
    for n in readouts:
        offset = walk_forward_offset(
            frame["GAME_DATE"],
            frame[f"{n}_raw"],
            frame["TOTAL_POINTS"],
            phases=phases,
            min_games=50,
        )
        frame[f"{n}_err"] = (frame["TOTAL_POINTS"] - frame[f"{n}_raw"] - offset).abs()
    scored = frame.dropna(subset=["v0_err", "v1_err"])
    diff = scored["v0_err"] - scored["v1_err"]
    print(
        f"  {len(scored)} calibrated games: total MAE v0 {scored['v0_err'].mean():.3f}, v1 "
        f"{scored['v1_err'].mean():.3f}; v1 better by {diff.mean():+.3f} ± {diff.std(ddof=1) / np.sqrt(len(diff)):.3f}"
    )
    print(
        f"  absence impact: mean |impact| v0 {frame['v0_impact'].abs().mean():.3f}, v1 "
        f"{frame['v1_impact'].abs().mean():.3f}; corr(v0, v1) {frame[['v0_impact', 'v1_impact']].corr().iloc[0, 1]:.3f}"
    )
    season_of = season_minutes.drop_duplicates("game_id").set_index("game_id")["season"]
    table["season"] = table["game_id"].map(season_of)
    frame["season"] = frame.index.map(season_of)
    out = args.local_root / "player_graph" / "rotation"
    table.to_parquet(
        out / f"pregame_minutes{args.tag}_season={label}.parquet", index=False
    )
    frame.to_parquet(out / f"pregame_totals{args.tag}_season={label}.parquet")
    if len(seasons) > 1:
        print("\n4. BY SEASON")
        rows_by = {}
        for season, part in table.groupby("season"):
            line = {
                f"MAE {c}": (part[c] - part["actual"]).abs().mean() for c in columns
            }
            per_team = (
                part.assign(
                    **{c: (part[c] - part["actual"]).abs() for c in ("v0", "v1")}
                )
                .groupby(["game_id", "team_id"])[["v0", "v1"]]
                .sum()
                / 2
            )
            line["misallocated v0"] = per_team["v0"].mean()
            line["misallocated v1"] = per_team["v1"].mean()
            games = frame.loc[frame["season"].eq(season)].dropna(
                subset=["v0_err", "v1_err"]
            )
            gain = games["v0_err"] - games["v1_err"]
            line["R1 games"] = len(games)
            line["R1 v1 better by"] = gain.mean()
            line["R1 SE"] = (
                gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else np.nan
            )
            rows_by[season] = line
        print(pd.DataFrame(rows_by).T.round(3).to_string())


if __name__ == "__main__":
    main()
