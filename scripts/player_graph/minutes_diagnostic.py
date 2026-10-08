"""Where 2_6's projected minutes fail, above all when someone is out (phase 4A,
step 1). Diagnostic only, development season 2018-19; nothing is fitted.

Projected minutes are 2_6's closing projection (``game_nights`` ->
``enumerate_scenarios`` -> ``allocate_minutes``), the same numbers the v0 game
graph carries; actual minutes are regulation minutes from the stints
(``nba_ou.data_processing.player_graph.minutes_diagnostics``). Only games
whose two teams had filed an injury report by tip-off, as for the v0 graphs.

Reports:

1. per-player error by projected role, actual role, sample size and report
   status, in games with and without someone out;
2. misallocated minutes per team-game (sum of |projected - actual| / 2) and
   where they come from (players who did not play, players not on the
   roster, players listed out who played, everyone else), by number and
   importance of the absences and by final margin (garbage time);
3. redistribution in games with someone out: who absorbs the absent minutes
   in reality vs in ``allocate_minutes`` (starters vs bench, position
   similarity, the top absorber);
4. physically invalid allocations (> 48 minutes), counted, v0 untouched;
5. the 2018-19 oracle tables: how much of the rotation oracle's gain comes
   from games with absences.

    python scripts/player_graph/minutes_diagnostic.py

Needs the database (rosters, injury reports). Writes the player and
team-game tables to ``data/player_graph/minutes_diagnostics/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import (
    game_nights,
    game_phase,
    walk_forward_offset,
)
from nba_ou.data_processing.player_graph.as_of import CLOSING, PointInTimeData
from nba_ou.data_processing.player_graph.minutes_diagnostics import (
    actual_minutes,
    minutes_table,
    projected_minutes,
)
from nba_ou.data_processing.player_graph.positions import PROBABILITY_COLUMNS

SEASON = 2018
CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
LOW_SAMPLE_GAMES = 10
IMPORTANCE_EDGES = (0, 15, 25, 32, 48.01)
IMPORTANCE_LABELS = ("< 15 min", "15-25", "25-32", "> 32")


def season_games() -> pd.DataFrame:
    file = pd.read_parquet(
        CLOSING_2_6,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "SEASON_YEAR",
            "TEAM_ID_TEAM_HOME",
            "TEAM_ID_TEAM_AWAY",
            "TOTAL_POINTS",
            total_line_col(),
        ],
    )
    file = file.loc[pd.to_numeric(file["SEASON_YEAR"], errors="coerce").eq(SEASON)]
    return pd.DataFrame(
        {
            "GAME_ID": file["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": pd.to_datetime(file["GAME_DATE"]).dt.normalize(),
            "HOME_TEAM_ID": file["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": file["TEAM_ID_TEAM_AWAY"].astype(str),
        }
    ).drop_duplicates("GAME_ID")


def mae_table(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = table.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "player-games": grouped.size(),
            "MAE": grouped["error"].apply(lambda e: e.abs().mean()),
            "bias (proj - actual)": grouped["error"].mean(),
            "mean actual": grouped["actual"].mean(),
            "played": grouped["played"].mean(),
        }
    )
    return out


def mean_se(values: pd.Series) -> str:
    values = values.dropna()
    return f"{values.mean():+.3f} ± {values.std(ddof=1) / np.sqrt(len(values)):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    started = time.time()
    data = PointInTimeData.load(range(2016, SEASON + 1), closing_injuries=True)
    state = data.injury_report(CLOSING)
    games = season_games()
    nights = game_nights(
        games,
        data.box_scores_2_6,
        player_out_probabilities(state.statuses),
        excluded=roster_exclusions(state.statuses),
    )
    covered = set(state.covered)
    both = {
        g
        for g, h, a in games[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]].itertuples(
            index=False
        )
        if (g, h) in covered and (g, a) in covered
    }
    # Games without validated stints have no actual minutes to compare with.
    both &= set(data.stints["game_id"])
    nights = {key: value for key, value in nights.items() if key[0] in both}
    projected = projected_minutes(nights)
    stints = data.stints.loc[data.stints["game_id"].isin(both)]
    table = minutes_table(projected, actual_minutes(stints))
    dates = games.set_index("GAME_ID")["GAME_DATE"]
    table["game_date"] = table["game_id"].map(dates)
    print(
        f"loaded in {time.time() - started:.0f}s; {len(both)} games with both "
        f"reports, {table.groupby(['game_id', 'team_id']).ngroups} team-games, "
        f"{len(table):,} player rows"
    )
    check = table.groupby(["game_id", "team_id"])[["projected", "actual"]].sum()
    print(
        "team sums: projected "
        f"{check['projected'].min():.2f}-{check['projected'].max():.2f}, actual "
        f"{check['actual'].min():.2f}-{check['actual'].max():.2f}"
    )

    # Labels: sample size and soft position as of the game date, actual starters.
    profiles = pd.concat(
        [
            pd.read_parquet(
                args.local_root
                / "player_graph"
                / "node_profiles"
                / f"season={s}.parquet",
                columns=[
                    "as_of_date",
                    "player_id",
                    "games_in_data",
                    *PROBABILITY_COLUMNS,
                ],
            )
            for s in (SEASON - 1, SEASON)
        ],
        ignore_index=True,
    ).rename(columns={"as_of_date": "game_date"})
    table = table.merge(profiles, on=["game_date", "player_id"], how="left")
    table["games_in_data"] = table["games_in_data"].fillna(0)
    box = data.box_scores.loc[data.box_scores["GAME_ID"].isin(both)]
    starters = set(
        zip(
            box.loc[
                box["START_POSITION"].fillna("").astype(str).str.strip().ne(""),
                "GAME_ID",
            ],
            box.loc[
                box["START_POSITION"].fillna("").astype(str).str.strip().ne(""),
                "PLAYER_ID",
            ].astype(str),
            strict=True,
        )
    )
    table["actual_role"] = [
        "starter" if (g, p) in starters else "bench"
        for g, p in zip(table["game_id"], table["player_id"], strict=True)
    ]
    table["sample"] = np.where(
        table["games_in_data"] <= LOW_SAMPLE_GAMES,
        f"<= {LOW_SAMPLE_GAMES} games",
        "more",
    )

    # Team-game context.
    team = table.groupby(["game_id", "team_id"])
    teams = pd.DataFrame(
        {
            "n_out": team["status"].apply(lambda s: (s == "out").sum()),
            "n_uncertain": team["status"].apply(lambda s: (s == "uncertain").sum()),
            "out_healthy": table.loc[table["status"].eq("out")]
            .groupby(["game_id", "team_id"])["healthy"]
            .sum(),
            "max_out_healthy": table.loc[table["status"].eq("out")]
            .groupby(["game_id", "team_id"])["healthy"]
            .max(),
            "misallocated": team["error"].apply(lambda e: e.abs().sum() / 2),
        }
    ).fillna({"out_healthy": 0.0, "max_out_healthy": 0.0})
    components = {
        "did not play (available)": (table["in_roster"] & ~table["played"])
        & table["status"].eq("available"),
        "did not play (uncertain)": (table["in_roster"] & ~table["played"])
        & table["status"].eq("uncertain"),
        "listed out but played": table["status"].eq("out") & table["played"],
        "not on roster": ~table["in_roster"],
    }
    rest = ~np.logical_or.reduce(list(components.values()))
    for name, mask in {**components, "everyone else": rest}.items():
        teams[name] = (
            (
                table.loc[mask, "error"]
                .abs()
                .groupby([table["game_id"], table["team_id"]])
                .sum()
                / 2
            )
            .reindex(teams.index)
            .fillna(0.0)
        )
    margin = (
        stints.assign(m=stints["home_pts"] - stints["away_pts"])
        .groupby("game_id")["m"]
        .sum()
    )
    teams["abs_margin"] = teams.index.get_level_values("game_id").map(margin.abs())
    teams["out_group"] = pd.cut(
        teams["n_out"], [-1, 0, 1, 2, 99], labels=["0 out", "1 out", "2 out", "3+ out"]
    )
    teams["importance"] = pd.cut(
        teams["max_out_healthy"], list(IMPORTANCE_EDGES), labels=list(IMPORTANCE_LABELS),
        right=False,
    )  # fmt: skip
    teams.loc[teams["n_out"].eq(0), "importance"] = np.nan
    table = table.join(teams[["n_out", "out_group"]], on=["game_id", "team_id"])
    table["game_type"] = np.where(table["n_out"] > 0, ">= 1 out", "nobody out")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n1. PER-PLAYER ERROR (projected - actual, regulation minutes)")
    roster_played = table.loc[table["status"].ne("out")]
    for by in (["game_type", "projected_role"], ["game_type", "actual_role"],
               ["game_type", "sample"], ["status"]):  # fmt: skip
        print(
            mae_table(roster_played if by != ["status"] else table, by)
            .round(2)
            .to_string()
        )
        print()

    print("2. MISALLOCATED MINUTES PER TEAM-GAME (sum |error| / 2) AND SOURCES")
    parts = ["misallocated", *components, "everyone else"]
    print(teams.groupby("out_group", observed=True)[parts].mean().round(2).to_string())
    print(
        f"  team-games per group: {teams['out_group'].value_counts().sort_index().to_dict()}"
    )
    print(teams.groupby("importance", observed=True)[parts].mean().round(2).to_string())
    blow = pd.cut(
        teams["abs_margin"], [-1, 9, 19, 99], labels=["margin < 10", "10-19", ">= 20"]
    )
    print(teams.groupby(blow, observed=True)[parts].mean().round(2).to_string())

    print("\n3. REDISTRIBUTION in team-games with someone listed out")
    out_games = teams.index[teams["n_out"] > 0]
    sub = table.set_index(["game_id", "team_id"]).loc[out_games].reset_index()
    absorbers = sub.loc[sub["status"].eq("available")].copy()
    absorbers["pred_delta"] = absorbers["projected"] - absorbers["healthy"]
    absorbers["actual_delta"] = absorbers["actual"] - absorbers["healthy"]
    unexpected = sub.loc[~sub["in_roster"]]
    absent = sub.loc[sub["status"].eq("out")]
    # Position similarity to the absentees, weighted by their full-health minutes.
    out_pos = (
        absent.assign(
            **{
                c: absent[c].fillna(1 / 3) * absent["healthy"]
                for c in PROBABILITY_COLUMNS
            }
        )
        .groupby(["game_id", "team_id"])[[*PROBABILITY_COLUMNS, "healthy"]]
        .sum()
    )
    for c in PROBABILITY_COLUMNS:
        out_pos[c] = out_pos[c] / out_pos["healthy"].where(out_pos["healthy"] > 0)
    joined = absorbers.join(
        out_pos[list(PROBABILITY_COLUMNS)], on=["game_id", "team_id"], rsuffix="_out"
    )
    absorbers["similarity"] = sum(
        joined[c].fillna(1 / 3) * joined[f"{c}_out"].fillna(1 / 3)
        for c in PROBABILITY_COLUMNS
    )
    total_pred = absorbers["pred_delta"].sum()
    total_actual = absorbers["actual_delta"].sum() + unexpected["actual"].sum()
    print(
        f"  {len(out_games)} team-games; absent full-health minutes "
        f"{absent['healthy'].sum():,.0f}; redistributed to available players: "
        f"projected {total_pred:,.0f}, actual {absorbers['actual_delta'].sum():,.0f} "
        f"+ {unexpected['actual'].sum():,.0f} to players not on the roster"
    )
    shares = pd.DataFrame(
        {
            "projected share": absorbers.groupby("projected_role")["pred_delta"].sum()
            / total_pred,
            "actual share": absorbers.groupby("projected_role")["actual_delta"].sum()
            / total_actual,
        }
    )
    shares.loc["not on roster"] = [0.0, unexpected["actual"].sum() / total_actual]
    print(shares.round(3).to_string())
    absorbers["similarity_group"] = pd.qcut(
        absorbers["similarity"],
        3,
        labels=["least similar third", "middle", "most similar third"],
    )
    by_sim = absorbers.groupby("similarity_group", observed=True)
    print(
        pd.DataFrame(
            {
                "projected share": by_sim["pred_delta"].sum() / total_pred,
                "actual share": by_sim["actual_delta"].sum() / total_actual,
                "mean similarity": by_sim["similarity"].mean(),
            }
        )
        .round(3)
        .to_string()
    )
    w = absorbers["pred_delta"]
    slope = np.polyfit(absorbers["pred_delta"], absorbers["actual_delta"], 1)[0]
    print(
        f"  available players: corr(projected delta, actual delta) "
        f"{np.corrcoef(absorbers['pred_delta'], absorbers['actual_delta'])[0, 1]:.3f}, "
        f"slope {slope:.2f}; corr(similarity, actual delta) "
        f"{np.corrcoef(absorbers['similarity'], absorbers['actual_delta'])[0, 1]:.3f}"
        f" (projected: {np.corrcoef(absorbers['similarity'], w)[0, 1]:.3f})"
    )
    single = teams.index[(teams["n_out"] == 1) & (teams["max_out_healthy"] >= 20)]
    tops = []
    for key in single:
        part = absorbers.set_index(["game_id", "team_id"]).loc[[key]]
        if part.empty:
            continue
        top = part.sort_values("actual_delta").iloc[-1]
        lost = teams.loc[key, "out_healthy"]
        tops.append(
            {
                "actual share of the absent minutes": top["actual_delta"] / lost,
                "projected share for him": top["pred_delta"] / lost,
                "bench": top["projected_role"] == "bench",
                "similarity rank": part["similarity"]
                .rank(ascending=False)[part["player_id"] == top["player_id"]]
                .iloc[0],
                "healthy rank": top["healthy_rank"],
            }
        )
    tops = pd.DataFrame(tops)
    print(
        f"  top actual absorber, {len(tops)} team-games with one player out "
        f"(>= 20 full-health min): takes {tops['actual share of the absent minutes'].median():.0%} "
        f"of the absent minutes (median; projected for him "
        f"{tops['projected share for him'].median():.0%}); a projected bench player in "
        f"{tops['bench'].mean():.0%}; the most position-similar available player in "
        f"{(tops['similarity rank'] <= 1).mean():.0%}; median full-health rank "
        f"{tops['healthy rank'].median():.0f}"
    )
    print("  available players' MAE: nobody out vs >= 1 out (same status)")
    avail = table.loc[table["status"].eq("available")]
    print(mae_table(avail, ["game_type", "projected_role"]).round(2).to_string())
    print(
        mae_table(
            avail.loc[avail["n_out"] > 0].join(
                teams["importance"], on=["game_id", "team_id"]
            ),
            ["importance"],
        )
        .round(2)
        .to_string()
    )

    print("\n4. ALLOCATIONS ABOVE 48 MINUTES (v0 unchanged)")
    over = table.groupby(["game_id", "team_id"]).agg(
        any_scenario=("max_scenario", "max"), expected=("projected", "max"),
        p_over=("p_over_48", "max"),
    )  # fmt: skip
    print(
        f"  team-games with a scenario above 48: {(over['any_scenario'] > 48).mean():.1%} "
        f"(max {over['any_scenario'].max():.1f}); probability-weighted "
        f"{over['p_over'].mean():.2%}; expected minutes above 48: "
        f"{(over['expected'] > 48).mean():.1%}; by number out: "
        + ", ".join(
            f"{g} {v:.1%}"
            for g, v in (over["any_scenario"] > 48)
            .groupby(teams["out_group"], observed=True)
            .mean()
            .items()
        )
    )

    print(
        "\n5. ORACLE ROTATION GAIN ON 2018-19 BY GAME TYPE (calibrated |total error|)"
    )
    oracle = pd.read_parquet(
        args.local_root / "player_graph" / "oracle" / f"season={SEASON}.parquet"
    )
    phases = game_phase(oracle["game_id"])
    for column in ("v0__r1_total", "oracle_rotation__r1_total"):
        offset = walk_forward_offset(
            oracle["GAME_DATE"], oracle[column], oracle["TOTAL_POINTS"], phases=phases
        )
        oracle[f"{column}_err"] = (
            oracle["TOTAL_POINTS"] - oracle[column] - offset
        ).abs()
    oracle["gain"] = (
        oracle["v0__r1_total_err"] - oracle["oracle_rotation__r1_total_err"]
    )
    per_game = teams.groupby(level="game_id").agg(
        n_out=("n_out", "sum"), max_out=("max_out_healthy", "max")
    )
    oracle = oracle.join(per_game, on="game_id")
    groups = {
        "all": oracle["gain"],
        "nobody out": oracle.loc[oracle["n_out"] == 0, "gain"],
        ">= 1 out": oracle.loc[oracle["n_out"] > 0, "gain"],
        "someone >= 25 min out": oracle.loc[oracle["max_out"] >= 25, "gain"],
        "key defender out": oracle.loc[oracle["key_defender_out"].astype(bool), "gain"],
    }
    for name, values in groups.items():
        print(
            f"  {name:24s} games {values.notna().sum():4d}  |error| drop {mean_se(values)}"
        )

    out = args.local_root / "player_graph" / "minutes_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / f"players_season={SEASON}.parquet", index=False)
    teams.reset_index().to_parquet(
        out / f"team_games_season={SEASON}.parquet", index=False
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
