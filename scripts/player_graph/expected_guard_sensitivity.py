"""Sensitivity of ``expected_guard`` v0 to the corrupt matchup games (phase 4B).

69 archive games carry doubled matchup exposure (or no points);
``PointInTimeData.without_corrupt_matchups`` drops them, whole, from the pair
history. This compares v0 as built with the history without them, nothing
retuned (same k, half-life and window):

``tables``: the stored v0 tables (``expected_guard/v0``) against the rebuild
without the games (``build_expected_guard.py --exclude-corrupt-matchups``):
raw ``r_hat``, ``r_prior`` and ``prior_weight`` differences per season and
month (when they start and how they fade as the window rolls forward), the
TVD between the two versions' normalized guard shares per attacker-game
(rates times the game's co-floor seconds, as the guard benchmark), and each
version's benchmark TVD against the observed shares (observed rows without
the corrupt games for both).

``graphs``: the closing v0 game graphs of ``--seasons`` built from both
histories: TVD of each attacker's guard-edge weights between the versions,
and the R2 readout (2_6's stored rating cache) of tonight's scenarios and the
full-health graph. Needs the database.

    python scripts/player_graph/expected_guard_sensitivity.py tables --seasons 2018-2025
    python scripts/player_graph/expected_guard_sensitivity.py graphs --seasons 2018-2019
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import game_nights, load_rating_book
from nba_ou.data_processing.player_graph.as_of import (
    CLOSING,
    CORRECTED_MATCHUP_HISTORY,
    FIRST_SEASON,
    ORIGINAL_MATCHUP_HISTORY,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.expected_guard import (
    read_seasons,
    stored_version,
)
from nba_ou.data_processing.player_graph.game_graph import build_game_graphs
from nba_ou.data_processing.player_graph.guard_benchmark import (
    ATTACKER,
    KEYS,
    benchmark_frame,
    implied_shares,
    share_tvd,
)
from nba_ou.data_processing.player_graph.matchup_defense import read_matchup_history
from nba_ou.data_processing.player_graph.oracle import (
    readouts,
    usage_as_of,
    usage_lookup,
)

ROOT = Path("data/player_graph/expected_guard")
CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
CHANGED = 1e-12


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def _quantiles(values: pd.Series) -> dict:
    values = values.abs()
    return {
        "mean": values.mean(),
        "p99": values.quantile(0.99),
        "p99.9": values.quantile(0.999),
        "max": values.max(),
    }


def tables(seasons: list[int], local_root: Path) -> None:
    original, meta_original = read_seasons(
        ROOT / stored_version(ORIGINAL_MATCHUP_HISTORY), seasons
    )
    corrected, meta_corrected = read_seasons(
        ROOT / stored_version(CORRECTED_MATCHUP_HISTORY), seasons
    )
    excluded = meta_corrected.pop("excluded_matchup_games", [])
    meta_original.pop("matchup_history", None)
    meta_corrected.pop("matchup_history", None)
    differ = {k: v for k, v in meta_original.items() if meta_corrected.get(k) != v}
    if differ:
        raise SystemExit(f"parameters differ: {differ}")
    keys = ["game_id", "off_player_id", "def_player_id"]
    both = original.merge(
        corrected, on=keys, suffixes=("", "_clean"), how="outer", indicator=True
    )
    # Pairs come from the stints of each date: a game whose stints were
    # rejected after v0 was built has rows only in v0. Compared: shared pairs.
    one_sided = both.loc[~both["_merge"].eq("both")]
    if len(one_sided):
        print(
            "pairs in one version only (stint store changed since v0 was built, "
            "not the matchup history): "
            + ", ".join(
                f"{game} {side} {n}"
                for (game, side), n in one_sided.groupby(
                    ["game_id", "_merge"], observed=True
                )
                .size()
                .loc[lambda x: x > 0]
                .items()
            )
        )
    one_sided_games = set(one_sided["game_id"])
    both = both.loc[both["_merge"].eq("both")].drop(columns="_merge")
    both["game_date"] = pd.to_datetime(both["game_date"])
    both["season"] = both["game_id"].str[3:5].astype(int) + 2000
    for column in ("r_hat", "r_prior", "prior_weight"):
        both[f"d_{column}"] = both[f"{column}_clean"] - both[column]
    changed = both["d_r_hat"].abs().gt(CHANGED) | both["d_r_prior"].abs().gt(CHANGED)
    pairs_games = pd.read_parquet(
        Path(local_root) / "player_graph" / "pair_game" / "season=2018.parquet",
        columns=["game_id", "game_date"],
    ).drop_duplicates("game_id")
    first_corrupt = pairs_games.loc[pairs_games["game_id"].isin(excluded), "game_date"]
    print(
        f"{len(excluded)} corrupt games excluded; first of them played "
        f"{pd.Timestamp(first_corrupt.min()).date() if len(first_corrupt) else 'n/a'}; "
        f"{len(both):,} pairs compared ({seasons[0]}-{seasons[-1]}); first as-of date "
        f"with any difference: {both.loc[changed, 'game_date'].min().date()}; last "
        f"identical date before it: "
        f"{both.loc[~changed & (both['game_date'] < both.loc[changed, 'game_date'].min()), 'game_date'].max().date()}"
    )
    pd.set_option("display.width", 250, "display.max_columns", 30)

    print("\n1. RAW RATES, corrected - original, per season (r_hat mean for scale)")
    rows = {}
    for season, part in both.groupby("season"):
        line = {
            "pairs": len(part),
            "changed": changed.loc[part.index].mean(),
            "r_hat mean": part["r_hat"].mean(),
        }
        for column in ("r_hat", "r_prior", "prior_weight"):
            line.update(
                {
                    f"|d {column}| {k}": v
                    for k, v in _quantiles(part[f"d_{column}"]).items()
                }
            )
        line["|d r_hat| / r_hat"] = (part["d_r_hat"].abs() / part["r_hat"]).mean()
        rows[season] = line
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.2e}"))

    print("\n   mean |d r_hat| by month (roll-off as the window moves forward)")
    month = both["game_date"].dt.to_period("M")
    monthly = both["d_r_hat"].abs().groupby(month).mean()
    print("   " + ", ".join(f"{m}: {v:.1e}" for m, v in monthly.items()))

    print("\n2. NORMALIZED GUARD SHARES (rate x the game's co-floor seconds)")
    observed, _ = read_matchup_history(seasons, local_root=local_root)
    frame_original = benchmark_frame(observed, original)
    frame_corrected = benchmark_frame(observed, corrected)
    frame = frame_original.merge(
        frame_corrected[[*KEYS, "r_hat"]], on=KEYS, suffixes=("", "_clean")
    )
    frame = frame.loc[~frame["game_id"].isin(one_sided_games)].reset_index(drop=True)
    e_original, _ = implied_shares(frame, frame["r_hat"])
    e_corrected, _ = implied_shares(frame, frame["r_hat_clean"])
    groups = [frame["game_id"], frame["off_player_id"]]
    between = 0.5 * (e_corrected - e_original).abs().groupby(groups).sum()
    weight = frame.groupby(ATTACKER)["matchup_seconds"].sum()
    season_of = frame.groupby(ATTACKER)["season_year"].first()
    tvd_original = share_tvd(frame, frame["r_hat"])
    tvd_corrected = share_tvd(frame, frame["r_hat_clean"])
    rows = {}
    for season in sorted(season_of.unique()):
        mask = season_of.eq(season)
        w = weight[mask]
        b = between[mask]
        rows[season] = {
            "attacker-games": int(mask.sum()),
            "TVD between versions, weighted mean": np.average(b, weights=w),
            "p99": b.quantile(0.99),
            "max": b.max(),
            "share > 0.01": (b > 0.01).mean(),
            "share > 0.05": (b > 0.05).mean(),
            "benchmark TVD original": np.average(tvd_original[mask], weights=w),
            "benchmark TVD corrected": np.average(tvd_corrected[mask], weights=w),
        }
    table = pd.DataFrame(rows).T
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))
    w = weight
    print(
        f"   all seasons: TVD between versions {np.average(between, weights=w):.5f} "
        f"(p99 {between.quantile(0.99):.5f}, max {between.max():.4f}); benchmark TVD "
        f"original {np.average(tvd_original, weights=w):.5f}, corrected "
        f"{np.average(tvd_corrected, weights=w):.5f}"
    )


def _season_games(season: int) -> pd.DataFrame:
    file = pd.read_parquet(
        CLOSING_2_6,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "SEASON_YEAR",
            "TEAM_ID_TEAM_HOME",
            "TEAM_ID_TEAM_AWAY",
        ],
    )
    file = file.loc[pd.to_numeric(file["SEASON_YEAR"], errors="coerce").eq(season)]
    return pd.DataFrame(
        {
            "GAME_ID": file["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": pd.to_datetime(file["GAME_DATE"]).dt.normalize(),
            "HOME_TEAM_ID": file["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": file["TEAM_ID_TEAM_AWAY"].astype(str),
        }
    ).drop_duplicates("GAME_ID")


def _inputs(data: PointInTimeData, book, graphs, dates: pd.Series):
    players = (
        pd.concat([g.nodes[["game_id", "player_id"]] for g in graphs])
        .drop_duplicates()
        .assign(game_date=lambda f: f["game_id"].map(dates))
    )
    ratings, usage = [], []
    for date, part in players.groupby("game_date", sort=True):
        rated = book.for_date(date)
        if rated is not None:
            team, ortg, pace = rated
            ratings.append(
                part.assign(
                    off=part["player_id"].map(team.offense).fillna(0.0),
                    **{"def": part["player_id"].map(team.defense).fillna(0.0)},
                    pace=part["player_id"].map(team.pace).fillna(0.0),
                    league_ortg=ortg,
                    league_pace=pace,
                )
            )
        usage.append(
            part.assign(
                usage=usage_lookup(usage_as_of(data.as_of(date)), part["player_id"])
            )
        )
    return (
        pd.concat(ratings, ignore_index=True).drop(columns="game_date"),
        pd.concat(usage, ignore_index=True)[["game_id", "player_id", "usage"]],
    )


def graphs(seasons: list[int]) -> None:
    started = time.time()
    data = PointInTimeData.load(
        range(max(FIRST_SEASON, seasons[0] - 3), seasons[-1] + 1),
        closing_injuries=True,
    )
    corrected, excluded = data.without_corrupt_matchups()
    state = data.injury_report(CLOSING)
    p_out = player_out_probabilities(state.statuses)
    excluded_players = roster_exclusions(state.statuses)
    book = load_rating_book()
    print(
        f"loaded in {time.time() - started:.0f}s; {len(excluded)} corrupt games "
        "dropped from the corrected history",
        flush=True,
    )
    pd.set_option("display.width", 250, "display.max_columns", 30)
    rows = {}
    for season in seasons:
        started = time.time()
        games = _season_games(season)
        dates = games.set_index("GAME_ID")["GAME_DATE"]
        nights = game_nights(
            games, data.box_scores_2_6, p_out, excluded=excluded_players
        )
        built = {
            name: build_game_graphs(source, games, nights, report_covered=state.covered)
            for name, source in (("original", data), ("corrected", corrected))
        }
        edges = {
            name: g.edges.loc[g.edges["relation"].eq("guards")].set_index(
                ["game_id", "scenario_id", "dst", "src"]
            )["weight"]
            for name, g in built.items()
        }
        joined = pd.concat(edges, axis=1, join="outer").fillna(0.0)
        per_attacker = (
            0.5
            * (joined["corrected"] - joined["original"])
            .abs()
            .groupby(level=["game_id", "scenario_id", "dst"])
            .sum()
        )
        ratings, usage = _inputs(data, book, list(built.values()), dates)
        out = {
            name: readouts(g, ratings, usage).set_index("game_id")
            for name, g in built.items()
        }
        r2 = out["corrected"]["r2_total"] - out["original"]["r2_total"]
        r2_healthy = (
            out["corrected"]["r2_total_healthy"] - out["original"]["r2_total_healthy"]
        )
        r1 = out["corrected"]["r1_total"] - out["original"]["r1_total"]
        rows[season] = {
            "games": games["GAME_ID"].nunique(),
            "guard edges": len(joined),
            "edge |d weight| max": (joined["corrected"] - joined["original"])
            .abs()
            .max(),
            "attacker TVD mean": per_attacker.mean(),
            "attacker TVD p99": per_attacker.quantile(0.99),
            "attacker TVD max": per_attacker.max(),
            "attackers TVD > 0.01": (per_attacker > 0.01).mean(),
            "|d R2| mean": r2.abs().mean(),
            "|d R2| max": r2.abs().max(),
            "|d R2 full health| max": r2_healthy.abs().max(),
            "|d R1| max": r1.abs().max(),
            "games with |d R2| > 0.1": int((r2.abs() > 0.1).sum()),
        }
        print(f"{season}: {time.time() - started:.0f}s", flush=True)
    print(
        "\n3. GAME GRAPHS (closing v0): corrected - original; TVD over each "
        "attacker's guard-edge weights per scenario; R2 / R1 totals in points"
    )
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.2e}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tables", "graphs"])
    parser.add_argument("--seasons", type=season_range, required=True)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "tables":
        tables(args.seasons, args.local_root)
    else:
        graphs(args.seasons)


if __name__ == "__main__":
    main()
