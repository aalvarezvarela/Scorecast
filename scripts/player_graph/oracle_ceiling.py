"""Oracle diagnostic for the v0 game graph: what would perfect weights buy?

**Diagnostic only, not a strict ceiling** (see
``nba_ou.data_processing.player_graph.oracle``): the oracle graphs use the
game's own minutes, rotation and guarding, which are never features and never
train anything. Four versions of every closing game graph are read out:

* ``v0``: projected minutes, expected overlap, expected rates;
* ``oracle_rotation`` (phase 4A): actual regulation minutes and overlap;
* ``oracle_guards`` (phase 4B): observed guarding rates;
* ``oracle_both``.

Readouts R1 (2_6 projection) and R2 (defense faced through guard shares) are
calibrated with 2_6's walk-forward level offset and scored on total MAE and
against the closing line (``LINE_ERROR``), overall, per season and on the
absence slices. All versions are compared on the same games.

    python scripts/player_graph/oracle_ceiling.py build --seasons 2018-2025
    python scripts/player_graph/oracle_ceiling.py summarize --seasons 2019-2025

(2018-19 is built only to warm up the level calibration of 2019-20.)

``build`` needs the database and writes ``data/player_graph/oracle/season=Y.parquet``
(skipping seasons already there unless ``--force``).
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
    load_rating_book,
    walk_forward_offset,
)
from nba_ou.data_processing.player_graph.as_of import CLOSING, PointInTimeData
from nba_ou.data_processing.player_graph.expected_guard import expected_guard
from nba_ou.data_processing.player_graph.game_graph import (
    FULL_HEALTH,
    GameGraphs,
    build_game_graphs,
)
from nba_ou.data_processing.player_graph.oracle import (
    VERSIONS,
    actual_graphs,
    readouts,
    regulation_stints,
    usage_as_of,
    usage_lookup,
    with_observed_rates,
)
from nba_ou.data_processing.player_graph.positions import positions_as_of

CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
OUT_DIR = Path("data/player_graph/oracle")
KEY_DEFENDER_MIN_MINUTES = 20.0
ABSENCE_IMPACT_SLICE = 3.0


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def season_games(season: int) -> pd.DataFrame:
    line = total_line_col()
    file = pd.read_parquet(
        CLOSING_2_6,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "SEASON_YEAR",
            "TEAM_ID_TEAM_HOME",
            "TEAM_ID_TEAM_AWAY",
            "TOTAL_POINTS",
            line,
        ],
    )
    file = file.loc[pd.to_numeric(file["SEASON_YEAR"], errors="coerce").eq(season)]
    return pd.DataFrame(
        {
            "GAME_ID": file["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": pd.to_datetime(file["GAME_DATE"]).dt.normalize(),
            "HOME_TEAM_ID": file["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": file["TEAM_ID_TEAM_AWAY"].astype(str),
            "TOTAL_POINTS": pd.to_numeric(file["TOTAL_POINTS"], errors="coerce"),
            "LINE": pd.to_numeric(file[line], errors="coerce"),
        }
    ).drop_duplicates("GAME_ID")


def _per_game_inputs(
    data: PointInTimeData, book, graphs: list[GameGraphs], dates: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As-of ratings and usage for every (game, player) in any version."""
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


def _expected_rates(data: PointInTimeData, stints: pd.DataFrame) -> pd.DataFrame:
    """r_hat as of each game's date for every actual opponent pair."""
    frames = []
    for date, part in stints.groupby("game_date", sort=True):
        pairs = []
        for row in part.drop_duplicates("game_id").itertuples(index=False):
            game = part.loc[part["game_id"].eq(row.game_id)]
            home = sorted({p for lineup in game["home_lineup"] for p in lineup})
            away = sorted({p for lineup in game["away_lineup"] for p in lineup})
            pairs += [(row.game_id, a, d) for a in home for d in away]
            pairs += [(row.game_id, a, d) for a in away for d in home]
        view = data.as_of(date)
        frames.append(
            expected_guard(
                view,
                pd.DataFrame(
                    pairs, columns=["game_id", "off_player_id", "def_player_id"]
                ),
                positions=positions_as_of(view),
            )[["game_id", "off_player_id", "def_player_id", "r_hat"]]
        )
    return pd.concat(frames, ignore_index=True).rename(columns={"r_hat": "rate"})


def _slices(v0: GameGraphs, ratings: pd.DataFrame) -> pd.DataFrame:
    """Per game: whether a key defender is expected out, and max minutes."""
    scenarios = v0.scenarios
    tonight = scenarios.loc[~scenarios["is_full_health"].astype(bool)]
    weight = tonight.set_index(["game_id", "scenario_id"])["weight"]
    weight = weight / weight.groupby(level="game_id").transform("sum")
    nodes = v0.nodes.join(weight.rename("w"), on=["game_id", "scenario_id"])
    plays = nodes.dropna(subset=["w"]).groupby(["game_id", "player_id"])["w"].sum()
    healthy = v0.nodes.loc[v0.nodes["scenario_id"].eq(FULL_HEALTH)].copy()
    healthy["p_absent"] = (
        1
        - plays.reindex(pd.MultiIndex.from_frame(healthy[["game_id", "player_id"]]))
        .fillna(0.0)
        .to_numpy()
    )
    healthy = healthy.merge(
        ratings[["game_id", "player_id", "def"]],
        on=["game_id", "player_id"],
        how="left",
    )
    rotation = healthy.loc[healthy["minutes"] >= KEY_DEFENDER_MIN_MINUTES].copy()
    rotation["rank"] = rotation.groupby(["game_id", "side"])["def"].rank(
        ascending=False
    )
    key_out = (
        rotation.loc[(rotation["rank"] <= 2) & (rotation["p_absent"] >= 0.5)]
        .groupby("game_id")
        .size()
    )
    out = pd.DataFrame({"game_id": scenarios["game_id"].unique()})
    out["key_defender_out"] = out["game_id"].map(key_out).fillna(0).gt(0)
    out["max_minutes"] = out["game_id"].map(
        v0.nodes.groupby("game_id")["minutes"].max()
    )
    return out


def build(seasons: list[int], force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [
        s for s in seasons if force or not (OUT_DIR / f"season={s}.parquet").exists()
    ]
    if not todo:
        print("nothing to build")
        return
    started = time.time()
    data = PointInTimeData.load(
        range(max(2016, todo[0] - 3), todo[-1] + 1), closing_injuries=True
    )
    state = data.injury_report(CLOSING)
    p_out = player_out_probabilities(state.statuses)
    excluded = roster_exclusions(state.statuses)
    book = load_rating_book()
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    for season in todo:
        started = time.time()
        games = season_games(season)
        dates = games.set_index("GAME_ID")["GAME_DATE"]
        nights = game_nights(games, data.box_scores, p_out, excluded=excluded)
        v0 = build_game_graphs(data, games, nights, report_covered=state.covered)
        stints = regulation_stints(
            data.stints.loc[data.stints["game_id"].isin(set(v0.scenarios["game_id"]))]
        )
        observed = data.pair_game.loc[
            data.pair_game["game_id"].isin(set(v0.scenarios["game_id"]))
            & data.pair_game["cofloor_seconds"].gt(0),
            ["game_id", "off_player_id", "def_player_id", "guard_rate"],
        ]
        expected = _expected_rates(data, stints)
        both_rates = expected.merge(
            observed, on=["game_id", "off_player_id", "def_player_id"], how="left"
        )
        both_rates["rate"] = both_rates["guard_rate"].where(
            both_rates["guard_rate"].notna(), both_rates["rate"]
        )
        graphs = {
            "v0": v0,
            "oracle_rotation": actual_graphs(stints, expected, dates),
            "oracle_guards": with_observed_rates(v0, observed),
            "oracle_both": actual_graphs(
                stints,
                both_rates[["game_id", "off_player_id", "def_player_id", "rate"]],
                dates,
            ),
        }
        ratings, usage = _per_game_inputs(data, book, list(graphs.values()), dates)
        results = []
        for version, graph in graphs.items():
            frame = readouts(graph, ratings, usage).set_index("game_id")
            results.append(frame[["r1_total", "r2_total"]].add_prefix(f"{version}__"))
            if version in ("v0", "oracle_guards"):
                results.append(
                    frame[["r1_total_healthy", "r2_total_healthy"]].add_prefix(
                        f"{version}__"
                    )
                )
        table = pd.concat(results, axis=1)
        # The rotation oracles have no actual full-health graph. The counterfactual
        # keeps projected minutes and changes only the rates its oracle changes:
        # v0's for the rotation oracle, the guard oracle's for both oracles.
        for version, source in (
            ("oracle_rotation", "v0"),
            ("oracle_both", "oracle_guards"),
        ):
            for readout in ("r1", "r2"):
                table[f"{version}__{readout}_total_healthy"] = table[
                    f"{source}__{readout}_total_healthy"
                ]
        table = table.join(_slices(v0, ratings).set_index("game_id"))
        table = table.join(
            games.set_index("GAME_ID")[["GAME_DATE", "TOTAL_POINTS", "LINE"]]
        )
        table["season"] = season
        table["observed_rate_share"] = (
            both_rates["guard_rate"].notna().groupby(both_rates["game_id"]).mean()
        )
        table.index.name = "game_id"
        table.reset_index().to_parquet(
            OUT_DIR / f"season={season}.parquet", index=False
        )
        complete = table.filter(like="__r1_total").notna().all(axis=1).sum()
        print(
            f"{season}: {len(table):,} games ({complete:,} with all four versions), "
            f"max minutes > 48 in {(table['max_minutes'] > 48).mean():.1%}, "
            f"{time.time() - started:.0f}s",
            flush=True,
        )


def _metrics(frame: pd.DataFrame, total: str, impact: str) -> dict:
    error = frame["TOTAL_POINTS"] - frame["LINE"]
    edge = frame[total] - frame["LINE"]
    decided = (error != 0) & (edge != 0)
    slope = np.polyfit(edge, error, 1)[0] if len(frame) > 2 else np.nan
    impact_slope = (
        np.polyfit(frame[impact], error, 1)[0] if frame[impact].std() > 0 else np.nan
    )
    return {
        "games": len(frame),
        "mae": float((frame[total] - frame["TOTAL_POINTS"]).abs().mean()),
        "edge_corr": float(np.corrcoef(edge, error)[0, 1]),
        "edge_slope": float(slope),
        "hit_rate": float((np.sign(edge) == np.sign(error))[decided].mean()),
        "impact_corr": (
            float(np.corrcoef(frame[impact], error)[0, 1])
            if frame[impact].std() > 0
            else np.nan
        ),
        "impact_slope": float(impact_slope),
    }


def summarize(seasons: list[int]) -> None:
    # The season before the first, when built, only warms up the level
    # calibration (50 earlier games of the same phase); it is not reported.
    warmup = OUT_DIR / f"season={seasons[0] - 1}.parquet"
    paths = [warmup] if warmup.exists() else []
    paths += [OUT_DIR / f"season={s}.parquet" for s in seasons]
    table = pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True
    ).sort_values(["GAME_DATE", "game_id"], kind="mergesort")
    totals = [f"{v}__{r}_total" for v in VERSIONS for r in ("r1", "r2")]
    table = table.dropna(subset=[*totals, "TOTAL_POINTS", "LINE"]).reset_index(
        drop=True
    )
    phases = game_phase(table["game_id"])
    for column in totals:
        offset = walk_forward_offset(
            table["GAME_DATE"], table[column], table["TOTAL_POINTS"], phases=phases
        )
        table[f"{column}_cal"] = table[column] + offset
        table[f"{column}_impact"] = table[column] - table[f"{column}_healthy"]
    table = table.dropna(subset=[f"{c}_cal" for c in totals])
    table = table.loc[table["season"].isin(seasons)]
    absence = (table["v0__r1_total_impact"]).abs() > ABSENCE_IMPACT_SLICE
    slices = {
        "all": np.ones(len(table), bool),
        f"|v0 R1 impact| > {ABSENCE_IMPACT_SLICE:g}": absence.to_numpy(),
        "key defender out": table["key_defender_out"].astype(bool).to_numpy(),
    }
    print(
        f"{len(table):,} games with all four versions and a calibrated level "
        f"(seasons {seasons[0]}-{seasons[-1]}); max minutes > 48 in "
        f"{(table['max_minutes'] > 48).mean():.1%} of v0 games; observed rate for "
        f"{table['observed_rate_share'].mean():.1%} of actual opponent pairs"
    )
    pd.set_option("display.width", 220)
    for name, mask in slices.items():
        rows = {}
        for readout in ("r1", "r2"):
            for version in VERSIONS:
                column = f"{version}__{readout}_total"
                rows[(readout.upper(), version)] = _metrics(
                    table.loc[mask], f"{column}_cal", f"{column}_impact"
                )
        print(f"\n== {name} ==")
        print(pd.DataFrame(rows).T.round(4).to_string())
    print("\n== paired difference vs v0 (same readout), mean ± SE ==")
    print("   |error| < 0 and hit > 0 are better than v0")
    paired = {}
    for name, mask in slices.items():
        part = table.loc[mask]
        error = part["TOTAL_POINTS"] - part["LINE"]
        for readout in ("r1", "r2"):
            base = f"v0__{readout}_total_cal"
            base_abs = (part[base] - part["TOTAL_POINTS"]).abs()
            base_hit = np.sign(part[base] - part["LINE"]) == np.sign(error)
            for version in VERSIONS[1:]:
                column = f"{version}__{readout}_total_cal"
                diff = (part[column] - part["TOTAL_POINTS"]).abs() - base_abs
                hit = (np.sign(part[column] - part["LINE"]) == np.sign(error)).astype(
                    float
                ) - base_hit.astype(float)
                paired[(name, readout.upper(), version)] = {
                    "abs_error": f"{diff.mean():+.3f} ± {diff.std() / np.sqrt(len(diff)):.3f}",
                    "hit_rate": f"{hit.mean():+.4f} ± {hit.std() / np.sqrt(len(hit)):.4f}",
                }
        r2 = (part["v0__r2_total_cal"] - part["TOTAL_POINTS"]).abs() - (
            part["v0__r1_total_cal"] - part["TOTAL_POINTS"]
        ).abs()
        paired[(name, "R2 - R1", "v0")] = {
            "abs_error": f"{r2.mean():+.3f} ± {r2.std() / np.sqrt(len(r2)):.3f}",
            "hit_rate": "",
        }
    print(pd.DataFrame(paired).T.to_string())

    print("\n== total MAE by season ==")
    by_season = {}
    for readout in ("r1", "r2"):
        for version in VERSIONS:
            column = f"{version}__{readout}_total_cal"
            by_season[(readout.upper(), version)] = (
                (table[column] - table["TOTAL_POINTS"])
                .abs()
                .groupby(table["season"])
                .mean()
            )
    print(pd.DataFrame(by_season).T.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "summarize"])
    parser.add_argument("--seasons", type=season_range, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build(args.seasons, args.force)
    else:
        summarize(args.seasons)


if __name__ == "__main__":
    main()
