"""Season-level stored prediction analysis; all UI labels are English."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import (
    MODEL_NAMES,
    PREDICTION_TARGET_LABELS,
)
from nba_ou.dashboard.domain.analysis import (
    MIN_CELL_BETS,
    SEASON_TYPES,
    aggregate_cells,
    paired_snapshot_summary,
    prepare_analysis,
    season_start,
)
from nba_ou.dashboard.domain.dates import madrid_time, slate_date
from nba_ou.dashboard.domain.strategies import (
    EARLY_WINDOW_MINUTES,
    MAX_COMBINATION_SIZE,
    TOP_COMBINATIONS,
    TOP_SLOTS,
    candidate_rows,
    compare_with_best,
    cumulative_win_rate,
    rank_candidates,
    rank_combinations,
    same_timepoint_combinations,
    selection_chance,
    timepoint_members,
    window_rows,
)
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.analysis_charts import (
    METRICS,
    closing_replay_timeline,
    closing_timeline,
    efficacy_chart,
    intermediate_replay_timeline,
    live_heatmap,
    snapshot_comparison,
    strategy_chart,
)
from nba_ou.dashboard.ui.controls import minimum_edge_select
from nba_ou.dashboard.ui.loaders import load_history

METHOD_NOTES = """
- **Slot:** model kind, schema, target, closing/intermediate, trained horizon and variant.
  Refits replace the build in a slot; they do not create extra votes or games.
- **Time:** intermediate and reconstructed closing rows use their stored input horizon;
  live closing rows use run start.
  A bucket covers the time since that snapshot passed. T−18h+ means more than 18 hours.
  Final is the last live closing output available before tip-off, or the reconstructed
  T−0 input in replay, matching Yesterday. A T−4h replay never stands in for Final.
- **Availability:** live input, prediction and run completion must precede tip-off.
  Input-time buckets describe what the model knew, not when the output was ready to bet.
  Tooltips report the median publication delay. Each game/slot/source/bucket appears once.
- **Sources:** live uses daily/TabPFN outputs; replay uses reconstructed snapshots, including closing models evaluated at their stored input timepoint.
  TabPFN own-horizon rows can appear in both sources when genuinely available pre-game.
  Replay never fills a live gap. Reconstructed inputs may contain later corrections.
- **Grading:** each output uses its own stored Bet365 line and the selected minimum edge.
  Win % = wins / (wins + losses); pushes, no-bet, missing-line, pending and void rows are excluded.
  MAE includes all valid predictions on final games, including no-bet rows.
- **Coverage:** settled bets / expected final-game prediction opportunities. Missing
  predictions reduce coverage; pooled rows include one opportunity per observed slot/time cell.
- **ROI:** one unit staked per priced bet, using recorded American odds. Pushes return zero;
  missing/invalid prices are excluded, never replaced by −110. See the priced-bet count.
- **Intervals:** Wilson 95% intervals apply per cell. Different cells share games and are
  correlated. Pooled prediction counts are not independent matches or independent trials.
  The matched comparison uses only games with a settled directional bet in both sources.
- **Reference:** 52.38% breaks even only at −110 odds. It is not a universal profitability
  threshold. Many cells and small samples can produce apparently strong results by chance.
  Ordinary independent-sample chi-square tests do not apply here.
- **Windows:** ≥ 6h pools intermediate models trained at T−6h or earlier; < 6h the rest.
- **Slots:** each intermediate model at its own snapshot, and each closing model at each
  input timepoint (Final included), with at least the minimum settled bets, in the chosen
  source. The three with the highest win % are shown. They are picked on the same games
  they are scored on, so they overstate what those slots will do next.
  Coin-flip chance: how often the best of the same number of coin-flip cells, with the
  same bet counts, reaches that win %. Cells share games, so it is a conservative guide,
  not a p-value. The grey band is where one coin flip lands 95% of the time.
- **Random baseline:** stored random OVER/UNDER picks (totals only, from 2026-10-26),
  graded at their stored line. Total points shows the same bets as line error.
- **Combinations:** only slots available at the same timepoint are combined, e.g. at T−8h
  the XGBoost and TabPFN T−8h intermediate models and the closing models on the T−8h input
  (optionally both totals targets). Every subset of 2–4 of them, on the games where all have
  a line, is scored two ways: the **mean edge** (an edge is a prediction minus the line it
  was graded at; bet with the minimum edge of the member published last) and a **majority
  vote** of the members' own signals (more than half of the members on one side; with two,
  both must agree). Both bet at the line and price of the member published last. The ten
  best combinations with the minimum settled bets are shown; they are picked from many, so
  the coin-flip chance counts all of them. The paired columns compare each with the best
  slot on the games both bet; the sign test uses the games only one of them won.
- **Hidden variants:** TabPFN thinking-high outputs are deprecated and not shown.
"""


def _summary_table(cells: pd.DataFrame) -> pd.DataFrame:
    names = {
        "model_label": "Model",
        "model_horizon_minutes": "Model horizon (min)",
        "source": "Source",
        "bucket": "Input bucket (min; 0 = Final)",
        "games": "Distinct games",
        "rows": "Prediction rows",
        "settled_games": "Final games with outputs",
        "wins": "Wins",
        "losses": "Losses",
        "pushes": "Pushes",
        "bets": "Bets",
        "win_pct": "Win %",
        "ci_low": "95% CI low",
        "ci_high": "95% CI high",
        "coverage": "Coverage %",
        "mae": "MAE (pts)",
        "roi": "ROI %",
        "priced_bets": "Priced bets",
        "median_lag_minutes": "Median publication lag (min)",
        "no_bet": "No bet",
        "no_line": "No line",
        "pending": "Pending",
        "void": "Void",
    }
    return (
        cells[[name for name in names if name in cells]].rename(columns=names).round(2)
    )


def _win_metric(label: str, rows: pd.DataFrame) -> None:
    if rows.empty:
        st.metric(label, "—", border=True)
        return
    pooled = aggregate_cells(rows, by=[]).iloc[0]
    value = "—" if pd.isna(pooled["win_pct"]) else f"{pooled['win_pct']:.1f}%"
    st.metric(
        label,
        value,
        border=True,
        help=f"{pooled['wins']:.0f} wins / {pooled['losses']:.0f} losses / {pooled['pushes']:.0f} pushes. "
        f"{pooled['games']:.0f} distinct games; {pooled['rows']:.0f} prediction rows. "
        "Pooled counts can repeat the same game across models/horizons; they are not independent trials.",
    )


def _random_baseline(baselines: pd.DataFrame) -> pd.DataFrame:
    if baselines.empty:
        return baselines
    # A random pick is a ±1-point edge; any positive edge places it.
    return prepare_analysis(baselines, min_edge={2: 0.0})


def _partner_rows(start, end, target, season_type, kind, thresholds, enabled):
    """The other totals target's rows, for line error + total points means."""
    if not enabled or target == 3:
        return None
    try:
        with st.spinner("Reading the other totals target for combinations…"):
            data = load_history(start, end, 2 if target == 1 else 1, season_type)
    except DashboardDataError as exc:
        st.warning(f"Cross-target combinations are unavailable: {exc}")
        return None
    raw = data.predictions
    if kind is not None and not raw.empty:
        raw = raw.loc[raw["kind"].eq(kind)]
    return prepare_analysis(raw, min_edge=thresholds)


ORDINALS = ("Best", "2nd best", "3rd best")


def _slot_card(rank: int, slot) -> None:
    with st.container(border=True, gap="xxsmall"):
        st.caption(f"{ORDINALS[rank]} slot")
        st.markdown(f"**{slot['candidate']}**")
        with st.container(horizontal=True, gap="small"):
            st.metric(
                "Win %",
                f"{slot['win_pct']:.1f}%",
                help=f"{slot['wins']:.0f} wins / {slot['losses']:.0f} losses / "
                f"{slot['pushes']:.0f} pushes. 95% CI "
                f"{slot['ci_low']:.1f}–{slot['ci_high']:.1f}%.",
            )
            st.metric("Bets", f"{slot['wins'] + slot['losses']:.0f}")
            st.metric(
                "ROI",
                "—" if pd.isna(slot["roi"]) else f"{slot['roi']:+.1f}%",
                help=f"{slot['priced_bets']:.0f} bets with a recorded price.",
            )


def _coin_flip_metric(chance: float, tried: int, win_pct: float, what: str) -> None:
    st.metric(
        "Coin-flip chance",
        "—" if pd.isna(chance) else f"{100 * chance:.0f}%",
        border=True,
        width="content",
        help=f"How often the best of {tried} coin-flip {what} with these bet counts "
        f"reaches {win_pct:.1f}%. High means the leader is indistinguishable from "
        "luck. Cells share games, so it is a conservative guide, not a p-value.",
    )


def _render_best_and_combinations(
    rows: pd.DataFrame,
    random_rows: pd.DataFrame,
    games: pd.DataFrame,
    *,
    thresholds,
    partner: pd.DataFrame | None,
) -> None:
    st.subheader("Best single slots")
    source = (
        st.segmented_control(
            "Source",
            ["Snapshot replay", "Live predictions"],
            default="Snapshot replay",
            key="history_best_source",
            help="Replay covers every game at every horizon; live covers only the "
            "snapshots a run happened to hit.",
        )
        or "Snapshot replay"
    )
    candidates = candidate_rows(rows, source)
    ranking = rank_candidates(candidates)
    eligible = ranking.loc[ranking["eligible"]] if not ranking.empty else ranking
    if eligible.empty:
        st.caption(
            f"No slot has {MIN_CELL_BETS} settled bets in {source.lower()} yet for these filters."
        )
        return
    top = eligible.head(TOP_SLOTS)
    best = top.iloc[0]
    slot_rows = {
        slot["candidate"]: candidates.loc[candidates["candidate"].eq(slot["candidate"])]
        for _, slot in top.iterrows()
    }
    best_rows = slot_rows[best["candidate"]]
    chance = selection_chance(eligible["wins"] + eligible["losses"], best["win_pct"])
    with st.container(horizontal=True, key="history_top_slots"):
        for rank, (_, slot) in enumerate(top.iterrows()):
            _slot_card(rank, slot)
        _coin_flip_metric(chance, len(eligible), best["win_pct"], "slots")
    st.caption(
        f"Ranked among {len(eligible)} slots with at least {MIN_CELL_BETS} settled bets "
        f"in {source.lower()}."
    )

    partner_candidates = (
        None if partner is None or partner.empty else candidate_rows(partner, source)
    )
    members = timepoint_members(candidates, partner_candidates)
    combined = same_timepoint_combinations(members, min_edge=thresholds)
    combo_ranking = rank_combinations(combined)
    eligible_combos = (
        combo_ranking.loc[combo_ranking["eligible"]]
        if not combo_ranking.empty
        else combo_ranking
    )
    top_combos = eligible_combos.head(TOP_COMBINATIONS)
    combo_rows = {
        name: combined.loc[combined["strategy"].eq(name)]
        for name in top_combos.get("strategy", [])
    }

    game_dates = games.set_index("game_id")["game_date"]
    roles = ("Best slot", "2nd best slot", "3rd best slot")
    curves = [
        cumulative_win_rate(frame, game_dates, name).assign(role=role)
        for (name, frame), role in zip(slot_rows.items(), roles, strict=False)
    ]
    own = candidates.loc[candidates["on_horizon"]]
    curves.append(
        cumulative_win_rate(own, game_dates, "All own snapshots (pooled)").assign(
            role="All own snapshots"
        )
    )
    if combo_rows:
        name = next(iter(combo_rows))
        curves.append(
            cumulative_win_rate(combo_rows[name], game_dates, name).assign(
                role="Best combination"
            )
        )
    if not random_rows.empty:
        curves.append(
            cumulative_win_rate(random_rows, game_dates, "Random baseline").assign(
                role="Random baseline"
            )
        )
    curves = [curve for curve in curves if "win_pct" in curve]
    st.altair_chart(
        efficacy_chart(pd.concat(curves, ignore_index=True)),
        width="stretch",
        alt="Cumulative win % of the three best slots by slate date",
    )
    st.caption(
        "Gold band: the best slot's 95% interval · Grey band: where a coin flip with the same bets lands 95% of the time · "
        "Dashed: pooled own snapshots and the random baseline · Dashed rule: 52.38% break-even at −110. "
        "The slots are chosen on these same games, so their curves flatter them."
    )
    with st.expander("Slot ranking"):
        st.dataframe(
            _summary_table(
                ranking.drop(columns="model_label").rename(
                    columns={"candidate": "model_label"}
                )
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Combinations · same timepoint")
    if combo_ranking.empty:
        st.caption(
            "No timepoint has two slots with lines on the same games in this source."
        )
        return
    if top_combos.empty:
        st.caption(
            f"{len(combo_ranking)} combinations tried; none has {MIN_CELL_BETS} settled bets yet."
        )
        return
    leader = top_combos.iloc[0]
    combo_chance = selection_chance(
        eligible_combos["wins"] + eligible_combos["losses"], leader["win_pct"]
    )
    table = compare_with_best(
        {
            **{f"{name} (slot)": frame for name, frame in slot_rows.items()},
            **combo_rows,
            **({"Random baseline": random_rows} if not random_rows.empty else {}),
        },
        best_rows,
    )
    table["role"] = "Combination"
    table.loc[table["strategy"].str.endswith(" (slot)"), "role"] = "Single slot"
    table.loc[table["strategy"].eq("Random baseline"), "role"] = "Random baseline"
    first = table.loc[table["strategy"].eq(leader["strategy"])].iloc[0]
    if leader["win_pct"] <= best["win_pct"]:
        st.info(
            f"No combination beats the best single slot ({best['win_pct']:.1f}%). "
            f"Highest combination: {leader['strategy']} at {leader['win_pct']:.1f}%."
        )
    else:
        p_value = first["sign_test_p"]
        verdict = (
            "a difference unlikely to be noise"
            if p_value < 0.05
            else "not distinguishable from the best slot yet"
        )
        notice = st.success if p_value < 0.05 else st.info
        notice(
            f"{leader['strategy']} wins {leader['win_pct']:.1f}% vs "
            f"{best['win_pct']:.1f}% for the best slot. On the "
            f"{first['matched_games']:.0f} games both bet it is "
            f"{first['matched_delta_pp']:+.1f} pp (sign test p = {p_value:.2f}), "
            f"{verdict}. It is plotted above as the best combination."
        )
    with st.container(horizontal=True, key="history_top_combination"):
        st.metric(
            "Best combination · win %",
            f"{leader['win_pct']:.1f}%",
            border=True,
            width="content",
            help=f"{leader['strategy']}: {leader['wins']:.0f} wins / "
            f"{leader['losses']:.0f} losses / {leader['pushes']:.0f} pushes.",
        )
        _coin_flip_metric(
            combo_chance, len(eligible_combos), leader["win_pct"], "combinations"
        )
    st.caption(
        f"Top {len(top_combos)} of {len(eligible_combos)} combinations with at least "
        f"{MIN_CELL_BETS} settled bets ({len(combo_ranking)} tried: subsets of 2–"
        f"{MAX_COMBINATION_SIZE} slots at one timepoint, mean edge and majority vote)."
    )
    ordered = pd.concat(
        [
            table.loc[table["role"].eq("Single slot")],
            table.loc[table["role"].eq("Combination")],
            table.loc[table["role"].eq("Random baseline")],
        ]
    )
    st.altair_chart(
        strategy_chart(ordered),
        width="stretch",
        alt="Win % of the three best single slots, the ten best combinations and the random baseline",
    )
    st.caption(
        "Each combination uses only slots available at that timepoint and bets at the line "
        "of the member published last. Dashed rule: 52.38% break-even."
    )
    with st.expander("Combination data"):
        st.dataframe(
            ordered.drop(columns=["role", "low_sample"])
            .rename(
                columns={
                    "strategy": "Strategy",
                    "wins": "Wins",
                    "losses": "Losses",
                    "pushes": "Pushes",
                    "bets": "Bets",
                    "win_pct": "Win %",
                    "ci_low": "95% CI low",
                    "ci_high": "95% CI high",
                    "roi": "ROI %",
                    "mae": "MAE (pts)",
                    "matched_games": "Games bet by both",
                    "only_strategy_won": "Only strategy won",
                    "only_best_won": "Only best won",
                    "matched_delta_pp": "Δ vs best on those (pp)",
                    "sign_test_p": "Sign test p",
                }
            )
            .round(2),
            hide_index=True,
            width="stretch",
        )
    with st.expander("All combinations tried"):
        st.dataframe(
            combo_ranking.rename(
                columns={
                    "strategy": "Strategy",
                    "method": "Method",
                    "timepoint": "Timepoint (min; 0 = Final)",
                    "members": "Members",
                    "member_names": "Slots",
                    "wins": "Wins",
                    "losses": "Losses",
                    "pushes": "Pushes",
                    "win_pct": "Win %",
                    "eligible": f"≥ {MIN_CELL_BETS} settled bets",
                }
            ).round(2),
            hide_index=True,
            width="stretch",
        )


def render_historical() -> None:
    today = slate_date()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        dates = st.date_input(
            "NBA slate dates",
            value=(season_start(today), today),
            format="DD/MM/YYYY",
            key="history_dates",
            width=280,
            help="Defaults to the current NBA season through today, including historical backfills. Inclusive NBA slate dates use the same Pacific date boundary as Today. Displayed timestamps use Madrid time.",
        )
        season_type = st.selectbox(
            "Season type",
            [2, 4, 5, 1, 3, None],
            format_func=lambda value: SEASON_TYPES.get(value, "All season types"),
            key="history_season_type",
            width=210,
        )
        kind = st.selectbox(
            "Model family",
            [None, 1, 3],
            format_func=lambda value: MODEL_NAMES.get(value, "All models"),
            key="history_kind",
            width=210,
        )
        refresh = st.button("Refresh", icon=":material/refresh:", key="history_refresh")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        target = (
            st.segmented_control(
                "Prediction target",
                list(PREDICTION_TARGET_LABELS),
                format_func=PREDICTION_TARGET_LABELS.get,
                default=1,
                key="history_target",
                width=450,
            )
            or 1
        )
        metric = (
            st.segmented_control(
                "Metric",
                list(METRICS),
                default="Win %",
                key="history_metric",
                width=350,
            )
            or "Win %"
        )
    with st.popover("Settings", icon=":material/tune:"):
        thresholds = {
            model_kind: minimum_edge_select(model_kind, page="history")
            for model_kind in MODEL_NAMES
        }
        cross_target = st.toggle(
            "Combine line error + total points",
            key="history_cross_target",
            disabled=target == 3,
            help="Adds the other totals target's slots as combination members at "
            "each timepoint. Reads it from Aiven once (cached).",
        )
    if len(dates) != 2:
        st.info("Choose a start and end date to view historical performance.")
        return
    start, end = dates
    st.caption(f"NBA slates · {start:%d %b %Y} – {end:%d %b %Y} · All times in Madrid")
    if refresh:
        load_history.clear(start, end, target, season_type)
    try:
        with st.spinner("Reading historical predictions and results from Aiven…"):
            data = load_history(start, end, target, season_type)
    except DashboardDataError as exc:
        st.error(str(exc), icon=":material/database:")
        return
    if data.games.empty:
        st.info("No games were recorded in Aiven for this date range and season type.")
        return
    final = (
        data.games["status"].eq(1)
        & data.games["home_points"].notna()
        & data.games["away_points"].notna()
    )
    settled_games = int(final.sum())
    pending_games = int(
        (data.games["status"].eq(0) & data.games["game_date"].lt(today)).sum()
    )
    if pending_games:
        st.caption(
            f"{pending_games} past games are awaiting final scores. Their predictions remain visible; win rate and MAE exclude unfinished results."
        )
    raw = data.predictions
    if kind is not None and not raw.empty:
        raw = raw.loc[raw["kind"].eq(kind)]
    rows = prepare_analysis(raw, min_edge=thresholds)
    if rows.empty:
        st.metric("Final games", f"{settled_games}/{len(data.games)}", border=True)
        st.info(
            "No stored predictions match this target and model family in the selected range."
        )
        return
    live = rows.loc[rows["source"].eq("Live predictions")]
    closing = live.loc[live["dataset_type"].eq(1)]
    closing_replay = rows.loc[
        rows["source"].eq("Snapshot replay") & rows["dataset_type"].eq(1)
    ]
    diagonal = rows.loc[rows["on_horizon"]]
    random_rows = _random_baseline(data.baselines)
    _render_best_and_combinations(
        rows,
        random_rows,
        data.games,
        thresholds=thresholds,
        partner=_partner_rows(
            start, end, target, season_type, kind, thresholds, cross_target
        ),
    )
    st.subheader("Performance overview")
    with st.container(horizontal=True):
        st.metric("Final games", f"{settled_games}/{len(data.games)}", border=True)
        _win_metric("Closing · Final", closing.loc[closing["bucket"].eq(0)])
        _win_metric("Closing · Replay", closing_replay)
        if not random_rows.empty:
            _win_metric("Random baseline", random_rows)
    early = f"T−{EARLY_WINDOW_MINUTES // 60}h"
    for source, short in (("Live predictions", "Live"), ("Snapshot replay", "Replay")):
        own = diagonal.loc[diagonal["source"].eq(source)]
        for column, (label, window) in zip(
            st.columns(3),
            (
                (f"Own snapshot · {short}", "All horizons"),
                (f"{short} · ≥ 6h", "≥ 6h"),
                (f"{short} · < 6h", "< 6h"),
            ),
            strict=True,
        ):
            with column:
                _win_metric(label, window_rows(own, window))
    st.caption(
        f"Own snapshot: each intermediate model at its trained horizon. ≥ 6h = models trained at {early} or earlier; < 6h = closer to tip-off. "
        "Win % pools prediction rows; several rows can refer to the same game. Live and replay can cover different games. Matched counts appear below."
    )
    st.info(
        f"Cells with fewer than {MIN_CELL_BETS} wins + losses are grey. Intervals are per cell; cells share games."
    )
    st.subheader("Closing by prediction time")
    if closing.empty:
        st.caption("No live closing predictions for these filters.")
    else:
        cells = aggregate_cells(closing, settled_games_total=settled_games)
        st.altair_chart(closing_timeline(cells, metric), width="stretch")
        if metric == "Win %":
            st.caption(
                "Bands: Wilson 95% intervals · Numbers: bets · Dashed line: 52.38% break-even at −110."
            )
        with st.expander("Closing data"):
            st.dataframe(_summary_table(cells), hide_index=True, width="stretch")
    st.subheader("Closing by input time · Replay")
    if closing_replay.empty:
        st.caption("No reconstructed closing predictions for these filters.")
    else:
        cells = aggregate_cells(closing_replay, settled_games_total=settled_games)
        st.altair_chart(
            closing_replay_timeline(cells, metric),
            width="stretch",
            alt="Closing model performance at reconstructed pre-game snapshots",
        )
        st.caption(
            "Reconstructed closing models use each stored input snapshot. Final means an exact T−0 reconstruction; earlier snapshots keep their own time. These outputs do not enter live metrics."
        )
        with st.expander("Closing replay data"):
            st.dataframe(
                _summary_table(cells),
                hide_index=True,
                width="stretch",
                alt="Closing replay counts, win rate, errors and coverage by model and snapshot",
            )
    st.subheader("Intermediate by model horizon · Replay")
    replay = diagonal.loc[diagonal["source"].eq("Snapshot replay")]
    if replay.empty:
        st.caption("No own-snapshot intermediate replay predictions for these filters.")
    else:
        cells = aggregate_cells(replay, settled_games_total=settled_games)
        st.altair_chart(intermediate_replay_timeline(cells, metric), width="stretch")
        st.caption(
            "Each point represents an intermediate model at its own horizon, summarizing its replay predictions across the selected games."
        )
        if metric == "Win %":
            st.caption(
                "Bands: Wilson 95% intervals · Numbers: bets · Dashed line: 52.38% break-even at −110."
            )
        with st.expander("Intermediate replay data"):
            st.dataframe(_summary_table(cells), hide_index=True, width="stretch")
    st.subheader("Live model × time")
    # TabPFN predicts only at its scheduled runs, so its live cells add noise.
    live_models = live.loc[live["kind"].ne(3)]
    if live_models.empty:
        st.caption(
            "No live XGBoost predictions for these filters (TabPFN is not shown here). Replay is shown separately below."
        )
    else:
        cells = aggregate_cells(live_models, settled_games_total=settled_games)
        st.altair_chart(live_heatmap(cells, metric), width="stretch")
        st.caption(
            "Gold outline: the model's own input-time bucket. Left: earlier inputs than training; right: fresher inputs. Hover for counts, coverage, prices and publication lag."
        )
        with st.expander("Live cell data and pooled margins"):
            st.dataframe(_summary_table(cells), hide_index=True, width="stretch")
            st.caption(
                "Row margins pool time cells; column margins pool model slots. Counts are prediction rows, not independent games. Pooled intervals are intentionally omitted."
            )
            for by in (
                ("model_label", "dataset_type", "model_horizon_minutes"),
                ("bucket",),
            ):
                margins = aggregate_cells(
                    live_models, by=by, settled_games_total=settled_games
                ).drop(columns=["ci_low", "ci_high"])
                st.dataframe(_summary_table(margins), hide_index=True, width="stretch")
    st.subheader("Own snapshot · Live vs replay")
    if diagonal.empty:
        st.caption("No own-horizon predictions for these filters.")
    else:
        cells = aggregate_cells(diagonal, settled_games_total=settled_games)
        st.altair_chart(snapshot_comparison(cells), width="stretch")
        paired = paired_snapshot_summary(rows)
        st.caption(
            "Matched comparison: only the same game/model/horizon with a win or loss in both sources. Pushes and no-bet rows are excluded from both percentages."
        )
        if paired.empty:
            st.caption("No matched settled live/replay bets are available yet.")
        else:
            st.dataframe(paired.round(2), hide_index=True, width="stretch")
        with st.expander("Own-snapshot counts and coverage"):
            st.dataframe(_summary_table(cells), hide_index=True, width="stretch")
    with st.expander("Method notes"):
        st.markdown(METHOD_NOTES)
    st.caption(
        f"Historical predictions read {madrid_time(data.fetched_at)} · cached for 15 minutes. Refresh reloads the selected range and target."
    )
