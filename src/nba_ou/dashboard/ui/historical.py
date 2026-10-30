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
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.analysis_charts import (
    METRICS,
    closing_replay_timeline,
    closing_timeline,
    intermediate_replay_timeline,
    live_heatmap,
    snapshot_comparison,
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
  threshold. Many cells and small samples can produce apparently strong results by chance;
  no best cell is selected. Ordinary independent-sample chi-square tests do not apply here.
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
    st.subheader("Performance overview")
    with st.container(horizontal=True):
        st.metric("Final games", f"{settled_games}/{len(data.games)}", border=True)
        _win_metric("Closing · Final", closing.loc[closing["bucket"].eq(0)])
        _win_metric("Closing · Replay", closing_replay)
        _win_metric(
            "Own snapshot · Live",
            diagonal.loc[diagonal["source"].eq("Live predictions")],
        )
        _win_metric(
            "Own snapshot · Replay",
            diagonal.loc[diagonal["source"].eq("Snapshot replay")],
        )
    st.caption(
        "Win % pools prediction rows; several rows can refer to the same game. Live and replay can cover different games. Matched counts appear below."
    )
    st.info(
        f"Cells with fewer than {MIN_CELL_BETS} wins + losses are grey. Intervals are per cell; cells share games. No best cell is selected."
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
    if live.empty:
        st.caption(
            "No live predictions for these filters. Replay is shown separately below."
        )
    else:
        cells = aggregate_cells(live, settled_games_total=settled_games)
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
                    live, by=by, settled_games_total=settled_games
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
