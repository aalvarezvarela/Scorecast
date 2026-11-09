"""Madrid betting-hour comparison and auditable per-game snapshot selection."""

import altair as alt
import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.analysis import MIN_CELL_BETS
from nba_ou.dashboard.domain.clock import (
    SERIES,
    SOURCES,
    clock_bets,
    clock_summary,
    hour_label,
)
from nba_ou.dashboard.domain.dates import madrid_time
from nba_ou.dashboard.repositories.clock import CLOCK_HOURS
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.analysis_charts import METRICS, MODEL_COLORS
from nba_ou.dashboard.ui.loaders import load_clock_history

SUMMARY_NAMES = {
    "series": "Series",
    "hour_label": "Madrid hour",
    "win_pct": "Win %",
    "wins": "Wins",
    "losses": "Losses",
    "pushes": "Pushes",
    "bets": "Bets",
    "ci_low": "95% CI low",
    "ci_high": "95% CI high",
    "roi": "ROI %",
    "priced_bets": "Priced bets",
    "mae": "MAE (pts)",
    "eligible_games": "Eligible final games",
    "covered_games": "Games with prediction + line",
    "coverage": "Coverage %",
    "missing": "Missing predictions",
    "no_line": "No line",
    "no_bet": "No bet",
    "pending": "Pending",
    "void": "Void",
    "mean_members": "Mean models per prediction",
    "median_quote_age_minutes": "Median quote age (min)",
}


def _clock_chart(summary: pd.DataFrame, metric: str):
    chart = alt.Chart(summary).encode(
        x=alt.X(
            "hour_label:O",
            sort=[hour_label(h) for h in CLOCK_HOURS],
            title="Betting hour · Madrid",
            axis=alt.Axis(labelAngle=-40),
        ),
        color=alt.Color(
            "series:N",
            title="Series",
            scale=alt.Scale(domain=list(SERIES), range=MODEL_COLORS[:3]),
        ),
    )
    line = chart.mark_line(point=True).encode(
        y=alt.Y(
            f"{METRICS[metric]}:Q",
            title="MAE (points)" if metric == "MAE" else metric,
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("hour_label:N", title="Madrid hour"),
            *[
                alt.Tooltip(f"{key}:Q", title=SUMMARY_NAMES[key], format=".1f")
                for key in (
                    "win_pct",
                    "bets",
                    "ci_low",
                    "ci_high",
                    "roi",
                    "priced_bets",
                    "coverage",
                    "eligible_games",
                    "covered_games",
                    "mae",
                    "mean_members",
                    "median_quote_age_minutes",
                )
            ],
        ],
    )
    if metric == "Win %":
        return (
            chart.mark_area(opacity=0.10).encode(y="ci_low:Q", y2="ci_high:Q") + line
        ).properties(height=320)
    return line.properties(height=320)


def _display_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = (
            pd.to_datetime(frame[column], utc=True)
            .dt.tz_convert("Europe/Madrid")
            .dt.strftime("%d/%m/%Y %H:%M %Z")
        )
    return frame


def render_clock_analysis(
    start, end, target, season_type, kind, thresholds, metric, refresh
):
    st.subheader("Betting hour · Madrid")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        source = (
            st.segmented_control(
                "Hourly source", SOURCES, default=SOURCES[0], key="history_clock_source"
            )
            or SOURCES[0]
        )
        hour = st.selectbox(
            "Betting hour (Madrid)",
            CLOCK_HOURS,
            index=CLOCK_HOURS.index(19),
            format_func=hour_label,
            key="history_clock_hour",
            width=230,
        )
        common = st.toggle(
            "Compare common games only",
            key="history_clock_common",
            help="Requires a prediction and betting line for every displayed series at every displayed hour. No bet due to minimum edge still counts as covered.",
        )
    first_hour, last_hour = st.select_slider(
        "Hours to compare (Madrid)",
        options=CLOCK_HOURS,
        value=(CLOCK_HOURS[0], CLOCK_HOURS[-1]),
        format_func=hour_label,
        key="history_clock_range",
        help="Narrow the comparison window to compare the same games across a practical set of hours.",
    )
    st.caption(
        "12:00–23:00 uses the NBA slate date; 00:00 (+1 day) is the following midnight in Madrid. "
        "Started games and future betting times are excluded. Each series places at most one bet per game and hour."
    )
    if refresh:
        load_clock_history.clear(start, end, target, season_type)
    try:
        with st.spinner("Reading predictions available at each Madrid hour…"):
            data = load_clock_history(start, end, target, season_type)
    except DashboardDataError as exc:
        st.error(str(exc), icon=":material/database:")
        return
    if data.games.empty:
        st.info("No games were recorded for this date range and season type.")
        return
    bets, members = clock_bets(
        data.games,
        data.predictions,
        source,
        CLOCK_HOURS,
        data.fetched_at,
        kind=kind,
        target=target,
        thresholds=thresholds,
    )
    if source == SOURCES[1]:
        st.info(
            "Snapshot replay uses reconstructed input times, not actual publication. Only builds trained before the game's date are included. Saved inputs may contain later corrections; this is a retrospective simulation."
        )
    st.caption(
        "TabPFN: closest earlier snapshot, one output. Closing: latest output per closing slot. "
        "Intermediate: latest output per model/horizon, then an equal-weight mean. "
        "Outputs are converted to points/home margin before averaging and compared with the same latest stored quote for that game/hour/source. "
        "An old quote is an approximation to the market at the betting hour; its age is shown. ROI stakes one unit per priced bet; missing prices never receive assumed odds."
    )
    comparison = (
        bets.loc[bets["hour"].between(first_hour, last_hour)]
        if not bets.empty
        else bets
    )
    summary = clock_summary(comparison, common_games=common)
    if summary.empty:
        st.info(
            "No eligible games for this comparison. Try a narrower hour window, another date range, source, model family or turn off the common-games filter."
        )
    else:
        st.altair_chart(
            _clock_chart(summary, metric),
            width="stretch",
            alt="Performance of TabPFN, closing and intermediate predictions by Madrid betting hour",
        )
        st.caption(
            f"Win % intervals are Wilson 95% per cell. Fewer than {MIN_CELL_BETS} wins + losses is a small sample. "
            "Hours share games and are correlated; the highest historical result is exploratory. "
            "Coverage counts final games with a prediction and line, including no-bet signals. "
            "Without the common-games filter, the games and available horizons can change across hours."
        )
        at_hour = summary.loc[summary["hour"].eq(hour)]
        if hour < first_hour or hour > last_hour:
            st.caption(
                "The selected detail hour is outside the chart's comparison window."
            )
        with st.container(horizontal=True):
            for row in at_hour.itertuples():
                with st.container(border=True):
                    st.caption(f"{row.series} · {hour_label(hour)} Madrid")
                    st.metric(
                        "Win %", "—" if pd.isna(row.win_pct) else f"{row.win_pct:.1f}%"
                    )
                    st.caption(
                        f"{row.wins} W / {row.losses} L / {row.pushes} P · {row.priced_bets} priced bets · ROI {'—' if pd.isna(row.roi) else f'{row.roi:.1f}%'}"
                    )
                    st.caption(
                        f"Coverage {row.covered_games}/{row.eligible_games} final games · {'Small sample' if row.wins + row.losses < MIN_CELL_BETS else '≥ 30 settled bets'}"
                    )
        with st.expander("Hourly results"):
            st.dataframe(
                summary[list(SUMMARY_NAMES)].rename(columns=SUMMARY_NAMES).round(2),
                hide_index=True,
                width="stretch",
                alt="Hourly results, uncertainty, coverage and missing outputs",
            )
    st.subheader("Selected-hour game detail")
    st.caption(
        "This audit table includes all eligible games at the selected hour, even when the chart is restricted to common games."
    )
    if bets.empty:
        st.info("No elapsed pre-game betting times in this range.")
        return
    selected = bets.loc[bets["hour"].eq(hour)]
    if selected.empty:
        st.info("No eligible games at the selected hour.")
        return
    detail_names = {
        "game_date": "NBA slate",
        "away_team": "Away",
        "home_team": "Home",
        "series": "Series",
        "cutoff_at": "Betting time (Madrid)",
        "tipoff_at": "Tip-off (Madrid)",
        "members": "Models",
        "predicted_value": "Predicted points / margin",
        "reference_line": "Betting line",
        "comparison_edge": "Edge",
        "signal": "Signal",
        "result": "Result",
        "bet_price": "American odds",
        "unit_profit": "Profit (units)",
        "input_at": "Oldest input (Madrid)",
        "latest_input_at": "Newest input (Madrid)",
        "quote_input_at": "Quote snapshot (Madrid)",
        "quote_age_minutes": "Quote age (min)",
    }
    detail = _display_times(
        selected,
        ["cutoff_at", "tipoff_at", "input_at", "latest_input_at", "quote_input_at"],
    )
    st.dataframe(
        detail[list(detail_names)].rename(columns=detail_names).round(2),
        hide_index=True,
        width="stretch",
        alt="One row per eligible game and series, with prediction, quote, age, bet and result",
    )
    if not members.empty:
        with st.expander("Snapshots used at the selected hour"):
            used = members.loc[members["hour"].eq(hour)]
            used = _display_times(used, ["input_at", "available_at"])
            names = {
                "game_date": "NBA slate",
                "away_team": "Away",
                "home_team": "Home",
                "series": "Series",
                "dataset_type": "Phase (1 closing / 2 intermediate)",
                "model_horizon_minutes": "Model horizon (min)",
                "variant": "Variant",
                "input_at": "Input time (Madrid)",
                "available_at": "Actual publication (Madrid)",
                "implied_prediction": "Predicted points / margin",
            }
            st.dataframe(
                used[list(names)].rename(columns=names).round(2),
                hide_index=True,
                width="stretch",
                alt="Exact selected snapshots and their input times, publication times and predictions",
            )
    st.caption(
        "TabPFN snapshots at the same input time prefer later publication; remaining ties follow a fixed rule. Madrid daylight-saving changes are handled automatically."
    )
    st.caption(
        f"Hourly predictions read {madrid_time(data.fetched_at)} · cached for 15 minutes. Refresh reloads the selected range and target."
    )
