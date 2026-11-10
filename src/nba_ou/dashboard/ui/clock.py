"""XGBoost and TabPFN, intermediate and closing, at each Madrid betting hour."""

import altair as alt
import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.analysis import BREAK_EVEN, MIN_CELL_BETS
from nba_ou.dashboard.domain.clock import (
    HOUR_COLUMN,
    SOURCES,
    best_cells,
    clock_bets,
    clock_summary,
    common_cohort,
    hour_label,
    hour_model_table,
    leader_checks,
    series_names,
)
from nba_ou.dashboard.domain.dates import madrid_time
from nba_ou.dashboard.repositories.clock import CLOCK_HOURS
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.analysis_charts import METRICS, MODEL_COLORS
from nba_ou.dashboard.ui.controls import coin_flip_metric, permutation_metric
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


def _clock_chart(summary: pd.DataFrame, metric: str, names: list[str]):
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
            scale=alt.Scale(domain=names, range=MODEL_COLORS[: len(names)]),
            legend=alt.Legend(orient="bottom"),
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


def _cell_style(win_pct: float, settled: float) -> str:
    if pd.isna(win_pct):
        return ""
    if settled < MIN_CELL_BETS:
        return "color: #8a8f98"
    distance = win_pct - BREAK_EVEN
    alpha = min(0.55, 0.12 + abs(distance) / 25)
    color = "46, 160, 67" if distance >= 0 else "218, 54, 51"
    return f"background-color: rgba({color}, {alpha:.2f})"


def _styled_table(summary: pd.DataFrame, table: pd.DataFrame, names: list[str]):
    """Green above break-even, red below; grey text when the sample is small."""
    cells = summary.assign(settled=summary["wins"] + summary["losses"])
    win_pct = cells.pivot(index="hour", columns="series", values="win_pct")
    settled = cells.pivot(index="hour", columns="series", values="settled")

    def styles(frame: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for position, hour in enumerate(sorted(win_pct.index)):
            for name in names:
                if name in win_pct:
                    css.loc[position, name] = _cell_style(
                        win_pct.loc[hour, name], settled.loc[hour, name]
                    )
        return css

    return table.style.apply(styles, axis=None)


def _render_best_cells(summary: pd.DataFrame, cohort: pd.DataFrame) -> None:
    st.subheader("Best hours and models")
    best = best_cells(summary)
    if best.empty:
        st.caption(
            f"No hour and series has {MIN_CELL_BETS} settled bets for these filters yet."
        )
        return
    chance, permutation, tried = leader_checks(cohort, summary)
    with st.container(horizontal=True, key="clock_best_cells"):
        for rank, row in enumerate(best.itertuples(), start=1):
            with st.container(border=True):
                st.caption(f"#{rank} · {row.hour_label} Madrid")
                st.metric(row.series, f"{row.win_pct:.1f}%")
                roi = "—" if pd.isna(row.roi) else f"{row.roi:+.1f}%"
                st.caption(
                    f"{row.wins} W / {row.losses} L / {row.pushes} P · ROI {roi} · "
                    f"Coverage {row.covered_games}/{row.eligible_games}"
                )
        with st.container(width="content", gap="small"):
            coin_flip_metric(chance, tried, best.loc[0, "win_pct"], "cells")
            permutation_metric(permutation, tried, "cells")
    st.caption(
        f"Picked from {tried} hour and series cells with at least {MIN_CELL_BETS} settled bets, "
        "on the same games they are scored on, so they flatter themselves. Hours share games and are correlated."
    )


def _render_hour_table(summary: pd.DataFrame, names: list[str]) -> None:
    st.subheader("Results by hour")
    table = hour_model_table(summary, names)
    st.dataframe(
        _styled_table(summary, table, names),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={
            HOUR_COLUMN: st.column_config.TextColumn(width="small", pinned=True)
        },
        alt="Win %, record and ROI of each series at each Madrid betting hour",
    )
    st.caption(
        f"Each cell: win %, wins-losses, ROI. Green beats the {BREAK_EVEN:.2f}% break-even at −110, red does not; "
        f"grey text has fewer than {MIN_CELL_BETS} wins + losses."
    )


def render_clock_analysis(
    start, end, target, season_type, kind, thresholds, metric, refresh
):
    st.subheader("Betting hour · Madrid")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        source = (
            st.segmented_control(
                "Hourly source", SOURCES, default=SOURCES[1], key="history_clock_source"
            )
            or SOURCES[1]
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
    if source == SOURCES[1]:
        st.info(
            "Snapshot replay uses reconstructed input times, not actual publication. Only builds trained before the game's date are included. "
            "Saved inputs may contain later corrections; this is a retrospective simulation."
        )
    st.caption(
        "At each betting hour every model contributes its previous available output. Intermediate series are the equal-weight mean of their models; closing series use the latest closing output, if there is one by then. "
        "All series are compared with the same latest stored Bet365 quote for that game and hour. "
        "12:00–23:00 uses the NBA slate date; 00:00 (+1 day) is the following midnight in Madrid. "
        "Started games and future betting times are excluded. An old quote is an approximation to the market at the betting hour; its age is shown."
    )
    bets, _ = clock_bets(
        data.games,
        data.predictions,
        data.quotes,
        source,
        CLOCK_HOURS,
        data.fetched_at,
        kind=kind,
        target=target,
        thresholds=thresholds,
    )
    comparison = (
        bets.loc[bets["hour"].between(first_hour, last_hour)]
        if not bets.empty
        else bets
    )
    cohort = common_cohort(comparison) if common else comparison
    summary = clock_summary(cohort)
    if summary.empty:
        st.info(
            "No eligible games for this comparison. Try a narrower hour window, another date range, source or model family, or turn off the common-games filter."
        )
    else:
        names = series_names(kind)
        _render_best_cells(summary, cohort)
        _render_hour_table(summary, names)
        st.altair_chart(
            _clock_chart(summary, metric, names),
            width="stretch",
            alt="Performance of XGBoost and TabPFN, intermediate and closing, by Madrid betting hour",
        )
        st.caption(
            f"Win % intervals are Wilson 95% per cell. Fewer than {MIN_CELL_BETS} wins + losses is a small sample. "
            "Hours share games and are correlated; the highest historical hour is exploratory. "
            "ROI stakes one unit per priced bet; missing prices never receive assumed odds. "
            "Coverage counts final games with a prediction and line, including no-bet signals."
        )
        with st.expander("Hourly results"):
            st.dataframe(
                summary[list(SUMMARY_NAMES)].rename(columns=SUMMARY_NAMES).round(2),
                hide_index=True,
                width="stretch",
                alt="Hourly results, uncertainty, coverage and missing outputs",
            )
    st.caption(
        f"Hourly predictions read {madrid_time(data.fetched_at)} · cached for 15 minutes. Refresh reloads the selected range and target."
    )
