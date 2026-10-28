import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

import sys

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from scripts.predict_nba_games import predict_nba_games as run_nba_predictor
except Exception:
    run_nba_predictor = None

from nba_ou.postgre_db.predictions.update.update_evaluation_predictions import (
    get_games_with_total_scored_points,
)
from nba_ou.utils.streamlit_utils import get_team_logo_url

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

MODEL_ORDER = ["full_dataset", "recent_games", "TabPFNRegressor"]
MODEL_LABELS = {
    "full_dataset": "Full Dataset",
    "recent_games": "Recent Games",
    "TabPFNRegressor": "TabPFN",
}
MODEL_PREFIXES = {
    "full_dataset": "full",
    "recent_games": "recent",
    "TabPFNRegressor": "tabpfn",
}
MODEL_WEIGHTS = {
    "full_dataset": 0.25,
    "recent_games": 0.25,
    "TabPFNRegressor": 0.50,
}


def set_runtime_env_from_secrets() -> None:
    try:
        os.environ["SUPABASE_DB_URL"] = st.secrets["DatabaseSupabase"][
            "SUPABASE_DB_URL"
        ]
        os.environ["SUPABASE_DB_PASSWORD"] = st.secrets["DatabaseSupabase"][
            "SUPABASE_DB_PASSWORD"
        ]
    except Exception:
        pass

    try:
        os.environ["ODDS_API_KEY"] = st.secrets["Odds"]["ODDS_API_KEY"]
    except Exception:
        pass


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
          /* Page container */
          .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.0rem;
            max-width: 1400px;
          }

          /* Sidebar polish */
          section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.15);
          }
          section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
          }

          /* Sidebar text sizing */
          section[data-testid="stSidebar"] h3 {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
          }
          section[data-testid="stSidebar"] div[role="radiogroup"] label {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
          }
          section[data-testid="stSidebar"] div[role="radiogroup"] label div {
            font-size: 1.5rem !important;
          }
          section[data-testid="stSidebar"] .stCaption {
            font-size: 1.15rem !important;
          }

          /* Typography */
          h1, h2, h3 {
            letter-spacing: -0.02em;
          }
          h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.2rem !important;
          }
          h2 {
            font-size: 1.8rem !important;
            font-weight: 750 !important;
          }
          h3 {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
          }

          /* Metrics */
          .stMetric label {
            font-size: 1.05rem !important;
            font-weight: 650 !important;
          }
          .stMetric [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
          }

          /* DataFrame readability */
          div[data-testid="stDataFrame"] div[role="gridcell"] {
            padding: 0.65rem !important;
          }
          div[data-testid="stDataFrame"] div[role="columnheader"] {
            font-weight: 750 !important;
            padding: 0.85rem !important;
          }

          /* "Hero" header container */
          .app-hero {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 18px 18px;
            background: linear-gradient(135deg,
              rgba(102, 126, 234, 0.15) 0%,
              rgba(118, 75, 162, 0.12) 100%);
            margin-bottom: 16px;
          }
          .app-subtitle {
            font-size: 1.05rem;
            opacity: 0.85;
            margin-top: 2px;
            margin-bottom: 10px;
          }
          .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 6px;
          }
          .chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(49, 51, 63, 0.10);
            font-size: 0.95rem;
            font-weight: 600;
          }

          /* Reduce visual noise on separators */
          hr {
            margin: 1.0rem 0;
            opacity: 0.25;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:12px;">
            <div style="flex: 1;">
              <div style="font-size: 0.95rem; font-weight: 700; opacity: 0.85;">NBA analytics</div>
              <div style="margin-top: 2px;">
                <span style="font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em;">
                  Over/Under Predictor
                </span>
              </div>
              <div class="app-subtitle">
                Predictions, results, and historical performance in one place.
              </div>
              <div class="chip-row">
                <span class="chip">Model: full_dataset</span>
                <span class="chip">Model: recent_games</span>
                <span class="chip">Model: TabPFNRegressor</span>
                <span class="chip">Timezone: Madrid (CEST)</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_model_type(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text in MODEL_ORDER:
        return text

    lowered = text.lower()
    if lowered == "full_dataset":
        return "full_dataset"
    if lowered == "recent_games":
        return "recent_games"
    if lowered in {"tabpfnregressor", "tabpfn"}:
        return "TabPFNRegressor"

    return None


def normalize_pick(value: object) -> str | float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    if text in {"OVER", "O", "1", "TRUE", "YES"}:
        return "OVER"
    if text in {"UNDER", "U", "0", "FALSE", "NO"}:
        return "UNDER"
    if text == "PUSH":
        return "PUSH"

    return np.nan


def format_madrid_datetime(series: pd.Series, fmt: str) -> pd.Series:
    dt_utc = pd.to_datetime(series, errors="coerce", utc=True)
    return dt_utc.dt.tz_convert("Europe/Madrid").dt.strftime(fmt)


def build_game_level_predictions(
    df: pd.DataFrame,
    prediction_cutoff: pd.Timestamp | None = None,
    model_version_filter: str | None = "1.0",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "model_type" not in work.columns:
        return pd.DataFrame()
    if "model_version" not in work.columns:
        work["model_version"] = np.nan

    work["model_type"] = work["model_type"].apply(normalize_model_type)
    work = work[work["model_type"].isin(MODEL_ORDER)].copy()

    if model_version_filter:
        normalized_version = str(model_version_filter).strip()
        work = work[
            work["model_version"].fillna("").astype(str).str.strip()
            == normalized_version
        ].copy()

    if work.empty:
        return pd.DataFrame()

    if "prediction_datetime" in work.columns:
        pred_dt = pd.to_datetime(work["prediction_datetime"], errors="coerce", utc=True)
    else:
        pred_dt = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")

    if "prediction_date" in work.columns:
        pred_date_dt = pd.to_datetime(
            work["prediction_date"], errors="coerce", utc=True
        )
    else:
        pred_date_dt = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
    work["prediction_datetime_utc"] = pred_dt.fillna(pred_date_dt)

    if prediction_cutoff is not None:
        cutoff = pd.Timestamp(prediction_cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        work = work[work["prediction_datetime_utc"] <= cutoff].copy()

    if work.empty:
        return pd.DataFrame()

    if "game_time" in work.columns:
        work["game_time_utc"] = pd.to_datetime(
            work["game_time"], errors="coerce", utc=True
        )
    else:
        work["game_time_utc"] = pd.NaT

    base_cols = [
        "game_id",
        "season_type",
        "game_date",
        "game_time",
        "game_time_utc",
        "team_name_team_home",
        "team_name_team_away",
        "total_over_under_line",
        "total_scored_points",
        "home_pts",
        "away_pts",
        "prediction_date",
        "prediction_datetime_utc",
        "time_to_match_minutes",
    ]
    available_base_cols = [col for col in base_cols if col in work.columns]

    base = (
        work.sort_values("prediction_datetime_utc")
        .groupby("game_id", as_index=False)
        .tail(1)[available_base_cols]
        .copy()
    )

    base = base.rename(
        columns={"prediction_datetime_utc": "latest_prediction_datetime"}
    )

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        per_model = work[work["model_type"] == model_type].copy()
        if per_model.empty:
            continue

        model_cols = [
            "game_id",
            "pred_pick",
            "pred_total_points",
            "pred_line_error",
            "prediction_datetime_utc",
        ]
        available_model_cols = [col for col in model_cols if col in per_model.columns]

        per_model = (
            per_model.sort_values("prediction_datetime_utc")
            .groupby("game_id", as_index=False)
            .tail(1)[available_model_cols]
        )

        rename_map = {
            "pred_pick": f"pick_{prefix}",
            "pred_total_points": f"pred_total_{prefix}",
            "pred_line_error": f"line_error_{prefix}",
            "prediction_datetime_utc": f"pred_dt_{prefix}",
        }
        per_model = per_model.rename(columns=rename_map)

        base = base.merge(per_model, on="game_id", how="left")

    line = pd.to_numeric(base.get("total_over_under_line"), errors="coerce")
    actual_total = pd.to_numeric(base.get("total_scored_points"), errors="coerce")

    # NumPy 2 raises DTypePromotionError when np.where mixes strings with np.nan.
    actual_side = pd.Series(index=base.index, dtype="object")
    actual_side.loc[actual_total > line] = "OVER"
    actual_side.loc[actual_total < line] = "UNDER"
    actual_side.loc[actual_total == line] = "PUSH"
    base["actual_side"] = actual_side

    def pick_from_diff(diff: pd.Series) -> pd.Series:
        out = pd.Series(index=diff.index, dtype="object")
        out.loc[diff > 0] = "OVER"
        out.loc[diff < 0] = "UNDER"
        out.loc[diff == 0] = "PUSH"
        return out

    model_pick_cols: list[str] = []
    model_total_cols: list[str] = []
    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        pick_col = f"pick_{prefix}"
        pred_total_col = f"pred_total_{prefix}"
        line_diff_col = f"line_diff_{prefix}"
        error_col = f"error_{prefix}"
        correct_col = f"correct_{prefix}"

        if pick_col not in base.columns:
            base[pick_col] = np.nan
        if pred_total_col not in base.columns:
            base[pred_total_col] = np.nan

        base[pick_col] = base[pick_col].apply(normalize_pick)
        base[pred_total_col] = pd.to_numeric(base[pred_total_col], errors="coerce")
        base[line_diff_col] = base[pred_total_col] - line

        derived_pick = pick_from_diff(base[line_diff_col])
        base[pick_col] = base[pick_col].where(base[pick_col].notna(), derived_pick)

        base[error_col] = base[pred_total_col] - actual_total
        base[correct_col] = (base[pick_col] == base["actual_side"]) & base[
            "actual_side"
        ].isin(["OVER", "UNDER"])

        model_pick_cols.append(pick_col)
        model_total_cols.append(pred_total_col)

    weighted_numerator = pd.Series(0.0, index=base.index)
    weighted_denominator = pd.Series(0.0, index=base.index)
    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        weight = MODEL_WEIGHTS[model_type]
        pred_total_col = f"pred_total_{prefix}"
        pred_total = pd.to_numeric(base[pred_total_col], errors="coerce")
        valid_mask = pred_total.notna()
        weighted_numerator += pred_total.fillna(0.0) * weight
        weighted_denominator += valid_mask.astype(float) * weight

    base["consensus_pred_total"] = weighted_numerator / weighted_denominator.replace(
        0, np.nan
    )
    base["consensus_line_diff"] = base["consensus_pred_total"] - line
    base["consensus_pick"] = pick_from_diff(base["consensus_line_diff"])
    base["consensus_error"] = base["consensus_pred_total"] - actual_total
    base["consensus_correct"] = (base["consensus_pick"] == base["actual_side"]) & base[
        "actual_side"
    ].isin(["OVER", "UNDER"])

    # Consensus without TabPFN (average of full_dataset and recent_games only)
    no_tabpfn_numerator = pd.Series(0.0, index=base.index)
    no_tabpfn_denominator = pd.Series(0.0, index=base.index)
    for model_type in ["full_dataset", "recent_games"]:
        prefix = MODEL_PREFIXES[model_type]
        pred_total_col = f"pred_total_{prefix}"
        pred_total = pd.to_numeric(base[pred_total_col], errors="coerce")
        valid_mask = pred_total.notna()
        no_tabpfn_numerator += pred_total.fillna(0.0)
        no_tabpfn_denominator += valid_mask.astype(float)

    base["consensus_no_tabpfn_pred_total"] = (
        no_tabpfn_numerator / no_tabpfn_denominator.replace(0, np.nan)
    )
    base["consensus_no_tabpfn_line_diff"] = (
        base["consensus_no_tabpfn_pred_total"] - line
    )
    base["consensus_no_tabpfn_pick"] = pick_from_diff(
        base["consensus_no_tabpfn_line_diff"]
    )
    base["consensus_no_tabpfn_error"] = (
        base["consensus_no_tabpfn_pred_total"] - actual_total
    )
    base["consensus_no_tabpfn_correct"] = (
        base["consensus_no_tabpfn_pick"] == base["actual_side"]
    ) & base["actual_side"].isin(["OVER", "UNDER"])

    base["all_models_available"] = base[model_total_cols].notna().all(axis=1)
    base["all_models_agree"] = (
        base[model_pick_cols].nunique(axis=1, dropna=True).eq(1)
        & base["all_models_available"]
    )

    if "game_time_utc" in base.columns:
        base = base.sort_values("game_time_utc")

    return base.reset_index(drop=True)


def build_upcoming_display(df: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame()
    display["Matchup"] = df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    display["Game Time (Madrid)"] = format_madrid_datetime(
        df["game_time_utc"], "%Y-%m-%d %H:%M"
    )
    display["O/U Line"] = pd.to_numeric(
        df["total_over_under_line"], errors="coerce"
    ).round(1)

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        label = MODEL_LABELS[model_type]
        display[f"{label} Total"] = pd.to_numeric(
            df[f"pred_total_{prefix}"], errors="coerce"
        ).round(1)
        display[f"{label} Diff"] = pd.to_numeric(
            df[f"line_diff_{prefix}"], errors="coerce"
        ).round(2)
        display[f"{label} Pick"] = df[f"pick_{prefix}"]

    display["Consensus Total"] = pd.to_numeric(
        df["consensus_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus Diff"] = pd.to_numeric(
        df["consensus_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus Pick"] = df["consensus_pick"]

    display["Consensus (No TabPFN) Total"] = pd.to_numeric(
        df["consensus_no_tabpfn_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus (No TabPFN) Diff"] = pd.to_numeric(
        df["consensus_no_tabpfn_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus (No TabPFN) Pick"] = df["consensus_no_tabpfn_pick"]

    if "time_to_match_minutes" in df.columns:
        display["Time to Game (min)"] = (
            pd.to_numeric(df["time_to_match_minutes"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return display


def build_past_display(df: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame()
    display["Matchup"] = df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    display["Game Time (Madrid)"] = format_madrid_datetime(df["game_time_utc"], "%H:%M")
    display["O/U Line"] = pd.to_numeric(
        df["total_over_under_line"], errors="coerce"
    ).round(1)
    display["Actual Total"] = pd.to_numeric(
        df["total_scored_points"], errors="coerce"
    ).round(1)
    display["Actual Side"] = df["actual_side"]

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        label = MODEL_LABELS[model_type]
        display[f"{label} Total"] = pd.to_numeric(
            df[f"pred_total_{prefix}"], errors="coerce"
        ).round(1)
        display[f"{label} Diff"] = pd.to_numeric(
            df[f"line_diff_{prefix}"], errors="coerce"
        ).round(2)
        display[f"{label} Pick"] = df[f"pick_{prefix}"]
        display[f"{label} Correct"] = df[f"correct_{prefix}"].map(
            {True: "✅", False: "❌"}
        )

    display["Consensus Total"] = pd.to_numeric(
        df["consensus_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus Diff"] = pd.to_numeric(
        df["consensus_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus Pick"] = df["consensus_pick"]
    display["Consensus Correct"] = df["consensus_correct"].map(
        {True: "✅", False: "❌"}
    )

    display["Consensus (No TabPFN) Total"] = pd.to_numeric(
        df["consensus_no_tabpfn_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus (No TabPFN) Diff"] = pd.to_numeric(
        df["consensus_no_tabpfn_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus (No TabPFN) Pick"] = df["consensus_no_tabpfn_pick"]
    display["Consensus (No TabPFN) Correct"] = df["consensus_no_tabpfn_correct"].map(
        {True: "✅", False: "❌"}
    )

    return display


def format_pick_label(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    text = str(value).upper()
    if text == "OVER":
        return "Over"
    if text == "UNDER":
        return "Under"
    if text == "PUSH":
        return "Push"
    if text == "MIXED":
        return "Mixed"
    return str(value)


def get_pick_icon(pick_label: str) -> str:
    if pick_label == "Under":
        return "🔵"
    if pick_label == "Over":
        return "🔴"
    return "⚪"


def render_prediction_cards(df: pd.DataFrame, include_actual: bool = False) -> None:
    df = df.sort_values("game_time_utc").reset_index(drop=True)
    cols_per_row = 2

    for idx in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            row_idx = idx + col_idx
            if row_idx >= len(df):
                break

            row = df.iloc[row_idx]

            with col:
                home_team = row["team_name_team_home"]
                away_team = row["team_name_team_away"]

                game_dt = pd.to_datetime(
                    row["game_time_utc"], errors="coerce", utc=True
                )
                if pd.isna(game_dt):
                    game_time = "TBD"
                    game_date = "TBD"
                else:
                    game_dt_madrid = game_dt.tz_convert("Europe/Madrid")
                    game_time = game_dt_madrid.strftime("%I:%M %p")
                    game_date = game_dt_madrid.strftime("%b %d, %Y")

                line_value = pd.to_numeric(
                    row.get("total_over_under_line"), errors="coerce"
                )
                line_text = f"{line_value:.1f}" if pd.notna(line_value) else "N/A"

                picks = {
                    "Full Dataset": format_pick_label(row.get("pick_full")),
                    "Recent Games": format_pick_label(row.get("pick_recent")),
                    "TabPFN": format_pick_label(row.get("pick_tabpfn")),
                }

                totals = {
                    "Full Dataset": pd.to_numeric(
                        row.get("pred_total_full"), errors="coerce"
                    ),
                    "Recent Games": pd.to_numeric(
                        row.get("pred_total_recent"), errors="coerce"
                    ),
                    "TabPFN": pd.to_numeric(
                        row.get("pred_total_tabpfn"), errors="coerce"
                    ),
                }

                if include_actual:
                    correct_flags = [
                        row.get("correct_full"),
                        row.get("correct_recent"),
                        row.get("correct_tabpfn"),
                    ]
                    valid_flags = [bool(v) for v in correct_flags if pd.notna(v)]
                    if not valid_flags:
                        border_color = "#9E9E9E"
                    elif all(valid_flags):
                        border_color = "#4CAF50"
                    elif any(valid_flags):
                        border_color = "#FFA500"
                    else:
                        border_color = "#F44336"
                else:
                    consensus_pick = row.get("consensus_pick")
                    border_color = (
                        "#4CAF50" if consensus_pick in {"OVER", "UNDER"} else "#FF9800"
                    )

                header_html = f"""
                <div style="
                    border: 2px solid {border_color};
                    border-radius: 12px 12px 0 0;
                    overflow: hidden;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.15);
                ">
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        color: white;
                    ">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="flex: 1; text-align: center;">
                                <img src="{get_team_logo_url(home_team)}" width="88" style="margin-bottom: 8px;">
                                <div style="font-size: 1.05rem; font-weight: 700;">{home_team}</div>
                            </div>
                            <div style="flex: 0.45; text-align: center;">
                                <div style="font-size: 2rem; font-weight: 900; margin-bottom: 5px;">VS</div>
                                <div style="font-size: 0.95rem; font-weight: 600;">{game_date}</div>
                                <div style="font-size: 1.05rem; font-weight: 700; margin-top: 3px;">🕐 {game_time}</div>
                            </div>
                            <div style="flex: 1; text-align: center;">
                                <img src="{get_team_logo_url(away_team)}" width="88" style="margin-bottom: 8px;">
                                <div style="font-size: 1.05rem; font-weight: 700;">{away_team}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """
                st.components.v1.html(header_html, height=192)

                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 2px solid {border_color}; border-top: none;
                             border-radius: 0 0 12px 12px; padding: 14px; margin-top: -5px;
                             background: white;">
                        """,
                        unsafe_allow_html=True,
                    )

                    if include_actual:
                        actual_total = pd.to_numeric(
                            row.get("total_scored_points"), errors="coerce"
                        )
                        actual_side = row.get("actual_side")
                        actual_side_text = (
                            actual_side if pd.notna(actual_side) else "N/A"
                        )
                        actual_total_text = (
                            f"{actual_total:.1f}" if pd.notna(actual_total) else "N/A"
                        )
                        actual_color = (
                            "#2196F3" if actual_side == "UNDER" else "#FF5722"
                        )
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin-bottom: 10px; padding: 10px;
                                 background: {actual_color}; color: white; border-radius: 8px;">
                                <span style="font-size: 1.1rem; font-weight: 700;">
                                    ⚡ ACTUAL: {actual_side_text} ({actual_total_text} points)
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        consensus = row.get("consensus_pick")
                        consensus_text = format_pick_label(consensus)
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin-bottom: 10px; padding: 10px;
                                 background: #eef2ff; color: #1f2937; border-radius: 8px;">
                                <span style="font-size: 1.0rem; font-weight: 700;">
                                    ✅ CONSENSUS: {consensus_text}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("O/U Line", line_text)
                    with col2:
                        val = totals["Full Dataset"]
                        st.metric(
                            "Full Total", f"{val:.1f}" if pd.notna(val) else "N/A"
                        )
                    with col3:
                        val = totals["Recent Games"]
                        st.metric(
                            "Recent Total", f"{val:.1f}" if pd.notna(val) else "N/A"
                        )
                    with col4:
                        val = totals["TabPFN"]
                        st.metric(
                            "TabPFN Total", f"{val:.1f}" if pd.notna(val) else "N/A"
                        )

                    consensus_total = pd.to_numeric(
                        row.get("consensus_pred_total"), errors="coerce"
                    )
                    consensus_diff = pd.to_numeric(
                        row.get("consensus_line_diff"), errors="coerce"
                    )
                    consensus_pick = format_pick_label(row.get("consensus_pick"))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Consensus Total",
                            f"{consensus_total:.1f}"
                            if pd.notna(consensus_total)
                            else "N/A",
                        )
                    with col2:
                        st.metric(
                            "Consensus Diff",
                            f"{consensus_diff:+.2f}"
                            if pd.notna(consensus_diff)
                            else "N/A",
                        )
                    with col3:
                        st.metric("Consensus Pick", consensus_pick)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        diff = pd.to_numeric(row.get("line_diff_full"), errors="coerce")
                        st.metric(
                            "Full Diff", f"{diff:+.2f}" if pd.notna(diff) else "N/A"
                        )
                    with col2:
                        diff = pd.to_numeric(
                            row.get("line_diff_recent"), errors="coerce"
                        )
                        st.metric(
                            "Recent Diff", f"{diff:+.2f}" if pd.notna(diff) else "N/A"
                        )
                    with col3:
                        diff = pd.to_numeric(
                            row.get("line_diff_tabpfn"), errors="coerce"
                        )
                        st.metric(
                            "TabPFN Diff", f"{diff:+.2f}" if pd.notna(diff) else "N/A"
                        )

                    model_cols = st.columns(3)
                    model_keys = [
                        ("Full Dataset", "correct_full"),
                        ("Recent Games", "correct_recent"),
                        ("TabPFN", "correct_tabpfn"),
                    ]

                    for mcol, (model_label, correct_col) in zip(model_cols, model_keys):
                        with mcol:
                            pick_label = picks[model_label]
                            icon = get_pick_icon(pick_label)
                            if include_actual:
                                flag = row.get(correct_col)
                                status = "✅" if pd.notna(flag) and bool(flag) else "❌"
                                st.metric(model_label, f"{icon} {pick_label} {status}")
                            else:
                                st.metric(model_label, f"{icon} {pick_label}")

                    st.markdown("</div>", unsafe_allow_html=True)


def summarize_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])
    rows = []

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        picks = df[f"pick_{prefix}"]
        valid_mask = resolved_mask & picks.isin(["OVER", "UNDER"])

        n_games = int(valid_mask.sum())
        accuracy = (
            float(df.loc[valid_mask, f"correct_{prefix}"].mean()) if n_games else np.nan
        )
        mean_error = (
            float(df.loc[valid_mask, f"error_{prefix}"].mean()) if n_games else np.nan
        )
        mae = (
            float(df.loc[valid_mask, f"error_{prefix}"].abs().mean())
            if n_games
            else np.nan
        )
        mean_abs_line_diff = (
            float(df.loc[valid_mask, f"line_diff_{prefix}"].abs().mean())
            if n_games
            else np.nan
        )

        rows.append(
            {
                "Model": MODEL_LABELS[model_type],
                "Games": n_games,
                "Accuracy (%)": None if pd.isna(accuracy) else round(accuracy * 100, 2),
                "Mean Error": None if pd.isna(mean_error) else round(mean_error, 2),
                "MAE": None if pd.isna(mae) else round(mae, 2),
                "Avg |Diff vs Line|": None
                if pd.isna(mean_abs_line_diff)
                else round(mean_abs_line_diff, 2),
            }
        )

    # Add consensus row
    consensus_picks = df["consensus_pick"]
    consensus_valid_mask = resolved_mask & consensus_picks.isin(["OVER", "UNDER"])

    n_consensus_games = int(consensus_valid_mask.sum())
    consensus_accuracy = (
        float(df.loc[consensus_valid_mask, "consensus_correct"].mean())
        if n_consensus_games
        else np.nan
    )
    consensus_mean_error = (
        float(df.loc[consensus_valid_mask, "consensus_error"].mean())
        if n_consensus_games
        else np.nan
    )
    consensus_mae = (
        float(df.loc[consensus_valid_mask, "consensus_error"].abs().mean())
        if n_consensus_games
        else np.nan
    )
    consensus_mean_abs_line_diff = (
        float(df.loc[consensus_valid_mask, "consensus_line_diff"].abs().mean())
        if n_consensus_games
        else np.nan
    )

    rows.append(
        {
            "Model": "Consensus",
            "Games": n_consensus_games,
            "Accuracy (%)": None
            if pd.isna(consensus_accuracy)
            else round(consensus_accuracy * 100, 2),
            "Mean Error": None
            if pd.isna(consensus_mean_error)
            else round(consensus_mean_error, 2),
            "MAE": None if pd.isna(consensus_mae) else round(consensus_mae, 2),
            "Avg |Diff vs Line|": None
            if pd.isna(consensus_mean_abs_line_diff)
            else round(consensus_mean_abs_line_diff, 2),
        }
    )

    # Add consensus (no TabPFN) row
    consensus_no_tabpfn_picks = df["consensus_no_tabpfn_pick"]
    consensus_no_tabpfn_valid_mask = resolved_mask & consensus_no_tabpfn_picks.isin(
        ["OVER", "UNDER"]
    )

    n_consensus_no_tabpfn_games = int(consensus_no_tabpfn_valid_mask.sum())
    consensus_no_tabpfn_accuracy = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_correct"].mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )
    consensus_no_tabpfn_mean_error = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_error"].mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )
    consensus_no_tabpfn_mae = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_error"]
            .abs()
            .mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )
    consensus_no_tabpfn_mean_abs_line_diff = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_line_diff"]
            .abs()
            .mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )

    rows.append(
        {
            "Model": "Consensus (No TabPFN)",
            "Games": n_consensus_no_tabpfn_games,
            "Accuracy (%)": None
            if pd.isna(consensus_no_tabpfn_accuracy)
            else round(consensus_no_tabpfn_accuracy * 100, 2),
            "Mean Error": None
            if pd.isna(consensus_no_tabpfn_mean_error)
            else round(consensus_no_tabpfn_mean_error, 2),
            "MAE": None
            if pd.isna(consensus_no_tabpfn_mae)
            else round(consensus_no_tabpfn_mae, 2),
            "Avg |Diff vs Line|": None
            if pd.isna(consensus_no_tabpfn_mean_abs_line_diff)
            else round(consensus_no_tabpfn_mean_abs_line_diff, 2),
        }
    )

    return pd.DataFrame(rows)


def compute_threshold_accuracy_table(
    df: pd.DataFrame,
    *,
    model_thresholds: dict[str, tuple[float, ...]] | None = None,
) -> pd.DataFrame:
    if model_thresholds is None:
        model_thresholds = {
            "full_dataset": (0.0, 1.0, 2.0),
            "recent_games": (0.0, 1.0, 2.0),
            "TabPFNRegressor": (0.0, 0.5, 1.0, 1.5, 2.0),
            "consensus": (0.0, 0.5, 1.0, 1.5, 2.0),
            "consensus_no_tabpfn": (0.0, 0.5, 1.0, 1.5, 2.0),
        }

    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])
    rows: list[dict] = []

    for model_type, thresholds in model_thresholds.items():
        if model_type == "consensus":
            pick_col = "consensus_pick"
            correct_col = "consensus_correct"
            line_diff_col = "consensus_line_diff"
            model_label = "Consensus"
        elif model_type == "consensus_no_tabpfn":
            pick_col = "consensus_no_tabpfn_pick"
            correct_col = "consensus_no_tabpfn_correct"
            line_diff_col = "consensus_no_tabpfn_line_diff"
            model_label = "Consensus (No TabPFN)"
        else:
            prefix = MODEL_PREFIXES[model_type]
            pick_col = f"pick_{prefix}"
            correct_col = f"correct_{prefix}"
            line_diff_col = f"line_diff_{prefix}"
            model_label = MODEL_LABELS[model_type]

        for threshold in thresholds:
            mask = (
                resolved_mask
                & df[pick_col].isin(["OVER", "UNDER"])
                & (pd.to_numeric(df[line_diff_col], errors="coerce").abs() >= threshold)
            )
            n_games = int(mask.sum())
            accuracy = float(df.loc[mask, correct_col].mean()) if n_games else np.nan

            rows.append(
                {
                    "Model": model_label,
                    "Filter": f"|Diff vs Line| >= {threshold:g}",
                    "Games": n_games,
                    "Accuracy (%)": None
                    if pd.isna(accuracy)
                    else round(accuracy * 100, 2),
                }
            )

    return pd.DataFrame(rows)


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out_rows = []
    temp = df.copy()
    temp["game_date_dt"] = pd.to_datetime(temp["game_date"], errors="coerce")
    temp = temp.dropna(subset=["game_date_dt"])  # keep only valid dates

    for game_date, group in temp.groupby(temp["game_date_dt"].dt.date):
        resolved = group[group["actual_side"].isin(["OVER", "UNDER"])]

        for model_type in MODEL_ORDER:
            prefix = MODEL_PREFIXES[model_type]
            valid = resolved[resolved[f"pick_{prefix}"].isin(["OVER", "UNDER"])]
            n_games = len(valid)

            out_rows.append(
                {
                    "game_date": pd.to_datetime(game_date),
                    "model_type": model_type,
                    "model_label": MODEL_LABELS[model_type],
                    "n_games": n_games,
                    "accuracy": valid[f"correct_{prefix}"].mean()
                    if n_games
                    else np.nan,
                    "mae": valid[f"error_{prefix}"].abs().mean() if n_games else np.nan,
                }
            )

        # Add consensus metrics for this date
        consensus_valid = resolved[resolved["consensus_pick"].isin(["OVER", "UNDER"])]
        n_consensus_games = len(consensus_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus",
                "model_label": "Consensus",
                "n_games": n_consensus_games,
                "accuracy": consensus_valid["consensus_correct"].mean()
                if n_consensus_games
                else np.nan,
                "mae": consensus_valid["consensus_error"].abs().mean()
                if n_consensus_games
                else np.nan,
            }
        )

        # Add consensus (no TabPFN) metrics for this date
        consensus_no_tabpfn_valid = resolved[
            resolved["consensus_no_tabpfn_pick"].isin(["OVER", "UNDER"])
        ]
        n_consensus_no_tabpfn_games = len(consensus_no_tabpfn_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus_no_tabpfn",
                "model_label": "Consensus (No TabPFN)",
                "n_games": n_consensus_no_tabpfn_games,
                "accuracy": consensus_no_tabpfn_valid[
                    "consensus_no_tabpfn_correct"
                ].mean()
                if n_consensus_no_tabpfn_games
                else np.nan,
                "mae": consensus_no_tabpfn_valid["consensus_no_tabpfn_error"]
                .abs()
                .mean()
                if n_consensus_no_tabpfn_games
                else np.nan,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["game_date", "model_type"])


def show_upcoming_predictions(model_version_filter: str | None) -> None:
    st.markdown("### 🔄 Update Predictions")
    st.caption(
        "Run the prediction model to generate fresh predictions for today's games."
    )
    st.markdown("")

    if run_nba_predictor is None:
        st.warning("Predictor module could not be imported in this environment.")
    elif st.button("Run Predictor Now", type="primary", use_container_width=True):
        try:
            with st.spinner("Running predictor. This may take a few minutes..."):
                run_nba_predictor(run_tabpfn_client=True)

            st.success("Predictions updated. Reloading...")
            time.sleep(1.5)
            st.rerun()
        except Exception as exc:
            st.error(f"Error running predictor: {exc}")
            st.exception(exc)

    st.markdown("---")

    with st.spinner("Loading upcoming predictions..."):
        raw = get_games_with_total_scored_points(only_null=True)
        games = build_game_level_predictions(
            raw,
            model_version_filter=model_version_filter,
        )

    if games.empty:
        st.info("No upcoming predictions found.")
        return

    latest_prediction_time = pd.to_datetime(
        games["latest_prediction_datetime"], errors="coerce", utc=True
    ).max()

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Upcoming Games", len(games))
    with metric_cols[1]:
        available_all = int(games["all_models_available"].sum())
        st.metric("3 Models Available", f"{available_all}/{len(games)}")
    with metric_cols[2]:
        if pd.notna(latest_prediction_time):
            st.metric(
                "Latest Prediction",
                latest_prediction_time.tz_convert("Europe/Madrid").strftime(
                    "%Y-%m-%d %H:%M"
                ),
            )
        else:
            st.metric("Latest Prediction", "N/A")

    if pd.notna(latest_prediction_time):
        st.caption(
            "Latest prediction: "
            + latest_prediction_time.tz_convert("Europe/Madrid").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " (Madrid)"
        )

    st.markdown("---")
    st.markdown("## Today's Predictions")
    st.caption("View upcoming games with AI-powered over/under predictions.")
    st.markdown("")
    use_cards = st.toggle("Use Card View", value=True)

    if use_cards:
        render_prediction_cards(games, include_actual=False)
    else:
        st.dataframe(
            build_upcoming_display(games),
            width="stretch",
            hide_index=True,
            height=600,
        )

    st.markdown("---")
    st.markdown("")
    with st.expander("ℹ️ **How to Read the Predictions**", expanded=False):
        st.markdown(
            """
        ### 📊 Understanding the Predictions

        - **🏀 Matchup**: Home team vs Away team with logos
        - **⏰ Game Time**: When the game starts (Madrid timezone)
        - **📏 O/U Line**: The bookmaker's over/under betting line
        - **📌 Full/Recent/TabPFN**: Predictions from each model
        - **✅ Consensus**: Combined agreement signal across models
        - **Time to Game**: Minutes until game starts

        **Note**: Predictions are updated periodically. Most recent prediction time shown above.
        """
        )


def show_past_games_results(model_version_filter: str | None) -> None:
    st.markdown("## Past Games Results")
    st.caption("Compare predictions vs actual totals for a selected date.")
    st.markdown("")

    default_date = datetime.now() - timedelta(days=1)
    selected_date = st.date_input("Select Date", value=default_date)
    date_str = selected_date.strftime("%Y-%m-%d")

    with st.spinner("Loading completed games..."):
        raw = get_games_with_total_scored_points(only_null=False, date=date_str)

    if raw.empty:
        st.warning(f"No completed games found for {date_str}.")
        return

    if "prediction_datetime" in raw.columns:
        prediction_dt = pd.to_datetime(
            raw["prediction_datetime"], errors="coerce", utc=True
        )
    else:
        prediction_dt = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")

    if "prediction_date" in raw.columns:
        prediction_date_dt = pd.to_datetime(
            raw["prediction_date"], errors="coerce", utc=True
        )
    else:
        prediction_date_dt = pd.Series(
            pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]"
        )
    raw["prediction_datetime_utc"] = prediction_dt.fillna(prediction_date_dt)

    unique_times = sorted(
        raw["prediction_datetime_utc"].dropna().unique(), reverse=True
    )
    if not unique_times:
        st.warning(f"No prediction timestamps found for {date_str}.")
        return

    st.markdown("### ⏰ Select Prediction Time")
    st.caption("Choose which prediction time to analyze (most recent is default).")
    mapping: dict[str, pd.Timestamp] = {}
    options: list[str] = []
    for ts in unique_times:
        ts_madrid = pd.Timestamp(ts).tz_convert("Europe/Madrid")
        label = ts_madrid.strftime("%Y-%m-%d %H:%M:%S")
        options.append(label)
        mapping[label] = pd.Timestamp(ts)

    selected_label = st.selectbox(
        "Prediction Time:",
        options=options,
        index=0,
        help="Select which prediction snapshot to analyze.",
    )

    selected_cutoff = mapping[selected_label]
    games = build_game_level_predictions(
        raw,
        prediction_cutoff=selected_cutoff,
        model_version_filter=model_version_filter,
    )

    if games.empty:
        st.warning("No games available at the selected prediction time.")
        return

    resolved_mask = games["actual_side"].isin(["OVER", "UNDER"])
    n_resolved = int(resolved_mask.sum())

    metrics_cols = st.columns(5)
    with metrics_cols[0]:
        st.metric("🎮 Games Played", n_resolved)

    metric_labels = {
        "full_dataset": "🧱 Full Correct",
        "recent_games": "⚡ Recent Correct",
        "TabPFNRegressor": "🧠 TabPFN Correct",
    }
    for idx, model_type in enumerate(MODEL_ORDER, start=1):
        prefix = MODEL_PREFIXES[model_type]
        model_mask = resolved_mask & games[f"pick_{prefix}"].isin(["OVER", "UNDER"])
        correct = int(games.loc[model_mask, f"correct_{prefix}"].sum())
        total = int(model_mask.sum())
        with metrics_cols[idx]:
            st.metric(metric_labels[model_type], f"{correct}/{total}")

    # Add consensus metric
    with metrics_cols[4]:
        consensus_mask = resolved_mask & games["consensus_pick"].isin(["OVER", "UNDER"])
        consensus_correct = int(games.loc[consensus_mask, "consensus_correct"].sum())
        consensus_total = int(consensus_mask.sum())
        st.metric("🎯 Consensus Correct", f"{consensus_correct}/{consensus_total}")

    # Add a second row of metrics for consensus no tabpfn
    st.markdown("")
    metrics_cols2 = st.columns(5)
    with metrics_cols2[4]:
        consensus_no_tabpfn_mask = resolved_mask & games[
            "consensus_no_tabpfn_pick"
        ].isin(["OVER", "UNDER"])
        consensus_no_tabpfn_correct = int(
            games.loc[consensus_no_tabpfn_mask, "consensus_no_tabpfn_correct"].sum()
        )
        consensus_no_tabpfn_total = int(consensus_no_tabpfn_mask.sum())
        st.metric(
            "📊 Consensus (No TabPFN)",
            f"{consensus_no_tabpfn_correct}/{consensus_no_tabpfn_total}",
        )

    st.markdown("---")
    st.markdown(f"### 🏀 Games on {date_str}")
    st.markdown("")
    use_cards = st.toggle("Use Card View", value=True, key="past_cards")

    if use_cards:
        render_prediction_cards(games, include_actual=True)
    else:
        st.dataframe(
            build_past_display(games), width="stretch", hide_index=True, height=600
        )


def show_historical_performance(model_version_filter: str | None) -> None:
    st.markdown("## Historical Betting Performance")
    st.caption("Analyze model accuracy and prediction error over time.")
    st.markdown("")

    use_date_filter = st.checkbox("Filter by Date Range", value=False)
    start_date = None
    end_date = None

    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2026-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    with st.spinner("Loading historical predictions..."):
        raw = get_games_with_total_scored_points(
            only_null=False,
            start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
            end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
        )
        games = build_game_level_predictions(
            raw,
            model_version_filter=model_version_filter,
        )

    if games.empty:
        st.warning("No historical rows found for the selected range.")
        return

    resolved_mask = games["actual_side"].isin(["OVER", "UNDER"])
    total_games = len(games)
    resolved_games = int(resolved_mask.sum())
    analyzed_days = pd.to_datetime(games["game_date"], errors="coerce").nunique()

    consensus_mask = resolved_mask & games["consensus_pick"].isin(["OVER", "UNDER"])
    consensus_n = int(consensus_mask.sum())
    consensus_acc = (
        (
            games.loc[consensus_mask, "consensus_pick"]
            == games.loc[consensus_mask, "actual_side"]
        ).mean()
        if consensus_n
        else np.nan
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("Games", total_games)
    with top_cols[1]:
        st.metric("Resolved", resolved_games)
    with top_cols[2]:
        st.metric("Days", int(analyzed_days))
    with top_cols[3]:
        st.metric(
            "Weighted Consensus",
            f"{(consensus_acc * 100):.2f}%" if pd.notna(consensus_acc) else "N/A",
        )

    st.markdown("### 💰 Overall Model Statistics")
    st.markdown("")
    summary_df = summarize_model_performance(games)
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.markdown("")
    st.markdown("### 🎯 Accuracy by |Diff vs O/U Line|")
    st.caption(
        "Full/Recent: thresholds at >=0, >=1, >=2. TabPFN/Consensus/Consensus(No TabPFN): thresholds at >=0, >=0.5, >=1, >=1.5, >=2."
    )
    threshold_df = compute_threshold_accuracy_table(games)
    st.dataframe(
        threshold_df,
        width="stretch",
        hide_index=True,
        height=240,
    )

    st.markdown("---")
    st.markdown("### 📅 Daily Accuracy")
    daily = compute_daily_metrics(games)

    if daily.empty:
        st.warning("No daily metrics available.")
        return

    accuracy_pivot = (
        daily.pivot(index="game_date", columns="model_label", values="accuracy")
        .sort_index()
        .mul(100)
    )

    st.dataframe(
        accuracy_pivot.reset_index().rename(columns={"game_date": "Date"}).round(2),
        width="stretch",
        hide_index=True,
        height=350,
    )

    smooth_window = st.slider("Smoothing window (days)", 1, 14, 1)

    fig_acc, ax_acc = plt.subplots(figsize=(14, 6), dpi=140)
    ax_acc.set_facecolor("white")
    ax_acc.grid(True, alpha=0.2)

    for model_label in accuracy_pivot.columns:
        y = accuracy_pivot[model_label]
        if smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        ax_acc.plot(accuracy_pivot.index, y, linewidth=2.5, label=model_label)
        ax_acc.scatter(
            accuracy_pivot.index, accuracy_pivot[model_label], s=16, alpha=0.25
        )

    ax_acc.axhline(50, linestyle="--", alpha=0.55, linewidth=1.3)
    ax_acc.set_title("Daily Accuracy by Model", fontsize=16, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Date")
    ax_acc.set_ylim(0, 100)
    ax_acc.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax_acc.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax_acc.xaxis.get_major_locator())
    )
    ax_acc.legend(frameon=False)
    fig_acc.tight_layout()
    st.pyplot(fig_acc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📉 Daily Mean Absolute Error")

    mae_pivot = daily.pivot(
        index="game_date", columns="model_label", values="mae"
    ).sort_index()

    fig_mae, ax_mae = plt.subplots(figsize=(14, 6), dpi=140)
    ax_mae.set_facecolor("white")
    ax_mae.grid(True, alpha=0.2)

    for model_label in mae_pivot.columns:
        y = mae_pivot[model_label]
        if smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        ax_mae.plot(mae_pivot.index, y, linewidth=2.5, label=model_label)
        ax_mae.scatter(mae_pivot.index, mae_pivot[model_label], s=16, alpha=0.25)

    ax_mae.set_title("Daily MAE by Model", fontsize=16, fontweight="bold")
    ax_mae.set_ylabel("MAE (points)")
    ax_mae.set_xlabel("Date")
    ax_mae.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax_mae.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax_mae.xaxis.get_major_locator())
    )
    ax_mae.legend(frameon=False)
    fig_mae.tight_layout()
    st.pyplot(fig_mae, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="NBA Over/Under Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": None,
            "Report a bug": None,
            "About": "NBA Over/Under Predictor: upcoming predictions, past results, and historical performance.",
        },
    )

    set_runtime_env_from_secrets()
    inject_global_css()

    with st.sidebar:
        st.markdown("### NBA Predictor Menu")
        view_option = st.radio(
            label="Go to",
            options=[
                "Upcoming Predictions",
                "Past Games Results",
                "Historical Performance",
            ],
            index=0,
        )
        model_version_filter = st.text_input(
            "Model Version",
            value="1.0",
            help="Exact model_version to display. Leave blank to include all versions.",
        ).strip()
        st.markdown("---")

    render_header()

    if view_option == "Upcoming Predictions":
        show_upcoming_predictions(model_version_filter or None)
    elif view_option == "Past Games Results":
        show_past_games_results(model_version_filter or None)
    else:
        show_historical_performance(model_version_filter or None)


if __name__ == "__main__":
    main()
