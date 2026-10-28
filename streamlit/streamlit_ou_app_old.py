import html
import inspect
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import streamlit.components.v1 as components  # noqa: E402
from nba_ou.postgre_db.predictions.shap_utils import (  # noqa: E402
    ShapFeatureContribution,
    parse_serialized_shap_contributions,
)
from nba_ou.postgre_db.predictions.update.update_evaluation_predictions import (  # noqa: E402
    get_available_training_code_tags,
    get_games_with_total_scored_points,
)
from nba_ou.postgre_db.predictions.update.update_total_points_predictions import (  # noqa: E402
    update_total_points_predictions as run_update_finished_matches,
)
from nba_ou.utils.streamlit_utils import get_team_logo_url  # noqa: E402

import streamlit as st  # noqa: E402
from scripts.predict_nba_games import (  # noqa: E402
    predict_nba_games as run_nba_predictor,
)

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")


@dataclass(frozen=True)
class PredictionModelDefinition:
    key: str
    label: str
    column_prefix: str
    is_total_points: bool = True


@dataclass(frozen=True)
class ModelCatalog:
    definitions: tuple[PredictionModelDefinition, ...] = ()

    @property
    def order(self) -> list[str]:
        return [model.key for model in self.definitions]

    @property
    def labels(self) -> dict[str, str]:
        return {model.key: model.label for model in self.definitions}

    @property
    def prefixes(self) -> dict[str, str]:
        return {model.key: model.column_prefix for model in self.definitions}

    @property
    def total_points_models(self) -> list[str]:
        return [model.key for model in self.definitions if model.is_total_points]

    @property
    def diff_from_line_models(self) -> list[str]:
        return [model.key for model in self.definitions if not model.is_total_points]


FIVE_THREE_TP_KEY = "consensus_tp_5y3y"
FIVE_THREE_TP_LABEL = "5Y Base + 3Y Over"
_NUMBER_WORDS = {"3": "three", "5": "five"}
BASELINE_MODEL_KEY_TOKENS = (
    "random_line_error_baseline",
    "historical_average_line_error_baseline",
    # Legacy names in case rows were generated before the terminology cleanup.
    "random_over_under_baseline",
    "historical_average_total_points_baseline",
)


def _render_iframe_html(markup: str, *, height: int) -> None:
    iframe = getattr(st, "iframe", None)
    if callable(iframe):
        try:
            params = inspect.signature(iframe).parameters
        except (TypeError, ValueError):
            params = {}

        if "html" in params:
            iframe(html=markup, height=height)
            return
        if "srcdoc" in params:
            iframe(srcdoc=markup, height=height)
            return
        if "src_doc" in params:
            iframe(src_doc=markup, height=height)
            return

        iframe(markup, height=height)
        return

    components.html(markup, height=height)


def _empty_text_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")


def _normalized_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return _empty_text_series(df.index)

    series = df[column].astype("string").str.strip()
    return series.mask(series.eq(""))


def _slugify_model_key(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or None


def _is_baseline_model_identifier(value: object) -> bool:
    slug = _slugify_model_key(value)
    if not slug:
        return False
    return any(token in slug for token in BASELINE_MODEL_KEY_TOKENS)


def _model_catalog_without_baselines(catalog: ModelCatalog) -> ModelCatalog:
    return ModelCatalog(
        definitions=tuple(
            model
            for model in catalog.definitions
            if not _is_baseline_model_identifier(model.key)
        )
    )


def exclude_baseline_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    baseline_mask = pd.Series(False, index=df.index)
    for col in ("_model_key", "model_type", "model_name", "prediction_source"):
        if col in df.columns:
            baseline_mask = baseline_mask | df[col].apply(_is_baseline_model_identifier)

    return df.loc[~baseline_mask].copy()


def _first_non_empty_text(*values: object) -> str | None:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _pretty_model_label(value: object) -> str:
    text = _first_non_empty_text(value)
    if text is None:
        return "Unknown Model"

    slug = _slugify_model_key(text) or ""
    if "tabpfn" in slug and "line_error" in slug:
        return "TabPFN Line Error"
    if "tabpfn" in slug:
        return "TabPFN"

    if "_" in text:
        return text.replace("_", " ").strip().title()
    return text


def _is_total_points_model(*values: object) -> bool:
    text = " ".join(str(value).lower() for value in values if pd.notna(value))
    if not text:
        return True
    if "line_error" in text or "diff_from_line" in text:
        return False
    if "total_points" in text or "tabpfn" in text:
        return True
    return True


def _build_model_metadata(
    *,
    model_type: object,
    model_name: object,
    prediction_source: object,
    prediction_value_type: object,
) -> tuple[str | None, str, bool]:
    canonical_text = _first_non_empty_text(model_type, model_name, prediction_source)
    key = _slugify_model_key(canonical_text)

    prediction_value_slug = _slugify_model_key(prediction_value_type)
    if key is None:
        key = prediction_value_slug
    elif prediction_value_slug and prediction_value_slug not in key:
        key = f"{key}_{prediction_value_slug}"

    label = _pretty_model_label(canonical_text)
    if prediction_value_slug == "diff_from_line" and "line error" not in label.lower():
        label = f"{label} Line Error"

    is_total_points = (
        str(prediction_value_type).strip().upper() == "TOTAL_POINTS"
        if pd.notna(prediction_value_type)
        else _is_total_points_model(
            prediction_value_type, model_type, model_name, prediction_source
        )
    )

    return key, label, is_total_points


def extract_model_catalog(df: pd.DataFrame) -> ModelCatalog:
    if df.empty:
        return ModelCatalog()

    work = df.copy()
    model_type = _normalized_text_series(work, "model_type")
    model_name = _normalized_text_series(work, "model_name")
    prediction_source = _normalized_text_series(work, "prediction_source")
    prediction_value_type = _normalized_text_series(work, "prediction_value_type")

    metadata = [
        _build_model_metadata(
            model_type=mt,
            model_name=mn,
            prediction_source=ps,
            prediction_value_type=pvt,
        )
        for mt, mn, ps, pvt in zip(
            model_type,
            model_name,
            prediction_source,
            prediction_value_type,
            strict=False,
        )
    ]
    work["_model_key"] = [item[0] for item in metadata]
    work["_model_label"] = [item[1] for item in metadata]
    work["_is_total_points"] = [item[2] for item in metadata]

    if "prediction_datetime" in work.columns:
        prediction_dt = pd.to_datetime(
            work["prediction_datetime"], errors="coerce", utc=True
        )
    else:
        prediction_dt = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")

    if "prediction_date" in work.columns:
        prediction_date = pd.to_datetime(
            work["prediction_date"], errors="coerce", utc=True
        )
    else:
        prediction_date = pd.Series(
            pd.NaT, index=work.index, dtype="datetime64[ns, UTC]"
        )
    work["_model_sort_ts"] = prediction_dt.fillna(prediction_date)
    work = work[work["_model_key"].notna()].copy()

    if work.empty:
        return ModelCatalog()

    latest_per_model = (
        work.sort_values(["_model_sort_ts", "_model_label"], na_position="last")
        .groupby("_model_key", as_index=False)
        .tail(1)
        .copy()
    )
    latest_per_model["_sort_group"] = latest_per_model["_model_label"].str.contains(
        "tabpfn", case=False, na=False
    )
    latest_per_model = latest_per_model.sort_values(
        ["_sort_group", "_model_label"],
        kind="stable",
    )

    definitions = tuple(
        PredictionModelDefinition(
            key=str(row["_model_key"]),
            label=str(row["_model_label"]),
            column_prefix=str(row["_model_key"]),
            is_total_points=bool(row["_is_total_points"]),
        )
        for row in latest_per_model.to_dict("records")
    )
    return ModelCatalog(definitions=definitions)


def get_model_catalog(df: pd.DataFrame | None) -> ModelCatalog:
    if df is None:
        return ModelCatalog()

    catalog = df.attrs.get("model_catalog")
    if isinstance(catalog, ModelCatalog):
        return catalog

    return extract_model_catalog(df)


def _ordered_model_types_by_prediction_target(catalog: ModelCatalog) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for model_type in [*catalog.total_points_models, *catalog.diff_from_line_models]:
        if model_type not in seen:
            ordered.append(model_type)
            seen.add(model_type)

    return ordered


def _prediction_target_label(catalog: ModelCatalog, model_type: str) -> str:
    if model_type in catalog.diff_from_line_models:
        return "Diff From Line"
    return "Total Points"


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


@st.cache_data(ttl=300, show_spinner=False)
def load_available_training_code_tags() -> list[str]:
    return get_available_training_code_tags()


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg-main: #fcfaf7;
            --bg-elevated: #fff7ef;
            --bg-panel: #ffffff;
            --bg-panel-2: #f8f1ff;
            --text-main: #1b1422;
            --text-muted: #5f4f73;
            --text-soft: #7c6b91;
            --border-main: rgba(155, 107, 255, 0.18);
            --border-strong: rgba(255, 151, 43, 0.30);
            --accent-orange: #ff9a2f;
            --accent-orange-2: #ff7a18;
            --accent-purple: #9b6bff;
            --accent-purple-2: #6f42ff;
            --success-glow: #ffd08a;
          }

          html, body, [data-testid="stAppViewContainer"], .stApp {
            background:
              radial-gradient(circle at top left, rgba(255, 122, 24, 0.10), transparent 24%),
              radial-gradient(circle at top right, rgba(111, 66, 255, 0.12), transparent 28%),
              linear-gradient(180deg, #fffdfa 0%, #fcfaf7 100%);
            color: var(--text-main);
          }

          /* Page container */
          .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.0rem;
            max-width: 1400px;
          }

          .main .block-container,
          .main .block-container p,
          .main .block-container li,
          .main .block-container label,
          .main .block-container div,
          .main .block-container span {
            color: var(--text-main);
          }

          /* Sidebar polish */
          section[data-testid="stSidebar"] {
            border-right: 1px solid var(--border-main);
            background:
              linear-gradient(180deg, rgba(250, 243, 255, 0.98), rgba(255, 248, 239, 0.98));
          }
          section[data-testid="stSidebar"] > div {
            background: transparent;
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
            color: var(--text-soft) !important;
          }

          section[data-testid="stSidebar"] .stSelectbox > div > div,
          section[data-testid="stSidebar"] .stDateInput > div > div,
          section[data-testid="stSidebar"] .stRadio > div,
          section[data-testid="stSidebar"] button {
            background: rgba(255, 255, 255, 0.94) !important;
            border-color: var(--border-main) !important;
            color: var(--text-main) !important;
          }

          section[data-testid="stSidebar"] .stButton button:hover {
            border-color: var(--border-strong) !important;
            box-shadow: 0 0 0 1px rgba(255, 154, 47, 0.22);
          }

          /* Typography */
          h1, h2, h3 {
            letter-spacing: -0.02em;
            color: var(--text-main) !important;
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
            color: var(--text-soft) !important;
          }
          .stMetric [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: var(--text-main) !important;
          }
          [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(252, 246, 255, 0.98));
            border: 1px solid var(--border-main);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            box-shadow: 0 14px 30px rgba(76, 45, 120, 0.08);
          }

          /* DataFrame readability */
          div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-main);
            border-radius: 16px;
            overflow: hidden;
          }
          div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
            background: rgba(255, 255, 255, 0.98);
          }
          div[data-testid="stDataFrame"] div[role="gridcell"] {
            padding: 0.65rem !important;
            background: rgba(255, 255, 255, 0.99) !important;
            color: var(--text-main) !important;
          }
          div[data-testid="stDataFrame"] div[role="columnheader"] {
            font-weight: 750 !important;
            padding: 0.85rem !important;
            background: linear-gradient(90deg, rgba(111, 66, 255, 0.24), rgba(255, 122, 24, 0.18)) !important;
            color: var(--text-main) !important;
          }

          /* "Hero" header container */
          .app-hero {
            border: 1px solid var(--border-main);
            border-radius: 16px;
            padding: 18px 18px;
            background:
              radial-gradient(circle at top right, rgba(255, 154, 47, 0.16), transparent 28%),
              linear-gradient(135deg,
                rgba(255, 255, 255, 0.98) 0%,
                rgba(247, 238, 255, 0.98) 52%,
                rgba(255, 243, 228, 0.98) 100%);
            margin-bottom: 16px;
            color: var(--text-main);
            box-shadow: 0 16px 36px rgba(93, 64, 145, 0.10);
          }
          .app-subtitle {
            font-size: 1.05rem;
            color: var(--text-muted);
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
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(255, 154, 47, 0.26);
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-main);
          }

          /* Reduce visual noise on separators */
          hr {
            margin: 1.0rem 0;
            opacity: 0.22;
            border-color: var(--border-main);
          }

          .stExpander,
          [data-testid="stExpander"] {
            border: 1px solid var(--border-main) !important;
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.92) !important;
          }

          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div {
            background: rgba(255, 255, 255, 0.96) !important;
            border-color: var(--border-main) !important;
            color: var(--text-main) !important;
          }

          .stButton button,
          .stDownloadButton button {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-orange-2)) !important;
            color: #fff7f0 !important;
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 24px rgba(111, 66, 255, 0.28);
          }

          .stButton button:hover,
          .stDownloadButton button:hover {
            filter: brightness(1.05);
            box-shadow: 0 14px 32px rgba(255, 122, 24, 0.22);
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
          }

          .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--border-main);
            border-radius: 12px 12px 0 0;
            color: var(--text-muted);
          }

          .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(111, 66, 255, 0.28), rgba(255, 122, 24, 0.18)) !important;
            color: var(--text-main) !important;
            border-color: var(--border-strong) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(catalog: ModelCatalog | None = None) -> None:
    model_labels = []
    if catalog is not None:
        model_labels = [
            catalog.labels[model_type]
            for model_type in _ordered_model_types_by_prediction_target(catalog)
        ]
        if (
            _find_last_n_total_points_model(catalog, 5) is not None
            and _find_last_n_total_points_model(catalog, 3) is not None
        ):
            model_labels.append(FIVE_THREE_TP_LABEL)

    chip_html = "".join(
        f'<span class="chip">📊 {html.escape(model_label)}</span>'
        for model_label in model_labels
    )
    if chip_html:
        chip_html += '<span class="chip">🕐 Madrid (CEST)</span>'
    else:
        chip_html = '<span class="chip">🕐 Madrid (CEST)</span>'

    st.markdown(
        f"""
        <div class="app-hero">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:12px;">
            <div style="flex: 1;">
              <div style="font-size: 0.95rem; font-weight: 700; color:#ffb36b;">NBA analytics</div>
              <div style="margin-top: 2px;">
                <span style="font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em; color:#1b1422;">
                  Over/Under Predictor
                </span>
              </div>
              <div class="app-subtitle">
                Predictions, results, and historical performance in one place.
              </div>
              <div class="chip-row">
                {chip_html}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _parse_prediction_timestamp_value(
    value: object, *, treat_naive_as_madrid_local: bool = False
) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT

    ts = pd.Timestamp(parsed)
    if ts.tzinfo is None:
        if treat_naive_as_madrid_local:
            return ts.tz_localize("Europe/Madrid").tz_convert("UTC")
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_prediction_timestamp_series(
    series: pd.Series, *, treat_naive_as_madrid_local: bool = False
) -> pd.Series:
    parsed = series.apply(
        _parse_prediction_timestamp_value,
        treat_naive_as_madrid_local=treat_naive_as_madrid_local,
    )
    return pd.to_datetime(parsed, errors="coerce", utc=True)


def pick_from_diff(diff: pd.Series) -> pd.Series:
    out = pd.Series(index=diff.index, dtype="object")
    out.loc[diff > 0] = "OVER"
    out.loc[diff < 0] = "UNDER"
    out.loc[diff == 0] = "PUSH"
    return out


def _numeric_scalar(value: object) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _contains_token_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    seq_len = len(sequence)
    if seq_len == 0 or seq_len > len(tokens):
        return False

    return any(
        tuple(tokens[idx : idx + seq_len]) == sequence
        for idx in range(len(tokens) - seq_len + 1)
    )


def _find_last_n_total_points_model(
    catalog: ModelCatalog, n_seasons: int
) -> PredictionModelDefinition | None:
    candidates: list[tuple[int, int, PredictionModelDefinition]] = []
    season_text = str(n_seasons)
    season_word = _NUMBER_WORDS.get(season_text)
    season_tokens = [season_text]
    if season_word:
        season_tokens.append(season_word)

    for position, model in enumerate(catalog.definitions):
        if not model.is_total_points:
            continue

        text = f"{model.key} {model.label} {model.column_prefix}".lower()
        slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        tokens = [token for token in slug.split("_") if token]
        score = 0

        for season_token in season_tokens:
            for period_token in ("seasons", "season", "years", "year"):
                if _contains_token_sequence(
                    tokens,
                    ("total", "points", "last", season_token, period_token),
                ):
                    score = max(score, 40)
                if _contains_token_sequence(
                    tokens, ("last", season_token, period_token)
                ):
                    score = max(score, 30)
                if _contains_token_sequence(tokens, (season_token, period_token)):
                    score = max(score, 20)

        if score:
            candidates.append((score, -position, model))

    if not candidates:
        return None

    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _latest_model_row(group: pd.DataFrame, model_key: str) -> pd.Series | None:
    rows = group[group["_model_key"] == model_key]
    if rows.empty:
        return None
    if "prediction_datetime_utc" in rows.columns:
        rows = rows.sort_values("prediction_datetime_utc")
    return rows.iloc[-1]


def _five_three_tp_pick_line_diff_from_group(
    group: pd.DataFrame, catalog: ModelCatalog
) -> tuple[object, float]:
    five_model = _find_last_n_total_points_model(catalog, 5)
    if five_model is None:
        return np.nan, np.nan

    five_row = _latest_model_row(group, five_model.key)
    three_model = _find_last_n_total_points_model(catalog, 3)
    three_row = (
        _latest_model_row(group, three_model.key) if three_model is not None else None
    )

    if three_row is not None and normalize_pick(three_row.get("pred_pick")) == "OVER":
        return "OVER", _numeric_scalar(three_row.get("pred_line_error"))

    if five_row is None:
        return np.nan, np.nan

    five_pick = normalize_pick(five_row.get("pred_pick"))
    if five_pick in {"OVER", "UNDER", "PUSH"}:
        return five_pick, _numeric_scalar(five_row.get("pred_line_error"))

    return np.nan, np.nan


def build_game_level_predictions(
    df: pd.DataFrame,
    prediction_cutoff: pd.Timestamp | None = None,
    training_code_tag_filter: str | None = "1.0",
    treat_prediction_timestamps_as_madrid_local: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "model_type" not in work.columns:
        work["model_type"] = np.nan
    if "model_name" not in work.columns:
        work["model_name"] = np.nan
    if "prediction_source" not in work.columns:
        work["prediction_source"] = np.nan
    if "prediction_value_type" not in work.columns:
        work["prediction_value_type"] = np.nan

    model_type_source = _normalized_text_series(work, "model_type")
    model_name_source = _normalized_text_series(work, "model_name")
    prediction_source = _normalized_text_series(work, "prediction_source")
    prediction_value_type = _normalized_text_series(work, "prediction_value_type")

    metadata = [
        _build_model_metadata(
            model_type=mt,
            model_name=mn,
            prediction_source=ps,
            prediction_value_type=pvt,
        )
        for mt, mn, ps, pvt in zip(
            model_type_source,
            model_name_source,
            prediction_source,
            prediction_value_type,
            strict=False,
        )
    ]
    work["_model_key"] = [item[0] for item in metadata]
    work["_model_label"] = [item[1] for item in metadata]
    work["_is_total_points"] = [item[2] for item in metadata]
    work = work[work["_model_key"].notna()].copy()

    if "training_code_tag" not in work.columns:
        work["training_code_tag"] = np.nan

    if training_code_tag_filter:
        normalized_tag = str(training_code_tag_filter).strip()
        work = work[
            work["training_code_tag"].fillna("").astype(str).str.strip()
            == normalized_tag
        ].copy()

    if work.empty:
        return pd.DataFrame()

    if "prediction_datetime" in work.columns:
        if treat_prediction_timestamps_as_madrid_local:
            pred_dt = parse_prediction_timestamp_series(
                work["prediction_datetime"], treat_naive_as_madrid_local=True
            )
        else:
            pred_dt = pd.to_datetime(
                work["prediction_datetime"], errors="coerce", utc=True
            )
    else:
        pred_dt = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")

    if "prediction_date" in work.columns:
        if treat_prediction_timestamps_as_madrid_local:
            pred_date_dt = parse_prediction_timestamp_series(
                work["prediction_date"], treat_naive_as_madrid_local=True
            )
        else:
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

    catalog = extract_model_catalog(work)
    model_order = _ordered_model_types_by_prediction_target(catalog)
    aggregation_catalog = _model_catalog_without_baselines(catalog)
    aggregation_model_order = _ordered_model_types_by_prediction_target(
        aggregation_catalog
    )
    aggregation_model_keys = set(aggregation_model_order)
    model_prefixes = catalog.prefixes
    if not model_order:
        return pd.DataFrame()

    if "game_time" in work.columns:
        work["game_time_utc"] = pd.to_datetime(
            work["game_time"], errors="coerce", utc=True
        )
    else:
        work["game_time_utc"] = pd.NaT

    line = pd.to_numeric(work.get("total_over_under_line"), errors="coerce")
    line_at_prediction = pd.to_numeric(
        work.get("total_bet365_line_at_prediction"), errors="coerce"
    )
    work["line_for_calc"] = line.fillna(line_at_prediction)
    work["pred_total_points"] = pd.to_numeric(
        work.get("pred_total_points"), errors="coerce"
    )
    work["pred_line_error"] = pd.to_numeric(
        work.get("pred_line_error"), errors="coerce"
    )

    # New schema allows either target value. Backfill missing side for downstream views.
    work["pred_total_points"] = work["pred_total_points"].where(
        work["pred_total_points"].notna(),
        work["line_for_calc"] + work["pred_line_error"],
    )
    work["pred_line_error"] = work["pred_line_error"].where(
        work["pred_line_error"].notna(),
        work["pred_total_points"] - work["line_for_calc"],
    )

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
        "na_columns_count",
        "na_columns_names",
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

    for model_type in model_order:
        prefix = model_prefixes[model_type]
        per_model = work[work["_model_key"] == model_type].copy()
        if per_model.empty:
            continue

        # Prefer rows that carry the direct total-points target when duplicates exist.
        if "prediction_value_type" in per_model.columns:
            per_model["_prediction_priority"] = (
                per_model["prediction_value_type"].astype(str).str.upper()
                == "TOTAL_POINTS"
            ).astype(int)
        else:
            per_model["_prediction_priority"] = 0

        model_cols = [
            "game_id",
            "pred_pick",
            "pred_total_points",
            "pred_line_error",
            "prediction_datetime_utc",
            "shap_base_value",
            "shap_top_positive_features",
            "shap_top_negative_features",
            "_prediction_priority",
        ]
        available_model_cols = [col for col in model_cols if col in per_model.columns]

        per_model = (
            per_model.sort_values(["prediction_datetime_utc", "_prediction_priority"])
            .groupby("game_id", as_index=False)
            .tail(1)[available_model_cols]
        )

        rename_map = {
            "pred_pick": f"pick_{prefix}",
            "pred_total_points": f"pred_total_{prefix}",
            "pred_line_error": f"line_error_{prefix}",
            "prediction_datetime_utc": f"pred_dt_{prefix}",
            "shap_base_value": f"shap_base_{prefix}",
            "shap_top_positive_features": f"shap_pos_{prefix}",
            "shap_top_negative_features": f"shap_neg_{prefix}",
        }
        per_model = per_model.rename(columns=rename_map)

        # Drop the priority column before merging to avoid conflicts
        if "_prediction_priority" in per_model.columns:
            per_model = per_model.drop(columns=["_prediction_priority"])

        base = base.merge(per_model, on="game_id", how="left")

    line = pd.to_numeric(base.get("total_over_under_line"), errors="coerce")
    actual_total = pd.to_numeric(base.get("total_scored_points"), errors="coerce")

    # NumPy 2 raises DTypePromotionError when np.where mixes strings with np.nan.
    actual_side = pd.Series(index=base.index, dtype="object")
    actual_side.loc[actual_total > line] = "OVER"
    actual_side.loc[actual_total < line] = "UNDER"
    actual_side.loc[actual_total == line] = "PUSH"
    base["actual_side"] = actual_side

    aggregation_pick_cols: list[str] = []
    aggregation_total_cols: list[str] = []
    aggregation_pick_cols_tp: list[str] = []
    aggregation_pick_cols_le: list[str] = []
    for model_type in model_order:
        prefix = model_prefixes[model_type]
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

        if model_type in aggregation_model_keys:
            aggregation_pick_cols.append(pick_col)
            aggregation_total_cols.append(pred_total_col)
            if model_type in aggregation_catalog.total_points_models:
                aggregation_pick_cols_tp.append(pick_col)
            if model_type in aggregation_catalog.diff_from_line_models:
                aggregation_pick_cols_le.append(pick_col)

    # Simple consensus: average of all available line diffs across models
    aggregation_diff_cols = [
        f"line_diff_{model_prefixes[m]}" for m in aggregation_model_order
    ]
    if aggregation_diff_cols:
        base["consensus_line_diff"] = base[aggregation_diff_cols].mean(
            axis=1, skipna=True
        )
    else:
        base["consensus_line_diff"] = np.nan
    base["consensus_pred_total"] = line + base["consensus_line_diff"]
    base["consensus_pick"] = pick_from_diff(base["consensus_line_diff"])
    base["consensus_error"] = base["consensus_pred_total"] - actual_total
    base["consensus_correct"] = (base["consensus_pick"] == base["actual_side"]) & base[
        "actual_side"
    ].isin(["OVER", "UNDER"])

    # Consensus without TabPFN (average of non-TabPFN model diffs)
    no_tabpfn_diff_cols = [
        f"line_diff_{model_prefixes[m]}"
        for m in aggregation_model_order
        if "tabpfn" not in m.lower()
    ]
    if no_tabpfn_diff_cols:
        base["consensus_no_tabpfn_line_diff"] = base[no_tabpfn_diff_cols].mean(
            axis=1, skipna=True
        )
    else:
        base["consensus_no_tabpfn_line_diff"] = np.nan
    base["consensus_no_tabpfn_pred_total"] = (
        line + base["consensus_no_tabpfn_line_diff"]
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

    # Majority vote consensus: direction decided by raw vote count across all models
    if aggregation_pick_cols:
        _vote_matrix = base[aggregation_pick_cols]
        base["consensus_vote_n_over"] = (_vote_matrix == "OVER").sum(axis=1)
        base["consensus_vote_n_under"] = (_vote_matrix == "UNDER").sum(axis=1)
    else:
        base["consensus_vote_n_over"] = 0
        base["consensus_vote_n_under"] = 0
    _vote_pick = pd.Series(index=base.index, dtype="object")
    _vote_pick.loc[base["consensus_vote_n_over"] > base["consensus_vote_n_under"]] = (
        "OVER"
    )
    _vote_pick.loc[base["consensus_vote_n_under"] > base["consensus_vote_n_over"]] = (
        "UNDER"
    )
    base["consensus_vote_pick"] = _vote_pick
    base["consensus_vote_correct"] = (
        base["consensus_vote_pick"] == base["actual_side"]
    ) & base["actual_side"].isin(["OVER", "UNDER"])

    # Majority vote - Total Points models only
    if aggregation_pick_cols_tp:
        _vote_matrix_tp = base[aggregation_pick_cols_tp]
        base["consensus_vote_tp_n_over"] = (_vote_matrix_tp == "OVER").sum(axis=1)
        base["consensus_vote_tp_n_under"] = (_vote_matrix_tp == "UNDER").sum(axis=1)
        _vote_pick_tp = pd.Series(index=base.index, dtype="object")
        _vote_pick_tp.loc[
            base["consensus_vote_tp_n_over"] > base["consensus_vote_tp_n_under"]
        ] = "OVER"
        _vote_pick_tp.loc[
            base["consensus_vote_tp_n_under"] > base["consensus_vote_tp_n_over"]
        ] = "UNDER"
        base["consensus_vote_tp_pick"] = _vote_pick_tp
        base["consensus_vote_tp_correct"] = (
            base["consensus_vote_tp_pick"] == base["actual_side"]
        ) & base["actual_side"].isin(["OVER", "UNDER"])
    else:
        base["consensus_vote_tp_n_over"] = 0
        base["consensus_vote_tp_n_under"] = 0
        base["consensus_vote_tp_pick"] = np.nan
        base["consensus_vote_tp_correct"] = False

    # Majority vote - Line Error models only
    if aggregation_pick_cols_le:
        _vote_matrix_le = base[aggregation_pick_cols_le]
        base["consensus_vote_le_n_over"] = (_vote_matrix_le == "OVER").sum(axis=1)
        base["consensus_vote_le_n_under"] = (_vote_matrix_le == "UNDER").sum(axis=1)
        _vote_pick_le = pd.Series(index=base.index, dtype="object")
        _vote_pick_le.loc[
            base["consensus_vote_le_n_over"] > base["consensus_vote_le_n_under"]
        ] = "OVER"
        _vote_pick_le.loc[
            base["consensus_vote_le_n_under"] > base["consensus_vote_le_n_over"]
        ] = "UNDER"
        base["consensus_vote_le_pick"] = _vote_pick_le
        base["consensus_vote_le_correct"] = (
            base["consensus_vote_le_pick"] == base["actual_side"]
        ) & base["actual_side"].isin(["OVER", "UNDER"])
    else:
        base["consensus_vote_le_n_over"] = 0
        base["consensus_vote_le_n_under"] = 0
        base["consensus_vote_le_pick"] = np.nan
        base["consensus_vote_le_correct"] = False

    # Bold Contrarian: prediction from the model with the highest absolute line diff
    if aggregation_diff_cols:
        abs_diffs = base[aggregation_diff_cols].abs()
        # idxmax gives the column name (per-row) with the largest absolute diff
        _bc_col_idx = abs_diffs.idxmax(axis=1)
        base["consensus_bold_contrarian_line_diff"] = pd.Series(
            [
                base.loc[idx, col] if pd.notna(col) else np.nan
                for idx, col in zip(base.index, _bc_col_idx, strict=False)
            ],
            index=base.index,
        )
    else:
        base["consensus_bold_contrarian_line_diff"] = np.nan
    base["consensus_bold_contrarian_pred_total"] = (
        line + base["consensus_bold_contrarian_line_diff"]
    )
    base["consensus_bold_contrarian_pick"] = pick_from_diff(
        base["consensus_bold_contrarian_line_diff"]
    )
    base["consensus_bold_contrarian_error"] = (
        base["consensus_bold_contrarian_pred_total"] - actual_total
    )
    base["consensus_bold_contrarian_correct"] = (
        base["consensus_bold_contrarian_pick"] == base["actual_side"]
    ) & base["actual_side"].isin(["OVER", "UNDER"])

    # 5Y/3Y total-points rule: follow 5Y unless the 3Y model says OVER.
    five_year_model = _find_last_n_total_points_model(aggregation_catalog, 5)
    three_year_model = _find_last_n_total_points_model(aggregation_catalog, 3)
    if five_year_model is not None:
        five_prefix = model_prefixes[five_year_model.key]
        five_pick_col = f"pick_{five_prefix}"
        five_diff_col = f"line_diff_{five_prefix}"

        three_pick = pd.Series(np.nan, index=base.index, dtype="object")
        three_line_diff = pd.Series(np.nan, index=base.index, dtype="float64")
        if three_year_model is not None:
            three_prefix = model_prefixes[three_year_model.key]
            three_pick = base.get(
                f"pick_{three_prefix}",
                pd.Series(np.nan, index=base.index, dtype="object"),
            )
            three_line_diff = pd.to_numeric(
                base.get(
                    f"line_diff_{three_prefix}",
                    pd.Series(np.nan, index=base.index, dtype="float64"),
                ),
                errors="coerce",
            )

        use_three_over = three_pick.eq("OVER")
        base["consensus_tp_5y3y_pick"] = base[five_pick_col].where(
            ~use_three_over, "OVER"
        )
        base["consensus_tp_5y3y_line_diff"] = pd.to_numeric(
            base[five_diff_col], errors="coerce"
        ).where(~use_three_over, three_line_diff)
    else:
        base["consensus_tp_5y3y_pick"] = np.nan
        base["consensus_tp_5y3y_line_diff"] = np.nan

    base["consensus_tp_5y3y_pred_total"] = line + base["consensus_tp_5y3y_line_diff"]
    base["consensus_tp_5y3y_error"] = (
        base["consensus_tp_5y3y_pred_total"] - actual_total
    )
    base["consensus_tp_5y3y_correct"] = (
        base["consensus_tp_5y3y_pick"] == base["actual_side"]
    ) & base["actual_side"].isin(["OVER", "UNDER"])

    if aggregation_total_cols:
        base["all_models_available"] = base[aggregation_total_cols].notna().all(axis=1)
        base["all_models_agree"] = (
            base[aggregation_pick_cols].nunique(axis=1, dropna=True).eq(1)
            & base["all_models_available"]
        )
    else:
        base["all_models_available"] = False
        base["all_models_agree"] = False

    if "game_time_utc" in base.columns:
        base = base.sort_values("game_time_utc")

    base = base.reset_index(drop=True)
    base.attrs["model_catalog"] = catalog
    return base


def build_upcoming_display(
    df: pd.DataFrame, show_pred_times: bool = False
) -> pd.DataFrame:
    catalog = get_model_catalog(df)
    model_order = _ordered_model_types_by_prediction_target(catalog)
    model_labels = catalog.labels
    model_prefixes = catalog.prefixes

    display = pd.DataFrame()
    display["Matchup"] = df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    display["Game Time (Madrid)"] = format_madrid_datetime(
        df["game_time_utc"], "%Y-%m-%d %H:%M"
    )
    display["O/U Line"] = pd.to_numeric(
        df["total_over_under_line"], errors="coerce"
    ).round(1)

    for model_type in model_order:
        prefix = model_prefixes[model_type]
        label = model_labels[model_type]
        display[f"{label} Total Points"] = pd.to_numeric(
            df[f"pred_total_{prefix}"], errors="coerce"
        ).round(1)
        display[f"{label} Line Error"] = pd.to_numeric(
            df[f"line_diff_{prefix}"], errors="coerce"
        ).round(2)
        display[f"{label} Pick"] = df[f"pick_{prefix}"]

        # Optionally show prediction time for each model
        if show_pred_times and f"pred_dt_{prefix}" in df.columns:
            display[f"{label} Time"] = format_madrid_datetime(
                df[f"pred_dt_{prefix}"], "%m-%d %H:%M"
            )

    display["Consensus Total Points"] = pd.to_numeric(
        df["consensus_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus Line Error"] = pd.to_numeric(
        df["consensus_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus Pick"] = df["consensus_pick"]

    display["Consensus (No TabPFN) Total Points"] = pd.to_numeric(
        df["consensus_no_tabpfn_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus (No TabPFN) Line Error"] = pd.to_numeric(
        df["consensus_no_tabpfn_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus (No TabPFN) Pick"] = df["consensus_no_tabpfn_pick"]

    display["Vote Pick"] = df["consensus_vote_pick"]
    display["Over Votes"] = pd.to_numeric(
        df["consensus_vote_n_over"], errors="coerce"
    ).astype("Int64")
    display["Under Votes"] = pd.to_numeric(
        df["consensus_vote_n_under"], errors="coerce"
    ).astype("Int64")

    display["Vote Pick (TP)"] = df["consensus_vote_tp_pick"]
    display["Over Votes (TP)"] = pd.to_numeric(
        df["consensus_vote_tp_n_over"], errors="coerce"
    ).astype("Int64")
    display["Under Votes (TP)"] = pd.to_numeric(
        df["consensus_vote_tp_n_under"], errors="coerce"
    ).astype("Int64")

    display[f"{FIVE_THREE_TP_LABEL} Total Points"] = pd.to_numeric(
        df["consensus_tp_5y3y_pred_total"], errors="coerce"
    ).round(1)
    display[f"{FIVE_THREE_TP_LABEL} Line Error"] = pd.to_numeric(
        df["consensus_tp_5y3y_line_diff"], errors="coerce"
    ).round(2)
    display[f"{FIVE_THREE_TP_LABEL} Pick"] = df["consensus_tp_5y3y_pick"]

    display["Vote Pick (LE)"] = df["consensus_vote_le_pick"]
    display["Over Votes (LE)"] = pd.to_numeric(
        df["consensus_vote_le_n_over"], errors="coerce"
    ).astype("Int64")
    display["Under Votes (LE)"] = pd.to_numeric(
        df["consensus_vote_le_n_under"], errors="coerce"
    ).astype("Int64")

    display["Bold Contrarian Total Points"] = pd.to_numeric(
        df["consensus_bold_contrarian_pred_total"], errors="coerce"
    ).round(1)
    display["Bold Contrarian Line Error"] = pd.to_numeric(
        df["consensus_bold_contrarian_line_diff"], errors="coerce"
    ).round(2)
    display["Bold Contrarian Pick"] = df["consensus_bold_contrarian_pick"]

    if "time_to_match_minutes" in df.columns:
        display["Time to Game (min)"] = (
            pd.to_numeric(df["time_to_match_minutes"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return display


def build_past_display(df: pd.DataFrame) -> pd.DataFrame:
    catalog = get_model_catalog(df)
    model_order = _ordered_model_types_by_prediction_target(catalog)
    model_labels = catalog.labels
    model_prefixes = catalog.prefixes

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

    for model_type in model_order:
        prefix = model_prefixes[model_type]
        label = model_labels[model_type]
        display[f"{label} Total Points"] = pd.to_numeric(
            df[f"pred_total_{prefix}"], errors="coerce"
        ).round(1)
        display[f"{label} Line Error"] = pd.to_numeric(
            df[f"line_diff_{prefix}"], errors="coerce"
        ).round(2)
        display[f"{label} Pick"] = df[f"pick_{prefix}"]
        display[f"{label} Correct"] = df[f"correct_{prefix}"].map(
            {True: "✅", False: "❌"}
        )

    display["Consensus Total Points"] = pd.to_numeric(
        df["consensus_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus Line Error"] = pd.to_numeric(
        df["consensus_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus Pick"] = df["consensus_pick"]
    display["Consensus Correct"] = df["consensus_correct"].map(
        {True: "✅", False: "❌"}
    )

    display["Consensus (No TabPFN) Total Points"] = pd.to_numeric(
        df["consensus_no_tabpfn_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus (No TabPFN) Line Error"] = pd.to_numeric(
        df["consensus_no_tabpfn_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus (No TabPFN) Pick"] = df["consensus_no_tabpfn_pick"]
    display["Consensus (No TabPFN) Correct"] = df["consensus_no_tabpfn_correct"].map(
        {True: "✅", False: "❌"}
    )

    display["Vote Pick"] = df["consensus_vote_pick"]
    display["Vote Correct"] = df["consensus_vote_correct"].map(
        {True: "✅", False: "❌"}
    )
    display["Over Votes"] = pd.to_numeric(
        df["consensus_vote_n_over"], errors="coerce"
    ).astype("Int64")
    display["Under Votes"] = pd.to_numeric(
        df["consensus_vote_n_under"], errors="coerce"
    ).astype("Int64")

    display["Vote Pick (TP)"] = df["consensus_vote_tp_pick"]
    display["Vote Correct (TP)"] = df["consensus_vote_tp_correct"].map(
        {True: "✅", False: "❌"}
    )
    display["Over Votes (TP)"] = pd.to_numeric(
        df["consensus_vote_tp_n_over"], errors="coerce"
    ).astype("Int64")
    display["Under Votes (TP)"] = pd.to_numeric(
        df["consensus_vote_tp_n_under"], errors="coerce"
    ).astype("Int64")

    display[f"{FIVE_THREE_TP_LABEL} Total Points"] = pd.to_numeric(
        df["consensus_tp_5y3y_pred_total"], errors="coerce"
    ).round(1)
    display[f"{FIVE_THREE_TP_LABEL} Line Error"] = pd.to_numeric(
        df["consensus_tp_5y3y_line_diff"], errors="coerce"
    ).round(2)
    display[f"{FIVE_THREE_TP_LABEL} Pick"] = df["consensus_tp_5y3y_pick"]
    display[f"{FIVE_THREE_TP_LABEL} Correct"] = df["consensus_tp_5y3y_correct"].map(
        {True: "✅", False: "❌"}
    )

    display["Vote Pick (LE)"] = df["consensus_vote_le_pick"]
    display["Vote Correct (LE)"] = df["consensus_vote_le_correct"].map(
        {True: "✅", False: "❌"}
    )
    display["Over Votes (LE)"] = pd.to_numeric(
        df["consensus_vote_le_n_over"], errors="coerce"
    ).astype("Int64")
    display["Under Votes (LE)"] = pd.to_numeric(
        df["consensus_vote_le_n_under"], errors="coerce"
    ).astype("Int64")

    display["Bold Contrarian Total Points"] = pd.to_numeric(
        df["consensus_bold_contrarian_pred_total"], errors="coerce"
    ).round(1)
    display["Bold Contrarian Line Error"] = pd.to_numeric(
        df["consensus_bold_contrarian_line_diff"], errors="coerce"
    ).round(2)
    display["Bold Contrarian Pick"] = df["consensus_bold_contrarian_pick"]
    display["Bold Contrarian Correct"] = df["consensus_bold_contrarian_correct"].map(
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


def _has_model_prediction(
    row: pd.Series, model_type: str, catalog: ModelCatalog
) -> bool:
    prefix = catalog.prefixes[model_type]
    return any(
        pd.notna(row.get(col))
        for col in (
            f"pred_total_{prefix}",
            f"line_diff_{prefix}",
            f"pick_{prefix}",
        )
    )


def _available_models_for_row(
    row: pd.Series,
    catalog: ModelCatalog,
    *,
    total_points_only: bool | None = None,
) -> list[str]:
    models = _ordered_model_types_by_prediction_target(catalog)
    if total_points_only is True:
        models = catalog.total_points_models
    elif total_points_only is False:
        models = catalog.diff_from_line_models

    return [
        model_type
        for model_type in models
        if _has_model_prediction(row, model_type, catalog)
    ]


def _format_prediction_timestamp(value: object) -> str:
    pred_dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(pred_dt):
        return "N/A"
    return pred_dt.tz_convert("Europe/Madrid").strftime("%Y-%m-%d %H:%M")


def _render_shap_reason_block(
    title: str,
    items: list[ShapFeatureContribution],
    *,
    accent_color: str,
    empty_text: str,
) -> None:
    if not items:
        body_html = (
            '<div style="padding:14px 12px;border-radius:12px;'
            "background:rgba(248,241,255,0.9);color:#5f4f73;"
            'font-size:0.95rem;font-weight:500;">'
            f"{html.escape(empty_text)}"
            "</div>"
        )
    else:
        rows_html = "".join(
            (
                '<div style="display:flex;align-items:center;justify-content:space-between;'
                "gap:10px;padding:10px 12px;margin-bottom:8px;border-radius:12px;"
                'background:rgba(255,255,255,0.96);border:1px solid rgba(155,107,255,0.18);">'
                '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
                'font-size:0.86rem;color:#1b1422;word-break:break-word;">'
                f"{html.escape(item.feature)}"
                "</div>"
                f'<div style="font-size:0.95rem;font-weight:800;color:{accent_color};'
                'white-space:nowrap;">'
                f"{item.value:+.3f}"
                "</div>"
                "</div>"
            )
            for item in items
        )
        body_html = rows_html

    st.markdown(
        f"""
        <div style="border:1px solid rgba(148,163,184,0.28);border-radius:16px;
                    padding:14px;background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,243,255,0.98));min-height:100%;
                    border-color:rgba(155,107,255,0.22);box-shadow:0 10px 24px rgba(93,64,145,0.06);">
          <div style="font-size:0.78rem;font-weight:800;color:{accent_color};
                      text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
            {html.escape(title)}
          </div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_reasoning_tab(
    row: pd.Series,
    model_type: str,
    catalog: ModelCatalog,
) -> None:
    prefix = catalog.prefixes[model_type]
    pick = format_pick_label(row.get(f"pick_{prefix}"))
    pred_total = pd.to_numeric(row.get(f"pred_total_{prefix}"), errors="coerce")
    line_diff = pd.to_numeric(row.get(f"line_diff_{prefix}"), errors="coerce")
    shap_base = pd.to_numeric(row.get(f"shap_base_{prefix}"), errors="coerce")

    base_label = (
        "SHAP Base Total"
        if model_type in catalog.total_points_models
        else "SHAP Base Margin"
    )
    explanation_text = (
        "Positive SHAP values push the predicted total points higher. "
        "Negative values pull it lower."
        if model_type in catalog.total_points_models
        else "Positive SHAP values push the predicted margin above the line "
        "(toward OVER). Negative values pull it below the line (toward UNDER)."
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Pick", pick)
    with metric_cols[1]:
        st.metric(
            "Predicted Total",
            f"{pred_total:.1f}" if pd.notna(pred_total) else "N/A",
        )
    with metric_cols[2]:
        st.metric(
            "Line Error",
            f"{line_diff:+.1f}" if pd.notna(line_diff) else "N/A",
        )
    with metric_cols[3]:
        st.metric(
            base_label,
            f"{shap_base:.3f}" if pd.notna(shap_base) else "N/A",
        )

    st.caption(
        "Prediction time: "
        f"{_format_prediction_timestamp(row.get(f'pred_dt_{prefix}'))}. "
        f"{explanation_text}"
    )

    positive_items = parse_serialized_shap_contributions(row.get(f"shap_pos_{prefix}"))
    negative_items = parse_serialized_shap_contributions(row.get(f"shap_neg_{prefix}"))

    col_up, col_down = st.columns(2)
    with col_up:
        _render_shap_reason_block(
            "Pushes Prediction Higher",
            positive_items,
            accent_color="#ff9a2f",
            empty_text="No positive SHAP drivers stored for this prediction.",
        )
    with col_down:
        _render_shap_reason_block(
            "Pulls Prediction Lower",
            negative_items,
            accent_color="#9b6bff",
            empty_text="No negative SHAP drivers stored for this prediction.",
        )


def _render_game_reasoning_content(row: pd.Series, catalog: ModelCatalog) -> None:
    available_models = [
        model_type
        for model_type in _ordered_model_types_by_prediction_target(catalog)
        if _has_model_prediction(row, model_type, catalog)
    ]
    if not available_models:
        st.info("No model details available for this game.")
        return

    st.caption(
        "Open a model tab to inspect the main SHAP features that pushed the prediction higher or lower."
    )
    tabs = st.tabs([catalog.labels[model_type] for model_type in available_models])
    for tab, model_type in zip(tabs, available_models, strict=False):
        with tab:
            _render_model_reasoning_tab(row, model_type, catalog)


def _format_reasoning_game_option(row: pd.Series) -> str:
    game_time = pd.to_datetime(row.get("game_time_utc"), errors="coerce", utc=True)
    if pd.isna(game_time):
        time_text = "TBD"
    else:
        time_text = game_time.tz_convert("Europe/Madrid").strftime("%m-%d %H:%M")

    return (
        f"{row['team_name_team_home']} vs {row['team_name_team_away']} "
        f"• {time_text} Madrid"
    )


def render_prediction_reasoning_selector(df: pd.DataFrame, *, key_prefix: str) -> None:
    if df.empty:
        return

    catalog = get_model_catalog(df)
    st.markdown("### Model Reasoning")
    with st.expander("Inspect SHAP drivers for a game", expanded=False):
        options = list(df.index)
        selected_idx = st.selectbox(
            "Game",
            options=options,
            format_func=lambda idx: _format_reasoning_game_option(df.loc[idx]),
            key=f"{key_prefix}_game_reasoning_selector",
        )
        _render_game_reasoning_content(df.loc[selected_idx], catalog)


def _build_model_cell_html(
    row: pd.Series,
    model_type: str,
    catalog: ModelCatalog,
    *,
    show_pred_total: bool = True,
) -> str:
    """Build HTML for a single model prediction cell."""
    prefix = catalog.prefixes[model_type]
    label = catalog.labels[model_type]
    diff = pd.to_numeric(row.get(f"line_diff_{prefix}"), errors="coerce")
    pick = row.get(f"pick_{prefix}")
    diff_text = f"{diff:+.1f}" if pd.notna(diff) else "—"

    if pd.notna(pick) and pick in ("OVER", "UNDER"):
        arrow = "▲" if pick == "OVER" else "▼"
        clr = "#ff9a2f" if pick == "OVER" else "#9b6bff"
        pk_text = str(pick)
    else:
        arrow, clr, pk_text = "—", "#a591ba", "N/A"

    pred_html = ""
    if show_pred_total:
        pred = pd.to_numeric(row.get(f"pred_total_{prefix}"), errors="coerce")
        pred_text = f"{pred:.1f}" if pd.notna(pred) else "—"
        pred_html = (
            f'<div style="font-size:1.3rem;font-weight:800;margin:2px 0;">'
            f"{pred_text}</div>"
        )

    return (
        f'<div style="min-width:0;text-align:center;padding:10px 6px;'
        f"background:linear-gradient(180deg, rgba(255,255,255,0.99), rgba(248,241,255,0.92));border-radius:10px;"
        f'border:1px solid rgba(155,107,255,0.18);box-shadow:0 8px 18px rgba(93,64,145,0.05);">'
        f'<div style="font-size:0.72rem;font-weight:700;color:#5f4f73;'
        f"text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;"
        f'line-height:1.25;word-break:break-word;overflow-wrap:anywhere;">'
        f"{label}</div>"
        f"{pred_html}"
        f'<div style="font-size:1.05rem;font-weight:700;color:{clr};">{diff_text}</div>'
        f'<div style="font-size:0.85rem;font-weight:700;color:{clr};margin-top:3px;">'
        f"{arrow} {pk_text}</div>"
        f"</div>"
    )


def _render_game_card(
    row: pd.Series,
    include_actual: bool,
    catalog: ModelCatalog,
) -> None:
    """Render a single game prediction card as self-contained HTML."""
    home_team = row["team_name_team_home"]
    away_team = row["team_name_team_away"]
    home_logo = get_team_logo_url(home_team)
    away_logo = get_team_logo_url(away_team)

    game_dt = pd.to_datetime(row["game_time_utc"], errors="coerce", utc=True)
    if pd.isna(game_dt):
        game_time, game_date = "TBD", "TBD"
    else:
        dt_madrid = game_dt.tz_convert("Europe/Madrid")
        game_time = dt_madrid.strftime("%I:%M %p")
        game_date = dt_madrid.strftime("%b %d, %Y")

    line_val = pd.to_numeric(row.get("total_over_under_line"), errors="coerce")
    line_text = f"{line_val:.1f}" if pd.notna(line_val) else "N/A"

    # NaN odds columns badge
    na_count_raw = row.get("na_columns_count")
    na_count = int(na_count_raw) if pd.notna(na_count_raw) else 0
    na_names_raw = row.get("na_columns_names")
    na_names_str = (
        str(na_names_raw).strip()
        if pd.notna(na_names_raw) and str(na_names_raw).strip()
        else ""
    )
    if na_count > 0 and na_names_str:
        na_columns_list = "".join(
            f'<div style="padding:2px 0;font-size:0.8rem;color:#7c5a00;'
            f'border-bottom:1px solid rgba(255,180,0,0.2);">{html.escape(col.strip())}</div>'
            for col in na_names_str.split(",")
            if col.strip()
        )
        nan_badge_html = (
            f'<details style="margin-top:6px;cursor:pointer;">'
            f'<summary style="display:inline-flex;align-items:center;gap:5px;'
            f"padding:4px 10px;border-radius:999px;"
            f"background:rgba(255,200,0,0.18);border:1px solid rgba(255,180,0,0.45);"
            f"font-size:0.78rem;font-weight:700;color:#7c5a00;list-style:none;"
            f'cursor:pointer;">'
            f"⚠️ {na_count} NaN feature col{'s' if na_count != 1 else ''} — click to inspect"
            f"</summary>"
            f'<div style="margin-top:6px;padding:8px 10px;background:rgba(255,250,220,0.95);'
            f"border:1px solid rgba(255,200,0,0.35);border-radius:8px;max-height:160px;"
            f'overflow-y:auto;">'
            f"{na_columns_list}"
            f"</div>"
            f"</details>"
        )
    elif na_count > 0:
        nan_badge_html = (
            f'<div style="margin-top:6px;display:inline-flex;align-items:center;gap:5px;'
            f"padding:4px 10px;border-radius:999px;"
            f"background:rgba(255,200,0,0.18);border:1px solid rgba(255,180,0,0.45);"
            f'font-size:0.78rem;font-weight:700;color:#7c5a00;">'
            f"⚠️ {na_count} NaN feature col{'s' if na_count != 1 else ''}"
            f"</div>"
        )
    else:
        nan_badge_html = ""

    if include_actual:
        primary_diff = pd.to_numeric(
            row.get("consensus_tp_5y3y_line_diff"), errors="coerce"
        )
        primary_pick = row.get("consensus_tp_5y3y_pick")
        primary_total = pd.to_numeric(
            row.get("consensus_tp_5y3y_pred_total"), errors="coerce"
        )
    else:
        primary_diff = pd.to_numeric(row.get("consensus_line_diff"), errors="coerce")
        primary_pick = row.get("consensus_pick")
        primary_total = pd.to_numeric(row.get("consensus_pred_total"), errors="coerce")

    if not (pd.notna(primary_pick) and primary_pick in ("OVER", "UNDER", "PUSH")):
        primary_diff = pd.to_numeric(row.get("consensus_line_diff"), errors="coerce")
        primary_pick = row.get("consensus_pick")
        primary_total = pd.to_numeric(row.get("consensus_pred_total"), errors="coerce")

    # Bet recommendation styling
    if pd.notna(primary_pick) and primary_pick in ("OVER", "UNDER"):
        is_over = primary_pick == "OVER"
        bet_label = "BET OVER ▲" if is_over else "BET UNDER ▼"
        accent = "#ff9a2f" if is_over else "#9b6bff"
        banner_bg = (
            "linear-gradient(90deg, rgba(255,154,47,0.18), rgba(255,122,24,0.10))"
            if is_over
            else "linear-gradient(90deg, rgba(155,107,255,0.20), rgba(111,66,255,0.10))"
        )
    else:
        bet_label = "PUSH —"
        accent = "#c8b6d8"
        banner_bg = "linear-gradient(90deg, rgba(80,62,102,0.35), rgba(34,18,53,0.22))"

    margin_text = f"{primary_diff:+.1f}" if pd.notna(primary_diff) else "—"
    cons_total_text = f"{primary_total:.1f}" if pd.notna(primary_total) else "—"

    # Model agreement / majority vote
    n_over_votes = int(row.get("consensus_vote_n_over") or 0)
    n_under_votes = int(row.get("consensus_vote_n_under") or 0)
    vote_pick = row.get("consensus_vote_pick")
    n_avail = n_over_votes + n_under_votes
    if n_avail:
        vote_label = str(vote_pick) if pd.notna(vote_pick) else "TIE"
        vote_text = f"🗳️ Vote: {vote_label} ({n_over_votes}↑ / {n_under_votes}↓)"
    else:
        vote_text = "No model votes"

    # Total Points models vote
    n_over_votes_tp = int(row.get("consensus_vote_tp_n_over") or 0)
    n_under_votes_tp = int(row.get("consensus_vote_tp_n_under") or 0)
    vote_pick_tp = row.get("consensus_vote_tp_pick")
    n_avail_tp = n_over_votes_tp + n_under_votes_tp
    if n_avail_tp:
        vote_label_tp = str(vote_pick_tp) if pd.notna(vote_pick_tp) else "TIE"
        vote_text_tp = (
            f"TP Vote: {vote_label_tp} ({n_over_votes_tp}↑ / {n_under_votes_tp}↓)"
        )
    else:
        vote_text_tp = ""

    hybrid_pick = row.get("consensus_tp_5y3y_pick")
    hybrid_line_diff = pd.to_numeric(
        row.get("consensus_tp_5y3y_line_diff"), errors="coerce"
    )
    if pd.notna(hybrid_pick) and hybrid_pick in ("OVER", "UNDER", "PUSH"):
        hybrid_diff_text = (
            f" ({hybrid_line_diff:+.1f})" if pd.notna(hybrid_line_diff) else ""
        )
        hybrid_text = f"{FIVE_THREE_TP_LABEL}: {hybrid_pick}{hybrid_diff_text}"
    else:
        hybrid_text = ""

    # Line Error models vote
    n_over_votes_le = int(row.get("consensus_vote_le_n_over") or 0)
    n_under_votes_le = int(row.get("consensus_vote_le_n_under") or 0)
    vote_pick_le = row.get("consensus_vote_le_pick")
    n_avail_le = n_over_votes_le + n_under_votes_le
    if n_avail_le:
        vote_label_le = str(vote_pick_le) if pd.notna(vote_pick_le) else "TIE"
        vote_text_le = (
            f"LE Vote: {vote_label_le} ({n_over_votes_le}↑ / {n_under_votes_le}↓)"
        )
    else:
        vote_text_le = ""

    # Build model cells using only models that actually have data for this game.
    available_total_models = _available_models_for_row(
        row, catalog, total_points_only=True
    )
    available_diff_models = _available_models_for_row(
        row, catalog, total_points_only=False
    )

    tp_cells = "".join(
        _build_model_cell_html(row, model_type, catalog, show_pred_total=True)
        for model_type in available_total_models
    )
    dl_cells = "".join(
        _build_model_cell_html(row, model_type, catalog, show_pred_total=False)
        for model_type in available_diff_models
    )

    total_grid_cols = 2 if len(available_total_models) > 1 else 1
    diff_grid_cols = 2 if len(available_diff_models) > 1 else 1
    total_points_grid_style = (
        f"display:grid;grid-template-columns:repeat({total_grid_cols},minmax(0,1fr));"
        "gap:8px;margin-bottom:12px;"
    )
    diff_grid_style = (
        f"display:grid;grid-template-columns:repeat({diff_grid_cols},minmax(0,1fr));"
        "gap:8px;"
    )
    total_section_html = (
        '<div style="font-size:0.75rem;font-weight:700;color:#999;text-transform:uppercase;'
        'letter-spacing:0.06em;margin-bottom:6px;">'
        "📊 Total Points Predictions"
        "</div>"
    )
    if tp_cells:
        total_section_html += f'<div style="{total_points_grid_style}">{tp_cells}</div>'
    else:
        total_section_html += (
            '<div style="padding:12px;border:1px dashed rgba(128,128,128,0.24);'
            "border-radius:10px;color:#5f4f73;font-size:0.9rem;margin-bottom:12px;"
            'background:rgba(248,241,255,0.7);border-color:rgba(155,107,255,0.18);">'
            "No total-points model output available for this game."
            "</div>"
        )

    diff_section_html = (
        '<div style="font-size:0.75rem;font-weight:700;color:#999;'
        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
        "📏 Diff from Line Predictions"
        "</div>"
    )
    if dl_cells:
        diff_section_html += f'<div style="{diff_grid_style}">{dl_cells}</div>'
    else:
        diff_section_html += (
            '<div style="padding:12px;border:1px dashed rgba(128,128,128,0.24);'
            "border-radius:10px;color:#5f4f73;font-size:0.9rem;"
            'background:rgba(248,241,255,0.7);border-color:rgba(155,107,255,0.18);">'
            "No line-error model output available for this game."
            "</div>"
        )

    # Get team scores for past games
    home_pts = pd.to_numeric(row.get("home_pts"), errors="coerce")
    away_pts = pd.to_numeric(row.get("away_pts"), errors="coerce")

    # Build score display for scoreboard (only for past games)
    home_score_html = ""
    away_score_html = ""
    if include_actual and pd.notna(home_pts) and pd.notna(away_pts):
        winner_style = (
            "background:linear-gradient(180deg, rgba(24,18,34,0.96), rgba(12,9,19,0.98));"
            "border:2px solid rgba(255,154,47,0.78);box-shadow:0 10px 18px rgba(0,0,0,0.22);"
        )
        loser_style = (
            "background:linear-gradient(180deg, rgba(20,14,30,0.94), rgba(10,8,16,0.98));"
            "border:2px solid rgba(155,107,255,0.52);box-shadow:0 10px 18px rgba(0,0,0,0.18);"
        )
        tie_style = (
            "background:linear-gradient(180deg, rgba(28,22,38,0.94), rgba(14,10,20,0.98));"
            "border:2px solid rgba(124,107,145,0.58);box-shadow:0 10px 18px rgba(0,0,0,0.16);"
        )

        home_style = (
            winner_style
            if home_pts > away_pts
            else loser_style if home_pts < away_pts else tie_style
        )
        away_style = (
            winner_style
            if away_pts > home_pts
            else loser_style if away_pts < home_pts else tie_style
        )

        home_score_html = (
            f'<div style="margin-top:8px;{home_style}border-radius:8px;'
            f'padding:6px 12px;display:inline-block;">'
            f'<span style="font-size:1.8rem;font-weight:900;color:#fff;">'
            f"{int(home_pts)}</span></div>"
        )
        away_score_html = (
            f'<div style="margin-top:8px;{away_style}border-radius:8px;'
            f'padding:6px 12px;display:inline-block;">'
            f'<span style="font-size:1.8rem;font-weight:900;color:#fff;">'
            f"{int(away_pts)}</span></div>"
        )

    # Actual result banner (past games only)
    actual_banner = ""
    if include_actual:
        actual_total = pd.to_numeric(row.get("total_scored_points"), errors="coerce")
        actual_side = row.get("actual_side")
        primary_correct = row.get("consensus_tp_5y3y_correct")
        if pd.isna(primary_correct):
            primary_correct = row.get("consensus_correct")

        if pd.notna(actual_side) and actual_side in ("OVER", "UNDER", "PUSH"):
            a_clr = (
                "#ff9a2f"
                if actual_side == "OVER"
                else "#9b6bff" if actual_side == "UNDER" else "#c8b6d8"
            )
            a_text = (
                f"{actual_side} ({actual_total:.1f} pts)"
                if pd.notna(actual_total)
                else actual_side
            )
        else:
            a_clr, a_text = "#a591ba", "Pending"

        # Icon logic: handle PUSH games separately
        if actual_side == "PUSH":
            icon = "⚖️"  # Balance/equal symbol for PUSH
        else:
            icon = (
                "✅"
                if pd.notna(primary_correct) and primary_correct
                else "❌" if pd.notna(primary_correct) else "⏳"
            )
        actual_banner = (
            f'<div style="background:{a_clr};color:white;text-align:center;'
            f'padding:10px;font-weight:700;font-size:1.05rem;">'
            f"{icon} RESULT: {a_text}</div>"
        )

    # Per-model correctness summary for past games
    model_results_html = ""
    if include_actual:
        actual_side = row.get("actual_side")
        is_push = actual_side == "PUSH"

        result_cells = ""
        for model_type in _available_models_for_row(row, catalog):
            p = catalog.prefixes[model_type]
            lbl = catalog.labels[model_type]
            flag = row.get(f"correct_{p}")

            if is_push:
                r_icon = "⚖️"  # PUSH games have no winner
            else:
                r_icon = (
                    "✅" if pd.notna(flag) and flag else "❌" if pd.notna(flag) else "—"
                )
            result_cells += (
                f'<div style="flex:1;text-align:center;font-size:0.9rem;'
                f'font-weight:600;">'
                f'<div style="font-size:0.75rem;color:#c8b6d8;'
                f'text-transform:uppercase;">{lbl}</div>'
                f"{r_icon}</div>"
            )

        # Add consensus vote results
        vote_correct = row.get("consensus_vote_correct")
        if is_push:
            vote_icon = "⚖️"
        else:
            vote_icon = (
                "✅"
                if pd.notna(vote_correct) and vote_correct
                else "❌" if pd.notna(vote_correct) else "—"
            )
        result_cells += (
            f'<div style="flex:1;text-align:center;font-size:0.9rem;font-weight:600;">'
            f'<div style="font-size:0.75rem;color:#c8b6d8;text-transform:uppercase;">Vote</div>'
            f"{vote_icon}</div>"
        )

        # Add TP vote results
        vote_tp_correct = row.get("consensus_vote_tp_correct")
        if is_push:
            vote_tp_icon = "⚖️"
        else:
            vote_tp_icon = (
                "✅"
                if pd.notna(vote_tp_correct) and vote_tp_correct
                else "❌" if pd.notna(vote_tp_correct) else "—"
            )
        result_cells += (
            f'<div style="flex:1;text-align:center;font-size:0.9rem;font-weight:600;">'
            f'<div style="font-size:0.75rem;color:#c8b6d8;text-transform:uppercase;">Vote TP</div>'
            f"{vote_tp_icon}</div>"
        )

        hybrid_correct = row.get("consensus_tp_5y3y_correct")
        if is_push:
            hybrid_icon = "⚖️"
        else:
            hybrid_icon = (
                "✅"
                if pd.notna(hybrid_correct) and hybrid_correct
                else "❌" if pd.notna(hybrid_correct) else "—"
            )
        result_cells += (
            f'<div style="flex:1;text-align:center;font-size:0.9rem;font-weight:600;">'
            f'<div style="font-size:0.75rem;color:#c8b6d8;text-transform:uppercase;">5Y/3Y</div>'
            f"{hybrid_icon}</div>"
        )

        # Add LE vote results
        vote_le_correct = row.get("consensus_vote_le_correct")
        if is_push:
            vote_le_icon = "⚖️"
        else:
            vote_le_icon = (
                "✅"
                if pd.notna(vote_le_correct) and vote_le_correct
                else "❌" if pd.notna(vote_le_correct) else "—"
            )
        result_cells += (
            f'<div style="flex:1;text-align:center;font-size:0.9rem;font-weight:600;">'
            f'<div style="font-size:0.75rem;color:#c8b6d8;text-transform:uppercase;">Vote LE</div>'
            f"{vote_le_icon}</div>"
        )

        model_results_html = (
            f'<div style="display:flex;gap:4px;padding:8px 14px 4px;'
            f'border-top:1px solid rgba(128,128,128,0.1);">'
            f"{result_cells}</div>"
        )

    card_html = f"""
    <div style="border:2px solid {accent};border-radius:16px;overflow:hidden;
                margin-bottom:16px;box-shadow:0 16px 34px rgba(93,64,145,0.10);
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                background:linear-gradient(180deg,#ffffff 0%, #fff9f2 100%);">
      <!-- Header -->
      <div style="background:linear-gradient(135deg,#ffffff 0%,#f7eeff 52%,#fff1df 100%);
                  padding:18px 16px;color:#1b1422;border-bottom:1px solid rgba(155,107,255,0.18);">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="flex:1;text-align:center;">
            <img src="{home_logo}" width="64" style="margin-bottom:4px;"
                 onerror="this.style.display='none'">
            <div style="font-size:1.05rem;font-weight:700;line-height:1.2;">
              {home_team}</div>
            {home_score_html}
          </div>
          <div style="flex:0.5;text-align:center;">
            <div style="font-size:0.85rem;opacity:0.8;">{game_date}</div>
            <div style="font-size:1.5rem;font-weight:900;margin:4px 0;">VS</div>
            <div style="font-size:1.0rem;font-weight:600;">🕐 {game_time}</div>
          </div>
          <div style="flex:1;text-align:center;">
            <img src="{away_logo}" width="64" style="margin-bottom:4px;"
                 onerror="this.style.display='none'">
            <div style="font-size:1.05rem;font-weight:700;line-height:1.2;">
              {away_team}</div>
            {away_score_html}
          </div>
        </div>
      </div>
      {actual_banner}
      <!-- Prediction Banner -->
      <div style="background:{banner_bg};padding:12px 16px;
                  border-bottom:1px solid rgba(155,107,255,0.18);">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:1.6rem;font-weight:900;color:{accent};
                        letter-spacing:-0.01em;">{bet_label}</div>
            <div style="font-size:0.8rem;color:#5f4f73;margin-top:1px;">{vote_text}</div>
            {f'<div style="font-size:0.75rem;color:#7c6b91;margin-top:2px;">📊 {vote_text_tp}</div>' if vote_text_tp else ""}
            {f'<div style="font-size:0.75rem;color:#7c6b91;margin-top:2px;">📊 {hybrid_text}</div>' if hybrid_text else ""}
            {f'<div style="font-size:0.75rem;color:#7c6b91;margin-top:2px;">📏 {vote_text_le}</div>' if vote_text_le else ""}
            {nan_badge_html}
          </div>
          <div style="text-align:center;">
            <div style="font-size:0.7rem;font-weight:600;color:#a591ba;
                        text-transform:uppercase;">O/U Line</div>
            <div style="font-size:1.5rem;font-weight:800;color:#1b1422;">{line_text}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:0.7rem;font-weight:600;color:#a591ba;
                        text-transform:uppercase;">Predicted</div>
            <div style="font-size:1.3rem;font-weight:800;color:#1b1422;">
              {cons_total_text}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.7rem;font-weight:600;color:#a591ba;
                        text-transform:uppercase;">Margin</div>
            <div style="font-size:1.3rem;font-weight:700;color:{accent};">
              {margin_text}</div>
          </div>
        </div>
      </div>
      <!-- Models -->
      <div style="padding:12px 14px;">
        {total_section_html}
        {diff_section_html}
      </div>
      {model_results_html}
    </div>
    """

    def _grid_block_height(n_items: int, cols: int) -> int:
        if n_items <= 0:
            return 88
        rows = int(np.ceil(n_items / max(cols, 1)))
        return 42 + (rows * 116) + max(rows - 1, 0) * 8

    model_section_height = _grid_block_height(
        len(available_total_models), total_grid_cols
    )
    model_section_height += _grid_block_height(
        len(available_diff_models), diff_grid_cols
    )
    if include_actual:
        model_section_height += 80
    actual_banner_height = 48 if include_actual else 0
    header_height = 170
    consensus_height = 128 + (50 if na_count > 0 else 0)
    card_height = header_height + actual_banner_height + consensus_height
    card_height += model_section_height + 40
    card_height = max(card_height, 520 if include_actual else 440)
    _render_iframe_html(card_html, height=card_height)


def render_prediction_cards(df: pd.DataFrame, include_actual: bool = False) -> None:
    catalog = get_model_catalog(df)
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
                with st.expander("🧠 Model Reasoning & SHAP Analysis", expanded=False):
                    _render_game_reasoning_content(row, catalog)
                _render_game_card(row, include_actual, catalog)


def _per_class_accuracy(
    picks: pd.Series, actual_sides: pd.Series
) -> tuple[float, float]:
    """Return (over_precision, under_precision) — of predicted OVERs how many are
    actually OVER, and of predicted UNDERs how many are actually UNDER."""
    valid = picks.isin(["OVER", "UNDER"]) & actual_sides.isin(["OVER", "UNDER"])
    p = picks[valid]
    a = actual_sides[valid]
    pred_over = p == "OVER"
    pred_under = p == "UNDER"
    over_prec = (
        float((a[pred_over] == "OVER").mean()) if pred_over.sum() > 0 else np.nan
    )
    under_prec = (
        float((a[pred_under] == "UNDER").mean()) if pred_under.sum() > 0 else np.nan
    )
    return over_prec, under_prec


def _pm_stat(pm_map: dict[str, dict[str, float]], label: str, key: str) -> float | None:
    """Safely extract a percent-rounded stat from the nested pre-midnight map."""
    entry = pm_map.get(label)
    if entry is None:
        return None
    val = entry.get(key, np.nan)
    return None if pd.isna(val) else round(val * 100, 2)


def summarize_model_performance(
    df: pd.DataFrame,
    *,
    raw: pd.DataFrame | None = None,
    training_code_tag_filter: str | None = "1.0",
) -> pd.DataFrame:
    catalog = get_model_catalog(df)
    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])
    pre_midnight_accuracy_map = (
        compute_pre_midnight_accuracy_map(
            raw, training_code_tag_filter=training_code_tag_filter
        )
        if raw is not None
        else {}
    )
    rows = []

    for model_type in _ordered_model_types_by_prediction_target(catalog):
        prefix = catalog.prefixes[model_type]
        picks = df[f"pick_{prefix}"]
        valid_mask = resolved_mask & picks.isin(["OVER", "UNDER"])

        n_games = int(valid_mask.sum())
        accuracy = (
            float(df.loc[valid_mask, f"correct_{prefix}"].mean()) if n_games else np.nan
        )
        over_acc, under_acc = (
            _per_class_accuracy(picks[valid_mask], df.loc[valid_mask, "actual_side"])
            if n_games
            else (np.nan, np.nan)
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

        model_label = catalog.labels[model_type]
        rows.append(
            {
                "Model": model_label,
                "Prediction Target": _prediction_target_label(catalog, model_type),
                "Games": n_games,
                "Accuracy (%)": None if pd.isna(accuracy) else round(accuracy * 100, 2),
                "Over Acc (%)": None if pd.isna(over_acc) else round(over_acc * 100, 2),
                "Under Acc (%)": (
                    None if pd.isna(under_acc) else round(under_acc * 100, 2)
                ),
                "Acc Before 00:00 (%)": _pm_stat(
                    pre_midnight_accuracy_map, model_label, "accuracy"
                ),
                "Over Acc Before 00:00 (%)": _pm_stat(
                    pre_midnight_accuracy_map, model_label, "over_accuracy"
                ),
                "Under Acc Before 00:00 (%)": _pm_stat(
                    pre_midnight_accuracy_map, model_label, "under_accuracy"
                ),
                "Mean Error": None if pd.isna(mean_error) else round(mean_error, 2),
                "MAE": None if pd.isna(mae) else round(mae, 2),
                "Avg |Line Error|": (
                    None
                    if pd.isna(mean_abs_line_diff)
                    else round(mean_abs_line_diff, 2)
                ),
            }
        )

    consensus_picks = df["consensus_pick"]
    consensus_valid_mask = resolved_mask & consensus_picks.isin(["OVER", "UNDER"])
    n_consensus_games = int(consensus_valid_mask.sum())
    consensus_accuracy = (
        float(df.loc[consensus_valid_mask, "consensus_correct"].mean())
        if n_consensus_games
        else np.nan
    )
    consensus_over_acc, consensus_under_acc = (
        _per_class_accuracy(
            consensus_picks[consensus_valid_mask],
            df.loc[consensus_valid_mask, "actual_side"],
        )
        if n_consensus_games
        else (np.nan, np.nan)
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
            "Prediction Target": "Diff From Line",
            "Games": n_consensus_games,
            "Accuracy (%)": (
                None
                if pd.isna(consensus_accuracy)
                else round(consensus_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None
                if pd.isna(consensus_over_acc)
                else round(consensus_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None
                if pd.isna(consensus_under_acc)
                else round(consensus_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus", "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus", "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus", "under_accuracy"
            ),
            "Mean Error": (
                None
                if pd.isna(consensus_mean_error)
                else round(consensus_mean_error, 2)
            ),
            "MAE": None if pd.isna(consensus_mae) else round(consensus_mae, 2),
            "Avg |Line Error|": (
                None
                if pd.isna(consensus_mean_abs_line_diff)
                else round(consensus_mean_abs_line_diff, 2)
            ),
        }
    )

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
    cntpfn_over_acc, cntpfn_under_acc = (
        _per_class_accuracy(
            consensus_no_tabpfn_picks[consensus_no_tabpfn_valid_mask],
            df.loc[consensus_no_tabpfn_valid_mask, "actual_side"],
        )
        if n_consensus_no_tabpfn_games
        else (np.nan, np.nan)
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
            "Prediction Target": "Diff From Line",
            "Games": n_consensus_no_tabpfn_games,
            "Accuracy (%)": (
                None
                if pd.isna(consensus_no_tabpfn_accuracy)
                else round(consensus_no_tabpfn_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None if pd.isna(cntpfn_over_acc) else round(cntpfn_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None if pd.isna(cntpfn_under_acc) else round(cntpfn_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (No TabPFN)", "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (No TabPFN)", "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (No TabPFN)", "under_accuracy"
            ),
            "Mean Error": (
                None
                if pd.isna(consensus_no_tabpfn_mean_error)
                else round(consensus_no_tabpfn_mean_error, 2)
            ),
            "MAE": (
                None
                if pd.isna(consensus_no_tabpfn_mae)
                else round(consensus_no_tabpfn_mae, 2)
            ),
            "Avg |Line Error|": (
                None
                if pd.isna(consensus_no_tabpfn_mean_abs_line_diff)
                else round(consensus_no_tabpfn_mean_abs_line_diff, 2)
            ),
        }
    )

    vote_picks = df["consensus_vote_pick"]
    vote_valid_mask = resolved_mask & vote_picks.isin(["OVER", "UNDER"])
    n_vote_games = int(vote_valid_mask.sum())
    vote_accuracy = (
        float(df.loc[vote_valid_mask, "consensus_vote_correct"].mean())
        if n_vote_games
        else np.nan
    )
    vote_over_acc, vote_under_acc = (
        _per_class_accuracy(
            vote_picks[vote_valid_mask],
            df.loc[vote_valid_mask, "actual_side"],
        )
        if n_vote_games
        else (np.nan, np.nan)
    )
    rows.append(
        {
            "Model": "Consensus (Majority Vote)",
            "Prediction Target": "Pick Vote",
            "Games": n_vote_games,
            "Accuracy (%)": (
                None if pd.isna(vote_accuracy) else round(vote_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None if pd.isna(vote_over_acc) else round(vote_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None if pd.isna(vote_under_acc) else round(vote_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Majority Vote)", "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Majority Vote)", "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Majority Vote)", "under_accuracy"
            ),
            "Mean Error": None,
            "MAE": None,
            "Avg |Line Error|": None,
        }
    )

    vote_tp_picks = df["consensus_vote_tp_pick"]
    vote_tp_valid_mask = resolved_mask & vote_tp_picks.isin(["OVER", "UNDER"])
    n_vote_tp_games = int(vote_tp_valid_mask.sum())
    vote_tp_accuracy = (
        float(df.loc[vote_tp_valid_mask, "consensus_vote_tp_correct"].mean())
        if n_vote_tp_games
        else np.nan
    )
    vote_tp_over_acc, vote_tp_under_acc = (
        _per_class_accuracy(
            vote_tp_picks[vote_tp_valid_mask],
            df.loc[vote_tp_valid_mask, "actual_side"],
        )
        if n_vote_tp_games
        else (np.nan, np.nan)
    )
    rows.append(
        {
            "Model": "Consensus (Vote TP)",
            "Prediction Target": "Pick Vote (TP only)",
            "Games": n_vote_tp_games,
            "Accuracy (%)": (
                None if pd.isna(vote_tp_accuracy) else round(vote_tp_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None if pd.isna(vote_tp_over_acc) else round(vote_tp_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None
                if pd.isna(vote_tp_under_acc)
                else round(vote_tp_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Vote TP)", "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Vote TP)", "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Vote TP)", "under_accuracy"
            ),
            "Mean Error": None,
            "MAE": None,
            "Avg |Line Error|": None,
        }
    )

    hybrid_picks = df["consensus_tp_5y3y_pick"]
    hybrid_valid_mask = resolved_mask & hybrid_picks.isin(["OVER", "UNDER"])
    n_hybrid_games = int(hybrid_valid_mask.sum())
    hybrid_accuracy = (
        float(df.loc[hybrid_valid_mask, "consensus_tp_5y3y_correct"].mean())
        if n_hybrid_games
        else np.nan
    )
    hybrid_over_acc, hybrid_under_acc = (
        _per_class_accuracy(
            hybrid_picks[hybrid_valid_mask],
            df.loc[hybrid_valid_mask, "actual_side"],
        )
        if n_hybrid_games
        else (np.nan, np.nan)
    )
    hybrid_mean_error = (
        float(df.loc[hybrid_valid_mask, "consensus_tp_5y3y_error"].mean())
        if n_hybrid_games
        else np.nan
    )
    hybrid_mae = (
        float(df.loc[hybrid_valid_mask, "consensus_tp_5y3y_error"].abs().mean())
        if n_hybrid_games
        else np.nan
    )
    hybrid_mean_abs_line_diff = (
        float(df.loc[hybrid_valid_mask, "consensus_tp_5y3y_line_diff"].abs().mean())
        if n_hybrid_games
        else np.nan
    )
    rows.append(
        {
            "Model": FIVE_THREE_TP_LABEL,
            "Prediction Target": "Total Points Rule",
            "Games": n_hybrid_games,
            "Accuracy (%)": (
                None if pd.isna(hybrid_accuracy) else round(hybrid_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None if pd.isna(hybrid_over_acc) else round(hybrid_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None if pd.isna(hybrid_under_acc) else round(hybrid_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, FIVE_THREE_TP_LABEL, "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, FIVE_THREE_TP_LABEL, "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, FIVE_THREE_TP_LABEL, "under_accuracy"
            ),
            "Mean Error": (
                None if pd.isna(hybrid_mean_error) else round(hybrid_mean_error, 2)
            ),
            "MAE": None if pd.isna(hybrid_mae) else round(hybrid_mae, 2),
            "Avg |Line Error|": (
                None
                if pd.isna(hybrid_mean_abs_line_diff)
                else round(hybrid_mean_abs_line_diff, 2)
            ),
        }
    )

    vote_le_picks = df["consensus_vote_le_pick"]
    vote_le_valid_mask = resolved_mask & vote_le_picks.isin(["OVER", "UNDER"])
    n_vote_le_games = int(vote_le_valid_mask.sum())
    vote_le_accuracy = (
        float(df.loc[vote_le_valid_mask, "consensus_vote_le_correct"].mean())
        if n_vote_le_games
        else np.nan
    )
    vote_le_over_acc, vote_le_under_acc = (
        _per_class_accuracy(
            vote_le_picks[vote_le_valid_mask],
            df.loc[vote_le_valid_mask, "actual_side"],
        )
        if n_vote_le_games
        else (np.nan, np.nan)
    )
    rows.append(
        {
            "Model": "Consensus (Vote LE)",
            "Prediction Target": "Pick Vote (LE only)",
            "Games": n_vote_le_games,
            "Accuracy (%)": (
                None if pd.isna(vote_le_accuracy) else round(vote_le_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None if pd.isna(vote_le_over_acc) else round(vote_le_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None
                if pd.isna(vote_le_under_acc)
                else round(vote_le_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Vote LE)", "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Vote LE)", "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Consensus (Vote LE)", "under_accuracy"
            ),
            "Mean Error": None,
            "MAE": None,
            "Avg |Line Error|": None,
        }
    )

    bc_picks = df["consensus_bold_contrarian_pick"]
    bc_valid_mask = resolved_mask & bc_picks.isin(["OVER", "UNDER"])
    n_bc_games = int(bc_valid_mask.sum())
    bc_accuracy = (
        float(df.loc[bc_valid_mask, "consensus_bold_contrarian_correct"].mean())
        if n_bc_games
        else np.nan
    )
    bc_over_acc, bc_under_acc = (
        _per_class_accuracy(
            bc_picks[bc_valid_mask],
            df.loc[bc_valid_mask, "actual_side"],
        )
        if n_bc_games
        else (np.nan, np.nan)
    )
    bc_mean_error = (
        float(df.loc[bc_valid_mask, "consensus_bold_contrarian_error"].mean())
        if n_bc_games
        else np.nan
    )
    bc_mae = (
        float(df.loc[bc_valid_mask, "consensus_bold_contrarian_error"].abs().mean())
        if n_bc_games
        else np.nan
    )
    bc_mean_abs_line_diff = (
        float(df.loc[bc_valid_mask, "consensus_bold_contrarian_line_diff"].abs().mean())
        if n_bc_games
        else np.nan
    )
    rows.append(
        {
            "Model": "Bold Contrarian",
            "Prediction Target": "Diff From Line",
            "Games": n_bc_games,
            "Accuracy (%)": (
                None if pd.isna(bc_accuracy) else round(bc_accuracy * 100, 2)
            ),
            "Over Acc (%)": (
                None if pd.isna(bc_over_acc) else round(bc_over_acc * 100, 2)
            ),
            "Under Acc (%)": (
                None if pd.isna(bc_under_acc) else round(bc_under_acc * 100, 2)
            ),
            "Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Bold Contrarian", "accuracy"
            ),
            "Over Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Bold Contrarian", "over_accuracy"
            ),
            "Under Acc Before 00:00 (%)": _pm_stat(
                pre_midnight_accuracy_map, "Bold Contrarian", "under_accuracy"
            ),
            "Mean Error": None if pd.isna(bc_mean_error) else round(bc_mean_error, 2),
            "MAE": None if pd.isna(bc_mae) else round(bc_mae, 2),
            "Avg |Line Error|": (
                None
                if pd.isna(bc_mean_abs_line_diff)
                else round(bc_mean_abs_line_diff, 2)
            ),
        }
    )

    return pd.DataFrame(rows)


def _get_pick_and_actual_for_model(
    df: pd.DataFrame, model_key: str, catalog: ModelCatalog
) -> tuple[pd.Series, pd.Series]:
    """Return (picks, actual_sides) filtered to resolved games with valid picks."""
    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])

    if model_key == "consensus":
        picks = df["consensus_pick"]
    elif model_key == "consensus_no_tabpfn":
        picks = df["consensus_no_tabpfn_pick"]
    elif model_key == "consensus_vote":
        picks = df["consensus_vote_pick"]
    elif model_key == "consensus_vote_tp":
        picks = df["consensus_vote_tp_pick"]
    elif model_key == FIVE_THREE_TP_KEY:
        picks = df["consensus_tp_5y3y_pick"]
    elif model_key == "consensus_vote_le":
        picks = df["consensus_vote_le_pick"]
    elif model_key == "consensus_bold_contrarian":
        picks = df["consensus_bold_contrarian_pick"]
    else:
        prefix = catalog.prefixes[model_key]
        picks = df[f"pick_{prefix}"]

    valid = resolved_mask & picks.isin(["OVER", "UNDER"])
    return picks[valid], df.loc[valid, "actual_side"]


def build_confusion_matrix(
    df: pd.DataFrame, model_key: str, catalog: ModelCatalog
) -> pd.DataFrame:
    """Build a 2x2 confusion matrix DataFrame (rows=Predicted, cols=Actual)."""
    picks, actuals = _get_pick_and_actual_for_model(df, model_key, catalog)
    labels = ["OVER", "UNDER"]
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for pred, actual in zip(picks, actuals, strict=False):
        if pred in labels and actual in labels:
            matrix.loc[pred, actual] += 1
    matrix.index.name = "Predicted"
    matrix.columns.name = "Actual"
    return matrix


def _build_pre_midnight_confusion_matrix(
    raw: pd.DataFrame,
    model_key: str,
    catalog: ModelCatalog,
    training_code_tag_filter: str | None = "1.0",
) -> pd.DataFrame:
    """Build a confusion matrix using only the latest prediction before midnight."""
    history = prepare_prediction_history(
        raw, training_code_tag_filter=training_code_tag_filter
    )
    labels = ["OVER", "UNDER"]
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    if history.empty:
        return matrix

    game_time_madrid = pd.to_datetime(
        history["game_time_utc"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Madrid")
    game_midnight_madrid = game_time_madrid.dt.normalize()

    if "game_date" in history.columns:
        game_date_local = pd.to_datetime(history["game_date"], errors="coerce")
        game_date_local = game_date_local.dt.tz_localize("Europe/Madrid")
        game_midnight_madrid = game_midnight_madrid.where(
            game_midnight_madrid.notna(), game_date_local
        )

    prediction_dt_madrid = pd.to_datetime(
        history["prediction_datetime_utc"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Madrid")
    pre_midnight = history[prediction_dt_madrid < game_midnight_madrid].copy()
    if pre_midnight.empty:
        return matrix

    pre_midnight = (
        pre_midnight.sort_values("prediction_datetime_utc")
        .groupby(["game_id", "_model_key"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # Determine if this is a consensus-level key or a per-model key
    consensus_keys = {
        "consensus",
        "consensus_no_tabpfn",
        "consensus_vote",
        "consensus_vote_tp",
        FIVE_THREE_TP_KEY,
        "consensus_vote_le",
        "consensus_bold_contrarian",
    }

    if model_key not in consensus_keys:
        # Per-model: filter to rows matching model_key
        model_rows = pre_midnight[pre_midnight["_model_key"] == model_key]
        valid = model_rows[
            model_rows["actual_side"].isin(labels)
            & model_rows["pred_pick"].isin(labels)
        ]
        for _, row in valid.iterrows():
            matrix.loc[row["pred_pick"], row["actual_side"]] += 1
    else:
        # Consensus: aggregate per game
        aggregation_catalog = _model_catalog_without_baselines(catalog)
        for _, group in pre_midnight.groupby("game_id"):
            actual_side = group["actual_side"].iloc[-1]
            if actual_side not in set(labels):
                continue

            aggregation_group = exclude_baseline_prediction_rows(group)
            if aggregation_group.empty:
                continue

            diffs = pd.to_numeric(aggregation_group["pred_line_error"], errors="coerce")
            valid_diffs = diffs.dropna()

            if model_key == "consensus" and not valid_diffs.empty:
                pick = pick_from_diff(pd.Series([valid_diffs.mean()])).iloc[0]
            elif model_key == "consensus_no_tabpfn":
                ntpfn = aggregation_group[
                    ~aggregation_group["_model_key"]
                    .astype(str)
                    .str.contains("tabpfn", case=False, na=False)
                ]
                ntpfn_diffs = pd.to_numeric(
                    ntpfn["pred_line_error"], errors="coerce"
                ).dropna()
                if ntpfn_diffs.empty:
                    continue
                pick = pick_from_diff(pd.Series([ntpfn_diffs.mean()])).iloc[0]
            elif model_key == "consensus_bold_contrarian":
                if valid_diffs.empty:
                    continue
                bold_idx = valid_diffs.abs().idxmax()
                pick = aggregation_group.loc[bold_idx, "pred_pick"]
            elif model_key == FIVE_THREE_TP_KEY:
                pick, _line_diff = _five_three_tp_pick_line_diff_from_group(
                    aggregation_group, aggregation_catalog
                )
            elif model_key in (
                "consensus_vote",
                "consensus_vote_tp",
                "consensus_vote_le",
            ):
                if model_key == "consensus_vote_tp":
                    sub = aggregation_group[
                        aggregation_group["_model_key"].isin(
                            aggregation_catalog.total_points_models
                        )
                    ]
                elif model_key == "consensus_vote_le":
                    sub = aggregation_group[
                        aggregation_group["_model_key"].isin(
                            aggregation_catalog.diff_from_line_models
                        )
                    ]
                else:
                    sub = aggregation_group
                vote_picks = sub["pred_pick"][sub["pred_pick"].isin(labels)]
                if vote_picks.empty:
                    continue
                n_over = int((vote_picks == "OVER").sum())
                n_under = int((vote_picks == "UNDER").sum())
                if n_over > n_under:
                    pick = "OVER"
                elif n_under > n_over:
                    pick = "UNDER"
                else:
                    continue
            else:
                continue

            if pick in labels:
                matrix.loc[pick, actual_side] += 1

    return matrix


def _cm_metrics(cm: pd.DataFrame) -> dict[str, float]:
    """Derive standard metrics from a 2x2 confusion matrix."""
    total = cm.values.sum()
    tp_over = int(cm.loc["OVER", "OVER"])
    fp_over = int(cm.loc["OVER", "UNDER"])
    tp_under = int(cm.loc["UNDER", "UNDER"])
    fp_under = int(cm.loc["UNDER", "OVER"])
    return {
        "total": total,
        "accuracy": (tp_over + tp_under) / total if total > 0 else np.nan,
        "precision_over": (
            tp_over / (tp_over + fp_over) if (tp_over + fp_over) > 0 else np.nan
        ),
        "precision_under": (
            tp_under / (tp_under + fp_under) if (tp_under + fp_under) > 0 else np.nan
        ),
        "recall_over": (
            tp_over / (tp_over + fp_under) if (tp_over + fp_under) > 0 else np.nan
        ),
        "recall_under": (
            tp_under / (tp_under + fp_over) if (tp_under + fp_over) > 0 else np.nan
        ),
    }


def _render_cm_html(cm: pd.DataFrame, title: str) -> str:
    """Return an HTML string for a labelled 2x2 confusion matrix."""
    total = cm.values.sum()
    if total == 0:
        cm_pct = cm.copy().astype(float)
    else:
        cm_pct = (cm / total * 100).round(1)

    cells = ""
    for pred_label in ["OVER", "UNDER"]:
        row_cells = ""
        for actual_label in ["OVER", "UNDER"]:
            count = int(cm.loc[pred_label, actual_label])
            pct = cm_pct.loc[pred_label, actual_label]
            is_correct = pred_label == actual_label
            bg = "rgba(34,197,94,0.18)" if is_correct else "rgba(239,68,68,0.12)"
            row_cells += (
                f'<td style="text-align:center;padding:14px 18px;'
                f'background:{bg};border:1px solid rgba(155,107,255,0.15);">'
                f'<div style="font-size:1.35rem;font-weight:800;">{count}</div>'
                f'<div style="font-size:0.8rem;color:#5f4f73;">{pct}%</div>'
                f"</td>"
            )
        cells += (
            f"<tr>"
            f'<td style="font-weight:700;padding:10px 12px;'
            f'background:rgba(255,255,255,0.95);border:1px solid rgba(155,107,255,0.15);">'
            f"{pred_label}</td>"
            f"{row_cells}</tr>"
        )

    return (
        f'<div style="font-size:0.85rem;font-weight:700;color:#5f4f73;margin-bottom:4px;">'
        f"{title} — {int(total)} games</div>"
        f'<table style="border-collapse:collapse;border-radius:12px;overflow:hidden;'
        f'border:1px solid rgba(155,107,255,0.22);margin:4px 0;">'
        f"<thead>"
        f"<tr>"
        f'<th colspan="3" style="padding:6px 10px;background:rgba(111,66,255,0.08);'
        f"border:1px solid rgba(155,107,255,0.15);font-size:0.8rem;"
        f'font-weight:700;text-align:center;color:#5f4f73;">← Actual Outcome →</th>'
        f"</tr>"
        f"<tr>"
        f'<th style="padding:8px 10px;background:rgba(111,66,255,0.12);'
        f"border:1px solid rgba(155,107,255,0.15);font-size:0.7rem;"
        f'color:#5f4f73;text-align:center;">↓ Predicted</th>'
        f'<th style="padding:8px 14px;background:rgba(111,66,255,0.12);'
        f'border:1px solid rgba(155,107,255,0.15);font-weight:700;">OVER</th>'
        f'<th style="padding:8px 14px;background:rgba(111,66,255,0.12);'
        f'border:1px solid rgba(155,107,255,0.15);font-weight:700;">UNDER</th>'
        f"</tr>"
        f"</thead>"
        f"<tbody>{cells}</tbody></table>"
    )


def render_confusion_matrix_selector(
    df: pd.DataFrame,
    *,
    raw: pd.DataFrame | None = None,
    training_code_tag_filter: str | None = "1.0",
    key_prefix: str = "cm",
) -> None:
    """Render an expander with a model selector and its confusion matrix."""
    catalog = get_model_catalog(df)
    model_order = _ordered_model_types_by_prediction_target(catalog)

    options: list[tuple[str, str]] = []
    for m in model_order:
        options.append((m, catalog.labels[m]))
    options.append(("consensus", "Consensus"))
    options.append(("consensus_no_tabpfn", "Consensus (No TabPFN)"))
    options.append(("consensus_vote", "Consensus (Majority Vote)"))
    options.append(("consensus_vote_tp", "Consensus (Vote TP)"))
    options.append((FIVE_THREE_TP_KEY, FIVE_THREE_TP_LABEL))
    options.append(("consensus_vote_le", "Consensus (Vote LE)"))
    options.append(("consensus_bold_contrarian", "Bold Contrarian"))

    with st.expander("🔢 Confusion Matrix", expanded=False):
        selected_label = st.selectbox(
            "Select model",
            [label for _, label in options],
            key=f"{key_prefix}_model_select",
        )
        selected_key = next(key for key, label in options if label == selected_label)

        # Build pre-midnight confusion matrix (shown on left)
        if raw is not None:
            cm_pm = _build_pre_midnight_confusion_matrix(
                raw, selected_key, catalog, training_code_tag_filter
            )
        else:
            cm_pm = pd.DataFrame(0, index=["OVER", "UNDER"], columns=["OVER", "UNDER"])

        # Build latest-prediction confusion matrix (for metrics on right)
        cm_latest = build_confusion_matrix(df, selected_key, catalog)

        pm_total = cm_pm.values.sum()
        latest_total = cm_latest.values.sum()

        if pm_total == 0 and latest_total == 0:
            st.info("No resolved games with valid predictions for this model.")
            return

        def _fmt(v: float) -> str:
            return f"{v * 100:.1f}%" if pd.notna(v) else "—"

        col_matrix, col_stats = st.columns([1.2, 1])

        with col_matrix:
            st.markdown(
                _render_cm_html(cm_pm, f"📅 Before 00:00 — {selected_label}"),
                unsafe_allow_html=True,
            )

        with col_stats:
            pm_m = _cm_metrics(cm_pm)
            latest_m = _cm_metrics(cm_latest)

            st.markdown(f"**{selected_label}**")
            header = "| Metric | Before 00:00 | Latest |\n| :--- | :---: | :---: |\n"
            rows_md = (
                f"| **Games** | {int(pm_m['total'])} | {int(latest_m['total'])} |\n"
                f"| **Overall Accuracy** | {_fmt(pm_m['accuracy'])} | {_fmt(latest_m['accuracy'])} |\n"
                f"| **Over Precision** | {_fmt(pm_m['precision_over'])} | {_fmt(latest_m['precision_over'])} |\n"
                f"| **Under Precision** | {_fmt(pm_m['precision_under'])} | {_fmt(latest_m['precision_under'])} |\n"
                f"| **Over Recall** | {_fmt(pm_m['recall_over'])} | {_fmt(latest_m['recall_over'])} |\n"
                f"| **Under Recall** | {_fmt(pm_m['recall_under'])} | {_fmt(latest_m['recall_under'])} |\n"
            )
            st.markdown(header + rows_md)
            st.caption(
                "Precision = of games *predicted* as X, how many were actually X. "
                "Recall = of games *actually* X, how many were predicted X. "
                "Over/Under Acc in the summary table = Precision."
            )


def compute_threshold_accuracy_table(
    df: pd.DataFrame,
    *,
    model_thresholds: dict[str, tuple[float, ...]] | None = None,
    pre_midnight_accuracy_maps: dict[float, dict[str, float]] | None = None,
) -> pd.DataFrame:
    catalog = get_model_catalog(df)
    if model_thresholds is None:
        model_thresholds = {
            m: (0.0, 0.5, 1.0, 1.5, 2.0)
            for m in _ordered_model_types_by_prediction_target(catalog)
        }
        model_thresholds["consensus"] = (0.0, 0.5, 1.0, 1.5, 2.0)
        model_thresholds["consensus_no_tabpfn"] = (0.0, 0.5, 1.0, 1.5, 2.0)
        model_thresholds[FIVE_THREE_TP_KEY] = (0.0, 0.5, 1.0, 1.5, 2.0)
        model_thresholds["consensus_bold_contrarian"] = (0.0, 0.5, 1.0, 1.5, 2.0)

    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])
    rows: list[dict] = []

    for model_type, thresholds in model_thresholds.items():
        if model_type == "consensus":
            pick_col = "consensus_pick"
            correct_col = "consensus_correct"
            line_diff_col = "consensus_line_diff"
            model_label = "Consensus"
            prediction_target = "Diff From Line"
        elif model_type == "consensus_no_tabpfn":
            pick_col = "consensus_no_tabpfn_pick"
            correct_col = "consensus_no_tabpfn_correct"
            line_diff_col = "consensus_no_tabpfn_line_diff"
            model_label = "Consensus (No TabPFN)"
            prediction_target = "Diff From Line"
        elif model_type == FIVE_THREE_TP_KEY:
            pick_col = "consensus_tp_5y3y_pick"
            correct_col = "consensus_tp_5y3y_correct"
            line_diff_col = "consensus_tp_5y3y_line_diff"
            model_label = FIVE_THREE_TP_LABEL
            prediction_target = "Total Points Rule"
        elif model_type == "consensus_bold_contrarian":
            pick_col = "consensus_bold_contrarian_pick"
            correct_col = "consensus_bold_contrarian_correct"
            line_diff_col = "consensus_bold_contrarian_line_diff"
            model_label = "Bold Contrarian"
            prediction_target = "Diff From Line"
        else:
            prefix = catalog.prefixes[model_type]
            pick_col = f"pick_{prefix}"
            correct_col = f"correct_{prefix}"
            line_diff_col = f"line_diff_{prefix}"
            model_label = catalog.labels[model_type]
            prediction_target = _prediction_target_label(catalog, model_type)

        for threshold in thresholds:
            mask = (
                resolved_mask
                & df[pick_col].isin(["OVER", "UNDER"])
                & (pd.to_numeric(df[line_diff_col], errors="coerce").abs() >= threshold)
            )
            n_games = int(mask.sum())
            accuracy = float(df.loc[mask, correct_col].mean()) if n_games else np.nan

            # Get pre-midnight accuracy for this model and threshold
            pre_midnight_acc = None
            if pre_midnight_accuracy_maps and threshold in pre_midnight_accuracy_maps:
                threshold_map = pre_midnight_accuracy_maps[threshold]
                if model_label in threshold_map:
                    pm_acc = threshold_map[model_label]
                    pre_midnight_acc = (
                        None if pd.isna(pm_acc) else round(pm_acc * 100, 2)
                    )

            rows.append(
                {
                    "Model": model_label,
                    "Prediction Target": prediction_target,
                    "Filter": f"|Line Error| >= {threshold:g}",
                    "Games": n_games,
                    "Accuracy (%)": (
                        None if pd.isna(accuracy) else round(accuracy * 100, 2)
                    ),
                    "Pre-Midnight Accuracy (%)": pre_midnight_acc,
                }
            )

    return pd.DataFrame(rows)


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    catalog = get_model_catalog(df)
    out_rows = []
    temp = df.copy()
    temp["game_date_dt"] = pd.to_datetime(temp["game_date"], errors="coerce")
    temp = temp.dropna(subset=["game_date_dt"])  # keep only valid dates

    for game_date, group in temp.groupby(temp["game_date_dt"].dt.date):
        resolved = group[group["actual_side"].isin(["OVER", "UNDER"])]

        for model_type in _ordered_model_types_by_prediction_target(catalog):
            prefix = catalog.prefixes[model_type]
            valid = resolved[resolved[f"pick_{prefix}"].isin(["OVER", "UNDER"])]
            n_games = len(valid)

            out_rows.append(
                {
                    "game_date": pd.to_datetime(game_date),
                    "model_type": model_type,
                    "model_label": catalog.labels[model_type],
                    "prediction_target": _prediction_target_label(catalog, model_type),
                    "n_games": n_games,
                    "accuracy": (
                        valid[f"correct_{prefix}"].mean() if n_games else np.nan
                    ),
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
                "prediction_target": "Diff From Line",
                "n_games": n_consensus_games,
                "accuracy": (
                    consensus_valid["consensus_correct"].mean()
                    if n_consensus_games
                    else np.nan
                ),
                "mae": (
                    consensus_valid["consensus_error"].abs().mean()
                    if n_consensus_games
                    else np.nan
                ),
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
                "prediction_target": "Diff From Line",
                "n_games": n_consensus_no_tabpfn_games,
                "accuracy": (
                    consensus_no_tabpfn_valid["consensus_no_tabpfn_correct"].mean()
                    if n_consensus_no_tabpfn_games
                    else np.nan
                ),
                "mae": (
                    consensus_no_tabpfn_valid["consensus_no_tabpfn_error"].abs().mean()
                    if n_consensus_no_tabpfn_games
                    else np.nan
                ),
            }
        )

        # Add majority vote metrics for this date
        vote_valid = resolved[resolved["consensus_vote_pick"].isin(["OVER", "UNDER"])]
        n_vote_games_daily = len(vote_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus_vote",
                "model_label": "Consensus (Majority Vote)",
                "prediction_target": "Pick Vote",
                "n_games": n_vote_games_daily,
                "accuracy": (
                    vote_valid["consensus_vote_correct"].mean()
                    if n_vote_games_daily
                    else np.nan
                ),
                "mae": np.nan,
            }
        )

        # Add TP vote metrics for this date
        vote_tp_valid = resolved[
            resolved["consensus_vote_tp_pick"].isin(["OVER", "UNDER"])
        ]
        n_vote_tp_games_daily = len(vote_tp_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus_vote_tp",
                "model_label": "Consensus (Vote TP)",
                "prediction_target": "Pick Vote (TP only)",
                "n_games": n_vote_tp_games_daily,
                "accuracy": (
                    vote_tp_valid["consensus_vote_tp_correct"].mean()
                    if n_vote_tp_games_daily
                    else np.nan
                ),
                "mae": np.nan,
            }
        )

        hybrid_valid = resolved[
            resolved["consensus_tp_5y3y_pick"].isin(["OVER", "UNDER"])
        ]
        n_hybrid_games_daily = len(hybrid_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": FIVE_THREE_TP_KEY,
                "model_label": FIVE_THREE_TP_LABEL,
                "prediction_target": "Total Points Rule",
                "n_games": n_hybrid_games_daily,
                "accuracy": (
                    hybrid_valid["consensus_tp_5y3y_correct"].mean()
                    if n_hybrid_games_daily
                    else np.nan
                ),
                "mae": (
                    hybrid_valid["consensus_tp_5y3y_error"].abs().mean()
                    if n_hybrid_games_daily
                    else np.nan
                ),
            }
        )

        # Add LE vote metrics for this date
        vote_le_valid = resolved[
            resolved["consensus_vote_le_pick"].isin(["OVER", "UNDER"])
        ]
        n_vote_le_games_daily = len(vote_le_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus_vote_le",
                "model_label": "Consensus (Vote LE)",
                "prediction_target": "Pick Vote (LE only)",
                "n_games": n_vote_le_games_daily,
                "accuracy": (
                    vote_le_valid["consensus_vote_le_correct"].mean()
                    if n_vote_le_games_daily
                    else np.nan
                ),
                "mae": np.nan,
            }
        )

        # Add Bold Contrarian metrics for this date
        bc_valid = resolved[
            resolved["consensus_bold_contrarian_pick"].isin(["OVER", "UNDER"])
        ]
        n_bc_games_daily = len(bc_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus_bold_contrarian",
                "model_label": "Bold Contrarian",
                "prediction_target": "Diff From Line",
                "n_games": n_bc_games_daily,
                "accuracy": (
                    bc_valid["consensus_bold_contrarian_correct"].mean()
                    if n_bc_games_daily
                    else np.nan
                ),
                "mae": (
                    bc_valid["consensus_bold_contrarian_error"].abs().mean()
                    if n_bc_games_daily
                    else np.nan
                ),
            }
        )

    daily = pd.DataFrame(out_rows)
    daily["_target_order"] = daily["prediction_target"].map(
        {"Total Points": 0, "Total Points Rule": 1, "Diff From Line": 2}
    )
    daily["_target_order"] = daily["_target_order"].fillna(3)
    return (
        daily.sort_values(["game_date", "_target_order", "model_type"])
        .drop(columns=["_target_order"])
        .reset_index(drop=True)
    )


def prepare_prediction_history(
    df: pd.DataFrame, training_code_tag_filter: str | None = "1.0"
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "model_type" not in work.columns:
        work["model_type"] = np.nan
    if "model_name" not in work.columns:
        work["model_name"] = np.nan
    if "prediction_source" not in work.columns:
        work["prediction_source"] = np.nan
    if "prediction_value_type" not in work.columns:
        work["prediction_value_type"] = np.nan
    if "training_code_tag" not in work.columns:
        work["training_code_tag"] = np.nan

    model_type_source = _normalized_text_series(work, "model_type")
    model_name_source = _normalized_text_series(work, "model_name")
    prediction_source = _normalized_text_series(work, "prediction_source")
    prediction_value_type = _normalized_text_series(work, "prediction_value_type")

    metadata = [
        _build_model_metadata(
            model_type=mt,
            model_name=mn,
            prediction_source=ps,
            prediction_value_type=pvt,
        )
        for mt, mn, ps, pvt in zip(
            model_type_source,
            model_name_source,
            prediction_source,
            prediction_value_type,
            strict=False,
        )
    ]
    work["_model_key"] = [item[0] for item in metadata]
    work["_model_label"] = [item[1] for item in metadata]
    work = work[work["_model_key"].notna()].copy()

    if training_code_tag_filter:
        normalized_tag = str(training_code_tag_filter).strip()
        work = work[
            work["training_code_tag"].fillna("").astype(str).str.strip()
            == normalized_tag
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
    work = work[work["prediction_datetime_utc"].notna()].copy()

    if work.empty:
        return pd.DataFrame()

    if "game_time" in work.columns:
        work["game_time_utc"] = pd.to_datetime(
            work["game_time"], errors="coerce", utc=True
        )
    else:
        work["game_time_utc"] = pd.NaT

    line = pd.to_numeric(work.get("total_over_under_line"), errors="coerce")
    line_at_prediction = pd.to_numeric(
        work.get("total_bet365_line_at_prediction"), errors="coerce"
    )
    work["line_for_calc"] = line.fillna(line_at_prediction)
    work["pred_total_points"] = pd.to_numeric(
        work.get("pred_total_points"), errors="coerce"
    )
    work["pred_line_error"] = pd.to_numeric(
        work.get("pred_line_error"), errors="coerce"
    )
    work["pred_total_points"] = work["pred_total_points"].where(
        work["pred_total_points"].notna(),
        work["line_for_calc"] + work["pred_line_error"],
    )
    work["pred_line_error"] = work["pred_line_error"].where(
        work["pred_line_error"].notna(),
        work["pred_total_points"] - work["line_for_calc"],
    )

    if "pred_pick" in work.columns:
        work["pred_pick"] = work["pred_pick"].apply(normalize_pick)
    else:
        work["pred_pick"] = pd.Series(np.nan, index=work.index, dtype="object")
    work["pred_pick"] = work["pred_pick"].where(
        work["pred_pick"].notna(), pick_from_diff(work["pred_line_error"])
    )

    if "time_to_match_minutes" in work.columns:
        time_to_match = pd.to_numeric(work["time_to_match_minutes"], errors="coerce")
    else:
        time_to_match = pd.Series(np.nan, index=work.index, dtype="float64")
    derived_time_to_match = (
        work["game_time_utc"] - work["prediction_datetime_utc"]
    ).dt.total_seconds() / 60.0
    work["time_to_match_minutes"] = time_to_match.fillna(derived_time_to_match)
    work["hours_to_tip"] = work["time_to_match_minutes"] / 60.0

    actual_total = pd.to_numeric(work.get("total_scored_points"), errors="coerce")
    work["actual_total"] = actual_total
    work["actual_side"] = pick_from_diff(actual_total - work["line_for_calc"])
    work["error"] = work["pred_total_points"] - actual_total

    if "prediction_value_type" in work.columns:
        work["_prediction_priority"] = (
            work["prediction_value_type"].astype(str).str.upper() == "TOTAL_POINTS"
        ).astype(int)
    else:
        work["_prediction_priority"] = 0

    keep_cols = [
        "game_id",
        "game_date",
        "_model_key",
        "_model_label",
        "prediction_datetime_utc",
        "game_time_utc",
        "time_to_match_minutes",
        "hours_to_tip",
        "line_for_calc",
        "pred_total_points",
        "pred_line_error",
        "pred_pick",
        "actual_total",
        "actual_side",
        "error",
        "_prediction_priority",
    ]
    available_keep_cols = [col for col in keep_cols if col in work.columns]
    work = (
        work.sort_values(
            ["game_id", "_model_key", "prediction_datetime_utc", "_prediction_priority"]
        )
        .groupby(["game_id", "_model_key", "prediction_datetime_utc"], as_index=False)
        .tail(1)[available_keep_cols]
        .reset_index(drop=True)
    )

    catalog = extract_model_catalog(df)
    work.attrs["model_catalog"] = catalog
    return work


def compute_pre_midnight_accuracy_with_threshold(
    raw: pd.DataFrame,
    training_code_tag_filter: str | None = "1.0",
    threshold: float = 0.0,
) -> dict[str, float]:
    """Compute pre-midnight accuracy for predictions with |line_error| >= threshold."""
    history = prepare_prediction_history(
        raw, training_code_tag_filter=training_code_tag_filter
    )
    if history.empty:
        return {}

    game_time_madrid = pd.to_datetime(
        history["game_time_utc"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Madrid")
    game_midnight_madrid = game_time_madrid.dt.normalize()

    if "game_date" in history.columns:
        game_date_local = pd.to_datetime(history["game_date"], errors="coerce")
        game_date_local = game_date_local.dt.tz_localize("Europe/Madrid")
        game_midnight_madrid = game_midnight_madrid.where(
            game_midnight_madrid.notna(), game_date_local
        )

    prediction_dt_madrid = pd.to_datetime(
        history["prediction_datetime_utc"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Madrid")
    pre_midnight = history[prediction_dt_madrid < game_midnight_madrid].copy()
    if pre_midnight.empty:
        return {}

    pre_midnight = (
        pre_midnight.sort_values("prediction_datetime_utc")
        .groupby(["game_id", "_model_key"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # Apply threshold filter
    line_error_abs = pd.to_numeric(
        pre_midnight["pred_line_error"], errors="coerce"
    ).abs()
    pre_midnight = pre_midnight[line_error_abs >= threshold].copy()

    if pre_midnight.empty:
        return {}

    catalog = history.attrs.get("model_catalog")
    if not isinstance(catalog, ModelCatalog):
        catalog = extract_model_catalog(raw)

    accuracy_map: dict[str, float] = {}

    for model_type in _ordered_model_types_by_prediction_target(catalog):
        model_rows = pre_midnight[pre_midnight["_model_key"] == model_type]
        valid_rows = model_rows[
            model_rows["actual_side"].isin(["OVER", "UNDER"])
            & model_rows["pred_pick"].isin(["OVER", "UNDER"])
        ]
        accuracy_map[catalog.labels[model_type]] = (
            float((valid_rows["pred_pick"] == valid_rows["actual_side"]).mean())
            if len(valid_rows)
            else np.nan
        )

    aggregation_catalog = _model_catalog_without_baselines(catalog)
    consensus_rows = []
    for _, group in pre_midnight.groupby("game_id"):
        actual_side = group["actual_side"].iloc[-1]
        if actual_side not in {"OVER", "UNDER"}:
            continue

        aggregation_group = exclude_baseline_prediction_rows(group)
        if aggregation_group.empty:
            continue

        diffs = pd.to_numeric(aggregation_group["pred_line_error"], errors="coerce")
        valid_diffs = diffs.dropna()
        if not valid_diffs.empty:
            consensus_pick = pick_from_diff(pd.Series([valid_diffs.mean()])).iloc[0]
            consensus_rows.append(
                {
                    "label": "Consensus",
                    "pick": consensus_pick,
                    "actual_side": actual_side,
                }
            )

        non_tabpfn_group = aggregation_group[
            ~aggregation_group["_model_key"]
            .astype(str)
            .str.contains("tabpfn", case=False, na=False)
        ]
        non_tabpfn_diffs = pd.to_numeric(
            non_tabpfn_group["pred_line_error"], errors="coerce"
        ).dropna()
        if not non_tabpfn_diffs.empty:
            no_tabpfn_pick = pick_from_diff(pd.Series([non_tabpfn_diffs.mean()])).iloc[
                0
            ]
            consensus_rows.append(
                {
                    "label": "Consensus (No TabPFN)",
                    "pick": no_tabpfn_pick,
                    "actual_side": actual_side,
                }
            )

        hybrid_pick, hybrid_line_diff = _five_three_tp_pick_line_diff_from_group(
            aggregation_group, aggregation_catalog
        )
        if hybrid_pick in {"OVER", "UNDER", "PUSH"} and (
            pd.notna(hybrid_line_diff) and abs(hybrid_line_diff) >= threshold
        ):
            consensus_rows.append(
                {
                    "label": FIVE_THREE_TP_LABEL,
                    "pick": hybrid_pick,
                    "actual_side": actual_side,
                }
            )

        if valid_diffs.empty:
            continue

        bold_idx = valid_diffs.abs().idxmax()
        bold_pick = aggregation_group.loc[bold_idx, "pred_pick"]
        consensus_rows.append(
            {"label": "Bold Contrarian", "pick": bold_pick, "actual_side": actual_side}
        )

    if consensus_rows:
        consensus_df = pd.DataFrame(consensus_rows)
        for label, group in consensus_df.groupby("label"):
            valid_rows = group[group["pick"].isin(["OVER", "UNDER"])]
            accuracy_map[label] = (
                float((valid_rows["pick"] == valid_rows["actual_side"]).mean())
                if len(valid_rows)
                else np.nan
            )

    return accuracy_map


def compute_pre_midnight_accuracy_map(
    raw: pd.DataFrame, training_code_tag_filter: str | None = "1.0"
) -> dict[str, dict[str, float]]:
    history = prepare_prediction_history(
        raw, training_code_tag_filter=training_code_tag_filter
    )
    if history.empty:
        return {}

    game_time_madrid = pd.to_datetime(
        history["game_time_utc"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Madrid")
    game_midnight_madrid = game_time_madrid.dt.normalize()

    if "game_date" in history.columns:
        game_date_local = pd.to_datetime(history["game_date"], errors="coerce")
        game_date_local = game_date_local.dt.tz_localize("Europe/Madrid")
        game_midnight_madrid = game_midnight_madrid.where(
            game_midnight_madrid.notna(), game_date_local
        )

    prediction_dt_madrid = pd.to_datetime(
        history["prediction_datetime_utc"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Madrid")
    pre_midnight = history[prediction_dt_madrid < game_midnight_madrid].copy()
    if pre_midnight.empty:
        return {}

    pre_midnight = (
        pre_midnight.sort_values("prediction_datetime_utc")
        .groupby(["game_id", "_model_key"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    catalog = history.attrs.get("model_catalog")
    if not isinstance(catalog, ModelCatalog):
        catalog = extract_model_catalog(raw)

    accuracy_map: dict[str, dict[str, float]] = {}

    for model_type in _ordered_model_types_by_prediction_target(catalog):
        model_rows = pre_midnight[pre_midnight["_model_key"] == model_type]
        valid_rows = model_rows[
            model_rows["actual_side"].isin(["OVER", "UNDER"])
            & model_rows["pred_pick"].isin(["OVER", "UNDER"])
        ]
        if len(valid_rows):
            over_acc, under_acc = _per_class_accuracy(
                valid_rows["pred_pick"], valid_rows["actual_side"]
            )
            accuracy_map[catalog.labels[model_type]] = {
                "accuracy": float(
                    (valid_rows["pred_pick"] == valid_rows["actual_side"]).mean()
                ),
                "over_accuracy": over_acc,
                "under_accuracy": under_acc,
            }
        else:
            accuracy_map[catalog.labels[model_type]] = {
                "accuracy": np.nan,
                "over_accuracy": np.nan,
                "under_accuracy": np.nan,
            }

    aggregation_catalog = _model_catalog_without_baselines(catalog)
    consensus_rows = []
    for _, group in pre_midnight.groupby("game_id"):
        actual_side = group["actual_side"].iloc[-1]
        if actual_side not in {"OVER", "UNDER"}:
            continue

        aggregation_group = exclude_baseline_prediction_rows(group)
        if aggregation_group.empty:
            continue

        diffs = pd.to_numeric(aggregation_group["pred_line_error"], errors="coerce")
        valid_diffs = diffs.dropna()
        if not valid_diffs.empty:
            consensus_pick = pick_from_diff(pd.Series([valid_diffs.mean()])).iloc[0]
            consensus_rows.append(
                {
                    "label": "Consensus",
                    "pick": consensus_pick,
                    "actual_side": actual_side,
                }
            )

        non_tabpfn_group = aggregation_group[
            ~aggregation_group["_model_key"]
            .astype(str)
            .str.contains("tabpfn", case=False, na=False)
        ]
        non_tabpfn_diffs = pd.to_numeric(
            non_tabpfn_group["pred_line_error"], errors="coerce"
        ).dropna()
        if not non_tabpfn_diffs.empty:
            no_tabpfn_pick = pick_from_diff(pd.Series([non_tabpfn_diffs.mean()])).iloc[
                0
            ]
            consensus_rows.append(
                {
                    "label": "Consensus (No TabPFN)",
                    "pick": no_tabpfn_pick,
                    "actual_side": actual_side,
                }
            )

        vote_picks = aggregation_group["pred_pick"][
            aggregation_group["pred_pick"].isin(["OVER", "UNDER"])
        ]
        if not vote_picks.empty:
            over_votes = int((vote_picks == "OVER").sum())
            under_votes = int((vote_picks == "UNDER").sum())
            vote_pick = np.nan
            if over_votes > under_votes:
                vote_pick = "OVER"
            elif under_votes > over_votes:
                vote_pick = "UNDER"
            consensus_rows.append(
                {
                    "label": "Consensus (Majority Vote)",
                    "pick": vote_pick,
                    "actual_side": actual_side,
                }
            )

        # Total Points models vote
        tp_picks = aggregation_group[
            aggregation_group["_model_key"].isin(
                aggregation_catalog.total_points_models
            )
        ]["pred_pick"]
        tp_picks = tp_picks[tp_picks.isin(["OVER", "UNDER"])]
        if not tp_picks.empty:
            tp_over_votes = int((tp_picks == "OVER").sum())
            tp_under_votes = int((tp_picks == "UNDER").sum())
            vote_tp_pick = np.nan
            if tp_over_votes > tp_under_votes:
                vote_tp_pick = "OVER"
            elif tp_under_votes > tp_over_votes:
                vote_tp_pick = "UNDER"
            consensus_rows.append(
                {
                    "label": "Consensus (Vote TP)",
                    "pick": vote_tp_pick,
                    "actual_side": actual_side,
                }
            )

        hybrid_pick, _hybrid_line_diff = _five_three_tp_pick_line_diff_from_group(
            aggregation_group, aggregation_catalog
        )
        if hybrid_pick in {"OVER", "UNDER", "PUSH"}:
            consensus_rows.append(
                {
                    "label": FIVE_THREE_TP_LABEL,
                    "pick": hybrid_pick,
                    "actual_side": actual_side,
                }
            )

        # Line Error models vote
        le_picks = aggregation_group[
            aggregation_group["_model_key"].isin(
                aggregation_catalog.diff_from_line_models
            )
        ]["pred_pick"]
        le_picks = le_picks[le_picks.isin(["OVER", "UNDER"])]
        if not le_picks.empty:
            le_over_votes = int((le_picks == "OVER").sum())
            le_under_votes = int((le_picks == "UNDER").sum())
            vote_le_pick = np.nan
            if le_over_votes > le_under_votes:
                vote_le_pick = "OVER"
            elif le_under_votes > le_over_votes:
                vote_le_pick = "UNDER"
            consensus_rows.append(
                {
                    "label": "Consensus (Vote LE)",
                    "pick": vote_le_pick,
                    "actual_side": actual_side,
                }
            )

        if valid_diffs.empty:
            continue

        bold_idx = valid_diffs.abs().idxmax()
        bold_pick = aggregation_group.loc[bold_idx, "pred_pick"]
        consensus_rows.append(
            {"label": "Bold Contrarian", "pick": bold_pick, "actual_side": actual_side}
        )

    if consensus_rows:
        consensus_df = pd.DataFrame(consensus_rows)
        for label, group in consensus_df.groupby("label"):
            valid_rows = group[group["pick"].isin(["OVER", "UNDER"])]
            if len(valid_rows):
                over_acc, under_acc = _per_class_accuracy(
                    valid_rows["pick"], valid_rows["actual_side"]
                )
                accuracy_map[label] = {
                    "accuracy": float(
                        (valid_rows["pick"] == valid_rows["actual_side"]).mean()
                    ),
                    "over_accuracy": over_acc,
                    "under_accuracy": under_acc,
                }
            else:
                accuracy_map[label] = {
                    "accuracy": np.nan,
                    "over_accuracy": np.nan,
                    "under_accuracy": np.nan,
                }

    return accuracy_map


def bucket_prediction_timing(time_to_match_minutes: pd.Series) -> pd.Series:
    labels = ["<=1h", "2-4h", "4-6h"]
    hours = pd.to_numeric(time_to_match_minutes, errors="coerce") / 60.0
    hours = hours.where(hours >= 0)
    buckets = pd.Series(pd.NA, index=hours.index, dtype="object")
    buckets.loc[hours <= 1] = labels[0]
    buckets.loc[(hours >= 2) & (hours < 4)] = labels[1]
    buckets.loc[(hours >= 4) & (hours <= 6)] = labels[2]
    return pd.Categorical(buckets, categories=labels, ordered=True)


def compute_prediction_timing_metrics(
    df: pd.DataFrame, training_code_tag_filter: str | None = "1.0"
) -> pd.DataFrame:
    history = prepare_prediction_history(
        df, training_code_tag_filter=training_code_tag_filter
    )
    if history.empty:
        return pd.DataFrame()
    catalog = get_model_catalog(history)

    history = history[history["time_to_match_minutes"].notna()].copy()
    history = history[history["time_to_match_minutes"] >= 0].copy()
    history = history[history["time_to_match_minutes"] <= 360].copy()
    if history.empty:
        return pd.DataFrame()

    history["time_bucket"] = bucket_prediction_timing(history["time_to_match_minutes"])
    history = history[history["time_bucket"].notna()].copy()
    if history.empty:
        return pd.DataFrame()

    latest_by_bucket = (
        history.sort_values("prediction_datetime_utc")
        .groupby(
            ["game_id", "_model_key", "time_bucket"], as_index=False, observed=True
        )
        .tail(1)
        .copy()
    )
    final_predictions = (
        history.sort_values("prediction_datetime_utc")
        .groupby(["game_id", "_model_key"], as_index=False)
        .tail(1)[["game_id", "_model_key", "pred_total_points", "pred_pick"]]
        .rename(
            columns={
                "pred_total_points": "final_pred_total_points",
                "pred_pick": "final_pred_pick",
            }
        )
    )
    latest_by_bucket = latest_by_bucket.merge(
        final_predictions, on=["game_id", "_model_key"], how="left"
    )
    latest_by_bucket["move_vs_final_points"] = (
        latest_by_bucket["pred_total_points"]
        - latest_by_bucket["final_pred_total_points"]
    ).abs()

    valid_final_pick_mask = latest_by_bucket["final_pred_pick"].isin(["OVER", "UNDER"])
    valid_bucket_pick_mask = latest_by_bucket["pred_pick"].isin(["OVER", "UNDER"])
    latest_by_bucket["pick_agrees_with_final"] = (
        valid_final_pick_mask
        & valid_bucket_pick_mask
        & (latest_by_bucket["pred_pick"] == latest_by_bucket["final_pred_pick"])
    )

    rows = []
    grouped = latest_by_bucket.groupby(
        ["_model_key", "_model_label", "time_bucket"], observed=True
    )
    for (model_key, model_label, time_bucket), group in grouped:
        resolved = group[
            group["actual_side"].isin(["OVER", "UNDER"])
            & group["pred_pick"].isin(["OVER", "UNDER"])
        ]
        with_final_pick = group[
            group["final_pred_pick"].isin(["OVER", "UNDER"])
            & group["pred_pick"].isin(["OVER", "UNDER"])
        ]

        n_games = len(group)
        accuracy = (
            float((resolved["pred_pick"] == resolved["actual_side"]).mean())
            if len(resolved)
            else np.nan
        )
        mae = float(resolved["error"].abs().mean()) if len(resolved) else np.nan
        mean_abs_line_diff = (
            float(group["pred_line_error"].abs().mean())
            if group["pred_line_error"].notna().any()
            else np.nan
        )
        move_vs_final = (
            float(group["move_vs_final_points"].mean())
            if group["move_vs_final_points"].notna().any()
            else np.nan
        )
        agreement_vs_final = (
            float(with_final_pick["pick_agrees_with_final"].mean())
            if len(with_final_pick)
            else np.nan
        )
        avg_hours_before_tip = (
            float(group["hours_to_tip"].mean())
            if group["hours_to_tip"].notna().any()
            else np.nan
        )

        rows.append(
            {
                "model_type": model_key,
                "Model": model_label,
                "Prediction Target": _prediction_target_label(catalog, model_key),
                "Time Bucket": str(time_bucket),
                "Games": n_games,
                "Avg Hours Before Tip": (
                    None
                    if pd.isna(avg_hours_before_tip)
                    else round(avg_hours_before_tip, 2)
                ),
                "Accuracy (%)": None if pd.isna(accuracy) else round(accuracy * 100, 2),
                "MAE": None if pd.isna(mae) else round(mae, 2),
                "Avg |Line Error|": (
                    None
                    if pd.isna(mean_abs_line_diff)
                    else round(mean_abs_line_diff, 2)
                ),
                "Avg |Move vs Final|": (
                    None if pd.isna(move_vs_final) else round(move_vs_final, 2)
                ),
                "Pick Agreement vs Final (%)": (
                    None
                    if pd.isna(agreement_vs_final)
                    else round(agreement_vs_final * 100, 2)
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    timing_df = pd.DataFrame(rows)
    bucket_order = ["<=1h", "2-4h", "4-6h"]
    timing_df["Time Bucket"] = pd.Categorical(
        timing_df["Time Bucket"], categories=bucket_order, ordered=True
    )
    timing_df["_target_order"] = timing_df["Prediction Target"].map(
        {"Total Points": 0, "Diff From Line": 1}
    )
    timing_df["_target_order"] = timing_df["_target_order"].fillna(2)
    return (
        timing_df.sort_values(["_target_order", "Model", "Time Bucket"])
        .drop(columns=["_target_order"])
        .reset_index(drop=True)
    )


def render_prediction_timing_analysis(
    raw: pd.DataFrame, training_code_tag_filter: str | None
) -> None:
    st.markdown("---")
    st.markdown("### ⏱ Prediction Timing Analysis")
    st.caption(
        "Only predictions made within 6 hours of tip-off are included. Buckets are <=1h, 2-4h, and 4-6h before tip-off. "
        "For each game/model/bucket, the latest prediction inside that bucket is used. Move vs final measures the gap to that model's final pregame prediction."
    )

    timing_df = compute_prediction_timing_metrics(
        raw, training_code_tag_filter=training_code_tag_filter
    )
    if timing_df.empty:
        st.info(
            "No timing-based prediction history is available for the selected range."
        )
        return

    st.dataframe(timing_df, width="stretch", hide_index=True, height=320)

    model_options = timing_df["Model"].dropna().drop_duplicates().tolist()
    selected_model = st.selectbox(
        "Timing detail by model",
        model_options,
        key="historical_prediction_timing_model",
    )
    model_df = timing_df[timing_df["Model"] == selected_model].copy()
    model_df["Time Bucket"] = model_df["Time Bucket"].astype(str)

    st.dataframe(
        model_df.drop(columns=["model_type"]),
        width="stretch",
        hide_index=True,
    )

    fig, (ax_acc, ax_move) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
    x = np.arange(len(model_df))

    ax_acc.set_facecolor("white")
    ax_acc.grid(True, alpha=0.2)
    ax_acc.plot(x, model_df["Accuracy (%)"], linewidth=2.5, marker="o")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(model_df["Time Bucket"], rotation=25, ha="right")
    ax_acc.set_ylim(0, 100)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Time Before Tip")
    ax_acc.set_title(f"{selected_model}: Accuracy by Prediction Time")

    ax_move.set_facecolor("white")
    ax_move.grid(True, alpha=0.2)
    ax_move.plot(x, model_df["Avg |Move vs Final|"], linewidth=2.5, marker="o")
    ax_move.set_xticks(x)
    ax_move.set_xticklabels(model_df["Time Bucket"], rotation=25, ha="right")
    ax_move.set_ylabel("Avg |Move vs Final|")
    ax_move.set_xlabel("Time Before Tip")
    ax_move.set_title(f"{selected_model}: Movement Toward Final Prediction")

    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def show_upcoming_predictions(training_code_tag_filter: str | None) -> None:
    st.markdown("### 🔄 Update Predictions")
    st.caption(
        "Run the prediction model to generate fresh predictions for today's games."
    )
    st.markdown("")

    if run_nba_predictor is None:
        st.warning("Predictor module could not be imported in this environment.")
    elif st.button("Run Predictor Now", type="primary", width="stretch"):
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

    # Load raw data first to extract available prediction times
    with st.spinner("Loading prediction data..."):
        raw = get_games_with_total_scored_points(only_null=True)

    if raw.empty:
        st.info("No upcoming predictions found.")
        return

    raw = exclude_baseline_prediction_rows(raw)
    if raw.empty:
        st.info("No upcoming predictions found.")
        return

    # Extract unique prediction times from raw data
    prediction_times = []
    for col in ["prediction_datetime", "prediction_date"]:
        if col in raw.columns:
            times = parse_prediction_timestamp_series(
                raw[col], treat_naive_as_madrid_local=True
            ).dropna()
            prediction_times.extend(times.tolist())

    if prediction_times:
        # Get unique times and sort descending (most recent first)
        unique_times = sorted(set(prediction_times), reverse=True)

        # Format times for display (Madrid timezone)
        time_options = ["Latest (All available)"]
        time_values = [None]  # None means no cutoff

        for utc_time in unique_times:
            madrid_time = pd.Timestamp(utc_time).tz_convert("Europe/Madrid")
            display_str = madrid_time.strftime("%Y-%m-%d %H:%M:%S (Madrid)")
            time_options.append(display_str)
            time_values.append(utc_time)

        # Add time selector dropdown
        st.markdown("### ⏰ Select Prediction Time")
        st.caption(
            "Choose a prediction time to see predictions as they were at that moment (latest available up to selected time)"
        )

        selected_index = st.selectbox(
            "Prediction Time",
            range(len(time_options)),
            format_func=lambda i: time_options[i],
            key="pred_time_selector",
        )

        prediction_cutoff = time_values[selected_index]

        if prediction_cutoff is not None:
            cutoff_madrid = pd.Timestamp(prediction_cutoff).tz_convert("Europe/Madrid")
            st.caption(
                f"📌 Using predictions up to: {cutoff_madrid.strftime('%Y-%m-%d %H:%M:%S')} Madrid"
            )
    else:
        prediction_cutoff = None

    # Build predictions with optional cutoff
    with st.spinner("Building predictions..."):
        games = build_game_level_predictions(
            raw,
            prediction_cutoff=prediction_cutoff,
            training_code_tag_filter=training_code_tag_filter,
            treat_prediction_timestamps_as_madrid_local=True,
        )

    render_header(get_model_catalog(games))

    if games.empty:
        st.info("No upcoming predictions found for the selected time.")
        return

    latest_prediction_time = pd.to_datetime(
        games["latest_prediction_datetime"], errors="coerce", utc=True
    ).max()

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Upcoming Games", len(games))
    with metric_cols[1]:
        available_all = int(games["all_models_available"].sum())
        st.metric("All Models Available", f"{available_all}/{len(games)}")
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
        cutoff_note = ""
        if prediction_cutoff is not None:
            cutoff_note = f" (filtered up to {prediction_cutoff.tz_convert('Europe/Madrid').strftime('%Y-%m-%d %H:%M:%S')} Madrid)"
        st.caption(
            "Latest prediction: "
            + latest_prediction_time.tz_convert("Europe/Madrid").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " (Madrid)"
            + cutoff_note
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
            build_upcoming_display(
                games, show_pred_times=(prediction_cutoff is not None)
            ),
            width="stretch",
            hide_index=True,
            height=600,
        )
        render_prediction_reasoning_selector(games, key_prefix="upcoming")

    st.markdown("---")
    st.markdown("")
    with st.expander("ℹ️ **How to Read the Predictions**", expanded=False):
        catalog = get_model_catalog(games)
        total_point_labels = [
            catalog.labels[model_type]
            for model_type in catalog.total_points_models
            if "tabpfn" not in model_type.lower()
        ]
        diff_labels = [
            catalog.labels[model_type] for model_type in catalog.diff_from_line_models
        ]
        configured_total_labels = ", ".join(total_point_labels) or "Configured models"
        diff_model_text = ""
        if diff_labels:
            diff_model_text = (
                "- **📏 Diff from Line Models**: "
                + ", ".join(diff_labels)
                + " — predict the difference from the line\n"
            )
        tabpfn_present = any(
            "tabpfn" in model_type.lower() for model_type in catalog.order
        )
        total_points_suffix = ", TabPFN" if tabpfn_present else ""
        st.markdown(
            f"""
        ### 📊 Understanding the Predictions

        - **🏀 Matchup**: Home team vs Away team with logos
        - **⏰ Game Time**: When the game starts (Madrid timezone)
        - **📏 O/U Line**: The bookmaker's over/under betting line
        - **📊 Total Points Models**: {configured_total_labels}{total_points_suffix} — predict total points directly
        {diff_model_text}
        - **🎯 Consensus**: Average of all model margins to decide OVER/UNDER
        - **📊 {FIVE_THREE_TP_LABEL}**: Follows the 5-year total-points model unless the 3-year total-points model says OVER
        - **Margin**: How far the consensus prediction is from the line (positive = OVER)
        - **🧠 Model Reasoning**: Open "Inspect model reasoning" to see SHAP drivers for each model

        ### ⏰ Time Filtering

        - **Latest Predictions** (default): Shows the most recent prediction for each model
        - **Filter by Time**: Select a specific date/time to see predictions as they were at that moment
          - Each model will show its most recent prediction up to the selected time
          - Useful for analyzing how predictions evolved over time

        **Note**: Predictions are updated periodically. Most recent prediction time shown above.
        """
        )


def show_past_games_results(training_code_tag_filter: str | None) -> None:
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

    raw = exclude_baseline_prediction_rows(raw)
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
        training_code_tag_filter=training_code_tag_filter,
    )

    render_header(get_model_catalog(games))

    if games.empty:
        st.warning("No games available at the selected prediction time.")
        return

    resolved_mask = games["actual_side"].isin(["OVER", "UNDER"])
    n_resolved = int(resolved_mask.sum())
    n_push = int((games["actual_side"] == "PUSH").sum())

    metrics_cols = st.columns(7)
    with metrics_cols[0]:
        metric_label = "🎮 Games (Resolved)" if n_push > 0 else "🎮 Games Played"
        metric_value = (
            f"{n_resolved}" if n_push == 0 else f"{n_resolved} ({n_push} push)"
        )
        st.metric(metric_label, metric_value)
    with metrics_cols[1]:
        consensus_mask = resolved_mask & games["consensus_pick"].isin(["OVER", "UNDER"])
        consensus_correct = int(games.loc[consensus_mask, "consensus_correct"].sum())
        consensus_total = int(consensus_mask.sum())
        st.metric("🎯 Consensus", f"{consensus_correct}/{consensus_total}")
    with metrics_cols[2]:
        consensus_no_tabpfn_mask = resolved_mask & games[
            "consensus_no_tabpfn_pick"
        ].isin(["OVER", "UNDER"])
        consensus_no_tabpfn_correct = int(
            games.loc[consensus_no_tabpfn_mask, "consensus_no_tabpfn_correct"].sum()
        )
        consensus_no_tabpfn_total = int(consensus_no_tabpfn_mask.sum())
        st.metric(
            "📊 No TabPFN",
            f"{consensus_no_tabpfn_correct}/{consensus_no_tabpfn_total}",
        )
    with metrics_cols[3]:
        vote_mask = resolved_mask & games["consensus_vote_pick"].isin(["OVER", "UNDER"])
        vote_correct = int(games.loc[vote_mask, "consensus_vote_correct"].sum())
        vote_total = int(vote_mask.sum())
        st.metric("🗳️ Vote", f"{vote_correct}/{vote_total}")
    with metrics_cols[4]:
        vote_tp_mask = resolved_mask & games["consensus_vote_tp_pick"].isin(
            ["OVER", "UNDER"]
        )
        vote_tp_correct = int(
            games.loc[vote_tp_mask, "consensus_vote_tp_correct"].sum()
        )
        vote_tp_total = int(vote_tp_mask.sum())
        st.metric("📊 Vote TP", f"{vote_tp_correct}/{vote_tp_total}")
    with metrics_cols[5]:
        hybrid_mask = resolved_mask & games["consensus_tp_5y3y_pick"].isin(
            ["OVER", "UNDER"]
        )
        hybrid_correct = int(games.loc[hybrid_mask, "consensus_tp_5y3y_correct"].sum())
        hybrid_total = int(hybrid_mask.sum())
        st.metric("📊 5Y/3Y", f"{hybrid_correct}/{hybrid_total}")
    with metrics_cols[6]:
        vote_le_mask = resolved_mask & games["consensus_vote_le_pick"].isin(
            ["OVER", "UNDER"]
        )
        vote_le_correct = int(
            games.loc[vote_le_mask, "consensus_vote_le_correct"].sum()
        )
        vote_le_total = int(vote_le_mask.sum())
        st.metric("📏 Vote LE", f"{vote_le_correct}/{vote_le_total}")

    st.markdown("")
    catalog = get_model_catalog(games)
    ordered_models = _ordered_model_types_by_prediction_target(catalog)
    model_metric_cols = st.columns(max(len(ordered_models), 1))
    for idx, model_type in enumerate(ordered_models):
        prefix = catalog.prefixes[model_type]
        label = catalog.labels[model_type]
        model_mask = resolved_mask & games[f"pick_{prefix}"].isin(["OVER", "UNDER"])
        correct = int(games.loc[model_mask, f"correct_{prefix}"].sum())
        total = int(model_mask.sum())
        with model_metric_cols[idx]:
            st.metric(label, f"{correct}/{total}")

    st.markdown("---")
    st.markdown(f"### 🏀 Games on {date_str}")

    # Show info about push games if any
    if n_push > 0:
        st.caption(
            f"📊 Showing all {len(games)} games ({n_resolved} resolved + {n_push} push). Metrics above exclude push games."
        )
    else:
        st.caption(f"📊 Showing all {len(games)} games.")

    st.markdown("")

    use_cards = st.toggle("Use Card View", value=True, key="past_cards")

    if use_cards:
        render_prediction_cards(games, include_actual=True)
    else:
        st.dataframe(
            build_past_display(games), width="stretch", hide_index=True, height=600
        )
        render_prediction_reasoning_selector(games, key_prefix="past")


def show_historical_performance(training_code_tag_filter: str | None) -> None:
    st.markdown("## Historical Betting Performance")
    st.caption("Analyze model accuracy and prediction error over time.")
    st.markdown("")

    use_date_filter = st.checkbox("Filter by Date Range", value=True)
    start_date = None
    end_date = None

    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2026-03-26"))
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
            training_code_tag_filter=training_code_tag_filter,
        )

    render_header(get_model_catalog(games))

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
    hybrid_mask = resolved_mask & games["consensus_tp_5y3y_pick"].isin(
        ["OVER", "UNDER"]
    )
    hybrid_n = int(hybrid_mask.sum())
    hybrid_acc = (
        (
            games.loc[hybrid_mask, "consensus_tp_5y3y_pick"]
            == games.loc[hybrid_mask, "actual_side"]
        ).mean()
        if hybrid_n
        else np.nan
    )

    top_cols = st.columns(5)
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
    with top_cols[4]:
        st.metric(
            FIVE_THREE_TP_LABEL,
            f"{(hybrid_acc * 100):.2f}%" if pd.notna(hybrid_acc) else "N/A",
        )

    st.markdown("### 💰 Overall Model Statistics")
    st.markdown("")
    summary_df = summarize_model_performance(
        games,
        raw=raw,
        training_code_tag_filter=training_code_tag_filter,
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)

    render_confusion_matrix_selector(
        games,
        raw=raw,
        training_code_tag_filter=training_code_tag_filter,
        key_prefix="hist_cm",
    )

    st.markdown("")
    st.markdown("### 🎯 Accuracy by |Line Error|")
    st.caption(
        "All models: thresholds at >=0, >=0.5, >=1, >=1.5, >=2. "
        "Pre-Midnight Accuracy shows latest prediction before 00h (Madrid time) "
        "filtered by the same |Line Error| threshold."
    )
    # Compute pre-midnight accuracy for each threshold
    pre_midnight_acc_maps = {
        threshold: compute_pre_midnight_accuracy_with_threshold(
            raw, training_code_tag_filter, threshold=threshold
        )
        for threshold in (0.0, 0.5, 1.0, 1.5, 2.0)
    }
    threshold_df = compute_threshold_accuracy_table(
        games, pre_midnight_accuracy_maps=pre_midnight_acc_maps
    )
    st.dataframe(
        threshold_df,
        width="stretch",
        hide_index=True,
        height=240,
    )

    render_prediction_timing_analysis(raw, training_code_tag_filter)

    st.markdown("---")
    st.markdown("### 📅 Daily Accuracy")
    daily = compute_daily_metrics(games)

    if daily.empty:
        st.warning("No daily metrics available.")
        return

    catalog = get_model_catalog(games)
    daily_model_order = [
        catalog.labels[model_type]
        for model_type in _ordered_model_types_by_prediction_target(catalog)
    ]
    daily_model_order.extend(
        ["Consensus", "Consensus (No TabPFN)", FIVE_THREE_TP_LABEL, "Bold Contrarian"]
    )
    accuracy_pivot = (
        daily.pivot(index="game_date", columns="model_label", values="accuracy")
        .sort_index()
        .mul(100)
    )
    accuracy_pivot = accuracy_pivot.reindex(
        columns=[
            label for label in daily_model_order if label in accuracy_pivot.columns
        ]
    )

    st.dataframe(
        accuracy_pivot.reset_index().rename(columns={"game_date": "Date"}).round(2),
        width="stretch",
        hide_index=True,
        height=350,
    )

    smooth_window = st.slider("Smoothing window (days)", 1, 14, 4)

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

    # Handle date locator with better edge case handling
    if len(accuracy_pivot.index) > 1:
        try:
            ax_acc.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=3, maxticks=9)
            )
            ax_acc.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax_acc.xaxis.get_major_locator())
            )
        except Exception:
            # Fallback for problematic date ranges
            ax_acc.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax_acc.legend(frameon=False)
    fig_acc.tight_layout()
    st.pyplot(fig_acc, width="stretch")

    st.markdown("---")
    st.markdown("### 📉 Daily Mean Absolute Error")

    mae_pivot = daily.pivot(
        index="game_date", columns="model_label", values="mae"
    ).sort_index()
    mae_pivot = mae_pivot.reindex(
        columns=[label for label in daily_model_order if label in mae_pivot.columns]
    )

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

    # Handle date locator with better edge case handling
    if len(mae_pivot.index) > 1:
        try:
            ax_mae.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=3, maxticks=9)
            )
            ax_mae.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax_mae.xaxis.get_major_locator())
            )
        except Exception:
            # Fallback for problematic date ranges
            ax_mae.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax_mae.legend(frameon=False)
    fig_mae.tight_layout()
    st.pyplot(fig_mae, width="stretch")


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
        available_training_code_tags = load_available_training_code_tags()
        training_code_tag_options = ["All available", *available_training_code_tags]
        default_training_code_tag = (
            "1.0"
            if "1.0" in available_training_code_tags
            else training_code_tag_options[0]
        )
        training_code_tag_filter = st.selectbox(
            "Training Code Tag",
            options=training_code_tag_options,
            index=training_code_tag_options.index(default_training_code_tag),
            help="Filter predictions by training_code_tag.",
        ).strip()
        st.markdown("---")
        st.markdown("### 🔄 Update Finished Matches")
        st.caption(
            "Fetch final scores for completed games and save them to the database."
        )
        st.markdown("")
        if run_update_finished_matches is None:
            st.warning("Update module could not be imported in this environment.")
        elif st.button(
            "Update Finished Matches", type="secondary", use_container_width=True
        ):
            try:
                with st.spinner("Updating finished matches. This may take a moment..."):
                    run_update_finished_matches()
                st.success("Finished matches updated successfully.")
                time.sleep(1.5)
                st.rerun()
            except Exception as exc:
                st.error(f"Error updating finished matches: {exc}")
                st.exception(exc)
        st.markdown("---")

    selected_training_code_tag = (
        None
        if training_code_tag_filter == "All available"
        else training_code_tag_filter
    )

    if view_option == "Upcoming Predictions":
        show_upcoming_predictions(selected_training_code_tag)
    elif view_option == "Past Games Results":
        show_past_games_results(selected_training_code_tag)
    else:
        show_historical_performance(selected_training_code_tag)


if __name__ == "__main__":
    main()
