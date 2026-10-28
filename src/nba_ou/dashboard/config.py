"""Display settings shared by the dashboard pages."""

from zoneinfo import ZoneInfo

MADRID = ZoneInfo("Europe/Madrid")
SLATE_TIMEZONE = ZoneInfo("America/Los_Angeles")
# Totals combines both regression targets for the same over/under market.
TARGET_LABELS = {1: "Totals", 3: "Spread"}
TARGET_GROUPS = {1: (1, 2), 3: (3,)}
PREDICTION_TARGET_LABELS = {1: "Line error", 2: "Total points", 3: "Spread error"}
DISPLAY_SCHEMA_VERSION = "2_5"
DEFAULT_TARGET = 1
REFERENCE_COLUMNS = {
    "Bet365": "bet365_line",
    "Market mean": "books_mean_line",
    "Market median": "books_median_line",
}
CACHE_TTL_SECONDS = 60
# Point-based levels used by the production-model evaluation campaigns, plus
# TabPFN's 0.01. Keep this lightweight: the dashboard must not import the
# training pipeline.
MIN_EDGE_LEVELS = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEFAULT_MIN_EDGE_POINTS = 0.1
# Per ``model_builds.kind``: 1 registry (XGBoost) models, 3 TabPFN, whose
# outputs sit much closer to the line.
MODEL_NAMES = {1: "XGBoost", 3: "TabPFN"}
DEFAULT_MIN_EDGE_BY_KIND = {1: DEFAULT_MIN_EDGE_POINTS, 3: 0.01}
SIGNAL_COLORS = {
    "OVER": "red",
    "UNDER": "blue",
    "HOME": "red",
    "AWAY": "blue",
    "NO EDGE": "gray",
    "UNAVAILABLE": "gray",
}
# Readable NBA red/blue tints on the dark dashboard, shared by charts/tables.
SIGNAL_HEX_COLORS = {
    "OVER": "#F37582",
    "UNDER": "#95B8FF",
    "HOME": "#F37582",
    "AWAY": "#95B8FF",
    "NO EDGE": "#99A7BF",
    "UNAVAILABLE": "#99A7BF",
}
