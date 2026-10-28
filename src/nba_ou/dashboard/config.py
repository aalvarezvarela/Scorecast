"""Display settings shared by the dashboard pages."""

from zoneinfo import ZoneInfo

MADRID = ZoneInfo("Europe/Madrid")
SLATE_TIMEZONE = ZoneInfo("America/Los_Angeles")
TARGET_LABELS = {2: "Total points", 1: "Line error", 3: "Spread error"}
REFERENCE_COLUMNS = {
    "Bet365": "bet365_line",
    "Market mean": "books_mean_line",
    "Market median": "books_median_line",
}
CACHE_TTL_SECONDS = 60
# Point-based levels used by the production-model evaluation campaigns. Keep
# this lightweight: the dashboard must not import the training pipeline.
MIN_EDGE_LEVELS = (0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEFAULT_MIN_EDGE_POINTS = 1.0
SIGNAL_COLORS = {
    "OVER": "orange",
    "UNDER": "blue",
    "HOME": "orange",
    "AWAY": "blue",
    "NO EDGE": "gray",
    "UNAVAILABLE": "gray",
}
