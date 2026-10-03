"""The compact Yahoo public-betting feature contract (schema 2_5 rebuild).

Closing models use twelve current-game percentages and twenty-four strictly
historical features. Intermediate models use only the historical block until
timestamped Yahoo percentages are available for their prediction horizons.
Yahoo's optional fill of missing BetMGM quotes is a separate, unchanged path.
"""

from collections.abc import Iterable

YAHOO_RAW_SOURCES: tuple[str, ...] = (
    "total_pct_bets_over",
    "total_pct_bets_under",
    "total_pct_money_over",
    "total_pct_money_under",
    "spread_pct_bets_away",
    "spread_pct_bets_home",
    "spread_pct_money_away",
    "spread_pct_money_home",
    "moneyline_pct_bets_away",
    "moneyline_pct_bets_home",
    "moneyline_pct_money_away",
    "moneyline_pct_money_home",
)

# Spread and moneyline histories follow the TEAM, irrespective of where it
# played. Totals-under history largely duplicates totals-over history.
YAHOO_HISTORY_SOURCES: tuple[str, ...] = (
    "total_pct_bets_over",
    "total_pct_money_over",
    "spread_pct_bets",
    "spread_pct_money",
    "moneyline_pct_bets",
    "moneyline_pct_money",
)
YAHOO_HISTORY_SUFFIXES: tuple[str, ...] = (
    "LAST_ALL_5_MATCHES_BEFORE",
    "TREND_SLOPE_LAST_5_GAMES_BEFORE",
)
YAHOO_RAW_COLUMNS = tuple(f"ODDS_{source}" for source in YAHOO_RAW_SOURCES)
YAHOO_HISTORY_COLUMNS = tuple(
    f"ODDS_{source}_{stat}_TEAM_{side}"
    for source in YAHOO_HISTORY_SOURCES
    for stat in YAHOO_HISTORY_SUFFIXES
    for side in ("HOME", "AWAY")
)
YAHOO_FEATURE_COLUMNS = YAHOO_RAW_COLUMNS + YAHOO_HISTORY_COLUMNS


def is_yahoo_percentage_column(column: str) -> bool:
    """Identify the percentage family, with or without the ODDS_ prefix."""
    name = column.lower()
    return "pct_bets" in name or "pct_money" in name


def compact_yahoo_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Recognise the reduced schema without changing archived full schemas.

    Require the complete historical block and no legacy Yahoo derivatives.
    The 24-column intermediate schema deliberately carries no raw percentages.
    """
    columns = set(columns)
    yahoo = {c for c in columns if is_yahoo_percentage_column(c)}
    if not set(YAHOO_HISTORY_COLUMNS).issubset(yahoo):
        return ()
    if not yahoo.issubset(YAHOO_FEATURE_COLUMNS):
        return ()
    raw = yahoo.intersection(YAHOO_RAW_COLUMNS)
    if raw and raw != set(YAHOO_RAW_COLUMNS):
        raise ValueError(
            "The compact Yahoo schema must include all twelve raw percentages "
            "or none (intermediate data). Missing columns: "
            f"{sorted(set(YAHOO_RAW_COLUMNS) - raw)}"
        )
    return tuple(c for c in YAHOO_FEATURE_COLUMNS if c in yahoo)
