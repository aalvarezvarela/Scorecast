"""Share the day page's opening total with the intermediate feature builder."""

from collections.abc import Iterable

import pandas as pd
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds

OPENER_COLUMNS = tuple(
    f"total_consensus_opener_{field}_{side}"
    for field in ("line", "price")
    for side in ("over", "under")
)


def merge_scheduled_opener(
    history: pd.DataFrame,
    day_odds: pd.DataFrame,
    game_ids: Iterable[str],
    *,
    normalize_total_lines: bool = True,
) -> pd.DataFrame:
    """Overlay scheduled games' raw opening quotes on normalized history.

    The day-page frame must already carry NBA game IDs, as the closing builder's
    ``df_odds_sportsbook_scheduled`` does. Only the opening total and its two
    prices cross this boundary: today's closing quotes, scores and public
    percentages must not become intermediate inputs. Earlier games are unchanged.

    Use the closing/training merge's price conversion and normalization, rather
    than copying a raw line or estimating this field from book-history ticks.
    An explicitly supplied day frame is authoritative for scheduled openers;
    absent quotes stay missing, even when a later historical row exists.
    """
    ids = list(dict.fromkeys(str(game) for game in game_ids))
    if not ids:
        return history
    quotes = day_odds.reindex(columns=["game_id", *OPENER_COLUMNS]).copy()
    quotes["game_id"] = quotes["game_id"].astype(str)
    quotes = quotes[quotes["game_id"].isin(ids)]
    if quotes["game_id"].duplicated().any():
        raise ValueError("More than one day-page opener for a scheduled game")
    quotes = quotes.set_index("game_id").reindex(ids).reset_index()
    for column in OPENER_COLUMNS:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce").astype(float)
    quotes = merge_yahoo_sportsbook_odds(
        pd.DataFrame(),
        quotes,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=False,
    ).set_index("game_id")

    out = history.copy()
    out["game_id"] = out["game_id"].astype(str)
    out = out.set_index("game_id")
    out = out.reindex(out.index.union(quotes.index, sort=False))
    for column in OPENER_COLUMNS:
        if column not in out:
            out[column] = float("nan")
        out.loc[quotes.index, column] = quotes[column]
    return out.reset_index()
