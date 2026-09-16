"""Referee features and their publication-time gate for intermediate snapshots."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.config.odds_columns import (
    get_main_book,
    spread_col,
    spread_line_home_col,
    total_line_col,
)
from nba_ou.data_processing.referees.add_referee_features import (
    REFEREE_METRICS,
    _normalize_referee_slots,
    compute_referee_features,
)
from nba_ou.data_processing.referees.referee_tendencies import (
    DEFAULT_REFEREE_HISTORY_SEASONS,
    add_referee_interaction_features,
    add_referee_tendency_features,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.injuries_refs.fetch_refs_db.get_refs_db import (
    get_refs_data_from_db,
)
from nba_ou.postgre_db.odds.merge_odds_data import (
    load_and_merge_odds_yahoo_sportsbookreview,
)

EASTERN = ZoneInfo("America/New_York")
REFEREE_RELEASE_HOUR_ET = 9


def _season_labels(first_year: int, last_year: int) -> list[str]:
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(first_year, last_year + 1)]


def _legacy_referee_features(
    team_games: pd.DataFrame,
    odds: pd.DataFrame,
    refs: pd.DataFrame,
    *,
    first_output_year: int,
    book: str,
) -> pd.DataFrame:
    """Run the existing legacy estimator once per game, using closing history."""
    team_games = team_games[team_games["SEASON_YEAR"] >= first_output_year - 1]
    games = team_games.groupby("GAME_ID", as_index=False).agg(
        GAME_DATE=("GAME_DATE", "first"),
        SEASON_YEAR=("SEASON_YEAR", "first"),
        TOTAL_POINTS=("PTS", "sum"),
        TOTAL_PF=("PF", "sum"),
    )
    raw_line = f"total_{book}_line_over"
    games = games.merge(
        odds[["game_id", raw_line]].rename(
            columns={"game_id": "GAME_ID", raw_line: total_line_col(book)}
        ),
        on="GAME_ID",
        how="left",
        validate="one_to_one",
    )
    named = refs.copy()
    named["GAME_ID"] = named["GAME_ID"].astype(str)
    named["FULL_NAME"] = named["FIRST_NAME"] + " " + named["LAST_NAME"]
    crews = (
        named.groupby("GAME_ID")["FULL_NAME"]
        .apply(lambda names: _normalize_referee_slots(names))
        .unstack()
        .reset_index()
    )
    games = games.merge(crews, on="GAME_ID", how="inner", validate="one_to_one")
    games["DIFF_FROM_LINE"] = games["TOTAL_POINTS"] - games[total_line_col(book)]
    estimated = compute_referee_features(games)
    columns = [
        f"REF_{stat}_{metric}_DIFF_BEFORE"
        for metric in REFEREE_METRICS
        for stat in ("AVG", "STD", "SUM")
    ]
    return estimated[["GAME_ID", *columns]]


def add_intermediate_referee_features(
    base: pd.DataFrame,
    *,
    history_seasons: int = DEFAULT_REFEREE_HISTORY_SEASONS,
    include_same_season_variants: bool = False,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    exclude_caesars: bool = False,
    combine_fanatics_and_caesars: bool | None = None,
) -> pd.DataFrame:
    """Attach the 2_5 referee families before games expand into snapshots."""
    first_year = int(base["SEASON_YEAR"].min())
    last_year = int(base["SEASON_YEAR"].max())
    seasons = _season_labels(first_year - history_seasons, last_year)
    team_games = load_games_from_db(seasons=seasons)
    if team_games is None or team_games.empty:
        raise RuntimeError("Could not load team games for referee history.")
    team_games.columns = team_games.columns.str.upper()
    team_games["GAME_ID"] = team_games["GAME_ID"].astype(str)
    odds = load_and_merge_odds_yahoo_sportsbookreview(
        season_years=seasons,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        exclude_caesars=exclude_caesars,
        combine_fanatics_and_caesars=combine_fanatics_and_caesars,
    )
    odds["game_id"] = odds["game_id"].astype(str)
    refs = get_refs_data_from_db(seasons)
    if refs is None or refs.empty:
        raise RuntimeError("Could not load referee assignments for intermediate games.")

    book = get_main_book()
    legacy = _legacy_referee_features(
        team_games, odds, refs, first_output_year=first_year, book=book
    )
    out = base.merge(legacy, on="GAME_ID", how="left", validate="one_to_one")
    return add_referee_tendency_features(
        out,
        df_team_games=team_games,
        df_refs=refs,
        df_odds=odds,
        total_line_book=book,
        spread_book=book,
        max_history_days=history_seasons * 365,
        include_same_season_variants=include_same_season_variants,
    )


def mask_referees_before_release(df: pd.DataFrame) -> pd.DataFrame:
    """Expose assignments only at/after 09:00 ET on the tipoff date."""
    referee_columns = [column for column in df if column.startswith("REF_")]
    if not referee_columns:
        return df
    tipoff = pd.to_datetime(df["TIPOFF_UTC"], utc=True).dt.tz_convert(EASTERN)
    snapshot = pd.to_datetime(df["SNAPSHOT_TS_UTC"], utc=True).dt.tz_convert(EASTERN)
    # Localize the wall-clock time itself. Adding nine elapsed hours to Eastern
    # midnight is wrong on the spring daylight-saving transition day.
    release = pd.to_datetime(
        tipoff.dt.strftime("%Y-%m-%d") + f" {REFEREE_RELEASE_HOUR_ET:02d}:00"
    ).dt.tz_localize(EASTERN)
    before_release = snapshot.isna() | tipoff.isna() | snapshot.lt(release)
    out = df.copy()
    out.loc[before_release, referee_columns] = float("nan")
    return out


def add_snapshot_referee_interactions(df: pd.DataFrame, *, book: str) -> pd.DataFrame:
    """Compute referee x spread interactions against the snapshot spread."""
    raw_spread_column = spread_col(book)
    out = df.copy()
    # The shared builder expects the raw home handicap (the negative of the
    # canonical market-implied home margin). Replace any closing value here.
    out[raw_spread_column] = -out[spread_line_home_col(book)]
    out = add_referee_interaction_features(out, spread_book=book)
    return out.drop(columns=[raw_spread_column])
