"""
Referee crew tendencies: recency-weighted, shrunk per-official effects.

For every official and every quantity ``y`` (a per-game residual against a
pre-game expectation), the tendency at game ``j`` is

    est_j = sum_i w_ij * y_i / (sum_i w_ij + k)
    w_ij  = 0.5 ** (age_days_ij / half_life_days)

over the official's games ``i`` played on a date strictly before game ``j``
and no more than ``max_history_days`` earlier. ``k`` shrinks officials with
little history toward zero, which is also the value an official with no
history at all receives. The crew feature is the sum over its officials; an
empty slot counts as an unknown official (0), and only crews above three
officials are scaled down.

Why this shape (measured 2026-09 on 2007-2025, forward test 2017+):

* Old history must be down-weighted rather than pooled: officials drift, and
  an equal-weight career average was worse than same-season history for fouls,
  free throws and pace. A 1-4 year half-life beat both.
* Most of the gain is at season openers, where same-season history is empty.
* The strongest totals signal is the crew's free-throw tendency, not its
  past line error.
* Spread-side quantities (home cover, home whistle edge, favourite cover)
  showed no stable forward signal. They are built so the spread model can
  prove or disprove that in-model, and are grouped as ``track="spread"``.
* Splitting by role (crew chief / referee / umpire) added nothing, and the
  historical source only encodes roles from 2021-22 on, so crews are treated
  as unordered sets.

Every quantity is centred on the league's season-to-date mean over strictly
earlier dates, so league-wide rule or officiating-emphasis changes are not
attributed to whichever officials happened to work that season.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nba_ou.config.constants import OVERTIME_THRESHOLD_MINUTES, REGULATION_GAME_MINUTES
from nba_ou.config.market_columns import spread_line_home_from_handicap
from nba_ou.config.odds_columns import spread_col
from nba_ou.data_processing.referees.add_referee_features import (
    canonicalize_referee_name,
)

TRACK_TOTALS = "totals"
TRACK_SPREAD = "spread"

#: Seasons of officiating history loaded before the first output season.
DEFAULT_REFEREE_HISTORY_SEASONS = 6
#: Games older than this never contribute. Makes a feature independent of how
#: much extra history happened to be loaded, so training and same-day
#: prediction produce identical values for the same game.
DEFAULT_MAX_HISTORY_DAYS = DEFAULT_REFEREE_HISTORY_SEASONS * 365

#: Crew features are expressed for a three-official crew. Empty slots count as
#: unknown officials; rare four-official games (an in-game replacement) are
#: scaled down to a three-official equivalent.
CREW_SIZE = 3
#: Completed games allowed to lack a crew. Measured 2014-2025: every
#: non-preseason game has one, so anything above a stray game is a broken or
#: lagging referee source.
DEFAULT_MAX_MISSING_CREW_SHARE = 0.01

PRIOR_GAMES_FEATURE = "REF_CREW_MIN_PRIOR_GAMES_BEFORE"
UNKNOWN_COUNT_FEATURE = "REF_CREW_UNKNOWN_COUNT_BEFORE"
#: Prefix for the optional same-season-only variants. Chosen so that
#: ``exclude_cols_containing`` substrings for the main features never match it.
SAME_SEASON_PREFIX = "REF_CREW_SS_"
UNMATCHED_OFFICIAL_PREFIX = "UNMATCHED:"

EXCLUDED_SEASON_TYPES = ("Preseason", "All Star")


@dataclass(frozen=True)
class RefereeTendencySpec:
    """One per-game quantity turned into one crew feature."""

    quantity: str
    feature: str
    half_life_days: float
    shrinkage_k: float
    track: str
    #: True when the quantity is expressed from the home side, so swapping
    #: home and away negates it.
    home_signed: bool = False

    def same_season_feature(self) -> str:
        return self.feature.replace("REF_CREW_", SAME_SEASON_PREFIX, 1)


#: Half-lives and shrinkage follow the forward-tested optimum per quantity
#: (selection 2012-2016). Tune them with walk-forward CV before changing.
DEFAULT_REFEREE_TENDENCY_SPECS: tuple[RefereeTendencySpec, ...] = (
    RefereeTendencySpec(
        "FTA_RESID", "REF_CREW_FTA_TENDENCY_BEFORE", 548.0, 100.0, TRACK_TOTALS
    ),
    RefereeTendencySpec(
        "PF_RESID", "REF_CREW_PF_TENDENCY_BEFORE", 365.0, 100.0, TRACK_TOTALS
    ),
    RefereeTendencySpec(
        "POSS_RESID", "REF_CREW_POSS_TENDENCY_BEFORE", 365.0, 600.0, TRACK_TOTALS
    ),
    RefereeTendencySpec(
        "LINE_ERROR", "REF_CREW_LINE_ERR_TENDENCY_BEFORE", 1460.0, 250.0, TRACK_TOTALS
    ),
    RefereeTendencySpec(
        "SPREAD_ERROR",
        "REF_CREW_SPREAD_ERROR_TENDENCY_BEFORE",
        1460.0,
        600.0,
        TRACK_SPREAD,
        home_signed=True,
    ),
    RefereeTendencySpec(
        "FAV_SPREAD_ERROR",
        "REF_CREW_FAV_SPREAD_ERROR_TENDENCY_BEFORE",
        730.0,
        250.0,
        TRACK_SPREAD,
    ),
    RefereeTendencySpec(
        "HOME_FTA_EDGE_RESID",
        "REF_CREW_HOME_FTA_EDGE_TENDENCY_BEFORE",
        1460.0,
        600.0,
        TRACK_SPREAD,
        home_signed=True,
    ),
    RefereeTendencySpec(
        "HOME_PF_EDGE_RESID",
        "REF_CREW_HOME_PF_EDGE_TENDENCY_BEFORE",
        730.0,
        100.0,
        TRACK_SPREAD,
        home_signed=True,
    ),
)

HISTORY_QUANTITIES: tuple[str, ...] = tuple(
    s.quantity for s in DEFAULT_REFEREE_TENDENCY_SPECS
)

#: Post-style-matchup combinations, see ``add_referee_interaction_features``.
FTA_X_EXPECTED_TOTAL_FTA_FEATURE = "REF_CREW_FTA_X_STYLE_EXPECTED_TOTAL_FTA_BEFORE"
FTA_X_ABS_SPREAD_FEATURE = "REF_CREW_FTA_X_ABS_SPREAD_BEFORE"
POSS_X_ABS_SPREAD_FEATURE = "REF_CREW_POSS_X_ABS_SPREAD_BEFORE"
FTA_X_HOME_FTA_RATE_EDGE_FEATURE = "REF_CREW_FTA_X_EXPECTED_HOME_FTA_RATE_EDGE_BEFORE"


def referee_tendency_feature_columns(
    specs: tuple[RefereeTendencySpec, ...] = DEFAULT_REFEREE_TENDENCY_SPECS,
    include_same_season_variants: bool = False,
) -> list[str]:
    columns = [spec.feature for spec in specs]
    if include_same_season_variants:
        columns += [spec.same_season_feature() for spec in specs]
    return columns + [PRIOR_GAMES_FEATURE, UNKNOWN_COUNT_FEATURE]


# ---------------------------------------------------------------------------
# Per-game history
# ---------------------------------------------------------------------------


def _prior_dates_mean(
    values: pd.Series, groups: list[pd.Series], dates: pd.Series
) -> pd.Series:
    """Mean of ``values`` within each group over strictly earlier dates.

    Rows sharing a date never see each other, whatever their order.
    """
    keys = [f"g{i}" for i in range(len(groups))]
    frame = pd.DataFrame(
        {
            "v": pd.to_numeric(values, errors="coerce").to_numpy(),
            **{key: group.to_numpy() for key, group in zip(keys, groups, strict=True)},
            "d": dates.to_numpy(),
        },
        index=values.index,
    )
    daily = frame.groupby([*keys, "d"])["v"].agg(["sum", "count"]).sort_index()
    by_group = list(range(len(keys)))
    prior_sum = daily["sum"].groupby(level=by_group).cumsum() - daily["sum"]
    prior_count = daily["count"].groupby(level=by_group).cumsum() - daily["count"]
    prior_mean = (prior_sum / prior_count.where(prior_count > 0)).rename("m")
    looked_up = frame[[*keys, "d"]].merge(
        prior_mean.reset_index(), on=[*keys, "d"], how="left"
    )["m"]
    return pd.Series(looked_up.to_numpy(), index=values.index)


def _season_to_date_mean(
    values: pd.Series, season: pd.Series, dates: pd.Series
) -> pd.Series:
    """Mean of ``values`` within the season over strictly earlier dates."""
    return _prior_dates_mean(values, [season], dates)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes"})


def build_referee_game_history(
    df_team_games: pd.DataFrame,
    df_odds: pd.DataFrame | None = None,
    *,
    total_line_book: str,
    spread_book: str,
) -> pd.DataFrame:
    """One row per completed game with every quantity officials are scored on.

    Args:
        df_team_games: Team-game rows (uppercase columns) with ``GAME_ID``,
            ``GAME_DATE``, ``SEASON_YEAR``, ``SEASON_TYPE``, ``TEAM_ID``,
            ``HOME``, ``PTS``, ``FTA``, ``PF``, ``POSS`` and ``MIN``.
        df_odds: Merged odds frame with ``game_id``,
            ``total_<total_line_book>_line_over`` and
            ``spread_<spread_book>_line_home`` (a home handicap). Optional;
            without it the line/spread quantities are missing.

    Returns:
        DataFrame keyed by ``GAME_ID`` with ``GAME_DATE``, ``SEASON_YEAR`` and
        the centred quantities in ``HISTORY_QUANTITIES``. Games whose box score
        is not final (missing points) are omitted.
    """
    required = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "TEAM_ID",
        "HOME",
        "PTS",
        "FTA",
        "PF",
        "POSS",
        "MIN",
    ]
    missing = [c for c in required if c not in df_team_games.columns]
    if missing:
        raise ValueError(
            f"Team games are missing columns for referee history: {missing}"
        )

    tg = df_team_games[
        required + (["SEASON_TYPE"] if "SEASON_TYPE" in df_team_games.columns else [])
    ].copy()
    if "SEASON_TYPE" in tg.columns:
        tg = tg[~tg["SEASON_TYPE"].isin(EXCLUDED_SEASON_TYPES)]
    tg["GAME_ID"] = tg["GAME_ID"].astype(str)
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"]).dt.normalize()
    tg["HOME"] = _as_bool(tg["HOME"])
    for column in ["PTS", "FTA", "PF", "POSS", "MIN"]:
        tg[column] = pd.to_numeric(tg[column], errors="coerce")
    tg = tg.dropna(subset=["PTS"]).drop_duplicates(["GAME_ID", "TEAM_ID"])

    shape = tg.groupby("GAME_ID")["HOME"].agg(["sum", "size"])
    valid_ids = shape[(shape["sum"] == 1) & (shape["size"] == 2)].index
    tg = tg[tg["GAME_ID"].isin(valid_ids)].sort_values(["GAME_DATE", "GAME_ID"])

    # Whistle volume is compared per regulation game; overtime would otherwise
    # be read as a tight crew.
    scale = np.where(
        tg["MIN"] >= OVERTIME_THRESHOLD_MINUTES,
        REGULATION_GAME_MINUTES / tg["MIN"],
        1.0,
    )
    for column in ["FTA", "PF", "POSS"]:
        tg[f"{column}_REG"] = tg[column] * scale

    opponent = tg[["GAME_ID", "TEAM_ID", "FTA_REG", "PF_REG"]].rename(
        columns={
            "TEAM_ID": "OPP_ID",
            "FTA_REG": "FTA_ALLOWED_REG",
            "PF_REG": "PF_DRAWN_REG",
        }
    )
    tg = tg.merge(opponent, on="GAME_ID")
    tg = tg[tg["TEAM_ID"] != tg["OPP_ID"]].sort_values(["GAME_DATE", "GAME_ID"])

    for column in ["FTA_REG", "PF_REG", "POSS_REG", "FTA_ALLOWED_REG", "PF_DRAWN_REG"]:
        # Strictly earlier dates, not the previous row: two rows for a team on
        # one date must not see each other's result.
        team_prior = _prior_dates_mean(
            tg[column], [tg["TEAM_ID"], tg["SEASON_YEAR"]], tg["GAME_DATE"]
        )
        league_prior = _season_to_date_mean(
            tg[column], tg["SEASON_YEAR"], tg["GAME_DATE"]
        )
        tg[f"{column}_EXP"] = team_prior.fillna(league_prior)

    home = tg[tg["HOME"]].set_index("GAME_ID")
    away = tg[~tg["HOME"]].set_index("GAME_ID").reindex(home.index)

    games = pd.DataFrame(
        {
            "GAME_DATE": home["GAME_DATE"],
            "SEASON_YEAR": home["SEASON_YEAR"],
            "TOTAL_POINTS": home["PTS"] + away["PTS"],
            "HOME_MARGIN": home["PTS"] - away["PTS"],
        }
    )
    # A side's expected free throws average what it usually earns with what
    # its opponent usually concedes; fouls committed likewise.
    fta_home_exp = (home["FTA_REG_EXP"] + away["FTA_ALLOWED_REG_EXP"]) / 2
    fta_away_exp = (away["FTA_REG_EXP"] + home["FTA_ALLOWED_REG_EXP"]) / 2
    pf_home_exp = (home["PF_REG_EXP"] + away["PF_DRAWN_REG_EXP"]) / 2
    pf_away_exp = (away["PF_REG_EXP"] + home["PF_DRAWN_REG_EXP"]) / 2

    raw = pd.DataFrame(index=games.index)
    raw["FTA_RESID"] = (home["FTA_REG"] + away["FTA_REG"]) - (
        fta_home_exp + fta_away_exp
    )
    raw["PF_RESID"] = (home["PF_REG"] + away["PF_REG"]) - (pf_home_exp + pf_away_exp)
    raw["POSS_RESID"] = (home["POSS_REG"] + away["POSS_REG"]) / 2 - (
        home["POSS_REG_EXP"] + away["POSS_REG_EXP"]
    ) / 2
    raw["HOME_FTA_EDGE_RESID"] = (home["FTA_REG"] - away["FTA_REG"]) - (
        fta_home_exp - fta_away_exp
    )
    # Positive = the whistle went against the away side more than expected.
    raw["HOME_PF_EDGE_RESID"] = (away["PF_REG"] - home["PF_REG"]) - (
        pf_away_exp - pf_home_exp
    )

    raw["LINE_ERROR"] = np.nan
    raw["SPREAD_ERROR"] = np.nan
    raw["FAV_SPREAD_ERROR"] = np.nan
    if df_odds is not None and not df_odds.empty and "game_id" in df_odds.columns:
        total_col = f"total_{total_line_book}_line_over"
        spread_home_col = f"spread_{spread_book}_line_home"
        odds = df_odds.copy()
        odds["game_id"] = odds["game_id"].astype(str)
        odds = odds.drop_duplicates("game_id", keep="last").set_index("game_id")
        if total_col in odds.columns:
            total_line = pd.to_numeric(odds[total_col], errors="coerce").reindex(
                games.index
            )
            raw["LINE_ERROR"] = games["TOTAL_POINTS"] - total_line
        if spread_home_col in odds.columns:
            spread_line_home = spread_line_home_from_handicap(
                odds[spread_home_col]
            ).reindex(games.index)
            raw["SPREAD_ERROR"] = games["HOME_MARGIN"] - spread_line_home
            favourite_sign = np.sign(spread_line_home).replace(0.0, np.nan)
            raw["FAV_SPREAD_ERROR"] = raw["SPREAD_ERROR"] * favourite_sign

    season = games["SEASON_YEAR"]
    dates = games["GAME_DATE"]
    for quantity in HISTORY_QUANTITIES:
        centre = _season_to_date_mean(raw[quantity], season, dates).fillna(0.0)
        games[quantity] = raw[quantity] - centre

    games.index.name = "GAME_ID"
    return games.reset_index()


# ---------------------------------------------------------------------------
# Crews
# ---------------------------------------------------------------------------


def build_official_name_index(df_refs: pd.DataFrame) -> dict[str, str]:
    """Canonical ``"First Last"`` -> ``OFFICIAL_ID`` of that name's latest game."""
    if df_refs is None or df_refs.empty:
        return {}
    refs = df_refs.copy()
    refs["NAME"] = (
        refs["FIRST_NAME"].astype(str) + " " + refs["LAST_NAME"].astype(str)
    ).map(canonicalize_referee_name)
    refs["GAME_DATE"] = pd.to_datetime(refs["GAME_DATE"])
    latest = (
        refs.dropna(subset=["NAME"])
        .sort_values("GAME_DATE")
        .drop_duplicates("NAME", keep="last")
    )
    return dict(zip(latest["NAME"], latest["OFFICIAL_ID"].astype(str), strict=True))


def build_official_crews(
    df_refs: pd.DataFrame,
    game_dates: pd.DataFrame,
    df_referees_scheduled: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long frame of ``GAME_ID, GAME_DATE, OFFICIAL_ID`` for historical and scheduled crews.

    Scheduled assignments carry only names, so they are resolved to
    ``OFFICIAL_ID`` through the historical name index. A name that cannot be
    resolved becomes ``UNMATCHED:<name>``: it has no history, contributes the
    neutral value 0 and is counted in ``REF_CREW_UNKNOWN_COUNT_BEFORE``.
    A scheduled crew replaces any stored crew for the same game.
    """
    dates = game_dates[["GAME_ID", "GAME_DATE"]].copy()
    dates["GAME_ID"] = dates["GAME_ID"].astype(str)
    dates["GAME_DATE"] = pd.to_datetime(dates["GAME_DATE"]).dt.normalize()
    dates = dates.drop_duplicates("GAME_ID", keep="last")

    frames = []
    if df_refs is not None and not df_refs.empty:
        hist = df_refs[["GAME_ID", "OFFICIAL_ID"]].copy()
        hist["GAME_ID"] = hist["GAME_ID"].astype(str)
        hist["OFFICIAL_ID"] = hist["OFFICIAL_ID"].astype(str)
        frames.append(hist.drop_duplicates())

    if df_referees_scheduled is not None and not df_referees_scheduled.empty:
        name_index = build_official_name_index(df_refs)
        slot_cols = [
            c for c in ("REF_1", "REF_2", "REF_3") if c in df_referees_scheduled.columns
        ]
        sched = df_referees_scheduled[["GAME_ID", *slot_cols]].melt(
            id_vars="GAME_ID", value_name="NAME"
        )
        sched["GAME_ID"] = sched["GAME_ID"].astype(str)
        sched["NAME"] = sched["NAME"].map(canonicalize_referee_name)
        sched = sched.dropna(subset=["NAME"])
        sched["OFFICIAL_ID"] = sched["NAME"].map(name_index)
        unmatched = sorted(sched.loc[sched["OFFICIAL_ID"].isna(), "NAME"].unique())
        if unmatched:
            warnings.warn(
                "Scheduled referee(s) without officiating history, treated as unknown "
                f"(neutral tendency): {unmatched}",
                stacklevel=2,
            )
        sched["OFFICIAL_ID"] = sched["OFFICIAL_ID"].fillna(
            UNMATCHED_OFFICIAL_PREFIX + sched["NAME"]
        )
        sched = sched[["GAME_ID", "OFFICIAL_ID"]].drop_duplicates()
        if frames:
            frames[0] = frames[0][~frames[0]["GAME_ID"].isin(set(sched["GAME_ID"]))]
        frames.append(sched)

    if not frames:
        return pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "OFFICIAL_ID"])
    crews = pd.concat(frames, ignore_index=True).merge(dates, on="GAME_ID", how="inner")
    return crews.sort_values(["GAME_DATE", "GAME_ID", "OFFICIAL_ID"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def compute_official_tendencies(
    crews: pd.DataFrame,
    game_history: pd.DataFrame,
    specs: tuple[RefereeTendencySpec, ...] = DEFAULT_REFEREE_TENDENCY_SPECS,
    max_history_days: int = DEFAULT_MAX_HISTORY_DAYS,
    include_same_season_variants: bool = False,
) -> pd.DataFrame:
    """Per official-game tendency estimates using only earlier games.

    Returns ``crews`` with one ``<feature>`` column per spec (and optional
    same-season variants) holding the official's estimate, plus
    ``PRIOR_GAMES`` = games inside the window before this date.
    """
    if max_history_days <= 0:
        raise ValueError("max_history_days must be positive.")

    history = game_history.copy()
    history["GAME_ID"] = history["GAME_ID"].astype(str)
    frame = crews.merge(
        history[["GAME_ID", "SEASON_YEAR", *[s.quantity for s in specs]]],
        on="GAME_ID",
        how="left",
    )
    # Scheduled games have no history row; their season comes from the date.
    fallback_season = frame["GAME_DATE"].dt.year - frame["GAME_DATE"].dt.month.le(
        7
    ).astype(int)
    frame["SEASON_YEAR"] = frame["SEASON_YEAR"].fillna(fallback_season).astype(int)
    frame = frame.sort_values(["OFFICIAL_ID", "GAME_DATE"]).reset_index(drop=True)

    out_columns = [spec.feature for spec in specs]
    if include_same_season_variants:
        out_columns += [spec.same_season_feature() for spec in specs]
    results = {column: np.zeros(len(frame)) for column in out_columns}
    prior_games = np.zeros(len(frame))

    for _, positions in frame.groupby("OFFICIAL_ID", sort=False).indices.items():
        days = (
            frame["GAME_DATE"]
            .to_numpy()[positions]
            .astype("datetime64[D]")
            .astype(np.int64)
        )
        seasons = frame["SEASON_YEAR"].to_numpy()[positions]
        age = (days[:, None] - days[None, :]).astype(float)
        in_window = (age > 0) & (age <= max_history_days)
        prior_games[positions] = in_window.sum(axis=1)
        same_season = in_window & (seasons[:, None] == seasons[None, :])

        for spec in specs:
            y = frame[spec.quantity].to_numpy(dtype=float)[positions]
            observed = ~np.isnan(y)
            decay = 0.5 ** (age / spec.half_life_days)
            masks = [(spec.feature, in_window)]
            if include_same_season_variants:
                masks.append((spec.same_season_feature(), same_season))
            for column, mask in masks:
                weights = np.where(mask, decay, 0.0)[:, observed]
                weighted_sum = weights @ y[observed]
                denominator = weights.sum(axis=1) + spec.shrinkage_k
                results[column][positions] = np.divide(
                    weighted_sum,
                    denominator,
                    out=np.zeros_like(weighted_sum),
                    where=denominator > 0,
                )

    for column, values in results.items():
        frame[column] = values
    frame["PRIOR_GAMES"] = prior_games
    return frame


def aggregate_crew_features(
    official_estimates: pd.DataFrame,
    specs: tuple[RefereeTendencySpec, ...] = DEFAULT_REFEREE_TENDENCY_SPECS,
    include_same_season_variants: bool = False,
) -> pd.DataFrame:
    """Collapse per-official estimates to one row per game.

    A crew with fewer than ``CREW_SIZE`` listed officials is treated as having
    unknown officials in the empty slots: they contribute the neutral 0, count
    toward ``REF_CREW_UNKNOWN_COUNT_BEFORE`` and pull the minimum prior games
    to 0. Only crews larger than ``CREW_SIZE`` (in-game replacements) are
    scaled down to a three-official equivalent.
    """
    columns = [spec.feature for spec in specs]
    if include_same_season_variants:
        columns += [spec.same_season_feature() for spec in specs]
    grouped = official_estimates.groupby("GAME_ID")
    officials = grouped.size()
    missing_slots = (CREW_SIZE - officials).clip(lower=0)

    crew = grouped[columns].mean().mul(officials.clip(upper=CREW_SIZE), axis=0)
    crew[PRIOR_GAMES_FEATURE] = (
        grouped["PRIOR_GAMES"].min().where(missing_slots == 0, 0)
    )
    crew[UNKNOWN_COUNT_FEATURE] = (
        official_estimates["PRIOR_GAMES"]
        .eq(0)
        .groupby(official_estimates["GAME_ID"])
        .sum()
        + missing_slots
    )
    return crew.reset_index()


def add_referee_tendency_features(
    df_merged: pd.DataFrame,
    *,
    df_team_games: pd.DataFrame,
    df_refs: pd.DataFrame,
    df_odds: pd.DataFrame | None,
    total_line_book: str,
    spread_book: str,
    df_referees_scheduled: pd.DataFrame | None = None,
    specs: tuple[RefereeTendencySpec, ...] = DEFAULT_REFEREE_TENDENCY_SPECS,
    max_history_days: int = DEFAULT_MAX_HISTORY_DAYS,
    include_same_season_variants: bool = False,
    max_missing_crew_share: float = DEFAULT_MAX_MISSING_CREW_SHARE,
) -> pd.DataFrame:
    """Attach ``REF_CREW_*_BEFORE`` features to the merged one-row-per-game frame.

    ``df_team_games``, ``df_refs`` and ``df_odds`` are the officiating
    history and must reach back ``max_history_days`` before the first game in
    ``df_merged`` for early rows to see their full window. They are loaded
    independently of the output seasons for exactly that reason.

    Raises:
        ValueError: if ``df_refs`` is empty, or if more than
            ``max_missing_crew_share`` of the completed games in ``df_merged``
            end up without a crew. Either means the referee source is broken
            or behind, and continuing would train on silently missing
            features. Scheduled games without an assignment only warn.
    """
    if df_refs is None or df_refs.empty:
        raise ValueError(
            "No referee assignments were provided for the REF_CREW_* features; "
            "the referee database load returned nothing."
        )

    history = build_referee_game_history(
        df_team_games, df_odds, total_line_book=total_line_book, spread_book=spread_book
    )
    game_dates = pd.concat(
        [
            history[["GAME_ID", "GAME_DATE"]],
            df_merged[["GAME_ID", "GAME_DATE"]].astype({"GAME_ID": str}),
        ],
        ignore_index=True,
    )
    crews = build_official_crews(df_refs, game_dates, df_referees_scheduled)
    estimates = compute_official_tendencies(
        crews,
        history,
        specs=specs,
        max_history_days=max_history_days,
        include_same_season_variants=include_same_season_variants,
    )
    crew_features = aggregate_crew_features(
        estimates, specs, include_same_season_variants
    )

    feature_columns = [c for c in crew_features.columns if c != "GAME_ID"]
    out = df_merged.drop(columns=feature_columns, errors="ignore").copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out = out.merge(crew_features, on="GAME_ID", how="left")

    without_crew = out[feature_columns[0]].isna()
    completed = out["GAME_ID"].isin(set(history["GAME_ID"]))
    missing_completed = out.loc[without_crew & completed, "GAME_ID"]
    missing_scheduled = out.loc[without_crew & ~completed, "GAME_ID"]
    n_completed = int(completed.sum())
    missing_share = len(missing_completed) / n_completed if n_completed else 0.0
    if missing_share > max_missing_crew_share:
        raise ValueError(
            f"{len(missing_completed)} of {n_completed} completed games "
            f"({missing_share:.1%}) have no referee crew, above the allowed "
            f"{max_missing_crew_share:.1%}. The referee data is incomplete or "
            f"behind. Examples: {sorted(missing_completed)[:10]}"
        )
    if not missing_scheduled.empty:
        warnings.warn(
            f"{len(missing_scheduled)} game(s) without a referee assignment get no "
            f"REF_CREW_* features: {sorted(missing_scheduled)[:10]}",
            stacklevel=2,
        )
    print(
        f"Referee tendency features attached to {(~without_crew).mean():.1%} of "
        f"{len(out)} games ({len(missing_completed)} completed games without crew)"
    )
    return out


# ---------------------------------------------------------------------------
# Combinations (run after style-matchup features exist)
# ---------------------------------------------------------------------------


def add_referee_interaction_features(
    df: pd.DataFrame, spread_book: str | None = None
) -> pd.DataFrame:
    """Crew tendency x game context. Skips any combination whose inputs are absent.

    * FTA tendency x expected total free throws: a whistle-heavy crew matters
      more for teams that live at the line.
    * FTA / pace tendency x |spread|: close games bring late fouling; large
      spreads bring variance (spread track).
    * FTA tendency x expected home free-throw-rate edge: a heavy whistle
      amplifies whichever side draws fouls (spread track).
    """
    out = df.copy()

    def numeric(column: str) -> pd.Series | None:
        return (
            pd.to_numeric(out[column], errors="coerce")
            if column in out.columns
            else None
        )

    fta = numeric("REF_CREW_FTA_TENDENCY_BEFORE")
    poss = numeric("REF_CREW_POSS_TENDENCY_BEFORE")
    expected_total_fta = numeric("STYLE_EXPECTED_TOTAL_FTA_BEFORE")
    handicap = numeric(spread_col(spread_book))
    rate_home = numeric("STYLE_EXPECTED_FTA_RATE_HOME_BEFORE")
    rate_away = numeric("STYLE_EXPECTED_FTA_RATE_AWAY_BEFORE")
    new_columns: dict[str, pd.Series] = {}

    if fta is not None and expected_total_fta is not None:
        new_columns[FTA_X_EXPECTED_TOTAL_FTA_FEATURE] = fta * expected_total_fta
    if handicap is not None:
        if fta is not None:
            new_columns[FTA_X_ABS_SPREAD_FEATURE] = fta * handicap.abs()
        if poss is not None:
            new_columns[POSS_X_ABS_SPREAD_FEATURE] = poss * handicap.abs()
    if fta is not None and rate_home is not None and rate_away is not None:
        new_columns[FTA_X_HOME_FTA_RATE_EDGE_FEATURE] = fta * (rate_home - rate_away)

    if new_columns:
        out = out.drop(columns=list(new_columns), errors="ignore")
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
    return out
