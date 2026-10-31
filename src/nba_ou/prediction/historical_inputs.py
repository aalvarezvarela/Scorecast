"""Prediction inputs read from the retrain partitions, without building history.

Only intermediate snapshot rows are prediction inputs, including for closing
models at T-240. A played game's closing row is NEVER used to fill a feature,
with one exception: its consensus opener (SBR's, BetMGM's first quote), which
the snapshot does not carry and which was published before the snapshot when
the snapshot already shows BetMGM's opener (``with_stored_openers``).
"""

from __future__ import annotations

import re
from collections import OrderedDict

import pandas as pd
from nba_ou.config.odds_columns import (
    apply_odds_prefix,
    moneyline_col,
    spread_col,
    total_line_col,
)
from nba_ou.config.yahoo_features import YAHOO_RAW_SOURCES
from nba_ou.create_training_data.select_intermediate_columns import (
    LEAKY_BEFORE_COLUMNS,
    SCORING_ONLY_PREFIXES,
    TARGET_COLUMNS,
)
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_high_value_features_for_team_points,
)
from nba_ou.data_processing.merged_home_away_data.odds_feature_engeneer import (
    engineer_odds_features,
)
from nba_ou.data_processing.odds.canonical_markets import add_canonical_market_columns
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds
from nba_ou.prediction.report_requirement import reports_filed
from nba_ou.utils.general_utils import get_season_year_from_date


class StoredInputsUnavailable(ValueError):
    """A saved row or feature is unavailable; never build or substitute a final value."""


def without_outcomes(frame):
    scoring = [column for column in frame if column.startswith(SCORING_ONLY_PREFIXES)]
    return frame.drop(
        columns=[*TARGET_COLUMNS, "IS_OVERTIME", *scoring], errors="ignore"
    ).copy()


#: The consensus opener's market stems in the merged odds frame, and the
#: snapshot market whose BetMGM opener shows it was published.
OPENER_MARKETS: dict[str, str] = {"total": "TOT", "spread": "SPR", "ml": "ML"}


#: This game's opener quote, not the earlier games' rolling ``..._BEFORE`` ones.
_OPENER_QUOTE = re.compile(
    rf"ODDS_({'|'.join(OPENER_MARKETS)})_consensus_opener_"
    r"(line|price)_(over|under|away|home)"
)


def opener_columns(closing_rows: pd.DataFrame) -> list[str]:
    """The stored closing rows' consensus opener quote columns (line, prices)."""
    return [column for column in closing_rows if _OPENER_QUOTE.fullmatch(column)]


def with_stored_openers(
    odds: pd.DataFrame, frame: pd.DataFrame, openers: pd.DataFrame
) -> pd.DataFrame:
    """``odds`` with the consensus opener of ``openers`` where it was published.

    SBR's consensus opener is BetMGM's first quote, line and prices. The
    snapshot keeps BetMGM's opener line only (normalized), so the closing
    spread consensus, its book count and cross-book spread are otherwise
    computed without it: on 2026-27 that moved the closing spread model by 0.4
    points and flipped 25 of 80 sides. ``openers`` (indexed by GAME_ID, the
    stored closing columns, ``ODDS_`` prefixed) fill a market only for games
    whose snapshot already shows BetMGM's opener: published before the
    snapshot. On 2025-26 and 2026-27 it always was, 11 hours or more before tip.
    """
    out = odds.copy()
    ids = frame.GAME_ID.astype(str)
    missing = pd.Series(float("nan"), index=frame.index)
    for market, prefix in OPENER_MARKETS.items():
        opened = pd.to_numeric(
            frame.get(f"ODDS_SNAP_{prefix}_BETMGM_OPENER_LINE", missing),
            errors="coerce",
        ).notna()
        for stored in opener_columns(openers):
            if not stored.startswith(f"ODDS_{market}_"):
                continue
            column = stored.removeprefix("ODDS_")
            value = ids.map(openers[stored])
            current = out[column] if column in out else missing
            out[column] = value.where(opened, current)
    return out


def closing_from_snapshot(snapshot, openers=None):
    """Run only the row-wise closing market formulas on saved snapshot quotes.

    Team/player/report/referee history already belongs to this snapshot. Yahoo
    and untimestamped consensus percentages stay unavailable. Closing's raw
    handicap and canonical implied-margin signs follow their existing helpers.
    ``openers``: the stored consensus opener per game (``with_stored_openers``);
    None leaves it missing but for the snapshot's opening total.
    """
    frame = without_outcomes(snapshot).drop(
        columns=list(LEAKY_BEFORE_COLUMNS), errors="ignore"
    )
    books = sorted(
        {
            match.group(1).lower()
            for column in frame
            for match in [
                re.fullmatch(r"ODDS_SNAP_(?:TOT|SPR|ML)_(.+)_HAS_QUOTE", column)
            ]
            if match and match.group(1) != "CONSENSUS"
        }
        | {"consensus_opener"}
    )
    raw = pd.DataFrame({"game_id": frame.GAME_ID.astype(str)}, index=frame.index)
    for book in books:
        for market, prefix in [("total", "TOT"), ("spread", "SPR"), ("ml", "ML")]:
            stem = f"ODDS_SNAP_{prefix}_{book.upper()}"
            quoted = (
                pd.to_numeric(
                    frame.get(f"{stem}_HAS_QUOTE", pd.Series(0, index=frame.index)),
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
            )

            def value(suffix, stem=stem, quoted=quoted):
                return pd.to_numeric(
                    frame.get(
                        f"{stem}_{suffix}", pd.Series(float("nan"), index=frame.index)
                    ),
                    errors="coerce",
                ).where(quoted)

            left, right = ("over", "under") if market == "total" else ("away", "home")
            raw[f"{market}_{book}_price_{left}"] = value("PRICE_LEFT")
            raw[f"{market}_{book}_price_{right}"] = value("PRICE_RIGHT")
            if market != "ml":
                raw[f"{market}_{book}_line_{left}"] = value("RAW_LINE")
                raw[f"{market}_{book}_line_{right}"] = value("RAW_LINE") * (
                    -1 if market == "spread" else 1
                )
    odds = merge_yahoo_sportsbook_odds(pd.DataFrame(), raw).set_index("game_id")
    odds = odds.reindex(frame.GAME_ID.astype(str)).set_axis(frame.index)
    if openers is not None:
        odds = with_stored_openers(odds, frame, openers)
    current = {}
    for book in books:
        current[total_line_col(book)] = odds[f"total_{book}_line_over"]
        for side in ("home", "away"):
            current[f"{spread_col(book)}_TEAM_{side.upper()}"] = odds[
                f"spread_{book}_line_{side}"
            ]
            current[f"{moneyline_col(book)}_TEAM_{side.upper()}"] = odds[
                f"ml_{book}_price_{side}"
            ]
    # The consensus opener line is already explicitly safe in the snapshot
    # partition. Its prices have no timestamped observation: keep them missing.
    opener = total_line_col("consensus_opener")
    if opener in frame:
        current[opener] = frame[opener]
    current.update({column: odds[column] for column in odds if column != "game_id"})
    unavailable = [
        *YAHOO_RAW_SOURCES,
        *(
            f"{market}_consensus_pct_{side}"
            for market, sides in [
                ("total", ("over", "under")),
                ("spread", ("away", "home")),
                ("moneyline", ("away", "home")),
            ]
            for side in sides
        ),
    ]
    current.update({column: float("nan") for column in unavailable})
    markets = pd.DataFrame(current, index=frame.index)
    # Preserve only earlier-game odds from the snapshot. Regenerate ALL of
    # this game's market-derived columns, even when an equal name exists.
    drop = [
        column
        for column in frame
        if column.startswith("ODDS_")
        and "_BEFORE" not in column
        and not column.startswith(("ODDS_SNAP_", "ODDS_LINE_HIST_"))
    ]
    frame = frame.drop(columns=drop)
    frame = pd.concat([frame, markets], axis=1)
    frame = add_canonical_market_columns(frame)
    frame = engineer_odds_features(frame, books=books)
    frame = add_high_value_features_for_team_points(frame)
    return apply_odds_prefix(frame)


class StoredPredictionInputs:
    """Pin current-season manifests and retain at most two decoded horizons."""

    def __init__(self, stores, *, tipoff_loader):
        self.store = stores.get("intermediate_line")
        self.closing_store = stores.get("closing_line")
        self.tipoff_loader = tipoff_loader
        self.manifests = None
        self.frames = OrderedDict()
        self.tipoffs = {}
        self.openers = {}

    def stored_openers(self, season: int) -> pd.DataFrame | None:
        """The season's stored consensus openers by GAME_ID; None without a
        closing partition (a run with no closing slot)."""
        if self.closing_store is None or not hasattr(self.closing_store, "read"):
            return None
        if season not in self.openers:
            rows = self.closing_store.read(season)
            columns = opener_columns(rows)
            self.openers[season] = rows.assign(
                GAME_ID=rows.GAME_ID.astype(str)
            ).set_index("GAME_ID")[columns]
        return self.openers[season]

    def read(self, game_date, horizon, ids, *, closing=False, required_features=()):
        if self.store is None or not hasattr(self.store, "read"):
            raise StoredInputsUnavailable("No readable intermediate season partitions")
        season = int(get_season_year_from_date(game_date))
        if self.manifests is None:
            self.manifests = self.store.manifests()
        manifest = self.manifests.get(season)
        if manifest is None:
            raise StoredInputsUnavailable(
                f"No stored intermediate partition for {season}"
            )
        key = (season, horizon)
        if key not in self.frames:
            self.frames[key] = self.store.read(
                season, horizon=horizon, manifest=manifest
            )
            while len(self.frames) > 2:
                self.frames.popitem(last=False)
        source = self.frames[key]
        dates = pd.to_datetime(source.GAME_DATE)
        frame = source.loc[
            dates.eq(pd.Timestamp(game_date))
            & source.GAME_ID.astype(str).isin(ids)
            & source.TIME_TO_MATCH_MIN.eq(horizon)
        ].copy()
        if frame.GAME_ID.duplicated().any():
            raise StoredInputsUnavailable("Duplicate stored game/snapshot rows")
        if frame.empty:
            raise StoredInputsUnavailable(
                "No stored rows for the requested games/timepoint"
            )
        if season not in self.tipoffs:
            games = self.tipoff_loader([season])
            self.tipoffs[season] = (
                games.assign(game_id=games.game_id.astype(str))
                .set_index("game_id")
                .tipoff_utc
            )
        tips = pd.to_datetime(
            frame.GAME_ID.astype(str).map(self.tipoffs[season]), utc=True
        )
        frame["SNAPSHOT_TS_UTC"] = tips - pd.Timedelta(minutes=horizon)
        frame = frame.loc[frame.SNAPSHOT_TS_UTC.notna()].copy()
        if frame.empty:
            raise StoredInputsUnavailable(
                "Stored snapshot has no known tipoff timestamp"
            )
        if closing:
            try:
                filed = reports_filed(frame)
            except ValueError as exc:
                raise StoredInputsUnavailable(str(exc)) from exc
            frame = frame.loc[filed].copy()
            if frame.empty:
                raise ValueError(
                    "No historical closing input has both injury reports filed"
                )
            frame = closing_from_snapshot(frame, openers=self.stored_openers(season))
        frame = without_outcomes(frame)
        missing = set(required_features) - set(frame.columns)
        if missing:
            raise StoredInputsUnavailable(
                f"Stored snapshot lacks {len(missing)} required features: {sorted(missing)[:12]}"
            )
        frame["SNAPSHOT_PASSED"] = True
        frame["GAME_TIME"] = frame.GAME_ID.astype(str).map(self.tipoffs[season])
        frame.attrs["input_source"] = {
            "source": "stored_snapshot",
            "dataset": "intermediate_line",
            "horizon_minutes": horizon,
            "partition_checksum": manifest.checksum,
            "partition_file": manifest.file,
            "games": sorted(frame.GAME_ID.astype(str)),
        }
        return frame
