"""Turn scraped line-history CSVs into rows for the Aiven store.

Encodes the Phase 0 findings (``docs/odds/line_history_phase0_findings.md``):

* Timestamps are naive ``Europe/Madrid``. 2019-20 and 2020-21 showed no clean
  DST step and are marked low confidence.
* Unmatched games are preseason and are dropped.
* SBR records in-play ticks that are indistinguishable from pre-game rows
  except by ``mins_to_tip``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MADRID = "Europe/Madrid"

# Per-season timezone, with the Phase 0 confidence attached. The two COVID-era
# seasons fit UTC/London about as well as Madrid, so they are held back unless
# explicitly requested rather than silently loaded under a guess.
SEASON_TIMEZONES: dict[int, tuple[str, str]] = {
    2019: (MADRID, "low"),
    2020: (MADRID, "low"),
    2021: (MADRID, "high"),
    2022: (MADRID, "high"),
    2023: (MADRID, "high"),
    2024: (MADRID, "high"),
    2025: (MADRID, "high"),
}

# SBR writes this where a book had no price up ("off the board").
NO_PRICE_SENTINEL = -100000
SMALLINT_MIN, SMALLINT_MAX = -32768, 32767

READ_COLUMNS = [
    "game_date",
    "season_year",
    "event_id",
    "team_home",
    "team_away",
    "bookmaker_slug",
    "market",
    "row_kind",
    "timestamp",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]

OUTPUT_COLUMNS = [
    "game_id",
    "season_year",
    "market_id",
    "book_id",
    "line_ts",
    "mins_to_tip",
    "is_pregame",
    "is_opener",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]


@dataclass
class TransformStats:
    source_rows: int = 0
    output_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    repaired: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    def repair(self, reason: str, count: int) -> None:
        if count:
            self.repaired[reason] = self.repaired.get(reason, 0) + int(count)


def season_timezone(season_year: int) -> tuple[str, str]:
    return SEASON_TIMEZONES.get(int(season_year), (MADRID, "unknown"))


def read_season_csvs(root: Path, season_label: str) -> pd.DataFrame:
    paths = sorted((root / season_label / "line_history").glob("*.csv"))
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, usecols=READ_COLUMNS))
        except (pd.errors.EmptyDataError, ValueError):
            continue

    # Empty frames would make concat warn and muddy the inferred dtypes.
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=READ_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def repair_spread_price_bleed(
    df: pd.DataFrame, spread_market_id: int | None
) -> tuple[pd.DataFrame, int]:
    """Move prices that the scraper parsed as spreads into the price columns.

    On a pick'em the SBR cell holds only a price ("-110") with no spread number,
    and the scraper's ``([+-]\\d+(?:\\.\\d+)?)`` pattern matches that price as
    the line. Such rows are recognisable because a genuine spread is mirrored --
    ``left_line == -right_line`` -- while these carry complementary *price*
    pairs (-110/-110, -115/-105).

    The value is relabelled as the price it demonstrably is; the spread is left
    NULL rather than inferred as 0, since the source never actually said so.
    """
    if spread_market_id is None or df.empty:
        return df, 0

    left = pd.to_numeric(df["left_line"], errors="coerce")
    right = pd.to_numeric(df["right_line"], errors="coerce")
    bleed = (
        (df["market_id"] == spread_market_id)
        & left.notna()
        & right.notna()
        & (left != -right)
    )
    if not bleed.any():
        return df, 0

    out = df.copy()
    out.loc[bleed, "left_price"] = pd.to_numeric(
        out.loc[bleed, "left_price"], errors="coerce"
    ).fillna(left[bleed])
    out.loc[bleed, "right_price"] = pd.to_numeric(
        out.loc[bleed, "right_price"], errors="coerce"
    ).fillna(right[bleed])
    out.loc[bleed, ["left_line", "right_line"]] = None
    return out, int(bleed.sum())


# Generous bounds on a *pre-game* line, used only to null values that cannot be
# real (a dropped decimal turns 228.5 into 2285). In-play rows are exempt: a
# live spread legitimately blows out past 30 during a rout.
PREGAME_LINE_BOUNDS: dict[str, tuple[float, float]] = {
    "totals": (150.0, 300.0),
    "point_spread": (-30.0, 30.0),
}


def null_implausible_pregame_lines(
    df: pd.DataFrame, market_ids: dict[str, int]
) -> tuple[pd.DataFrame, int]:
    """Drop pre-game line values outside what the market can produce.

    Prices and the row itself are kept -- only the impossible line is cleared,
    so nothing else about the timepoint is lost.
    """
    if df.empty:
        return df, 0

    out = df.copy()
    affected = pd.Series(False, index=out.index)

    for code, (low, high) in PREGAME_LINE_BOUNDS.items():
        market_id = market_ids.get(code)
        if market_id is None:
            continue
        for column in ["left_line", "right_line"]:
            values = pd.to_numeric(out[column], errors="coerce")
            bad = (
                (out["market_id"] == market_id)
                & out["is_pregame"]
                & values.notna()
                & ((values < low) | (values > high))
            )
            if bad.any():
                out.loc[bad, column] = None
                affected |= bad

    return out, int(affected.sum())


def _encode_line(values: pd.Series) -> pd.Series:
    """Half-points doubled, as a nullable small integer."""
    numeric = pd.to_numeric(values, errors="coerce")
    doubled = (numeric * 2).round()
    out_of_range = doubled.notna() & (
        (doubled < SMALLINT_MIN) | (doubled > SMALLINT_MAX)
    )
    doubled = doubled.mask(out_of_range)
    return doubled.astype("Int64")


def _encode_price(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.mask(numeric <= NO_PRICE_SENTINEL)
    out_of_range = numeric.notna() & (
        (numeric < SMALLINT_MIN) | (numeric > SMALLINT_MAX)
    )
    numeric = numeric.mask(out_of_range)
    return numeric.round().astype("Int64")


def transform_season(
    raw: pd.DataFrame,
    *,
    season_year: int,
    games: pd.DataFrame,
    schedule: pd.DataFrame,
    book_ids: dict[str, int],
    market_ids: dict[str, int],
    normalize_team: callable,
) -> tuple[pd.DataFrame, pd.DataFrame, TransformStats]:
    """Return (fact rows, game dimension, drop stats).

    ``games`` supplies (game_date, team_home, team_away) -> game_id; ``schedule``
    supplies game_id -> tipoff_utc.
    """
    stats = TransformStats(source_rows=len(raw))
    empty = (
        pd.DataFrame(columns=OUTPUT_COLUMNS),
        pd.DataFrame(columns=GAME_DIM_COLUMNS),
        stats,
    )
    if raw.empty:
        return empty

    df = raw.copy()

    before = len(df)
    df = df.dropna(subset=["timestamp"])
    stats.drop("unparseable_timestamp", before - len(df))

    # game_id: unmatched rows are preseason, which nba_games does not carry.
    df["team_home"] = df["team_home"].map(normalize_team)
    df["team_away"] = df["team_away"].map(normalize_team)
    df = df.merge(
        games[["game_id", "game_date", "team_home", "team_away"]],
        on=["game_date", "team_home", "team_away"],
        how="left",
    )
    before = len(df)
    df = df.dropna(subset=["game_id"])
    stats.drop("preseason_or_unmatched_game", before - len(df))

    df = df.merge(schedule[["game_id", "tipoff_utc"]], on="game_id", how="left")
    before = len(df)
    df = df.dropna(subset=["tipoff_utc"])
    stats.drop("no_tipoff", before - len(df))

    if df.empty:
        return empty

    game_dim = build_game_dimension(df, schedule, season_year)

    # Localize. Rows in a DST gap or repeated hour are dropped rather than
    # guessed: mins_to_tip drives the leakage filter, so a silent 1h error there
    # is worse than losing a handful of rows on one night a season.
    tz, _confidence = season_timezone(season_year)
    localized = df["timestamp"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    before = len(df)
    df = df.assign(line_ts=localized.dt.tz_convert("UTC")).dropna(subset=["line_ts"])
    stats.drop("dst_ambiguous_or_nonexistent", before - len(df))

    df["mins_to_tip"] = (
        (df["line_ts"] - df["tipoff_utc"]).dt.total_seconds() / 60.0
    ).round()
    before = len(df)
    df = df.dropna(subset=["mins_to_tip"])
    stats.drop("no_mins_to_tip", before - len(df))

    df["is_pregame"] = df["mins_to_tip"] < 0
    df["is_opener"] = df["row_kind"].astype(str).str.lower().eq("opener")

    df["market_id"] = df["market"].map(market_ids)
    df["book_id"] = (
        df["bookmaker_slug"].astype(str).str.strip().str.lower().map(book_ids)
    )
    before = len(df)
    df = df.dropna(subset=["market_id", "book_id"])
    stats.drop("unknown_market_or_book", before - len(df))

    df, bled = repair_spread_price_bleed(df, market_ids.get("point_spread"))
    stats.repair("spread_price_bleed", bled)

    df, implausible = null_implausible_pregame_lines(df, market_ids)
    stats.repair("implausible_pregame_line", implausible)

    for column in ["left_line", "right_line"]:
        df[column] = _encode_line(df[column])
    for column in ["left_price", "right_price"]:
        df[column] = _encode_price(df[column])

    df["season_year"] = season_year
    df["mins_to_tip"] = df["mins_to_tip"].astype("Int64")
    df["market_id"] = df["market_id"].astype("Int64")
    df["book_id"] = df["book_id"].astype("Int64")
    df["game_id"] = df["game_id"].astype(str)

    # The scrape overlaps across days, and 2021-22 was scraped twice.
    before = len(df)
    df = df.sort_values(["game_id", "market_id", "book_id", "line_ts", "is_opener"])
    df = df.drop_duplicates(
        subset=["game_id", "market_id", "book_id", "line_ts"], keep="last"
    )
    stats.drop("duplicate_timepoint", before - len(df))

    out = df[OUTPUT_COLUMNS].reset_index(drop=True)
    stats.output_rows = len(out)

    # Keep the dimension to games that actually survived.
    game_dim = game_dim[game_dim["game_id"].isin(out["game_id"].unique())]
    return out, game_dim.reset_index(drop=True), stats


GAME_DIM_COLUMNS = [
    "game_id",
    "game_date",
    "season_year",
    "tipoff_utc",
    "event_id",
    "team_home",
    "team_away",
]


def build_game_dimension(
    matched: pd.DataFrame,
    schedule: pd.DataFrame,
    season_year: int,
) -> pd.DataFrame:
    """One row per game, for ``lh_game``.

    Takes the post-join frame so ``game_id`` and the SBR ``event_id`` are
    already on the same row -- they cannot be re-associated by date afterwards,
    since a date holds many games.
    """
    if matched.empty:
        return pd.DataFrame(columns=GAME_DIM_COLUMNS)

    dim = (
        matched.assign(
            event_id=pd.to_numeric(matched["event_id"], errors="coerce").astype("Int64")
        )
        .sort_values("game_id")
        .drop_duplicates("game_id")[["game_id", "event_id"]]
    )

    dim = dim.merge(
        schedule[["game_id", "game_date", "tipoff_utc", "team_home", "team_away"]],
        on="game_id",
        how="inner",
    )
    dim["season_year"] = season_year
    return dim[GAME_DIM_COLUMNS]
