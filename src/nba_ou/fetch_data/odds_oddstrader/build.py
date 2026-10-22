"""Raw OddsTrader archive -> ``lh_*`` Parquet tables in S3.

Everything between the raw bundle and the table rows is the SBR ingest path:
``parse_bundle`` yields the SBR scraper's ``ScrapedGame`` and
``line_history_aiven.ingest.build_frames`` resolves ``game_id`` against
``nba_games``, applies the two load-time repairs and the SMALLINT encodings,
and counts every drop by reason. Only the destination differs: Parquet in S3
instead of ``COPY`` into Aiven.

Deterministic, so a rebuild reproduces the tables:

* ``book_id`` is the OddsTrader ``paid`` (also kept as ``source_book_id``), so
  ids never depend on the order books were first seen.
* ``team_home``/``team_away`` in ``lh_game`` are the OddsTrader tricodes, the
  same form SBR's ``lh_game`` holds where the schedule feed supplied it.
* When two events resolve to the same game (a relisted or moved game), the one
  with more ticks is kept and the other is counted as
  ``duplicate_event_for_game``.

Games ``nba_games`` does not carry are dropped and counted, as for SBR: the
preseason, and every game before 2014-15, where the table starts. Those stay in
the raw archive only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, date, datetime

import pandas as pd

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import ScrapedGame
from nba_ou.postgre_db.line_history_aiven import ingest as ingest_mod
from nba_ou.postgre_db.line_history_aiven import schema as schema_mod
from nba_ou.postgre_db.line_history_aiven.transform import (
    GAME_DIM_COLUMNS,
    OUTPUT_COLUMNS,
)

from .books import book_for_paid
from .parse import parse_bundle, teams
from .store import EVENTS_NAME, RAW_PREFIX, OddsTraderStore, line_table_name

#: Parser/build version, recorded in the manifest.
BUILD_VERSION = 1

DOWNLOAD_THREADS = 16


def raw_game_keys_by_day(existing: Iterable[str]) -> dict[date, list[str]]:
    by_day: dict[date, list[str]] = defaultdict(list)
    for key in existing:
        if not key.startswith(f"{RAW_PREFIX}/") or key.endswith(f"/{EVENTS_NAME}"):
            continue
        by_day[date.fromisoformat(key.rsplit("/", 2)[-2])].append(key)
    return {day: sorted(keys) for day, keys in sorted(by_day.items())}


def _download(store: OddsTraderStore, keys: list[str]) -> list[dict]:
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        return list(pool.map(store.get_raw, keys))


def _drop_duplicate_events(
    games: list[ScrapedGame],
    game_index: pd.DataFrame,
    stats: ingest_mod.IngestStats,
) -> list[ScrapedGame]:
    """Keep one event per resolved game_id: the one with the most ticks."""
    lookup = ingest_mod.build_game_lookup(game_index)
    best: dict[str, ScrapedGame] = {}
    keep: list[ScrapedGame] = []
    for game in games:
        hit = ingest_mod.resolve_game_id(
            lookup,
            game_date=game.game_date,
            team_away=game.team_away,
            team_home=game.team_home,
        )
        if hit is None:
            keep.append(game)  # build_frames counts it as unmatched
            continue
        current = best.get(hit[0])
        if current is None or len(game.ticks) > len(current.ticks):
            best[hit[0]] = game
    duplicates = len(games) - len(keep) - len(best)
    stats.drop("duplicate_event_for_game", duplicates)
    return keep + list(best.values())


def build_tables(
    store: OddsTraderStore,
    *,
    games_df: pd.DataFrame,
    season_years: Iterable[int] | None = None,
    existing: set[str] | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> dict:
    """Rebuild ``line_history/`` from ``raw/``; returns the manifest."""
    existing = store.list_raw_keys() if existing is None else existing
    by_day = raw_game_keys_by_day(existing)
    wanted = None if season_years is None else {int(s) for s in season_years}
    if wanted is not None:
        # The bubble's October 2020 dates belong to 2019-20 in nba_games, so
        # neighbouring date-seasons are read too and rows filtered by season.
        reach = wanted | {s + 1 for s in wanted}
        by_day = {d: k for d, k in by_day.items() if season_year_for_date(d) in reach}

    game_index = ingest_mod.build_game_index(games_df)
    market_ids = schema_mod.market_ids()

    rows_by_season: dict[int, list[pd.DataFrame]] = defaultdict(list)
    dims: list[pd.DataFrame] = []
    stats_by_season: dict[int, ingest_mod.IngestStats] = {}
    books: dict[int, tuple[str, str]] = {}
    abbrs: dict[int, tuple[str, str]] = {}
    unreadable: list[str] = []
    table_info: dict[str, dict] = {}
    meta_rows: list[dict] = []
    loaded_at = datetime.now(UTC)

    def write(name: str, frame: pd.DataFrame) -> None:
        if dry_run:
            table_info[name] = {"rows": len(frame), "bytes": None, "key": None}
            return
        key, size = store.put_table(name, frame)
        table_info[name] = {"rows": len(frame), "bytes": size, "key": key}
        if progress:
            print(
                f"  wrote {name}: {len(frame):,} rows, {size / 1e6:.1f} MB", flush=True
            )

    def flush(season: int) -> None:
        """Write one season's fact table and free its rows."""
        parts = rows_by_season.pop(season)
        if wanted is not None and season not in wanted:
            return
        frame = _typed_lines(pd.concat(parts, ignore_index=True))
        write(line_table_name(season), frame)
        stats = stats_by_season.get(season, ingest_mod.IngestStats())
        meta_rows.append(
            {
                "season_year": season,
                "timezone": "UTC",
                "confidence": "high",
                # Counted per date-season, so the bubble's October 2020 rows
                # are counted under 2020 while loaded under 2019.
                "source_rows": stats.source_ticks,
                "loaded_rows": len(frame),
                "dropped_rows": json.dumps(stats.dropped, sort_keys=True),
                "repaired_rows": json.dumps(stats.repaired, sort_keys=True),
                "loaded_at": loaded_at,
            }
        )

    groups: dict[int, list[date]] = defaultdict(list)
    for day in by_day:
        groups[season_year_for_date(day)].append(day)

    for date_season, days in sorted(groups.items()):
        keys = [key for day in days for key in by_day[day]]
        stats = stats_by_season.setdefault(date_season, ingest_mod.IngestStats())
        games: list[ScrapedGame] = []
        for key, bundle in zip(keys, _download(store, keys), strict=True):
            try:
                games.append(parse_bundle(bundle))
            except Exception as exc:  # noqa: BLE001 -- count, report, keep going
                unreadable.append(f"{key}: {exc}")
                continue
            away, home = teams(bundle["event"])
            abbrs[int(bundle["event"]["eid"])] = (
                str((home.get("source") or {}).get("abbr") or ""),
                str((away.get("source") or {}).get("abbr") or ""),
            )
            for entry in bundle.get("history") or []:
                books[int(entry["paid"])] = book_for_paid(entry["paid"])

        book_ids = {slug: paid for paid, (slug, _) in books.items()}
        games = _drop_duplicate_events(games, game_index, stats)
        rows, dim, stats = ingest_mod.build_frames(
            games,
            game_index=game_index,
            book_ids=book_ids,
            market_ids=market_ids,
            stats=stats,
        )
        del games
        if not dim.empty:
            codes = dim["event_id"].map(lambda e: abbrs.get(int(e), ("", "")))
            dim["team_home"] = [
                c[0] or h for c, h in zip(codes, dim["team_home"], strict=True)
            ]
            dim["team_away"] = [
                c[1] or a for c, a in zip(codes, dim["team_away"], strict=True)
            ]
            dims.append(dim)
        for season, part in rows.groupby("season_year"):
            rows_by_season[int(season)].append(part)
        if progress:
            print(
                f"  date-season {date_season}: {stats.scraped_games} games parsed, "
                f"{stats.matched_games} matched, {len(rows):,} rows",
                flush=True,
            )
        # Later date-seasons only add rows to their own season or, for the
        # bubble, to the one before; anything older is final.
        for season in [s for s in rows_by_season if s < date_season]:
            flush(season)

    for season in sorted(rows_by_season):
        flush(season)

    game_dim = (
        pd.concat(dims, ignore_index=True).drop_duplicates("game_id")
        if dims
        else pd.DataFrame(columns=GAME_DIM_COLUMNS)
    )
    if wanted is not None:
        game_dim = game_dim[game_dim["season_year"].isin(wanted)]

    write(
        "lh_book",
        pd.DataFrame(
            [
                {"book_id": paid, "slug": slug, "name": name, "source_book_id": paid}
                for paid, (slug, name) in sorted(books.items())
            ],
            columns=["book_id", "slug", "name", "source_book_id"],
        ).astype({"book_id": "int16", "source_book_id": "int16"}),
    )
    write(
        "lh_market",
        pd.DataFrame(schema_mod.MARKETS, columns=["market_id", "code"]).astype(
            {"market_id": "int16"}
        ),
    )
    write("lh_game", _typed_game_dim(game_dim))
    write(
        "lh_load_meta",
        pd.DataFrame(
            meta_rows,
            columns=[
                "season_year", "timezone", "confidence", "source_rows",
                "loaded_rows", "dropped_rows", "repaired_rows", "loaded_at",
            ],
        ).astype({"season_year": "int16"}),
    )  # fmt: skip

    manifest = {
        "source": "oddstrader",
        "build_version": BUILD_VERSION,
        "generated_at": loaded_at.isoformat(),
        "seasons": sorted(int(r["season_year"]) for r in meta_rows),
        "raw_games": sum(len(k) for k in by_day.values()),
        "unreadable_bundles": unreadable,
        "stats_by_date_season": {
            str(s): {
                k: v
                for k, v in asdict(st).items()
                if k not in ("unmatched_games", "tipoff_disagreements", "retimed_games")
            }
            | {"unmatched_games": len(st.unmatched_games)}
            for s, st in sorted(stats_by_season.items())
        },
        "tables": table_info,
    }
    if not dry_run:
        store.put_manifest(manifest)
    return manifest


def _typed_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """The Aiven column types: SMALLINT ids and encodings, INTEGER minutes."""
    out = rows[OUTPUT_COLUMNS].copy()
    out["game_id"] = out["game_id"].astype(str)
    out["line_ts"] = pd.to_datetime(out["line_ts"], utc=True)
    out["mins_to_tip"] = out["mins_to_tip"].astype("int32")
    out["is_pregame"] = out["is_pregame"].astype(bool)
    out["is_opener"] = out["is_opener"].astype(bool)
    for column in ["season_year", "market_id", "book_id"]:
        out[column] = out[column].astype("int16")
    for column in ["left_line", "left_price", "right_line", "right_price"]:
        out[column] = out[column].astype("Int16")
    return out.sort_values(["game_id", "market_id", "book_id", "line_ts"]).reset_index(
        drop=True
    )


def _typed_game_dim(dim: pd.DataFrame) -> pd.DataFrame:
    out = dim[GAME_DIM_COLUMNS].copy()
    out["game_id"] = out["game_id"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out["season_year"] = out["season_year"].astype("int16")
    out["tipoff_utc"] = pd.to_datetime(out["tipoff_utc"], utc=True)
    out["event_id"] = out["event_id"].astype("int32")
    return out.sort_values("game_id").reset_index(drop=True)
