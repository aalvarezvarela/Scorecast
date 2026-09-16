"""End-to-end ingest: archived PDFs -> spans in Aiven.

Season at a time, because a span may only be built from a *contiguous* run of
reports. Loading a season in two disjoint halves would merge across the gap and
invent a span covering reports nobody actually read.

The phases are separable and each is independently re-runnable:

    A  parse    PDF bytes -> tidy rows                     (parse.py)
    B  resolve  attach game_id / team_id / player_id       (resolve.py)
    C  spans    observations -> validity intervals         (spans.py)
    D  load     dimensions and facts into Aiven            (load.py)

Two kinds of observation are *derived* here rather than read off a single row,
because the report expresses them by omission:

* **Removal.** A player who was listed and then drops off a *filed* team's list
  while the game is still upcoming is no longer designated. Measured in ~2% of
  games, usually 1-3 players. Without an explicit NULL-status observation his
  last span would run to tipoff and keep reading "Doubtful".
* **Filing.** A team of a listed game that is not ``NOT YET SUBMITTED`` has
  filed, even if it lists nobody. And a game can vanish from the report once
  both teams have filed empty lists (seen in the 2026 East Finals). Recording
  only "not submitted" would leave every filing span open until tip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import psycopg
from tqdm import tqdm

from nba_ou.fetch_data.injury_reports.archive import manifest as mf

from . import load as loader
from . import resolve as R
from .parse import LegacyLayoutError, ParsedReport, category_code, parse_report
from .schema import ERAS, NOT_LISTED_CATEGORY, create_season_partition
from .spans import build_filing_spans, build_spans

_GAME_KEYS = ["game_date", "team_away", "team_home"]


@dataclass
class IngestSummary:
    season: str
    reports_read: int = 0
    reports_legacy: int = 0
    reports_failed: int = 0
    observations: int = 0
    removals: int = 0
    spans: int = 0
    filing_spans: int = 0
    tipoffs_from_reports: int = 0
    #: Real NBA games (present in nba_games) that no source could place.
    games_without_tipoff: int = 0
    #: Listed "games" that are not NBA games at all: Summer League, postponed
    #: dates, and "if necessary" playoff games that were never played.
    listed_non_games: int = 0
    #: Rows deleted per table when the season was reloaded with ``replace``.
    replaced: dict[str, int] = field(default_factory=dict)
    resolution: R.ResolutionReport = field(default_factory=R.ResolutionReport)

    def describe(self) -> str:
        return (
            f"{self.season}: {self.reports_read} reports "
            f"({self.reports_legacy} unreadable legacy skipped, {self.reports_failed} failed) -> "
            f"{self.observations:,} observations (+{self.removals:,} removals) -> "
            f"{self.spans:,} spans (+{self.filing_spans:,} filing); "
            f"{self.resolution.summary()}"
            + (
                f"; {self.tipoffs_from_reports} tipoffs taken from reports"
                if self.tipoffs_from_reports
                else ""
            )
            + (
                f"; {self.games_without_tipoff} NBA games dropped for want of a tipoff"
                if self.games_without_tipoff
                else ""
            )
            + (
                f"; {self.listed_non_games} listed non-games skipped "
                "(Summer League / postponed / if-necessary)"
                if self.listed_non_games
                else ""
            )
            + (
                "; replaced existing rows: "
                + ", ".join(f"{t}={n:,}" for t, n in self.replaced.items())
                if self.replaced
                else ""
            )
        )


@dataclass(frozen=True)
class _Read:
    report: ParsedReport
    era: int
    season_year: int


def read_reports(
    manifest: pd.DataFrame, storage, *, limit: int | None = None, quiet: bool = False
) -> tuple[list[_Read], int, int]:
    """Parse every downloaded report in ``manifest``, oldest first.

    Reports missing from storage are skipped silently -- the manifest records
    what was *discovered*, which is a superset of what was downloaded (reports
    whose header sits outside the era tolerance are rejected and never uploaded).

    Each report is placed at its *publication* instant (``mf.published_utc``:
    the PDF header when it differs from the filename, else the derived time).
    If two files claim the same instant, only the later-labelled one is kept, so
    ``observed_at`` stays unique.
    """
    rows = manifest.loc[manifest["nba_available"] == "true"].copy()
    rows["published_at"] = mf.published_utc(rows)
    # On a tie a downloaded file beats one that never reached storage.
    rows["is_stored"] = rows.get("download_status", pd.Series(index=rows.index)).eq(
        mf.DOWNLOAD_STORED
    )
    rows = rows.sort_values(["published_at", "is_stored", "report_datetime_utc"])
    rows = rows.drop_duplicates(subset=["published_at"], keep="last")
    if limit:
        rows = rows.head(limit)

    parsed: list[_Read] = []
    legacy = failed = 0
    bar = tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="parse",
        unit="pdf",
        disable=True if quiet else None,
    )
    try:
        for row in bar:
            key = getattr(row, "s3_key", None)
            if not key or pd.isna(key):
                continue
            data = storage.get(key)
            if data is None:
                continue
            observed_at = pd.Timestamp(row.published_at).to_pydatetime()
            try:
                report = parse_report(data, observed_at)
            except LegacyLayoutError:
                legacy += 1
                continue
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failed += 1
                tqdm.write(f"  parse failed {key}: {exc}")
                continue
            parsed.append(
                _Read(
                    report,
                    ERAS.get(str(getattr(row, "source_era", "")), 0),
                    int(row.season_year),
                )
            )
    finally:
        bar.close()
    return parsed, legacy, failed


def _stack(parsed: list[_Read], attr: str) -> pd.DataFrame:
    frames = [
        getattr(r.report, attr).assign(observed_at=r.report.observed_at)
        for r in parsed
        if not getattr(r.report, attr).empty
    ]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["observed_at"] = pd.to_datetime(out["observed_at"], utc=True)
    return out


def build_report_frame(parsed: list[_Read]) -> pd.DataFrame:
    """Every parsed report -- including ones listing no players at all.

    A report whose games are all ``NOT YET SUBMITTED`` still proves the feed was
    live at that instant, which is exactly what coverage has to know.
    """
    return pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                [r.report.observed_at for r in parsed], utc=True
            ),
            "season_year": [r.season_year for r in parsed],
            "era": [r.era for r in parsed],
            "n_rows": [r.report.n_rows for r in parsed],
            "parse_ok": True,
        }
    ).drop_duplicates(subset=["observed_at"])


def build_removal_observations(
    observations: pd.DataFrame,
    listed: pd.DataFrame,
    not_filed: pd.DataFrame,
    report_times: pd.Series,
) -> pd.DataFrame:
    """NULL-status observations for players who dropped off a filed list.

    ``listed`` is ``(game_id, observed_at, tipoff_utc)`` for every report that
    lists the game; ``not_filed`` is ``(game_id, raw_team, observed_at)``.
    """
    if observations.empty or listed.empty:
        return observations.iloc[0:0]

    carry = [
        "game_id",
        "player_id",
        "team_id",
        "nba_team_id",
        "season_year",
        "tipoff_utc",
        "raw_name",
        "raw_team",
    ]
    players = (
        observations.sort_values("observed_at")
        .drop_duplicates(subset=["game_id", "player_id"], keep="first")[
            carry + ["observed_at"]
        ]
        .rename(columns={"observed_at": "first_seen"})
    )

    times = listed[["game_id", "observed_at"]].drop_duplicates()

    # A game that stops being listed while still upcoming has had both lists
    # filed empty; its first unlisted report removes whoever was left.
    last = listed.groupby("game_id").agg(
        last_listed=("observed_at", "max"), tipoff_utc=("tipoff_utc", "first")
    )
    all_times = pd.Series(sorted(pd.to_datetime(report_times, utc=True).unique()))
    vanished = []
    for game_id, row in last.iterrows():
        after = all_times[(all_times > row.last_listed) & (all_times < row.tipoff_utc)]
        if len(after):
            vanished.append((game_id, after.iloc[0]))
    if vanished:
        times = pd.concat(
            [times, pd.DataFrame(vanished, columns=["game_id", "observed_at"])],
            ignore_index=True,
        )

    cand = players.merge(times, on="game_id")
    cand = cand.loc[
        (cand["observed_at"] > cand["first_seen"])
        & (cand["observed_at"] < cand["tipoff_utc"])
    ]
    # Absent because his team has not filed yet is unknown, not removed.
    if not not_filed.empty:
        cand = cand.merge(
            not_filed[["game_id", "raw_team", "observed_at"]].assign(_nys=True),
            how="left",
            on=["game_id", "raw_team", "observed_at"],
        )
        cand = cand.loc[cand["_nys"].isna()].drop(columns=["_nys"])
    present = observations[["game_id", "player_id", "observed_at"]].assign(_here=True)
    cand = cand.merge(present, how="left", on=["game_id", "player_id", "observed_at"])
    removed = cand.loc[cand["_here"].isna()].drop(columns=["_here", "first_seen"])

    label = NOT_LISTED_CATEGORY[1]
    return removed.assign(
        status_id=pd.NA, reason_category=label, reason_detail=""
    ).reset_index(drop=True)


def build_filing_observations(
    listed: pd.DataFrame,
    not_filed: pd.DataFrame,
    report_times: pd.Series,
    tricode_to_team_id: dict[str, int],
) -> pd.DataFrame:
    """One ``submitted`` observation per team per report that lists its game."""
    columns = [
        "game_id",
        "team_id",
        "season_year",
        "observed_at",
        "tipoff_utc",
        "submitted",
    ]
    if listed.empty:
        return pd.DataFrame(columns=columns)

    sides = pd.concat(
        [
            listed.assign(tricode=listed["team_home"]),
            listed.assign(tricode=listed["team_away"]),
        ],
        ignore_index=True,
    )
    sides["team_id"] = sides["tricode"].map(tricode_to_team_id)
    sides = sides.dropna(subset=["team_id"])

    nys = set()
    if not not_filed.empty:
        nys = set(
            zip(
                not_filed["game_id"],
                not_filed["team_id"],
                not_filed["observed_at"],
                strict=True,
            )
        )
    sides["submitted"] = [
        (g, t, o) not in nys
        for g, t, o in zip(
            sides["game_id"], sides["team_id"], sides["observed_at"], strict=True
        )
    ]

    # Vanished-while-upcoming games: both teams filed empty lists.
    last = listed.groupby("game_id").agg(
        last_listed=("observed_at", "max"),
        tipoff_utc=("tipoff_utc", "first"),
        season_year=("season_year", "first"),
        team_home=("team_home", "first"),
        team_away=("team_away", "first"),
    )
    all_times = pd.Series(sorted(pd.to_datetime(report_times, utc=True).unique()))
    extra = []
    for game_id, row in last.iterrows():
        after = all_times[(all_times > row.last_listed) & (all_times < row.tipoff_utc)]
        if not len(after):
            continue
        for tricode in (row.team_home, row.team_away):
            team_id = tricode_to_team_id.get(tricode)
            if team_id is not None:
                extra.append(
                    (
                        game_id,
                        team_id,
                        row.season_year,
                        after.iloc[0],
                        row.tipoff_utc,
                        True,
                    )
                )
    out = sides[columns]
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, columns=columns)], ignore_index=True)
    return out.reset_index(drop=True)


def ingest_season(
    season: str,
    *,
    manifest: pd.DataFrame,
    storage,
    aiven: psycopg.Connection | None,
    source: psycopg.Connection,
    limit: int | None = None,
    dry_run: bool = False,
    quiet: bool = False,
    replace: bool = False,
) -> IngestSummary:
    """Run all four phases for one season.

    ``replace`` deletes the season's existing spans, filing spans, reports and
    unresolved-name counts before writing. It is required whenever reports are
    added to a season that is already loaded: spans are inserted with
    ``ON CONFLICT DO NOTHING``, so a reload over old rows would keep each old
    span's ``valid_to`` and leave it overlapping the new, shorter spans. The
    delete runs only after every report has been parsed and resolved.
    """
    summary = IngestSummary(season=season)
    parsed, summary.reports_legacy, summary.reports_failed = read_reports(
        manifest, storage, limit=limit, quiet=quiet
    )
    summary.reports_read = len(parsed)
    if not parsed:
        return summary

    players = _stack(parsed, "players")
    listings = _stack(parsed, "games")
    not_submitted = _stack(parsed, "not_submitted")
    reports = build_report_frame(parsed)
    if listings.empty:
        return summary

    seasons = sorted(
        {(d.year - 1 if d.month < 8 else d.year) for d in listings["game_date"]}
    )

    # --- dimensions ---------------------------------------------------------
    teams = R.load_team_dimension(source)
    games = R.load_game_dimension(seasons, source)
    unscheduled = R.load_unscheduled_games(source, seasons, set(games["game_id"]))
    from_reports = R.tipoffs_from_reports(unscheduled, listings)
    summary.tipoffs_from_reports = len(from_reports)
    if not from_reports.empty:
        games = pd.concat([games, from_reports], ignore_index=True)
    games["tipoff_utc"] = pd.to_datetime(games["tipoff_utc"], utc=True)
    rosters = R.load_rosters(source, seasons)

    if dry_run:
        team_map = dict(zip(teams["report_name"], teams["nba_team_id"], strict=True))
    else:
        team_map = loader.upsert_teams(aiven, teams)
        for season_year in seasons:
            create_season_partition(aiven, season_year)
        loader.upsert_games(aiven, games)
    tricode_to_team_id = {
        tricode: team_map[name]
        for tricode, name in zip(teams["tricode"], teams["report_name"], strict=True)
        if name in team_map
    }

    # --- which games each report lists ----------------------------------------
    listing_report = R.ResolutionReport()
    listed = R.resolve_games(listings, games, listing_report)
    unmatched = {(u["game_date"], u["matchup"]) for u in listing_report.unmatched_games}
    # An unmatched listing is a real gap only if nba_games knows the game;
    # otherwise the report was describing something that is not an NBA game.
    placed = set(from_reports["game_id"]) if not from_reports.empty else set()
    still_missing = unscheduled.loc[~unscheduled["game_id"].isin(placed)]
    known = {
        (d, f"{a}@{h}")
        for d, a, h in zip(
            still_missing["game_date"],
            still_missing["team_away"],
            still_missing["team_home"],
            strict=True,
        )
    }
    summary.games_without_tipoff = len(unmatched & known)
    summary.listed_non_games = len(unmatched - known)
    not_filed = pd.DataFrame()
    if not not_submitted.empty:
        not_filed = R.resolve_games(not_submitted, games, R.ResolutionReport())
        not_filed["team_id"] = not_filed["raw_team"].map(team_map)

    # --- player observations --------------------------------------------------
    observations = pd.DataFrame()
    if not players.empty:
        players = players.merge(
            teams[["nba_team_id", "report_name"]],
            how="left",
            left_on="raw_team",
            right_on="report_name",
        )
        observations = R.resolve_games(players, games, summary.resolution)
        observations = R.resolve_players(observations, rosters, summary.resolution)
    summary.observations = len(observations)

    if not observations.empty:
        observations["team_id"] = observations["raw_team"].map(team_map)
        removals = build_removal_observations(
            observations, listed, not_filed, reports["observed_at"]
        )
        summary.removals = len(removals)
        observations = pd.concat([observations, removals], ignore_index=True)

    # --- reports, reasons, ids ------------------------------------------------
    if replace and not dry_run:
        # Only the season being reloaded. Listings can reach into a neighbouring
        # season, whose rows this run does not rebuild and must not delete.
        summary.replaced = loader.clear_season(aiven, int(season.split("-")[0]))
    report_map = (
        dict(zip(reports["observed_at"], reports["observed_at"], strict=True))
        if dry_run
        else loader.upsert_reports(aiven, reports)
    )

    spans = pd.DataFrame()
    if not observations.empty:
        observations["report_id"] = observations["observed_at"].map(report_map)
        pairs = {
            (category_code(c), c, d)
            for c, d in zip(
                observations["reason_category"],
                observations["reason_detail"],
                strict=True,
            )
        }
        if dry_run:
            observations["reason_id"] = (
                observations["reason_category"].map(category_code)
                + "|"
                + observations["reason_detail"]
            )
        else:
            reason_map = loader.upsert_reasons(aiven, sorted(pairs))
            observations["reason_id"] = [
                reason_map.get((category_code(c), d))
                for c, d in zip(
                    observations["reason_category"],
                    observations["reason_detail"],
                    strict=True,
                )
            ]
        spans = build_spans(observations)
    summary.spans = len(spans)

    filings = build_filing_observations(
        listed, not_filed, reports["observed_at"], tricode_to_team_id
    )
    filing_spans = build_filing_spans(filings)
    summary.filing_spans = len(filing_spans)

    if not dry_run:
        loader.insert_spans(aiven, spans)
        loader.insert_filing_spans(aiven, filing_spans)
        resolved = observations.loc[observations["status_id"].notna()]
        loader.record_aliases(aiven, resolved)
        loader.record_unresolved(aiven, summary.resolution.unresolved)
    return summary
