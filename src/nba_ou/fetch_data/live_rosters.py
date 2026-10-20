"""Official team rosters, archived as immutable, timestamped S3 snapshots.

No roster is written to PostgreSQL. A small per-day/team manifest allows a
direct read on failure and selection of a snapshot observed before a cutoff.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging

import pandas as pd
from botocore.config import Config
from nba_api.stats.endpoints.commonteamroster import CommonTeamRoster

from nba_ou.config.settings import SETTINGS
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import make_s3_client

log = logging.getLogger(__name__)
PREFIX = "train_data/live_inputs/v1/rosters"
PLAYER_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "POSITION"]
SNAPSHOT_COLUMNS = ["TEAM_ID", "SEASON", *PLAYER_COLUMNS, "FETCHED_AT_UTC", "SOURCE"]


def empty_snapshots():
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def season_for_game(game) -> str:
    year = int(get_season_year_from_date(pd.Timestamp(game.GAME_DATE_EST)))
    return f"{year}-{str(year + 1)[-2:]}"


def normalize_players(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={"PLAYER": "PLAYER_NAME"}).copy()
    if not {"PLAYER_ID", "PLAYER_NAME"}.issubset(out):
        raise ValueError("Roster is missing player IDs or names")
    ids = pd.to_numeric(out.PLAYER_ID, errors="coerce")
    if out.empty or ids.isna().any() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("Empty roster or invalid player IDs")
    out["PLAYER_ID"] = ids.astype("int64").astype(str)
    out["PLAYER_NAME"] = out.PLAYER_NAME.astype("string").str.strip()
    if out.PLAYER_NAME.isna().any() or out.PLAYER_NAME.eq("").any():
        raise ValueError("Roster contains unnamed players")
    if out.PLAYER_ID.duplicated().any():
        raise ValueError("Duplicate roster player IDs")
    out = out.reindex(columns=PLAYER_COLUMNS).astype("string")
    return out.sort_values("PLAYER_ID").reset_index(drop=True)


class RosterStore:
    def __init__(self, client, bucket):
        self.client, self.bucket = client, bucket
        self._etags = {}

    @staticmethod
    def base(day, season, team):
        return f"{PREFIX}/game_date={day}/season={season}/team_id={team}"

    def get(self, key):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"]
            self._etags[key] = response.get("ETag")
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {
                "NoSuchKey",
                "404",
                "NotFound",
            }:
                self._etags[key] = None
                return None
            raise
        try:
            return body.read()
        finally:
            body.close()

    def entries(self, day, season, team):
        payload = self.get(f"{self.base(day, season, team)}/index.json")
        entries = [] if payload is None else json.loads(payload)
        if not isinstance(entries, list):
            raise ValueError("Invalid roster manifest")
        for entry in entries:
            stamp = pd.Timestamp(entry["fetched_at_utc"])
            if (
                stamp.tzinfo is None
                or stamp.tz_convert("America/New_York").date().isoformat() != day
            ):
                raise ValueError("Invalid roster snapshot date")
            label = stamp.tz_convert("UTC").strftime("%Y-%m-%dT%H-%M-%S.%fZ")
            if (
                entry["key"]
                != f"{self.base(day, season, team)}/fetched_at={label}/players.parquet"
            ):
                raise ValueError("Invalid roster snapshot key")
        return entries

    def read(self, entries, season, team, as_of):
        frames = []
        for entry in entries:
            stamp = pd.Timestamp(entry["fetched_at_utc"])
            if stamp > as_of:
                continue
            payload = self.get(entry["key"])
            if (
                payload is None
                or hashlib.sha256(payload).hexdigest() != entry["parquet_sha256"]
            ):
                raise ValueError("Missing or corrupt roster snapshot")
            players = normalize_players(pd.read_parquet(io.BytesIO(payload)))
            frames.append(
                players.assign(
                    TEAM_ID=str(team), SEASON=season, FETCHED_AT_UTC=stamp, SOURCE="s3"
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else empty_snapshots()

    def save(self, day, season, team, players, fetched_at, entries):
        digest = hashlib.sha256(
            players.to_json(orient="split", index=False).encode()
        ).hexdigest()
        if entries and entries[-1]["content_sha256"] == digest:
            return
        # Older/replayed observations cannot replace a more recent manifest.
        if entries and pd.Timestamp(entries[-1]["fetched_at_utc"]) >= fetched_at:
            return
        stamp = fetched_at.strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        key = f"{self.base(day, season, team)}/fetched_at={stamp}/players.parquet"
        buffer = io.BytesIO()
        players.to_parquet(buffer, index=False, compression="zstd")
        payload = buffer.getvalue()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        entry = dict(
            key=key,
            fetched_at_utc=fetched_at.isoformat(),
            content_sha256=digest,
            parquet_sha256=hashlib.sha256(payload).hexdigest(),
        )
        manifest = f"{self.base(day, season, team)}/index.json"
        etag = self._etags.get(manifest)
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=manifest,
                Body=json.dumps([*entries, entry]).encode(),
                ContentType="application/json",
                **condition,
            )
            self._etags[manifest] = response.get("ETag")
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "412",
                "409",
            }:
                log.warning(
                    "Concurrent roster archive update for team %s; retained existing manifest",
                    team,
                )
                return
            raise


def _store():
    if not SETTINGS.s3_bucket:
        raise ValueError("S3 bucket is not configured")
    return RosterStore(
        make_s3_client(
            profile=SETTINGS.s3_aws_profile,
            region=SETTINGS.s3_aws_region,
            config=Config(
                connect_timeout=3, read_timeout=5, retries={"total_max_attempts": 2}
            ),
        ),
        SETTINGS.s3_bucket,
    )


def load_live_rosters(scheduled_games: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    """Fetch once per team/season; retain eligible same-day snapshots for replay.

    A past as_of never triggers a live request. Endpoint/S3 failures are isolated
    per team, leaving that team's existing historical inference available.
    """
    if scheduled_games.empty:
        return empty_snapshots()
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("Roster as_of must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    try:
        store = _store()
    except Exception as exc:
        log.warning("Roster S3 cache unavailable: %s", exc)
        store = None
    requests = {
        (
            pd.Timestamp(game.GAME_DATE_EST).date().isoformat(),
            season_for_game(game),
            str(team),
        )
        for game in scheduled_games.itertuples()
        for team in (game.HOME_TEAM_ID, game.VISITOR_TEAM_ID)
    }
    frames = []
    for day, season, team in sorted(requests):
        live = None
        # Explicit historical cutoffs must only read archived observations.
        if (
            as_of is None
            and day == now.tz_convert("America/New_York").date().isoformat()
        ):
            try:
                response = CommonTeamRoster(
                    team_id=int(team),
                    season=season,
                    league_id_nullable="00",
                    timeout=10,
                )
                raw = response.common_team_roster.get_data_frame()
                if "TeamID" not in raw or not raw.TeamID.astype(str).eq(team).all():
                    raise ValueError("Roster team mismatch")
                if (
                    "SEASON" not in raw
                    or not raw.SEASON.astype(str).isin([season, season[:4]]).all()
                ):
                    raise ValueError("Roster season mismatch")
                live = normalize_players(raw)
                fetched_at = pd.Timestamp.now(tz="UTC")
                frames.append(
                    live.assign(
                        TEAM_ID=team,
                        SEASON=season,
                        FETCHED_AT_UTC=fetched_at,
                        SOURCE="nba",
                    )
                )
            except Exception as exc:
                log.warning("Live roster failed for team %s: %s", team, exc)
        try:
            if store is not None:
                entries = store.entries(day, season, team)
                cached = store.read(entries, season, team, cutoff)
                if not cached.empty:
                    frames.append(cached)
                if live is not None:
                    store.save(day, season, team, live, fetched_at, entries)
        except Exception as exc:
            log.warning("Roster cache failed for team %s: %s", team, exc)
        if live is None and not any(
            not f.empty and f.TEAM_ID.eq(team).any() and f.SEASON.eq(season).any()
            for f in frames
        ):
            log.warning(
                "No eligible official roster for team %s; using historical inference",
                team,
            )
    return (
        pd.concat(frames, ignore_index=True).reindex(columns=SNAPSHOT_COLUMNS)
        if frames
        else empty_snapshots()
    )


def roster_overrides(scheduled_games, snapshots, *, as_of=None):
    """Select the newest same-day roster per team-game observed by ``as_of``.

    Returns ``(game_id, team_id) -> {player_id: name}``. A missing key requests
    historical inference; it never means an empty or healthy team.
    """
    result = {}
    if snapshots is None or snapshots.empty:
        return result
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    stamps = pd.to_datetime(snapshots.FETCHED_AT_UTC, utc=True)
    for game in scheduled_games.itertuples():
        game_id = str(game.GAME_ID)
        for team in (str(game.HOME_TEAM_ID), str(game.VISITOR_TEAM_ID)):
            eligible = snapshots.loc[
                snapshots.TEAM_ID.eq(team)
                & snapshots.SEASON.eq(season_for_game(game))
                & stamps.le(cutoff)
                & stamps.dt.tz_convert("America/New_York").dt.date.eq(
                    pd.Timestamp(game.GAME_DATE_EST).date()
                )
            ]
            if not eligible.empty:
                latest = pd.to_datetime(eligible.FETCHED_AT_UTC, utc=True).max()
                players = eligible.loc[
                    pd.to_datetime(eligible.FETCHED_AT_UTC, utc=True).eq(latest)
                ]
                result[(game_id, team)] = dict(
                    zip(players.PLAYER_ID.astype(str), players.PLAYER_NAME, strict=True)
                )
    return result
