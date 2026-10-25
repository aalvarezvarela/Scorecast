"""Training datasets kept in S3 one season per file, updated in place.

The daily refit needs every dataset's whole history, and building it from the
database takes an hour for the intermediate dataset (12 minutes for the
closing one). But a game's row depends only on what came before it -- the
point-in-time parity of ``scripts/perf/point_in_time_parity.py`` -- so a
finished season's rows never change, and each day only adds games. So each
dataset lives in S3 as one Parquet file per season, plus a JSON manifest::

    train_data/2_5/seasons/closing_line/season=2025.json
    train_data/2_5/seasons/closing_line/season=2025.9f3c0b1a2d4e5f60.parquet

and a day's update (``update_partitions``) builds only the games since the
last cutoff, in target mode (``target_game_ids``: each row exactly the row a
full build gives), and rewrites the partitions they belong to. There is one
file per season, never one per day.

Every update rebuilds the last ``OVERLAP_DAYS`` before the previous cutoff,
or starts the day after the newest stored game when that is earlier: a game
that could not be built yet (its line history not ingested) is retried every
day until it is, and a finished season is never rebuilt day after day (which
would recompute its whole history each time). A game without line history is
not built at all, closing rows included (``ready_game_ids``): its close would
be SBR's day-page number, which during a feed freeze is a line from hours
before tip. It waits, and pulls the window back to it, for up to
``MAX_WAIT_DAYS``; after that it is built with what there is, and reported.
Rebuilt rows must come back identical, so a difference is reported: a code change
that moved values, or a database correction (late box scores, a line-history
repair), that a rebuild of the affected seasons
(``scripts/create_train_data/build_season_partitions.py rebuild``) should
carry into the older rows.

A data file is never overwritten: it is named by its checksum, and the
manifest that names it is written after it, so a write that stops halfway
leaves the season as it was. The file the manifest replaced is deleted once
the new manifest is in place -- one data file per season.

Rows are stored in a full build's order, so the concatenated seasons are the
full build's frame, row order included -- which the refit's training window
and its seeded subsampling both depend on. That order also keeps a game's
horizons together, which compresses five times better than grouping by
horizon (39 against 198 MB for 2025-26), and decoding a season to keep one
horizon takes half a second anyway.

S3 is the store. A local copy of each file (``cache_dir``) is used only when
its checksum is the manifest's, and is checked once per ``SeasonPartitions``:
a runner on a slow link downloads a finished season once, and reading 16
horizons does not hash every file 16 times.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from nba_ou.create_training_data.train_data_store import (
    parse_season_partition_key,
    season_partition_key,
    season_partitions_prefix,
)
from nba_ou.utils.s3_models import (
    list_s3_objects,
    read_s3_object_bytes,
    upload_bytes_to_s3,
)

CLOSING_LINE = "closing_line"
INTERMEDIATE_LINE = "intermediate_line"
DATE_COLUMN = "GAME_DATE"
GAME_ID_COLUMN = "GAME_ID"
SEASON_COLUMN = "SEASON_YEAR"
SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"

#: A full build's row order, per dataset type.
ROW_ORDER: dict[str, tuple[str, ...]] = {
    CLOSING_LINE: (DATE_COLUMN, GAME_ID_COLUMN),
    INTERMEDIATE_LINE: (DATE_COLUMN, GAME_ID_COLUMN, SNAPSHOT_COLUMN),
}

#: Days before the last cutoff that every update rebuilds and checks.
OVERLAP_DAYS = 3

#: Days a finished game waits for its line history before it is built without
#: it. The line-history update holds a game whose history stops long before
#: tip for 3 days (``TRUNCATED_HOLD_DAYS``) and stores it after; this outlasts
#: that hold, so a frozen SBR day-page close only gets in as a last resort.
MAX_WAIT_DAYS = 7

#: Relative tolerance of the overlap check. Not exact: twelve roster-continuity
#: columns vary in their last bit with the hash seed (PYTHONHASHSEED).
OVERLAP_RTOL = 1e-12

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "cache" / "season_partitions"
)


def cache_dir() -> Path | None:
    """The local copies' directory; ``NBA_OU_PARTITION_CACHE_DIR=""`` turns
    them off. The workflows point it outside the checkout, which
    ``actions/checkout`` cleans on every run."""
    value = os.environ.get("NBA_OU_PARTITION_CACHE_DIR")
    if value is None:
        return DEFAULT_CACHE_DIR
    return Path(value).expanduser() if value else None


def dataset_checksum(data: bytes) -> str:
    """``sha256:<first 16 hex>``: how a fit records its dataset's bytes
    (training_pipeline.data.compute_file_checksum, FitRecord.dataset_checksum)."""
    return f"sha256:{hashlib.sha256(data).hexdigest()[:16]}"


class PartitionError(RuntimeError):
    """The season partitions are missing, inconsistent or cannot take rows."""


class PartitionManifest(BaseModel):
    """The JSON naming a season's data file, written after it: the manifest is
    the season, and moving it is the only step a reader can see."""

    model_config = ConfigDict(extra="forbid")

    dataset_type: str
    schema_version: str
    season: int
    #: The cutoff (``recent_limit_to_include``) of the build these rows equal.
    limit_date: date
    n_rows: int
    n_games: int
    first_game_date: date | None
    last_game_date: date | None
    checksum: str
    #: The data file beside the manifest, named by ``checksum``.
    file: str
    #: How the rows were made: split from a full build, or an update.
    source: str
    git_commit: str | None = None
    written_at: datetime


@dataclass(frozen=True)
class PartitionedFrame:
    """Every season's rows (or one horizon of them), in a full build's order."""

    df: pd.DataFrame
    #: ``closing_line_data_2_5_20261021``: named as the builders name a file.
    build_id: str
    #: One checksum over every season's checksum.
    checksum: str
    limit_date: date
    prefix: str


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return None


def _to_date(value) -> date | None:
    return None if pd.isna(value) else pd.Timestamp(value).date()


class SeasonPartitions:
    """One dataset type's season files of one schema version."""

    def __init__(
        self,
        *,
        s3_client,
        bucket: str,
        dataset_type: str,
        schema_version: str,
        local_dir: Path | None = None,
        use_cache: bool = True,
    ):
        if dataset_type not in ROW_ORDER:
            raise ValueError(f"No season partitions for {dataset_type!r}.")
        self.s3_client = s3_client
        self.bucket = bucket
        self.dataset_type = dataset_type
        self.schema_version = schema_version
        self.prefix = season_partitions_prefix(
            dataset_type=dataset_type, schema_version=schema_version
        )
        root = local_dir if local_dir is not None else cache_dir()
        self.local_dir = root / self.prefix if (use_cache and root) else None
        #: Data files already checked against their manifest in this run:
        #: file name -> local path, or the bytes when nothing is kept locally.
        self._verified: dict[str, Path | bytes] = {}

    # -- keys and manifests -------------------------------------------------

    def _manifest_key(self, season: int) -> str:
        return season_partition_key(
            dataset_type=self.dataset_type,
            schema_version=self.schema_version,
            season=season,
            ext="json",
        )

    def _file_name(self, season: int, checksum: str) -> str:
        key = season_partition_key(
            dataset_type=self.dataset_type,
            schema_version=self.schema_version,
            season=season,
            digest=checksum.removeprefix("sha256:"),
        )
        return key.rsplit("/", 1)[-1]

    def manifests(self) -> dict[int, PartitionManifest]:
        """Every stored season's manifest, by season."""
        found = {}
        for obj in list_s3_objects(
            s3_client=self.s3_client, bucket=self.bucket, prefix=self.prefix
        ):
            parsed = parse_season_partition_key(obj["Key"])
            if parsed is None or parsed[1] != "json":
                continue
            raw = read_s3_object_bytes(
                s3_client=self.s3_client, bucket=self.bucket, key=obj["Key"]
            )
            found[parsed[0]] = PartitionManifest.model_validate_json(raw)
        return dict(sorted(found.items()))

    def limit_date(self) -> date | None:
        """The newest cutoff: the date the partitions are current to."""
        manifests = self.manifests()
        return max((m.limit_date for m in manifests.values()), default=None)

    # -- bytes --------------------------------------------------------------

    def _source(self, manifest: PartitionManifest) -> Path | bytes:
        """The season's data file, checked against its manifest once per run:
        the local copy when it is the manifest's, else downloaded (and kept)."""
        if manifest.file in self._verified:
            return self._verified[manifest.file]
        local = self.local_dir / manifest.file if self.local_dir is not None else None
        if local is not None and local.exists():
            if dataset_checksum(local.read_bytes()) == manifest.checksum:
                self._verified[manifest.file] = local
                return local
        key = self.prefix + manifest.file
        data = read_s3_object_bytes(
            s3_client=self.s3_client, bucket=self.bucket, key=key
        )
        if dataset_checksum(data) != manifest.checksum:
            raise PartitionError(
                f"s3://{self.bucket}/{key} is {dataset_checksum(data)}, and its "
                f"manifest says {manifest.checksum}: the file was changed after "
                "it was written. Rebuild the season."
            )
        stored = self._keep_local(manifest.file, data)
        self._verified[manifest.file] = stored if stored is not None else data
        return self._verified[manifest.file]

    def _keep_local(self, file: str, data: bytes) -> Path | None:
        """A local copy of a checked data file; older copies of the season go."""
        if self.local_dir is None:
            return None
        self.local_dir.mkdir(parents=True, exist_ok=True)
        path = self.local_dir / file
        temporary = path.with_suffix(".parquet.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        season = parse_season_partition_key(file)[0]
        for other in self.local_dir.glob(f"season={season:04d}*.parquet"):
            if other.name != file:
                other.unlink(missing_ok=True)
        return path

    # -- read and write -----------------------------------------------------

    def read(
        self,
        season: int,
        *,
        horizon: int | None = None,
        manifest: PartitionManifest | None = None,
    ) -> pd.DataFrame:
        """One season's rows, in a full build's order; ``horizon`` keeps one
        intermediate horizon."""
        import pyarrow.parquet as pq

        if horizon is not None and self.dataset_type != INTERMEDIATE_LINE:
            raise ValueError("Only intermediate partitions have horizons.")
        manifest = manifest or self.manifests().get(season)
        if manifest is None:
            raise PartitionError(f"No {self.dataset_type} partition for {season}.")
        stored = self._source(manifest)
        source = io.BytesIO(stored) if isinstance(stored, bytes) else str(stored)
        if horizon is None:
            df = pq.read_table(source).to_pandas()
        else:
            df = pq.read_table(
                source, filters=[(SNAPSHOT_COLUMN, "=", int(horizon))]
            ).to_pandas()
        return self._in_build_order(df)

    def _in_build_order(self, df: pd.DataFrame) -> pd.DataFrame:
        order = list(ROW_ORDER[self.dataset_type])
        # The keys are unique per row, so this order is the only one.
        return df.sort_values(order, kind="mergesort").reset_index(drop=True)

    def write(
        self, season: int, df: pd.DataFrame, *, limit_date: date | str, source: str
    ) -> PartitionManifest:
        """Store ``df`` as the season's rows: a new data file, then the
        manifest naming it, then the file it replaced is deleted."""
        if df.empty:
            raise PartitionError(f"Refusing to store an empty {season} partition.")
        seasons = set(pd.to_numeric(df[SEASON_COLUMN]).astype(int))
        if seasons != {int(season)}:
            raise PartitionError(
                f"Rows of seasons {sorted(seasons)} given for {season}."
            )
        keys = list(ROW_ORDER[self.dataset_type])
        if df.duplicated(keys).any():
            raise PartitionError(f"The {season} rows repeat a {keys} key.")
        buffer = io.BytesIO()
        self._in_build_order(df).to_parquet(buffer, index=False, compression="zstd")
        data = buffer.getvalue()
        checksum = dataset_checksum(data)
        dates = pd.to_datetime(df[DATE_COLUMN])
        manifest = PartitionManifest(
            dataset_type=self.dataset_type,
            schema_version=self.schema_version,
            season=int(season),
            limit_date=pd.Timestamp(limit_date).date(),
            n_rows=len(df),
            n_games=int(df[GAME_ID_COLUMN].nunique()),
            first_game_date=_to_date(dates.min()),
            last_game_date=_to_date(dates.max()),
            checksum=checksum,
            file=self._file_name(season, checksum),
            source=source,
            git_commit=_git_commit(),
            written_at=datetime.now(tz=UTC),
        )
        upload_bytes_to_s3(
            s3_client=self.s3_client,
            bucket=self.bucket,
            key=self.prefix + manifest.file,
            data=data,
        )
        upload_bytes_to_s3(
            s3_client=self.s3_client,
            bucket=self.bucket,
            key=self._manifest_key(season),
            data=manifest.model_dump_json(indent=2).encode(),
        )
        self._delete_unreferenced(season, keep=manifest.file)
        stored = self._keep_local(manifest.file, data)
        self._verified[manifest.file] = stored if stored is not None else data
        return manifest

    def _delete_unreferenced(self, season: int, *, keep: str) -> None:
        """Every data file of the season but the manifest's: the one it
        replaced, and any a write that stopped halfway left behind."""
        for obj in list_s3_objects(
            s3_client=self.s3_client, bucket=self.bucket, prefix=self.prefix
        ):
            name = obj["Key"].rsplit("/", 1)[-1]
            parsed = parse_season_partition_key(name)
            if parsed == (season, "parquet") and name != keep:
                self.s3_client.delete_object(Bucket=self.bucket, Key=obj["Key"])

    def touch(
        self, manifest: PartitionManifest, *, limit_date: date | str, source: str
    ) -> PartitionManifest:
        """Move a season's cutoff without changing its rows (a day without
        games): the manifest only."""
        updated = manifest.model_copy(
            update={
                "limit_date": pd.Timestamp(limit_date).date(),
                "source": source,
                "written_at": datetime.now(tz=UTC),
            }
        )
        upload_bytes_to_s3(
            s3_client=self.s3_client,
            bucket=self.bucket,
            key=self._manifest_key(manifest.season),
            data=updated.model_dump_json(indent=2).encode(),
        )
        return updated

    def load(self, *, horizon: int | None = None) -> PartitionedFrame:
        """Every season, oldest first: a full build's frame."""
        manifests = self.manifests()
        if not manifests:
            raise PartitionError(
                f"No {self.dataset_type} {self.schema_version} season partitions "
                f"under s3://{self.bucket}/{self.prefix}; create them with "
                "scripts/create_train_data/build_season_partitions.py split."
            )
        parts = [
            self.read(season, horizon=horizon, manifest=manifest)
            for season, manifest in manifests.items()
        ]
        columns = list(parts[0].columns)
        for season, part in zip(manifests, parts, strict=True):
            if list(part.columns) != columns:
                raise PartitionError(
                    f"The {season} partition's columns differ from "
                    f"{next(iter(manifests))}'s: rebuild it."
                )
        df = pd.concat(parts, ignore_index=True)
        limit = max(m.limit_date for m in manifests.values())
        digest = hashlib.sha256(
            "\n".join(f"{s}:{m.checksum}" for s, m in manifests.items()).encode()
        ).hexdigest()
        return PartitionedFrame(
            df=df,
            build_id=f"{self.dataset_type}_data_{self.schema_version}_{limit:%Y%m%d}",
            checksum=f"sha256:{digest[:16]}",
            limit_date=limit,
            prefix=self.prefix,
        )


# -- creating and updating ----------------------------------------------------


def split_into_partitions(
    df: pd.DataFrame,
    partitions: SeasonPartitions,
    *,
    limit_date: date | str,
    source: str,
) -> dict[int, PartitionManifest]:
    """Store a full build as its seasons (the starting point)."""
    seasons = sorted(pd.to_numeric(df[SEASON_COLUMN]).astype(int).unique())
    return {
        int(season): partitions.write(
            int(season),
            df.loc[pd.to_numeric(df[SEASON_COLUMN]).astype(int) == season],
            limit_date=limit_date,
            source=source,
        )
        for season in seasons
    }


@dataclass
class OverlapCheck:
    """The rebuilt rows of games that were already stored."""

    season: int
    n_games: int
    #: Column -> share of rows that differ beyond OVERLAP_RTOL.
    differing: dict[str, float] = field(default_factory=dict)
    #: Stored games the rebuild did not return; their stored rows are kept.
    missing_games: list[str] = field(default_factory=list)


@dataclass
class UpdateReport:
    dataset_type: str
    window_start: date
    limit_date: date
    n_target_games: int
    rows_built: int
    written: dict[int, PartitionManifest] = field(default_factory=dict)
    overlap: list[OverlapCheck] = field(default_factory=list)
    #: Finished games the build did not return and nothing stores: retried
    #: by the next update while they are newer than every stored game.
    not_built: list[str] = field(default_factory=list)
    #: Finished games held back without line history; retried until they are
    #: ``MAX_WAIT_DAYS`` old.
    waiting: list[str] = field(default_factory=list)
    #: Games built without line history after waiting ``MAX_WAIT_DAYS``.
    overdue: list[str] = field(default_factory=list)

    @property
    def drifted(self) -> bool:
        return any(c.differing or c.missing_games for c in self.overlap)

    def describe(self) -> str:
        lines = [
            f"{self.dataset_type}: built {self.n_target_games} game(s) from "
            f"{self.window_start} to {self.limit_date} ({self.rows_built} rows); "
            f"wrote season(s) {sorted(self.written) or 'none'}"
        ]
        if self.not_built:
            lines.append(
                f"  {len(self.not_built)} finished game(s) not built (no line "
                f"history yet?): {self.not_built[:10]}"
            )
        if self.waiting:
            lines.append(
                f"  {len(self.waiting)} finished game(s) wait for their line "
                f"history: {self.waiting[:10]}"
            )
        if self.overdue:
            lines.append(
                f"  {len(self.overdue)} game(s) built without line history, "
                f"done waiting: {self.overdue[:10]}"
            )
        for check in self.overlap:
            if not (check.differing or check.missing_games):
                status = "identical"
            else:
                status = f"{len(check.differing)} column(s) differ"
                if check.missing_games:
                    status += f", not rebuilt (kept): {check.missing_games[:5]}"
            lines.append(
                f"  overlap {check.season}: {check.n_games} stored game(s) rebuilt, "
                f"{status}"
            )
            for column, share in sorted(check.differing.items())[:10]:
                lines.append(f"    {column}: {share:.1%} of rows")
        return "\n".join(lines)


def _differing_columns(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
    """Column -> share of rows that differ; rows aligned beforehand."""
    differing = {}
    for column in old.columns:
        a, b = old[column], new[column]
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            x = a.to_numpy(dtype=float)
            y = b.to_numpy(dtype=float)
            same = np.isclose(x, y, rtol=OVERLAP_RTOL, atol=0.0, equal_nan=True)
        else:
            a_na, b_na = a.isna().to_numpy(), b.isna().to_numpy()
            equal = a.astype(object).to_numpy() == b.astype(object).to_numpy()
            same = (a_na & b_na) | (~a_na & ~b_na & equal)
        if not same.all():
            differing[column] = float((~same).mean())
    return differing


def _check_overlap(
    season: int, stored: pd.DataFrame, rebuilt: pd.DataFrame, keys: list[str]
) -> OverlapCheck:
    stored_ids = set(stored[GAME_ID_COLUMN].astype(str))
    rebuilt_ids = set(rebuilt[GAME_ID_COLUMN].astype(str))
    common = stored_ids & rebuilt_ids
    check = OverlapCheck(
        season=season,
        n_games=len(common),
        missing_games=sorted(stored_ids - rebuilt_ids),
    )
    if common and list(stored.columns) == list(rebuilt.columns):
        old = stored[stored[GAME_ID_COLUMN].astype(str).isin(common)]
        new = rebuilt[rebuilt[GAME_ID_COLUMN].astype(str).isin(common)]
        old = old.sort_values(keys, kind="mergesort").reset_index(drop=True)
        new = new.sort_values(keys, kind="mergesort").reset_index(drop=True)
        if len(old) != len(new) or not old[keys].astype(str).equals(
            new[keys].astype(str)
        ):
            check.differing["<rows>"] = 1.0
        else:
            check.differing = _differing_columns(old, new)
    return check


def _match_stored_schema(rows: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with ``like``'s columns and dtypes, as one full build gives
    them: a target build infers each dtype over its own rows only."""
    if list(rows.columns) != list(like.columns):
        missing = [c for c in like.columns if c not in rows.columns]
        extra = [c for c in rows.columns if c not in like.columns]
        if missing or extra:
            raise PartitionError(
                f"The new rows' columns differ from the stored ones (missing "
                f"{missing[:5]}, extra {extra[:5]}): the builder changed; rebuild "
                "every season."
            )
        rows = rows[list(like.columns)]
    for column in like.columns:
        if rows[column].dtype != like[column].dtype:
            try:
                rows[column] = rows[column].astype(like[column].dtype)
            except (TypeError, ValueError) as exc:
                raise PartitionError(
                    f"{column}: new rows are {rows[column].dtype}, stored ones "
                    f"{like[column].dtype}, and do not convert ({exc})."
                ) from exc
    return rows


def _stored_game_ids(
    partitions: SeasonPartitions,
    manifests: dict[int, PartitionManifest],
    *,
    since: date,
) -> set[str]:
    """The stored games of every season with a game dated on or after ``since``."""
    stored_ids: set[str] = set()
    for season, manifest in manifests.items():
        if manifest.last_game_date is not None and manifest.last_game_date >= since:
            stored_ids |= set(
                partitions.read(season, manifest=manifest)[GAME_ID_COLUMN].astype(str)
            )
    return stored_ids


def update_partitions(
    partitions: SeasonPartitions,
    *,
    limit_date: date | str,
    game_ids_between: Callable[[date, date], Sequence[str]],
    build_rows: Callable[[Sequence[str], date], pd.DataFrame],
    ready_game_ids: Callable[[Sequence[str]], Sequence[str]] | None = None,
    overlap_days: int = OVERLAP_DAYS,
    max_wait_days: int = MAX_WAIT_DAYS,
) -> UpdateReport:
    """Bring the partitions to ``limit_date``: rebuild the games from
    ``overlap_days`` before the last cutoff (or the day after the newest stored
    game, if earlier) on, and rewrite the seasons they belong to.

    ``game_ids_between(start, end)`` lists the finished games dated in
    ``[start, end]``; ``build_rows(game_ids, limit_date)`` builds their rows as
    a full build with that cutoff would. Stored rows dated before the window
    are kept as they are; those inside it are replaced by the rebuild, after
    being compared with it (``UpdateReport.overlap``). A stored game the
    rebuild does not return keeps its row, and is reported.

    ``ready_game_ids(game_ids)`` returns those whose inputs are complete (their
    line history is stored). The others are not built while they are under
    ``max_wait_days`` old (``UpdateReport.waiting``), and the window reaches
    back to the oldest of them; older ones are built anyway
    (``UpdateReport.overdue``).
    """
    limit = pd.Timestamp(limit_date).date()
    manifests = partitions.manifests()
    if not manifests:
        raise PartitionError(
            f"No {partitions.dataset_type} season partitions to update; create "
            "them with scripts/create_train_data/build_season_partitions.py split."
        )
    last_limit = max(m.limit_date for m in manifests.values())
    if limit < last_limit:
        raise PartitionError(
            f"The partitions are already at {last_limit}; refusing to move them "
            f"back to {limit}."
        )
    newest_game = max(
        (m.last_game_date for m in manifests.values() if m.last_game_date),
        default=last_limit,
    )
    start = min(
        last_limit - timedelta(days=overlap_days), newest_game + timedelta(days=1)
    )
    # A game still waiting may sit behind games stored after it, which moved
    # the window past it: reach back while it can still wait.
    wait_floor = limit - timedelta(days=max_wait_days)
    if ready_game_ids is not None and wait_floor < start:
        older = set(map(str, game_ids_between(wait_floor, start - timedelta(days=1))))
        if older - _stored_game_ids(partitions, manifests, since=wait_floor):
            start = wait_floor
    game_ids = sorted(set(map(str, game_ids_between(start, limit))))
    waiting: list[str] = []
    overdue: list[str] = []
    if ready_game_ids is not None and game_ids:
        ready = set(map(str, ready_game_ids(game_ids)))
        due = (
            set(map(str, game_ids_between(start, wait_floor)))
            if start <= wait_floor
            else set()
        )
        waiting = sorted(set(game_ids) - ready - due)
        overdue = sorted(due - ready)
        game_ids = sorted(set(game_ids) - set(waiting))
    rebuilt = build_rows(game_ids, limit) if game_ids else pd.DataFrame()
    report = UpdateReport(
        dataset_type=partitions.dataset_type,
        window_start=start,
        limit_date=limit,
        n_target_games=len(game_ids),
        rows_built=len(rebuilt),
        waiting=waiting,
        overdue=overdue,
    )
    keys = list(ROW_ORDER[partitions.dataset_type])
    newest = max(manifests)
    if not rebuilt.empty:
        rebuilt = rebuilt.loc[pd.to_datetime(rebuilt[DATE_COLUMN]).dt.date >= start]
    built = set(rebuilt[GAME_ID_COLUMN].astype(str)) if not rebuilt.empty else set()
    unbuilt = set(game_ids) - built
    if unbuilt:
        stored_ids = _stored_game_ids(partitions, manifests, since=start)
        report.not_built = sorted(unbuilt - stored_ids)
    seasons = (
        sorted(set(pd.to_numeric(rebuilt[SEASON_COLUMN]).astype(int)))
        if not rebuilt.empty
        else []
    )
    for season in seasons:
        if season in manifests:
            stored = partitions.read(season, manifest=manifests[season])
        else:
            # A new season: typed like the newest one.
            stored = partitions.read(newest, manifest=manifests[newest]).iloc[:0]
        fresh = rebuilt.loc[pd.to_numeric(rebuilt[SEASON_COLUMN]).astype(int) == season]
        fresh = _match_stored_schema(fresh.copy(), stored)
        in_window = pd.to_datetime(stored[DATE_COLUMN]).dt.date >= start
        replaced = in_window & stored[GAME_ID_COLUMN].astype(str).isin(
            set(fresh[GAME_ID_COLUMN].astype(str))
        )
        if in_window.any():
            report.overlap.append(
                _check_overlap(season, stored.loc[in_window], fresh, keys)
            )
        merged = pd.concat([stored.loc[~replaced], fresh], ignore_index=True)
        report.written[season] = partitions.write(
            season, merged, limit_date=limit, source=f"update {start}..{limit}"
        )
    if not report.written:
        # No games (off season): the rows are already current to the cutoff.
        report.written[newest] = partitions.touch(
            manifests[newest],
            limit_date=limit,
            source=f"update {start}..{limit} (no games)",
        )
    return report


# -- the builders ---------------------------------------------------------------


def _games_table(seasons: Sequence[int]) -> pd.DataFrame:
    from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
        load_games_from_db,
    )
    from nba_ou.utils.seasons import drop_exhibition_games

    games = load_games_from_db(seasons=[f"{y}-{(y + 1) % 100:02d}" for y in seasons])
    if games is None:
        raise PartitionError("Could not load the games table.")
    # No dataset has a row for a preseason or All-Star game, so none is a
    # game to build or to wait for.
    return drop_exhibition_games(games, "game_id")


def finished_game_ids(start: date, end: date) -> list[str]:
    """The games in the games table (finished ones only) dated in [start, end]."""
    from nba_ou.utils.general_utils import get_season_year_from_date

    first = int(get_season_year_from_date(pd.Timestamp(start).strftime("%Y-%m-%d")))
    last = int(get_season_year_from_date(pd.Timestamp(end).strftime("%Y-%m-%d")))
    games = _games_table(range(first, last + 1))
    dates = pd.to_datetime(games["game_date"]).dt.date
    keep = (dates >= start) & (dates <= end)
    return sorted(games.loc[keep, "game_id"].astype(str).unique())


def season_game_ids(season: int, end: date) -> list[str]:
    """The finished games of one season dated up to ``end`` -- by season, not
    by date: the 2019-20 season ran to October 2020."""
    games = _games_table([season])
    keep = (pd.to_numeric(games["season_year"]) == season) & (
        pd.to_datetime(games["game_date"]).dt.date <= end
    )
    return sorted(games.loc[keep, "game_id"].astype(str).unique())


def build_rows(
    dataset_type: str, game_ids: Sequence[str], limit_date: date
) -> pd.DataFrame:
    """The target games' rows, as a full build with this cutoff gives them.

    The builders' defaults are the ones the stored 2_5 datasets were built
    with (their manifests' ``build_args``), so target rows equal them.
    """
    limit = pd.Timestamp(limit_date).strftime("%Y-%m-%d")
    if dataset_type == CLOSING_LINE:
        from nba_ou.create_training_data.create_df_to_predict import (
            create_df_to_predict,
        )

        return create_df_to_predict(
            todays_prediction=False,
            recent_limit_to_include=limit,
            older_season_limit=None,
            target_game_ids=list(game_ids),
        )
    from nba_ou.create_training_data.create_intermediate_line_df import (
        create_intermediate_line_df,
    )

    # A game without stored line history has no snapshot rows (and the
    # builder refuses a set of targets with none): it waits for ingestion.
    with_ticks = _line_history_game_ids(game_ids)
    if not with_ticks:
        return pd.DataFrame()
    return create_intermediate_line_df(
        recent_limit_to_include=limit, target_game_ids=with_ticks
    )


def _line_history_game_ids(game_ids: Sequence[str]) -> list[str]:
    """The games among ``game_ids`` the line-history stores hold: SBR's, or
    Covers' (its fallback in every line-history read,
    ``line_history_covers.fallback``)."""
    from psycopg import sql

    from nba_ou.postgre_db.line_history_aiven.fetch import (
        SCHEMA,
        connect_line_history_db,
    )
    from nba_ou.postgre_db.line_history_covers import schema as covers_schema
    from nba_ou.postgre_db.line_history_covers import store as covers_store

    ids = list(map(str, game_ids))
    conn = connect_line_history_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT game_id FROM {}.lh_game WHERE game_id = ANY(%s)"
                ).format(sql.Identifier(SCHEMA)),
                (ids,),
            )
            found = {str(row[0]) for row in cur.fetchall()}
        if covers_schema.schema_exists(conn):
            found |= covers_store.stored_game_ids(conn, ids)
        return sorted(found)
    finally:
        conn.close()


def update_from_database(
    partitions: SeasonPartitions, *, limit_date: date | str
) -> UpdateReport:
    """``update_partitions`` with the real builders."""
    return update_partitions(
        partitions,
        limit_date=limit_date,
        game_ids_between=finished_game_ids,
        build_rows=lambda ids, limit: build_rows(partitions.dataset_type, ids, limit),
        ready_game_ids=_line_history_game_ids,
    )


def manifest_summary(manifests: dict[int, PartitionManifest]) -> str:
    return json.dumps(
        {
            s: {
                "limit_date": str(m.limit_date),
                "games": m.n_games,
                "rows": m.n_rows,
                "last_game": str(m.last_game_date),
                "checksum": m.checksum,
                "source": m.source,
            }
            for s, m in manifests.items()
        },
        indent=2,
    )
