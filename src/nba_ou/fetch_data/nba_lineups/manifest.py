"""Small, atomic, per-season resume manifests for raw lineup responses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_ou.fetch_data.injury_reports.archive.storage import Storage

COLUMNS = (
    "game_id",
    "endpoint",
    "fetched_at",
    "status",
    "bytes",
    "http_status",
    "elapsed_s",
    "source",
)

#: ``source`` of a rotation rebuilt from play-by-play rather than fetched; an
#: empty ``source`` is the NBA API itself.
REBUILT_SOURCE = "rebuilt_from_playbyplayv2"

#: ``ok`` -- archived. ``empty`` -- the NBA has nothing for this game: a
#: sub-second 5xx or a 200 without rows. ``server_timeout`` -- a slow 5xx, the
#: backend timing out on that attempt; it says nothing about the game (see
#: ``client.ServerTimeout``), so it must not be counted as a hole. ``failed`` --
#: the circuit breaker opened on it. Everything but ``ok`` is retried next run.
STATUSES = frozenset({"ok", "empty", "server_timeout", "failed"})


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Give a manifest every column, including one written before they existed.

    Manifests from before 2026-09-24 carry no ``http_status``/``elapsed_s``,
    and from before 2026-09-27 no ``source``; their rows read as unknown rather
    than failing to load.
    """
    frame = frame.copy()
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["http_status"] = pd.array(
        pd.to_numeric(frame["http_status"], errors="coerce"), dtype="Int64"
    )
    frame["elapsed_s"] = pd.to_numeric(frame["elapsed_s"], errors="coerce").astype(
        "float64"
    )
    frame["source"] = frame["source"].astype("object")
    return frame[list(COLUMNS)]


class Manifest:
    def __init__(self, root: Path, *, mirror: Storage | None = None) -> None:
        self.root = Path(root)
        self.mirror = mirror
        self._frames: dict[int, pd.DataFrame] = {}

    def path(self, season_year: int) -> Path:
        return self.root / f"season={season_year}" / "manifest.parquet"

    def load(self, season_year: int) -> pd.DataFrame:
        if season_year not in self._frames:
            path = self.path(season_year)
            if not path.exists() and self.mirror is not None:
                raw = self.mirror.get(self.mirror_key(season_year))
                if raw is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
            self._frames[season_year] = _normalise(
                pd.read_parquet(path)
                if path.exists()
                else pd.DataFrame(columns=list(COLUMNS))
            )
        return self._frames[season_year]

    @staticmethod
    def mirror_key(season_year: int) -> str:
        return f"nba_api_raw/manifest/season={season_year}/manifest.parquet"

    def sync_all(self) -> None:
        if self.mirror is None:
            return
        for season_year in self._frames:
            self.mirror.put(
                self.mirror_key(season_year),
                self.path(season_year).read_bytes(),
                {"content_type": "application/octet-stream"},
            )

    def is_ok(self, season_year: int, game_id: str, endpoint: str) -> bool:
        frame = self.load(season_year)
        rows = frame.loc[
            frame.game_id.eq(str(game_id)) & frame.endpoint.eq(endpoint), "status"
        ]
        return not rows.empty and rows.iloc[-1] == "ok"

    def record_many(
        self,
        season_year: int,
        rows: list[tuple[str, str, str, int]],
        *,
        source: str | None = None,
    ) -> None:
        """Record many results with a single parquet write.

        Importing a season settles over a thousand rows at once; recording them
        one at a time rewrites the whole manifest each time.
        """
        if not rows:
            return
        frame = self.load(season_year)
        now = pd.Timestamp.now(tz="UTC")
        incoming = _normalise(
            pd.DataFrame(
                [
                    (str(game_id), endpoint, now, status, nbytes)
                    for game_id, endpoint, status, nbytes in rows
                ],
                columns=list(COLUMNS[:5]),
            )
        )
        incoming["source"] = source
        if not frame.empty:
            superseded = pd.MultiIndex.from_frame(
                frame[["game_id", "endpoint"]]
            ).isin(list(zip(incoming.game_id, incoming.endpoint, strict=True)))
            frame = frame.loc[~superseded]
        self._frames[season_year] = (
            incoming if frame.empty else pd.concat([frame, incoming], ignore_index=True)
        )
        self._write(season_year)

    def _write(self, season_year: int) -> None:
        path = self.path(season_year)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        self._frames[season_year].to_parquet(tmp, index=False)
        tmp.replace(path)

    def record(
        self,
        season_year: int,
        game_id: str,
        endpoint: str,
        status: str,
        nbytes: int = 0,
        *,
        http_status: int | None = None,
        elapsed_s: float | None = None,
        source: str | None = None,
    ) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        frame = self.load(season_year)
        mask = frame.game_id.eq(str(game_id)) & frame.endpoint.eq(endpoint)
        row = _normalise(
            pd.DataFrame(
                [
                    (
                        str(game_id),
                        endpoint,
                        pd.Timestamp.now(tz="UTC"),
                        status,
                        nbytes,
                        http_status,
                        elapsed_s,
                        source,
                    )
                ],
                columns=list(COLUMNS),
            )
        )
        existing = frame.loc[~mask]
        self._frames[season_year] = (
            row if existing.empty else pd.concat([existing, row], ignore_index=True)
        )
        self._write(season_year)
