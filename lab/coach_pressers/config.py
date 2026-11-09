"""Run configuration and file/S3 layout of the coach presser archive."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lab.coach_pressers.teams import TEAMS
from lab.game_reports.season_reports import DEFAULT_PHASES, PHASES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = PROJECT_ROOT / "data" / "coach_pressers"
PACKAGE_DIR = Path(__file__).resolve().parent
CHANNELS_CSV = PACKAGE_DIR / "channels.csv"
OVERRIDES_CSV = PACKAGE_DIR / "coaches_overrides.csv"
S3_PREFIX = "train_data/coach_pressers/v1"
SCHEMA_VERSION = 1


def season_label(season_year: int) -> str:
    return f"{season_year}-{(season_year + 1) % 100:02d}"


@dataclass(frozen=True)
class PresserConfig:
    season_year: int
    local_root: Path = LOCAL_ROOT
    bucket: str = ""
    prefix: str = S3_PREFIX
    phases: tuple[str, ...] = DEFAULT_PHASES
    teams: tuple[str, ...] = ()
    language: str = "en"
    js_runtime: str = "node"
    # Pacing, in seconds.
    youtube_interval: float = 1.0
    wikipedia_interval: float = 2.0
    download_sleep: tuple[float, float] = (5.0, 15.0)
    pause_every: int = 200
    pause_seconds: float = 600.0
    # LLM tie-break (step 7).
    use_llm: bool = True
    llm_model: str = "gpt-6-luna"
    # gpt-6-luna accepts none, low, medium, high, xhigh, max (not minimal).
    llm_reasoning: str = "none"
    llm_budget_usd: float = 5.0
    llm_price_in_per_m: float = 0.10
    llm_price_out_per_m: float = 0.50
    # Step 6b: judge every window upload of a game without a candidate.
    gap_fill: bool = True
    # Archive (step 9).
    captions: bool = True
    audio: bool = True
    keep_local: bool = False
    channels_csv: Path = CHANNELS_CSV
    overrides_csv: Path = OVERRIDES_CSV
    extra: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.season_year < 2019:
            raise ValueError("The archive starts with the 2019-20 season")
        if not self.phases or set(self.phases) - set(PHASES.values()):
            raise ValueError(f"Unknown phases: {self.phases}")
        if set(self.teams) - set(TEAMS):
            raise ValueError(
                f"Unknown team codes: {sorted(set(self.teams) - set(TEAMS))}"
            )
        if not self.prefix.strip("/"):
            raise ValueError("S3 prefix must not be empty")
        if self.js_runtime not in {"node", "deno"}:
            raise ValueError("js_runtime must be node or deno")
        low, high = self.download_sleep
        if not (0 <= low <= high and math.isfinite(high)):
            raise ValueError("download_sleep must be 0 <= min <= max")
        if self.llm_budget_usd < 0:
            raise ValueError("llm_budget_usd cannot be negative")

    # ------------------------------------------------------------ local layout
    @property
    def season(self) -> str:
        return season_label(self.season_year)

    @property
    def root(self) -> Path:
        return Path(self.local_root)

    @property
    def directory(self) -> Path:
        return self.root / "seasons" / f"season={self.season}"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def coaches_path(self) -> Path:
        return self.directory / "coaches.json"

    @property
    def candidates_path(self) -> Path:
        return self.directory / "candidates.json"

    @property
    def reports_dir(self) -> Path:
        return self.directory / "reports"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def channels_dir(self) -> Path:
        return self.root / "channels"

    @property
    def llm_cache_path(self) -> Path:
        return self.root / "llm_decisions.jsonl"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def local_object(self, kind: str, name: str) -> Path:
        """Local copy of an archived object (audio/, captions/, info/)."""
        return self.directory / kind / name

    # --------------------------------------------------------------- S3 layout
    @property
    def s3_root(self) -> str:
        return self.prefix.strip("/")

    @property
    def s3_season_prefix(self) -> str:
        return f"{self.s3_root}/season={self.season}"

    def object_key(self, kind: str, name: str) -> str:
        return f"{self.s3_season_prefix}/{kind}/{name}"

    def season_file_key(self, name: str) -> str:
        return f"{self.s3_season_prefix}/{name}"

    def shared_key(self, relative: str) -> str:
        return f"{self.s3_root}/{relative}"


def object_stem(game_id: str, team: str, video_id: str) -> str:
    return f"{game_id}_{team}_{video_id}"
