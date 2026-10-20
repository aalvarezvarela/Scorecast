"""The pieces every schema layer is made of.

A *layer* turns a dataset of its parent version into its own version by adding
columns -- never by touching the ones already there. It is handed only the
parent's row keys plus the few columns it declares in ``requires``, and returns
only the columns it declares in ``columns``. The base frame itself never
reaches it, which is what makes "2_6 is 2_5 plus these columns" a property of
the code rather than a promise.

Inputs that are expensive to load (player box scores, the injury report state,
the lineup rating cache) live in a :class:`LayerContext`. A builder that already
holds them in memory seeds the context; a layer run over a file on disk lets the
context load them on first use. Either way a chain of layers loads each input
once.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import pandas as pd

#: The two generated datasets. Same strings as
#: ``training_pipeline.config.DatasetType`` (asserted in the tests); spelled out
#: here because ``nba_ou`` does not import ``training_pipeline``.
CLOSING_LINE = "closing_line"
INTERMEDIATE_LINE = "intermediate_line"
DATASET_TYPES = (CLOSING_LINE, INTERMEDIATE_LINE)

GAME_ID_COLUMN = "GAME_ID"
#: Minutes before tip-off; ``training_pipeline.config.SNAPSHOT_COLUMN``.
SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"

#: Columns that identify a row. A layer's output is aligned to these.
ROW_KEYS: Mapping[str, tuple[str, ...]] = {
    CLOSING_LINE: (GAME_ID_COLUMN,),
    INTERMEDIATE_LINE: (GAME_ID_COLUMN, SNAPSHOT_COLUMN),
}

_VERSION_RE = re.compile(r"^(\d+)_(\d+)$")


def parse_version(version: str) -> tuple[int, int]:
    """``"2_6"`` -> ``(2, 6)``, so versions compare numerically (2_10 > 2_9)."""
    match = _VERSION_RE.match(str(version))
    if match is None:
        raise ValueError(f"Schema versions look like '2_6'. Got {version!r}.")
    return int(match.group(1)), int(match.group(2))


def check_dataset_type(dataset_type: str) -> str:
    if dataset_type not in DATASET_TYPES:
        raise ValueError(
            f"dataset_type must be one of {DATASET_TYPES}. Got {dataset_type!r}."
        )
    return dataset_type


def game_id_keys(values: pd.Series) -> pd.Series:
    """GAME_ID as the 10-character string the database uses.

    A CSV read without an explicit dtype turns ``0022500001`` into the integer
    ``22500001`` (or ``22500001.0`` once a NaN is present), so keys read from a
    file and keys held in memory only match after this.
    """
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(10)


def team_id_keys(values: pd.Series) -> pd.Series:
    """TEAM_ID as a plain integer string, whatever dtype it arrived in."""
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


@dataclass
class LayerContext:
    """Shared, lazily loaded inputs for a chain of layers.

    Seed a field to reuse an object already in memory; leave it ``None`` and the
    first layer that asks loads it. ``cache`` holds anything else a layer wants
    to share with a later one under a name of its choosing. ``provenance``
    collects what a build read from live sources (where snapshot times came
    from, digests of the report states); file builds store it in the manifest.
    """

    df_players: pd.DataFrame | None = None
    injury_report_state: Any | None = None
    cache: dict[str, Any] = field(default_factory=dict)
    #: Optional scoring-sidecar rows: keys plus TIPOFF_UTC/SNAPSHOT_TS_UTC.
    snapshot_times: pd.DataFrame | None = None
    #: Explicitly seeded report states for these snapshots, keyed by horizon.
    snapshot_injury_states: dict[int, Any] | None = None
    #: Whether snapshot tipoffs may be read from the live line-history schedule
    #: when neither the inputs nor ``snapshot_times`` carry them. File builds
    #: turn this off so a missing scoring sidecar fails instead of silently
    #: depending on the database's current state.
    allow_schedule_tipoffs: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def get_or_load(self, name: str, loader: Callable[[], Any]) -> Any:
        if name not in self.cache:
            self.cache[name] = loader()
        return self.cache[name]

    def players(self, game_dates: pd.Series) -> pd.DataFrame:
        """Cleaned player box scores covering ``game_dates`` plus one prior season."""
        if self.df_players is None:
            from .inputs import load_player_history

            self.df_players = load_player_history(game_dates)
        return self.df_players

    def injury_report(self) -> Any:
        """``InjuryReportState``: the last report before each tip-off."""
        if self.injury_report_state is None:
            from nba_ou.data_processing.injury_status.report_state import (
                load_injury_report_state,
            )

            self.injury_report_state = load_injury_report_state()
        if "injury_report_state_sha256" not in self.provenance:
            from nba_ou.data_processing.injury_status.report_state import (
                report_state_digest,
            )

            self.provenance["injury_report_state_sha256"] = report_state_digest(
                self.injury_report_state
            )
        return self.injury_report_state

    def snapshot_cutoffs(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Validated UTC cutoffs from input timestamps, sidecar or schedule."""
        from .inputs import resolve_snapshot_cutoffs

        if self.snapshot_times is not None:
            source = "scoring_sidecar"
        elif {"TIPOFF_UTC", "SNAPSHOT_TS_UTC"} & set(inputs.columns):
            source = "embedded"
        elif self.allow_schedule_tipoffs:
            source = "line_history_schedule"
        else:
            raise ValueError(
                "Snapshot cutoffs need the base's scoring sidecar (TIPOFF_UTC / "
                "SNAPSHOT_TS_UTC). Pass --scoring-path, or --allow-schedule-tipoffs "
                "to read the live line-history schedule instead."
            )
        cutoffs = resolve_snapshot_cutoffs(inputs, self.snapshot_times)
        self.provenance["snapshot_time_source"] = source
        return cutoffs

    def snapshot_reports(self, cutoffs: pd.DataFrame) -> dict[int, Any]:
        """States at the requested cutoffs, never the closing report state."""
        if self.snapshot_injury_states is not None:
            missing = set(cutoffs["snapshot_minutes"]) - set(
                self.snapshot_injury_states
            )
            if missing:
                raise ValueError(
                    f"Missing snapshot report states for {sorted(missing)}"
                )
            states = self.snapshot_injury_states
        else:
            from nba_ou.data_processing.injury_status.report_state import (
                load_snapshot_report_states,
            )

            ordered = cutoffs.sort_values(["game_id", "snapshot_minutes"])
            digest = sha256(
                pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
            ).hexdigest()
            states = self.get_or_load(
                f"snapshot_reports:{digest}",
                lambda: load_snapshot_report_states(cutoffs),
            )
        from nba_ou.data_processing.injury_status.report_state import (
            report_state_digest,
        )

        self.provenance["snapshot_report_states_sha256"] = report_state_digest(
            {horizon: states[horizon] for horizon in set(cutoffs["snapshot_minutes"])}
        )
        return states


#: ``build(inputs, ctx, dataset_type) -> new columns``. ``inputs`` holds the row
#: keys and the layer's ``requires`` columns, indexed like the base frame; the
#: result must carry the same index and exactly the declared columns.
LayerBuild = Callable[[pd.DataFrame, LayerContext, str], pd.DataFrame]


@dataclass(frozen=True)
class SchemaLayer:
    """One schema version expressed as columns added to its parent."""

    version: str
    parent: str
    summary: str
    #: Declared outputs per dataset type. A dataset type that is absent gets no
    #: new columns from this layer -- its version still advances.
    columns: Mapping[str, tuple[str, ...]]
    #: Columns of the parent the layer reads, beyond the row keys. Read only.
    requires: tuple[str, ...]
    build: LayerBuild
    #: Columns read when the parent has them (e.g. ``TOTAL_POINTS``, absent
    #: from some same-day prediction frames). Read only.
    optional: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if parse_version(self.version) <= parse_version(self.parent):
            raise ValueError(
                f"Layer {self.version} must be newer than its parent {self.parent}."
            )
        for dataset_type in self.columns:
            check_dataset_type(dataset_type)

    def columns_for(self, dataset_type: str) -> tuple[str, ...]:
        return tuple(self.columns.get(check_dataset_type(dataset_type), ()))
