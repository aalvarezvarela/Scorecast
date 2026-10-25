"""Collect one prediction run and store it: S3 run archive + Aiven tables.

``docs/serving/prediction_storage.md`` is the design. A run (one invocation of
``predict_nba_games.py`` or ``predict_past_snapshots.py``) gives a
``PredictionRunRecorder`` its schedule, its frames and every slot's output; at
the end ``persist_prediction_run`` writes the S3 archive first (the complete
record) and then the Aiven rows, in one transaction.

Each slot's output is validated **when it is added**, against the schedule and
the frame it was predicted from. An output that breaks the store's contract is
rejected on its own, reported as that slot's failure, and the rest of the run
is still stored: one broken slot never costs the others. The contract:

* every game is scheduled and has a tip-off;
* every row is a row of the frame of the model's dataset type -- the latest
  or own snapshot of an intermediate model, as ``INPUT_MODE`` says;
* an ``own`` row is the model's own horizon;
* the line the output is relative to is that row's line;
* values are finite, and no (row, build) is predicted twice in a run.

What the recorder derives, so the scripts need not know the store's shape:

* ``games``: one row per scheduled game that was predicted.
* ``inputs``: one row per frame row a model read, keyed by
  (game, input mode, snapshot horizon). Its lines and prices come from the
  frame, not from each model's output: in ``latest`` mode the 48 intermediate
  models read the same row, and the training-support limits never touch odds.
* ``model_builds``: one row per (slot, fit) or baseline that predicted.
* ``predictions``: one row per model output, with the input and build keys.
* ``slots``: per slot, whether it predicted, was skipped or failed, and why,
  with the games its NaN budget left out (the S3 manifest).

Bet lines. Intermediate rows carry the main book's raw quote at the snapshot
(``ODDS_SNAP_{TOT,SPR}_<BOOK>_*``): the line, both prices and the quote's age.
In the line history ``left`` is OVER for totals and the AWAY side for spread
(``line_history.snapshots``), and the raw spread line is already the implied
home margin. The closing frame normalizes lines and prices in place, rounded
back to half-points, so it keeps no raw quote. A closing row takes its quote
from the game's latest intermediate row instead (``LATEST_INPUT``: the run's
own moment), when the run built one; otherwise its bet line stays null.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import (
    get_main_book,
    spread_line_home_col,
    total_line_col,
)
from nba_ou.postgre_db.predictions_aiven.schema import (
    DATASET_TYPE_CODES,
    INPUT_MODE_CLOSING,
    INPUT_MODE_CODES,
    INPUT_MODE_OWN,
    KIND_BASELINE,
    KIND_REGISTRY,
    KIND_TABPFN,
    LINE_SOURCE_CODES,
    RUN_STATUS_OK,
    RUN_STATUS_PARTIAL,
    RUN_STATUS_S3_FAILED,
    TARGET_CODES,
)
from nba_ou.prediction.line_source import LINE_SOURCE_COLUMN
from nba_ou.utils.general_utils import get_season_year_from_date

CLOSING_LINE = "closing_line"
INTERMEDIATE_LINE = "intermediate_line"

#: Set by ``load_and_predict_model_for_nba_games`` on its output: the games its
#: NaN budget (the model's ``max_na_per_row``) left out.
DROPPED_NA_ATTR = "dropped_na_games"

#: The model output each target's ``prediction_value`` is.
PREDICTION_VALUE_COLUMNS: dict[str, str] = {
    "line_error": "PRED_LINE_ERROR",
    "total_points": "PRED_TOTAL_POINTS",
    "spread_error": "PRED_SPREAD_ERROR",
}
#: The line each target's output is relative to, as the output reports it.
MODEL_LINE_COLUMNS: dict[str, str] = {
    "line_error": "TOTAL_OVER_UNDER_LINE",
    "total_points": "TOTAL_OVER_UNDER_LINE",
    "spread_error": "SPREAD_LINE_AT_PREDICTION",
}
PICK_SIDES: dict[str, int] = {"OVER": 1, "HOME": 1, "UNDER": -1, "AWAY": -1}
#: What every slot output must carry, whatever its target.
REQUIRED_OUTPUT_COLUMNS = (
    "GAME_ID",
    "PRED_PICK",
    "PREDICTED_AT_UTC",
    "MODEL_TARGET",
    "MODEL_DATASET_TYPE",
    "MODEL_HORIZON_MINUTES",
    "MODEL_SCHEMA_VERSION",
    "MODEL_VARIANT",
    "SPEC_ID",
    "FIT_ID",
)

#: The (game, input mode, snapshot horizon) key of an input row.
INPUT_KEY = ["game_id", "input_mode", "snapshot_horizon_minutes"]
#: The identity of a model build, as ``model_builds`` keys it.
BUILD_KEY = [
    "kind",
    "schema_version",
    "dataset_type",
    "target",
    "horizon_minutes",
    "variant",
    "fit_id",
]
BUILD_COLUMNS = [
    *BUILD_KEY,
    "spec_id",
    "train_date_min",
    "train_date_max",
    "training_code_tag",
    "anchor_book",
    "s3_build_prefix",
]
BET_COLUMNS = [
    "total_bet_line_x2",
    "total_price_over",
    "total_price_under",
    "total_line_age_minutes",
    "spread_bet_line_x2",
    "spread_price_home",
    "spread_price_away",
    "spread_line_age_minutes",
]
INPUT_COLUMNS = [
    *INPUT_KEY,
    "tipoff_at",
    "total_model_line",
    "spread_model_line",
    *BET_COLUMNS,
    "line_source",
]
PREDICTION_COLUMNS = [
    *INPUT_KEY,
    *BUILD_KEY,
    "slot",
    "predicted_at",
    "time_to_tipoff_seconds",
    "prediction_value",
    "pick_side",
    "na_feature_count",
    "na_feature_names",
    "input_limited_count",
]
#: An accepted output's columns: ``PREDICTION_COLUMNS`` before the time to
#: tip (which needs the input row's tip-off), plus the build's other fields.
ACCEPTED_COLUMNS = [
    *[c for c in PREDICTION_COLUMNS if c != "time_to_tipoff_seconds"],
    *BUILD_COLUMNS[len(BUILD_KEY) :],
]
GAME_COLUMNS = [
    "game_id",
    "tipoff_at",
    "home_team_id",
    "away_team_id",
    "game_date",
    "season_year",
    "season_type",
    "home_team",
    "away_team",
]


class OutputRejected(ValueError):
    """A slot output that breaks the store's contract (module docstring)."""


def game_id_int(game_ids: pd.Series) -> pd.Series:
    """NBA game ids as integers: ``"0022500001"`` -> ``22500001``."""
    return pd.to_numeric(game_ids.astype(str), errors="raise").astype("int64")


def _half_points_x2(values: pd.Series) -> pd.Series:
    """``values`` doubled where they are half-points (224.5 -> 449), else NA."""
    doubled = pd.to_numeric(values, errors="coerce") * 2
    exact = doubled.notna() & np.isclose(doubled, doubled.round(), atol=1e-6)
    return doubled.round().where(exact).astype("Int64")


def _american_odds(values: pd.Series) -> pd.Series:
    """American odds as integers; NA where missing or not a valid price."""
    odds = pd.to_numeric(values, errors="coerce").round()
    return odds.where(odds.abs() >= 100).astype("Int64")


def _optional(frame: pd.DataFrame, name: str) -> pd.Series:
    """``frame[name]``, or a column of nulls when ``frame`` lacks it."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(None, index=frame.index, dtype="object")


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def line_source_codes(frame: pd.DataFrame) -> pd.Series:
    """Each frame row's line-source code (``LINE_SOURCE_CODES``); NA when the
    frame does not say."""
    if LINE_SOURCE_COLUMN not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Int16")
    return frame[LINE_SOURCE_COLUMN].map(LINE_SOURCE_CODES).astype("Int16")


def bet_line_columns(
    frame: pd.DataFrame, dataset_type: str, book: str | None = None
) -> pd.DataFrame:
    """Per row of ``frame``, the main book's quote being bet (``inputs`` columns).

    Only an intermediate frame carries raw quotes; a closing frame's rows get
    nulls (``PredictionRunRecorder`` fills them from the latest snapshot).
    """
    book = book or get_main_book()
    out = pd.DataFrame(index=frame.index)
    if dataset_type == INTERMEDIATE_LINE:
        for market, prefix, sides in (
            ("total", "TOT", ("over", "under")),
            # Spread: left is the AWAY side, right the HOME side.
            ("spread", "SPR", ("away", "home")),
        ):
            stem = f"ODDS_SNAP_{prefix}_{book.upper()}"
            quoted = _column(frame, f"{stem}_HAS_QUOTE").fillna(0).gt(0)
            line = _column(frame, f"{stem}_RAW_LINE").where(quoted)
            out[f"{market}_bet_line_x2"] = _half_points_x2(line)
            out[f"{market}_price_{sides[0]}"] = _american_odds(
                _column(frame, f"{stem}_PRICE_LEFT").where(quoted)
            )
            out[f"{market}_price_{sides[1]}"] = _american_odds(
                _column(frame, f"{stem}_PRICE_RIGHT").where(quoted)
            )
            out[f"{market}_line_age_minutes"] = _column(
                frame, f"{stem}_LINE_AGE_MINUTES"
            ).where(quoted)
    elif dataset_type == CLOSING_LINE:
        # Normalized in place: no raw quote survives (see the module docstring).
        for name in BET_COLUMNS:
            if name.endswith("_age_minutes"):
                out[name] = np.nan
            else:
                out[name] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    else:
        raise ValueError(f"Unknown dataset type {dataset_type!r}")
    return out[BET_COLUMNS]


def games_rows(scheduled_games: pd.DataFrame, game_date: str) -> pd.DataFrame:
    """``games`` rows from the schedule (``GAME_ID``, ``GAME_TIME``, team ids).

    ``game_date`` is the schedule's own date (``GAME_DATE_EST``, the NBA's
    Eastern date) where it has one, else the run's date.
    """
    from nba_api.stats.static import teams

    def abbreviation(team_id: int) -> str:
        team = teams.find_team_name_by_id(int(team_id))
        return team["abbreviation"] if team else str(team_id)

    if scheduled_games.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    game_ids = scheduled_games["GAME_ID"].astype(str)
    tipoff = pd.to_datetime(scheduled_games["GAME_TIME"], utc=True)
    if "GAME_DATE_EST" in scheduled_games.columns:
        dates = pd.to_datetime(scheduled_games["GAME_DATE_EST"], errors="coerce")
        dates = dates.fillna(pd.Timestamp(game_date))
    else:
        dates = pd.Series(pd.Timestamp(game_date), index=scheduled_games.index)
    home = scheduled_games["HOME_TEAM_ID"].astype("int64")
    away = scheduled_games["VISITOR_TEAM_ID"].astype("int64")
    rows = pd.DataFrame(
        {
            "game_id": game_id_int(game_ids).to_numpy(),
            "tipoff_at": tipoff.to_numpy(),
            "home_team_id": home.to_numpy(),
            "away_team_id": away.to_numpy(),
            "game_date": dates.dt.date.to_numpy(),
            "season_year": [int(get_season_year_from_date(d)) for d in dates],
            # The third digit of a game id is its type: 1 pre-season,
            # 2 regular season, 4 playoffs, 5 play-in, 6 cup final.
            "season_type": game_ids.str[2].astype(int).to_numpy(),
            "home_team": [abbreviation(team) for team in home],
            "away_team": [abbreviation(team) for team in away],
        }
    )
    return rows.drop_duplicates("game_id").reset_index(drop=True)


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def git_sha() -> str | None:
    """The commit the run executes: ``GITHUB_SHA`` in CI, else ``git rev-parse``."""
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dropped_na_games(predictions: pd.DataFrame | None) -> list[str]:
    """The games a slot's NaN budget left out, as its output reports them."""
    if predictions is None:
        return []
    return [str(game) for game in predictions.attrs.get(DROPPED_NA_ATTR, [])]


@dataclass
class RunTables:
    """One run's rows for the Aiven store, keyed by natural keys, not ids."""

    run: dict
    games: pd.DataFrame
    model_builds: pd.DataFrame
    inputs: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class PredictionRunRecorder:
    """Everything one run predicted, ready for ``persist_prediction_run``.

    ``workflow`` is ``schema.WORKFLOW_DAILY``, ``WORKFLOW_PAST_SNAPSHOTS`` or
    ``WORKFLOW_TABPFN``.
    Call ``set_schedule`` and ``add_frame`` before adding outputs: an output is
    validated against them as it arrives.
    """

    workflow: int
    game_date: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_uuid: uuid.UUID = field(default_factory=uuid.uuid4)
    book: str = field(default_factory=get_main_book)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    frame_versions: dict[str, str] = field(default_factory=dict)
    scheduled_games: pd.DataFrame | None = None
    #: Accepted outputs, each already in ``PREDICTION_COLUMNS`` shape.
    outputs: list[pd.DataFrame] = field(default_factory=list)
    #: Per slot: label, status (predicted/skipped/failed), reason, rows,
    #: games left out by the NaN budget.
    slots: list[dict] = field(default_factory=list)
    #: Why the schedule or a frame cannot be stored ("schedule" or a dataset
    #: type -> error). The outputs that need it are rejected with this reason;
    #: the run itself carries on.
    setup_errors: dict[str, str] = field(default_factory=dict)
    _frame_keys: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _games: pd.DataFrame | None = field(default=None, repr=False)

    # -- run inputs ---------------------------------------------------------

    def set_schedule(self, scheduled_games: pd.DataFrame) -> None:
        self.scheduled_games = scheduled_games
        try:
            self._games = games_rows(scheduled_games, self.game_date)
        except Exception as exc:  # noqa: BLE001 -- rejects outputs, not the run
            self.setup_errors["schedule"] = f"{type(exc).__name__}: {exc}"
            self._games = pd.DataFrame(columns=GAME_COLUMNS)

    def add_frame(
        self, dataset_type: str, frame: pd.DataFrame, *, version: str | None = None
    ) -> None:
        """A frame the run's models read; ``version`` is its schema version."""
        if dataset_type not in DATASET_TYPE_CODES:
            raise ValueError(f"Unknown dataset type {dataset_type!r}")
        self.frames[dataset_type] = frame
        if version is not None:
            self.frame_versions[dataset_type] = str(version)
        try:
            self._frame_keys[dataset_type] = self._keys(dataset_type, frame)
        except Exception as exc:  # noqa: BLE001 -- rejects outputs, not the run
            self.setup_errors[dataset_type] = f"{type(exc).__name__}: {exc}"
            self._frame_keys.pop(dataset_type, None)

    @staticmethod
    def _keys(dataset_type: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Per frame row, its (game id, snapshot horizon) input key."""
        keys = pd.DataFrame(
            {"game_id": game_id_int(frame["GAME_ID"]).to_numpy()}, index=frame.index
        )
        if dataset_type == INTERMEDIATE_LINE:
            keys["snapshot_horizon_minutes"] = pd.to_numeric(
                frame["TIME_TO_MATCH_MIN"]
            ).astype("Int64")
        else:
            keys["snapshot_horizon_minutes"] = pd.Series(
                pd.NA, index=frame.index, dtype="Int64"
            )
        if keys.duplicated().any():
            raise ValueError(f"the {dataset_type} frame repeats a (game, snapshot)")
        return keys

    # -- slot outcomes ------------------------------------------------------

    def record_slot(
        self,
        label: str,
        status: str,
        *,
        reason: str | None = None,
        rows: int = 0,
        dropped_na: list[str] | None = None,
    ) -> None:
        """A slot's outcome for the manifest: predicted, skipped or failed."""
        self.slots.append(
            {
                "slot": label,
                "status": status,
                "reason": reason,
                "rows": int(rows),
                "dropped_na_games": list(dropped_na or []),
            }
        )

    def add_predictions(self, predictions: pd.DataFrame, *, label: str) -> str | None:
        """A registry slot's output (``load_registry_model_and_predict``).

        Returns why the output was rejected, or None once it is accepted. An
        empty output records the slot as skipped.
        """
        return self._add(predictions, label=label, kind=KIND_REGISTRY)

    def add_tabpfn(self, predictions: pd.DataFrame, *, label: str) -> str | None:
        """A TabPFN output (``prediction_tabpfn_client.predict_with_tabpfn``).

        Shaped like a registry slot's output, and checked the same way; only
        ``model_builds.kind`` tells them apart.
        """
        return self._add(predictions, label=label, kind=KIND_TABPFN)

    def add_baseline(
        self, name: str, predictions: pd.DataFrame, *, schema_version: str
    ) -> str | None:
        """A baseline's output: a line-error prediction on the closing rows.

        A baseline whose output depends on a history window (the historical
        average) is a new build per window: its fit id carries the window's
        last date, so each stored value keeps the calculation that produced it.
        """
        if predictions is None or predictions.empty:
            return self._add(predictions, label=name, kind=KIND_BASELINE)
        version = predictions["MODEL_VERSION"].astype(str)
        window_end = pd.to_datetime(_optional(predictions, "TRAIN_DATE_MAX"))
        fit_id = version.where(
            window_end.isna(), version + "-" + window_end.dt.strftime("%Y%m%d")
        )
        baseline = predictions.assign(
            MODEL_TARGET="line_error",
            MODEL_DATASET_TYPE=CLOSING_LINE,
            MODEL_HORIZON_MINUTES=0,
            MODEL_SCHEMA_VERSION=schema_version,
            MODEL_VARIANT=name,
            SPEC_ID=name,
            FIT_ID=fit_id,
            MODEL_BUILD_PREFIX=None,
            PREDICTED_AT_UTC=datetime.now(UTC),
        )
        return self._add(baseline, label=name, kind=KIND_BASELINE)

    def _add(self, predictions, *, label: str, kind: int) -> str | None:
        dropped = dropped_na_games(predictions)
        if predictions is None or predictions.empty:
            self.record_slot(
                label, "skipped", reason="no rows to predict", dropped_na=dropped
            )
            return None
        try:
            rows = self._normalize(predictions, kind=kind, label=label)
        except OutputRejected as exc:
            self.record_slot(label, "failed", reason=f"rejected: {exc}")
            return str(exc)
        self.outputs.append(rows)
        self.record_slot(label, "predicted", rows=len(rows), dropped_na=dropped)
        return None

    def _normalize(self, out: pd.DataFrame, *, kind: int, label: str) -> pd.DataFrame:
        """``out`` in ``PREDICTION_COLUMNS`` shape, or ``OutputRejected``."""
        if self.scheduled_games is None:
            raise ValueError("set_schedule() must come before any output")
        if "schedule" in self.setup_errors:
            raise OutputRejected(
                f"the schedule is unusable ({self.setup_errors['schedule']})"
            )
        missing = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in out.columns]
        if missing:
            raise OutputRejected(f"missing columns {missing}")
        target = out["MODEL_TARGET"].astype(str)
        unknown = sorted(set(target) - set(TARGET_CODES))
        if unknown:
            raise OutputRejected(f"unknown target(s) {unknown}")
        dataset_type = out["MODEL_DATASET_TYPE"].astype(str)
        for name in sorted(set(dataset_type)):
            if name in self.setup_errors:
                raise OutputRejected(
                    f"the {name} frame is unusable ({self.setup_errors[name]})"
                )
        unframed = sorted(set(dataset_type) - set(self._frame_keys))
        if unframed:
            raise OutputRejected(f"no {unframed} frame was added to the run")

        value = pd.Series(np.nan, index=out.index)
        model_line = pd.Series(np.nan, index=out.index)
        for name in sorted(set(target)):
            rows = target.eq(name)
            for column in (PREDICTION_VALUE_COLUMNS[name], MODEL_LINE_COLUMNS[name]):
                if column not in out.columns:
                    raise OutputRejected(f"a {name} output without {column}")
            value[rows] = _column(out, PREDICTION_VALUE_COLUMNS[name])[rows]
            model_line[rows] = _column(out, MODEL_LINE_COLUMNS[name])[rows]
        if not np.isfinite(value).all():
            raise OutputRejected("a prediction value is missing or not finite")

        closing = dataset_type.eq(CLOSING_LINE)
        mode = _optional(out, "INPUT_MODE").map(INPUT_MODE_CODES)
        mode = mode.where(~closing, INPUT_MODE_CLOSING)
        if mode.isna().any():
            raise OutputRejected("an intermediate output without INPUT_MODE own/latest")
        snapshot = (
            pd.to_numeric(_optional(out, "TIME_TO_MATCH_MIN"), errors="coerce")
            .where(~closing)
            .astype("Int64")
        )
        if snapshot[~closing].isna().any():
            raise OutputRejected("an intermediate output without TIME_TO_MATCH_MIN")
        horizon = pd.to_numeric(out["MODEL_HORIZON_MINUTES"], errors="coerce").astype(
            "Int64"
        )
        own = mode.eq(INPUT_MODE_OWN).to_numpy(dtype=bool)
        on_horizon = snapshot.eq(horizon).fillna(False).to_numpy(dtype=bool)
        if (own & ~on_horizon).any():
            raise OutputRejected(
                "an own-snapshot row that is not the model's horizon "
                f"(T-{sorted(set(snapshot[own].dropna()))} for "
                f"T-{sorted(set(horizon[own].dropna()))})"
            )

        game_id = game_id_int(out["GAME_ID"])
        unscheduled = sorted(set(game_id) - set(self._games["game_id"]))
        if unscheduled:
            raise OutputRejected(f"game(s) {unscheduled} are not in the schedule")
        tipoffs = self._games.set_index("game_id")["tipoff_at"]
        if tipoffs.reindex(game_id).isna().any():
            raise OutputRejected("a game without a tip-off")

        rows = pd.DataFrame(
            {
                "game_id": game_id.to_numpy(),
                "input_mode": mode.astype(int).to_numpy(),
                "snapshot_horizon_minutes": snapshot.to_numpy(),
                "kind": int(kind),
                "schema_version": out["MODEL_SCHEMA_VERSION"].astype(str).to_numpy(),
                "dataset_type": dataset_type.map(DATASET_TYPE_CODES).to_numpy(),
                "target": target.map(TARGET_CODES).to_numpy(),
                "horizon_minutes": horizon.to_numpy(),
                "variant": out["MODEL_VARIANT"].astype(str).to_numpy(),
                "fit_id": out["FIT_ID"].astype(str).to_numpy(),
                "spec_id": out["SPEC_ID"].astype(str).to_numpy(),
                "train_date_min": pd.to_datetime(
                    _optional(out, "TRAIN_DATE_MIN"), errors="coerce"
                ).dt.date.to_numpy(),
                "train_date_max": pd.to_datetime(
                    _optional(out, "TRAIN_DATE_MAX"), errors="coerce"
                ).dt.date.to_numpy(),
                "training_code_tag": _optional(out, "TRAINING_CODE_TAG").to_numpy(),
                "anchor_book": self.book,
                "s3_build_prefix": _optional(out, "MODEL_BUILD_PREFIX").to_numpy(),
                "slot": label,
                "predicted_at": pd.to_datetime(
                    out["PREDICTED_AT_UTC"], utc=True
                ).to_numpy(),
                "prediction_value": value.astype("float32").to_numpy(),
                "pick_side": out["PRED_PICK"]
                .map(PICK_SIDES)
                .fillna(0)
                .astype(int)
                .to_numpy(),
                "na_feature_count": _column(out, "NA_COLUMNS_COUNT")
                .fillna(0)
                .astype(int)
                .to_numpy(),
                "na_feature_names": _optional(out, "NA_COLUMNS_NAMES").to_numpy(),
                "input_limited_count": _column(out, "INPUT_LIMITED_COUNT")
                .fillna(0)
                .astype(int)
                .to_numpy(),
                "_dataset": dataset_type.to_numpy(),
                "_model_line": model_line.to_numpy(),
            }
        )
        if rows["predicted_at"].isna().any():
            raise OutputRejected("an output without its inference time")

        # Each row must be a row of its frame, carrying that row's line.
        for name, part in rows.groupby("_dataset"):
            keys = self._frame_keys[str(name)]
            located = part[["game_id", "snapshot_horizon_minutes"]].merge(
                keys.reset_index(names="_frame_index"),
                on=["game_id", "snapshot_horizon_minutes"],
                how="left",
            )
            if located["_frame_index"].isna().any():
                raise OutputRejected(f"a row that is not in the {name} frame")
            frame = self.frames[str(name)].loc[located["_frame_index"].astype(int)]
            expected = np.where(
                part["target"].eq(TARGET_CODES["spread_error"]).to_numpy(),
                _column(frame, spread_line_home_col(self.book)).to_numpy(),
                _column(frame, total_line_col(self.book)).to_numpy(),
            ).astype(float)
            got = part["_model_line"].to_numpy(dtype=float)
            agrees = np.isclose(got, expected, atol=1e-6) | (
                np.isnan(got) & np.isnan(expected)
            )
            if not agrees.all():
                raise OutputRejected(
                    "a model's line differs from its input row's line, for game(s) "
                    f"{sorted(part.loc[~agrees, 'game_id'].unique().tolist())}"
                )

        identity = [*INPUT_KEY, *BUILD_KEY]
        seen = pd.concat(
            [*(o[identity] for o in self.outputs), rows[identity]], ignore_index=True
        )
        if seen.duplicated().any():
            raise OutputRejected("a (row, build) already predicted in this run")
        return rows.drop(columns=["_dataset", "_model_line"])

    # -- derived rows -------------------------------------------------------

    def _accepted(self) -> pd.DataFrame:
        if not self.outputs:
            return pd.DataFrame(columns=ACCEPTED_COLUMNS)
        return pd.concat(self.outputs, ignore_index=True)[ACCEPTED_COLUMNS]

    def input_frames(self) -> dict[str, pd.DataFrame]:
        """Per dataset type, the frame rows at least one model read (for S3)."""
        accepted = self._accepted()
        result = {}
        for dataset_type, keys in self._frame_keys.items():
            used = accepted.loc[
                accepted["dataset_type"].eq(DATASET_TYPE_CODES[dataset_type]),
                ["game_id", "snapshot_horizon_minutes"],
            ].drop_duplicates()
            if used.empty:
                continue
            index = keys.reset_index(names="_i").merge(
                used.astype({"snapshot_horizon_minutes": "Int64"}),
                on=["game_id", "snapshot_horizon_minutes"],
            )["_i"]
            result[dataset_type] = self.frames[dataset_type].loc[index]
        return result

    def _latest_quotes(self) -> pd.DataFrame:
        """Per game id, the quote of its latest intermediate row (none: empty)."""
        frame = self.frames.get(INTERMEDIATE_LINE)
        if frame is None or "LATEST_INPUT" not in frame.columns:
            # No raw quote: the closing branch's typed nulls.
            return bet_line_columns(pd.DataFrame(), CLOSING_LINE, self.book)
        latest = frame[frame["LATEST_INPUT"].fillna(False).astype(bool)]
        quotes = bet_line_columns(latest, INTERMEDIATE_LINE, self.book)
        quotes.index = game_id_int(latest["GAME_ID"]).to_numpy()
        return quotes[~quotes.index.duplicated()]

    def _inputs(self, accepted: pd.DataFrame) -> pd.DataFrame:
        parts = []
        tipoffs = self._games.set_index("game_id")["tipoff_at"]
        for code, wanted in accepted.groupby("dataset_type"):
            dataset_type = {v: k for k, v in DATASET_TYPE_CODES.items()}[int(code)]
            wanted = wanted[INPUT_KEY].drop_duplicates()
            keys = self._frame_keys[dataset_type].reset_index(names="_i")
            located = wanted.merge(
                keys, on=["game_id", "snapshot_horizon_minutes"], how="left"
            )
            frame = (
                self.frames[dataset_type]
                .loc[located["_i"].astype(int)]
                .reset_index(drop=True)
            )
            if dataset_type == INTERMEDIATE_LINE:
                tipoff = pd.to_datetime(frame["SNAPSHOT_TS_UTC"], utc=True) + (
                    pd.to_timedelta(
                        located["snapshot_horizon_minutes"].astype(float), unit="m"
                    )
                )
                bets = bet_line_columns(frame, dataset_type, self.book)
            else:
                tipoff = located["game_id"].map(tipoffs)
                bets = self._latest_quotes().reindex(located["game_id"].to_numpy())
                bets.index = located.index
            part = located[INPUT_KEY].assign(
                tipoff_at=pd.to_datetime(tipoff, utc=True).to_numpy(),
                total_model_line=_column(frame, total_line_col(self.book)).to_numpy(),
                spread_model_line=_column(
                    frame, spread_line_home_col(self.book)
                ).to_numpy(),
            )
            part = pd.concat([part, bets.reset_index(drop=True)], axis=1)
            part["line_source"] = line_source_codes(frame).to_numpy()
            parts.append(part)
        if not parts:
            return pd.DataFrame(columns=INPUT_COLUMNS)
        inputs = pd.concat(parts, ignore_index=True)[INPUT_COLUMNS]
        if inputs["tipoff_at"].isna().any():
            raise ValueError("an input row has no tip-off; time to tip is unknown")
        return inputs

    def tables(self, *, status: int, finished_at: datetime | None = None) -> RunTables:
        """The run's rows for the Aiven store (natural keys; the writer maps ids).

        A run with no accepted output still has its ``runs`` row, so coverage
        can be measured; its other tables are empty.
        """
        if self.scheduled_games is None:
            raise ValueError("set_schedule() was never called")
        accepted = self._accepted()
        inputs = self._inputs(accepted)
        games = self._games[self._games["game_id"].isin(inputs["game_id"])]
        builds = (
            accepted[BUILD_COLUMNS].drop_duplicates(BUILD_KEY).reset_index(drop=True)
        )
        predictions = accepted[ACCEPTED_COLUMNS[: len(PREDICTION_COLUMNS) - 1]].merge(
            inputs[[*INPUT_KEY, "tipoff_at"]], on=INPUT_KEY, how="left"
        )
        seconds = (
            pd.to_datetime(predictions["tipoff_at"], utc=True)
            - pd.to_datetime(predictions["predicted_at"], utc=True)
        ).dt.total_seconds()
        predictions["time_to_tipoff_seconds"] = np.floor(seconds).astype("int64")
        predictions = predictions.drop(columns="tipoff_at")[PREDICTION_COLUMNS]
        run = {
            "run_uuid": self.run_uuid,
            "started_at": _utc(self.started_at),
            "finished_at": _utc(finished_at or datetime.now(UTC)),
            "workflow": int(self.workflow),
            "status": int(status),
            "git_sha": git_sha(),
        }
        return RunTables(
            run=run,
            games=games.reset_index(drop=True),
            model_builds=builds,
            inputs=inputs,
            predictions=predictions.reset_index(drop=True),
        )


def persist_prediction_run(
    recorder: PredictionRunRecorder,
    *,
    s3_client,
    bucket: str,
    partial: bool,
    conn=None,
) -> list[str]:
    """Store a run: S3 archive first, then the Aiven rows. Returns the failures.

    ``partial`` says some slot failed (``runs.status`` 2). ``conn`` defaults to
    a new Aiven connection, closed afterwards.
    """
    from nba_ou.postgre_db.predictions_aiven.store import write_run
    from nba_ou.utils.s3_prediction_runs import upload_prediction_run

    failures: list[str] = []
    finished_at = datetime.now(UTC)
    status = RUN_STATUS_PARTIAL if partial else RUN_STATUS_OK
    try:
        tables = recorder.tables(status=status, finished_at=finished_at)
    except Exception as exc:  # noqa: BLE001 -- the run's predictions still printed
        print(f"  ✖ Run could not be assembled for storage: {exc}")
        return [f"prediction store: {exc}"]

    try:
        prefix, upload_failures = upload_prediction_run(
            s3_client=s3_client,
            bucket=bucket,
            recorder=recorder,
            tables=tables,
        )
    except Exception as exc:  # noqa: BLE001 -- the SQL rows are still written
        prefix, upload_failures = None, [f"manifest: {exc}"]
    if upload_failures:
        for failure in upload_failures:
            print(f"  ✖ S3 run archive: {failure}")
        failures += [f"S3 run archive: {failure}" for failure in upload_failures]
        tables.run["status"] = RUN_STATUS_S3_FAILED
    else:
        print(f"  ✓ Run archived to s3://{bucket}/{prefix}")

    own_conn = conn is None
    try:
        if own_conn:
            from nba_ou.postgre_db.config.db_config import connect_nba_db

            conn = connect_nba_db(env="aiven")
        written = write_run(conn, tables)
    except Exception as exc:  # noqa: BLE001 -- reported with the run's failures
        print(f"  ✖ Aiven prediction store write failed: {exc}")
        failures.append(f"Aiven prediction store: {exc}")
    else:
        if written:
            print(
                f"  ✓ Stored {len(tables.predictions)} prediction(s) over "
                f"{len(tables.inputs)} input row(s) in Aiven"
            )
        else:
            print(f"  ✓ Run {recorder.run_uuid} was already stored; nothing written")
    finally:
        if own_conn and conn is not None:
            conn.close()
    return failures
