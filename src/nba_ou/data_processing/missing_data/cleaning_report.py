"""A record of what cleaning removed, and why.

The cleaning pipeline drops on the order of 350 columns and a few dozen rows
across nine steps, and until this existed it said so only by printing. That made
"why is this column not in my model?" a question you answered by re-running with
``verbose=2`` and reading several hundred lines of output -- if you still had the
same data and settings to re-run with.

The report is a plain record, not a policy: it never changes what is dropped.
Every field is filled from a decision the pipeline already made, so keeping it
costs a dict append per dropped column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CleaningReport:
    """Which columns and rows were removed, by which step, and for what reason.

    Steps are recorded in execution order, so reading ``column_drops`` top to
    bottom reconstructs the run. A column appears at most once: each step drops
    from what the previous ones left.
    """

    #: (column, step, reason) for every column removed.
    column_drops: list[dict[str, str]] = field(default_factory=list)
    #: (step, rows_before, rows_after, reason) for every row-removing step.
    row_drops: list[dict[str, Any]] = field(default_factory=list)
    #: Per-group row retention, when the caller named a grouping column. See
    #: ``record_group_survival``.
    group_survival: dict[str, Any] = field(default_factory=dict)
    #: Which rows and columns the correlation step actually judged, when a
    #: repeated-measures policy was in force. See ``record_redundancy_view``.
    redundancy_view: dict[str, Any] = field(default_factory=dict)
    #: Columns held out of the correlation comparison for having no spread on
    #: the view. They survive cleaning. See ``record_correlation_exclusions``.
    correlation_exclusions: list[str] = field(default_factory=list)
    columns_in: int = 0
    columns_out: int = 0
    rows_in: int = 0
    rows_out: int = 0

    def _record(self, column: str, step: str, reason: str) -> None:
        """First step to catch a column owns it.

        Several steps scan the full column list and accumulate into one set
        before anything is dropped, so a column can satisfy more than one --
        GAME_ID is both a pure-string column and an ``_ID`` column. It is
        removed once, so it must be reported once, or ``columns_dropped_by_step``
        sums to more than the columns that actually went.
        """
        if any(entry["column"] == column for entry in self.column_drops):
            return
        self.column_drops.append({"column": column, "step": step, "reason": reason})

    def drop_columns(self, columns: list[str], *, step: str, reason: str) -> None:
        for column in columns:
            self._record(column, step, reason)

    def drop_columns_with_reasons(self, reasons: dict[str, str], *, step: str) -> None:
        """Record drops that each carry their own reason, e.g. a correlation
        partner and the correlation with it."""
        for column, reason in reasons.items():
            self._record(column, step, reason)

    def record_rows(self, *, step: str, before: int, after: int, reason: str) -> None:
        if before == after:
            return
        self.row_drops.append(
            {
                "step": step,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_dropped": int(before - after),
                "reason": reason,
            }
        )

    def record_group_survival(
        self,
        *,
        group_col: str,
        before_counts: dict[Any, int],
        after_counts: dict[Any, int],
    ) -> None:
        """How evenly the row filters fell across the values of ``group_col``.

        Row cleaning is stated as one global number -- ``max_na_per_row`` drops
        rows over a NaN budget -- but nothing makes that budget bite evenly. On
        the intermediate-line dataset, where a group is one pre-game snapshot
        horizon, it does not: the long horizons carry more NaNs because their
        look-back windows reach into less tick history, so a threshold set for
        the dataset as a whole deletes 12.2% of the 12-hours-out rows against
        8.6% of the 30-minutes-out ones. That silently re-weights the snapshot
        mix, which is the single axis the dataset exists to compare.

        Recorded rather than enforced. There is no correct retention spread to
        assert -- some unevenness is inherent to the data -- and failing a build
        over it would be worse than the imbalance. The point is that it stops
        being invisible: it lands in ``cleaning_report.json`` on every run,
        where a comparison across runs can read it.
        """
        # Natural order where the keys allow it, so snapshot horizons read
        # 0, 30, 60, ... rather than the 0, 120, 180, 240, 30 that sorting them
        # as text gives -- the whole point of the table is reading a trend down
        # the horizon axis. Falls back to text for mixed or unorderable keys.
        keys = list(set(before_counts) | set(after_counts))
        try:
            keys.sort()
        except TypeError:
            keys.sort(key=repr)
        groups = []
        for key in keys:
            before = int(before_counts.get(key, 0))
            after = int(after_counts.get(key, 0))
            groups.append(
                {
                    "group": key,
                    "rows_before": before,
                    "rows_after": after,
                    "retention_pct": (
                        round(100.0 * after / before, 2) if before else None
                    ),
                }
            )
        retentions = [
            g["retention_pct"] for g in groups if g["retention_pct"] is not None
        ]
        self.group_survival = {
            "group_col": group_col,
            "groups": groups,
            "retention_spread_pp": (
                round(max(retentions) - min(retentions), 2) if retentions else None
            ),
        }

    def record_redundancy_view(
        self,
        *,
        group_col: str,
        snapshot_col: str | None,
        target_snapshot: float,
        rows_in_view: int,
        rows_total: int,
        exempt_columns: list[str],
        snapshots_used: dict[Any, int] | None = None,
    ) -> None:
        """What the correlation step was actually shown.

        Without this, two very different fates are indistinguishable in the
        report. A snapshot/market column that survived was never a candidate --
        it is exempt by policy. A historical column that survived was judged and
        kept. Recording the view makes "was this column even eligible?"
        answerable, and ``snapshots_used`` shows how often the preferred
        horizon was actually available rather than assumed.

        Drops themselves stay distinguishable by step name: exact duplicates
        under ``duplicate_columns`` / ``absolute_value_match``, correlation
        under ``correlated_columns_one_row_per_group``.
        """
        self.redundancy_view = {
            "group_col": group_col,
            "snapshot_col": snapshot_col,
            "target_snapshot": target_snapshot,
            "rows_in_view": int(rows_in_view),
            "rows_total": int(rows_total),
            "n_exempt_columns": len(exempt_columns),
            "exempt_columns": sorted(exempt_columns),
            "snapshots_used": (
                None
                if snapshots_used is None
                else {str(k): int(v) for k, v in sorted(snapshots_used.items())}
            ),
        }

    def record_correlation_exclusions(self, columns: list[str]) -> None:
        """Columns the correlation step refused to judge, for lack of spread.

        Kept apart from ``column_drops`` because these columns survive: the
        record answers "why was this never compared?", not "why did this die?".
        A constant correlates with nothing, so judging it either way is
        meaningless -- but leaving it in the comparison makes it a
        divide-by-zero that reads as perfect redundancy, which is what this
        exclusion exists to prevent.
        """
        self.correlation_exclusions = sorted(columns)

    def why_dropped(self, column: str) -> dict[str, str] | None:
        """The record for one column, or None if it survived."""
        for entry in self.column_drops:
            if entry["column"] == column:
                return entry
        return None

    def columns_by_step(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.column_drops:
            counts[entry["step"]] = counts.get(entry["step"], 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns_in": self.columns_in,
            "columns_out": self.columns_out,
            "columns_dropped": len(self.column_drops),
            "columns_dropped_by_step": self.columns_by_step(),
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_in - self.rows_out,
            "row_drops": self.row_drops,
            "group_survival": self.group_survival,
            "redundancy_view": self.redundancy_view,
            "correlation_exclusions": self.correlation_exclusions,
            "column_drops": self.column_drops,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def summary(self) -> str:
        lines = [
            f"columns {self.columns_in} -> {self.columns_out} "
            f"({len(self.column_drops)} dropped)",
        ]
        for step, count in self.columns_by_step().items():
            lines.append(f"    {count:>5} {step}")
        lines.append(f"rows {self.rows_in} -> {self.rows_out}")
        for entry in self.row_drops:
            lines.append(f"    {entry['rows_dropped']:>5} {entry['step']}")
        if self.redundancy_view:
            view = self.redundancy_view
            lines.append(
                f"correlation judged on one row per {view['group_col']}: "
                f"{view['rows_in_view']:,} of {view['rows_total']:,} rows, "
                f"{view['n_exempt_columns']} snapshot columns exempt"
            )
        if self.correlation_exclusions:
            lines.append(
                f"    {len(self.correlation_exclusions)} constant columns excluded "
                f"from the comparison (kept): "
                f"{', '.join(self.correlation_exclusions[:5])}"
                + (" ..." if len(self.correlation_exclusions) > 5 else "")
            )
        if self.group_survival:
            spread = self.group_survival.get("retention_spread_pp")
            lines.append(
                f"row retention by {self.group_survival['group_col']} "
                f"(spread {spread}pp)"
            )
            for group in self.group_survival["groups"]:
                lines.append(
                    f"    {str(group['group']):>8}  {group['rows_before']:>6} -> "
                    f"{group['rows_after']:>6}  {group['retention_pct']}%"
                )
        return "\n".join(lines)
