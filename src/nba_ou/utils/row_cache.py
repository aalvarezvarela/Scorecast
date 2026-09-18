"""Row results reused across repeated calls of a row-by-row feature builder."""

from __future__ import annotations


class RowCache:
    """Per-row outputs of a builder that is called repeatedly on the same inputs.

    The snapshot injury stage runs each team-game builder once per horizon on
    the same games and players; only the as-of injury report changes. A
    builder that accepts a ``RowCache`` keys every row on everything it reads
    that can differ between calls, and reuses the stored output when the key
    was seen before, so the result is identical to rebuilding every row.

    A cache is only valid across calls whose remaining inputs are identical;
    ``bind`` rejects a call with a different configuration.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple, object] = {}
        self.hits = 0
        self.misses = 0
        self._signature: tuple | None = None

    def bind(self, signature: tuple) -> None:
        if self._signature is None:
            self._signature = signature
        elif self._signature != signature:
            raise ValueError(
                "RowCache reused with different inputs: "
                f"{self._signature} != {signature}"
            )
