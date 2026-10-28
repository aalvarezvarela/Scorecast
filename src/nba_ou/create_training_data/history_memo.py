"""Per-season results of history-only work, reused across target builds.

A target build (``create_df_to_predict(target_game_ids=...)``) still does some
work for every earlier game, because target rows read it: the player step
records each history row's availability and top-player ids, and the
availability effects compute each history row's raw effects for the
shrinkage fit. For a finished season those results are the same every day,
so this keeps them per season and reuses them.

Each season's entry is keyed on

* the code (``code_key``). By default this is the frozen base schema: its
  version, the memo's root modules and the main sportsbook. The 2_5 builders
  are frozen (a change to a 2_5 value is a new base, ``3_0``), so a deploy
  keeps every entry. ``NBA_OU_HISTORY_MEMO_FROZEN=0`` keys on the source
  instead: every ``nba_ou`` source file the memo's root module can execute --
  its import closure, followed statically, imports inside functions included
  -- the versions of the distributions that closure imports, the Python
  version and the main sportsbook. A change to any of them refuses every
  entry of that memo; a change elsewhere in the package leaves it alone. The
  root is the module that computes and stores the entries, or the builder
  whose callables or prepared frames they are computed from. A code change
  that does move a 2_5 value must therefore come with a rebuild of the frozen
  entries: with ``NBA_OU_HISTORY_MEMO_REBUILD=1`` a build reads no entry and
  writes every one it computes; and
* a digest of the season's inputs as this build loaded them -- games, player
  box scores, odds after the closing-line repair, injuries and injury
  reports -- chained over every earlier season too. A history row of season
  ``S`` reads nothing from later seasons, so the chain up to ``S`` covers
  everything it can depend on, and a repaired or backfilled season (or any
  earlier one) refuses its entry.

Only seasons before the targets' first season are kept: the current season
gains games every day, so its digest never repeats. The parity test
(``scripts/perf/target_parity.py``) checks a build that reads every entry
against the full build, bit for bit.

Entries live in S3 next to the input cache (``.../memo/<name>/<code>/
season=S.pkl``) or, with ``NBA_OU_HISTORY_MEMO_DIR``, in a local directory.
``NBA_OU_HISTORY_MEMO=0`` turns it off.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import sys
import zlib
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from nba_ou.utils import warm_gate

MEMO_PREFIX = "train_data/input_cache/v1/memo"

#: The root of the memos the intermediate builder's injury and line-history
#: stages keep: their entries are computed from callables and frames the
#: builder prepares, so its whole closure is their code.
INTERMEDIATE_BUILDER = "nba_ou.create_training_data.create_intermediate_line_df"

PACKAGE = "nba_ou"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Calls that import code a static walk cannot see. A closure that makes one
#: is keyed on the whole package instead.
_DYNAMIC_IMPORT_NAMES = frozenset({"__import__", "exec", "eval", "import_module"})
_DYNAMIC_IMPORT_ATTRS = frozenset(
    {"import_module", "run_module", "run_path", "iter_modules", "walk_packages"}
)


def history_memo_enabled() -> bool:
    return os.environ.get("NBA_OU_HISTORY_MEMO", "1") != "0"


def history_memo_frozen() -> bool:
    """``NBA_OU_HISTORY_MEMO_FROZEN=0`` keys entries on the source code."""
    return os.environ.get("NBA_OU_HISTORY_MEMO_FROZEN", "1") != "0"


def history_memo_rebuild() -> bool:
    """``NBA_OU_HISTORY_MEMO_REBUILD=1``: read no entry, write every one."""
    return os.environ.get("NBA_OU_HISTORY_MEMO_REBUILD", "0") == "1"


def _module_path(module: str, package_root: Path, package: str) -> Path | None:
    parts = module.split(".")
    if parts[0] != package:
        return None
    base = package_root.joinpath(*parts[1:])
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _module_name(path: Path, package_root: Path, package: str) -> str:
    parts = [package, *path.relative_to(package_root).with_suffix("").parts]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _scan_imports(
    path: Path, package_root: Path, package: str
) -> tuple[set[str], set[str], bool]:
    """``path``'s package imports, its external top-level imports, and whether
    it imports dynamically. Every import counts, wherever it sits."""
    module = _module_name(path, package_root, package)
    tree = ast.parse(path.read_bytes(), filename=str(path))
    internal, external, dynamic = set(), set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == package:
                    internal.add(alias.name)
                else:
                    external.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = module.split(".")
                if path.name != "__init__.py":
                    anchor = anchor[:-1]
                anchor = anchor[: len(anchor) - (node.level - 1)]
                source = ".".join(anchor + ([node.module] if node.module else []))
            else:
                source = node.module or ""
                if source.split(".")[0] != package:
                    external.add(source.split(".")[0])
                    continue
            # ``from a.b import c`` runs a.b, and c too when it is a submodule.
            internal.add(source)
            internal.update(f"{source}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id in _DYNAMIC_IMPORT_NAMES) or (
                isinstance(func, ast.Attribute) and func.attr in _DYNAMIC_IMPORT_ATTRS
            ):
                dynamic = True
    return internal, external, dynamic


def module_closure(
    roots: Iterable[str],
    *,
    package_root: Path = PACKAGE_ROOT,
    package: str = PACKAGE,
) -> tuple[list[Path], set[str], bool]:
    """The package files ``roots`` can execute, the external top-level modules
    they import, and whether any of them imports dynamically.

    Importing a module runs every parent package's ``__init__`` first, so
    those are in the closure too.
    """
    seen: dict[str, Path] = {}
    external: set[str] = set()
    dynamic = False
    todo = list(roots)
    while todo:
        module = todo.pop()
        parts = module.split(".")
        for name in (".".join(parts[:i]) for i in range(1, len(parts) + 1)):
            if name in seen:
                continue
            path = _module_path(name, package_root, package)
            if path is None:
                continue
            seen[name] = path
            internal, imported, uses_dynamic = _scan_imports(
                path, package_root, package
            )
            todo.extend(internal)
            external |= imported
            dynamic |= uses_dynamic
    for root in roots:
        if root not in seen:
            raise ValueError(f"{root} is not a module of {package}")
    return sorted(seen.values()), external, dynamic


def _distribution_versions(modules: Iterable[str]) -> list[str]:
    """``dist==version`` of each installed distribution providing ``modules``.

    Standard-library modules have none; the Python version covers them.
    """
    providers = metadata.packages_distributions()
    dists = {dist for module in modules for dist in providers.get(module, [])}
    versions = []
    for dist in sorted(dists):
        try:
            versions.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{dist}==?")
    return versions


def _closure_key(
    roots: tuple[str, ...], *, package_root: Path, package: str, settings: str
) -> str:
    files, external, dynamic = module_closure(
        roots, package_root=package_root, package=package
    )
    if dynamic:
        # Code imported by name could be anything in the package.
        files = sorted(package_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    for line in _distribution_versions(external):
        digest.update(line.encode())
    digest.update(f"python={sys.version_info[0]}.{sys.version_info[1]}".encode())
    digest.update(settings.encode())
    return digest.hexdigest()[:20]


def code_key(*roots: str) -> str:
    """The code a memo rooted at ``roots`` (module names) depends on.

    Frozen (the default, ``history_memo_frozen``): the base schema version,
    the roots and the main sportsbook. Otherwise ``source_code_key``.
    """
    if history_memo_frozen():
        return frozen_code_key(*roots)
    return source_code_key(*roots)


@cache
def frozen_code_key(*roots: str) -> str:
    """The base schema version, ``roots`` and the main sportsbook.

    The roots keep memos of different builders apart, as the source key does.
    """
    from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
    from nba_ou.config.odds_columns import get_main_book

    tag = hashlib.sha256("|".join(roots).encode()).hexdigest()[:8]
    return f"frozen-{BASE_SCHEMA_VERSION}-{get_main_book()}-{tag}"


@cache
def source_code_key(*roots: str) -> str:
    """Every source file ``roots`` can execute, its libraries and settings.

    ``config.ini`` is not hashed -- it changes with every promotion -- but
    its one setting that changes feature values, the main sportsbook, is.
    """
    from nba_ou.config.odds_columns import get_main_book

    return _closure_key(
        roots,
        package_root=PACKAGE_ROOT,
        package=PACKAGE,
        settings=f"main_book={get_main_book()}",
    )


def season_of_game_ids(game_ids: pd.Series) -> np.ndarray:
    """Season start year from NBA game ids (``00SYYNNNNN``); -1 if unparseable."""
    ids = game_ids.astype(str)
    text = ids.str.zfill(10).str[3:5].where(ids.str.isdigit())
    years = pd.to_numeric(text, errors="coerce")
    return np.where(years.notna(), 2000 + years.fillna(0), -1).astype("int64")


def season_of_game_id(game_id) -> int | None:
    """``season_of_game_ids`` for one id."""
    text = str(game_id)
    return 2000 + int(text.zfill(10)[3:5]) if text.isdigit() else None


def _frame_digest(frame: pd.DataFrame) -> bytes:
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(str(column).encode())
        digest.update(str(frame[column].dtype).encode())
        try:
            hashed = pd.util.hash_pandas_object(frame[column], index=False)
        except TypeError:  # unhashable values: lists, dicts
            hashed = pd.util.hash_pandas_object(frame[column].map(repr), index=False)
        digest.update(hashed.to_numpy().tobytes())
    return digest.digest()


def _season_values(frame: pd.DataFrame) -> pd.DataFrame:
    """A season's rows as their values, without the schema other rows impose.

    Rows a build appends for tonight's games change the whole frame's schema,
    not the earlier rows: their NaN outcomes turn integer columns into floats
    (or nullable ``Int64``), and columns only they carry (``GAME_TIME``) are
    NaN for every earlier row. So a column that is empty in the season is
    left out, and numeric ones (bool included) are compared as float64.
    """
    columns = frame.columns[frame.notna().any().to_numpy()]
    return pd.DataFrame(
        {
            column: (
                frame[column].astype("float64")
                if pd.api.types.is_numeric_dtype(frame[column])
                else frame[column]
            )
            for column in columns
        },
        index=frame.index,
    )


def season_input_digests(
    inputs: dict[str, tuple[pd.DataFrame, str]],
    seasons: Iterable[int],
    *,
    values_only: bool = False,
    parts: dict[int, dict[str, str]] | None = None,
) -> dict[int, str]:
    """Chained digest per season of every input frame's rows up to it.

    ``inputs`` maps a name to ``(frame, game_id_column)``; a row belongs to
    the season its game id names. Rows keep the frame's order. The code is
    not part of it: it is in every entry's path (``HistoryMemo.code``).

    ``values_only`` digests each season's rows through ``_season_values``, so
    a build that also holds tonight's games and one that does not agree on
    every earlier season.

    ``parts``, when given, is filled with each season's own digest of each
    input (not chained): what ``HistoryMemo`` names when a season misses.
    """
    per_frame = {}
    for name, (frame, game_id_column) in sorted(inputs.items()):
        if frame is None or frame.empty:
            per_frame[name] = (frame, np.array([], dtype="int64"))
            continue
        per_frame[name] = (frame, season_of_game_ids(frame[game_id_column]))
    out = {}
    chain = b""
    for season in sorted(set(seasons)):
        digest = hashlib.sha256(chain)
        digest.update(str(season).encode())
        season_parts = {}
        for name, (frame, frame_seasons) in sorted(per_frame.items()):
            digest.update(name.encode())
            if frame is not None and len(frame_seasons):
                rows = frame[frame_seasons == season]
                if values_only:
                    rows = _season_values(rows)
                rows_digest = _frame_digest(rows)
                digest.update(rows_digest)
                season_parts[name] = rows_digest.hex()
        chain = digest.digest()
        out[season] = digest.hexdigest()
        if parts is not None:
            parts[season] = season_parts
    return out


class _LocalBackend:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def get(self, key: str) -> bytes | None:
        path = self.root / key
        return path.read_bytes() if path.exists() else None

    def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


class _S3Backend:
    def __init__(self, store) -> None:
        self.store = store

    def get(self, key: str) -> bytes | None:
        return self.store._get(f"{MEMO_PREFIX}/{key}")

    def put(self, key: str, data: bytes) -> None:
        self.store._put(f"{MEMO_PREFIX}/{key}", data)


def default_backend():
    """Local directory if ``NBA_OU_HISTORY_MEMO_DIR`` is set, else S3, else None."""
    if not history_memo_enabled():
        return None
    local = os.environ.get("NBA_OU_HISTORY_MEMO_DIR")
    if local:
        return _LocalBackend(Path(local))
    from nba_ou.postgre_db.season_store import default_store

    store = default_store()
    return _S3Backend(store) if store is not None else None


class HistoryMemo:
    """Season entries of named results, valid for the given input digests.

    ``max_season`` is the last season whose entries may be read or written.
    ``code`` is the memo's ``code_key``. Any failure to read or write is
    reported and treated as a miss: the caller computes the season itself, so
    a broken store costs time, never a value.

    ``parts`` names what each season's digest is made of, as
    ``{season: {part: digest}}``. It is stored with the entry and only read
    to say which parts changed when the digest does ("inputs changed (odds)");
    the digest alone decides whether an entry is used.
    """

    def __init__(
        self,
        backend,
        digests: dict[int, str],
        max_season: int,
        *,
        code: str,
        parts: dict[int, dict[str, str]] | None = None,
    ) -> None:
        self.backend = backend
        self.digests = digests
        self.max_season = max_season
        self.code = code
        self.parts = parts or {}
        self.report: list[str] = []
        self._tables: dict[tuple[str, int], dict] = {}
        self._sizes: dict[tuple[str, int], int] = {}
        self._pending: dict[tuple[str, int], Future] = {}
        #: Tables whose stored entry was refused: computed again, for sure.
        self._refused: set[tuple[str, int]] = set()

    def prefetch(self, names: Iterable[str]) -> None:
        """Start downloading every usable season of ``names`` in the background.

        Entries are a few MB each and mostly transfer time, so they come down
        in parallel while the build works on other stages; ``table`` waits
        for its own entry only.
        """
        keys = [
            (name, season)
            for name in names
            for season in sorted(self.digests)
            if self.usable(season) and (name, season) not in self._tables
        ]
        if not keys:
            return
        pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="history-memo")
        for key in keys:
            self._pending[key] = pool.submit(self.load, *key)
        pool.shutdown(wait=False)

    def wait(self) -> None:
        """Finish every prefetch, so the tables are in memory before a fork.

        A forked process has none of the download threads: a table still
        pending in it would never arrive.
        """
        for key in list(self._pending):
            self.table(*key)

    def table(self, name: str, season) -> dict | None:
        """The season's entries as a dict to read and add to; None if unusable.

        Loaded once per build and shared by every caller; ``flush`` writes the
        tables that gained entries.

        Computing a season is reported to the warm gate
        (``nba_ou.utils.warm_gate``): a refused entry as it is loaded, a table
        with no entry stored when it gains its first. A table that stays empty
        (a season with nothing to keep) is no work, and holds no one.
        """
        if not self.usable(season):
            return None
        key = (name, int(season))
        if key not in self._tables:
            pending = self._pending.pop(key, None)
            loaded = pending.result() if pending is not None else self.load(*key)
            if isinstance(loaded, dict):
                self._tables[key] = loaded
            elif key in self._refused:
                warm_gate.cold(name, key[1])
                self._tables[key] = {}
            else:
                self._tables[key] = _MissedTable(name, key[1])
            self._sizes[key] = len(self._tables[key])
        return self._tables[key]

    def flush(self) -> None:
        for (name, season), table in self._tables.items():
            if len(table) > self._sizes[(name, season)]:
                self.save(name, season, table)
                self._sizes[(name, season)] = len(table)
        loaded = sum(self._sizes.values())
        print(
            f"History memo: {len(self._tables)} season tables, {loaded:,} entries"
            + (f"; {'; '.join(_grouped(self.report))}" if self.report else "")
        )

    def usable(self, season) -> bool:
        try:
            season = int(season)
        except (TypeError, ValueError):
            return False
        return season <= self.max_season and season in self.digests

    def _key(self, name: str, season: int, suffix: str) -> str:
        return f"{name}/{self.code}/season={season}.{suffix}"

    def load(self, name: str, season: int):
        if not self.usable(season):
            return None
        if history_memo_rebuild():
            self.report.append(f"{name} {season}: rebuilt")
            self._refused.add((name, season))
            return None
        try:
            raw = self.backend.get(self._key(name, season, "json"))
            if raw is None:
                return None
            meta = json.loads(raw)
            if meta.get("digest") != self.digests[season]:
                self.report.append(
                    f"{name} {season}: inputs changed"
                    + self._changed_parts(meta.get("parts"), season)
                )
                self._refused.add((name, season))
                return None
            payload = self.backend.get(self._key(name, season, "pkl"))
            if payload is None or hashlib.sha256(payload).hexdigest() != meta["sha256"]:
                self.report.append(f"{name} {season}: payload missing or corrupt")
                self._refused.add((name, season))
                return None
            if meta.get("compression") == "zlib":
                payload = zlib.decompress(payload)
            return pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001 -- a broken store is a miss
            self.report.append(f"{name} {season}: read failed ({exc})")
            self._refused.add((name, season))
            return None

    def _changed_parts(self, stored: dict | None, season: int) -> str:
        """`` (odds, players)``: the parts whose digest differs from the
        entry's; `` (an earlier season)`` when none does, as a chained digest
        moves with any earlier season's input."""
        current = self.parts.get(season)
        if current is None or not isinstance(stored, dict):
            return ""
        changed = sorted(
            part
            for part in set(current) | set(stored)
            if current.get(part) != stored.get(part)
        )
        return f" ({', '.join(changed) if changed else 'an earlier season'})"

    def save(self, name: str, season: int, value) -> None:
        if not self.usable(season):
            return
        try:
            # zlib: the pickled tables shrink ~6x, and they are read over S3.
            if isinstance(value, _MissedTable):
                value = dict(value)
            payload = zlib.compress(
                pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), 6
            )
            meta = {
                "digest": self.digests[season],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "compression": "zlib",
            }
            if season in self.parts:
                meta["parts"] = self.parts[season]
            self.backend.put(self._key(name, season, "pkl"), payload)
            self.backend.put(self._key(name, season, "json"), json.dumps(meta).encode())
        except Exception as exc:  # noqa: BLE001
            self.report.append(f"{name} {season}: write failed ({exc})")


def _grouped(report: list[str]) -> list[str]:
    """``report``'s lines with one per table and cause, its seasons listed:
    ``"rows 2018, 2020: inputs changed"`` for ``"rows 2018: inputs changed"``
    and ``"rows 2020: inputs changed"``. A line of another shape stays as is."""
    seasons: dict[tuple[str, str], list[str]] = {}
    for line in report:
        head, sep, cause = line.partition(": ")
        name, _, season = head.rpartition(" ")
        if not sep or not name or not season.isdigit():
            name, season, cause = line, "", ""
        seasons.setdefault((name, cause), []).append(season)
    return [
        (
            f"{name} {', '.join(sorted(filter(None, listed)))}: {cause}"
            if cause
            else name
        )
        for (name, cause), listed in seasons.items()
    ]


class _MissedTable(dict):
    """A season table with no entry stored: its first entry tells the warm
    gate this build is computing a past season."""

    def __init__(self, name: str, season: int) -> None:
        super().__init__()
        self._missed: tuple[str, int] | None = (name, season)

    def _computed(self) -> None:
        if self._missed is not None:
            missed, self._missed = self._missed, None
            warm_gate.cold(*missed)

    def __setitem__(self, key, value) -> None:
        self._computed()
        super().__setitem__(key, value)

    def update(self, *args, **kwargs) -> None:
        other = dict(*args, **kwargs)
        if other:
            self._computed()
        super().update(other)

    def setdefault(self, key, default=None):
        if key not in self:
            self._computed()
        return super().setdefault(key, default)


class _SeasonRows:
    """``RowCache.rows`` over a memo's season tables, plus the build's own rows.

    A key whose season the memo covers, and that ``persist`` admits, is read
    from and written to that season's table; any other key lives only in this
    build, as in a plain ``RowCache``.
    """

    def __init__(self, memo: HistoryMemo, name: str, season_of, persist) -> None:
        self.memo, self.name = memo, name
        self.season_of, self.persist = season_of, persist
        self.local: dict = {}
        #: Entries this process added to the season tables, by season.
        self.added: dict[int, dict] = {}

    def _table(self, key):
        if not self.persist(key):
            return None
        return self.memo.table(self.name, self.season_of(key))

    def get(self, key, default=None):
        if key in self.local:
            return self.local[key]
        table = self._table(key)
        if table is None:
            return default
        return table.get(key, default)

    def __setitem__(self, key, value) -> None:
        table = self._table(key)
        if table is None:
            self.local[key] = value
        else:
            table[key] = value
            self.added.setdefault(int(self.season_of(key)), {})[key] = value


class PersistentRowCache:
    """A ``RowCache`` whose rows of earlier games outlive the build.

    The row builders key each row on everything it reads that can differ
    between their calls (``RowCache``); what stays the same within a build --
    the game history, the players, the injuries -- the memo's season digests
    cover across builds. So a row cached under its key in a season table is
    the row any later build with the same digests computes. ``persist`` keeps
    tonight's rows out: their inputs can still change before tip-off.

    ``bind`` checks the configuration within a build only; a build's sizes,
    which ``RowCache`` signatures include, differ from build to build.
    """

    def __init__(
        self,
        memo: HistoryMemo,
        name: str,
        *,
        season_of,
        persist=lambda key: True,
        size_positions: tuple[int, ...] = (),
    ) -> None:
        self.rows = _SeasonRows(memo, name, season_of, persist)
        self.hits = 0
        self.misses = 0
        self._size_positions = size_positions
        self._signature = None

    def added_entries(self) -> dict[int, dict]:
        """The season entries this process computed (for a parent to merge)."""
        return self.rows.added

    def merge_entries(self, entries: dict[int, dict]) -> None:
        """Add entries another process computed under the same memo."""
        for season, rows in entries.items():
            table = self.rows.memo.table(self.rows.name, season)
            if table is not None:
                table.update(rows)

    def bind(self, signature: tuple) -> None:
        kept = tuple(
            value
            for position, value in enumerate(signature)
            if position not in self._size_positions
        )
        if self._signature is None:
            self._signature = kept
        elif self._signature != kept:
            raise ValueError(
                f"PersistentRowCache reused with different inputs: "
                f"{self._signature} != {kept}"
            )
