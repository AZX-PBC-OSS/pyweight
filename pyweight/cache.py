"""Persistent SQLite cache for parsed modules and cost/timing/memory results."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pyweight.models import CostEstimate, MemoryResult, ModuleInfo, TimingResult

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS: int = 3600

_SCHEMA_VERSION = 1

_DDL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    tool_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS module_info (
    file_path TEXT NOT NULL,
    mtime REAL NOT NULL,
    file_size INTEGER NOT NULL,
    python_version TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (file_path, mtime, file_size, python_version, tool_version)
);

CREATE TABLE IF NOT EXISTS cost_estimate (
    module TEXT NOT NULL,
    python_version TEXT NOT NULL,
    venv_hash TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (module, python_version, venv_hash)
);

CREATE TABLE IF NOT EXISTS timing_result (
    module TEXT NOT NULL,
    python_version TEXT NOT NULL,
    venv_hash TEXT NOT NULL,
    stored_at REAL NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (module, python_version, venv_hash)
);

CREATE TABLE IF NOT EXISTS memory_result (
    module TEXT NOT NULL,
    python_version TEXT NOT NULL,
    venv_hash TEXT NOT NULL,
    stored_at REAL NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (module, python_version, venv_hash)
);
"""

_DROP_ALL = """\
DROP TABLE IF EXISTS memory_result;
DROP TABLE IF EXISTS timing_result;
DROP TABLE IF EXISTS cost_estimate;
DROP TABLE IF EXISTS module_info;
DROP TABLE IF EXISTS schema_version;
"""


class Cache:
    """Persistent SQLite cache for parsed modules and cost estimates."""

    SCHEMA_VERSION = _SCHEMA_VERSION

    def __init__(
        self,
        db_path: Path | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        if db_path is None:
            db_path = Path.home() / ".cache" / "pyweight" / "cache.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._ttl = ttl_seconds
        # isolation_level="IMMEDIATE" ensures `with self._conn:` starts an
        # IMMEDIATE transaction, avoiding the double-BEGIN conflict that occurs
        # when a manual BEGIN is issued inside a connection context manager.
        self._conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level="IMMEDIATE")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def __enter__(self) -> Cache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._conn.close()

    @property
    def db_path(self) -> Path:
        """Path to the underlying SQLite database file."""
        return self._db_path

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()

        if row is not None:
            version_row = self._conn.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            if version_row is not None and int(version_row[0]) == _SCHEMA_VERSION:
                return
            found_version = int(version_row[0]) if version_row is not None else "missing"
            logger.warning(
                "Cache schema version mismatch (found %s, expected %s); rebuilding cache",
                found_version,
                _SCHEMA_VERSION,
            )

        # Drop and recreate all tables atomically so a crash mid-migration
        # does not leave a partially migrated schema on disk.
        migration_script = (
            "BEGIN;\n"
            + _DROP_ALL
            + _DDL
            + f"\nINSERT INTO schema_version (id, version, tool_version)"
            f" VALUES (1, {_SCHEMA_VERSION}, '0.1.0');\n"
            "COMMIT;"
        )
        self._conn.executescript(migration_script)

    # ------------------------------------------------------------------
    # Module info cache
    # ------------------------------------------------------------------

    def get_module_info(
        self,
        path: Path,
        mtime: float,
        size: int,
        python_version: str,
        tool_version: str,
    ) -> ModuleInfo | None:
        sql = (
            "SELECT data FROM module_info"
            " WHERE file_path=? AND mtime=? AND file_size=?"
            " AND python_version=? AND tool_version=?"
        )
        row = self._conn.execute(
            sql, (str(path), mtime, size, python_version, tool_version)
        ).fetchone()
        if row is None:
            return None
        return ModuleInfo.from_dict(json.loads(row[0]))

    def put_module_info(
        self,
        path: Path,
        mtime: float,
        size: int,
        python_version: str,
        tool_version: str,
        info: ModuleInfo,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO module_info"
                " (file_path, mtime, file_size, python_version, tool_version, data)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (str(path), mtime, size, python_version, tool_version, json.dumps(info.to_dict())),
            )

    # ------------------------------------------------------------------
    # Cost estimate cache
    # ------------------------------------------------------------------

    def get_cost(
        self,
        module: str,
        python_version: str,
        venv_hash: str,
    ) -> CostEstimate | None:
        row = self._conn.execute(
            "SELECT data FROM cost_estimate WHERE module=? AND python_version=? AND venv_hash=?",
            (module, python_version, venv_hash),
        ).fetchone()
        if row is None:
            return None
        return CostEstimate.from_dict(json.loads(row[0]))

    def put_cost(
        self,
        module: str,
        python_version: str,
        venv_hash: str,
        estimate: CostEstimate,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cost_estimate"
                " (module, python_version, venv_hash, data)"
                " VALUES (?, ?, ?, ?)",
                (module, python_version, venv_hash, json.dumps(estimate.to_dict())),
            )

    # ------------------------------------------------------------------
    # Timing result cache (with TTL)
    # ------------------------------------------------------------------

    def get_timing(
        self,
        module: str,
        python_version: str,
        venv_hash: str,
    ) -> TimingResult | None:
        row = self._conn.execute(
            "SELECT data, stored_at FROM timing_result"
            " WHERE module=? AND python_version=? AND venv_hash=?",
            (module, python_version, venv_hash),
        ).fetchone()
        if row is None:
            return None
        stored_at = float(row[1])
        if time.time() - stored_at > self._ttl:
            return None
        return TimingResult.from_dict(json.loads(row[0]))

    def put_timing(
        self,
        module: str,
        python_version: str,
        venv_hash: str,
        result: TimingResult,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO timing_result"
                " (module, python_version, venv_hash, stored_at, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (module, python_version, venv_hash, time.time(), json.dumps(result.to_dict())),
            )

    # ------------------------------------------------------------------
    # Memory result cache (with TTL)
    # ------------------------------------------------------------------

    def get_memory(
        self,
        module: str,
        python_version: str,
        venv_hash: str,
    ) -> MemoryResult | None:
        row = self._conn.execute(
            "SELECT data, stored_at FROM memory_result"
            " WHERE module=? AND python_version=? AND venv_hash=?",
            (module, python_version, venv_hash),
        ).fetchone()
        if row is None:
            return None
        stored_at = float(row[1])
        if time.time() - stored_at > self._ttl:
            return None
        return MemoryResult.from_dict(json.loads(row[0]))

    def put_memory(
        self,
        module: str,
        python_version: str,
        venv_hash: str,
        result: MemoryResult,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_result"
                " (module, python_version, venv_hash, stored_at, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (module, python_version, venv_hash, time.time(), json.dumps(result.to_dict())),
            )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM module_info")
            self._conn.execute("DELETE FROM cost_estimate")
            self._conn.execute("DELETE FROM timing_result")
            self._conn.execute("DELETE FROM memory_result")

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    @property
    def journal_mode(self) -> str:
        """Return the current SQLite journal mode (e.g. 'wal', 'delete')."""
        row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else ""
