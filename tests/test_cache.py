"""Comprehensive tests for pyweight.cache."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from pyweight.cache import Cache
from pyweight.models import (
    Confidence,
    CostEstimate,
    MemoryResult,
    ModuleInfo,
    TimingResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.db"


@pytest.fixture
def cache(db_path: Path) -> Generator[Cache]:
    c = Cache(db_path=db_path)
    yield c
    c.close()


def _module_info(path: Path | None = None) -> ModuleInfo:
    from pyweight.models import ImportRef

    p = path or Path("/tmp/fake_module.py")
    ref = ImportRef(
        module="os",
        names=("path",),
        alias=None,
        lineno=1,
        scope="module",
        level=0,
        is_type_checking=False,
        is_conditional=False,
        is_star=False,
    )
    return ModuleInfo(
        path=p,
        qualified_name="fake_module",
        imports=(ref,),
        is_init=False,
        has_getattr=False,
        has_all=True,
        all_names=frozenset({"foo", "bar"}),
        defined_names=frozenset({"baz"}),
        star_imports=(),
        purity="pure",
        usage_scopes={"os": frozenset({"module"})},
    )


def _cost_estimate(module: str = "requests") -> CostEstimate:
    return CostEstimate(
        module=module,
        dist_name="requests",
        direct_size_bytes=100_000,
        transitive_size_bytes=200_000,
        confidence=Confidence.HIGH,
        provenance="test",
    )


def _timing_result(module: str = "requests") -> TimingResult:
    return TimingResult(
        module=module,
        self_time_us=1500,
        cumulative_time_us=8000,
        dependencies=("urllib3", "certifi"),
    )


def _memory_result(module: str = "requests") -> MemoryResult:
    return MemoryResult(
        module=module,
        python_heap_bytes=4_096_000,
        rss_bytes=12_000_000,
    )


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(cache: Cache, tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text("import os\n", encoding="utf-8")
    assert cache.get_module_info(p, 1.0, 10, "3.13", "0.1.0") is None
    assert cache.get_cost("requests", "3.13", "abc123") is None
    assert cache.get_timing("requests", "3.13", "abc123") is None
    assert cache.get_memory("requests", "3.13", "abc123") is None


# ---------------------------------------------------------------------------
# Cache hit after put
# ---------------------------------------------------------------------------


def test_cache_hit_after_put(cache: Cache, tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    info = _module_info(p)
    cache.put_module_info(p, 1.0, 10, "3.13", "0.1.0", info)
    result = cache.get_module_info(p, 1.0, 10, "3.13", "0.1.0")
    assert result is not None
    assert result.qualified_name == info.qualified_name


def test_cost_cache_hit_after_put(cache: Cache) -> None:
    est = _cost_estimate()
    cache.put_cost("requests", "3.13", "abc123", est)
    result = cache.get_cost("requests", "3.13", "abc123")
    assert result is not None
    assert result.module == "requests"


def test_timing_cache_hit_after_put(cache: Cache) -> None:
    tr = _timing_result()
    cache.put_timing("requests", "3.13", "abc123", tr)
    result = cache.get_timing("requests", "3.13", "abc123")
    assert result is not None
    assert result.module == "requests"


def test_memory_cache_hit_after_put(cache: Cache) -> None:
    mr = _memory_result()
    cache.put_memory("requests", "3.13", "abc123", mr)
    result = cache.get_memory("requests", "3.13", "abc123")
    assert result is not None
    assert result.module == "requests"


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


def test_ttl_expiration_timing(db_path: Path) -> None:
    with Cache(db_path=db_path, ttl_seconds=1) as c:
        tr = _timing_result()
        c.put_timing("requests", "3.13", "abc123", tr)
        assert c.get_timing("requests", "3.13", "abc123") is not None

        # Mock time.time() to be past TTL
        future = time.time() + 3600
        with patch("pyweight.cache.time") as mock_time:
            mock_time.time.return_value = future
            assert c.get_timing("requests", "3.13", "abc123") is None


def test_ttl_expiration_memory(db_path: Path) -> None:
    with Cache(db_path=db_path, ttl_seconds=1) as c:
        mr = _memory_result()
        c.put_memory("requests", "3.13", "abc123", mr)
        assert c.get_memory("requests", "3.13", "abc123") is not None

        future = time.time() + 3600
        with patch("pyweight.cache.time") as mock_time:
            mock_time.time.return_value = future
            assert c.get_memory("requests", "3.13", "abc123") is None


# ---------------------------------------------------------------------------
# Cache invalidation on mtime change
# ---------------------------------------------------------------------------


def test_cache_invalidation_mtime(cache: Cache, tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    info = _module_info(p)
    cache.put_module_info(p, 1000.0, 10, "3.13", "0.1.0", info)
    # Same path, different mtime → miss
    assert cache.get_module_info(p, 2000.0, 10, "3.13", "0.1.0") is None
    # Original mtime still works
    assert cache.get_module_info(p, 1000.0, 10, "3.13", "0.1.0") is not None


def test_cache_invalidation_size(cache: Cache, tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    info = _module_info(p)
    cache.put_module_info(p, 1000.0, 100, "3.13", "0.1.0", info)
    # Different file_size → miss
    assert cache.get_module_info(p, 1000.0, 200, "3.13", "0.1.0") is None


# ---------------------------------------------------------------------------
# Environment scoping
# ---------------------------------------------------------------------------


def test_environment_scoping(cache: Cache) -> None:
    est = _cost_estimate()
    cache.put_cost("requests", "3.13", "venv_a_hash", est)
    # Different venv_hash → miss
    assert cache.get_cost("requests", "3.13", "venv_b_hash") is None
    # Original venv still present
    assert cache.get_cost("requests", "3.13", "venv_a_hash") is not None


def test_timing_environment_scoping(cache: Cache) -> None:
    tr = _timing_result()
    cache.put_timing("requests", "3.13", "venv_a", tr)
    assert cache.get_timing("requests", "3.13", "venv_b") is None
    assert cache.get_timing("requests", "3.13", "venv_a") is not None


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_empties_all(cache: Cache, tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    cache.put_module_info(p, 1.0, 10, "3.13", "0.1.0", _module_info(p))
    cache.put_cost("requests", "3.13", "h", _cost_estimate())
    cache.put_timing("requests", "3.13", "h", _timing_result())
    cache.put_memory("requests", "3.13", "h", _memory_result())

    cache.clear()

    assert cache.get_module_info(p, 1.0, 10, "3.13", "0.1.0") is None
    assert cache.get_cost("requests", "3.13", "h") is None
    assert cache.get_timing("requests", "3.13", "h") is None
    assert cache.get_memory("requests", "3.13", "h") is None


# ---------------------------------------------------------------------------
# DB file creation
# ---------------------------------------------------------------------------


def test_cache_file_created(tmp_path: Path) -> None:
    db = tmp_path / "subdir" / "cache.db"
    with Cache(db_path=db) as c:
        assert db.exists()
        _ = c  # used via context manager


def test_cache_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with Cache() as c:
        expected = tmp_path / ".cache" / "pyweight" / "cache.db"
        assert expected.exists()
        _ = c  # used via context manager


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_schema_migration(db_path: Path) -> None:
    # Create a cache, then overwrite the schema version via a separate connection
    # to simulate an old cache file.
    with Cache(db_path=db_path) as c1:
        c1.put_cost("requests", "3.13", "h", _cost_estimate())

    # Corrupt the version using an independent connection (not through Cache).
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("UPDATE schema_version SET version = 0 WHERE id = 1")
        raw.commit()
    finally:
        raw.close()

    # Re-opening should detect mismatch, drop all tables, and recreate.
    with Cache(db_path=db_path) as c2:
        # Data from the old schema must be gone.
        assert c2.get_cost("requests", "3.13", "h") is None
        # Schema version must be current.
        raw2 = sqlite3.connect(str(db_path))
        try:
            row = raw2.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        finally:
            raw2.close()
        assert row is not None
        assert int(row[0]) == Cache.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# WAL mode
# ---------------------------------------------------------------------------


def test_wal_mode(cache: Cache) -> None:
    assert cache.journal_mode == "wal"


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------


def test_roundtrip_module_info(cache: Cache, tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    original = _module_info(p)
    cache.put_module_info(p, 1.0, 42, "3.13", "0.1.0", original)
    retrieved = cache.get_module_info(p, 1.0, 42, "3.13", "0.1.0")
    assert retrieved is not None
    assert retrieved.qualified_name == original.qualified_name
    assert retrieved.imports == original.imports
    assert retrieved.is_init == original.is_init
    assert retrieved.has_getattr == original.has_getattr
    assert retrieved.has_all == original.has_all
    assert retrieved.all_names == original.all_names
    assert retrieved.defined_names == original.defined_names
    assert retrieved.star_imports == original.star_imports
    assert retrieved.purity == original.purity
    assert retrieved.usage_scopes == original.usage_scopes


def test_roundtrip_cost_estimate(cache: Cache) -> None:
    original = _cost_estimate()
    cache.put_cost("requests", "3.13", "h", original)
    retrieved = cache.get_cost("requests", "3.13", "h")
    assert retrieved is not None
    assert retrieved.module == original.module
    assert retrieved.dist_name == original.dist_name
    assert retrieved.direct_size_bytes == original.direct_size_bytes
    assert retrieved.transitive_size_bytes == original.transitive_size_bytes
    assert retrieved.confidence == original.confidence
    assert retrieved.provenance == original.provenance


def test_roundtrip_timing_result(cache: Cache) -> None:
    original = _timing_result()
    cache.put_timing("requests", "3.13", "h", original)
    retrieved = cache.get_timing("requests", "3.13", "h")
    assert retrieved is not None
    assert retrieved.module == original.module
    assert retrieved.self_time_us == original.self_time_us
    assert retrieved.cumulative_time_us == original.cumulative_time_us
    assert retrieved.dependencies == original.dependencies


def test_roundtrip_memory_result(cache: Cache) -> None:
    original = _memory_result()
    cache.put_memory("requests", "3.13", "h", original)
    retrieved = cache.get_memory("requests", "3.13", "h")
    assert retrieved is not None
    assert retrieved.module == original.module
    assert retrieved.python_heap_bytes == original.python_heap_bytes
    assert retrieved.rss_bytes == original.rss_bytes


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(db_path: Path) -> None:
    with Cache(db_path=db_path) as c:
        est = _cost_estimate()
        c.put_cost("requests", "3.13", "h", est)
        result = c.get_cost("requests", "3.13", "h")
        assert result is not None
        assert result.module == "requests"
    # Connection should be closed; accessing it would raise ProgrammingError.
    with pytest.raises(sqlite3.ProgrammingError):
        c._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# db_path property
# ---------------------------------------------------------------------------


def test_db_path_property(db_path: Path) -> None:
    with Cache(db_path=db_path) as c:
        assert c.db_path == db_path


# ---------------------------------------------------------------------------
# Same-key overwrite (M-5)
# ---------------------------------------------------------------------------


def test_put_module_info_overwrite(cache: Cache, tmp_path: Path) -> None:
    """put_module_info with the same key twice must return the latest value."""
    p = tmp_path / "mod.py"

    first = _module_info(p)
    cache.put_module_info(p, 1.0, 10, "3.13", "0.1.0", first)

    # Build a second ModuleInfo with a different qualified_name to distinguish it.
    from pyweight.models import ImportRef

    ref2 = ImportRef(
        module="sys",
        names=("argv",),
        alias=None,
        lineno=1,
        scope="module",
        level=0,
        is_type_checking=False,
        is_conditional=False,
        is_star=False,
    )
    second = ModuleInfo(
        path=p,
        qualified_name="updated_module",
        imports=(ref2,),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
        usage_scopes={},
    )
    cache.put_module_info(p, 1.0, 10, "3.13", "0.1.0", second)

    result = cache.get_module_info(p, 1.0, 10, "3.13", "0.1.0")
    assert result is not None
    assert result.qualified_name == "updated_module"


def test_put_cost_overwrite(cache: Cache) -> None:
    """put_cost with the same key twice must return the latest value."""
    first = CostEstimate(
        module="requests",
        dist_name="requests",
        direct_size_bytes=100_000,
        transitive_size_bytes=200_000,
        confidence=Confidence.HIGH,
        provenance="first",
    )
    cache.put_cost("requests", "3.13", "abc123", first)

    second = CostEstimate(
        module="requests",
        dist_name="requests",
        direct_size_bytes=50_000,
        transitive_size_bytes=90_000,
        confidence=Confidence.MEDIUM,
        provenance="second",
    )
    cache.put_cost("requests", "3.13", "abc123", second)

    result = cache.get_cost("requests", "3.13", "abc123")
    assert result is not None
    assert result.provenance == "second"
    assert result.direct_size_bytes == 50_000


# ---------------------------------------------------------------------------
# Double-open fast-path (L-2)
# ---------------------------------------------------------------------------


def test_second_open_hits_fast_path(db_path: Path) -> None:
    """Opening the same db_path twice must not recreate the schema on the second open."""
    with Cache(db_path=db_path) as c1:
        c1.put_cost("requests", "3.13", "h", _cost_estimate())

    # Second open: schema already at current version — data must survive.
    with Cache(db_path=db_path) as c2:
        result = c2.get_cost("requests", "3.13", "h")
        assert result is not None, "Data must survive a second open when schema version matches"
        assert result.module == "requests"
