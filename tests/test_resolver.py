"""Tests for pyweight.resolver.PackageResolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyweight.models import Confidence
from pyweight.resolver import PackageResolver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_resolver(sample_package_path: Path) -> PackageResolver:
    return PackageResolver(sample_package_path)


@pytest.fixture
def ns_resolver(namespace_package_path: Path) -> PackageResolver:
    return PackageResolver(namespace_package_path)


# ---------------------------------------------------------------------------
# all_modules discovery
# ---------------------------------------------------------------------------


def test_all_modules_includes_package_init(sample_resolver: PackageResolver) -> None:
    modules = sample_resolver.all_modules
    assert "sample_package" in modules
    assert modules["sample_package"].name == "__init__.py"


def test_all_modules_includes_submodules(sample_resolver: PackageResolver) -> None:
    modules = sample_resolver.all_modules
    expected = {
        "sample_package.core",
        "sample_package.models",
        "sample_package.hub",
        "sample_package.heavy",
        "sample_package.guarded",
        "sample_package.conditional",
        "sample_package.side_effects",
        "sample_package.deferred_candidate",
        "sample_package.utils",
        "sample_package.utils.helpers",
    }
    assert expected <= set(modules.keys())


def test_all_modules_returns_copy(sample_resolver: PackageResolver) -> None:
    """Mutating the returned dict must not affect the resolver's internal state."""
    m1 = sample_resolver.all_modules
    m1["injected"] = Path("/tmp/fake.py")
    m2 = sample_resolver.all_modules
    assert "injected" not in m2


# ---------------------------------------------------------------------------
# Absolute import resolution
# ---------------------------------------------------------------------------


def test_resolve_absolute_known(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    path, conf = sample_resolver.resolve("sample_package.core", None, level=0)
    assert path == sample_package_path / "core.py"
    assert conf == Confidence.HIGH


def test_resolve_absolute_package_init(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    path, conf = sample_resolver.resolve("sample_package", None, level=0)
    assert path == sample_package_path / "__init__.py"
    assert conf == Confidence.HIGH


def test_resolve_absolute_nested(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    path, conf = sample_resolver.resolve("sample_package.utils.helpers", None, level=0)
    assert path == sample_package_path / "utils" / "helpers.py"
    assert conf == Confidence.HIGH


def test_resolve_absolute_unknown_returns_none(sample_resolver: PackageResolver) -> None:
    path, conf = sample_resolver.resolve("nonexistent_module", None, level=0)
    assert path is None
    assert conf == Confidence.LOW


def test_resolve_stdlib_returns_none(sample_resolver: PackageResolver) -> None:
    path, conf = sample_resolver.resolve("os", None, level=0)
    assert path is None
    assert conf == Confidence.LOW


def test_resolve_third_party_returns_none(sample_resolver: PackageResolver) -> None:
    path, conf = sample_resolver.resolve("numpy", None, level=0)
    assert path is None
    assert conf == Confidence.LOW


# ---------------------------------------------------------------------------
# Relative import resolution
# ---------------------------------------------------------------------------


def test_resolve_relative_level1_sibling(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    """from .core import CoreClass — level=1, from models.py"""
    # models.py lives at sample_package.models; from .core means sample_package.core
    path, conf = sample_resolver.resolve("core", "sample_package.models", level=1)
    assert path == sample_package_path / "core.py"
    assert conf == Confidence.HIGH


def test_resolve_relative_level2_from_utils_helpers(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    """from .. import core — level=2, from sample_package.utils.helpers"""
    # base_parts = ["sample_package"], module_name="" → resolves to "sample_package"
    path, conf = sample_resolver.resolve("", "sample_package.utils.helpers", level=2)
    assert path == sample_package_path / "__init__.py"
    assert conf == Confidence.HIGH


def test_resolve_relative_level1_package_itself(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    """from . import models — level=1, module_name="" from sample_package.hub"""
    path, conf = sample_resolver.resolve("", "sample_package.hub", level=1)
    assert path == sample_package_path / "__init__.py"
    assert conf == Confidence.HIGH


def test_resolve_relative_none_from_module(sample_resolver: PackageResolver) -> None:
    """Relative import with no from_module context → (None, LOW)."""
    path, conf = sample_resolver.resolve("core", None, level=1)
    assert path is None
    assert conf == Confidence.LOW


def test_resolve_relative_level_too_deep(sample_resolver: PackageResolver) -> None:
    """Level deeper than the module's nesting returns (None, LOW)."""
    path, conf = sample_resolver.resolve("x", "sample_package", level=5)
    assert path is None
    assert conf == Confidence.LOW


# ---------------------------------------------------------------------------
# is_external
# ---------------------------------------------------------------------------


def test_is_external_stdlib(sample_resolver: PackageResolver) -> None:
    assert sample_resolver.is_external("os") is True


def test_is_external_third_party(sample_resolver: PackageResolver) -> None:
    assert sample_resolver.is_external("numpy") is True


def test_is_external_own_module(sample_resolver: PackageResolver) -> None:
    assert sample_resolver.is_external("sample_package.core") is False


def test_is_external_own_top_level(sample_resolver: PackageResolver) -> None:
    assert sample_resolver.is_external("sample_package") is False


# ---------------------------------------------------------------------------
# Namespace packages
# ---------------------------------------------------------------------------


def test_namespace_pkg_submodules_discovered(ns_resolver: PackageResolver) -> None:
    """Modules under namespace-package subdirs are discovered."""
    modules = ns_resolver.all_modules
    # Qualified names are prefixed by the root dir name: "namespace_package.subpkg_a" etc.
    assert any("subpkg_a" in k for k in modules)
    assert any("subpkg_b" in k for k in modules)


def test_namespace_pkg_child_module_confidence_low(ns_resolver: PackageResolver) -> None:
    """Modules under a namespace package (no __init__.py at root) resolve with Confidence.LOW."""
    modules = ns_resolver.all_modules
    # e.g. "namespace_package.subpkg_a.mod" — parent "namespace_package" is a namespace pkg
    ns_modules = [k for k in modules if "subpkg" in k and k.endswith(".mod")]
    assert ns_modules, f"Expected namespace-pkg child modules, got: {list(modules)}"
    for name in ns_modules:
        _path, conf = ns_resolver.resolve(name, None, level=0)
        # namespace_package has no __init__.py → _has_namespace_ancestor returns True → LOW
        assert conf == Confidence.LOW


def test_namespace_root_not_in_all_modules(ns_resolver: PackageResolver) -> None:
    """The namespace root dir itself has no __init__.py → not in all_modules."""
    # The namespace_package dir has no __init__.py; it's a namespace package.
    modules = ns_resolver.all_modules
    # "namespace_package" should NOT be in modules (no __init__.py at root)
    assert "namespace_package" not in modules


def test_qname_for_path_known(sample_resolver: PackageResolver, sample_package_path: Path) -> None:
    """qname_for_path returns the qualified name for a known path."""
    assert sample_resolver.qname_for_path(sample_package_path / "core.py") == "sample_package.core"


def test_qname_for_path_init(sample_resolver: PackageResolver, sample_package_path: Path) -> None:
    """qname_for_path returns the package qname for an __init__.py path."""
    assert sample_resolver.qname_for_path(sample_package_path / "__init__.py") == "sample_package"


def test_qname_for_path_unknown(
    sample_resolver: PackageResolver,
) -> None:
    """qname_for_path returns None for a path not in the package."""
    assert sample_resolver.qname_for_path(Path("/not/in/package.py")) is None


# ---------------------------------------------------------------------------
# __init__.py relative import resolution
# ---------------------------------------------------------------------------


def test_resolve_relative_from_init_level1(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    """from .helpers import X inside utils/__init__.py (level=1, is_init=True).

    from_module="sample_package.utils" which IS the package, so level=1 must
    stay within that package: resolves to sample_package.utils.helpers.
    """
    path, conf = sample_resolver.resolve("helpers", "sample_package.utils", level=1, is_init=True)
    assert path == sample_package_path / "utils" / "helpers.py"
    assert conf == Confidence.HIGH


def test_resolve_relative_from_regular_file_level1_unchanged(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    """from .core import X inside models.py (level=1, is_init=False, default).

    Regular file: from_module="sample_package.models", strip=1 → base="sample_package".
    Resolves to sample_package.core — unchanged by the is_init fix.
    """
    path, conf = sample_resolver.resolve("core", "sample_package.models", level=1, is_init=False)
    assert path == sample_package_path / "core.py"
    assert conf == Confidence.HIGH


def test_resolve_relative_from_init_level2(
    sample_resolver: PackageResolver, sample_package_path: Path
) -> None:
    """from .. import core inside utils/__init__.py (level=2, is_init=True).

    from_module="sample_package.utils", is_init=True → strip=1 → base=["sample_package"].
    Resolves to sample_package.core.
    """
    path, conf = sample_resolver.resolve("core", "sample_package.utils", level=2, is_init=True)
    assert path == sample_package_path / "core.py"
    assert conf == Confidence.HIGH


def test_namespace_pkg_logged(
    namespace_package_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Namespace packages should produce an info log message."""
    import logging

    with caplog.at_level(logging.INFO, logger="pyweight.resolver"):
        PackageResolver(namespace_package_path)
    # The namespace root dir (no __init__.py) should trigger a log
    info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert info_msgs, "Expected at least one info log for namespace package detection"
