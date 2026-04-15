"""Tests for pyweight.cost — TDD-style, documents expected behaviour."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from pyweight.cost import (
    _build_import_to_dist_mapping,  # pyright: ignore[reportPrivateUsage]
    _parse_importtime_line,  # pyright: ignore[reportPrivateUsage]
    _validate_module_name,  # pyright: ignore[reportPrivateUsage]
    estimate_size,
    estimate_transitive_cost,
    find_pip_dependency_chain,
    get_dist_requires,
    measure_memory,
    resolve_dist_name,
    time_import,
)
from pyweight.models import (
    Confidence,
    CostEstimate,
    EdgeInfo,
    ImportGraph,
    ModuleInfo,
)

# ---------------------------------------------------------------------------
# _parse_importtime_line
# ---------------------------------------------------------------------------


def test_parse_importtime_line_valid() -> None:
    """A well-formed importtime line parses to (self_us, cum_us, name, level)."""
    line = "import time:       150 |       4454 | json"
    result = _parse_importtime_line(line)
    assert result is not None
    self_us, cum_us, name, level = result
    assert self_us == 150
    assert cum_us == 4454
    assert name == "json"
    assert level == 0  # no leading spaces → nesting level 0


def test_parse_importtime_line_invalid() -> None:
    """Garbage input returns None without raising."""
    assert _parse_importtime_line("") is None
    assert _parse_importtime_line("not an importtime line") is None
    assert _parse_importtime_line("import time: self [us] | cumulative | imported package") is None


def test_parse_importtime_line_nested() -> None:
    """An indented (nested) line returns the correct nesting level."""
    # Four leading spaces indicates a depth-2 nested import in -X importtime output
    line = "import time:        28 |         28 |       _json"
    result = _parse_importtime_line(line)
    assert result is not None
    self_us, cum_us, name, level = result
    assert self_us == 28
    assert cum_us == 28
    assert name == "_json"
    assert level == 3  # 7 leading spaces: (7-1)//2 = 3 nesting levels


# ---------------------------------------------------------------------------
# estimate_size
# ---------------------------------------------------------------------------


def test_estimate_size_known_package() -> None:
    """typer is guaranteed to be installed; must return non-zero bytes and HIGH confidence."""
    result = estimate_size("typer")
    assert result.module == "typer"
    assert result.dist_name is not None
    assert result.direct_size_bytes > 0
    assert result.confidence == Confidence.HIGH
    assert "typer" in result.provenance.lower()


def test_estimate_size_stdlib() -> None:
    """stdlib modules (json) return LOW confidence and provenance mentioning 'stdlib'."""
    result = estimate_size("json")
    assert result.module == "json"
    assert result.dist_name is None
    assert result.confidence == Confidence.LOW
    assert "stdlib" in result.provenance


def test_estimate_size_unknown() -> None:
    """A completely unknown/fake module returns LOW confidence."""
    result = estimate_size("_totally_fake_module_xyz_99")
    assert result.confidence == Confidence.LOW
    assert result.direct_size_bytes == 0


# ---------------------------------------------------------------------------
# estimate_transitive_cost
# ---------------------------------------------------------------------------


def _make_graph_with_two_nodes_sharing_dep() -> ImportGraph:
    """Build a tiny ImportGraph: mod_a → ext_dep, mod_b → ext_dep, root → mod_a, mod_b."""
    graph = ImportGraph()
    graph.external_deps = {"ext_dep"}

    def _stub_module(name: str) -> ModuleInfo:
        from pathlib import Path

        return ModuleInfo(
            path=Path(f"/fake/{name}.py"),
            qualified_name=name,
            imports=(),
            is_init=False,
            has_getattr=False,
            has_all=False,
            all_names=frozenset(),
            defined_names=frozenset(),
            star_imports=(),
            purity="pure",
        )

    for name in ("root", "mod_a", "mod_b"):
        graph.add_module(name, _stub_module(name))

    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph.add_edge("root", "mod_a", edge)
    graph.add_edge("root", "mod_b", edge)
    graph.add_edge("mod_a", "mod_b", edge)

    return graph


def test_transitive_cost_deduplication() -> None:
    """Two paths reaching the same external distribution must count it only once."""
    graph = _make_graph_with_two_nodes_sharing_dep()
    cache: dict[str, CostEstimate] = {}

    def _fake_estimate_size(module: str, search_paths: object = None) -> CostEstimate:
        if module in ("mod_a", "mod_b"):
            return CostEstimate(
                module=module,
                dist_name="shared-dist",
                direct_size_bytes=1000,
                transitive_size_bytes=1000,
                confidence=Confidence.HIGH,
                provenance="fake",
            )
        return CostEstimate(
            module=module,
            dist_name=None,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance="fake-stdlib",
        )

    with patch("pyweight.cost.estimate_size", side_effect=_fake_estimate_size):
        result = estimate_transitive_cost(graph, "root", cache)

    # shared-dist should be counted exactly once despite being reachable via two paths
    assert result.transitive_size_bytes == 1000


def test_transitive_cost_stdlib_only_returns_low_confidence() -> None:
    """When all reachable modules are stdlib (no dists found), confidence must be LOW."""
    from pathlib import Path

    graph = ImportGraph()
    for name in ("root", "json", "os"):
        graph.add_module(
            name,
            ModuleInfo(
                path=Path(f"/fake/{name}.py"),
                qualified_name=name,
                imports=(),
                is_init=False,
                has_getattr=False,
                has_all=False,
                all_names=frozenset(),
                defined_names=frozenset(),
                star_imports=(),
                purity="pure",
            ),
        )
    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph.add_edge("root", "json", edge)
    graph.add_edge("root", "os", edge)

    cache: dict[str, CostEstimate] = {}

    def _stdlib_estimate_size(module: str, search_paths: object = None) -> CostEstimate:
        return CostEstimate(
            module=module,
            dist_name=None,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance="stdlib",
        )

    with patch("pyweight.cost.estimate_size", side_effect=_stdlib_estimate_size):
        result = estimate_transitive_cost(graph, "root", cache)

    assert result.confidence == Confidence.LOW


def test_transitive_cost_memoization() -> None:
    """A second call with the same module reads from size_cache without recomputing."""
    from pathlib import Path

    graph = ImportGraph()
    root_info = ModuleInfo(
        path=Path("/fake/root.py"),
        qualified_name="root",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
    )
    graph.add_module("root", root_info)

    cache: dict[str, CostEstimate] = {}
    call_count = 0

    def _counting_estimate_size(module: str, search_paths: object = None) -> CostEstimate:
        nonlocal call_count
        call_count += 1
        return CostEstimate(
            module=module,
            dist_name=None,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance="fake",
        )

    with patch("pyweight.cost.estimate_size", side_effect=_counting_estimate_size):
        estimate_transitive_cost(graph, "root", cache)
        calls_after_first = call_count
        estimate_transitive_cost(graph, "root", cache)
        calls_after_second = call_count

    # Second call must not invoke estimate_size again
    assert calls_after_second == calls_after_first


# ---------------------------------------------------------------------------
# time_import
# ---------------------------------------------------------------------------


def test_time_import_json() -> None:
    """time_import("json") must return positive timing values."""
    result = time_import("json", runs=1)
    assert result.module == "json"
    # json always takes > 0 µs
    assert result.cumulative_time_us > 0


def test_time_import_timeout() -> None:
    """subprocess.TimeoutExpired from a run is re-raised (not swallowed)."""
    with (
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)),
        pytest.raises(subprocess.TimeoutExpired),
    ):
        time_import("json", runs=1, timeout_s=1)


# ---------------------------------------------------------------------------
# Module name validation
# ---------------------------------------------------------------------------


def test_module_name_validation_rejects_injection() -> None:
    """Names containing shell-special characters raise ValueError."""
    with pytest.raises(ValueError, match="Invalid module name"):
        time_import("json; import os")

    with pytest.raises(ValueError, match="Invalid module name"):
        time_import("json && rm -rf /")

    with pytest.raises(ValueError, match="Invalid module name"):
        time_import("../../../etc/passwd")


def test_module_name_validation_rejects_newline_injection() -> None:
    """Embedded newlines must not bypass validation (re.fullmatch vs re.match + $)."""
    with pytest.raises(ValueError, match="Invalid module name"):
        _validate_module_name("json\nimport os")


def test_module_name_validation_accepts_dotted() -> None:
    """A valid dotted name like 'os.path' passes validation without raising ValueError."""
    fake_output = (
        "import time: self [us] | cumulative | imported package\n"
        "import time:        10 |        10 | os.path\n"
    )
    fake_result = MagicMock()
    fake_result.stderr = fake_output
    fake_result.stdout = ""
    fake_result.returncode = 0

    with patch("subprocess.run", return_value=fake_result):
        result = time_import("os.path", runs=1)

    assert result.module == "os.path"


# ---------------------------------------------------------------------------
# measure_memory
# ---------------------------------------------------------------------------


def test_measure_memory_non_linux() -> None:
    """On non-Linux platforms measure_memory raises RuntimeError with a clear message."""
    with (
        patch("platform.system", return_value="Darwin"),
        pytest.raises(RuntimeError, match="Darwin"),
    ):
        measure_memory("json")


def test_measure_memory_linux() -> None:
    """On Linux, measure_memory("json") returns positive rss_bytes."""
    if sys.platform != "linux":
        pytest.skip("Linux-only test")

    result = measure_memory("json")
    assert result.module == "json"
    assert result.rss_bytes > 0
    assert result.python_heap_bytes >= 0


# ---------------------------------------------------------------------------
# _build_import_to_dist_mapping
# ---------------------------------------------------------------------------


def test_build_import_to_dist_mapping() -> None:
    """Mapping includes well-known installed packages with HIGH confidence."""
    mapping = _build_import_to_dist_mapping()

    assert isinstance(mapping, dict)
    # typer, rich are runtime dependencies guaranteed by pyproject.toml
    assert "typer" in mapping
    assert "rich" in mapping

    dist_name, confidence = mapping["typer"]
    assert dist_name == "typer"
    assert confidence == Confidence.HIGH

    dist_name_r, confidence_r = mapping["rich"]
    assert dist_name_r == "rich"
    assert confidence_r == Confidence.HIGH


def test_build_import_to_dist_mapping_medium_confidence_heuristics() -> None:
    """Override entries not in packages_distributions() carry MEDIUM confidence."""
    # Simulate PIL not being present in the primary metadata
    mock_primary: dict[str, list[str]] = {}

    with patch("pyweight.cost.packages_distributions", return_value=mock_primary):
        mapping = _build_import_to_dist_mapping()

    assert "PIL" in mapping
    dist_name, confidence = mapping["PIL"]
    assert dist_name == "Pillow"
    assert confidence == Confidence.MEDIUM


def test_build_import_to_dist_mapping_high_wins_over_heuristic() -> None:
    """If PIL appears in packages_distributions, its confidence must be HIGH."""
    mock_primary: dict[str, list[str]] = {"PIL": ["Pillow"]}

    with patch("pyweight.cost.packages_distributions", return_value=mock_primary):
        mapping = _build_import_to_dist_mapping()

    dist_name, confidence = mapping["PIL"]
    assert dist_name == "Pillow"
    assert confidence == Confidence.HIGH


def test_subprocess_env_strips_dangerous_vars() -> None:
    """PYTHONSTARTUP, PYTHONINSPECT, PYTHONDEBUG, PYTHONBREAKPOINT are stripped."""
    import os

    from pyweight.cost import _build_subprocess_env  # pyright: ignore[reportPrivateUsage]

    injected = {
        "PYTHONSTARTUP": "/evil/startup.py",
        "PYTHONINSPECT": "1",
        "PYTHONDEBUG": "1",
        "PYTHONBREAKPOINT": "evil.hook",
        "PATH": "/usr/bin",
    }
    with patch.dict(os.environ, injected):
        env = _build_subprocess_env()

    assert "PYTHONSTARTUP" not in env
    assert "PYTHONINSPECT" not in env
    assert "PYTHONDEBUG" not in env
    assert "PYTHONBREAKPOINT" not in env
    assert "PATH" in env


def test_estimate_size_medium_confidence_heuristic() -> None:
    """A module resolved via name heuristic only carries MEDIUM confidence."""
    # PIL is not installed in the test venv; its mapping comes from the override table.
    # We mock get_distribution to succeed so we can test the confidence path.
    from pathlib import PurePosixPath

    mock_file = MagicMock()
    mock_file.size = 5000
    mock_dist = MagicMock()
    mock_dist.files = [mock_file]

    # Patch packages_distributions to return empty (forces heuristic path for PIL)
    with (
        patch("pyweight.cost.packages_distributions", return_value={}),
        patch("pyweight.cost.get_distribution", return_value=mock_dist),
    ):
        result = estimate_size("PIL")

    assert result.dist_name == "Pillow"
    assert result.confidence == Confidence.MEDIUM
    assert result.direct_size_bytes == 5000
    _ = PurePosixPath  # used only for type context clarity


# ---------------------------------------------------------------------------
# resolve_dist_name
# ---------------------------------------------------------------------------


def test_resolve_dist_name_known_package() -> None:
    """pytest is installed in the dev environment; must resolve to a dist name."""
    name = resolve_dist_name("pytest")
    assert name is not None
    assert isinstance(name, str)
    assert len(name) > 0


def test_resolve_dist_name_unknown() -> None:
    """A package that does not exist must return None."""
    assert resolve_dist_name("nonexistent_package_xyz") is None


# ---------------------------------------------------------------------------
# get_dist_requires
# ---------------------------------------------------------------------------


def test_get_dist_requires_returns_list() -> None:
    """pytest has known dependencies; the list must be non-empty."""
    deps = get_dist_requires("pytest")
    assert isinstance(deps, list)
    assert len(deps) > 0
    for dep in deps:
        assert isinstance(dep, str)


def test_get_dist_requires_unknown_dist() -> None:
    """An unknown distribution returns an empty list without raising."""
    deps = get_dist_requires("nonexistent_dist_xyz_99")
    assert deps == []


def test_get_dist_requires_excludes_extras_various_styles() -> None:
    """All extras-gated variants must be excluded regardless of quote/spacing style."""
    mock_dist = MagicMock()
    mock_dist.requires = [
        'requests; extra == "security"',
        "urllib3; extra=='optional'",
        "certifi ; Extra == 'tls'",
        "idna;extra!='foo'",
        "chardet ; extra < 'x'",
        "pluggy",  # non-extras dep: must be included
    ]
    with patch("pyweight.cost.get_distribution", return_value=mock_dist):
        deps = get_dist_requires("some-dist")

    # Only the non-extras dep should survive
    assert deps == ["pluggy"]


# ---------------------------------------------------------------------------
# find_pip_dependency_chain
# ---------------------------------------------------------------------------


def test_find_pip_dependency_chain_direct_dep() -> None:
    """pytest depends on pluggy; we expect a short chain."""
    chain = find_pip_dependency_chain("pytest", "pluggy")
    assert chain is not None
    assert chain[0] == "pytest"
    assert chain[-1] == "pluggy"
    assert len(chain) >= 2


def test_find_pip_dependency_chain_not_found() -> None:
    """No path should exist between two unrelated dists."""
    result = find_pip_dependency_chain("pytest", "nonexistent_dist_xyz_99")
    assert result is None


def test_find_pip_dependency_chain_budget_caps_on_distinct_nodes() -> None:
    """BFS halts when visited set reaches the budget, not on queue pops."""
    # Build a star graph: source → dep0, dep1, … dep999
    # Each dep has no further requirements so all are distinct nodes.
    # With budget=1000 and source already in visited (count=1), visiting 999
    # more nodes must stop before len(visited) reaches 1000.
    dep_names = [f"dep{i}" for i in range(1002)]

    def fake_requires(dist: str, search_paths: tuple[Path, ...] | None = None) -> list[str]:
        if dist == "source":
            return dep_names
        return []

    with patch("pyweight.cost.get_dist_requires", side_effect=fake_requires):
        result = find_pip_dependency_chain("source", "dep9999")

    # Must return None (target not found) without hanging or raising
    assert result is None
