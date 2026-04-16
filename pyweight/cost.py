"""Cost estimation: size, import timing, and memory profiling for Python modules."""

from __future__ import annotations

import platform
import re
import statistics
import subprocess
import sys
from collections import deque
from importlib.metadata import (
    DistributionFinder,
    MetadataPathFinder,
    PackageNotFoundError,
    packages_distributions,
)
from importlib.metadata import distribution as get_distribution
from typing import TYPE_CHECKING

from pyweight.graph import reachable_from
from pyweight.models import (
    Confidence,
    CostEstimate,
    ImportGraph,
    MemoryResult,
    TimingResult,
)

if TYPE_CHECKING:
    import importlib.metadata
    from pathlib import Path

# ---------------------------------------------------------------------------
# Name heuristics for packages with non-matching import names
# ---------------------------------------------------------------------------

_IMPORT_NAME_OVERRIDES: dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "attr": "attrs",
    "gi": "PyGObject",
    "wx": "wxPython",
    "usb": "pyusb",
    "serial": "pyserial",
    "OpenSSL": "pyOpenSSL",
    "dotenv": "python-dotenv",
    "google.cloud": "google-cloud",
}

# ---------------------------------------------------------------------------
# Module name validation
# ---------------------------------------------------------------------------

_MODULE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")


def _validate_module_name(module: str) -> None:
    """Raise ValueError if *module* is not a valid dotted Python identifier."""
    if not _MODULE_NAME_RE.fullmatch(module):
        raise ValueError(
            f"Invalid module name {module!r}: must match "
            r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
        )


# ---------------------------------------------------------------------------
# Foreign-venv metadata helpers
# ---------------------------------------------------------------------------


def _distribution(
    dist_name: str,
    search_paths: tuple[Path, ...] | None,
) -> importlib.metadata.Distribution:
    """Return Distribution for *dist_name* from *search_paths*.

    When *search_paths* is None the default importlib.metadata search is used
    (i.e. pyweight's own environment).
    """
    if search_paths is None:
        return get_distribution(dist_name)
    ctx = DistributionFinder.Context(
        name=dist_name,
        path=[str(p) for p in search_paths],
    )
    dists = list(MetadataPathFinder.find_distributions(ctx))
    if not dists:
        raise PackageNotFoundError(dist_name)
    return dists[0]


def packages_distributions_for(
    search_paths: tuple[Path, ...] | None,
) -> dict[str, list[str]]:
    """Return import-name → dist-name mapping for *search_paths*."""
    if search_paths is None:
        return dict(packages_distributions())
    ctx = DistributionFinder.Context(path=[str(p) for p in search_paths])
    result: dict[str, list[str]] = {}
    for dist in MetadataPathFinder.find_distributions(ctx):
        top_level_text = dist.read_text("top_level.txt")
        if top_level_text:
            for pkg in top_level_text.splitlines():
                pkg = pkg.strip()
                if pkg:
                    result.setdefault(pkg, []).append(dist.name)
        else:
            # Fallback: normalise dist name to import name
            import_name = dist.name.replace("-", "_").lower()
            result.setdefault(import_name, []).append(dist.name)
    return result


# ---------------------------------------------------------------------------
# Import-to-distribution mapping
# ---------------------------------------------------------------------------

# Keyed by search_paths tuple (None = current env).
_dist_mapping_cache: dict[tuple[Path, ...] | None, dict[str, tuple[str, Confidence]]] = {}


def _get_dist_mapping(
    search_paths: tuple[Path, ...] | None = None,
) -> dict[str, tuple[str, Confidence]]:
    if search_paths not in _dist_mapping_cache:
        _dist_mapping_cache[search_paths] = _build_import_to_dist_mapping(search_paths)
    return _dist_mapping_cache[search_paths]


def clear_caches() -> None:
    """Clear all module-level caches. Useful for test isolation."""
    _dist_mapping_cache.clear()


def _build_import_to_dist_mapping(
    search_paths: tuple[Path, ...] | None = None,
) -> dict[str, tuple[str, Confidence]]:
    """Return a dict mapping import top-level name → (dist_name, Confidence).

    Primary source: importlib.metadata.packages_distributions() → Confidence.HIGH.
    Fallback for known overrides not in metadata: _IMPORT_NAME_OVERRIDES → Confidence.MEDIUM.
    """
    result: dict[str, tuple[str, Confidence]] = {}

    primary = packages_distributions_for(search_paths)
    for import_name, dist_names in primary.items():
        if dist_names:
            result[import_name] = (dist_names[0], Confidence.HIGH)

    for import_name, dist_name in _IMPORT_NAME_OVERRIDES.items():
        if import_name not in result:
            result[import_name] = (dist_name, Confidence.MEDIUM)

    return result


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------


def estimate_size(
    module: str,
    search_paths: tuple[Path, ...] | None = None,
) -> CostEstimate:
    """Return a :class:`CostEstimate` for *module*'s installed distribution.

    When *search_paths* is provided, metadata is queried from those
    site-packages directories (the target project's venv) instead of
    pyweight's own environment.
    """
    top_level = module.split(".")[0]

    # Stdlib: no dist metadata exists
    if top_level in sys.stdlib_module_names:
        return CostEstimate(
            module=module,
            dist_name=None,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance="stdlib — no dist metadata",
        )

    mapping = _get_dist_mapping(search_paths)
    mapping_entry = mapping.get(top_level)

    if mapping_entry is None:
        return CostEstimate(
            module=module,
            dist_name=None,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance=f"no distribution found for {top_level!r}",
        )

    dist_name, confidence = mapping_entry
    provenance_parts: list[str] = [f"dist={dist_name!r}"]
    if confidence == Confidence.MEDIUM:
        provenance_parts.append("via name heuristic")

    try:
        dist = _distribution(dist_name, search_paths)
    except PackageNotFoundError:
        return CostEstimate(
            module=module,
            dist_name=dist_name,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance=f"dist={dist_name!r} mapped but not installed",
        )

    files = dist.files
    if files is None:
        return CostEstimate(
            module=module,
            dist_name=dist_name,
            direct_size_bytes=0,
            transitive_size_bytes=0,
            confidence=Confidence.LOW,
            provenance=f"dist={dist_name!r} has no RECORD file",
        )

    total_bytes = sum(f.size for f in files if f.size is not None)
    provenance_parts.append(f"files={len(files)}")
    provenance_parts.append(f"bytes={total_bytes}")

    return CostEstimate(
        module=module,
        dist_name=dist_name,
        direct_size_bytes=total_bytes,
        transitive_size_bytes=total_bytes,
        confidence=confidence,
        provenance=", ".join(provenance_parts),
    )


def estimate_transitive_cost(
    graph: ImportGraph,
    module: str,
    size_cache: dict[str, CostEstimate],
    search_paths: tuple[Path, ...] | None = None,
) -> CostEstimate:
    """Sum external dependency sizes reachable from *module*.

    Each external distribution is counted at most once regardless of how many
    paths reach it.  Results are memoized in *size_cache*.

    When *search_paths* is provided, metadata is queried from the target
    project's site-packages directories.
    """
    if module in size_cache:
        return size_cache[module]

    reachable = reachable_from(graph, module, include_type_checking=False)

    direct = estimate_size(module, search_paths)
    total_bytes = direct.direct_size_bytes
    seen_dists: set[str] = set()
    min_confidence: Confidence | None = None

    if direct.dist_name:
        seen_dists.add(direct.dist_name)
        min_confidence = direct.confidence

    for mod in reachable:
        est = estimate_size(mod, search_paths)
        dist = est.dist_name
        if dist is None:
            continue
        if dist in seen_dists:
            continue
        seen_dists.add(dist)
        total_bytes += est.direct_size_bytes
        if min_confidence is None or est.confidence.rank() < min_confidence.rank():
            min_confidence = est.confidence

    # Also account for external dependencies imported by reachable modules
    all_reachable = reachable | {module}
    for mod in all_reachable:
        for ext_top in graph.module_external_imports.get(mod, set()):
            ext_est = estimate_size(ext_top, search_paths)
            ext_dist = ext_est.dist_name
            if ext_dist is None:
                continue
            if ext_dist in seen_dists:
                continue
            seen_dists.add(ext_dist)
            total_bytes += ext_est.direct_size_bytes
            if min_confidence is None or ext_est.confidence.rank() < min_confidence.rank():
                min_confidence = ext_est.confidence

    final_confidence = min_confidence if min_confidence is not None else Confidence.LOW

    result = CostEstimate(
        module=module,
        dist_name=direct.dist_name,
        direct_size_bytes=direct.direct_size_bytes,
        transitive_size_bytes=total_bytes,
        confidence=final_confidence,
        provenance=f"transitive sum of {len(seen_dists)} dist(s) reachable from {module!r}",
    )
    size_cache[module] = result
    return result


# ---------------------------------------------------------------------------
# Import timing via -X importtime
# ---------------------------------------------------------------------------

_IMPORTTIME_RE = re.compile(r"^import time:\s*(\d+)\s*\|\s*(\d+)\s*\|(\s*)(.*?)\s*$")


def _parse_importtime_line(line: str) -> tuple[int, int, str, int] | None:
    """Parse one line of ``-X importtime`` stderr output.

    Returns ``(self_us, cumulative_us, name, nesting_level)`` or ``None``.

    The ``-X importtime`` format is::

        import time: <self_us> | <cumulative_us> | <indent><name>

    where indent is at least one space (top-level = 1 space, each additional
    nesting level adds 2 spaces).  nesting_level = (len(indent) - 1) // 2.
    """
    match = _IMPORTTIME_RE.match(line)
    if match is None:
        return None
    self_us = int(match.group(1))
    cumulative_us = int(match.group(2))
    indent = match.group(3)
    name = match.group(4)
    if not name:
        return None
    # indent is at minimum 1 space for top-level entries; each extra 2 spaces
    # represents one additional level of nesting.
    nesting_level = (len(indent) - 1) // 2
    return (self_us, cumulative_us, name, nesting_level)


def time_import(
    module: str,
    runs: int = 3,
    timeout_s: int = 30,
    target_python: Path | None = None,
) -> TimingResult:
    """Run ``python -X importtime -c "import <module>"`` N times.

    When *target_python* is provided, that interpreter is used instead of
    ``sys.executable`` — allowing timing against the target project's venv.
    """
    _validate_module_name(module)

    python = str(target_python) if target_python is not None else sys.executable
    env = _build_subprocess_env()
    cmd = [python, "-X", "importtime", "-c", f"import {module}"]

    all_self_times: list[int] = []
    all_cumulative_times: list[int] = []
    seen_deps: set[str] = set()

    for _ in range(runs):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired(
                cmd, timeout_s, stderr=f"time_import({module!r}) timed out"
            ) from exc

        self_us, cum_us, deps = _extract_module_timing(result.stderr, module)
        all_self_times.append(self_us)
        all_cumulative_times.append(cum_us)
        seen_deps.update(deps)

    median_self = int(statistics.median(all_self_times))
    median_cum = int(statistics.median(all_cumulative_times))

    return TimingResult(
        module=module,
        self_time_us=median_self,
        cumulative_time_us=median_cum,
        dependencies=tuple(sorted(seen_deps - {module})),
    )


def _build_subprocess_env() -> dict[str, str]:
    """Build a clean environment for importtime subprocesses."""
    import os

    _strip = frozenset({"PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONDEBUG", "PYTHONBREAKPOINT"})
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _strip:
            continue
        env[key] = value
    return env


def _extract_module_timing(stderr: str, module: str) -> tuple[int, int, list[str]]:
    """Extract self/cumulative timing for *module* from importtime stderr.

    Returns (self_us, cumulative_us, dependency_names).

    Raises:
        RuntimeError: if the target module line is not found in the output.
    """
    self_us: int | None = None
    cum_us: int | None = None
    deps: list[str] = []

    for line in stderr.splitlines():
        parsed = _parse_importtime_line(line)
        if parsed is None:
            continue
        s, c, name, _level = parsed
        deps.append(name)
        if name == module:
            self_us = s
            cum_us = c

    if self_us is None or cum_us is None:
        raise RuntimeError(
            f"_extract_module_timing: module {module!r} not found in importtime output"
        )

    return self_us, cum_us, deps


# ---------------------------------------------------------------------------
# Memory profiling via subprocess (Linux-only, opt-in)
# ---------------------------------------------------------------------------

_MEMORY_PROBE_SCRIPT = """\
import sys
import tracemalloc

tracemalloc.start()
__import__(sys.argv[1])
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_kb = 0
try:
    with open("/proc/self/status") as _f:
        for _line in _f:
            if _line.startswith("VmRSS:"):
                rss_kb = int(_line.split()[1])
                break
except OSError:
    pass

print(peak)
print(rss_kb * 1024)
"""


def measure_memory(
    module: str,
    timeout_s: int = 30,
    target_python: Path | None = None,
) -> MemoryResult:
    """Measure peak tracemalloc heap and RSS for importing *module*.

    Linux-only: reads /proc/self/status for VmRSS.
    When *target_python* is provided, that interpreter is used.
    """
    _validate_module_name(module)

    if platform.system() != "Linux":
        raise RuntimeError(
            f"measure_memory() requires Linux; current platform is {platform.system()!r}"
        )

    python = str(target_python) if target_python is not None else sys.executable
    env = _build_subprocess_env()

    try:
        result = subprocess.run(
            [python, "-c", _MEMORY_PROBE_SCRIPT, module],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise subprocess.TimeoutExpired(
            [sys.executable], timeout_s, stderr=f"measure_memory({module!r}) timed out"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"measure_memory({module!r}): subprocess exited with code"
            f" {result.returncode}: {result.stderr!r}"
        )

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError(
            f"measure_memory({module!r}): unexpected subprocess output: {result.stdout!r}"
        )

    try:
        python_heap_bytes = int(lines[0])
        rss_bytes = int(lines[1])
    except ValueError as exc:
        raise RuntimeError(
            f"measure_memory({module!r}): could not parse output: {result.stdout!r}"
        ) from exc

    return MemoryResult(
        module=module,
        python_heap_bytes=python_heap_bytes,
        rss_bytes=rss_bytes,
    )


# ---------------------------------------------------------------------------
# Distribution name resolution
# ---------------------------------------------------------------------------

_DIST_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")

# Matches any marker expression that gates a requirement on an extra, e.g.:
#   extra == "security", extra=='foo', Extra != "bar", extra<"x"
_EXTRA_MARKER_RE = re.compile(r"\bextra\s*[!=<>]", re.IGNORECASE)


def resolve_dist_name(
    import_or_package_name: str,
    search_paths: tuple[Path, ...] | None = None,
) -> str | None:
    """Resolve an import name or package name to its distribution name.

    Tries ``importlib.metadata.distribution(name)`` directly first, then falls
    back to the ``packages_distributions()`` mapping.  Returns ``None`` when the
    name cannot be resolved to any installed distribution.
    """
    try:
        dist = _distribution(import_or_package_name, search_paths)
        return dist.name
    except PackageNotFoundError:
        pass

    pkg_map = packages_distributions_for(search_paths)
    dists = pkg_map.get(import_or_package_name)
    if dists:
        return dists[0]

    return None


def get_dist_requires(
    dist_name: str,
    search_paths: tuple[Path, ...] | None = None,
) -> list[str]:
    """Return non-extra dependency distribution names for *dist_name*.

    Parses PEP 508 requirement strings from the distribution metadata and
    skips entries that are optional extras (``; extra ==``).  Returns an empty
    list when the distribution is not installed or has no dependencies.
    """
    try:
        dist = _distribution(dist_name, search_paths)
    except PackageNotFoundError:
        return []

    requires = dist.requires
    if requires is None:
        return []

    result: list[str] = []
    for req in requires:
        if _EXTRA_MARKER_RE.search(req):
            continue
        match = _DIST_NAME_RE.match(req)
        if match:
            result.append(match.group(1))

    return result


def find_pip_dependency_chain(
    source_dist: str,
    target_dist: str,
    search_paths: tuple[Path, ...] | None = None,
) -> tuple[str, ...] | None:
    """Return the BFS shortest dependency path from *source_dist* to *target_dist*.

    Uses ``get_dist_requires`` at each hop.  Returns ``None`` when no path
    exists.  Visits at most 1000 nodes to guard against pathological graphs.
    """
    if source_dist == target_dist:
        return (source_dist,)

    visited: set[str] = {source_dist.lower()}
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source_dist, (source_dist,))])
    budget = 1000

    while queue:
        current, path = queue.popleft()
        for dep in get_dist_requires(current, search_paths):
            dep_lower = dep.lower()
            if dep_lower in visited:
                continue
            visited.add(dep_lower)
            if len(visited) >= budget:
                return None
            new_path = (*path, dep)
            if dep_lower == target_dist.lower():
                return new_path
            queue.append((dep, new_path))

    return None
