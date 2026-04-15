"""High-level analysis functions built on ImportGraph primitives."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

from pyweight.cost import find_pip_dependency_chain, resolve_dist_name

if TYPE_CHECKING:
    from pathlib import Path
from pyweight.graph import fan_in, reachable_from, shortest_path, to_networkx
from pyweight.models import (
    BarrelInfo,
    Confidence,
    CostEstimate,
    DeadExport,
    DeferralCandidate,
    EscapeLevel,
    ExternalWhyResult,
    Hotspot,
    HubDependency,
    ImportGraph,
    ImportRef,
    Purity,
    ReachabilityResult,
    RiskLevel,
    WhyResult,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eager_import_count(graph: ImportGraph, module: str) -> int:
    """Count module-level, non-TYPE_CHECKING, non-conditional imports for a module."""
    info = graph.modules.get(module)
    if info is None:
        return 0
    return sum(
        1
        for ref in info.imports
        if ref.scope == "module" and not ref.is_type_checking and not ref.is_conditional
    )


def _reexported_count(graph: ImportGraph, module: str) -> int:
    """Count re-exported symbols: defined_names + explicitly imported names."""
    info = graph.modules.get(module)
    if info is None:
        return 0
    explicitly_imported: set[str] = set()
    for ref in info.imports:
        if not ref.is_star:
            for name in ref.names:
                explicitly_imported.add(name)
    return len(info.defined_names | explicitly_imported)


def _fan_out_runtime(graph: ImportGraph, module: str) -> int:
    """Count runtime (non-TYPE_CHECKING) outgoing edges for a module."""
    count = 0
    for _target, edge_info in graph.edges.get(module, {}).items():
        if not edge_info.is_type_checking:
            count += 1
    return count


def _make_aggregate_cost(
    module: str,
    modules_set: frozenset[str],
    costs: dict[str, CostEstimate],
    graph: ImportGraph | None = None,
    search_paths: tuple[Path, ...] | None = None,
) -> CostEstimate | None:
    """Aggregate cost for a set of modules with external dependency deduplication.

    Collects all unique external distributions imported (directly or transitively)
    by any module in *modules_set* and sums their installed sizes.  Internal
    module source sizes are negligible and not included.
    """
    from pyweight.cost import estimate_size

    # Collect all unique external top-level imports across the module set
    ext_tops: set[str] = set()
    if graph is not None:
        for mod in modules_set:
            ext_tops.update(graph.module_external_imports.get(mod, set()))

    # Sum unique dist sizes
    seen_dists: set[str] = set()
    total_bytes = 0
    min_confidence: Confidence | None = None

    for ext_top in ext_tops:
        est = estimate_size(ext_top, search_paths)
        dist = est.dist_name
        if dist is None:
            continue
        if dist in seen_dists:
            continue
        seen_dists.add(dist)
        total_bytes += est.direct_size_bytes
        est_rank = est.confidence.rank()
        if min_confidence is None or est_rank < min_confidence.rank():
            min_confidence = est.confidence

    if min_confidence is None:
        return None

    return CostEstimate(
        module=module,
        dist_name=None,
        direct_size_bytes=total_bytes,
        transitive_size_bytes=total_bytes,
        confidence=min_confidence,
        provenance=f"aggregate over {len(modules_set)} modules, {len(seen_dists)} ext dist(s)",
    )


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def find_barrels(graph: ImportGraph) -> list[BarrelInfo]:
    """Return all __init__.py barrel modules, sorted by reexported_count descending."""
    results: list[BarrelInfo] = []

    for name, info in graph.modules.items():
        if not info.is_init:
            continue

        eager = _eager_import_count(graph, name)
        reexported = _reexported_count(graph, name)

        results.append(
            BarrelInfo(
                path=info.path,
                qualified_name=name,
                star_imports=info.star_imports,
                eager_imports=eager,
                reexported_count=reexported,
                has_lazy_loading=info.has_getattr,
                has_all=info.has_all,
            )
        )

    results.sort(key=lambda b: b.reexported_count, reverse=True)
    return results


def _bound_names(import_ref: ImportRef) -> tuple[str, ...]:
    """Return the names bound in the local namespace by this import statement.

    - ``import json`` → (``json``,)  [top-level module name or alias]
    - ``from x import Foo`` → (``Foo``,) or alias if present
    - ``from x import *`` → ()  [star import; no specific bound names]
    """
    if import_ref.is_star:
        return ()
    if import_ref.names:
        # from X import a [as b]
        if import_ref.alias:
            return (import_ref.alias,)
        return import_ref.names
    # plain ``import X [as Y]``
    if import_ref.alias:
        return (import_ref.alias,)
    # ``import a.b.c`` binds only ``a`` in the local namespace
    return (import_ref.module.split(".")[0],)


def classify_escape_level(
    import_ref: ImportRef,
    usage_scopes: dict[str, frozenset[str]],
) -> EscapeLevel:
    """Classify how far an imported name escapes its import scope.

    For multi-name imports, classifies each name independently and returns the
    maximum (highest-escaping) level.  Names with no usage data default to
    MODULE_SCOPE (conservative).
    """
    if import_ref.is_star:
        return EscapeLevel.MODULE_SCOPE

    names = _bound_names(import_ref)
    if not names:
        return EscapeLevel.MODULE_SCOPE

    def _level_for_name(name: str) -> EscapeLevel:
        scopes = usage_scopes.get(name)
        if scopes is None:
            # No usage data: conservative default
            return EscapeLevel.MODULE_SCOPE

        # REEXPORTED: listed in __all__ or explicitly re-exported
        if "__all__" in scopes or "reexported" in scopes:
            return EscapeLevel.REEXPORTED

        max_level = EscapeLevel.FUNCTION_LOCAL

        for scope in scopes:
            if scope == "module":
                return EscapeLevel.MODULE_SCOPE
            parts = scope.split(".")
            # Single-part PascalCase scope → class body.  Multi-part or
            # snake_case → function body (stays at FUNCTION_LOCAL).
            if len(parts) == 1 and parts[0][:1].isupper() and max_level < EscapeLevel.CLASS_BODY:
                max_level = EscapeLevel.CLASS_BODY

        return max_level

    levels = [_level_for_name(name) for name in names]
    return max(levels)


def find_deferral_candidates(
    graph: ImportGraph,
    costs: dict[str, CostEstimate] | None = None,
    entry_point: str | None = None,
) -> list[DeferralCandidate]:
    """Find imports that could be moved inside the function/class that uses them.

    Only FUNCTION_LOCAL and CLASS_BODY escape levels are considered deferrable.
    Sorted by estimated cost descending.
    """
    results: list[DeferralCandidate] = []

    for module_name, info in graph.modules.items():
        for ref in info.imports:
            # Only module-level, non-TYPE_CHECKING, non-conditional imports are candidates
            if ref.scope != "module":
                continue
            if ref.is_type_checking:
                continue
            if ref.is_conditional:
                continue

            level = classify_escape_level(ref, info.usage_scopes)

            if level not in (EscapeLevel.FUNCTION_LOCAL, EscapeLevel.CLASS_BODY):
                continue

            # Risk from purity
            purity = info.purity
            if purity == Purity.PURE:
                risk: str = RiskLevel.SAFE
            elif purity == Purity.SIDE_EFFECTS:
                risk = RiskLevel.HAS_SIDE_EFFECTS
            else:
                risk = RiskLevel.UNKNOWN

            # Find all scopes where the bound names are used; use _bound_names to
            # correctly handle plain `import X` (empty ref.names) style imports.
            bound = _bound_names(ref)
            used_scopes: set[str] = set()
            for name in bound:
                scopes = info.usage_scopes.get(name)
                if scopes:
                    used_scopes.update(scopes)

            # Estimated cost for the imported module
            estimated_cost: CostEstimate | None = None
            if costs is not None:
                estimated_cost = costs.get(ref.module)

            # Evidence trail via shortest path from entry_point
            evidence: list[str] = []
            if entry_point is not None:
                path = shortest_path(graph, entry_point, module_name)
                if path is not None:
                    evidence = list(path)

            # Confidence: HIGH if we have usage data, MEDIUM otherwise
            has_usage = any(name in info.usage_scopes for name in bound)
            confidence = Confidence.HIGH if has_usage else Confidence.MEDIUM

            results.append(
                DeferralCandidate(
                    file=info.path,
                    import_ref=ref,
                    used_in_scopes=frozenset(used_scopes),
                    escape_level=level,
                    estimated_cost=estimated_cost,
                    confidence=confidence,
                    risk=risk,
                    evidence=tuple(evidence),
                )
            )

    results.sort(
        key=lambda c: c.estimated_cost.transitive_size_bytes if c.estimated_cost else 0,
        reverse=True,
    )
    return results


def find_hub_dependencies(
    graph: ImportGraph,
    costs: dict[str, CostEstimate] | None = None,
    min_fan: int = 3,
) -> list[HubDependency]:
    """Find modules with high fan-in * fan-out (hub score), excluding TYPE_CHECKING edges.

    Only modules where BOTH fan_in >= min_fan AND fan_out >= min_fan are flagged.
    Sorted by hub_score descending.
    """
    results: list[HubDependency] = []

    for module_name, info in graph.modules.items():
        fi = fan_in(graph, module_name, include_type_checking=False)
        fo = _fan_out_runtime(graph, module_name)

        if fi < min_fan or fo < min_fan:
            continue

        score = float(fi * fo)
        transitive_cost: CostEstimate | None = None
        if costs is not None:
            transitive_cost = costs.get(module_name)

        results.append(
            HubDependency(
                module=module_name,
                fan_in=fi,
                fan_out=fo,
                hub_score=score,
                is_init=info.is_init,
                transitive_cost=transitive_cost,
            )
        )

    results.sort(key=lambda h: h.hub_score, reverse=True)
    return results


def find_hotspots(
    graph: ImportGraph,
    costs: dict[str, CostEstimate],
    min_cost_bytes: int = 0,
) -> list[Hotspot]:
    """Find modules that are both expensive and widely imported.

    score = fan_in * transitive_size_bytes.
    Sorted by score descending.
    """
    results: list[Hotspot] = []

    for module_name in graph.modules:
        cost = costs.get(module_name)
        if cost is None:
            continue

        if cost.transitive_size_bytes < min_cost_bytes:
            continue

        fi = fan_in(graph, module_name, include_type_checking=False)
        score = float(fi * cost.transitive_size_bytes)

        # Gather evidence: shortest paths from importers
        evidence: list[str] = []
        for importer in graph.reverse_edges.get(module_name, {}):
            path = shortest_path(graph, importer, module_name)
            if path is not None:
                evidence.append(" -> ".join(path))
            if len(evidence) >= 3:
                break

        results.append(
            Hotspot(
                module=module_name,
                fan_in=fi,
                transitive_cost=cost,
                score=score,
                evidence=tuple(evidence),
            )
        )

    results.sort(key=lambda h: h.score, reverse=True)
    return results


def analyze_reachability(
    graph: ImportGraph,
    entry_point: str,
    costs: dict[str, CostEstimate],
    search_paths: tuple[Path, ...] | None = None,
) -> ReachabilityResult:
    """BFS from entry_point to determine required vs unreachable modules.

    Root causes identify __init__.py barrels whose eager imports pull in
    unreachable modules.
    """
    all_modules: frozenset[str] = frozenset(graph.modules.keys())

    if entry_point not in graph.modules:
        return ReachabilityResult(
            entry_point=entry_point,
            required_modules=frozenset(),
            unreachable_modules=all_modules,
            required_cost=None,
            unreachable_cost=_make_aggregate_cost(
                entry_point,
                all_modules,
                costs,
                graph,
                search_paths,
            ),
            root_causes=(),
            import_paths={},
            confidence=Confidence.LOW,
            unknown_origins=(),
        )

    # BFS excluding TYPE_CHECKING edges
    reachable = reachable_from(graph, entry_point, include_type_checking=False)
    required: frozenset[str] = frozenset(reachable) | {entry_point}
    unreachable: frozenset[str] = all_modules - required

    # Build import_paths for required modules
    import_paths: dict[str, tuple[str, ...]] = {}
    for mod in required:
        if mod == entry_point:
            import_paths[mod] = (entry_point,)
            continue
        path = shortest_path(graph, entry_point, mod)
        if path is not None:
            import_paths[mod] = path

    # Root causes: barrels with eager imports that pull in unreachable modules
    root_causes: list[str] = []
    unknown_origins: list[str] = []

    for mod in required:
        info = graph.modules[mod]
        if info.is_init:
            for ref in info.imports:
                if ref.is_type_checking or ref.is_conditional:
                    continue
                if ref.scope != "module":
                    continue
                if ref.is_star:
                    # Star import: if source is unresolved, track as unknown origin
                    if ref.module and ref.module not in graph.modules:
                        unknown_origins.append(ref.module)
                else:
                    # Check if the specific resolved target of this ref is unreachable
                    resolved_target = _resolve_import_module(ref, mod, is_init=info.is_init)
                    if (
                        resolved_target is not None
                        and resolved_target in unreachable
                        and mod not in root_causes
                    ):
                        root_causes.append(mod)

    required_cost = _make_aggregate_cost(
        entry_point,
        required,
        costs,
        graph,
        search_paths,
    )
    unreachable_cost = _make_aggregate_cost(
        entry_point,
        unreachable,
        costs,
        graph,
        search_paths,
    )

    # Overall confidence: LOW if any cost estimate is LOW or missing
    confidence = Confidence.HIGH
    for mod in required:
        est = costs.get(mod)
        if est is None or est.confidence == Confidence.LOW:
            confidence = Confidence.LOW
            break
        if est.confidence == Confidence.MEDIUM and confidence == Confidence.HIGH:
            confidence = Confidence.MEDIUM

    return ReachabilityResult(
        entry_point=entry_point,
        required_modules=required,
        unreachable_modules=unreachable,
        required_cost=required_cost,
        unreachable_cost=unreachable_cost,
        root_causes=tuple(root_causes),
        import_paths=import_paths,
        confidence=confidence,
        unknown_origins=tuple(unknown_origins),
    )


def _all_paths_bfs(
    graph: ImportGraph,
    source: str,
    target: str,
    max_paths: int = 5,
    max_expansions: int = 10_000,
) -> list[tuple[str, ...]]:
    """BFS to find up to max_paths distinct paths from source to target.

    Stops after max_expansions queue pops to bound memory on dense graphs.
    """
    if source == target:
        return [(source,)]

    found: list[tuple[str, ...]] = []
    # Queue of (current_node, path_so_far)
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
    expansions = 0

    while queue and len(found) < max_paths and expansions < max_expansions:
        current, path = queue.popleft()
        expansions += 1
        for neighbour, edge_info in graph.edges.get(current, {}).items():
            if edge_info.is_type_checking:
                continue
            if neighbour in path:
                continue
            new_path = (*path, neighbour)
            if neighbour == target:
                found.append(new_path)
                if len(found) >= max_paths:
                    break
            else:
                queue.append((neighbour, new_path))

    return found


def explain_import(
    graph: ImportGraph,
    entry_point: str,
    target: str,
    costs: dict[str, CostEstimate],
    max_paths: int = 5,
) -> WhyResult:
    """Explain why entry_point imports target, showing all paths.

    Paths are ranked by cost (heaviest first).  The critical edge per path is
    the edge with the highest-cost target.  suggested_cuts uses networkx minimum
    cut if available, else None.
    """
    if entry_point not in graph.modules or target not in graph.modules:
        return WhyResult(
            entry_point=entry_point,
            target=target,
            paths=(),
            critical_edges=(),
            suggested_cuts=None,
            total_cost=None,
        )

    raw_paths = _all_paths_bfs(graph, entry_point, target, max_paths=max_paths)

    if not raw_paths:
        return WhyResult(
            entry_point=entry_point,
            target=target,
            paths=(),
            critical_edges=(),
            suggested_cuts=None,
            total_cost=None,
        )

    def _path_cost(path: tuple[str, ...]) -> int:
        total = 0
        for node in path:
            est = costs.get(node)
            if est is not None:
                total += est.transitive_size_bytes
        return total

    # Sort paths: heaviest cost first
    raw_paths.sort(key=_path_cost, reverse=True)

    # Critical edge per path: the edge whose TARGET has the highest transitive cost
    critical_edges: list[tuple[str, str]] = []
    for path in raw_paths:
        best_edge: tuple[str, str] | None = None
        best_cost = -1
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            tgt_cost = costs.get(tgt)
            tgt_bytes = tgt_cost.transitive_size_bytes if tgt_cost else 0
            if tgt_bytes > best_cost:
                best_cost = tgt_bytes
                best_edge = (src, tgt)
        # best_edge is None only when the path has no edges (single node), which
        # _all_paths_bfs never produces — every path has source and target.
        if best_edge is not None:
            critical_edges.append(best_edge)

    # Try networkx min-cut for suggested cuts
    suggested_cuts: tuple[str, ...] | None = None
    try:
        import networkx as nx

        dg = to_networkx(graph, include_type_checking=False)

        if dg.has_node(entry_point) and dg.has_node(target):
            try:
                # Assign unit capacity to all edges so min-cut is well-defined
                for _u, _v, data in dg.edges(data=True):
                    data["capacity"] = 1
                _cut_val, partitions_raw = nx.minimum_cut(  # type: ignore[no-untyped-call]
                    dg, entry_point, target
                )
                partitions = cast("tuple[set[str], set[str]]", partitions_raw)
                sink_nodes: set[str] = partitions[1]
                # Nodes in the sink partition that entry_point directly imports
                cuts: list[str] = [
                    n for n in sink_nodes if n != target and n in graph.edges.get(entry_point, {})
                ]
                suggested_cuts = tuple(cuts) if cuts else None
            except (nx.NetworkXError, nx.NetworkXUnbounded, ValueError):
                suggested_cuts = None
    except ImportError:
        suggested_cuts = None

    total_cost = costs.get(target)

    return WhyResult(
        entry_point=entry_point,
        target=target,
        paths=tuple(tuple(p) for p in raw_paths),
        critical_edges=tuple(critical_edges),
        suggested_cuts=suggested_cuts,
        total_cost=total_cost,
    )


def explain_external_import(
    graph: ImportGraph,
    entry_point: str,
    target_package: str,
    search_paths: tuple[Path, ...] | None = None,
) -> ExternalWhyResult:
    """Explain why importing *entry_point* causes external package *target_package* to load.

    Algorithm:
    1. Resolve *target_package* to its dist name and discover matching top-level import names.
    2. Find which internal modules directly import those top-level names.
    3. For each direct importer, compute the shortest path from *entry_point*.
    4. If no direct importers are found, search for a transitive pip dependency chain.

    When *search_paths* is provided, metadata queries use the target project's
    site-packages instead of pyweight's own environment.
    """
    from pyweight.cost import packages_distributions_for

    if entry_point not in graph.modules:
        return ExternalWhyResult(
            entry_point=entry_point,
            target_package=target_package,
            target_import_names=(),
            direct_importers=(),
            paths=(),
            pip_dependency_chain=None,
            confidence=Confidence.LOW,
        )

    target_dist = resolve_dist_name(target_package, search_paths)

    # Build set of top-level import names that map to this distribution
    target_import_names: list[str] = []
    pkg_dist_map = packages_distributions_for(search_paths)
    for import_name, dist_list in pkg_dist_map.items():
        if not dist_list:
            continue
        resolved = resolve_dist_name(import_name, search_paths)
        if (
            resolved is not None and resolved.lower() == (target_dist or target_package).lower()
        ) or target_package.lower() in (d.lower() for d in dist_list):
            target_import_names.append(import_name)

    # Also include the target_package itself as a potential import name
    if target_package not in target_import_names:
        target_import_names.append(target_package)

    target_names_set = set(target_import_names)

    # Find modules that directly import any of those top-level names
    direct_importers: list[str] = []
    for module, ext_imports in graph.module_external_imports.items():
        if ext_imports & target_names_set:
            direct_importers.append(module)

    # Compute shortest path from entry_point to each direct importer
    paths: list[tuple[str, ...]] = []
    reachable_importers: list[str] = []
    for importer in direct_importers:
        path = shortest_path(graph, entry_point, importer)
        if path is not None:
            paths.append(path)
            reachable_importers.append(importer)

    if reachable_importers:
        return ExternalWhyResult(
            entry_point=entry_point,
            target_package=target_package,
            target_import_names=tuple(sorted(target_names_set)),
            direct_importers=tuple(reachable_importers),
            paths=tuple(paths),
            pip_dependency_chain=None,
            confidence=Confidence.HIGH,
        )

    # No direct importers reachable — search for transitive pip dependency chain
    if target_dist is not None:
        reachable_mods = reachable_from(graph, entry_point, include_type_checking=False)
        reachable_mods.add(entry_point)

        seen_ext_dists: set[str] = set()
        for mod in reachable_mods:
            for ext_top in graph.module_external_imports.get(mod, set()):
                ext_dist = resolve_dist_name(ext_top, search_paths)
                if ext_dist is not None and ext_dist not in seen_ext_dists:
                    seen_ext_dists.add(ext_dist)
                    chain = find_pip_dependency_chain(ext_dist, target_dist, search_paths)
                    if chain is not None:
                        return ExternalWhyResult(
                            entry_point=entry_point,
                            target_package=target_package,
                            target_import_names=tuple(sorted(target_names_set)),
                            direct_importers=(),
                            paths=(),
                            pip_dependency_chain=chain,
                            confidence=Confidence.LOW,
                        )

    return ExternalWhyResult(
        entry_point=entry_point,
        target_package=target_package,
        target_import_names=tuple(sorted(target_names_set)),
        direct_importers=(),
        paths=(),
        pip_dependency_chain=None,
        confidence=Confidence.LOW,
    )


def _resolve_import_module(
    ref: ImportRef, importer_qname: str, is_init: bool = False
) -> str | None:
    """Resolve a potentially-relative ImportRef to a fully-qualified module name.

    Relative imports (level > 0) are resolved against the importer's package.
    Returns None if resolution is not possible.

    *is_init* must be True when the importing file is an ``__init__.py``.
    For ``__init__.py`` the qualified name IS the package, so level=1 means
    "same package" rather than "parent package"; strip count is reduced by one.
    """
    if ref.level == 0:
        return ref.module if ref.module else None

    # Relative import: resolve against importer's package.
    # For __init__.py the qname already represents the package itself, so
    # level=1 stays at the same package (strip=0) and level=2 goes one up.
    parts = importer_qname.split(".")
    strip = ref.level - 1 if is_init else ref.level
    if strip >= len(parts):
        return None

    parent_parts = parts[: len(parts) - strip] if strip > 0 else parts

    if ref.module:
        return ".".join(parent_parts) + "." + ref.module
    return ".".join(parent_parts)


def find_dead_exports(
    graph: ImportGraph,
    is_library: bool = False,
) -> list[DeadExport]:
    """Find exported symbols that are never imported by any other module.

    Confidence:
    - HIGH for __all__-listed symbols (capped at MEDIUM when is_library=True)
    - MEDIUM for public defined names
    - LOW for star re-exports

    Sorted by module, then symbol_name.
    """
    # Build a set of (resolved_module, symbol) pairs that ARE imported somewhere,
    # and a set of modules that have at least one named (non-star) importer.
    imported_symbols: set[tuple[str, str]] = set()
    modules_with_named_importers: set[str] = set()

    for importer_name, info in graph.modules.items():
        for ref in info.imports:
            if ref.is_star:
                continue
            resolved = _resolve_import_module(ref, importer_name, is_init=info.is_init)
            if resolved is None:
                continue
            modules_with_named_importers.add(resolved)
            for name in ref.names:
                imported_symbols.add((resolved, name))

    results: list[DeadExport] = []

    for module_name, info in graph.modules.items():
        if info.has_all and info.all_names:
            # __all__-listed symbols: HIGH confidence (MEDIUM for libraries)
            for sym in sorted(info.all_names):
                if (module_name, sym) not in imported_symbols:
                    confidence = Confidence.MEDIUM if is_library else Confidence.HIGH
                    results.append(
                        DeadExport(
                            module=module_name,
                            symbol_name=sym,
                            export_mechanism="__all__",
                            confidence=confidence,
                        )
                    )
        else:
            # No __all__, or __all__ is empty: fall through to defined_names at MEDIUM confidence.
            # When has_all=True but all_names is empty we still check defined_names so that
            # dead-export detection is never silently skipped.
            for sym in sorted(info.defined_names):
                if sym.startswith("_"):
                    continue
                if (module_name, sym) not in imported_symbols:
                    results.append(
                        DeadExport(
                            module=module_name,
                            symbol_name=sym,
                            export_mechanism="public_def",
                            confidence=Confidence.MEDIUM,
                        )
                    )

        # Star re-exports from __init__.py barrels: flag at LOW confidence
        # when no one imports from this module via named imports
        if info.is_init and info.star_imports and module_name not in modules_with_named_importers:
            for star_src in info.star_imports:
                results.append(
                    DeadExport(
                        module=module_name,
                        symbol_name=f"*from:{star_src}",
                        export_mechanism="star_reexport",
                        confidence=Confidence.LOW,
                    )
                )

    results.sort(key=lambda d: (d.module, d.symbol_name))
    return results
