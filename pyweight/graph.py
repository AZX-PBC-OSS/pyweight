from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from pyweight.models import EdgeInfo, ImportGraph
from pyweight.parser import parse_module
from pyweight.resolver import PackageResolver

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx


def build_graph(
    package_root: Path,
    entry_point: str | None = None,
) -> ImportGraph:
    """Build an :class:`ImportGraph` from *package_root*.

    If *entry_point* is provided the graph is pruned to the subgraph reachable
    from that module (BFS over all edges including type-checking ones).
    """
    resolver = PackageResolver(package_root)
    graph = ImportGraph()

    # Parse every discovered module and add it to the graph
    for qname, path in resolver.all_modules.items():
        info = parse_module(path, qname)
        graph.add_module(qname, info)

    # Second pass: resolve imports and add edges
    for qname, info in graph.modules.items():
        for ref in info.imports:
            resolved_path, confidence = resolver.resolve(
                ref.module, qname, ref.level, is_init=info.is_init
            )

            if resolved_path is None:
                # Resolution failure already implies external or unresolvable.
                if ref.level == 0 and ref.module:
                    ext_top = ref.module.split(".")[0]
                    graph.external_deps.add(ext_top)
                    if qname not in graph.module_external_imports:
                        graph.module_external_imports[qname] = set()
                    graph.module_external_imports[qname].add(ext_top)
                continue

            # Find the target qualified name from the resolved path
            target_qname = _path_to_qname(resolved_path, resolver)
            if target_qname is None:
                continue

            edge = EdgeInfo(
                confidence=confidence,
                is_type_checking=ref.is_type_checking,
                is_conditional=ref.is_conditional,
            )
            graph.add_edge(qname, target_qname, edge)

    if entry_point is not None:
        graph = _prune_to_reachable(graph, entry_point)

    return graph


def _path_to_qname(path: Path, resolver: PackageResolver) -> str | None:
    """Reverse-map a resolved path back to a qualified name."""
    return resolver.qname_for_path(path)


def _prune_to_reachable(graph: ImportGraph, entry_point: str) -> ImportGraph:
    """Return a new ImportGraph containing only nodes reachable from *entry_point*."""
    if entry_point not in graph.modules:
        return graph

    reachable = reachable_from(graph, entry_point, include_type_checking=True)
    reachable.add(entry_point)

    pruned = ImportGraph()
    for name in reachable:
        if name in graph.modules:
            pruned.add_module(name, graph.modules[name])

    for source, targets in graph.edges.items():
        if source not in reachable:
            continue
        for target, edge_info in targets.items():
            if target in reachable:
                pruned.add_edge(source, target, edge_info)

    pruned.module_external_imports = {
        mod: deps for mod, deps in graph.module_external_imports.items() if mod in reachable
    }
    pruned.external_deps = {
        ext for deps in pruned.module_external_imports.values() for ext in deps
    }
    return pruned


def reachable_from(
    graph: ImportGraph,
    module: str,
    depth: int | None = None,
    include_type_checking: bool = False,
) -> set[str]:
    """BFS from *module*; return set of reachable module names (excluding start).

    When *depth* is given, traversal stops after that many hops.
    By default type-checking edges are excluded.
    """
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(module, 0)])

    while queue:
        current, current_depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if depth is not None and current_depth >= depth:
            continue

        for neighbour, edge_info in graph.edges.get(current, {}).items():
            if not include_type_checking and edge_info.is_type_checking:
                continue
            if neighbour not in visited:
                queue.append((neighbour, current_depth + 1))

    visited.discard(module)
    return visited


def shortest_path(
    graph: ImportGraph,
    source: str,
    target: str,
    include_type_checking: bool = False,
) -> tuple[str, ...] | None:
    """Return BFS shortest path from *source* to *target*, or None if unreachable."""
    if source == target:
        return (source,)

    visited: set[str] = {source}
    # Queue stores (current_node, path_so_far)
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])

    while queue:
        current, path = queue.popleft()
        for neighbour, edge_info in graph.edges.get(current, {}).items():
            if not include_type_checking and edge_info.is_type_checking:
                continue
            if neighbour in visited:
                continue
            new_path = (*path, neighbour)
            if neighbour == target:
                return new_path
            visited.add(neighbour)
            queue.append((neighbour, new_path))

    return None


def fan_in(
    graph: ImportGraph,
    module: str,
    include_type_checking: bool = False,
) -> int:
    """Count the number of modules that import *module* (reverse edges)."""
    count = 0
    for _source, edge_info in graph.reverse_edges.get(module, {}).items():
        if not include_type_checking and edge_info.is_type_checking:
            continue
        count += 1
    return count


def to_networkx(
    graph: ImportGraph,
    include_type_checking: bool = False,
) -> nx.DiGraph[str]:
    """Convert *graph* to a :class:`networkx.DiGraph`.

    networkx is imported lazily so callers without it installed only fail here.
    """
    import networkx as nx

    dg: nx.DiGraph[str] = nx.DiGraph()

    for name in graph.modules:
        dg.add_node(name)

    for source, targets in graph.edges.items():
        for target, edge_info in targets.items():
            if not include_type_checking and edge_info.is_type_checking:
                continue
            dg.add_edge(
                source,
                target,
                confidence=edge_info.confidence,
                is_type_checking=edge_info.is_type_checking,
                is_conditional=edge_info.is_conditional,
            )

    return dg
