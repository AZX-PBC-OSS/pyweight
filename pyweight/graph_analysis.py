from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from pyweight.models import Confidence, CostEstimate, DSMResult, ImportGraph

if TYPE_CHECKING:
    from networkx import DiGraph as NxDiGraph


def _require_networkx() -> None:
    try:
        import importlib

        importlib.import_module("networkx")
    except ImportError as exc:
        raise ImportError(
            "networkx is required for graph analysis. "
            "Install it with: pip install pyweight[analysis]"
        ) from exc


def find_dominators(graph: ImportGraph, entry_point: str) -> dict[str, str]:
    """Return immediate dominator mapping for all nodes reachable from *entry_point*.

    The entry point is manually mapped to itself (NX omits it from the return dict).
    """
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    G = to_networkx(graph)
    doms = cast(
        "dict[str, str]",
        nx.immediate_dominators(G, entry_point),  # type: ignore[reportUnknownMemberType]
    )
    # nx.immediate_dominators excludes the start node; insert it explicitly.
    doms[entry_point] = entry_point
    return doms


def find_chokepoints(graph: ImportGraph) -> list[str]:
    """Return modules whose removal disconnects the import graph (articulation points).

    Operates on the undirected projection of the import graph.
    """
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    di_graph = to_networkx(graph)
    undirected: nx.Graph[str] = nx.Graph(di_graph)
    return list(nx.articulation_points(undirected))


def find_cycles(graph: ImportGraph) -> list[tuple[str, ...]]:
    """Return all non-trivial strongly connected components as tuples of module names."""
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    G = to_networkx(graph)
    return [tuple(sorted(scc)) for scc in nx.strongly_connected_components(G) if len(scc) > 1]


def condense_sccs(
    graph: ImportGraph,
    *,
    nx_graph: NxDiGraph[str] | None = None,
) -> tuple[NxDiGraph[int], dict[int, frozenset[str]]]:
    """Condense SCCs into a DAG.

    Returns the condensed DiGraph (nodes are integer SCC IDs) and a mapping from
    SCC ID to the frozenset of original module names it contains.

    Pass a pre-built *nx_graph* to avoid a second ``to_networkx`` call when the
    caller already holds one.
    """
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    G: NxDiGraph[str] = nx_graph if nx_graph is not None else to_networkx(graph)
    condensed: NxDiGraph[int] = nx.condensation(G)
    members: dict[int, frozenset[str]] = {
        node: frozenset(cast("set[str]", condensed.nodes[node]["members"]))
        for node in condensed.nodes
    }
    return condensed, members


def find_min_cut_modules(
    graph: ImportGraph,
    source: str,
    sink: str,
    costs: dict[str, CostEstimate],
) -> tuple[frozenset[str], float]:
    """Find the minimum node-cut between *source* and *sink* by cost.

    Uses node-splitting: each module becomes an (in, out) node pair with capacity
    equal to its ``direct_size_bytes``.  Cross-edges carry infinite capacity.
    Source and sink get infinite internal capacity so they are never part of the cut.

    Returns a frozenset of module names in the cut-set and the total cost saved
    (sum of ``direct_size_bytes`` for each cut module).
    """
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    if source not in graph.modules:
        raise ValueError(f"source {source!r} not in graph")
    if sink not in graph.modules:
        raise ValueError(f"sink {sink!r} not in graph")

    original = to_networkx(graph)

    split: nx.DiGraph[str] = nx.DiGraph()

    for node in original.nodes:
        if node in (source, sink):
            # Exclude source/sink from the cut set by giving them infinite capacity.
            capacity: float = math.inf
        else:
            capacity = costs[node].direct_size_bytes if node in costs else 1
        split.add_edge(f"{node}_in", f"{node}_out", capacity=capacity)

    for u, v in original.edges:
        split.add_edge(f"{u}_out", f"{v}_in", capacity=math.inf)

    _cut_value, raw_partition = cast(
        "tuple[float, tuple[set[str], set[str]]]",
        nx.minimum_cut(  # type: ignore[reportUnknownMemberType]
            split, f"{source}_in", f"{sink}_out", capacity="capacity"
        ),
    )
    reachable: set[str] = raw_partition[0]

    cut_modules: set[str] = set()
    for node in original.nodes:
        if f"{node}_in" in reachable and f"{node}_out" not in reachable:
            cut_modules.add(node)

    total_cost = sum(costs[m].direct_size_bytes for m in cut_modules if m in costs)
    return frozenset(cut_modules), float(total_cost)


def find_redundant_edges(
    graph: ImportGraph,
) -> list[tuple[tuple[str, str], Confidence]]:
    """Return edges that are transitively redundant.

    SCC-internal edges get Confidence.LOW (cyclic, hard to remove safely).
    Non-SCC edges that are covered by a transitive path get Confidence.HIGH.
    """
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    G = to_networkx(graph)
    # Pass the already-built NX graph so condense_sccs does not rebuild it.
    condensed, members = condense_sccs(graph, nx_graph=G)

    # Reverse map: module name → SCC id
    module_to_scc: dict[str, int] = {}
    for scc_id, mod_set in members.items():
        for mod in mod_set:
            module_to_scc[mod] = scc_id

    # Transitive reduction on condensed DAG (acyclic)
    reduced: NxDiGraph[int] = nx.transitive_reduction(condensed)

    condensed_edges: set[tuple[int, int]] = set(condensed.edges())
    reduced_edges: set[tuple[int, int]] = set(reduced.edges())
    removed_condensed: set[tuple[int, int]] = condensed_edges - reduced_edges

    result: list[tuple[tuple[str, str], Confidence]] = []

    for u, v in G.edges():
        u_scc = module_to_scc[u]
        v_scc = module_to_scc[v]

        if u_scc == v_scc:
            # Edge is within an SCC (cycle) — low confidence removal
            result.append(((u, v), Confidence.LOW))
        elif (u_scc, v_scc) in removed_condensed:
            # Edge maps to a condensed edge that was removed by transitive reduction
            result.append(((u, v), Confidence.HIGH))

    return result


def rank_by_centrality(
    graph: ImportGraph,
    costs: dict[str, CostEstimate],
) -> list[tuple[str, float]]:
    """Rank modules by weighted betweenness centrality.

    Edge weight = 1 / (transitive_size_bytes + 1) so that cheaper targets get
    higher weight (shorter "distance") in the centrality computation.

    For graphs with more than 500 modules an approximate algorithm is used.
    """
    _require_networkx()
    import networkx as nx

    from pyweight.graph import to_networkx

    G = to_networkx(graph)

    for u, v in G.edges():
        transitive = costs[v].transitive_size_bytes if v in costs else 0
        G[u][v]["weight"] = 1.0 / (transitive + 1)

    if len(G) > 500:
        k = min(100, len(G))
        centrality: dict[str, float] = nx.betweenness_centrality(  # type: ignore[reportUnknownMemberType]
            G, weight="weight", k=k
        )
    else:
        centrality = nx.betweenness_centrality(G, weight="weight")  # type: ignore[reportUnknownMemberType]

    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)


def compute_dsm(graph: ImportGraph) -> DSMResult:
    """Compute a Design Structure Matrix for the import graph.

    SCCs are condensed first to get a topological order.  Violations are
    edges from a higher layer index to a lower one (above-diagonal in the DSM).
    ``layering_health`` = 1 - violations/total_edges (1.0 when there are no edges).
    """
    _require_networkx()
    import networkx as nx

    condensed, members = condense_sccs(graph)

    # Topological order over condensed DAG → layer index per SCC
    topo_order: list[int] = list(nx.topological_sort(condensed))
    scc_layer: dict[int, int] = {scc_id: idx for idx, scc_id in enumerate(topo_order)}

    # Reverse map module → SCC id
    module_to_scc: dict[str, int] = {}
    for scc_id, mod_set in members.items():
        for mod in mod_set:
            module_to_scc[mod] = scc_id

    # Expand topological order back to individual modules, preserving SCC grouping
    ordered_modules: list[str] = []
    for scc_id in topo_order:
        ordered_modules.extend(sorted(members[scc_id]))

    n = len(ordered_modules)
    module_index: dict[str, int] = {m: i for i, m in enumerate(ordered_modules)}

    # Build boolean matrix
    matrix_rows: list[tuple[bool, ...]] = []
    for src in ordered_modules:
        row: list[bool] = [False] * n
        for tgt in graph.edges.get(src, {}):
            if tgt in module_index:
                row[module_index[tgt]] = True
        matrix_rows.append(tuple(row))

    # Identify violations: edge (src → tgt) where src has a higher layer than tgt
    violations: list[tuple[str, str]] = []
    total_edges = 0
    for src in ordered_modules:
        src_layer = scc_layer[module_to_scc[src]]
        for tgt in graph.edges.get(src, {}):
            if tgt not in module_index:
                continue
            total_edges += 1
            tgt_layer = scc_layer[module_to_scc[tgt]]
            if src_layer > tgt_layer:
                violations.append((src, tgt))

    layering_health = 1.0 if total_edges == 0 else 1.0 - len(violations) / total_edges

    layer_assignments: dict[str, int] = {m: scc_layer[module_to_scc[m]] for m in ordered_modules}

    return DSMResult(
        modules=tuple(ordered_modules),
        matrix=tuple(matrix_rows),
        layer_assignments=layer_assignments,
        violations=tuple(violations),
        layering_health=layering_health,
    )


def compute_topological_depth(graph: ImportGraph) -> dict[str, int]:
    """Assign a topological depth to each module.

    Leaves (out-degree 0 in the condensed DAG) receive depth 0.  Depth increases
    towards roots.  All modules in the same SCC share the same depth.
    """
    _require_networkx()
    import networkx as nx

    condensed, members = condense_sccs(graph)

    depths: dict[int, int] = {}
    for scc_id in reversed(list(nx.topological_sort(condensed))):
        successors = list(condensed.successors(scc_id))
        if not successors:
            depths[scc_id] = 0
        else:
            depths[scc_id] = max(depths[s] for s in successors) + 1

    result: dict[str, int] = {}
    for scc_id, mod_set in members.items():
        depth = depths.get(scc_id, 0)
        for mod in mod_set:
            result[mod] = depth

    return result
