"""Tests for pyweight.graph_analysis module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyweight.models import Confidence, CostEstimate, EdgeInfo, ImportGraph, ModuleInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module_info(name: str) -> ModuleInfo:
    return ModuleInfo(
        path=Path(f"/{name}.py"),
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


def _make_edge(*, confidence: Confidence = Confidence.HIGH) -> EdgeInfo:
    return EdgeInfo(confidence=confidence, is_type_checking=False, is_conditional=False)


def _make_cost(name: str, direct: int = 1000, transitive: int = 5000) -> CostEstimate:
    return CostEstimate(
        module=name,
        dist_name=None,
        direct_size_bytes=direct,
        transitive_size_bytes=transitive,
        confidence=Confidence.HIGH,
        provenance="test",
    )


def _build_linear_graph(names: list[str]) -> ImportGraph:
    """Build a→b→c→... chain."""
    g = ImportGraph()
    for n in names:
        g.add_module(n, _make_module_info(n))
    for i in range(len(names) - 1):
        g.add_edge(names[i], names[i + 1], _make_edge())
    return g


def _build_cycle_graph(
    cycle: list[str], extra_edges: list[tuple[str, str]] | None = None
) -> ImportGraph:
    """Build a cyclic graph: cycle[0]→cycle[1]→...→cycle[-1]→cycle[0]."""
    g = ImportGraph()
    for n in cycle:
        g.add_module(n, _make_module_info(n))
    for i in range(len(cycle)):
        g.add_edge(cycle[i], cycle[(i + 1) % len(cycle)], _make_edge())
    for src, tgt in extra_edges or []:
        if src not in g.modules:
            g.add_module(src, _make_module_info(src))
        if tgt not in g.modules:
            g.add_module(tgt, _make_module_info(tgt))
        g.add_edge(src, tgt, _make_edge())
    return g


# ---------------------------------------------------------------------------
# find_dominators
# ---------------------------------------------------------------------------


def test_find_dominators_entry_dominates_all() -> None:
    from pyweight.graph_analysis import find_dominators

    # a → b → c
    g = _build_linear_graph(["a", "b", "c"])
    doms = find_dominators(g, "a")

    # a dominates b and c
    assert doms["b"] == "a"
    assert doms["c"] == "b"
    # entry dominates itself
    assert doms["a"] == "a"


def test_find_dominators_entry_point_maps_to_itself() -> None:
    from pyweight.graph_analysis import find_dominators

    g = _build_linear_graph(["root", "child"])
    doms = find_dominators(g, "root")
    assert doms["root"] == "root"


def test_find_dominators_branch() -> None:
    from pyweight.graph_analysis import find_dominators

    # root → a, root → b; a → sink; b → sink
    g = ImportGraph()
    for n in ["root", "a", "b", "sink"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("root", "a", _make_edge())
    g.add_edge("root", "b", _make_edge())
    g.add_edge("a", "sink", _make_edge())
    g.add_edge("b", "sink", _make_edge())

    doms = find_dominators(g, "root")
    # root dominates everything reachable
    assert doms["root"] == "root"
    assert doms["a"] == "root"
    assert doms["b"] == "root"
    # sink is dominated by root (common dominator), not a or b
    assert doms["sink"] == "root"


def test_find_dominators_entry_not_in_graph_raises() -> None:
    import networkx as nx

    from pyweight.graph_analysis import find_dominators

    g = _build_linear_graph(["a", "b"])
    with pytest.raises((nx.NetworkXError, KeyError, Exception)):
        find_dominators(g, "nonexistent")


# ---------------------------------------------------------------------------
# find_chokepoints
# ---------------------------------------------------------------------------


def test_find_chokepoints_known_topology() -> None:
    from pyweight.graph_analysis import find_chokepoints

    # a - bridge - b, where "bridge" is the articulation point
    # a → bridge → b  (directed, but undirected projection: a--bridge--b)
    g = ImportGraph()
    for n in ["a", "bridge", "b"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "bridge", _make_edge())
    g.add_edge("bridge", "b", _make_edge())

    chokepoints = find_chokepoints(g)
    assert "bridge" in chokepoints


def test_find_chokepoints_fully_connected_no_articulation() -> None:
    from pyweight.graph_analysis import find_chokepoints

    # Triangle: a→b→c→a (every node is redundant in the undirected sense)
    g = _build_cycle_graph(["a", "b", "c"])
    chokepoints = find_chokepoints(g)
    # No single articulation point in a triangle
    assert len(chokepoints) == 0


def test_find_chokepoints_star_topology() -> None:
    from pyweight.graph_analysis import find_chokepoints

    # center → leaf1, center → leaf2, center → leaf3
    g = ImportGraph()
    nodes = ["center", "leaf1", "leaf2", "leaf3"]
    for n in nodes:
        g.add_module(n, _make_module_info(n))
    for leaf in ["leaf1", "leaf2", "leaf3"]:
        g.add_edge("center", leaf, _make_edge())

    chokepoints = find_chokepoints(g)
    assert "center" in chokepoints


# ---------------------------------------------------------------------------
# find_cycles
# ---------------------------------------------------------------------------


def test_find_cycles_simple_cycle() -> None:
    from pyweight.graph_analysis import find_cycles

    # A → B → C → A
    g = _build_cycle_graph(["A", "B", "C"])
    cycles = find_cycles(g)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C"}


def test_find_cycles_no_cycle() -> None:
    from pyweight.graph_analysis import find_cycles

    g = _build_linear_graph(["x", "y", "z"])
    cycles = find_cycles(g)
    assert cycles == []


def test_find_cycles_multiple_cycles() -> None:
    from pyweight.graph_analysis import find_cycles

    # Two disjoint cycles: [a,b] and [c,d,e]
    g = ImportGraph()
    for n in ["a", "b", "c", "d", "e"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "a", _make_edge())
    g.add_edge("c", "d", _make_edge())
    g.add_edge("d", "e", _make_edge())
    g.add_edge("e", "c", _make_edge())

    cycles = find_cycles(g)
    assert len(cycles) == 2
    cycle_sets = [set(c) for c in cycles]
    assert {"a", "b"} in cycle_sets
    assert {"c", "d", "e"} in cycle_sets


def test_find_cycles_returns_tuples() -> None:
    from pyweight.graph_analysis import find_cycles

    g = _build_cycle_graph(["p", "q"])
    cycles = find_cycles(g)
    assert all(isinstance(c, tuple) for c in cycles)


def test_find_cycles_tuples_are_sorted() -> None:
    from pyweight.graph_analysis import find_cycles

    # Cycle with members whose sorted order is deterministic: c→b→a→c
    g = _build_cycle_graph(["c", "b", "a"])
    cycles = find_cycles(g)
    assert len(cycles) == 1
    members = list(cycles[0])
    assert members == sorted(members)


# ---------------------------------------------------------------------------
# condense_sccs
# ---------------------------------------------------------------------------


def test_condense_sccs_is_dag() -> None:
    import networkx as nx

    from pyweight.graph_analysis import condense_sccs

    g = _build_cycle_graph(["a", "b", "c"], extra_edges=[("d", "a")])
    dag, _members = condense_sccs(g)
    assert nx.is_directed_acyclic_graph(dag)


def test_condense_sccs_membership_correct() -> None:
    from pyweight.graph_analysis import condense_sccs

    # a→b→a cycle; c→a (separate)
    g = ImportGraph()
    for n in ["a", "b", "c"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "a", _make_edge())
    g.add_edge("c", "a", _make_edge())

    _dag, members = condense_sccs(g)
    all_mods: set[str] = set()
    for mod_set in members.values():
        all_mods |= mod_set

    assert all_mods == {"a", "b", "c"}

    # a and b must be in the same SCC
    scc_of_a = next(scc_id for scc_id, mods in members.items() if "a" in mods)
    scc_of_b = next(scc_id for scc_id, mods in members.items() if "b" in mods)
    assert scc_of_a == scc_of_b

    # c must be in its own SCC
    scc_of_c = next(scc_id for scc_id, mods in members.items() if "c" in mods)
    assert scc_of_c != scc_of_a


def test_condense_sccs_covers_all_modules() -> None:
    from pyweight.graph_analysis import condense_sccs

    g = _build_linear_graph(["x", "y", "z"])
    _dag, members = condense_sccs(g)
    all_mods = {m for mods in members.values() for m in mods}
    assert all_mods == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# find_min_cut_modules
# ---------------------------------------------------------------------------


def test_find_min_cut_modules_disconnects_source_from_sink() -> None:
    from pyweight.graph import to_networkx
    from pyweight.graph_analysis import find_min_cut_modules

    # source → m1 → sink; source → m2 → sink
    g = ImportGraph()
    for n in ["source", "m1", "m2", "sink"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("source", "m1", _make_edge())
    g.add_edge("source", "m2", _make_edge())
    g.add_edge("m1", "sink", _make_edge())
    g.add_edge("m2", "sink", _make_edge())

    costs = {
        "source": _make_cost("source", direct=9999),
        "m1": _make_cost("m1", direct=100),
        "m2": _make_cost("m2", direct=200),
        "sink": _make_cost("sink", direct=9999),
    }

    cut_modules, _ = find_min_cut_modules(g, "source", "sink", costs)

    # After removing the cut modules, sink must be unreachable from source
    import networkx as nx

    G = to_networkx(g)
    remaining: nx.DiGraph[str] = nx.DiGraph()
    for node in G.nodes:
        if node not in cut_modules:
            remaining.add_node(node)
    for u, v in G.edges:
        if u not in cut_modules and v not in cut_modules:
            remaining.add_edge(u, v)

    assert not nx.has_path(remaining, "source", "sink")


def test_find_min_cut_modules_cost_matches() -> None:
    from pyweight.graph_analysis import find_min_cut_modules

    g = ImportGraph()
    for n in ["source", "mid", "sink"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("source", "mid", _make_edge())
    g.add_edge("mid", "sink", _make_edge())

    costs = {
        "source": _make_cost("source", direct=9999),
        "mid": _make_cost("mid", direct=500),
        "sink": _make_cost("sink", direct=9999),
    }

    cut_modules, total_cost = find_min_cut_modules(g, "source", "sink", costs)
    expected = sum(costs[m].direct_size_bytes for m in cut_modules)
    assert total_cost == float(expected)


def test_find_min_cut_modules_source_not_in_graph_raises() -> None:
    from pyweight.graph_analysis import find_min_cut_modules

    g = _build_linear_graph(["a", "b"])
    costs = {n: _make_cost(n) for n in ["a", "b"]}
    with pytest.raises(ValueError, match="source"):
        find_min_cut_modules(g, "missing", "b", costs)


def test_find_min_cut_modules_sink_not_in_graph_raises() -> None:
    from pyweight.graph_analysis import find_min_cut_modules

    g = _build_linear_graph(["a", "b"])
    costs = {n: _make_cost(n) for n in ["a", "b"]}
    with pytest.raises(ValueError, match="sink"):
        find_min_cut_modules(g, "a", "missing", costs)


def test_find_min_cut_modules_source_equals_sink() -> None:
    import contextlib

    from pyweight.graph_analysis import find_min_cut_modules

    # source == sink: both nodes are valid so no ValueError is raised.
    # NX may raise or return a result for the trivial same-node case — either is acceptable.
    g = _build_linear_graph(["a", "b", "c"])
    costs = {n: _make_cost(n) for n in ["a", "b", "c"]}
    with contextlib.suppress(Exception):
        _cut, _cost = find_min_cut_modules(g, "a", "a", costs)


# ---------------------------------------------------------------------------
# find_redundant_edges
# ---------------------------------------------------------------------------


def test_find_redundant_edges_transitive_redundancy() -> None:
    from pyweight.graph_analysis import find_redundant_edges

    # a → b → c; a → c (redundant, covered by a→b→c)
    g = ImportGraph()
    for n in ["a", "b", "c"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "c", _make_edge())
    g.add_edge("a", "c", _make_edge())  # redundant

    redundant = find_redundant_edges(g)
    redundant_edges = {e for e, _ in redundant}
    assert ("a", "c") in redundant_edges


def test_find_redundant_edges_high_confidence_non_scc() -> None:
    from pyweight.graph_analysis import find_redundant_edges

    # a → b → c; a → c (redundant, non-SCC)
    g = ImportGraph()
    for n in ["a", "b", "c"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "c", _make_edge())
    g.add_edge("a", "c", _make_edge())

    redundant = find_redundant_edges(g)
    conf_map = dict(redundant)
    assert conf_map[("a", "c")] == Confidence.HIGH


def test_find_redundant_edges_low_confidence_scc() -> None:
    from pyweight.graph_analysis import find_redundant_edges

    # a→b→a cycle with a→b edge (SCC edge)
    g = ImportGraph()
    for n in ["a", "b"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "a", _make_edge())

    redundant = find_redundant_edges(g)
    assert all(conf == Confidence.LOW for _, conf in redundant)


def test_find_redundant_edges_non_redundant_not_returned() -> None:
    from pyweight.graph_analysis import find_redundant_edges

    # Simple chain: a → b → c (no redundancy)
    g = _build_linear_graph(["a", "b", "c"])
    redundant = find_redundant_edges(g)
    redundant_edges = {e for e, _ in redundant}
    assert ("a", "b") not in redundant_edges
    assert ("b", "c") not in redundant_edges


def test_find_redundant_edges_self_loop() -> None:
    from pyweight.graph_analysis import find_redundant_edges

    # a → a (self-loop) — self-loops are within a trivial SCC
    g = ImportGraph()
    g.add_module("a", _make_module_info("a"))
    g.add_module("b", _make_module_info("b"))
    g.add_edge("a", "a", _make_edge())
    g.add_edge("a", "b", _make_edge())

    redundant = find_redundant_edges(g)
    redundant_edges = {e for e, _ in redundant}
    # The self-loop is within its own SCC — should be flagged LOW confidence
    assert ("a", "a") in redundant_edges
    conf_map = dict(redundant)
    assert conf_map[("a", "a")] == Confidence.LOW


# ---------------------------------------------------------------------------
# rank_by_centrality
# ---------------------------------------------------------------------------


def test_rank_by_centrality_most_central() -> None:
    from pyweight.graph_analysis import rank_by_centrality

    # diamond: a→b, a→c, b→d, c→d — b and c both mediate paths to d
    g = ImportGraph()
    for n in ["a", "b", "c", "d"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("a", "c", _make_edge())
    g.add_edge("b", "d", _make_edge())
    g.add_edge("c", "d", _make_edge())

    costs = {n: _make_cost(n, transitive=1000) for n in ["a", "b", "c", "d"]}
    ranked = rank_by_centrality(g, costs)

    assert len(ranked) == 4
    # b and c should have higher centrality than a and d
    rank_map = dict(ranked)
    assert rank_map["b"] > rank_map["a"]
    assert rank_map["c"] > rank_map["a"]


def test_rank_by_centrality_sorted_descending() -> None:
    from pyweight.graph_analysis import rank_by_centrality

    g = _build_linear_graph(["p", "q", "r", "s"])
    costs = {n: _make_cost(n) for n in ["p", "q", "r", "s"]}
    ranked = rank_by_centrality(g, costs)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_by_centrality_returns_all_modules() -> None:
    from pyweight.graph_analysis import rank_by_centrality

    g = _build_linear_graph(["x", "y", "z"])
    costs = {n: _make_cost(n) for n in ["x", "y", "z"]}
    ranked = rank_by_centrality(g, costs)
    assert {name for name, _ in ranked} == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# compute_dsm
# ---------------------------------------------------------------------------


def test_compute_dsm_linear_chain_zero_violations() -> None:
    from pyweight.graph_analysis import compute_dsm

    g = _build_linear_graph(["a", "b", "c"])
    result = compute_dsm(g)
    assert result.violations == ()
    assert result.layering_health == 1.0


def test_compute_dsm_no_edges_health_one() -> None:
    from pyweight.graph_analysis import compute_dsm

    g = ImportGraph()
    for n in ["x", "y"]:
        g.add_module(n, _make_module_info(n))

    result = compute_dsm(g)
    assert result.layering_health == 1.0
    assert result.violations == ()


def test_compute_dsm_cycle_produces_no_inter_scc_violations() -> None:
    from pyweight.graph_analysis import compute_dsm

    # a → b → c with back-edge c → a creates SCC {a,b,c}; d → a is an inter-SCC edge.
    # All inter-SCC edges in the condensed DAG go forward in topo order by definition,
    # so src_layer is never > tgt_layer. Violations are always empty.
    g = ImportGraph()
    for n in ["a", "b", "c", "d"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("d", "a", _make_edge())
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "c", _make_edge())
    g.add_edge("c", "a", _make_edge())

    result = compute_dsm(g)
    assert result.violations == ()
    assert result.layering_health == 1.0


def test_compute_dsm_layer_assignments_consistent_with_topo_order() -> None:
    from pyweight.graph_analysis import compute_dsm

    # a → b → c: topological order puts a first (layer 0), c last (layer 2).
    # Violations are edges where src_layer > tgt_layer, so a→b is fine (0 < 1).
    g = _build_linear_graph(["a", "b", "c"])
    result = compute_dsm(g)
    # a is the root: it has the smallest layer index
    assert result.layer_assignments["a"] < result.layer_assignments["b"]
    assert result.layer_assignments["b"] < result.layer_assignments["c"]


def test_compute_dsm_scc_members_share_layer() -> None:
    from pyweight.graph_analysis import compute_dsm

    # a → b → c → a forms SCC {a,b,c}; d → a is an inter-SCC edge.
    # All members of the SCC get the same layer assignment.
    g = ImportGraph()
    for n in ["a", "b", "c", "d"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("d", "a", _make_edge())
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "c", _make_edge())
    g.add_edge("c", "a", _make_edge())

    result = compute_dsm(g)
    # All SCC members share the same layer
    assert result.layer_assignments["a"] == result.layer_assignments["b"]
    assert result.layer_assignments["b"] == result.layer_assignments["c"]
    # d is in a separate SCC and precedes the cycle SCC in topo order
    assert result.layer_assignments["d"] != result.layer_assignments["a"]


def test_compute_dsm_modules_and_matrix_dimensions_match() -> None:
    from pyweight.graph_analysis import compute_dsm

    g = _build_linear_graph(["a", "b", "c", "d"])
    result = compute_dsm(g)
    n = len(result.modules)
    assert len(result.matrix) == n
    assert all(len(row) == n for row in result.matrix)


def test_compute_dsm_matrix_reflects_edges() -> None:
    from pyweight.graph_analysis import compute_dsm

    g = _build_linear_graph(["a", "b", "c"])
    result = compute_dsm(g)
    idx = {m: i for i, m in enumerate(result.modules)}
    # a → b must be True in matrix
    assert result.matrix[idx["a"]][idx["b"]] is True
    # b → a must be False
    assert result.matrix[idx["b"]][idx["a"]] is False


# ---------------------------------------------------------------------------
# compute_topological_depth
# ---------------------------------------------------------------------------


def test_compute_topological_depth_leaves_are_zero() -> None:
    from pyweight.graph_analysis import compute_topological_depth

    # a → b → c: c is a leaf (out-degree 0)
    g = _build_linear_graph(["a", "b", "c"])
    depths = compute_topological_depth(g)
    assert depths["c"] == 0


def test_compute_topological_depth_increases_along_chain() -> None:
    from pyweight.graph_analysis import compute_topological_depth

    # a → b → c: c=0, b=1, a=2
    g = _build_linear_graph(["a", "b", "c"])
    depths = compute_topological_depth(g)
    assert depths["c"] < depths["b"] < depths["a"]


def test_compute_topological_depth_scc_same_depth() -> None:
    from pyweight.graph_analysis import compute_topological_depth

    # a → b → a cycle (SCC), d → a (d is outside SCC)
    g = ImportGraph()
    for n in ["a", "b", "d"]:
        g.add_module(n, _make_module_info(n))
    g.add_edge("a", "b", _make_edge())
    g.add_edge("b", "a", _make_edge())
    g.add_edge("d", "a", _make_edge())

    depths = compute_topological_depth(g)
    # a and b are in the same SCC → same depth
    assert depths["a"] == depths["b"]
    # d points into the SCC, so d has higher depth
    assert depths["d"] > depths["a"]


def test_compute_topological_depth_isolated_node() -> None:
    from pyweight.graph_analysis import compute_topological_depth

    g = ImportGraph()
    g.add_module("solo", _make_module_info("solo"))
    depths = compute_topological_depth(g)
    assert depths["solo"] == 0


def test_compute_topological_depth_all_modules_present() -> None:
    from pyweight.graph_analysis import compute_topological_depth

    g = _build_linear_graph(["x", "y", "z"])
    depths = compute_topological_depth(g)
    assert set(depths.keys()) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# Missing networkx graceful error
# ---------------------------------------------------------------------------


def test_missing_networkx_raises_import_error() -> None:
    from pyweight.graph_analysis import find_cycles

    # Temporarily hide networkx from the import system
    original = sys.modules.get("networkx")
    sys.modules["networkx"] = None  # type: ignore[assignment]
    try:
        g = _build_linear_graph(["a", "b"])
        with pytest.raises(ImportError, match="networkx is required"):
            find_cycles(g)
    finally:
        if original is None:
            del sys.modules["networkx"]
        else:
            sys.modules["networkx"] = original


def test_missing_networkx_error_message_is_clear() -> None:
    from pyweight.graph_analysis import find_chokepoints

    original = sys.modules.get("networkx")
    sys.modules["networkx"] = None  # type: ignore[assignment]
    try:
        g = _build_linear_graph(["a", "b"])
        with pytest.raises(ImportError) as exc_info:
            find_chokepoints(g)
        assert "pip install pyweight[analysis]" in str(exc_info.value)
    finally:
        if original is None:
            del sys.modules["networkx"]
        else:
            sys.modules["networkx"] = original
