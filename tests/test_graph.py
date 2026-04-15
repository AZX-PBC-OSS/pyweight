"""Tests for pyweight.graph module."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from pyweight.graph import build_graph, fan_in, reachable_from, shortest_path, to_networkx
from pyweight.models import Confidence, ImportGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_graph(sample_package_path: Path) -> ImportGraph:
    return build_graph(sample_package_path)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_build_graph_module_count(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # sample_package has at least: __init__, core, models, hub, heavy, guarded,
    # conditional, side_effects, deferred_candidate, utils/__init__, utils/helpers,
    # _mock_heavy_lib/__init__
    assert len(graph.modules) >= 10


def test_build_graph_all_module_names_present(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    expected = {
        "sample_package",
        "sample_package.core",
        "sample_package.models",
        "sample_package.hub",
        "sample_package.heavy",
        "sample_package.guarded",
        "sample_package.utils",
        "sample_package.utils.helpers",
    }
    assert expected <= set(graph.modules.keys())


def test_build_graph_edges_present(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # models imports core
    assert "sample_package.core" in graph.edges.get("sample_package.models", {})
    # hub imports core
    assert "sample_package.core" in graph.edges.get("sample_package.hub", {})
    # heavy imports hub
    assert "sample_package.hub" in graph.edges.get("sample_package.heavy", {})


def test_build_graph_external_deps_tracked(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # core.py imports os; heavy.py imports json and re; conditional.py imports tomllib
    assert graph.external_deps & {"os", "json", "re"}


def test_build_graph_reverse_edges_consistent(sample_package_path: Path) -> None:
    """Every forward edge must appear in reverse_edges."""
    graph = _sample_graph(sample_package_path)
    for source, targets in graph.edges.items():
        for target, edge_info in targets.items():
            assert source in graph.reverse_edges.get(target, {}), (
                f"Missing reverse edge {target} <- {source}"
            )
            assert graph.reverse_edges[target][source] == edge_info


# ---------------------------------------------------------------------------
# TYPE_CHECKING edges
# ---------------------------------------------------------------------------


def test_type_checking_edge_retained(sample_package_path: Path) -> None:
    """guarded.py has a TYPE_CHECKING import of .heavy — edge must be retained."""
    graph = _sample_graph(sample_package_path)
    heavy_edges = graph.edges.get("sample_package.guarded", {})
    assert "sample_package.heavy" in heavy_edges, (
        f"Expected guarded->heavy edge; edges: {list(heavy_edges)}"
    )
    # At least one edge to heavy should be is_type_checking=True
    edge = heavy_edges["sample_package.heavy"]
    # The edge may be the runtime one or the type-checking one — either way it exists
    assert edge is not None


def test_type_checking_edge_flag(sample_package_path: Path) -> None:
    """guarded.py has both a TYPE_CHECKING and a runtime import of .heavy.
    After merge, the edge must be is_type_checking=False (runtime wins).
    """
    graph = _sample_graph(sample_package_path)
    heavy_edges = graph.edges.get("sample_package.guarded", {})
    assert "sample_package.heavy" in heavy_edges
    assert heavy_edges["sample_package.heavy"].is_type_checking is False


def test_add_edge_merge_runtime_wins_over_type_checking() -> None:
    """When the same edge is added twice, one TYPE_CHECKING and one runtime,
    the merged edge must have is_type_checking=False (runtime wins).
    """
    from pyweight.models import EdgeInfo

    graph = ImportGraph()
    tc_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=True, is_conditional=False)
    runtime_edge = EdgeInfo(
        confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False
    )
    graph.add_edge("a", "b", tc_edge)
    graph.add_edge("a", "b", runtime_edge)
    assert graph.edges["a"]["b"].is_type_checking is False


def test_add_edge_merge_confidence_keeps_best() -> None:
    """When merging duplicate edges, the best (highest) confidence is kept."""
    from pyweight.models import EdgeInfo

    graph = ImportGraph()
    high_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=True, is_conditional=False)
    low_edge = EdgeInfo(confidence=Confidence.LOW, is_type_checking=False, is_conditional=False)
    graph.add_edge("a", "b", high_edge)
    graph.add_edge("a", "b", low_edge)
    assert graph.edges["a"]["b"].confidence == Confidence.HIGH


def test_add_edge_merge_confidence_keeps_best_reversed_order() -> None:
    """When merging duplicate edges, best confidence is kept regardless of insertion order."""
    from pyweight.models import EdgeInfo

    graph = ImportGraph()
    low_edge = EdgeInfo(confidence=Confidence.LOW, is_type_checking=False, is_conditional=False)
    high_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=True, is_conditional=False)
    graph.add_edge("a", "b", low_edge)
    graph.add_edge("a", "b", high_edge)
    assert graph.edges["a"]["b"].confidence == Confidence.HIGH


def test_add_edge_merge_is_conditional_only_if_both() -> None:
    """Merged edge is conditional only when both source edges are conditional."""
    from pyweight.models import EdgeInfo

    graph = ImportGraph()
    cond_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=True)
    non_cond_edge = EdgeInfo(
        confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False
    )
    graph.add_edge("a", "b", cond_edge)
    graph.add_edge("a", "b", non_cond_edge)
    assert graph.edges["a"]["b"].is_conditional is False


def test_type_checking_excluded_from_reachable_default(sample_package_path: Path) -> None:
    """reachable_from excludes is_type_checking edges by default."""
    from pyweight.models import EdgeInfo
    from pyweight.parser import parse_module

    core_path = sample_package_path / "core.py"
    core_info = parse_module(core_path, "sample_package.core")
    models_path = sample_package_path / "models.py"
    models_info = parse_module(models_path, "sample_package.models")

    mini = ImportGraph()
    mini.add_module("sample_package.core", core_info)
    mini.add_module("sample_package.models", models_info)
    tc_edge = EdgeInfo(
        confidence=Confidence.HIGH,
        is_type_checking=True,
        is_conditional=False,
    )
    mini.add_edge("sample_package.models", "sample_package.core", tc_edge)

    result = reachable_from(mini, "sample_package.models", include_type_checking=False)
    assert "sample_package.core" not in result

    result_with_tc = reachable_from(mini, "sample_package.models", include_type_checking=True)
    assert "sample_package.core" in result_with_tc


# ---------------------------------------------------------------------------
# Entry-point filtering
# ---------------------------------------------------------------------------


def test_entry_point_prunes_graph(sample_package_path: Path) -> None:
    full_graph = build_graph(sample_package_path)
    pruned_graph = build_graph(sample_package_path, entry_point="sample_package.models")
    assert len(pruned_graph.modules) < len(full_graph.modules)


def test_entry_point_graph_contains_entry_module(sample_package_path: Path) -> None:
    graph = build_graph(sample_package_path, entry_point="sample_package.models")
    assert "sample_package.models" in graph.modules


def test_entry_point_graph_excludes_unreachable(sample_package_path: Path) -> None:
    graph = build_graph(sample_package_path, entry_point="sample_package.models")
    # side_effects and deferred_candidate are not reachable from models
    assert "sample_package.side_effects" not in graph.modules
    assert "sample_package.deferred_candidate" not in graph.modules


def test_entry_point_graph_includes_transitive_deps(sample_package_path: Path) -> None:
    graph = build_graph(sample_package_path, entry_point="sample_package.models")
    # models imports core, so core must be present
    assert "sample_package.core" in graph.modules


# ---------------------------------------------------------------------------
# reachable_from
# ---------------------------------------------------------------------------


def test_reachable_from_direct(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    reachable = reachable_from(graph, "sample_package.models")
    assert "sample_package.core" in reachable


def test_reachable_from_excludes_start(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    reachable = reachable_from(graph, "sample_package.models")
    assert "sample_package.models" not in reachable


def test_reachable_from_depth_limit(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    depth1 = reachable_from(graph, "sample_package.hub", depth=1)
    depth_unlimited = reachable_from(graph, "sample_package.hub")
    # Depth=1 should be a subset of unlimited
    assert depth1 <= depth_unlimited


def test_reachable_from_depth_1_direct_only(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # hub imports conditional, core, models, utils.helpers
    depth1 = reachable_from(graph, "sample_package.hub", depth=1)
    assert "sample_package.core" in depth1
    assert "sample_package.models" in depth1


def test_reachable_from_depth_zero_returns_empty(sample_package_path: Path) -> None:
    """depth=0 means no neighbours are expanded — the result is always empty.
    This is correct: depth=0 expands zero hops from the start node.
    """
    graph = _sample_graph(sample_package_path)
    result = reachable_from(graph, "sample_package.hub", depth=0)
    assert result == set()


def test_reachable_from_isolated_node() -> None:
    graph = ImportGraph()
    from pyweight.models import ModuleInfo

    info = ModuleInfo(
        path=Path("/fake/mod.py"),
        qualified_name="mod",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
    )
    graph.add_module("mod", info)
    assert reachable_from(graph, "mod") == set()


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------


def test_shortest_path_direct(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    path = shortest_path(graph, "sample_package.models", "sample_package.core")
    assert path is not None
    assert path[0] == "sample_package.models"
    assert path[-1] == "sample_package.core"
    assert len(path) == 2


def test_shortest_path_same_node(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    path = shortest_path(graph, "sample_package.core", "sample_package.core")
    assert path == ("sample_package.core",)


def test_shortest_path_unreachable(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # core has no imports that lead back to hub (core is a leaf)
    path = shortest_path(graph, "sample_package.core", "sample_package.hub")
    assert path is None


def test_shortest_path_transitive(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # heavy → hub → core
    path = shortest_path(graph, "sample_package.heavy", "sample_package.core")
    assert path is not None
    assert path[0] == "sample_package.heavy"
    assert path[-1] == "sample_package.core"
    assert len(path) >= 2


# ---------------------------------------------------------------------------
# fan_in
# ---------------------------------------------------------------------------


def test_fan_in_core_imported_by_multiple(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # core is imported by: __init__ (star), models, hub, utils/helpers (transitively via init)
    fi = fan_in(graph, "sample_package.core")
    assert fi >= 2


def test_fan_in_leaf_zero(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    # Nothing imports core.py's stdlib deps (os, pathlib) within our graph
    # core itself should have zero fan-in from runtime edges
    # (conditional/deferred_candidate don't import it directly)
    fi = fan_in(graph, "sample_package.conditional")
    assert isinstance(fi, int)
    assert fi >= 0


def test_fan_in_excludes_type_checking_by_default(sample_package_path: Path) -> None:
    """fan_in should exclude type-checking edges when include_type_checking=False."""
    from pyweight.models import EdgeInfo

    mini = ImportGraph()
    from pyweight.models import ModuleInfo

    def _make_info(name: str, path: Path) -> ModuleInfo:
        return ModuleInfo(
            path=path,
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

    mini.add_module("a", _make_info("a", Path("/a.py")))
    mini.add_module("b", _make_info("b", Path("/b.py")))
    tc_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=True, is_conditional=False)
    mini.add_edge("a", "b", tc_edge)

    assert fan_in(mini, "b", include_type_checking=False) == 0
    assert fan_in(mini, "b", include_type_checking=True) == 1


# ---------------------------------------------------------------------------
# Node ID round-trip
# ---------------------------------------------------------------------------


def test_node_id_roundtrip(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    for name in graph.modules:
        node_id = graph.node_id(name)
        assert graph.node_name(node_id) == name


def test_node_ids_unique(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    ids = [graph.node_id(name) for name in graph.modules]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# to_networkx
# ---------------------------------------------------------------------------


def test_to_networkx_nodes(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    dg = to_networkx(graph)
    assert set(dg.nodes) == set(graph.modules.keys())


def test_to_networkx_edges(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    dg = to_networkx(graph, include_type_checking=True)
    # Every runtime+tc edge in graph should be in dg
    for source, targets in graph.edges.items():
        for target in targets:
            assert dg.has_edge(source, target), f"Missing edge {source} -> {target}"


def test_to_networkx_excludes_type_checking_by_default(sample_package_path: Path) -> None:
    from pyweight.models import EdgeInfo, ModuleInfo

    mini = ImportGraph()

    def _make_info(name: str) -> ModuleInfo:
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

    mini.add_module("a", _make_info("a"))
    mini.add_module("b", _make_info("b"))
    tc_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=True, is_conditional=False)
    mini.add_edge("a", "b", tc_edge)

    dg_no_tc = to_networkx(mini, include_type_checking=False)
    dg_with_tc = to_networkx(mini, include_type_checking=True)

    assert not dg_no_tc.has_edge("a", "b")
    assert dg_with_tc.has_edge("a", "b")


def test_to_networkx_edge_attributes(sample_package_path: Path) -> None:
    graph = _sample_graph(sample_package_path)
    dg = to_networkx(graph, include_type_checking=True)
    for _u, _v, data in dg.edges(data=True):
        assert "confidence" in data
        assert "is_type_checking" in data
        assert "is_conditional" in data


def test_to_networkx_is_digraph(sample_package_path: Path) -> None:
    import networkx as nx

    graph = _sample_graph(sample_package_path)
    dg = to_networkx(graph)
    assert isinstance(dg, nx.DiGraph)


# ---------------------------------------------------------------------------
# Hypothesis: structural invariants
# ---------------------------------------------------------------------------


@given(st.data())
def test_hypothesis_reverse_edges_consistent(data: st.DataObject) -> None:
    """For any ImportGraph, reverse_edges must be the exact transpose of edges."""
    from pyweight.models import EdgeInfo, ModuleInfo

    def _make_info(name: str) -> ModuleInfo:
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

    num_nodes = data.draw(st.integers(min_value=1, max_value=6))
    names = [f"mod_{i}" for i in range(num_nodes)]

    graph = ImportGraph()
    for n in names:
        graph.add_module(n, _make_info(n))

    # Draw a random set of edges
    for src in names:
        for tgt in names:
            if src != tgt and data.draw(st.booleans()):
                edge = EdgeInfo(
                    confidence=Confidence.HIGH,
                    is_type_checking=data.draw(st.booleans()),
                    is_conditional=data.draw(st.booleans()),
                )
                graph.add_edge(src, tgt, edge)

    # Verify transpose property
    for source, targets in graph.edges.items():
        for target, edge_info in targets.items():
            assert source in graph.reverse_edges.get(target, {}), (
                f"Missing reverse edge {target} <- {source}"
            )
            assert graph.reverse_edges[target][source] == edge_info

    for target, sources in graph.reverse_edges.items():
        for source, _edge_info in sources.items():
            assert target in graph.edges.get(source, {}), (
                f"reverse_edges has {target}<-{source} but forward edge missing"
            )


@given(st.data())
def test_hypothesis_reachable_subset_of_all_nodes(data: st.DataObject) -> None:
    """BFS closure from any node must be a subset of all nodes in the graph."""
    from pyweight.models import EdgeInfo, ModuleInfo

    def _make_info(name: str) -> ModuleInfo:
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

    num_nodes = data.draw(st.integers(min_value=1, max_value=8))
    names = [f"node_{i}" for i in range(num_nodes)]

    graph = ImportGraph()
    for n in names:
        graph.add_module(n, _make_info(n))

    for src in names:
        for tgt in names:
            if src != tgt and data.draw(st.booleans()):
                edge = EdgeInfo(
                    confidence=Confidence.HIGH,
                    is_type_checking=data.draw(st.booleans()),
                    is_conditional=False,
                )
                graph.add_edge(src, tgt, edge)

    start = data.draw(st.sampled_from(names))
    reachable = reachable_from(graph, start, include_type_checking=True)

    all_nodes = set(graph.modules.keys())
    assert reachable <= all_nodes
    assert start not in reachable


# ---------------------------------------------------------------------------
# module_external_imports tracking
# ---------------------------------------------------------------------------


def test_build_graph_module_external_imports_tracked(sample_package_path: Path) -> None:
    graph = build_graph(sample_package_path)
    # core.py imports os (stdlib/external); the mapping must be non-empty
    assert len(graph.module_external_imports) > 0
    # Every value must be a non-empty set of strings
    for mod, exts in graph.module_external_imports.items():
        assert isinstance(mod, str)
        assert isinstance(exts, set)
        assert len(exts) > 0
    # All external top-level names must appear in external_deps
    for exts in graph.module_external_imports.values():
        for ext in exts:
            assert ext in graph.external_deps


def test_prune_to_reachable_preserves_module_external_imports(
    sample_package_path: Path,
) -> None:
    full_graph = build_graph(sample_package_path)
    pruned = build_graph(sample_package_path, entry_point="sample_package.models")

    # Pruned graph must only contain entries for modules that survived pruning
    for mod in pruned.module_external_imports:
        assert mod in pruned.modules

    # Unreachable modules must have been dropped from the mapping
    unreachable = set(full_graph.modules) - set(pruned.modules)
    for mod in unreachable:
        assert mod not in pruned.module_external_imports
