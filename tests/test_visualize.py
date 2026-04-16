"""Tests for pyweight.visualize module."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pyweight.models import (
    Confidence,
    CostEstimate,
    EdgeInfo,
    ImportGraph,
    ModuleInfo,
    ReachabilityResult,
)
from pyweight.visualize import generate_graph_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(name: str, *, is_init: bool = False) -> ModuleInfo:
    return ModuleInfo(
        path=Path(f"/{name.replace('.', '/')}.py"),
        qualified_name=name,
        imports=(),
        is_init=is_init,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
    )


def _make_cost(module: str, size: int = 2048) -> CostEstimate:
    return CostEstimate(
        module=module,
        dist_name=None,
        direct_size_bytes=size,
        transitive_size_bytes=size,
        confidence=Confidence.HIGH,
        provenance="test",
    )


def _build_small_graph() -> tuple[ImportGraph, dict[str, CostEstimate]]:
    g = ImportGraph()
    for name, is_init in [("pkg", True), ("pkg.core", False), ("pkg.utils", False)]:
        g.add_module(name, _make_module(name, is_init=is_init))
    g.add_edge(
        "pkg.core",
        "pkg.utils",
        EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False),
    )
    g.add_edge(
        "pkg",
        "pkg.core",
        EdgeInfo(confidence=Confidence.MEDIUM, is_type_checking=True, is_conditional=False),
    )
    costs = {name: _make_cost(name) for name in g.modules}
    return g, costs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_graph_html_returns_valid_html() -> None:
    g, costs = _build_small_graph()
    html = generate_graph_html(g, costs)

    assert "<!DOCTYPE html>" in html
    assert "<script" in html
    assert "</html>" in html


def test_generate_graph_html_embeds_node_data() -> None:
    g, costs = _build_small_graph()
    html = generate_graph_html(g, costs)

    assert '"pkg"' in html
    assert '"pkg.core"' in html
    assert '"pkg.utils"' in html


def test_generate_graph_html_embeds_edge_data() -> None:
    g, costs = _build_small_graph()
    html = generate_graph_html(g, costs)

    # The edges JSON block must contain source/target module names
    assert '"pkg.core"' in html
    assert '"pkg.utils"' in html
    # TYPE_CHECKING edge flag embedded
    assert '"is_type_checking":true' in html or '"is_type_checking": true' in html


def test_generate_graph_html_with_reachability() -> None:
    g, costs = _build_small_graph()
    reachability = ReachabilityResult(
        entry_point="pkg",
        required_modules=frozenset({"pkg", "pkg.core"}),
        unreachable_modules=frozenset({"pkg.utils"}),
        required_cost=costs["pkg.core"],
        unreachable_cost=costs["pkg.utils"],
        root_causes=(),
        import_paths={},
        confidence=Confidence.HIGH,
        unknown_origins=(),
    )
    html = generate_graph_html(g, costs, reachability=reachability)

    assert '"required"' in html
    assert '"unreachable"' in html


def test_generate_graph_html_empty_graph() -> None:
    g = ImportGraph()
    html = generate_graph_html(g, {})

    assert "<!DOCTYPE html>" in html
    assert "</html>" in html
    # Nodes array should be empty
    assert "[]" in html


def test_generate_graph_html_contains_d3_reference() -> None:
    g, costs = _build_small_graph()
    html = generate_graph_html(g, costs)

    assert "d3js.org" in html or "d3.v7" in html


def test_generate_graph_html_node_size_bytes_from_costs() -> None:
    g = ImportGraph()
    g.add_module("mod.a", _make_module("mod.a"))
    costs = {"mod.a": _make_cost("mod.a", size=99999)}
    html = generate_graph_html(g, costs)

    assert "99999" in html


def test_generate_graph_html_missing_cost_defaults_to_zero() -> None:
    g = ImportGraph()
    g.add_module("mod.a", _make_module("mod.a"))
    html = generate_graph_html(g, {})  # no costs provided

    # size_bytes should default to 0
    assert '"size_bytes":0' in html or '"size_bytes": 0' in html


def test_generate_graph_html_is_init_flag() -> None:
    g = ImportGraph()
    g.add_module("pkg", _make_module("pkg", is_init=True))
    g.add_module("pkg.sub", _make_module("pkg.sub", is_init=False))
    costs = {name: _make_cost(name) for name in g.modules}
    html = generate_graph_html(g, costs)

    # Both true and false values for is_init must appear
    assert '"is_init":true' in html or '"is_init": true' in html
    assert '"is_init":false' in html or '"is_init": false' in html


def test_generate_graph_html_cdn_param_accepted() -> None:
    """cdn parameter is accepted without error (reserved for future use)."""
    g, costs = _build_small_graph()
    html_cdn_true = generate_graph_html(g, costs, cdn=True)
    html_cdn_false = generate_graph_html(g, costs, cdn=False)

    assert "<!DOCTYPE html>" in html_cdn_true
    assert "<!DOCTYPE html>" in html_cdn_false


def test_generate_graph_html_embedded_json_is_valid() -> None:
    """Verify the embedded nodes/links JSON is syntactically valid."""
    g, costs = _build_small_graph()
    html = generate_graph_html(g, costs)

    # Extract node and link JSON arrays from the JS assignments
    nodes_match = re.search(r"const nodes = (\[.*?\]);", html, re.DOTALL)
    links_match = re.search(r"const links = (\[.*?\]);", html, re.DOTALL)

    assert nodes_match is not None, "nodes assignment not found"
    assert links_match is not None, "links assignment not found"

    nodes_data = json.loads(nodes_match.group(1))
    links_data = json.loads(links_match.group(1))

    assert len(nodes_data) == 3
    assert len(links_data) == 2
    assert all("id" in n for n in nodes_data)
    assert all("source" in e and "target" in e for e in links_data)
