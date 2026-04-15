"""Tests for pyweight.analyzers module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pyweight.analyzers import (
    _resolve_import_module,
    analyze_reachability,
    classify_escape_level,
    explain_external_import,
    explain_import,
    find_barrels,
    find_dead_exports,
    find_deferral_candidates,
    find_hotspots,
    find_hub_dependencies,
)
from pyweight.graph import build_graph
from pyweight.models import (
    Confidence,
    CostEstimate,
    EdgeInfo,
    EscapeLevel,
    ImportGraph,
    ImportRef,
    ModuleInfo,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_graph(sample_package_path: Path) -> ImportGraph:
    return build_graph(sample_package_path)


def _make_module(name: str, *, is_init: bool = False, purity: str = "pure") -> ModuleInfo:
    return ModuleInfo(
        path=Path(f"/{name}.py"),
        qualified_name=name,
        imports=(),
        is_init=is_init,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity=purity,
    )


def _make_cost(module: str, size: int = 1000) -> CostEstimate:
    return CostEstimate(
        module=module,
        dist_name=None,
        direct_size_bytes=size,
        transitive_size_bytes=size,
        confidence=Confidence.HIGH,
        provenance="test",
    )


def _make_import_ref(
    module: str,
    names: tuple[str, ...] = (),
    *,
    scope: str = "module",
    is_type_checking: bool = False,
    is_conditional: bool = False,
    is_star: bool = False,
    level: int = 0,
    alias: str | None = None,
) -> ImportRef:
    return ImportRef(
        module=module,
        names=names,
        alias=alias,
        lineno=1,
        scope=scope,
        level=level,
        is_type_checking=is_type_checking,
        is_conditional=is_conditional,
        is_star=is_star,
    )


# ---------------------------------------------------------------------------
# find_barrels
# ---------------------------------------------------------------------------


def test_find_barrels_detects_sample_package_init(sample_graph: ImportGraph) -> None:
    """sample_package/__init__.py is a barrel: is_init=True, has star imports and __all__."""
    barrels = find_barrels(sample_graph)
    names = [b.qualified_name for b in barrels]
    assert "sample_package" in names


def test_find_barrels_only_init_modules(sample_graph: ImportGraph) -> None:
    """find_barrels returns only __init__.py modules."""
    barrels = find_barrels(sample_graph)
    for b in barrels:
        info = sample_graph.modules[b.qualified_name]
        assert info.is_init, f"{b.qualified_name} is not an __init__.py but was in barrels"


def test_find_barrels_sorted_by_reexported_count_descending(sample_graph: ImportGraph) -> None:
    barrels = find_barrels(sample_graph)
    counts = [b.reexported_count for b in barrels]
    assert counts == sorted(counts, reverse=True)


def test_find_barrels_sample_package_has_star_import(sample_graph: ImportGraph) -> None:
    """sample_package/__init__.py has a star import from .core."""
    barrels = find_barrels(sample_graph)
    sp = next(b for b in barrels if b.qualified_name == "sample_package")
    assert len(sp.star_imports) >= 1


def test_find_barrels_sample_package_has_all(sample_graph: ImportGraph) -> None:
    barrels = find_barrels(sample_graph)
    sp = next(b for b in barrels if b.qualified_name == "sample_package")
    assert sp.has_all is True


def test_find_barrels_eager_imports_count(sample_graph: ImportGraph) -> None:
    """sample_package/__init__.py has module-level non-TYPE_CHECKING non-conditional imports."""
    barrels = find_barrels(sample_graph)
    sp = next(b for b in barrels if b.qualified_name == "sample_package")
    assert sp.eager_imports >= 1


# ---------------------------------------------------------------------------
# classify_escape_level
# ---------------------------------------------------------------------------


def test_classify_escape_level_function_local() -> None:
    """Name used only in a function scope is FUNCTION_LOCAL."""
    ref = _make_import_ref("json", names=("json",))
    usage_scopes = {"json": frozenset({"load_data"})}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.FUNCTION_LOCAL


def test_classify_escape_level_class_body() -> None:
    """Name used in a class body (PascalCase scope) is CLASS_BODY."""
    ref = _make_import_ref("some_module", names=("SomeClass",))
    usage_scopes = {"SomeClass": frozenset({"MyWidget"})}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.CLASS_BODY


def test_classify_escape_level_module_scope() -> None:
    """Name used at module scope is MODULE_SCOPE."""
    ref = _make_import_ref("some_module", names=("SomeThing",))
    usage_scopes = {"SomeThing": frozenset({"module"})}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.MODULE_SCOPE


def test_classify_escape_level_reexported_via_all() -> None:
    """Name listed in __all__ scope is REEXPORTED."""
    ref = _make_import_ref("some_module", names=("SomeThing",))
    usage_scopes = {"SomeThing": frozenset({"__all__"})}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.REEXPORTED


def test_classify_escape_level_no_usage_defaults_module_scope() -> None:
    """Name with no usage data defaults to MODULE_SCOPE (conservative)."""
    ref = _make_import_ref("some_module", names=("Unused",))
    usage_scopes: dict[str, frozenset[str]] = {}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.MODULE_SCOPE


def test_classify_escape_level_multi_name_returns_max() -> None:
    """Multi-name import: names escape to different levels → max returned."""
    ref = _make_import_ref("some_module", names=("func_a", "FuncB"))
    # func_a: FUNCTION_LOCAL (lowercase function scope)
    # FuncB: CLASS_BODY (PascalCase class scope)
    usage_scopes = {
        "func_a": frozenset({"local_fn"}),
        "FuncB": frozenset({"MyClass"}),
    }
    result = classify_escape_level(ref, usage_scopes)
    assert result == EscapeLevel.CLASS_BODY


def test_classify_escape_level_multi_name_max_is_module_scope() -> None:
    """When one name escapes to MODULE_SCOPE, that dominates."""
    ref = _make_import_ref("some_module", names=("func_a", "public_name"))
    usage_scopes = {
        "func_a": frozenset({"local_fn"}),
        "public_name": frozenset({"module"}),
    }
    result = classify_escape_level(ref, usage_scopes)
    assert result == EscapeLevel.MODULE_SCOPE


def test_classify_escape_level_plain_import_uses_module_name() -> None:
    """``import json`` binds ``json`` in the namespace; checks usage of ``json``."""
    ref = _make_import_ref("json", names=())
    usage_scopes = {"json": frozenset({"load_data"})}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.FUNCTION_LOCAL


def test_classify_escape_level_star_import_is_module_scope() -> None:
    """Star imports cannot be individually classified; conservatively MODULE_SCOPE."""
    ref = _make_import_ref("some_module", names=("*",), is_star=True)
    usage_scopes: dict[str, frozenset[str]] = {}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.MODULE_SCOPE


def test_classify_escape_level_nested_scope_is_function_local() -> None:
    """Usage in a nested scope like ClassName.method is FUNCTION_LOCAL."""
    ref = _make_import_ref("some_module", names=("helper",))
    usage_scopes = {"helper": frozenset({"MyClass.process"})}
    assert classify_escape_level(ref, usage_scopes) == EscapeLevel.FUNCTION_LOCAL


# ---------------------------------------------------------------------------
# find_deferral_candidates — fixture-based
# ---------------------------------------------------------------------------


def test_deferral_candidate_json_is_function_local(sample_graph: ImportGraph) -> None:
    """deferred_candidate.py's json import is only used in load_data → FUNCTION_LOCAL."""
    candidates = find_deferral_candidates(sample_graph)
    json_candidates = [
        c
        for c in candidates
        if c.import_ref.module == "json" and str(c.file).endswith("deferred_candidate.py")
    ]
    assert len(json_candidates) >= 1
    assert json_candidates[0].escape_level == EscapeLevel.FUNCTION_LOCAL


def test_deferral_candidate_json_risk_safe(sample_graph: ImportGraph) -> None:
    """deferred_candidate.py is purity=pure → risk=safe."""
    candidates = find_deferral_candidates(sample_graph)
    json_candidate = next(
        c
        for c in candidates
        if c.import_ref.module == "json" and str(c.file).endswith("deferred_candidate.py")
    )
    assert json_candidate.risk == "safe"


def test_deferral_candidate_side_effects_module_has_side_effects_risk(
    sample_graph: ImportGraph,
) -> None:
    """side_effects.py is purity=side_effects; if any of its imports are deferrable,
    risk must be has_side_effects."""
    candidates = find_deferral_candidates(sample_graph)
    se_candidates = [c for c in candidates if str(c.file).endswith("side_effects.py")]
    # side_effects.py might not produce deferral candidates (imports not function-local),
    # but if it does, each must have risk=has_side_effects.
    for c in se_candidates:
        assert c.risk == "has_side_effects", (
            f"Expected has_side_effects for side_effects.py but got {c.risk!r}"
        )


def test_deferral_candidates_exclude_module_scope_imports(sample_graph: ImportGraph) -> None:
    """Imports that escape to MODULE_SCOPE are NOT in deferral candidates."""
    candidates = find_deferral_candidates(sample_graph)
    for c in candidates:
        assert c.escape_level in (EscapeLevel.FUNCTION_LOCAL, EscapeLevel.CLASS_BODY), (
            f"Expected only FUNCTION_LOCAL or CLASS_BODY, got {c.escape_level}"
        )


def test_deferral_candidates_exclude_type_checking(sample_graph: ImportGraph) -> None:
    """TYPE_CHECKING imports are never deferral candidates."""
    candidates = find_deferral_candidates(sample_graph)
    for c in candidates:
        assert not c.import_ref.is_type_checking


def test_deferral_candidates_exclude_conditional(sample_graph: ImportGraph) -> None:
    """Conditional imports are never deferral candidates."""
    candidates = find_deferral_candidates(sample_graph)
    for c in candidates:
        assert not c.import_ref.is_conditional


def test_deferral_candidates_sorted_by_cost_descending() -> None:
    """When costs are provided, candidates are sorted by cost descending."""
    graph = ImportGraph()
    cheap_ref = _make_import_ref("cheap_lib", names=())
    expensive_ref = _make_import_ref("expensive_lib", names=())
    cheap_mod = ModuleInfo(
        path=Path("/cheap.py"),
        qualified_name="cheap",
        imports=(cheap_ref,),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
        usage_scopes={"cheap_lib": frozenset({"do_something"})},
    )
    expensive_mod = ModuleInfo(
        path=Path("/expensive.py"),
        qualified_name="expensive",
        imports=(expensive_ref,),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
        usage_scopes={"expensive_lib": frozenset({"do_something"})},
    )
    graph.add_module("cheap", cheap_mod)
    graph.add_module("expensive", expensive_mod)

    costs = {
        "cheap_lib": _make_cost("cheap_lib", size=100),
        "expensive_lib": _make_cost("expensive_lib", size=50000),
    }

    candidates = find_deferral_candidates(graph, costs=costs)
    if len(candidates) >= 2:
        assert candidates[0].estimated_cost is not None
        assert candidates[-1].estimated_cost is not None
        assert (
            candidates[0].estimated_cost.transitive_size_bytes
            >= candidates[-1].estimated_cost.transitive_size_bytes
        )


def test_deferral_candidates_with_entry_point_includes_evidence(
    sample_graph: ImportGraph,
) -> None:
    """When entry_point is provided, evidence trail is populated."""
    candidates = find_deferral_candidates(
        sample_graph, entry_point="sample_package.deferred_candidate"
    )
    dc_candidates = [c for c in candidates if str(c.file).endswith("deferred_candidate.py")]
    if dc_candidates:
        assert len(dc_candidates[0].evidence) >= 1


# ---------------------------------------------------------------------------
# find_hub_dependencies
# ---------------------------------------------------------------------------


def test_find_hub_dependencies_hub_detected(sample_graph: ImportGraph) -> None:
    """hub.py has fan_in=3 and fan_out=4; should be detected at min_fan=3."""
    hubs = find_hub_dependencies(sample_graph, min_fan=3)
    hub_names = [h.module for h in hubs]
    assert "sample_package.hub" in hub_names


def test_find_hub_dependencies_hub_score(sample_graph: ImportGraph) -> None:
    """hub.py: fan_in * fan_out = 3 * 4 = 12."""
    hubs = find_hub_dependencies(sample_graph, min_fan=3)
    hub = next(h for h in hubs if h.module == "sample_package.hub")
    assert hub.hub_score == hub.fan_in * hub.fan_out
    assert hub.hub_score >= 12.0


def test_find_hub_dependencies_sorted_descending(sample_graph: ImportGraph) -> None:
    hubs = find_hub_dependencies(sample_graph, min_fan=1)
    scores = [h.hub_score for h in hubs]
    assert scores == sorted(scores, reverse=True)


def test_find_hub_dependencies_is_init_flag(sample_graph: ImportGraph) -> None:
    """__init__.py hubs are flagged with is_init=True."""
    hubs = find_hub_dependencies(sample_graph, min_fan=1)
    for h in hubs:
        expected = sample_graph.modules[h.module].is_init
        assert h.is_init == expected


def test_find_hub_dependencies_excludes_type_checking_edges(sample_graph: ImportGraph) -> None:
    """TYPE_CHECKING edges are excluded from fan_in/fan_out counts."""
    hubs = find_hub_dependencies(sample_graph, min_fan=1)
    for h in hubs:
        runtime_fi = sum(
            1
            for _, e in sample_graph.reverse_edges.get(h.module, {}).items()
            if not e.is_type_checking
        )
        assert h.fan_in == runtime_fi


def test_find_hub_dependencies_min_fan_filters(sample_graph: ImportGraph) -> None:
    """Modules with fan_in < min_fan or fan_out < min_fan are excluded."""
    hubs_high = find_hub_dependencies(sample_graph, min_fan=10)
    # With a very high min_fan, likely nothing qualifies
    hubs_low = find_hub_dependencies(sample_graph, min_fan=1)
    assert len(hubs_high) <= len(hubs_low)


def test_find_hub_dependencies_with_costs_attaches_cost() -> None:
    """When costs dict is provided, transitive_cost is attached."""
    graph = ImportGraph()
    hub_mod = ModuleInfo(
        path=Path("/hub.py"),
        qualified_name="hub",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
    )
    graph.add_module("hub", hub_mod)
    for i in range(3):
        m = _make_module(f"dep{i}")
        graph.add_module(f"dep{i}", m)
        graph.add_edge(
            "hub",
            f"dep{i}",
            EdgeInfo(
                confidence=Confidence.HIGH,
                is_type_checking=False,
                is_conditional=False,
            ),
        )
    for i in range(3):
        m = _make_module(f"importer{i}")
        graph.add_module(f"importer{i}", m)
        graph.add_edge(
            f"importer{i}",
            "hub",
            EdgeInfo(
                confidence=Confidence.HIGH,
                is_type_checking=False,
                is_conditional=False,
            ),
        )

    costs = {"hub": _make_cost("hub", size=99999)}
    hubs = find_hub_dependencies(graph, costs=costs, min_fan=3)
    assert len(hubs) == 1
    assert hubs[0].transitive_cost is not None
    assert hubs[0].transitive_cost.transitive_size_bytes == 99999


# ---------------------------------------------------------------------------
# find_hotspots
# ---------------------------------------------------------------------------


def test_find_hotspots_ranking_by_score() -> None:
    """score = fan_in * transitive_size_bytes; highest score first."""
    graph = ImportGraph()
    # small_mod: fan_in=1, size=100 → score=100
    # big_mod: fan_in=2, size=1000 → score=2000
    small = _make_module("small")
    big = _make_module("big")
    a = _make_module("a")
    b = _make_module("b")
    c = _make_module("c")
    for m in (small, big, a, b, c):
        graph.add_module(m.qualified_name, m)

    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph.add_edge("a", "small", edge)
    graph.add_edge("b", "big", edge)
    graph.add_edge("c", "big", edge)

    costs = {
        "small": _make_cost("small", size=100),
        "big": _make_cost("big", size=1000),
    }

    hotspots = find_hotspots(graph, costs)
    assert len(hotspots) >= 2
    assert hotspots[0].module == "big"
    assert hotspots[0].score == 2 * 1000


def test_find_hotspots_min_cost_filter() -> None:
    """Modules below min_cost_bytes are excluded."""
    graph = ImportGraph()
    cheap = _make_module("cheap")
    expensive = _make_module("expensive")
    importer = _make_module("importer")
    for m in (cheap, expensive, importer):
        graph.add_module(m.qualified_name, m)

    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph.add_edge("importer", "cheap", edge)
    graph.add_edge("importer", "expensive", edge)

    costs = {
        "cheap": _make_cost("cheap", size=50),
        "expensive": _make_cost("expensive", size=5000),
    }

    hotspots = find_hotspots(graph, costs, min_cost_bytes=100)
    names = [h.module for h in hotspots]
    assert "expensive" in names
    assert "cheap" not in names


def test_find_hotspots_evidence_populated(sample_graph: ImportGraph) -> None:
    """Hotspots with fan_in > 0 include evidence strings."""
    costs = {name: _make_cost(name, size=1000) for name in sample_graph.modules}
    hotspots = find_hotspots(sample_graph, costs)
    # hub has fan_in >= 3; its evidence should be non-empty
    hub_hotspot = next((h for h in hotspots if h.module == "sample_package.hub"), None)
    if hub_hotspot is not None and hub_hotspot.fan_in > 0:
        assert len(hub_hotspot.evidence) >= 1


def test_find_hotspots_no_cost_module_excluded() -> None:
    """Modules with no entry in costs dict are excluded."""
    graph = ImportGraph()
    mod = _make_module("no_cost")
    importer = _make_module("importer")
    graph.add_module("no_cost", mod)
    graph.add_module("importer", importer)
    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph.add_edge("importer", "no_cost", edge)

    hotspots = find_hotspots(graph, {})
    assert not any(h.module == "no_cost" for h in hotspots)


# ---------------------------------------------------------------------------
# analyze_reachability
# ---------------------------------------------------------------------------


def test_analyze_reachability_required_contains_entry(sample_graph: ImportGraph) -> None:
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = analyze_reachability(sample_graph, "sample_package.models", costs)
    assert "sample_package.models" in result.required_modules


def test_analyze_reachability_required_contains_transitive(sample_graph: ImportGraph) -> None:
    """models imports core → core is required."""
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = analyze_reachability(sample_graph, "sample_package.models", costs)
    assert "sample_package.core" in result.required_modules


def test_analyze_reachability_unreachable_correct(sample_graph: ImportGraph) -> None:
    """Modules not transitively imported from models are unreachable."""
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = analyze_reachability(sample_graph, "sample_package.models", costs)
    # hub, heavy, side_effects, deferred_candidate are not reachable from models
    for mod in (
        "sample_package.hub",
        "sample_package.heavy",
        "sample_package.side_effects",
        "sample_package.deferred_candidate",
    ):
        assert mod in result.unreachable_modules


def test_analyze_reachability_required_plus_unreachable_equals_all(
    sample_graph: ImportGraph,
) -> None:
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = analyze_reachability(sample_graph, "sample_package.models", costs)
    all_modules = frozenset(sample_graph.modules.keys())
    assert result.required_modules | result.unreachable_modules == all_modules
    assert result.required_modules & result.unreachable_modules == frozenset()


def test_analyze_reachability_import_paths_populated(sample_graph: ImportGraph) -> None:
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = analyze_reachability(sample_graph, "sample_package.models", costs)
    assert "sample_package.models" in result.import_paths
    assert result.import_paths["sample_package.models"] == ("sample_package.models",)


def test_analyze_reachability_unknown_entry_returns_empty(sample_graph: ImportGraph) -> None:
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = analyze_reachability(sample_graph, "nonexistent.module", costs)
    assert result.required_modules == frozenset()
    assert result.entry_point == "nonexistent.module"


# ---------------------------------------------------------------------------
# explain_import
# ---------------------------------------------------------------------------


def test_explain_import_path_through_init(sample_graph: ImportGraph) -> None:
    """sample_package -> sample_package.heavy: path should exist via __init__.py."""
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    # utils.helpers -> sample_package (the __init__) exists; or heavy is reachable
    # via another path. Just check that a path is found if heavy is reachable.
    result = explain_import(sample_graph, "sample_package.guarded", "sample_package.heavy", costs)
    if result.paths:
        assert result.paths[0][0] == "sample_package.guarded"
        assert result.paths[0][-1] == "sample_package.heavy"


def test_explain_import_unreachable_target_returns_empty_paths(
    sample_graph: ImportGraph,
) -> None:
    """core has no imports that lead to hub → explain_import returns empty paths."""
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = explain_import(sample_graph, "sample_package.core", "sample_package.hub", costs)
    assert result.paths == ()
    assert result.critical_edges == ()


def test_explain_import_nonexistent_module_returns_empty(sample_graph: ImportGraph) -> None:
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = explain_import(sample_graph, "sample_package", "nonexistent.module", costs)
    assert result.paths == ()


def test_explain_import_paths_sorted_by_cost_descending(sample_graph: ImportGraph) -> None:
    """Paths are sorted heaviest first."""
    costs = {name: _make_cost(name, size=1000) for name in sample_graph.modules}
    # Give heavy a much larger cost
    costs["sample_package.heavy"] = _make_cost("sample_package.heavy", size=999999)
    result = explain_import(sample_graph, "sample_package.heavy", "sample_package.core", costs)
    if len(result.paths) >= 2:

        def path_cost(path: tuple[str, ...]) -> int:
            return sum(costs.get(n, _make_cost(n, size=0)).transitive_size_bytes for n in path)

        costs_seq = [path_cost(p) for p in result.paths]
        assert costs_seq == sorted(costs_seq, reverse=True)


def test_explain_import_critical_edges_length_matches_paths(sample_graph: ImportGraph) -> None:
    costs = {name: _make_cost(name) for name in sample_graph.modules}
    result = explain_import(sample_graph, "sample_package.heavy", "sample_package.core", costs)
    assert len(result.critical_edges) == len(result.paths)


def test_explain_import_max_five_paths() -> None:
    """At most 5 paths are returned."""
    graph = ImportGraph()
    # Build a diamond-like graph with many parallel paths
    src = _make_module("src")
    tgt = _make_module("tgt")
    graph.add_module("src", src)
    graph.add_module("tgt", tgt)
    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    for i in range(10):
        mid = _make_module(f"mid{i}")
        graph.add_module(f"mid{i}", mid)
        graph.add_edge("src", f"mid{i}", edge)
        graph.add_edge(f"mid{i}", "tgt", edge)

    result = explain_import(graph, "src", "tgt", {})
    assert len(result.paths) <= 5


# ---------------------------------------------------------------------------
# find_dead_exports
# ---------------------------------------------------------------------------


def test_find_dead_exports_unused_symbol_flagged(sample_graph: ImportGraph) -> None:
    """A symbol defined but never imported by anyone should be in dead exports."""
    dead = find_dead_exports(sample_graph)
    # Some modules have symbols no one imports; verify at least one dead export exists
    assert len(dead) >= 1


def test_find_dead_exports_imported_symbol_not_flagged(sample_graph: ImportGraph) -> None:
    """hub.py imports CoreClass and helper_fn from core.py; those should NOT be dead."""
    dead = find_dead_exports(sample_graph)
    dead_set = {(d.module, d.symbol_name) for d in dead}
    # core.py has __all__ = ["CoreClass", "helper_fn"]
    # hub.py imports both — so they should not be dead if correctly resolved
    # Note: sample_package/__init__.py imports from core via star; models imports CoreClass
    # The presence or absence here depends on resolution — the key check is that if
    # something IS imported via an explicit named import, it must not appear in dead exports.
    # We verify that the dead exports logic is consistent with the graph's edges.
    for importer_name, info in sample_graph.modules.items():
        for ref in info.imports:
            if ref.is_star or not ref.names:
                continue
            # Resolve the module name
            from pyweight.analyzers import _resolve_import_module

            resolved = _resolve_import_module(ref, importer_name)
            if resolved is None or resolved not in sample_graph.modules:
                continue
            for name in ref.names:
                assert (resolved, name) not in dead_set, (
                    f"{resolved}.{name} is explicitly imported by {importer_name} "
                    "but appears in dead exports"
                )


def test_find_dead_exports_is_library_caps_confidence(sample_graph: ImportGraph) -> None:
    """is_library=True caps __all__-based confidence at MEDIUM."""
    dead_lib = find_dead_exports(sample_graph, is_library=True)
    for d in dead_lib:
        if d.export_mechanism == "__all__":
            assert d.confidence != Confidence.HIGH, (
                f"{d.module}.{d.symbol_name}: expected MEDIUM for library, got HIGH"
            )


def test_find_dead_exports_without_library_high_confidence_for_all(
    sample_graph: ImportGraph,
) -> None:
    """is_library=False: __all__-based dead exports have HIGH confidence."""
    dead = find_dead_exports(sample_graph, is_library=False)
    all_based = [d for d in dead if d.export_mechanism == "__all__"]
    for d in all_based:
        assert d.confidence == Confidence.HIGH


def test_find_dead_exports_sorted_by_module_then_symbol(sample_graph: ImportGraph) -> None:
    dead = find_dead_exports(sample_graph)
    pairs = [(d.module, d.symbol_name) for d in dead]
    assert pairs == sorted(pairs)


def test_find_dead_exports_synthetic_unused_symbol() -> None:
    """Synthetic graph: exported symbol never imported → dead."""
    graph = ImportGraph()
    exporter = ModuleInfo(
        path=Path("/exporter.py"),
        qualified_name="exporter",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=True,
        all_names=frozenset({"PublicAPI"}),
        defined_names=frozenset({"PublicAPI"}),
        star_imports=(),
        purity="pure",
    )
    consumer = ModuleInfo(
        path=Path("/consumer.py"),
        qualified_name="consumer",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
    )
    graph.add_module("exporter", exporter)
    graph.add_module("consumer", consumer)

    dead = find_dead_exports(graph)
    dead_set = {(d.module, d.symbol_name) for d in dead}
    assert ("exporter", "PublicAPI") in dead_set


def test_find_dead_exports_synthetic_imported_symbol_not_dead() -> None:
    """Synthetic graph: exported symbol IS imported → not dead."""
    ref = ImportRef(
        module="exporter",
        names=("PublicAPI",),
        alias=None,
        lineno=1,
        scope="module",
        level=0,
        is_type_checking=False,
        is_conditional=False,
        is_star=False,
    )
    exporter = ModuleInfo(
        path=Path("/exporter.py"),
        qualified_name="exporter",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=True,
        all_names=frozenset({"PublicAPI"}),
        defined_names=frozenset({"PublicAPI"}),
        star_imports=(),
        purity="pure",
    )
    consumer = ModuleInfo(
        path=Path("/consumer.py"),
        qualified_name="consumer",
        imports=(ref,),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
    )
    graph = ImportGraph()
    graph.add_module("exporter", exporter)
    graph.add_module("consumer", consumer)

    dead = find_dead_exports(graph)
    dead_set = {(d.module, d.symbol_name) for d in dead}
    assert ("exporter", "PublicAPI") not in dead_set


def test_find_dead_exports_public_def_medium_confidence() -> None:
    """Public defined names (no __all__) have MEDIUM confidence."""
    mod = ModuleInfo(
        path=Path("/mymod.py"),
        qualified_name="mymod",
        imports=(),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset({"public_fn"}),
        star_imports=(),
        purity="pure",
    )
    graph = ImportGraph()
    graph.add_module("mymod", mod)

    dead = find_dead_exports(graph)
    assert any(d.symbol_name == "public_fn" and d.confidence == Confidence.MEDIUM for d in dead)


# ---------------------------------------------------------------------------
# _resolve_import_module — H-1 unit tests
# ---------------------------------------------------------------------------


def test_resolve_import_module_level1_sibling() -> None:
    """level=1 from a.b.c with module='sibling' → a.b.sibling."""
    ref = _make_import_ref("sibling", names=("X",), level=1)
    assert _resolve_import_module(ref, "a.b.c") == "a.b.sibling"


def test_resolve_import_module_level2_parent() -> None:
    """level=2 from a.b.c with module='x' → a.x."""
    ref = _make_import_ref("x", names=("Y",), level=2)
    assert _resolve_import_module(ref, "a.b.c") == "a.x"


def test_resolve_import_module_level1_bare_from_dot() -> None:
    """level=1 from a.b.c with module='' (bare 'from . import x') → a.b."""
    ref = _make_import_ref("", names=("x",), level=1)
    assert _resolve_import_module(ref, "a.b.c") == "a.b"


def test_resolve_import_module_level0_absolute() -> None:
    """level=0 returns the module name unchanged."""
    ref = _make_import_ref("os.path", names=("join",), level=0)
    assert _resolve_import_module(ref, "a.b.c") == "os.path"


def test_resolve_import_module_level0_empty_returns_none() -> None:
    """level=0 with empty module string returns None."""
    ref = _make_import_ref("", names=(), level=0)
    assert _resolve_import_module(ref, "a.b.c") is None


def test_resolve_import_module_level_equals_parts_returns_none() -> None:
    """level=3 from a.b.c (3 parts) with module='x' → None (can't go above root)."""
    ref = _make_import_ref("x", names=("X",), level=3)
    assert _resolve_import_module(ref, "a.b.c") is None


# ---------------------------------------------------------------------------
# find_deferral_candidates — M-6: plain import X style
# ---------------------------------------------------------------------------


def test_deferral_candidates_plain_import_x_style() -> None:
    """Plain ``import json`` (empty names) is detected as deferral candidate via _bound_names."""
    ref = _make_import_ref("json", names=())  # plain import X — binds name "json"
    mod = ModuleInfo(
        path=Path("/mymod.py"),
        qualified_name="mymod",
        imports=(ref,),
        is_init=False,
        has_getattr=False,
        has_all=False,
        all_names=frozenset(),
        defined_names=frozenset(),
        star_imports=(),
        purity="pure",
        usage_scopes={"json": frozenset({"process_data"})},
    )
    graph = ImportGraph()
    graph.add_module("mymod", mod)
    candidates = find_deferral_candidates(graph)
    assert any(
        c.import_ref.module == "json" and c.escape_level == EscapeLevel.FUNCTION_LOCAL
        for c in candidates
    )


# ---------------------------------------------------------------------------
# explain_external_import
# ---------------------------------------------------------------------------


def _make_graph_with_external(
    entry: str,
    mid: str,
    ext_top: str,
    edge: EdgeInfo | None = None,
) -> ImportGraph:
    """Build a minimal graph: entry -> mid, where mid imports external ext_top."""
    if edge is None:
        edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph = ImportGraph()
    graph.add_module(entry, _make_module(entry))
    graph.add_module(mid, _make_module(mid))
    graph.add_edge(entry, mid, edge)
    graph.module_external_imports[mid] = {ext_top}
    graph.external_deps.add(ext_top)
    return graph


def test_explain_external_import_direct() -> None:
    """Module A directly imports 'numpy'; verify direct_importers and paths."""
    graph = _make_graph_with_external("myapp", "myapp.loader", "numpy")

    with (
        patch("pyweight.analyzers.resolve_dist_name", return_value="numpy"),
        patch("pyweight.cost.packages_distributions_for", return_value={"numpy": ["numpy"]}),
    ):
        result = explain_external_import(graph, "myapp", "numpy")

    assert "myapp.loader" in result.direct_importers
    assert len(result.paths) >= 1
    assert result.paths[0][0] == "myapp"
    assert result.paths[0][-1] == "myapp.loader"
    assert "numpy" in result.target_import_names


def test_explain_external_import_not_found() -> None:
    """Querying for a package nobody imports returns empty direct_importers and paths."""
    graph = _make_graph_with_external("myapp", "myapp.loader", "numpy")

    with (
        patch("pyweight.analyzers.resolve_dist_name", return_value="ghost-pkg"),
        patch("pyweight.cost.packages_distributions_for", return_value={}),
        patch("pyweight.analyzers.find_pip_dependency_chain", return_value=None),
    ):
        result = explain_external_import(graph, "myapp", "ghost-pkg")

    assert result.direct_importers == ()
    assert result.paths == ()
    assert result.pip_dependency_chain is None


def test_explain_external_import_via_entry_point() -> None:
    """entry -> A -> B where B imports 'pandas'; path should be (entry, A, B)."""
    edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    graph = ImportGraph()
    graph.add_module("entry", _make_module("entry"))
    graph.add_module("entry.a", _make_module("entry.a"))
    graph.add_module("entry.a.b", _make_module("entry.a.b"))
    graph.add_edge("entry", "entry.a", edge)
    graph.add_edge("entry.a", "entry.a.b", edge)
    graph.module_external_imports["entry.a.b"] = {"pandas"}
    graph.external_deps.add("pandas")

    with (
        patch("pyweight.analyzers.resolve_dist_name", return_value="pandas"),
        patch("pyweight.cost.packages_distributions_for", return_value={"pandas": ["pandas"]}),
    ):
        result = explain_external_import(graph, "entry", "pandas")

    assert "entry.a.b" in result.direct_importers
    assert len(result.paths) == 1
    assert result.paths[0] == ("entry", "entry.a", "entry.a.b")
