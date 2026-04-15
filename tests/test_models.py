from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyweight.models import (
    BarrelInfo,
    Confidence,
    CostEstimate,
    DeadExport,
    DeferralCandidate,
    DSMResult,
    EdgeInfo,
    EscapeLevel,
    ExternalWhyResult,
    FullReport,
    Hotspot,
    HubDependency,
    ImportGraph,
    ImportRef,
    MemoryResult,
    ModuleInfo,
    Purity,
    ReachabilityResult,
    RefactoringKind,
    RiskLevel,
    SuggestedRefactoring,
    TimingResult,
    WhyResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_import_ref(
    module: str = "os",
    names: tuple[str, ...] = ("path",),
    alias: str | None = None,
    lineno: int = 1,
    scope: str = "module",
    level: int = 0,
    is_type_checking: bool = False,
    is_conditional: bool = False,
    is_star: bool = False,
) -> ImportRef:
    return ImportRef(
        module=module,
        names=names,
        alias=alias,
        lineno=lineno,
        scope=scope,
        level=level,
        is_type_checking=is_type_checking,
        is_conditional=is_conditional,
        is_star=is_star,
    )


def make_cost_estimate(
    module: str = "requests",
    dist_name: str | None = "requests",
    direct_size_bytes: int = 1024,
    transitive_size_bytes: int = 4096,
    confidence: Confidence = Confidence.HIGH,
    provenance: str = "wheel",
) -> CostEstimate:
    return CostEstimate(
        module=module,
        dist_name=dist_name,
        direct_size_bytes=direct_size_bytes,
        transitive_size_bytes=transitive_size_bytes,
        confidence=confidence,
        provenance=provenance,
    )


def make_module_info(
    path: Path = Path("/pkg/foo.py"),
    qualified_name: str = "pkg.foo",
    imports: tuple[ImportRef, ...] = (),
    is_init: bool = False,
    has_getattr: bool = False,
    has_all: bool = False,
    all_names: frozenset[str] = frozenset(),
    defined_names: frozenset[str] = frozenset({"MyClass"}),
    star_imports: tuple[str, ...] = (),
    purity: Purity = Purity.PURE,
    usage_scopes: dict[str, frozenset[str]] | None = None,
) -> ModuleInfo:
    return ModuleInfo(
        path=path,
        qualified_name=qualified_name,
        imports=imports,
        is_init=is_init,
        has_getattr=has_getattr,
        has_all=has_all,
        all_names=all_names,
        defined_names=defined_names,
        star_imports=star_imports,
        purity=purity,
        usage_scopes=usage_scopes if usage_scopes is not None else {},
    )


# ---------------------------------------------------------------------------
# Confidence enum
# ---------------------------------------------------------------------------


def test_confidence_values() -> None:
    assert Confidence.HIGH == "high"
    assert Confidence.MEDIUM == "medium"
    assert Confidence.LOW == "low"


def test_confidence_str() -> None:
    assert str(Confidence.HIGH) == "high"
    assert str(Confidence.LOW) == "low"


def test_confidence_from_string() -> None:
    assert Confidence("high") is Confidence.HIGH
    assert Confidence("medium") is Confidence.MEDIUM
    assert Confidence("low") is Confidence.LOW


# ---------------------------------------------------------------------------
# EscapeLevel ordering
# ---------------------------------------------------------------------------


def test_escape_level_ordering_all_pairs() -> None:
    order = [
        EscapeLevel.FUNCTION_LOCAL,
        EscapeLevel.CLASS_BODY,
        EscapeLevel.MODULE_SCOPE,
        EscapeLevel.REEXPORTED,
    ]
    for i, lower in enumerate(order):
        for j, higher in enumerate(order):
            if i < j:
                assert lower < higher, f"{lower} should be < {higher}"
                assert higher > lower, f"{higher} should be > {lower}"
                assert lower != higher
            elif i == j:
                assert not (lower < higher)
                assert not (lower > higher)
                assert lower == higher
            else:
                assert not (lower < higher)
                assert lower > higher


def test_escape_level_not_lexicographic() -> None:
    # Lexicographically "class_body" < "function_local" would be True
    # but our semantic ordering says FUNCTION_LOCAL < CLASS_BODY
    assert "class_body" < "function_local"  # lexicographic order
    assert EscapeLevel.FUNCTION_LOCAL < EscapeLevel.CLASS_BODY  # semantic order


def test_escape_level_total_ordering() -> None:
    levels = sorted(
        [
            EscapeLevel.REEXPORTED,
            EscapeLevel.FUNCTION_LOCAL,
            EscapeLevel.MODULE_SCOPE,
            EscapeLevel.CLASS_BODY,
        ]
    )
    assert levels == [
        EscapeLevel.FUNCTION_LOCAL,
        EscapeLevel.CLASS_BODY,
        EscapeLevel.MODULE_SCOPE,
        EscapeLevel.REEXPORTED,
    ]


def test_escape_level_hash() -> None:
    s: set[EscapeLevel] = {EscapeLevel.FUNCTION_LOCAL, EscapeLevel.CLASS_BODY}
    assert EscapeLevel.FUNCTION_LOCAL in s
    assert EscapeLevel.MODULE_SCOPE not in s


# ---------------------------------------------------------------------------
# ImportRef
# ---------------------------------------------------------------------------


def test_import_ref_construction() -> None:
    ref = make_import_ref()
    assert ref.module == "os"
    assert ref.names == ("path",)
    assert ref.alias is None
    assert ref.lineno == 1
    assert ref.level == 0
    assert not ref.is_type_checking
    assert not ref.is_conditional
    assert not ref.is_star


def test_import_ref_frozen() -> None:
    ref = make_import_ref()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.module = "sys"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EdgeInfo
# ---------------------------------------------------------------------------


def test_edge_info_construction() -> None:
    ei = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=True)
    assert ei.confidence is Confidence.HIGH
    assert ei.is_conditional


def test_edge_info_frozen() -> None:
    ei = EdgeInfo(confidence=Confidence.LOW, is_type_checking=False, is_conditional=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ei.confidence = Confidence.HIGH  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModuleInfo
# ---------------------------------------------------------------------------


def test_module_info_construction() -> None:
    ref = make_import_ref()
    mi = make_module_info(imports=(ref,), all_names=frozenset({"foo", "bar"}))
    assert mi.qualified_name == "pkg.foo"
    assert len(mi.imports) == 1
    assert "foo" in mi.all_names


def test_module_info_frozen() -> None:
    mi = make_module_info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        mi.qualified_name = "other"  # type: ignore[misc]


def test_module_info_default_usage_scopes() -> None:
    mi = make_module_info()
    assert mi.usage_scopes == {}


# ---------------------------------------------------------------------------
# CostEstimate
# ---------------------------------------------------------------------------


def test_cost_estimate_construction() -> None:
    ce = make_cost_estimate()
    assert ce.module == "requests"
    assert ce.dist_name == "requests"
    assert ce.direct_size_bytes == 1024
    assert ce.confidence is Confidence.HIGH


def test_cost_estimate_frozen() -> None:
    ce = make_cost_estimate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ce.module = "other"  # type: ignore[misc]


def test_cost_estimate_no_dist_name() -> None:
    ce = make_cost_estimate(dist_name=None)
    assert ce.dist_name is None


# ---------------------------------------------------------------------------
# TimingResult
# ---------------------------------------------------------------------------


def test_timing_result_construction() -> None:
    tr = TimingResult(
        module="mymod",
        self_time_us=100,
        cumulative_time_us=500,
        dependencies=("dep1", "dep2"),
    )
    assert tr.self_time_us == 100
    assert tr.dependencies == ("dep1", "dep2")


def test_timing_result_frozen() -> None:
    tr = TimingResult(module="m", self_time_us=1, cumulative_time_us=2, dependencies=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        tr.module = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryResult
# ---------------------------------------------------------------------------


def test_memory_result_construction() -> None:
    mr = MemoryResult(module="mymod", python_heap_bytes=2048, rss_bytes=8192)
    assert mr.python_heap_bytes == 2048
    assert mr.rss_bytes == 8192


def test_memory_result_frozen() -> None:
    mr = MemoryResult(module="m", python_heap_bytes=0, rss_bytes=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mr.module = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DeferralCandidate
# ---------------------------------------------------------------------------


def test_deferral_candidate_construction() -> None:
    dc = DeferralCandidate(
        file=Path("/a/b.py"),
        import_ref=make_import_ref(),
        used_in_scopes=frozenset({"fn_a"}),
        escape_level=EscapeLevel.FUNCTION_LOCAL,
        estimated_cost=make_cost_estimate(),
        confidence=Confidence.MEDIUM,
        risk=RiskLevel.SAFE,
        evidence=("used once",),
    )
    assert dc.escape_level is EscapeLevel.FUNCTION_LOCAL
    assert "fn_a" in dc.used_in_scopes


def test_deferral_candidate_frozen() -> None:
    dc = DeferralCandidate(
        file=Path("/a/b.py"),
        import_ref=make_import_ref(),
        used_in_scopes=frozenset(),
        escape_level=EscapeLevel.MODULE_SCOPE,
        estimated_cost=None,
        confidence=Confidence.LOW,
        risk=RiskLevel.HAS_SIDE_EFFECTS,
        evidence=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        dc.risk = RiskLevel.SAFE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BarrelInfo
# ---------------------------------------------------------------------------


def test_barrel_info_construction() -> None:
    bi = BarrelInfo(
        path=Path("/pkg/__init__.py"),
        qualified_name="pkg",
        star_imports=("pkg.a", "pkg.b"),
        eager_imports=5,
        reexported_count=12,
        has_lazy_loading=False,
        has_all=True,
    )
    assert bi.eager_imports == 5
    assert bi.has_all


def test_barrel_info_frozen() -> None:
    bi = BarrelInfo(
        path=Path("/p/__init__.py"),
        qualified_name="p",
        star_imports=(),
        eager_imports=0,
        reexported_count=0,
        has_lazy_loading=False,
        has_all=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bi.eager_imports = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HubDependency
# ---------------------------------------------------------------------------


def test_hub_dependency_construction() -> None:
    hd = HubDependency(
        module="core.utils",
        fan_in=20,
        fan_out=3,
        hub_score=0.85,
        is_init=False,
        transitive_cost=make_cost_estimate(),
    )
    assert hd.fan_in == 20
    assert hd.hub_score == pytest.approx(0.85)


def test_hub_dependency_no_cost() -> None:
    hd = HubDependency(
        module="core.utils",
        fan_in=1,
        fan_out=1,
        hub_score=0.1,
        is_init=False,
        transitive_cost=None,
    )
    assert hd.transitive_cost is None


def test_hub_dependency_frozen() -> None:
    hd = HubDependency(
        module="m", fan_in=0, fan_out=0, hub_score=0.0, is_init=False, transitive_cost=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hd.module = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DSMResult
# ---------------------------------------------------------------------------


def test_dsm_result_construction() -> None:
    dsm = DSMResult(
        modules=("a", "b", "c"),
        matrix=((False, True, False), (False, False, True), (False, False, False)),
        layer_assignments={"a": 0, "b": 1, "c": 2},
        violations=(("c", "a"),),
        layering_health=0.9,
    )
    assert len(dsm.modules) == 3
    assert dsm.layering_health == pytest.approx(0.9)
    assert dsm.violations == (("c", "a"),)


def test_dsm_result_frozen() -> None:
    dsm = DSMResult(
        modules=(),
        matrix=(),
        layer_assignments={},
        violations=(),
        layering_health=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        dsm.layering_health = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hotspot
# ---------------------------------------------------------------------------


def test_hotspot_construction() -> None:
    hs = Hotspot(
        module="heavy.lib",
        fan_in=15,
        transitive_cost=make_cost_estimate(),
        score=0.95,
        evidence=("imported by 15 modules",),
    )
    assert hs.fan_in == 15
    assert hs.score == pytest.approx(0.95)


def test_hotspot_frozen() -> None:
    hs = Hotspot(
        module="m",
        fan_in=1,
        transitive_cost=make_cost_estimate(),
        score=0.5,
        evidence=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hs.score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReachabilityResult
# ---------------------------------------------------------------------------


def test_reachability_result_construction() -> None:
    rr = ReachabilityResult(
        entry_point="main",
        required_modules=frozenset({"a", "b"}),
        unreachable_modules=frozenset({"c"}),
        required_cost=make_cost_estimate(),
        unreachable_cost=None,
        root_causes=("unused feature",),
        import_paths={"a": ("main", "a"), "b": ("main", "b")},
        confidence=Confidence.HIGH,
        unknown_origins=(),
    )
    assert "a" in rr.required_modules
    assert "c" in rr.unreachable_modules


def test_reachability_result_frozen() -> None:
    rr = ReachabilityResult(
        entry_point="e",
        required_modules=frozenset(),
        unreachable_modules=frozenset(),
        required_cost=None,
        unreachable_cost=None,
        root_causes=(),
        import_paths={},
        confidence=Confidence.LOW,
        unknown_origins=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rr.entry_point = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SuggestedRefactoring
# ---------------------------------------------------------------------------


def test_suggested_refactoring_construction() -> None:
    sr = SuggestedRefactoring(
        id="REF-001",
        impact=0.7,
        kind=RefactoringKind.DEFER_IMPORT,
        file=Path("/a/b.py"),
        lineno=42,
        description="Move import inside function",
        current_code="import heavy",
        proposed_pattern_pre315="def fn():\n    import heavy",
        proposed_pattern_pep810=None,
        estimated_savings_mb=5.0,
        risk=RiskLevel.SAFE,
        purity_check="pass",
        confidence=Confidence.HIGH,
        evidence=("used only in fn",),
    )
    assert sr.id == "REF-001"
    assert sr.lineno == 42
    assert sr.proposed_pattern_pep810 is None


def test_suggested_refactoring_frozen() -> None:
    sr = SuggestedRefactoring(
        id="x",
        impact=0.0,
        kind=RefactoringKind.DEFER_IMPORT,
        file=Path("/f.py"),
        lineno=1,
        description="d",
        current_code="c",
        proposed_pattern_pre315="p",
        proposed_pattern_pep810=None,
        estimated_savings_mb=0.0,
        risk=RiskLevel.SAFE,
        purity_check="q",
        confidence=Confidence.LOW,
        evidence=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        sr.impact = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WhyResult
# ---------------------------------------------------------------------------


def test_why_result_construction() -> None:
    wr = WhyResult(
        entry_point="main",
        target="heavy.lib",
        paths=(("main", "util", "heavy.lib"),),
        critical_edges=(("util", "heavy.lib"),),
        suggested_cuts=("util",),
        total_cost=make_cost_estimate(),
    )
    assert wr.target == "heavy.lib"
    assert len(wr.paths) == 1
    assert wr.suggested_cuts == ("util",)


def test_why_result_no_cuts() -> None:
    wr = WhyResult(
        entry_point="e",
        target="t",
        paths=(),
        critical_edges=(),
        suggested_cuts=None,
        total_cost=None,
    )
    assert wr.suggested_cuts is None


def test_why_result_frozen() -> None:
    wr = WhyResult(
        entry_point="e",
        target="t",
        paths=(),
        critical_edges=(),
        suggested_cuts=None,
        total_cost=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        wr.target = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DeadExport
# ---------------------------------------------------------------------------


def test_dead_export_construction() -> None:
    de = DeadExport(
        module="pkg.utils",
        symbol_name="helper",
        export_mechanism="__all__",
        confidence=Confidence.MEDIUM,
    )
    assert de.symbol_name == "helper"
    assert de.confidence is Confidence.MEDIUM


def test_dead_export_frozen() -> None:
    de = DeadExport(
        module="m",
        symbol_name="s",
        export_mechanism="direct",
        confidence=Confidence.LOW,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        de.symbol_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FullReport
# ---------------------------------------------------------------------------


def test_full_report_construction() -> None:
    report = FullReport(
        schema_version="1.0",
        tool_version="0.1.0",
        timestamp="2026-01-01T00:00:00Z",
        package="mypkg",
        entry_point="mypkg.main",
        reachability=None,
        barrels=(),
        hotspots=(),
        deferral_candidates=(),
        hub_dependencies=(),
        dead_exports=(),
        dsm=None,
        chokepoints=(),
        cycles=(),
        suggested_refactorings=(),
    )
    assert report.schema_version == "1.0"
    assert report.reachability is None


def test_full_report_frozen() -> None:
    report = FullReport(
        schema_version="1.0",
        tool_version="0.1.0",
        timestamp="2026-01-01T00:00:00Z",
        package="p",
        entry_point="e",
        reachability=None,
        barrels=(),
        hotspots=(),
        deferral_candidates=(),
        hub_dependencies=(),
        dead_exports=(),
        dsm=None,
        chokepoints=(),
        cycles=(),
        suggested_refactorings=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.package = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ImportGraph (mutable)
# ---------------------------------------------------------------------------


def test_import_graph_add_module() -> None:
    g = ImportGraph()
    mi = make_module_info()
    g.add_module("pkg.foo", mi)
    assert "pkg.foo" in g.modules
    assert g.modules["pkg.foo"] is mi


def test_import_graph_node_id_roundtrip() -> None:
    g = ImportGraph()
    g.add_module("pkg.foo", make_module_info())
    g.add_module("pkg.bar", make_module_info(qualified_name="pkg.bar"))
    id_foo = g.node_id("pkg.foo")
    id_bar = g.node_id("pkg.bar")
    assert id_foo != id_bar
    assert g.node_name(id_foo) == "pkg.foo"
    assert g.node_name(id_bar) == "pkg.bar"


def test_import_graph_add_edge() -> None:
    g = ImportGraph()
    ei = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    g.add_edge("pkg.foo", "pkg.bar", ei)
    assert "pkg.bar" in g.edges["pkg.foo"]
    assert g.edges["pkg.foo"]["pkg.bar"] is ei
    assert "pkg.foo" in g.reverse_edges["pkg.bar"]
    assert g.reverse_edges["pkg.bar"]["pkg.foo"] is ei


def test_import_graph_edge_assigns_ids() -> None:
    g = ImportGraph()
    ei = EdgeInfo(confidence=Confidence.MEDIUM, is_type_checking=False, is_conditional=False)
    g.add_edge("a", "b", ei)
    assert g.node_id("a") != g.node_id("b")
    assert g.node_name(g.node_id("a")) == "a"
    assert g.node_name(g.node_id("b")) == "b"


def test_import_graph_id_map_stable_across_additions() -> None:
    g = ImportGraph()
    g.add_module("x", make_module_info(qualified_name="x"))
    id_x = g.node_id("x")
    g.add_module("y", make_module_info(qualified_name="y"))
    g.add_module("z", make_module_info(qualified_name="z"))
    assert g.node_id("x") == id_x  # stable after further additions


def test_import_graph_add_edge_idempotent_id() -> None:
    g = ImportGraph()
    ei = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    g.add_module("a", make_module_info(qualified_name="a"))
    id_a_before = g.node_id("a")
    g.add_edge("a", "b", ei)
    assert g.node_id("a") == id_a_before  # no re-assignment


def test_import_graph_add_multiple_edges_same_source() -> None:
    g = ImportGraph()
    ei = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    g.add_edge("a", "b", ei)
    g.add_edge("a", "c", ei)
    assert set(g.edges["a"].keys()) == {"b", "c"}


def test_import_graph_add_multiple_edges_same_target() -> None:
    g = ImportGraph()
    ei = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    g.add_edge("a", "b", ei)
    g.add_edge("c", "b", ei)
    assert set(g.reverse_edges["b"].keys()) == {"a", "c"}


def test_import_graph_is_mutable() -> None:
    g = ImportGraph()
    g.external_deps.add("numpy")
    assert "numpy" in g.external_deps
    g.external_deps.discard("numpy")
    assert "numpy" not in g.external_deps


def test_import_graph_merge_edge_keeps_best_confidence_high_over_low() -> None:
    g = ImportGraph()
    high_edge = EdgeInfo(confidence=Confidence.HIGH, is_type_checking=False, is_conditional=False)
    low_edge = EdgeInfo(confidence=Confidence.LOW, is_type_checking=False, is_conditional=False)
    g.add_edge("a", "b", high_edge)
    g.add_edge("a", "b", low_edge)
    assert g.edges["a"]["b"].confidence is Confidence.HIGH


def test_import_graph_merge_edge_keeps_best_confidence_medium_over_low() -> None:
    g = ImportGraph()
    medium_edge = EdgeInfo(
        confidence=Confidence.MEDIUM, is_type_checking=False, is_conditional=False
    )
    low_edge = EdgeInfo(confidence=Confidence.LOW, is_type_checking=False, is_conditional=False)
    g.add_edge("a", "b", medium_edge)
    g.add_edge("a", "b", low_edge)
    assert g.edges["a"]["b"].confidence is Confidence.MEDIUM


# ---------------------------------------------------------------------------
# to_dict / from_dict roundtrips (manual)
# ---------------------------------------------------------------------------


def test_cost_estimate_roundtrip() -> None:
    ce = make_cost_estimate(confidence=Confidence.MEDIUM)
    assert CostEstimate.from_dict(ce.to_dict()) == ce


def test_cost_estimate_roundtrip_no_dist() -> None:
    ce = make_cost_estimate(dist_name=None)
    assert CostEstimate.from_dict(ce.to_dict()) == ce


def test_timing_result_roundtrip() -> None:
    tr = TimingResult(module="m", self_time_us=10, cumulative_time_us=100, dependencies=("a", "b"))
    assert TimingResult.from_dict(tr.to_dict()) == tr


def test_memory_result_roundtrip() -> None:
    mr = MemoryResult(module="m", python_heap_bytes=512, rss_bytes=1024)
    assert MemoryResult.from_dict(mr.to_dict()) == mr


def test_module_info_roundtrip_empty() -> None:
    mi = make_module_info()
    assert ModuleInfo.from_dict(mi.to_dict()) == mi


def test_module_info_roundtrip_with_imports_and_scopes() -> None:
    ref = ImportRef(
        module="os",
        names=("path", "getcwd"),
        alias=None,
        lineno=5,
        scope="module",
        level=0,
        is_type_checking=False,
        is_conditional=True,
        is_star=False,
    )
    mi = ModuleInfo(
        path=Path("/src/pkg/mod.py"),
        qualified_name="pkg.mod",
        imports=(ref,),
        is_init=False,
        has_getattr=True,
        has_all=True,
        all_names=frozenset({"A", "B"}),
        defined_names=frozenset({"A", "B", "_private"}),
        star_imports=("pkg.base",),
        purity=Purity.SIDE_EFFECTS,
        usage_scopes={"fn_x": frozenset({"scope1", "scope2"})},
    )
    assert ModuleInfo.from_dict(mi.to_dict()) == mi


def test_module_info_to_dict_uses_sorted_lists() -> None:
    mi = make_module_info(all_names=frozenset({"z", "a", "m"}))
    d = mi.to_dict()
    assert d["all_names"] == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Hypothesis property-based roundtrip tests
# ---------------------------------------------------------------------------

confidence_st = st.sampled_from(list(Confidence))


@given(
    module=st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="._"),
    ),
    dist_name=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    direct_size=st.integers(min_value=0, max_value=10**9),
    transitive_size=st.integers(min_value=0, max_value=10**9),
    confidence=confidence_st,
    provenance=st.text(min_size=0, max_size=50),
)
@settings(max_examples=100)
def test_cost_estimate_hypothesis_roundtrip(
    module: str,
    dist_name: str | None,
    direct_size: int,
    transitive_size: int,
    confidence: Confidence,
    provenance: str,
) -> None:
    ce = CostEstimate(
        module=module,
        dist_name=dist_name,
        direct_size_bytes=direct_size,
        transitive_size_bytes=transitive_size,
        confidence=confidence,
        provenance=provenance,
    )
    assert CostEstimate.from_dict(ce.to_dict()) == ce


@given(
    module=st.text(min_size=1, max_size=30),
    self_time=st.integers(min_value=0, max_value=10**9),
    cumulative_time=st.integers(min_value=0, max_value=10**9),
    deps=st.lists(st.text(min_size=1, max_size=20), max_size=10),
)
@settings(max_examples=100)
def test_timing_result_hypothesis_roundtrip(
    module: str,
    self_time: int,
    cumulative_time: int,
    deps: list[str],
) -> None:
    tr = TimingResult(
        module=module,
        self_time_us=self_time,
        cumulative_time_us=cumulative_time,
        dependencies=tuple(deps),
    )
    assert TimingResult.from_dict(tr.to_dict()) == tr


@given(
    module=st.text(min_size=1, max_size=30),
    heap=st.integers(min_value=0, max_value=10**9),
    rss=st.integers(min_value=0, max_value=10**9),
)
@settings(max_examples=100)
def test_memory_result_hypothesis_roundtrip(
    module: str,
    heap: int,
    rss: int,
) -> None:
    mr = MemoryResult(module=module, python_heap_bytes=heap, rss_bytes=rss)
    assert MemoryResult.from_dict(mr.to_dict()) == mr


_identifier_st = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
)

_import_ref_st = st.builds(
    ImportRef,
    module=_identifier_st,
    names=st.lists(_identifier_st, max_size=5).map(tuple),
    alias=st.one_of(st.none(), _identifier_st),
    lineno=st.integers(min_value=1, max_value=9999),
    scope=_identifier_st,
    level=st.integers(min_value=0, max_value=3),
    is_type_checking=st.booleans(),
    is_conditional=st.booleans(),
    is_star=st.booleans(),
)


@given(
    path=st.builds(
        lambda parts: Path("/") / Path(*parts),
        parts=st.lists(
            st.text(
                min_size=1,
                max_size=10,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll"), whitelist_characters="_"
                ),
            ),
            min_size=1,
            max_size=3,
        ).map(lambda ps: [f"{p}.py" if i == len(ps) - 1 else p for i, p in enumerate(ps)]),
    ),
    qualified_name=_identifier_st,
    imports=st.lists(_import_ref_st, max_size=5).map(tuple),
    is_init=st.booleans(),
    has_getattr=st.booleans(),
    has_all=st.booleans(),
    all_names=st.frozensets(_identifier_st, max_size=5),
    defined_names=st.frozensets(_identifier_st, max_size=5),
    star_imports=st.lists(_identifier_st, max_size=3).map(tuple),
    purity=st.sampled_from(list(Purity)),
    usage_scopes=st.dictionaries(
        _identifier_st,
        st.frozensets(_identifier_st, max_size=3),
        max_size=3,
    ),
)
@settings(max_examples=80)
def test_module_info_hypothesis_roundtrip(
    path: Path,
    qualified_name: str,
    imports: tuple[ImportRef, ...],
    is_init: bool,
    has_getattr: bool,
    has_all: bool,
    all_names: frozenset[str],
    defined_names: frozenset[str],
    star_imports: tuple[str, ...],
    purity: Purity,
    usage_scopes: dict[str, frozenset[str]],
) -> None:
    mi = ModuleInfo(
        path=path,
        qualified_name=qualified_name,
        imports=imports,
        is_init=is_init,
        has_getattr=has_getattr,
        has_all=has_all,
        all_names=all_names,
        defined_names=defined_names,
        star_imports=star_imports,
        purity=purity,
        usage_scopes=usage_scopes,
    )
    assert ModuleInfo.from_dict(mi.to_dict()) == mi


# ---------------------------------------------------------------------------
# ImportGraph.module_external_imports
# ---------------------------------------------------------------------------


def test_import_graph_module_external_imports_field() -> None:
    g = ImportGraph()
    assert g.module_external_imports == {}
    g.module_external_imports["pkg.foo"] = {"numpy", "pandas"}
    assert g.module_external_imports["pkg.foo"] == {"numpy", "pandas"}
    g.module_external_imports["pkg.foo"].add("scipy")
    assert "scipy" in g.module_external_imports["pkg.foo"]


# ---------------------------------------------------------------------------
# ExternalWhyResult
# ---------------------------------------------------------------------------


def test_external_why_result_frozen() -> None:
    result = ExternalWhyResult(
        entry_point="pkg.main",
        target_package="numpy",
        target_import_names=("numpy",),
        direct_importers=("pkg.utils",),
        paths=(("pkg.main", "pkg.utils"),),
        pip_dependency_chain=("myapp", "numpy"),
        confidence=Confidence.HIGH,
    )
    assert result.entry_point == "pkg.main"
    assert result.target_package == "numpy"
    assert result.pip_dependency_chain == ("myapp", "numpy")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.target_package = "scipy"  # type: ignore[misc]
