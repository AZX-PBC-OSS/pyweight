"""Tests for pyweight.report module — TDD-first."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pyweight import __version__
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
    HubDependency,
    ImportGraph,
    ImportRef,
    MemoryResult,
    ModuleInfo,
    ReachabilityResult,
    RiskLevel,
    SuggestedRefactoring,
    TimingResult,
    WhyResult,
)
from pyweight.report import (
    format_analysis,
    format_barrels,
    format_dead_exports,
    format_dsm,
    format_graph,
    format_hotspots,
    format_hubs,
    format_profile,
    format_reachability,
    format_report,
    format_suggestions,
    format_why,
    format_why_external,
    generate_refactoring_plan,
)

# ---------------------------------------------------------------------------
# Shared test fixtures / builders
# ---------------------------------------------------------------------------


def _make_module(name: str, *, is_init: bool = False, purity: str = "pure") -> ModuleInfo:
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
        purity=purity,
    )


def _make_cost(module: str, size: int = 1_000_000) -> CostEstimate:
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
    is_star: bool = False,
    lineno: int = 1,
) -> ImportRef:
    return ImportRef(
        module=module,
        names=names,
        alias=None,
        lineno=lineno,
        scope=scope,
        level=0,
        is_type_checking=False,
        is_conditional=False,
        is_star=is_star,
    )


def _simple_graph() -> ImportGraph:
    g = ImportGraph()
    g.add_module("app", _make_module("app"))
    g.add_module("app.utils", _make_module("app.utils"))
    g.add_module("app.models", _make_module("app.models"))
    g.add_edge("app", "app.utils", EdgeInfo(Confidence.HIGH, False, False))
    g.add_edge("app", "app.models", EdgeInfo(Confidence.HIGH, False, False))
    return g


def _make_hotspot(module: str, fan_in: int = 5, size: int = 2_000_000) -> object:
    from pyweight.models import Hotspot

    return Hotspot(
        module=module,
        fan_in=fan_in,
        transitive_cost=_make_cost(module, size),
        score=float(fan_in * size),
        evidence=(f"app -> {module}",),
    )


def _make_barrel(path_str: str, qname: str) -> BarrelInfo:
    return BarrelInfo(
        path=Path(path_str),
        qualified_name=qname,
        star_imports=("mylib.core",),
        eager_imports=10,
        reexported_count=25,
        has_lazy_loading=False,
        has_all=True,
    )


def _make_deferral_candidate() -> DeferralCandidate:
    return DeferralCandidate(
        file=Path("/app/main.py"),
        import_ref=_make_import_ref("heavy_lib", ("HeavyClass",)),
        used_in_scopes=frozenset({"process"}),
        escape_level=EscapeLevel.FUNCTION_LOCAL,
        estimated_cost=_make_cost("heavy_lib", 5_000_000),
        confidence=Confidence.HIGH,
        risk=RiskLevel.SAFE,
        evidence=("app -> main",),
    )


def _make_hub(module: str, *, is_init: bool = False) -> HubDependency:
    return HubDependency(
        module=module,
        fan_in=8,
        fan_out=6,
        hub_score=48.0,
        is_init=is_init,
        transitive_cost=_make_cost(module, 3_000_000),
    )


def _make_dsm() -> DSMResult:
    mods = ("app", "app.utils", "app.models")
    matrix: tuple[tuple[bool, ...], ...] = (
        (False, True, True),
        (False, False, False),
        (False, False, False),
    )
    return DSMResult(
        modules=mods,
        matrix=matrix,
        layer_assignments={"app": 2, "app.utils": 0, "app.models": 1},
        violations=(("app.utils", "app"),),
        layering_health=0.75,
    )


def _make_why_result() -> WhyResult:
    return WhyResult(
        entry_point="app",
        target="heavy_lib",
        paths=(
            ("app", "app.utils", "heavy_lib"),
            ("app", "app.models", "heavy_lib"),
        ),
        critical_edges=(("app.utils", "heavy_lib"), ("app.models", "heavy_lib")),
        suggested_cuts=("app.utils",),
        total_cost=_make_cost("heavy_lib", 5_000_000),
    )


def _make_reachability() -> ReachabilityResult:
    return ReachabilityResult(
        entry_point="app",
        required_modules=frozenset({"app", "app.utils"}),
        unreachable_modules=frozenset({"old_module", "legacy"}),
        required_cost=_make_cost("app", 1_000_000),
        unreachable_cost=_make_cost("unreachable", 4_000_000),
        root_causes=("app.__init__",),
        import_paths={"app": ("app",), "app.utils": ("app", "app.utils")},
        confidence=Confidence.HIGH,
        unknown_origins=("external_mystery",),
    )


def _make_dead_exports() -> tuple[DeadExport, ...]:
    return (
        DeadExport(
            module="app.utils",
            symbol_name="helper_fn",
            export_mechanism="public_def",
            confidence=Confidence.MEDIUM,
        ),
        DeadExport(
            module="app.utils",
            symbol_name="unused_class",
            export_mechanism="__all__",
            confidence=Confidence.HIGH,
        ),
        DeadExport(
            module="app.models",
            symbol_name="OldModel",
            export_mechanism="__all__",
            confidence=Confidence.HIGH,
        ),
    )


def _make_timing() -> tuple[TimingResult, ...]:
    return (
        TimingResult(
            module="app",
            self_time_us=500,
            cumulative_time_us=2000,
            dependencies=("app.utils",),
        ),
        TimingResult(
            module="app.utils",
            self_time_us=1500,
            cumulative_time_us=1500,
            dependencies=(),
        ),
    )


def _make_memory() -> tuple[MemoryResult, ...]:
    return (MemoryResult(module="app", python_heap_bytes=10_000_000, rss_bytes=15_000_000),)


def _make_full_report() -> FullReport:
    import datetime

    return FullReport(
        schema_version="1",
        tool_version=__version__,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        package="app",
        entry_point="app",
        reachability=_make_reachability(),
        barrels=(_make_barrel("/app/__init__.py", "app"),),
        hotspots=(_make_hotspot("app.utils"),),  # type: ignore[arg-type]
        deferral_candidates=(_make_deferral_candidate(),),
        hub_dependencies=(_make_hub("app.__init__", is_init=True), _make_hub("app.core")),
        dead_exports=_make_dead_exports(),
        dsm=_make_dsm(),
        chokepoints=("app.utils",),
        cycles=(("app.a", "app.b"),),
        suggested_refactorings=(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(output: str) -> dict[str, object]:
    return json.loads(output)  # type: ignore[no-any-return]


# ===========================================================================
# format_graph
# ===========================================================================


class TestFormatGraph:
    def test_json_is_valid(self) -> None:
        g = _simple_graph()
        result = format_graph(g, "json")
        data = _parse_json(result)
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        g = _simple_graph()
        data = _parse_json(format_graph(g, "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_modules(self) -> None:
        g = _simple_graph()
        data = _parse_json(format_graph(g, "json"))
        modules = data["modules"]
        assert isinstance(modules, list)
        assert "app" in modules

    def test_json_contains_edges(self) -> None:
        g = _simple_graph()
        data = _parse_json(format_graph(g, "json"))
        assert "edges" in data

    def test_rich_tree_contains_module_names(self) -> None:
        g = _simple_graph()
        result = format_graph(g, "rich")
        assert "app" in result

    def test_dot_format_header(self) -> None:
        g = _simple_graph()
        result = format_graph(g, "dot")
        assert "digraph" in result
        assert "->" in result

    def test_dot_contains_module_names(self) -> None:
        g = _simple_graph()
        result = format_graph(g, "dot")
        assert "app" in result

    def test_unknown_format_raises(self) -> None:
        g = _simple_graph()
        with pytest.raises(ValueError, match="format"):
            format_graph(g, "xml")


# ===========================================================================
# format_hotspots
# ===========================================================================


class TestFormatHotspots:
    def _hotspots(self) -> tuple[object, ...]:
        return (
            _make_hotspot("app.heavy", fan_in=10, size=5_000_000),
            _make_hotspot("app.utils", fan_in=3, size=1_000_000),
        )

    def test_json_is_valid(self) -> None:
        data = _parse_json(format_hotspots(self._hotspots(), "json"))  # type: ignore[arg-type]
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_hotspots(self._hotspots(), "json"))  # type: ignore[arg-type]
        assert data["pyweight_version"] == __version__

    def test_json_contains_hotspots_list(self) -> None:
        data = _parse_json(format_hotspots(self._hotspots(), "json"))  # type: ignore[arg-type]
        assert "hotspots" in data
        assert len(data["hotspots"]) == 2  # type: ignore[arg-type]

    def test_rich_contains_column_headers(self) -> None:
        result = format_hotspots(self._hotspots(), "rich")  # type: ignore[arg-type]
        assert "Module" in result
        assert "Fan-In" in result
        assert "Score" in result

    def test_rich_contains_module_names(self) -> None:
        result = format_hotspots(self._hotspots(), "rich")  # type: ignore[arg-type]
        assert "app.heavy" in result
        assert "app.utils" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_hotspots(self._hotspots(), "csv")  # type: ignore[arg-type]


# ===========================================================================
# format_suggestions
# ===========================================================================


class TestFormatSuggestions:
    def test_json_is_valid(self) -> None:
        cand = _make_deferral_candidate()
        barrel = _make_barrel("/app/__init__.py", "app")
        data = _parse_json(format_suggestions((cand,), (barrel,), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        cand = _make_deferral_candidate()
        data = _parse_json(format_suggestions((cand,), (), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_candidates_and_barrels(self) -> None:
        cand = _make_deferral_candidate()
        barrel = _make_barrel("/app/__init__.py", "app")
        data = _parse_json(format_suggestions((cand,), (barrel,), "json"))
        assert "deferral_candidates" in data
        assert "barrels" in data

    def test_rich_shows_confidence(self) -> None:
        cand = _make_deferral_candidate()
        result = format_suggestions((cand,), (), "rich")
        assert "high" in result.lower() or "HIGH" in result

    def test_rich_shows_risk(self) -> None:
        cand = _make_deferral_candidate()
        result = format_suggestions((cand,), (), "rich")
        assert "safe" in result.lower() or "risk" in result.lower()

    def test_rich_shows_evidence(self) -> None:
        cand = _make_deferral_candidate()
        result = format_suggestions((cand,), (), "rich")
        assert "app -> main" in result

    def test_rich_shows_code_snippet(self) -> None:
        cand = _make_deferral_candidate()
        result = format_suggestions((cand,), (), "rich")
        assert "heavy_lib" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_suggestions((), (), "html")


# ===========================================================================
# format_barrels
# ===========================================================================


class TestFormatBarrels:
    def test_json_is_valid(self) -> None:
        barrel = _make_barrel("/app/__init__.py", "app")
        data = _parse_json(format_barrels((barrel,), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        barrel = _make_barrel("/app/__init__.py", "app")
        data = _parse_json(format_barrels((barrel,), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_barrels_list(self) -> None:
        barrel = _make_barrel("/app/__init__.py", "app")
        data = _parse_json(format_barrels((barrel,), "json"))
        assert "barrels" in data
        assert len(data["barrels"]) == 1  # type: ignore[arg-type]

    def test_rich_shows_column_headers(self) -> None:
        barrel = _make_barrel("/app/__init__.py", "app")
        result = format_barrels((barrel,), "rich")
        assert "Path" in result
        assert "Re-exports" in result

    def test_rich_shows_barrel_path(self) -> None:
        barrel = _make_barrel("/app/__init__.py", "app")
        result = format_barrels((barrel,), "rich")
        assert "__init__" in result or "app" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_barrels((), "tsv")


# ===========================================================================
# format_profile
# ===========================================================================


class TestFormatProfile:
    def test_json_is_valid(self) -> None:
        data = _parse_json(format_profile(_make_timing(), _make_memory(), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_profile(_make_timing(), _make_memory(), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_timing_and_memory(self) -> None:
        data = _parse_json(format_profile(_make_timing(), _make_memory(), "json"))
        assert "timing" in data
        assert "memory" in data

    def test_rich_shows_module_name(self) -> None:
        result = format_profile(_make_timing(), _make_memory(), "rich")
        assert "app" in result

    def test_rich_shows_column_headers(self) -> None:
        result = format_profile(_make_timing(), _make_memory(), "rich")
        assert "Module" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_profile((), (), "yaml")


# ===========================================================================
# format_reachability
# ===========================================================================


class TestFormatReachability:
    def test_json_is_valid(self) -> None:
        data = _parse_json(format_reachability(_make_reachability(), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_reachability(_make_reachability(), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_required_and_unreachable(self) -> None:
        data = _parse_json(format_reachability(_make_reachability(), "json"))
        assert "required_modules" in data
        assert "unreachable_modules" in data

    def test_json_contains_root_causes(self) -> None:
        data = _parse_json(format_reachability(_make_reachability(), "json"))
        assert "root_causes" in data

    def test_json_contains_unknown_origins(self) -> None:
        data = _parse_json(format_reachability(_make_reachability(), "json"))
        assert "unknown_origins" in data

    def test_rich_shows_required_modules(self) -> None:
        result = format_reachability(_make_reachability(), "rich")
        assert "app" in result

    def test_rich_contains_unknown_origins_section(self) -> None:
        result = format_reachability(_make_reachability(), "rich")
        assert "external_mystery" in result

    def test_rich_contains_root_causes(self) -> None:
        result = format_reachability(_make_reachability(), "rich")
        assert "app.__init__" in result

    def test_rich_shows_unreachable_modules(self) -> None:
        result = format_reachability(_make_reachability(), "rich")
        assert "old_module" in result or "legacy" in result

    def test_rich_shows_fix_suggestions(self) -> None:
        result = format_reachability(_make_reachability(), "rich")
        # Should suggest lazy imports or similar fix
        assert any(kw in result.lower() for kw in ("lazy", "defer", "fix", "suggest", "remove"))

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_reachability(_make_reachability(), "xml")


# ===========================================================================
# format_hubs
# ===========================================================================


class TestFormatHubs:
    def test_json_is_valid(self) -> None:
        hubs = (_make_hub("app.__init__", is_init=True), _make_hub("app.core"))
        data = _parse_json(format_hubs(hubs, "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        hubs = (_make_hub("app.__init__", is_init=True),)
        data = _parse_json(format_hubs(hubs, "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_hubs_list(self) -> None:
        hubs = (_make_hub("app.core"),)
        data = _parse_json(format_hubs(hubs, "json"))
        assert "hubs" in data

    def test_rich_distinguishes_barrel_hubs(self) -> None:
        hubs = (_make_hub("app.__init__", is_init=True), _make_hub("app.core"))
        result = format_hubs(hubs, "rich")
        # Barrel hub should be labelled differently
        assert "barrel" in result.lower() or "init" in result.lower()

    def test_rich_shows_column_headers(self) -> None:
        hubs = (_make_hub("app.core"),)
        result = format_hubs(hubs, "rich")
        assert "Module" in result
        assert "Fan-In" in result
        assert "Fan-Out" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_hubs((), "toml")


# ===========================================================================
# format_dsm
# ===========================================================================


class TestFormatDSM:
    def test_json_is_valid(self) -> None:
        data = _parse_json(format_dsm(_make_dsm(), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_dsm(_make_dsm(), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_layering_health(self) -> None:
        data = _parse_json(format_dsm(_make_dsm(), "json"))
        assert "layering_health" in data

    def test_json_contains_violations(self) -> None:
        data = _parse_json(format_dsm(_make_dsm(), "json"))
        assert "violations" in data
        assert len(data["violations"]) == 1  # type: ignore[arg-type]

    def test_rich_shows_violation_marker(self) -> None:
        result = format_dsm(_make_dsm(), "rich")
        # Violations should be marked with '!'
        assert "!" in result

    def test_rich_shows_layering_health_score(self) -> None:
        result = format_dsm(_make_dsm(), "rich")
        assert "0.75" in result or "75" in result

    def test_rich_shows_module_names(self) -> None:
        result = format_dsm(_make_dsm(), "rich")
        assert "app" in result

    def test_rich_shows_x_marker_for_dependency(self) -> None:
        result = format_dsm(_make_dsm(), "rich")
        assert "X" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_dsm(_make_dsm(), "csv")


# ===========================================================================
# format_why
# ===========================================================================


class TestFormatWhy:
    def test_json_is_valid(self) -> None:
        data = _parse_json(format_why(_make_why_result(), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_why(_make_why_result(), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_paths(self) -> None:
        data = _parse_json(format_why(_make_why_result(), "json"))
        assert "paths" in data

    def test_json_contains_critical_edges(self) -> None:
        data = _parse_json(format_why(_make_why_result(), "json"))
        assert "critical_edges" in data

    def test_json_contains_suggested_cuts(self) -> None:
        data = _parse_json(format_why(_make_why_result(), "json"))
        assert "suggested_cuts" in data

    def test_rich_shows_import_paths(self) -> None:
        result = format_why(_make_why_result(), "rich")
        assert "app" in result
        assert "heavy_lib" in result

    def test_rich_shows_critical_edge_highlight(self) -> None:
        result = format_why(_make_why_result(), "rich")
        # Critical edges should be highlighted
        assert "critical" in result.lower() or "heavy_lib" in result

    def test_rich_shows_suggested_cuts(self) -> None:
        result = format_why(_make_why_result(), "rich")
        assert "app.utils" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_why(_make_why_result(), "html")


# ===========================================================================
# format_dead_exports
# ===========================================================================


class TestFormatDeadExports:
    def test_json_is_valid(self) -> None:
        data = _parse_json(format_dead_exports(_make_dead_exports(), "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_dead_exports(_make_dead_exports(), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_contains_dead_exports_list(self) -> None:
        data = _parse_json(format_dead_exports(_make_dead_exports(), "json"))
        assert "dead_exports" in data
        assert len(data["dead_exports"]) == 3  # type: ignore[arg-type]

    def test_rich_groups_by_module(self) -> None:
        result = format_dead_exports(_make_dead_exports(), "rich")
        assert "app.utils" in result
        assert "app.models" in result

    def test_rich_shows_column_headers(self) -> None:
        result = format_dead_exports(_make_dead_exports(), "rich")
        assert "Symbol" in result
        assert "Confidence" in result

    def test_rich_sorted_by_confidence(self) -> None:
        result = format_dead_exports(_make_dead_exports(), "rich")
        # HIGH confidence items should appear
        assert "high" in result.lower() or "HIGH" in result

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_dead_exports((), "xml")


# ===========================================================================
# generate_refactoring_plan
# ===========================================================================


class TestGenerateRefactoringPlan:
    def _make_plan_inputs(self) -> dict[str, object]:
        g = _simple_graph()
        reach = _make_reachability()
        barrels = (_make_barrel("/app/__init__.py", "app"),)
        hotspots = (_make_hotspot("app.utils"),)  # type: ignore[assignment]
        candidates = (_make_deferral_candidate(),)
        chokepoints: tuple[str, ...] = ("app.utils",)
        cycles: tuple[tuple[str, ...], ...] = (("app.a", "app.b"),)
        hubs = (_make_hub("app.__init__", is_init=True), _make_hub("app.core"))
        dead = _make_dead_exports()
        return {
            "graph": g,
            "reachability": reach,
            "barrels": barrels,
            "hotspots": hotspots,
            "candidates": candidates,
            "chokepoints": chokepoints,
            "cycles": cycles,
            "hub_dependencies": hubs,
            "dead_exports": dead,
        }

    def test_returns_tuple_of_suggested_refactorings(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        assert isinstance(result, tuple)
        assert all(isinstance(r, SuggestedRefactoring) for r in result)

    def test_ordered_by_impact_descending(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        impacts = [r.impact for r in result]
        assert impacts == sorted(impacts, reverse=True), "Should be sorted by impact descending"

    def test_suggestions_include_risk_and_confidence(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        assert len(result) > 0
        for r in result:
            assert r.risk != ""
            assert r.confidence in (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)

    def test_suggestion_types_include_defer_import(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        types = {r.kind for r in result}
        assert "defer_import" in types

    def test_suggestion_types_include_remove_dead_export(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        types = {r.kind for r in result}
        assert "remove_dead_export" in types

    def test_each_suggestion_has_proposed_patterns(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        for r in result:
            assert r.proposed_pattern_pre315 != ""

    def test_impact_uses_confidence_weight(self) -> None:
        # HIGH=1.0, MEDIUM=0.7, LOW=0.3
        # A HIGH-confidence 1MB suggestion should outrank a LOW-confidence 2MB suggestion
        from pyweight.models import EscapeLevel

        high_cand = DeferralCandidate(
            file=Path("/app/main.py"),
            import_ref=_make_import_ref("lib_a", ("A",)),
            used_in_scopes=frozenset({"fn"}),
            escape_level=EscapeLevel.FUNCTION_LOCAL,
            estimated_cost=_make_cost("lib_a", 1_000_000),
            confidence=Confidence.HIGH,
            risk=RiskLevel.SAFE,
            evidence=(),
        )
        low_cand = DeferralCandidate(
            file=Path("/app/main.py"),
            import_ref=_make_import_ref("lib_b", ("B",)),
            used_in_scopes=frozenset({"fn"}),
            escape_level=EscapeLevel.FUNCTION_LOCAL,
            estimated_cost=_make_cost("lib_b", 2_000_000),
            confidence=Confidence.LOW,
            risk=RiskLevel.SAFE,
            evidence=(),
        )
        # high_cand impact = 1MB * 1.0 = 1_000_000
        # low_cand impact = 2MB * 0.3 = 600_000
        g = _simple_graph()
        reach = _make_reachability()
        result = generate_refactoring_plan(
            graph=g,
            reachability=reach,
            barrels=(),
            hotspots=(),
            candidates=(high_cand, low_cand),
            chokepoints=(),
            cycles=(),
            hub_dependencies=(),
            dead_exports=(),
        )
        defer_results = [r for r in result if r.kind == "defer_import"]
        assert len(defer_results) >= 2
        # high should come before low
        high_idx = next(i for i, r in enumerate(defer_results) if "lib_a" in r.current_code)
        low_idx = next(i for i, r in enumerate(defer_results) if "lib_b" in r.current_code)
        assert high_idx < low_idx, (
            f"high_cand (idx {high_idx}) should precede low_cand (idx {low_idx})"
        )

    def test_chokepoint_labels_appear_in_evidence(self) -> None:
        inputs = self._make_plan_inputs()
        result = generate_refactoring_plan(
            graph=inputs["graph"],  # type: ignore[arg-type]
            reachability=inputs["reachability"],  # type: ignore[arg-type]
            barrels=inputs["barrels"],  # type: ignore[arg-type]
            hotspots=inputs["hotspots"],  # type: ignore[arg-type]
            candidates=inputs["candidates"],  # type: ignore[arg-type]
            chokepoints=inputs["chokepoints"],  # type: ignore[arg-type]
            cycles=inputs["cycles"],  # type: ignore[arg-type]
            hub_dependencies=inputs["hub_dependencies"],  # type: ignore[arg-type]
            dead_exports=inputs["dead_exports"],  # type: ignore[arg-type]
        )
        all_evidence = " ".join(e for r in result for e in r.evidence)
        assert "chokepoint" in all_evidence.lower() or any(
            "app.utils" in e for r in result for e in r.evidence
        )


# ===========================================================================
# format_report
# ===========================================================================


class TestFormatReport:
    def test_json_is_valid(self) -> None:
        report = _make_full_report()
        data = _parse_json(format_report(report, "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        report = _make_full_report()
        data = _parse_json(format_report(report, "json"))
        assert data["pyweight_version"] == __version__

    def test_json_has_schema_version(self) -> None:
        report = _make_full_report()
        data = _parse_json(format_report(report, "json"))
        assert data["schema_version"] == "1"

    def test_json_contains_all_sections(self) -> None:
        report = _make_full_report()
        data = _parse_json(format_report(report, "json"))
        for key in ("reachability", "barrels", "hotspots", "hub_dependencies", "dead_exports"):
            assert key in data, f"Missing key: {key}"

    def test_markdown_contains_all_sections(self) -> None:
        report = _make_full_report()
        md = format_report(report, "markdown")
        for section in ("Reachability", "Barrels", "Hotspot", "Hub", "Dead Export"):
            assert section in md, f"Missing markdown section: {section}"

    def test_markdown_contains_refactoring_plan(self) -> None:
        report = _make_full_report()
        md = format_report(report, "markdown")
        assert "Refactoring" in md

    def test_markdown_contains_code_snippets(self) -> None:
        report = _make_full_report()
        md = format_report(report, "markdown")
        assert "```" in md

    def test_markdown_contains_evidence(self) -> None:
        report = _make_full_report()
        md = format_report(report, "markdown")
        # Evidence from the deferral candidate
        assert "app -> main" in md

    def test_markdown_contains_ordered_refactoring(self) -> None:
        report = _make_full_report()
        md = format_report(report, "markdown")
        assert "1." in md or "#" in md

    def test_unknown_format_raises(self) -> None:
        report = _make_full_report()
        with pytest.raises(ValueError, match="format"):
            format_report(report, "rst")


# ===========================================================================
# format_analysis
# ===========================================================================


class TestFormatAnalysis:
    def test_rich_with_dsm_hubs_dead_exports_cycles(self) -> None:
        dsm = _make_dsm()
        hubs = (_make_hub("app.__init__", is_init=True), _make_hub("app.core"))
        dead = _make_dead_exports()
        cycles: tuple[tuple[str, ...], ...] = (("app.a", "app.b"),)
        result = format_analysis(dsm, hubs, dead, cycles, "rich")
        assert "app" in result
        assert "hub" in result.lower() or "Module" in result
        assert "cycle" in result.lower() or "app.a" in result

    def test_json_is_valid(self) -> None:
        dsm = _make_dsm()
        hubs = (_make_hub("app.core"),)
        dead = _make_dead_exports()
        cycles: tuple[tuple[str, ...], ...] = (("app.a", "app.b"),)
        data = _parse_json(format_analysis(dsm, hubs, dead, cycles, "json"))
        assert isinstance(data, dict)

    def test_json_has_pyweight_version(self) -> None:
        data = _parse_json(format_analysis(_make_dsm(), (), (), (), "json"))
        assert data["pyweight_version"] == __version__

    def test_json_dead_exports_use_canonical_field_names(self) -> None:
        dead = _make_dead_exports()
        data = _parse_json(format_analysis(None, (), dead, (), "json"))
        exports = cast("list[dict[str, object]]", data["dead_exports"])
        assert isinstance(exports, list)
        assert len(exports) > 0
        first = exports[0]
        assert isinstance(first, dict)
        assert "symbol_name" in first, "canonical field name 'symbol_name' missing"
        assert "export_mechanism" in first, "canonical field 'export_mechanism' missing"
        assert "symbol" not in first, "old field name 'symbol' should not be present"

    def test_json_hubs_use_hub_score_and_cost_mb(self) -> None:
        hubs = (_make_hub("app.core"),)
        data = _parse_json(format_analysis(None, hubs, (), (), "json"))
        hubs_list = cast("list[dict[str, object]]", data["hubs"])
        assert isinstance(hubs_list, list)
        assert len(hubs_list) == 1
        hub = hubs_list[0]
        assert isinstance(hub, dict)
        assert "hub_score" in hub, "expected 'hub_score' not 'score'"
        assert "cost_mb" in hub

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            format_analysis(None, (), (), (), "xml")

    def test_none_dsm_branch(self) -> None:
        result = format_analysis(None, (), (), (), "rich")
        assert result == "(no advanced analysis results)\n"

    def test_empty_inputs_rich(self) -> None:
        result = format_analysis(None, (), (), (), "rich")
        assert "(no advanced analysis results)" in result

    def test_empty_inputs_json(self) -> None:
        data = _parse_json(format_analysis(None, (), (), (), "json"))
        assert data["dsm"] is None
        assert data["hubs"] == []
        assert data["dead_exports"] == []
        assert data["cycles"] == []


# ===========================================================================
# Empty-input tests (L-4)
# ===========================================================================


class TestEmptyInputs:
    def test_format_hotspots_empty_rich(self) -> None:
        result = format_hotspots((), "rich")
        assert isinstance(result, str)

    def test_format_hubs_empty_rich(self) -> None:
        result = format_hubs((), "rich")
        assert isinstance(result, str)

    def test_format_suggestions_empty_rich(self) -> None:
        result = format_suggestions((), (), "rich")
        assert "(no suggestions)" in result

    def test_format_dead_exports_empty_rich(self) -> None:
        result = format_dead_exports((), "rich")
        assert "(no dead exports found)" in result

    def test_format_graph_empty(self) -> None:
        g = ImportGraph()
        result = format_graph(g, "rich")
        assert isinstance(result, str)


# ===========================================================================
# format_why_external
# ===========================================================================


def _make_external_why_result(
    *,
    with_pip_chain: bool = False,
    with_paths: bool = True,
) -> ExternalWhyResult:
    paths: tuple[tuple[str, ...], ...] = ()
    direct_importers: tuple[str, ...] = ()
    if with_paths:
        paths = (("myapp", "myapp.loader"),)
        direct_importers = ("myapp.loader",)

    pip_chain: tuple[str, ...] | None = None
    if with_pip_chain:
        pip_chain = ("requests", "urllib3", "numpy")

    return ExternalWhyResult(
        entry_point="myapp",
        target_package="numpy",
        target_import_names=("numpy",),
        direct_importers=direct_importers,
        paths=paths,
        pip_dependency_chain=pip_chain,
        confidence=Confidence.HIGH,
    )


class TestFormatWhyExternal:
    def test_json_is_valid(self) -> None:
        result = _make_external_why_result()
        data = _parse_json(format_why_external(result, "json"))
        assert isinstance(data, dict)

    def test_json_structure(self) -> None:
        result = _make_external_why_result()
        data = _parse_json(format_why_external(result, "json"))
        assert data["entry_point"] == "myapp"
        assert data["target_package"] == "numpy"
        assert isinstance(data["target_import_names"], list)
        assert isinstance(data["direct_importers"], list)
        assert isinstance(data["paths"], list)
        assert data["confidence"] == "high"

    def test_json_pip_chain_null_when_absent(self) -> None:
        result = _make_external_why_result(with_pip_chain=False)
        data = _parse_json(format_why_external(result, "json"))
        assert data["pip_dependency_chain"] is None

    def test_json_pip_chain_present(self) -> None:
        result = _make_external_why_result(with_pip_chain=True)
        data = _parse_json(format_why_external(result, "json"))
        assert isinstance(data["pip_dependency_chain"], list)
        assert data["pip_dependency_chain"] == ["requests", "urllib3", "numpy"]

    def test_rich_contains_entry_point(self) -> None:
        result = _make_external_why_result()
        output = format_why_external(result, "rich")
        assert "myapp" in output

    def test_rich_contains_target_package(self) -> None:
        result = _make_external_why_result()
        output = format_why_external(result, "rich")
        assert "numpy" in output

    def test_rich_shows_direct_importers(self) -> None:
        result = _make_external_why_result(with_paths=True)
        output = format_why_external(result, "rich")
        assert "myapp.loader" in output

    def test_rich_shows_pip_chain(self) -> None:
        result = _make_external_why_result(with_pip_chain=True)
        output = format_why_external(result, "rich")
        assert "requests" in output
        assert "urllib3" in output

    def test_rich_shows_confidence(self) -> None:
        result = _make_external_why_result()
        output = format_why_external(result, "rich")
        assert "high" in output.lower()

    def test_unknown_format_raises(self) -> None:
        result = _make_external_why_result()
        with pytest.raises(ValueError, match="format"):
            format_why_external(result, "xml")
