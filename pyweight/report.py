"""Formatters and refactoring plan generation for pyweight analysis results."""

from __future__ import annotations

import io
import json
import textwrap
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from pyweight import __version__
from pyweight.models import (
    BarrelInfo,
    Confidence,
    CostEstimate,
    DeadExport,
    DeferralCandidate,
    DSMResult,
    FullReport,
    HubDependency,
    ImportGraph,
    MemoryResult,
    ReachabilityResult,
    RefactoringKind,
    RiskLevel,
    SuggestedRefactoring,
    TimingResult,
    WhyResult,
)

if TYPE_CHECKING:
    from pyweight.models import ExternalWhyResult, Hotspot

# ---------------------------------------------------------------------------
# Confidence weights and sort order for impact scoring
# ---------------------------------------------------------------------------

_CONFIDENCE_WEIGHT: dict[Confidence, float] = {
    Confidence.HIGH: 1.0,
    Confidence.MEDIUM: 0.7,
    Confidence.LOW: 0.3,
}

_CONFIDENCE_SORT_ORDER: dict[Confidence, int] = {
    Confidence.HIGH: 0,
    Confidence.MEDIUM: 1,
    Confidence.LOW: 2,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rich_str(renderable: object) -> str:
    """Render a rich renderable to a plain string via StringIO."""
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True, width=120)
    console.print(renderable)
    return buf.getvalue()


def _invalid_format(fmt: str) -> ValueError:
    return ValueError(f"Unsupported format {fmt!r}. Valid formats depend on the formatter.")


def _bytes_to_mb(n: int) -> float:
    return n / 1_048_576


def _cost_mb(cost: CostEstimate | None) -> float:
    if cost is None:
        return 0.0
    return _bytes_to_mb(cost.transitive_size_bytes)


def _version_envelope(data: dict[str, object]) -> dict[str, object]:
    return {"pyweight_version": __version__, **data}


def _dump(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# 1. format_graph
# ---------------------------------------------------------------------------


def format_graph(graph: ImportGraph, fmt: str) -> str:
    """Format an ImportGraph as rich tree, JSON, or DOT."""
    if fmt == "json":
        edges: list[dict[str, object]] = []
        for src, targets in graph.edges.items():
            for tgt, edge_info in targets.items():
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "confidence": edge_info.confidence.value,
                        "is_type_checking": edge_info.is_type_checking,
                        "is_conditional": edge_info.is_conditional,
                    }
                )
        return _dump(
            _version_envelope(
                {
                    "modules": sorted(graph.modules.keys()),
                    "edges": edges,
                    "external_deps": sorted(graph.external_deps),
                }
            )
        )

    if fmt == "dot":
        lines = ["digraph imports {", '  rankdir="LR";']
        for src, targets in graph.edges.items():
            for tgt in targets:
                lines.append(f'  "{src}" -> "{tgt}";')
        lines.append("}")
        return "\n".join(lines)

    if fmt == "rich":
        roots = set(graph.modules.keys()) - {
            tgt for targets in graph.edges.values() for tgt in targets
        }
        if not roots:
            roots = set(graph.modules.keys())
        tree = Tree("[bold]Import Graph[/bold]")
        for root in sorted(roots):
            _add_tree_node(tree, root, graph, set(), depth=0)
        return _rich_str(tree)

    raise _invalid_format(fmt)


def _add_tree_node(
    parent: Tree,
    module: str,
    graph: ImportGraph,
    visited: set[str],
    depth: int,
) -> None:
    label = f"[cyan]{module}[/cyan]" if depth == 0 else module
    node = parent.add(label)
    if module in visited or depth > 6:
        return
    visited = visited | {module}
    for child in sorted(graph.edges.get(module, {}).keys()):
        _add_tree_node(node, child, graph, visited, depth + 1)


# ---------------------------------------------------------------------------
# 2. format_hotspots
# ---------------------------------------------------------------------------


def format_hotspots(hotspots: tuple[Hotspot, ...], fmt: str) -> str:
    """Format hotspots as rich table or JSON."""
    if fmt == "json":
        items = [
            {
                "module": h.module,
                "fan_in": h.fan_in,
                "cost_mb": _cost_mb(h.transitive_cost),
                "score": h.score,
                "evidence": list(h.evidence),
            }
            for h in hotspots
        ]
        return _dump(_version_envelope({"hotspots": items}))

    if fmt == "rich":
        table = Table(title="Import Hotspots", show_lines=False)
        table.add_column("Module", style="cyan")
        table.add_column("Fan-In", justify="right")
        table.add_column("Cost MB", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Evidence")
        for h in hotspots:
            table.add_row(
                h.module,
                str(h.fan_in),
                f"{_cost_mb(h.transitive_cost):.2f}",
                f"{h.score:.0f}",
                "; ".join(h.evidence),
            )
        return _rich_str(table)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 3. format_suggestions
# ---------------------------------------------------------------------------


def format_suggestions(
    candidates: tuple[DeferralCandidate, ...],
    barrels: tuple[BarrelInfo, ...],
    fmt: str,
) -> str:
    """Format deferral candidates and barrel suggestions."""
    if fmt == "json":
        cand_items = [
            {
                "file": str(c.file),
                "module": c.import_ref.module,
                "names": list(c.import_ref.names),
                "escape_level": c.escape_level.value,
                "cost_mb": _cost_mb(c.estimated_cost),
                "confidence": c.confidence.value,
                "risk": c.risk,
                "evidence": list(c.evidence),
            }
            for c in candidates
        ]
        barrel_items = [
            {
                "path": str(b.path),
                "qualified_name": b.qualified_name,
                "star_imports": list(b.star_imports),
                "eager_imports": b.eager_imports,
                "reexported_count": b.reexported_count,
                "has_lazy_loading": b.has_lazy_loading,
                "has_all": b.has_all,
            }
            for b in barrels
        ]
        return _dump(
            _version_envelope({"deferral_candidates": cand_items, "barrels": barrel_items})
        )

    if fmt == "rich":
        parts: list[str] = []
        for c in candidates:
            cost_str = f"{_cost_mb(c.estimated_cost):.2f} MB" if c.estimated_cost else "unknown"
            if c.import_ref.names:
                current_snippet = (
                    f"from {c.import_ref.module} import {', '.join(c.import_ref.names)}"
                )
                import_label = current_snippet
            else:
                current_snippet = f"import {c.import_ref.module}"
                import_label = current_snippet
            body = (
                f"[bold]File:[/bold] {c.file}\n"
                f"[bold]Import:[/bold] {import_label}\n"
                f"[bold]Cost:[/bold] {cost_str}\n"
                f"[bold]Escape level:[/bold] {c.escape_level.value}\n"
                f"[bold]Confidence:[/bold] {c.confidence.value}  "
                f"[bold]Risk:[/bold] {c.risk}\n"
                f"[bold]Evidence:[/bold] {'; '.join(c.evidence)}\n"
                f"\n[italic]Current code:[/italic]\n"
                f"  {current_snippet}  # line {c.import_ref.lineno}\n"
            )
            panel = Panel(body, title=f"Defer: {c.import_ref.module}", border_style="yellow")
            parts.append(_rich_str(panel))
        for b in barrels:
            body = (
                f"[bold]Path:[/bold] {b.path}\n"
                f"[bold]Eager imports:[/bold] {b.eager_imports}  "
                f"[bold]Re-exports:[/bold] {b.reexported_count}\n"
                f"[bold]Lazy loading:[/bold] {b.has_lazy_loading}  "
                f"[bold]__all__:[/bold] {b.has_all}\n"
                f"[bold]Star imports:[/bold] {', '.join(b.star_imports)}\n"
            )
            panel = Panel(body, title=f"Barrel: {b.qualified_name}", border_style="blue")
            parts.append(_rich_str(panel))
        return "\n".join(parts) if parts else "(no suggestions)\n"

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 4. format_barrels
# ---------------------------------------------------------------------------


def format_barrels(barrels: tuple[BarrelInfo, ...], fmt: str) -> str:
    """Format barrel info as rich table or JSON."""
    if fmt == "json":
        items = [
            {
                "path": str(b.path),
                "qualified_name": b.qualified_name,
                "star_imports": list(b.star_imports),
                "eager_imports": b.eager_imports,
                "reexported_count": b.reexported_count,
                "has_lazy_loading": b.has_lazy_loading,
                "has_all": b.has_all,
            }
            for b in barrels
        ]
        return _dump(_version_envelope({"barrels": items}))

    if fmt == "rich":
        table = Table(title="Barrel Modules (__init__.py)", show_lines=False)
        table.add_column("Path", style="cyan")
        table.add_column("Star Imports", justify="right")
        table.add_column("Re-exports", justify="right")
        table.add_column("Lazy?", justify="center")
        table.add_column("__all__?", justify="center")
        for b in barrels:
            table.add_row(
                str(b.path),
                str(len(b.star_imports)),
                str(b.reexported_count),
                "yes" if b.has_lazy_loading else "no",
                "yes" if b.has_all else "no",
            )
        return _rich_str(table)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 5. format_profile
# ---------------------------------------------------------------------------


def format_profile(
    timing: tuple[TimingResult, ...],
    memory: tuple[MemoryResult, ...],
    fmt: str,
) -> str:
    """Format profiling results as rich table or JSON."""
    if fmt == "json":
        return _dump(
            _version_envelope(
                {
                    "timing": [t.to_dict() for t in timing],
                    "memory": [m.to_dict() for m in memory],
                }
            )
        )

    if fmt == "rich":
        timing_table = Table(title="Import Timing", show_lines=False)
        timing_table.add_column("Module", style="cyan")
        timing_table.add_column("Self (µs)", justify="right")
        timing_table.add_column("Cumulative (µs)", justify="right")
        timing_table.add_column("Dependencies")
        for t in timing:
            timing_table.add_row(
                t.module,
                str(t.self_time_us),
                str(t.cumulative_time_us),
                ", ".join(t.dependencies),
            )

        memory_table = Table(title="Import Memory", show_lines=False)
        memory_table.add_column("Module", style="cyan")
        memory_table.add_column("Heap (MB)", justify="right")
        memory_table.add_column("RSS (MB)", justify="right")
        for m in memory:
            memory_table.add_row(
                m.module,
                f"{_bytes_to_mb(m.python_heap_bytes):.2f}",
                f"{_bytes_to_mb(m.rss_bytes):.2f}",
            )

        return _rich_str(timing_table) + _rich_str(memory_table)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 6. format_reachability — "killer output"
# ---------------------------------------------------------------------------


def format_reachability(result: ReachabilityResult, fmt: str) -> str:
    """Format reachability analysis — the diagnostic centrepiece."""
    if fmt == "json":
        return _dump(
            _version_envelope(
                {
                    "entry_point": result.entry_point,
                    "required_modules": sorted(result.required_modules),
                    "unreachable_modules": sorted(result.unreachable_modules),
                    "required_cost_mb": _cost_mb(result.required_cost),
                    "unreachable_cost_mb": _cost_mb(result.unreachable_cost),
                    "root_causes": list(result.root_causes),
                    "unknown_origins": list(result.unknown_origins),
                    "import_paths": {k: list(v) for k, v in result.import_paths.items()},
                    "confidence": result.confidence.value,
                }
            )
        )

    if fmt == "rich":
        parts: list[str] = []

        # Required modules + cost
        req_table = Table(title="Required Modules", show_lines=False)
        req_table.add_column("Module", style="green")
        req_table.add_column("Import Path")
        for mod in sorted(result.required_modules):
            path = result.import_paths.get(mod, ())
            req_table.add_row(mod, " -> ".join(path))
        parts.append(_rich_str(req_table))

        # Cost summary
        req_mb = _cost_mb(result.required_cost)
        unr_mb = _cost_mb(result.unreachable_cost)
        cost_panel = Panel(
            f"Required: [green]{req_mb:.2f} MB[/green]   "
            f"Unreachable: [red]{unr_mb:.2f} MB[/red]   "
            f"Confidence: {result.confidence.value}",
            title="Cost Summary",
        )
        parts.append(_rich_str(cost_panel))

        # Unreachable modules
        if result.unreachable_modules:
            unr_table = Table(title="Unreachable Modules", show_lines=False)
            unr_table.add_column("Module", style="red")
            for mod in sorted(result.unreachable_modules):
                unr_table.add_row(mod)
            parts.append(_rich_str(unr_table))

        # Root causes
        if result.root_causes:
            rc_text = "\n".join(f"  • {rc}" for rc in result.root_causes)
            parts.append(
                _rich_str(
                    Panel(
                        rc_text,
                        title="Root Causes (barrels pulling in unreachable modules)",
                        border_style="red",
                    )
                )
            )

        # Unknown origins
        if result.unknown_origins:
            uo_text = "\n".join(f"  • {u}" for u in result.unknown_origins)
            parts.append(
                _rich_str(
                    Panel(
                        uo_text,
                        title="Unknown Origins (unresolved star imports)",
                        border_style="yellow",
                    )
                )
            )

        # Fix suggestions
        suggestions: list[str] = []
        if result.root_causes:
            suggestions.append(
                "Consider lazy imports in barrel __init__.py files listed under Root Causes."
            )
        if result.unreachable_modules:
            suggestions.append(
                "Defer or remove imports of unreachable modules to reduce startup cost."
            )
        if result.unknown_origins:
            suggestions.append(
                "Resolve unknown star-import origins to enable accurate reachability analysis."
            )
        if suggestions:
            fix_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(suggestions))
            parts.append(_rich_str(Panel(fix_text, title="Fix Suggestions", border_style="green")))

        return "\n".join(parts)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 7. format_hubs
# ---------------------------------------------------------------------------


def format_hubs(hubs: tuple[HubDependency, ...], fmt: str) -> str:
    """Format hub dependencies, distinguishing barrel hubs from code hubs."""
    if fmt == "json":
        items = [
            {
                "module": h.module,
                "fan_in": h.fan_in,
                "fan_out": h.fan_out,
                "hub_score": h.hub_score,
                "hub_type": "barrel" if h.is_init else "code",
                "cost_mb": _cost_mb(h.transitive_cost),
            }
            for h in hubs
        ]
        return _dump(_version_envelope({"hubs": items}))

    if fmt == "rich":
        table = Table(title="Hub Dependencies", show_lines=False)
        table.add_column("Module", style="cyan")
        table.add_column("Type", justify="center")
        table.add_column("Fan-In", justify="right")
        table.add_column("Fan-Out", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Cost MB", justify="right")
        for h in hubs:
            hub_type = "[yellow]barrel (init)[/yellow]" if h.is_init else "code"
            table.add_row(
                h.module,
                hub_type,
                str(h.fan_in),
                str(h.fan_out),
                f"{h.hub_score:.0f}",
                f"{_cost_mb(h.transitive_cost):.2f}",
            )
        return _rich_str(table)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 8. format_dsm
# ---------------------------------------------------------------------------


def format_dsm(dsm_result: DSMResult, fmt: str) -> str:
    """Format DSM as ASCII matrix with violation markers or JSON."""
    if fmt == "json":
        return _dump(
            _version_envelope(
                {
                    "modules": list(dsm_result.modules),
                    "layer_assignments": dsm_result.layer_assignments,
                    "violations": [list(v) for v in dsm_result.violations],
                    "layering_health": dsm_result.layering_health,
                }
            )
        )

    if fmt == "rich":
        header_row, data_rows, footer_rows = _render_dsm_cells(dsm_result)
        lines: list[str] = [header_row, *data_rows, *footer_rows]
        return "\n".join(lines) + "\n"

    raise _invalid_format(fmt)


def _render_dsm_cells(dsm: DSMResult) -> tuple[str, list[str], list[str]]:
    """Return (header_row, data_rows, footer_rows) for a DSM as plain text.

    Both the rich formatter and the markdown renderer delegate here so the
    matrix-walk and label-truncation logic live in exactly one place.
    """
    violation_set = set(dsm.violations)
    n = len(dsm.modules)
    max_label = min(max((len(m) for m in dsm.modules), default=4), 20)

    header = " " * (max_label + 4)
    for j in range(n):
        header += f"{j:>4}"

    data_rows: list[str] = []
    for i, src in enumerate(dsm.modules):
        label = src[:max_label].ljust(max_label)
        row = f"{i:>2} {label} "
        for j, tgt in enumerate(dsm.modules):
            if i == j:
                row += "   ."
            elif dsm.matrix[i][j]:
                marker = "!" if (src, tgt) in violation_set else "X"
                row += f"   {marker}"
            else:
                row += "    "
        data_rows.append(row)

    health_pct = dsm.layering_health * 100
    footer_rows: list[str] = [
        f"\nLayering health: {dsm.layering_health:.2f} ({health_pct:.1f}%)"
        f"  Violations: {len(dsm.violations)}/{n}",
        "  X = dependency  ! = layering violation (back-edge)",
    ]
    for i, m in enumerate(dsm.modules):
        layer = dsm.layer_assignments.get(m, -1)
        footer_rows.append(f"  [{i:>2}] {m}  (layer {layer})")

    return header, data_rows, footer_rows


# ---------------------------------------------------------------------------
# 9. format_why
# ---------------------------------------------------------------------------


def format_why(why_result: WhyResult, fmt: str) -> str:
    """Format WhyResult — import paths ranked by cost, critical edges, suggested cuts."""
    if fmt == "json":
        return _dump(
            _version_envelope(
                {
                    "entry_point": why_result.entry_point,
                    "target": why_result.target,
                    "paths": [list(p) for p in why_result.paths],
                    "critical_edges": [list(e) for e in why_result.critical_edges],
                    "suggested_cuts": list(why_result.suggested_cuts)
                    if why_result.suggested_cuts
                    else None,
                    "total_cost_mb": _cost_mb(why_result.total_cost),
                }
            )
        )

    if fmt == "rich":
        parts: list[str] = []

        # Summary panel
        cost_str = (
            f"{_cost_mb(why_result.total_cost):.2f} MB" if why_result.total_cost else "unknown"
        )
        summary = (
            f"[bold]Entry:[/bold] {why_result.entry_point}  "
            f"[bold]Target:[/bold] {why_result.target}  "
            f"[bold]Cost:[/bold] {cost_str}"
        )
        parts.append(_rich_str(Panel(summary, title="Why Import?")))

        # Import paths
        for i, path in enumerate(why_result.paths):
            critical = why_result.critical_edges[i] if i < len(why_result.critical_edges) else None
            path_parts: list[str] = []
            for k in range(len(path)):
                node = path[k]
                if critical and k + 1 < len(path) and (path[k], path[k + 1]) == critical:
                    path_parts.append(f"[bold red]{node}[/bold red]")
                else:
                    path_parts.append(node)
            path_str = " -> ".join(path_parts)
            critical_label = (
                f"  [dim](critical edge: {critical[0]} -> {critical[1]})[/dim]" if critical else ""
            )
            markup = f"  Path {i + 1}: {path_str}{critical_label}"
            parts.append(_rich_str(Text.from_markup(markup)))

        # Suggested cuts
        if why_result.suggested_cuts:
            cuts_text = "\n".join(f"  • {c}" for c in why_result.suggested_cuts)
            parts.append(
                _rich_str(
                    Panel(cuts_text, title="Suggested Cuts (min-cut)", border_style="yellow")
                )
            )

        return "\n".join(parts)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 9b. format_why_external
# ---------------------------------------------------------------------------


def format_why_external(result: ExternalWhyResult, fmt: str = "rich") -> str:
    """Format an ExternalWhyResult as rich or JSON output."""
    if fmt == "json":
        return _dump(
            _version_envelope(
                {
                    "entry_point": result.entry_point,
                    "target_package": result.target_package,
                    "target_import_names": list(result.target_import_names),
                    "direct_importers": list(result.direct_importers),
                    "paths": [list(p) for p in result.paths],
                    "pip_dependency_chain": list(result.pip_dependency_chain)
                    if result.pip_dependency_chain
                    else None,
                    "confidence": result.confidence.value,
                }
            )
        )

    if fmt == "rich":
        parts: list[str] = []

        header = (
            f"[bold]Entry:[/bold] {result.entry_point}  "
            f"[bold]Package:[/bold] {result.target_package}"
        )
        parts.append(_rich_str(Panel(header, title="Why External?")))

        # Import names mapping
        if result.target_import_names:
            names_str = ", ".join(f"[cyan]{n}[/cyan]" for n in result.target_import_names)
            parts.append(_rich_str(Text.from_markup(f"  Import names: {names_str}")))

        if result.direct_importers:
            table = Table(title="Direct Importers → Import Paths", show_lines=False)
            table.add_column("Importer", style="cyan")
            table.add_column("Path")
            for i, importer in enumerate(result.direct_importers):
                path_str = " -> ".join(result.paths[i]) if i < len(result.paths) else importer
                table.add_row(importer, path_str)
            parts.append(_rich_str(table))
        else:
            parts.append(_rich_str(Text.from_markup("  [dim]No direct importers found.[/dim]")))

        if result.pip_dependency_chain:
            chain_str = " -> ".join(result.pip_dependency_chain)
            parts.append(
                _rich_str(
                    Panel(
                        f"pip chain: {chain_str}",
                        title="Pip Dependency Chain",
                        border_style="yellow",
                    )
                )
            )

        conf_color = "green" if result.confidence == Confidence.HIGH else "yellow"
        parts.append(
            _rich_str(
                Text.from_markup(
                    f"  Confidence: [{conf_color}]{result.confidence.value}[/{conf_color}]"
                )
            )
        )

        return "\n".join(parts)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 10. format_compare
# ---------------------------------------------------------------------------


def format_compare(
    entry_points: tuple[str, ...],
    results: dict[str, ReachabilityResult],
    all_modules: set[str],
    graph_modules: set[str],
    costs: dict[str, CostEstimate],
    fmt: str = "rich",
) -> str:
    """Format entry-point comparison results as rich table or JSON."""
    if fmt == "json":
        ep_sets: dict[str, list[str]] = {
            ep: sorted(r.required_modules) for ep, r in results.items()
        }
        shared = sorted(
            m for m in all_modules if all(m in r.required_modules for r in results.values())
        )
        dead = sorted(graph_modules - all_modules)
        return _dump(
            _version_envelope(
                {
                    "entry_points": ep_sets,
                    "shared": shared,
                    "dead": dead,
                }
            )
        )

    if fmt == "rich":
        table = Table(title="Entry Point Comparison")
        table.add_column("Module")
        for ep in entry_points:
            table.add_column(ep, justify="center")
        table.add_column("Cost MB", justify="right")
        table.add_column("Status")

        all_ep_sets = [results[ep].required_modules for ep in entry_points]
        for mod in sorted(all_modules):
            ep_marks = ["Y" if mod in ep_set else "-" for ep_set in all_ep_sets]
            cost_mb = costs[mod].transitive_size_bytes / 1024**2 if mod in costs else 0.0
            if all(mod in ep_set for ep_set in all_ep_sets):
                status = "shared"
            else:
                present_in = [ep for i, ep in enumerate(entry_points) if mod in all_ep_sets[i]]
                status = f"unique to {present_in[0]}" if len(present_in) == 1 else "partial"
            table.add_row(mod, *ep_marks, f"{cost_mb:.2f}", status)

        for mod in sorted(graph_modules - all_modules):
            ep_marks = ["-"] * len(entry_points)
            cost_mb = costs[mod].transitive_size_bytes / 1024**2 if mod in costs else 0.0
            table.add_row(mod, *ep_marks, f"{cost_mb:.2f}", "dead")

        return _rich_str(table)

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 11. format_dead_exports
# ---------------------------------------------------------------------------


def format_dead_exports(dead_exports: tuple[DeadExport, ...], fmt: str) -> str:
    """Format dead exports grouped by module, sorted by confidence."""
    if fmt == "json":
        items = [
            {
                "module": d.module,
                "symbol_name": d.symbol_name,
                "export_mechanism": d.export_mechanism,
                "confidence": d.confidence.value,
            }
            for d in dead_exports
        ]
        return _dump(_version_envelope({"dead_exports": items}))

    if fmt == "rich":
        # Sort by module then by confidence (high first) then symbol
        sorted_exports = sorted(
            dead_exports,
            key=lambda d: (d.module, _CONFIDENCE_SORT_ORDER[d.confidence], d.symbol_name),
        )
        parts: list[str] = []
        for module, group_iter in groupby(sorted_exports, key=lambda d: d.module):
            table = Table(title=f"Dead Exports: {module}", show_lines=False)
            table.add_column("Symbol", style="cyan")
            table.add_column("Mechanism")
            table.add_column("Confidence", justify="center")
            for d in group_iter:
                conf_style = "red" if d.confidence == Confidence.HIGH else "yellow"
                table.add_row(
                    d.symbol_name,
                    d.export_mechanism,
                    Text(d.confidence.value, style=conf_style),
                )
            parts.append(_rich_str(table))
        return "\n".join(parts) if parts else "(no dead exports found)\n"

    raise _invalid_format(fmt)


# ---------------------------------------------------------------------------
# 11. format_analysis (combined advanced results)
# ---------------------------------------------------------------------------


def format_analysis(
    dsm: DSMResult | None,
    hubs: tuple[HubDependency, ...],
    dead_exports: tuple[DeadExport, ...],
    cycles: tuple[tuple[str, ...], ...],
    fmt: str,
) -> str:
    """Format combined advanced analysis output."""
    if fmt == "json":
        return _dump(
            _version_envelope(
                {
                    "dsm": _dsm_to_dict(dsm) if dsm else None,
                    "hubs": [_hub_to_dict(h) for h in hubs],
                    "dead_exports": [_dead_export_to_dict(d) for d in dead_exports],
                    "cycles": [list(c) for c in cycles],
                }
            )
        )

    if fmt == "rich":
        parts: list[str] = []
        if dsm:
            parts.append(format_dsm(dsm, "rich"))
        if hubs:
            parts.append(format_hubs(hubs, "rich"))
        if dead_exports:
            parts.append(format_dead_exports(dead_exports, "rich"))
        if cycles:
            table = Table(title="Import Cycles", show_lines=False)
            table.add_column("Cycle Members")
            for cycle in cycles:
                table.add_row(" -> ".join(cycle))
            parts.append(_rich_str(table))
        return "\n".join(parts) if parts else "(no advanced analysis results)\n"

    raise _invalid_format(fmt)


def _dsm_to_dict(dsm: DSMResult) -> dict[str, object]:
    return {
        "modules": list(dsm.modules),
        "layer_assignments": dsm.layer_assignments,
        "violations": [list(v) for v in dsm.violations],
        "layering_health": dsm.layering_health,
    }


def _dead_export_to_dict(d: DeadExport) -> dict[str, object]:
    return {
        "module": d.module,
        "symbol_name": d.symbol_name,
        "export_mechanism": d.export_mechanism,
        "confidence": d.confidence.value,
    }


# ---------------------------------------------------------------------------
# 12. generate_refactoring_plan
# ---------------------------------------------------------------------------


def generate_refactoring_plan(
    graph: ImportGraph,
    reachability: ReachabilityResult | None,
    barrels: tuple[BarrelInfo, ...],
    hotspots: tuple[Hotspot, ...],
    candidates: tuple[DeferralCandidate, ...],
    chokepoints: tuple[str, ...],
    cycles: tuple[tuple[str, ...], ...],
    hub_dependencies: tuple[HubDependency, ...],
    dead_exports: tuple[DeadExport, ...],
    dsm: DSMResult | None = None,
) -> tuple[SuggestedRefactoring, ...]:
    """Synthesise all analysis results into an ordered set of refactoring suggestions.

    Ordered by: estimated_savings_mb * confidence_weight, descending.
    """
    # Parameters reserved for future suggestion types (not yet consumed)
    _ = (graph, reachability, barrels, hotspots, dsm)
    chokepoint_set = frozenset(chokepoints)
    items: list[SuggestedRefactoring] = []
    counter = 0

    # --- defer_import from deferral candidates ---
    for c in candidates:
        counter += 1
        savings_mb = _cost_mb(c.estimated_cost)
        weight = _CONFIDENCE_WEIGHT[c.confidence]
        impact = savings_mb * weight
        names_str = ", ".join(c.import_ref.names) or c.import_ref.module
        module = c.import_ref.module

        extra_evidence: list[str] = list(c.evidence)
        if module in chokepoint_set or str(c.file) in chokepoint_set:
            extra_evidence.append(f"chokepoint: {module} is an articulation point")

        current = f"import {module}  # line {c.import_ref.lineno}"
        pre315 = textwrap.dedent(f"""\
            def your_function(...):
                from {module} import {names_str}
                ...""")
        pep810 = textwrap.dedent(f"""\
            # PEP 810 (deferred imports)
            import {module} defer
            # then inside function body: use {names_str}""")

        items.append(
            SuggestedRefactoring(
                id=f"refactor-{counter:04d}",
                impact=impact,
                kind=RefactoringKind.DEFER_IMPORT,
                file=c.file,
                lineno=c.import_ref.lineno,
                description=(
                    f"Defer import of '{module}' — used only in "
                    f"{c.escape_level.value} scope. "
                    f"Risk: {c.risk}. Confidence: {c.confidence.value}."
                ),
                current_code=current,
                proposed_pattern_pre315=pre315,
                proposed_pattern_pep810=pep810,
                estimated_savings_mb=savings_mb,
                risk=c.risk,
                purity_check="",
                confidence=c.confidence,
                evidence=tuple(extra_evidence),
            )
        )

    # --- remove_dead_export from dead exports ---
    for d in sorted(dead_exports, key=lambda x: _CONFIDENCE_SORT_ORDER[x.confidence]):
        counter += 1
        weight = _CONFIDENCE_WEIGHT[d.confidence]
        # Dead exports don't have a direct cost; use a nominal 0.1 MB scaled by confidence weight
        nominal_mb = 0.1 * weight
        impact = nominal_mb
        items.append(
            SuggestedRefactoring(
                id=f"refactor-{counter:04d}",
                impact=impact,
                kind=RefactoringKind.REMOVE_DEAD_EXPORT,
                file=Path(d.module.replace(".", "/") + ".py"),
                lineno=0,
                description=(
                    f"Remove dead export '{d.symbol_name}' from '{d.module}' "
                    f"({d.export_mechanism}). Confidence: {d.confidence.value}."
                ),
                current_code=f"# {d.export_mechanism}: {d.symbol_name}",
                proposed_pattern_pre315=f"# Delete or un-export '{d.symbol_name}' from {d.module}",
                proposed_pattern_pep810=None,
                estimated_savings_mb=nominal_mb,
                risk=RiskLevel.SAFE,
                purity_check="",
                confidence=d.confidence,
                evidence=(f"Symbol '{d.symbol_name}' never imported by any module",),
            )
        )

    # --- break_cycle from cycles ---
    for cycle in cycles:
        counter += 1
        impact = 0.05  # nominal
        items.append(
            SuggestedRefactoring(
                id=f"refactor-{counter:04d}",
                impact=impact,
                kind=RefactoringKind.BREAK_CYCLE,
                file=Path(cycle[0].replace(".", "/") + ".py"),
                lineno=0,
                description=(
                    f"Break import cycle: {' -> '.join(cycle)}. "
                    "Extract shared code to a new module or use TYPE_CHECKING guards."
                ),
                current_code=f"# Cycle: {' -> '.join(cycle)}",
                proposed_pattern_pre315=(
                    "# Extract shared types to a separate module;\n"
                    "# use 'from __future__ import annotations' + TYPE_CHECKING guards."
                ),
                proposed_pattern_pep810=None,
                estimated_savings_mb=0.0,
                risk=RiskLevel.HAS_SIDE_EFFECTS,
                purity_check="",
                confidence=Confidence.MEDIUM,
                evidence=(f"Cycle detected: {' -> '.join(cycle)}",),
            )
        )

    # --- decouple_hub from hub_dependencies ---
    hub_modules = {h.module for h in hub_dependencies}
    for h in hub_dependencies:
        counter += 1
        savings_mb = _cost_mb(h.transitive_cost)
        weight = _CONFIDENCE_WEIGHT[Confidence.MEDIUM]
        impact = savings_mb * weight * 0.5  # partial savings — not all callers will be decoupled
        extra: list[str] = [f"hub_score: {h.hub_score:.0f}"]
        if h.module in chokepoint_set:
            extra.append(f"chokepoint: {h.module} is an articulation point")
        items.append(
            SuggestedRefactoring(
                id=f"refactor-{counter:04d}",
                impact=impact,
                kind=RefactoringKind.DECOUPLE_HUB,
                file=Path(h.module.replace(".", "/") + ("/__init__.py" if h.is_init else ".py")),
                lineno=0,
                description=(
                    f"Decouple hub '{h.module}' (fan-in={h.fan_in}, fan-out={h.fan_out}, "
                    f"type={'barrel' if h.is_init else 'code'}). "
                    "Split responsibilities or introduce an abstraction layer."
                ),
                current_code=f"# Hub: {h.module}  fan-in={h.fan_in}  fan-out={h.fan_out}",
                proposed_pattern_pre315=(
                    "# Split into focused sub-modules; depend on interfaces not concrete hubs."
                ),
                proposed_pattern_pep810=None,
                estimated_savings_mb=savings_mb,
                risk=RiskLevel.HAS_SIDE_EFFECTS,
                purity_check="",
                confidence=Confidence.MEDIUM,
                evidence=tuple(extra),
            )
        )

    # --- decouple_hub for chokepoints not already flagged as hubs ---
    for cp in chokepoints:
        if cp in hub_modules:
            continue
        counter += 1
        items.append(
            SuggestedRefactoring(
                id=f"refactor-{counter:04d}",
                impact=0.01,
                kind=RefactoringKind.DECOUPLE_HUB,
                file=Path(cp.replace(".", "/") + ".py"),
                lineno=0,
                description=(
                    f"Module '{cp}' is a chokepoint (articulation point) — "
                    "removing it would disconnect the import graph. Consider decoupling."
                ),
                current_code=f"# Chokepoint: {cp}",
                proposed_pattern_pre315=(
                    "# Introduce an abstraction or split into smaller modules."
                ),
                proposed_pattern_pep810=None,
                estimated_savings_mb=0.0,
                risk=RiskLevel.HAS_SIDE_EFFECTS,
                purity_check="",
                confidence=Confidence.LOW,
                evidence=(f"chokepoint: {cp} is an articulation point",),
            )
        )

    # Sort by impact descending
    items.sort(key=lambda r: r.impact, reverse=True)
    return tuple(items)


# ---------------------------------------------------------------------------
# 13. format_report
# ---------------------------------------------------------------------------


def format_report(report: FullReport, fmt: str) -> str:
    """Render a FullReport as JSON (versioned schema) or self-contained Markdown."""
    if fmt == "json":
        data: dict[str, object] = {
            "pyweight_version": __version__,
            "schema_version": "1",
            "timestamp": report.timestamp,
            "package": report.package,
            "entry_point": report.entry_point,
            "reachability": _reachability_to_dict(report.reachability),
            "required_module_count": len(report.reachability.required_modules)
            if report.reachability
            else 0,
            "unreachable_module_count": len(report.reachability.unreachable_modules)
            if report.reachability
            else 0,
            "barrels": [_barrel_to_dict(b) for b in report.barrels],
            "hotspots": [_hotspot_to_dict(h) for h in report.hotspots],
            "hub_dependencies": [_hub_to_dict(h) for h in report.hub_dependencies],
            "dead_exports": [_dead_export_to_dict(d) for d in report.dead_exports],
            "deferral_candidates": [_candidate_to_dict(c) for c in report.deferral_candidates],
            "chokepoints": list(report.chokepoints),
            "cycles": [list(c) for c in report.cycles],
            "dsm": _dsm_to_dict(report.dsm) if report.dsm else None,
            "suggested_refactorings": [
                _refactoring_to_dict(r) for r in report.suggested_refactorings
            ],
        }
        return _dump(data)

    if fmt == "markdown":
        return _render_markdown(report)

    raise _invalid_format(fmt)


def _render_dsm_plain(dsm: DSMResult) -> str:
    """Render a DSM as plain ASCII text (no Rich markup) for embedding in Markdown."""
    header, data_rows, footer_rows = _render_dsm_cells(dsm)
    return "\n".join([header, *data_rows, *footer_rows])


def _render_markdown(report: FullReport) -> str:
    lines: list[str] = [
        f"# pyweight Report: {report.package}",
        "",
        f"**Version:** {report.tool_version}  **Generated:** {report.timestamp}",
        f"**Entry point:** `{report.entry_point}`",
        "",
    ]

    # Reachability
    lines += ["## Reachability Analysis", ""]
    if report.reachability:
        r = report.reachability
        req_mb = _cost_mb(r.required_cost)
        unr_mb = _cost_mb(r.unreachable_cost)
        lines += [
            f"- Required modules: {len(r.required_modules)} ({req_mb:.2f} MB)",
            f"- Unreachable modules: {len(r.unreachable_modules)} ({unr_mb:.2f} MB)",
            f"- Confidence: {r.confidence.value}",
            "",
        ]
        if r.root_causes:
            lines += ["**Root Causes:**", ""]
            for rc in r.root_causes:
                lines.append(f"- `{rc}`")
            lines.append("")
        if r.unknown_origins:
            lines += ["**Unknown Origins:**", ""]
            for uo in r.unknown_origins:
                lines.append(f"- `{uo}`")
            lines.append("")
    else:
        lines += ["_No reachability data._", ""]

    # Barrels
    lines += ["## Barrels", ""]
    if report.barrels:
        header_row = "| Path | Star Imports | Re-exports | Lazy? | __all__? |"
        lines += [header_row, "|---|---|---|---|---|"]
        for b in report.barrels:
            lines.append(
                f"| `{b.path}` | {len(b.star_imports)} | {b.reexported_count} "
                f"| {'yes' if b.has_lazy_loading else 'no'} | {'yes' if b.has_all else 'no'} |"
            )
        lines.append("")
    else:
        lines += ["_No barrels found._", ""]

    # Hotspots
    lines += ["## Hotspots", ""]
    if report.hotspots:
        lines += ["| Module | Fan-In | Cost MB | Score |", "|---|---|---|---|"]
        for h in report.hotspots:
            lines.append(
                f"| `{h.module}` | {h.fan_in} "
                f"| {_cost_mb(h.transitive_cost):.2f} | {h.score:.0f} |"
            )
        lines.append("")
    else:
        lines += ["_No hotspots found._", ""]

    # Hub Dependencies
    lines += ["## Hub Dependencies", ""]
    if report.hub_dependencies:
        lines += ["| Module | Type | Fan-In | Fan-Out | Score |", "|---|---|---|---|---|"]
        for h in report.hub_dependencies:
            hub_type = "barrel" if h.is_init else "code"
            lines.append(
                f"| `{h.module}` | {hub_type} | {h.fan_in} | {h.fan_out} | {h.hub_score:.0f} |"
            )
        lines.append("")
    else:
        lines += ["_No hub dependencies found._", ""]

    # Dead Exports
    lines += ["## Dead Exports", ""]
    if report.dead_exports:
        sorted_dead = sorted(
            report.dead_exports,
            key=lambda d: (d.module, _CONFIDENCE_SORT_ORDER[d.confidence], d.symbol_name),
        )
        lines += ["| Module | Symbol | Mechanism | Confidence |", "|---|---|---|---|"]
        for d in sorted_dead:
            lines.append(
                f"| `{d.module}` | `{d.symbol_name}` "
                f"| {d.export_mechanism} | {d.confidence.value} |"
            )
        lines.append("")
    else:
        lines += ["_No dead exports found._", ""]

    # Deferral Candidates (Suggestions)
    lines += ["## Import Deferral Suggestions", ""]
    if report.deferral_candidates:
        for c in report.deferral_candidates:
            cost_str = f"{_cost_mb(c.estimated_cost):.2f} MB" if c.estimated_cost else "unknown"
            names_str = ", ".join(c.import_ref.names) or c.import_ref.module
            evidence_str = "; ".join(c.evidence) if c.evidence else "n/a"
            lines += [
                f"### Defer `{c.import_ref.module}` in `{c.file}`",
                "",
                (
                    f"- **Cost:** {cost_str}  "
                    f"**Confidence:** {c.confidence.value}  **Risk:** {c.risk}"
                ),
                f"- **Escape level:** {c.escape_level.value}",
                f"- **Evidence:** {evidence_str}",
                "",
                "**Current:**",
                "```python",
                f"from {c.import_ref.module} import {names_str}  # line {c.import_ref.lineno}",
                "```",
                "",
                "**Proposed (pre-3.15):**",
                "```python",
                "def your_function(...):",
                f"    from {c.import_ref.module} import {names_str}",
                "```",
                "",
            ]
    else:
        lines += ["_No deferral candidates found._", ""]

    # DSM
    if report.dsm:
        lines += [
            "## Design Structure Matrix (DSM)",
            "",
            f"Layering health: {report.dsm.layering_health:.2%}  "
            f"Violations: {len(report.dsm.violations)}",
            "",
            "```",
            _render_dsm_plain(report.dsm),
            "```",
            "",
        ]

    # Refactoring Plan
    lines += ["## Refactoring Plan", ""]
    # Generate plan if not already present
    refactorings = list(report.suggested_refactorings)
    if not refactorings:
        refactorings = list(
            generate_refactoring_plan(
                graph=ImportGraph(),
                reachability=report.reachability,
                barrels=report.barrels,
                hotspots=report.hotspots,
                candidates=report.deferral_candidates,
                chokepoints=report.chokepoints,
                cycles=report.cycles,
                hub_dependencies=report.hub_dependencies,
                dead_exports=report.dead_exports,
                dsm=report.dsm,
            )
        )

    if refactorings:
        for i, r in enumerate(refactorings, 1):
            lines += [
                f"### {i}. {r.kind.value.replace('_', ' ').title()}: `{r.file}`",
                "",
                f"**Impact:** {r.impact:.3f} MB  **Risk:** {r.risk}  "
                f"**Confidence:** {r.confidence.value}",
                "",
                r.description,
                "",
            ]
            if r.evidence:
                lines.append("**Evidence:** " + "; ".join(r.evidence))
                lines.append("")
            lines += [
                "**Current:**",
                "```python",
                r.current_code,
                "```",
                "",
                "**Proposed (pre-3.15):**",
                "```python",
                r.proposed_pattern_pre315,
                "```",
                "",
            ]
            if r.proposed_pattern_pep810:
                lines += [
                    "**Proposed (PEP 810):**",
                    "```python",
                    r.proposed_pattern_pep810,
                    "```",
                    "",
                ]
    else:
        lines += ["_No refactoring suggestions generated._", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _reachability_to_dict(r: ReachabilityResult | None) -> dict[str, object] | None:
    if r is None:
        return None
    return {
        "entry_point": r.entry_point,
        "required_modules": sorted(r.required_modules),
        "unreachable_modules": sorted(r.unreachable_modules),
        "required_cost_mb": _cost_mb(r.required_cost),
        "unreachable_cost_mb": _cost_mb(r.unreachable_cost),
        "root_causes": list(r.root_causes),
        "unknown_origins": list(r.unknown_origins),
        "import_paths": {k: list(v) for k, v in r.import_paths.items()},
        "confidence": r.confidence.value,
    }


def _barrel_to_dict(b: BarrelInfo) -> dict[str, object]:
    return {
        "path": str(b.path),
        "qualified_name": b.qualified_name,
        "star_imports": list(b.star_imports),
        "eager_imports": b.eager_imports,
        "reexported_count": b.reexported_count,
        "has_lazy_loading": b.has_lazy_loading,
        "has_all": b.has_all,
    }


def _hotspot_to_dict(h: Hotspot) -> dict[str, object]:
    return {
        "module": h.module,
        "fan_in": h.fan_in,
        "cost_mb": _cost_mb(h.transitive_cost),
        "score": h.score,
        "evidence": list(h.evidence),
    }


def _hub_to_dict(h: HubDependency) -> dict[str, object]:
    return {
        "module": h.module,
        "fan_in": h.fan_in,
        "fan_out": h.fan_out,
        "hub_score": h.hub_score,
        "hub_type": "barrel" if h.is_init else "code",
        "cost_mb": _cost_mb(h.transitive_cost),
    }


def _candidate_to_dict(c: DeferralCandidate) -> dict[str, object]:
    return {
        "file": str(c.file),
        "module": c.import_ref.module,
        "names": list(c.import_ref.names),
        "escape_level": c.escape_level.value,
        "cost_mb": _cost_mb(c.estimated_cost),
        "confidence": c.confidence.value,
        "risk": c.risk,
        "evidence": list(c.evidence),
    }


def _refactoring_to_dict(r: SuggestedRefactoring) -> dict[str, object]:
    return {
        "id": r.id,
        "impact": r.impact,
        "type": r.kind.value,
        "file": str(r.file),
        "lineno": r.lineno,
        "description": r.description,
        "estimated_savings_mb": r.estimated_savings_mb,
        "risk": str(r.risk),
        "confidence": r.confidence.value,
        "evidence": list(r.evidence),
    }
