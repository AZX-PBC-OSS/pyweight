from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.console import Console

from pyweight import __version__
from pyweight.analyzers import (
    analyze_reachability,
    explain_external_import,
    explain_import,
    find_barrels,
    find_deferral_candidates,
    find_hotspots,
    find_hub_dependencies,
)
from pyweight.cost import estimate_transitive_cost, measure_memory, time_import
from pyweight.graph import build_graph
from pyweight.report import (
    format_barrels,
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
from pyweight.venv import TargetEnv, discover_target_env, env_from_python

if TYPE_CHECKING:
    from pyweight.models import (
        CostEstimate,
        FullReport,
        ImportGraph,
        MemoryResult,
        ReachabilityResult,
        TimingResult,
    )

app = typer.Typer(
    name="pyweight",
    help="Analyze Python import graphs to detect and fix import bloat.",
    no_args_is_help=True,
)

stderr = Console(stderr=True)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

type _Config = dict[str, Any]


def _load_config(path: Path) -> _Config:
    """Read [tool.pyweight] from pyproject.toml in the target directory, if present."""
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        pyproject = path.parent / "pyproject.toml"
    if not pyproject.exists():
        return {}
    try:
        with pyproject.open("rb") as f:
            data: _Config = tomllib.load(f)
        tool_cfg = data.get("tool", {})
        if not isinstance(tool_cfg, dict):
            return {}
        pw_cfg = cast("_Config", tool_cfg).get("pyweight", {})
        return cast("_Config", pw_cfg) if isinstance(pw_cfg, dict) else {}
    except tomllib.TOMLDecodeError as exc:
        stderr.print(f"Warning: could not parse pyproject.toml: {exc}")
        return {}
    except (KeyError, TypeError):
        return {}


def _parse_size(s: str) -> int:
    """Parse human-readable size like '50MB', '500KB', '1GB' to bytes.

    Raises ValueError for negative values or unrecognised suffixes — this
    indicates a malformed budget in pyproject.toml [tool.pyweight].
    """
    original = s.strip()
    upper = original.upper()

    multiplier: int
    if upper.endswith("GB"):
        raw, multiplier = upper[:-2], 1024**3
    elif upper.endswith("MB"):
        raw, multiplier = upper[:-2], 1024**2
    elif upper.endswith("KB"):
        raw, multiplier = upper[:-2], 1024
    else:
        # Bare integer (bytes); reject anything with a non-numeric suffix
        try:
            value = float(upper)
        except ValueError:
            raise ValueError(
                f"Unrecognised size suffix in budget config: {original!r}. "
                "Expected a value like '50MB', '500KB', '1GB', or a bare integer."
            ) from None
        if value < 0:
            raise ValueError(
                f"Negative size value in budget config: {original!r}. "
                "Budget must be a positive number."
            )
        return int(value)

    value = float(raw)
    if value < 0:
        raise ValueError(
            f"Negative size value in budget config: {original!r}. "
            "Budget must be a positive number."
        )
    return int(value * multiplier)


def _resolve_entry_point(graph_modules: set[str], ep: str) -> str:
    """Resolve entry-point: try full path, then strip trailing component."""
    if ep in graph_modules:
        return ep
    parts = ep.rsplit(".", 1)
    while len(parts) > 1:
        candidate = parts[0]
        if candidate in graph_modules:
            return candidate
        parts = candidate.rsplit(".", 1)
    return ep


def _resolve_env(pkg_path: Path, cli_python: str | None) -> TargetEnv | None:
    """Resolve the target project's environment."""
    if cli_python:
        return env_from_python(Path(cli_python))
    return discover_target_env(pkg_path)


def _compute_costs(
    g: ImportGraph,
    search_paths: tuple[Path, ...] | None = None,
) -> dict[str, CostEstimate]:
    """Compute transitive costs for all modules in an ImportGraph."""
    costs: dict[str, CostEstimate] = {}
    for mod in g.modules:
        costs[mod] = estimate_transitive_cost(g, mod, costs, search_paths)
    return costs


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pyweight {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """Analyze Python import graphs to detect and fix import bloat."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def graph(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    entry_point: Annotated[
        str | None, typer.Option("--entry-point", help="Prune graph to entry point")
    ] = None,
    depth: Annotated[int | None, typer.Option("--depth", help="Maximum traversal depth")] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", help="Output format: rich, json, dot, html")
    ] = "rich",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Visualize import graph (rich/JSON/DOT/HTML formats)."""
    try:
        pkg_path = Path(path).resolve()
        g = build_graph(pkg_path, entry_point=entry_point)

        if depth is not None:
            if entry_point is None:
                stderr.print("Warning: --depth requires --entry-point; ignoring --depth")
            else:
                from pyweight.graph import reachable_from

                resolved_ep = _resolve_entry_point(set(g.modules.keys()), entry_point)
                if resolved_ep not in g.modules:
                    stderr.print(
                        f"[yellow]Warning:[/yellow] entry point {entry_point!r} not found in graph"
                    )
                reachable = reachable_from(g, resolved_ep, depth=depth)
                reachable.add(resolved_ep)
                # Filter graph modules and edges to the depth-limited reachable set
                from pyweight.models import ImportGraph as _IG

                filtered = _IG()
                for name in reachable:
                    if name in g.modules:
                        filtered.add_module(name, g.modules[name])
                for source, targets in g.edges.items():
                    if source not in reachable:
                        continue
                    for target, edge_info in targets.items():
                        if target in reachable:
                            filtered.add_edge(source, target, edge_info)
                filtered.external_deps = set(g.external_deps)
                g = filtered

        if fmt == "html":
            from pyweight.visualize import generate_graph_html

            env = _resolve_env(pkg_path, python)
            sp = env.site_packages if env else None
            costs = _compute_costs(g, sp)
            html = generate_graph_html(g, costs)
            output_path = Path("pyweight-graph.html")
            output_path.write_text(html)
            typer.echo(f"Visualization written to {output_path}")
        else:
            output = format_graph(g, fmt)
            typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def hotspots(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    min_cost_mb: Annotated[
        float | None, typer.Option("--min-cost-mb", help="Minimum cost in MB")
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Modules with high fan-in x cost."""
    try:
        pkg_path = Path(path).resolve()
        cfg = _load_config(pkg_path)
        if min_cost_mb is None:
            min_cost_mb = float(cfg.get("min-cost-mb", 0.0))
        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        costs = _compute_costs(g, sp)
        min_bytes = int(min_cost_mb * 1024 * 1024)
        spots = find_hotspots(g, costs, min_cost_bytes=min_bytes)
        output = format_hotspots(tuple(spots), fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def suggest(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    entry_point: Annotated[
        str | None, typer.Option("--entry-point", help="Entry point module")
    ] = None,
    target_python: Annotated[
        str | None, typer.Option("--target-python", help="Target Python version")
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Include side-effect modules in suggestions"),
    ] = False,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Deferral candidates with escape analysis safety proof."""
    try:
        pkg_path = Path(path).resolve()
        cfg = _load_config(pkg_path)
        if entry_point is None and "entry-point" in cfg:
            entry_point = str(cfg["entry-point"])
        if target_python is None and "target-python" in cfg:
            target_python = str(cfg["target-python"])
        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        costs = _compute_costs(g, sp)
        candidates = find_deferral_candidates(g, costs, entry_point=entry_point)
        if not force:
            candidates = [c for c in candidates if c.risk != "has_side_effects"]
        barrel_list = find_barrels(g)
        output = format_suggestions(tuple(candidates), tuple(barrel_list), fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def profile(
    module: Annotated[str, typer.Argument(help="Module name to profile")],
    runs: Annotated[int, typer.Option("--runs", help="Number of timing runs")] = 3,
    memory: Annotated[bool, typer.Option("--memory", help="Also measure memory usage")] = False,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    timeout: Annotated[int, typer.Option("--timeout", help="Subprocess timeout in seconds")] = 30,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
) -> None:
    """Import timing and memory measurement."""
    try:
        target_py = Path(python) if python else None
        timing_result: TimingResult = time_import(
            module,
            runs=runs,
            timeout_s=timeout,
            target_python=target_py,
        )
        memory_results: list[MemoryResult] = []
        if memory:
            memory_results.append(
                measure_memory(module, timeout_s=timeout, target_python=target_py),
            )
        output = format_profile((timing_result,), tuple(memory_results), fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def barrels(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
) -> None:
    """Identify re-export aggregators (__init__.py files)."""
    try:
        pkg_path = Path(path).resolve()
        g = build_graph(pkg_path)
        barrel_list = find_barrels(g)
        output = format_barrels(tuple(barrel_list), fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def reach(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    entry_point: Annotated[str, typer.Option("--entry-point", help="Entry-point module")],
    baseline: Annotated[
        str | None, typer.Option("--baseline", help="Path to baseline JSON file")
    ] = None,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression", help="Exit 1 if metrics increase beyond tolerance"),
    ] = False,
    fail_on_budget: Annotated[
        bool,
        typer.Option("--fail-on-budget", help="Exit 1 if import cost exceeds configured budget"),
    ] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable cache (reserved)")] = False,
    tolerance_pct: Annotated[
        float | None,
        typer.Option("--tolerance-pct", help="Tolerance percentage for dynamic metrics"),
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json, html")] = "rich",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Reachability analysis from entry point."""
    try:
        pkg_path = Path(path).resolve()
        cfg = _load_config(pkg_path)
        if tolerance_pct is None:
            tolerance_pct = float(cfg.get("tolerance-pct", 5.0))

        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        resolved_ep = _resolve_entry_point(set(g.modules.keys()), entry_point)
        if resolved_ep not in g.modules:
            stderr.print(
                f"[yellow]Warning:[/yellow] entry point {entry_point!r} not found in graph"
            )
        costs = _compute_costs(g, sp)
        result = analyze_reachability(g, resolved_ep, costs, search_paths=sp)

        if fmt == "html":
            from pyweight.visualize import generate_graph_html

            html = generate_graph_html(g, costs, reachability=result)
            output_path = Path("pyweight-reach.html")
            output_path.write_text(html)
            typer.echo(f"Visualization written to {output_path}")
        else:
            output = format_reachability(result, fmt)
            typer.echo(output, nl=False)

        exit_code = 0

        # Budget check
        budgets_raw = cfg.get("budgets", {})
        if isinstance(budgets_raw, dict) and (fail_on_budget or budgets_raw):
            budgets: _Config = cast("_Config", budgets_raw)
            budget_str = budgets.get(entry_point) or budgets.get(resolved_ep)
            if budget_str is None:
                if fail_on_budget:
                    stderr.print(
                        f"Warning: --fail-on-budget specified but no budget configured "
                        f"for {resolved_ep!r}"
                    )
            elif isinstance(budget_str, str):
                budget_bytes = _parse_size(budget_str)
                required_cost = result.required_cost
                actual_bytes = required_cost.transitive_size_bytes if required_cost else 0
                if actual_bytes > budget_bytes:
                    budget_mb = budget_bytes / 1024**2
                    actual_mb = actual_bytes / 1024**2
                    stderr.print(
                        f"[red]Budget exceeded:[/red] {resolved_ep} costs "
                        f"{actual_mb:.2f} MB (budget: {budget_mb:.2f} MB)"
                    )
                    exit_code = 1

        # Regression check
        if baseline is not None and fail_on_regression:
            baseline_path = Path(baseline)
            if baseline_path.exists():
                with baseline_path.open() as f:
                    baseline_data: dict[str, Any] = json.load(f)
                current_req = len(result.required_modules)
                current_unr = len(result.unreachable_modules)
                baseline_req = int(baseline_data.get("required_module_count", current_req))
                baseline_unr = int(baseline_data.get("unreachable_module_count", current_unr))
                # Static metrics: 0% tolerance (deterministic)
                if current_req > baseline_req or current_unr > baseline_unr:
                    stderr.print(
                        f"[red]Regression:[/red] required={current_req} (was {baseline_req}), "
                        f"unreachable={current_unr} (was {baseline_unr})"
                    )
                    exit_code = 1

        if exit_code != 0:
            raise typer.Exit(code=exit_code)
    except typer.Exit:
        raise
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def hubs(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    min_fan: Annotated[
        int | None, typer.Option("--min-fan", help="Minimum fan-in AND fan-out threshold")
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Hub-like dependency detection."""
    try:
        pkg_path = Path(path).resolve()
        cfg = _load_config(pkg_path)
        if min_fan is None:
            min_fan = int(cfg.get("min-fan", 3))
        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        costs = _compute_costs(g, sp)
        hub_list = find_hub_dependencies(g, costs, min_fan=min_fan)
        output = format_hubs(tuple(hub_list), fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def analyze(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    entry_point: Annotated[
        str | None, typer.Option("--entry-point", help="Entry point module")
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Advanced analysis (DSM, dominators, cycles, centrality)."""
    try:
        import importlib

        importlib.import_module("networkx")
    except ImportError as exc:
        stderr.print("Install pyweight[analysis] for advanced graph analysis", markup=False)
        raise typer.Exit(code=1) from exc

    try:
        from pyweight.analyzers import find_dead_exports
        from pyweight.graph_analysis import compute_dsm, find_cycles
        from pyweight.report import format_analysis

        pkg_path = Path(path).resolve()
        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        costs = _compute_costs(g, sp)
        dsm = compute_dsm(g)
        cycles = find_cycles(g)
        hub_list = find_hub_dependencies(g, costs)
        dead = find_dead_exports(g, is_library=library)
        output = format_analysis(dsm, tuple(hub_list), tuple(dead), tuple(cycles), fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def why(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    target: Annotated[
        str, typer.Option("--target", help="Target module or external package to explain")
    ],
    entry_point: Annotated[str, typer.Option("--entry-point", help="Entry-point module")],
    max_paths: Annotated[int, typer.Option("--max-paths", help="Maximum paths to show")] = 5,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Causal attribution — why is a module or package loaded?"""
    try:
        pkg_path = Path(path).resolve()
        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        resolved_ep = _resolve_entry_point(set(g.modules.keys()), entry_point)
        if resolved_ep not in g.modules:
            stderr.print(
                f"[yellow]Warning:[/yellow] entry point {entry_point!r} not found in graph"
            )

        if target in g.modules:
            costs = _compute_costs(g, sp)
            why_result = explain_import(g, resolved_ep, target, costs, max_paths=max_paths)
            output = format_why(why_result, fmt)
        else:
            result = explain_external_import(g, resolved_ep, target, search_paths=sp)
            output = format_why_external(result, fmt)

        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def compare(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    entry_points: Annotated[
        list[str], typer.Option("--entry-points", help="Entry-point modules to compare")
    ],
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: rich, json")] = "rich",
) -> None:
    """Multi-entry-point dependency breakdown."""
    try:
        from pyweight.report import format_compare

        pkg_path = Path(path).resolve()
        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        costs = _compute_costs(g, sp)

        results: dict[str, ReachabilityResult] = {}
        for ep in entry_points:
            resolved = _resolve_entry_point(set(g.modules.keys()), ep)
            results[ep] = analyze_reachability(g, resolved, costs, search_paths=sp)

        all_modules: set[str] = set()
        for r in results.values():
            all_modules |= r.required_modules

        output = format_compare(
            entry_points=tuple(entry_points),
            results=results,
            all_modules=all_modules,
            graph_modules=set(g.modules.keys()),
            costs=costs,
            fmt=fmt,
        )
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def report(
    path: Annotated[str, typer.Argument(help="Path to package to analyze")],
    entry_point: Annotated[str, typer.Option("--entry-point", help="Entry-point module")],
    target_python: Annotated[
        str | None, typer.Option("--target-python", help="Target Python version")
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="Path to target project's Python interpreter"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: json, markdown")] = "json",
    library: Annotated[bool, typer.Option("--library", help="Mark package as a library")] = False,
) -> None:
    """Comprehensive report with refactoring plan."""
    try:
        import datetime

        from pyweight.analyzers import find_dead_exports
        from pyweight.models import FullReport as _FullReport

        pkg_path = Path(path).resolve()
        cfg = _load_config(pkg_path)
        if target_python is None and "target-python" in cfg:
            target_python = str(cfg["target-python"])

        env = _resolve_env(pkg_path, python)
        sp = env.site_packages if env else None
        g = build_graph(pkg_path)
        resolved_ep = _resolve_entry_point(set(g.modules.keys()), entry_point)
        if resolved_ep not in g.modules:
            stderr.print(
                f"[yellow]Warning:[/yellow] entry point {entry_point!r} not found in graph"
            )
        costs = _compute_costs(g, sp)

        reachability = analyze_reachability(g, resolved_ep, costs, search_paths=sp)
        barrel_list = find_barrels(g)
        hotspot_list = find_hotspots(g, costs)
        candidates = find_deferral_candidates(g, costs, entry_point=resolved_ep)
        hub_list = find_hub_dependencies(g, costs)
        dead = find_dead_exports(g, is_library=library)

        dsm = None
        chokepoints: tuple[str, ...] = ()
        cycles: tuple[tuple[str, ...], ...] = ()
        try:
            from pyweight.graph_analysis import compute_dsm, find_chokepoints, find_cycles

            dsm = compute_dsm(g)
            chokepoints = tuple(find_chokepoints(g))
            cycles = tuple(find_cycles(g))
        except ImportError:
            pass

        refactorings = generate_refactoring_plan(
            graph=g,
            reachability=reachability,
            barrels=tuple(barrel_list),
            hotspots=tuple(hotspot_list),
            candidates=tuple(candidates),
            chokepoints=chokepoints,
            cycles=cycles,
            hub_dependencies=tuple(hub_list),
            dead_exports=tuple(dead),
            dsm=dsm,
        )

        full: FullReport = _FullReport(
            schema_version="1",
            tool_version=__version__,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            package=pkg_path.name,
            entry_point=resolved_ep,
            reachability=reachability,
            barrels=tuple(barrel_list),
            hotspots=tuple(hotspot_list),
            deferral_candidates=tuple(candidates),
            hub_dependencies=tuple(hub_list),
            dead_exports=tuple(dead),
            dsm=dsm,
            chokepoints=chokepoints,
            cycles=cycles,
            suggested_refactorings=refactorings,
        )

        output = format_report(full, fmt)
        typer.echo(output, nl=False)
    except Exception as exc:
        stderr.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
