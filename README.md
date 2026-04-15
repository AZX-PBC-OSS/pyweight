# pyweight

Analyze Python import graphs to detect and fix import bloat.

pyweight statically walks a package's import graph to identify which modules
pull in heavy external dependencies, where barrel files cause unnecessary
eager loading, and which imports can be safely deferred. It queries the
**target project's own virtual environment** for package size metadata, so
`uvx pyweight` works correctly when analyzing a project you don't have
installed.

- GitHub: https://github.com/AZX-PBC-OSS/pyweight
- License: MIT
- Requires: Python 3.12+

---

## Quick Start

```bash
# Install
pip install pyweight

# Find the heaviest import chains in your package
pyweight hotspots ./mypackage

# See what's reachable from your app's entry point
pyweight reach ./mypackage --entry-point myapp.main

# Get suggested lazy-import refactors
pyweight suggest ./mypackage

# Produce a full machine-readable report
pyweight report ./mypackage --entry-point myapp.main
```

Run without installation using uvx:

```bash
uvx pyweight hotspots ./mypackage
```

---

## Installation

```bash
# Core (graph traversal, hotspots, suggestions, barrels, reach, hubs, why, compare, report)
pip install pyweight
# or
uv add pyweight

# With advanced analysis (DSM, cycle detection, centrality via networkx)
pip install "pyweight[analysis]"
```

---

## What pyweight Finds

| Problem | Command |
|---|---|
| Modules with high fan-in and high transitive cost | `hotspots` |
| Barrel `__init__.py` files that re-export everything | `barrels` |
| Imports safe to defer (with escape analysis) | `suggest` |
| Hub dependencies (high fan-in AND fan-out) | `hubs` |
| Modules unreachable from an entry point | `reach` |
| Why a specific module or external package is loaded | `why` |
| Which entry points pull in which modules | `compare` |
| DSM, import cycles, chokepoints, dead exports | `analyze` |

---

## Commands

### `graph` — Visualize the import graph

```
pyweight graph <path> [--entry-point <mod>] [--depth <n>] [--format rich|json|dot|html]
```

Renders the full import graph. `--entry-point` prunes to modules reachable from
that module. `--depth` limits traversal depth (requires `--entry-point`).
`--format html` writes `pyweight-graph.html` in the current directory.

### `hotspots` — Modules with high fan-in × cost

```
pyweight hotspots <path> [--min-cost-mb <n>] [--format rich|json]
```

Lists modules ranked by (number of importers) × (transitive dependency size).
`--min-cost-mb` filters to modules above a cost threshold.

### `suggest` — Deferral candidates

```
pyweight suggest <path> [--entry-point <mod>] [--force] [--format rich|json]
```

Identifies imports that could be converted to lazy imports, with an escape
analysis proof of safety. By default, modules flagged as having side effects are
excluded. Pass `--force` to include them anyway.

### `profile` — Import timing and memory

```
pyweight profile <module> [--runs <n>] [--memory] [--timeout <s>] [--format rich|json]
```

Measures wall-clock import time across `--runs` subprocess invocations (default 3)
and optionally peak memory usage with `--memory`.

### `barrels` — Barrel file detection

```
pyweight barrels <path> [--format rich|json]
```

Identifies `__init__.py` files that aggregate and re-export symbols from
submodules, causing those submodules to be loaded even when only a subset is
needed.

### `reach` — Reachability from an entry point

```
pyweight reach <path> --entry-point <mod> [--baseline <file>] \
    [--fail-on-regression] [--fail-on-budget] [--tolerance-pct <n>] \
    [--format rich|json|html]
```

Reports required modules (reachable from the entry point) and unreachable
modules. Supports CI gating via exit codes — see [CI Integration](#ci-integration).
`--format html` writes `pyweight-reach.html`.

### `hubs` — Hub dependency detection

```
pyweight hubs <path> [--min-fan <n>] [--format rich|json]
```

Finds modules that are both heavily imported and import many others (high fan-in
AND fan-out). `--min-fan` sets the threshold for both directions (default: 3).

### `analyze` — Advanced analysis

```
pyweight analyze <path> [--entry-point <mod>] [--format rich|json]
```

Requires `pyweight[analysis]`. Produces a dependency structure matrix (DSM),
cycle report, hub list, centrality scores, and dead exports.

### `why` — Causal attribution

```
pyweight why <path> --target <mod_or_pkg> --entry-point <mod> \
    [--max-paths <n>] [--format rich|json]
```

Explains why a module or external package is loaded from a given entry point,
showing the import paths responsible. Works for both internal modules and
external package names.

### `compare` — Multi-entry-point comparison

```
pyweight compare <path> --entry-points <mod1> --entry-points <mod2> \
    [--format rich|json]
```

Compares which modules are required by each entry point, and identifies shared,
unique, and dead modules across all of them.

### `report` — Full report with refactoring plan

```
pyweight report <path> --entry-point <mod> [--format json|markdown]
```

Combines reachability, barrels, hotspots, deferral candidates, hub dependencies,
dead exports, and (if `pyweight[analysis]` is installed) DSM, cycles, and
chokepoints into a single report with a prioritized refactoring plan.
Default output format is `json`.

---

## Global Options

All commands accept these options:

| Option | Description |
|---|---|
| `--python <path>` | Explicit path to the target project's Python interpreter |
| `--format <fmt>` | Output format (varies by command; see above) |
| `--library` | Mark the package as a library (affects dead export confidence) |
| `--no-cache` | Disable caching (reserved) |
| `--verbose` | Verbose output (reserved) |

---

## Configuration via pyproject.toml

Place a `[tool.pyweight]` section in your project's `pyproject.toml` to set
defaults. CLI flags override config values.

```toml
[tool.pyweight]
entry-point = "myapp.main"
target-python = "3.12"
min-cost-mb = 1.0
min-fan = 3
tolerance-pct = 5.0

[tool.pyweight.budgets]
"myapp.main" = "50MB"
"myapp.api" = "100MB"
```

Supported keys:

| Key | Type | Used by |
|---|---|---|
| `entry-point` | string | `suggest`, `reach`, `report` |
| `target-python` | string | `suggest`, `report` |
| `min-cost-mb` | float | `hotspots` |
| `min-fan` | int | `hubs` |
| `tolerance-pct` | float | `reach` |
| `budgets` | table (module → size string) | `reach --fail-on-budget` |

Budget values accept `KB`, `MB`, `GB` suffixes or a bare integer (bytes):
`"50MB"`, `"512KB"`, `"1GB"`.

---

## Venv Discovery

pyweight needs the target project's site-packages to estimate external
dependency sizes. Discovery runs in this order:

1. If `--python` is passed, use that interpreter directly.
2. Walk upward from the package root looking for `.venv/` or `venv/` with a
   `bin/python` (Unix) or `Scripts/python.exe` (Windows). Stops at the first
   directory containing `pyproject.toml` or `setup.py`.
3. Fall back to the `VIRTUAL_ENV` environment variable.

The directory walk takes priority over `VIRTUAL_ENV`. This prevents pyweight
from accidentally reading its own venv (installed via `uvx`) when you are
analyzing a different project.

If no environment is found, cost estimates fall back to the current interpreter.

---

## CI Integration

`pyweight reach` exits with code 1 when a budget or regression check fails.

**Budget gate** — fail if import cost exceeds a configured limit:

```bash
pyweight reach ./mypackage --entry-point myapp.main --fail-on-budget
```

Requires `[tool.pyweight.budgets]` in `pyproject.toml` for the entry point.
If `--fail-on-budget` is passed but no budget is configured for the entry point,
a warning is printed and the command exits 0.

**Regression gate** — fail if module counts increase beyond a baseline:

```bash
# Save a baseline (pipe JSON output to a file)
pyweight reach ./mypackage --entry-point myapp.main --format json > baseline.json

# In CI, compare against the baseline (0% tolerance on static module counts)
pyweight reach ./mypackage --entry-point myapp.main \
    --baseline baseline.json \
    --fail-on-regression
```

`--tolerance-pct` applies to dynamic (cost) metrics. Module counts are compared
exactly (0% tolerance) because they are deterministic.

---

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/AZX-PBC-OSS/pyweight
cd pyweight
uv sync

# Run tests
uv run pytest

# Lint and type-check
uv run ruff check .
uv run pyright
```

The `analysis` extra (networkx) is included in the `test` dependency group and
is available automatically after `uv sync`.
