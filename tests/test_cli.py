"""Integration tests for pyweight CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pyweight.cli import _parse_size, app  # pyright: ignore[reportPrivateUsage]

RUNNER = CliRunner()
SAMPLE = str(Path(__file__).parent / "fixtures" / "sample_package")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def invoke(*args: str) -> tuple[int, str]:
    """Invoke the CLI with args and return (exit_code, output)."""
    result = RUNNER.invoke(app, list(args))
    return result.exit_code, result.output


def invoke_json(*args: str) -> tuple[int, dict[str, object]]:
    """Invoke CLI with --format json and parse the JSON output."""
    code, output = invoke(*args)
    if code != 0:
        return code, {}
    return code, cast("dict[str, object]", json.loads(output))


# ---------------------------------------------------------------------------
# --help for each subcommand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subcmd",
    [
        "graph",
        "hotspots",
        "suggest",
        "profile",
        "barrels",
        "reach",
        "hubs",
        "analyze",
        "why",
        "compare",
        "report",
    ],
)
def test_help(subcmd: str) -> None:
    code, output = invoke(subcmd, "--help")
    assert code == 0
    assert subcmd in output.lower() or "help" in output.lower()


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


def test_graph_json() -> None:
    code, data = invoke_json("graph", SAMPLE, "--format", "json")
    assert code == 0
    assert "modules" in data
    assert "edges" in data
    assert "sample_package" in cast("dict[str, object]", data["modules"])


def test_graph_dot() -> None:
    code, output = invoke("graph", SAMPLE, "--format", "dot")
    assert code == 0
    assert "digraph imports" in output


def test_graph_rich() -> None:
    code, output = invoke("graph", SAMPLE, "--format", "rich")
    assert code == 0
    assert len(output) > 0


def test_graph_with_entry_point() -> None:
    code, data = invoke_json(
        "graph", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    assert code == 0
    assert "sample_package.core" in cast("dict[str, object]", data["modules"])


def test_graph_invalid_path() -> None:
    code, _ = invoke("graph", "/nonexistent/path")
    assert code == 1


def test_graph_depth_without_entry_point_warns() -> None:
    """--depth without --entry-point should warn and still succeed."""
    result = RUNNER.invoke(app, ["graph", SAMPLE, "--depth", "2", "--format", "rich"])
    assert result.exit_code == 0
    combined = (result.output or "") + (result.stderr or "")
    assert "Warning" in combined


def test_graph_depth_with_entry_point() -> None:
    """--depth with --entry-point should filter graph to depth-limited reachable set."""
    code, data = invoke_json(
        "graph", SAMPLE, "--entry-point", "sample_package.core", "--depth", "1", "--format", "json"
    )
    assert code == 0
    assert "modules" in data


# ---------------------------------------------------------------------------
# hotspots
# ---------------------------------------------------------------------------


def test_hotspots_json() -> None:
    code, data = invoke_json("hotspots", SAMPLE, "--format", "json")
    assert code == 0
    assert "hotspots" in data
    assert isinstance(data["hotspots"], list)


def test_hotspots_rich() -> None:
    code, _output = invoke("hotspots", SAMPLE, "--format", "rich")
    assert code == 0


def test_hotspots_min_cost_filter() -> None:
    # Very high min cost — should produce empty list
    code, data = invoke_json("hotspots", SAMPLE, "--min-cost-mb", "999999", "--format", "json")
    assert code == 0
    assert data["hotspots"] == []


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------


def test_suggest_json() -> None:
    code, data = invoke_json("suggest", SAMPLE, "--format", "json")
    assert code == 0
    assert "deferral_candidates" in data
    assert "barrels" in data


def test_suggest_rich() -> None:
    code, _output = invoke("suggest", SAMPLE, "--format", "rich")
    assert code == 0


def test_suggest_force_flag() -> None:
    # With --force, side-effect candidates should be included
    code, data = invoke_json("suggest", SAMPLE, "--force", "--format", "json")
    assert code == 0
    assert isinstance(data, dict)


def test_suggest_with_entry_point() -> None:
    code, data = invoke_json(
        "suggest", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    assert code == 0
    assert isinstance(data, dict)


def test_suggest_side_effects_excluded_without_force() -> None:
    # Without --force, no candidate with risk=has_side_effects should appear
    code, data = invoke_json("suggest", SAMPLE, "--format", "json")
    assert code == 0
    candidates = cast("list[dict[str, object]]", data["deferral_candidates"])
    for c in candidates:
        assert c["risk"] != "has_side_effects"


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def test_profile_json() -> None:
    code, data = invoke_json("profile", "json", "--runs", "1", "--format", "json")
    assert code == 0
    assert "timing" in data
    timing = cast("list[dict[str, object]]", data["timing"])
    assert len(timing) == 1
    assert timing[0]["module"] == "json"


def test_profile_rich() -> None:
    code, output = invoke("profile", "json", "--runs", "1", "--format", "rich")
    assert code == 0
    assert len(output) > 0


def test_profile_invalid_module_name() -> None:
    # Name with invalid chars fails validation in cost.py
    code, _ = invoke("profile", "not/a/valid.module!")
    assert code == 1


def test_profile_memory_flag() -> None:
    """--memory flag triggers memory measurement alongside timing."""
    code, data = invoke_json("profile", "json", "--runs", "1", "--memory", "--format", "json")
    assert code == 0
    assert "timing" in data
    assert "memory" in data
    memory = cast("list[object]", data["memory"])
    assert len(memory) == 1


# ---------------------------------------------------------------------------
# barrels
# ---------------------------------------------------------------------------


def test_barrels_json() -> None:
    code, data = invoke_json("barrels", SAMPLE, "--format", "json")
    assert code == 0
    assert "barrels" in data
    # sample_package has an __init__.py barrel
    barrel_names = [
        cast("dict[str, object]", b)["qualified_name"]
        for b in cast("list[object]", data["barrels"])
    ]
    assert "sample_package" in barrel_names


def test_barrels_rich() -> None:
    code, _output = invoke("barrels", SAMPLE, "--format", "rich")
    assert code == 0


def test_barrels_invalid_path() -> None:
    code, _ = invoke("barrels", "/no/such/path")
    assert code == 1


# ---------------------------------------------------------------------------
# reach
# ---------------------------------------------------------------------------


def test_reach_json() -> None:
    code, data = invoke_json(
        "reach", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    assert code == 0
    assert "entry_point" in data
    assert data["entry_point"] == "sample_package.core"
    assert "required_modules" in data
    assert "unreachable_modules" in data


def test_reach_rich() -> None:
    code, _output = invoke(
        "reach", SAMPLE, "--entry-point", "sample_package.core", "--format", "rich"
    )
    assert code == 0


def test_reach_entry_point_resolution() -> None:
    # Passing a symbol-qualified name should resolve to the module
    code, data = invoke_json(
        "reach",
        SAMPLE,
        "--entry-point",
        "sample_package.core.CoreClass",
        "--format",
        "json",
    )
    assert code == 0
    assert data["entry_point"] == "sample_package.core"


def test_reach_fail_on_regression_triggers() -> None:
    """--fail-on-regression exits 1 when baseline has fewer required modules."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"required_module_count": 0, "unreachable_module_count": 0}, f)
        baseline_path = f.name

    try:
        code, _ = invoke(
            "reach",
            SAMPLE,
            "--entry-point",
            "sample_package",
            "--baseline",
            baseline_path,
            "--fail-on-regression",
        )
        assert code == 1
    finally:
        Path(baseline_path).unlink(missing_ok=True)


def test_reach_fail_on_regression_no_regression() -> None:
    """--fail-on-regression exits 0 when baseline matches or exceeds actual."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"required_module_count": 9999, "unreachable_module_count": 9999}, f)
        baseline_path = f.name

    try:
        code, _ = invoke(
            "reach",
            SAMPLE,
            "--entry-point",
            "sample_package.core",
            "--baseline",
            baseline_path,
            "--fail-on-regression",
        )
        assert code == 0
    finally:
        Path(baseline_path).unlink(missing_ok=True)


def test_reach_no_cache_flag() -> None:
    code, _ = invoke_json(
        "reach", SAMPLE, "--entry-point", "sample_package.core", "--no-cache", "--format", "json"
    )
    assert code == 0


def test_reach_fail_on_budget_no_budget_configured() -> None:
    """--fail-on-budget with no budget in config warns but exits 0."""
    code, _ = invoke(
        "reach",
        SAMPLE,
        "--entry-point",
        "sample_package.core",
        "--fail-on-budget",
    )
    assert code == 0


def test_reach_fail_on_budget_exceeds() -> None:
    """--fail-on-budget exits 1 when cost exceeds configured budget."""
    import shutil
    from unittest.mock import MagicMock, patch

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copytree(SAMPLE, tmp_path / "sample_package")
        pkg_path = tmp_path / "sample_package"

        pyproject = pkg_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.pyweight]\n[tool.pyweight.budgets]\n"sample_package.core" = "1MB"\n',
            encoding="utf-8",
        )

        # Mock required_cost so the actual bytes exceed the 1 MB budget
        mock_cost = MagicMock()
        mock_cost.transitive_size_bytes = 10 * 1024 * 1024  # 10 MB

        mock_result = MagicMock()
        mock_result.required_modules = frozenset({"sample_package.core"})
        mock_result.unreachable_modules = frozenset()
        mock_result.required_cost = mock_cost

        with (
            patch("pyweight.cli.analyze_reachability", return_value=mock_result),
            patch("pyweight.cli.format_reachability", return_value=""),
        ):
            code, _ = invoke(
                "reach",
                str(pkg_path),
                "--entry-point",
                "sample_package.core",
                "--fail-on-budget",
            )
        assert code == 1


# ---------------------------------------------------------------------------
# hubs
# ---------------------------------------------------------------------------


def test_hubs_json() -> None:
    code, data = invoke_json("hubs", SAMPLE, "--format", "json")
    assert code == 0
    assert "hubs" in data


def test_hubs_rich() -> None:
    code, _output = invoke("hubs", SAMPLE, "--format", "rich")
    assert code == 0


def test_hubs_min_fan() -> None:
    # High min-fan threshold — should yield no hubs
    code, data = invoke_json("hubs", SAMPLE, "--min-fan", "999", "--format", "json")
    assert code == 0
    assert data["hubs"] == []


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_json() -> None:
    code, data = invoke_json("analyze", SAMPLE, "--format", "json")
    assert code == 0
    assert "dsm" in data
    assert "hubs" in data
    assert "cycles" in data


def test_analyze_rich() -> None:
    code, _output = invoke("analyze", SAMPLE, "--format", "rich")
    assert code == 0


def test_analyze_graceful_degradation_when_networkx_missing() -> None:
    """analyze should print helpful message and exit 1 when networkx is not installed."""
    with patch("importlib.import_module", side_effect=ImportError("no module named networkx")):
        result = RUNNER.invoke(app, ["analyze", SAMPLE])
    assert result.exit_code == 1
    # The error message goes to stderr; check that it ran without traceback
    assert "pyweight[analysis]" in (result.output + (result.stderr or ""))


# ---------------------------------------------------------------------------
# why
# ---------------------------------------------------------------------------


def test_why_internal_json() -> None:
    """why --target <internal_module> uses explain_import path."""
    code, data = invoke_json(
        "why",
        SAMPLE,
        "--target",
        "sample_package.heavy",
        "--entry-point",
        "sample_package",
        "--format",
        "json",
    )
    assert code == 0
    assert "entry_point" in data
    assert "target" in data
    assert "paths" in data


def test_why_internal_rich() -> None:
    code, _output = invoke(
        "why",
        SAMPLE,
        "--target",
        "sample_package.heavy",
        "--entry-point",
        "sample_package",
        "--format",
        "rich",
    )
    assert code == 0


def test_why_external_json() -> None:
    """why --target <external_package> uses explain_external_import path."""
    code, data = invoke_json(
        "why",
        SAMPLE,
        "--target",
        "pytest",
        "--entry-point",
        "sample_package",
        "--format",
        "json",
    )
    assert code == 0
    assert "entry_point" in data
    assert "target_package" in data
    assert data["target_package"] == "pytest"
    assert "direct_importers" in data
    assert "paths" in data
    assert "confidence" in data


def test_why_external_unknown_package() -> None:
    """Querying for a nonexistent package exits 0 with graceful empty result."""
    code, data = invoke_json(
        "why",
        SAMPLE,
        "--target",
        "nonexistent-package-xyz-12345",
        "--entry-point",
        "sample_package",
        "--format",
        "json",
    )
    assert code == 0
    assert data.get("direct_importers") == []
    assert data.get("paths") == []


def test_why_external_rich_output() -> None:
    """why --target <external_package> with rich format exits 0 and produces output."""
    code, output = invoke(
        "why",
        SAMPLE,
        "--target",
        "pytest",
        "--entry-point",
        "sample_package",
        "--format",
        "rich",
    )
    assert code == 0
    assert len(output) > 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_json() -> None:
    code, data = invoke_json(
        "compare",
        SAMPLE,
        "--entry-points",
        "sample_package.core",
        "--entry-points",
        "sample_package.models",
        "--format",
        "json",
    )
    assert code == 0
    assert "entry_points" in data
    assert "shared" in data
    assert "dead" in data


def test_compare_rich() -> None:
    code, _output = invoke(
        "compare",
        SAMPLE,
        "--entry-points",
        "sample_package.core",
        "--entry-points",
        "sample_package.models",
        "--format",
        "rich",
    )
    assert code == 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_json() -> None:
    code, data = invoke_json(
        "report", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    assert code == 0
    assert "schema_version" in data
    assert "entry_point" in data
    assert data["entry_point"] == "sample_package.core"
    assert "suggested_refactorings" in data


def test_report_markdown() -> None:
    code, output = invoke(
        "report", SAMPLE, "--entry-point", "sample_package.core", "--format", "markdown"
    )
    assert code == 0
    assert "# pyweight Report" in output


def test_report_json_schema_stability() -> None:
    """JSON output must have the expected top-level keys (schema stability)."""
    code, data = invoke_json(
        "report", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    assert code == 0
    expected_keys = {
        "pyweight_version",
        "schema_version",
        "timestamp",
        "package",
        "entry_point",
        "reachability",
        "barrels",
        "hotspots",
        "hub_dependencies",
        "dead_exports",
        "deferral_candidates",
        "chokepoints",
        "cycles",
        "suggested_refactorings",
    }
    assert expected_keys <= set(data.keys())


# ---------------------------------------------------------------------------
# Config support (pyproject.toml [tool.pyweight])
# ---------------------------------------------------------------------------


def test_config_entry_point_from_pyproject(tmp_path: Path) -> None:
    """CLI reads entry-point from [tool.pyweight] in pyproject.toml without explicit flag."""
    import shutil

    shutil.copytree(SAMPLE, tmp_path / "sample_package")
    pkg_path = tmp_path / "sample_package"

    pyproject = pkg_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.pyweight]\nentry-point = "sample_package.core"\n',
        encoding="utf-8",
    )

    # suggest reads entry-point from config; do NOT pass --entry-point on CLI
    code, data = invoke_json("suggest", str(pkg_path), "--format", "json")
    assert code == 0
    assert "deferral_candidates" in data


def test_config_entry_point_not_in_pyproject(tmp_path: Path) -> None:
    """When no config present, suggest works without entry-point."""
    import shutil

    shutil.copytree(SAMPLE, tmp_path / "sample_package")
    pkg_path = tmp_path / "sample_package"

    # No pyproject.toml — must not crash, just run without entry-point filter
    code, data = invoke_json("suggest", str(pkg_path), "--format", "json")
    assert code == 0
    assert "deferral_candidates" in data


def test_config_malformed_toml_warns(tmp_path: Path) -> None:
    """Malformed pyproject.toml prints a warning and does not crash."""
    import shutil

    shutil.copytree(SAMPLE, tmp_path / "sample_package")
    pkg_path = tmp_path / "sample_package"

    pyproject = pkg_path / "pyproject.toml"
    pyproject.write_text("this is not [ valid toml !!!", encoding="utf-8")

    result = RUNNER.invoke(app, ["suggest", str(pkg_path), "--format", "json"])
    # Should succeed (fall back to empty config) with a warning
    assert result.exit_code == 0
    combined = (result.output or "") + (result.stderr or "")
    assert "Warning" in combined or "could not parse" in combined


# ---------------------------------------------------------------------------
# _parse_size unit tests
# ---------------------------------------------------------------------------


def test_parse_size_kb() -> None:
    assert _parse_size("500KB") == 500 * 1024


def test_parse_size_mb() -> None:
    assert _parse_size("50MB") == 50 * 1024**2


def test_parse_size_gb() -> None:
    assert _parse_size("2GB") == 2 * 1024**3


def test_parse_size_bare_integer() -> None:
    assert _parse_size("1024") == 1024


def test_parse_size_negative_raises() -> None:
    with pytest.raises(ValueError, match="Negative"):
        _parse_size("-10MB")


def test_parse_size_unknown_suffix_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognised"):
        _parse_size("10TB")
