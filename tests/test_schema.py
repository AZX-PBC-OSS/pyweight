"""Schema contract tests: verify JSON output structure matches golden snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from pyweight.cli import app

RUNNER = CliRunner()
SAMPLE = str(Path(__file__).parent / "fixtures" / "sample_package")
GOLDEN = Path(__file__).parent / "fixtures" / "golden"

_REPORT_REQUIRED_FIELDS = {
    "pyweight_version",
    "schema_version",
    "timestamp",
    "package",
    "entry_point",
    "reachability",
    "barrels",
    "hotspots",
    "deferral_candidates",
    "hub_dependencies",
    "dead_exports",
    "dsm",
    "chokepoints",
    "cycles",
    "suggested_refactorings",
}


def _load_golden(name: str) -> object:
    return json.loads((GOLDEN / name).read_text())


def _invoke_json(*args: str) -> dict[str, object]:
    result = RUNNER.invoke(app, list(args))
    assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
    return cast("dict[str, object]", json.loads(result.output))


def _assert_same_structure(actual: object, expected: object, path: str = "") -> None:
    """Assert actual and expected have same JSON structure (keys, types) but values can differ."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict at {path!r}, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"Key mismatch at {path!r}: {set(actual.keys()) ^ set(expected.keys())}"
        )
        for key in expected:
            _assert_same_structure(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"Expected list at {path!r}, got {type(actual)}"
        if expected and actual:
            _assert_same_structure(actual[0], expected[0], f"{path}[0]")
    elif expected is None:
        # Nullable field — actual may also be None or the same nullable type.
        # We can't enforce a specific type from a null golden value, but we
        # verify the actual is at least valid JSON.
        assert actual is None or isinstance(actual, (str, int, float, bool, list, dict)), (
            f"Unexpected type at {path!r}: {type(actual)}"
        )
    else:
        assert type(actual) == type(expected), (  # noqa: E721
            f"Type mismatch at {path!r}: expected {type(expected)}, got {type(actual)}"
        )


# ---------------------------------------------------------------------------
# Golden structure tests
# ---------------------------------------------------------------------------


def test_golden_graph_structure() -> None:
    golden = _load_golden("graph.json")
    actual = _invoke_json("graph", SAMPLE, "--format", "json")
    _assert_same_structure(actual, golden, "graph")


def test_golden_hotspots_structure() -> None:
    golden = _load_golden("hotspots.json")
    actual = _invoke_json("hotspots", SAMPLE, "--format", "json")
    _assert_same_structure(actual, golden, "hotspots")


def test_golden_barrels_structure() -> None:
    golden = _load_golden("barrels.json")
    actual = _invoke_json("barrels", SAMPLE, "--format", "json")
    _assert_same_structure(actual, golden, "barrels")


def test_golden_reach_structure() -> None:
    golden = _load_golden("reach.json")
    actual = _invoke_json(
        "reach", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    _assert_same_structure(actual, golden, "reach")


def test_golden_suggest_structure() -> None:
    golden = _load_golden("suggest.json")
    actual = _invoke_json(
        "suggest", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    _assert_same_structure(actual, golden, "suggest")


def test_golden_hubs_structure() -> None:
    golden = _load_golden("hubs.json")
    actual = _invoke_json("hubs", SAMPLE, "--format", "json")
    _assert_same_structure(actual, golden, "hubs")


def test_golden_why_structure() -> None:
    golden = _load_golden("why.json")
    actual = _invoke_json(
        "why",
        SAMPLE,
        "--target",
        "sample_package.heavy",
        "--entry-point",
        "sample_package.core",
        "--format",
        "json",
    )
    _assert_same_structure(actual, golden, "why")


def test_golden_report_structure() -> None:
    golden = _load_golden("report.json")
    actual = _invoke_json(
        "report", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    _assert_same_structure(actual, golden, "report")


def test_golden_analyze_structure() -> None:
    golden = _load_golden("analyze.json")
    actual = _invoke_json(
        "analyze", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    _assert_same_structure(actual, golden, "analyze")


def test_golden_compare_structure() -> None:
    golden = _load_golden("compare.json")
    actual = _invoke_json(
        "compare",
        SAMPLE,
        "--entry-points",
        "sample_package.core",
        "--entry-points",
        "sample_package.models",
        "--format",
        "json",
    )
    _assert_same_structure(actual, golden, "compare")


# ---------------------------------------------------------------------------
# report required fields
# ---------------------------------------------------------------------------


def test_golden_report_required_fields() -> None:
    actual = _invoke_json(
        "report", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    missing = _REPORT_REQUIRED_FIELDS - set(actual.keys())
    assert not missing, f"report JSON missing required fields: {missing}"


# ---------------------------------------------------------------------------
# pyweight_version present in all per-command outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "golden_name",
    [
        "graph.json",
        "hotspots.json",
        "barrels.json",
        "reach.json",
        "suggest.json",
        "hubs.json",
        "why.json",
        "report.json",
        "analyze.json",
        "compare.json",
    ],
)
def test_golden_pyweight_version_present(golden_name: str) -> None:
    data = _load_golden(golden_name)
    assert isinstance(data, dict), f"{golden_name}: expected top-level dict"
    assert "pyweight_version" in data, f"{golden_name}: missing 'pyweight_version' key"


# ---------------------------------------------------------------------------
# confidence values
# ---------------------------------------------------------------------------


_VALID_CONFIDENCE = {"high", "medium", "low"}


def _collect_confidence_values(obj: object, path: str = "") -> list[tuple[str, str]]:
    """Recursively find all 'confidence' field values as (path, value) pairs."""
    results: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        if "confidence" in obj:
            val = obj["confidence"]
            if isinstance(val, str):
                results.append((f"{path}.confidence", val))
        for k, v in obj.items():
            results.extend(_collect_confidence_values(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(_collect_confidence_values(item, f"{path}[{i}]"))
    return results


def test_golden_confidence_values() -> None:
    """All confidence fields across golden outputs must be high/medium/low."""
    files = [
        "graph.json",
        "hotspots.json",
        "barrels.json",
        "reach.json",
        "suggest.json",
        "hubs.json",
        "why.json",
        "report.json",
        "analyze.json",
        "compare.json",
    ]
    violations: list[str] = []
    for name in files:
        data = _load_golden(name)
        for path, val in _collect_confidence_values(data, name):
            if val not in _VALID_CONFIDENCE:
                violations.append(f"{path}: {val!r}")
    assert not violations, "Invalid confidence values:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# numeric field type validation
# ---------------------------------------------------------------------------


def test_golden_report_numeric_fields() -> None:
    """Spot-check that key numeric fields in report JSON are actually numbers."""
    data = _invoke_json(
        "report", SAMPLE, "--entry-point", "sample_package.core", "--format", "json"
    )
    hotspots = cast("list[dict[str, object]]", data["hotspots"])
    for hs in hotspots:
        assert isinstance(hs["fan_in"], int), f"hotspot fan_in not int: {hs['fan_in']!r}"
        assert isinstance(hs["score"], float), f"hotspot score not float: {hs['score']!r}"

    hubs = cast("list[dict[str, object]]", data["hub_dependencies"])
    for hub in hubs:
        assert isinstance(hub["fan_in"], int), f"hub fan_in not int: {hub['fan_in']!r}"
        assert isinstance(hub["fan_out"], int), f"hub fan_out not int: {hub['fan_out']!r}"
        assert isinstance(hub["hub_score"], float), (
            f"hub hub_score not float: {hub['hub_score']!r}"
        )
