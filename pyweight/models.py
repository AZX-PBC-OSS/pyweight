from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import NotImplementedType


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def rank(self) -> int:
        return {"high": 2, "medium": 1, "low": 0}[self.value]


class Purity(StrEnum):
    PURE = "pure"
    SIDE_EFFECTS = "side_effects"
    UNKNOWN = "unknown"


class RiskLevel(StrEnum):
    SAFE = "safe"
    HAS_SIDE_EFFECTS = "has_side_effects"
    UNKNOWN = "unknown"


class RefactoringKind(StrEnum):
    DEFER_IMPORT = "defer_import"
    REMOVE_DEAD_EXPORT = "remove_dead_export"
    BREAK_CYCLE = "break_cycle"
    DECOUPLE_HUB = "decouple_hub"


_ESCAPE_LEVEL_ORDER: dict[str, int] = {
    "function_local": 0,
    "class_body": 1,
    "module_scope": 2,
    "reexported": 3,
}


class EscapeLevel(StrEnum):
    FUNCTION_LOCAL = "function_local"
    CLASS_BODY = "class_body"
    MODULE_SCOPE = "module_scope"
    REEXPORTED = "reexported"

    # All four rich-comparison methods must be defined explicitly: StrEnum
    # inherits from str, which already provides __gt__, __le__, __ge__ based on
    # lexicographic ordering. @functools.total_ordering cannot override inherited
    # str methods, so each one must be replaced manually here.

    def __lt__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, EscapeLevel):
            return NotImplemented
        return _ESCAPE_LEVEL_ORDER[self.value] < _ESCAPE_LEVEL_ORDER[other.value]

    def __gt__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, EscapeLevel):
            return NotImplemented
        return _ESCAPE_LEVEL_ORDER[self.value] > _ESCAPE_LEVEL_ORDER[other.value]

    def __le__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, EscapeLevel):
            return NotImplemented
        return _ESCAPE_LEVEL_ORDER[self.value] <= _ESCAPE_LEVEL_ORDER[other.value]

    def __ge__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, EscapeLevel):
            return NotImplemented
        return _ESCAPE_LEVEL_ORDER[self.value] >= _ESCAPE_LEVEL_ORDER[other.value]

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True, slots=True)
class ImportRef:
    module: str
    names: tuple[str, ...]
    alias: str | None
    lineno: int
    scope: str
    level: int
    is_type_checking: bool
    is_conditional: bool
    is_star: bool


@dataclass(frozen=True, slots=True)
class EdgeInfo:
    confidence: Confidence
    is_type_checking: bool
    is_conditional: bool


type JsonDict = dict[str, object]


def _import_ref_to_dict(r: ImportRef) -> JsonDict:
    return {
        "module": r.module,
        "names": list(r.names),
        "alias": r.alias,
        "lineno": r.lineno,
        "scope": r.scope,
        "level": r.level,
        "is_type_checking": r.is_type_checking,
        "is_conditional": r.is_conditional,
        "is_star": r.is_star,
    }


def _import_ref_from_dict(d: JsonDict) -> ImportRef:
    names_raw = cast("list[object]", d["names"])
    return ImportRef(
        module=cast("str", d["module"]),
        names=tuple(cast("str", n) for n in names_raw),
        alias=cast("str | None", d["alias"]),
        lineno=cast("int", d["lineno"]),
        scope=cast("str", d["scope"]),
        level=cast("int", d["level"]),
        is_type_checking=cast("bool", d["is_type_checking"]),
        is_conditional=cast("bool", d["is_conditional"]),
        is_star=cast("bool", d["is_star"]),
    )


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    path: Path
    qualified_name: str
    imports: tuple[ImportRef, ...]
    is_init: bool
    has_getattr: bool
    has_all: bool
    all_names: frozenset[str]
    defined_names: frozenset[str]
    star_imports: tuple[str, ...]
    purity: Purity
    usage_scopes: dict[str, frozenset[str]] = field(default_factory=dict[str, frozenset[str]])

    def to_dict(self) -> JsonDict:
        return {
            "path": str(self.path),
            "qualified_name": self.qualified_name,
            "imports": [_import_ref_to_dict(r) for r in self.imports],
            "is_init": self.is_init,
            "has_getattr": self.has_getattr,
            "has_all": self.has_all,
            "all_names": sorted(self.all_names),
            "defined_names": sorted(self.defined_names),
            "star_imports": list(self.star_imports),
            "purity": self.purity,
            "usage_scopes": {k: sorted(v) for k, v in self.usage_scopes.items()},
        }

    @classmethod
    def from_dict(cls, d: JsonDict) -> ModuleInfo:
        imports_raw = cast("list[JsonDict]", d["imports"])
        imports = tuple(_import_ref_from_dict(r) for r in imports_raw)
        scopes_raw = cast("dict[str, list[str]]", d["usage_scopes"])
        usage_scopes: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in scopes_raw.items()}
        all_names_raw = cast("list[str]", d["all_names"])
        defined_names_raw = cast("list[str]", d["defined_names"])
        star_imports_raw = cast("list[str]", d["star_imports"])
        return cls(
            path=Path(cast("str", d["path"])),
            qualified_name=cast("str", d["qualified_name"]),
            imports=imports,
            is_init=cast("bool", d["is_init"]),
            has_getattr=cast("bool", d["has_getattr"]),
            has_all=cast("bool", d["has_all"]),
            all_names=frozenset(all_names_raw),
            defined_names=frozenset(defined_names_raw),
            star_imports=tuple(star_imports_raw),
            purity=Purity(cast("str", d["purity"])),
            usage_scopes=usage_scopes,
        )


@dataclass(frozen=True, slots=True)
class CostEstimate:
    module: str
    dist_name: str | None
    direct_size_bytes: int
    transitive_size_bytes: int
    confidence: Confidence
    provenance: str

    def to_dict(self) -> JsonDict:
        return {
            "module": self.module,
            "dist_name": self.dist_name,
            "direct_size_bytes": self.direct_size_bytes,
            "transitive_size_bytes": self.transitive_size_bytes,
            "confidence": self.confidence.value,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: JsonDict) -> CostEstimate:
        return cls(
            module=cast("str", d["module"]),
            dist_name=cast("str | None", d["dist_name"]),
            direct_size_bytes=cast("int", d["direct_size_bytes"]),
            transitive_size_bytes=cast("int", d["transitive_size_bytes"]),
            confidence=Confidence(cast("str", d["confidence"])),
            provenance=cast("str", d["provenance"]),
        )


@dataclass(frozen=True, slots=True)
class TimingResult:
    module: str
    self_time_us: int
    cumulative_time_us: int
    dependencies: tuple[str, ...]

    def to_dict(self) -> JsonDict:
        return {
            "module": self.module,
            "self_time_us": self.self_time_us,
            "cumulative_time_us": self.cumulative_time_us,
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, d: JsonDict) -> TimingResult:
        deps_raw = cast("list[str]", d["dependencies"])
        return cls(
            module=cast("str", d["module"]),
            self_time_us=cast("int", d["self_time_us"]),
            cumulative_time_us=cast("int", d["cumulative_time_us"]),
            dependencies=tuple(deps_raw),
        )


@dataclass(frozen=True, slots=True)
class MemoryResult:
    module: str
    python_heap_bytes: int
    rss_bytes: int

    def to_dict(self) -> JsonDict:
        return {
            "module": self.module,
            "python_heap_bytes": self.python_heap_bytes,
            "rss_bytes": self.rss_bytes,
        }

    @classmethod
    def from_dict(cls, d: JsonDict) -> MemoryResult:
        return cls(
            module=cast("str", d["module"]),
            python_heap_bytes=cast("int", d["python_heap_bytes"]),
            rss_bytes=cast("int", d["rss_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class DeferralCandidate:
    file: Path
    import_ref: ImportRef
    used_in_scopes: frozenset[str]
    escape_level: EscapeLevel
    estimated_cost: CostEstimate | None
    confidence: Confidence
    risk: RiskLevel
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BarrelInfo:
    path: Path
    qualified_name: str
    star_imports: tuple[str, ...]
    eager_imports: int
    reexported_count: int
    has_lazy_loading: bool
    has_all: bool


@dataclass(frozen=True, slots=True)
class HubDependency:
    module: str
    fan_in: int
    fan_out: int
    hub_score: float
    is_init: bool
    transitive_cost: CostEstimate | None


@dataclass(frozen=True, slots=True)
class DSMResult:
    modules: tuple[str, ...]
    matrix: tuple[tuple[bool, ...], ...]
    layer_assignments: dict[str, int]
    violations: tuple[tuple[str, str], ...]
    layering_health: float


@dataclass(frozen=True, slots=True)
class Hotspot:
    module: str
    fan_in: int
    transitive_cost: CostEstimate
    score: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    entry_point: str
    required_modules: frozenset[str]
    unreachable_modules: frozenset[str]
    required_cost: CostEstimate | None
    unreachable_cost: CostEstimate | None
    root_causes: tuple[str, ...]
    import_paths: dict[str, tuple[str, ...]]
    confidence: Confidence
    unknown_origins: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuggestedRefactoring:
    id: str
    impact: float
    kind: RefactoringKind
    file: Path
    lineno: int
    description: str
    current_code: str
    proposed_pattern_pre315: str
    proposed_pattern_pep810: str | None
    estimated_savings_mb: float
    risk: RiskLevel
    purity_check: str
    confidence: Confidence
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WhyResult:
    entry_point: str
    target: str
    paths: tuple[tuple[str, ...], ...]
    critical_edges: tuple[tuple[str, str], ...]
    suggested_cuts: tuple[str, ...] | None
    total_cost: CostEstimate | None


@dataclass(frozen=True, slots=True)
class ExternalWhyResult:
    entry_point: str
    target_package: str
    target_import_names: tuple[str, ...]
    direct_importers: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...]
    pip_dependency_chain: tuple[str, ...] | None
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class DeadExport:
    module: str
    symbol_name: str
    export_mechanism: str
    confidence: Confidence


# Schema versioning policy for FullReport:
#   - Adding new optional fields: no schema_version bump needed.
#   - Removing fields or changing field types: bump schema_version.
#   - Current schema_version: "1"
@dataclass(frozen=True, slots=True)
class FullReport:
    schema_version: str
    tool_version: str
    timestamp: str
    package: str
    entry_point: str
    reachability: ReachabilityResult | None
    barrels: tuple[BarrelInfo, ...]
    hotspots: tuple[Hotspot, ...]
    deferral_candidates: tuple[DeferralCandidate, ...]
    hub_dependencies: tuple[HubDependency, ...]
    dead_exports: tuple[DeadExport, ...]
    dsm: DSMResult | None
    chokepoints: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]
    suggested_refactorings: tuple[SuggestedRefactoring, ...]


@dataclass(slots=True)
class ImportGraph:
    modules: dict[str, ModuleInfo] = field(default_factory=dict[str, ModuleInfo])
    edges: dict[str, dict[str, EdgeInfo]] = field(default_factory=dict[str, dict[str, EdgeInfo]])
    reverse_edges: dict[str, dict[str, EdgeInfo]] = field(
        default_factory=dict[str, dict[str, EdgeInfo]]
    )
    external_deps: set[str] = field(default_factory=set[str])
    module_external_imports: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    _name_to_id: dict[str, int] = field(default_factory=dict[str, int])
    _id_to_name: dict[int, str] = field(default_factory=dict[int, str])

    def _ensure_id(self, name: str) -> None:
        if name not in self._name_to_id:
            new_id = len(self._name_to_id)
            self._name_to_id[name] = new_id
            self._id_to_name[new_id] = name

    def add_module(self, name: str, info: ModuleInfo) -> None:
        self.modules[name] = info
        self._ensure_id(name)

    def add_edge(self, source: str, target: str, edge_info: EdgeInfo) -> None:
        self._ensure_id(source)
        self._ensure_id(target)
        if source not in self.edges:
            self.edges[source] = {}
        if target in self.edges[source]:
            existing = self.edges[source][target]
            # Runtime edge wins over TYPE_CHECKING; keep best confidence; conditional
            # only if both are conditional.
            edge_info = EdgeInfo(
                confidence=max(
                    existing.confidence,
                    edge_info.confidence,
                    key=lambda c: c.rank(),
                ),
                is_type_checking=existing.is_type_checking and edge_info.is_type_checking,
                is_conditional=existing.is_conditional and edge_info.is_conditional,
            )
        self.edges[source][target] = edge_info
        if target not in self.reverse_edges:
            self.reverse_edges[target] = {}
        self.reverse_edges[target][source] = edge_info

    def node_id(self, name: str) -> int:
        return self._name_to_id[name]

    def node_name(self, node_id: int) -> str:
        return self._id_to_name[node_id]
