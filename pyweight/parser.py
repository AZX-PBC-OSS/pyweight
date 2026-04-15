from __future__ import annotations

import ast
import logging
import platform
from typing import TYPE_CHECKING, Protocol

from pyweight import __version__ as _tool_version
from pyweight.models import Confidence, ImportRef, ModuleInfo, Purity

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache protocol — the real Cache implementation lives elsewhere; this stub
# lets the parser accept a cache without a circular import.
# ---------------------------------------------------------------------------


class Cache(Protocol):
    """Minimal interface the parser requires from a cache object."""

    def get_module_info(
        self,
        path: Path,
        mtime: float,
        size: int,
        python_version: str,
        tool_version: str,
    ) -> ModuleInfo | None: ...

    def put_module_info(
        self,
        path: Path,
        mtime: float,
        size: int,
        python_version: str,
        tool_version: str,
        info: ModuleInfo,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Internal visitors
# ---------------------------------------------------------------------------


class _ImportVisitor(ast.NodeVisitor):
    """Collect ImportRef instances from a parsed AST."""

    def __init__(self) -> None:
        self._scope_stack: list[str] = ["module"]
        self.imports: list[ImportRef] = []
        self.has_getattr: bool = False
        self.all_names: frozenset[str] = frozenset()
        self.defined_names: list[str] = []
        self.star_imports: list[str] = []
        # Whether the current node walk is inside a TYPE_CHECKING guard
        self._in_type_checking: bool = False
        # Whether the current node walk is inside a conditional (non-TYPE_CHECKING)
        self._in_conditional: bool = False

    @property
    def _scope(self) -> str:
        return self._scope_stack[-1]

    # ------------------------------------------------------------------
    # Scope management helpers
    # ------------------------------------------------------------------

    def _push_scope(self, name: str) -> None:
        # Build fully-qualified scope: e.g. "Foo.process" rather than bare "process"
        if len(self._scope_stack) == 1:
            # Directly under module — no prefix
            self._scope_stack.append(name)
        else:
            self._scope_stack.append(f"{self._scope_stack[-1]}.{name}")

    def _pop_scope(self) -> None:
        self._scope_stack.pop()

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if len(self._scope_stack) == 1:
            # Top-level definition — record as defined name
            self.defined_names.append(node.name)
        self._push_scope(node.name)
        self.generic_visit(node)
        self._pop_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "__getattr__" and self._scope == "module":
            self.has_getattr = True
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node)

    # ------------------------------------------------------------------
    # __all__ extraction
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._scope == "module":
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    self.all_names = _extract_all(node.value)
                else:
                    # Simple name assignments at module level count as defined names
                    if isinstance(target, ast.Name):
                        self.defined_names.append(target.id)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # TYPE_CHECKING guard detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_type_checking_test(test: ast.expr) -> bool:
        """Return True if *test* is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        return (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and test.value.id == "typing"
        )

    def visit_If(self, node: ast.If) -> None:
        if self._is_type_checking_test(node.test):
            prev_tc = self._in_type_checking
            self._in_type_checking = True
            for stmt in node.body:
                self.visit(stmt)
            self._in_type_checking = prev_tc
            # Visit the else branch as a regular conditional (not type-checking)
            if node.orelse:
                prev_cond = self._in_conditional
                self._in_conditional = True
                self._in_type_checking = False
                for stmt in node.orelse:
                    self.visit(stmt)
                self._in_conditional = prev_cond
                self._in_type_checking = prev_tc
        else:
            prev = self._in_conditional
            self._in_conditional = True
            self.generic_visit(node)
            self._in_conditional = prev

    def visit_Try(self, node: ast.Try) -> None:
        prev = self._in_conditional
        self._in_conditional = True
        self.generic_visit(node)
        self._in_conditional = prev

    def visit_TryStar(self, node: ast.TryStar) -> None:
        prev = self._in_conditional
        self._in_conditional = True
        self.generic_visit(node)
        self._in_conditional = prev

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            ref = ImportRef(
                module=alias.name,
                names=(),
                alias=alias.asname,
                lineno=node.lineno,
                scope=self._scope,
                level=0,
                is_type_checking=self._in_type_checking,
                is_conditional=self._in_conditional,
                is_star=False,
            )
            self.imports.append(ref)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        level = node.level or 0
        is_star = len(node.names) == 1 and node.names[0].name == "*"

        if is_star:
            self.star_imports.append(module)
            ref = ImportRef(
                module=module,
                names=("*",),
                alias=None,
                lineno=node.lineno,
                scope=self._scope,
                level=level,
                is_type_checking=self._in_type_checking,
                is_conditional=self._in_conditional,
                is_star=True,
            )
            self.imports.append(ref)
        else:
            for alias in node.names:
                ref = ImportRef(
                    module=module,
                    names=(alias.name,),
                    alias=alias.asname,
                    lineno=node.lineno,
                    scope=self._scope,
                    level=level,
                    is_type_checking=self._in_type_checking,
                    is_conditional=self._in_conditional,
                    is_star=False,
                )
                self.imports.append(ref)


# ---------------------------------------------------------------------------


class _UsageVisitor(ast.NodeVisitor):
    """Track which names are used in which scopes."""

    def __init__(self) -> None:
        self._scope_stack: list[str] = ["module"]
        # name -> set of scopes where it was used
        self.usage: dict[str, set[str]] = {}
        # Names already recorded as attribute chain roots; skip in visit_Name
        self._attr_roots: set[ast.AST] = set()

    @property
    def _scope(self) -> str:
        return self._scope_stack[-1]

    def _record(self, name: str) -> None:
        if name not in self.usage:
            self.usage[name] = set()
        self.usage[name].add(self._scope)

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        # Use fully-qualified names to avoid scope collisions for same-named
        # nested functions/methods across different classes.
        if len(self._scope_stack) == 1:
            self._scope_stack.append(node.name)
        else:
            self._scope_stack.append(f"{self._scope_stack[-1]}.{node.name}")
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node not in self._attr_roots:
            self._record(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Walk to the root of the attribute chain and record only the root name.
        # Mark the root Name node so visit_Name skips it (avoids double-counting).
        # Then call generic_visit so all sub-expressions (subscript slices,
        # call args, etc.) inside the chain are still visited — visit_Name will
        # skip the already-recorded root via _attr_roots.
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and isinstance(root.ctx, ast.Load):
            self._attr_roots.add(root)
            self._record(root.id)
        self.generic_visit(node)


# ---------------------------------------------------------------------------


class _PurityClassifier:
    """Classify a module's purity from its top-level AST statements."""

    def classify(self, tree: ast.Module) -> Purity:
        has_side_effects = False
        has_unknown = False
        for stmt in tree.body:
            verdict = self._check_stmt(stmt)
            if verdict is Purity.SIDE_EFFECTS:
                has_side_effects = True
            elif verdict is Purity.UNKNOWN:
                has_unknown = True
        if has_side_effects:
            return Purity.SIDE_EFFECTS
        if has_unknown:
            return Purity.UNKNOWN
        return Purity.PURE

    def _check_stmt(self, stmt: ast.stmt) -> Purity:
        # Bare expression that is a Call — definitely a side effect
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return Purity.SIDE_EFFECTS
        # Assignment whose RHS is a Call — ambiguous (could be side-effectful)
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            return Purity.UNKNOWN
        return Purity.PURE


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _extract_all(node: ast.expr) -> frozenset[str]:
    """Extract string elements from a list or tuple literal."""
    names: list[str] = []
    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
    return frozenset(names)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_module(
    path: Path,
    qualified_name: str,
    cache: Cache | None = None,
) -> ModuleInfo:
    """Parse a Python source file and return a :class:`ModuleInfo`.

    When *cache* is provided and the module is already cached (same path,
    mtime, size, Python version, and tool version), the cached result is
    returned immediately.  After parsing, the result is stored so subsequent
    calls are fast.

    If the file cannot be decoded as UTF-8 or contains a syntax error, a
    warning is emitted via the ``logging`` module and a minimal
    :class:`ModuleInfo` with no imports and ``purity="unknown"`` is returned
    so the module still appears in the dependency graph without contributing
    any edges.
    """
    python_version = platform.python_version()
    tool_version = _tool_version

    # Snapshot stat() once to avoid a TOCTOU race between cache lookup and write.
    stat = path.stat() if cache is not None else None

    if cache is not None and stat is not None:
        cached = cache.get_module_info(
            path, stat.st_mtime, stat.st_size, python_version, tool_version
        )
        if cached is not None:
            return cached

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("Skipping %s: cannot decode as UTF-8: %s", path, exc)
        return ModuleInfo(
            path=path,
            qualified_name=qualified_name,
            imports=(),
            is_init=path.name == "__init__.py",
            has_getattr=False,
            has_all=False,
            all_names=frozenset(),
            defined_names=frozenset(),
            star_imports=(),
            purity=Purity.UNKNOWN,
        )

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return ModuleInfo(
            path=path,
            qualified_name=qualified_name,
            imports=(),
            is_init=path.name == "__init__.py",
            has_getattr=False,
            has_all=False,
            all_names=frozenset(),
            defined_names=frozenset(),
            star_imports=(),
            purity=Purity.UNKNOWN,
        )

    import_visitor = _ImportVisitor()
    import_visitor.visit(tree)

    usage_visitor = _UsageVisitor()
    usage_visitor.visit(tree)

    purity_classifier = _PurityClassifier()
    purity = purity_classifier.classify(tree)

    usage_scopes: dict[str, frozenset[str]] = {
        name: frozenset(scopes) for name, scopes in usage_visitor.usage.items()
    }

    info = ModuleInfo(
        path=path,
        qualified_name=qualified_name,
        imports=tuple(import_visitor.imports),
        is_init=path.name == "__init__.py",
        has_getattr=import_visitor.has_getattr,
        has_all=bool(import_visitor.all_names),
        all_names=import_visitor.all_names,
        defined_names=frozenset(import_visitor.defined_names),
        star_imports=tuple(import_visitor.star_imports),
        purity=purity,
        usage_scopes=usage_scopes,
    )

    if cache is not None and stat is not None:
        cache.put_module_info(
            path, stat.st_mtime, stat.st_size, python_version, tool_version, info
        )

    return info


# ---------------------------------------------------------------------------
# Symbol origin resolution
# ---------------------------------------------------------------------------


def get_symbol_origin(
    module_info: ModuleInfo,
    symbol_name: str,
    all_names_by_module: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str | None, Confidence]:
    """Return the probable origin module for *symbol_name* and a confidence level.

    - Explicit ``from X import symbol_name`` → ``(X, Confidence.HIGH)``
    - Star import where the source module declares ``__all__`` containing the
      symbol (via *all_names_by_module*) → ``(source, Confidence.MEDIUM)``
    - Star import with no ``__all__`` information → ``(source, Confidence.LOW)``
    """
    # Explicit named import — highest confidence
    for ref in module_info.imports:
        if not ref.is_star and symbol_name in ref.names:
            return (ref.module, Confidence.HIGH)

    # Star imports — check __all__ from source module if available.
    # Accumulate a low_candidate and only fall through to LOW after
    # exhausting all star-import candidates so we don't short-circuit
    # before finding a MEDIUM match in a later star import.
    low_candidate: str | None = None
    for ref in module_info.imports:
        if ref.is_star:
            source = ref.module if ref.module else None
            if (
                all_names_by_module is not None
                and source is not None
                and source in all_names_by_module
                and symbol_name in all_names_by_module[source]
            ):
                return (source, Confidence.MEDIUM)
            if low_candidate is None:
                low_candidate = source
    if low_candidate is not None:
        return (low_candidate, Confidence.LOW)

    return (None, Confidence.LOW)
