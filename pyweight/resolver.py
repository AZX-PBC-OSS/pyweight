from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003

from pyweight.models import Confidence

logger = logging.getLogger(__name__)


class PackageResolver:
    """Resolve module qualified names to filesystem paths within a package root.

    Uses pure filesystem traversal — no code execution, no importlib.
    """

    def __init__(self, package_root: Path) -> None:
        self._root = package_root
        # qualified_name → Path (for regular packages and module files)
        self._modules: dict[str, Path] = {}
        # Reverse map: Path → qualified_name (O(1) lookup for _path_to_qname)
        self._path_to_name: dict[Path, str] = {}
        # qualified names that are namespace packages (dirs without __init__.py)
        self._namespace_pkgs: set[str] = set()
        # Walk starting from the package root, seeding parts with its own name
        self._walk(package_root, [package_root.name])

    def _walk(self, directory: Path, parts: list[str]) -> None:
        """Recursively walk *directory* and populate the module mapping."""
        has_init = (directory / "__init__.py").exists()
        qname = ".".join(parts)

        if has_init:
            init_path = directory / "__init__.py"
            self._modules[qname] = init_path
            self._path_to_name[init_path] = qname
        else:
            # Namespace package — a directory without __init__.py
            self._namespace_pkgs.add(qname)
            logger.info("Detected namespace package %s", qname)

        for child in sorted(directory.iterdir()):
            # Skip hidden entries and __pycache__
            if child.name.startswith(".") or child.name == "__pycache__":
                continue

            if child.is_dir():
                self._walk(child, [*parts, child.name])
            elif child.is_file() and child.suffix == ".py" and child.name != "__init__.py":
                stem = child.stem
                child_qname = ".".join([*parts, stem])
                self._modules[child_qname] = child
                self._path_to_name[child] = child_qname

    def resolve(
        self,
        module_name: str,
        from_module: str | None,
        level: int = 0,
        is_init: bool = False,
    ) -> tuple[Path | None, Confidence]:
        """Resolve *module_name* to a Path and confidence.

        For relative imports (*level* > 0) *from_module* must be provided.
        Returns ``(None, Confidence.LOW)`` for external or unresolvable names.

        *is_init* must be True when the importing file is an ``__init__.py``.
        For ``__init__.py``, the module's qualified name IS the package, so
        level=1 means "same package" rather than "parent package". We subtract
        one from the strip count to compensate.
        """
        if level > 0:
            if from_module is None:
                return (None, Confidence.LOW)
            # from_module is a dotted module name (e.g. "sample_package.utils.helpers").
            # level=1 means same package as from_module, so strip 1 trailing component.
            # level=2 means parent package, strip 2 trailing components, etc.
            # For __init__.py the qualified name already IS the package, so level=1
            # requires no stripping and level=2 strips only one component.
            parts = from_module.split(".")
            strip = level - 1 if is_init else level
            if strip == 0:
                base_parts = parts
            elif strip < len(parts):
                base_parts = parts[:-strip]
            else:
                return (None, Confidence.LOW)
            if module_name:
                absolute = ".".join([*base_parts, module_name]) if base_parts else module_name
            else:
                absolute = ".".join(base_parts) if base_parts else ""
            return self._lookup(absolute)

        # Absolute import
        return self._lookup(module_name)

    def _lookup(self, module_name: str) -> tuple[Path | None, Confidence]:
        if not module_name:
            return (None, Confidence.LOW)

        if module_name in self._modules:
            # Intentional per ticket spec: any module whose ancestor chain includes a
            # namespace package (no __init__.py) is resolved with LOW confidence, because
            # namespace packages can be split across multiple directories on sys.path and
            # we cannot guarantee we found all contributions.
            confidence = (
                Confidence.LOW if self._has_namespace_ancestor(module_name) else Confidence.HIGH
            )
            return (self._modules[module_name], confidence)

        if module_name in self._namespace_pkgs:
            # Namespace package itself has no __init__.py → no Path
            return (None, Confidence.LOW)

        return (None, Confidence.LOW)

    def _has_namespace_ancestor(self, module_name: str) -> bool:
        """Return True if any ancestor package of *module_name* is a namespace package."""
        parts = module_name.split(".")
        return any(".".join(parts[:i]) in self._namespace_pkgs for i in range(1, len(parts)))

    def is_external(self, module_name: str) -> bool:
        """Return True if *module_name* is not resolvable within this package root."""
        if not module_name:
            return True
        top = module_name.split(".")[0]
        return top not in self._modules and top not in self._namespace_pkgs

    @property
    def all_modules(self) -> dict[str, Path]:
        """Return the full qualified_name → Path mapping."""
        return dict(self._modules)

    def qname_for_path(self, path: Path) -> str | None:
        """Return the qualified name for *path*, or None if not in this package."""
        return self._path_to_name.get(path)
