"""Target project virtual environment discovery."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetEnv:
    """Describes a target project's Python environment."""

    python: Path
    """Absolute path to the Python interpreter."""

    site_packages: tuple[Path, ...]
    """Site-packages directories (may be multiple for conda/system)."""


def discover_target_env(package_root: Path) -> TargetEnv | None:
    """Discover the target project's venv from *package_root*.

    Returns ``None`` when no environment is found (caller should fall back to
    the current interpreter).

    Discovery order:
    1. Walk upward from *package_root* looking for ``.venv/``, ``venv/`` dirs
       with a ``bin/python`` (stops at first ``pyproject.toml`` or filesystem root)
    2. ``VIRTUAL_ENV`` environment variable as fallback (only if the walk finds
       nothing — avoids pyweight's own venv when analyzing a different project)
    """
    env = _walk_for_venv(package_root)
    if env is not None:
        return env

    env = _try_virtual_env_var()
    if env is not None:
        return env

    return None


def env_from_python(python: Path) -> TargetEnv:
    """Build a :class:`TargetEnv` from an explicit Python interpreter path.

    Does NOT resolve symlinks — if the user passes a venv python, we must
    invoke it as-is so ``site.getsitepackages()`` returns venv site-packages.
    """
    python = python.absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    site_pkgs = _find_site_packages(python)
    return TargetEnv(python=python, site_packages=tuple(site_pkgs))


def _try_virtual_env_var() -> TargetEnv | None:
    venv_dir = os.environ.get("VIRTUAL_ENV")
    if not venv_dir:
        return None
    venv_path = Path(venv_dir)
    python = _find_python_in_venv(venv_path)
    if python is None:
        return None
    site_pkgs = _find_site_packages(python)
    return TargetEnv(python=python, site_packages=tuple(site_pkgs))


def _walk_for_venv(start: Path) -> TargetEnv | None:
    """Walk upward from *start* looking for a venv directory."""
    current = start.resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        for name in (".venv", "venv"):
            candidate = current / name
            python = _find_python_in_venv(candidate)
            if python is not None:
                site_pkgs = _find_site_packages(python)
                return TargetEnv(python=python, site_packages=tuple(site_pkgs))
        # Stop at project boundary
        boundary_markers = ("pyproject.toml", "setup.py", "setup.cfg")
        if any((current / m).exists() for m in boundary_markers):
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _find_python_in_venv(venv_dir: Path) -> Path | None:
    """Return the python binary inside *venv_dir*, or None.

    Intentionally does NOT resolve symlinks — venv python must be invoked
    via the venv path so ``site.getsitepackages()`` returns the venv's
    own site-packages rather than the base interpreter's.
    """
    if not venv_dir.is_dir():
        return None
    # Unix
    candidate = venv_dir / "bin" / "python"
    if candidate.is_file():
        return candidate.absolute()
    # Windows
    candidate = venv_dir / "Scripts" / "python.exe"
    if candidate.is_file():
        return candidate.absolute()
    return None


def _find_site_packages(python: Path) -> list[Path]:
    """Query *python* for its site-packages directories."""
    try:
        result = subprocess.run(
            [str(python), "-c", "import site; print(*site.getsitepackages(), sep='\\n')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return []

    if result.returncode != 0:
        return []

    paths: list[Path] = []
    for line in result.stdout.strip().splitlines():
        p = Path(line.strip())
        if p.is_dir():
            paths.append(p)
    return paths


def is_current_env(env: TargetEnv) -> bool:
    """Return True if *env* points to the currently running interpreter."""
    return env.python.resolve() == Path(sys.executable).resolve()
