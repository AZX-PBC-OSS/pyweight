"""Tests for pyweight.venv module."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyweight.venv import (
    TargetEnv,
    _find_python_in_venv,
    _find_site_packages,
    _walk_for_venv,
    discover_target_env,
    env_from_python,
    is_current_env,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unix_venv(base: Path, name: str = ".venv") -> Path:
    """Create a minimal Unix-style venv directory under *base*."""
    venv_dir = base / name
    python = venv_dir / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    return venv_dir


def _fake_site_packages_run(site_dirs: list[Path]):  # type: ignore[return]
    """Return a side_effect callable that fakes a successful subprocess.run result."""

    def _side_effect(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = "\n".join(str(p) for p in site_dirs)
        return result

    return _side_effect


# ---------------------------------------------------------------------------
# TargetEnv dataclass
# ---------------------------------------------------------------------------


def test_target_env_construction(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    sp = tmp_path / "site-packages"
    sp.mkdir()
    env = TargetEnv(python=python, site_packages=(sp,))
    assert env.python == python
    assert env.site_packages == (sp,)


def test_target_env_frozen(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    env = TargetEnv(python=python, site_packages=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        env.python = tmp_path / "other"  # type: ignore[misc]


def test_target_env_empty_site_packages(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    env = TargetEnv(python=python, site_packages=())
    assert env.site_packages == ()


# ---------------------------------------------------------------------------
# _find_python_in_venv
# ---------------------------------------------------------------------------


def test_find_python_in_venv_unix(tmp_path: Path) -> None:
    venv_dir = _make_unix_venv(tmp_path)
    result = _find_python_in_venv(venv_dir)
    assert result is not None
    assert result.name == "python"


def test_find_python_in_venv_windows_scripts(tmp_path: Path) -> None:
    venv_dir = tmp_path / ".venv"
    python_exe = venv_dir / "Scripts" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.touch()
    result = _find_python_in_venv(venv_dir)
    assert result is not None
    assert result.name == "python.exe"


def test_find_python_in_venv_missing_dir(tmp_path: Path) -> None:
    result = _find_python_in_venv(tmp_path / "nonexistent")
    assert result is None


def test_find_python_in_venv_dir_exists_but_no_python(tmp_path: Path) -> None:
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    result = _find_python_in_venv(venv_dir)
    assert result is None


def test_find_python_in_venv_returns_resolved_path(tmp_path: Path) -> None:
    venv_dir = _make_unix_venv(tmp_path)
    result = _find_python_in_venv(venv_dir)
    assert result is not None
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# _find_site_packages
# ---------------------------------------------------------------------------


def test_find_site_packages_returns_existing_dirs(tmp_path: Path) -> None:
    sp1 = tmp_path / "site-packages"
    sp1.mkdir()
    sp2 = tmp_path / "dist-packages"
    sp2.mkdir()
    python = tmp_path / "python"
    python.touch()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp1, sp2])):
        result = _find_site_packages(python)

    assert sp1 in result
    assert sp2 in result


def test_find_site_packages_filters_nonexistent_dirs(tmp_path: Path) -> None:
    real_sp = tmp_path / "site-packages"
    real_sp.mkdir()
    ghost_sp = tmp_path / "ghost"  # not created
    python = tmp_path / "python"
    python.touch()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([real_sp, ghost_sp])):
        result = _find_site_packages(python)

    assert real_sp in result
    assert ghost_sp not in result


def test_find_site_packages_nonzero_returncode(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()

    def _fail(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=_fail):
        result = _find_site_packages(python)

    assert result == []


def test_find_site_packages_timeout(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=10)):
        result = _find_site_packages(python)

    assert result == []


def test_find_site_packages_file_not_found(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()

    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = _find_site_packages(python)

    assert result == []


def test_find_site_packages_permission_error(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()

    with patch("subprocess.run", side_effect=PermissionError):
        result = _find_site_packages(python)

    assert result == []


# ---------------------------------------------------------------------------
# _walk_for_venv
# ---------------------------------------------------------------------------


def test_walk_for_venv_finds_dot_venv_in_start_dir(tmp_path: Path) -> None:
    _make_unix_venv(tmp_path, ".venv")
    sp = tmp_path / "sp"
    sp.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp])):
        result = _walk_for_venv(tmp_path)

    assert result is not None
    assert result.python.name == "python"


def test_walk_for_venv_finds_venv_dir(tmp_path: Path) -> None:
    _make_unix_venv(tmp_path, "venv")
    sp = tmp_path / "sp"
    sp.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp])):
        result = _walk_for_venv(tmp_path)

    assert result is not None


def test_walk_for_venv_finds_venv_in_parent(tmp_path: Path) -> None:
    _make_unix_venv(tmp_path, ".venv")
    nested = tmp_path / "sub" / "pkg"
    nested.mkdir(parents=True)
    sp = tmp_path / "sp"
    sp.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp])):
        result = _walk_for_venv(nested)

    assert result is not None
    assert result.python.name == "python"


def test_walk_for_venv_stops_at_pyproject_toml(tmp_path: Path) -> None:
    # pyproject.toml in the start dir prevents walking to grandparent
    _make_unix_venv(tmp_path, ".venv")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    # venv is one level above the pyproject.toml boundary

    with patch("subprocess.run", side_effect=_fake_site_packages_run([])):
        result = _walk_for_venv(project_dir)

    # The venv is in tmp_path, but we stopped at project_dir because of pyproject.toml
    assert result is None


def test_walk_for_venv_stops_at_setup_py(tmp_path: Path) -> None:
    _make_unix_venv(tmp_path, ".venv")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "setup.py").write_text("from setuptools import setup\nsetup()\n")

    with patch("subprocess.run", side_effect=_fake_site_packages_run([])):
        result = _walk_for_venv(project_dir)

    assert result is None


def test_walk_for_venv_returns_none_when_no_venv(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    # No venv anywhere, no pyproject.toml to stop early — relies on filesystem root

    with patch("subprocess.run", side_effect=_fake_site_packages_run([])):
        result = _walk_for_venv(project_dir)

    assert result is None


def test_walk_for_venv_prefers_dot_venv_over_venv(tmp_path: Path) -> None:
    _make_unix_venv(tmp_path, ".venv")
    _make_unix_venv(tmp_path, "venv")
    sp = tmp_path / "sp"
    sp.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp])):
        result = _walk_for_venv(tmp_path)

    assert result is not None
    # .venv is checked first per the implementation order
    assert ".venv" in str(result.python)


# ---------------------------------------------------------------------------
# discover_target_env
# ---------------------------------------------------------------------------


def test_discover_target_env_uses_virtual_env_var(tmp_path: Path) -> None:
    venv_dir = _make_unix_venv(tmp_path, ".venv")
    sp = tmp_path / "sp"
    sp.mkdir()

    with (
        patch.dict("os.environ", {"VIRTUAL_ENV": str(venv_dir)}, clear=False),
        patch("subprocess.run", side_effect=_fake_site_packages_run([sp])),
    ):
        result = discover_target_env(tmp_path / "project")

    assert result is not None
    assert result.python.name == "python"
    assert sp in result.site_packages


def test_discover_target_env_ignores_invalid_virtual_env_var(tmp_path: Path) -> None:
    _make_unix_venv(tmp_path, ".venv")
    sp = tmp_path / "sp"
    sp.mkdir()

    with (
        patch.dict("os.environ", {"VIRTUAL_ENV": str(tmp_path / "nonexistent_venv")}, clear=False),
        patch("subprocess.run", side_effect=_fake_site_packages_run([sp])),
    ):
        # Falls back to walking from tmp_path which has .venv
        result = discover_target_env(tmp_path)

    assert result is not None


def test_discover_target_env_falls_back_to_walk_when_no_virtual_env_var(
    tmp_path: Path,
) -> None:
    _make_unix_venv(tmp_path, ".venv")
    sp = tmp_path / "sp"
    sp.mkdir()
    env_without_virtual_env = {
        k: v for k, v in __import__("os").environ.items() if k != "VIRTUAL_ENV"
    }

    with (
        patch.dict("os.environ", env_without_virtual_env, clear=True),
        patch("subprocess.run", side_effect=_fake_site_packages_run([sp])),
    ):
        result = discover_target_env(tmp_path)

    assert result is not None


def test_discover_target_env_returns_none_when_no_venv_found(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    env_without_virtual_env = {
        k: v for k, v in __import__("os").environ.items() if k != "VIRTUAL_ENV"
    }

    with (
        patch.dict("os.environ", env_without_virtual_env, clear=True),
        patch("subprocess.run", side_effect=_fake_site_packages_run([])),
    ):
        result = discover_target_env(project_dir)

    assert result is None


def test_discover_target_env_walk_takes_priority_over_virtual_env_var(tmp_path: Path) -> None:
    # Two separate venvs: one referenced by VIRTUAL_ENV, one found by walking.
    # Walk should win to avoid picking up the analyzing tool's own venv.
    venv_via_var = tmp_path / "env_var_venv"
    python_via_var = venv_via_var / "bin" / "python"
    python_via_var.parent.mkdir(parents=True)
    python_via_var.touch()

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _make_unix_venv(project_dir, ".venv")

    sp = tmp_path / "sp"
    sp.mkdir()

    with (
        patch.dict("os.environ", {"VIRTUAL_ENV": str(venv_via_var)}, clear=False),
        patch("subprocess.run", side_effect=_fake_site_packages_run([sp])),
    ):
        result = discover_target_env(project_dir)

    assert result is not None
    # Walk should win — found .venv in project_dir before checking VIRTUAL_ENV
    assert ".venv" in str(result.python)


# ---------------------------------------------------------------------------
# env_from_python
# ---------------------------------------------------------------------------


def test_env_from_python_valid_interpreter(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    sp = tmp_path / "site-packages"
    sp.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp])):
        result = env_from_python(python)

    assert result.python == python.resolve()
    assert sp in result.site_packages


def test_env_from_python_raises_for_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Python interpreter not found"):
        env_from_python(tmp_path / "nonexistent_python")


def test_env_from_python_raises_for_directory(tmp_path: Path) -> None:
    directory = tmp_path / "a_directory"
    directory.mkdir()
    with pytest.raises(FileNotFoundError, match="Python interpreter not found"):
        env_from_python(directory)


def test_env_from_python_resolves_path(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    sp = tmp_path / "sp"
    sp.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp])):
        result = env_from_python(python)

    assert result.python.is_absolute()


def test_env_from_python_site_packages_as_tuple(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.touch()
    sp1 = tmp_path / "sp1"
    sp1.mkdir()
    sp2 = tmp_path / "sp2"
    sp2.mkdir()

    with patch("subprocess.run", side_effect=_fake_site_packages_run([sp1, sp2])):
        result = env_from_python(python)

    assert isinstance(result.site_packages, tuple)
    assert len(result.site_packages) == 2


# ---------------------------------------------------------------------------
# is_current_env
# ---------------------------------------------------------------------------


def test_is_current_env_true_for_running_interpreter() -> None:
    python = Path(sys.executable)
    env = TargetEnv(python=python, site_packages=())
    assert is_current_env(env) is True


def test_is_current_env_false_for_different_interpreter(tmp_path: Path) -> None:
    other_python = tmp_path / "other_python"
    other_python.touch()
    env = TargetEnv(python=other_python, site_packages=())
    assert is_current_env(env) is False


def test_is_current_env_resolves_symlinks(tmp_path: Path) -> None:
    # A symlink that resolves to the current executable should return True
    link = tmp_path / "python_link"
    link.symlink_to(sys.executable)
    env = TargetEnv(python=link, site_packages=())
    assert is_current_env(env) is True
