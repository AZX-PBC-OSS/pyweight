from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_package_path(fixtures_path: Path) -> Path:
    return fixtures_path / "sample_package"


@pytest.fixture
def namespace_package_path(fixtures_path: Path) -> Path:
    return fixtures_path / "namespace_package"
