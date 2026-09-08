"""Locate test and example assets for both editable checkouts and wheels."""

from __future__ import annotations

from pathlib import Path

BUNDLED_DIRNAME = "_bundled"


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def import_root() -> Path:
    """Directory that should be on ``PYTHONPATH`` so ``import pelican_nlp`` works.

    In an editable checkout this is the repo root. After a wheel install it is
    ``site-packages``.
    """
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path | None:
    """Repo root when tests/ and examples/ live next to the package source."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "tests").is_dir():
        return candidate
    return None


def pytest_root() -> Path:
    """Directory that contains ``pytest.ini`` and ``tests/``."""
    root = repo_root()
    if root is not None:
        return root
    bundled = package_dir() / BUNDLED_DIRNAME
    if (bundled / "tests").is_dir():
        return bundled
    raise FileNotFoundError(
        "Could not find Pelican tests. Reinstall from the project source "
        "or from a wheel that includes bundled tests."
    )


def tests_root() -> Path:
    return pytest_root() / "tests"


def examples_root() -> Path:
    root = repo_root()
    if root is not None:
        return root / "examples"
    bundled = package_dir() / BUNDLED_DIRNAME / "examples"
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        "Could not find Pelican example projects. Reinstall from the project "
        "source or from a wheel that includes bundled examples."
    )
