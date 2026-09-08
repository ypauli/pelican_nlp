"""Copy tests/ and examples/ into the package when building a wheel."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py

_SKIP_EXAMPLE_FOLDERS = frozenset({"example_acoustic-features", "example_general"})
_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.py[cod]",
    ".pytest_cache",
    ".DS_Store",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def copy_bundled_assets(project_root: Path | None = None) -> Path:
    """Copy pytest assets into ``pelican_nlp/testing/_bundled/``. Return dest."""
    project_root = project_root or _project_root()
    dest = project_root / "pelican_nlp" / "testing" / "_bundled"
    tests_src = project_root / "tests"
    examples_src = project_root / "examples"
    pytest_ini = project_root / "pytest.ini"
    if not tests_src.is_dir() or not examples_src.is_dir():
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if pytest_ini.is_file():
        shutil.copy2(pytest_ini, dest / "pytest.ini")
    shutil.copytree(tests_src, dest / "tests", ignore=_IGNORE)

    examples_dst = dest / "examples"
    examples_dst.mkdir()
    for child in sorted(examples_src.iterdir()):
        if not child.is_dir() or child.name in _SKIP_EXAMPLE_FOLDERS:
            continue
        if child.name.startswith("."):
            continue
        shutil.copytree(child, examples_dst / child.name, ignore=_IGNORE)
    return dest


class build_py(_build_py):
    def run(self):
        if not getattr(self, "editable_mode", False):
            copy_bundled_assets(_project_root())
        super().run()
