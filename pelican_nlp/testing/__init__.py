"""Helpers for example-project regression tests.

The pipeline is generic: point it at any directory of Pelican projects
(YAML + ``participants/`` + optional golden ``derivatives/``), select a
subset, run each project in a temp copy, and compare outputs to goldens.
"""

from .example_projects import (
    CONFIG_ONLY_EXAMPLES,
    ExampleProject,
    ExampleSettings,
    NO_GOLDENS_SKIP,
    discover_example_projects,
    load_example_settings,
    select_example_projects,
)
from .golden import compare_derivative_trees
from .paths import examples_root, import_root, pytest_root, tests_root
from .runner import run_example_golden

__all__ = [
    "CONFIG_ONLY_EXAMPLES",
    "ExampleProject",
    "ExampleSettings",
    "NO_GOLDENS_SKIP",
    "compare_derivative_trees",
    "discover_example_projects",
    "examples_root",
    "import_root",
    "load_example_settings",
    "pytest_root",
    "run_example_golden",
    "select_example_projects",
    "tests_root",
]
