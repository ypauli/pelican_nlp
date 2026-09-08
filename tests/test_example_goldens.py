"""Opt-in end-to-end golden tests for every example project under a root.

Default pytest deselects this module. Run with:

    pytest --run-examples --examples perplexity
    pelican-run --run-tests --examples perplexity
    pelican-test-examples --examples perplexity

Example runs print a short progress summary (example N of M, pipeline stage,
elapsed time). Use ``--example-logs`` for the full pipeline dump:

    pelican-run --run-tests --examples --example-logs
"""

from pathlib import Path

import pytest

from pelican_nlp.testing import (
    CONFIG_ONLY_EXAMPLES,
    NO_GOLDENS_SKIP,
    discover_example_projects,
    run_example_golden,
    select_example_projects,
)

pytestmark = pytest.mark.example


def pytest_generate_tests(metafunc):
    if "example_project" not in metafunc.fixturenames:
        return
    root = Path(metafunc.config.getoption("--example-root"))
    projects = discover_example_projects(root)
    selected = [
        project
        for project in select_example_projects(
            projects, metafunc.config.getoption("--examples")
        )
        if project.name not in CONFIG_ONLY_EXAMPLES
    ]
    if not selected:
        metafunc.parametrize(
            "example_project",
            [
                pytest.param(
                    None,
                    marks=pytest.mark.skip(
                        reason="config-only examples are not golden-tested"
                    ),
                )
            ],
            ids=["none"],
        )
        return
    metafunc.parametrize(
        "example_project",
        selected,
        ids=[project.name for project in selected],
    )


def test_example_matches_golden(example_project, pytestconfig):
    update = pytestconfig.getoption("--update-goldens")
    if example_project.skip_reason:
        if not (update and example_project.skip_reason == NO_GOLDENS_SKIP):
            hint = ""
            if example_project.skip_reason == NO_GOLDENS_SKIP:
                hint = (
                    " Pass --update-goldens to run it once and freeze derivatives/."
                )
            pytest.skip(example_project.skip_reason + hint)
    run_example_golden(
        example_project,
        update_goldens=update,
    )
