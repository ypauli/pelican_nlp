"""Run packaged Pelican tests (unit tests and optional example goldens)."""

from __future__ import annotations

import sys
from pathlib import Path


def run_pytest_suite(
    *,
    examples: str | None = None,
    example_root: str | Path | None = None,
    update_goldens: bool = False,
    example_logs: bool = False,
    pytest_args: list[str] | None = None,
) -> int:
    """Invoke pytest on the bundled or checkout test tree. Return the exit code."""
    try:
        import pytest
    except ImportError as error:
        raise SystemExit(
            "pytest is required. Install with: pip install 'pelican_nlp[dev]'"
        ) from error

    from .paths import examples_root, pytest_root

    root = pytest_root()
    tests_dir = root / "tests"
    command = [str(tests_dir), f"--rootdir={root}"]
    ini = root / "pytest.ini"
    if ini.is_file():
        command.extend(["-c", str(ini)])

    if examples is not None:
        command.append("--run-examples")
        command.append(f"--examples={examples}")
        resolved_examples = example_root or examples_root()
        command.append(f"--example-root={resolved_examples}")
        if update_goldens:
            command.append("--update-goldens")
        if example_logs:
            command.append("--example-logs")
    elif example_root:
        command.append(f"--example-root={example_root}")

    extra = list(pytest_args or [])
    if examples is not None and not any(
        item == "-s" or item.startswith("--capture") for item in extra
    ):
        command.append("-s")
    command.extend(extra)
    return pytest.main(command)


def main(argv=None) -> None:
    """Backward-compatible alias: run unit tests plus example goldens."""
    argv = list(sys.argv[1:] if argv is None else argv)
    forwarded = ["--run-tests"]
    has_examples = any(
        item == "--examples" or item.startswith("--examples=") for item in argv
    )
    if not has_examples:
        forwarded.append("--examples")
    forwarded.extend(argv)
    from pelican_nlp.cli import main as pelican_main

    pelican_main(forwarded)


if __name__ == "__main__":
    main()
