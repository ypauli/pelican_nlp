"""Command-line entry for the Pelican pipeline and packaged tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pelican_nlp.utils.setup_functions import resolve_project_config


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split_at = argv.index("--")
        extra = argv[split_at + 1 :]
        argv = argv[:split_at]

    parser = argparse.ArgumentParser(
        prog="pelican-run",
        description="Run a Pelican project pipeline, or the packaged test suite.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run packaged unit tests instead of the project pipeline.",
    )
    parser.add_argument(
        "--examples",
        nargs="?",
        const="all",
        default=None,
        metavar="SELECTION",
        help=(
            "With --run-tests, also run example golden tests. "
            "Optional comma-separated names, or omit for all."
        ),
    )
    parser.add_argument(
        "--example-root",
        default=None,
        help="Directory of example projects (default: packaged examples).",
    )
    parser.add_argument(
        "--update-goldens",
        action="store_true",
        help="With --run-tests --examples, replace frozen derivatives/.",
    )
    parser.add_argument(
        "--example-logs",
        action="store_true",
        help="With --run-tests --examples, print full pipeline logs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug details and third-party progress (Hugging Face, tqdm).",
    )
    args = parser.parse_args(argv)

    if args.examples is not None and not args.run_tests:
        parser.error("--examples requires --run-tests")
    if args.update_goldens and not args.run_tests:
        parser.error("--update-goldens requires --run-tests")
    if args.example_root is not None and not args.run_tests:
        parser.error("--example-root requires --run-tests")
    if args.example_logs and not args.run_tests:
        parser.error("--example-logs requires --run-tests")
    if extra and not args.run_tests:
        parser.error("extra pytest arguments after -- require --run-tests")

    if args.run_tests:
        from pelican_nlp.testing.cli import run_pytest_suite

        raise SystemExit(
            run_pytest_suite(
                examples=args.examples,
                example_root=args.example_root,
                update_goldens=args.update_goldens,
                example_logs=args.example_logs,
                pytest_args=extra,
            )
        )

    _run_pipeline(verbose=args.verbose)


def _run_pipeline(verbose: bool = False) -> None:
    from pelican_nlp.main import Pelican

    try:
        config_file = str(resolve_project_config(Path.cwd()))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    try:
        pelican = Pelican(config_file, verbose=verbose)
        pelican.run()
    except Exception as e:
        print(f"Error: {e}")
        return
