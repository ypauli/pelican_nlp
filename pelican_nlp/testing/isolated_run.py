"""Child process entry: run one example without nesting another subprocess."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from pelican_nlp.testing.runner import GoldenRegressionError, run_example_golden


def _write_error_file(payload: dict, text: str) -> None:
    error_file = payload.get("error_file")
    if not error_file:
        return
    Path(error_file).write_text(text, encoding="utf-8")


def _system_exit_message(error: SystemExit) -> str:
    code = error.code
    if isinstance(code, int):
        return f"sys.exit({code})"
    if code is None:
        return "sys.exit()"
    return str(code)


def main(argv=None) -> int:
    payload: dict = {}
    try:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(line_buffering=True)
                sys.stderr.reconfigure(line_buffering=True)
            except Exception:
                pass
        raw = sys.stdin.read() if argv is None else None
        if argv:
            payload = json.loads(argv[0]) if argv else {}
        else:
            payload = json.loads(raw or "{}")

        example_dir = Path(payload["example_dir"])
        update_goldens = bool(payload.get("update_goldens", False))
        work_dir = payload.get("work_dir")
        work_path = Path(work_dir) if work_dir else None

        from pelican_nlp.testing.example_projects import inspect_example_project

        project = inspect_example_project(example_dir)
        print(f"Starting isolated run: {project.name}", flush=True)
        run_example_golden(
            project,
            update_goldens=update_goldens,
            work_dir=work_path,
            isolate=False,
        )
    except GoldenRegressionError as error:
        message = str(error)
        _write_error_file(payload, message)
        print(message, file=sys.stderr, flush=True)
        return 1
    except SystemExit as error:
        if error.code in (0, None):
            return 0
        message = _system_exit_message(error)
        _write_error_file(payload, message)
        print(message, file=sys.stderr, flush=True)
        return error.code if isinstance(error.code, int) else 1
    except Exception:
        message = traceback.format_exc()
        _write_error_file(payload, message)
        print(message, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
