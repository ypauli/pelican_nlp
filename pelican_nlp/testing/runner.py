"""Run one example project in a temp directory and compare to goldens."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .example_projects import NO_GOLDENS_SKIP, ExampleProject
from .golden import compare_derivative_trees
from .paths import import_root

_LOG_TAIL_CHARS = 8000
_HEARTBEAT_SECONDS = 30
EXAMPLE_LOGS_ENV = "PELICAN_EXAMPLE_LOGS"
_QUIET_CHILD_ENV = {
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}
_PROGRESS_MARKERS = (
    "Starting isolated run",
    "Instantiating all participants",
    "Processing corpus:",
    "No documents for corpus",
    "Extracting ",
    "Logits [",
    "Embeddings [",
    "Perplexity [",
    "Pipeline ran successfully",
    "GPU memory cleared",
    "Loading checkpoint",
    "Downloading",
    "Skipping output directory",
    "Text-from-transcriptions",
    "Fitting BERTopic",
    "Performing per-document topic",
    "Performing corpus-level topic",
)


class GoldenRegressionError(AssertionError):
    """Raised when an example run does not match its frozen derivatives."""


def example_logs_verbose() -> bool:
    return os.environ.get(EXAMPLE_LOGS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def is_progress_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(marker in stripped for marker in _PROGRESS_MARKERS)


def run_example_golden(
    project: ExampleProject,
    *,
    update_goldens: bool = False,
    work_dir: Path | None = None,
    isolate: bool = True,
) -> Path:
    """Copy the example, run Pelican, compare ``derivatives/`` to the golden.

    By default each example runs in a child process so GPU memory is returned
    to the OS before the next example starts. Pass ``isolate=False`` only from
    that child (or from tests that mock Pelican).
    """
    bootstrapping = update_goldens and project.skip_reason == NO_GOLDENS_SKIP
    if project.skip_reason and not bootstrapping:
        raise GoldenRegressionError(
            f"{project.name} is not a complete example: {project.skip_reason}"
        )
    if project.config_path is None:
        raise GoldenRegressionError(f"{project.name} has no YAML config")

    if isolate:
        return _run_in_subprocess(project, update_goldens=update_goldens, work_dir=work_dir)

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix=f"pelican-{project.name}-") as tmp:
            return _run_in(project, Path(tmp), update_goldens=update_goldens)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return _run_in(project, work_dir, update_goldens=update_goldens)


def _run_in_subprocess(
    project: ExampleProject,
    *,
    update_goldens: bool,
    work_dir: Path | None,
) -> Path:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(import_root()), existing) if part
    )
    env["PYTHONUNBUFFERED"] = "1"
    error_path = Path(
        tempfile.NamedTemporaryFile(
            prefix=f"pelican-{project.name}-isolated-",
            suffix=".err",
            delete=False,
        ).name
    )
    log_path = Path(
        tempfile.NamedTemporaryFile(
            prefix=f"pelican-{project.name}-isolated-",
            suffix=".log",
            delete=False,
        ).name
    )
    payload = {
        "example_dir": str(project.path),
        "update_goldens": bool(update_goldens),
        "work_dir": str(work_dir) if work_dir is not None else None,
        "error_file": str(error_path),
    }
    print(
        f"  {project.name}  isolated process (GPU memory is released after this example)",
        flush=True,
    )
    started = time.monotonic()
    try:
        returncode = _stream_isolated_run(payload, env, log_path)
        elapsed = format_duration(time.monotonic() - started)
        if returncode != 0:
            print(f"  {project.name}  FAILED ({elapsed})", flush=True)
            raise GoldenRegressionError(
                _isolated_failure_message(
                    project.name, returncode, error_path, log_path
                )
            )
        print(f"  {project.name}  passed ({elapsed})", flush=True)
    finally:
        error_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
    if work_dir is not None:
        return Path(work_dir) / "derivatives"
    return project.path / "derivatives"


def _stream_isolated_run(payload: dict, env: dict, log_path: Path) -> int:
    """Run the child, keep a full log, and print only progress by default."""
    child_env = env.copy()
    verbose = example_logs_verbose()
    if not verbose:
        for key, value in _QUIET_CHILD_ENV.items():
            child_env.setdefault(key, value)

    process = subprocess.Popen(
        [sys.executable, "-m", "pelican_nlp.testing.isolated_run"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_env,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(payload))
    process.stdin.close()

    state = {
        "last_print": time.monotonic(),
        "last_seen": "",
        "start": time.monotonic(),
    }
    stop = threading.Event()

    def _heartbeat() -> None:
        while not stop.wait(_HEARTBEAT_SECONDS):
            if process.poll() is not None:
                return
            if time.monotonic() - state["last_print"] < _HEARTBEAT_SECONDS:
                continue
            elapsed = format_duration(time.monotonic() - state["start"])
            last = state["last_seen"] or "no pipeline line yet (model load is often quiet)"
            print(f"  still running ({elapsed}); last: {last[:120]}", flush=True)
            state["last_print"] = time.monotonic()

    def _pump() -> None:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                stripped = line.strip()
                if stripped:
                    state["last_seen"] = stripped
                if verbose or is_progress_line(line):
                    sys.stdout.write(line if line.endswith("\n") else line + "\n")
                    sys.stdout.flush()
                    state["last_print"] = time.monotonic()

    pump = threading.Thread(target=_pump, name="pelican-isolated-log", daemon=True)
    beat = threading.Thread(target=_heartbeat, name="pelican-isolated-heartbeat", daemon=True)
    pump.start()
    beat.start()
    returncode = process.wait()
    stop.set()
    pump.join(timeout=5)
    beat.join(timeout=1)
    return returncode


def _isolated_failure_message(
    name: str,
    returncode: int,
    error_path: Path,
    log_path: Path,
) -> str:
    parts = [
        f"{name} failed in an isolated process (exit {returncode})."
    ]
    if error_path.exists():
        detail = error_path.read_text(encoding="utf-8").strip()
        if detail:
            parts.append(detail)
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8").strip()
        if log_text:
            tail = log_text[-_LOG_TAIL_CHARS:]
            if len(log_text) > _LOG_TAIL_CHARS:
                tail = "...(log truncated)...\n" + tail
            if not (len(parts) > 1 and parts[1] in tail):
                parts.append("Isolated-run log (tail):\n" + tail)
    if len(parts) == 1:
        parts.append(
            "The child process left no traceback. It may have been killed "
            "(out of memory) or exited via sys.exit()."
        )
    return "\n".join(parts)


def _run_in(project: ExampleProject, work_dir: Path, update_goldens: bool) -> Path:
    config_path = _stage_example(project, work_dir)
    from pelican_nlp.main import Pelican

    Pelican(str(config_path)).run()

    actual_dir = work_dir / "derivatives"
    golden_dir = project.path / "derivatives"
    if update_goldens:
        _replace_tree(golden_dir, actual_dir)
        return actual_dir

    mismatches = compare_derivative_trees(actual_dir, golden_dir, project.settings)
    if mismatches:
        details = "\n".join(mismatches[:50])
        raise GoldenRegressionError(
            f"{project.name} derivatives differ from golden:\n{details}"
        )
    return actual_dir


def _stage_example(project: ExampleProject, work_dir: Path) -> Path:
    """Copy YAML and participants/ only — never the golden derivatives."""
    staged_config = work_dir / project.config_path.name
    shutil.copy2(project.config_path, staged_config)
    source_participants = project.path / "participants"
    target_participants = work_dir / "participants"
    if target_participants.exists():
        shutil.rmtree(target_participants)
    shutil.copytree(
        source_participants,
        target_participants,
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"),
    )
    return staged_config


def _replace_tree(destination: Path, source: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination
