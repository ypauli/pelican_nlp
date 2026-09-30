"""Write ``derivatives/pelican_runtime.json`` after a successful outermost run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pelican_nlp._version import __version__
from pelican_nlp.config_defaults import canonical_asr_model_id

RUNTIME_FILENAME = "pelican_runtime.json"
NESTED_PHASE_ENV = "PELICAN_NESTED_PHASE"
_NESTED_TRUE = {"1", "true", "yes", "on"}


def is_nested_phase() -> bool:
    """True in the text-from-transcriptions process spawned by an audio run."""
    return os.environ.get(NESTED_PHASE_ENV, "").strip().lower() in _NESTED_TRUE


def clear_runtime_file(output_directory) -> None:
    """Remove a previous runtime file so a failed run cannot look successful."""
    path = Path(output_directory) / RUNTIME_FILENAME
    try:
        path.unlink()
    except FileNotFoundError:
        return


def write_runtime_file(output_directory, record: dict) -> Path:
    """Atomically replace ``pelican_runtime.json`` in ``output_directory``."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / RUNTIME_FILENAME
    fd, tmp_name = tempfile.mkstemp(
        prefix=".pelican_runtime.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def config_file_sha256(config_path) -> str:
    """SHA-256 of the configuration file bytes, not the defaults-filled dict."""
    digest = hashlib.sha256()
    with Path(config_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_models(config: dict) -> dict:
    """Model ids that this config actually runs, in a stable order."""
    config = config or {}
    metrics = [str(name) for name in (config.get("metrics_to_extract") or [])]
    models = {}

    if config.get("input_file") == "audio" and config.get("transcription"):
        transcription = config.get("transcription")
        raw = transcription.get("transcription_model") if isinstance(transcription, dict) else None
        models["transcription"] = canonical_asr_model_id(raw if isinstance(raw, str) else None)

    if any(name in metrics for name in ("embeddings", "topic_modeling")):
        name = _model_name((config.get("options_embeddings") or {}))
        if name:
            models["embeddings"] = name

    if any(name in metrics for name in ("logits", "perplexity")):
        name = _model_name((config.get("options_logits") or {}))
        if name:
            models["logits"] = name

    return models


def build_runtime_record(
    *,
    config: dict,
    config_path,
    started_at: datetime,
    finished_at: datetime,
    n_documents: int,
    n_units: int,
    device: str,
) -> dict:
    """Flat record of one completed run. No config dump and no absolute paths."""
    started = _as_utc(started_at)
    finished = _as_utc(finished_at)
    duration = round((finished - started).total_seconds(), 1)
    metrics = [str(name) for name in ((config or {}).get("metrics_to_extract") or [])]
    return {
        "pelican_nlp": __version__,
        "python": platform.python_version(),
        "config_file": Path(config_path).name,
        "config_sha256": config_file_sha256(config_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": duration,
        "status": "completed",
        "input_file": _plain((config or {}).get("input_file")),
        "language": _plain((config or {}).get("language")),
        "task_name": _plain((config or {}).get("task_name")),
        "documents": int(n_documents),
        "units": int(n_units),
        "metrics": metrics,
        "models": runtime_models(config),
        "device": device,
    }


def current_runtime_device() -> str:
    """``cuda``, ``mps``, or ``cpu`` for the process that is writing the record."""
    from pelican_nlp.utils.gpu_budget import runtime_torch_device

    return str(runtime_torch_device().type)


def _model_name(options) -> str | None:
    if not isinstance(options, dict):
        return None
    return _plain(options.get("model_name"))


def _plain(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
