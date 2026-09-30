"""Runtime metadata written to derivatives/pelican_runtime.json."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pelican_nlp._version import __version__
from pelican_nlp.main import Pelican
from pelican_nlp.testing.golden import compare_derivative_trees
from pelican_nlp.testing.runner import _replace_tree
from pelican_nlp.utils.runtime_metadata import (
    NESTED_PHASE_ENV,
    RUNTIME_FILENAME,
    build_runtime_record,
    clear_runtime_file,
    runtime_models,
    write_runtime_file,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project(tmp_path: Path) -> Path:
    config = tmp_path / "config_demo.yml"
    config.write_text(
        "\n".join(
            [
                'input_file: "text"',
                'language: "german"',
                "task_name: null",
                "metrics_to_extract: []",
                "pipeline_options:",
                "  quality_check: false",
                "  clean_text: false",
                "  tokenize_text: false",
                "  normalize_text: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write(tmp_path / "participants" / "stories" / "hansel.txt", "Once upon a time.")
    return config


def _record(tmp_path: Path, config: dict, *, name: str = "config_demo.yml") -> dict:
    path = tmp_path / name
    path.write_text("input_file: text\n", encoding="utf-8")
    started = datetime(2026, 9, 30, 14, 23, 1, tzinfo=timezone.utc)
    finished = datetime(2026, 9, 30, 14, 41, 12, tzinfo=timezone.utc)
    return build_runtime_record(
        config=config,
        config_path=path,
        started_at=started,
        finished_at=finished,
        n_documents=12,
        n_units=4,
        device="cpu",
    )


def test_record_keeps_version_config_name_and_hash(tmp_path):
    config_path = tmp_path / "config_interview.yml"
    payload = b'input_file: "audio"\nlanguage: "german"\n'
    config_path.write_bytes(payload)
    record = build_runtime_record(
        config={
            "input_file": "audio",
            "language": "german",
            "task_name": "interview",
            "metrics_to_extract": ["embeddings", "logits"],
            "transcription": {"transcription_model": "whisper-medium"},
            "options_embeddings": {"model_name": "xlm-roberta-base"},
            "options_logits": {"model_name": "gpt2"},
        },
        config_path=config_path,
        started_at=datetime(2026, 9, 30, 14, 23, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 30, 14, 41, 12, tzinfo=timezone.utc),
        n_documents=12,
        n_units=4,
        device="cuda",
    )
    assert record["pelican_nlp"] == __version__
    assert record["config_file"] == "config_interview.yml"
    assert record["config_sha256"] == hashlib.sha256(payload).hexdigest()
    assert str(config_path.parent) not in json.dumps(record)
    assert record["started_at"] == "2026-09-30T14:23:01+00:00"
    assert record["finished_at"] == "2026-09-30T14:41:12+00:00"
    assert record["duration_seconds"] == 1091.0
    assert record["status"] == "completed"
    assert record["documents"] == 12
    assert record["units"] == 4
    assert record["metrics"] == ["embeddings", "logits"]
    assert record["models"] == {
        "transcription": "openai/whisper-medium",
        "embeddings": "xlm-roberta-base",
        "logits": "gpt2",
    }
    assert record["device"] == "cuda"


def test_models_only_include_steps_this_config_runs(tmp_path):
    text = _record(
        tmp_path,
        {
            "input_file": "text",
            "language": "  ",
            "task_name": None,
            "metrics_to_extract": ["perplexity"],
            "transcription": {"transcription_model": "whisper-medium"},
            "options_embeddings": {"model_name": "fastText"},
            "options_logits": {"model_name": "gpt2"},
        },
    )
    assert text["language"] is None
    assert text["task_name"] is None
    assert text["models"] == {"logits": "gpt2"}

    topics = runtime_models(
        {
            "input_file": "text",
            "metrics_to_extract": ["topic_modeling"],
            "options_embeddings": {"model_name": "fastText"},
        }
    )
    assert topics == {"embeddings": "fastText"}

    blank = runtime_models(
        {
            "input_file": "audio",
            "metrics_to_extract": ["embeddings"],
            "options_embeddings": {"model_name": "  "},
        }
    )
    assert blank == {}


def test_write_replaces_the_runtime_file(tmp_path):
    target = tmp_path / RUNTIME_FILENAME
    target.write_text('{"status": "old"}\n', encoding="utf-8")
    written = write_runtime_file(tmp_path, {"status": "completed", "pelican_nlp": "0.4.7"})
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "completed"
    assert list(tmp_path.glob(".pelican_runtime.*.tmp")) == []

    clear_runtime_file(tmp_path)
    assert not target.exists()
    clear_runtime_file(tmp_path)


def test_runtime_json_is_ignored_by_golden_comparison(tmp_path):
    golden = tmp_path / "golden"
    actual = tmp_path / "actual"
    _write(golden / "keep" / "a.txt", "hello\n")
    _write(actual / "keep" / "a.txt", "hello\n")
    _write(actual / RUNTIME_FILENAME, '{"status": "completed"}\n')
    assert compare_derivative_trees(actual, golden) == []

    _write(actual / "keep" / "a.txt", "changed\n")
    messages = compare_derivative_trees(actual, golden)
    assert any("a.txt" in message for message in messages)
    assert all(RUNTIME_FILENAME not in message for message in messages)


def test_golden_update_does_not_freeze_runtime_file(tmp_path):
    source = tmp_path / "actual"
    _write(source / "embeddings" / "a.csv", "Token\nhello\n")
    _write(source / RUNTIME_FILENAME, "{}\n")
    destination = tmp_path / "golden"
    _replace_tree(destination, source)
    assert (destination / "embeddings" / "a.csv").is_file()
    assert not (destination / RUNTIME_FILENAME).exists()


def test_successful_run_writes_runtime_file(tmp_path):
    config = _project(tmp_path)
    Pelican(str(config)).run()
    path = tmp_path / "derivatives" / RUNTIME_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["pelican_nlp"] == __version__
    assert record["config_file"] == "config_demo.yml"
    assert record["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert record["status"] == "completed"
    assert record["input_file"] == "text"
    assert record["language"] == "german"
    assert record["documents"] == 1
    assert record["units"] == 1
    assert record["metrics"] == []
    assert record["models"] == {}
    assert record["device"] in {"cpu", "cuda", "mps"}
    assert list((tmp_path / "derivatives").glob(".pelican_runtime.*.tmp")) == []


def test_second_successful_run_replaces_stale_runtime_file(tmp_path):
    config = _project(tmp_path)
    stale = tmp_path / "derivatives" / RUNTIME_FILENAME
    _write(stale, '{"status": "stale"}\n')
    Pelican(str(config)).run()
    record = json.loads(stale.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["config_file"] == "config_demo.yml"


def test_failed_run_removes_stale_runtime_file(tmp_path, monkeypatch):
    config = _project(tmp_path)
    stale = tmp_path / "derivatives" / RUNTIME_FILENAME
    _write(stale, '{"status": "stale"}\n')

    def boom(self, corpus_entity, documents):
        raise RuntimeError("stop")

    monkeypatch.setattr(Pelican, "_run_on_documents", boom)
    with pytest.raises(RuntimeError, match="stop"):
        Pelican(str(config)).run()
    assert not stale.exists()


def test_nested_phase_does_not_touch_runtime_file(tmp_path, monkeypatch):
    config = _project(tmp_path)
    stale = tmp_path / "derivatives" / RUNTIME_FILENAME
    _write(stale, '{"status": "parent"}\n')
    monkeypatch.setenv(NESTED_PHASE_ENV, "1")
    Pelican(str(config)).run()
    assert stale.read_text(encoding="utf-8") == '{"status": "parent"}\n'


def test_text_phase_subprocess_is_marked_nested(tmp_path, monkeypatch):
    monkeypatch.delenv(NESTED_PHASE_ENV, raising=False)
    config = _project(tmp_path)
    pelican = Pelican(str(config))
    pelican.config["input_file"] = "audio"
    pelican.config["transcription"] = {"transcription_model": None}
    pelican.config["metrics_to_extract"] = ["embeddings"]
    seen = {}

    def fake_run(cmd, env=None, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = env

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("pelican_nlp.main.subprocess.run", fake_run)

    class Corpus:
        def transcribe_audio(self, skip_existing=True):
            return None

    pelican._process_audio_corpus(Corpus(), "stories")
    assert seen["env"][NESTED_PHASE_ENV] == "1"
    assert "--text-from-transcriptions" in seen["cmd"]
    assert NESTED_PHASE_ENV not in __import__("os").environ
