import os
from io import StringIO
from types import SimpleNamespace

import pelican_nlp.config as cfg
from pelican_nlp.utils.progress import (
    NullReporter,
    PipelineReporter,
    UnitTracker,
    apply_verbosity,
    documents_by_unit,
    pipeline_stage_labels,
    short_model_name,
    walk_units,
)


def test_debug_mode_defaults_quiet(monkeypatch):
    monkeypatch.setattr(cfg, "DEBUG_MODE", False)
    monkeypatch.delenv("PELICAN_DEBUG", raising=False)
    monkeypatch.delenv("PELICAN_VERBOSE", raising=False)
    assert cfg.debug_enabled() is False


def test_debug_enabled_from_env(monkeypatch):
    monkeypatch.setattr(cfg, "DEBUG_MODE", False)
    monkeypatch.setenv("PELICAN_DEBUG", "1")
    assert cfg.debug_enabled() is True


def test_apply_verbosity_sets_library_env(monkeypatch):
    monkeypatch.setattr(cfg, "DEBUG_MODE", True)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    assert apply_verbosity(False) is False
    assert cfg.DEBUG_MODE is False
    assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
    assert os.environ.get("TRANSFORMERS_VERBOSITY") == "error"
    assert apply_verbosity(True) is True
    assert cfg.DEBUG_MODE is True


def test_pipeline_stage_labels_text_and_audio():
    text = {
        "input_file": "text",
        "metrics_to_extract": ["embeddings", "logits"],
        "options_embeddings": {"model_name": "fastText"},
        "options_logits": {"model_name": "DiscoResearch/Llama3-German-8B-32k"},
        "create_aggregation_of_results": True,
    }
    assert pipeline_stage_labels(text) == [
        "preprocess",
        "embeddings (fastText)",
        "logits (Llama3-German-8B-32k)",
        "aggregation",
    ]
    audio = {
        "input_file": "audio",
        "transcription": {"hf_token": ""},
        "opensmile_feature_extraction": True,
        "metrics_to_extract": ["embeddings"],
    }
    assert pipeline_stage_labels(audio) == [
        "transcription",
        "opensmile",
        "text-from-transcriptions",
    ]
    assert pipeline_stage_labels(audio, text_from_transcriptions=True)[:1] == ["preprocess"]


def test_short_model_name():
    assert short_model_name("DiscoResearch/Llama3-German-8B-32k") == "Llama3-German-8B-32k"
    assert short_model_name("fastText") == "fastText"
    assert short_model_name(None) is None


def test_documents_by_unit_preserves_order():
    docs = [
        SimpleNamespace(source_folder="part-02", name="b.txt", lpds_entities={}),
        SimpleNamespace(source_folder="part-01", name="a.txt", lpds_entities={}),
        SimpleNamespace(source_folder="part-02", name="c.txt", lpds_entities={}),
    ]
    grouped = documents_by_unit(docs)
    assert list(grouped) == ["part-02", "part-01"]
    assert [doc.name for doc in grouped["part-02"]] == ["b.txt", "c.txt"]


def test_reporter_plain_lines_two_units():
    buf = StringIO()
    reporter = PipelineReporter(stream=buf, force_plain=True)
    reporter.print_summary(
        {"input_file": "text", "language": "german"},
        2,
        3,
        ["preprocess", "embeddings (fastText)"],
    )
    seen = []
    docs = [
        SimpleNamespace(source_folder="part-01", name="a.txt", lpds_entities={}),
        SimpleNamespace(source_folder="part-01", name="b.txt", lpds_entities={}),
        SimpleNamespace(source_folder="part-02", name="c.txt", lpds_entities={}),
    ]
    for unit, unit_docs in walk_units(reporter, docs, "embeddings", label="embeddings (fastText)"):
        seen.append(unit)
        for document in unit_docs:
            reporter.advance_item(document.name)
    reporter.close()
    text = buf.getvalue()
    assert "Pelican  text  german  2 participants / 3 documents" in text
    assert "Stages: preprocess → embeddings (fastText)" in text
    assert "Now: embeddings (fastText)" in text
    assert "embeddings  1/2  part-01" in text
    assert "embeddings  2/2  part-02" in text
    assert seen == ["part-01", "part-02"]


def test_unit_tracker_plain_lines():
    buf = StringIO()
    reporter = PipelineReporter(stream=buf, force_plain=True)
    docs = [
        SimpleNamespace(source_folder="part-01", name="a.txt", lpds_entities={}),
        SimpleNamespace(source_folder="part-02", name="b.txt", lpds_entities={}),
        SimpleNamespace(source_folder="part-01", name="c.txt", lpds_entities={}),
    ]
    tracker = UnitTracker(reporter, docs, "logits")
    for document in docs:
        tracker.finish_document(document)
    reporter.close()
    text = buf.getvalue()
    assert "Now: logits" in text
    assert "logits  1/2  part-02" in text
    assert "logits  2/2  part-01" in text


def test_null_reporter_is_silent_except_warnings(capsys):
    reporter = NullReporter()
    reporter.print_summary({"input_file": "text"}, 1, 1, ["preprocess"])
    reporter.start_stage("preprocess", ["part-01"])
    reporter.start_unit("part-01", 1)
    reporter.advance_item("a.txt")
    reporter.finish_unit()
    reporter.status("should not show")
    reporter.warn("disk full")
    captured = capsys.readouterr()
    assert "should not show" not in captured.err
    assert "Pelican" not in captured.err
    assert "Warning: disk full" in captured.err
