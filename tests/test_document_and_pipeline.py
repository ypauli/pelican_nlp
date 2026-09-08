from pathlib import Path

from pelican_nlp.core.document import Document
from pelican_nlp.preprocessing.pipeline import TextPreprocessingPipeline
from pelican_nlp.preprocessing.text_cleaner import TextCleaner
from pelican_nlp.utils.setup_functions import load_config


def _write_doc(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_document_loads_and_detects_single_section(tmp_path):
    _write_doc(tmp_path, "part-01_task-fluency_text.txt", "Ameise; Affe; Baer")
    document = Document(
        file_path=str(tmp_path),
        name="part-01_task-fluency_text.txt",
        task="fluency",
        has_sections=False,
    )
    document.detect_sections()
    assert len(document.sections) == 1
    section_text = next(iter(document.sections.values()))
    assert "Ameise" in section_text


def test_document_clean_uses_cleaner_not_dead_fluency_flag(tmp_path):
    _write_doc(tmp_path, "part-01_task-fluency_text.txt", "Hello World")
    document = Document(
        file_path=str(tmp_path),
        name="part-01_task-fluency_text.txt",
        task="fluency",
        fluency=True,
        has_sections=False,
    )
    document.detect_sections()
    cleaner = TextCleaner(
        {
            "remove_timestamps": False,
            "lowercase": True,
            "general_cleaning": True,
            "fluency_task": False,
        }
    )
    document.clean_text(cleaner)
    cleaned = next(iter(document.cleaned_sections.values()))
    assert cleaned == "hello world"


def test_pipeline_tokenize_whitespace_does_not_crash(tmp_path):
    _write_doc(tmp_path, "part-01_task-fluency_text.txt", "one two three")
    document = Document(
        file_path=str(tmp_path),
        name="part-01_task-fluency_text.txt",
        task="fluency",
        has_sections=False,
    )
    document.detect_sections()
    pipeline = TextPreprocessingPipeline(
        {
            "pipeline_options": {
                "clean_text": True,
                "tokenize_text": True,
                "normalize_text": False,
            },
            "cleaning_options": {
                "remove_timestamps": False,
                "lowercase": True,
                "general_cleaning": True,
                "fluency_task": False,
            },
            "tokenization_options": {
                "method": "whitespace",
                "purpose": "embeddings",
            },
        }
    )
    pipeline.process_document(document)
    assert document.tokens_embeddings
    assert document.tokens_embeddings[0] == ["one", "two", "three"]


def test_load_config_applies_defaults(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("task_name: fluency\ninput_file: text\n", encoding="utf-8")
    config = load_config(str(config_path))
    assert config["task_name"] == "fluency"
    assert config["opensmile_feature_extraction"] is False
    assert config["options_embeddings"]["distance-from-randomness"] is False
