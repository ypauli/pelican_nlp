import subprocess
import sys


def _assert_isolated(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def test_model_registry_import_does_not_load_accelerate():
    sys.modules.pop("pelican_nlp.extraction.language_model", None)
    sys.modules.pop("accelerate", None)

    from pelican_nlp.extraction.model_registry import resolve_model_kind

    assert resolve_model_kind("fastText") == "static"
    assert "pelican_nlp.extraction.language_model" not in sys.modules
    assert "accelerate" not in sys.modules


def test_cli_import_does_not_load_main_or_torch():
    _assert_isolated(
        "import pelican_nlp.cli\n"
        "import sys\n"
        "assert 'pelican_nlp.main' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
    )


def test_preprocessing_lpds_import_does_not_load_tokenizer_or_torch():
    _assert_isolated(
        "from pelican_nlp.preprocessing import LPDS\n"
        "import sys\n"
        "assert LPDS.__name__ == 'LPDS'\n"
        "assert 'pelican_nlp.preprocessing.text_tokenizer' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
    )


def test_transcription_import_does_not_load_embeddings_stack():
    _assert_isolated(
        "import pelican_nlp.preprocessing.transcription\n"
        "import sys\n"
        "assert 'pelican_nlp.extraction.extract_embeddings' not in sys.modules\n"
        "assert 'pelican_nlp.extraction.language_model' not in sys.modules\n"
        "assert 'pelican_nlp.extraction.extract_logits' not in sys.modules\n"
        "assert 'fasttext' not in sys.modules\n"
    )


def test_whitespace_tokenizer_does_not_import_torch():
    _assert_isolated(
        "from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer\n"
        "import sys\n"
        "tokenizer = TextTokenizer('whitespace')\n"
        "assert tokenizer.tokenize_text('one two') == ['one', 'two']\n"
        "assert 'torch' not in sys.modules\n"
    )


def test_package_import_does_not_load_pelican_class():
    _assert_isolated(
        "import pelican_nlp\n"
        "import sys\n"
        "assert pelican_nlp.__version__\n"
        "assert 'pelican_nlp.main' not in sys.modules\n"
    )
