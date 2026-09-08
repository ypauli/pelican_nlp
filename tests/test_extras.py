import pytest

from pelican_nlp.extras import (
    extras_required_by_config,
    extra_install_hint,
    require_config_extras,
    require_extra,
)


def test_extras_required_by_transcription_config():
    assert extras_required_by_config({"transcription": {"hf_token": ""}}) == ["transcription"]


def test_extras_required_by_acoustic_flags():
    assert extras_required_by_config({"opensmile_feature_extraction": True}) == ["acoustic"]
    assert extras_required_by_config({"prosogram_extraction": True}) == ["acoustic"]


def test_extras_required_by_text_metrics():
    assert extras_required_by_config(
        {"metrics_to_extract": ["embeddings", "logits", "perplexity", "topic_modeling"]}
    ) == ["embeddings", "topic"]


def test_extras_required_by_normalize_text():
    assert extras_required_by_config(
        {"pipeline_options": {"normalize_text": True}}
    ) == ["nlp"]


def test_extras_required_empty_config():
    assert extras_required_by_config({}) == []
    assert extras_required_by_config(None) == []


def test_extra_install_hint():
    assert extra_install_hint(["transcription"]) == "pip install 'pelican_nlp[transcription]'"
    assert extra_install_hint(["transcription", "embeddings"]) == (
        "pip install 'pelican_nlp[transcription,embeddings]'"
    )


def test_require_extra_unknown():
    with pytest.raises(ValueError, match="Unknown extra"):
        require_extra("not-an-extra")


def test_require_extra_missing_markers(monkeypatch):
    monkeypatch.setattr("pelican_nlp.extras.importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError) as exc:
        require_extra("nlp")
    message = str(exc.value)
    assert "nlp" in message
    assert "pelican_nlp[nlp]" in message
    assert "spacy" in message


def test_require_config_extras_missing(monkeypatch):
    monkeypatch.setattr("pelican_nlp.extras.importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError) as exc:
        require_config_extras(
            {
                "transcription": {"hf_token": ""},
                "metrics_to_extract": ["embeddings"],
            }
        )
    message = str(exc.value)
    assert "transcription" in message
    assert "embeddings" in message
    assert "pelican_nlp[transcription,embeddings]" in message


def test_require_config_extras_complete_on_full_install():
    require_config_extras({"metrics_to_extract": ["embeddings"]})


def test_pelican_is_exported_lazily():
    from pelican_nlp import Pelican

    assert Pelican.__name__ == "Pelican"
