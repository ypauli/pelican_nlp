import pytest

from types import SimpleNamespace

from pelican_nlp.preprocessing.text_tokenizer import (
    TOKENIZATION_MODEL,
    TOKENIZATION_WHITESPACE,
    TextTokenizer,
    model_encode_kwargs,
    normalize_tokenization_method,
)


def test_normalize_accepts_whitespace_and_model():
    assert normalize_tokenization_method("whitespace") == TOKENIZATION_WHITESPACE
    assert normalize_tokenization_method("model") == TOKENIZATION_MODEL
    assert normalize_tokenization_method(" MODEL ") == TOKENIZATION_MODEL


def test_normalize_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="whitespace"):
        normalize_tokenization_method("mmBERT")
    with pytest.raises(ValueError, match="tokenization_method"):
        normalize_tokenization_method("")
    with pytest.raises(ValueError, match="tokenization_method"):
        normalize_tokenization_method(None)


def test_model_roberta_is_deprecated_alias_for_model():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert normalize_tokenization_method("model_roberta") == TOKENIZATION_MODEL


def test_whitespace_tokenizer_splits():
    tokenizer = TextTokenizer("whitespace")
    assert tokenizer.tokenize_text("one two  three") == ["one", "two", "three"]
    with pytest.raises(ValueError, match="string"):
        tokenizer.tokenize_text(["not", "a", "string"])


def test_model_encode_kwargs_skips_padding_without_pad_token():
    no_pad = SimpleNamespace(pad_token=None)
    kwargs = model_encode_kwargs(no_pad, max_length=None)
    assert kwargs["padding"] is False
    assert kwargs["truncation"] is False
    assert "max_length" not in kwargs

    with_pad = SimpleNamespace(pad_token="<pad>")
    padded = model_encode_kwargs(with_pad, max_length=512)
    assert padded["padding"] is True
    assert padded["truncation"] is True
    assert padded["max_length"] == 512


def test_model_tokenize_matches_encode_without_pad_token():
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained("DiscoResearch/Llama3-German-8B-32k")
    assert hf.pad_token is None
    wrapper = TextTokenizer("model", model_name="DiscoResearch/Llama3-German-8B-32k")
    text = "Hallo Welt"
    encoded = hf.encode(text, add_special_tokens=True)
    batched = wrapper.tokenize_text(text)
    assert batched["input_ids"][0].tolist() == encoded
