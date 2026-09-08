from types import SimpleNamespace

import pytest

from pelican_nlp.extraction.model_registry import (
    MODEL_KIND_CAUSAL_LM,
    MODEL_KIND_ENCODER,
    MODEL_KIND_STATIC,
    infer_kind_from_hf_config,
    normalize_model_kind,
    resolve_model_kind,
)


def test_overrides_cover_existing_and_mmbert_names():
    assert resolve_model_kind("fastText") == MODEL_KIND_STATIC
    assert resolve_model_kind("xlm-roberta-base") == MODEL_KIND_ENCODER
    assert resolve_model_kind("DiscoResearch/Llama3-German-8B-32k") == MODEL_KIND_CAUSAL_LM
    assert resolve_model_kind("jhu-clsp/mmBERT-base") == MODEL_KIND_ENCODER


def test_explicit_kind_wins_over_override():
    assert resolve_model_kind("xlm-roberta-base", explicit_kind="causal_lm") == MODEL_KIND_CAUSAL_LM


def test_kind_aliases():
    assert normalize_model_kind("MLM") == MODEL_KIND_ENCODER
    assert normalize_model_kind("causal") == MODEL_KIND_CAUSAL_LM
    assert normalize_model_kind(None) is None


def test_infer_from_modernbert_and_llama_configs():
    mmbert = SimpleNamespace(
        architectures=["ModernBertForMaskedLM"],
        model_type="modernbert",
        is_decoder=False,
        is_encoder_decoder=False,
    )
    llama = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        is_decoder=True,
        is_encoder_decoder=False,
    )
    assert infer_kind_from_hf_config(mmbert) == MODEL_KIND_ENCODER
    assert infer_kind_from_hf_config(llama) == MODEL_KIND_CAUSAL_LM
    assert resolve_model_kind("some/unknown-encoder", hf_config=mmbert) == MODEL_KIND_ENCODER


def test_unknown_config_requires_explicit_kind():
    empty = SimpleNamespace(architectures=[], model_type="unknown", is_decoder=False, is_encoder_decoder=False)
    with pytest.raises(ValueError, match="model_kind"):
        resolve_model_kind("org/mystery-model", hf_config=empty)
