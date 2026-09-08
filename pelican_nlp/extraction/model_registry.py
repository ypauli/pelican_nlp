"""Resolve how Hugging Face and static embedding models should be loaded.

To add a new model, in order of preference:

1. Set ``model_name`` to a Hugging Face id (or local checkpoint path) in YAML.
   Encoder vs causal LM is inferred from the model config.
2. If inference fails, set ``model_kind`` in YAML to ``encoder``, ``causal_lm``,
   or ``static``.
3. Optionally add an exact-name override in ``MODEL_KIND_OVERRIDES`` for aliases
   or models whose Hub config is ambiguous.
"""

from __future__ import annotations

from typing import Any, Optional

MODEL_KIND_STATIC = "static"
MODEL_KIND_ENCODER = "encoder"
MODEL_KIND_CAUSAL_LM = "causal_lm"

VALID_MODEL_KINDS = {
    MODEL_KIND_STATIC,
    MODEL_KIND_ENCODER,
    MODEL_KIND_CAUSAL_LM,
}

MODEL_KIND_ALIASES = {
    "static": MODEL_KIND_STATIC,
    "fasttext": MODEL_KIND_STATIC,
    "encoder": MODEL_KIND_ENCODER,
    "mlm": MODEL_KIND_ENCODER,
    "masked_lm": MODEL_KIND_ENCODER,
    "bert": MODEL_KIND_ENCODER,
    "causal_lm": MODEL_KIND_CAUSAL_LM,
    "causal": MODEL_KIND_CAUSAL_LM,
    "decoder": MODEL_KIND_CAUSAL_LM,
    "llm": MODEL_KIND_CAUSAL_LM,
}

# Case-insensitive exact names. Loading still uses the original model_name string.
MODEL_KIND_OVERRIDES = {
    "fasttext": MODEL_KIND_STATIC,
    "xlm-roberta-base": MODEL_KIND_ENCODER,
    "discoresearch/llama3-german-8b-32k": MODEL_KIND_CAUSAL_LM,
    "jhu-clsp/mmbert-base": MODEL_KIND_ENCODER,
}

# Used when Hub `architectures` is missing or uninformative.
ENCODER_MODEL_TYPES = {
    "albert",
    "bert",
    "camembert",
    "convbert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "ernie",
    "flaubert",
    "funnel",
    "layoutlm",
    "longformer",
    "megatron-bert",
    "mobilebert",
    "modernbert",
    "mpnet",
    "nystromformer",
    "roberta",
    "squeezebert",
    "xlm-roberta",
    "xlm-roberta-xl",
}

CAUSAL_LM_MODEL_TYPES = {
    "bloom",
    "cohere",
    "falcon",
    "gemma",
    "gemma2",
    "gemma3",
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "olmo",
    "olmo2",
    "opt",
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "stablelm",
    "starcoder2",
}


def normalize_model_kind(kind: Optional[str]) -> Optional[str]:
    """Return a canonical kind or None if kind is empty."""
    if kind is None:
        return None
    if not isinstance(kind, str):
        raise ValueError(f"model_kind must be a string, got {type(kind).__name__}")
    normalized = kind.strip().lower()
    if not normalized:
        return None
    if normalized in MODEL_KIND_ALIASES:
        return MODEL_KIND_ALIASES[normalized]
    raise ValueError(
        f"Unknown model_kind '{kind}'. Use one of: {', '.join(sorted(VALID_MODEL_KINDS))}."
    )


def infer_kind_from_hf_config(config: Any) -> Optional[str]:
    """Infer encoder vs causal LM from a Hugging Face config object."""
    architectures = getattr(config, "architectures", None) or []
    for architecture in architectures:
        architecture_name = str(architecture).lower()
        if "forcausallm" in architecture_name:
            return MODEL_KIND_CAUSAL_LM
        if "formaskedlm" in architecture_name:
            return MODEL_KIND_ENCODER

    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type in CAUSAL_LM_MODEL_TYPES:
        return MODEL_KIND_CAUSAL_LM
    if model_type in ENCODER_MODEL_TYPES:
        return MODEL_KIND_ENCODER

    if getattr(config, "is_decoder", False) and not getattr(config, "is_encoder_decoder", False):
        return MODEL_KIND_CAUSAL_LM

    return None


def _load_hf_config(model_name: str, trust_remote_code: bool = False):
    from transformers import AutoConfig

    from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs

    return AutoConfig.from_pretrained(
        model_name,
        **huggingface_from_pretrained_kwargs(trust_remote_code=trust_remote_code),
    )


def resolve_model_kind(
    model_name: str,
    explicit_kind: Optional[str] = None,
    trust_remote_code: bool = False,
    hf_config: Any = None,
) -> str:
    """Return ``static``, ``encoder``, or ``causal_lm`` for ``model_name``.

    Resolution order: explicit YAML kind, known-name override, registered
    static families (fastText and any family added via
    ``register_static_family``), then Hugging Face config.
    """
    if not model_name or not str(model_name).strip():
        raise ValueError("model_name must be a non-empty string.")

    normalized_explicit = normalize_model_kind(explicit_kind)
    if normalized_explicit is not None:
        return normalized_explicit

    override = MODEL_KIND_OVERRIDES.get(str(model_name).strip().lower())
    if override is not None:
        return override

    from pelican_nlp.utils.model_cache import matching_static_family

    if matching_static_family(model_name) is not None:
        return MODEL_KIND_STATIC

    config = hf_config if hf_config is not None else _load_hf_config(
        model_name, trust_remote_code=trust_remote_code
    )
    inferred = infer_kind_from_hf_config(config)
    if inferred is not None:
        return inferred

    raise ValueError(
        f"Could not determine how to load '{model_name}'. "
        "Set model_kind in your YAML to 'encoder' (BERT-like embeddings), "
        "'causal_lm' (decoder models for logits/perplexity), or 'static' (fastText)."
    )
