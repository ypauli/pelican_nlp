import warnings

TOKENIZATION_WHITESPACE = "whitespace"
TOKENIZATION_MODEL = "model"

_VALID_TOKENIZATION_METHODS = {
    TOKENIZATION_WHITESPACE,
    TOKENIZATION_MODEL,
}

# Old YAML used "model_roberta" for the Hugging Face tensor call. That is
# the same as "model": the tokenizer still comes from model_name.
_TOKENIZATION_ALIASES = {
    "model_roberta": TOKENIZATION_MODEL,
}


def normalize_tokenization_method(method):
    """Return ``whitespace`` or ``model``.

    ``model`` uses ``AutoTokenizer`` for ``model_name``. The tokenizer is not
    selected by this field.
    """
    if method is None or (isinstance(method, str) and not method.strip()):
        raise ValueError(
            "tokenization_method must be 'whitespace' or 'model'."
        )
    if not isinstance(method, str):
        raise ValueError(
            f"tokenization_method must be a string, got {type(method).__name__}."
        )
    normalized = method.strip().lower()
    if normalized in _TOKENIZATION_ALIASES:
        canonical = _TOKENIZATION_ALIASES[normalized]
        warnings.warn(
            f"tokenization_method '{method}' is deprecated; use '{canonical}'. "
            "The tokenizer is taken from model_name, not from this field.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    if normalized not in _VALID_TOKENIZATION_METHODS:
        raise ValueError(
            f"Unknown tokenization_method '{method}'. Use 'whitespace' "
            "(split on spaces) or 'model' (the tokenizer from model_name)."
        )
    return normalized


def model_encode_kwargs(tokenizer, max_length=None):
    """Kwargs for ``tokenizer(text)`` so ``model`` matches the old encode path.

    Causal LMs often have no pad token. ``padding=True`` then raises, which
    broke logits examples after ``model_roberta`` was folded into ``model``.
    """
    kwargs = {
        "return_tensors": "pt",
        "add_special_tokens": True,
        "padding": tokenizer.pad_token is not None,
    }
    if max_length:
        kwargs["truncation"] = True
        kwargs["max_length"] = max_length
    else:
        # Logits chunk after tokenization; do not cut to model_max_length here.
        kwargs["truncation"] = False
    return kwargs


class TextTokenizer:
    def __init__(self, method, model_name=None, max_length=None, trust_remote_code=False):
        self.tokenization_method = normalize_tokenization_method(method)
        self.model_name = model_name
        self.max_sequence_length = max_length
        self.trust_remote_code = trust_remote_code

        self.tokenizer = self.get_tokenizer()

    def tokenize_text(self, text):
        if not isinstance(text, str):
            raise ValueError(
                f"to tokenize a text it must be a in string format, but it is in format {type(text)}"
            )

        if self.tokenization_method == TOKENIZATION_WHITESPACE:
            return text.split()

        return self.tokenizer(
            text, **model_encode_kwargs(self.tokenizer, self.max_sequence_length)
        )

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_tokenizer(self):
        if self.tokenization_method == TOKENIZATION_MODEL:
            from transformers import AutoTokenizer
            from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs

            if not self.model_name:
                raise ValueError("model_name must be provided for model-based tokenization methods")
            return AutoTokenizer.from_pretrained(
                self.model_name,
                **huggingface_from_pretrained_kwargs(
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=True,
                ),
            )
        return None
