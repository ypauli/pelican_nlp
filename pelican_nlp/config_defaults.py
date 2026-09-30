"""Defaults for optional YAML keys that the pipeline used to require.

``input_file`` is still required. ``task_name``, ``corpus_key``, and
``corpus_values`` may be omitted: files without those tags are grouped by
unit folder instead.
"""

from copy import deepcopy

_OPTIONAL_DEFAULTS = {
    "metrics_to_extract": [],
    "opensmile_feature_extraction": False,
    "prosogram_extraction": False,
    "create_aggregation_of_results": False,
    "output_document_information": False,
    "discourse": False,
    "fluency_task": False,
    "multiple_sessions": False,
    "has_multiple_sections": False,
    "has_section_titles": False,
    "number_of_speakers": 1,
    "skip_existing": True,
}

_EMBEDDING_DEFAULTS = {
    "distance-from-randomness": False,
    "semantic-similarity": False,
    "keep_speakertags": False,
    "clean_embedding_tokens": True,
}

_PIPELINE_DEFAULTS = {
    "quality_check": False,
    "clean_text": True,
    "tokenize_text": False,
    "normalize_text": False,
}

DEFAULT_TRANSCRIPTION_MODEL = "openai/whisper-medium"
_PATCHING_TRUE = {True, 1, "1", "true", "yes", "on"}
_PATCHING_FALSE = {False, 0, "0", "false", "no", "off", None, ""}


def canonical_asr_model_id(model) -> str:
    """Normalize a Whisper/HF id so ``whisper-medium`` matches the default."""
    if not isinstance(model, str) or not model.strip():
        return DEFAULT_TRANSCRIPTION_MODEL
    name = model.strip()
    if "/" not in name and name.lower().startswith("whisper-"):
        return f"openai/{name}"
    return name


def asr_model_filename_slug(model) -> str:
    """Filesystem-safe model id (``openai/whisper-medium`` → ``openai-whisper-medium``)."""
    return canonical_asr_model_id(model).replace("/", "-")


def resolve_transcription_patching(config) -> bool:
    """Return whether a second ASR pass should patch the primary transcript."""
    if not isinstance(config, dict):
        return False
    transcription = config.get("transcription")
    if not isinstance(transcription, dict):
        return False
    value = transcription.get("transcription_patching")
    if isinstance(value, str):
        value = value.strip().lower()
    if value in _PATCHING_TRUE:
        return True
    if value in _PATCHING_FALSE:
        return False
    return bool(value)


def resolve_num_speakers(config, default: int = 1) -> int:
    """Return the expected speaker count from a config.

    ``transcription.num_speakers`` wins over the top-level ``number_of_speakers``.
    An explicit ``null`` in the YAML means "not set" rather than "no speakers": it
    used to leave the count at ``None``, which silently skipped diarization and
    labeled every word ``UNKNOWN`` even for interview recordings.
    """
    if not isinstance(config, dict):
        return default

    transcription = config.get("transcription")
    candidates = []
    if isinstance(transcription, dict):
        candidates.append(transcription.get("num_speakers"))
    candidates.append(config.get("number_of_speakers"))

    for value in candidates:
        if value is None or isinstance(value, bool):
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return default


def resolve_transcription_language(config):
    """Return the top-level ``language`` to pin for ASR, or ``None`` for auto-detection.

    An empty string or ``null`` means "not set". Transcription does not have its
    own language key.
    """
    if not isinstance(config, dict):
        return None
    value = config.get("language")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def apply_config_defaults(config):
    """Return a new dict with optional keys filled in. ``config`` is not mutated."""
    if not config:
        filled = {}
    else:
        filled = deepcopy(config)

    for key, value in _OPTIONAL_DEFAULTS.items():
        filled.setdefault(key, deepcopy(value) if not isinstance(value, bool) else value)

    # Keep the top-level count in sync with the transcription block so every
    # consumer (documents, diarization, text output) sees the same number.
    filled["number_of_speakers"] = resolve_num_speakers(
        filled, default=_OPTIONAL_DEFAULTS["number_of_speakers"]
    )

    pipeline = filled.setdefault("pipeline_options", {})
    if isinstance(pipeline, dict):
        for key, value in _PIPELINE_DEFAULTS.items():
            pipeline.setdefault(key, value)

    embeddings = filled.setdefault("options_embeddings", {})
    if isinstance(embeddings, dict):
        for key, value in _EMBEDDING_DEFAULTS.items():
            embeddings.setdefault(key, value)
        if embeddings.get("divergence_from_optimality") and not embeddings.get(
            "distance-from-randomness"
        ):
            embeddings["distance-from-randomness"] = True

    return filled
