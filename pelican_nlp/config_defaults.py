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


def apply_config_defaults(config):
    """Return a new dict with optional keys filled in. ``config`` is not mutated."""
    if not config:
        filled = {}
    else:
        filled = deepcopy(config)

    for key, value in _OPTIONAL_DEFAULTS.items():
        filled.setdefault(key, deepcopy(value) if not isinstance(value, bool) else value)

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
