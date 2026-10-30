"""Named optional-dependency extras and missing-install checks.

``pip install pelican_nlp`` still ships the full stack. Extra names document
which libraries a YAML config needs, and they become real subsets if default
dependencies are slimmed later.
"""

from __future__ import annotations

import importlib.util
from typing import Iterable, List, Mapping, Sequence

EXTRA_MARKERS: Mapping[str, Sequence[str]] = {
    "transcription": (
        "torch",
        "torchaudio",
        "transformers",
        "huggingface_hub",
        "pyannote.audio",
        "uroman",
        "librosa",
        "soundfile",
        "pydub",
        "audiofile",
    ),
    "acoustic": (
        "opensmile",
        "parselmouth",
        "audiofile",
        "librosa",
        "soundfile",
        "pydub",
    ),
    "embeddings": (
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "fasttext",
        "sklearn",
        "scipy",
        "statsmodels",
        "ortools",
    ),
    "nlp": ("spacy",),
    "topic": ("bertopic",),
}

_TEXT_METRICS = frozenset({"embeddings", "logits", "perplexity"})


def _missing_modules(modules: Iterable[str]) -> List[str]:
    return [name for name in modules if importlib.util.find_spec(name) is None]


def extra_install_hint(extras: Sequence[str]) -> str:
    names = [name.strip() for name in extras if name and str(name).strip()]
    if not names:
        raise ValueError("at least one extra name is required")
    joined = ",".join(names)
    return f"pip install 'pelican_nlp[{joined}]'"


def require_extra(name: str) -> None:
    """Raise ImportError if marker libraries for ``name`` are missing."""
    markers = EXTRA_MARKERS.get(name)
    if markers is None:
        raise ValueError(f"Unknown extra: {name}")
    missing = _missing_modules(markers)
    if not missing:
        return
    hint = extra_install_hint([name])
    raise ImportError(
        f"The '{name}' extra is required (missing: {', '.join(missing)}). "
        f"Install with: {hint}"
    )


def extras_required_by_config(config: Mapping | None) -> List[str]:
    """Return extra names implied by a loaded Pelican YAML config."""
    if not config:
        return []

    extras: List[str] = []
    if config.get("transcription"):
        extras.append("transcription")
    if config.get("opensmile_feature_extraction") or config.get("prosogram_extraction"):
        extras.append("acoustic")

    metrics = config.get("metrics_to_extract") or []
    if any(metric in _TEXT_METRICS for metric in metrics):
        extras.append("embeddings")
    if "topic_modeling" in metrics:
        extras.append("topic")

    pipeline = config.get("pipeline_options") or {}
    if isinstance(pipeline, dict) and pipeline.get("normalize_text"):
        extras.append("nlp")

    return extras


def _model_name(config, key: str) -> str:
    block = (config or {}).get(key) or {}
    if not isinstance(block, dict):
        return ""
    return str(block.get("model_name") or "").strip().lower()


def _uses_fasttext(config) -> bool:
    name = _model_name(config, "options_embeddings")
    return name.startswith("fasttext") or name.startswith("cc.")


def _uses_distance(config) -> bool:
    block = (config or {}).get("options_embeddings") or {}
    if not isinstance(block, dict):
        return False
    return bool(block.get("distance-from-randomness") or block.get("divergence_from_optimality"))


def modules_for_extra(name: str, config: Mapping | None = None) -> List[str]:
    """Libraries this config will import for ``name``.

    The embeddings extra lists fastText, scikit-learn, and OR-Tools, but an
    encoder-only config never imports them. Requiring them blocks a run that
    does not use those paths.
    """
    if name != "embeddings":
        return list(EXTRA_MARKERS[name])
    modules = ["torch", "transformers", "accelerate", "huggingface_hub"]
    metrics = (config or {}).get("metrics_to_extract") or []
    block = (config or {}).get("options_embeddings") or {}
    if isinstance(block, dict) and "embeddings" in metrics and block.get("semantic-similarity"):
        modules.append("scipy")
    if _uses_fasttext(config):
        modules.append("fasttext")
    if _uses_distance(config):
        modules.extend(["sklearn", "ortools"])
    return modules


def require_config_extras(config: Mapping | None) -> None:
    """Fail once if any extra required by ``config`` is incomplete."""
    needed = extras_required_by_config(config)
    missing_extras = []
    missing_modules = []
    for name in needed:
        markers = modules_for_extra(name, config)
        absent = _missing_modules(markers)
        if absent:
            missing_extras.append(name)
            missing_modules.extend(absent)
    if not missing_extras:
        return
    unique_modules = list(dict.fromkeys(missing_modules))
    hint = extra_install_hint(missing_extras)
    raise ImportError(
        "This config needs pelican_nlp extras that are not installed: "
        f"{', '.join(missing_extras)} (missing: {', '.join(unique_modules)}). "
        f"Install with: {hint}"
    )
