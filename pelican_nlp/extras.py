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


def require_config_extras(config: Mapping | None) -> None:
    """Fail once if any extra required by ``config`` is incomplete."""
    needed = extras_required_by_config(config)
    missing_extras = []
    missing_modules = []
    for name in needed:
        markers = EXTRA_MARKERS[name]
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
