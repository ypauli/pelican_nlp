"""Device-level model cache used by every loader.

``pelican-run`` still uses the current working directory only to find the YAML
and ``participants/``. Weights are resolved from the machine, never from that
project folder.

Cache roots (first existing override wins):

* ``PELICAN_CACHE_DIR`` — PELICAN static artifacts (default
  ``~/.cache/pelican-nlp``)
* ``HF_HOME`` / ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE`` — Hub models
  (default ``~/.cache/huggingface``; ``TRANSFORMERS_CACHE`` is still read if set)
* ``TORCH_HOME`` — torchaudio / torch hub (default ``~/.cache/torch``)

Static families (fastText today) are registered objects. A new family is a
class with ``matches`` / ``artifact`` / ``load``; it does not belong in the
loaders. Language-specific FastText files use ``cc.{lang}.300.bin``; ``fastText``
with no language still means German for existing configs.
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


def _home() -> Path:
    return Path.home()


def cache_root() -> Path:
    override = os.environ.get("PELICAN_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _home() / ".cache" / "pelican-nlp"


def huggingface_home() -> Path:
    override = os.environ.get("HF_HOME")
    if override:
        return Path(override).expanduser()
    return _home() / ".cache" / "huggingface"


def huggingface_hub_cache() -> Path:
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        override = os.environ.get(key)
        if override:
            return Path(override).expanduser()
    return huggingface_home() / "hub"


def torch_home() -> Path:
    override = os.environ.get("TORCH_HOME")
    if override:
        return Path(override).expanduser()
    return _home() / ".cache" / "torch"


def configure_device_caches() -> Path:
    """Pin Hub and torch caches to the device. Does not change the CWD."""
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)

    hf_home = huggingface_home()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))

    hub = huggingface_hub_cache()
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))

    torch_dir = torch_home()
    torch_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_dir))
    return root


def huggingface_from_pretrained_kwargs(**extra):
    """Kwargs so Hub loaders never fall back to the project directory."""
    configure_device_caches()
    kwargs = {"cache_dir": str(huggingface_hub_cache())}
    kwargs.update({key: value for key, value in extra.items() if value is not None})
    return kwargs


def _unique_paths(paths):
    seen = set()
    ordered = []
    for path in paths:
        key = str(Path(path).expanduser())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(Path(path).expanduser())
    return ordered


@dataclass(frozen=True)
class ArtifactSpec:
    """One downloadable file on the device (any static family)."""

    filename: str
    url: str
    subdirectory: str
    extra_search: tuple[Path, ...] = field(default_factory=tuple)
    env_path: str | None = None


class StaticFamily(Protocol):
    def matches(self, model_name: str) -> bool: ...
    def artifact(self, model_name: str) -> ArtifactSpec: ...
    def load(self, path: Path): ...


_STATIC_FAMILIES: dict[str, StaticFamily] = {}


def register_static_family(name: str, family: StaticFamily) -> None:
    _STATIC_FAMILIES[name] = family


def unregister_static_family(name: str) -> None:
    _STATIC_FAMILIES.pop(name, None)


def matching_static_family(model_name: str) -> StaticFamily | None:
    for family in _STATIC_FAMILIES.values():
        if family.matches(model_name):
            return family
    return None


def download_artifact(url: str, target: Path) -> Path:
    """Download ``url`` to ``target``. Never writes into the CWD."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / (target.name + ".partial")
    unpacked = target.parent / (target.name + ".tmp")
    from pelican_nlp.utils.progress import active_reporter

    reporter = active_reporter()
    reporter.status(f"Downloading model to {target} ...")
    try:
        urllib.request.urlretrieve(url, partial)
        if url.endswith(".gz"):
            reporter.status("Decompressing model file...")
            with gzip.open(partial, "rb") as compressed:
                with open(unpacked, "wb") as out:
                    shutil.copyfileobj(compressed, out)
            unpacked.replace(target)
        else:
            partial.replace(target)
        reporter.status("Model stored on the device cache")
    finally:
        if partial.exists():
            partial.unlink()
        if unpacked.exists():
            unpacked.unlink()
    return target


def ensure_artifact(spec: ArtifactSpec) -> Path:
    """Return ``spec`` from the device, downloading into the cache if needed."""
    configure_device_caches()
    candidates = []
    if spec.env_path:
        env_value = os.environ.get(spec.env_path)
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.extend(spec.extra_search)
    candidates.append(cache_root() / spec.subdirectory / spec.filename)

    for path in _unique_paths(candidates):
        if path.is_file():
            return path

    if spec.env_path and os.environ.get(spec.env_path):
        target = Path(os.environ[spec.env_path]).expanduser()
    else:
        target = cache_root() / spec.subdirectory / spec.filename
    return download_artifact(spec.url, target)


def ensure_static_model(model_name: str) -> Path:
    family = matching_static_family(model_name)
    if family is None:
        raise ValueError(
            f"No on-device static model family is registered for '{model_name}'."
        )
    return ensure_artifact(family.artifact(model_name))


def load_static_model(model_name: str):
    family = matching_static_family(model_name)
    if family is None:
        raise ValueError(
            f"No on-device static model family is registered for '{model_name}'."
        )
    path = ensure_artifact(family.artifact(model_name))
    try:
        return family.load(path), path
    except ValueError:
        from pelican_nlp.utils.progress import active_reporter

        active_reporter().warn(f"Existing model file is corrupted, re-downloading from {path}...")
        path.unlink(missing_ok=True)
        path = ensure_artifact(family.artifact(model_name))
        return family.load(path), path


def load_spacy_model(name: str):
    """Load a spaCy package from the environment, then from the device cache."""
    import spacy

    candidates = []
    env_path = os.environ.get("PELICAN_SPACY_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(name)
    cached = cache_root() / "spacy" / name
    if cached.exists():
        candidates.append(str(cached))

    last_error = None
    for candidate in candidates:
        try:
            return spacy.load(candidate)
        except OSError as error:
            last_error = error
    raise OSError(
        f"spaCy model '{name}' is not installed on this device. "
        f"Install it in the environment or place it under {cache_root() / 'spacy' / name}."
    ) from last_error


class FastTextFamily:
    """Crawl vectors: ``fastText``, ``fasttext-en``, ``cc.fr.300``."""

    url_template = (
        "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz"
    )
    filename_template = "cc.{lang}.300.bin"
    default_language = "de"

    def matches(self, model_name: str) -> bool:
        lower = str(model_name).strip().lower()
        return lower.startswith("fasttext") or lower.startswith("cc.")

    def language_code(self, model_name: str) -> str:
        lower = str(model_name).strip().lower().replace("fast-text", "fasttext")
        if lower.startswith("cc."):
            parts = lower.split(".")
            if len(parts) >= 2 and parts[1]:
                return parts[1]
        if lower.startswith("fasttext"):
            rest = lower[len("fasttext"):].lstrip("-_:/")
            if not rest:
                return self.default_language
            if rest.startswith("cc."):
                return rest.split(".")[1]
            lang = rest.split(".")[0]
            if 2 <= len(lang) <= 3 and lang.isalpha():
                return lang
            return self.default_language
        raise ValueError(f"Cannot parse a FastText language from '{model_name}'.")

    def artifact(self, model_name: str) -> ArtifactSpec:
        lang = self.language_code(model_name)
        filename = self.filename_template.format(lang=lang)
        return ArtifactSpec(
            filename=filename,
            url=self.url_template.format(lang=lang),
            subdirectory="fasttext",
            extra_search=(_home() / ".fasttext" / filename,),
            env_path="FASTTEXT_MODEL_PATH",
        )

    def load(self, path: Path):
        import fasttext

        return fasttext.load_model(str(path))


register_static_family("fasttext", FastTextFamily())
