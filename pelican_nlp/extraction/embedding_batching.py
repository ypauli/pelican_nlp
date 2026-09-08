"""Auto cross-document encoder batching.

When ``options_embeddings.batch_size`` is omitted, encoder models on CUDA pick
a batch size from free GPU memory and encode texts from several documents in
one forward. An explicit integer, including ``1``, always wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pelican_nlp.extraction.model_registry import MODEL_KIND_CAUSAL_LM, MODEL_KIND_STATIC
from pelican_nlp.extraction.sectioning import iter_section_groups

MAX_AUTO_BATCH = 64
# Rough activations: batch * seq * hidden * layers * this * dtype_bytes.
ACTIVATION_FUDGE = 8
FREE_MEMORY_FRACTION = 0.75


@dataclass
class EmbeddingJob:
    document: Any
    section_key: Any
    part_index: int
    text: Any
    embeddings: Any = None
    token_count: int = 0


def explicit_embedding_batch_size(embedding_options) -> Optional[int]:
    """Return a user-set batch size, or ``None`` to auto-select."""
    if not isinstance(embedding_options, dict) or "batch_size" not in embedding_options:
        return None
    raw = embedding_options.get("batch_size")
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"", "auto", "none"}:
            return None
        raw = text
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return None


def embedding_batch_size_or_one(embedding_options) -> int:
    """Batch size for a single ``extract_embeddings_from_text`` call (no auto)."""
    explicit = explicit_embedding_batch_size(embedding_options)
    return 1 if explicit is None else explicit


def activation_bytes_per_text(
    seq_len: int,
    hidden: int,
    layers: int,
    bytes_per_elem: int = 4,
) -> int:
    seq_len = max(1, int(seq_len or 1))
    hidden = max(1, int(hidden or 1))
    layers = max(1, int(layers or 1))
    bytes_per_elem = max(1, int(bytes_per_elem or 4))
    return seq_len * hidden * layers * ACTIVATION_FUDGE * bytes_per_elem


def auto_encoder_batch_size(
    *,
    free_bytes: Optional[int],
    seq_len: int,
    hidden: int,
    layers: int,
    n_texts: int,
    cap: int = MAX_AUTO_BATCH,
    bytes_per_elem: int = 4,
) -> int:
    """Largest batch that should fit in ``free_bytes``. ``1`` means do not batch."""
    if n_texts < 2:
        return 1
    if not free_bytes or free_bytes <= 0:
        return 1
    per_item = activation_bytes_per_text(seq_len, hidden, layers, bytes_per_elem)
    budget = int(free_bytes * FREE_MEMORY_FRACTION)
    n = budget // per_item
    if n < 2:
        return 1
    return max(2, min(n, cap, n_texts))


def encoder_shape_from_model(model_instance, default_hidden=768, default_layers=12):
    """``(hidden_size, num_layers)`` from a Hugging Face config, with fallbacks."""
    cfg = getattr(model_instance, "config", None)
    hidden = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
        or default_hidden
    )
    layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
        or default_layers
    )
    try:
        hidden = int(hidden)
    except (TypeError, ValueError):
        hidden = default_hidden
    try:
        layers = int(layers)
    except (TypeError, ValueError):
        layers = default_layers
    return max(1, hidden), max(1, layers)


def dtype_bytes_from_model(model_instance) -> int:
    try:
        import torch

        dtype = next(model_instance.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16):
            return 2
    except Exception:
        pass
    return 4


def model_is_on_cuda(model_instance) -> bool:
    try:
        from pelican_nlp.utils.gpu_budget import input_device_for_model

        device = input_device_for_model(model_instance)
    except Exception:
        return False
    return getattr(device, "type", None) == "cuda"


def model_has_cpu_or_disk_offload(model_instance) -> bool:
    device_map = getattr(model_instance, "hf_device_map", None)
    if not device_map:
        return False
    for device in device_map.values():
        if isinstance(device, (list, tuple)):
            device = device[0] if device else "cpu"
        if device in (None, "cpu", "disk"):
            return True
        if str(device).lower() in {"cpu", "disk"}:
            return True
    return False


def encoder_auto_batch_eligible(model_kind, pytorch_based, model_instance) -> bool:
    if not pytorch_based:
        return False
    if model_kind in {MODEL_KIND_CAUSAL_LM, MODEL_KIND_STATIC}:
        return False
    if model_instance is None:
        return False
    if not model_is_on_cuda(model_instance):
        return False
    if model_has_cpu_or_disk_offload(model_instance):
        return False
    return True


def free_bytes_for_activations() -> Optional[int]:
    """Physical free VRAM, capped by the process GPU budget minus allocated bytes."""
    from pelican_nlp.utils import gpu_budget

    free, _total = gpu_budget.cuda_memory_bytes()
    if free is None:
        return None
    remaining = int(free)
    try:
        import torch

        allocated = int(torch.cuda.memory_allocated())
    except Exception:
        allocated = 0
    limit = gpu_budget.allocator_limit_bytes()
    if limit is not None:
        remaining = min(remaining, max(0, int(limit) - allocated))
    return remaining


def collect_document_jobs(documents, config, keep_speakertags=False):
    """``[(document, [EmbeddingJob, ...]), ...]`` in corpus order."""
    jobs_by_doc = []
    for document in documents:
        doc_jobs = []
        for key, section_parts in iter_section_groups(
            document, config, keep_speakertags=keep_speakertags
        ):
            for part_index, text in enumerate(section_parts):
                doc_jobs.append(
                    EmbeddingJob(
                        document=document,
                        section_key=key,
                        part_index=part_index,
                        text=text,
                    )
                )
        jobs_by_doc.append((document, doc_jobs))
    return jobs_by_doc


def count_jobs(jobs_by_doc) -> int:
    return sum(len(jobs) for _document, jobs in jobs_by_doc)


def iter_document_windows(jobs_by_doc, batch_size: int):
    """Yield document groups whose text count is at least ``batch_size``.

    Always includes at least one document. The last window may be smaller.
    """
    batch_size = max(1, int(batch_size or 1))
    window = []
    n_texts = 0
    for item in jobs_by_doc:
        window.append(item)
        n_texts += len(item[1])
        if n_texts >= batch_size:
            yield window
            window = []
            n_texts = 0
    if window:
        yield window


def flatten_window_jobs(window) -> list:
    jobs = []
    for _document, doc_jobs in window:
        jobs.extend(doc_jobs)
    return jobs


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        return True
    text = f"{name} {exc}".lower()
    return "out of memory" in text and "cuda" in text


def should_window_batch(*, pytorch_based: bool, model_kind, batch_size: int) -> bool:
    if not pytorch_based:
        return False
    if model_kind == MODEL_KIND_CAUSAL_LM:
        return False
    return int(batch_size or 1) > 1
