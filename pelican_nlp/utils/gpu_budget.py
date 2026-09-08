"""Process-level compute budget.

Keeps GPU headroom for the display, applies CPU offload during Hub load,
and exposes a CPU worker count for non-GPU work. Disk offload is off
unless explicitly enabled — swapping to disk can freeze a laptop as well.
"""

from __future__ import annotations

import os

DEFAULT_GPU_RESERVE_GB = 2.0
DEFAULT_CPU_RESERVE_GB = 3.0
MIN_FRACTION = 0.1
MAX_FRACTION = 0.95
MIN_GPU_PLACEMENT_BYTES = int(0.5 * (1024 ** 3))
# Extra room for activations when asking whether the *whole* model fits.
WEIGHT_FIT_MARGIN = 1.15
# If we cannot count parameters, treat unknown causal LMs as 8B-class (fp16).
UNKNOWN_CAUSAL_FP16_BYTES = int(16 * (1024 ** 3))

_applied = False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def gpu_reserve_gb() -> float:
    return max(0.0, _env_float("PELICAN_GPU_RESERVE_GB", DEFAULT_GPU_RESERVE_GB))


def cpu_reserve_gb() -> float:
    return max(0.0, _env_float("PELICAN_CPU_RESERVE_GB", DEFAULT_CPU_RESERVE_GB))


def allow_disk_offload() -> bool:
    return _env_flag("PELICAN_GPU_DISK_OFFLOAD")


def cpu_worker_count() -> int:
    """Workers for CPU-only steps (openSMILE). Default 1 = sequential."""
    raw = os.environ.get("PELICAN_CPU_WORKERS")
    if raw is None or str(raw).strip() == "":
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def _mps() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def accelerate_device_map_supported() -> bool:
    """``device_map='auto'`` is unreliable on MPS-only machines."""
    return not (_mps() and not _cuda())


def cuda_memory_bytes():
    """``(free, total)`` physical bytes, or ``(None, None)`` without CUDA."""
    if not _cuda():
        return None, None
    import torch

    try:
        return torch.cuda.mem_get_info()
    except Exception:
        return None, None


def gpu_limit_bytes() -> int | None:
    """Bytes this process may place on GPU 0. ``None`` means keep the model off GPU."""
    free, total = cuda_memory_bytes()
    if total is None:
        return None
    reserve = int(gpu_reserve_gb() * (1024 ** 3))
    reserve = min(reserve, int(total * (1.0 - MIN_FRACTION)))
    from_total = max(0, total - reserve)
    # Do not subtract the reserve twice: display use already shows up in ``free``.
    usable = min(from_total, free or 0)
    if usable < MIN_GPU_PLACEMENT_BYTES:
        return None
    return usable


def allocator_limit_bytes() -> int | None:
    """Cap for ``set_per_process_memory_fraction`` (based on device total)."""
    _free, total = cuda_memory_bytes()
    if total is None:
        return None
    reserve = int(gpu_reserve_gb() * (1024 ** 3))
    reserve = min(reserve, int(total * (1.0 - MIN_FRACTION)))
    return max(int(total * MIN_FRACTION), total - reserve)


def memory_fraction() -> float | None:
    limit = allocator_limit_bytes()
    _free, total = cuda_memory_bytes()
    if limit is None or not total:
        return None
    override = os.environ.get("PELICAN_GPU_MEMORY_FRACTION")
    if override is not None and str(override).strip() != "":
        try:
            return min(MAX_FRACTION, max(MIN_FRACTION, float(override)))
        except ValueError:
            pass
    return min(MAX_FRACTION, max(MIN_FRACTION, limit / total))


def gpu_can_hold(needed_bytes: int | None) -> bool:
    """True when ``needed_bytes`` fits in the GPU budget, including a margin."""
    if needed_bytes is None:
        return False
    limit = gpu_limit_bytes()
    if limit is None:
        return False
    return int(needed_bytes * WEIGHT_FIT_MARGIN) <= limit


def weight_bytes_from_config(config, bytes_per_param: int = 2) -> int | None:
    """Rough transformer weight size from an HF config. Overestimates slightly."""
    hidden = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    vocab = getattr(config, "vocab_size", None)
    intermediate = getattr(config, "intermediate_size", None)
    if not hidden or not layers or not vocab:
        return None
    if intermediate is None:
        intermediate = 4 * hidden
    attention = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate
    embeddings = 2 * vocab * hidden
    params = layers * (attention + mlp) + embeddings
    return int(params * bytes_per_param)


def estimate_pretrained_weight_bytes(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    bytes_per_param: int = 2,
) -> int | None:
    """Best-effort weight size without loading checkpoints. ``None`` if unknown."""
    try:
        from transformers import AutoConfig
        from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs

        config = AutoConfig.from_pretrained(
            model_name,
            **huggingface_from_pretrained_kwargs(trust_remote_code=trust_remote_code),
        )
    except Exception:
        return None
    return weight_bytes_from_config(config, bytes_per_param=bytes_per_param)


def hub_max_memory(needed_bytes: int | None = None) -> dict:
    """``max_memory`` for ``from_pretrained(device_map='auto')``. No disk by default.

    Always fills the GPU up to the budget when CUDA is usable. Layers that do
    not fit go to CPU. The whole model is not forced onto CPU just because it
    is larger than VRAM.
    """
    import psutil

    cpu_total = psutil.virtual_memory().total
    cpu_reserve = int(cpu_reserve_gb() * (1024 ** 3))
    cpu_bytes = max(int(1 * (1024 ** 3)), cpu_total - cpu_reserve)
    mapping: dict = {"cpu": _bytes_to_gib(cpu_bytes)}

    gpu_bytes = gpu_limit_bytes()
    if gpu_bytes is not None:
        mapping[0] = _bytes_to_gib(gpu_bytes)
        if needed_bytes is not None and not gpu_can_hold(needed_bytes):
            print(
                f"Model weights (~{needed_bytes / (1024 ** 3):.1f} GiB) exceed the "
                f"GPU budget ({gpu_bytes / (1024 ** 3):.1f} GiB). Filling the GPU "
                "and offloading leftover layers to CPU.",
                flush=True,
            )
    if allow_disk_offload():
        mapping["disk"] = os.environ.get("PELICAN_GPU_DISK_OFFLOAD_SIZE", "50GiB")
    return mapping


def hub_model_load_kwargs(**extra) -> dict:
    """Kwargs so Hub *weights* load under the budget (cache dir + device map)."""
    from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs

    needed_bytes = extra.pop("needed_bytes", None)
    use_half = extra.pop("use_half", False)
    apply_gpu_budget()
    kwargs = huggingface_from_pretrained_kwargs(**extra)
    kwargs.setdefault("low_cpu_mem_usage", True)
    if use_half:
        import torch

        kwargs.setdefault("torch_dtype", torch.float16)
    if accelerate_device_map_supported():
        kwargs.setdefault("device_map", "auto")
        kwargs["max_memory"] = hub_max_memory(needed_bytes=needed_bytes)
    return kwargs


def prefer_cuda(min_free_gb: float = 0.0) -> bool:
    """True when CUDA is usable without eating the display reserve."""
    apply_gpu_budget()
    if not _cuda():
        return False
    free, _total = cuda_memory_bytes()
    if free is None:
        return False
    reserve = int(gpu_reserve_gb() * (1024 ** 3))
    if free < reserve:
        print(
            f"GPU has {free / (1024 ** 3):.1f} GiB free; "
            f"need {gpu_reserve_gb():.1f} GiB display headroom. Using CPU."
        )
        return False
    if min_free_gb and free < min_free_gb * (1024 ** 3):
        print(
            f"GPU has {free / (1024 ** 3):.1f} GiB free; "
            f"need {min_free_gb:.1f} GiB for this model. Using CPU."
        )
        return False
    return True


def runtime_torch_device(*, min_free_gb: float = 0.0, allow_mps: bool = True):
    """CUDA if headroom remains, else MPS (optional), else CPU."""
    import torch

    apply_gpu_budget()
    if prefer_cuda(min_free_gb=min_free_gb):
        return torch.device("cuda")
    if allow_mps and _mps():
        return torch.device("mps")
    return torch.device("cpu")


def input_device_for_model(model):
    """Device for encoder inputs when the model may be split by ``device_map``."""
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        first = next(iter(device_map.values()))
        if isinstance(first, (list, tuple)):
            first = first[0] if first else "cpu"
        if first in (None, "disk"):
            first = "cpu"
        if isinstance(first, int):
            return torch.device(f"cuda:{first}")
        return torch.device(str(first))
    try:
        return next(model.parameters()).device
    except (StopIteration, TypeError):
        return torch.device("cpu")


def apply_gpu_budget() -> dict:
    """Set the CUDA allocator cap once per process. Safe to call often."""
    global _applied
    snapshot = {
        "cuda": False,
        "fraction": None,
        "max_memory": hub_max_memory(),
        "reserve_gb": gpu_reserve_gb(),
    }
    if not _cuda():
        return snapshot
    snapshot["cuda"] = True
    fraction = memory_fraction()
    snapshot["fraction"] = fraction
    if _applied:
        return snapshot
    _applied = True
    if fraction is None:
        return snapshot
    import torch

    try:
        torch.cuda.set_per_process_memory_fraction(fraction)
        limit = allocator_limit_bytes() or 0
        print(
            f"GPU budget: {limit / (1024 ** 3):.1f} GiB for PyTorch "
            f"({gpu_reserve_gb():.1f} GiB reserved), memory_fraction={fraction:.2f}."
        )
    except Exception as error:
        print(f"Could not set CUDA memory fraction: {error}")
    return snapshot


def reset_gpu_budget_for_tests() -> None:
    global _applied
    _applied = False


def _bytes_to_gib(value: int) -> str:
    gib = max(1, int(value / (1024 ** 3)))
    return f"{gib}GiB"
