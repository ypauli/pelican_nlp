"""Release model objects and GPU cache after an extraction step."""

from __future__ import annotations

import gc


def release_gpu(*objects) -> None:
    """Drop references to models and clear the CUDA cache when available."""
    for obj in objects:
        if obj is None:
            continue
        for attr in ("model_instance", "model", "Tokenizer", "tokenizer"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, None)
                except Exception:
                    pass
        try:
            del obj
        except Exception:
            pass

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()
