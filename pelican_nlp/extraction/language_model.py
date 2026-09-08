import torch
import psutil

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM, AutoModel

from pelican_nlp.utils.gpu_budget import (
    UNKNOWN_CAUSAL_FP16_BYTES,
    apply_gpu_budget,
    estimate_pretrained_weight_bytes,
    gpu_can_hold,
    hub_max_memory,
    hub_model_load_kwargs,
)
from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs, load_static_model
from pelican_nlp.extraction.model_registry import (
    MODEL_KIND_CAUSAL_LM,
    MODEL_KIND_STATIC,
    resolve_model_kind,
)


class Model:
    def __init__(self, model_name, model_kind=None):
        self.model_name = model_name
        self.model_instance = None
        self.device_map = None
        self.kind = None
        self._requested_kind = model_kind
        self._needed_bytes = None

    def load_model(self, empty_weights=False, trust_remote_code=False, model_kind=None):
        """Loads and configures the model"""
        requested_kind = model_kind if model_kind is not None else self._requested_kind
        self.kind = resolve_model_kind(
            self.model_name,
            explicit_kind=requested_kind,
            trust_remote_code=trust_remote_code,
        )

        if self.kind == MODEL_KIND_STATIC:
            self._load_static()
            return

        model_cls = AutoModelForCausalLM if self.kind == MODEL_KIND_CAUSAL_LM else AutoModel
        self.model_instance = self._from_pretrained(
            model_cls,
            empty_weights=empty_weights,
            trust_remote_code=trust_remote_code,
        )
        print(f'{self.model_name} loaded ({self.kind}).')

        already_placed = bool(getattr(self.model_instance, "hf_device_map", None))
        if empty_weights or not already_placed:
            self.device_map_creation()
            self.model_instance = dispatch_model(self.model_instance, device_map=self.device_map)
            print('Model dispatched to appropriate devices.')
        else:
            self.device_map = dict(self.model_instance.hf_device_map)
            gpu_modules = sum(
                1
                for device in self.device_map.values()
                if device == 0 or str(device).startswith("cuda")
            )
            cpu_modules = sum(
                1 for device in self.device_map.values() if device == "cpu"
            )
            print(
                f"Model placed during load: {gpu_modules} module(s) on GPU, "
                f"{cpu_modules} on CPU.",
                flush=True,
            )

    def _from_pretrained(self, model_cls, empty_weights=False, trust_remote_code=False):
        if empty_weights:
            kwargs = huggingface_from_pretrained_kwargs(
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
            )
            with init_empty_weights():
                return model_cls.from_pretrained(self.model_name, **kwargs)
        needed_bytes, use_half = self._placement_for_load(trust_remote_code=trust_remote_code)
        self._needed_bytes = needed_bytes
        kwargs = hub_model_load_kwargs(
            trust_remote_code=trust_remote_code,
            use_safetensors=True,
            needed_bytes=needed_bytes,
            use_half=use_half,
        )
        dtype = kwargs.get("torch_dtype", "default")
        device_map = kwargs.get("device_map", "none")
        print(
            f"Loading {self.model_name} ({self.kind}), dtype={dtype}, "
            f"device_map={device_map}. This step has no inner progress bar "
            "and can take several minutes...",
            flush=True,
        )
        import time

        started = time.monotonic()
        model = model_cls.from_pretrained(self.model_name, **kwargs)
        print(
            f"Finished loading {self.model_name} in {time.monotonic() - started:.0f}s.",
            flush=True,
        )
        return model

    def _placement_for_load(self, trust_remote_code=False):
        """Choose fp16 vs fp32. Oversized models still use the GPU up to the budget."""
        fp16_bytes = estimate_pretrained_weight_bytes(
            self.model_name, trust_remote_code=trust_remote_code, bytes_per_param=2
        )
        fp32_bytes = None if fp16_bytes is None else fp16_bytes * 2
        if fp16_bytes is None and self.kind == MODEL_KIND_CAUSAL_LM:
            fp16_bytes = UNKNOWN_CAUSAL_FP16_BYTES
            fp32_bytes = UNKNOWN_CAUSAL_FP16_BYTES * 2
        if self.kind != MODEL_KIND_CAUSAL_LM:
            return fp32_bytes, False
        if fp32_bytes is not None and gpu_can_hold(fp32_bytes):
            return fp32_bytes, False
        print(
            "Causal LM does not fit in fp32 on the GPU budget; "
            "loading float16 and filling the GPU, leftover layers on CPU.",
            flush=True,
        )
        return fp16_bytes, True

    def _load_static(self):
        self.model_instance, model_path = load_static_model(self.model_name)
        print(f"Static model loaded successfully from {model_path}")

    def device_map_creation(self):
        apply_gpu_budget()
        max_memory = hub_max_memory(needed_bytes=self._needed_bytes)
        if torch.cuda.is_available():
            device_type = "cuda"
            print(f'{torch.cuda.get_device_name(0)} available.')
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            print("Apple Metal (MPS) available.")
            cpu_budget = max_memory.get(
                "cpu",
                f"{int(psutil.virtual_memory().total / (1024 ** 3))}GiB",
            )
            mps_memory = {"mps": cpu_budget, "cpu": cpu_budget}
            if "disk" in max_memory:
                mps_memory["disk"] = max_memory["disk"]
            max_memory = mps_memory
        else:
            device_type = "cpu"
            print("Careful: No GPU available, using CPU. This will be slow.")

        self.device_map = infer_auto_device_map(self.model_instance, max_memory=max_memory)
        return device_type
