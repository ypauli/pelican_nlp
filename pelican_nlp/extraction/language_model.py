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
    MODEL_KIND_ENCODER,
    MODEL_KIND_STATIC,
    resolve_model_kind,
)

# Buffers that are rebuilt at load time. They are not pretrained weights.
_BENIGN_MISSING_SUFFIXES = ("inv_freq", "position_ids")
# The matrix a checkpoint may store only on a tied language-model head.
_EMBEDDING_WEIGHT_MARKERS = (
    "tok_embeddings.weight",
    "word_embeddings.weight",
    "embed_tokens.weight",
    "wte.weight",
)


def serious_missing_keys(missing_keys):
    """Checkpoint parameters that were left at their random initialization."""
    return [
        key
        for key in (missing_keys or [])
        if not str(key).endswith(_BENIGN_MISSING_SUFFIXES)
    ]


def missing_embedding_keys(missing_keys):
    """Embedding matrices missing because a tied copy lived only on the LM head.

    Any other missing parameter is returned as an empty list so the caller
    refuses the load instead of treating it as this one recoverable case.
    """
    missing = serious_missing_keys(missing_keys)
    if missing and all(any(marker in key for marker in _EMBEDDING_WEIGHT_MARKERS) for key in missing):
        return missing
    return []


class Model:
    def __init__(self, model_name, model_kind=None, revision=None):
        self.model_name = model_name
        self.model_instance = None
        self.device_map = None
        self.kind = None
        self._requested_kind = model_kind
        self._needed_bytes = None
        self.revision = revision or None

    def load_model(self, empty_weights=False, trust_remote_code=False, model_kind=None):
        """Loads and configures the model"""
        requested_kind = model_kind if model_kind is not None else self._requested_kind
        self.kind = resolve_model_kind(
            self.model_name,
            explicit_kind=requested_kind,
            trust_remote_code=trust_remote_code,
            revision=self.revision,
        )

        if self.kind == MODEL_KIND_STATIC:
            self._load_static()
            return

        from pelican_nlp.config import debug_print

        model_cls = AutoModelForCausalLM if self.kind == MODEL_KIND_CAUSAL_LM else AutoModel
        self.model_instance = self._from_pretrained(
            model_cls,
            empty_weights=empty_weights,
            trust_remote_code=trust_remote_code,
        )
        debug_print(f'{self.model_name} loaded ({self.kind}).')

        already_placed = bool(getattr(self.model_instance, "hf_device_map", None))
        if empty_weights or not already_placed:
            self.device_map_creation()
            self.model_instance = dispatch_model(self.model_instance, device_map=self.device_map)
            debug_print('Model dispatched to appropriate devices.')
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
            debug_print(
                f"Model placed during load: {gpu_modules} module(s) on GPU, "
                f"{cpu_modules} on CPU.",
            )

    def _from_pretrained(self, model_cls, empty_weights=False, trust_remote_code=False):
        if empty_weights:
            kwargs = huggingface_from_pretrained_kwargs(
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
                revision=self.revision,
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
            revision=self.revision,
        )
        dtype = kwargs.get("torch_dtype", "default")
        device_map = kwargs.get("device_map", "none")
        from pelican_nlp.utils.progress import active_reporter

        reporter = active_reporter()
        reporter.status(
            f"Loading {self.model_name} ({self.kind}), dtype={dtype}, "
            f"device_map={device_map}..."
        )
        import time

        started = time.monotonic()
        if self.kind == MODEL_KIND_ENCODER and self._config_ties_masked_embeddings(kwargs):
            # The token matrix is stored once, on the masked-LM head. Loading the
            # encoder alone initializes that matrix randomly.
            model = self._encoder_from_masked_lm(kwargs)
            self._reject_unloaded_embeddings(model)
        else:
            model, info = self._call_from_pretrained(model_cls, kwargs)
            if self.kind == MODEL_KIND_ENCODER:
                model = self._require_pretrained_encoder(model, info, kwargs)
        reporter.status(
            f"Loaded {self.model_name} in {time.monotonic() - started:.0f}s."
        )
        return model

    def _config_ties_masked_embeddings(self, kwargs) -> bool:
        """True when this checkpoint is a masked LM with tied input embeddings."""
        from transformers import AutoConfig

        from pelican_nlp.utils.model_cache import huggingface_from_pretrained_kwargs

        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                **huggingface_from_pretrained_kwargs(
                    trust_remote_code=kwargs.get("trust_remote_code", False),
                    revision=self.revision,
                ),
            )
        except OSError:
            return False
        architectures = [str(name) for name in (getattr(config, "architectures", None) or [])]
        tied = getattr(config, "tie_word_embeddings", True)
        return bool(tied) and any("MaskedLM" in name for name in architectures)

    def _call_from_pretrained(self, model_cls, kwargs):
        """Load weights and report which checkpoint tensors were not applied."""
        load_kwargs = dict(kwargs)
        load_kwargs["output_loading_info"] = True
        try:
            return model_cls.from_pretrained(self.model_name, **load_kwargs)
        except OSError as exc:
            # A checkpoint that only has pytorch_model.bin cannot satisfy
            # use_safetensors=True. The bin file is a valid load.
            if load_kwargs.get("use_safetensors") is False or "safetensors" not in str(exc).lower():
                raise
            load_kwargs["use_safetensors"] = False
            return model_cls.from_pretrained(self.model_name, **load_kwargs)

    def _require_pretrained_encoder(self, model, info, kwargs):
        """Keep a load only when the encoder weights came from the checkpoint.

        Masked-LM checkpoints often store the tied token matrix once, on the
        decoder head. ``AutoModel`` then leaves ``tok_embeddings`` at its
        random initialization and only logs the missing key. Load the full
        masked LM so the tie is applied, and refuse to continue if the matrix
        is still absent.
        """
        missing = serious_missing_keys((info or {}).get("missing_keys"))
        recoverable = missing_embedding_keys(missing)
        if missing and not recoverable:
            raise RuntimeError(
                f"Checkpoint for '{self.model_name}' is missing encoder weights "
                f"{missing}. Refusing to continue with randomly initialized parameters."
            )
        if recoverable:
            del model
            model = self._encoder_from_masked_lm(kwargs)
        self._reject_unloaded_embeddings(model)
        return model

    def _encoder_from_masked_lm(self, kwargs):
        """Load the masked LM without ``device_map`` and return its encoder.

        ``device_map`` forces ``low_cpu_mem_usage=True``, which leaves a tied
        embedding on the meta device. Loading on CPU first materializes the
        pretrained matrix; placement happens afterwards.
        """
        from transformers import AutoModelForMaskedLM

        recover = dict(kwargs)
        recover.pop("device_map", None)
        recover.pop("max_memory", None)
        recover["low_cpu_mem_usage"] = False
        masked, info = self._call_from_pretrained(AutoModelForMaskedLM, recover)
        still_missing = missing_embedding_keys(serious_missing_keys((info or {}).get("missing_keys")))
        if still_missing:
            raise RuntimeError(
                f"Checkpoint for '{self.model_name}' is missing token embeddings "
                f"{still_missing}. Refusing to continue with randomly initialized parameters."
            )
        encoder = getattr(masked, "base_model", masked)
        return encoder

    def _reject_unloaded_embeddings(self, model):
        embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        weight = None if embeddings is None else getattr(embeddings, "weight", None)
        unloaded = weight is None or getattr(weight, "is_meta", False) or str(getattr(weight, "device", "")) == "meta"
        if unloaded:
            raise RuntimeError(
                f"Token embeddings for '{self.model_name}' were not loaded from the checkpoint. "
                "Refusing to continue with randomly initialized parameters."
            )

    def _placement_for_load(self, trust_remote_code=False):
        """Choose fp16 vs fp32. Oversized models still use the GPU up to the budget."""
        fp16_bytes = estimate_pretrained_weight_bytes(
            self.model_name,
            trust_remote_code=trust_remote_code,
            bytes_per_param=2,
            revision=self.revision,
        )
        fp32_bytes = None if fp16_bytes is None else fp16_bytes * 2
        if fp16_bytes is None and self.kind == MODEL_KIND_CAUSAL_LM:
            fp16_bytes = UNKNOWN_CAUSAL_FP16_BYTES
            fp32_bytes = UNKNOWN_CAUSAL_FP16_BYTES * 2
        if self.kind != MODEL_KIND_CAUSAL_LM:
            return fp32_bytes, False
        from pelican_nlp.utils.progress import active_reporter

        if fp32_bytes is not None and gpu_can_hold(fp32_bytes):
            return fp32_bytes, False
        active_reporter().status(
            "Causal LM does not fit in fp32 on the GPU budget; "
            "loading float16 and filling the GPU, leftover layers on CPU."
        )
        return fp16_bytes, True

    def _load_static(self):
        from pelican_nlp.utils.progress import active_reporter

        self.model_instance, model_path = load_static_model(self.model_name)
        active_reporter().status(f"Loaded static model from {model_path}")

    def device_map_creation(self):
        from pelican_nlp.config import debug_print
        from pelican_nlp.utils.progress import active_reporter

        apply_gpu_budget()
        max_memory = hub_max_memory(needed_bytes=self._needed_bytes)
        if torch.cuda.is_available():
            device_type = "cuda"
            debug_print(f'{torch.cuda.get_device_name(0)} available.')
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            debug_print("Apple Metal (MPS) available.")
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
            active_reporter().warn("No GPU available, using CPU. This will be slow.")

        self.device_map = infer_auto_device_map(self.model_instance, max_memory=max_memory)
        return device_type
