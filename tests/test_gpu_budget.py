from types import SimpleNamespace

import pytest

from pelican_nlp.utils import gpu_budget
from pelican_nlp.utils.gpu_budget import (
    allow_disk_offload,
    apply_gpu_budget,
    cpu_worker_count,
    gpu_can_hold,
    hub_max_memory,
    hub_model_load_kwargs,
    input_device_for_model,
    memory_fraction,
    prefer_cuda,
    reset_gpu_budget_for_tests,
    weight_bytes_from_config,
)


SIXTEEN_GIB = 16 * (1024 ** 3)
FOUR_GIB = 4 * (1024 ** 3)


@pytest.fixture(autouse=True)
def _isolate_gpu_budget(monkeypatch):
    reset_gpu_budget_for_tests()
    for key in (
        "PELICAN_GPU_RESERVE_GB",
        "PELICAN_GPU_MEMORY_FRACTION",
        "PELICAN_GPU_DISK_OFFLOAD",
        "PELICAN_GPU_DISK_OFFLOAD_SIZE",
        "PELICAN_CPU_RESERVE_GB",
        "PELICAN_CPU_WORKERS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("torch.cuda.set_per_process_memory_fraction", lambda *a, **k: None)
    yield
    reset_gpu_budget_for_tests()


def _fake_cuda(monkeypatch, *, free=SIXTEEN_GIB, total=SIXTEEN_GIB):
    monkeypatch.setattr(gpu_budget, "_cuda", lambda: True)
    monkeypatch.setattr(gpu_budget, "_mps", lambda: False)
    monkeypatch.setattr(gpu_budget, "cuda_memory_bytes", lambda: (free, total))
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: SimpleNamespace(total=32 * (1024 ** 3)),
    )


def test_hub_max_memory_has_no_disk_by_default(monkeypatch):
    _fake_cuda(monkeypatch)
    mapping = hub_max_memory()
    assert "disk" not in mapping
    assert 0 in mapping
    assert "cpu" in mapping
    assert mapping[0] == "14GiB"


def test_disk_offload_is_opt_in(monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setenv("PELICAN_GPU_DISK_OFFLOAD", "1")
    mapping = hub_max_memory()
    assert mapping["disk"] == "50GiB"
    assert allow_disk_offload() is True


def test_prefer_cuda_requires_display_headroom(monkeypatch):
    _fake_cuda(monkeypatch, free=int(1.5 * (1024 ** 3)))
    assert prefer_cuda() is False
    _fake_cuda(monkeypatch, free=int(8 * (1024 ** 3)))
    reset_gpu_budget_for_tests()
    assert prefer_cuda() is True


def test_apply_gpu_budget_sets_fraction_once(monkeypatch):
    _fake_cuda(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "torch.cuda.set_per_process_memory_fraction",
        lambda fraction: calls.append(fraction),
    )
    first = apply_gpu_budget()
    second = apply_gpu_budget()
    assert len(calls) == 1
    assert calls[0] == pytest.approx(0.875)
    assert first["fraction"] == pytest.approx(0.875)
    assert second["fraction"] == pytest.approx(0.875)


def test_memory_fraction_env_override(monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setenv("PELICAN_GPU_MEMORY_FRACTION", "0.5")
    assert memory_fraction() == pytest.approx(0.5)


def test_hub_model_load_kwargs_place_weights_under_budget(monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(gpu_budget, "apply_gpu_budget", lambda: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.model_cache.huggingface_from_pretrained_kwargs",
        lambda **extra: {"cache_dir": "/tmp/hub", **extra},
    )
    kwargs = hub_model_load_kwargs(trust_remote_code=True, use_safetensors=True)
    assert kwargs["cache_dir"] == "/tmp/hub"
    assert kwargs["device_map"] == "auto"
    assert kwargs["low_cpu_mem_usage"] is True
    assert "disk" not in kwargs["max_memory"]
    assert 0 in kwargs["max_memory"]
    assert kwargs["trust_remote_code"] is True


def test_hub_model_load_kwargs_skip_device_map_on_mps_only(monkeypatch):
    monkeypatch.setattr(gpu_budget, "_cuda", lambda: False)
    monkeypatch.setattr(gpu_budget, "_mps", lambda: True)
    monkeypatch.setattr(gpu_budget, "apply_gpu_budget", lambda: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.model_cache.huggingface_from_pretrained_kwargs",
        lambda **extra: {"cache_dir": "/tmp/hub", **extra},
    )
    kwargs = hub_model_load_kwargs()
    assert "device_map" not in kwargs
    assert "max_memory" not in kwargs


def test_cpu_worker_count_defaults_to_one(monkeypatch):
    assert cpu_worker_count() == 1
    monkeypatch.setenv("PELICAN_CPU_WORKERS", "4")
    assert cpu_worker_count() == 4


def test_input_device_for_model_uses_device_map():
    model = SimpleNamespace(hf_device_map={"encoder.layer.0": 0})
    device = input_device_for_model(model)
    assert str(device) == "cuda:0"


def test_weight_bytes_from_config_llama_class():
    config = SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=128256,
        intermediate_size=14336,
    )
    nbytes = weight_bytes_from_config(config, bytes_per_param=2)
    assert nbytes is not None
    assert nbytes > 14 * (1024 ** 3)


def test_weight_bytes_from_config_whisper_encoder_decoder():
    config = SimpleNamespace(
        d_model=1280,
        encoder_layers=32,
        decoder_layers=32,
        num_hidden_layers=32,
        vocab_size=51866,
        encoder_ffn_dim=5120,
    )
    nbytes = weight_bytes_from_config(config, bytes_per_param=4)
    encoder_only = weight_bytes_from_config(
        SimpleNamespace(
            d_model=1280,
            num_hidden_layers=32,
            vocab_size=51866,
            encoder_ffn_dim=5120,
        ),
        bytes_per_param=4,
    )
    assert nbytes is not None
    assert encoder_only is not None
    assert nbytes > encoder_only * 1.5
    assert nbytes > 5 * (1024 ** 3)
    assert nbytes < 12 * (1024 ** 3)


def test_gpu_can_hold_uses_margin(monkeypatch):
    _fake_cuda(monkeypatch)
    limit = gpu_budget.gpu_limit_bytes()
    assert gpu_can_hold(int(limit * 0.5)) is True
    assert gpu_can_hold(limit) is False
    assert gpu_can_hold(None) is False


def test_hub_max_memory_keeps_gpu_when_model_is_larger_than_budget(monkeypatch):
    _fake_cuda(monkeypatch)
    mapping = hub_max_memory(needed_bytes=20 * (1024 ** 3))
    assert 0 in mapping
    assert mapping[0] == "14GiB"
    assert "cpu" in mapping


def test_hub_max_memory_keeps_gpu_for_small_model(monkeypatch):
    _fake_cuda(monkeypatch)
    mapping = hub_max_memory(needed_bytes=1 * (1024 ** 3))
    assert 0 in mapping


def test_hub_model_load_kwargs_half_still_uses_gpu_for_large_model(monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(gpu_budget, "apply_gpu_budget", lambda: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.model_cache.huggingface_from_pretrained_kwargs",
        lambda **extra: {"cache_dir": "/tmp/hub", **extra},
    )
    kwargs = hub_model_load_kwargs(
        needed_bytes=20 * (1024 ** 3),
        use_half=True,
        use_safetensors=True,
    )
    import torch

    assert 0 in kwargs["max_memory"]
    assert kwargs["torch_dtype"] == torch.float16
