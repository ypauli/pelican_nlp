import gzip
from pathlib import Path

from pelican_nlp.extraction.model_registry import MODEL_KIND_STATIC, resolve_model_kind
from pelican_nlp.utils import model_cache
from pelican_nlp.utils.model_cache import (
    ArtifactSpec,
    FastTextFamily,
    configure_device_caches,
    ensure_static_model,
    huggingface_from_pretrained_kwargs,
    matching_static_family,
    register_static_family,
    unregister_static_family,
)


def _isolate_homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(model_cache, "_home", lambda: home)
    monkeypatch.delenv("FASTTEXT_MODEL_PATH", raising=False)
    monkeypatch.delenv("PELICAN_CACHE_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.chdir(project)
    return home, project


def test_existing_fasttext_on_device_is_used(tmp_path, monkeypatch):
    home, project = _isolate_homes(tmp_path, monkeypatch)
    model = home / ".fasttext" / "cc.de.300.bin"
    model.parent.mkdir()
    model.write_bytes(b"weights")

    assert ensure_static_model("fastText") == model
    assert list(project.iterdir()) == []


def test_fasttext_language_is_not_hardcoded_to_german():
    family = FastTextFamily()
    assert family.language_code("fastText") == "de"
    assert family.language_code("fasttext-en") == "en"
    assert family.language_code("cc.fr.300") == "fr"
    assert family.artifact("fasttext-en").filename == "cc.en.300.bin"
    assert "cc.en.300.bin.gz" in family.artifact("fasttext-en").url


def test_fasttext_language_resolves_as_static():
    assert resolve_model_kind("fasttext-fr") == MODEL_KIND_STATIC
    assert matching_static_family("cc.it.300.bin") is not None


def test_env_path_wins_over_home_cache(tmp_path, monkeypatch):
    home, _project = _isolate_homes(tmp_path, monkeypatch)
    home_model = home / ".fasttext" / "cc.de.300.bin"
    home_model.parent.mkdir()
    home_model.write_bytes(b"home")
    override = tmp_path / "shared" / "cc.de.300.bin"
    override.parent.mkdir()
    override.write_bytes(b"shared")
    monkeypatch.setenv("FASTTEXT_MODEL_PATH", str(override))

    assert ensure_static_model("fastText") == override


def test_download_writes_to_device_cache_not_cwd(tmp_path, monkeypatch):
    home, project = _isolate_homes(tmp_path, monkeypatch)
    captured = {}

    def fake_urlretrieve(url, filename):
        captured["url"] = url
        Path(filename).write_bytes(gzip.compress(b"fake-fasttext"))

    monkeypatch.setattr(model_cache.urllib.request, "urlretrieve", fake_urlretrieve)

    path = ensure_static_model("fasttext-en")
    assert path == home / ".cache" / "pelican-nlp" / "fasttext" / "cc.en.300.bin"
    assert path.read_bytes() == b"fake-fasttext"
    assert "cc.en.300.bin.gz" in captured["url"]
    assert list(project.iterdir()) == []


def test_huggingface_kwargs_use_device_hub_cache(tmp_path, monkeypatch):
    home, project = _isolate_homes(tmp_path, monkeypatch)
    kwargs = huggingface_from_pretrained_kwargs(trust_remote_code=True)
    assert kwargs["cache_dir"] == str(home / ".cache" / "huggingface" / "hub")
    assert kwargs["trust_remote_code"] is True
    assert "device_map" not in kwargs
    assert Path(kwargs["cache_dir"]).is_dir()
    assert list(project.iterdir()) == []


def test_configure_device_caches_does_not_change_cwd(tmp_path, monkeypatch):
    _home, project = _isolate_homes(tmp_path, monkeypatch)
    configure_device_caches()
    assert Path.cwd() == project


def test_register_static_family_is_used_for_new_models(tmp_path, monkeypatch):
    home, project = _isolate_homes(tmp_path, monkeypatch)

    class DummyFamily:
        def matches(self, model_name):
            return str(model_name).startswith("dummy-")

        def artifact(self, model_name):
            return ArtifactSpec(
                filename=f"{model_name}.bin",
                url=f"https://example.invalid/{model_name}.bin.gz",
                subdirectory="dummy",
            )

        def load(self, path):
            return path.read_bytes()

    register_static_family("dummy", DummyFamily())
    try:
        def fake_urlretrieve(_url, filename):
            Path(filename).write_bytes(gzip.compress(b"dummy-weights"))

        monkeypatch.setattr(model_cache.urllib.request, "urlretrieve", fake_urlretrieve)
        path = ensure_static_model("dummy-vectors")
        assert path == home / ".cache" / "pelican-nlp" / "dummy" / "dummy-vectors.bin"
        assert path.read_bytes() == b"dummy-weights"
        assert resolve_model_kind("dummy-vectors") == MODEL_KIND_STATIC
        assert list(project.iterdir()) == []
    finally:
        unregister_static_family("dummy")


def test_pelican_run_cwd_still_finds_yaml_only():
    from pelican_nlp.cli import _run_pipeline
    import inspect

    source = inspect.getsource(_run_pipeline)
    assert "Path.cwd()" in source
    assert "chdir" not in source
