from types import SimpleNamespace

from pelican_nlp.extraction.embedding_batching import (
    auto_encoder_batch_size,
    collect_document_jobs,
    embedding_batch_size_or_one,
    encoder_auto_batch_eligible,
    encoder_shape_from_model,
    explicit_embedding_batch_size,
    flatten_window_jobs,
    is_cuda_oom,
    iter_document_windows,
    model_has_cpu_or_disk_offload,
    should_window_batch,
)
from pelican_nlp.extraction.extract_embeddings import EmbeddingsExtractor
from pelican_nlp.extraction.model_registry import (
    MODEL_KIND_CAUSAL_LM,
    MODEL_KIND_ENCODER,
    MODEL_KIND_STATIC,
)


def test_explicit_batch_size_none_means_auto():
    assert explicit_embedding_batch_size({}) is None
    assert explicit_embedding_batch_size({"batch_size": None}) is None
    assert explicit_embedding_batch_size({"batch_size": "auto"}) is None
    assert explicit_embedding_batch_size({"batch_size": 1}) == 1
    assert explicit_embedding_batch_size({"batch_size": 16}) == 16
    assert embedding_batch_size_or_one({}) == 1
    assert embedding_batch_size_or_one({"batch_size": "auto"}) == 1


def test_auto_encoder_batch_size_fills_free_memory():
    twelve_gib = 12 * (1024 ** 3)
    n = auto_encoder_batch_size(
        free_bytes=twelve_gib,
        seq_len=512,
        hidden=768,
        layers=12,
        n_texts=200,
    )
    assert n == 64
    assert (
        auto_encoder_batch_size(
            free_bytes=int(0.2 * (1024 ** 3)),
            seq_len=512,
            hidden=768,
            layers=12,
            n_texts=200,
        )
        == 1
    )
    assert (
        auto_encoder_batch_size(
            free_bytes=twelve_gib,
            seq_len=512,
            hidden=768,
            layers=12,
            n_texts=1,
        )
        == 1
    )


def test_auto_encoder_batch_size_caps_at_text_count():
    n = auto_encoder_batch_size(
        free_bytes=12 * (1024 ** 3),
        seq_len=512,
        hidden=768,
        layers=12,
        n_texts=5,
    )
    assert n == 5


def test_document_windows_fill_to_batch_size():
    class _Doc:
        def __init__(self, name, text):
            self.name = name
            self.cleaned_sections = {"s": text}

    docs = [_Doc(f"d{i}", f"t{i}") for i in range(5)]
    jobs_by_doc = collect_document_jobs(docs, {"discourse": False})
    windows = list(iter_document_windows(jobs_by_doc, 2))
    assert [len(window) for window in windows] == [2, 2, 1]
    assert [job.text for job in flatten_window_jobs(windows[0])] == ["t0", "t1"]


def test_encoder_auto_batch_eligible_rejects_cpu_causal_and_offload():
    cpu_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8, num_hidden_layers=2),
        hf_device_map=None,
    )
    cpu_model.parameters = lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])
    assert encoder_auto_batch_eligible(MODEL_KIND_ENCODER, True, cpu_model) is False
    assert encoder_auto_batch_eligible(MODEL_KIND_CAUSAL_LM, True, object()) is False
    assert encoder_auto_batch_eligible(MODEL_KIND_STATIC, True, object()) is False
    assert encoder_auto_batch_eligible(MODEL_KIND_ENCODER, False, object()) is False
    offloaded = SimpleNamespace(hf_device_map={"layer.0": 0, "layer.1": "cpu"})
    assert model_has_cpu_or_disk_offload(offloaded) is True


def test_should_window_batch():
    assert should_window_batch(pytorch_based=True, model_kind=MODEL_KIND_ENCODER, batch_size=8)
    assert not should_window_batch(pytorch_based=True, model_kind=MODEL_KIND_ENCODER, batch_size=1)
    assert not should_window_batch(pytorch_based=True, model_kind=MODEL_KIND_CAUSAL_LM, batch_size=8)


def test_is_cuda_oom():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))


def test_encoder_shape_from_model():
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=1024, num_hidden_layers=24))
    assert encoder_shape_from_model(model) == (1024, 24)


class _Doc:
    def __init__(self, name, text):
        self.name = name
        self.cleaned_sections = {"section": text}
        self.embeddings = []
        self.fluency_word_count = None


class _Corpus:
    def __init__(self, documents, batch_size=None):
        self.documents = documents
        self.task = "fluency"
        self.derivatives_dir = "/tmp"
        options = {
            "clean_embedding_tokens": False,
            "semantic-similarity": False,
            "distance-from-randomness": False,
            "keep_speakertags": False,
        }
        if batch_size is not None:
            options["batch_size"] = batch_size
        self.config = {"discourse": False, "options_embeddings": options}


def _stub_extractor():
    extractor = EmbeddingsExtractor.__new__(EmbeddingsExtractor)
    extractor.embeddings_configurations = {
        "pytorch_based_model": True,
        "max_length": 32,
        "model_name": "stub-encoder",
    }
    extractor.model = SimpleNamespace(kind=MODEL_KIND_ENCODER)
    extractor.model_instance = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8, num_hidden_layers=2),
        hf_device_map=None,
    )
    return extractor


def test_windowed_process_encodes_two_documents_together(monkeypatch):
    extractor = _stub_extractor()
    calls = []

    def fake_batched(text_list, embedding_options, batch_size):
        calls.append((list(text_list), batch_size))
        return ([[(text, [1.0])] for text in text_list], [1] * len(text_list))

    extractor._extract_embeddings_encoder_batched = fake_batched
    extractor._resolve_encoder_batch_size = lambda *args, **kwargs: 2
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.store_features_to_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.release_gpu",
        lambda *args, **kwargs: None,
    )

    docs = [_Doc("a.txt", "hello"), _Doc("b.txt", "world")]
    extractor.process_corpus(_Corpus(docs))
    assert calls == [(["hello", "world"], 2)]
    assert docs[0].embeddings[0][0][0][0] == "hello"
    assert docs[1].embeddings[0][0][0][0] == "world"
    assert docs[0].fluency_word_count == 1


def test_explicit_batch_size_one_stays_sequential(monkeypatch):
    extractor = _stub_extractor()
    per_text = []

    def fake_from_text(text_list, embedding_options):
        per_text.append(list(text_list))
        return [[("t", [0.0])]], 1

    extractor.extract_embeddings_from_text = fake_from_text
    extractor._extract_embeddings_encoder_batched = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("should not use cross-document batching")
    )
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.store_features_to_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.release_gpu",
        lambda *args, **kwargs: None,
    )

    docs = [_Doc("a.txt", "hello"), _Doc("b.txt", "world")]
    extractor.process_corpus(_Corpus(docs, batch_size=1))
    assert per_text == [["hello"], ["world"]]


def test_auto_resolve_picks_batch_size_without_yaml(monkeypatch, capsys):
    extractor = _stub_extractor()
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.encoder_auto_batch_eligible",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.free_bytes_for_activations",
        lambda: 12 * (1024 ** 3),
    )
    monkeypatch.setattr(
        "pelican_nlp.extraction.extract_embeddings.dtype_bytes_from_model",
        lambda model: 4,
    )
    docs = [_Doc("a.txt", "hello"), _Doc("b.txt", "world")]
    corpus = _Corpus(docs)
    from pelican_nlp.extraction.extract_embeddings import _embedding_write_options

    write_opts = _embedding_write_options(corpus, corpus.config["options_embeddings"])
    n = extractor._resolve_encoder_batch_size(
        corpus, corpus.config["options_embeddings"], write_opts
    )
    assert n == 2
    assert "auto batch_size=2" in capsys.readouterr().out


def test_encode_jobs_halves_batch_after_cuda_oom():
    extractor = _stub_extractor()
    sizes = []

    def fake_batched(text_list, embedding_options, batch_size):
        sizes.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return ([[("t", [0.0])] for _ in text_list], [1] * len(text_list))

    extractor._extract_embeddings_encoder_batched = fake_batched
    jobs = collect_document_jobs(
        [_Doc("a.txt", "a"), _Doc("b.txt", "b"), _Doc("c.txt", "c"), _Doc("d.txt", "d")],
        {"discourse": False},
    )
    flat = flatten_window_jobs(jobs)
    used = extractor._encode_jobs(flat, {}, 4)
    assert sizes == [4, 2]
    assert used == 2
    assert [job.embeddings[0][0] for job in flat] == ["t", "t", "t", "t"]
