from types import SimpleNamespace
from unittest.mock import MagicMock

import json
import pytest
import torch

from pelican_nlp.utils import gpu_budget


def _fake_pipeline_capture(monkeypatch, captured):
    import pelican_nlp.preprocessing.transcription as tr

    def fake_pipeline(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    return tr


def test_audio_transcriber_keeps_fp32_when_weights_fit(monkeypatch):
    captured = {}
    tr = _fake_pipeline_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 2 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: True)

    transcriber = tr.AudioTranscriber(model="openai/whisper-large-v3")
    assert captured["task"] == "automatic-speech-recognition"
    assert captured["model"] == "openai/whisper-large-v3"
    assert "torch_dtype" not in captured
    assert transcriber.model == "openai/whisper-large-v3"


def test_audio_transcriber_uses_float16_when_fp32_does_not_fit(monkeypatch):
    captured = {}
    tr = _fake_pipeline_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 20 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: False)

    tr.AudioTranscriber(model="openai/whisper-large-v3")
    assert captured["torch_dtype"] == torch.float16


def test_audio_transcriber_skips_float16_on_cpu(monkeypatch):
    captured = {}
    tr = _fake_pipeline_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cpu"),
    )
    tr.AudioTranscriber(model="openai/whisper-medium")
    assert "torch_dtype" not in captured


def test_audio_transcriber_uses_float16_when_fp32_working_set_does_not_fit(monkeypatch):
    captured = {}
    tr = _fake_pipeline_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 7 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: n < 10 * (1024 ** 3))

    tr.AudioTranscriber(model="openai/whisper-large-v3")
    assert captured["torch_dtype"] == torch.float16


def _chunk(text_export=b"RIFF"):
    class Segment:
        def __len__(self):
            return 5000

        def export(self, buf, format="wav"):
            buf.write(text_export)

    return SimpleNamespace(
        audio_segment=Segment(),
        transcript="",
        whisper_alignments=[],
        start_time=0.0,
    )


def test_audio_transcriber_converts_in_place_to_float16_after_oom(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    converted = {}

    class Model:
        def to(self, *a, **k):
            converted["dtype"] = k.get("dtype") if "dtype" in k else (a[0] if a else None)
            return self

    class Fp32ThenOk:
        def __init__(self):
            self.model = Model()

        def __call__(self, *a, **k):
            if converted.get("dtype") == torch.float16:
                return {"text": "hello", "chunks": []}
            raise torch.cuda.OutOfMemoryError("oom")

    pipe = Fp32ThenOk()
    builds = {"n": 0}

    def fake_pipeline(task, **kwargs):
        builds["n"] += 1
        return pipe

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 2 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: True)
    monkeypatch.setattr(tr, "_clear_cuda", lambda: None)

    transcriber = tr.AudioTranscriber(model="openai/whisper-medium")
    audio = SimpleNamespace(chunks=[_chunk()])
    audio.register_model = lambda *a, **k: None
    transcriber.transcribe(audio)

    assert builds["n"] == 1
    assert converted["dtype"] == torch.float16
    assert audio.chunks[0].transcript == "hello"
    assert transcriber.transcriber is pipe


def test_audio_transcriber_reloads_float16_after_oom(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    dtypes = []

    class BoomPipeline:
        def __call__(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("oom")

    class OkPipeline:
        def __call__(self, *a, **k):
            return {"text": "hello", "chunks": []}

    def fake_pipeline(task, **kwargs):
        dtypes.append(kwargs.get("torch_dtype"))
        if kwargs.get("torch_dtype") == torch.float16:
            return OkPipeline()
        return BoomPipeline()

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 2 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: True)
    monkeypatch.setattr(tr, "_clear_cuda", lambda: None)

    transcriber = tr.AudioTranscriber(model="openai/whisper-medium")
    audio = SimpleNamespace(chunks=[_chunk()])
    audio.register_model = lambda *a, **k: None
    transcriber.transcribe(audio)

    assert torch.float16 in dtypes
    assert audio.chunks[0].transcript == "hello"
    assert transcriber._torch_dtype == torch.float16
    assert transcriber.transcriber is not None


def test_float16_reload_failure_does_not_break_later_chunks(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    class BoomPipeline:
        def __call__(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("oom")

    def fake_pipeline(task, **kwargs):
        if kwargs.get("torch_dtype") == torch.float16:
            raise torch.cuda.OutOfMemoryError("load oom")
        return BoomPipeline()

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 2 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: True)
    monkeypatch.setattr(tr, "_clear_cuda", lambda: None)

    transcriber = tr.AudioTranscriber(model="openai/whisper-medium")
    audio = SimpleNamespace(chunks=[_chunk(), _chunk()])
    audio.register_model = lambda *a, **k: None
    transcriber.transcribe(audio)

    assert hasattr(transcriber, "transcriber")
    assert audio.chunks[0].transcript == ""
    assert audio.chunks[1].transcript == ""


def test_process_single_delays_aligner_until_after_asr(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    order = []

    class DummyTranscriber:
        def transcribe(self, audio_file):
            order.append("asr")
            audio_file.chunks = [SimpleNamespace(transcript="hello")]

        def park_on_cpu(self):
            order.append("park")

        def restore_to_device(self):
            order.append("restore")

    class DummyAligner:
        def __init__(self):
            order.append("aligner_init")

        def align(self, audio_file):
            order.append("align")

    monkeypatch.setattr(tr, "ForcedAligner", DummyAligner)
    monkeypatch.setattr(tr, "SpeakerDiarizer", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(tr, "release_transcription_models", lambda *a, **k: None)

    audio = SimpleNamespace(
        file="/tmp/none.wav",
        chunks=[],
        forced_alignments=[],
        whisper_alignments=[{"word": "hello"}],
        num_speakers=1,
    )
    audio.load_audio = lambda: None
    audio.rms_normalization = lambda output_dir=None: None
    audio.split_on_silence = lambda **k: None
    audio.combine_chunks = lambda: None
    audio.combine_alignment_and_diarization = lambda src: None
    audio.aggregate_to_utterances = lambda **k: None
    audio.register_model = lambda *a, **k: None

    tr.process_single_audio_file(
        audio_file=audio,
        hf_token="",
        num_speakers=1,
        transcription_model="openai/whisper-large-v3",
        transcriber=DummyTranscriber(),
        aligner=None,
        diarizer=None,
        release_models=False,
    )
    assert order == ["asr", "park", "aligner_init", "align", "restore"]


def test_process_single_can_skip_restoring_transcriber(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    order = []

    class DummyTranscriber:
        def transcribe(self, audio_file):
            audio_file.chunks = [SimpleNamespace(transcript="hello")]

        def park_on_cpu(self):
            order.append("park")

        def restore_to_device(self):
            order.append("restore")

    class DummyAligner:
        def align(self, audio_file):
            order.append("align")

        def park_on_cpu(self):
            order.append("aligner_park")

        def restore_to_device(self):
            order.append("aligner_restore")

    monkeypatch.setattr(tr, "ForcedAligner", DummyAligner)
    monkeypatch.setattr(tr, "SpeakerDiarizer", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(tr, "release_transcription_models", lambda *a, **k: None)

    audio = SimpleNamespace(
        file="/tmp/none.wav",
        chunks=[],
        forced_alignments=[],
        whisper_alignments=[{"word": "hello"}],
        num_speakers=1,
    )
    audio.load_audio = lambda: None
    audio.rms_normalization = lambda output_dir=None: None
    audio.split_on_silence = lambda **k: None
    audio.combine_chunks = lambda: None
    audio.combine_alignment_and_diarization = lambda src: None
    audio.aggregate_to_utterances = lambda **k: None
    audio.register_model = lambda *a, **k: None

    tr.process_single_audio_file(
        audio_file=audio,
        hf_token="",
        num_speakers=1,
        transcriber=DummyTranscriber(),
        aligner=DummyAligner(),
        diarizer=None,
        release_models=False,
        restore_transcriber=False,
    )
    assert "restore" not in order
    assert "park" in order


def test_independent_patch_stores_then_unloads_before_second_model(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    order = []
    stored = []

    class PrimaryFile:
        file_path = "/tmp"
        name = "a.wav"
        file = "/tmp/a.wav"
        target_rms_db = -20
        participant_ID = None
        source_folder = None
        unit_kind = None
        task = None
        num_speakers = 1
        _normalized_audio_dir = "/tmp/norm"

    def fake_process(audio_file, **kwargs):
        order.append(("process", kwargs.get("transcription_model"), audio_file is primary))
        return audio_file

    def fake_unload(pool):
        order.append("unload")
        pool.clear()

    def fake_clone(audio_file):
        order.append("clone")
        clone = PrimaryFile()
        clone.name = "clone.wav"
        return clone

    monkeypatch.setattr(tr, "process_single_audio_file", fake_process)
    monkeypatch.setattr(tr, "unload_transcription_runtime", fake_unload)
    monkeypatch.setattr(tr, "clone_audio_file_for_independent_run", fake_clone)

    primary = PrimaryFile()

    def store_primary(processed):
        stored.append(processed)
        order.append("store")

    first, second = tr.transcribe_with_independent_patch(
        primary,
        process_kwargs={"hf_token": ""},
        primary_model="openai/whisper-large-v3",
        patch_model="openai/whisper-medium",
        store_primary=store_primary,
    )
    assert stored == [primary]
    assert first is primary
    assert second is not primary
    assert order == [
        ("process", "openai/whisper-large-v3", True),
        "store",
        "unload",
        "clone",
        ("process", "openai/whisper-medium", False),
        "unload",
    ]


def test_word_timestamp_merge_error_falls_back_to_segment_timestamps(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    class MergeBugPipeline:
        def __call__(self, *a, **kwargs):
            if kwargs.get("return_timestamps") is True:
                return {"text": "hello world", "chunks": []}
            raise TypeError(
                "'<=' not supported between instances of 'NoneType' and 'float'"
            )

    monkeypatch.setattr(tr, "pipeline", lambda *a, **k: MergeBugPipeline())
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 2 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: True)
    monkeypatch.setattr(tr, "_clear_cuda", lambda: None)

    transcriber = tr.AudioTranscriber(model="openai/whisper-medium")
    audio = SimpleNamespace(chunks=[_chunk()])
    audio.register_model = lambda *a, **k: None
    transcriber.transcribe(audio)

    chunk = audio.chunks[0]
    assert chunk.transcript == "hello world"
    assert len(chunk.whisper_alignments) == 2
    assert chunk.whisper_alignments[0]["word"] == "hello"


def test_tokenizer_overrides_list_extra_special_tokens(tmp_path):
    from pelican_nlp.preprocessing.transcription import asr_tokenizer_from_pretrained_kwargs

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|de|>", "<|transcribe|>"]}),
        encoding="utf-8",
    )
    assert asr_tokenizer_from_pretrained_kwargs(str(tmp_path), {}) == {
        "extra_special_tokens": {}
    }


def test_feature_extractor_from_nested_processor_config(tmp_path):
    from pelican_nlp.preprocessing.transcription import load_asr_feature_extractor

    (tmp_path / "processor_config.json").write_text(
        json.dumps({
            "processor_class": "WhisperProcessor",
            "feature_extractor": {
                "chunk_length": 30,
                "dither": 0.0,
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": 128,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_side": "right",
                "padding_value": 0.0,
                "return_attention_mask": False,
                "sampling_rate": 16000,
            },
        }),
        encoding="utf-8",
    )
    extractor = load_asr_feature_extractor(str(tmp_path), {})
    assert extractor.feature_size == 128
    assert extractor.sampling_rate == 16000
    assert extractor.chunk_length == 30


def test_pipeline_receives_compat_preprocessors(monkeypatch):
    captured = {}
    tokenizer = object()
    feature_extractor = object()
    tr = _fake_pipeline_capture(monkeypatch, captured)
    monkeypatch.setattr(
        tr,
        "asr_pipeline_preprocessor_kwargs",
        lambda *a, **k: {"tokenizer": tokenizer, "feature_extractor": feature_extractor},
    )
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cpu"),
    )
    tr.AudioTranscriber(model="Flix-AI/flix-swissgerman-full")
    assert captured["tokenizer"] is tokenizer
    assert captured["feature_extractor"] is feature_extractor
    assert captured["model"] == "Flix-AI/flix-swissgerman-full"


def test_transcribe_accepts_list_pipeline_output(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    class ListPipeline:
        def __call__(self, *a, **k):
            return [
                {"text": "hallo", "chunks": [{"text": "hallo", "timestamp": (0.0, 0.2)}]},
                {"text": "welt", "chunks": [{"text": "welt", "timestamp": (0.2, 0.4)}]},
            ]

    monkeypatch.setattr(tr, "pipeline", lambda *a, **k: ListPipeline())
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cpu"),
    )
    monkeypatch.setattr(tr, "_clear_cuda", lambda: None)

    transcriber = tr.AudioTranscriber(model="openai/whisper-medium")
    audio = SimpleNamespace(chunks=[_chunk()])
    audio.register_model = lambda *a, **k: None
    transcriber.transcribe(audio)
    assert audio.chunks[0].transcript == "hallo welt"
    assert [item["word"] for item in audio.chunks[0].whisper_alignments] == ["hallo", "welt"]


def test_sanitize_drops_null_forced_decoder_ids():
    from pelican_nlp.preprocessing.transcription import _sanitize_whisper_generation_config

    pipe = SimpleNamespace(
        generation_config=SimpleNamespace(
            forced_decoder_ids=[[1, None], [2, 50360]],
        )
    )
    _sanitize_whisper_generation_config(pipe)
    assert pipe.generation_config.forced_decoder_ids == [[2, 50360]]


def test_transcribe_audio_raises_when_every_file_fails(tmp_path, monkeypatch):
    from pelican_nlp.core.corpus import Corpus

    wav = tmp_path / "part-VP1_task-interview_n-1.wav"
    wav.write_bytes(b"RIFF")
    document = SimpleNamespace(
        file=str(wav),
        name=wav.name,
        source_folder="part-VP1",
    )

    def boom(*args, **kwargs):
        raise AttributeError("'list' object has no attribute 'keys'")

    monkeypatch.setattr("pelican_nlp.extras.require_extra", lambda *a, **k: None)
    monkeypatch.setattr(
        "pelican_nlp.preprocessing.transcription.AudioTranscriber",
        boom,
    )

    corpus = Corpus(
        "part-VP1",
        [document],
        {
            "transcription": {
                "transcription_model": "Flix-AI/flix-swissgerman-full",
                "hf_token": "x",
            }
        },
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="produced no outputs"):
        corpus.transcribe_audio()
    assert not (tmp_path / "derivatives" / "transcription" / f"{wav.stem}_transcript.txt").exists()

