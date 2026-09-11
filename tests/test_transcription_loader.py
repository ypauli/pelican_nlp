from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from pelican_nlp.utils import gpu_budget


def _fake_pipeline_capture(monkeypatch, captured):
    import pelican_nlp.preprocessing.transcription as tr

    def fake_pipeline(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
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
    audio.aggregate_to_utterances = lambda: None
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
