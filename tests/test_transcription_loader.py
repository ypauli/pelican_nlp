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

    class Segment:
        def __len__(self):
            return 5000

        def export(self, buf, format="wav"):
            buf.write(b"RIFF")

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device",
        lambda **k: torch.device("cuda"),
    )
    monkeypatch.setattr(
        gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: 2 * (1024 ** 3)
    )
    monkeypatch.setattr(gpu_budget, "gpu_can_hold", lambda n: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    transcriber = tr.AudioTranscriber(model="openai/whisper-medium")
    chunk = SimpleNamespace(
        audio_segment=Segment(),
        transcript="",
        whisper_alignments=[],
        start_time=0.0,
    )
    audio = SimpleNamespace(chunks=[chunk])
    audio.register_model = lambda *a, **k: None
    transcriber.transcribe(audio)

    assert torch.float16 in dtypes
    assert chunk.transcript == "hello"
    assert transcriber._torch_dtype == torch.float16


def test_process_single_delays_aligner_until_after_asr(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    order = []

    class DummyTranscriber:
        def transcribe(self, audio_file):
            order.append("asr")
            audio_file.chunks = [SimpleNamespace(transcript="hello")]

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
    assert order == ["asr", "aligner_init", "align"]
