"""Apple Silicon memory release during transcription.

CUDA ``empty_cache`` does not run on MPS, and the MPS caching allocator keeps
freed blocks in unified memory until ``torch.mps.empty_cache()``. These tests
drive the real cleanup branches with MPS reported available and a budget that
fits one resident model plus one file of activations — the next chunk, the
aligner, and the next file succeed only when that cache is released.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not hasattr(torch.backends, "mps"),
    reason="this torch build has no MPS backend",
)


class _Budget:
    """Unified-memory stand-in: freed MPS blocks stay reserved until empty_cache."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.live = 0
        self.cached = 0
        self.mps_empty_calls = 0
        self.cuda_empty_calls = 0

    def allocate(self, nbytes):
        if self.live + self.cached + nbytes > self.capacity:
            raise RuntimeError(
                "MPS backend out of memory: "
                f"tried to allocate {nbytes} with live={self.live} "
                f"cached={self.cached} capacity={self.capacity}"
            )
        self.live += nbytes

    def release(self, nbytes):
        self.live -= nbytes
        self.cached += nbytes

    def empty_cache(self):
        self.cached = 0
        self.mps_empty_calls += 1

    def note_cuda(self):
        self.cuda_empty_calls += 1


class _Weight:
    def __init__(self, budget, nbytes):
        self.budget = budget
        self.nbytes = nbytes
        self.device = "cpu"
        self.to("mps")

    def to(self, *args, **kwargs):
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        else:
            device = "cpu"
        target = getattr(device, "type", None) or str(device)
        if target == self.device:
            return self
        if self.device == "mps":
            self.budget.release(self.nbytes)
        if target == "mps":
            self.budget.allocate(self.nbytes)
        self.device = target
        return self

    def __del__(self):
        budget = getattr(self, "budget", None)
        if budget is None or getattr(self, "device", None) != "mps":
            return
        nbytes = getattr(self, "nbytes", 0)
        self.device = "cpu"
        self.nbytes = 0
        if nbytes:
            budget.release(nbytes)


def _use_apple_silicon(monkeypatch, budget):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", budget.empty_cache)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", budget.note_cuda)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def _chunk():
    class Segment:
        def __len__(self):
            return 5000

        def export(self, buf, format="wav"):
            buf.write(b"RIFFstub")

    return SimpleNamespace(
        audio_segment=Segment(),
        transcript="",
        whisper_alignments=[],
        start_time=0.0,
    )


def test_clear_cuda_releases_mps_cache_and_leaves_cuda_alone(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    calls = []
    budget = _Budget(10)
    _use_apple_silicon(monkeypatch, budget)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))

    tr._clear_cuda()

    assert calls == ["mps", "sync", "mps"]
    assert budget.cuda_empty_calls == 0


def test_clear_cuda_prefers_cuda_when_both_devices_report_available(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("cuda-sync"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))

    tr._clear_cuda()

    assert calls == ["cuda", "cuda-sync", "cuda"]
    assert "mps" not in calls


def test_clear_cuda_is_a_noop_without_an_accelerator(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    tr._clear_cuda()


def test_release_transcription_models_returns_mps_memory(monkeypatch):
    import pelican_nlp.preprocessing.transcription as tr

    budget = _Budget(capacity=100)
    _use_apple_silicon(monkeypatch, budget)
    holder = SimpleNamespace(transcriber=SimpleNamespace(model=_Weight(budget, 40)))
    assert budget.live == 40

    tr.release_transcription_models(holder, None, None)

    assert budget.live == 0
    assert budget.cached == 0
    assert budget.mps_empty_calls >= 1
    assert budget.cuda_empty_calls == 0


def test_chunk_and_park_fallback_release_mps_cache(monkeypatch):
    """Per-chunk cleanup and the no-park fallback both have their own CUDA branch."""
    import pelican_nlp.preprocessing.transcription as tr

    budget = _Budget(capacity=100)
    _use_apple_silicon(monkeypatch, budget)

    class Pipe:
        def __init__(self):
            self.model = _Weight(budget, 40)

        def __call__(self, wav_bytes, **kwargs):
            budget.allocate(50)
            budget.release(50)
            return {"text": "guten tag", "chunks": []}

    transcriber = tr.AudioTranscriber.__new__(tr.AudioTranscriber)
    transcriber.device = torch.device("mps")
    transcriber.model = "openai/whisper-medium"
    transcriber.language = None
    transcriber.extra_generate_kwargs = {}
    transcriber._generate_kwargs_disabled = False
    transcriber._torch_dtype = None
    transcriber.transcriber = Pipe()

    audio = SimpleNamespace(chunks=[_chunk(), _chunk()], register_model=lambda *a, **k: None)
    transcriber.transcribe(audio)

    assert [chunk.transcript for chunk in audio.chunks] == ["guten tag", "guten tag"]
    assert budget.cached == 0
    assert budget.cuda_empty_calls == 0

    class BareTranscriber:
        def transcribe(self, audio_file):
            audio_file.chunks = [SimpleNamespace(transcript="hello")]

    class BareAligner:
        def align(self, audio_file):
            audio_file.forced_alignments = []

    bare_audio = SimpleNamespace(
        file="/tmp/none.wav",
        chunks=[],
        forced_alignments=[],
        whisper_alignments=[],
        num_speakers=1,
    )
    bare_audio.load_audio = lambda: None
    bare_audio.rms_normalization = lambda output_dir=None: None
    bare_audio.split_on_silence = lambda **k: None
    bare_audio.combine_chunks = lambda: None
    bare_audio.combine_alignment_and_diarization = lambda src: None
    bare_audio.aggregate_to_utterances = lambda **k: None
    bare_audio.register_model = lambda *a, **k: None
    before = budget.mps_empty_calls
    tr.process_single_audio_file(
        audio_file=bare_audio,
        hf_token="",
        num_speakers=1,
        transcriber=BareTranscriber(),
        aligner=BareAligner(),
        diarizer=None,
        release_models=False,
    )
    assert budget.mps_empty_calls == before + 1


def test_mac_transcription_succeeds_across_files(tmp_path, monkeypatch):
    """Two files on a unified-memory budget that cannot hold a leaked MPS cache.

    Whisper weights are 40, each file holds 20 of audio, a chunk allocates 30,
    and the aligner needs 55. Capacity is 100, so:

    * the second chunk OOMs unless the per-chunk branch calls ``empty_cache``
    * the aligner OOMs unless parking calls ``empty_cache``
    * the second file OOMs unless the corpus loop calls ``empty_cache``
    * weights stay cached at the end unless model release calls ``empty_cache``
    """
    import pelican_nlp.preprocessing.transcription as tr
    from pelican_nlp.core.corpus import Corpus

    whisper, audio_bytes, activation, aligner_bytes, capacity = 40, 20, 30, 55, 100
    budget = _Budget(capacity)
    _use_apple_silicon(monkeypatch, budget)
    monkeypatch.setattr("pelican_nlp.extras.require_extra", lambda *a, **k: None)

    class Pipe:
        def __init__(self):
            self.model = _Weight(budget, whisper)

        def __call__(self, wav_bytes, **kwargs):
            budget.allocate(activation)
            budget.release(activation)
            return {
                "text": "guten tag",
                "chunks": [
                    {"text": "guten", "timestamp": (0.0, 0.4)},
                    {"text": "tag", "timestamp": (0.4, 0.8)},
                ],
            }

    class MacTranscriber(tr.AudioTranscriber):
        def __init__(self, model=None, language=None, generate_kwargs=None):
            self.device = torch.device("mps")
            self.model = model or "openai/whisper-medium"
            self.language = tr.normalize_language(language)
            self.extra_generate_kwargs = dict(generate_kwargs or {})
            self._generate_kwargs_disabled = False
            self._torch_dtype = None
            self.transcriber = Pipe()

    class MacAligner:
        def __init__(self, device=None):
            self.device = torch.device("mps")
            self.model = _Weight(budget, aligner_bytes)

        def align(self, audio_file):
            audio_file.forced_alignments = []

        park_on_cpu = tr.ForcedAligner.park_on_cpu
        restore_to_device = tr.ForcedAligner.restore_to_device

    monkeypatch.setattr(tr, "AudioTranscriber", MacTranscriber)
    monkeypatch.setattr(tr, "ForcedAligner", MacAligner)

    class Recording:
        def __init__(self, path):
            self.file = str(path)
            self.name = path.name
            self.file_path = str(path.parent)
            self.source_folder = "part-VP1"
            self.num_speakers = 1
            self._held = 0
            self.chunks = []
            self.forced_alignments = []
            self.whisper_alignments = []
            self.combined_data = []
            self.combined_utterances = []
            self.transcript_text = ""
            self.speaker_segments = []
            self.metadata = {"models_used": {}}

        def load_audio(self):
            budget.allocate(audio_bytes)
            self._held = audio_bytes

        def clear_audio_data(self):
            if self._held:
                budget.release(self._held)
                self._held = 0

        def rms_normalization(self, output_dir=None):
            return None

        def split_on_silence(self, **kwargs):
            self.chunks = [_chunk(), _chunk()]

        def combine_chunks(self):
            self.transcript_text = " ".join(
                chunk.transcript for chunk in self.chunks if getattr(chunk, "transcript", "")
            ).strip()

        def combine_alignment_and_diarization(self, source):
            self.combined_data = [{"text": self.transcript_text}]

        def aggregate_to_utterances(self, max_gap=1.0):
            self.combined_utterances = [{"text": self.transcript_text, "speaker": "SPEAKER_00"}]

        def register_model(self, *args, **kwargs):
            return None

        def save_as_json(self, output_file):
            Path(output_file).write_text(self.transcript_text, encoding="utf-8")

        def save_as_text(self, output_file):
            Path(output_file).write_text(self.transcript_text + "\n", encoding="utf-8")

    documents = []
    for name in ("part-VP1_task-interview_n-1.wav", "part-VP1_task-interview_n-2.wav"):
        path = tmp_path / name
        path.write_bytes(b"RIFF")
        documents.append(Recording(path))

    corpus = Corpus(
        "part-VP1",
        documents,
        {
            "input_file": "audio",
            "language": "german",
            "number_of_speakers": 1,
            "transcription": {
                "transcription_model": "openai/whisper-medium",
                "hf_token": "",
                "num_speakers": 1,
            },
        },
        tmp_path,
    )
    corpus.transcribe_audio()

    transcription_dir = tmp_path / "derivatives" / "transcription"
    for document in documents:
        text_path = transcription_dir / f"{Path(document.file).stem}_transcript.txt"
        assert text_path.read_text(encoding="utf-8").strip() == "guten tag guten tag"
    assert budget.cuda_empty_calls == 0
    assert budget.mps_empty_calls >= 1
    assert budget.live == 0
    assert budget.cached == 0


def test_pipeline_gpu_clear_uses_mps_when_cuda_is_absent(monkeypatch):
    from pelican_nlp.main import Pelican

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    Pelican._clear_gpu_memory()

    assert calls == ["mps"]
