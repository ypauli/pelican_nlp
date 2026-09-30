"""Regression tests for the audio transcription pipeline.

Each test pins one defect that produced wrong transcripts rather than an error:
silent language drift, forced alignment overwriting the transcript, utterances
that never end, words losing their speaker, and clipped normalization.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pelican_nlp.config_defaults import (
    DEFAULT_TRANSCRIPTION_MODEL,
    apply_config_defaults,
    canonical_asr_model_id,
    resolve_num_speakers,
    resolve_transcription_language,
    resolve_transcription_patching,
)
from pelican_nlp.core.audio_document import AudioFile
from pelican_nlp.utils import gpu_budget


def _audio_file(**kwargs):
    """AudioFile without touching the filesystem."""
    audio = AudioFile.__new__(AudioFile)
    audio.file = kwargs.get("file", "/tmp/example.wav")
    audio.name = os.path.basename(audio.file)
    audio.file_path = os.path.dirname(audio.file)
    audio.target_rms_db = kwargs.get("target_rms_db", -20)
    audio.normalized_path = None
    audio.audio = kwargs.get("audio")
    audio.sample_rate = kwargs.get("sample_rate", 16000)
    audio.chunks = []
    audio.speaker_segments = kwargs.get("speaker_segments", [])
    audio.metadata = {"models_used": {}}
    audio.num_speakers = kwargs.get("num_speakers", 1)
    audio.transcript_text = kwargs.get("transcript_text")
    audio.transcription_file = None
    audio.transcription_text_file = None
    audio.whisper_alignments = kwargs.get("whisper_alignments", [])
    audio.forced_alignments = kwargs.get("forced_alignments", [])
    audio.combined_data = []
    audio.combined_utterances = kwargs.get("combined_utterances", [])
    return audio


def _words(spec):
    """Build word alignments from ``(word, start, end)`` tuples."""
    return [{"word": w, "start_time": s, "end_time": e} for w, s, e in spec]


# --------------------------------------------------------------------------
# Language pinning
# --------------------------------------------------------------------------


def _transcriber(monkeypatch, captured, pipeline_impl=None, **kwargs):
    import pelican_nlp.preprocessing.transcription as tr

    def fake_pipeline(task, **pipeline_kwargs):
        captured["pipeline"] = pipeline_kwargs
        return pipeline_impl if pipeline_impl is not None else (lambda *a, **k: {"text": "", "chunks": []})

    monkeypatch.setattr(tr, "pipeline", fake_pipeline)
    monkeypatch.setattr(tr, "asr_pipeline_preprocessor_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pelican_nlp.utils.gpu_budget.runtime_torch_device", lambda **k: torch.device("cpu")
    )
    monkeypatch.setattr(gpu_budget, "estimate_pretrained_weight_bytes", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_clear_cuda", lambda: None)
    return tr, tr.AudioTranscriber(model="openai/whisper-large-v3", **kwargs)


def test_configured_language_is_pinned_for_generation(monkeypatch):
    calls = []

    class Pipe:
        def __call__(self, _audio, **kwargs):
            calls.append(kwargs)
            return {"text": "hallo welt", "chunks": []}

    _tr, transcriber = _transcriber(monkeypatch, {}, Pipe(), language="german")
    audio = SimpleNamespace(chunks=[_asr_chunk()], register_model=lambda *a, **k: None)
    transcriber.transcribe(audio)

    assert calls[0]["generate_kwargs"] == {"language": "german", "task": "transcribe"}


def test_missing_language_leaves_detection_untouched(monkeypatch):
    calls = []

    class Pipe:
        def __call__(self, _audio, **kwargs):
            calls.append(kwargs)
            return {"text": "hallo", "chunks": []}

    _tr, transcriber = _transcriber(monkeypatch, {}, Pipe(), language="  ")
    audio = SimpleNamespace(chunks=[_asr_chunk()], register_model=lambda *a, **k: None)
    transcriber.transcribe(audio)

    assert transcriber.language is None
    assert "generate_kwargs" not in calls[0]


def test_extra_generate_kwargs_are_forwarded(monkeypatch):
    calls = []

    class Pipe:
        def __call__(self, _audio, **kwargs):
            calls.append(kwargs)
            return {"text": "hallo", "chunks": []}

    _tr, transcriber = _transcriber(
        monkeypatch,
        {},
        Pipe(),
        language="de",
        generate_kwargs={"condition_on_prev_tokens": False},
    )
    audio = SimpleNamespace(chunks=[_asr_chunk()], register_model=lambda *a, **k: None)
    transcriber.transcribe(audio)

    assert calls[0]["generate_kwargs"]["condition_on_prev_tokens"] is False
    assert calls[0]["generate_kwargs"]["language"] == "de"


def test_model_rejecting_language_falls_back_instead_of_failing(monkeypatch):
    calls = []

    class EnglishOnlyPipe:
        def __call__(self, _audio, **kwargs):
            calls.append(kwargs)
            if "generate_kwargs" in kwargs:
                raise ValueError("Cannot specify `task` or `language` for an English-only model")
            return {"text": "hello", "chunks": []}

    _tr, transcriber = _transcriber(monkeypatch, {}, EnglishOnlyPipe(), language="german")
    audio = SimpleNamespace(chunks=[_asr_chunk()], register_model=lambda *a, **k: None)
    transcriber.transcribe(audio)

    assert len(calls) == 2
    assert "generate_kwargs" not in calls[1]
    assert audio.chunks[0].transcript == "hello"


def _asr_chunk():
    class Segment:
        def __len__(self):
            return 5000

        def export(self, buf, format="wav"):
            buf.write(b"RIFF")

    return SimpleNamespace(
        audio_segment=Segment(), transcript="", whisper_alignments=[], start_time=0.0
    )


# --------------------------------------------------------------------------
# Forced alignment keeps readable text
# --------------------------------------------------------------------------


class _Span:
    def __init__(self, start, end, score=1.0):
        self.start = start
        self.end = end
        self.score = score


def test_merge_token_spans_keeps_surface_words_and_scores():
    from pelican_nlp.preprocessing.transcription import _merge_token_spans

    # "Über" and "z.B." keep their casing/punctuation, and the two tokens that
    # "z.B." normalizes to are merged back into a single word entry.
    surface_words = ["Über", "z.B."]
    owners = [0, 1, 1]
    token_spans = [[_Span(0, 10, 0.9)], [_Span(10, 20, 0.5)], [_Span(20, 30, 0.7)]]

    merged = _merge_token_spans(
        token_spans, owners, surface_words, ratio=1.0, sample_rate=10, chunk_start=100.0
    )

    assert [entry["word"] for entry in merged] == ["Über", "z.B."]
    assert merged[0]["start_time"] == pytest.approx(100.0)
    assert merged[1]["start_time"] == pytest.approx(101.0)
    assert merged[1]["end_time"] == pytest.approx(103.0)
    assert merged[0]["score"] == pytest.approx(0.9)
    assert merged[1]["score"] == pytest.approx(0.6)


def test_normalize_uroman_folds_typographic_apostrophes():
    from pelican_nlp.preprocessing.transcription import ForcedAligner

    aligner = ForcedAligner.__new__(ForcedAligner)
    assert ForcedAligner.normalize_uroman(aligner, "Don\u2019t") == "don't"


def test_normalize_language_treats_blank_as_auto_detect():
    from pelican_nlp.preprocessing.transcription import normalize_language

    assert normalize_language("") is None
    assert normalize_language("  Auto ") is None
    assert normalize_language(None) is None
    assert normalize_language(" German ") == "german"


# --------------------------------------------------------------------------
# Utterance aggregation
# --------------------------------------------------------------------------


def test_utterances_split_without_any_punctuation():
    audio = _audio_file()
    audio.combined_data = [
        {**word, "speaker": "SPEAKER_00"}
        for word in _words([("das", 0.0, 0.4), ("ist", 0.4, 0.8), ("gut", 5.0, 5.4)])
    ]
    audio.aggregate_to_utterances(max_gap=1.0)

    assert [u["text"] for u in audio.combined_utterances] == ["das ist", "gut"]


def test_utterances_split_on_cjk_sentence_marks():
    audio = _audio_file()
    audio.combined_data = [
        {**word, "speaker": "SPEAKER_00"}
        for word in _words([("首先\u3002", 0.0, 0.5), ("然後", 0.5, 1.0), ("好\uff01", 1.0, 1.4)])
    ]
    audio.aggregate_to_utterances(max_gap=None)

    assert [u["text"] for u in audio.combined_utterances] == ["首先\u3002", "然後 好\uff01"]


def test_utterances_split_on_speaker_change_mid_sentence():
    audio = _audio_file(num_speakers=2)
    audio.combined_data = [
        {**word, "speaker": speaker}
        for word, speaker in zip(
            _words([("und", 0.0, 0.3), ("dann", 0.3, 0.6), ("genau", 0.6, 1.0), ("ja", 1.0, 1.3)]),
            ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"],
        )
    ]
    audio.aggregate_to_utterances(max_gap=None)

    assert [(u["speaker"], u["text"]) for u in audio.combined_utterances] == [
        ("SPEAKER_00", "und dann"),
        ("SPEAKER_01", "genau ja"),
    ]
    assert all(u["confidence"] == 1.0 for u in audio.combined_utterances)


def test_gap_splitting_can_be_disabled():
    audio = _audio_file()
    audio.combined_data = [
        {**word, "speaker": "SPEAKER_00"}
        for word in _words([("a", 0.0, 0.1), ("b", 60.0, 60.1)])
    ]
    audio.aggregate_to_utterances(max_gap=None)

    assert len(audio.combined_utterances) == 1


# --------------------------------------------------------------------------
# Word to speaker assignment
# --------------------------------------------------------------------------


def test_out_of_order_words_still_get_their_speaker():
    audio = _audio_file(
        num_speakers=2,
        speaker_segments=[
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ],
    )
    # "vier" starts before the words stored ahead of it: Whisper does this at chunk
    # boundaries, and a forward-only segment pointer used to label it UNKNOWN.
    audio.whisper_alignments = _words(
        [("eins", 0.5, 1.0), ("zwei", 6.0, 6.5), ("drei", 6.6, 7.0), ("vier", 1.5, 2.0)]
    )
    audio.combine_alignment_and_diarization("whisper_alignments")

    assert [w["speaker"] for w in audio.combined_data] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_00",
    ]


def test_zero_duration_words_are_assigned_not_dropped():
    audio = _audio_file(
        num_speakers=2,
        speaker_segments=[{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
    )
    audio.whisper_alignments = _words([("kollabiert", 2.0, 2.0)])
    audio.combine_alignment_and_diarization("whisper_alignments")

    assert audio.combined_data[0]["speaker"] == "SPEAKER_00"


def test_diarization_gap_inherits_surrounding_speaker():
    audio = _audio_file(
        num_speakers=2,
        speaker_segments=[
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_00"},
        ],
    )
    audio.whisper_alignments = _words(
        [("eins", 0.1, 0.5), ("luecke", 1.5, 2.0), ("zwei", 3.1, 3.5)]
    )
    audio.combine_alignment_and_diarization("whisper_alignments")

    assert [w["speaker"] for w in audio.combined_data] == ["SPEAKER_00"] * 3


def test_single_word_speaker_flip_is_smoothed():
    words = [
        {"word": "a", "speaker": "SPEAKER_00"},
        {"word": "b", "speaker": "SPEAKER_01"},
        {"word": "c", "speaker": "SPEAKER_00"},
    ]
    assert AudioFile._smooth_word_speakers(words) == 1
    assert [w["speaker"] for w in words] == ["SPEAKER_00"] * 3


def test_genuine_speaker_turns_are_not_smoothed_away():
    words = [
        {"word": "a", "speaker": "SPEAKER_00"},
        {"word": "b", "speaker": "SPEAKER_01"},
        {"word": "c", "speaker": "SPEAKER_01"},
        {"word": "d", "speaker": "SPEAKER_00"},
    ]
    assert AudioFile._smooth_word_speakers(words) == 0


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_normalization_gain_is_capped_to_avoid_clipping(tmp_path):
    # Quiet signal with one loud transient: the raw RMS gain would push the peak
    # far past 1.0 and clip on 16-bit WAV write.
    samples = np.full(16000, 0.01, dtype=np.float32)
    samples[0] = 0.9
    audio = _audio_file(audio=samples, file=str(tmp_path / "quiet.wav"))

    audio.rms_normalization(output_dir=str(tmp_path))

    import soundfile as sf

    written, _ = sf.read(audio.normalized_path)
    assert np.max(np.abs(written)) <= 1.0
    assert audio.metadata["normalization"]["peak_limited"] is True


def test_silent_audio_does_not_produce_invalid_samples(tmp_path):
    audio = _audio_file(
        audio=np.zeros(1600, dtype=np.float32), file=str(tmp_path / "silence.wav")
    )
    audio.rms_normalization(output_dir=str(tmp_path))

    import soundfile as sf

    written, _ = sf.read(audio.normalized_path)
    assert np.all(np.isfinite(written))


def test_normalized_path_never_overwrites_a_non_wav_source(tmp_path):
    source = tmp_path / "interview.mp3"
    source.write_bytes(b"not really audio")
    audio = _audio_file(audio=np.full(1600, 0.1, dtype=np.float32), file=str(source))

    audio.rms_normalization()

    assert audio.normalized_path != str(source)
    assert audio.normalized_path.endswith("_normalized.wav")
    assert source.read_bytes() == b"not really audio"


# --------------------------------------------------------------------------
# Chunk splitting
# --------------------------------------------------------------------------


def test_oversized_span_is_cut_at_silence():
    # Silence midpoint at 9500 ms; a blind split would cut at 10000 ms, mid-word.
    pieces = AudioFile._split_oversized_buffer(0, 20000, 12000, candidate_points=[9500])
    assert pieces == [(0, 9500), (9500, 20000)]


def test_oversized_span_ignores_silence_far_from_the_target():
    # A silence 500 ms in would leave a 19.5 s tail; keep the balanced cut instead.
    pieces = AudioFile._split_oversized_buffer(0, 20000, 12000, candidate_points=[500])
    assert pieces == [(0, 10000), (10000, 20000)]


def test_split_pieces_stay_contiguous_and_gapless():
    pieces = AudioFile._split_oversized_buffer(
        0, 35000, 10000, candidate_points=[4000, 9000, 17000, 26000, 34000]
    )
    assert pieces[0][0] == 0
    assert pieces[-1][1] == 35000
    for left, right in zip(pieces, pieces[1:]):
        assert left[1] == right[0]
    assert all(end - start <= 10000 for start, end in pieces)


def test_adjust_intervals_covers_audio_without_silence_candidates():
    audio = _audio_file()
    intervals = [(0, 100000), (100000, 260000)]
    adjusted = AudioFile._adjust_intervals_by_length(
        audio, intervals, min_length=90000, max_length=120000, candidate_points=[]
    )
    assert adjusted[0][0] == 0
    assert adjusted[-1][1] == 260000
    assert all(end - start <= 120000 for start, end in adjusted)


# --------------------------------------------------------------------------
# Text output
# --------------------------------------------------------------------------


def test_text_output_is_built_from_utterances(tmp_path):
    audio = _audio_file(
        num_speakers=1,
        transcript_text="raw chunk join that may differ",
        combined_utterances=[
            {"text": "Erster Satz.", "speaker": "SPEAKER_0"},
            {"text": "Zweiter Satz.", "speaker": "SPEAKER_0"},
        ],
    )
    out = tmp_path / "out.txt"
    audio.save_as_text(str(out))

    assert out.read_text(encoding="utf-8") == "Erster Satz. Zweiter Satz."


def test_text_output_labels_speakers_when_diarization_found_several(tmp_path):
    audio = _audio_file(
        num_speakers=2,
        transcript_text="ignored",
        combined_utterances=[
            {"text": "Frage?", "speaker": "SPEAKER_00"},
            {"text": "Antwort.", "speaker": "SPEAKER_01"},
        ],
    )
    out = tmp_path / "out.txt"
    audio.save_as_text(str(out))

    assert out.read_text(encoding="utf-8") == "SPEAKER_00: Frage?\nSPEAKER_01: Antwort.\n"


def test_text_output_falls_back_to_raw_transcript(tmp_path):
    audio = _audio_file(transcript_text="only raw text", combined_utterances=[])
    out = tmp_path / "out.txt"
    audio.save_as_text(str(out))

    assert out.read_text(encoding="utf-8") == "only raw text"


# --------------------------------------------------------------------------
# Speaker count resolution
# --------------------------------------------------------------------------


def test_transcription_block_overrides_top_level_speaker_count():
    config = {"number_of_speakers": None, "transcription": {"num_speakers": 3}}
    assert resolve_num_speakers(config) == 3


def test_explicit_null_speaker_count_falls_back_to_one():
    assert resolve_num_speakers({"number_of_speakers": None}) == 1
    assert resolve_num_speakers({}) == 1


def test_transcription_language_uses_top_level_only():
    assert resolve_transcription_language({"language": "german", "transcription": {}}) == "german"
    assert (
        resolve_transcription_language(
            {"language": "german", "transcription": {"language": "english"}}
        )
        == "german"
    )


def test_blank_language_everywhere_means_auto_detect():
    assert resolve_transcription_language({"language": "", "transcription": {}}) is None
    assert resolve_transcription_language({}) is None


def test_config_defaults_propagate_transcription_speaker_count():
    filled = apply_config_defaults(
        {"input_file": "audio", "number_of_speakers": None, "transcription": {"num_speakers": 2}}
    )
    assert filled["number_of_speakers"] == 2


def test_skip_existing_requires_fusion_sidecar_when_patching(tmp_path):
    from pelican_nlp.core.corpus import _transcription_outputs_ready

    json_path = tmp_path / "a_allOutputs.json"
    txt_path = tmp_path / "a_transcript.txt"
    fusion_path = tmp_path / "a_fusion.json"
    json_path.write_text("{}", encoding="utf-8")
    txt_path.write_text("hello", encoding="utf-8")

    assert _transcription_outputs_ready(
        str(json_path),
        str(txt_path),
        str(fusion_path),
        patching_enabled=False,
        primary_model_id="openai/whisper-large-v3",
        patch_model_id="openai/whisper-medium",
    )
    assert not _transcription_outputs_ready(
        str(json_path),
        str(txt_path),
        str(fusion_path),
        patching_enabled=True,
        primary_model_id="openai/whisper-large-v3",
        patch_model_id="openai/whisper-medium",
    )
    fusion_path.write_text(
        '{"models": {"primary": "openai/whisper-large-v3", "patch": "openai/whisper-medium"}}',
        encoding="utf-8",
    )
    assert _transcription_outputs_ready(
        str(json_path),
        str(txt_path),
        str(fusion_path),
        patching_enabled=True,
        primary_model_id="openai/whisper-large-v3",
        patch_model_id="openai/whisper-medium",
    )


def test_canonical_asr_id_maps_short_whisper_names():
    assert canonical_asr_model_id(None) == DEFAULT_TRANSCRIPTION_MODEL
    assert canonical_asr_model_id("") == DEFAULT_TRANSCRIPTION_MODEL
    assert canonical_asr_model_id("whisper-medium") == DEFAULT_TRANSCRIPTION_MODEL
    assert canonical_asr_model_id("openai/whisper-large-v3") == "openai/whisper-large-v3"


def test_transcription_patching_coerces_yaml_yes():
    assert resolve_transcription_patching({"transcription": {"transcription_patching": True}})
    assert resolve_transcription_patching({"transcription": {"transcription_patching": "yes"}})
    assert not resolve_transcription_patching({"transcription": {"transcription_patching": "no"}})
    assert not resolve_transcription_patching({})
    assert not resolve_transcription_patching({"transcription": {}})
