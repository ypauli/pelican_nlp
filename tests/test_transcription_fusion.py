"""Unit tests for primary/default Whisper fusion."""

from pelican_nlp.preprocessing.transcription_fusion import (
    SOURCE_PATCH,
    SOURCE_PRIMARY,
    analyze_text,
    apply_fusion_to_audio_file,
    emitible,
    fuse_transcriptions,
    is_clean,
    build_utterances,
)


def _source(utterances, model, segs=None):
    return {
        "audio_file_path": "/tmp/a.wav",
        "metadata": {
            "length_seconds": 10.0,
            "models_used": {"Transcription": {"model": model}},
        },
        "utterance_data": utterances,
        "speaker_segments": segs or [],
    }


def _utt(text, start, end, speaker="SPEAKER_00"):
    return {"text": text, "start_time": start, "end_time": end, "speaker": speaker}


def test_mandarin_is_clean_when_language_is_chinese():
    text = "首先我會用一個圖片策略"
    q = analyze_text(text, language="chinese")
    assert q.high_severity is False
    assert emitible(text, language="chinese") is True


def test_mandarin_is_tainted_when_language_is_german():
    text = "首先我會用一個圖片策略然後在對這六個"
    q = analyze_text(text, language="german")
    assert q.high_severity is True
    assert any(flag.startswith("non_latin") for flag in q.flags)


def test_english_participant_is_not_english_burst():
    text = "And then with the next one, I did like someone that matches that and that."
    q = analyze_text(text, language="english")
    assert "english_burst" not in q.flags
    assert q.high_severity is False


def test_english_burst_flags_when_language_is_german():
    text = "And then with the next one I did like someone that matches"
    q = analyze_text(text, language="german")
    assert q.high_severity is True
    assert "english_burst" in q.flags


def test_no_language_does_not_drop_other_scripts():
    q = analyze_text("首先我會用一個圖片策略然後", language=None)
    assert q.high_severity is False


def test_short_repetition_is_kept():
    assert emitible("Hallo hallo", language="german") is True
    q = analyze_text("Hallo hallo", language="german")
    assert q.high_severity is False


def test_long_word_loop_is_high_severity():
    text = "und " * 8
    q = analyze_text(text, language="german")
    assert q.high_severity is True


def test_boilerplate_is_high_severity():
    q = analyze_text("Thanks for watching this video please subscribe", language="german")
    assert q.high_severity is True


def test_keeps_clean_primary_without_requiring_agreement():
    primary = _source(
        [_utt("Das ist ein sauberer Satz.", 0.0, 2.0)],
        "openai/whisper-large-v3",
    )
    patch = _source(
        [_utt("Das ist ein anderer Satz.", 0.0, 2.0)],
        "openai/whisper-medium",
    )
    payload = fuse_transcriptions(primary, patch, language="german")
    kept = payload["kept_segments"]
    assert len(kept) == 1
    assert kept[0]["text"] == "Das ist ein sauberer Satz."
    assert kept[0]["chosen_source"] == SOURCE_PRIMARY
    assert kept[0]["decision"] == "keep_primary_clean"


def test_patches_tainted_primary_from_clean_medium():
    loop = "und " * 10
    primary = _source(
        [_utt(loop.strip(), 1.0, 4.0)],
        "openai/whisper-large-v3",
    )
    patch = _source(
        [_utt("Und dann bin ich gegangen.", 1.1, 3.5)],
        "openai/whisper-medium",
    )
    payload = fuse_transcriptions(primary, patch, language="german")
    kept = payload["kept_segments"]
    assert len(kept) == 1
    assert kept[0]["text"] == "Und dann bin ich gegangen."
    assert kept[0]["chosen_source"] == SOURCE_PATCH
    assert kept[0]["decision"] == "patch_from_default_high_taint"
    assert payload["n_dropped"] == 0


def test_does_not_hole_fill_over_kept_primary():
    primary = _source(
        [_utt("Das bleibt stehen.", 0.0, 2.0)],
        "openai/whisper-large-v3",
    )
    patch = _source(
        [_utt("Unerwuenschtes Duplikat.", 0.2, 1.8)],
        "openai/whisper-medium",
    )
    payload = fuse_transcriptions(primary, patch, language="german")
    texts = [row["text"] for row in payload["kept_segments"]]
    assert texts == ["Das bleibt stehen."]


def test_hole_fill_only_outside_primary_spans():
    primary = _source(
        [_utt("Anfang.", 0.0, 1.0)],
        "openai/whisper-large-v3",
    )
    patch = _source(
        [
            _utt("Anfang anders.", 0.0, 1.0),
            _utt("Nur im Loch.", 5.0, 6.0),
        ],
        "openai/whisper-medium",
    )
    payload = fuse_transcriptions(primary, patch, language="german")
    texts = [row["text"] for row in payload["kept_segments"]]
    assert texts == ["Anfang.", "Nur im Loch."]
    decisions = [row["decision"] for row in payload["kept_segments"]]
    assert decisions == ["keep_primary_clean", "hole_fill_from_default"]


def test_both_tainted_is_dropped_and_recorded():
    loop = "go " * 10
    primary = _source([_utt(loop.strip(), 0.0, 3.0)], "openai/whisper-large-v3")
    patch = _source([_utt(loop.strip(), 0.0, 3.0)], "openai/whisper-medium")
    payload = fuse_transcriptions(primary, patch, language="german")
    assert payload["kept_segments"] == []
    assert payload["n_dropped"] == 1
    assert payload["segments"][0]["decision"] == "dropped_both_tainted"
    assert payload["segments"][0]["text"] == ""


def test_identical_utterance_run_is_tainted():
    repeated = [_utt("Ich weiss nicht.", i * 1.0, i * 1.0 + 0.8) for i in range(6)]
    utts = build_utterances({"utterance_data": repeated}, language="german")
    assert all(not is_clean(u, language="german") for u in utts)


def test_apply_fusion_sets_canonical_utterances():
    class Dummy:
        combined_utterances = []
        transcript_text = ""
        models = {}

        def register_model(self, name, params):
            self.models[name] = params

    primary = _source([_utt("Satz eins.", 0.0, 1.0)], "openai/whisper-large-v3")
    payload = fuse_transcriptions(primary, None, language="german")
    audio = Dummy()
    apply_fusion_to_audio_file(audio, payload)
    assert audio.combined_utterances[0]["text"] == "Satz eins."
    assert audio.transcript_text == "Satz eins."
    assert "TranscriptionPatching" in audio.models
