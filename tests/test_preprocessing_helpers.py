from pelican_nlp.preprocessing.text_cleaner import (
    FluencyCleaner,
    TextCleaner,
    _remove_brackets_and_bracketcontent,
    lowercase,
    remove_punctuation,
    remove_timestamps,
)
from pelican_nlp.preprocessing.speaker_diarization import TextDiarizer


class _Doc:
    fluency_duplicate_count = None
    number_of_duplicates = None
    number_of_hyphenated_words = None


def test_lowercase_string_and_list():
    assert lowercase("AbC") == "abc"
    assert lowercase(["Ab", "C"]) == ["ab", "c"]


def test_remove_punctuation():
    assert remove_punctuation("hello, world!") == "hello world"
    assert remove_punctuation(["hello,", "world!"]) == ["hello", "world"]


def test_remove_brackets():
    text = "keep (drop) and [also] {this} <x>"
    cleaned = _remove_brackets_and_bracketcontent(text)
    assert "drop" not in cleaned
    assert "keep" in cleaned


def test_remove_timestamps():
    text = "hello #00:00:23-00# world"
    assert remove_timestamps(text, "#00:00:23-00#") == "hello  world"


def test_general_cleaner_lowercase():
    cleaner = TextCleaner(
        {
            "remove_timestamps": False,
            "lowercase": True,
            "general_cleaning": True,
            "fluency_task": False,
        }
    )
    assert cleaner.clean(_Doc(), "Hello   World") == "hello world"


def test_fluency_cleaner_splits_and_dedupes():
    doc = _Doc()
    cleaner = FluencyCleaner(
        {
            "word_splitter": ";",
            "remove_hyphens": True,
            "remove_duplicates": True,
            "lowercase": True,
        }
    )
    result = cleaner.cleanFluency(doc, "Katze; Hund; Katze; A-ffe")
    assert result == "katze hund affe"
    assert doc.fluency_duplicate_count == 1


def test_parse_speaker_keeps_matching_turns():
    text = "A: hello there\nB: I am the participant\nA: thanks"
    parts = TextDiarizer.parse_speaker(text, "B", keep_speakertag=False)
    assert parts == ["I am the participant"]


def test_parse_speaker_no_tags_returns_plain_text():
    assert TextDiarizer.parse_speaker("plain text", "B") == ["plain text"]


def test_parse_speaker_unmatched_tags_returns_empty():
    text = "A: only investigator"
    assert TextDiarizer.parse_speaker(text, "B") == []
