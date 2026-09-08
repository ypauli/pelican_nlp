"""Split cleaned document sections into analysis parts.

Extractors should use this helper instead of reimplementing discourse
speaker filtering.
"""


def split_section_text(section, config, keep_speakertags=False):
    """Return a list of text parts for one cleaned section.

    With ``discourse`` and a participant speaker tag, speaker turns are
    parsed. Otherwise the section is a single part.
    """
    if not config.get("discourse"):
        return [section] if section is not None else []

    from pelican_nlp.preprocessing.speaker_diarization import TextDiarizer

    participant_speakertag = config.get("participant_speakertag")
    if participant_speakertag is None:
        return [section] if section is not None else []
    return TextDiarizer.parse_speaker(section, participant_speakertag, keep_speakertags)


def iter_section_groups(document, config, keep_speakertags=False):
    """Yield ``(section_key, parts)`` from ``document.cleaned_sections``."""
    cleaned_sections = getattr(document, "cleaned_sections", None) or {}
    for key, section in cleaned_sections.items():
        yield key, split_section_text(section, config, keep_speakertags=keep_speakertags)
