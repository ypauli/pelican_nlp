# Import core classes lazily so `import pelican_nlp.core.document` does not load models.
def __getattr__(name):
    if name == "Corpus":
        from .corpus import Corpus
        return Corpus
    if name == "Document":
        from .document import Document
        return Document
    if name == "AudioFile":
        from .audio_document import AudioFile
        return AudioFile
    if name == "Participant":
        from .participant import Participant
        return Participant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Corpus", "Document", "AudioFile", "Participant"]
