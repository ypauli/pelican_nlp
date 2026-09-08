# Bind the LPDS class explicitly: the module is also named LPDS, so
# `from pelican_nlp.preprocessing import LPDS` would otherwise get the module.
from .LPDS import LPDS


# Import other preprocessing classes lazily so `from pelican_nlp.preprocessing import LPDS`
# does not load the tokenizer (or torch).
def __getattr__(name):
    if name == "TextImporter":
        from .text_importer import TextImporter
        return TextImporter
    if name == "TextCleaner":
        from .text_cleaner import TextCleaner
        return TextCleaner
    if name == "TextTokenizer":
        from .text_tokenizer import TextTokenizer
        return TextTokenizer
    if name == "TextNormalizer":
        from .text_normalizer import TextNormalizer
        return TextNormalizer
    if name == "TextPreprocessingPipeline":
        from .pipeline import TextPreprocessingPipeline
        return TextPreprocessingPipeline
    if name == "SectionIdentificator":
        from .section_identificator import SectionIdentificator
        return SectionIdentificator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TextImporter",
    "TextCleaner",
    "TextTokenizer",
    "TextNormalizer",
    "TextPreprocessingPipeline",
    "LPDS",
    "SectionIdentificator",
]
