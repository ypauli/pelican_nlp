from .main import Pelican  # Now importing Pelican

from .preprocessing import LPDS
from .preprocessing import TextCleaner
from .preprocessing import TextImporter
from .preprocessing import TextTokenizer
from .preprocessing import TextNormalizer
from .preprocessing import TextPreprocessingPipeline

from .core.corpus import Corpus
from .core.document import Document
from .core.subject import Subject
from .core.audio_document import AudioFile

from .extraction.extract_embeddings import EmbeddingsExtractor
from .extraction.extract_acoustic_features import AudioFeatureExtractor
from .extraction.extract_logits import LogitsExtractor

# Version and metadata
__version__ = "0.1.0"
__author__ = "Yves Pauli"

