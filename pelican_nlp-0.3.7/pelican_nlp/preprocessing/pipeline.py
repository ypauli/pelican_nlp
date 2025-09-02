from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer
from pelican_nlp.preprocessing.text_cleaner import TextCleaner, FluencyCleaner
from pelican_nlp.preprocessing.text_normalizer import TextNormalizer

class TextPreprocessingPipeline:
    """Pipeline for text preprocessing operations."""
    
    def __init__(self, config):
        """Initialize pipeline with configuration.
        
        Args:
            config: Dictionary of configuration options
        """
        self.config = config
        self.pipeline_options = config.get('pipeline_options')
        self.cleaner = None
        self.normalizer = None
        self.tokenizer = None

    def process_document(self, document):
        """Process a document through configured pipeline steps.
        
        Args:
            document: Document object to process
        """
        print('Processing document (pipeline.py)')
        
        for option, enabled in self.pipeline_options.items():
            if enabled:
                processor = getattr(self, f"_{option}")
                processor(document)

    def _clean_text(self, document):
        """Clean document text."""
        self.cleaner = TextCleaner(self.config['cleaning_options'])
        document.clean_text(self.cleaner)

    def _tokenize_text(self, document):
        """Tokenize document text."""
        self.tokenizer = TextTokenizer(self.config['tokenization_options'])
        document.tokenize_text(self.tokenizer)

    def _normalize_text(self, document):
        """Normalize document text."""
        self.normalizer = TextNormalizer(self.config['normalization_options'])
        document.normalize_text(self.normalizer)

    def _quality_check(self, document):
        """Placeholder for quality check implementation."""
        pass