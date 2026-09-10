from pelican_nlp.config import debug_print


class TextPreprocessingPipeline:
    """Pipeline for text preprocessing operations."""
    
    def __init__(self, config):
        """Initialize pipeline with configuration.
        
        Args:
            config: Dictionary of configuration options
        """
        self.config = config
        self.pipeline_options = config.get('pipeline_options', {})
        self.cleaner = None
        self.normalizer = None
        self.tokenizer = None

    def process_document(self, document):
        """Process a document through configured pipeline steps.
        
        Args:
            document: Document object to process
        """
        debug_print('Processing document (pipeline.py)')
        
        if not self.pipeline_options:
            from pelican_nlp.utils.progress import active_reporter

            active_reporter().warn(
                "No pipeline_options found in config. Skipping preprocessing pipeline."
            )
            return
        
        for option, enabled in self.pipeline_options.items():
            if enabled:
                processor = getattr(self, f"_{option}")
                processor(document)

    def _clean_text(self, document):
        """Clean document text."""
        from pelican_nlp.preprocessing.text_cleaner import TextCleaner

        self.cleaner = TextCleaner(self.config['cleaning_options'])
        document.clean_text(self.cleaner)

    def _tokenize_text(self, document):
        """Tokenize document text."""
        from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer

        opts = self.config.get("tokenization_options") or {}
        method = opts.get("method", "whitespace")
        self.tokenizer = TextTokenizer(
            method,
            model_name=opts.get("model_name"),
            max_length=opts.get("max_length"),
            trust_remote_code=opts.get("trust_remote_code", False),
        )
        document.tokenize_text(self.tokenizer, purpose=opts.get("purpose", "embeddings"))

    def _normalize_text(self, document):
        """Normalize document text."""
        from pelican_nlp.preprocessing.text_normalizer import TextNormalizer

        self.normalizer = TextNormalizer(self.config['normalization_options'])
        document.normalize_text(self.normalizer)

    def _quality_check(self, document):
        """Placeholder for quality check implementation."""
        pass
