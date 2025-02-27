from pelican.preprocessing.text_tokenizer import TextTokenizer
from pelican.preprocessing.text_cleaner import TextCleaner, FluencyCleaner
from pelican.preprocessing.text_normalizer import TextNormalizer

class TextPreprocessingPipeline:
    def __init__(self, config):

        self.config = config
        self.pipeline_options = config['pipeline_options']
        self.cleaner = None
        self.normalizer = None
        self.tokenizer = None

    def process_document(self, document):

        print('processing document (pipeline.py)')

        for option in self.pipeline_options:
            if self.pipeline_options.get(option):
                getattr(self, f"_{option}")(document)


    def _clean_text(self, document):
        if self.config['fluency_task']:
            self.cleaner = FluencyCleaner(self.config['cleaning_options'])
        else:
            self.cleaner = TextCleaner(self.config['cleaning_options'])
        document.clean_text(self.cleaner)
    def _quality_check(self, document):
        return
    def _tokenize_text(self, document):
        self.tokenizer = TextTokenizer(self.config['tokenization_options'])
        document.tokenize_text(self.tokenizer)
    def _normalize_text(self, document):
        self.normalizer = TextNormalizer(self.config['normalization_options'])
        document.normalize_text(self.normalizer)