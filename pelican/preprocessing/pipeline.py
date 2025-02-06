from pelican.preprocessing.text_cleaner import TextCleaner
from pelican.preprocessing.text_normalizer import TextNormalizer

class TextPreprocessingPipeline:
    def __init__(self, config):

        self.config = config
        self.cleaner = TextCleaner(config['cleaning_options'])
        self.normalizer = TextNormalizer(config['normalization_options'])

    def process_document(self, document):

        print('processing document (pipeline.py)')

        # Clean, tokenize, and normalize chapters or whole document
        document.clean_text(self.cleaner, document.num_speakers)
        #document.normalize_text(self.normalizer)
