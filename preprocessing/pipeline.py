from preprocessing.text_cleaner import TextCleaner
from preprocessing.text_tokenizer import TextTokenizer
from preprocessing.text_normalizer import TextNormalizer

class TextPreprocessingPipeline:
    def __init__(self, config):

        self.cleaner = TextCleaner(config['cleaning_options'])
        self.tokenizer = TextTokenizer(config['tokenization_options'])
        self.normalizer = TextNormalizer(config['normalization_options'])

    def process_document(self, document, is_dialog=False):
        """
        Processes a single document, handling individual chapters if present.
        """
        # Clean, tokenize, and normalize chapters or whole document
        document.clean_text(self.cleaner, is_dialog=is_dialog)
        document.tokenize_text(self.tokenizer)
        #document.normalize_text(self.normalizer)

        # Return processed chapters if available, otherwise whole document
        return document.get_processed_text()
