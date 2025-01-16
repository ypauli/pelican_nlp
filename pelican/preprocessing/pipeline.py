from pelican.preprocessing.text_cleaner import TextCleaner
from pelican.preprocessing.text_tokenizer import TextTokenizer
from pelican.preprocessing.text_normalizer import TextNormalizer

class TextPreprocessingPipeline:
    def __init__(self, config):

        self.config = config
        self.cleaner = TextCleaner(config['cleaning_options'])
        self.tokenizer_logits = TextTokenizer(config['tokenization_options_logits'])
        self.tokenizer_embeddings = TextTokenizer(config['tokenization_options_embeddings'])
        self.normalizer = TextNormalizer(config['normalization_options'])

    def process_document(self, document, is_dialog=False):

        print('processing document (pipeline.py)')

        # Clean, tokenize, and normalize chapters or whole document
        document.clean_text(self.cleaner, is_dialog=is_dialog)
        if self.config['extract_logits']:
            document.tokenize_text(self.tokenizer_logits, 'logits')
        if self.config['extract_embeddings']:
            document.tokenize_text(self.tokenizer_embeddings, 'embeddings')
        #document.normalize_text(self.normalizer)
