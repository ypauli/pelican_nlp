import spacy

class TextNormalizer:
    def __init__(self, options):
        self.options = options
        self.nlp = None  # Initialize as None, load only when needed

    def _load_model(self):
        """Load spaCy model if not already loaded."""
        if self.nlp is None:
            self.nlp = spacy.load('de_core_news_sm')

    def normalize(self, tokens):
        method = self.options.get('method')

        if method == 'lemmatization':
            self._load_model()  # Load model only when lemmatization is needed
            return [self.nlp(token)[0].lemma_ for token in tokens]
        elif method == 'stemming':
            self._load_model()  # Load model only when stemming is needed
            doc = self.nlp(" ".join(tokens))
            return [token._.stemmed for token in doc]
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
