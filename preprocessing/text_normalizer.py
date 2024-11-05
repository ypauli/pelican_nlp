import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('wordnet')
class TextNormalizer:
    def __init__(self, options):
        self.options = options
        self.lemmatizer = WordNetLemmatizer() if self.options.get('method') == 'lemmatization' else None
        self.stemmer = PorterStemmer() if self.options.get('method') == 'stemming' else None

    def normalize(self, tokens):
        method = self.options.get('method')

        #lemmatization cuts off wrong ending (e.g. 'es' wrongly converts to 'e')...
        if method == 'lemmatization':
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        elif method == 'stemming':
            return [self.stemmer.stem(token) for token in tokens]
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
