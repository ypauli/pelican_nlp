import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy

nltk.download('wordnet')
class TextNormalizer:
    def __init__(self, options):
        self.options = options

    def normalize(self, tokens):
        method = self.options.get('method')

        if method == 'lemmatization':
            nlp = spacy.load('de_core_news_sm')
            return [nlp(token)[0].lemma_ for token in tokens]
        elif method == 'stemming':
            stemmer =  PorterStemmer
            return [stemmer.stem(token) for token in tokens]
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
