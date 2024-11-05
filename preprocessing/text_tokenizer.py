class TextTokenizer:
    def __init__(self, options):
        self.options = options

    def tokenize(self, text):
        method = self.options.get('method', 'whitespace')

        if method == 'whitespace':
            return text.split()
        # Add other tokenization methods like regex or NLTK here
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")
