from transformers import AutoTokenizer, LlamaTokenizer
import torch

class TextTokenizer:
    def __init__(self, options):
        self.options = options
        self.device_used = 'cuda' if torch.cuda.is_available()==True else 'cpu'
        self.model = None
        self.tokenizer = None

    def tokenize(self, text):
        method = self.options.get('method')

        if self.options.get('remove_punctuation', True):
            text = self._remove_punctuation(text)
        if self.options.get('lowercase', True):
            text = text.lower()

        if method == 'whitespace':
            return text.split()
        if method == 'model_instance':
            self.model = self.options.get('model_name')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            return self.tokenizer.encode(text, return_tensors='pt').to(self.device_used)
        # Add other tokenization methods like regex or NLTK here
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")

    def convert_IDs_to_tokens(self, chunk):
        return self.tokenizer.convert_ids_to_tokens(chunk)
    
    def _remove_punctuation(self, text):
        import string
        return text.translate(str.maketrans('', '', string.punctuation))
