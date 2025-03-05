import torch

class TextTokenizer:
    def __init__(self, method, model_name=None, max_length=None):
        self.tokenization_method = method
        self.model_name = model_name
        self.max_sequence_length=max_length

        self.tokenizer = self.get_tokenizer()

    def tokenize_text(self, text):

        method = self.tokenization_method

        if not isinstance(text, str):
            raise ValueError(f"to tokenize a text it must be a in string format, but it is in format {type(text)}")

        if method == 'whitespace':
            # Tokenize by whitespace
            return text.split()
        elif method == 'model_roberta':
            # Tokenize using the model's tokenizer
            return self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=self.max_sequence_length).to(self.device_used)
        elif method == 'model':
            return self.tokenizer.encode(text, return_tensors='pt')
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_tokenizer(self):
        if self.tokenization_method == 'model':
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.model_name) #, use_fast=True
        else:
            return None
