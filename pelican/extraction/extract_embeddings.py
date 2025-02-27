from pelican.extraction.language_model import Model
from pelican.preprocessing.text_tokenizer import TextTokenizer
from pelican.preprocessing.text_cleaner import TextCleaner

class EmbeddingsExtractor:
    def __init__(self, embeddings_configurations, project_path):
        self.embeddings_configurations = embeddings_configurations
        self.model_name = embeddings_configurations['model_name']  # Embedding model instance (e.g., fastText, RoBERTa)
        self.model = Model(self.model_name, project_path)
        self.Tokenizer = None

    def extract_embeddings_from_text(self, text_list):

        doc_entry_list = []

        self.model.load_model()
        model = self.model.model_instance

        self.Tokenizer = TextTokenizer(self.embeddings_configurations['tokenization_method'], self.model_name,
                                       self.embeddings_configurations['max_length'])

        for text in text_list:

            print(f'The text is: {text}')

            embeddings = {}

            # Tokenize the input text
            inputs = self.Tokenizer.tokenize_text(text)
            #print(f'inputs: {inputs}')

            if self.embeddings_configurations['pytorch_based_model']:
                #e.g. RoBERTa Model
                import torch
                with torch.no_grad():
                    outputs = model(**inputs)

                #print(f'outputs: {outputs}')

                # Get word embeddings (last hidden state)
                word_embeddings = outputs.last_hidden_state

                # Extract input_ids and convert them back to tokens
                input_ids = inputs['input_ids'][0].tolist()
                tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(input_ids)

                print(f'Tokens backconversion: {tokens}')

                # Now align the tokens and embeddings
                for token, embedding in zip(tokens, word_embeddings[0]):
                    embeddings[token]=embedding.tolist()

            else:
                if self.model_name == 'fastText':
                    for token in inputs:
                        token = TextCleaner._remove_punctuation(token)
                        token = TextCleaner._lowercase(token)
                        embeddings[token]=model.get_word_vector(token)

            doc_entry_list.append(embeddings)

        return doc_entry_list