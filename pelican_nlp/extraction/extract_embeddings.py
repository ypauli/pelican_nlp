from pelican_nlp.extraction.language_model import Model
from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer

class EmbeddingsExtractor:
    def __init__(self, embeddings_configurations, project_path):
        self.embeddings_configurations = embeddings_configurations
        self.model_name = embeddings_configurations['model_name']  # Embedding model instance (e.g., fastText, RoBERTa)
        self.model = Model(self.model_name, project_path)
        self.Tokenizer = TextTokenizer(self.embeddings_configurations['tokenization_method'], self.model_name,
                                       self.embeddings_configurations['max_length'])

        self.model.load_model()
        self.model_instance = self.model.model_instance

    def extract_embeddings_from_text(self, text_list):

        doc_entry_list = []

        for text in text_list:

            embeddings = {}

            # Tokenize the input text
            inputs = self.Tokenizer.tokenize_text(text)
            print(f'inputs are: {inputs}')

            if self.embeddings_configurations['pytorch_based_model']:
                #e.g. RoBERTa Model or Llama Model
                import torch
                with torch.no_grad():
                    if 'llama' in self.model_name.lower():
                        # Handle Llama models which expect input_ids directly
                        outputs = self.model_instance(input_ids=inputs['input_ids'])
                    else:
                        # Handle RoBERTa and other models that accept **inputs
                        outputs = self.model_instance(**inputs)

                # Get word embeddings (last hidden state)
                word_embeddings = outputs.last_hidden_state

                # Extract input_ids and convert them back to tokens
                input_ids = inputs['input_ids'][0].tolist()
                tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(input_ids)

                # Now align the tokens and embeddings
                for token, embedding in zip(tokens, word_embeddings[0]):
                    embeddings[token]=embedding.tolist()

            else:
                if self.model_name == 'fastText':
                    embeddings = []
                    for token in inputs:
                        embeddings.append((token, self.model_instance.get_word_vector(token)))

            doc_entry_list.append(embeddings)

        return doc_entry_list, len(inputs)