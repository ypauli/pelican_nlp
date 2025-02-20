import numpy as np
from concurrent.futures import ProcessPoolExecutor

from pelican.extraction.language_model import Model
from pelican.preprocessing.text_tokenizer import TextTokenizer
from pelican.preprocessing.text_cleaner import TextCleaner
from pelican.csv_functions import store_features_to_csv

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

            # Tokenize the input text
            inputs = self.Tokenizer.tokenize_text(text)
            #print(f'inputs: {inputs}')

            if self.embeddings_configurations['pytorch_based_model']:
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

                embeddings = {}
                # Now align the tokens and embeddings
                for token, embedding in zip(tokens, word_embeddings[0]):
                    embeddings[token]=embedding.tolist()

                doc_entry_list.append(embeddings)

            else:
                if self.model_name == 'fastText':
                    # Assuming fastText is being used for semantic embeddings
                    for token in tokens:
                        embeddings.append(self.model.get_word_vector(token))
                return np.array(embeddings)
        return doc_entry_list

    @staticmethod
    def extract_embeddings_Morteza(text):
        from transformers import AutoTokenizer, AutoModel
        import torch

        print(f'Text is: {text}')
        print(f'type is: {type(text)}')

        # Load the XLM-RoBERTa model and tokenizer
        model_name = "xlm-roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(tokenizer)
        model = AutoModel.from_pretrained(model_name)
        print(model)

        # Tokenize the text and get tensor format suitable for the model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        print(f'inputs: {inputs}')

        # Pass the inputs through the model to get the outputs
        with torch.no_grad():
            outputs = model(**inputs)
        print(f'outputs: {outputs}')

        # Extract the word embeddings (last hidden state)
        word_embeddings = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, 768]
        print(f'word embeddings: {word_embeddings}')

        # Get the individual token embeddings
        tokens = tokenizer.tokenize(text)
        print(f'tokens: {tokens}')
        embeddings = {}
        for token, embedding in zip(tokens, word_embeddings[0]):
            clean_token = token
            if clean_token:  # Ignore empty or non-relevant tokens
                embeddings[clean_token] = embedding.tolist()  # Save the full 768-dimensional embedding
        print(f'embeddings: {embeddings}')
        return embeddings

    def process_text(self, document, embeddings_configurations, window_sizes, metric_function=None, parallel=False,
                     speakertag=None):

        document.embeddings = []

        if document.num_speakers:
            print(f'🎤 Speaker-based processing: {document.cleaned_sections}')
            for key, section in document.cleaned_sections.items():
                speaker_tokens = self.extract_speaker_tokens(section, speakertag, embeddings_configurations)

                if self.model_name != 'xlm-roberta-base':
                    speaker_tokens = [self._preprocess_token(token) for token in
                                      speaker_tokens]  # ✅ Apply preprocessing

                    # Create a new list to store cleaned tokens
                    cleaned_speaker_tokens = []

                    for token in speaker_tokens:
                        cleaned_token = TextCleaner.clean_subword_token_RoBERTa(token)
                        cleaned_speaker_tokens.append(cleaned_token)

                    # Replace the old speaker_tokens with the cleaned ones
                    speaker_tokens = cleaned_speaker_tokens


                speaker_tokens = [token for token in speaker_tokens if token and token is not None]
                print(f'speaker tokens are {speaker_tokens}')
                embeddings = self.get_vector(speaker_tokens)

                doc_entry = {
                    'tokens': speaker_tokens,  # ✅ First key is 'tokens' (preprocessed)
                    'embeddings': embeddings
                }

                document.embeddings.append(doc_entry)

            print(f'✅ Speaker-based embeddings stored: {document.embeddings}')
            store_features_to_csv(document.embeddings, document.results_path, document.corpus_name)
            return

        # Standard tokenization for non-speaker data
        tokenizer = TextTokenizer(embeddings_configurations)
        document.tokenize_text(tokenizer, 'embeddings')

        for token_list in document.tokens_embeddings:
            token_list = [self._preprocess_token(token) for token in token_list]  # ✅ Apply preprocessing

            embeddings = self.get_vector(token_list)
            similarity_matrix = self.pairwise_similarities(embeddings, metric_function)

            results = {'tokens': token_list, 'embeddings': embeddings}  # ✅ Store preprocessed tokens

            for window_size in window_sizes:
                if window_size <= 0 or window_size > len(token_list):
                    continue

                if parallel:
                    with ProcessPoolExecutor() as executor:
                        futures = []
                        for start in range(0, len(token_list), window_size):
                            end = min(start + window_size, len(token_list))
                            window_similarities = similarity_matrix[start:end, start:end]
                            window_values = window_similarities[np.triu_indices_from(window_similarities, k=1)]
                            futures.append(executor.submit(self.aggregate_window, window_values))

                        for future in futures:
                            window_stats = future.result()
                            results.update(window_stats)
                else:
                    results.update(self.compute_window_statistics(similarity_matrix, window_size))

            document.embeddings.append(results)

        store_features_to_csv(document.embeddings, document.results_path, document.corpus_name)