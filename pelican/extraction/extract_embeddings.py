import numpy as np
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import pdist, squareform

from pelican.preprocessing.speaker_diarization import TextDiarizer
from pelican.preprocessing.text_tokenizer import TextTokenizer
from pelican.csv_functions import store_features_to_csv


class EmbeddingsExtractor:
    def __init__(self, model_name, mode='semantic'):
        self.model_name = model_name  # Embedding model instance (e.g., fastText, Epitran)
        self.mode = mode  # 'semantic' or 'phonetic'
        self.model = self._load_model()

    def _load_model(self):
        if self.mode == 'semantic':
            import fasttext.util
            fasttext.util.download_model('de', if_exists='ignore')
            model = fasttext.load_model('cc.de.300.bin')
            print('✅ FastText model loaded.')
        elif self.mode == 'phonetic':
            if not self.model_name:
                raise ValueError("❌ A phonetic model instance is required for 'phonetic' mode.")
            model = self.model_name
            print('✅ Phonetic model loaded.')
        else:
            raise ValueError("❌ Mode should be 'semantic' or 'phonetic'.")
        return model

    def get_vector(self, tokens):
        """Compute embeddings for a list of tokens."""

        print(f'📌 Processing embeddings for tokens: {tokens}')

        embeddings = []

        for token in tokens:
            if self.mode == 'semantic':
                embeddings.append(self.model.get_word_vector(token))
            elif self.mode == 'phonetic':
                ipa_transcription = self.model.transliterate(token)
                embeddings.append(ipa_to_features(ipa_transcription))
            else:
                raise ValueError("❌ Mode should be 'semantic' or 'phonetic'.")

        return np.array(embeddings)

    def pairwise_similarities(self, embeddings, metric_function=None):
        """Compute pairwise similarities between embeddings."""
        if self.mode == 'semantic':
            distance_matrix = pdist(embeddings, metric='cosine')
            similarity_matrix = 1 - squareform(distance_matrix)
        elif self.mode == 'phonetic':
            num_embeddings = len(embeddings)
            similarity_matrix = np.zeros((num_embeddings, num_embeddings))
            for i, j in combinations(range(num_embeddings), 2):
                sim = metric_function(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        else:
            raise ValueError("❌ Mode should be 'semantic' or 'phonetic'.")
        return similarity_matrix

    def compute_window_statistics(self, similarities, window_size, aggregation_functions=[np.mean]):
        """Compute aggregated statistics over a given window size."""
        num_tokens = similarities.shape[0]
        stats = {}

        for start in range(0, num_tokens, window_size):
            end = min(start + window_size, num_tokens)
            window_similarities = similarities[start:end, start:end]
            window_values = window_similarities[np.triu_indices_from(window_similarities, k=1)]

            for func in aggregation_functions:
                key = f'{func.__name__}_window_{window_size}'
                stats.setdefault(key, []).append(func(window_values))

        return {key: np.mean(values) for key, values in stats.items()}

    def _preprocess_token(self, token):
        """Apply lowercase and punctuation removal."""
        from pelican.preprocessing.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        return cleaner._remove_punctuation(cleaner._lowercase(token))

    def process_text(self, document, embeddings_configurations, window_sizes, metric_function=None, parallel=False,
                     speakertag=None):

        document.embeddings = []

        if document.num_speakers:
            print(f'🎤 Speaker-based processing: {document.cleaned_sections}')
            for key, section in document.cleaned_sections.items():
                speaker_tokens = self.extract_speaker_tokens(section, speakertag)
                speaker_tokens = [self._preprocess_token(token) for token in speaker_tokens]  # ✅ Apply preprocessing

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

    def extract_speaker_tokens(self, text, speaker):
        """Extracts tokens for a specific speaker."""
        import re
        speaker_tokens, all_tokens = [], []
        current_speaker, utterance_buffer = None, []

        for line in text.split('\n'):
            match = re.match(r'^(\w+):\s*(.*)', line)
            if match:
                if current_speaker:
                    all_tokens.extend(utterance_buffer)
                    if current_speaker == speaker:
                        speaker_tokens.extend(utterance_buffer)

                current_speaker, content = match.groups()
                utterance_buffer = content.split()
            elif current_speaker:
                utterance_buffer.extend(line.split())

        if current_speaker:
            all_tokens.extend(utterance_buffer)
            if current_speaker == speaker:
                speaker_tokens.extend(utterance_buffer)

        return speaker_tokens

    @staticmethod
    def aggregate_window(window_values, aggregation_functions=[np.mean]):
        """Aggregates window values using specified functions."""
        return {func.__name__: func(window_values) for func in aggregation_functions}
