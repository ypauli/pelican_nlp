import numpy as np
import pandas as pd
from itertools import combinations, islice
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cosine, pdist, squareform
from collections import Counter

class EmbeddingsExtractor:
    def __init__(self, model, mode='semantic'):

        self.model = model #embedding model (e.g., fastText, Epitran instance)
        self.mode = mode #semantic or phonetic

    def get_vector(self, tokens):
        """
        Get embeddings for a list of tokens using the provided model.

        Parameters:
        - tokens: List of tokens (words).

        Returns:
        - List of embeddings corresponding to the tokens.
        """
        embeddings = []
        for token in tokens:
            if self.mode == 'semantic':
                embeddings.append(self.model.get_word_vector(token))
            elif self.mode == 'phonetic':
                ipa_transcription = self.model.transliterate(token)
                # Convert IPA transcription to feature vectors (e.g., using panphon)
                # Here we assume a function ipa_to_features exists
                embeddings.append(ipa_to_features(ipa_transcription))
            else:
                raise ValueError("Mode should be 'semantic' or 'phonetic'")
        return embeddings

    def compute_similarity(self, vec1, vec2, metric_function):
        #Compute similarity between two embeddings using the provided metric function.
        return metric_function(vec1, vec2)

    def pairwise_similarities(self, embeddings, metric_function):
        """
        Compute pairwise similarities between all embeddings.

        Parameters:
        - embeddings: List of embedding vectors.
        - metric_function: Function to compute similarity between two vectors.

        Returns:
        - Numpy array of pairwise similarities.
        """
        if self.mode == 'semantic':
            # Compute cosine similarities
            distance_matrix = pdist(embeddings, metric='cosine')
            similarity_matrix = 1 - squareform(distance_matrix)
        elif self.mode == 'phonetic':
            # Compute similarities using the metric_function
            num_embeddings = len(embeddings)
            similarity_matrix = np.zeros((num_embeddings, num_embeddings))
            for i, j in combinations(range(num_embeddings), 2):
                sim = metric_function(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        else:
            raise ValueError("Mode should be 'semantic' or 'phonetic'")
        return similarity_matrix

    def compute_window_statistics(self, similarities, window_size, aggregation_functions):
        """
        Compute aggregated statistics over specified window sizes.

        Parameters:
        - similarities: Numpy array of pairwise similarities.
        - window_size: Size of the window (number of tokens).
        - aggregation_functions: List of functions to aggregate similarities.

        Returns:
        - Dictionary of aggregated statistics.
        """
        num_tokens = similarities.shape[0]
        stats = {}
        for start in range(0, num_tokens, window_size):
            end = min(start + window_size, num_tokens)
            window_similarities = similarities[start:end, start:end]
            window_values = window_similarities[np.triu_indices_from(window_similarities, k=1)]
            for func in aggregation_functions:
                key = f'{func.__name__}_window_{window_size}'
                stats.setdefault(key, []).append(func(window_values))
        # Compute overall statistics
        overall_stats = {key: np.mean(values) for key, values in stats.items()}
        return overall_stats

    def process_tokens(self, tokens, window_sizes, metric_function, aggregation_functions, parallel=False):
        """
        Process the list of tokens to compute embeddings and aggregated statistics.

        Parameters:
        - tokens: List of tokens (words).
        - window_sizes: List of window sizes to compute statistics over.
        - metric_function: Function to compute similarity between two embeddings.
        - aggregation_functions: List of functions to aggregate similarities.
        - parallel: Boolean, whether to use parallel processing.

        Returns:
        - Dictionary containing embeddings and aggregated statistics.
        """
        embeddings = self.get_vector(tokens)
        embeddings = np.array(embeddings)

        # Compute pairwise similarities
        similarity_matrix = self.pairwise_similarities(embeddings, metric_function)

        # Prepare results
        results = {'embeddings': embeddings, 'tokens': tokens}

        # Compute statistics for each window size
        for window_size in window_sizes:
            if window_size <= 0 or window_size > len(tokens):
                continue  # Skip invalid window sizes

            if parallel:
                # Use multiprocessing for window computations
                with ProcessPoolExecutor() as executor:
                    futures = []
                    num_tokens = len(tokens)
                    for start in range(0, num_tokens, window_size):
                        end = min(start + window_size, num_tokens)
                        window_similarities = similarity_matrix[start:end, start:end]
                        window_values = window_similarities[np.triu_indices_from(window_similarities, k=1)]
                        futures.append(executor.submit(self.aggregate_window, window_values, aggregation_functions))
                    for future in futures:
                        window_stats = future.result()
                        for key, value in window_stats.items():
                            results.setdefault(key, []).append(value)
            else:
                # Sequential processing
                window_stats = self.compute_window_statistics(similarity_matrix, window_size, aggregation_functions)
                results.update(window_stats)

        return results

    @staticmethod
    def aggregate_window(window_values, aggregation_functions):
        """
        Aggregate window values using provided functions.

        Parameters:
        - window_values: Numpy array of similarity values in the window.
        - aggregation_functions: List of functions to aggregate similarities.

        Returns:
        - Dictionary of aggregated values for the window.
        """
        stats = {}
        for func in aggregation_functions:
            key = func.__name__
            stats[key] = func(window_values)
        return stats