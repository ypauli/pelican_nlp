import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

class embeddings_metrics_statistics():
    @staticmethod
    def pairwise_similarities(embeddings, metric_function=None):
        """Compute pairwise similarities between embeddings."""
        distance_matrix = pdist(embeddings, metric='cosine')
        similarity_matrix = 1 - squareform(distance_matrix)
        return similarity_matrix

    @staticmethod
    def compute_window_statistics(similarities, window_size, aggregation_functions=[np.mean]):
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


    @staticmethod
    def aggregate_window(window_values, aggregation_functions=[np.mean]):
        """Aggregates window values using specified functions."""
        return {func.__name__: func(window_values) for func in aggregation_functions}