import numpy as np
import scipy
from scipy.spatial.distance import cdist
import pandas as pd

def calculate_semantic_similarity(embedding_vectors, word_by_word_similarity=False):
    tokens = list(embedding_vectors.keys())
    vectors = list(embedding_vectors.values())

    similarities = get_pairwise_similarities(vectors)
    mean_similarity = np.nanmean(similarities)

    if word_by_word_similarity:
        word_similarities = [item for sublist in zip(tokens[1:], similarities) for item in sublist]
        return mean_similarity, word_similarities

    return mean_similarity

def get_pairwise_similarities(vectors):
    return [1 - scipy.spatial.distance.cosine(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))]

def get_cosine_similarity_matrix(embedding_vectors):
    vectors = list(embedding_vectors.values())
    similarity_matrix = 1 - cdist(vectors, vectors, 'cosine')
    np.fill_diagonal(similarity_matrix, np.nan)
    return similarity_matrix

def get_semantic_similarity_windows(embedding_vectors, window_size):
    tokens = list(embedding_vectors.keys())
    vectors = list(embedding_vectors.values())

    if len(tokens) < 2:
        return np.nan, np.nan, np.nan, np.nan

    if window_size == 'all':
        cosine_similarity_matrix = get_cosine_similarity_matrix(embedding_vectors)
        return calculate_window_statistics(cosine_similarity_matrix)

    window_means, window_stds = zip(*[
        calculate_window_statistics(get_cosine_similarity_matrix({token: vector for token, vector in zip(tokens[i:i + window_size], vectors[i:i + window_size])}))
        for i in range(len(tokens) - window_size + 1)
    ])

    return np.mean(window_means), np.std(window_means), np.mean(window_stds), np.std(window_stds)

def calculate_window_statistics(cosine_similarity_matrix):
    matrix_values = pd.DataFrame(cosine_similarity_matrix).stack()
    return matrix_values.mean(), matrix_values.std()
