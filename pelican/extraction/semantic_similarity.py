import numpy as np
import scipy
from scipy.spatial.distance import cdist
import pandas as pd

def calculate_semantic_similarity(embedding_vectors):

    vectors = list(embedding_vectors.values())

    consecutive_similarities = get_consecutive_vector_similarities(vectors)
    print(f'consec similarities: {consecutive_similarities}')
    mean_similarity = np.nanmean(consecutive_similarities)

    return consecutive_similarities, mean_similarity

def get_consecutive_vector_similarities(vectors):
    return [1 - scipy.spatial.distance.cosine(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))]

def get_cosine_similarity_matrix(embedding_vectors):
    #print(f'embedding_vectors for cosine-similarity-matrix: {embedding_vectors}')
    vectors = list(embedding_vectors.values())
    similarity_matrix = 1 - cdist(vectors, vectors, 'cosine')
    np.fill_diagonal(similarity_matrix, np.nan)

    #upper_triangle_indices = np.triu_indices_from(similarity_matrix)
    #similarity_matrix[upper_triangle_indices] = np.nan


    #print(f'similarity matrix: {similarity_matrix}')
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
