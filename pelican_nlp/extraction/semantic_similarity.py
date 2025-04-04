import numpy as np
import scipy
from scipy.spatial.distance import cdist
import pandas as pd

def calculate_semantic_similarity(embedding_vectors):
    # Extract just the vectors from the list of tuples
    vectors = [vector for _, vector in embedding_vectors]
    consecutive_similarities = get_consecutive_vector_similarities(vectors)
    mean_similarity = np.nanmean(consecutive_similarities)
    return consecutive_similarities, mean_similarity

def get_consecutive_vector_similarities(vectors):
    return [1 - scipy.spatial.distance.cosine(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))]

def get_cosine_similarity_matrix(embedding_vectors):
    # Extract just the vectors from the list of tuples
    vectors = [vector for _, vector in embedding_vectors]
    similarity_matrix = 1 - cdist(vectors, vectors, 'cosine')
    np.fill_diagonal(similarity_matrix, np.nan)
    return similarity_matrix

def get_semantic_similarity_windows(embedding_vectors, window_size):
    # Extract tokens and vectors from the list of tuples
    tokens, vectors = zip(*embedding_vectors)

    # Early return if not enough tokens
    if len(tokens) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Early return if window size is larger than sequence
    if window_size > len(tokens):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if window_size == 'all':
        cosine_similarity_matrix = get_cosine_similarity_matrix(embedding_vectors)
        return calculate_window_statistics(cosine_similarity_matrix)

    # Collect window statistics
    window_statistics = []
    for i in range(len(tokens) - window_size + 1):
        window_vectors = list(zip(tokens[i:i + window_size], vectors[i:i + window_size]))
        if window_vectors:  # Make sure window is not empty
            sim_matrix = get_cosine_similarity_matrix(window_vectors)
            window_statistics.append(calculate_window_statistics(sim_matrix))
    
    # Handle case where no valid windows were found
    if not window_statistics:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        
    # Unzip the statistics
    window_means, window_stds, window_medians = zip(*window_statistics)
    
    return (np.mean(window_means), np.std(window_means), 
            np.mean(window_stds), np.std(window_stds), 
            np.mean(window_medians))

def calculate_window_statistics(cosine_similarity_matrix):
    matrix_values = pd.DataFrame(cosine_similarity_matrix).stack()
    return matrix_values.mean(), matrix_values.std(), matrix_values.median()
