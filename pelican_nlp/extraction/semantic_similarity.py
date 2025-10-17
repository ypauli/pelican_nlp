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
    # Guard against zero vectors to avoid divide-by-zero warnings in cosine distance
    similarities = []
    for i in range(1, len(vectors)):
        u = vectors[i - 1]
        v = vectors[i]
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        if uu == 0.0 or vv == 0.0:
            similarities.append(np.nan)
            continue
        similarities.append(1 - scipy.spatial.distance.cosine(u, v))
    return similarities

def get_cosine_similarity_matrix(embedding_vectors):
    # Extract just the vectors from the list of tuples
    vectors = [vector for _, vector in embedding_vectors]
    if len(vectors) == 0:
        return np.array([[]], dtype=float)
    # Identify zero vectors
    norms = np.array([np.dot(v, v) for v in vectors])
    zero_mask = norms == 0.0
    # Compute cosine similarities via cdist, which may emit warnings for zero rows/cols
    sim = 1 - cdist(vectors, vectors, 'cosine')
    # Where either vector is zero, set similarity to NaN
    if sim.size > 0 and zero_mask.any():
        # Broadcast masks to rows/cols
        sim[zero_mask, :] = np.nan
        sim[:, zero_mask] = np.nan
    # Diagonal should be NaN (self-similarity not considered in window stats)
    np.fill_diagonal(sim, np.nan)
    return sim

def get_semantic_similarity_windows(embedding_vectors, window_size):
    # Extract tokens and vectors from the list of tuples
    tokens, vectors = zip(*embedding_vectors)

    # Early return if not enough tokens
    if len(tokens) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Handle sentence-wise similarity
    if window_size == 'sentence':
        return get_sentence_wise_similarity(embedding_vectors)

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

def get_sentence_wise_similarity(embedding_vectors):
    """
    Calculate semantic similarity grouped by sentences.
    Groups tokens by sentence boundaries (., !, ?) and computes similarity between sentences.
    """
    import re
    
    # Extract tokens and vectors from the list of tuples
    tokens, vectors = zip(*embedding_vectors)
    
    # Find sentence boundaries - look for sentence-ending punctuation
    sentence_boundaries = []
    for i, token in enumerate(tokens):
        # Check if token ends with sentence punctuation
        if re.search(r'[.!?]$', str(token)):
            sentence_boundaries.append(i)
    
    # If no sentence boundaries found, treat entire utterance as one sentence
    if not sentence_boundaries:
        # If only one token, return NaN
        if len(tokens) < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        # Otherwise, treat as single sentence and return NaN (no inter-sentence similarity)
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Group tokens by sentences
    sentences = []
    start_idx = 0
    for boundary in sentence_boundaries:
        end_idx = boundary + 1  # Include the punctuation token
        sentence_tokens = tokens[start_idx:end_idx]
        sentence_vectors = vectors[start_idx:end_idx]
        sentences.append(list(zip(sentence_tokens, sentence_vectors)))
        start_idx = end_idx
    
    # Add remaining tokens as last sentence if any
    if start_idx < len(tokens):
        sentence_tokens = tokens[start_idx:]
        sentence_vectors = vectors[start_idx:]
        sentences.append(list(zip(sentence_tokens, sentence_vectors)))
    
    # Filter out empty sentences
    sentences = [sent for sent in sentences if sent]
    
    # Need at least 2 sentences for inter-sentence similarity
    if len(sentences) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Calculate sentence-level embeddings (mean of token embeddings in each sentence)
    sentence_embeddings = []
    for sentence in sentences:
        if not sentence:
            continue
        # Get vectors for this sentence
        sent_vectors = [vector for _, vector in sentence]
        # Calculate mean embedding for the sentence
        if sent_vectors:
            mean_embedding = np.mean(sent_vectors, axis=0)
            sentence_embeddings.append(mean_embedding)
    
    # Need at least 2 sentence embeddings
    if len(sentence_embeddings) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Calculate similarity matrix between sentences
    sentence_sim_matrix = get_cosine_similarity_matrix(
        [(f"sentence_{i}", emb) for i, emb in enumerate(sentence_embeddings)]
    )
    
    # Calculate statistics for inter-sentence similarities
    return calculate_window_statistics(sentence_sim_matrix)

def calculate_window_statistics(cosine_similarity_matrix):
    matrix_values = pd.DataFrame(cosine_similarity_matrix).stack()
    return matrix_values.mean(), matrix_values.std(), matrix_values.median()
