import numpy as np
import scipy

def calculate_semantic_similarity(embedding_vectors, word_by_word_similarity=False):

    tokens = list(embedding_vectors.keys())
    vectors = list(embedding_vectors.values())

    similarities = np.array([])

    for position in range(1, len(vectors)):
        vec1 = vectors[position - 1]
        vec2 = vectors[position]

        similarity = get_semantic_similarity(vec1, vec2)
        similarities = np.append(similarities, similarity)

    mean_similarity = similarities[~np.isnan(similarities)].mean() if similarities.size > 0 else np.nan

    if word_by_word_similarity:
        word_similarities = [i for sublist in list(zip(tokens[1:], similarities)) for i in sublist]
        return mean_similarity, word_similarities

    return mean_similarity

def get_semantic_similarity(vec1, vec2) -> float:
    try:
        return 1 - scipy.spatial.distance.cosine(vec1, vec2)
    except ValueError:
        return np.nan