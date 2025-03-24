import numpy as np
import scipy
from typing import Dict, List, Any

#Type aliases
DistanceMatrix = np.ndarray
EmbeddingDict = Dict[str, Dict[str, List[Any]]]

def get_distance_from_randomness(embeddings, config, parallel=False):

    if parallel:
        print(f'parallel computing not yet set up... '
              f'continuing without calculating divergence from optimality')
        return

    else:
        results_dict = {}
        result = optimality(
            embeddings, config['window_size'], config['bootstrap'], config['shuffle_mode']
        )
        results_dict[f'section'] = result
        return results_dict


def optimality(embeddings_dict, min_len, bootstrap, shuffle_mode):

    words = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())

    answer_res = []
    answer_len = len(words)

    for i in range((answer_len - min_len) + 1):

        window = embeddings[i:i + min_len]
        dist_matrix = create_semantic_distance_matrix(window)

        # Calculate costs for actual sequence and permutations
        perm_costs = []
        for j in range(bootstrap):
            order = (np.arange(len(window)) if j == 0
                     else get_shuffled_order(len(window), shuffle_mode, j))
            cost = calculate_total_distance_covered(dist_matrix, order)
            perm_costs.append(cost)

            if j == 0:
                all_pairs_avg = average_similarity(dist_matrix)

        # Normalize costs by number of edges
        costs_per_edge = np.array(perm_costs) / (min_len - 1)
        true_cost = costs_per_edge[0]

        # Store results for this window
        window_results = {
            "window_index": i,
            "all_pairs_average": all_pairs_avg,
            "actual_dist": true_cost,
            "average_dist": np.mean(costs_per_edge[1:]),
            "std_dist": np.std(costs_per_edge[1:])
        }
        answer_res.append(window_results)

    return answer_res


def create_semantic_distance_matrix(embedding_list: List[np.ndarray]) -> DistanceMatrix:

    distances = scipy.spatial.distance.cdist(
        np.array(embedding_list),
        np.array(embedding_list),
        'cosine'
    )
    np.fill_diagonal(distances, 0)
    return distances

def get_shuffled_order(n: int, shuffle_mode: str, seed: int) -> np.ndarray:

    np.random.seed(seed)

    if shuffle_mode == "include0_includeN":
        order = np.arange(n)
        np.random.shuffle(order)
    elif shuffle_mode == "exclude0_includeN":
        rest = np.arange(1, n)
        np.random.shuffle(rest)
        order = np.concatenate(([0], rest))
    elif shuffle_mode == "exclude0_excludeN":
        middle = np.arange(1, n - 1)
        np.random.shuffle(middle)
        order = np.concatenate(([0], middle, [n - 1]))
    else:
        raise ValueError(f"Invalid shuffle mode: {shuffle_mode}")

    return order

def calculate_total_distance_covered(dist_matrix: DistanceMatrix, order: np.ndarray) -> float:
    distances = dist_matrix[order[:-1], order[1:]]
    return float(np.sum(distances))

def average_similarity(matrix: DistanceMatrix) -> float:

    n = matrix.shape[0]

    # Only count upper triangle to avoid double counting
    upper_tri = np.triu(matrix, k=1)
    total = np.sum(upper_tri)
    count = (n * (n - 1)) // 2  # Number of pairs

    return float(total / count) if count > 0 else 0.0