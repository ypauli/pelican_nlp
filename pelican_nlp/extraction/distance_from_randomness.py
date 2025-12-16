import numpy as np
import scipy
import random
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances

#Type aliases
DistanceMatrix = np.ndarray
EmbeddingDict = Dict[str, Dict[str, List[Any]]]

def get_distance_from_randomness(embeddings, config, parallel=False):
    """
    Calculate distance from randomness metrics.
    
    Supports two modes:
    1. Sliding window mode (existing): Uses window_size and sliding windows
    2. TSP divergence mode (new): Calculates global/local divergence with TSP optimal path
    
    Args:
        embeddings: Embedding dictionary (token -> embedding) or list of embeddings
        config: Configuration dictionary with options
        parallel: Whether to use parallel computing (not yet implemented)
        
    Returns:
        Dictionary with results (format depends on mode)
    """
    if parallel:
        print(f'parallel computing not yet set up... '
              f'continuing without calculating divergence from optimality')
        return

    # Check if TSP divergence mode is enabled
    use_tsp_divergence = config.get('use_tsp_divergence', False)
    
    if use_tsp_divergence:
        # New TSP-based divergence mode
        return _calculate_tsp_divergence(embeddings, config)
    else:
        # Existing sliding window mode (backward compatible)
        # Normalize embeddings to a dict: token_id -> embedding
        if isinstance(embeddings, dict):
            embeddings_dict = embeddings
        elif isinstance(embeddings, list):
            # List format: list of (token, embedding) tuples or list of embeddings
            embeddings_dict = {}
            if len(embeddings) > 0 and isinstance(embeddings[0], tuple):
                # List of (token, embedding) tuples – keep token as key
                for idx, (token, emb) in enumerate(embeddings):
                    # Ensure numpy array for distance calculations
                    emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb
                    embeddings_dict[str(token) if token is not None else str(idx)] = emb_array
            else:
                # Plain list of embeddings – use index as key
                for idx, emb in enumerate(embeddings):
                    emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb
                    embeddings_dict[str(idx)] = emb_array
        else:
            raise ValueError(f"Unsupported embeddings format for sliding window mode: {type(embeddings)}")

        results_dict = {}
        result = optimality(
            embeddings_dict,
            config.get('window_size', 8),
            config.get('bootstrap', 1000),
            config.get('shuffle_mode', 'include0_includeN'),
        )
        results_dict['section'] = result
        return results_dict


def _calculate_tsp_divergence(embeddings, config) -> Dict[str, Any]:
    """
    Calculate TSP-based divergence metrics for the entire sequence.
    
    Args:
        embeddings: Embedding dictionary (token -> embedding) or list of embeddings
        config: Configuration dictionary
        
    Returns:
        Dictionary with divergence metrics
    """
    # Convert embeddings to list format
    if isinstance(embeddings, dict):
        # Dictionary format: {token: embedding}
        embeddings_list = list(embeddings.values())
    elif isinstance(embeddings, list):
        # List format: list of (token, embedding) tuples or list of embeddings
        if len(embeddings) > 0 and isinstance(embeddings[0], tuple):
            # List of tuples: extract embeddings
            embeddings_list = [emb[1] for emb in embeddings]
        else:
            # List of embeddings
            embeddings_list = embeddings
    else:
        raise ValueError(f"Unsupported embeddings format: {type(embeddings)}")
    
    # Convert to numpy arrays if needed
    embeddings_list = [np.array(emb) if not isinstance(emb, np.ndarray) else emb
                       for emb in embeddings_list]

    # Safety guard: limit sequence length passed to TSP solver
    # Large n can make OR-Tools unstable or extremely slow; we truncate to a configurable maximum.
    max_tsp_tokens = config.get('max_tsp_tokens', 100)
    if len(embeddings_list) > max_tsp_tokens:
        print(f"[distance_from_randomness] Truncating embeddings from {len(embeddings_list)} "
              f"to {max_tsp_tokens} tokens for TSP divergence.")
        embeddings_list = embeddings_list[:max_tsp_tokens]
    
    # Get configuration parameters
    n_permute = config.get('n_permute', config.get('bootstrap', 1000))
    random_seed = config.get('random_seed', None)
    
    # Calculate divergence metrics
    results = calculate_divergence_metrics(embeddings_list, n_permute, random_seed)
    
    # Format for storage (remove numpy arrays that can't be serialized easily)
    results_dict = {
        'section': [{
            'total_distance': results['total_distance'],
            'avg_distance': results['avg_distance'],
            'global_divergence': results['global_divergence'],
            'local_divergence': results['local_divergence'],
            'global_div_z': results['global_div_z'],
            'local_div_z': results['local_div_z'],
            'optimal_path': results['optimal_path']
        }]
    }
    
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


def tsp_optimal_path(vectors: np.ndarray) -> Tuple[List[int], DistanceMatrix, float, float]:
    """
    Compute optimal TSP path using OR-Tools and return path, distance matrix, observed length, and optimal length.
    
    Args:
        vectors: Array of embedding vectors (n x d)
        
    Returns:
        Tuple of (optimal_path, distance_matrix, observed_length, optimal_length)
        - optimal_path: List of indices representing the optimal TSP path
        - distance_matrix: Cosine distance matrix
        - observed_length: Total distance of observed order (0->1->2->...->n-1)
        - optimal_length: Total distance of optimal TSP path
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        raise ImportError("ortools is required for TSP optimal path calculation. Install with: pip install ortools")
    
    n = len(vectors)
    if n <= 1:
        return list(range(n)), np.zeros((n, n)), 0.0, 0.0
    
    # Create cosine distance matrix
    dist_matrix = cosine_distances(vectors)
    
    # Set up TSP solver
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        """Callback function for distance between nodes."""
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        # Convert to integer (multiply by 1e6 for precision)
        return int(dist_matrix[f][t] * 1e6)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Set search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    # Solve TSP
    solution = routing.SolveWithParameters(search_params)
    
    if solution is None:
        # Fallback to observed order if TSP fails
        path = list(range(n))
        obs_len = sum(dist_matrix[i, i+1] for i in range(n-1))
        opt_len = obs_len
        return path, dist_matrix, obs_len, opt_len
    
    # Extract optimal path
    path = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        path.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    
    # Calculate observed and optimal path lengths
    obs_len = sum(dist_matrix[i, i+1] for i in range(n-1)) if n > 1 else 0.0
    opt_len = sum(dist_matrix[path[i], path[i+1]] for i in range(len(path)-1)) if len(path) > 1 else 0.0
    
    return path, dist_matrix, obs_len, opt_len


def local_divergence(obs_order: List[int], opt_path: List[int]) -> float:
    """
    Calculate local divergence: sum of absolute position differences in optimal path for consecutive observed items.
    
    Args:
        obs_order: Observed order (list of indices)
        opt_path: Optimal TSP path (list of indices)
        
    Returns:
        Local divergence value
    """
    if len(obs_order) <= 1 or len(opt_path) <= 1:
        return 0.0
    
    # Create mapping from node to its position in optimal path
    index_in_opt = {node: i for i, node in enumerate(opt_path)}
    
    div = 0.0
    for i in range(len(obs_order) - 1):
        a, b = obs_order[i], obs_order[i+1]
        if a in index_in_opt and b in index_in_opt:
            div += abs(index_in_opt[a] - index_in_opt[b])
    
    return div


def compute_z_score(real_val: float, perm_vals: List[float]) -> float:
    """
    Calculate z-score: (real_value - mean(permutations)) / std(permutations)
    
    Args:
        real_val: Real observed value
        perm_vals: List of permutation values
        
    Returns:
        Z-score (0 if std is 0)
    """
    if not perm_vals:
        return 0.0
    
    mu = np.mean(perm_vals)
    sigma = np.std(perm_vals)
    
    return (real_val - mu) / sigma if sigma > 0 else 0.0


def calculate_divergence_metrics(embeddings_list: List[np.ndarray], 
                                 n_permute: int = 1000,
                                 random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Calculate global and local divergence metrics with z-scores from permutation testing.
    
    Args:
        embeddings_list: List of embedding vectors
        n_permute: Number of permutations for null distribution
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with divergence metrics including:
        - total_distance: Observed total distance
        - avg_distance: Average distance per edge
        - global_divergence: obs_len - opt_len
        - local_divergence: Local divergence value
        - global_div_z: Z-score for global divergence
        - local_div_z: Z-score for local divergence
        - optimal_path: Optimal TSP path
        - distance_matrix: Distance matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)
    
    if len(embeddings_list) < 2:
        return {
            "total_distance": 0.0,
            "avg_distance": 0.0,
            "global_divergence": 0.0,
            "local_divergence": 0.0,
            "global_div_z": 0.0,
            "local_div_z": 0.0,
            "optimal_path": list(range(len(embeddings_list))),
            "distance_matrix": np.zeros((len(embeddings_list), len(embeddings_list)))
        }
    
    # Convert to numpy array
    vectors = np.array(embeddings_list)
    
    # Calculate optimal path and distances
    opt_path, dist_matrix, obs_len, opt_len = tsp_optimal_path(vectors)
    
    # Observed order is just sequential (0, 1, 2, ..., n-1)
    obs_order = list(range(len(embeddings_list)))
    
    # Calculate divergences
    g_div = obs_len - opt_len
    l_div = local_divergence(obs_order, opt_path)
    
    # Permutation testing for null distribution
    g_perms = []
    l_perms = []
    
    for _ in range(n_permute):
        perm_order = obs_order[:]
        random.shuffle(perm_order)
        
        # Global divergence for permutation
        perm_g = sum(dist_matrix[perm_order[i], perm_order[i+1]] for i in range(len(perm_order)-1)) - opt_len
        g_perms.append(perm_g)
        
        # Local divergence for permutation
        perm_l = local_divergence(perm_order, opt_path)
        l_perms.append(perm_l)
    
    # Calculate z-scores
    g_z = compute_z_score(g_div, g_perms)
    l_z = compute_z_score(l_div, l_perms)
    
    # Calculate average distance
    n_edges = len(obs_order) - 1 if len(obs_order) > 1 else 1
    avg_dist = obs_len / n_edges if n_edges > 0 else 0.0
    
    return {
        "total_distance": obs_len,
        "avg_distance": avg_dist,
        "global_divergence": g_div,
        "local_divergence": l_div,
        "global_div_z": g_z,
        "local_div_z": l_z,
        "optimal_path": opt_path,
        "distance_matrix": dist_matrix
    }