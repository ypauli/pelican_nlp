import numpy as np
import scipy
from scipy.spatial.distance import cdist
import pandas as pd
import string
from pelican_nlp.config import debug_print

SPECIAL_TOKENS = {'<s>', '</s>', '<pad>', '<unk>'}

def _is_punctuation_token(token):
    token_str = str(token)
    if token_str in SPECIAL_TOKENS:
        return True
    token_core = token_str.replace('▁', '').strip()
    if token_core == '':
        return True
    return all(char in string.punctuation for char in token_core)

def filter_punctuation_tokens(embedding_vectors):
    filtered = []
    removed = 0
    for token, vector in embedding_vectors:
        if _is_punctuation_token(token):
            removed += 1
            continue
        filtered.append((token, vector))
    if removed > 0:
        debug_print(f"[filter_punctuation_tokens] Removed {removed} punctuation token(s)")
    return filtered

def _token_to_text_piece(token_str):
    token_str = str(token_str)
    if not token_str:
        return '', False
    has_word_boundary = token_str.startswith('▁')
    core = token_str.lstrip('▁')
    return core, has_word_boundary

def _clean_token_text(token):
    core, _ = _token_to_text_piece(token)
    return core

def tokens_to_text(tokens):
    text = ''
    for token in tokens:
        core, has_boundary = _token_to_text_piece(token)
        if not core:
            continue
        if has_boundary:
            if core in string.punctuation:
                text += core
            else:
                if text and not text.endswith(' '):
                    text += ' '
                text += core
        else:
            text += core
    return ' '.join(text.split())

def calculate_semantic_similarity(embedding_vectors):
    # Check if embedding_vectors is empty
    if not embedding_vectors:
        return [], np.nan
    
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
    tokens = [token for token, _ in embedding_vectors]
    
    debug_print(f"  [get_cosine_similarity_matrix] Processing {len(vectors)} vectors")
    
    if len(vectors) == 0:
        debug_print("  [get_cosine_similarity_matrix] ERROR: Empty vectors list")
        return np.array([[]], dtype=float)
    
    # Identify zero vectors
    norms = np.array([np.dot(v, v) for v in vectors])
    zero_mask = norms == 0.0
    zero_count = np.sum(zero_mask)
    
    if zero_count > 0:
        zero_tokens = [tokens[i] for i in range(len(tokens)) if zero_mask[i]]
        debug_print(f"  [get_cosine_similarity_matrix] WARNING: Found {zero_count} zero vector(s): {zero_tokens}")
    
    # Compute cosine similarities via cdist, which may emit warnings for zero rows/cols
    sim = 1 - cdist(vectors, vectors, 'cosine')
    
    # Where either vector is zero, set similarity to NaN
    if sim.size > 0 and zero_mask.any():
        # Broadcast masks to rows/cols
        sim[zero_mask, :] = np.nan
        sim[:, zero_mask] = np.nan
        debug_print(f"  [get_cosine_similarity_matrix] Set {np.sum(zero_mask)} row(s)/column(s) to NaN due to zero vectors")
    
    # Diagonal should be NaN (self-similarity not considered in window stats)
    np.fill_diagonal(sim, np.nan)
    
    # Count valid (non-NaN) values
    valid_count = np.sum(~np.isnan(sim))
    total_count = sim.size
    debug_print(f"  [get_cosine_similarity_matrix] Similarity matrix: {sim.shape}, valid values: {valid_count}/{total_count}, NaN values: {total_count - valid_count}")
    
    if valid_count == 0:
        debug_print(f"  [get_cosine_similarity_matrix] ERROR: All values are NaN in similarity matrix!")
        debug_print(f"  [get_cosine_similarity_matrix] Matrix:\n{sim}")
    
    return sim

def _format_window_detail(index, tokens, stats):
    """Helper to build a detail entry for a sliding window."""
    return {
        'window_index': index,
        'start_token': _clean_token_text(tokens[0]) if tokens else None,
        'end_token': _clean_token_text(tokens[-1]) if tokens else None,
        'token_span': tokens_to_text(tokens),
        'mean': stats[0],
        'std': stats[1],
        'median': stats[2]
    }

def _nan_summary():
    return (np.nan, np.nan, np.nan, np.nan, np.nan)

def get_semantic_similarity_windows(embedding_vectors, window_size, return_details=False, exclude_punctuation=False):
    debug_print(f"\n[get_semantic_similarity_windows] === START: window_size={window_size} ===")
    
    # Check if embedding_vectors is empty
    if not embedding_vectors:
        debug_print(f"[get_semantic_similarity_windows] ERROR: Empty embedding_vectors, returning NaN")
        summary = _nan_summary()
        return (summary, []) if return_details else summary
    
    # Optionally filter punctuation tokens (except for sentence processing where punctuation is needed)
    if exclude_punctuation and window_size != 'sentence':
        embedding_vectors = filter_punctuation_tokens(embedding_vectors)
        if not embedding_vectors:
            debug_print("[get_semantic_similarity_windows] All tokens removed after punctuation filtering")
            summary = _nan_summary()
            return (summary, []) if return_details else summary

    # Extract tokens and vectors from the list of tuples
    tokens, vectors = zip(*embedding_vectors)
    debug_print(f"[get_semantic_similarity_windows] Total tokens: {len(tokens)}")
    debug_print(f"[get_semantic_similarity_windows] First 5 tokens: {list(tokens[:5])}")

    # Early return if not enough tokens
    if len(tokens) < 2:
        debug_print(f"[get_semantic_similarity_windows] ERROR: Not enough tokens ({len(tokens)} < 2), returning NaN")
        summary = _nan_summary()
        return (summary, []) if return_details else summary

    # Handle sentence-wise similarity
    if window_size == 'sentence':
        debug_print(f"[get_semantic_similarity_windows] Processing sentence-wise similarity")
        return get_sentence_wise_similarity(embedding_vectors, return_details=return_details, exclude_punctuation=exclude_punctuation)

    # Early return if window size is larger than sequence
    if window_size > len(tokens):
        debug_print(f"[get_semantic_similarity_windows] ERROR: Window size ({window_size}) > token count ({len(tokens)}), returning NaN")
        summary = _nan_summary()
        return (summary, []) if return_details else summary

    if window_size == 'all':
        debug_print(f"[get_semantic_similarity_windows] Processing 'all' window size")
        cosine_similarity_matrix = get_cosine_similarity_matrix(embedding_vectors)
        result = calculate_window_statistics(cosine_similarity_matrix, skip_std=False)
        debug_print(f"[get_semantic_similarity_windows] Result for 'all': {result}")
        return (result, []) if return_details else result

    # Collect window statistics
    num_windows = len(tokens) - window_size + 1
    debug_print(f"[get_semantic_similarity_windows] Processing {num_windows} windows of size {window_size}")
    
    window_statistics = []
    window_details = [] if return_details else None
    nan_window_count = 0
    
    for i in range(num_windows):
        window_vectors = list(zip(tokens[i:i + window_size], vectors[i:i + window_size]))
        window_tokens = tokens[i:i + window_size]
        
        debug_print(f"\n  [get_semantic_similarity_windows] Window {i+1}/{num_windows}: tokens={list(window_tokens)}")
        
        if window_vectors:  # Make sure window is not empty
            sim_matrix = get_cosine_similarity_matrix(window_vectors)
            # Skip std calculation for window_size 2 (std is always 0 with only 1 unique value)
            skip_std = (window_size == 2)
            stats = calculate_window_statistics(sim_matrix, skip_std=skip_std)
            window_statistics.append(stats)
            if return_details:
                detail = _format_window_detail(i + 1, window_tokens, stats)
                window_details.append(detail)
            
            # Check if this window returned all NaN
            if all(pd.isna(v) for v in stats):
                nan_window_count += 1
                debug_print(f"  [get_semantic_similarity_windows] WARNING: Window {i+1} returned all NaN statistics")
        else:
            debug_print(f"  [get_semantic_similarity_windows] ERROR: Window {i+1} is empty!")
    
    debug_print(f"\n[get_semantic_similarity_windows] Processed {len(window_statistics)} windows, {nan_window_count} returned all NaN")
    
    # Handle case where no valid windows were found
    if not window_statistics:
        debug_print(f"[get_semantic_similarity_windows] ERROR: No valid windows found, returning NaN")
        summary = _nan_summary()
        return (summary, window_details or []) if return_details else summary
        
    # Unzip the statistics
    window_means, window_stds, window_medians = zip(*window_statistics)
    
    # Count NaN values in each statistic type
    nan_means = sum(1 for m in window_means if pd.isna(m))
    nan_stds = sum(1 for s in window_stds if pd.isna(s))
    nan_medians = sum(1 for m in window_medians if pd.isna(m))
    
    debug_print(f"[get_semantic_similarity_windows] Window statistics summary:")
    debug_print(f"  - Window means: {len(window_means)} total, {nan_means} NaN, {len(window_means) - nan_means} valid")
    debug_print(f"  - Window stds: {len(window_stds)} total, {nan_stds} NaN, {len(window_stds) - nan_stds} valid")
    debug_print(f"  - Window medians: {len(window_medians)} total, {nan_medians} NaN, {len(window_medians) - nan_medians} valid")
    
    # Calculate final statistics
    final_mean = np.mean(window_means)
    
    # For window_size 2, skip std calculations (they're always 0 or NaN)
    if window_size == 2:
        final_std_of_means = np.nan
        final_mean_of_stds = np.nan
        final_std_of_stds = np.nan
        debug_print(f"[get_semantic_similarity_windows] Skipping std calculations for window_size 2")
    else:
        final_std_of_means = np.std(window_means)
        # Filter out NaN stds before calculating mean/std of stds
        valid_stds = [s for s in window_stds if not pd.isna(s)]
        if valid_stds:
            final_mean_of_stds = np.mean(valid_stds)
            final_std_of_stds = np.std(valid_stds) if len(valid_stds) > 1 else np.nan
        else:
            final_mean_of_stds = np.nan
            final_std_of_stds = np.nan
    
    final_mean_of_medians = np.mean(window_medians)
    
    debug_print(f"[get_semantic_similarity_windows] Final aggregated statistics:")
    debug_print(f"  - Mean of window means: {final_mean:.6f} (NaN: {pd.isna(final_mean)})")
    debug_print(f"  - Std of window means: {final_std_of_means:.6f} (NaN: {pd.isna(final_std_of_means)})")
    debug_print(f"  - Mean of window stds: {final_mean_of_stds:.6f} (NaN: {pd.isna(final_mean_of_stds)})")
    debug_print(f"  - Std of window stds: {final_std_of_stds:.6f} (NaN: {pd.isna(final_std_of_stds)})")
    debug_print(f"  - Mean of window medians: {final_mean_of_medians:.6f} (NaN: {pd.isna(final_mean_of_medians)})")
    
    result = (final_mean, final_std_of_means, final_mean_of_stds, final_std_of_stds, final_mean_of_medians)
    debug_print(f"[get_semantic_similarity_windows] === END: window_size={window_size}, result={result} ===\n")
    
    if return_details:
        return result, window_details or []
    return result

def get_sentence_wise_similarity(embedding_vectors, return_details=False, exclude_punctuation=False):
    """
    Calculate semantic similarity grouped by sentences.
    Groups tokens by sentence boundaries (., !, ?) and computes similarity between sentences.
    """
    import re
    
    # Check if embedding_vectors is empty
    if not embedding_vectors:
        summary = _nan_summary()
        return (summary, []) if return_details else summary
    
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
            debug_print(f"[get_sentence_wise_similarity] WARNING: Not enough tokens ({len(tokens)} < 2) for sentence similarity")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        # Otherwise, treat as single sentence and return NaN (no inter-sentence similarity)
        debug_print(f"[get_sentence_wise_similarity] WARNING: No sentence boundaries found (no tokens ending with . ! or ?). "
                   f"This may occur if punctuation was removed during tokenization. "
                   f"Found {len(tokens)} tokens. First few tokens: {list(tokens[:5])}")
        summary = _nan_summary()
        return (summary, []) if return_details else summary
    
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
        debug_print(f"[get_sentence_wise_similarity] WARNING: Found {len(sentences)} sentence(s), but need at least 2 for inter-sentence similarity")
        summary = _nan_summary()
        return (summary, []) if return_details else summary
    
    # Optionally remove punctuation within each sentence
    if exclude_punctuation:
        sentences = [filter_punctuation_tokens(sentence) for sentence in sentences]
        sentences = [sent for sent in sentences if sent]

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
        summary = _nan_summary()
        return (summary, []) if return_details else summary
    
    # Calculate similarity matrix between sentences
    sentence_sim_matrix = get_cosine_similarity_matrix(
        [(f"sentence_{i}", emb) for i, emb in enumerate(sentence_embeddings)]
    )
    
    # Calculate statistics (returns 3-tuple: mean, std, median)
    mean_val, std_val, median_val = calculate_window_statistics(sentence_sim_matrix)
    
    # Convert to 5-tuple format to match regular window sizes:
    # (mean_of_window_means, std_of_window_means, mean_of_window_stds, std_of_window_stds, mean_of_window_medians)
    # For sentence similarity, there's only one "window" (the sentence similarity matrix),
    # so std_of_means and std_of_stds are NaN
    summary = (mean_val, np.nan, std_val, np.nan, median_val)
    
    if not return_details:
        return summary
    
    sentence_texts = [tokens_to_text([token for token, _ in sentence]) for sentence in sentences]
    
    detail_rows = []
    for i in range(len(sentence_embeddings)):
        for j in range(i + 1, len(sentence_embeddings)):
            similarity_value = sentence_sim_matrix[i, j]
            if pd.isna(similarity_value):
                continue
            detail_rows.append({
                'sentence_i_index': i + 1,
                'sentence_j_index': j + 1,
                'sentence_i_text': sentence_texts[i],
                'sentence_j_text': sentence_texts[j],
                'similarity': similarity_value
            })
    
    return summary, detail_rows

def calculate_window_statistics(cosine_similarity_matrix, skip_std=False):
    matrix_values = pd.DataFrame(cosine_similarity_matrix).stack()
    
    # Filter out NaN values for debugging
    valid_values = matrix_values.dropna()
    nan_count = len(matrix_values) - len(valid_values)
    
    debug_print(f"    [calculate_window_statistics] Total values: {len(matrix_values)}, Valid: {len(valid_values)}, NaN: {nan_count}")
    
    if len(valid_values) == 0:
        debug_print(f"    [calculate_window_statistics] ERROR: All values are NaN! Returning NaN for all statistics.")
        debug_print(f"    [calculate_window_statistics] Matrix shape: {cosine_similarity_matrix.shape}")
        debug_print(f"    [calculate_window_statistics] Matrix:\n{cosine_similarity_matrix}")
        return np.nan, np.nan, np.nan
    
    mean_val = matrix_values.mean()
    
    # Skip std calculation if requested (e.g., for window_size 2 where std is always 0)
    if skip_std:
        std_val = np.nan
        debug_print(f"    [calculate_window_statistics] Skipping std calculation (skip_std=True)")
    else:
        # Also check if there's only 1 unique value (which makes std meaningless)
        unique_values = valid_values.unique()
        if len(unique_values) <= 1:
            std_val = np.nan
            debug_print(f"    [calculate_window_statistics] Skipping std calculation (only {len(unique_values)} unique value(s))")
        else:
            std_val = matrix_values.std()
    
    median_val = matrix_values.median()
    
    debug_print(f"    [calculate_window_statistics] Statistics - Mean: {mean_val:.6f}, Std: {std_val:.6f}, Median: {median_val:.6f}")
    
    return mean_val, std_val, median_val
