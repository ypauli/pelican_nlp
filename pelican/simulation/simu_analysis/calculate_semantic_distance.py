import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

##################################################
#               HELPER FUNCTIONS
##################################################

def cosine_distance(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    Returns a scalar value in the range [0,1], or 0 if the inputs are invalid.
    """
    # Check if either vector contains NaN or infinite values
    if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)) or np.any(np.isinf(vec1)) or np.any(np.isinf(vec2)):
        return 0  # Return 0 if invalid inputs are detected
    
    try:
        return cosine_similarity([vec1], [vec2])[0][0]
    except Exception:
        return 0  # Return 0 if an error occurs during computation


def parse_section(rows, sequences_to_ignore, punctuation_tokens):
    """
    Given all the rows (tokens + embeddings) for one section,
    this function identifies and stores any 'prompt' tokens
    matching sequences_to_ignore, then returns:
    
    prompt_tokens, prompt_embs, normal_tokens, normal_embs
    """
    prompt_tokens = []
    prompt_embs = []
    normal_tokens = []
    normal_embs = []
    
    i = 0
    while i < len(rows):
        token = str(rows[i][0]).strip('"')
        emb = np.array(rows[i][1:], dtype=float)
        
        # Try to match each ignore/prompt sequence
        matched_sequence = None
        for seq in sequences_to_ignore:
            seq_len = len(seq)
            if i + seq_len <= len(rows):
                # Extract the tokens in the window [i : i+seq_len]
                window_tokens = [r[0] for r in rows[i : i+seq_len]]
                if window_tokens == seq:
                    matched_sequence = seq
                    break
        
        if matched_sequence:
            # We found a prompt sequence => store in prompt arrays
            seq_len = len(matched_sequence)
            seq_rows = rows[i : i+seq_len]
            
            # Gather all tokens/embeddings for this matched sequence
            these_tokens = [r[0] for r in seq_rows]
            these_embs   = np.array([r[1:] for r in seq_rows], dtype=float)
            
            prompt_tokens.append(these_tokens)
            prompt_embs.append(these_embs)
            
            i += seq_len  # jump past these tokens
        else:
            # No matched prompt => treat as normal token
            # Optionally exclude punctuation
            if token not in punctuation_tokens:
                normal_tokens.append(token)
                normal_embs.append(emb)
            i += 1
    
    normal_embs = np.array(normal_embs, dtype=float) if normal_embs else np.array([])
    return prompt_tokens, prompt_embs, normal_tokens, normal_embs


##################################################
#      MAIN DATA-LOADING + METRIC FUNCTIONS
##################################################

def load_embeddings(csv_file):
    """
    Reads the CSV file (which has lines with 'Token' to mark sections)
    and returns a list of sections. Each section is a tuple:
    
      (
        [ [prompt_seq1_tokens], [prompt_seq2_tokens], ... ],  # prompt_tokens
        [ prompt_seq1_embs,    prompt_seq2_embs,    ... ],    # prompt_embs
        [normal_token_1, normal_token_2, ...],                # tokens
        np.array([...])                                       # embeddings
      )
    """
    # Define the sequences of tokens to ignore -> treat as prompts
    sequences_to_ignore = [
        ["seit", "letzter", "woche", "habe", "ich"],
        ["von", "hier", "aus", "bis", "zum", "nächsten", "supermarkt", "gelangt", "man"],
        ["in", "meinem", "letzten", "traum"],
        ["ich", "werde", "so", "viele", "tiere", "aufzählen", "wie", "möglich:", "pelikan,"]
    ]

    # Punctuation tokens to ignore from normal text
    punctuation_tokens = {",", ";", "!", "?", ":", "(", ")", "[", "]", "{", "}", "\"", "'"}

    df = pd.read_csv(csv_file, header=None)

    # First, split the DataFrame into sections, ignoring lines where row[0] == 'Token' 
    # except as a marker for new sections
    raw_sections = []
    current_rows = []
    for row in df.itertuples(index=False):
        # row is a tuple (col0, col1, col2, ...)
        if row[0] == 'Token':
            # We reached a boundary => close off the current section (if not empty)
            if current_rows:
                raw_sections.append(current_rows)
                current_rows = []
        else:
            current_rows.append(row)
    
    # Append the last section if it has data
    if current_rows:
        raw_sections.append(current_rows)

    # Now parse each section to split out prompt vs. normal tokens
    sections = []
    for rows in raw_sections:
        prompt_tokens, prompt_embs, normal_tokens, normal_embs = parse_section(
            rows, sequences_to_ignore, punctuation_tokens
        )
        sections.append((prompt_tokens, prompt_embs, normal_tokens, normal_embs))
    
    return sections


def average_semantic_distances(section_data):
    """
    Calculates:
     1) average cosine similarity between consecutive normal tokens
     2) average cosine similarity between ALL pairs of normal tokens
    For each section, returns two lists (consecutive_distances, all_pairs_distances).
    """
    consecutive_distances = []
    all_pairs_distances   = []
    
    for (prompt_tokens, prompt_embs, tokens, embeddings) in section_data:
        if len(tokens) < 2:
            # Not enough tokens for pairwise calculations
            consecutive_distances.append(0.0)
            all_pairs_distances.append(0.0)
            continue
        
        # Distances between consecutive tokens
        distances_consec = []
        for i in range(len(tokens) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            distances_consec.append(cosine_distance(vec1, vec2))
        
        # Distances between all token pairs
        distances_all = []
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                vec1 = embeddings[i]
                vec2 = embeddings[j]
                distances_all.append(cosine_distance(vec1, vec2))
        
        # Averages
        avg_consec = np.mean(distances_consec) if distances_consec else 0.0
        avg_all    = np.mean(distances_all)    if distances_all    else 0.0
        
        consecutive_distances.append(avg_consec)
        all_pairs_distances.append(avg_all)

    return consecutive_distances, all_pairs_distances


def average_sentence_distances(section_data):
    """
    Splits each section's normal text into sentences by '.' tokens,
    computes:
      - average cosine similarity among all pairs of sentence-mean embeddings
      - a word mover’s distance (WMD-like measure) among all pairs
    Returns two lists, each containing the per-section averages.
    """
    sentence_distance_list = []
    wmd_distance_list      = []
    
    # Define punctuation that does NOT break sentences
    punctuation_tokens = {",", ";", "!", "?", ":", "(", ")", "[", "]", "{", "}", "\"", "'"}

    for (prompt_tokens, prompt_embs, tokens, embeddings) in section_data:
        if len(tokens) == 0:
            sentence_distance_list.append(0.0)
            wmd_distance_list.append(0.0)
            continue

        # Collect sentence embeddings
        sentences     = []
        sentence_vecs = []
        curr_sentence = []
        
        for t, emb in zip(tokens, embeddings):
            if t not in punctuation_tokens: 
                curr_sentence.append(emb)
            # If t i . or t contains ., we've reached the end of a sentence
            if t == '.' or '.' in t:
                if curr_sentence:
                    sentences.append(np.array(curr_sentence))
                    sentence_vecs.append(np.mean(curr_sentence, axis=0))
                curr_sentence = []
        
        # If there's any remainder after the last '.' 
        if curr_sentence:
            sentences.append(np.array(curr_sentence))
            sentence_vecs.append(np.mean(curr_sentence, axis=0))
        
        # If we have fewer than 2 sentences, distance among them doesn't apply
        if len(sentence_vecs) < 2:
            sentence_distance_list.append(0.0)
            wmd_distance_list.append(0.0)
            continue
        
        # Compute pairwise distances among sentence vectors
        cos_dists = []
        wmd_dists = []
        for i in range(len(sentence_vecs)):
            for j in range(i+1, len(sentence_vecs)):
                cos_dists.append(cosine_distance(sentence_vecs[i], sentence_vecs[j]))
                
                # Word Mover's Distance (approx): for each token in sentence i,
                # find its minimal distance to a token in sentence j (via cdist).
                pairwise = cdist(sentences[i], sentences[j], metric='cosine')
                wmd_ij = np.mean(np.min(pairwise, axis=1))
                wmd_dists.append(wmd_ij)

        sentence_distance_list.append(np.mean(cos_dists) if cos_dists else 0.0)
        wmd_distance_list.append(np.mean(wmd_dists) if wmd_dists else 0.0)

    return sentence_distance_list, wmd_distance_list


def average_prompt_distances(section_data):
    """
    For each section, treats all prompt tokens as "sentence 0," 
    then computes the average distance of every normal sentence
    to that prompt (both cosine and WMD).
    
    - We combine *all* prompt embeddings in the section into one 
      big "prompt" array, then average them for a single prompt vector. 
    - If multiple prompt sequences occur, they all contribute 
      to the final "prompt" embedding.
    """
    all_prompt_cosine = []
    all_prompt_wmd    = []
    all_prompt_sentence_cosine = []
    all_prompt_sentence_wmd    = []

    punctuation_tokens = {",", ";", "!", "?", ":", "(", ")", "[", "]", "{", "}", "\"", "'"}

    for (prompt_tokens, prompt_embs, tokens, embeddings) in section_data:
        
        # Flatten all prompt embeddings into a single array (if any)
        if len(prompt_embs) == 0:
            # No prompts => no prompt-based distance
            all_prompt_cosine.append(0.0)
            all_prompt_wmd.append(0.0)
            continue
        
        # Combine all prompt embeddings into one big array
        # shape: (P, D) where P is total prompt tokens, D is embedding dim
        prompt_arrays = [arr for arr in prompt_embs]  # each arr is shape (seq_len, D)
        big_prompt    = np.concatenate(prompt_arrays, axis=0)  # (sum_of_seq_lens, D)
        prompt_mean   = np.mean(big_prompt, axis=0)            # shape (D,)

        # Now split normal tokens into sentences
        sentences     = []
        sentence_vecs = []
        curr_sentence = []
        
        for t, emb in zip(tokens, embeddings):
            if t not in punctuation_tokens:
                curr_sentence.append(emb)
            # If t i . or t contains ., we've reached the end of a sentence
            if t == '.' or '.' in t:
                if curr_sentence:
                    sentences.append(np.array(curr_sentence))
                    sentence_vecs.append(np.mean(curr_sentence, axis=0))
                curr_sentence = []
        # leftover
        if curr_sentence:
            sentences.append(np.array(curr_sentence))
            sentence_vecs.append(np.mean(curr_sentence, axis=0))

        if len(sentence_vecs) == 0:
            # No normal sentences
            all_prompt_sentence_cosine.append(0.0)
            all_prompt_sentence_wmd.append(0.0)
            all_prompt_cosine.append(0.0)
            all_prompt_wmd.append(0.0)
            continue

        # For each sentence, compute distance to prompt
        cos_dists = []
        wmd_dists = []
        for sent_vec, sent_arr in zip(sentence_vecs, sentences):
            # Cosine similarity to prompt mean
            cos_val = cosine_distance(prompt_mean, sent_vec)
            cos_dists.append(cos_val)
            
            # Word-mover-like distance
            pairwise = cdist(big_prompt, sent_arr, metric='cosine')
            wmd_val  = np.mean(np.min(pairwise, axis=1))
            wmd_dists.append(wmd_val)

        # Store average of distances for this entire section
        all_prompt_sentence_cosine.append(cos_dists)
        all_prompt_sentence_wmd.append(wmd_dists)
        
        all_prompt_cosine.append(np.mean(cos_dists))
        all_prompt_wmd.append(np.mean(wmd_dists))

    return all_prompt_sentence_cosine, all_prompt_sentence_wmd, all_prompt_cosine, all_prompt_wmd

##################################################
#                    RUN
##################################################

def run(csv_path):
    """
    Orchestrates loading embeddings from the CSV file, computes:
      1) average token-to-token distances (consecutive & all-pairs)
      2) average sentence-to-sentence distances
      3) average distance of each sentence to its prompt
    and prints/returns them.
    """
    # 1) Load sections with both prompt and normal embeddings
    section_data = load_embeddings(csv_path)

    # 2) Compute average semantic distances among normal tokens
    avg_consec, avg_all_pairs = average_semantic_distances(section_data)

    # 3) Compute average sentence distances
    avg_sentence_distances, wmd_sentence_distances = average_sentence_distances(section_data)

    # 4) Compute average prompt distances (sentence 0)
    all_prompt_sentence_cosine, all_prompt_sentence_wmd, avg_prompt_cosine, avg_prompt_wmd = average_prompt_distances(section_data)

    # Print results
    print("Avg. consecutive token distances:", avg_consec)
    print("Avg. all-pair token distances:",    avg_all_pairs)
    print("Avg. sentence-to-sentence distances (cosine):", avg_sentence_distances)
    print("Avg. sentence-to-sentence WMD:",               wmd_sentence_distances)
    print("Avg. prompt-to-sentence distances (cosine):",  avg_prompt_cosine)
    print("Avg. prompt-to-sentence WMD:",                 avg_prompt_wmd)
    print("Avg. cosine distance from prompt for each sentence:", all_prompt_sentence_cosine)
    print("Avg. WMD distance from prompt for each sentence:",    all_prompt_sentence_wmd)

    return (
        avg_consec, 
        avg_all_pairs, 
        avg_sentence_distances, 
        wmd_sentence_distances, 
        avg_prompt_cosine,
        avg_prompt_wmd,
        all_prompt_sentence_cosine,
        all_prompt_sentence_wmd
    )