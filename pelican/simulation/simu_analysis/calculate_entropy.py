import pandas as pd
import numpy as np
import io

def run(csv_path):
    """
    Reads the CSV at csv_path, splits it into sections,
    extracts prompt rows (sentence 0) if present, splits the remainder
    into sentences, and computes average entropy per sentence (including prompt).
    
    Returns a dict:
      {
        "Section 1": {
            "Sentence 0 (Prompt)": <avg_entropy or None>,
            "Sentence 1": <avg_entropy>,
            "Sentence 2": <avg_entropy>,
            ...
        },
        "Section 2": { ... },
        ...
      }
    """
    section_entropy = calculate_entropy_per_section(csv_path)
    
    print(section_entropy)
    
    return section_entropy

def calculate_entropy_per_section(file_path):
    # Define the sequences to ignore (prompt)
    sequences_to_ignore = [
        ["se", "it", "Ġletz", "ter", "Ġwo", "che", "Ġhabe", "Ġich"],
        ["von", "Ġhier", "Ġaus", "Ġbis", "Ġzum", "ĠnÃ¤chsten", "Ġsuper", "markt", "Ġgel", "ang", "t", "Ġman"],
        ["als", "Ġletz", "tes", "Ġhabe", "Ġich", "Ġget", "r", "Ã¤", "um", "t"],
        ["ich", "Ġwerde", "Ġso", "Ġviele", "Ġti", "ere", "Ġauf", "z", "Ã¤hlen", "Ġwie", "ĠmÃ¶glich", "Ġpel", "ikan"]
    ]
    
    # Read the file line by line to detect sections
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Break into sections whenever we see the header line
    sections = []
    current_section = []
    header_line = "token,logprob_actual,logprob_max,entropy,most_likely_token"

    for line in lines:
        if line.strip() == header_line:
            # If there's an existing section, save it
            if current_section:
                sections.append(current_section)
            # Start a new section with the header
            current_section = [line.strip()]
        else:
            # Add line to the current section
            current_section.append(line.strip())
    # Add the last section if it exists
    if current_section:
        sections.append(current_section)

    section_results = {}
    
    for idx, section_lines in enumerate(sections, start=1):
        section_name = f"Section {idx}"
        
        # Build a DataFrame from these lines
        section_data = "\n".join(section_lines)
        try:
            df = pd.read_csv(io.StringIO(section_data))
        except pd.errors.EmptyDataError:
            # No valid data
            section_results[section_name] = {}
            continue

        # Ensure columns are stripped of whitespace
        df.columns = df.columns.str.strip()

        # We need at least "token" and "entropy"
        if not {"token", "entropy"}.issubset(df.columns):
            section_results[section_name] = {}
            continue
        
        # ----------------------------------------------------
        # Extract prompt rows if they match a sequence at start
        # ----------------------------------------------------
        prompt_df = pd.DataFrame()  # empty by default
        for sequence in sequences_to_ignore:
            seq_len = len(sequence)
            # Compare the top N tokens to the sequence
            if len(df) >= seq_len:
                if list(df["token"].iloc[:seq_len]) == sequence:
                    # Prompt found => store it
                    prompt_df = df.iloc[:seq_len].copy()
                    # Remove from the main DF
                    df = df.iloc[seq_len:].reset_index(drop=True)
                    break  # only handle 1 matched sequence at the start
        
        # ----------------------------------------------------
        # Compute "Sentence 0" (prompt) average entropy
        # ----------------------------------------------------
        if prompt_df.empty:
            prompt_entropy = None
        else:
            prompt_entropy = prompt_df["entropy"].mean()

        # ----------------------------------------------------
        # Split remaining tokens by '.' => separate sentences
        # ----------------------------------------------------
        sentence_list = split_df_into_sentences(df)
        
        # Compute average entropy for each sentence
        results_for_section = {}
        
        # Sentence 0 => prompt
        results_for_section["Sentence 0 (Prompt)"] = prompt_entropy
        
        # Sentences 1..N => from the main text
        for i, sent_df in enumerate(sentence_list, start=1):
            # If no rows, skip
            if sent_df.empty:
                avg_ent = None
            else:
                avg_ent = sent_df["entropy"].mean()
            results_for_section[f"Sentence {i}"] = avg_ent
        
        section_results[section_name] = results_for_section

    return section_results

def split_df_into_sentences(df):
    """
    Splits a DataFrame into a list of smaller DataFrames, where
    each sub-DataFrame ends when the token == '.'.
    
    Example:
      tokens = [A, B, ., C, D, ., E]
      => 2 sentences: [A,B], [C,D], leftover: [E] if E not ended with '.' 
    """
    sentences = []
    current_rows = []
    
    for idx in df.index:
        row = df.loc[idx]
        current_rows.append(row)
        if row["token"] == ".":
            # End of sentence
            sentence_df = pd.DataFrame(current_rows).reset_index(drop=True)
            sentences.append(sentence_df)
            current_rows = []
    
    # If leftover rows exist and didn't end with '.'
    if current_rows:
        sentence_df = pd.DataFrame(current_rows).reset_index(drop=True)
        sentences.append(sentence_df)
    
    return sentences

run("/home/ubuntu/emilia/pel_output_normal_partial/Outputs/subject_0/ses-0/a/sub-0_ses-0_group-a_timepoint-0_results_logits.csv")