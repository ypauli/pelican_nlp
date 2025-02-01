import pandas as pd
import numpy as np
import io

def run(csv_path):
    """
    Reads the CSV file at csv_path, splits it into sections,
    extracts & removes any prompt (ignore-sequence) at the start
    of each section, then computes both:
      1. Overall average entropy & perplexity for the section's main text (prompt excluded).
      2. Sentence-level average entropy & perplexity, with the prompt as "Sentence 0."

    Returns four lists, each of length = number_of_sections:
      avg_entropy_per_section               -> [float, float, ...]
      avg_perplexity_per_section            -> [float, float, ...]
      avg_sentence_entropy_per_section      -> [[ent0, ent1, ent2, ...], [...], ...]
      avg_sentence_perplexity_per_section   -> [[ppl0, ppl1, ppl2, ...], [...], ...]
    """
    sections = read_sections(csv_path)
    
    # Prepare final results
    avg_entropy_per_section = []
    avg_perplexity_per_section = []
    avg_sentence_entropy_per_section = []
    avg_sentence_perplexity_per_section = []
    
    for df_section in sections:
        # df_section is a DataFrame for this section
        
        if df_section.empty:
            # If section is completely empty, store placeholder results
            avg_entropy_per_section.append(None)
            avg_perplexity_per_section.append(None)
            avg_sentence_entropy_per_section.append([])
            avg_sentence_perplexity_per_section.append([])
            continue
        
        # --------------------------------------------------------------
        # 1) Remove prompt from the start (if any), treat it as Sentence 0
        # --------------------------------------------------------------
        prompt_df, main_df = extract_prompt(df_section)

        # --------------------------------------------------------------
        # 2) Compute overall metrics on the main text (prompt excluded)
        # --------------------------------------------------------------
        # Overall average entropy for the main text
        if "entropy" in main_df.columns and not main_df.empty:
            overall_entropy = main_df["entropy"].mean()
        else:
            overall_entropy = None
        
        # Overall perplexity for the main text
        # We need "logprob_actual" for perplexity
        if "logprob_actual" in main_df.columns and not main_df.empty:
            n_main = len(main_df)
            logprob_sum_main = main_df["logprob_actual"].sum()
            overall_perplexity = np.exp(-logprob_sum_main / n_main)
        else:
            overall_perplexity = None
        
        avg_entropy_per_section.append(overall_entropy)
        avg_perplexity_per_section.append(overall_perplexity)
        
        # --------------------------------------------------------------
        # 3) Compute sentence-level metrics (prompt => Sentence 0)
        # --------------------------------------------------------------
        # 3a) Prompt metrics (if prompt_df is non-empty)
        sentence_entropies = []
        sentence_perplexities = []
        
        # Prompt is always sentence 0
        if not prompt_df.empty:
            # Average entropy of prompt
            if "entropy" in prompt_df.columns:
                prompt_entropy = prompt_df["entropy"].mean()
            else:
                prompt_entropy = None
            
            # Perplexity of prompt
            if "logprob_actual" in prompt_df.columns:
                n_prompt = len(prompt_df)
                logprob_sum_prompt = prompt_df["logprob_actual"].sum()
                prompt_perplexity = np.exp(-logprob_sum_prompt / n_prompt)
            else:
                prompt_perplexity = None
        else:
            # No prompt found
            prompt_entropy = None
            prompt_perplexity = None
        
        sentence_entropies.append(prompt_entropy)
        sentence_perplexities.append(prompt_perplexity)
        
        # 3b) Split the main DF into sentences
        sentences = split_df_into_sentences(main_df)
        
        for sent_df in sentences:
            if sent_df.empty:
                # No data in this sentence
                sentence_entropies.append(None)
                sentence_perplexities.append(None)
            else:
                # Average entropy for this sentence
                if "entropy" in sent_df.columns:
                    sent_entropy = sent_df["entropy"].mean()
                else:
                    sent_entropy = None
                
                # Perplexity for this sentence
                if "logprob_actual" in sent_df.columns:
                    n_sent = len(sent_df)
                    logprob_sum = sent_df["logprob_actual"].sum()
                    sent_perplexity = np.exp(-logprob_sum / n_sent)
                else:
                    sent_perplexity = None
                
                sentence_entropies.append(sent_entropy)
                sentence_perplexities.append(sent_perplexity)
        
        # Collect the sentence-level lists
        avg_sentence_entropy_per_section.append(sentence_entropies)
        avg_sentence_perplexity_per_section.append(sentence_perplexities)
    
    # Print or inspect results
    # print("Average Perplexity per Section:", avg_perplexity_per_section)
    # print("Average Entropy per Section:", avg_entropy_per_section)
    # print("Sentence-level Perplexity:", avg_sentence_perplexity_per_section)
    # print("Sentence-level Entropy:", avg_sentence_entropy_per_section)
        
    # Return the four aggregated lists
    return (
        avg_perplexity_per_section,
        avg_entropy_per_section,
        avg_sentence_perplexity_per_section,
        avg_sentence_entropy_per_section,
    )


# -------------------------------------------------------------------------
#         HELPER FUNCTIONS
# -------------------------------------------------------------------------

def read_sections(file_path):
    """
    Reads the file line by line, splitting into sections
    whenever it encounters the known header line:
      'token,logprob_actual,logprob_max,entropy,most_likely_token'

    Returns a list of DataFrames, one per section.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    header_line = "token,logprob_actual,logprob_max,entropy,most_likely_token"
    sections = []
    current_section = []
    
    for line in lines:
        line = str(line)
        if line.strip() == header_line:
            # If there's an existing section, finish it
            if current_section:
                df_sec = lines_to_df(current_section)
                sections.append(df_sec)
            # Start a new section with the header
            current_section = [line.strip()]
        else:
            current_section.append(line.strip())
    
    # Add the final section if present
    if current_section:
        df_sec = lines_to_df(current_section)
        sections.append(df_sec)
    
    return sections


def lines_to_df(lines):
    """
    Converts a list of strings into a single DataFrame by joining
    them into a CSV structure in memory.
    """
    section_data = "\n".join(lines)
    try:
        df = pd.read_csv(io.StringIO(section_data))
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def extract_prompt(df):
    """
    Checks if the start of df matches any known prompt (ignore) sequence.
    If so, returns (prompt_df, remainder_df). If no match, returns (empty_df, df).
    
    The prompt is "Sentence 0".
    """
    sequences_to_ignore = [
        ["se", "it", "Ġletz", "ter", "Ġwo", "che", "Ġhabe", "Ġich"],
        ["von", "Ġhier", "Ġaus", "Ġbis", "Ġzum", "ĠnÃ¤chsten", "Ġsuper", "markt", "Ġgel", "ang", "t", "Ġman"],
        ["in", "Ġmeinem", "Ġletzten", "Ġtra", "um"],
        ['ich', 'Ġwerde', 'Ġso', 'Ġviele', 'Ġti', 'ere', 'Ġauf', 'z', 'Ã¤hlen', 'Ġwie', 'ĠmÃ¶glich', ':', 'Ġpel']
    ]
    
    if df.empty:
        return pd.DataFrame(), df
    
    for sequence in sequences_to_ignore:
        seq_len = len(sequence)
        if len(df) >= seq_len:
            # Check if top N tokens match
            top_tokens = list(df["token"].iloc[:seq_len])
            top_tokens = [str(token).strip('"') for token in top_tokens]
            if top_tokens == sequence:
                # Prompt found
                prompt_part = df.iloc[:seq_len].copy()
                remainder_part = df.iloc[seq_len:].copy().reset_index(drop=True)
                return prompt_part, remainder_part
    
    # If no match, no prompt
    return pd.DataFrame(), df


def split_df_into_sentences(df):
    """
    Splits df into a list of smaller DataFrames by token == '.'.
    
    Each returned DataFrame is one sentence, ending right
    after encountering '.' in the "token" column.
    If leftover rows do not end with '.', they become the
    final 'sentence' as well.
    """
    sentences = []
    current_rows = []
    
    for idx in df.index:
        row = df.loc[idx]
        current_rows.append(row)
        # If the the token contains a ., the sentence ends
        if "." in str(row["token"]):
            # End of sentence
            sentence_df = pd.DataFrame(current_rows).reset_index(drop=True)
            sentences.append(sentence_df)
            current_rows = []
    
    # If leftover rows remain (didn't end on '.'), treat them as one sentence
    if current_rows:
        sentence_df = pd.DataFrame(current_rows).reset_index(drop=True)
        sentences.append(sentence_df)
    
    return sentences