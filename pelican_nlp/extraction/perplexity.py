import pandas as pd
import numpy as np
import io


def run(csv_path):
    """
    Reads the CSV file at csv_path, splits it into sections,
    extracts & removes any prompt (ignore-sequence) at the start
    of each section, then computes perplexity for:
      1. The section's main text (prompt excluded).
      2. Sentence-level perplexity, with the prompt as "Sentence 0."

    Returns two lists, each of length = number_of_sections:
      avg_perplexity_per_section            -> [float, float, ...]
      avg_sentence_perplexity_per_section   -> [[ppl0, ppl1, ppl2, ...], [...], ...]
    """
    sections = read_sections(csv_path)

    # Prepare final results
    avg_perplexity_per_section = []
    avg_sentence_perplexity_per_section = []

    for df_section in sections:
        # df_section is a DataFrame for this section
        if df_section.empty:
            # If section is completely empty, store placeholder results
            avg_perplexity_per_section.append(None)
            avg_sentence_perplexity_per_section.append([])
            continue

        # --------------------------------------------------------------
        # 1) Remove prompt from the start (if any), treat it as Sentence 0
        # --------------------------------------------------------------
        prompt_df, main_df = extract_prompt(df_section)

        # --------------------------------------------------------------
        # 2) Compute overall perplexity on the main text (prompt excluded)
        # --------------------------------------------------------------
        if "logprob_actual" in main_df.columns and not main_df.empty:
            n_main = len(main_df)
            logprob_sum_main = main_df["logprob_actual"].sum()
            overall_perplexity = np.exp(-logprob_sum_main / n_main)
        else:
            overall_perplexity = None

        avg_perplexity_per_section.append(overall_perplexity)

        # --------------------------------------------------------------
        # 3) Compute sentence-level perplexity (prompt => Sentence 0)
        # --------------------------------------------------------------
        sentence_perplexities = []

        # Prompt is always sentence 0
        if not prompt_df.empty and "logprob_actual" in prompt_df.columns:
            n_prompt = len(prompt_df)
            logprob_sum_prompt = prompt_df["logprob_actual"].sum()
            prompt_perplexity = np.exp(-logprob_sum_prompt / n_prompt)
        else:
            prompt_perplexity = None

        sentence_perplexities.append(prompt_perplexity)

        # 3b) Split the main DF into sentences
        sentences = split_df_into_sentences(main_df)

        for sent_df in sentences:
            if sent_df.empty or "logprob_actual" not in sent_df.columns:
                sentence_perplexities.append(None)
            else:
                n_sent = len(sent_df)
                logprob_sum = sent_df["logprob_actual"].sum()
                sent_perplexity = np.exp(-logprob_sum / n_sent)
                sentence_perplexities.append(sent_perplexity)

        avg_sentence_perplexity_per_section.append(sentence_perplexities)

    return (
        avg_perplexity_per_section,
        avg_sentence_perplexity_per_section,
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
            if current_section:
                df_sec = lines_to_df(current_section)
                sections.append(df_sec)
            current_section = [line.strip()]
        else:
            current_section.append(line.strip())

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
            top_tokens = list(df["token"].iloc[:seq_len])
            top_tokens = [str(token).strip('"') for token in top_tokens]
            if top_tokens == sequence:
                prompt_part = df.iloc[:seq_len].copy()
                remainder_part = df.iloc[seq_len:].copy().reset_index(drop=True)
                return prompt_part, remainder_part

    return pd.DataFrame(), df


def split_df_into_sentences(df):
    """
    Splits df into a list of smaller DataFrames by token == '.'.
    Each returned DataFrame is one sentence, ending right after encountering '.'
    in the "token" column. If leftover rows do not end with '.', they become the
    final 'sentence' as well.
    """
    sentences = []
    current_rows = []

    for idx in df.index:
        row = df.loc[idx]
        current_rows.append(row)
        if "." in str(row["token"]):
            sentence_df = pd.DataFrame(current_rows).reset_index(drop=True)
            sentences.append(sentence_df)
            current_rows = []

    if current_rows:
        sentence_df = pd.DataFrame(current_rows).reset_index(drop=True)
        sentences.append(sentence_df)

    return sentences