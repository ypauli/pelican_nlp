import pandas as pd
import numpy as np
import io

def run(csv_path):

    # Calculate perplexity for each section
    section_perplexities = calculate_entropy_per_section(csv_path)
    
    # Print results
    # for section, perplexity in section_perplexities.items():
    #     if perplexity is not None:
    #         print(f"{section}: Perplexity = {perplexity:.4f}")
    #     else:
            
    #         print(f"{section}: Skipped (no valid data after ignoring sequences).")
    
    return section_perplexities


def calculate_entropy_per_section(file_path):
    # Define the sequences to ignore
    sequences_to_ignore = [
        ["se", "it", "Ġletz", "ter", "Ġwo", "che", "Ġhabe", "Ġich"],
        ["von", "Ġhier", "Ġaus", "Ġbis", "Ġzum", "ĠnÃ¤chsten", "Ġsuper", "markt", "Ġgel", "ang", "t", "Ġman"],
        ["als", "Ġletz", "tes", "Ġhabe", "Ġich", "Ġget", "r", "Ã¤", "um", "t"],
        ["ich", "Ġwerde", "Ġso", "Ġviele", "Ġti", "ere", "Ġauf", "z", "Ã¤hlen", "Ġwie", "ĠmÃ¶glich", "Ġpel", "ikan"]
    ]

    # Read the file line by line to detect sections
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sections = []
    current_section = []
    
    for line in lines:
        # Check if the line is a header
        if line.strip() == "token,logprob_actual,logprob_max,entropy,most_likely_token":
            # If there's an existing section, save it
            if current_section:
                sections.append(current_section)
            # Start a new section
            current_section = []
            current_section.append(line.strip())  # Include the header
        else:
            # Add line to the current section
            current_section.append(line.strip())
    
    # Add the last section if it exists
    if current_section:
        sections.append(current_section)
    
    # Process each section
    section_results = {}
    for idx, section in enumerate(sections):

        # Create a DataFrame from the section
        section_data = "\n".join(section)
        try:
            df = pd.read_csv(io.StringIO(section_data))
        except pd.errors.EmptyDataError:
            # Handle cases where the section might be empty
            section_results[f"Section {idx + 1}"] = None
            continue

        # Ensure no extra spaces in column names
        df.columns = df.columns.str.strip()

        # Verify the necessary columns exist
        required_columns = {"token", "entropy"}
        if not required_columns.issubset(df.columns):
            section_results[f"Section {idx + 1}"] = None
            continue

        # Remove rows corresponding to sequences to ignore at the beginning of the section
        for sequence in sequences_to_ignore:
            seq_length = len(sequence)
            if list(df['token'].iloc[:seq_length]) == sequence:
                df = df.iloc[seq_length:]
                break
        
        # Skip processing if the section is empty after filtering
        if df.empty:
            section_results[f"Section {idx + 1}"] = None
            continue

        # Calculate perplexity
        average_entropy = df['entropy'].mean()
        # print(f"Entropy: {average_entropy}")
                
        section_results[idx] = average_entropy
    
    return section_results