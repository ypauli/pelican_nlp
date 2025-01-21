import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def run(csv_path):
        
    # Load embeddings and calculate the average semantic distance for each section
    section_embeddings = load_embeddings(csv_path)
    avg_distances = average_semantic_distance_per_section(section_embeddings)

    # Output the average distance for each section
    # for section_start, section_end, avg_distance in avg_distances:
    #     print(f"Average semantic distance for section from '{section_start}' to '{section_end}': {avg_distance}")
    
    return avg_distances
    

def load_embeddings(csv_file):
    # Define the sequences of tokens to ignore
    sequences_to_ignore = [
        ["seit", "letzter", "woche", "habe", "ich"],
        ["von", "hier", "aus", "bis", "zum", "nächsten", "supermarkt", "gelangt", "man"],
        ["als", "letztes", "habe", "ich", "geträumt"],
        ["ich", "werde", "so", "viele", "tiere", "aufzählen", "wie", "möglich", "pelikan"]
    ]

    # Read the CSV file and parse it into a DataFrame
    df = pd.read_csv(csv_file, header=None)
    sections = []
    current_section = []
    current_tokens = []  # To track tokens at the beginning of a section
    skipping = True  # Flag to determine if we are still in the "ignoring" phase

    for row in df.itertuples(index=False):
        if row[0] == 'Token':  # Start of a new section
            if current_section:  # If there's a previous section, save it
                sections.append(current_section)
            current_section = []  # Start a new section
            current_tokens = []  # Reset tokens at the start of the new section
            skipping = True  # Reset the skipping flag for the new section
        else:
            if skipping:
                current_tokens.append(row[0])  # Add the current token
                # Check if current tokens match any sequence to ignore
                if any(current_tokens == seq[:len(current_tokens)] for seq in sequences_to_ignore):
                    if any(current_tokens == seq for seq in sequences_to_ignore):  # Full match
                        current_tokens = []  # Reset tokens and keep skipping
                    continue
                else:
                    skipping = False  # Stop skipping once no match is found
                    current_section.extend((token, *row[1:]) for token in current_tokens)  # Add skipped tokens
                    current_tokens = []  # Clear the buffer
            current_section.append(row)  # Add the current embedding to the section

    if current_section:  # Append the last section
        sections.append(current_section)

    # Process each section to get tokens and embeddings
    section_embeddings = []
    for section in sections:
        tokens = [row[0] for row in section]
        embeddings = np.array([row[1:] for row in section], dtype=float)
        section_embeddings.append((tokens, embeddings))

    return section_embeddings


def cosine_distance(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


def average_semantic_distance_per_section(section_embeddings):
    avg_distances = []
    
    for tokens, embeddings in section_embeddings:
        distances = []
        
        for i in range(1, len(tokens)):
            vec1 = embeddings[i-1]
            vec2 = embeddings[i]
            distance = cosine_distance(vec1, vec2)
            distances.append(distance)
        
        avg_distance = np.mean(distances) if distances else 0
        # avg_distances.append((tokens[0], tokens[-1], avg_distance))
        avg_distances.append(avg_distance)
    
    return avg_distances