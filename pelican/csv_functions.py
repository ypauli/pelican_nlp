import csv
import numpy as np
import os

def store_features_to_csv(input_data, output_file):

    print('Storing results as CSV...')
    print('The type of the input data is:', type(input_data))

    # Determine the type of data and adjust the file name
    if isinstance(input_data, dict) and 'embeddings' in input_data:
        suffix = '_embeddings'
    elif isinstance(input_data, dict) and 'tokens_logits' in input_data:
        suffix = '_logits'
    elif isinstance(input_data, list) and all(isinstance(d, dict) for d in input_data):
        suffix = '_logits'  # Assuming list of dictionaries relates to logits
    else:
        raise ValueError("Input data must be a dictionary with 'embeddings' or 'tokens_logits', or a list of dictionaries.")

    # Append the suffix before the file extension
    base, ext = os.path.splitext(output_file)
    output_file = f"{base}{suffix}{ext}"

    print(f"Updated file name to: {output_file}")

    if isinstance(input_data, dict):
        embeddings = input_data.get('embeddings')
        tokens = input_data.get('tokens_logits')

        print(f'Tokens: {tokens}')
        print(f'Embeddings: {embeddings}')

        if embeddings is not None:
            if len(embeddings) != len(tokens):
                raise ValueError("The number of embeddings must match the number of tokens.")

        # Ensure embeddings is a numpy array for consistent handling
        embeddings = np.array(embeddings, dtype=np.float32)

        # Open file in append mode and write a new header before appending
        with open(output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Token'] + [f"Dim_{i}" for i in range(embeddings.shape[1])]
            writer.writerow(header)

            for token, embedding in zip(tokens, embeddings):
                writer.writerow([token] + embedding.tolist())

    elif isinstance(input_data, list):
        # Handle list of dictionaries
        if not all(isinstance(d, dict) for d in input_data):
            raise ValueError("Each item in the list must be a dictionary.")

        fieldnames = list(input_data[0].keys())

        # Open file in append mode and write a new header before appending
        with open(output_file, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in input_data:
                writer.writerow(row)

    print(f"Data successfully stored to {output_file}")