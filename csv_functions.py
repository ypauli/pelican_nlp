import csv
import numpy as np

def store_features_to_csv(input_data, output_file):

    if isinstance(input_data, dict):
        embeddings = input_data.get('embeddings')
        tokens = input_data.get('tokens_logits')

        if embeddings is None or tokens is None:
            raise ValueError("The input dictionary must contain 'embeddings' and 'tokens_logits'.")

        if len(embeddings) != len(tokens):
            raise ValueError("The number of embeddings must match the number of tokens.")

        # Ensure embeddings is a numpy array for consistent handling
        embeddings = np.array(embeddings, dtype=np.float32)
        # Write to CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write header
            header = ['Token'] + [f"Dim_{i}" for i in range(embeddings.shape[1])]
            writer.writerow(header)

            # Write each token and its corresponding embedding
            for token, embedding in zip(tokens, embeddings):
                writer.writerow([token] + embedding.tolist())

    if isinstance(input_data, list):
        headers = input_data[0].keys()
        with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(input_data)

    else:
        raise ValueError("Input must be a dictionary or a list.")

    print(f"Data successfully stored to {output_file}")