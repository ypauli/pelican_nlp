import csv
import os
import numpy as np

def store_features_to_csv(input_data, output_file):

    print('The type of the input data is: ', type(input_data))

    if isinstance(input_data, dict):

        embeddings = input_data.get('embeddings')
        tokens = input_data.get('tokens_logits')

        if embeddings is not None:
            if len(embeddings) != len(tokens):
                raise ValueError("The number of embeddings must match the number of tokens.")

        # Ensure embeddings is a numpy array for consistent handling
        embeddings = np.array(embeddings, dtype=np.float32)

        # Always open in append mode and write a new header before appending
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

        new_fieldnames = list(input_data[0].keys())

        # Always open in append mode and write a new header before appending
        with open(output_file, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=new_fieldnames)
            writer.writeheader()
            for row in input_data:
                writer.writerow(row)

    print(f"Data successfully stored to {output_file}")