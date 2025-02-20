import os
import csv
import numpy as np

def store_features_to_csv(input_data, results_path, corpus, metric):

    if not isinstance(input_data, dict) or not input_data:
        raise ValueError("Input data must be a non-empty dictionary.")

    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)

    # Extract subject, session, task from the path
    parts = results_path.split(os.sep)
    if len(parts) < 4:
        raise ValueError("Invalid results_path format. Expected 'project/subject/session/task'.")
    _, subject, session, task = parts[-4:]

    # Input data: keys are tokens, values are their corresponding embeddings
    tokens = list(input_data.keys())
    metric_values = np.array(list(input_data.values()), dtype=np.float32)

    # Ensure token count and embedding shape match
    if len(tokens) != len(metric_values):
        raise ValueError(f"Mismatch: {len(tokens)} tokens but {len(metric_values)} metric values.")

    output_filename = f"{subject}_{session}_{task}_{corpus}_{metric}.csv"
    output_filepath = os.path.join(results_path, output_filename)

    file_exists = os.path.exists(output_filepath)

    with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Get number of embedding dimensions
        num_dimensions = metric_values.shape[1]
        header = ['Token'] + [f"Dim_{i}" for i in range(num_dimensions)]

        if not file_exists:
            writer.writerow(header)
        else:
            writer.writerow([])  # Separate sections
            writer.writerow([f"New Section"])
            writer.writerow(header)

        # Write token and its corresponding embedding
        for token, embedding in zip(tokens, metric_values):
            writer.writerow([token] + embedding.tolist())