import os
import csv
import numpy as np

def store_features_to_csv(input_data, results_path, corpus, metric):
    # Get the base derivatives path (up to 'derivatives' folder)
    derivatives_path = os.path.dirname(os.path.dirname(os.path.dirname(results_path)))

    # Determine the appropriate metric folder path
    if metric.startswith('semantic-similarity') or metric in ['consecutive-similarities', 'cosine-similarity-matrix']:
        metric_folder = 'semantic-similarity'
    else:
        metric_folder = 'embeddings'

    # Extract subject, session, task from the original path
    parts = results_path.split(os.sep)
    if len(parts) < 4:
        raise ValueError("Invalid results_path format. Expected 'project/subject/session/task'.")
    _, subject, session, task = parts[-4:]

    # Construct the new results path
    final_results_path = os.path.join(derivatives_path, metric_folder, subject, session, task)

    # Ensure results directory exists
    os.makedirs(final_results_path, exist_ok=True)

    output_filename = f"{subject}_{session}_{task}_{corpus}_{metric}.csv"
    output_filepath = os.path.join(final_results_path, output_filename)
    file_exists = os.path.exists(output_filepath)

    if metric=='embeddings':
        if not isinstance(input_data, dict) or not input_data:
            raise ValueError("Input data must be a non-empty dictionary.")

        # Input data: keys are tokens, values are their corresponding embeddings
        tokens = list(input_data.keys())
        metric_values = np.array(list(input_data.values()), dtype=np.float32)

        # Ensure token count and embedding shape match
        if len(tokens) != len(metric_values):
            raise ValueError(f"Mismatch: {len(tokens)} tokens but {len(metric_values)} metric values.")

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

    elif metric == 'consecutive-similarities':
        with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Index', 'Consecutive_Similarity', 'Mean_Similarity'])
            else:
                writer.writerow([])  # Separate sections
                writer.writerow(['New Section'])
                writer.writerow(['Index', 'Consecutive_Similarity', 'Mean_Similarity'])

            for idx, sim in enumerate(input_data['consecutive_similarities']):
                writer.writerow([idx, sim, input_data['mean_similarity']])

    elif metric == 'cosine-similarity-matrix':
        with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Matrix'])
            else:
                writer.writerow([])  # Separate sections
                writer.writerow(['New Section'])

            # Write the matrix
            for row in input_data:
                writer.writerow(row)

    elif metric.startswith('semantic-similarity-window-'):
        with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Metric', 'Similarity_Score'])
            else:
                writer.writerow([])  # Separate sections
                writer.writerow(['New Section'])
                writer.writerow(['Metric', 'Similarity_Score'])

            metrics = list(input_data.keys())
            for metric in metrics:
                writer.writerow([metric, input_data[metric]])

    elif metric=='logits':
        with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = list(input_data[0].keys()) if input_data else []

            if not file_exists:
                writer.writerow(header)
            else:
                writer.writerow([])  # Separate sections
                writer.writerow([f"New Section"])
                writer.writerow(header)

            for entry in input_data:
                writer.writerow(entry.values())