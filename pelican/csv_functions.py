import os
import csv
import numpy as np


def store_features_to_csv(input_data, results_path, corpus):
    print('📝 Storing results as CSV...')

    if not isinstance(input_data, list) or not input_data:
        raise ValueError("❌ Input data must be a non-empty list of dictionaries.")

    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)

    # Extract subject, session, task from the path
    parts = results_path.split(os.sep)
    if len(parts) < 4:
        raise ValueError("❌ Invalid results_path format. Expected 'project/subject/session/task'.")
    _, subject, session, task = parts[-4:]

    print(f'📌 Corpus: {corpus}')

    for idx, section in enumerate(input_data):
        if 'tokens' not in section or 'embeddings' not in section:
            raise ValueError(f"❌ Section {idx} is missing 'tokens' or 'embeddings' key.")

        tokens = section['tokens']
        embeddings = np.array(section['embeddings'], dtype=np.float32)

        if len(tokens) != len(embeddings):
            raise ValueError(f"❌ Mismatch: {len(tokens)} tokens but {len(embeddings)} embeddings.")

        # Determine the metric name (assumes first key after 'tokens' and 'embeddings')
        metric_keys = [key for key in section.keys() if key not in ['tokens', 'embeddings']]
        metric_name = metric_keys[0] if metric_keys else 'embeddings'

        output_filename = f"{subject}_{session}_{task}_{corpus}_{metric_name}.csv"
        output_filepath = os.path.join(results_path, output_filename)

        file_exists = os.path.exists(output_filepath)

        with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Header with embedding dimensions
            num_dimensions = embeddings.shape[1]
            header = ['Token'] + [f"Dim_{i}" for i in range(num_dimensions)]

            if not file_exists:
                writer.writerow(header)
            else:
                writer.writerow([])  # Separate sections
                writer.writerow([f"New Section"])
                writer.writerow(header)

            # Write token and its corresponding embedding
            for token, embedding in zip(tokens, embeddings):
                writer.writerow([token] + embedding.tolist())

        print(f"✅ Data stored at: {output_filepath}")


'''import os
import csv
import numpy as np

def store_features_to_csv(input_data, results_path, corpus):

    print('Storing results as CSV...')
    print(f'The input data is: {input_data}')

    # Ensure the results_path exists
    os.makedirs(results_path, exist_ok=True)

    # Determine the metric name based on the first key in input_data
    if not isinstance(input_data, dict) or not input_data:
        raise ValueError("Input data must be a non-empty dictionary.")
    metric = f"{next(iter(input_data.keys()))}"

    # Extract subject, session, task from the path
    parts = results_path.split(os.sep)
    if len(parts) < 4:
        raise ValueError("Invalid results_path format. Expected 'project/subject/session/task'.")
    _, subject, session, task = parts[-4:]

    print(f'corpus is {corpus}')

    # Create the output filename
    output_filename = f"{subject}_{session}_{task}_{corpus}_{metric}.csv"
    output_filepath = os.path.join(results_path, output_filename)

    embeddings = input_data.get('embeddings')

    if embeddings is not None:
        embeddings = np.array(embeddings, dtype=np.float32)

        tokens_embeddings = input_data.get('tokens_embeddings')
        if len(embeddings) != len(tokens_embeddings):
            raise ValueError("The number of embeddings must match the number of tokens.")

        file_exists = os.path.exists(output_filepath)
        with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Token'] + [f"Dim_{i}" for i in range(embeddings.shape[1])]

            if not file_exists:
                writer.writerow(header)
            else:
                writer.writerow([])  # Insert an empty row for separation
                writer.writerow([f"New Paragraph Section"])
                writer.writerow(header)

            for token, embedding in zip(tokens_embeddings, embeddings):
                writer.writerow([token] + embedding.tolist())

    print(f"Data successfully stored at: {output_filepath}")'''