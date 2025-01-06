import csv
import numpy as np
import os

def store_features_to_csv(input_data, output_file, append=False):

    print('The type of the input data is: ', type(input_data))

    if isinstance(input_data, dict):

        embeddings = input_data.get('embeddings')
        tokens = input_data.get('tokens_logits')

        if embeddings is not None:
            if len(embeddings) != len(tokens):
                raise ValueError("The number of embeddings must match the number of tokens.")

        # Ensure embeddings is a numpy array for consistent handling
        embeddings = np.array(embeddings, dtype=np.float32)

        # Write to CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Token'] + [f"Dim_{i}" for i in range(embeddings.shape[1])]
            writer.writerow(header)
            for token, embedding in zip(tokens, embeddings):
                writer.writerow([token] + embedding.tolist())

    elif isinstance(input_data, list):

        # Handle list of dictionaries
        if not all(isinstance(d, dict) for d in input_data):
            raise ValueError("Each item in the list must be a dictionary.")

        # Read existing file data, if any
        if os.path.exists(output_file):
            with open(output_file, mode='r', newline='', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                existing_data = list(reader)
                existing_fieldnames = reader.fieldnames or []
        else:
            existing_data = []
            existing_fieldnames = []

        # Generate new fieldnames based on input data

        new_fieldnames = []

        for idx, key in enumerate(input_data[0].keys()):
            new_fieldnames.append(f"{key}_{len(existing_fieldnames)}")

        # Update the fieldnames with new headers

        updated_fieldnames = existing_fieldnames + new_fieldnames

        # Align existing data with new headers

        for row in existing_data:
            for field in new_fieldnames:
                if field not in row:
                    row[field] = None

        # Add new data as additional columns

        for i, row in enumerate(existing_data):
            if i < len(input_data):
                for key, value in input_data[i].items():
                    row[f"{key}_{len(existing_fieldnames)}"] = value

        # Handle additional rows if input_data has more entries than existing_data

        for i in range(len(existing_data), len(input_data)):

            new_row = {field: None for field in updated_fieldnames}  # Initialize all fields with None
            for key, value in input_data[i].items():
                new_row[f"{key}_{len(existing_fieldnames)}"] = value
            existing_data.append(new_row)

        # Write the updated data back to the file

        with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=updated_fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)

    print(f"Data successfully stored to {output_file}")

'''
    elif isinstance(input_data, list):

        # Handle list of dictionaries
        if not all(isinstance(d, dict) for d in input_data):
            raise ValueError("Each item in the list must be a dictionary.")

        mode = 'a' if append else 'w'  # Append mode or write mode
        with open(output_file, mode=mode, newline='', encoding='utf-8') as csv_file:

            headers = input_data[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=headers)

            # Write header only if not appending
            if not append:
                writer.writeheader()
            writer.writerows(input_data)
'''