import os
import csv
import numpy as np

def store_features_to_csv(input_data, derivatives_dir, doc_class, metric):
    """Store various types of features to CSV files with consistent formatting.
    
    Args:
        input_data: The data to be stored in CSV format
        derivatives_dir: Base directory for all derivatives
        doc_class: Document class containing subject, session (optional), task, and task_addition (optional) info
        metric: Type of metric being stored
    """
    # Get the appropriate metric folder
    metric_folder = metric
    
    # Build base filename parts from doc_class
    filename_parts = [
        doc_class.subject_ID,
        doc_class.task,
        doc_class.corpus_name
    ]
    
    # Add session to filename if it exists
    if hasattr(doc_class, 'session') and doc_class.session:
        filename_parts.insert(1, doc_class.session)
    
    # Join the base parts with underscores
    filename = "_".join(filename_parts)
    
    # Add task_addition with underscore if it exists
    if hasattr(doc_class, 'task_addition') and doc_class.task_addition:
        filename += f"_{doc_class.task_addition}"
    
    # Add the metric with an underscore
    filename += f"_{metric}.csv"

    # Build the full path
    path_components = [
        derivatives_dir,
        metric_folder,
        doc_class.subject_ID,
    ]

    # Add session to path if it exists
    if hasattr(doc_class, 'session') and doc_class.session:
        path_components.append(doc_class.session)

    path_components.append(doc_class.task)
    
    # Create directory and get final filepath
    final_results_path = os.path.join(*path_components)
    os.makedirs(final_results_path, exist_ok=True)
    
    output_filepath = os.path.join(final_results_path, filename)
    file_exists = os.path.exists(output_filepath)
    
    # Write data based on metric type
    with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if metric == 'embeddings':
            if not isinstance(input_data, list) or not input_data:
                raise ValueError("Input data must be a non-empty list of tuples.")
            
            # Get the dimensionality from the first embedding
            embedding_dim = len(input_data[0][1])
            header = ['Token'] + [f"Dim_{i}" for i in range(embedding_dim)]
            _write_csv_header(writer, header, file_exists)
            
            for token, embedding in input_data:
                # Handle both list and tensor/array types
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                elif isinstance(embedding, list):
                    embedding_list = embedding
                else:
                    raise ValueError(f"Embedding must be either a list or have tolist() method, got {type(embedding)}")
                writer.writerow([token] + embedding_list)

        elif metric == 'cosine-similarity-matrix':
            _write_csv_header(writer, ['Matrix'], file_exists)
            for row in input_data:
                writer.writerow(row)
                
        elif metric.startswith('semantic-similarity-window-'):
            header = ['Metric', 'Similarity_Score']
            _write_csv_header(writer, header, file_exists)
            
            for metric_name, score in input_data.items():
                writer.writerow([metric_name, score])

        elif metric == 'distance-from-randomness':
            header = ['window_index', 'all_pairs_average', 'actual_dist', 'average_dist', 'std_dist']
            _write_csv_header(writer, header, file_exists)

            # Input data is a dictionary with 'section' key containing list of window results
            for window_result in input_data['section']:
                writer.writerow([
                    window_result['window_index'],
                    window_result['all_pairs_average'],
                    window_result['actual_dist'],
                    window_result['average_dist'],
                    window_result['std_dist']
                ])

        elif metric == 'logits':
            if not input_data:
                return
            header = list(input_data[0].keys())
            _write_csv_header(writer, header, file_exists)
            
            for entry in input_data:
                writer.writerow(entry.values())

        elif metric == 'opensmile-features':
            if not input_data:
                return
                
            # Get all column names from the first entry
            csv_columns = list(input_data[0].keys()) if isinstance(input_data, list) else list(input_data.keys())
            
            # Only write header if file doesn't exist
            if not file_exists:
                writer.writerow(csv_columns)
            
            # Handle both list of dictionaries and single dictionary cases
            if isinstance(input_data, list):
                for entry in input_data:
                    # Create a new array for the row data
                    row_data = []
                    for column in csv_columns:
                        # Convert numerical values to float
                        value = entry[column]
                        if isinstance(value, (int, float)):
                            value = float(value)
                        row_data.append(value)
                    writer.writerow(row_data)
            else:
                # Handle single dictionary case
                row_data = []
                for column in csv_columns:
                    value = input_data[column]
                    if isinstance(value, (int, float)):
                        value = float(value)
                    row_data.append(value)
                writer.writerow(row_data)


def _build_filename_parts(path_parts, corpus, metric, config=None):
    """Helper function to build filename components."""
    filename_config = config.get('filename_components', {}) if config else {}

    # Extract mandatory components
    if len(path_parts) < 3:
        raise ValueError("Invalid path format. Expected at least 'project/subject/task'.")

    subject = path_parts[-3]
    task = path_parts[-1]

    # Build filename components
    parts = [subject]

    # Add optional session
    if filename_config.get('session', False) and len(path_parts) >= 4:
        parts.append(path_parts[-3])

    parts.append(task)

    # Add optional components
    if filename_config.get('corpus', True):
        parts.append(corpus)
    parts.extend(filename_config.get('additional_tags', []))
    parts.append(metric)

    return parts


def _get_metric_folder(metric):
    """Determine the appropriate metric folder."""
    if metric.startswith('semantic-similarity') or metric in ['consecutive-similarities', 'cosine-similarity-matrix']:
        return 'semantic-similarity'
    return 'embeddings'


def _write_csv_header(writer, header, file_exists):
    """Write CSV header with section separation if file exists."""
    if not file_exists:
        writer.writerow(header)
    else:
        writer.writerow([])  # Separate sections
        writer.writerow(['New Section'])
        writer.writerow(header)