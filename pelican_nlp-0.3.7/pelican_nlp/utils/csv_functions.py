import os
import csv
from .filename_parser import parse_lpds_filename
from pelican_nlp.config import debug_print

def store_features_to_csv(input_data, derivatives_dir, doc_class, metric):
    """Store various types of features to CSV files with consistent formatting."""

    # Parse entities from the document name
    entities = parse_lpds_filename(doc_class.name)
    
    # Get the base filename without extension and current suffix
    base_filename = os.path.splitext(doc_class.name)[0]  # Remove extension
    
    # If there's a suffix in the entities, remove it from the base filename
    if 'suffix' in entities:
        # Remove the current suffix
        base_filename = base_filename.replace(f"_{entities['suffix']}", "")
    
    # Create the new filename with the metric as suffix
    filename = f"{base_filename}_{metric}.csv"
    
    # Extract core information from entities for directory structure
    participant_ID = f"part-{entities['part']}" if 'part' in entities else None
    if not participant_ID:
        raise ValueError(f"Missing required 'part' entity in filename: {doc_class.name}")
    
    session = f"ses-{entities['ses']}" if 'ses' in entities else None
    task = f"task-{entities['task']}" if 'task' in entities else None
    
    # Build the full path components
    path_components = [
        derivatives_dir,
        metric,  # Use metric as the folder name
        participant_ID,
    ]

    # Add session to path if it exists
    if session:
        path_components.append(session)

    # Add task to path if it exists
    if task:
        path_components.append(task)
    
    # Create directory and get final filepath
    # Ensure all components have compatible types by using str() conversion
    base_path = os.path.join(str(derivatives_dir), str(metric), str(participant_ID))
    
    # Build path incrementally with explicit type conversion
    if session:
        final_results_path = os.path.join(base_path, str(session))
    else:
        final_results_path = base_path
        
    if task:
        final_results_path = os.path.join(final_results_path, str(task))


    debug_print(final_results_path)
    os.makedirs(final_results_path, exist_ok=True)
    
    output_filepath = os.path.join(final_results_path, str(filename))
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

        elif metric == 'prosogram-features':
            if input_data is None or (hasattr(input_data, 'empty') and input_data.empty):
                return
                
            # Handle pandas DataFrame (main case)
            if hasattr(input_data, 'to_csv'):
                # Only write header if file doesn't exist
                if not file_exists:
                    # Write column headers
                    writer.writerow(input_data.columns.tolist())
                
                # Write data rows
                for _, row in input_data.iterrows():
                    writer.writerow(row.tolist())
            
            # Handle dictionary of multiple files (fallback case)
            elif isinstance(input_data, dict):
                # Write each file type as a separate section
                for file_type, data in input_data.items():
                    if not file_exists:
                        writer.writerow([f"File_Type: {file_type}"])
                    
                    if hasattr(data, 'to_csv'):  # DataFrame
                        if not file_exists:
                            writer.writerow(data.columns.tolist())
                        for _, row in data.iterrows():
                            writer.writerow(row.tolist())
                    else:  # Text data
                        if not file_exists:
                            writer.writerow([f"Content: {file_type}"])
                        # Write text content line by line
                        for line in str(data).split('\n'):
                            if line.strip():
                                writer.writerow([line.strip()])
                    
                    if not file_exists:
                        writer.writerow([])  # Empty line between sections

    return output_filepath


def _build_filename_parts(path_parts, corpus, metric, config=None):
    """Helper function to build filename components."""
    filename_config = config.get('filename_components', {}) if config else {}

    # Extract mandatory components
    if len(path_parts) < 3:
        raise ValueError("Invalid path format. Expected at least 'project/participant/task'.")

    participant = path_parts[-3]
    task = path_parts[-1]

    # Build filename components
    parts = [participant]

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